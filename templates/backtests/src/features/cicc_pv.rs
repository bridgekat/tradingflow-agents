//! The CICC price-volume factor handbook (中金《量化多因子系列（7）：价量因子
//! 手册》), ported to the expression vocabulary plus two custom window
//! reductions.
//!
//! The handbook groups its factors into eight categories. Five are built from
//! daily OHLCV alone and are ported here — 动量&反转 (momentum & reversal,
//! 图表4), 波动率 (volatility, 图表16), 流动性 (liquidity, 图表28), 量价相关性
//! (price-volume correlation, 图表40) and 筹码分布 (chip distribution, 图表52).
//! The other three — 资金流 (order-size money flow), 北向资金 (northbound
//! holdings) and 融资融券 (margin trading) — need intraday order-size,
//! Stock Connect and margin data the Parquet panels do not carry, and are
//! omitted wholesale.
//!
//! Most factors are formula strings lowered by [`build_features_cicc_pv`]
//! through the shared [price-field context](super::context), extended here
//! with two more fields:
//!
//! | Field | Meaning |
//! | --- | --- |
//! | `$turn` | daily turnover 换手率: raw volume / circulating shares |
//! | `$amount` | daily dollar volume (turnover value, CNY; adjustment-invariant) |
//! | `$path` | 日K线最短路径 `2(high − low) − \|open − close\|` in **raw** prices |
//!
//! `$path` is registered as a wire (rather than spelled from `$high` etc.)
//! because the context's price fields are forward-adjusted: the path is in
//! absolute price units and is divided by the raw CNY `$amount`, so the
//! adjusted spelling would scale each stock by its adjustment multiplier.
//! Every other price-shaped expression is either a same-day ratio (amplitude,
//! shadows), where the multiplier cancels exactly, or an across-day return,
//! where the adjustment is wanted.
//!
//! Two factor families are not expressible in the vocabulary and are custom
//! rolling reductions instead, built on the public
//! [`rolling::Accumulator`]/[`rolling::Rolling`] base over a stacked `(N, 2)`
//! window whose channel wires still come from the shared context:
//!
//! * **振幅调整动量 `mmt_range_{M,A}`** — sum of daily returns on the top-20%
//!   amplitude days minus the bottom-20% amplitude days of the window.
//! * **筹码分布 `distribution_*`** — a turnover-decayed holder-cost
//!   distribution: each day deposits its turnover as weight at that day's
//!   (adjusted) close, older chips decay by `Π(1 − turnover)`, and the ten
//!   factors are moments and profit/loss-band proportions of the implied
//!   holder-return distribution, over a 250-day window.
//!
//! # Porting decisions
//!
//! * **Returns are log returns** (`Delta(Log($close), 1)` of the adjusted
//!   close), so the summed families (`mmt_intraday`, `mmt_overnight`,
//!   `mmt_off_limit`, `mmt_route`) telescope exactly and
//!   `intraday + overnight = close-to-close` holds identically. The handbook's
//!   simple returns are rank-equivalent for the single-span factors and
//!   near-equivalent for the sums.
//! * **Windows are trading-day counts on the daily clock**: 1M = 21, 3M = 63,
//!   6M = 126, 1Y = 252 ticks (20 where the handbook says 20 个交易日). A
//!   stock's own non-trading days are `NaN` samples, which the rolling
//!   operators skip.
//! * **The limit-day mask is `|log return| ≥ 9.5%`** (`mmt_off_limit_*`),
//!   approximating the ±10% boards; ST (±5%) and 创业板/科创板 (±20%) boards
//!   are not distinguished, and "一字板" (locked-limit all day) is not
//!   distinguishable from a limit close.
//! * **`mmt_discrete` spells up% − down% as an explicit ratio** over the days
//!   the stock actually traded (`Sum(If(...)) / Count(...)`), because `Sign`
//!   maps `+0.0` to `1` (an unchanged close would count as an up day) and a
//!   comparison with a `NaN` non-trading day is `false`.
//! * **`mmt_highest_days_A` uses the high price** (the handbook's 最高价), as
//!   `252 - IdxMax($high, 252)`: `IdxMax` is the 1-based window position of
//!   the extreme (oldest = 1), so the difference counts ticks since it.
//! * **Downside/upside returns preserve `NaN`** via `Less($ret, $ret * 0)`
//!   (resp. `Greater`): `x * 0` is `0` for finite `x` and `NaN` for `NaN`, so
//!   a suspended day stays missing instead of entering the std as a zero.
//! * **Shadows are normalized per the handbook**: upper shadows (标准化上影线,
//!   威廉上影线) divide by the high, lower shadows by the low.
//! * **报告期动量 (`mmt_report_overnight`, `mmt_report_jump`,
//!   `mmt_report_period`) are omitted**: they anchor windows to earnings
//!   announcement dates, which needs event-triggered sampling the expression
//!   clock cannot express. The chip window is 250 days (the handbook does not
//!   state one in its table).

use std::ops::Range;

use tradingflow::{
    data::{Array, ArrayView, Instant, Retention, SeriesView},
    expr::Context,
    graph::{Builder, OperatorExt},
    operators::{
        array::stack,
        elem,
        rolling::{self, Accumulator},
        series::buffer,
    },
    ports::ArrayPortHandle,
    time::UnixTime,
};

use crate::data::MarketData;

/// ~1 trading month, in ticks of the daily clock.
const M: usize = 21;
/// ~1 trading year, in ticks of the daily clock.
const Y: usize = 252;
/// The chip-distribution look-back, in ticks of the daily clock.
const CHIP_WINDOW: usize = 250;

/// Daily building blocks, as named spellings inlined into the factor strings
/// below — hash-consing during lowering makes each one a single subgraph no
/// matter how many factors reference it.
const RET: &str = "Delta(Log($close), 1)"; // adjusted close-to-close log return
const INTRA: &str = "(Log($close) - Log($open))"; // 日内 intraday log return
const OVER: &str = "(Log($open) - Ref(Log($close), 1))"; // 隔夜 overnight log return
const MASKED: &str = // daily log return, limit days (≥ ~9.5%) masked to 0
    "If(Abs(Delta(Log($close), 1)) < 0.095, Delta(Log($close), 1), 0)";
const HIGHLOW: &str = "($high / $low)"; // 日内振幅 daily amplitude
const TURND: &str = "Delta($turn, 1)"; // 换手率变动 turnover change

/// The 动量&反转 (momentum & reversal) factors expressible as formulas
/// (图表4), as `(name, expression)` built by [`momentum`]. `_M` is the
/// trailing month, `_A` the trailing year (minus the month, where the
/// handbook nets the reversal out of the momentum).
fn momentum() -> Vec<(String, String)> {
    let f = |name: &str, src: String| (name.to_string(), src);
    vec![
        // 1个月收益率 / 12个月收益率 − 1个月收益率.
        f("mmt_normal_M", format!("Delta(Log($close), {M})")),
        f(
            "mmt_normal_A",
            format!("Ref(Log($close), {M}) - Ref(Log($close), {Y})"),
        ),
        // 当期收盘价 / 20日均价; 1个月前收盘价 / 1年均价.
        f("mmt_avg_M", "$close / Mean($close, 20)".into()),
        f("mmt_avg_A", format!("Ref($close, {M}) / Mean($close, {Y})")),
        // 日内 / 隔夜涨跌幅之和.
        f("mmt_intraday_M", format!("Sum({INTRA}, {M})")),
        f(
            "mmt_intraday_A",
            format!("Sum({INTRA}, {Y}) - Sum({INTRA}, {M})"),
        ),
        f("mmt_overnight_M", format!("Sum({OVER}, {M})")),
        f(
            "mmt_overnight_A",
            format!("Sum({OVER}, {Y}) - Sum({OVER}, {M})"),
        ),
        // 去涨跌停动量: the same sums over the limit-masked daily return.
        f("mmt_off_limit_M", format!("Sum({MASKED}, {M})")),
        f(
            "mmt_off_limit_A",
            format!("Sum({MASKED}, {Y}) - Sum({MASKED}, {M})"),
        ),
        // 路径调整动量: span return / Σ|daily return|.
        f(
            "mmt_route_M",
            format!("Delta(Log($close), {M}) / Sum(Abs({RET}), {M})"),
        ),
        f(
            "mmt_route_A",
            format!("Delta(Log($close), {Y}) / Sum(Abs({RET}), {Y})"),
        ),
        // 信息离散度: 上涨天数占比 − 下跌天数占比, over traded days.
        f(
            "mmt_discrete_M",
            format!(
                "(Sum(If({RET} > 0, 1, 0), {M}) - Sum(If({RET} < 0, 1, 0), {M})) \
                 / Count({RET}, {M})"
            ),
        ),
        f(
            "mmt_discrete_A",
            format!(
                "(Sum(If({RET} > 0, 1, 0), {Y}) - Sum(If({RET} < 0, 1, 0), {Y})) \
                 / Count({RET}, {Y})"
            ),
        ),
        // 横截面 rank 动量: daily cross-sectional return rank, time-averaged.
        f("mmt_sec_rank_M", format!("Mean(CSRank({RET}), 20)")),
        f("mmt_sec_rank_A", format!("Mean(CSRank({RET}), {Y})")),
        // 时序 rank 动量: daily 1-year time-series price rank, 20-day mean.
        f(
            "mmt_time_rank_M",
            format!("Mean(Rank(Log($close), {Y}), 20)"),
        ),
        // 最高价距今天数: ticks since the 1-year high.
        f("mmt_highest_days_A", format!("{Y} - IdxMax($high, {Y})")),
    ]
}

/// The windowed factor families (波动率 图表16, 流动性 图表28), as
/// `(name prefix, expression template)` instantiated per window by
/// [`windowed`]: `{d}` is substituted with the tick count and the `1M`/`3M`/
/// `6M` tag is appended to the prefix.
#[rustfmt::skip]
const WINDOWED: [(&str, &str); 20] = [
    // ---- 波动率: 收益波动率 (plain / downside / upside) ----
    ("vol_std",             "Std({ret}, {d})"),
    ("vol_down_std",        "Std(Less({ret}, {ret} * 0), {d})"),
    ("vol_up_std",          "Std(Greater({ret}, {ret} * 0), {d})"),
    // ---- 波动率: 振幅与影线 (amplitude and candlestick shadows) ----
    ("vol_highlow_avg",     "Mean({hl}, {d})"),
    ("vol_highlow_std",     "Std({hl}, {d})"),
    ("vol_upshadow_avg",    "Mean(($high - Greater($open, $close)) / $high, {d})"),
    ("vol_upshadow_std",    "Std(($high - Greater($open, $close)) / $high, {d})"),
    ("vol_downshadow_avg",  "Mean((Less($open, $close) - $low) / $low, {d})"),
    ("vol_downshadow_std",  "Std((Less($open, $close) - $low) / $low, {d})"),
    ("vol_w_upshadow_avg",  "Mean(($high - $close) / $high, {d})"),
    ("vol_w_upshadow_std",  "Std(($high - $close) / $high, {d})"),
    ("vol_w_downshadow_avg", "Mean(($close - $low) / $low, {d})"),
    ("vol_w_downshadow_std", "Std(($close - $low) / $low, {d})"),
    // ---- 流动性: 换手率 (turnover level and dispersion) ----
    ("liq_turn_avg",        "Mean($turn, {d})"),
    ("liq_turn_std",        "Std($turn, {d})"),
    // ---- 流动性: 价格弹性 (price elasticity) ----
    // These carry CNY scales spanning many orders of magnitude across the
    // cross-section, so the rank transform (`CSRank`) is part of their
    // definition — the same idiom as the fundamental catalogs.
    ("liq_vstd",            "CSRank(Sum($amount, {d}) / Std({ret}, {d}))"), // 成交波动比
    ("liq_amihud_avg",      "CSRank(Mean(Abs({ret}) / $amount, {d}))"),
    ("liq_amihud_std",      "CSRank(Std(Abs({ret}) / $amount, {d}))"),
    ("liq_shortcut_avg",    "CSRank(Mean($path / $amount, {d}))"), // 最短路径非流动
    ("liq_shortcut_std",    "CSRank(Std($path / $amount, {d}))"),
];

/// Instantiates every [`WINDOWED`] family at 1M/3M/6M.
fn windowed() -> Vec<(String, String)> {
    WINDOWED
        .iter()
        .flat_map(|&(name, src)| {
            [("1M", M), ("3M", 63), ("6M", 126)].map(|(tag, d)| {
                let src = src
                    .replace("{d}", &d.to_string())
                    .replace("{ret}", RET)
                    .replace("{hl}", HIGHLOW);
                (format!("{name}_{tag}"), src)
            })
        })
        .collect()
}

/// The 量价相关性 (price-volume correlation) factors (图表40): 20-day
/// correlations of turnover (and its change) against price and return, in the
/// synchronous (量价同步), volume-leading (量领先一期, turnover lagged) and
/// price-leading (价领先一期, price lagged) alignments.
fn correlation() -> Vec<(String, String)> {
    let f = |name: &str, src: String| (name.to_string(), src);
    vec![
        f("corr_price_turn_1M", "Corr($turn, $close, 20)".into()),
        f(
            "corr_price_turn_post_1M",
            "Corr(Ref($turn, 1), $close, 20)".into(),
        ),
        f(
            "corr_price_turn_prior_1M",
            "Corr($turn, Ref($close, 1), 20)".into(),
        ),
        f("corr_ret_turn_1M", format!("Corr($turn, {RET}, 20)")),
        f(
            "corr_ret_turn_post_1M",
            format!("Corr(Ref($turn, 1), {RET}, 20)"),
        ),
        f(
            "corr_ret_turn_prior_1M",
            format!("Corr($turn, Ref({RET}, 1), 20)"),
        ),
        f("corr_ret_turnd_1M", format!("Corr({TURND}, {RET}, 20)")),
        f(
            "corr_ret_turnd_post_1M",
            format!("Corr(Ref({TURND}, 1), {RET}, 20)"),
        ),
        f(
            "corr_ret_turnd_prior_1M",
            format!("Corr({TURND}, Ref({RET}, 1), 20)"),
        ),
    ]
}

// ---------------------------------------------------------------------------
// Custom window reductions: joint per-lane statistics over a stacked (N, 2)
// window, which the expression vocabulary cannot express.
// ---------------------------------------------------------------------------

/// A window-scanning [`Accumulator`] over a two-channel `(N, 2)` series,
/// applying `f(channel 0, channel 1)` per lane once the window is full.
struct Scan2 {
    window: usize,
    f: fn(&[f64], &[f64]) -> f64,
    out_len: usize,
}

impl Accumulator<f64, 2, f64, 1> for Scan2 {
    fn init(&mut self, extents: [usize; 2]) -> Array<f64, 1> {
        assert_eq!(extents[1], 2, "Scan2 expects an (N, 2) input");
        self.out_len = extents[0];
        Array::full([extents[0]], f64::NAN)
    }

    fn add(&mut self, _: ArrayView<f64, 2>) {}
    fn remove(&mut self, _: ArrayView<f64, 2>) {}

    fn write(&mut self, out: &mut Array<f64, 1>, window: SeriesView<'_, f64, 2>) {
        let w = self.window;
        let out = out.data_mut();
        if window.len() < w {
            out.fill(f64::NAN);
            return;
        }
        let start = window.range().end - w;
        let rows: Vec<_> = (0..w)
            .map(|k| window.at(start + k).1.to_contiguous())
            .collect();
        let (mut c0, mut c1) = (vec![0.0f64; w], vec![0.0f64; w]);
        for (j, o) in out.iter_mut().enumerate() {
            for k in 0..w {
                c0[k] = rows[k][j * 2];
                c1[k] = rows[k][j * 2 + 1];
            }
            *o = (self.f)(&c0, &c1);
        }
    }
}

/// 振幅调整动量: Σ(returns of the top-20%-amplitude days) − Σ(bottom-20%);
/// `amp` is the amplitude channel, `ret` the daily-return channel. `NaN`
/// (non-trading) days are excluded before the 20% split; fewer than 5 traded
/// days is `NaN`.
fn range_mom(amp: &[f64], ret: &[f64]) -> f64 {
    let mut pairs: Vec<(f64, f64)> = amp
        .iter()
        .zip(ret)
        .filter(|(a, r)| a.is_finite() && r.is_finite())
        .map(|(a, r)| (*a, *r))
        .collect();
    let m = pairs.len();
    if m < 5 {
        return f64::NAN;
    }
    pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    let k = ((m as f64) * 0.2).round().max(1.0) as usize;
    let bottom: f64 = pairs[..k].iter().map(|p| p.1).sum();
    let top: f64 = pairs[m - k..].iter().map(|p| p.1).sum();
    top - bottom
}

/// The 筹码分布 factor names, in [`ChipDist`] column order.
const CHIP_NAMES: [&str; 10] = [
    "distribution_ret_avg",      // 筹码平均收益率
    "distribution_ret_std",      // 筹码标准差
    "distribution_ret_skew",     // 筹码偏度
    "distribution_ret_kurt",     // 筹码峰度
    "distribution_max_prob_ret", // 最大筹码收益率 (modal bin)
    "distribution_bal",          // −2%..2% 筹码占比
    "distribution_profit_l",     // >10% 筹码占比
    "distribution_profit_s",     // 2%..10% 筹码占比
    "distribution_loss_s",       // −10%..−2% 筹码占比
    "distribution_loss_l",       // <−10% 筹码占比
];

/// 筹码分布 (chip distribution) [`Accumulator`]: maintains a recent
/// transaction-volume cost distribution from a `(adjusted close, turnover)`
/// window. Each day deposits `turnover` weight at that day's close; older
/// chips decay by later days' turnover (`× Π(1 − turnover)`). The holder
/// return of a chip bought at `C`, valued at the current close `P`, is
/// `P/C − 1`; the output is one `(N, 10)` cross-section of the
/// [`CHIP_NAMES`] statistics of that distribution.
///
/// # Incremental design
///
/// A day's turnover multiplies every older chip by the same survival factor,
/// so a chip's weight relative to its cohort never changes after deposit.
/// The state is therefore a ring of `(weight, 1/price)` slots kept in
/// **current scale**: each new day rescales the live slots and every running
/// sum by its survival factor `1 − t` — the [`rolling::mean_exp`] idiom —
/// and then deposits weight `t` at that day's price. Daily rescaling keeps
/// every number bounded (no cumulative survival product to underflow, no
/// deposit-scale ratios to overflow), and arithmetic error injected at any
/// tick decays under the same factor as the data, so the sums need no
/// periodic resnap. A slot leaving the window subtracts its (already
/// decayed) contribution and frees its ring position for the arriving row.
///
/// The moments are O(1) per tick: with `x = 1/price`, the holder return is
/// `r = P·x − 1`, so `E[rᵏ]` is a degree-`k` polynomial in today's close `P`
/// whose coefficients are the running power sums `Tₖ = Σ weight · xᵏ`
/// (`k = 0..=4`); readout substitutes `P` and converts raw to central
/// moments, as [`rolling::skew`]/[`rolling::kurt`] do. The conversion's
/// cancellation noise (order `ε · E[(P/price)²]`) sets the degeneracy guard:
/// a variance at or below it — a one-chip distribution, e.g. right after a
/// full-turnover day — reads as zero std and `NaN` skew/kurtosis, where the
/// windowed form's centered sums hit an exact 0. The profit/loss bands
/// and the modal bin have thresholds that move with `P`, so they remain a
/// linear scan over the live slots — one fused multiply-free pass, instead
/// of the full reconstruction (weights, moments and bins) the window-scanning
/// form recomputes per tick.
struct ChipDist {
    window: usize,
    lanes: usize,
    /// The absolute input-series range already folded into the state.
    done: Range<usize>,
    /// Ring slots, lane-major `[lanes][window]`, addressed by absolute row
    /// index `mod window`; a dead slot holds zero weight. The slot keeps the
    /// deposit *price* rather than its inverse so the band scan can divide —
    /// bit-identical to the windowed reconstruction, which matters exactly at
    /// the bin edges (today's chip sits precisely on the `r = 0` edge, and
    /// `p · (1/p) − 1` misses it by an ulp).
    wgt: Vec<f64>,
    price: Vec<f64>,
    /// The running power sums `Tₖ[lane] = Σ wgt · (1/price)ᵏ`, `k = 0..=4`.
    sums: [Vec<f64>; 5],
}

impl ChipDist {
    fn new(window: usize) -> Self {
        Self {
            window,
            lanes: 0,
            done: 0..0,
            wgt: Vec::new(),
            price: Vec::new(),
            sums: std::array::from_fn(|_| Vec::new()),
        }
    }
}

impl Accumulator<f64, 2, f64, 2> for ChipDist {
    fn init(&mut self, extents: [usize; 2]) -> Array<f64, 2> {
        assert_eq!(extents[1], 2, "ChipDist expects an (N, 2) input");
        let (n, w) = (extents[0], self.window);
        self.lanes = n;
        self.done = 0..0;
        self.wgt = vec![0.0; n * w];
        self.price = vec![0.0; n * w];
        self.sums = std::array::from_fn(|_| vec![0.0; n]);
        Array::full([n, CHIP_NAMES.len()], f64::NAN)
    }

    fn add(&mut self, _: ArrayView<f64, 2>) {}
    fn remove(&mut self, _: ArrayView<f64, 2>) {}

    fn write(&mut self, out: &mut Array<f64, 2>, window: SeriesView<'_, f64, 2>) {
        const C: usize = CHIP_NAMES.len();
        let (w, n) = (self.window, self.lanes);
        let range = window.range();

        // Evict the rows that left the window: subtract each expired slot's
        // current-scale contribution and free its ring position (exactly the
        // position the same-numbered arriving row will reuse). `1/price`
        // recomputes to the same bits as at deposit, so the subtraction
        // cancels the deposit exactly up to the interleaved decay rescales.
        for i in self.done.start..range.start.min(self.done.end) {
            let pos = i % w;
            for j in 0..n {
                let s = j * w + pos;
                let wg = self.wgt[s];
                if wg > 0.0 {
                    let ip = 1.0 / self.price[s];
                    let ip2 = ip * ip;
                    self.sums[0][j] -= wg;
                    self.sums[1][j] -= wg * ip;
                    self.sums[2][j] -= wg * ip2;
                    self.sums[3][j] -= wg * ip2 * ip;
                    self.sums[4][j] -= wg * ip2 * ip2;
                }
                self.wgt[s] = 0.0;
                self.price[s] = 0.0;
            }
        }

        // Ingest the new rows in time order: the day's turnover rescales
        // every live chip (and the sums with them), then the day deposits
        // its own chip. An invalid price still decays older chips but
        // deposits nothing, and an invalid turnover is no trade at all —
        // both as in the windowed form.
        for i in self.done.end.max(range.start)..range.end {
            let row = window.at(i).1.to_contiguous();
            let pos = i % w;
            for j in 0..n {
                let (p, t) = (row[j * 2], row[j * 2 + 1]);
                let t = if t.is_finite() {
                    t.clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let d = 1.0 - t;
                if d < 1.0 {
                    for wg in &mut self.wgt[j * w..(j + 1) * w] {
                        *wg *= d;
                    }
                    for sums in &mut self.sums {
                        sums[j] *= d;
                    }
                }
                let s = j * w + pos;
                if t > 0.0 && p.is_finite() && p > 0.0 {
                    let ip = 1.0 / p;
                    let ip2 = ip * ip;
                    self.wgt[s] = t;
                    self.price[s] = p;
                    self.sums[0][j] += t;
                    self.sums[1][j] += t * ip;
                    self.sums[2][j] += t * ip2;
                    self.sums[3][j] += t * ip2 * ip;
                    self.sums[4][j] += t * ip2 * ip2;
                } else {
                    self.wgt[s] = 0.0;
                    self.price[s] = 0.0;
                }
            }
        }
        self.done = range.clone();

        let out = out.data_mut();
        if window.len() < w {
            out.fill(f64::NAN);
            return;
        }
        let newest = window.at(range.end - 1).1.to_contiguous();
        for j in 0..n {
            let base = j * C;
            let p = newest[j * 2];
            let t0 = self.sums[0][j];
            if !(p.is_finite() && p > 0.0) || t0 <= 0.0 {
                out[base..base + C].fill(f64::NAN);
                continue;
            }
            // Raw moments of `r = p·x − 1` from the power sums —
            // `qₖ = E[(p·x)ᵏ]` — then raw → central.
            let q1 = self.sums[1][j] / t0 * p;
            let q2 = self.sums[2][j] / t0 * (p * p);
            let q3 = self.sums[3][j] / t0 * (p * p * p);
            let q4 = self.sums[4][j] / t0 * (p * p * p * p);
            let avg = q1 - 1.0;
            let r2 = q2 - 2.0 * q1 + 1.0;
            let r3 = q3 - 3.0 * q2 + 3.0 * q1 - 1.0;
            let r4 = q4 - 4.0 * q3 + 6.0 * q2 - 4.0 * q1 + 1.0;
            let m2 = r2 - avg * avg;
            let m3 = r3 - 3.0 * avg * r2 + 2.0 * avg * avg * avg;
            let m4 = r4 - 4.0 * avg * r3 + 6.0 * (avg * avg) * r2 - 3.0 * avg.powi(4);
            // The raw→central subtraction carries cancellation noise of
            // order `ε · q₂`, so the degeneracy guard scales with it (the
            // windowed form's centered sums hit an exact 0 there): a
            // degenerate distribution reads as zero std and `NaN`
            // skew/kurtosis. Any real chip spread sits many orders above.
            let m2_min = 1e-16f64.max(1e-13 * (q2.abs() + q1 * q1));
            let (std, skew, kurt) = if m2 > m2_min {
                (m2.sqrt(), m3 / (m2 * m2.sqrt()), m4 / (m2 * m2))
            } else {
                (0.0, f64::NAN, f64::NAN)
            };
            // Profit/loss-band chip proportions + the modal-bin return
            // ([-0.5, 0.5] in 0.02-wide bins, extremes clamped into the
            // ends): the thresholds move with `p`, so this stays a scan.
            let (mut bal, mut pl, mut ps, mut ls, mut ll) = (0.0, 0.0, 0.0, 0.0, 0.0);
            let mut bins = [0.0f64; 50];
            for s in j * w..(j + 1) * w {
                let wg = self.wgt[s];
                if wg <= 0.0 {
                    continue;
                }
                let r = p / self.price[s] - 1.0;
                if r > 0.10 {
                    pl += wg;
                } else if r > 0.02 {
                    ps += wg;
                } else if r >= -0.02 {
                    bal += wg;
                } else if r >= -0.10 {
                    ls += wg;
                } else {
                    ll += wg;
                }
                let b = (((r + 0.5) / 0.02).floor() as isize).clamp(0, 49) as usize;
                bins[b] += wg;
            }
            // Normalize the bands by their own sum (every live chip falls in
            // exactly one band), which keeps the proportions summing to 1
            // independent of any residual drift in `T₀`.
            let total = pl + ps + bal + ls + ll;
            if total <= 0.0 {
                out[base..base + C].fill(f64::NAN);
                continue;
            }
            let (mut maxb, mut maxv) = (0usize, bins[0]);
            for (b, &v) in bins.iter().enumerate().skip(1) {
                if v > maxv {
                    maxv = v;
                    maxb = b;
                }
            }
            let max_prob_ret = -0.5 + (maxb as f64 + 0.5) * 0.02;
            out[base..base + C].copy_from_slice(&[
                avg,
                std,
                skew,
                kurt,
                max_prob_ret,
                bal / total,
                pl / total,
                ps / total,
                ls / total,
                ll / total,
            ]);
        }
    }
}

/// Lowers the whole price-volume catalog through `ctx`, returning one
/// `(name, raw value)` entry per factor. Registers the extra context fields
/// (`$turn`, `$amount`, `$path` — see the [module docs](self)) from `m`, then
/// lowers the formula tables and wires the two custom reductions from
/// context-lowered channel wires, so every shared subexpression stays a
/// single hash-consed node.
pub fn build_features_cicc_pv(
    b: &mut Builder<Instant, UnixTime>,
    ctx: &mut Context<1>,
    m: &MarketData,
) -> Vec<(String, ArrayPortHandle<f64, 1>)> {
    // Extra fields over the shared price context. `$turn` divides raw volume
    // by raw circulating shares (both unadjusted, so the ratio is exact);
    // `$path` is built from the raw OHLC (see the module docs).
    let turn = b.op(elem::div(), (m.volume, m.field("shares.circulating")));
    ctx.add_field("turn", turn);
    ctx.add_field("amount", m.amount);
    let hl = b.op(elem::sub(), (m.high, m.low));
    let oc = b.op(elem::sub(), (m.open, m.close));
    let abs_oc = b.op(elem::abs(), oc);
    let hl2 = b.op(elem::add(), (hl, hl));
    let path = b.op(elem::sub(), (hl2, abs_oc));
    ctx.add_field("path", path);

    let mut entries: Vec<(String, ArrayPortHandle<f64, 1>)> = Vec::new();
    for (name, src) in momentum()
        .into_iter()
        .chain(windowed())
        .chain(correlation())
    {
        let wire = ctx
            .lower_str(b, &src)
            .unwrap_or_else(|e| panic!("{name}: {e}\n  {src}"));
        entries.push((name, wire));
    }

    // 振幅调整动量: (amplitude, daily return) pairs reduced per window.
    let amp = ctx.lower_str(b, HIGHLOW).expect("amplitude lowering");
    let ret = ctx.lower_str(b, RET).expect("return lowering");
    let amp_ret = b.op(stack::<f64, 1, 2>(1), &[amp, ret][..]);
    for (name, w) in [("mmt_range_M", M), ("mmt_range_A", Y)] {
        let acc = Scan2 {
            window: w,
            f: range_mom,
            out_len: 0,
        };
        let h = b.op(
            buffer(Retention::count(w)).then(rolling::Rolling::new(Retention::count(w), acc)),
            (m.daily, amp_ret),
        );
        entries.push((name.to_string(), h));
    }

    // 筹码分布: (adjusted close, turnover) → (N, 10), one column per factor.
    let close = ctx.lower_str(b, "$close").expect("close lowering");
    let chip_in = b.op(stack::<f64, 1, 2>(1), &[close, turn][..]);
    let chip = b.op(
        buffer(Retention::count(CHIP_WINDOW)).then(rolling::Rolling::new(
            Retention::count(CHIP_WINDOW),
            ChipDist::new(CHIP_WINDOW),
        )),
        (m.daily, chip_in),
    );
    for (c, name) in CHIP_NAMES.into_iter().enumerate() {
        let h = b.op(tradingflow::operators::array::select_at(c, 1), chip);
        entries.push((name.to_string(), h));
    }

    entries
}

#[cfg(test)]
mod tests {
    use tradingflow::data::{Duration, Series};
    use tradingflow::graph::Pool;
    use tradingflow::operators::series::record_all;
    use tradingflow::sources::sync;

    use super::*;

    const LANES: usize = 3;
    const W: usize = 60;
    const TICKS: usize = 300;

    /// Deterministic uniform in `[0, 1)` (an LCG, so the test is hermetic).
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Synthetic `(price, turnover)` rows: random-walk prices and skewed
    /// turnovers, with suspended days (`NaN` price), invalid turnovers
    /// (`NaN`), zero-turnover days, one full-turnover wipe (`t = 1`) and one
    /// out-of-range turnover (`t = 2.5`, exercising the clamp).
    fn synthetic_rows() -> Vec<Vec<(f64, f64)>> {
        let mut state = 42u64;
        let mut prices = [50.0, 8.0, 300.0];
        (0..TICKS)
            .map(|i| {
                (0..LANES)
                    .map(|j| {
                        prices[j] *= ((lcg(&mut state) - 0.5) * 0.1).exp();
                        let mut p = prices[j];
                        let mut t = lcg(&mut state).powi(2) * 0.4;
                        if (i + 7 * j) % 37 == 0 {
                            p = f64::NAN; // suspended: no quote today
                        }
                        if (i + 11 * j) % 53 == 0 {
                            t = f64::NAN; // invalid turnover
                        }
                        if (i + 13 * j) % 29 == 0 {
                            t = 0.0; // no volume
                        }
                        if j == 0 && i == 150 {
                            t = 1.0; // full turnover: wipes all older chips
                        }
                        if j == 0 && i == 170 {
                            t = 2.5; // out of range: clamps to 1
                        }
                        (p, t)
                    })
                    .collect()
            })
            .collect()
    }

    /// The windowed reference: the O(w)-reconstruction algorithm the
    /// incremental [`ChipDist`] replaced, ported verbatim. Returns the
    /// [`CHIP_NAMES`] row for one lane's last-`W` window.
    fn reference(price: &[f64], turn: &[f64]) -> [f64; 10] {
        let w = price.len();
        let mut weight = vec![0.0f64; w];
        let mut surv = 1.0f64;
        for i in (0..w).rev() {
            let t = if turn[i].is_finite() {
                turn[i].clamp(0.0, 1.0)
            } else {
                0.0
            };
            weight[i] = if price[i].is_finite() && price[i] > 0.0 {
                t * surv
            } else {
                0.0
            };
            surv *= 1.0 - t;
        }
        let p_cur = price[w - 1];
        let total: f64 = weight.iter().sum();
        if !(p_cur.is_finite() && p_cur > 0.0) || total <= 0.0 {
            return [f64::NAN; 10];
        }
        let mut ret = vec![0.0f64; w];
        for i in 0..w {
            weight[i] /= total;
            ret[i] = if weight[i] > 0.0 {
                p_cur / price[i] - 1.0
            } else {
                0.0
            };
        }
        let mut avg = 0.0;
        for i in 0..w {
            avg += weight[i] * ret[i];
        }
        let (mut m2, mut m3, mut m4) = (0.0, 0.0, 0.0);
        for i in 0..w {
            let d = ret[i] - avg;
            let (wd, d2) = (weight[i], d * d);
            m2 += wd * d2;
            m3 += wd * d2 * d;
            m4 += wd * d2 * d2;
        }
        let std = m2.sqrt();
        let skew = if m2 > 1e-16 {
            m3 / (m2 * std)
        } else {
            f64::NAN
        };
        let kurt = if m2 > 1e-16 { m4 / (m2 * m2) } else { f64::NAN };
        let (mut bal, mut pl, mut ps, mut ls, mut ll) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let mut bins = [0.0f64; 50];
        for i in 0..w {
            let (r, wd) = (ret[i], weight[i]);
            if wd <= 0.0 {
                continue;
            }
            if r > 0.10 {
                pl += wd;
            } else if r > 0.02 {
                ps += wd;
            } else if r >= -0.02 {
                bal += wd;
            } else if r >= -0.10 {
                ls += wd;
            } else {
                ll += wd;
            }
            let b = (((r + 0.5) / 0.02).floor() as isize).clamp(0, 49) as usize;
            bins[b] += wd;
        }
        let (mut maxb, mut maxv) = (0usize, bins[0]);
        for (b, &v) in bins.iter().enumerate().skip(1) {
            if v > maxv {
                maxv = v;
                maxb = b;
            }
        }
        let max_prob_ret = -0.5 + (maxb as f64 + 0.5) * 0.02;
        [avg, std, skew, kurt, max_prob_ret, bal, pl, ps, ls, ll]
    }

    fn assert_close(name: &str, tick: usize, lane: usize, got: f64, want: f64, tol: f64) {
        let ok = (got.is_nan() && want.is_nan()) || (got - want).abs() <= tol * (1.0 + want.abs());
        assert!(ok, "{name} tick {tick} lane {lane}: got {got}, want {want}");
    }

    /// Drives [`ChipDist`] through a real graph over the synthetic series and
    /// compares every tick of every lane against [`reference`].
    #[tokio::test]
    async fn chip_dist_matches_windowed_reference() {
        let rows = synthetic_rows();
        let timestamps: Vec<Instant> = (0..TICKS)
            .map(|i| Instant::from_offset(Duration::from_days(i as i64)))
            .collect();
        let values: Vec<Array<f64, 2>> = rows
            .iter()
            .map(|row| {
                let flat: Vec<f64> = row.iter().flat_map(|&(p, t)| [p, t]).collect();
                Array::from_parts([LANES, 2], flat.into())
            })
            .collect();

        let mut b = Builder::new(UnixTime);
        let (signal, array) = b.source(sync::array_series(Series::from((timestamps, values))));
        let chip = b.op(
            buffer(Retention::count(W))
                .then(rolling::Rolling::new(Retention::count(W), ChipDist::new(W))),
            (signal, array),
        );
        let recorded = b.op(record_all(), (signal, chip));
        let mut g = b.build();
        g.run(&mut Pool::new(2), |_, _| {}).await;

        let series = g.view(recorded);
        assert_eq!(series.range().len(), TICKS);
        for tick in 0..TICKS {
            let row = series.at(tick).1.to_contiguous();
            for lane in 0..LANES {
                let want = if tick + 1 < W {
                    [f64::NAN; 10] // warm-up: window not yet full
                } else {
                    let window = &rows[tick + 1 - W..=tick];
                    let price: Vec<f64> = window.iter().map(|r| r[lane].0).collect();
                    let turn: Vec<f64> = window.iter().map(|r| r[lane].1).collect();
                    reference(&price, &turn)
                };
                for (c, name) in CHIP_NAMES.into_iter().enumerate() {
                    // The std/skew/kurt go through the raw→central
                    // conversion, whose cancellation noise (≈ √ε for the
                    // std of a degenerate one-chip distribution, e.g. right
                    // after a full-turnover wipe) the centered reference
                    // does not share — so they get a looser tolerance than
                    // the mean and the directly-summed bands.
                    let tol = match c {
                        1..=3 => 1e-6,
                        _ => 1e-9,
                    };
                    assert_close(name, tick, lane, row[lane * 10 + c], want[c], tol);
                }
            }
        }
    }
}
