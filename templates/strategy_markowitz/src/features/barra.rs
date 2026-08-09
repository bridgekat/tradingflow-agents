//! The BARRA-style risk factor catalog: the classic style descriptors of
//! Barra CNE5 / USE4, one raw entry per descriptor, for the covariance
//! predictor to regress realized returns on. The country intercept and the
//! industry dummy columns that complete a BARRA exposure matrix live outside
//! this catalog — see [`build_features`](super::build_features) and the
//! [`industry`](super::industry) set.
//!
//! # Catalog
//!
//! Style descriptors (finalized by winsorize → cross-sectional z-score in
//! [`build_features`](super::build_features)):
//!
//! | Name | CNE5 factor | Spelling here |
//! | --- | --- | --- |
//! | `SIZE` | Size | ln(total market cap) |
//! | `BETA` | Beta | 1-year regression of daily returns on `$mkt` |
//! | `MOM` | Momentum | `RSTR`: 2-year log return, most recent month excluded |
//! | `DASTD` | Residual Volatility | 1-year std of daily returns |
//! | `CMRA` | Residual Volatility | 1-year log price range (max − min) |
//! | `HSIGMA` | Residual Volatility | residual std of the `BETA` regression |
//! | `NLSIZE` | Non-linear Size | cube of z-scored `SIZE`, orthogonalized to it |
//! | `BTOP` | Book-to-Price | 归母净资产 / total market cap |
//! | `STOM` | Liquidity | ln(1-month turnover sum) |
//! | `STOQ` | Liquidity | ln(monthly-average turnover over 3 months) |
//! | `STOA` | Liquidity | ln(monthly-average turnover over 12 months) |
//! | `ETOP` | Earnings Yield | net profit TTM / total market cap |
//! | `CETOP` | Earnings Yield | operating cash flow TTM / total market cap |
//! | `EGRO` | Growth | YoY growth of net profit TTM |
//! | `SGRO` | Growth | YoY growth of revenue TTM |
//! | `MLEV` | Leverage | (market cap + long-term debt) / market cap |
//! | `DTOA` | Leverage | total liabilities / total assets |
//! | `BLEV` | Leverage | (book equity + long-term debt) / book equity |
//!
//! The market series `$mkt` is the cap-weighted cross-sectional mean of the
//! daily log returns (weighted by the previous trading day's circulating
//! market cap), broadcast to every lane so the pairwise expression operators
//! can read it like any other field.
//!
//! # Deviations from CNE5, beyond spelling
//!
//! * **Descriptors stay separate columns** instead of being averaged into
//!   composite factors (CNE5's Residual Volatility = 0.74·DASTD + 0.16·CMRA +
//!   0.10·HSIGMA, etc.): the factor regression estimates one return per
//!   descriptor directly, which subsumes the fixed descriptor weights.
//! * **Plain rolling windows** replace the exponential halflife weights
//!   (63-day for `BETA`/`HSIGMA`, 42-day for `DASTD`, 126-day for `MOM`),
//!   which the expression vocabulary does not carry for pair statistics.
//! * **`CMRA` is the 1-year log price range**: the cumulative ranges
//!   `Z(T) = ln P_now − ln P_{−T}` satisfy `max_T Z − min_T Z =
//!   max_τ ln P_τ − min_τ ln P_τ` over the window, so the daily-sampled range
//!   equals the handbook's monthly-sampled one up to sampling density.
//! * **Growth is the year-over-year TTM growth** ([`growth`]) rather than the
//!   5-year regression slope of annual reports — the panels' history is too
//!   short to spend five years warming up.
//! * **`EPFWD` (forecast earnings yield) is omitted**: no analyst-estimate
//!   panel. `HSIGMA` derives from the same-window identity
//!   `Var[resid] = Var[r] − β²·Var[mkt]`, clamped at zero, instead of a
//!   separate residual pass.
//! * **Descriptors are not cap-weighted at standardization**: the finalizer
//!   z-scores each cross-section equal-weighted.
//!
//! `MOM` looks back [`MOM_T`] trading days, the longest window of any
//! catalog: the covariance predictor sees no complete row before it fills,
//! so give `--warmup-days` around 1100 (≈ 750 calendar days for the window
//! plus the predictor's own `--min-periods`) before trusting its output.

use tradingflow::{
    data::{Array, ArrayView, Instant, Retention},
    expr::Context,
    graph::Builder,
    operators::{array, elem, rolling, stats},
    ports::ArrayPortHandle,
    time::UnixTime,
};

use super::common::{annualized, growth, parent_equity, ttm};
use crate::data::MarketData;

/// ~1 trading month / quarter / year, in ticks of the daily clock.
const M: usize = 21;
const Q: usize = 63;
const Y: usize = 252;
/// The `MOM` (`RSTR`) look-back and its most-recent-month exclusion lag.
const MOM_T: usize = 504;
const MOM_L: usize = 21;

/// Adjusted close-to-close daily log return — the same spelling as the
/// price-volume catalog, so the two hash-cons onto one subgraph.
const RET: &str = "Delta(Log($close), 1)";

/// The cap-weighted cross-sectional mean return, broadcast to every lane:
/// `Σ w·r / Σ w` over the lanes where both the return and the weight are
/// usable, `NaN` when none are (e.g. the very first tick, before the lagged
/// caps exist).
fn market_return_into(
    out: &mut Array<f64, 1>,
    ret: ArrayView<'_, f64, 1>,
    weight: ArrayView<'_, f64, 1>,
) {
    let mut sum = 0.0;
    let mut sum_w = 0.0;
    for (&r, &w) in ret.iter().zip(weight.iter()) {
        if r.is_finite() && w.is_finite() && w > 0.0 {
            sum += w * r;
            sum_w += w;
        }
    }
    let mkt = if sum_w > 0.0 { sum / sum_w } else { f64::NAN };
    out.data_mut().fill(mkt);
}

/// `NLSIZE` from the z-scored `SIZE`: the cube, cross-sectionally
/// orthogonalized to `z` by projecting out the `Σz⁴/Σz²` regression
/// component (both moments over the finite lanes).
fn nlsize_into(out: &mut Array<f64, 1>, z: ArrayView<'_, f64, 1>) {
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for &v in z.iter() {
        if v.is_finite() {
            sxy += v * v * v * v;
            sxx += v * v;
        }
    }
    let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    for (o, &v) in out.data_mut().iter_mut().zip(z.iter()) {
        *o = if v.is_finite() {
            v * v * v - slope * v
        } else {
            f64::NAN
        };
    }
}

/// Builds the style catalog, returning the descriptors as raw
/// `(name, value)` entries for the winsorize → z-score finalizer.
/// Registers `$turn` and `$mkt` on the shared context (overwriting an
/// existing `$turn` with an identically-spelled wire — already-lowered
/// expressions keep theirs).
pub fn build_features_barra(
    b: &mut Builder<Instant, UnixTime>,
    ctx: &mut Context<1>,
    m: &MarketData,
) -> Vec<(String, ArrayPortHandle<f64, 1>)> {
    let daily = m.daily;

    // ---- Extra context fields ----
    // Daily turnover, exactly the price-volume catalog's `$turn`.
    let turn = b.op(elem::div(), (m.volume, m.field("shares.circulating")));
    ctx.add_field("turn", turn);
    // `$mkt`: daily returns weighted by the previous trading day's
    // circulating cap, so the weights are known before the return realizes.
    let ret = ctx.lower_str(b, RET).expect("return lowering");
    let circ_cap = b.op(elem::mul(), (m.close, m.field("shares.circulating")));
    let cap_lag = b.op(rolling::lag(Retention::count(1)), (daily, circ_cap));
    let mkt = b.op(
        array::array_binary_map_inplace(
            |r, w| {
                let mut out = Array::zeros(r.extents());
                market_return_into(&mut out, r, w);
                out
            },
            market_return_into,
        ),
        (ret, cap_lag),
    );
    ctx.add_field("mkt", mkt);

    let mut styles: Vec<(String, ArrayPortHandle<f64, 1>)> = Vec::new();

    // ---- Size ----
    let mc = b.op(elem::mul(), (m.close, m.field("shares.total")));
    let size = b.op(elem::ln(), mc);
    styles.push(("SIZE".to_owned(), size));

    // ---- Expression styles ----
    // `BETA` appears inside `HSIGMA` too; hash-consing shares the subgraph,
    // along with every `Var`/`Std` the two spellings have in common. The
    // `Greater(x, x * 0)` guard clamps a negative variance difference to `0`
    // while letting `NaN` stay missing (`Greater` lowers to `f64::max`,
    // which would swallow it against a plain `0`).
    let beta = format!("(Cov({RET}, $mkt, {Y}) / Var($mkt, {Y}))");
    let vdiff = format!("(Var({RET}, {Y}) - Power({beta}, 2) * Var($mkt, {Y}))");
    for (name, src) in [
        ("BETA", beta.clone()),
        (
            "MOM",
            format!("Ref(Log($close), {MOM_L}) - Ref(Log($close), {MOM_T})"),
        ),
        ("DASTD", format!("Std({RET}, {Y})")),
        (
            "CMRA",
            format!("Max(Log($close), {Y}) - Min(Log($close), {Y})"),
        ),
        ("HSIGMA", format!("Sqrt(Greater({vdiff}, {vdiff} * 0))")),
        ("STOM", format!("Log(Sum($turn, {M}))")),
        ("STOQ", format!("Log(Sum($turn, {Q}) / 3)")),
        ("STOA", format!("Log(Sum($turn, {Y}) / 12)")),
    ] {
        let wire = ctx
            .lower_str(b, &src)
            .unwrap_or_else(|e| panic!("{name}: {e}\n  {src}"));
        styles.push((name.to_string(), wire));
    }

    // ---- Non-linear size ----
    // The cube of the z-scored size, orthogonalized to it; the z-score here
    // is its own node (the finalizer standardizes only the final entries).
    let size_z = b.op(stats::standardize(), size);
    let nlsize = b.op(
        array::array_map_inplace(
            |z| {
                let mut out = Array::zeros(z.extents());
                nlsize_into(&mut out, z);
                out
            },
            nlsize_into,
        ),
        size_z,
    );
    styles.push(("NLSIZE".to_owned(), nlsize));

    // ---- Valuation, earnings yield and growth ----
    let be = parent_equity(b, m);
    styles.push(("BTOP".to_owned(), b.op(elem::div(), (be, mc))));
    let np = annualized(b, m, m.income_report_signals, "income_statement.profit");
    let np_ttm = ttm(b, daily, np);
    styles.push(("ETOP".to_owned(), b.op(elem::div(), (np_ttm, mc))));
    let ocf = annualized(
        b,
        m,
        m.cf_report_signals,
        "cash_flow_statement.change.operating",
    );
    let ocf_ttm = ttm(b, daily, ocf);
    styles.push(("CETOP".to_owned(), b.op(elem::div(), (ocf_ttm, mc))));
    let rev = annualized(
        b,
        m,
        m.income_report_signals,
        "income_statement.profit.operating.income.revenue",
    );
    let rev_ttm = ttm(b, daily, rev);
    styles.push(("EGRO".to_owned(), growth(b, daily, np_ttm)));
    styles.push(("SGRO".to_owned(), growth(b, daily, rev_ttm)));

    // ---- Leverage ----
    // Liabilities are stored credit-negative; long-term debt is the
    // non-current remainder `total − current`.
    let ta = m.field("balance_sheet.assets");
    let tl = b.op(elem::neg(), m.field("balance_sheet.liab"));
    let cl = b.op(elem::neg(), m.field("balance_sheet.liab.current"));
    let ncl = b.op(elem::sub(), (tl, cl));
    let mc_ncl = b.op(elem::add(), (mc, ncl));
    styles.push(("MLEV".to_owned(), b.op(elem::div(), (mc_ncl, mc))));
    styles.push(("DTOA".to_owned(), b.op(elem::div(), (tl, ta))));
    let be_ncl = b.op(elem::add(), (be, ncl));
    styles.push(("BLEV".to_owned(), b.op(elem::div(), (be_ncl, be))));

    styles
}
