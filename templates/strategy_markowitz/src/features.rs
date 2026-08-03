//! The cross-sectional feature catalog.
//!
//! Currently, every feature is finalized by [`rank_impute`]: cross-sectionally
//! percentile-ranked to `[0, 1]`, then missing values imputed with the
//! neutral median `0.5`. Imputation is considered as part of the feature's
//! definition. Because every feature is rank-transformed, `0.5` is the common
//! neutral fill, and the panel's scale is uniform regardless of how many
//! features join it.

use tradingflow::{
    data::{Duration, Instant, Retention, Schema},
    graph::{Builder, OperatorExt},
    operators::{array, elem, rolling, stats},
    ports::{ArrayPortHandle, SignalPortHandle},
    time::UnixTime,
};

use crate::data::MarketData;

/// Retention window lengths for month / quarter / year.
const MONTH: Retention = Retention::duration(Duration::from_days(30));
const QUARTER: Retention = Retention::duration(Duration::from_days(90));
const YEAR: Retention = Retention::duration(Duration::from_days(365));

/// The model-ready feature panel.
pub struct Features {
    /// Feature names, in column order.
    pub schema: Schema,
    /// The live `(num_stocks, num_features)` panel the predictors regress on.
    pub panel: ArrayPortHandle<f64, 2>,
}

/// Turns a raw feature value into its model-ready form: cross-sectionally
/// percentile-ranked to `[0, 1]`, then impute a missing value (a `NaN` rank)
/// with the neutral median `0.5`.
pub fn rank_impute(
    b: &mut Builder<Instant, UnixTime>,
    h: ArrayPortHandle<f64, 1>,
) -> ArrayPortHandle<f64, 1> {
    b.op(stats::percentile().then(elem::fill_nan(0.5)), h)
}

/// Trailing-twelve-month of an annualized flow: a 365-day rolling mean of the
/// annualized (effective-date-aligned) report stream, ingesting one sample
/// per report pulse.
pub fn ttm(
    b: &mut Builder<Instant, UnixTime>,
    reports: SignalPortHandle<0>,
    flow: ArrayPortHandle<f64, 1>,
) -> ArrayPortHandle<f64, 1> {
    b.op(rolling::mean(YEAR, 1), (reports, flow))
}

/// Year-over-year delta.
#[allow(dead_code)]
pub fn change(
    b: &mut Builder<Instant, UnixTime>,
    daily: SignalPortHandle<0>,
    level: ArrayPortHandle<f64, 1>,
) -> ArrayPortHandle<f64, 1> {
    b.op(rolling::diff(YEAR), (daily, level))
}

/// Year-over-year growth.
#[allow(dead_code)]
pub fn growth(
    b: &mut Builder<Instant, UnixTime>,
    daily: SignalPortHandle<0>,
    level: ArrayPortHandle<f64, 1>,
) -> ArrayPortHandle<f64, 1> {
    b.op(rolling::pct_change(YEAR), (daily, level))
}

/// Build the feature catalog and stack it into the model-ready `(N, F)` panel.
pub fn build_features(b: &mut Builder<Instant, UnixTime>, m: &MarketData) -> Features {
    let mut names = Vec::new();
    let mut raw = Vec::new();
    let mut add = |name: &str, h: ArrayPortHandle<f64, 1>| {
        names.push(name.to_string());
        raw.push(h);
    };
    let daily = m.daily;

    // Shared building blocks (computed once, reused by several features).
    let market_cap = b.op(elem::mul(), (m.close, m.total_shares));
    let np_ttm = ttm(b, m.income_reports, m.net_profit);

    // ---- Momentum & reversal ----
    let rev_1m = b.op(rolling::diff(MONTH), (daily, m.log_adj)); // trailing 1-month log return
    let mom_1y = b.op(rolling::diff(YEAR), (daily, m.log_adj));
    add("REV_1M", rev_1m);
    add("MOM_12M", b.op(elem::sub(), (mom_1y, rev_1m))); // 12-month minus most recent month

    // ---- Volatility ----
    add(
        "VOL_3M",
        b.op(rolling::std_dev(QUARTER, 0), (daily, m.log_return)),
    );

    // ---- Size ----
    add("LN_MC", b.op(elem::ln(), market_cap));

    // ---- Valuation ----
    add("BP_LR", b.op(elem::div(), (m.parent_equity, market_cap))); // 净资产 / 总市值
    add("EP_TTM", b.op(elem::div(), (np_ttm, market_cap))); // 净利润 TTM / 总市值

    // ---- Liquidity ----
    let turnover = b.op(elem::div(), (m.volume, m.circ_shares)); // daily 换手率
    add("TURN_1M", b.op(rolling::mean(MONTH, 0), (daily, turnover)));

    // Finalize each entry into its model-ready feature and stack the panel.
    let features: Vec<_> = raw.into_iter().map(|h| rank_impute(b, h)).collect();
    let panel = b.op(array::stack(1), &features[..]);

    Features {
        schema: Schema::new(names),
        panel,
    }
}
