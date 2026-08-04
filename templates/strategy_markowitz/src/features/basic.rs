use tradingflow::{
    data::{Duration, Instant, Retention},
    graph::Builder,
    operators::{elem, feature, rolling},
    ports::{ArrayPortHandle, SignalPortHandle},
    time::UnixTime,
};

use crate::data::MarketData;

/// Retention window lengths for month / quarter / year.
const MONTH: Retention = Retention::duration(Duration::from_days(30));
const QUARTER: Retention = Retention::duration(Duration::from_days(90));
const YEAR: Retention = Retention::duration(Duration::from_days(365));

/// Annualized (YTD → per-year) report flow named `name`, gated per element by
/// the panel's per-stock report signals. The name's first segment is the
/// panel's field prefix (e.g. `income_statement`), which locates the
/// `report_year` / `report_day_of_year` metadata columns.
pub fn annualized(
    b: &mut Builder<Instant, UnixTime>,
    m: &MarketData,
    report_signals: SignalPortHandle<1>,
    name: &str,
) -> ArrayPortHandle<f64, 1> {
    let prefix = name.split('.').next().unwrap();
    let year = b.op(elem::as_(), m.field(&format!("{prefix}.report_year")));
    let doy = b.op(
        elem::as_(),
        m.field(&format!("{prefix}.report_day_of_year")),
    );
    b.op(
        feature::annualize(),
        (report_signals, m.field(name), year, doy),
    )
}

/// 归母净资产 (carried, positive): the balance-sheet equity components are
/// stored credit-negative, so their sum is negated.
pub fn parent_equity(
    b: &mut Builder<Instant, UnixTime>,
    m: &MarketData,
) -> ArrayPortHandle<f64, 1> {
    let capital_reserves = b.op(
        elem::add(),
        (
            m.field("balance_sheet.equity.capital"),
            m.field("balance_sheet.equity.reserves"),
        ),
    );
    let parent_sum = b.op(
        elem::add(),
        (
            capital_reserves,
            m.field("balance_sheet.equity.parent_interests"),
        ),
    );
    b.op(elem::neg(), parent_sum)
}

/// Trailing-twelve-month of an annualized flow: a 365-day rolling mean of the
/// annualized (effective-date-aligned) report stream.
pub fn ttm(
    b: &mut Builder<Instant, UnixTime>,
    daily: SignalPortHandle<0>,
    flow: ArrayPortHandle<f64, 1>,
) -> ArrayPortHandle<f64, 1> {
    b.op(rolling::mean(YEAR, 1), (daily, flow))
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

/// The baseline feature catalog: a small hand-written set of `(name, raw
/// value)` entries, one per feature, not yet rank-imputed.
pub fn build_features_basic(
    b: &mut Builder<Instant, UnixTime>,
    m: &MarketData,
) -> Vec<(String, ArrayPortHandle<f64, 1>)> {
    let mut features = Vec::new();
    let mut add = |name: &str, h: ArrayPortHandle<f64, 1>| {
        features.push((name.to_string(), h));
    };
    let daily = m.daily;

    // Shared building blocks (computed once, reused by several features).
    let market_cap = b.op(elem::mul(), (m.close, m.field("shares.total")));
    let net_profit = annualized(b, m, m.income_report_signals, "income_statement.profit");
    let np_ttm = ttm(b, m.daily, net_profit);
    let parent_equity = parent_equity(b, m);

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
    add("BP_LR", b.op(elem::div(), (parent_equity, market_cap))); // 净资产 / 总市值
    add("EP_TTM", b.op(elem::div(), (np_ttm, market_cap))); // 净利润 TTM / 总市值

    // ---- Liquidity ----
    let turnover = b.op(elem::div(), (m.volume, m.field("shares.circulating"))); // daily 换手率
    add("TURN_1M", b.op(rolling::mean(MONTH, 0), (daily, turnover)));

    features
}
