use tradingflow::{
    data::{Duration, Instant, Retention},
    graph::Builder,
    operators::{elem, feature, rolling},
    ports::{ArrayPortHandle, SignalPortHandle},
    time::UnixTime,
};

use crate::data::MarketData;

/// The one-year retention window of the TTM and year-over-year helpers.
pub const YEAR: Retention = Retention::duration(Duration::from_days(365));

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
pub fn change(
    b: &mut Builder<Instant, UnixTime>,
    daily: SignalPortHandle<0>,
    level: ArrayPortHandle<f64, 1>,
) -> ArrayPortHandle<f64, 1> {
    b.op(rolling::diff(YEAR), (daily, level))
}

/// Year-over-year growth.
pub fn growth(
    b: &mut Builder<Instant, UnixTime>,
    daily: SignalPortHandle<0>,
    level: ArrayPortHandle<f64, 1>,
) -> ArrayPortHandle<f64, 1> {
    b.op(rolling::pct_change(YEAR), (daily, level))
}
