//! The price-field factor-expression context shared by the ported catalogs.
//!
//! [`build_price_context`] assembles a [`tradingflow::expr`] [`Context`] over
//! the whole-market panel, clocked by the daily pulse (rolling windows tick
//! once per trading day), holding the six daily series Qlib's `Alpha158` and
//! `Alpha360` read. All price-shaped fields are forward-adjusted for dividends
//! and splits with the same multipliers as [`MarketData::adj_close`], and
//! every field is signaled-or-NaN: `NaN` where a stock did not trade, which
//! the rolling operators skip and the comparisons treat as `false`.
//!
//! | Field | Meaning |
//! | --- | --- |
//! | `$open`, `$high`, `$low`, `$close` | adjusted daily prices |
//! | `$volume` | daily volume in shares, split-adjusted (divided by the multiplier) |
//! | `$vwap` | adjusted daily volume-weighted average price (`amount / volume`, adjusted) |
//!
//! The [Alpha101 context](super::alpha101) registers these same six fields,
//! wired identically, among the extra ones its own catalog needs (`$amount`,
//! `$returns`, `$cap`, the `$adv{d}` averages and the `$industry` grouping).
//! It is therefore built in place of this one whenever Alpha101 is selected,
//! so that every selected catalog lowers through a single [`Context`] and
//! hash-consing gives each common subexpression exactly one graph node — see
//! [`build_features`](super::build_features).

use tradingflow::{data::Instant, expr::Context, graph::Builder, operators::elem, time::UnixTime};

use crate::data::MarketData;

/// Builds the shared price-field [`Context`]. See the [module docs](self) for
/// the registered fields.
pub fn build_price_context(b: &mut Builder<Instant, UnixTime>, m: &MarketData) -> Context<1> {
    let mult = m.adj_multipliers;
    let adjust = |b: &mut Builder<Instant, UnixTime>, h| b.op(elem::mul(), (h, mult));
    let open = adjust(b, m.open);
    let high = adjust(b, m.high);
    let low = adjust(b, m.low);
    let volume = b.op(elem::div(), (m.volume, mult));
    let raw_vwap = b.op(elem::div(), (m.amount, m.volume));
    let vwap = adjust(b, raw_vwap);

    Context::new(m.daily, [m.n])
        .with_field("open", open)
        .with_field("high", high)
        .with_field("low", low)
        .with_field("close", m.adj_close)
        .with_field("volume", volume)
        .with_field("vwap", vwap)
}
