//! The synthetic quote book.
//!
//! The Parquet panel data carries neither an order book nor listing state,
//! so both are approximated from the closing prices alone:
//!
//! * **Quotes.** A stock that traded today quotes `close ∓ tick`, a one-tick
//!   spread around the close, so a round trip pays `2 × tick` before fees.
//! * **Price limits.** A-shares trade inside a ±`limit` price interval around
//!   previous close (here the previous *known* close, so the interval survives
//!   suspensions), rounded to the tick size. A quote outside the interval
//!   cannot execute, so it becomes the infinity on its side: `ask = +inf` at
//!   limit-up, and `bid = -inf` at limit-down.
//! * **Suspension.** A stock with no closing price today (a `NaN` price) has
//!   no quote at all: `(-inf, +inf)`, which holds the trader's mark at the
//!   last real price instead of writing the position down.
//! * **Delisting.** Lacking a listing-state column, a stock quoteless for
//!   more than `delist_days` consecutive trading days is treated as delisted
//!   (`flag = false`), which marks it to zero — **excluding it from NAV** —
//!   and blocks trading in it. A later quote can relist it, so a long
//!   suspension that eventually resumes is written down and then written back
//!   up rather than stranding the shares.
//!
//! `NaN` never reaches the trader: every stock is quoted on every price
//! pulse, and a stock with nothing to quote says so with infinities.

use tradingflow::{
    data::{Array, ArrayView, Instant},
    graph::{Builder, Operator},
    ports::{ArrayPort, ArrayPortHandle, SignalPort, SignalPortHandle},
    time::UnixTime,
};

/// Daily price-limit range (±10% for most A-shares; the ±20% boards are not
/// distinguished, a documented approximation).
pub const PRICE_LIMIT: f64 = 0.10;
/// Half-spread of the synthetic quote book, in yuan (A-shares tick at 0.01).
pub const TICK_SIZE: f64 = 0.01;
/// A stock with `NaN` prices for more than this many consecutive trading
/// days counts as delisted (the panel carries no listing state).
pub const DELIST_DAYS: u32 = 252;

/// One-pass synthetic quote book over `(daily_signals, carried_close)`.
/// Set/stale is stated by the signal array (a per-element pulse on trading
/// days), not by `NaN` probing: where the signal is set the carried close is
/// that day's close.
struct SyntheticQuotes {
    limit: f64,
    tick: f64,
    delist_days: u32,
}

struct QuotesState {
    limit: f64,
    tick: f64,
    delist_days: u32,
    /// Last known close per stock (`NaN` before listing) — the range anchor.
    prev_close: Vec<f64>,
    /// Length of the current quoteless run, in trading days.
    run: Vec<u32>,
    flags: Array<bool, 1>,
    bids: Array<f64, 1>,
    asks: Array<f64, 1>,
}

impl Operator for SyntheticQuotes {
    type Inputs = (SignalPort<1>, ArrayPort<f64, 1>);
    type Outputs = (ArrayPort<bool, 1>, ArrayPort<f64, 1>, ArrayPort<f64, 1>);
    type Context = Instant;
    type State = QuotesState;

    fn init(self, (_, closes): (ArrayView<'_, bool, 1>, ArrayView<'_, f64, 1>)) -> Self::State {
        let n = closes.extents()[0];
        QuotesState {
            limit: self.limit,
            tick: self.tick,
            delist_days: self.delist_days,
            prev_close: vec![f64::NAN; n],
            run: vec![0; n],
            flags: Array::full([n], true),
            bids: Array::full([n], f64::NEG_INFINITY),
            asks: Array::full([n], f64::INFINITY),
        }
    }

    fn reset<'a, 'b: 'a>(
        _: (ArrayView<'a, bool, 1>, ArrayView<'a, f64, 1>),
        state: &'b mut Self::State,
    ) -> (
        ArrayView<'a, bool, 1>,
        ArrayView<'a, f64, 1>,
        ArrayView<'a, f64, 1>,
    ) {
        (state.flags.view(), state.bids.view(), state.asks.view())
    }

    fn compute<'a, 'b: 'a>(
        (signals, closes): (ArrayView<'a, bool, 1>, ArrayView<'a, f64, 1>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (
        ArrayView<'a, bool, 1>,
        ArrayView<'a, f64, 1>,
        ArrayView<'a, f64, 1>,
    ) {
        // Round to ticks, as the exchange does when determining the interval.
        let round = |x: f64| (x * 100.0).round() / 100.0;
        let flags = state.flags.data_mut();
        let bids = state.bids.data_mut();
        let asks = state.asks.data_mut();
        for (i, (&set, &close)) in signals.iter().zip(closes.iter()).enumerate() {
            if set && close.is_finite() {
                // The price interval is anchored on the last *known* close, so
                // a stock resuming after a suspension is intervaled against its
                // pre-suspension price. The first trading day has no limits.
                let prev = state.prev_close[i];
                let (lower, upper) = if prev.is_finite() {
                    (
                        round(prev * (1.0 - state.limit)),
                        round(prev * (1.0 + state.limit)),
                    )
                } else {
                    (f64::NEG_INFINITY, f64::INFINITY)
                };
                let (bid, ask) = (close - state.tick, close + state.tick);
                bids[i] = if bid < lower { f64::NEG_INFINITY } else { bid };
                asks[i] = if ask > upper { f64::INFINITY } else { ask };
                state.prev_close[i] = close;
                state.run[i] = 0;
            } else {
                bids[i] = f64::NEG_INFINITY;
                asks[i] = f64::INFINITY;
                state.run[i] = state.run[i].saturating_add(1);
            }
            flags[i] = state.run[i] <= state.delist_days;
        }
        (state.flags.view(), state.bids.view(), state.asks.view())
    }
}

/// Wires the synthetic quote book: `(flags, bids, asks)`, refreshed on the
/// daily pulse and retained in between.
pub fn build_quotes(
    b: &mut Builder<Instant, UnixTime>,
    daily_signals: SignalPortHandle<1>,
    close_carried: ArrayPortHandle<f64, 1>,
) -> (
    ArrayPortHandle<bool, 1>,
    ArrayPortHandle<f64, 1>,
    ArrayPortHandle<f64, 1>,
) {
    b.op(
        SyntheticQuotes {
            limit: PRICE_LIMIT,
            tick: TICK_SIZE,
            delist_days: DELIST_DAYS,
        },
        (daily_signals, close_carried),
    )
}
