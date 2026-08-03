//! Cap-weighted top-`k` universe.
//!
//! The universe is defined by its `[num_stocks]` weight vector: positive for
//! stocks in the universe (the predictors and the optimizer read "positive
//! means active"), zero otherwise. [`build_cap_weighted_universe`] selects the
//! top `k` stocks by circulating market cap, then weighs them proportionally to
//! cap, so the same vector doubles as the cap-weighted index weights: the
//! baseline the strategy variants compare against.
//!
//! Market cap is computed from the **signaled-or-NaN** closing prices, so a
//! stock that did not trade on the most recent trading day (suspended,
//! delisted, or not yet listed) has `NaN` market cap and drops out of the
//! selection. Note that the signaled-or-NaN closing prices still retain their
//! values outside of trading days, making it suitable for use here (the
//! rebalance signal may pulse on non-trading days).

use tradingflow::{
    data::{Array, ArrayView, Instant},
    graph::{Builder, Operator},
    operators::elem,
    ports::{ArrayPort, ArrayPortHandle, SignalPort, SignalPortHandle},
    time::UnixTime,
};

use crate::data::MarketData;

/// Market-cap-weighted top-`k` selection, recomputed per rebalance pulse.
struct CapWeightedTopK {
    k: usize,
}

struct CapWeightedTopKState {
    k: usize,
    order: Vec<usize>,
    out: Array<f64, 1>,
}

impl Operator for CapWeightedTopK {
    type Inputs = (SignalPort<0>, ArrayPort<f64, 1>);
    type Outputs = ArrayPort<f64, 1>;
    type Context = Instant;
    type State = CapWeightedTopKState;

    fn init(self, (_, caps): (ArrayView<'_, bool, 0>, ArrayView<'_, f64, 1>)) -> Self::State {
        CapWeightedTopKState {
            k: self.k,
            order: Vec::new(),
            out: Array::zeros(caps.extents()),
        }
    }

    fn reset<'a, 'b: 'a>(
        _: (ArrayView<'a, bool, 0>, ArrayView<'a, f64, 1>),
        state: &'b mut Self::State,
    ) -> ArrayView<'a, f64, 1> {
        state.out.view()
    }

    fn compute<'a, 'b: 'a>(
        (signal, caps): (ArrayView<'a, bool, 0>, ArrayView<'a, f64, 1>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> ArrayView<'a, f64, 1> {
        if *signal {
            let caps = caps.to_contiguous();
            let order = &mut state.order;
            order.clear();
            order.extend((0..caps.len()).filter(|&i| caps[i].is_finite() && caps[i] > 0.0));
            let out = state.out.data_mut();
            out.fill(0.0);
            let k = state.k.min(order.len());
            if k > 0 {
                // Top `k` by cap, descending; a full sort would also do, but
                // the selection only needs the split.
                order
                    .select_nth_unstable_by(k - 1, |&a, &b| caps[b].partial_cmp(&caps[a]).unwrap());
                let total: f64 = order[..k].iter().map(|&i| caps[i]).sum();
                if total > 0.0 {
                    for &i in &order[..k] {
                        out[i] = caps[i] / total;
                    }
                }
            }
        }
        state.out.view()
    }
}

/// The cap-weighted top-`k` universe over circulating market cap
/// (close × circulating shares), on the rebalance pulse.
pub fn build_cap_weighted_universe(
    b: &mut Builder<Instant, UnixTime>,
    m: &MarketData,
    rebalance: SignalPortHandle<0>,
    k: usize,
) -> ArrayPortHandle<f64, 1> {
    let circ_cap = b.op(elem::mul(), (m.close, m.circ_shares));
    b.op(CapWeightedTopK { k }, (rebalance, circ_cap))
}
