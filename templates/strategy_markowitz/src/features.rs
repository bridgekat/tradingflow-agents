//! The cross-sectional feature catalog.
//!
//! Every feature is finalized by [`winsorize_impute`]: cross-sectionally
//! winsorized at the 1% tails, then missing values imputed with the
//! cross-sectional mean of the surviving values. A feature whose raw
//! cross-section is too heavy-tailed even winsorized makes a scale of its own
//! part of its definition instead — the basic catalog's report-derived
//! fundamentals rank themselves to `[0, 1]` percentiles, for which the
//! finalizer is benign: winsorization barely moves a uniform rank, and the
//! mean imputation lands at the neutral median `0.5`. The panel's scale is
//! not uniform across features, which the mean predictor absorbs: it
//! standardizes every feature from its pooled moments before fitting.
//!
//! Fundamental building blocks are derived here from the raw carried report
//! fields in [`MarketData::fields`]. The market data store assets and incomes
//! debit-positive, liabilities, equity and expenses credit-negative; formulas
//! negate them where a positive magnitude is wanted.

mod alpha101;
mod alpha158;
mod alpha360;
mod basic;
mod context;

use std::collections::HashMap;
use tradingflow::{
    data::{Array, ArrayView, Instant, Schema},
    graph::{Builder, OperatorExt},
    operators::{array, stats},
    ports::ArrayPortHandle,
    time::UnixTime,
};

use crate::data::MarketData;
use alpha101::{build_context_alpha101, build_features_alpha101};
use alpha158::build_features_alpha158;
use alpha360::build_features_alpha360;
use basic::build_features_basic;
use context::build_price_context;

/// The selectable feature sets (`--features`). Doc comments double as the
/// CLI help text.
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum FeatureSet {
    /// The hand-written baseline catalog (`build_features_basic`).
    Basic,
    /// The WorldQuant Alpha101 catalog (`build_features_alpha101`). Costly:
    /// thousands of graph nodes, many of them window-scanning.
    Alpha101,
    /// The Qlib Alpha158 catalog (`build_features_alpha158`). Costly:
    /// hundreds of graph nodes, many of them window-scanning.
    Alpha158,
    /// The Qlib Alpha360 catalog (`build_features_alpha360`): the trailing
    /// 60-day window of six daily series. Costly in memory: 354 lag nodes,
    /// each buffering its own window of the whole cross-section.
    Alpha360,
}

/// The model-ready feature panel.
pub struct Features {
    /// Feature names, in column order.
    pub schema: Schema,
    /// The live `(num_stocks, num_features)` panel the predictors regress on.
    pub panel: ArrayPortHandle<f64, 2>,
}

/// Turns a raw feature value into its model-ready form: cross-sectionally
/// winsorized at the 1% tails, then impute a missing value (a `NaN`) with the
/// cross-sectional mean of the winsorized values — the neutral fill on the
/// feature's own scale. A fully-missing cross-section stays `NaN`, so those
/// timestamps drop out of the predictor's fit and mask rather than entering
/// as a made-up constant.
pub fn winsorize_impute(
    b: &mut Builder<Instant, UnixTime>,
    h: ArrayPortHandle<f64, 1>,
) -> ArrayPortHandle<f64, 1> {
    // Winsorization clamps ±∞ to the finite tail bounds, so only `NaN` is
    // left to impute here.
    fn impute_into(out: &mut Array<f64, 1>, x: ArrayView<'_, f64, 1>) {
        let mut count = 0usize;
        let mut sum = 0.0;
        for &v in x.iter() {
            if v.is_finite() {
                count += 1;
                sum += v;
            }
        }
        let mean = sum / count as f64; // `NaN` when nothing is finite
        let out = out.data_mut();
        for (i, &v) in x.iter().enumerate() {
            out[i] = if v.is_nan() { mean } else { v };
        }
    }
    let impute = array::array_map_inplace(
        |x| {
            let mut out = Array::zeros(x.extents());
            impute_into(&mut out, x);
            out
        },
        impute_into,
    );
    b.op(stats::winsorize(0.01).then(impute), h)
}

/// Build the feature panel from the selected [`FeatureSet`]s, with every
/// entry finalized by [`winsorize_impute`] and stacked into the model-ready
/// `(N, F)` panel. `industries` tags each symbol for `IndNeutralize`.
pub fn build_features(
    b: &mut Builder<Instant, UnixTime>,
    m: &MarketData,
    sets: &[FeatureSet],
    industries: &[String],
) -> Features {
    let mut entries = Vec::new();
    if sets.contains(&FeatureSet::Basic) {
        entries.extend(build_features_basic(b, m));
    }

    // The ported catalogs lower through one shared `Context`, so their common
    // subexpressions hash-cons into single graph nodes. Alpha101's fields are
    // a superset of the price fields the other two read, so it supplies the
    // context whenever it is selected.
    let alpha101 = sets.contains(&FeatureSet::Alpha101);
    let alpha158 = sets.contains(&FeatureSet::Alpha158);
    let alpha360 = sets.contains(&FeatureSet::Alpha360);
    if alpha101 || alpha158 || alpha360 {
        let mut ctx = match alpha101 {
            true => build_context_alpha101(b, m, industries),
            false => build_price_context(b, m),
        };
        if alpha101 {
            entries.extend(build_features_alpha101(b, &mut ctx));
        }
        if alpha158 {
            entries.extend(build_features_alpha158(b, &mut ctx));
        }
        if alpha360 {
            entries.extend(build_features_alpha360(b, &mut ctx));
        }
    }
    assert!(!entries.is_empty(), "no features selected");

    // Qlib names four features the same way in both Alpha158 and Alpha360
    // (`OPEN0`, `HIGH0`, `LOW0`, `VWAP0`), for the same expression. Keep the
    // first of each, and check that the later one really did lower to the same
    // wire — so two genuinely different features sharing a name still fail
    // loudly rather than silently losing one.
    let mut seen: HashMap<String, ArrayPortHandle<f64, 1>> = HashMap::new();
    entries.retain(|(name, h)| match seen.insert(name.clone(), *h) {
        None => true,
        Some(first) => {
            assert_eq!(
                first.index(),
                h.index(),
                "two different features are both named {name:?}"
            );
            false
        }
    });

    // Finalize each entry into its model-ready feature and stack the panel.
    let (names, raw): (Vec<_>, Vec<_>) = entries.into_iter().unzip();
    let features: Vec<_> = raw.into_iter().map(|h| winsorize_impute(b, h)).collect();
    let panel = b.op(array::stack(1), &features[..]);

    Features {
        schema: Schema::new(names),
        panel,
    }
}
