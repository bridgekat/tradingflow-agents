//! The cross-sectional feature catalog.
//!
//! Currently, every feature is finalized by [`rank_impute`]: cross-sectionally
//! percentile-ranked to `[0, 1]`, then missing values imputed with the
//! neutral median `0.5`. Imputation is considered as part of the feature's
//! definition. Because every feature is rank-transformed, `0.5` is the common
//! neutral fill, and the panel's scale is uniform regardless of how many
//! features join it.
//!
//! Fundamental building blocks are derived here from the raw carried report
//! fields in [`MarketData::fields`]. The market data store assets and incomes
//! debit-positive, liabilities, equity and expenses credit-negative; formulas
//! negate them where a positive magnitude is wanted.

mod alpha101;
mod basic;

use tradingflow::{
    data::{Instant, Schema},
    graph::{Builder, OperatorExt},
    operators::{array, elem, stats},
    ports::ArrayPortHandle,
    time::UnixTime,
};

use crate::data::MarketData;
use alpha101::{build_context_alpha101, build_features_alpha101};
use basic::build_features_basic;

/// The selectable feature sets (`--features`). Doc comments double as the
/// CLI help text.
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum FeatureSet {
    /// The hand-written baseline catalog (`build_features_basic`).
    Basic,
    /// The WorldQuant Alpha101 catalog (`build_features_alpha101`). Costly:
    /// thousands of graph nodes, many of them window-scanning.
    Alpha101,
}

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

/// Build the feature panel from the selected [`FeatureSet`]s, with every
/// entry finalized by [`rank_impute`] and stacked into the model-ready
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
    if sets.contains(&FeatureSet::Alpha101) {
        let mut ctx = build_context_alpha101(b, m, industries);
        entries.extend(build_features_alpha101(b, &mut ctx));
    }
    assert!(!entries.is_empty(), "no features selected");

    // Finalize each entry into its model-ready feature and stack the panel.
    let (names, raw): (Vec<_>, Vec<_>) = entries.into_iter().unzip();
    let features: Vec<_> = raw.into_iter().map(|h| rank_impute(b, h)).collect();
    let panel = b.op(array::stack(1), &features[..]);

    Features {
        schema: Schema::new(names),
        panel,
    }
}
