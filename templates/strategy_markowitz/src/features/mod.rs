//! The cross-sectional feature catalog: the alpha panel the mean predictor
//! regresses on, and the risk panel the covariance predictor reads as factor
//! exposures — both built by [`build_features`].
//!
//! Every alpha feature is finalized by [`winsorize_impute`]: cross-sectionally
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
//! The risk panel is assembled from the selected [`RiskFeatureSet`]s, plus
//! an always-on `COUNTRY` intercept column (constant `1`) — the factor
//! regression needs a market column whatever else is selected, and with
//! nothing else it degrades to a plain single-factor market model. Style
//! descriptors ([`barra`]) go through [`winsorize_impute`] and then a
//! cross-sectional z-score — the risk model reads the panel as exposures
//! directly, so the standardization that the mean predictor does internally
//! has to happen on the wire here (and an imputed value lands at the neutral
//! `0`); the dummy columns ([`industry`]) bypass the finalizer to stay `0/1`.
//!
//! Fundamental building blocks are derived here from the raw carried report
//! fields in [`MarketData::fields`]. The market data store assets and incomes
//! debit-positive, liabilities, equity and expenses credit-negative; formulas
//! negate them where a positive magnitude is wanted.

mod alpha101;
mod alpha158;
mod alpha360;
mod barra;
mod cicc_fund;
mod cicc_pv;
mod common;
mod context;
mod industry;

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
use barra::build_features_barra;
use cicc_fund::build_features_cicc_fund;
use cicc_pv::build_features_cicc_pv;
use context::build_price_context;
use industry::build_features_industry;

/// The selectable alpha feature sets (`--alpha-feature-sets`). Doc comments
/// double as the CLI help text.
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum AlphaFeatureSet {
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
    /// The CICC fundamental factor handbook catalog
    /// (`build_features_cicc_fund`), 基本面因子手册's implementable subset.
    /// Needs `--warmup-days` around 1000: a 365-day TTM under a 365-day lag,
    /// plus the predictors' min-periods, before the first prediction.
    CiccFund,
    /// The CICC price-volume factor handbook catalog
    /// (`build_features_cicc_pv`), 价量因子手册's OHLCV-derivable subset.
    /// Needs `--warmup-days` around 700: 252-tick windows plus the
    /// predictors' min-periods, before the first prediction.
    CiccPv,
}

/// The selectable risk feature sets (`--risk-feature-sets`). The risk panel
/// always carries the `COUNTRY` intercept on top of the selection. Doc
/// comments double as the CLI help text.
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum RiskFeatureSet {
    /// The BARRA CNE5-style descriptor catalog (`build_features_barra`).
    /// Needs `--warmup-days` around 1100: the 504-tick momentum window plus
    /// the predictors' min-periods, before the first prediction.
    Barra,
    /// Industry dummy exposures (`build_features_industry`): one static
    /// `0/1` column per industry tag of the symbol list. Mind the column
    /// count against small `--universe-size` values — the factor regression
    /// sees one column per industry.
    Industry,
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

/// Drops later entries whose name a catalog already used: Qlib names four
/// features identically in Alpha158 and Alpha360 (`OPEN0`, `HIGH0`, `LOW0`,
/// `VWAP0`) for the same expression, which hash-consing collapses to the same
/// wire; hand-wired nodes are not hash-consed, so catalogs sharing a formula
/// still differ in wires. Keep the first of each name, silently when the
/// wires are identical and with a warning otherwise — so an unintended
/// collision of genuinely different features is at least visible.
fn dedup_entries(entries: &mut Vec<(String, ArrayPortHandle<f64, 1>)>) {
    let mut seen: HashMap<String, ArrayPortHandle<f64, 1>> = HashMap::new();
    entries.retain(|(name, h)| match seen.insert(name.clone(), *h) {
        None => true,
        Some(first) => {
            if first.index() != h.index() {
                eprintln!("warning: duplicate feature {name:?}; keeping the first");
            }
            false
        }
    });
}

/// Build the two feature panels: the alpha panel from the selected
/// [`AlphaFeatureSet`]s, every entry finalized by [`winsorize_impute`], and
/// the risk panel from the selected [`RiskFeatureSet`]s plus the always-on
/// `COUNTRY` intercept, its style entries additionally z-scored (see the
/// [module docs](self)) — each stacked into its model-ready `(N, F)` panel.
/// `industries` tags each symbol for `IndNeutralize` and the risk panel's
/// industry dummies.
pub fn build_features(
    b: &mut Builder<Instant, UnixTime>,
    m: &MarketData,
    alpha_sets: &[AlphaFeatureSet],
    risk_sets: &[RiskFeatureSet],
    industries: &[String],
) -> (Features, Features) {
    let mut entries = Vec::new();
    if alpha_sets.contains(&AlphaFeatureSet::CiccFund) {
        entries.extend(build_features_cicc_fund(b, m));
    }

    // The expression catalogs — alpha and risk alike — lower through one
    // shared `Context`, so their common subexpressions hash-cons into single
    // graph nodes. Alpha101's fields are a superset of the price fields the
    // others read, so it supplies the context whenever it is selected.
    let alpha101 = alpha_sets.contains(&AlphaFeatureSet::Alpha101);
    let mut ctx = match alpha101 {
        true => build_context_alpha101(b, m, industries),
        false => build_price_context(b, m),
    };
    if alpha101 {
        entries.extend(build_features_alpha101(b, &mut ctx));
    }
    if alpha_sets.contains(&AlphaFeatureSet::Alpha158) {
        entries.extend(build_features_alpha158(b, &mut ctx));
    }
    if alpha_sets.contains(&AlphaFeatureSet::Alpha360) {
        entries.extend(build_features_alpha360(b, &mut ctx));
    }
    if alpha_sets.contains(&AlphaFeatureSet::CiccPv) {
        entries.extend(build_features_cicc_pv(b, &mut ctx, m));
    }
    assert!(!entries.is_empty(), "no alpha features selected");
    dedup_entries(&mut entries);

    // Finalize each entry into its model-ready feature and stack the panel.
    let (names, raw): (Vec<_>, Vec<_>) = entries.into_iter().unzip();
    let features: Vec<_> = raw.into_iter().map(|h| winsorize_impute(b, h)).collect();
    let panel = b.op(array::stack(1), &features[..]);
    let alpha = Features {
        schema: Schema::new(names),
        panel,
    };

    // The risk panel: the selected style entries winsorized and z-scored
    // into exposures, then the intercept and dummy columns as-is.
    let mut entries = Vec::new();
    if risk_sets.contains(&RiskFeatureSet::Barra) {
        entries.extend(build_features_barra(b, &mut ctx, m));
    }
    dedup_entries(&mut entries);
    let (mut names, raw): (Vec<String>, Vec<_>) = entries.into_iter().unzip();
    let mut features: Vec<_> = raw
        .into_iter()
        .map(|h| {
            let h = winsorize_impute(b, h);
            b.op(stats::standardize(), h)
        })
        .collect();
    names.push("COUNTRY".to_string());
    features.push(b.val(array::constant(Array::full([m.n], 1.0))));
    if risk_sets.contains(&RiskFeatureSet::Industry) {
        for (name, h) in build_features_industry(b, industries) {
            names.push(name);
            features.push(h);
        }
    }
    let panel = b.op(array::stack(1), &features[..]);
    let risk = Features {
        schema: Schema::new(names),
        panel,
    };

    (alpha, risk)
}
