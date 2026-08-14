use std::collections::HashMap;
use tradingflow::{
    data::{Array, Duration, Instant, Retention, Schema},
    graph::{Builder, OperatorExt},
    operators::{array, elem, feature, rolling, stats},
    ports::{ArrayPortHandle, SignalPortHandle},
    time::UnixTime,
};

use super::alpha101::{build_context_alpha101, build_features_alpha101};
use super::alpha158::build_features_alpha158;
use super::alpha360::build_features_alpha360;
use super::barra::build_features_barra;
use super::cicc_fund::build_features_cicc_fund;
use super::cicc_pv::build_features_cicc_pv;
use super::context::build_price_context;
use super::industry::build_features_industry;
use crate::data::MarketData;

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
    /// The Barra CNE5-style descriptor catalog (`build_features_barra`).
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

/// The one-year retention window of the TTM and year-over-year helpers.
pub const YEAR: Retention = Retention::duration(Duration::from_days(365));

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

/// The z-score a normalized feature is clamped to. Beyond about this far out
/// a cross-sectional value is a data artifact rather than an opinion, and
/// what matters is bounding its leverage over the models' normal equations
/// rather than preserving its magnitude. Roughly the Barra convention.
const Z_LIMIT: f64 = 5.0;

/// Turns a raw feature value into its exposure form: cross-sectionally
/// winsorized at the 1% tails, cross-sectionally z-scored to mean `0` and
/// standard deviation `1` over the values present, and clamped to ±[`Z_LIMIT`]
/// - the normalized scale both models require (see the [module docs](super)).
pub fn normalize(
    b: &mut Builder<Instant, UnixTime>,
    h: ArrayPortHandle<f64, 1>,
) -> ArrayPortHandle<f64, 1> {
    b.op(
        stats::winsorize(0.01)
            .then(stats::standardize_or_zero())
            .then(elem::clampf(-Z_LIMIT, Z_LIMIT)),
        h,
    )
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

/// Build the two feature panels — the alpha panel from the selected
/// [`AlphaFeatureSet`]s and the risk panel from the selected
/// [`RiskFeatureSet`]s plus the always-on `COUNTRY` intercept — each entry
/// finalized by [`normalize`] and stacked into its model-ready `(N, F)`
/// panel. Only the risk panel's constant and dummy columns bypass the
/// finalizer, since a `0/1` indicator is already on a normalized scale.
///
/// `industries` tags each symbol for `IndNeutralize` and the risk panel's
/// industry dummies.
pub fn build_features(
    b: &mut Builder<Instant, UnixTime>,
    m: &MarketData,
    alpha_sets: &[AlphaFeatureSet],
    risk_sets: &[RiskFeatureSet],
    industries: &[String],
) -> (Features, Features) {
    // The expression catalogs — alpha and risk alike — lower through one
    // shared `Context`, so their common subexpressions hash-cons into
    // single graph nodes. Alpha101's fields are a superset of the price
    // fields the others read, so it supplies the context whenever it is
    // selected.
    let has_alpha101 = alpha_sets.contains(&AlphaFeatureSet::Alpha101);
    let mut ctx = match has_alpha101 {
        true => build_context_alpha101(b, m, industries),
        false => build_price_context(b, m),
    };

    // The alpha panel: the selected alpha entries, normalized.
    let mut entries = Vec::new();
    if alpha_sets.contains(&AlphaFeatureSet::CiccFund) {
        entries.extend(build_features_cicc_fund(b, m));
    }
    if has_alpha101 {
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
    dedup_entries(&mut entries);

    let (names, raw): (Vec<_>, Vec<_>) = entries.into_iter().unzip();
    let features: Vec<_> = raw.into_iter().map(|h| normalize(b, h)).collect();

    // `stack` needs at least one view; an empty selection (e.g. a backtest
    // that only evaluates risk models) still gets a well-formed `(N, 0)`
    // panel.
    let panel = match features.is_empty() {
        true => b.val(array::constant(Array::full([m.n, 0], 0.0))),
        false => b.op(array::stack(1), &features[..]),
    };
    let alpha = Features {
        schema: Schema::new(names),
        panel,
    };

    // The risk panel: the selected style entries normalized into exposures,
    // then the intercept and dummy columns as-is.
    let mut entries = Vec::new();
    if risk_sets.contains(&RiskFeatureSet::Barra) {
        entries.extend(build_features_barra(b, &mut ctx, m));
    }
    dedup_entries(&mut entries);

    let (mut names, raw): (Vec<String>, Vec<_>) = entries.into_iter().unzip();
    let mut features: Vec<_> = raw.into_iter().map(|h| normalize(b, h)).collect();

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
