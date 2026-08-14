//! Evaluation of risk model (covariance forecast) quality.
//!
//! Each selected risk model variant runs exactly as it would inside the
//! strategy — same features, universe, rebalance calendar and warm-up — but
//! instead of feeding an optimizer, its prediction feeds the closed-form
//! GMV (global minimum variance) portfolio `portfolio.minimum_variance` and a
//! set of independent scoring operators. Every scored port passes through a
//! one-tick lag first, so a day is never scored against a prediction that
//! has already trained on its return. Three metrics per variant, all
//! independent of any alpha model:
//!
//! 1. **Log-likelihood** (`metric.log_likelihood`): the Gaussian
//!    log-density of each realized return cross-section under `N(0, Σ)`,
//!    per asset. Higher is better; both over- and under-forecast
//!    covariances lose.
//! 2. **Realized GMV volatility** (`metric::portfolio::period_return`): the
//!    GMV book `w = Σ⁻¹𝟙 / (𝟙ᵀΣ⁻¹𝟙)` is held between rebalances and its
//!    realized volatility measured. Lower is better — minimizing it is
//!    exactly what a covariance forecast is for.
//! 3. **Bias ratio** (`metric.portfolio_variance`): the standard deviation
//!    of the GMV returns divided by their predicted volatility `√(wᵀΣw)`.
//!    Ideal is 1: above 1 the model under-forecasts its own portfolio's
//!    risk, below 1 it over-forecasts.
//!
//! Four more columns describe the prediction and its book rather than score
//! them: the mean predicted correlation (`metric.mean_correlation`) and the
//! factor share of variance (`metric.factor_share`), and the GMV book's
//! breadth and short share of gross exposure (`metric::portfolio`).
//!
//! The metrics that read the `Σ = X F Xᵀ + D` triple are Python operators,
//! since that decomposition is a convention of these risk models rather than
//! anything TradingFlow knows about; the ones that read only weights and
//! returns are TradingFlow's own.
//!
//! The readout is one summary line per variant plus a wide CSV of the daily
//! metric series; see [`report::RiskModelTable`].
//!
//! This backtest requires an embedded Python interpreter with NumPy and
//! SciPy; see `strategy_base` for how the virtual environment is discovered.
//! The environment variable `OPENBLAS_NUM_THREADS=1` is required if NumPy
//! is linked against OpenBLAS (which is common).

use clap::Parser;
use indicatif::ProgressBar;
use pyo3::prelude::*;
use tradingflow::{
    data::{Duration, Instant},
    graph::{Builder, OperatorExt, Pool},
    operators::{elem, metric, rolling, series, signal},
    ports::{ArrayPort, SignalPort},
    python::{py_operator_module, py_params},
    sources::sync,
    time::UnixTime,
};

use backtests::{data, features, python, report, universe, utils};

/// The selectable covariance predictors (`--risk-models`).
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum RiskModelKind {
    /// The incremental factor model (`risk_models.factor_model`): regresses
    /// returns on the risk feature panel cross-sectionally.
    FactorModel,
    /// The shrinkage estimator (`risk_models.shrinkage`): a sample
    /// covariance over the last `--risk-window` samples, shrunk towards each
    /// of `--risk-shrinkage-targets` (one variant per target) and truncated
    /// to its top `--risk-rank` eigenpairs; ignores the risk feature panel.
    Shrinkage,
}

/// The selectable targets of the shrinkage risk model
/// (`--risk-shrinkage-targets`).
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum RiskShrinkageTargetKind {
    /// No shrinkage.
    None,
    /// The diagonal of the sample covariance: correlations shrink towards
    /// zero, variances are kept.
    Diagonal,
    /// Constant correlation (Ledoit-Wolf 2004): correlations shrink towards
    /// their cross-sectional mean, variances are kept.
    ConstantCorrelation,
    /// Common covariance (Schäfer-Strimmer target C): correlations and
    /// variances all shrink towards their cross-sectional means.
    CommonCovariance,
    /// Single index (Ledoit-Wolf 2003): correlations shrink towards a
    /// one-factor market model, with the market taken as the equal-weighted
    /// portfolio of standardized returns; variances are kept.
    SingleIndex,
}

impl RiskShrinkageTargetKind {
    /// The target name passed to the shrinkage risk model.
    fn name(self) -> &'static str {
        match self {
            RiskShrinkageTargetKind::None => "none",
            RiskShrinkageTargetKind::Diagonal => "diagonal",
            RiskShrinkageTargetKind::ConstantCorrelation => "constant-correlation",
            RiskShrinkageTargetKind::CommonCovariance => "common-covariance",
            RiskShrinkageTargetKind::SingleIndex => "single-index",
        }
    }
}

/// Evaluation of risk model (covariance forecast) quality.
#[derive(Parser)]
struct Args {
    /// Directory containing the data CSV files.
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/a_shares_crawler/panels"))]
    data_dir: String,

    /// Directory containing the Python operator modules.
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/python"))]
    python_ops_dir: String,

    /// Path of the daily metric series CSV to write: one column per metric
    /// per evaluated variant.
    #[arg(long, default_value = "risk_models.csv")]
    output: String,

    /// First date to evaluate (inclusive); also the anchor of the rebalance
    /// calendar. The default covers the whole history.
    #[arg(long, value_parser = utils::parse_date, default_value = "1990-01-01")]
    start: Instant,

    /// Last date to evaluate (exclusive).
    #[arg(long, value_parser = utils::parse_date)]
    end: Option<Instant>,

    /// Calendar days of data read before `--start` to warm up the rolling
    /// features and the predictors' training windows.
    #[arg(long, default_value_t = 1000)]
    warmup_days: usize,

    /// Refresh the predictions every this many calendar days, mirroring the
    /// strategy's rebalance calendar.
    #[arg(long, default_value_t = 30)]
    rebalance_every: usize,

    /// Number of stocks in the tradable universe: the top by circulating
    /// market cap, re-selected at every rebalance; `0` selects the whole
    /// market.
    #[arg(long, default_value_t = 0)]
    universe_size: usize,

    /// Risk feature sets (repeatable or comma-separated); the risk panel
    /// always carries the COUNTRY intercept on top of the selection.
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_values = ["barra"]
    )]
    risk_feature_sets: Vec<features::RiskFeatureSet>,

    /// Sampling periods a stock needs before a model emits a prediction.
    #[arg(long, default_value_t = 120)]
    min_periods: usize,

    /// The covariance predictors to evaluate (repeatable or
    /// comma-separated); `shrinkage` expands into one variant per
    /// `--risk-shrinkage-targets` entry.
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_values = ["factor-model,shrinkage"]
    )]
    risk_models: Vec<RiskModelKind>,

    /// Number of eigenpairs the shrinkage risk model keeps: its output
    /// factor rank. This truncation is itself a shrinkage (to a low-rank
    /// target, at intensity 1) and empirically the stronger one, so it
    /// largely determines how the shrinkage targets compare - keep it high
    /// when evaluating them. Sweepable (repeatable or comma-separated), one
    /// variant per entry. Clamped to the symbol count; unused by the factor
    /// model.
    #[arg(long, value_delimiter = ',', default_values_t = [20])]
    risk_rank: Vec<usize>,

    /// Trading days of history the shrinkage risk model keeps: its moment
    /// estimates run over this rolling window (which truncates the halflife
    /// decays). Sweepable (repeatable or comma-separated), one variant per
    /// entry. Unused by the factor model.
    #[arg(long, value_delimiter = ',', default_values_t = [500])]
    risk_window: Vec<usize>,

    /// The shrinkage targets to evaluate (repeatable or comma-separated),
    /// one shrinkage variant per entry. Unused by the factor model.
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_values = ["none,diagonal,constant-correlation,common-covariance,single-index"]
    )]
    risk_shrinkage_targets: Vec<RiskShrinkageTargetKind>,

    /// Halflife (in number of trading days) of the exponential decay for
    /// the factor covariances in the risk models. Sweepable (repeatable or
    /// comma-separated), one variant per entry.
    #[arg(long, value_delimiter = ',', default_values_t = [250.0])]
    covariance_halflife: Vec<f64>,

    /// Halflife (in number of trading days) of the exponential decay for
    /// the specific variances in the risk models. Sweepable (repeatable or
    /// comma-separated), one variant per entry.
    #[arg(long, value_delimiter = ',', default_values_t = [150.0])]
    specific_halflife: Vec<f64>,

    /// Print one diagnostic line per shrinkage model fit: the shrinkage
    /// intensity, the variance the kept eigenpairs explain, and the leading
    /// eigenvalues as annualized volatilities.
    #[arg(long)]
    verbose: bool,
}

/// One evaluated risk model configuration.
struct Variant {
    /// Column label, carrying whichever sweep dimensions vary.
    label: String,
    /// The Python operator module implementing the model.
    module: &'static str,
    /// The shrinkage target, for the shrinkage module only.
    target: Option<&'static str>,
    rank: usize,
    window: usize,
    covariance_halflife: f64,
    specific_halflife: f64,
}

impl Args {
    /// Where the panel sources start reading: `--start` minus the warm-up.
    pub fn data_start(&self) -> Option<Instant> {
        Some(
            self.start
                .saturating_sub(Duration::from_days(self.warmup_days as i64)),
        )
    }

    /// The rebalance calendar: every `--rebalance-every` calendar days from
    /// `--start` up to `--end` (or today when open-ended).
    pub fn rebalance_instants(&self) -> Vec<Instant> {
        let now = Instant::from_offset(Duration::from_nanos(
            chrono::Utc::now().timestamp_nanos_opt().unwrap(),
        ));
        let end = self.end.unwrap_or(now).min(now);
        let step = Duration::from_days(self.rebalance_every as i64);
        let mut instants = Vec::new();
        let mut t = self.start;
        while t < end {
            instants.push(t);
            t = t.saturating_add(step);
        }
        instants
    }

    /// The tradable universe size, parsed from the CLI value `n`,
    /// with `0` meaning the whole market.
    pub fn resolved_universe_size(&self, n: usize) -> usize {
        match self.universe_size {
            0 => n,
            k => k,
        }
    }

    /// The evaluated variants: the selected models crossed with every swept
    /// parameter. A sweep dimension appears in the label only when it has
    /// more than one value, so single-configuration runs keep short names.
    /// The factor model ignores the rank and window dimensions, so it is
    /// only crossed with the halflives.
    pub fn variants(&self) -> Vec<Variant> {
        let mut variants = Vec::new();
        for model in &self.risk_models {
            // `(base label, module, target)`; the sample baseline is the
            // shrinkage estimator with the `none` target, whose intensity is
            // 0 by construction.
            let bases: Vec<(String, &'static str, Option<&'static str>)> = match model {
                RiskModelKind::FactorModel => {
                    vec![("factor_model".to_string(), "risk_models.factor_model", None)]
                }
                RiskModelKind::Shrinkage => self
                    .risk_shrinkage_targets
                    .iter()
                    .map(|target| {
                        let name = target.name();
                        (
                            format!("shrinkage_{}", name.replace('-', "_")),
                            "risk_models.shrinkage",
                            Some(name),
                        )
                    })
                    .collect(),
            };

            for (base, module, target) in bases {
                // The factor model reads neither dimension: collapse them so
                // it contributes one variant per halflife pair, not a
                // duplicate per rank and window.
                let sizes: Vec<(usize, usize)> = match target.is_some() {
                    true => self
                        .risk_rank
                        .iter()
                        .flat_map(|&rank| self.risk_window.iter().map(move |&w| (rank, w)))
                        .collect(),
                    false => vec![(0, 0)],
                };
                for (rank, window) in sizes {
                    for &covariance_halflife in &self.covariance_halflife {
                        for &specific_halflife in &self.specific_halflife {
                            let mut label = base.clone();
                            if target.is_some() && self.risk_rank.len() > 1 {
                                label += &format!("_r{rank}");
                            }
                            if target.is_some() && self.risk_window.len() > 1 {
                                label += &format!("_w{window}");
                            }
                            if self.covariance_halflife.len() > 1 {
                                label += &format!("_ch{covariance_halflife}");
                            }
                            if self.specific_halflife.len() > 1 {
                                label += &format!("_sh{specific_halflife}");
                            }
                            variants.push(Variant {
                                label,
                                module,
                                target,
                                rank,
                                window,
                                covariance_halflife,
                                specific_halflife,
                            });
                        }
                    }
                }
            }
        }
        variants
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // Start the embedded interpreter as the repository's virtual environment,
    // with the strategy-local operators resolvable by `py_operator_module`.
    python::initialize(&args.python_ops_dir);

    // Print run parameters.
    println!("data dir: \"{}\"", args.data_dir);
    println!(
        "data start: {}",
        args.data_start()
            .map_or("(none)".to_string(), utils::format_date)
    );
    println!(
        "data end: {}",
        args.end.map_or("(none)".to_string(), utils::format_date)
    );
    println!("rebalance interval: {} calendar days", args.rebalance_every);

    // Load the complete symbol list with its industry tags.
    let data::SymbolList {
        symbols,
        industries,
    } = data::read_symbol_list(&format!("{}/symbol_list.parquet", args.data_dir));

    let k = args.resolved_universe_size(symbols.len());
    assert!(k >= 2, "the tradable universe needs at least 2 symbols");

    println!("symbol axis: {} symbols", symbols.len());
    println!("tradable universe: top {k} by market cap");

    // Build the computation graph.
    let mut pool = Pool::new(std::thread::available_parallelism().unwrap().get());
    let mut b = Builder::new(UnixTime);

    // Market data: every panel column (via `data.fields`) plus derived fields.
    let m = data::build_market_data(
        &mut b,
        &args.data_dir,
        args.data_start(),
        args.end,
        &symbols,
    );

    // Trading day signal.
    let daily = m.daily;

    // Rebalance calendar signal.
    let rebalance = b.source(sync::signal_iter(args.rebalance_instants().into_iter()));

    // Extract the risk feature panel from market data (no alpha features).
    let (_, risk_features) =
        features::build_features(&mut b, &m, &[], &args.risk_feature_sets, &industries);

    println!(
        "risk features: {}",
        risk_features.schema.labels().join(", ")
    );

    // Cap-weighted top-`k` universe weights.
    let universe = universe::build_cap_weighted_universe(&mut b, &m, rebalance, k);

    // Prediction targets.
    let returns = b.op(rolling::pct_change(1), (daily, m.adj_close));

    // The scored state advances on trading days and rebalances alike;
    // lagging a port one tick of this clock yields, at each trading day, the
    // latest value set strictly *before* it. Scoring only lagged ports is
    // what keeps a return from ever being scored by a prediction that has
    // already trained on it, or by a book rebalanced on it.
    let score = b.op(signal::or(), (daily, rebalance));
    let lag_universe = b.op(rolling::lag(1), (score, universe));

    // One risk model per variant plus its scoring operators; see
    // `strategy_base` for the risk model interface. The model's prediction
    // feeds its closed-form GMV book (`portfolio.minimum_variance`), and the
    // seven metric columns come from independent operators over the lagged
    // ports: the Gaussian log-likelihood (`metric.log_likelihood`), the
    // realized GMV return and its predicted variance, and the prediction
    // and book diagnostics from `tradingflow`'s `metric` operators.
    type RiskInputs = (
        SignalPort<0>,     // sample (trading day) signal
        ArrayPort<f64, 2>, // raw risk features
        ArrayPort<f64, 1>, // realized returns
        SignalPort<0>,     // rebalance signal
        ArrayPort<f64, 1>, // index weights (positive means in-universe)
    );
    type RiskOutputs = (
        ArrayPort<f64, 2>, // risk factor exposures: X
        ArrayPort<f64, 2>, // factor covariance matrix: F
        ArrayPort<f64, 1>, // specific variances: diagonal D
    );
    type GmvInputs = (
        SignalPort<0>,     // rebalance signal
        ArrayPort<f64, 1>, // index weights (positive means in-universe)
        ArrayPort<f64, 2>, // risk factor exposures: X
        ArrayPort<f64, 2>, // factor covariance matrix: F
        ArrayPort<f64, 1>, // specific variances: diagonal D
    );
    type GmvOutputs = ArrayPort<f64, 1>; // GMV weights
    type LogLikelihoodInputs = (
        SignalPort<0>,     // sample (trading day) signal
        ArrayPort<f64, 1>, // realized returns
        ArrayPort<f64, 1>, // lagged index weights
        ArrayPort<f64, 2>, // lagged risk factor exposures: X
        ArrayPort<f64, 2>, // lagged factor covariance matrix: F
        ArrayPort<f64, 1>, // lagged specific variances: diagonal D
    );
    // The prediction readouts share one shape: a cross-sectional vector —
    // the lagged universe for the diagnostics, the lagged GMV book for the
    // predicted variance — read against the lagged prediction.
    type ReadoutInputs = (
        SignalPort<0>,     // sample (trading day) signal
        ArrayPort<f64, 1>, // lagged index weights, or the lagged GMV book
        ArrayPort<f64, 2>, // lagged risk factor exposures: X
        ArrayPort<f64, 2>, // lagged factor covariance matrix: F
        ArrayPort<f64, 1>, // lagged specific variances: diagonal D
    );
    type MetricOutput = ArrayPort<f64, 0>; // the day's scalar metric

    let mut variants = Vec::new();
    for v in args.variants() {
        println!("risk model: {} ({})", v.label, v.module);
        let label = v.label.clone();
        let verbose = args.verbose;
        let (exposures, covariance, specific) = b.op(
            py_operator_module::<RiskInputs, RiskOutputs>(
                v.module,
                py_params(|d| {
                    d.set_item("universe_size", k)?;
                    d.set_item("target_offset", 1)?; // features[t] pairs with target[t + 1]
                    d.set_item("min_periods", args.min_periods)?;
                    d.set_item("covariance_halflife", v.covariance_halflife)?;
                    d.set_item("specific_halflife", v.specific_halflife)?;
                    if let Some(target) = v.target {
                        d.set_item("rank", v.rank)?;
                        d.set_item("window", v.window)?;
                        d.set_item("target", target)?;
                        if verbose {
                            d.set_item("label", label)?;
                        }
                    }
                    Ok(())
                }),
            ),
            (daily, risk_features.panel, returns, rebalance, universe),
        );

        // The closed-form GMV book of this variant's prediction.
        let weights = b.op(
            py_operator_module::<GmvInputs, GmvOutputs>("portfolio.minimum_variance", None),
            (rebalance, universe, exposures, covariance, specific),
        );

        // The scored state: everything one tick behind the scoring clock.
        // The book is filled before the first rebalance sets one — an absent
        // book is no positions, which the weight metrics require to be a
        // finite zero rather than a missing value.
        let x = b.op(rolling::lag(1), (score, exposures));
        let f = b.op(rolling::lag(1), (score, covariance));
        let d = b.op(rolling::lag(1), (score, specific));
        let w = b.op(rolling::lag(1).then(elem::fill_nan(0.0)), (score, weights));

        // Score each trading day against that state.
        let log_likelihood = b.op(
            py_operator_module::<LogLikelihoodInputs, MetricOutput>("metric.log_likelihood", None),
            (daily, returns, lag_universe, x, f, d),
        );
        let gmv_return = b.op(metric::portfolio::period_return(), (daily, w, returns));
        let gmv_variance = b.op(
            py_operator_module::<ReadoutInputs, MetricOutput>("metric.portfolio_variance", None),
            (daily, w, x, f, d),
        );
        let mean_correlation = b.op(
            py_operator_module::<ReadoutInputs, MetricOutput>("metric.mean_correlation", None),
            (daily, lag_universe, x, f, d),
        );
        let factor_share = b.op(
            py_operator_module::<ReadoutInputs, MetricOutput>("metric.factor_share", None),
            (daily, lag_universe, x, f, d),
        );
        let gmv_breadth = b.op(metric::portfolio::breadth(), (daily, w));

        // The short share of the book's gross exposure: `0` for a long-only
        // book, `NaN` for an empty one (both legs are zero).
        let short = b.op(metric::portfolio::short_exposure(), (daily, w));
        let gross = b.op(metric::portfolio::gross_exposure(), (daily, w));
        let gmv_short_share = b.op(elem::div(), (short, gross));

        // One recorded daily series per metric, in `RISK_MODEL_METRICS` order.
        let metric_series = [
            log_likelihood,
            gmv_return,
            gmv_variance,
            mean_correlation,
            factor_share,
            gmv_breadth,
            gmv_short_share,
        ]
        .map(|port| b.op(series::record_all(), (daily, port)));
        variants.push((v.label, metric_series));
    }

    // Run the event loop until all sources are exhausted, with a progress bar
    // over the estimated total row count.
    let mut g = b.build();
    let bar = ProgressBar::new(g.size_hint().unwrap_or(0) as u64);
    g.run(&mut pool, |g, _| bar.set_position(g.num_events() as u64))
        .await;
    bar.finish();

    // Print summary statistics per variant, write the wide metric CSV.
    let mut table = report::RiskModelTable::default();
    for (label, metric_series) in variants {
        table.add(&g, label, Some(args.start), metric_series);
    }
    table.print();
    table.write(&args.output);
}
