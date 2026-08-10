//! A baseline factor model strategy.
//!
//! The strategy does the following:
//!
//! 1. Calculate selected features from the price and volume series.
//! 2. Feed the features and past returns into a mean predictor
//!    (the "alpha model") to obtain a cross-sectional vector of expected
//!    future returns.
//! 3. Feed the features and past returns into a covariance predictor
//!    (the "risk model") to obtain a cross-section of factor exposures,
//!    factor covariance matrix and specific variances. Together they define
//!    a full covariance matrix of future returns.
//! 4. Feed both the mean vector and covariance matrix components into a
//!    mean-variance portfolio optimizer to obtain a vector of weights.
//! 5. Simulate frictionless trading using the weights, output NAV curves
//!    and stats.
//!
//! Steps 4 and 5 repeat once per `--risk-aversion` value. The whole sweep
//! runs in a single pass over the data: the variants share the feature, alpha
//! and risk model nodes, and fork only at the optimizer. The optimizers are
//! run in parallel: they are scheduled on independent threads, and CVXPY
//! releases the Python GIL during backend solves (although it makes little
//! difference on the small example dataset).
//!
//! This backtest requires an embedded Python interpreter with NumPy, SciPy
//! and CVXPY. The build script automatically discovers the active virtual
//! environment, and `python::initialize()` sets up the interpreter to act
//! like the one in the discovered virtual environment. Therefore, it suffices
//! to install dependencies in the virtual environment, and run `cargo run`
//! with it active.
//!
//! The environment variable `OPENBLAS_NUM_THREADS=1` is required if NumPy
//! is linked against OpenBLAS (which is common).

use clap::Parser;
use indicatif::ProgressBar;
use pyo3::prelude::*;
use tradingflow::{
    data::{Duration, Instant},
    graph::{Builder, OperatorExt, Pool},
    operators::{elem, rolling, series, stats, trader},
    ports::{ArrayPort, SignalPort},
    python::{py_operator_module, py_params},
    sources::sync,
    time::UnixTime,
};

use backtests::{data, features, python, quotes, report, universe, utils};

/// The selectable mean predictors (`--alpha-model`).
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum AlphaModelKind {
    /// The incremental factor model (`alpha_models.factor_model`): regresses
    /// returns on the alpha feature panel cross-sectionally.
    FactorModel,
}

/// The selectable covariance predictors (`--risk-model`).
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum RiskModelKind {
    /// The incremental factor model (`risk_models.factor_model`): regresses
    /// returns on the risk feature panel cross-sectionally.
    FactorModel,
    /// The shrinkage estimator (`risk_models.shrinkage`): a sample
    /// covariance over the last `--risk-window` samples, shrunk towards its
    /// diagonal and truncated to its top `--risk-rank` eigenpairs; ignores
    /// the risk feature panel.
    Shrinkage,
}

/// The selectable portfolio optimizers (`--portfolio`).
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum PortfolioKind {
    /// The Markowitz mean-variance optimizer (`portfolio.markowitz_cvxpy`):
    /// maximizes `expected returns - risk aversion * variance of returns`.
    MarkowitzCvxpy,
}

/// A baseline factor model strategy.
#[derive(Parser)]
struct Args {
    /// Directory containing the data CSV files.
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/a_shares_crawler/panels"))]
    data_dir: String,

    /// Directory containing the Python operator modules.
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/python"))]
    python_ops_dir: String,

    /// Path of the NAV curve CSV to write: one column per sweep variant,
    /// plus the cap-weighted index.
    #[arg(long, default_value = "strategy.csv")]
    output: String,

    /// Portfolio weights below this threshold are trimmed from the weights
    /// CSV; `0` keeps every holding (only floating-point dust is dropped).
    #[arg(long, default_value_t = 0.005)]
    min_weight: f64,

    /// First date to trade (inclusive), e.g. 2018-01-01; also the anchor
    /// of the rebalance calendar. The default covers the whole history.
    #[arg(long, value_parser = utils::parse_date, default_value = "1990-01-01")]
    start: Instant,

    /// Last date to trade (exclusive).
    #[arg(long, value_parser = utils::parse_date)]
    end: Option<Instant>,

    /// Calendar days of data read before `--start` to warm up the rolling
    /// features, the TTM aggregates and the predictors' training windows.
    #[arg(long, default_value_t = 1000)]
    warmup_days: usize,

    /// Rebalance every this many calendar days. A rebalance falling between
    /// trading days executes at the next trading day's quotes.
    #[arg(long, default_value_t = 30)]
    rebalance_every: usize,

    /// Number of stocks in the tradable universe: the top by circulating
    /// market cap, re-selected at every rebalance; `0` selects the whole
    /// market. Also sizes the Markowitz solver and the predictors' active
    /// blocks.
    #[arg(long, default_value_t = 0)]
    universe_size: usize,

    /// Alpha feature sets (repeatable or comma-separated).
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_values = ["cicc-fund,cicc-pv"]
    )]
    alpha_feature_sets: Vec<features::AlphaFeatureSet>,

    /// Risk feature sets (repeatable or comma-separated); the risk panel
    /// always carries the COUNTRY intercept on top of the selection.
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_values = ["barra"]
    )]
    risk_feature_sets: Vec<features::RiskFeatureSet>,

    /// Sampling periods a stock needs before either model emits a prediction.
    #[arg(long, default_value_t = 120)]
    min_periods: usize,

    /// The mean predictor feeding the optimizer.
    #[arg(long, value_enum, default_value = "factor-model")]
    alpha_model: AlphaModelKind,

    /// Ridge regression regularization term for the alpha model.
    #[arg(long, default_value_t = 1.0)]
    ridge_l2: f64,

    /// Halflife (in number of trading days) of the exponential decay for
    /// the factor premiums in the alpha model.
    #[arg(long, default_value_t = 250.0)]
    premium_halflife: f64,

    /// The covariance predictor feeding the optimizer.
    #[arg(long, value_enum, default_value = "factor-model")]
    risk_model: RiskModelKind,

    /// Number of eigenpairs the shrinkage risk model keeps: its output
    /// factor rank. Clamped to the symbol count; unused by the factor model.
    #[arg(long, default_value_t = 20)]
    risk_rank: usize,

    /// Trading days of history the shrinkage risk model keeps: its moment
    /// estimates run over this rolling window (which truncates the halflife
    /// decays), and its memory is `2 × window × symbols` floats. Unused by
    /// the factor model.
    #[arg(long, default_value_t = 500)]
    risk_window: usize,

    /// Halflife (in number of trading days) of the exponential decay for
    /// the factor covariances in the risk model.
    #[arg(long, default_value_t = 250.0)]
    covariance_halflife: f64,

    /// Halflife (in number of trading days) of the exponential decay for
    /// the specific variances in the risk model.
    #[arg(long, default_value_t = 150.0)]
    specific_halflife: f64,

    /// The portfolio optimizer.
    #[arg(long, value_enum, default_value = "markowitz-cvxpy")]
    portfolio: PortfolioKind,

    /// Measure risk relative to the cap-weighted index, rather than in
    /// absolute terms.
    #[arg(long)]
    benchmark_relative: bool,

    /// Risk aversion parameter sweep: the optimizer maximizes
    /// `expected returns - risk aversion * variance of returns`.
    #[arg(long, value_delimiter = ',', default_value = "0.5,1,2,5,10,25,50,100")]
    risk_aversion: Vec<f64>,

    /// Initial cash.
    #[arg(long, default_value_t = 1_000_000.0)]
    initial_cash: f64,
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

    /// The alpha model module path, selected by `--alpha-model`.
    pub fn alpha_module(&self) -> &'static str {
        match self.alpha_model {
            AlphaModelKind::FactorModel => "alpha_models.factor_model",
        }
    }

    /// The risk model module path, selected by `--risk-model`.
    pub fn risk_module(&self) -> &'static str {
        match self.risk_model {
            RiskModelKind::FactorModel => "risk_models.factor_model",
            RiskModelKind::Shrinkage => "risk_models.shrinkage",
        }
    }

    /// The portfolio optimizer module path, selected by `--portfolio`.
    pub fn portfolio_module(&self) -> &'static str {
        match self.portfolio {
            PortfolioKind::MarkowitzCvxpy => "portfolio.markowitz_cvxpy",
        }
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
    println!("risk-aversion sweep: {:?}", args.risk_aversion);

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
    let share_divs = m.field("dividends.share");
    let cash_divs = m.field("dividends.cash");

    // Trading day signal.
    let daily = m.daily;

    // Rebalance calendar signal.
    let rebalance = b.source(sync::signal_iter(args.rebalance_instants().into_iter()));

    // Extract the alpha and risk feature panels from market data.
    let (alpha_features, risk_features) = features::build_features(
        &mut b,
        &m,
        &args.alpha_feature_sets,
        &args.risk_feature_sets,
        &industries,
    );

    println!(
        "alpha features: {}",
        alpha_features.schema.labels().join(", ")
    );
    println!(
        "risk features: {}",
        risk_features.schema.labels().join(", ")
    );
    println!("alpha model: {}", args.alpha_module());
    println!("risk model: {}", args.risk_module());
    println!("portfolio optimizer: {}", args.portfolio_module());

    // Cap-weighted top-`k` universe weights.
    let universe = universe::build_cap_weighted_universe(&mut b, &m, rebalance, k);

    // Synthetic quote book.
    let (flags, bids, asks) =
        quotes::build_quotes(&mut b, m.price_signals, m.field("prices.close"));

    // The cap-weighted index portfolio for baseline.
    let (_positions, _cash, index_nav) = b.op(
        trader::fixed::benchmark(true, args.initial_cash),
        (
            (daily, flags, bids, asks),
            (m.div_signals, share_divs, cash_divs),
            (rebalance, universe),
        ),
    );
    let index_nav_series = b.op(series::record_all(), (daily, index_nav));

    // Prediction targets.
    let returns = b.op(rolling::pct_change(1), (daily, m.adj_close));
    let returns_demeaned = b.op(stats::demean(), returns);

    // The mean predictor (alpha model).
    //
    // Prediction is updated per rebalance signal, while observing one sample
    // of alpha feature matrix and realized return vector per trading day.
    type AlphaInputs = (
        SignalPort<0>,     // sample (trading day) signal
        ArrayPort<f64, 2>, // raw alpha features
        ArrayPort<f64, 1>, // realized returns (demeaned)
        SignalPort<0>,     // rebalance signal
        ArrayPort<f64, 1>, // index weights (positive means in-universe)
    );
    type AlphaOutputs = ArrayPort<f64, 1>; // predicted returns (demeaned)
    let mu = b.op(
        py_operator_module::<AlphaInputs, AlphaOutputs>(
            args.alpha_module(),
            py_params(|d| {
                d.set_item("universe_size", k)?;
                d.set_item("target_offset", 1)?; // features[t] pairs with target[t + 1]
                d.set_item("min_periods", args.min_periods)?;
                d.set_item("ridge_l2", args.ridge_l2)?;
                d.set_item("premium_halflife", args.premium_halflife)?;
                Ok(())
            }),
        ),
        (
            daily,
            alpha_features.panel,
            returns_demeaned,
            rebalance,
            universe,
        ),
    );

    // The covariance predictor (risk model).
    //
    // Emits the three objects of the covariance matrix `Σ = X F Xᵀ + D`
    // on three ports, updated per rebalance signal, while observing one sample
    // of risk feature matrix and realized return vector per trading day.
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
    let (exposures, covariance, specific) = b.op(
        py_operator_module::<RiskInputs, RiskOutputs>(
            args.risk_module(),
            py_params(|d| {
                d.set_item("universe_size", k)?;
                d.set_item("target_offset", 1)?; // features[t] pairs with target[t + 1]
                d.set_item("min_periods", args.min_periods)?;
                d.set_item("covariance_halflife", args.covariance_halflife)?;
                d.set_item("specific_halflife", args.specific_halflife)?;
                if args.risk_model == RiskModelKind::Shrinkage {
                    d.set_item("rank", args.risk_rank)?;
                    d.set_item("window", args.risk_window)?;
                }
                Ok(())
            }),
        ),
        (daily, risk_features.panel, returns, rebalance, universe),
    );

    // The risk aversion parameter sweep.
    let mut variants = Vec::new();
    for risk_aversion in args.risk_aversion.iter() {
        // The portfolio optimizer.
        type PortfolioInputs = (
            SignalPort<0>,     // rebalance signal
            ArrayPort<f64, 1>, // index weights
            ArrayPort<f64, 1>, // predicted returns (demeaned)
            ArrayPort<f64, 2>, // risk factor exposures: X
            ArrayPort<f64, 2>, // factor covariance matrix: F
            ArrayPort<f64, 1>, // specific variances: diagonal D
        );
        type PortfolioOutputs = ArrayPort<f64, 1>; // portfolio weights
        let weights = b.op(
            py_operator_module::<PortfolioInputs, PortfolioOutputs>(
                args.portfolio_module(),
                py_params(|d| {
                    d.set_item("universe_size", k)?; // size the solver to the universe
                    d.set_item("benchmark_relative", args.benchmark_relative)?;
                    d.set_item("risk_aversion", risk_aversion)?;
                    d.set_item("long_only", true)?; // aim for long-only portfolios
                    d.set_item("full_position", true)?; // aim for fully invested portfolios
                    Ok(())
                }),
            ),
            (rebalance, universe, mu, exposures, covariance, specific),
        );

        // Limit weights to long-only and non-leveraged.
        let weights = b.op(
            elem::clampf(0.0, f64::INFINITY).then(stats::scale_down(1.0)),
            weights,
        );

        // Simulate frictionless trading using `weight`.
        let (_positions, _cash, nav) = b.op(
            trader::fixed::benchmark(true, args.initial_cash),
            (
                (daily, flags, bids, asks),
                (m.div_signals, share_divs, cash_divs),
                (rebalance, weights),
            ),
        );
        let nav_series = b.op(series::record_all(), (daily, nav));
        let book_series = b.op(series::record_all(), (rebalance, weights));
        variants.push((risk_aversion, nav_series, book_series));
    }

    // Run the event loop until all sources are exhausted, with a progress
    // bar over the estimated total row count.
    let mut g = b.build();
    let bar = ProgressBar::new(g.size_hint().unwrap_or(0) as u64);
    g.run(&mut pool, |g, _| bar.set_position(g.num_events() as u64))
        .await;
    bar.finish();

    // Print summary statistics per variant, write the wide NAV CSV and the
    // long-format weights CSV alongside it.
    let mut table = report::ReportTable::default();
    let mut books = report::WeightsTable::default();
    table.add(&g, "index", Some(args.start), index_nav_series);
    for (risk_aversion, nav, book) in variants {
        let label = format!("risk_aversion_{risk_aversion}");
        table.add(&g, label.as_str(), Some(args.start), nav);
        books.add(&g, label, &symbols, args.min_weight, book);
    }
    table.write(&args.output);
    books.write(&report::weights_path(&args.output));
}
