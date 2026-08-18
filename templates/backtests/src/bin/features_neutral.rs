//! Risk-neutralized evaluation of feature predictive power.
//!
//! Same measurement as [`features_base`](../features_base/index.html) — the
//! cross-sectional information coefficient of each alpha feature against the
//! forward `--horizon` return, inside the top-`--universe-size` tradable
//! cross-section — with one change: before the correlation, each feature is
//! **projected orthogonally to the risk model's exposure matrix** on the same
//! day (`screens.neutralize`).
//!
//! The point of doing that is the question a portfolio actually asks of a
//! feature. A mean-variance optimizer fed a covariance `Σ = B F Bᵀ + D` prices
//! every unit of exposure to the columns of `B`; a feature whose predictive
//! power lives in the span of `B` therefore reaches the book only as a factor
//! bet the same matrix charges for, and the optimizer hedges it away. What
//! survives the projection is what the optimizer can spend. A raw IC screen
//! cannot see the difference, which is why the largest in-universe ICs in this
//! search have belonged to features (turnover level, volatility, size) that
//! are Barra descriptors in disguise.
//!
//! `--neutralize` toggles the projection off, which reproduces
//! `features_base` exactly and gives the paired control arm.
//!
//! Unlike `features_base` this binary takes `--warmup-days`: the Barra panel's
//! `MOM` descriptor looks back 504 trading days, so a neutralized screen that
//! started reading data on the evaluation start date would have no risk matrix
//! (and so no residual) for the first two years of it. Data is read from
//! `--start` minus the warm-up; scoring the window itself is a matter of
//! slicing the emitted curve at `--start` (see `ic_neutral.py`).

use clap::Parser;
use indicatif::ProgressBar;
use tradingflow::{
    data::{Duration, Instant},
    graph::{Builder, OperatorExt, Pool},
    operators::{array, elem, metric, rolling, series},
    ports::{ArrayPort, SignalPort},
    python::{py_operator_module, py_params},
    time::UnixTime,
};

use backtests::{data, features, python, report, universe, utils};

/// Risk-neutralized evaluation of feature predictive power.
#[derive(Parser)]
struct Args {
    /// Directory containing the data CSV files.
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/a_shares_crawler/panels"))]
    data_dir: String,

    /// Directory containing the Python operator modules.
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/python"))]
    python_ops_dir: String,

    /// Path of the cumulative IC curve CSV to write.
    #[arg(long, default_value = "features.csv")]
    output: String,

    /// First date to score (inclusive). Data is read from this date minus
    /// `--warmup-days`; the curve carries the warm-up rows too, so slice it
    /// here when summarizing.
    #[arg(long, value_parser = utils::parse_date)]
    start: Option<Instant>,

    /// Last date to score (exclusive).
    #[arg(long, value_parser = utils::parse_date)]
    end: Option<Instant>,

    /// Calendar days of data read before `--start` to warm up the rolling
    /// features — the Barra `MOM` descriptor alone needs 504 trading days.
    #[arg(long, default_value_t = 1100)]
    warmup_days: usize,

    /// Alpha feature sets to score (repeatable or comma-separated).
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_values = ["cicc-fund,cicc-pv"]
    )]
    alpha_feature_sets: Vec<features::AlphaFeatureSet>,

    /// Risk feature sets the features are neutralized against; the panel
    /// always carries the COUNTRY intercept on top of the selection.
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_values = ["barra"]
    )]
    risk_feature_sets: Vec<features::RiskFeatureSet>,

    /// Project each feature orthogonally to the risk panel before scoring it.
    /// Off reproduces `features_base` on the same dates (the control arm).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    neutralize: bool,

    /// Use the rank (Spearman) IC instead of the Pearson IC.
    #[arg(long)]
    rank: bool,

    /// Forward-return horizon in trading days.
    #[arg(long, default_value_t = 21)]
    horizon: usize,

    /// Restrict the cross-section to the top-`k` stocks by circulating market
    /// cap, re-selected every trading day. The projection runs over the same
    /// cross-section: it must be the one the optimizer performs.
    #[arg(long, default_value_t = 300)]
    universe_size: usize,
}

impl Args {
    /// Where the panel sources start reading: `--start` minus the warm-up.
    fn data_start(&self) -> Option<Instant> {
        self.start
            .map(|s| s.saturating_sub(Duration::from_days(self.warmup_days as i64)))
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    python::initialize(&args.python_ops_dir);

    assert!(
        args.universe_size >= 2,
        "the neutralized screen needs a tradable cross-section"
    );

    // Load the complete symbol list with its industry tags.
    let data::SymbolList {
        symbols,
        industries,
    } = data::read_symbol_list(&format!("{}/symbol_list.parquet", args.data_dir));

    // Build the computation graph.
    let mut pool = Pool::new(std::thread::available_parallelism().unwrap().get());
    let mut b = Builder::new(UnixTime);

    // Market data: every panel column (via `data.fields`) plus derived fields.
    let m = data::build_market_data(&mut b, &args.data_dir, args.data_start(), args.end, &symbols);

    // Trading day signal.
    let daily = m.daily;

    // Extract the alpha and risk feature panels from market data.
    let (alpha_features, risk_features) = features::build_features(
        &mut b,
        &m,
        &args.alpha_feature_sets,
        &args.risk_feature_sets,
        &industries,
    );

    let alpha_labels = alpha_features.schema.labels();
    println!("alpha features: {}", alpha_labels.join(", "));
    println!("risk features: {}", risk_features.schema.labels().join(", "));
    println!("neutralize: {}", args.neutralize);
    println!(
        "data start: {}",
        args.data_start()
            .map_or("(none)".to_string(), utils::format_date)
    );

    // Prediction targets.
    let returns = b.op(rolling::pct_change(args.horizon), (daily, m.adj_close));

    // The tradable cross-section: top-`k` by circulating cap, re-selected
    // every trading day, and the 1.0/NaN mask derived from it.
    let univ = universe::build_cap_weighted_universe(&mut b, &m, daily, args.universe_size);
    let univ_mask = b.op(
        elem::fill_where(|&w: &f64| w <= 0.0, f64::NAN).then(elem::signum()),
        univ,
    );
    println!(
        "universe filter: top {} by circulating market cap (daily)",
        args.universe_size
    );

    // The neutralized alpha panel: each column, on each day, projected
    // orthogonally to the risk exposures over the tradable cross-section.
    type NeutralizeInputs = (
        SignalPort<0>,     // sample (trading day) signal
        ArrayPort<f64, 2>, // alpha features
        ArrayPort<f64, 2>, // risk features (the exposure matrix B)
        ArrayPort<f64, 1>, // universe weights (positive means in-universe)
    );
    type NeutralizeOutputs = ArrayPort<f64, 2>; // residual alpha features
    let panel = match args.neutralize {
        false => alpha_features.panel,
        true => b.op(
            py_operator_module::<NeutralizeInputs, NeutralizeOutputs>(
                "screens.neutralize",
                py_params(|_d| Ok(())),
            ),
            (daily, alpha_features.panel, risk_features.panel, univ),
        ),
    };

    // Calculate the information coefficient of each (residual) feature: the
    // feature is lagged the full horizon so it is correlated against the
    // return it could have predicted, never one it was computed from.
    let mut ic_series = Vec::new();
    for (i, name) in alpha_labels.iter().enumerate() {
        let values = b.op(array::select_at(i, 1), panel);
        let values = b.op(elem::mul(), (values, univ_mask));
        let lagged = b.op(rolling::lag(args.horizon), (daily, values));
        let ic = if args.rank {
            b.op(metric::feature::rank_ic(), (daily, lagged, returns))
        } else {
            b.op(metric::feature::ic(), (daily, lagged, returns))
        };
        ic_series.push((name.as_str(), b.op(series::record_all(), (daily, ic))));
    }

    // Run the event loop until all sources are exhausted.
    let mut g = b.build();
    let bar = ProgressBar::new(g.size_hint().unwrap_or(0) as u64);
    g.run(&mut pool, |g, _| bar.set_position(g.num_events() as u64))
        .await;
    bar.finish();

    // Print summary statistics per feature, write the wide cumulative IC CSV.
    let mut table = report::FeatureTable::default();
    for (name, ic) in ic_series {
        table.add(&g, name, ic);
    }
    table.print();
    table.write(&args.output);
}
