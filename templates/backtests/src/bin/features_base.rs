//! Evaluation of feature predictive power.
//!
//! Each alpha feature is scored by its daily **information coefficient**: the
//! cross-sectional correlation between the feature value on one trading day
//! and the realized return of the next, which is the feature's edge over the
//! cross-sectional average on that day. The readout is the running sum of
//! those daily ICs — the cumulative IC curve — plus a summary line per feature;
//! see [`report::FeatureTable`] for how to read either.

use clap::Parser;
use indicatif::ProgressBar;
use tradingflow::{
    data::Instant,
    graph::{Builder, Pool},
    operators::{array, metric, rolling, series},
    time::UnixTime,
};

use backtests::{data, features, report, utils};

/// Evaluation of feature predictive power.
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

    /// First date to backtest (inclusive).
    #[arg(long, value_parser = utils::parse_date)]
    start: Option<Instant>,

    /// Last date to backtest (exclusive).
    #[arg(long, value_parser = utils::parse_date)]
    end: Option<Instant>,

    /// Alpha feature sets to backtest (repeatable or comma-separated).
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_values = ["cicc-fund,cicc-pv"]
    )]
    alpha_feature_sets: Vec<features::AlphaFeatureSet>,

    /// Use the rank (Spearman) IC instead of the Pearson IC.
    #[arg(long)]
    rank: bool,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // Load the complete symbol list with its industry tags.
    let data::SymbolList {
        symbols,
        industries,
    } = data::read_symbol_list(&format!("{}/symbol_list.parquet", args.data_dir));

    // Build the computation graph.
    let mut pool = Pool::new(std::thread::available_parallelism().unwrap().get());
    let mut b = Builder::new(UnixTime);

    // Market data: every panel column (via `data.fields`) plus derived fields.
    let m = data::build_market_data(&mut b, &args.data_dir, args.start, args.end, &symbols);

    // Trading day signal.
    let daily = m.daily;

    // Extract the alpha and risk feature panels from market data.
    let (alpha_features, _) =
        features::build_features(&mut b, &m, &args.alpha_feature_sets, &[], &industries);

    println!(
        "alpha features: {}",
        alpha_features.schema.labels().join(", ")
    );

    // Prediction targets.
    let returns = b.op(rolling::pct_change(1), (daily, m.adj_close));

    // Calculate the information coefficient of each alpha feature: the
    // feature is lagged a day so it is correlated against the return it
    // could have predicted, never the one it was computed from.
    let mut ic_series = Vec::new();
    for (i, name) in alpha_features.schema.labels().iter().enumerate() {
        let values = b.op(array::select_at(i, 1), alpha_features.panel);
        let lagged = b.op(rolling::lag(1), (daily, values));
        let ic = if args.rank {
            b.op(metric::feature::rank_ic(), (daily, lagged, returns))
        } else {
            b.op(metric::feature::ic(), (daily, lagged, returns))
        };
        ic_series.push((name.as_str(), b.op(series::record_all(), (daily, ic))));
    }

    // Run the event loop until all sources are exhausted, with a progress bar
    // over the estimated total row count.
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
