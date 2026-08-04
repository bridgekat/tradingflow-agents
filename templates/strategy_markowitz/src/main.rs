//! Markowitz mean-variance strategy over the full A-shares history, read from
//! the long-format Parquet panels written by
//! `data/a_shares_crawler/export_parquet.py`.
//!
//! The pipeline: [`data`] -> [`features`] -> [`universe`] -> predictors ->
//! Markowitz portfolio optimizers → frictionless traders against the synthetic
//! [`quotes`] book → NAV columns and summary statistics.
//!
//! # Design notes
//!
//! - The symbol axis covers the whole market. The traded subset is chosen
//!   through the `universe` vector, which the predictors and the optimizer
//!   read as "positive means in-universe". Sizing the solvers by
//!   `--universe-size` (not the axis) is what keeps the whole-market run fast.
//! - The variance predictor and the optimizers emit on the rebalance
//!   calendar, and the covariance matrix crosses the wire as a
//!   `[FACTOR_RANK + 1, N]` low-rank-plus-diagonal approximation (loadings +
//!   idiosyncratic std-dev) rather than the dense `[N, N]` matrix. The mean
//!   predictor samples daily, but refitting and predicting on the rebalance
//!   calendar.
//! - The three Python operators — the incremental ridge mean predictor, the
//!   shrinkage variance predictor and the Markowitz optimizer — are loaded
//!   from strategy-local, self-contained files. The `main` function appends
//!   the crate's `python/` directory to the embedded interpreter's `sys.path`,
//!   so that `py_operator_module` can resolve them like installed packages.
//!   This makes the whole modeling pipeline easier to modify and experiment
//!   with. The Markowitz solves release the GIL, so the sweep's variants solve
//!   in parallel on the pool.

mod args;
mod data;
mod features;
mod quotes;
mod report;
mod universe;

use clap::Parser;
use indicatif::ProgressBar;
use pyo3::prelude::*;
use tradingflow::{
    graph::{Builder, Pool},
    operators::{portfolio, predictor, series, stats, trader},
    python::{attach, py_operator_module, py_params},
    sources::sync,
    time::UnixTime,
};

use args::Args;
use data::{build_market_data, read_symbol_list};
use features::build_features;
use quotes::build_quotes;
use report::{ReportTable, WeightsTable, weights_path};
use universe::build_cap_weighted_universe;

/// Variance training window, in sampling (trading) days.
const COV_WINDOW: usize = 252;
/// Minimum valid observations for a stock to be considered predictable.
const MIN_PERIODS: usize = 63;
/// Rank of the factor-model covariance the Markowitz solve optimizes against.
const FACTOR_RANK: usize = 20;

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // Make `py_operator_module` resolve the strategy-local operators.
    let code = std::ffi::CString::new(format!(
        "import sys; sys.path.append({:?})",
        concat!(env!("CARGO_MANIFEST_DIR"), "/python"),
    ))
    .unwrap();
    attach(|py| py.run(&code, None, None).expect("cannot extend sys.path"));

    // Load the complete symbol list with its industry tags.
    let data::SymbolList {
        symbols,
        industries,
    } = read_symbol_list(&format!("{}/symbol_list.parquet", args.data_dir));
    let k = args.resolved_universe_size(symbols.len());
    assert!(k >= 2, "the tradable universe needs at least 2 symbols");
    println!(
        "symbol axis: {} symbols, tradable universe: top {k} by market cap, \
         risk-aversion sweep: {:?}",
        symbols.len(),
        args.risk_aversion,
    );

    // Build the computation graph.
    let mut pool = Pool::new(std::thread::available_parallelism().unwrap().get());
    let mut b = Builder::new(UnixTime);

    // Market data: every panel column (via `data.fields`) plus derived fields.
    let data = build_market_data(&mut b, &symbols, &args);
    let share_divs = data.field("dividends.share");
    let cash_divs = data.field("dividends.cash");

    // Trading day signal.
    let daily = data.daily;

    // Rebalance calendar signal.
    let rebalance = b.source(sync::signal_iter(args.rebalance_instants().into_iter()));

    // Extract features from market data.
    let features = build_features(&mut b, &data, &args.features, &industries);

    // Cap-weighted top-`k` universe weights.
    let universe = build_cap_weighted_universe(&mut b, &data, rebalance, k);

    // Synthetic quote book.
    let (flags, bids, asks) = build_quotes(&mut b, data.price_signals, data.field("prices.close"));

    // Log-returns as prediction target (raw and demeaned).
    let target = b.op(stats::winsorize(0.01), data.log_return);
    let demeaned = b.op(stats::demean(), target);

    println!("features: {}", features.schema.labels().join(", "));

    // Mean predictor: regresses the feature panel against next-day demeaned
    // log returns, adding samples daily and refitting per rebalance.
    // Masked and sized to the universe.
    let (_, predicted_returns) = b.op(
        py_operator_module::<predictor::mean::Inputs, predictor::mean::Outputs>(
            "mean",
            py_params(|d| {
                d.set_item("target_offset", 1)?; // features[t] predict target[t + 1]
                d.set_item("refit_every", 1)?; // refit on every rebalance pulse
                d.set_item("max_periods", None::<usize>)?; // fit on all history
                d.set_item("min_periods", MIN_PERIODS)?;
                d.set_item("universe_size", k)?;
                d.set_item("alpha", args.ridge_alpha)
            }),
        ),
        (daily, features.panel, demeaned, rebalance, universe),
    );

    // Variance predictor: re-estimates a shrunk sample covariance of raw
    // log returns over the trailing year, and emits it as a
    // `[FACTOR_RANK + 1, N]` low-rank-plus-diagonal approximation.
    // Masked and sized to the universe.
    let (_, predicted_cov) = b.op(
        py_operator_module::<predictor::variance::Inputs, predictor::variance::Outputs>(
            "variance",
            py_params(|d| {
                d.set_item("refit_every", 1)?;
                d.set_item("max_periods", COV_WINDOW)?;
                d.set_item("min_periods", MIN_PERIODS)?;
                d.set_item("universe_size", k)?;
                d.set_item("factor_rank", FACTOR_RANK)?; // must match the optimizer's
                d.set_item("target", predictor::variance::Target::SingleIndex as u8)
            }),
        ),
        (daily, features.panel, target, rebalance, universe),
    );

    // Parameter sweep: one Markowitz portfolio optimizer and one frictionless
    // trader per risk-aversion value, all sharing the predictors above.
    // Sweeping other parameters can be done in a similar fashion: build
    // several parallel paths with different parameters.
    let mut variants = Vec::new();
    for delta in args.risk_aversion.iter() {
        let (rebalance, weights) =
            b.op(
                py_operator_module::<
                    portfolio::mean_variance::Inputs,
                    portfolio::mean_variance::Outputs,
                >(
                    "portfolio",
                    py_params(|d| {
                        d.set_item("logarithmic", true)?; // moments are log returns
                        d.set_item("max_universe_size", k)?; // size the solver to the universe
                        d.set_item(
                            "mode",
                            portfolio::mean_variance::Mode::MinMeanVariance as u8,
                        )?;
                        d.set_item("bound", delta)?;
                        d.set_item("long_only", true)?; // produce no short or leveraged positions
                        d.set_item("full_position", false)?; // may hold cash
                        d.set_item("factor_rank", FACTOR_RANK) // matches the risk panel
                    }),
                ),
                (rebalance, universe, predicted_returns, predicted_cov),
            );
        let (_positions, _cash, nav) = b.op(
            trader::fixed::benchmark(true, args.initial_cash),
            (
                (daily, flags, bids, asks),
                (data.div_signals, share_divs, cash_divs),
                (rebalance, weights),
            ),
        );
        let nav_series = b.op(series::record_all(), (daily, nav));
        // Record the optimizer's target book on its rebalance pulse — one
        // `[N]` cross-section per rebalance — for the weights CSV.
        let book_series = b.op(series::record_all(), (rebalance, weights));
        variants.push((delta, nav_series, book_series));
    }

    // The cap-weighted index over the same universe, as the benchmark.
    let (_positions, _cash, nav) = b.op(
        trader::fixed::benchmark(true, args.initial_cash),
        (
            (daily, flags, bids, asks),
            (data.div_signals, share_divs, cash_divs),
            (rebalance, universe),
        ),
    );
    let index_nav_series = b.op(series::record_all(), (daily, nav));

    // Run the event loop until all sources are exhausted, with a progress
    // bar over the estimated total row count.
    let mut g = b.build();
    let bar = ProgressBar::new(g.size_hint().unwrap_or(0) as u64);
    g.run(&mut pool, |g, _| bar.set_position(g.num_events() as u64))
        .await;
    bar.finish();

    // Print summary statistics per variant, write the wide NAV CSV and the
    // long-format weights CSV alongside it.
    let mut table = ReportTable::default();
    let mut books = WeightsTable::default();
    table.add(&g, "index", Some(args.start), index_nav_series);
    for (delta, nav, book) in variants {
        let label = format!("delta_{delta}");
        table.add(&g, label.as_str(), Some(args.start), nav);
        books.add(&g, label, &symbols, args.min_weight, book);
    }
    table.write(&args.output);
    books.write(&weights_path(&args.output));
}
