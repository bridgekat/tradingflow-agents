//! Ridge + Markowitz strategy over the full A-shares history, read from the
//! long-format Parquet panels written by `templates/export_parquet.py`.
//!
//! Adapted from the `strategy_markowitz_panel` example in the `tradingflow`
//! crate. The differences:
//!
//! - The panel sources are `panel::parquet` over `daily_prices.parquet`,
//!   `dividends.parquet` and `equity_structures.parquet`, whose value columns
//!   keep the crawler's hierarchical names (`prices.close`, `dividends.cash`,
//!   `shares.total`, ...).
//! - The symbol axis covers the *whole* market, read from
//!   `symbol_list.parquet` — the panel source panics on labels missing from
//!   its schema, so the schema must span every symbol in the panels. The
//!   traded subset is instead chosen through the `universe` vector
//!   (`--symbols` and/or `--industry`), which the predictors and the
//!   optimizer read as "positive means active".
//! - The covariance predictor and the optimizer rebalance on a decimated
//!   (every `--rebalance-every` trading days) signal rather than daily: a
//!   covariance emission is a full `[N, N]` cross-section (N ≈ 5900, ~280 MB),
//!   so emitting one per day is unaffordable on the market-wide axis. The
//!   Ridge mean predictor still samples and predicts daily, refitting on the
//!   same monthly cadence as the original example.
//! - The three Python operators — the incremental Ridge mean predictor, the
//!   shrinkage covariance predictor and the Markowitz optimizer — are loaded
//!   from *strategy-local, self-contained* files (`python/ridge_incr.py`,
//!   `python/shrinkage.py`, `python/markowitz.py`, assembled from the
//!   corresponding `tradingflow` Python modules) instead of the installed
//!   `tradingflow` package: `main` appends the crate's `python/` directory to
//!   the embedded interpreter's `sys.path`, and `py_operator_module` then
//!   resolves them like installed packages. The whole modeling pipeline is
//!   thus local source code, open to modification without touching the
//!   framework.

use std::collections::HashMap;
use std::fs::File;

use arrow::array::{Array as _, StringArray};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use chrono::{DateTime, NaiveDate};
use clap::Parser;
use indicatif::ProgressBar;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use pyo3::prelude::*;

use tradingflow::{
    data::{ArrayView, Axis, Duration, Instant, Schema},
    graph::{Builder, Pool},
    operators::{
        array, elem, feature, metric, portfolio, predictor, rolling, series, signal, stats, trader,
    },
    python::{attach, py_operator_module, py_params},
    sources::panel,
    time::UnixTime,
};

/// The default universe when neither `--symbols` nor `--industry` is given:
/// the six symbols of the original example.
const DEFAULT_SYMBOLS: [&str; 6] = [
    "000001.SZ",
    "000002.SZ",
    "000858.SZ",
    "600000.SH",
    "600519.SH",
    "601398.SH",
];

/// Trading days per year, for annualizing the daily statistics.
const DAYS_PER_YEAR: f64 = 252.0;

/// Ridge + Markowitz strategy over real market history, read from Parquet panels.
#[derive(Parser)]
struct Args {
    /// Directory containing the merged long-format Parquet tables.
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/a_shares_crawler/panels"))]
    data_dir: String,
    /// Path of the NAV curve CSV to write.
    #[arg(long, default_value = "target/strategy_markowitz.csv")]
    output: String,
    /// Symbols to include in the tradable universe (comma-separated).
    #[arg(long, value_delimiter = ',')]
    symbols: Option<Vec<String>>,
    /// Include every symbol of this industry (exact match, e.g. 白酒) in the
    /// tradable universe. Unions with `--symbols` when both are given.
    #[arg(long)]
    industry: Option<String>,
    /// First date to backtest (inclusive), e.g. 2018-01-01.
    #[arg(long, value_parser = parse_date)]
    start: Option<Instant>,
    /// Last date to backtest (exclusive).
    #[arg(long, value_parser = parse_date)]
    end: Option<Instant>,
    /// Rebalance the portfolio every this many trading days.
    #[arg(long, default_value_t = 21)]
    rebalance_every: u64,
    /// Initial cash.
    #[arg(long, default_value_t = 1_000_000.0)]
    initial_cash: f64,
    /// Markowitz risk-aversion coefficient.
    #[arg(long, default_value_t = 25.0)]
    risk_aversion: f64,
    /// Ridge regression L2 penalty.
    #[arg(long, default_value_t = 0.01)]
    ridge_alpha: f64,
}

/// Parses a `YYYY-MM-DD` date into its midnight [`Instant`].
fn parse_date(s: &str) -> Result<Instant, String> {
    let date = NaiveDate::parse_from_str(s, "%Y-%m-%d").map_err(|e| e.to_string())?;
    let ns = date
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp_nanos_opt()
        .unwrap();
    Ok(Instant::from_offset(Duration::from_nanos(ns)))
}

/// Formats an [`Instant`] back into a `YYYY-MM-DD` date.
fn format_date(t: Instant) -> String {
    let dt = DateTime::from_timestamp_nanos(t.as_offset().as_nanos());
    dt.date_naive().format("%Y-%m-%d").to_string()
}

/// Reads `(symbols, industries)` from `symbol_list.parquet`, in file order.
fn read_symbol_list(path: &str) -> (Vec<String>, Vec<String>) {
    let file = File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .and_then(|b| b.build())
        .unwrap_or_else(|e| panic!("{path}: {e}"));

    let (mut symbols, mut industries) = (Vec::new(), Vec::new());
    for batch in reader {
        let batch = batch.unwrap_or_else(|e| panic!("{path}: {e}"));
        // `industry` is dictionary-encoded; cast both columns to plain UTF-8.
        let column = |name: &str| {
            let column = batch
                .column_by_name(name)
                .unwrap_or_else(|| panic!("{path}: missing column {name:?}"));
            cast(column.as_ref(), &DataType::Utf8)
                .unwrap_or_else(|e| panic!("{path}: column {name:?}: {e}"))
        };
        let (symbol, industry) = (column("symbol"), column("industry"));
        let symbol = symbol.as_any().downcast_ref::<StringArray>().unwrap();
        let industry = industry.as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..batch.num_rows() {
            symbols.push(symbol.value(i).to_string());
            industries.push(if industry.is_null(i) {
                String::new()
            } else {
                industry.value(i).to_string()
            });
        }
    }
    (symbols, industries)
}

/// Marks the tradable universe over the symbol axis from the CLI selection.
fn select_universe(args: &Args, symbols: &[String], industries: &[String]) -> Vec<bool> {
    let index: HashMap<&str, usize> = symbols
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let mut selected = vec![false; symbols.len()];

    if let Some(industry) = &args.industry {
        for (i, name) in industries.iter().enumerate() {
            if name == industry {
                selected[i] = true;
            }
        }
        assert!(
            selected.iter().any(|&s| s),
            "no symbol belongs to industry {industry:?}"
        );
    }
    let explicit = match (&args.symbols, &args.industry) {
        (Some(list), _) => list.clone(),
        (None, None) => DEFAULT_SYMBOLS.map(String::from).to_vec(),
        (None, Some(_)) => Vec::new(),
    };
    for symbol in &explicit {
        let i = index
            .get(symbol.as_str())
            .unwrap_or_else(|| panic!("symbol {symbol:?} is not in the symbol list"));
        selected[*i] = true;
    }
    selected
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let dir = &args.data_dir;

    // The symbol axis spans the whole market; the strategy trades `selected`.
    let (symbols, industries) = read_symbol_list(&format!("{dir}/symbol_list.parquet"));
    let selected = select_universe(&args, &symbols, &industries);
    let n = symbols.len();
    let k = selected.iter().filter(|&&s| s).count();
    assert!(k >= 2, "the tradable universe needs at least 2 symbols");
    println!("symbol axis: {n} symbols, tradable universe: {k}");

    // Strategy-local Python operators live in `python/` next to this crate;
    // put that directory on the embedded interpreter's module path, so
    // `py_operator_module` resolves them like any installed package. (The
    // venv's own `site-packages` is found automatically — the binary records
    // which interpreter PyO3 linked against and impersonates it at startup.)
    let code = std::ffi::CString::new(format!(
        "import sys; sys.path.append({:?})",
        concat!(env!("CARGO_MANIFEST_DIR"), "/python"),
    ))
    .unwrap();
    attach(|py| py.run(&code, None, None).expect("cannot extend sys.path"));

    // Create the thread pool.
    let mut pool = Pool::new(std::thread::available_parallelism().unwrap().get());

    // Build the graph. Each table is a long-format Parquet panel source over
    // the shared symbol axis: one signal array marking the rows present at
    // each date, plus one carried cross-section per value column.
    let mut b = Builder::new(UnixTime);
    let schema = Schema::new(&symbols);

    let (price_signals, prices) = b.source(
        panel::parquet(
            format!("{dir}/daily_prices.parquet"),
            "date",
            [("symbol".into(), Axis::Labeled(schema.clone()))],
            vec!["prices.close".into()],
        )
        .with_time_range(args.start, args.end),
    );
    let (div_signals, divs) = b.source(
        panel::parquet(
            format!("{dir}/dividends.parquet"),
            "date",
            [("symbol".into(), Axis::Labeled(schema.clone()))],
            vec!["dividends.share".into(), "dividends.cash".into()],
        )
        .with_time_range(args.start, args.end),
    );
    let (_equity_signals, equity) = b.source(
        panel::parquet(
            format!("{dir}/equity_structures.parquet"),
            "date",
            [("symbol".into(), Axis::Labeled(schema))],
            vec!["shares.total".into()],
        )
        .with_time_range(args.start, args.end),
    );
    let (close, share_divs, cash_divs) = (prices[0], divs[0], divs[1]);
    let total_shares = equity[0];

    // One scalar pulse per trading day (any symbol has a row), and a decimated
    // pulse every `rebalance_every`-th trading day. On the market-wide axis a
    // covariance emission is a full [N, N] cross-section, so the covariance
    // predictor and the optimizer run on the slower cadence.
    let daily = b.op(signal::any(), price_signals);
    let rebalance_every = args.rebalance_every;
    let mut ticks = 0u64;
    let monthly = b.op(
        signal::filter(move |_: ArrayView<'_, f64, 1>| {
            let fire = ticks % rebalance_every == 0;
            ticks += 1;
            fire
        }),
        (daily, close),
    );

    // Forward-adjust closes for dividends, then take daily log returns —
    // the prediction target, and an input to the volatility feature.
    let (_multipliers, adj_close) = b.op(
        feature::forward_adjust(),
        ((price_signals, close), (div_signals, share_divs, cash_divs)),
    );
    let log_price = b.op(elem::ln(), adj_close);
    let log_return = b.op(rolling::diff(1), (daily, log_price));

    // A few cross-sectional features, standardized across the cross-section:
    // 21-day momentum, 63-day volatility, and log market cap.
    let momentum = b.op(rolling::diff(21), (daily, log_price));
    let volatility = b.op(rolling::std_dev(63, 21), (daily, log_return));
    let market_cap = b.op(elem::mul(), (close, total_shares));
    let size = b.op(elem::ln(), market_cap);
    let f1 = b.op(stats::standardize(), momentum);
    let f2 = b.op(stats::standardize(), volatility);
    let f3 = b.op(stats::standardize(), size);
    let features = b.op(array::stack(1), &[f1, f2, f3][..]); // [N, 3] panel

    // Predict next-day moments: an incremental Ridge regression of the
    // features against next-day log returns for the means (sampling and
    // predicting daily, refitting on the rebalance cadence), and a shrunk
    // sample covariance for the risk (recomputed at each rebalance). Both are
    // strategy-local Python operators from `python/`; the
    // `py_params` dicts become the keyword arguments of their `build`.
    let universe = b.val(array::constant(
        selected
            .iter()
            .map(|&s| if s { 1.0 } else { 0.0 })
            .collect::<Vec<f64>>(),
    ));
    let (_, predicted_returns) = b.op(
        py_operator_module::<predictor::mean::Inputs, predictor::mean::Outputs>(
            "ridge_incr",
            py_params(|d| {
                d.set_item("target_offset", 1)?; // features[t] predict target[t + 1]
                d.set_item("refit_every", args.rebalance_every)?; // refit monthly
                d.set_item("max_periods", None::<usize>)?; // fit on all history
                d.set_item("min_periods", 63)?; // a quarter of coverage first
                d.set_item("alpha", args.ridge_alpha)
            }),
        ),
        (daily, features, log_return, daily, universe),
    );
    let (_, predicted_cov) = b.op(
        py_operator_module::<predictor::variance::Inputs, predictor::variance::Outputs>(
            "shrinkage",
            py_params(|d| {
                d.set_item("target_offset", 1)?;
                d.set_item("refit_every", 1)?; // every rebalance tick, i.e. monthly
                d.set_item("max_periods", 252)?; // one-year covariance window
                d.set_item("min_periods", 63)?;
                d.set_item("target", predictor::variance::Target::SingleIndex as u8)
            }),
        ),
        (daily, features, log_return, monthly, universe),
    );

    // The Markowitz optimizer trades predicted return against predicted risk:
    // maximize `μᵀx - δ·xᵀΣx` subject to long-only weights summing to at most
    // one. Its default `Config` maps the predictors' log-return moments to the
    // linear ones optimization needs; the solver is sized to the tradable
    // universe rather than the whole symbol axis.
    let (rebalance, weights) = b.op(
        py_operator_module::<portfolio::mean_variance::Inputs, portfolio::mean_variance::Outputs>(
            "markowitz",
            py_params(|d| {
                d.set_item("logarithmic", true)?; // moments are log returns
                d.set_item("max_universe_size", k)?; // size the solver to the universe
                d.set_item("mode", portfolio::mean_variance::Mode::MinMeanVariance as u8)?;
                d.set_item("bound", args.risk_aversion)?;
                d.set_item("long_only", true)?;
                d.set_item("full_position", false)?; // may hold cash
                d.set_item("factor_rank", (k - 1).min(20)) // covariance factor rank
            }),
        ),
        (monthly, universe, predicted_returns, predicted_cov),
    );

    // Frictionless trading at the last known close (assuming best bid = best
    // ask = close, so a suspended symbol trades at its carried price), with
    // real dividend events credited to the book. Only the selected universe
    // is tradable, and only once a symbol has a (carried) close — the trader
    // panics on a tradable NaN quote, and a symbol listed after the backtest
    // start has no price until its first row arrives.
    let selected_flags = b.val(array::constant(selected.clone()));
    let has_price = b.op(elem::is_finite(), close);
    let flags = b.op(elem::and(), (selected_flags, has_price));
    let (_positions, _cash, nav) = b.op(
        trader::fixed::benchmark(false, args.initial_cash),
        (
            (daily, flags, close, close),
            (div_signals, share_divs, cash_divs),
            (rebalance, weights),
        ),
    );

    // Record the NAV curve and its daily performance statistics.
    let nav_series = b.op(series::record_all(), (daily, nav));
    let comp_return = b.op(metric::performance::comp_return(), (daily, nav));
    let sharpe = b.op(metric::performance::return_sharpe(), (daily, nav));
    let vol = b.op(metric::performance::return_vol(), (daily, nav));

    // Run the event loop until all sources are exhausted, with a progress bar
    // over the estimated total row count.
    let mut g = b.build();
    let bar = ProgressBar::new(g.size_hint().unwrap_or(0) as u64);
    g.run(&mut pool, |g, _| bar.set_position(g.num_events() as u64))
        .await;
    bar.finish();

    // Log the NAV curve.
    let series = g.view(nav_series);
    let (instants, values) = (series.instants(), series.to_contiguous());
    let mut csv = String::from("date,nav\n");
    for (&t, &nav) in instants.iter().zip(values.iter()) {
        csv += &format!("{},{nav}\n", format_date(t));
    }
    if let Some(parent) = std::path::Path::new(&args.output).parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(&args.output, csv).unwrap();
    println!("wrote {} NAV points to {}", instants.len(), args.output);

    // Print summary statistics (daily metrics annualized).
    let final_nav = *values.last().unwrap();
    let years = (*instants.last().unwrap() - instants[0]).as_days() as f64 / 365.25;
    let mut peak = f64::MIN;
    let mdd = values.iter().fold(0.0f64, |mdd, &nav| {
        peak = peak.max(nav);
        mdd.min(nav / peak - 1.0)
    });
    println!("final NAV:         {final_nav:.0} ({years:.1} years)");
    println!(
        "total return:      {:+.2}%",
        (final_nav / args.initial_cash - 1.0) * 100.0
    );
    println!(
        "annual return:     {:+.2}%",
        ((1.0 + *g.view(comp_return)).powf(DAYS_PER_YEAR) - 1.0) * 100.0
    );
    println!(
        "annual volatility: {:.2}%",
        *g.view(vol) * DAYS_PER_YEAR.sqrt() * 100.0
    );
    println!(
        "annual Sharpe:     {:.3}",
        *g.view(sharpe) * DAYS_PER_YEAR.sqrt()
    );
    println!("max drawdown:      {:.2}%", mdd * 100.0);
}
