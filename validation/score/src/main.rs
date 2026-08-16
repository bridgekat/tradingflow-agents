//! The trusted scorer.
//!
//! Reads the weights a submission produced and simulates trading them, to
//! answer the only question validation asks: what would holding this book have
//! returned, net of costs?
//!
//! It loads no submitted code. Not the Rust features, not the Python models —
//! this crate does not even build `tradingflow`'s `python` feature, so an
//! embedded interpreter is not available to be reached. Its whole input
//! surface is a weights CSV, the market panels and its own command line. That
//! is what makes its output evidence rather than a claim: a submission decides
//! *what to hold*, and nothing about *how it is scored*.
//!
//! # The weights format
//!
//! One CSV per portfolio, read by [`sources::panel::csv`]:
//!
//! ```text
//! date,symbol,weight
//! 2020-01-02,000001.SZ,0.031
//! 2020-01-02,600519.SH,0.024
//! ```
//!
//! Rows must ascend by `date` (the reader bisects on byte offsets to find a
//! time range). A book is a complete cross-section: a symbol absent from a
//! date is held at zero, not carried over from the last date it appeared on,
//! and the residual `1 - Σw` is cash. The portfolio's label is the file stem.
//!
//! # What the scorer fixes, and why
//!
//! Every choice that determines the number rather than the strategy is the
//! scorer's, not the submission's:
//!
//! - **The scoring window** (`--start`, `--end`), so a book cannot select the
//!   slice that flatters it. Weights outside the window are read for warm-up
//!   and never traded.
//! - **Execution is delayed**: a book dated `D` fills at the *next* trading
//!   day's quotes. Filling at `D`'s own close would let a submission trade on
//!   a price it had already seen.
//! - **Costs** (`--fee-rate-buy`, `--fee-rate-sell`, `--fee-min`), charged as
//!   `max(|amount| * rate, min)` per fill.
//! - **Leverage** (`--max-gross`), checked against each book's `Σ|w|`.
//! - **The benchmark**: a cap-weighted top-`k` index on the scorer's own
//!   calendar, traded through the same costs, so both sides of the comparison
//!   are net of the same frictions and neither is chosen by the submission.

use clap::Parser;
use indicatif::ProgressBar;
use serde::Serialize;
use tradingflow::{
    data::{Axis, Duration, Instant, Schema},
    graph::{Builder, OperatorExt, Pool},
    operators::{elem, metric, reduce, series, signal, trader},
    sources::{panel, sync},
    time::UnixTime,
};

mod data;
mod quotes;
mod report;
mod universe;
mod utils;

use report::{Recorded, Report};

/// Scores submitted portfolio weights against market data.
#[derive(Parser)]
struct Args {
    /// Directory containing the Parquet market panels.
    #[arg(long)]
    data_dir: String,

    /// A submitted weights CSV (repeatable); the portfolio's label is the
    /// file stem.
    #[arg(long = "weights", required = true)]
    weights: Vec<String>,

    /// Path of the NAV curve CSV to write; the JSON summary is written as
    /// `<stem>_summary.json` beside it.
    #[arg(long, default_value = "score.csv")]
    output: String,

    /// First date to score (inclusive). Also anchors the benchmark calendar.
    #[arg(long, value_parser = utils::parse_date)]
    start: Instant,

    /// Last date to score (exclusive).
    #[arg(long, value_parser = utils::parse_date)]
    end: Option<Instant>,

    /// Calendar days of price history read before `--start`, to warm the
    /// quote book: the price limits need a previous close to anchor on, and
    /// the delisting heuristic needs to have seen a stock quoteless for
    /// [`quotes::DELIST_DAYS`] *trading* days before it writes it off. Too
    /// short a warm-up leaves long-suspended stocks looking live, which moves
    /// the benchmark's market-cap ranking.
    #[arg(long, default_value_t = 500)]
    warmup_days: usize,

    /// Rebalance interval of the benchmark index, in calendar days.
    #[arg(long, default_value_t = 30)]
    benchmark_rebalance_every: usize,

    /// Number of stocks in the benchmark index: the top by circulating
    /// market cap, re-selected at every benchmark rebalance.
    #[arg(long, default_value_t = 300)]
    benchmark_size: usize,

    /// Maximum gross exposure `Σ|w|` a book may ask for. Exceeding it fails
    /// the run.
    #[arg(long, default_value_t = 1.0)]
    max_gross: f64,

    /// Initial cash.
    #[arg(long, default_value_t = 1_000_000.0)]
    initial_cash: f64,

    /// Fee charged on a buy, as a fraction of the traded amount.
    #[arg(long, default_value_t = 0.0005)]
    fee_rate_buy: f64,

    /// Fee charged on a sell, as a fraction of the traded amount; the default
    /// adds the A-share stamp duty (0.1%) to the commission.
    #[arg(long, default_value_t = 0.0015)]
    fee_rate_sell: f64,

    /// Minimum fee per fill, in the currency of `--initial-cash`.
    #[arg(long, default_value_t = 0.0)]
    fee_min: f64,
}

impl Args {
    /// Where the panel sources start reading: `--start` minus the warm-up.
    fn data_start(&self) -> Option<Instant> {
        Some(
            self.start
                .saturating_sub(Duration::from_days(self.warmup_days as i64)),
        )
    }

    /// The benchmark's rebalance calendar.
    fn benchmark_instants(&self) -> Vec<Instant> {
        let now = Instant::from_offset(Duration::from_nanos(
            chrono::Utc::now().timestamp_nanos_opt().unwrap(),
        ));
        let end = self.end.unwrap_or(now).min(now);
        let step = Duration::from_days(self.benchmark_rebalance_every as i64);
        let mut instants = Vec::new();
        let mut t = self.start;
        while t < end {
            instants.push(t);
            t = t.saturating_add(step);
        }
        instants
    }

    /// The trader configuration, shared by every book and the benchmark so
    /// their NAV curves are comparable.
    fn trader_params(&self) -> trader::fixed::FractionalParams {
        trader::fixed::FractionalParams {
            // A book dated `D` fills at the next trading day's quotes.
            delayed: true,
            initial_cash: self.initial_cash,
            fee_base_buy: self.fee_min,
            fee_base_sell: self.fee_min,
            fee_rate_buy: self.fee_rate_buy,
            fee_rate_sell: self.fee_rate_sell,
        }
    }
}

/// The scorer's own configuration, recorded in the summary so a ledger entry
/// says what the number was measured under.
#[derive(Serialize)]
struct Summary<'a> {
    schema: u32,
    scorer: &'a str,
    data_dir: &'a str,
    weights: &'a [String],
    start: String,
    end: Option<String>,
    warmup_days: usize,
    benchmark_size: usize,
    benchmark_rebalance_every: usize,
    max_gross: f64,
    initial_cash: f64,
    fee_rate_buy: f64,
    fee_rate_sell: f64,
    fee_min: f64,
    portfolios: &'a [report::PortfolioStats],
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let symbols = data::read_symbols(&format!("{}/symbol_list.parquet", args.data_dir));
    let schema = Schema::new(&symbols);
    println!("data dir: \"{}\"", args.data_dir);
    println!("symbol axis: {} symbols", symbols.len());
    println!(
        "scoring window: {} to {}",
        utils::format_date(args.start),
        args.end.map_or("(open)".to_string(), utils::format_date),
    );
    println!(
        "fees: buy {:.3}%, sell {:.3}%, min {} per fill",
        args.fee_rate_buy * 100.0,
        args.fee_rate_sell * 100.0,
        args.fee_min,
    );

    let mut pool = Pool::new(std::thread::available_parallelism().unwrap().get());
    let mut b = Builder::new(UnixTime);

    let m = data::build_market_data(
        &mut b,
        &args.data_dir,
        args.data_start(),
        args.end,
        &symbols,
    );
    let daily = m.daily;
    let (flags, bids, asks) = quotes::build_quotes(&mut b, m.price_signals, m.close_carried);

    // Everything that gets traded, scored identically: the benchmark first so
    // it heads the report, then one entry per submitted book.
    let mut books: Vec<(String, _, _)> = Vec::new();

    let bench_rebalance = b.source(sync::signal_iter(args.benchmark_instants().into_iter()));
    let bench_weights =
        universe::build_cap_weighted_universe(&mut b, &m, bench_rebalance, args.benchmark_size);
    books.push((
        format!("index_top{}", args.benchmark_size),
        bench_rebalance,
        bench_weights,
    ));

    for path in &args.weights {
        let label = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_else(|| panic!("{path}: cannot derive a portfolio label from the filename"))
            .to_string();

        // The panel source carries values, so a name dropped from a book would
        // keep its previous weight. `collect` re-reads each rebalance as the
        // cross-section it actually is — this date's values, `NaN` elsewhere —
        // and the absences fill to zero.
        let (row_signals, values) = b.source(
            panel::csv(
                path.clone(),
                "date",
                [("symbol".to_string(), Axis::Labeled(schema.clone()))],
                vec!["weight".into()],
            )
            .with_time_range(None, args.end),
        );
        let carried = values[0];
        let rebalance = b.op(signal::any(), row_signals);
        let weights = b.op(
            signal::collect().then(elem::fill_nan(0.0)),
            (row_signals, carried, rebalance),
        );
        println!("weights: {label} <- {path}");
        books.push((label, rebalance, weights));
    }

    let mut recorded = Vec::new();
    for (label, rebalance, weights) in books {
        let (_positions, _cash, nav) = b.op(
            trader::fixed::fractional(args.trader_params()),
            (
                (daily, flags, bids, asks),
                (m.div_signals, m.share_divs, m.cash_divs),
                (rebalance, weights),
            ),
        );
        let turnover = b.op(metric::portfolio::turnover(), (rebalance, weights));
        let gross = b.op(elem::abs().then(reduce::sum_finite(0)), weights);

        recorded.push((
            label,
            Recorded {
                nav: b.op(series::record_all(), (daily, nav)),
                turnover: b.op(series::record_all(), (daily, turnover)),
                gross: b.op(series::record_all(), (daily, gross)),
            },
        ));
    }

    let mut g = b.build();
    let bar = ProgressBar::new(g.size_hint().unwrap_or(0) as u64);
    g.run(&mut pool, |g, _| bar.set_position(g.num_events() as u64))
        .await;
    bar.finish();

    let mut out = Report::default();
    for (label, r) in recorded {
        out.add(&g, label, Some(args.start), r);
    }
    out.write_nav(&args.output);

    let summary = Summary {
        schema: 1,
        scorer: "validation/score",
        data_dir: &args.data_dir,
        weights: &args.weights,
        start: utils::format_date(args.start),
        end: args.end.map(utils::format_date),
        warmup_days: args.warmup_days,
        benchmark_size: args.benchmark_size,
        benchmark_rebalance_every: args.benchmark_rebalance_every,
        max_gross: args.max_gross,
        initial_cash: args.initial_cash,
        fee_rate_buy: args.fee_rate_buy,
        fee_rate_sell: args.fee_rate_sell,
        fee_min: args.fee_min,
        portfolios: out.stats(),
    };
    let path = report::summary_path(&args.output);
    report::write_file(
        &path,
        &format!("{}\n", serde_json::to_string_pretty(&summary).unwrap()),
    );
    println!("wrote run summary to {path}");

    // Last, so an inadmissible book is still reported before it is rejected.
    out.enforce_gross(args.max_gross);
}
