use chrono::{DateTime, NaiveDate};
use clap::Parser;
use tradingflow::data::{Duration, Instant};

use crate::features::FeatureSet;

/// CLI arguments.
#[derive(Parser)]
pub struct Args {
    /// Directory containing the merged long-format Parquet tables.
    #[arg(long)]
    pub data_dir: String,
    /// Path of the NAV curve CSV to write (one column per variant).
    #[arg(long)]
    pub output: String,
    /// First date to backtest (inclusive), e.g. 2018-01-01; also the anchor
    /// of the rebalance calendar. The default covers the whole history.
    #[arg(long, value_parser = parse_date, default_value = "1990-01-01")]
    pub start: Instant,
    /// Last date to backtest (exclusive).
    #[arg(long, value_parser = parse_date)]
    pub end: Option<Instant>,
    /// Calendar days of data read before `--start` to warm up the rolling
    /// features, the TTM aggregates and the predictors' training windows.
    /// Trading (and the reported NAV) still begins at `--start`. The default
    /// covers the `basic` and alpha catalogs; the CICC catalogs' year-lagged
    /// features need ~700 (cicc-pv) to ~1000 (cicc-fund) days before the
    /// predictors see enough finite history to emit a first prediction.
    #[arg(long, default_value_t = 400)]
    pub warmup_days: i64,
    /// Rebalance the portfolios every this many calendar days. A rebalance
    /// falling between trading days executes at the next trading day's quotes.
    #[arg(long, default_value_t = 30)]
    pub rebalance_every: i64,
    /// Number of stocks in the tradable universe: the top by circulating
    /// market cap, re-selected at every rebalance; `0` selects the whole
    /// market. Also sizes the Markowitz solver and the predictors' active
    /// blocks.
    #[arg(long, default_value_t = 0)]
    pub universe_size: usize,
    /// Initial cash.
    #[arg(long, default_value_t = 1_000_000.0)]
    pub initial_cash: f64,
    /// Feature set(s) to build the panel from (repeatable or comma-separated).
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_values = ["basic"]
    )]
    pub features: Vec<FeatureSet>,
    /// Ridge regression L2 penalty.
    #[arg(long, default_value_t = 0.01)]
    pub ridge_alpha: f64,
    /// Markowitz risk-aversion coefficient(s); a comma-separated list sweeps
    /// one portfolio per value against shared predictors.
    #[arg(
        long,
        value_delimiter = ',',
        default_values = ["0.5", "1", "2", "5", "10", "25", "50", "100"]
    )]
    pub risk_aversion: Vec<f64>,
}

impl Args {
    /// The tradable universe size, parsed from the CLI value `n`,
    /// with `0` meaning the whole market.
    pub fn resolved_universe_size(&self, n: usize) -> usize {
        match self.universe_size {
            0 => n,
            k => k,
        }
    }

    /// Where the panel sources start reading: `--start` minus the warm-up.
    pub fn data_start(&self) -> Option<Instant> {
        Some(
            self.start
                .saturating_sub(Duration::from_days(self.warmup_days)),
        )
    }

    /// The rebalance calendar: every `--rebalance-every` calendar days from
    /// `--start` up to `--end` (or today when open-ended).
    pub fn rebalance_instants(&self) -> Vec<Instant> {
        assert!(
            self.rebalance_every > 0,
            "--rebalance-every must be positive"
        );
        let now = Instant::from_offset(Duration::from_nanos(
            chrono::Utc::now().timestamp_nanos_opt().unwrap(),
        ));
        let end = self.end.unwrap_or(now).min(now);
        let step = Duration::from_days(self.rebalance_every);
        let mut instants = Vec::new();
        let mut t = self.start;
        while t < end {
            instants.push(t);
            t = t.saturating_add(step);
        }
        instants
    }
}

/// Parses a `YYYY-MM-DD` date into its midnight [`Instant`].
pub fn parse_date(s: &str) -> Result<Instant, String> {
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
pub fn format_date(t: Instant) -> String {
    let dt = DateTime::from_timestamp_nanos(t.as_offset().as_nanos());
    dt.date_naive().format("%Y-%m-%d").to_string()
}
