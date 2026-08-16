//! The market data the scorer trades against.
//!
//! Deliberately a fraction of what a strategy loads: the scorer never computes
//! a feature, so it reads closing prices, dividends and share counts and
//! nothing else. Every column it does not read is a column a submission cannot
//! influence the score through.
//!
//! Panel sources emit a `([N] signal, [N] value array)` stream: signal element
//! `i` pulses on the timestamps stock `i` has a row, and the value arrays are
//! *carried*, so a cell with no row keeps its last value.
//!
//! The three panels are read differently, which matters for correctness rather
//! than speed. Prices and dividends are events, and only those inside the
//! scoring window (plus a warm-up, so the quote book has a previous close to
//! anchor its price limits on and its delisting counter has had time to
//! saturate) affect the result.
//!
//! Share counts are a *level*, so that panel is read with `prefill`: its
//! pre-window history folds into the cross-section at the window's start
//! rather than being cut away. Truncating it instead loses every stock whose
//! last issue or buyback predates the warm-up — several hundred of them here,
//! including some of the largest names in the market — which would silently
//! drop them out of the benchmark's market-cap ranking.

use arrow::array::{Array as _, StringArray};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use tradingflow::{
    data::{Axis, Instant, Schema},
    graph::Builder,
    operators::signal,
    ports::{ArrayPortHandle, SignalPortHandle},
    sources::panel,
    time::UnixTime,
};

/// Reads the sorted symbol list — the cross-section axis — from
/// `symbol_list.parquet`.
pub fn read_symbols(path: &str) -> Vec<String> {
    let file = File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .and_then(|b| b.build())
        .unwrap_or_else(|e| panic!("{path}: {e}"));

    let mut symbols = Vec::new();
    for batch in reader {
        let batch = batch.unwrap_or_else(|e| panic!("{path}: {e}"));
        let column = batch
            .column_by_name("symbol")
            .unwrap_or_else(|| panic!("{path}: missing column \"symbol\""));
        let column = cast(column.as_ref(), &DataType::Utf8)
            .unwrap_or_else(|e| panic!("{path}: column \"symbol\": {e}"));
        let column = column.as_any().downcast_ref::<StringArray>().unwrap();
        symbols.extend((0..column.len()).map(|i| column.value(i).to_string()));
    }
    symbols
}

/// The market data wired by [`build_market_data`].
pub struct MarketData {
    /// One pulse per trading day (when any stock has a row).
    pub daily: SignalPortHandle<0>,
    /// Per-stock signals: element `i` pulses on stock `i`'s trading days.
    pub price_signals: SignalPortHandle<1>,
    /// Per-stock dividend-event signals.
    pub div_signals: SignalPortHandle<1>,
    /// Closing prices, carried — what the quote book reads.
    pub close_carried: ArrayPortHandle<f64, 1>,
    /// Closing prices, signaled-or-NaN: that day's close, or `NaN` where the
    /// stock did not trade. The benchmark's market cap reads this, so a
    /// suspended or unlisted stock drops out of the ranking instead of
    /// ranking on a stale price.
    pub close: ArrayPortHandle<f64, 1>,
    /// Share dividends per share held.
    pub share_divs: ArrayPortHandle<f64, 1>,
    /// Cash dividends per share held.
    pub cash_divs: ArrayPortHandle<f64, 1>,
    /// Circulating share count, carried.
    pub circulating: ArrayPortHandle<f64, 1>,
}

/// Loads the Parquet panels and wires the cross-sectional fields.
pub fn build_market_data(
    b: &mut Builder<Instant, UnixTime>,
    data_dir: &str,
    data_start: Option<Instant>,
    data_end: Option<Instant>,
    symbols: &[String],
) -> MarketData {
    let schema = Schema::new(symbols);
    let axes = || [("symbol".to_string(), Axis::Labeled(schema.clone()))];

    let (price_signals, prices) = b.source(
        panel::parquet(
            format!("{data_dir}/daily_prices.parquet"),
            "date",
            axes(),
            vec!["prices.close".into()],
        )
        .with_time_range(data_start, data_end),
    );

    let (div_signals, divs) = b.source(
        panel::parquet(
            format!("{data_dir}/dividends.parquet"),
            "date",
            axes(),
            vec!["dividends.share".into(), "dividends.cash".into()],
        )
        .with_time_range(data_start, data_end),
    );

    // Prefilled: see the module docs on why this one carries its history in.
    let (_equity_signals, equity) = b.source(
        panel::parquet(
            format!("{data_dir}/equity_structures.parquet"),
            "date",
            axes(),
            vec!["shares.circulating".into()],
        )
        .with_time_range(data_start, data_end)
        .with_prefill(true),
    );

    let (close_carried, share_divs, cash_divs, circulating) =
        (prices[0], divs[0], divs[1], equity[0]);

    let daily = b.op(signal::any(), price_signals);
    let close = b.op(signal::collect(), (price_signals, close_carried, daily));

    MarketData {
        daily,
        price_signals,
        div_signals,
        close_carried,
        close,
        share_divs,
        cash_divs,
        circulating,
    }
}
