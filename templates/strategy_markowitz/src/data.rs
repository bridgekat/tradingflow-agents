//! The cross-sectional market panel data.
//!
//! Every panel source emits a `([N] signal, M × [N] value arrays)` stream:
//! signal element `i` pulses on the timestamps stock `i` has a row, and the
//! value arrays are *carried*: a cell with no row on a timestamp keeps its
//! last emitted value. The carried value arrays are in [`MarketData::fields`],
//! one cross-sectional array handle for each value column.
//!
//! The daily price fields also want *signaled-or-NaN* representation:
//! that day's value, or `NaN` where the stock did not trade.
//! The distinction matters downstream: a suspended stock has no quote today,
//! but may have a last known price. [`tradingflow::operators::signal::collect`]
//! constructs this representation. The signaled-or-NaN daily price fields and
//! other derived nodes are separate struct fields.
//!
//! Derived fundamentals — annualized report flows, TTM aggregates, parent
//! equity — are built in [`crate::features`] from these raw carried fields.

use arrow::array::{Array as _, StringArray};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::BTreeMap;
use std::fs::File;
use tradingflow::{
    data::{Axis, Instant, Schema},
    graph::Builder,
    operators::{elem, feature, rolling, signal},
    ports::{ArrayPortHandle, SignalPortHandle},
    sources::panel,
    time::UnixTime,
};

/// The symbol table read from `symbol_list.parquet`.
pub struct SymbolList {
    /// Stock symbols, sorted — the cross-section axis.
    pub symbols: Vec<String>,
    /// Industry tag per symbol, parallel to `symbols` (empty when missing).
    pub industries: Vec<String>,
}

/// Reads the symbol list (symbols and industry tags) from
/// `symbol_list.parquet`.
pub fn read_symbol_list(path: &str) -> SymbolList {
    let file = File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .and_then(|b| b.build())
        .unwrap_or_else(|e| panic!("{path}: {e}"));

    let mut symbols = Vec::new();
    let mut industries = Vec::new();
    for batch in reader {
        let batch = batch.unwrap_or_else(|e| panic!("{path}: {e}"));
        let read = |name: &str, out: &mut Vec<String>| {
            let column = batch
                .column_by_name(name)
                .unwrap_or_else(|| panic!("{path}: missing column {name:?}"));
            let column = cast(column.as_ref(), &DataType::Utf8)
                .unwrap_or_else(|e| panic!("{path}: column {name:?}: {e}"));
            let column = column.as_any().downcast_ref::<StringArray>().unwrap();
            out.extend((0..column.len()).map(|i| column.value(i).to_string()));
        };
        read("symbol", &mut symbols);
        read("industry", &mut industries);
    }
    SymbolList {
        symbols,
        industries,
    }
}

/// Lists the value columns of a Parquet panel file, in file order: every
/// numeric or boolean column of the schema.
fn read_value_columns(path: &str) -> Vec<String> {
    let file = File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let builder =
        ParquetRecordBatchReaderBuilder::try_new(file).unwrap_or_else(|e| panic!("{path}: {e}"));
    builder
        .schema()
        .fields()
        .iter()
        .filter(|f| f.data_type().is_numeric() || *f.data_type() == DataType::Boolean)
        .map(|f| f.name().clone())
        .collect()
}

/// Cross-sectional market panel data produced by [`build_market_data`].
///
/// [`fields`](Self::fields) holds every raw (carried) column of every panel
/// file; [`field`](Self::field) looks one up by name. The daily price fields
/// are additionally exposed signaled-or-NaN, along with the derived
/// adjusted-close returns.
#[allow(dead_code)]
pub struct MarketData {
    /// One pulse per trading day (when any stock has a row).
    pub daily: SignalPortHandle<0>,
    /// Per-stock signals: element `i` pulses on stock `i`'s trading days.
    pub price_signals: SignalPortHandle<1>,
    /// Per-stock dividend-event signals.
    pub div_signals: SignalPortHandle<1>,
    /// Per-stock equity structure change signals.
    pub equity_signals: SignalPortHandle<1>,
    /// Per-stock balance sheet report signals.
    pub balance_report_signals: SignalPortHandle<1>,
    /// Per-stock income statement report signals.
    pub income_report_signals: SignalPortHandle<1>,
    /// Per-stock cash flow statement report signals.
    pub cf_report_signals: SignalPortHandle<1>,

    /// Per-stock row signals of every panel kind, keyed by file stem
    /// (e.g. `balance_sheets`): element `i` pulses when stock `i` has a row.
    pub row_signals: BTreeMap<String, SignalPortHandle<1>>,
    /// Every value column of every panel file, as emitted by the sources
    /// (carried), keyed under the panel's field prefix (see [`PANELS`]).
    pub fields: BTreeMap<String, ArrayPortHandle<f64, 1>>,

    // Daily price fields (per-stock signaled-or-NaN).
    pub close: ArrayPortHandle<f64, 1>,
    pub open: ArrayPortHandle<f64, 1>,
    pub high: ArrayPortHandle<f64, 1>,
    pub low: ArrayPortHandle<f64, 1>,
    pub volume: ArrayPortHandle<f64, 1>,
    pub amount: ArrayPortHandle<f64, 1>,
    /// Cumulative forward-adjustment multipliers (carried, starts at `1`):
    /// `adjusted price = raw price × multiplier`, and adjusted volume divides.
    pub adj_multipliers: ArrayPortHandle<f64, 1>,
    /// Forward-adjusted close.
    pub adj_close: ArrayPortHandle<f64, 1>,
    /// Log forward-adjusted close — the source of returns.
    pub log_adj: ArrayPortHandle<f64, 1>,
    /// Daily log return of the adjusted close (`NaN` around non-trading days).
    pub log_return: ArrayPortHandle<f64, 1>,

    /// Number of symbols on the axis.
    pub n: usize,
}

impl MarketData {
    /// The raw carried column named `name`, panicking on an unknown name
    /// (print [`fields`](Self::fields)`.keys()` to list what is available).
    pub fn field(&self, name: &str) -> ArrayPortHandle<f64, 1> {
        *self
            .fields
            .get(name)
            .unwrap_or_else(|| panic!("no field {name:?} among the {} loaded", self.fields.len()))
    }
}

/// Loads the long-format Parquet panels and wire the cross-sectional fields.
pub fn build_market_data(
    b: &mut Builder<Instant, UnixTime>,
    data_dir: &str,
    data_start: Option<Instant>,
    data_end: Option<Instant>,
    symbols: &[String],
) -> MarketData {
    // Whole-market axis: the schema must cover every label its table carries.
    let schema = Schema::new(symbols);

    /// The panel files read, as `(file stem, field prefix)` pairs. Every value
    /// column of a panel keys into [`MarketData::fields`] under the prefix: a
    /// column the prefix already namespaces (`prices.close` under `prices`, or
    /// the root field named exactly the prefix) keys as-is, anything else gets
    /// it prepended (e.g. `income_statement.report_year`).
    const PANELS: [(&str, &str); 6] = [
        ("daily_prices", "prices"),
        ("dividends", "dividends"),
        ("equity_structures", "shares"),
        ("balance_sheets", "balance_sheet"),
        ("income_statements", "income_statement"),
        ("cash_flow_statements", "cash_flow_statement"),
        // Unused due to low data availability.
        // ("indirect_statements", "indirect_statement"),
    ];

    // One panel source per file, reading every value column.
    let mut fields = BTreeMap::new();
    let mut row_signals = BTreeMap::new();
    for (kind, prefix) in PANELS {
        let names = read_value_columns(&format!("{data_dir}/{kind}.parquet"));
        let (signals, values) = b.source(
            panel::parquet(
                format!("{data_dir}/{kind}.parquet"),
                "date",
                [("symbol".into(), Axis::Labeled(schema.clone()))],
                names.clone(),
            )
            .with_time_range(data_start, data_end),
        );
        row_signals.insert(kind.to_string(), signals);
        for (name, &handle) in names.iter().zip(values.iter()) {
            let key = match name.strip_prefix(prefix) {
                Some(rest) if rest.is_empty() || rest.starts_with('.') => name.clone(),
                _ => format!("{prefix}.{name}"),
            };
            let prev = fields.insert(key, handle);
            assert!(
                prev.is_none(),
                "{data_dir}/{kind}.parquet: duplicate field key"
            );
        }
    }

    let field = |name: &str| -> ArrayPortHandle<f64, 1> {
        *fields
            .get(name)
            .unwrap_or_else(|| panic!("{data_dir}: no field {name:?} in the Parquet panels"))
    };
    let signals = |kind: &str| -> SignalPortHandle<1> {
        *row_signals
            .get(kind)
            .unwrap_or_else(|| panic!("{data_dir}: no panel file {kind}.parquet"))
    };
    let price_signals = signals("daily_prices");
    let div_signals = signals("dividends");
    let equity_signals = signals("equity_structures");
    let balance_report_signals = signals("balance_sheets");
    let income_report_signals = signals("income_statements");
    let cf_report_signals = signals("cash_flow_statements");

    // Batch-level cadences: any stock has a row on this timestamp.
    let daily = b.op(signal::any(), price_signals);

    // Daily price fields, converted to signaled-or-NaN representation.
    let collect = |b: &mut Builder<Instant, UnixTime>, carried| {
        b.op(signal::collect(), (price_signals, carried, daily))
    };
    let close = collect(b, field("prices.close"));
    let open = collect(b, field("prices.open"));
    let high = collect(b, field("prices.high"));
    let low = collect(b, field("prices.low"));
    let volume = collect(b, field("prices.volume"));
    let amount = collect(b, field("prices.amount"));

    // Forward adjustment for dividends, whole cross-section at once.
    let (adj_multipliers, adj_close) = b.op(
        feature::forward_adjust(),
        (
            (price_signals, close),
            (
                div_signals,
                field("dividends.share"),
                field("dividends.cash"),
            ),
        ),
    );
    let adj_close = collect(b, adj_close);
    let log_adj = b.op(elem::ln(), adj_close);
    let log_return = b.op(rolling::diff(1), (daily, log_adj));

    MarketData {
        daily,
        price_signals,
        div_signals,
        equity_signals,
        balance_report_signals,
        income_report_signals,
        cf_report_signals,
        row_signals,
        fields,
        close,
        open,
        high,
        low,
        volume,
        amount,
        adj_multipliers,
        adj_close,
        log_adj,
        log_return,
        n: symbols.len(),
    }
}
