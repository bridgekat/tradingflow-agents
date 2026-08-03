//! The cross-sectional market panel data.
//!
//! Every panel source emits a `([N] signal, M × [N] value arrays)` stream:
//! signal element `i` pulses on the timestamps stock `i` has a row, and the
//! value arrays are *carried*: a cell with no row on a timestamp keeps its
//! last emitted value. Fundamental fields are therefore point-in-time carried
//! state for free. The daily price fields instead want *signaled-or-NaN*
//! representation: that day's value, or `NaN` where the stock did not trade.
//! [`tradingflow::operators::signal::collect`] constructs this representation.
//! The distinction matters downstream: a suspended stock has no quote today,
//! but may have a last known price (carried).
//!
//! Income and cash-flow statements are annualized via [`feature::annualize`].
//! A trailing-twelve-month (TTM) figure is a 365-day rolling mean of the
//! annualized report stream (see [`ttm`](crate::features::ttm)). The Parquet
//! files store assets debit-positive and liabilities credit-negative;
//! factor formulas negate them where a positive magnitude is wanted.

use arrow::array::{Array as _, StringArray};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use tradingflow::{
    data::{Axis, Instant, Schema},
    graph::Builder,
    operators::{elem, feature, rolling, signal},
    ports::{ArrayPortHandle, SignalPortHandle},
    sources::panel,
    time::UnixTime,
};

use crate::args::Args;

/// Reads the symbol list from `symbol_list.parquet`.
pub fn read_symbol_list(path: &str) -> Vec<String> {
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
        for i in 0..batch.num_rows() {
            symbols.push(column.value(i).to_string());
        }
    }
    symbols
}

/// Cross-sectional market panel data produced by [`build_market_data`].
///
/// Daily price fields are signaled-or-NaN; `close_carried` is the carried
/// (last known) closing price. Fundamental fields are carried, in which income
/// and cash-flow statement fields are annualized.
#[allow(dead_code)]
pub struct MarketData {
    /// One pulse per trading day (when any stock has a row).
    pub daily: SignalPortHandle<0>,
    /// Per-stock signals: element `i` pulses on stock `i`'s trading days.
    pub price_signals: SignalPortHandle<1>,
    /// One pulse per income-statement effective date (when any stock reports).
    pub income_reports: SignalPortHandle<0>,
    /// Per-stock report signals, for TTM sampling.
    pub income_report_signals: SignalPortHandle<1>,
    /// One pulse per cash-flow-statement effective date.
    pub cf_reports: SignalPortHandle<0>,
    /// Per-stock report signals, for TTM sampling.
    pub cf_report_signals: SignalPortHandle<1>,

    // Daily price fields (per-stock signaled-or-NaN).
    pub close: ArrayPortHandle<f64, 1>,
    pub open: ArrayPortHandle<f64, 1>,
    pub high: ArrayPortHandle<f64, 1>,
    pub low: ArrayPortHandle<f64, 1>,
    pub volume: ArrayPortHandle<f64, 1>,
    pub amount: ArrayPortHandle<f64, 1>,
    /// Forward-adjusted close.
    pub adj_close: ArrayPortHandle<f64, 1>,
    /// Log forward-adjusted close — the source of returns.
    pub log_adj: ArrayPortHandle<f64, 1>,
    /// Daily log return of the adjusted close (`NaN` around non-trading days).
    pub log_return: ArrayPortHandle<f64, 1>,

    /// Carried closing price, `NaN` before listing.
    pub close_carried: ArrayPortHandle<f64, 1>,

    // Dividend events.
    pub div_signals: SignalPortHandle<1>,
    pub share_divs: ArrayPortHandle<f64, 1>,
    pub cash_divs: ArrayPortHandle<f64, 1>,

    // Equity structure (carried).
    pub total_shares: ArrayPortHandle<f64, 1>,
    pub circ_shares: ArrayPortHandle<f64, 1>,

    // Balance-sheet stocks (carried; liabilities credit-negative).
    pub parent_equity: ArrayPortHandle<f64, 1>, // 归母净资产, positive
    pub total_assets: ArrayPortHandle<f64, 1>,
    pub total_liab: ArrayPortHandle<f64, 1>, // NEGATIVE
    pub current_assets: ArrayPortHandle<f64, 1>,
    pub current_liab: ArrayPortHandle<f64, 1>, // NEGATIVE
    pub cash: ArrayPortHandle<f64, 1>,
    pub inventories: ArrayPortHandle<f64, 1>,
    pub receivables: ArrayPortHandle<f64, 1>,

    // Annualized income and cash flows (carried; costs credit-negative).
    pub net_profit: ArrayPortHandle<f64, 1>,
    pub operating_profit: ArrayPortHandle<f64, 1>,
    pub revenue: ArrayPortHandle<f64, 1>,
    pub operating_cost: ArrayPortHandle<f64, 1>, // NEGATIVE
    pub net_operating_cashflow: ArrayPortHandle<f64, 1>,

    /// Number of symbols on the axis.
    pub n: usize,
}

/// Loads the long-format Parquet panels and wire the cross-sectional fields.
pub fn build_market_data(
    b: &mut Builder<Instant, UnixTime>,
    symbols: &[String],
    args: &Args,
) -> MarketData {
    let dir = &args.data_dir;
    let start = args.data_start();
    let end = args.end;

    // Whole-market axis: the schema must cover every label its table carries.
    let schema = Schema::new(symbols);

    // Helper function for registering Parquet panel sources.
    let panel = |b: &mut Builder<Instant, UnixTime>,
                 kind: &str,
                 columns: Vec<String>|
     -> (SignalPortHandle<1>, Box<[ArrayPortHandle<f64, 1>]>) {
        b.source(
            panel::parquet(
                format!("{dir}/{kind}.parquet"),
                "date",
                [("symbol".into(), Axis::Labeled(schema.clone()))],
                columns,
            )
            .with_time_range(start, end),
        )
    };

    let (price_signals, prices) = panel(
        b,
        "daily_prices",
        vec![
            "prices.close".into(),  // 0
            "prices.open".into(),   // 1
            "prices.high".into(),   // 2
            "prices.low".into(),    // 3
            "prices.volume".into(), // 4 (shares)
            "prices.amount".into(), // 5 (turnover value, 成交额)
        ],
    );
    let (div_signals, divs) = panel(
        b,
        "dividends",
        vec!["dividends.share".into(), "dividends.cash".into()],
    );
    let (_equity_signals, equity) = panel(
        b,
        "equity_structures",
        vec!["shares.total".into(), "shares.circulating".into()],
    );
    let (_balance_signals, balance) = panel(
        b,
        "balance_sheets",
        vec![
            // [0..3]: parent-equity components (summed and negated below).
            "balance_sheet.equity.capital".into(),
            "balance_sheet.equity.reserves".into(),
            "balance_sheet.equity.parent_interests".into(),
            // [3..10]: balance stocks for the fundamental factors
            // (assets debit-positive, liabilities credit-negative).
            "balance_sheet.assets".into(),
            "balance_sheet.liab".into(),
            "balance_sheet.assets.current".into(),
            "balance_sheet.liab.current".into(),
            "balance_sheet.assets.current.cash".into(),
            "balance_sheet.assets.current.inventories".into(),
            "balance_sheet.assets.current.receivables.notes_and_accounts".into(),
        ],
    );
    let (income_report_signals, income) = panel(
        b,
        "income_statements",
        vec![
            "report_year".into(),
            "report_day_of_year".into(),
            "income_statement.profit".into(),
            "income_statement.profit.operating".into(),
            "income_statement.profit.operating.income.revenue".into(),
            "income_statement.profit.operating.expenses.costs".into(), // negative
        ],
    );
    let (cf_report_signals, cashflow) = panel(
        b,
        "cash_flow_statements",
        vec![
            "report_year".into(),
            "report_day_of_year".into(),
            "cash_flow_statement.change.operating".into(),
        ],
    );

    // Batch-level cadences: any stock has a row on this timestamp.
    let daily = b.op(signal::any(), price_signals);
    let income_reports = b.op(signal::any(), income_report_signals);
    let cf_reports = b.op(signal::any(), cf_report_signals);

    // Daily price fields.
    let close = b.op(signal::collect(), (price_signals, prices[0], daily));
    let open = b.op(signal::collect(), (price_signals, prices[1], daily));
    let high = b.op(signal::collect(), (price_signals, prices[2], daily));
    let low = b.op(signal::collect(), (price_signals, prices[3], daily));
    let volume = b.op(signal::collect(), (price_signals, prices[4], daily));
    let amount = b.op(signal::collect(), (price_signals, prices[5], daily));

    // Forward adjustment for dividends, whole cross-section at once.
    let (share_divs, cash_divs) = (divs[0], divs[1]);
    let (_multipliers, adj_close) = b.op(
        feature::forward_adjust(),
        ((price_signals, close), (div_signals, share_divs, cash_divs)),
    );
    let adj_close = b.op(signal::collect(), (price_signals, adj_close, daily));
    let log_adj = b.op(elem::ln(), adj_close);
    let log_return = b.op(rolling::diff(1), (daily, log_adj));

    // parent_equity = -(capital + reserves + parent_interests): the Parquet
    // files store equity credit-negative.
    let capital_reserves = b.op(elem::add(), (balance[0], balance[1]));
    let parent_sum = b.op(elem::add(), (capital_reserves, balance[2]));
    let parent_equity = b.op(elem::neg(), parent_sum);

    // Annualized income and cash flows (YTD -> annualized), gated per element
    // by the report signal.
    let income_year = b.op(elem::as_(), income[0]);
    let income_doy = b.op(elem::as_(), income[1]);
    let annualize = |b: &mut Builder<Instant, UnixTime>, ytd| {
        b.op(
            feature::annualize(),
            (income_report_signals, ytd, income_year, income_doy),
        )
    };
    let net_profit = annualize(b, income[2]);
    let operating_profit = annualize(b, income[3]);
    let revenue = annualize(b, income[4]);
    let operating_cost = annualize(b, income[5]);

    let cf_year = b.op(elem::as_(), cashflow[0]);
    let cf_doy = b.op(elem::as_(), cashflow[1]);
    let net_operating_cashflow = b.op(
        feature::annualize(),
        (cf_report_signals, cashflow[2], cf_year, cf_doy),
    );

    MarketData {
        daily,
        price_signals,
        income_reports,
        income_report_signals,
        cf_reports,
        cf_report_signals,
        close,
        open,
        high,
        low,
        volume,
        amount,
        adj_close,
        log_adj,
        log_return,
        close_carried: prices[0],
        div_signals,
        share_divs,
        cash_divs,
        total_shares: equity[0],
        circ_shares: equity[1],
        parent_equity,
        total_assets: balance[3],
        total_liab: balance[4],
        current_assets: balance[5],
        current_liab: balance[6],
        cash: balance[7],
        inventories: balance[8],
        receivables: balance[9],
        net_profit,
        operating_profit,
        revenue,
        operating_cost,
        net_operating_cashflow,
        n: symbols.len(),
    }
}
