//! The scorer's readouts: the summary statistics of each scored book, the NAV
//! CSV, and the JSON record a research ledger appends.
//!
//! Unlike the template's report, this one is authoritative: it is computed by
//! the scorer from the weights a submission produced, not by the submission
//! itself, so the numbers it writes are evidence rather than a claim. The
//! statistics struct and the JSON are one definition — a metric cannot be
//! printed but forgotten in the record.
//!
//! `serde_json` renders a non-finite float as `null`, which is the behaviour
//! wanted here: a book too short to score belongs in the ledger as a recorded
//! failure, not as a file that fails to parse.

use serde::Serialize;
use std::fmt::Write as _;
use tradingflow::{data::Instant, graph::Graph, ports::SeriesPortHandle, time::UnixTime};

use crate::utils::format_date;

/// Trading days per year, for annualizing the daily statistics.
const DAYS_PER_YEAR: f64 = 252.0;

/// Summary statistics of one scored book.
#[derive(Serialize)]
pub struct PortfolioStats {
    pub label: String,
    pub final_value: f64,
    pub cagr: f64,
    pub vol: f64,
    pub sharpe: f64,
    pub max_drawdown: f64,
    /// Annualized two-way turnover: the `Σ|Δw|` the book accumulated per year.
    /// At fee rates `buy` and `sell` the annual drag is about
    /// `turnover * (buy + sell) / 2`.
    pub turnover: f64,
    /// The largest `Σ|w|` any of this book's rebalances asked for — the
    /// leverage check, reported whether or not it passed.
    pub max_gross: f64,
}

/// Statistics from a daily NAV curve, the cumulative turnover curve and the
/// per-rebalance gross exposure recorded beside it. The ratios are `NaN` for a
/// curve with fewer than 10 finite positive samples.
fn stats(label: String, nav: &[f64], turnover: &[f64], gross: &[f64]) -> PortfolioStats {
    let max_gross = gross
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(0.0f64, f64::max);
    let turnover = annualized(turnover);
    let s: Vec<f64> = nav
        .iter()
        .copied()
        .filter(|x| x.is_finite() && *x > 0.0)
        .collect();
    let final_value = s.last().copied().unwrap_or(f64::NAN);
    if s.len() < 10 {
        return PortfolioStats {
            label,
            final_value,
            cagr: f64::NAN,
            vol: f64::NAN,
            sharpe: f64::NAN,
            max_drawdown: f64::NAN,
            turnover,
            max_gross,
        };
    }
    let years = s.len() as f64 / DAYS_PER_YEAR;
    let cagr = (s[s.len() - 1] / s[0]).powf(1.0 / years) - 1.0;
    let returns: Vec<f64> = s.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    let std = var.sqrt();
    let mut peak = f64::MIN;
    let max_drawdown = s.iter().fold(0.0f64, |mdd, &nav| {
        peak = peak.max(nav);
        mdd.min(nav / peak - 1.0)
    });
    PortfolioStats {
        label,
        final_value,
        cagr,
        vol: std * DAYS_PER_YEAR.sqrt(),
        sharpe: match std > 0.0 {
            true => mean / std * DAYS_PER_YEAR.sqrt(),
            false => f64::NAN,
        },
        max_drawdown,
        turnover,
        max_gross,
    }
}

/// Annualizes a cumulative curve: what it accumulated over its span, divided
/// by that span in years. `NaN` below 10 finite samples.
///
/// The turnover curve steps once per rebalance by the L1 distance between the
/// outgoing and incoming books, so it measures *intended* trading: it does not
/// see the drift of held weights between rebalances, and its first step
/// includes the one-off cost of opening the book. Read it as the lower bound
/// it is.
fn annualized(curve: &[f64]) -> f64 {
    let s: Vec<f64> = curve.iter().copied().filter(|x| x.is_finite()).collect();
    if s.len() < 10 {
        return f64::NAN;
    }
    (s[s.len() - 1] - s[0]) / (s.len() as f64 / DAYS_PER_YEAR)
}

/// The recorded series of one book, as handed to [`Report::add`].
pub struct Recorded {
    /// Daily NAV.
    pub nav: SeriesPortHandle<f64, 0>,
    /// Cumulative turnover, on the daily pulse so it shares the NAV's axis.
    pub turnover: SeriesPortHandle<f64, 0>,
    /// Gross exposure `Σ|w|`, on the daily pulse likewise.
    pub gross: SeriesPortHandle<f64, 0>,
}

/// Accumulates the scored books: their statistics, and the NAV columns written
/// to CSV. All curves are recorded on the same daily pulse, so they share one
/// date axis.
#[derive(Default)]
pub struct Report {
    instants: Vec<Instant>,
    stats: Vec<PortfolioStats>,
    navs: Vec<Vec<f64>>,
}

impl Report {
    /// Read one book's recorded series, trim the warm-up before `start`, keep
    /// its statistics and NAV column, and print its summary line.
    pub fn add(
        &mut self,
        g: &Graph<Instant, UnixTime>,
        label: impl Into<String>,
        start: Option<Instant>,
        r: Recorded,
    ) {
        let series = g.view(r.nav);
        let (instants, nav) = (series.instants(), series.to_contiguous());
        let keep = instants
            .iter()
            .position(|&t| start.is_none_or(|s| t >= s))
            .unwrap_or(instants.len());
        let (instants, nav) = (&instants[keep..], &nav[keep..]);
        if self.stats.is_empty() {
            self.instants = instants.to_vec();
        } else {
            assert_eq!(self.instants.len(), instants.len(), "misaligned NAV curves");
        }

        // Both companions share the NAV's daily pulse, so its trim applies.
        let turnover = g.view(r.turnover).to_contiguous();
        let gross = g.view(r.gross).to_contiguous();
        assert_eq!(turnover.len(), keep + nav.len(), "misaligned turnover curve");
        assert_eq!(gross.len(), keep + nav.len(), "misaligned exposure curve");

        let s = stats(label.into(), nav, &turnover[keep..], &gross[keep..]);
        println!(
            "{:>20}: final={:>12.0}  cagr={:>+6.2}%  vol={:>5.2}%  sharpe={:>6.3}  mdd={:>6.2}%  turn={:>5.2}x  gross={:>4.2}",
            s.label,
            s.final_value,
            s.cagr * 100.0,
            s.vol * 100.0,
            s.sharpe,
            s.max_drawdown * 100.0,
            s.turnover,
            s.max_gross,
        );
        self.stats.push(s);
        self.navs.push(nav.to_vec());
    }

    /// The scored books, in the order they were added.
    pub fn stats(&self) -> &[PortfolioStats] {
        &self.stats
    }

    /// Reject any book whose gross exposure exceeded `max_gross`, naming all
    /// of them rather than only the first.
    ///
    /// The check runs after scoring rather than at read time: the numbers are
    /// worth seeing even when the book is inadmissible, and a validator would
    /// rather read why a submission failed than watch it abort mid-run.
    pub fn enforce_gross(&self, max_gross: f64) {
        // Weights are printed to a CSV, so allow the last decimal to round.
        let over: Vec<&PortfolioStats> = (self.stats.iter())
            .filter(|s| s.max_gross > max_gross + 1e-6)
            .collect();
        assert!(
            over.is_empty(),
            "gross exposure above the scorer's limit of {max_gross}: {}",
            over.iter()
                .map(|s| format!("{} at {:.4}", s.label, s.max_gross))
                .collect::<Vec<_>>()
                .join(", "),
        );
    }

    /// Write the NAV curves as `date,<label>,...` CSV.
    pub fn write_nav(&self, path: &str) {
        let mut csv = String::from("date");
        for s in &self.stats {
            write!(csv, ",{}", s.label).unwrap();
        }
        csv.push('\n');
        for (i, &t) in self.instants.iter().enumerate() {
            write!(csv, "{}", format_date(t)).unwrap();
            for nav in &self.navs {
                write!(csv, ",{}", nav[i]).unwrap();
            }
            csv.push('\n');
        }
        write_file(path, &csv);
        println!("wrote {} NAV points to {path}", self.instants.len());
    }
}

/// Writes `contents` to `path`, creating the parent directory if needed.
pub fn write_file(path: &str, contents: &str) {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, contents).unwrap_or_else(|e| panic!("write {path}: {e}"));
}

/// The summary path: `<stem>_summary.json` next to the NAV CSV.
pub fn summary_path(output: &str) -> String {
    let p = std::path::Path::new(output);
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
    p.with_file_name(format!("{stem}_summary.json"))
        .to_string_lossy()
        .into_owned()
}
