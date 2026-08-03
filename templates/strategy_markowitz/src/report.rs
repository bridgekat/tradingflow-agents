//! NAV readout: summary statistics and the wide CSV the plot scripts read.

use std::fmt::Write as _;
use tradingflow::{data::Instant, graph::Graph, ports::SeriesPortHandle, time::UnixTime};

use crate::args::format_date;

/// Trading days per year, for annualizing the daily statistics.
const DAYS_PER_YEAR: f64 = 252.0;

/// Summary statistics of a daily NAV curve.
pub struct ReportStats {
    pub final_value: f64,
    pub cagr: f64,
    pub sharpe: f64,
    pub vol: f64,
    pub max_drawdown: f64,
}

/// Statistics from a daily NAV series; ratios are NaN for curves with fewer
/// than 10 finite positive samples.
pub fn nav_stats(values: &[f64]) -> ReportStats {
    let s: Vec<f64> = values
        .iter()
        .copied()
        .filter(|x| x.is_finite() && *x > 0.0)
        .collect();
    let final_value = s.last().copied().unwrap_or(f64::NAN);
    if s.len() < 10 {
        return ReportStats {
            final_value,
            cagr: f64::NAN,
            sharpe: f64::NAN,
            vol: f64::NAN,
            max_drawdown: f64::NAN,
        };
    }
    let years = s.len() as f64 / DAYS_PER_YEAR;
    let cagr = (s[s.len() - 1] / s[0]).powf(1.0 / years) - 1.0;
    let returns: Vec<f64> = s.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    let std = var.sqrt();
    let sharpe = if std > 0.0 {
        mean / std * DAYS_PER_YEAR.sqrt()
    } else {
        f64::NAN
    };
    let mut peak = f64::MIN;
    let max_drawdown = s.iter().fold(0.0f64, |mdd, &nav| {
        peak = peak.max(nav);
        mdd.min(nav / peak - 1.0)
    });
    ReportStats {
        final_value,
        cagr,
        sharpe,
        vol: std * DAYS_PER_YEAR.sqrt(),
        max_drawdown,
    }
}

/// Accumulates the labelled NAV columns the strategy writes to CSV. All
/// curves are recorded on the same daily pulse, so they share one date axis.
#[derive(Default)]
pub struct ReportTable {
    instants: Vec<Instant>,
    columns: Vec<(String, Vec<f64>)>,
}

impl ReportTable {
    /// Read a recorded NAV, trim the warm-up before `start`, add it as a
    /// column, and print its summary line.
    pub fn add(
        &mut self,
        g: &Graph<Instant, UnixTime>,
        label: impl Into<String>,
        start: Option<Instant>,
        h: SeriesPortHandle<f64, 0>,
    ) {
        let label = label.into();
        let series = g.view(h);
        let (instants, values) = (series.instants(), series.to_contiguous());
        let keep = instants
            .iter()
            .position(|&t| start.is_none_or(|s| t >= s))
            .unwrap_or(instants.len());
        let (instants, values) = (&instants[keep..], &values[keep..]);
        if self.columns.is_empty() {
            self.instants = instants.to_vec();
        } else {
            assert_eq!(self.instants.len(), instants.len(), "misaligned NAV curves");
        }
        let stats = nav_stats(values);
        println!(
            "{label:>12}: final={:>12.0}  cagr={:>+6.2}%  vol={:>5.2}%  sharpe={:>6.3}  mdd={:>6.2}%",
            stats.final_value,
            stats.cagr * 100.0,
            stats.vol * 100.0,
            stats.sharpe,
            stats.max_drawdown * 100.0,
        );
        self.columns.push((label, values.to_vec()));
    }

    /// Write the accumulated columns as `date,<label>,...` CSV.
    pub fn write(&self, path: &str) {
        let mut csv = String::from("date");
        for (label, _) in &self.columns {
            write!(csv, ",{label}").unwrap();
        }
        csv.push('\n');
        for (i, &t) in self.instants.iter().enumerate() {
            write!(csv, "{}", format_date(t)).unwrap();
            for (_, values) in &self.columns {
                write!(csv, ",{}", values[i]).unwrap();
            }
            csv.push('\n');
        }
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, csv).unwrap_or_else(|e| panic!("write {path}: {e}"));
        println!("wrote {} NAV points to {path}", self.instants.len());
    }
}
