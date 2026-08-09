//! NAV readout: summary statistics, the wide CSV the plot scripts read, and
//! the long-format weights CSV written alongside it.

use std::fmt::Write as _;
use tradingflow::{data::Instant, graph::Graph, ports::SeriesPortHandle, time::UnixTime};

use crate::utils::format_date;

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

/// The weights CSV path: `<stem>_weights.<ext>` next to the NAV CSV.
pub fn weights_path(output: &str) -> String {
    let p = std::path::Path::new(output);
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
    let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("csv");
    p.with_file_name(format!("{stem}_weights.{ext}"))
        .to_string_lossy()
        .into_owned()
}

/// Accumulates each portfolio's recorded rebalance books and writes them as a
/// long-format CSV — `date,portfolio,symbol,weight` — trimmed to significant
/// holdings.
///
/// The weights are the optimizer's targets set on each rebalance date
/// (executed at the next trading day's quotes, so a limit-locked or suspended
/// name may not fill exactly), one block of rows per date in descending
/// weight order. The residual `1 − Σ weights` — the book's cash — is
/// reported under the pseudo-symbol `CASH`, so a block sums to 1 up to the
/// trimmed dust.
#[derive(Default)]
pub struct WeightsTable {
    /// `(portfolio, date, symbol, weight)` rows, in write order.
    rows: Vec<(String, Instant, String, f64)>,
}

impl WeightsTable {
    /// Read a recorded rebalance-book series and add its rows: holdings with
    /// `weight ≥ min_weight` (and the `CASH` residual), largest first per
    /// date. `symbols` is the cross-section axis the book indexes.
    pub fn add(
        &mut self,
        g: &Graph<Instant, UnixTime>,
        label: impl Into<String>,
        symbols: &[String],
        min_weight: f64,
        h: SeriesPortHandle<f64, 1>,
    ) {
        // Suppress pure floating-point dust even when `min_weight` is zero.
        let threshold = min_weight.max(1e-9);
        let label = label.into();
        let series = g.view(h);
        let instants = series.instants();
        for (k, &date) in instants.iter().enumerate() {
            let book = series.at(k).1.to_contiguous();
            let mut sum = 0.0;
            let mut kept: Vec<(usize, f64)> = Vec::new();
            for (i, &w) in book.iter().enumerate() {
                if w.is_finite() && w > 0.0 {
                    sum += w;
                    if w >= threshold {
                        kept.push((i, w));
                    }
                }
            }
            kept.sort_by(|a, b| b.1.total_cmp(&a.1));
            for (i, w) in kept {
                self.rows.push((label.clone(), date, symbols[i].clone(), w));
            }
            let cash = 1.0 - sum;
            if cash >= threshold {
                self.rows.push((label.clone(), date, "CASH".into(), cash));
            }
        }
    }

    /// Write the accumulated rows as `date,portfolio,symbol,weight` CSV.
    pub fn write(&self, path: &str) {
        let mut csv = String::from("date,portfolio,symbol,weight\n");
        for (portfolio, date, symbol, weight) in &self.rows {
            writeln!(csv, "{},{portfolio},{symbol},{weight}", format_date(*date)).unwrap();
        }
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, csv).unwrap_or_else(|e| panic!("write {path}: {e}"));
        println!("wrote {} weight rows to {path}", self.rows.len());
    }
}
