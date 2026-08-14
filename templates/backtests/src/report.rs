//! Backtest readouts: the NAV summary statistics and the wide CSV the plot
//! scripts read ([`ReportTable`]), the long-format weights CSV written
//! alongside it ([`WeightsTable`]), the per-feature diagnostics
//! ([`FeatureTable`]) — for now each feature's cumulative information
//! coefficient and the statistics of the IC series behind it — and the
//! per-risk-model diagnostics ([`RiskModelTable`]): the daily metric series
//! recorded from the risk-model scoring operators and their summary statistics.

use std::fmt::Write as _;
use tradingflow::{data::Instant, graph::Graph, ports::SeriesPortHandle, time::UnixTime};

use crate::utils::format_date;

/// Trading days per year, for annualizing the daily statistics.
const DAYS_PER_YEAR: f64 = 252.0;

/// Write a `date,<label>,...` wide CSV: one row per instant, one column per
/// labelled series (each as long as `instants`). `what` names the rows in the
/// line printed afterwards.
fn write_wide_csv(path: &str, instants: &[Instant], columns: &[(&str, &[f64])], what: &str) {
    let mut csv = String::from("date");
    for (label, _) in columns {
        write!(csv, ",{label}").unwrap();
    }
    csv.push('\n');
    for (i, &t) in instants.iter().enumerate() {
        write!(csv, "{}", format_date(t)).unwrap();
        for (_, values) in columns {
            write!(csv, ",{}", values[i]).unwrap();
        }
        csv.push('\n');
    }
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, csv).unwrap_or_else(|e| panic!("write {path}: {e}"));
    println!("wrote {} {what} to {path}", instants.len());
}

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
        let columns: Vec<(&str, &[f64])> = self
            .columns
            .iter()
            .map(|(label, values)| (label.as_str(), values.as_slice()))
            .collect();
        write_wide_csv(path, &self.instants, &columns, "NAV points");
    }
}

/// Summary statistics of one feature's daily information-coefficient (IC)
/// series — currently the whole of what [`FeatureTable`] scores a feature on.
pub struct FeatureStats {
    /// Finite daily observations: the days whose cross-section had enough
    /// pairwise-valid symbols to correlate at all.
    pub count: usize,
    /// Mean IC over those observations — the feature's average per-day
    /// cross-sectional correlation with the next day's return.
    pub mean: f64,
    /// Sample standard deviation of the IC: how much that edge wobbles.
    pub std: f64,
    /// `mean / std`: the per-day information ratio of the IC.
    pub icir: f64,
    /// `icir * sqrt(count)`: the t-statistic against a mean IC of zero.
    pub t_stat: f64,
    /// Share of the observations whose IC is positive.
    pub positive: f64,
    /// Final cumulative IC — the sum of the finite observations, i.e. the last
    /// point of the curve [`FeatureTable`] writes.
    pub cumulative: f64,
}

/// Statistics of a daily IC series over its finite entries; the dispersion
/// ratios are `NaN` for a series with fewer than two of them, and everything
/// but `cumulative` is `NaN` for one with none.
pub fn feature_stats(values: &[f64]) -> FeatureStats {
    let s: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    let count = s.len();
    let cumulative: f64 = s.iter().sum();
    let mean = cumulative / count as f64; // `NaN` when nothing is finite
    let positive = s.iter().filter(|&&x| x > 0.0).count() as f64 / count as f64;
    if count < 2 {
        return FeatureStats {
            count,
            mean,
            std: f64::NAN,
            icir: f64::NAN,
            t_stat: f64::NAN,
            positive,
            cumulative,
        };
    }
    let var = s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (count - 1) as f64;
    let std = var.sqrt();
    let icir = if std > 0.0 { mean / std } else { f64::NAN };
    FeatureStats {
        count,
        mean,
        std,
        icir,
        t_stat: icir * (count as f64).sqrt(),
        positive,
        cumulative,
    }
}

/// Accumulates the per-feature diagnostics: today each feature's cumulative
/// IC curve and the [`FeatureStats`] of the IC series behind it. All the IC
/// series are recorded on the same daily pulse, so they share one date axis.
///
/// The curve is the running sum of the daily ICs: a feature that predicts the
/// next day's cross-section with a stable edge climbs steadily, one with no
/// edge wanders around zero, and the slope at any point is the local mean IC.
/// A day the feature could not be scored (too few pairwise-valid symbols, so
/// the IC is `NaN` — typically the warm-up before its windows fill) carries
/// the curve flat rather than breaking it, which keeps the summed edge
/// comparable across features that start at different dates. It also means a
/// flat segment reads as "not scored", not as "scored zero"; the `n` column of
/// the printed summary is how many days actually counted.
#[derive(Default)]
pub struct FeatureTable {
    instants: Vec<Instant>,
    /// `(label, statistics, cumulative curve)`, in insertion order.
    columns: Vec<(String, FeatureStats, Vec<f64>)>,
}

impl FeatureTable {
    /// Read a feature's recorded daily IC series, add its cumulative curve as
    /// a column and keep its statistics for [`print`](Self::print).
    pub fn add(
        &mut self,
        g: &Graph<Instant, UnixTime>,
        label: impl Into<String>,
        ic: SeriesPortHandle<f64, 0>,
    ) {
        let label = label.into();
        let series = g.view(ic);
        let (instants, values) = (series.instants(), series.to_contiguous());
        if self.columns.is_empty() {
            self.instants = instants.to_vec();
        } else {
            assert_eq!(self.instants.len(), instants.len(), "misaligned IC curves");
        }
        let stats = feature_stats(&values);
        let mut sum = 0.0;
        let curve = values
            .iter()
            .map(|v| {
                if v.is_finite() {
                    sum += v;
                }
                sum
            })
            .collect();
        self.columns.push((label, stats, curve));
    }

    /// Print one summary line per feature, strongest mean IC first. A feature
    /// that was never scored sorts last, whatever its `NaN` statistics compare
    /// as.
    pub fn print(&self) {
        let mut order: Vec<&(String, FeatureStats, Vec<f64>)> = self.columns.iter().collect();
        let key = |s: &FeatureStats| match s.mean.is_finite() {
            true => s.mean.abs(),
            false => f64::NEG_INFINITY,
        };
        order.sort_by(|a, b| key(&b.1).total_cmp(&key(&a.1)));
        println!(
            "{:<28}{:>7}{:>10}{:>9}{:>8}{:>8}{:>8}{:>9}",
            "feature", "n", "mean IC", "std", "ICIR", "t", "pos", "cum IC",
        );
        for (label, s, _) in order {
            println!(
                "{label:<28}{:>7}{:>+10.4}{:>9.4}{:>+8.3}{:>+8.2}{:>7.1}%{:>+9.2}",
                s.count,
                s.mean,
                s.std,
                s.icir,
                s.t_stat,
                s.positive * 100.0,
                s.cumulative,
            );
        }
    }

    /// Write the accumulated cumulative IC curves as `date,<label>,...` CSV,
    /// the columns in the order the features were added.
    pub fn write(&self, path: &str) {
        let columns: Vec<(&str, &[f64])> = self
            .columns
            .iter()
            .map(|(label, _, curve)| (label.as_str(), curve.as_slice()))
            .collect();
        write_wide_csv(path, &self.instants, &columns, "cumulative IC points");
    }
}

/// The daily metric series a risk model is scored on, in recording order.
pub const RISK_MODEL_METRICS: [&str; 7] = [
    "log_likelihood",
    "gmv_return",
    "gmv_variance",
    "mean_correlation",
    "factor_share",
    "gmv_breadth",
    "gmv_short_share",
];

/// Summary statistics of one risk model's daily metric series.
pub struct RiskModelStats {
    /// Days on which the GMV portfolio was scored (finite return and
    /// positive predicted variance).
    pub days: usize,
    /// Mean per-asset Gaussian log-likelihood of the realized cross-sections
    /// under the predicted covariances. Higher is better.
    pub mean_log_likelihood: f64,
    /// Annualized realized volatility of the GMV portfolio built from the
    /// predicted covariances. Lower is better: minimizing this is exactly
    /// what the covariance forecast is for.
    pub realized_vol: f64,
    /// Annualized predicted volatility of the same portfolio (the mean of
    /// the model's own per-period forecasts).
    pub predicted_vol: f64,
    /// Standard deviation of the GMV returns standardized by their
    /// predicted volatility. Ideal is 1: above 1 the model under-forecasts
    /// its own portfolio's risk, below 1 it over-forecasts.
    pub bias_ratio: f64,
    /// Mean predicted off-diagonal correlation across the scored days.
    pub mean_correlation: f64,
    /// Mean share of predicted variance carried by the factors.
    pub factor_share: f64,
    /// Mean effective number of GMV holdings, `1 / Σ w²`.
    pub gmv_breadth: f64,
    /// Mean share of the GMV book's gross exposure carried by its short
    /// leg, `Σ max(-wᵢ, 0) / Σ |wᵢ|`.
    pub gmv_short_share: f64,
}

/// Mean of the finite entries; `NaN` when there are none.
fn finite_mean(xs: &[f64]) -> f64 {
    let s: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    s.iter().sum::<f64>() / s.len() as f64
}

/// Statistics from the daily metric series (columns in the order of
/// [`RISK_MODEL_METRICS`]); the volatility and bias numbers are `NaN` with
/// fewer than 10 scored days.
pub fn risk_model_stats(columns: &[Vec<f64>]) -> RiskModelStats {
    let mean_log_likelihood = finite_mean(&columns[0]);
    let mean_correlation = finite_mean(&columns[3]);
    let factor_share = finite_mean(&columns[4]);
    let gmv_breadth = finite_mean(&columns[5]);
    let gmv_short_share = finite_mean(&columns[6]);

    let scored: Vec<(f64, f64)> = columns[1]
        .iter()
        .zip(&columns[2])
        .filter(|(r, v)| r.is_finite() && v.is_finite() && **v > 0.0)
        .map(|(&r, &v)| (r, v))
        .collect();
    let days = scored.len();
    if days < 10 {
        return RiskModelStats {
            days,
            mean_log_likelihood,
            realized_vol: f64::NAN,
            predicted_vol: f64::NAN,
            bias_ratio: f64::NAN,
            mean_correlation,
            factor_share,
            gmv_breadth,
            gmv_short_share,
        };
    }

    let sample_std = |xs: &[f64]| {
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        (xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (xs.len() - 1) as f64).sqrt()
    };
    let rets: Vec<f64> = scored.iter().map(|(r, _)| *r).collect();
    let zs: Vec<f64> = scored.iter().map(|(r, v)| r / v.sqrt()).collect();
    let mean_pred = scored.iter().map(|(_, v)| v.sqrt()).sum::<f64>() / days as f64;
    RiskModelStats {
        days,
        mean_log_likelihood,
        realized_vol: sample_std(&rets) * DAYS_PER_YEAR.sqrt(),
        predicted_vol: mean_pred * DAYS_PER_YEAR.sqrt(),
        bias_ratio: sample_std(&zs),
        mean_correlation,
        factor_share,
        gmv_breadth,
        gmv_short_share,
    }
}

/// Accumulates the per-risk-model diagnostics: the daily metric series
/// recorded from the scoring operators (one rank-0 series per
/// [`RISK_MODEL_METRICS`] entry) and the [`RiskModelStats`] behind them.
/// All series are recorded on the same daily pulse, so they share one date
/// axis.
#[derive(Default)]
pub struct RiskModelTable {
    instants: Vec<Instant>,
    /// `(label, statistics, one series per [`RISK_MODEL_METRICS`] column)`,
    /// in insertion order.
    columns: Vec<(String, RiskModelStats, Vec<Vec<f64>>)>,
}

impl RiskModelTable {
    /// Read one risk model's recorded daily metric series (in the order of
    /// [`RISK_MODEL_METRICS`]), trim the warm-up before `start`, add its
    /// columns and keep its statistics for [`print`](Self::print).
    pub fn add(
        &mut self,
        g: &Graph<Instant, UnixTime>,
        label: impl Into<String>,
        start: Option<Instant>,
        hs: [SeriesPortHandle<f64, 0>; RISK_MODEL_METRICS.len()],
    ) {
        let label = label.into();
        let mut columns = Vec::with_capacity(RISK_MODEL_METRICS.len());
        for h in hs {
            let series = g.view(h);
            let instants = series.instants();
            let keep = instants
                .iter()
                .position(|&t| start.is_none_or(|s| t >= s))
                .unwrap_or(instants.len());
            if self.columns.is_empty() && columns.is_empty() {
                self.instants = instants[keep..].to_vec();
            } else {
                assert_eq!(
                    self.instants.len(),
                    instants.len() - keep,
                    "misaligned metric series"
                );
            }
            columns.push(series.to_contiguous()[keep..].to_vec());
        }
        let stats = risk_model_stats(&columns);
        self.columns.push((label, stats, columns));
    }

    /// Print one summary line per risk model, highest mean log-likelihood
    /// first. A model that was never scored sorts last, whatever its `NaN`
    /// statistics compare as.
    pub fn print(&self) {
        let mut order: Vec<&(String, RiskModelStats, Vec<Vec<f64>>)> =
            self.columns.iter().collect();
        let key = |s: &RiskModelStats| match s.mean_log_likelihood.is_finite() {
            true => s.mean_log_likelihood,
            false => f64::NEG_INFINITY,
        };
        order.sort_by(|a, b| key(&b.1).total_cmp(&key(&a.1)));
        println!(
            "{:<40}{:>6}{:>11}{:>10}{:>10}{:>7}{:>8}{:>8}{:>9}{:>8}",
            "risk model",
            "days",
            "logL/asset",
            "GMV vol",
            "GMV pred",
            "bias",
            "corr",
            "factor",
            "N_eff",
            "short",
        );
        for (label, s, _) in order {
            println!(
                "{label:<40}{:>6}{:>+11.4}{:>9.2}%{:>9.2}%{:>7.2}{:>8.3}{:>7.0}%{:>9.0}{:>7.0}%",
                s.days,
                s.mean_log_likelihood,
                s.realized_vol * 100.0,
                s.predicted_vol * 100.0,
                s.bias_ratio,
                s.mean_correlation,
                s.factor_share * 100.0,
                s.gmv_breadth,
                s.gmv_short_share * 100.0,
            );
        }
    }

    /// Write the accumulated metric series as a `date,<label>_...` wide CSV,
    /// one column per metric per risk model, in the order the models were
    /// added.
    pub fn write(&self, path: &str) {
        let labels: Vec<Vec<String>> = self
            .columns
            .iter()
            .map(|(label, _, _)| {
                RISK_MODEL_METRICS
                    .iter()
                    .map(|metric| format!("{label}_{metric}"))
                    .collect()
            })
            .collect();
        let mut columns: Vec<(&str, &[f64])> = Vec::new();
        for ((_, _, series), names) in self.columns.iter().zip(&labels) {
            for (values, name) in series.iter().zip(names) {
                columns.push((name.as_str(), values.as_slice()));
            }
        }
        write_wide_csv(path, &self.instants, &columns, "metric points");
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
