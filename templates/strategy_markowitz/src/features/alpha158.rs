//! The Qlib Alpha158 catalog, ported to the expression vocabulary.
//!
//! The catalog is Qlib's default `Alpha158` configuration — 9 K-line shape
//! features ([`KBAR`]), 4 raw price ratios ([`PRICE`]) and 29 rolling families
//! ([`ROLLING`]) over the windows `[5, 10, 20, 30, 60]`, i.e.
//! `9 + 4 + 29 × 5 = 158` features. Each is a string expression rather than
//! hand-written graph wiring, lowered by [`build_features_alpha158`] through
//! the shared [price-field context](super::context), whose six fields are
//! exactly the ones `Alpha158` reads.
//!
//! Every name is Qlib's. The expression vocabulary is modeled on Qlib's too,
//! so the formulas transcribe near-verbatim; three decisions were forced
//! during the port, each of which changes values rather than merely spelling
//! (on top of the panel-level differences in the [module docs](super)):
//!
//! * **The up/down day counts are spelled as an explicit ratio.** Qlib writes
//!   `Mean($close > Ref($close, 1), d)`, feeding a boolean to a rolling mean.
//!   Rolling operators here take numeric wires, and the obvious wrapping
//!   `Mean(If($close > Ref($close, 1), 1, 0), d)` would count a non-trading
//!   day as a down day, because a comparison against `NaN` is `false`.
//!   `CNTP`/`CNTN`/`CNTD` are therefore written
//!   `Sum(If(…), d) / Count($close - Ref($close, 1), d)` — the same ratio, but
//!   over the days the stock actually traded, which is what Qlib's
//!   per-instrument calendar gives it.
//! * **`Greater`/`Less` ignore `NaN`** (they are Rust's `f64::max`/`min`), so
//!   `Greater($close - Ref($close, 1), 0)` is `0` on a non-trading day rather
//!   than `NaN`. That is what the `SUMP`/`SUMN`/`SUMD` and `VSUM*` families
//!   want: a day without a quote contributes nothing to the numerator, and
//!   nothing to the `NaN`-skipping `Sum(Abs(…))` denominator either.
//! * **`Corr` is not masked at near-zero variance.** Qlib nulls a rolling
//!   correlation whose either leg is near-constant over the window
//!   (`atol = 2e-5`); here `Corr` lowers to `Cov / (Std * Std)`, which is
//!   `NaN` only when a leg is *exactly* constant, and merely extreme just
//!   below that. Such a `CORR`/`CORD` is at most clamped to a winsorization
//!   bound instead of being imputed to the neutral fill.

use tradingflow::{
    data::Instant, expr::Context, graph::Builder, ports::ArrayPortHandle, time::UnixTime,
};

/// The rolling window lengths (in trading days) every [`ROLLING`] family is
/// instantiated at — Qlib's `Alpha158` default.
const WINDOWS: [usize; 5] = [5, 10, 20, 30, 60];

/// The K-line (candlestick) shape features, as `(name, expression)`: where the
/// close sits inside the day's bar, each ratio normalized by the open or by
/// the day's range. The `1e-12` guards are Qlib's, against a zero-range bar.
#[rustfmt::skip]
pub const KBAR: [(&str, &str); 9] = [
    ("KMID",  "($close - $open) / $open"),
    ("KLEN",  "($high - $low) / $open"),
    ("KMID2", "($close - $open) / ($high - $low + 1e-12)"),
    ("KUP",   "($high - Greater($open, $close)) / $open"),
    ("KUP2",  "($high - Greater($open, $close)) / ($high - $low + 1e-12)"),
    ("KLOW",  "(Less($open, $close) - $low) / $open"),
    ("KLOW2", "(Less($open, $close) - $low) / ($high - $low + 1e-12)"),
    ("KSFT",  "(2 * $close - $high - $low) / $open"),
    ("KSFT2", "(2 * $close - $high - $low) / ($high - $low + 1e-12)"),
];

/// The raw price features, as `(name, expression)`: each of the day's other
/// prices divided by the close, which removes the unit. The trailing `0` in
/// the name is Qlib's price window — today rather than `Ref(…, d)` days ago.
#[rustfmt::skip]
pub const PRICE: [(&str, &str); 4] = [
    ("OPEN0", "$open / $close"),
    ("HIGH0", "$high / $close"),
    ("LOW0",  "$low / $close"),
    ("VWAP0", "$vwap / $close"),
];

/// The rolling feature families, as `(name prefix, expression template)`. Each
/// is instantiated once per window in [`WINDOWS`]: `{d}` is substituted with
/// the window and the window is appended to the prefix, so `ROC` becomes
/// `ROC5`, `ROC10`, … Each comes out unitless: the price-valued ones are
/// divided by the close, the index-valued ones by the window length, the
/// volume-valued ones by today's volume.
#[rustfmt::skip]
pub const ROLLING: [(&str, &str); 29] = [
    // ---- Trend: where the close sits relative to its own recent past ----
    // Rate of change, and the moving average / dispersion of the close.
    ("ROC",  "Ref($close, {d}) / $close"),
    ("MA",   "Mean($close, {d}) / $close"),
    ("STD",  "Std($close, {d}) / $close"),
    // The OLS of the close against window position: slope, fit quality and
    // the newest residual — how strong and how linear the trend is.
    ("BETA", "Slope($close, {d}) / $close"),
    ("RSQR", "Rsquare($close, {d})"),
    ("RESI", "Resi($close, {d}) / $close"),

    // ---- Price level within the window's range ----
    ("MAX",  "Max($high, {d}) / $close"),
    ("MIN",  "Min($low, {d}) / $close"),
    ("QTLU", "Quantile($close, {d}, 0.8) / $close"),
    ("QTLD", "Quantile($close, {d}, 0.2) / $close"),
    ("RANK", "Rank($close, {d})"),
    // The stochastic oscillator's %K: the close between support and
    // resistance.
    ("RSV",  "($close - Min($low, {d})) / (Max($high, {d}) - Min($low, {d}) + 1e-12)"),

    // ---- Aroon: how recently the extremes were set ----
    ("IMAX", "IdxMax($high, {d}) / {d}"),
    ("IMIN", "IdxMin($low, {d}) / {d}"),
    ("IMXD", "(IdxMax($high, {d}) - IdxMin($low, {d})) / {d}"),

    // ---- Price/volume co-movement, in levels and in changes ----
    ("CORR", "Corr($close, Log($volume + 1), {d})"),
    ("CORD", "Corr($close / Ref($close, 1), Log($volume / Ref($volume, 1) + 1), {d})"),

    // ---- Up/down day counts (see the module docs on the spelling) ----
    ("CNTP", "Sum(If($close > Ref($close, 1), 1, 0), {d}) \
              / Count($close - Ref($close, 1), {d})"),
    ("CNTN", "Sum(If($close < Ref($close, 1), 1, 0), {d}) \
              / Count($close - Ref($close, 1), {d})"),
    ("CNTD", "(Sum(If($close > Ref($close, 1), 1, 0), {d}) \
              - Sum(If($close < Ref($close, 1), 1, 0), {d})) \
              / Count($close - Ref($close, 1), {d})"),

    // ---- RSI-shaped gain/loss decomposition of the close ----
    ("SUMP", "Sum(Greater($close - Ref($close, 1), 0), {d}) \
              / (Sum(Abs($close - Ref($close, 1)), {d}) + 1e-12)"),
    ("SUMN", "Sum(Greater(Ref($close, 1) - $close, 0), {d}) \
              / (Sum(Abs($close - Ref($close, 1)), {d}) + 1e-12)"),
    ("SUMD", "(Sum(Greater($close - Ref($close, 1), 0), {d}) \
              - Sum(Greater(Ref($close, 1) - $close, 0), {d})) \
              / (Sum(Abs($close - Ref($close, 1)), {d}) + 1e-12)"),

    // ---- Volume: level, dispersion, and the volume-weighted volatility ----
    ("VMA",   "Mean($volume, {d}) / ($volume + 1e-12)"),
    ("VSTD",  "Std($volume, {d}) / ($volume + 1e-12)"),
    ("WVMA",  "Std(Abs($close / Ref($close, 1) - 1) * $volume, {d}) \
               / (Mean(Abs($close / Ref($close, 1) - 1) * $volume, {d}) + 1e-12)"),

    // ---- The same RSI decomposition applied to volume ----
    ("VSUMP", "Sum(Greater($volume - Ref($volume, 1), 0), {d}) \
               / (Sum(Abs($volume - Ref($volume, 1)), {d}) + 1e-12)"),
    ("VSUMN", "Sum(Greater(Ref($volume, 1) - $volume, 0), {d}) \
               / (Sum(Abs($volume - Ref($volume, 1)), {d}) + 1e-12)"),
    ("VSUMD", "(Sum(Greater($volume - Ref($volume, 1), 0), {d}) \
               - Sum(Greater(Ref($volume, 1) - $volume, 0), {d})) \
               / (Sum(Abs($volume - Ref($volume, 1)), {d}) + 1e-12)"),
];

/// The catalog is 158 features, as advertised.
const _: () = assert!(KBAR.len() + PRICE.len() + ROLLING.len() * WINDOWS.len() == 158);

/// Lowers the whole Alpha158 catalog through `ctx`, returning one
/// `(name, raw value)` entry per feature.
///
/// Taking the context rather than building one lets the catalog share
/// subexpressions with any other factor lowered through it — the rolling
/// families overlap heavily across windows (`Ref($close, 1)`, the gain/loss
/// sums, `Abs($close / Ref($close, 1) - 1) * $volume`), and hash-consing makes
/// each distinct subexpression exactly one graph node.
pub fn build_features_alpha158(
    b: &mut Builder<Instant, UnixTime>,
    ctx: &mut Context<1>,
) -> Vec<(String, ArrayPortHandle<f64, 1>)> {
    let fixed = KBAR
        .iter()
        .chain(PRICE.iter())
        .map(|&(name, src)| (name.to_string(), src.to_string()));
    let rolling = ROLLING.iter().flat_map(|&(name, src)| {
        WINDOWS
            .iter()
            .map(move |d| (format!("{name}{d}"), src.replace("{d}", &d.to_string())))
    });
    fixed
        .chain(rolling)
        .map(|(name, src)| {
            let wire = ctx
                .lower_str(b, &src)
                .unwrap_or_else(|e| panic!("{name}: {e}\n  {src}"));
            (name, wire)
        })
        .collect()
}
