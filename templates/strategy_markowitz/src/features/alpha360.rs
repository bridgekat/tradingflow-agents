//! The Qlib Alpha360 catalog, ported to the expression vocabulary.
//!
//! Alpha360 is not a factor catalog so much as a raw window: the last
//! [`LAGS`] trading days of each of the six daily [`SERIES`], normalized by
//! today's value so the panel is unitless — the prices by `$close`, the volume
//! by `$volume`. `6 × 60 = 360` features, lowered by
//! [`build_features_alpha360`] through the shared
//! [price-field context](super::context), whose six fields are exactly the
//! ones `Alpha360` reads. It is meant for models that learn their own features
//! from the sequence, and the column order is Qlib's — one series at a time,
//! oldest lag first — so a `(60, 6)` reshape lands the way those models
//! expect.
//!
//! Nothing here needs an operator outside `Ref` and division, so every
//! expression transcribes verbatim from Qlib — including its two spellings of
//! lag `0`, which drop the `Ref` rather than passing it a zero window. That is
//! worth preserving: `Ref(x, 0)` *is* `x` here, the same Qlib-compatible
//! identity, but hash-consing is syntactic, so `Ref($open, 0) / $close` would
//! build a second node beside Alpha158's identically-valued `OPEN0`.
//!
//! Two consequences of Qlib's own design, called out in its source:
//!
//! * `CLOSE0` is `1` wherever the stock traded, and `VOLUME0` is `1` up to the
//!   `1e-12` guard. Both survive into the panel for faithfulness;
//!   [`winsorize_impute`](super::winsorize_impute) passes them through as
//!   (near-)constant columns.
//! * `OPEN0`, `HIGH0`, `LOW0` and `VWAP0` are the same name *and* the same
//!   expression as four of [Alpha158](super::alpha158)'s, which is why
//!   [`build_features`](super::build_features) deduplicates by name when both
//!   catalogs are selected. That leaves the four in Alpha158's column
//!   positions rather than this catalog's, so select Alpha360 alone if the
//!   `(60, 6)` layout matters.
//!
//! Beyond the panel-level differences in the [module docs](super), nothing
//! here departs from Qlib: every name and every expression is its own.

use tradingflow::{
    data::Instant, expr::Context, graph::Builder, ports::ArrayPortHandle, time::UnixTime,
};

/// The number of trailing trading days the window spans, lag `0` (today)
/// through lag `59` — Qlib's `Alpha360` length.
const LAGS: usize = 60;

/// The daily series the window is taken over, as `(name prefix, lagged
/// expression, today's expression)`. `{d}` is substituted with the lag in
/// trading days and appended to the prefix, so `CLOSE` becomes `CLOSE59`,
/// `CLOSE58`, … `CLOSE1`, with the third column supplying `CLOSE0`. The prices
/// are normalized by today's close and the volume by today's volume, which is
/// what makes the window comparable across stocks and dates.
#[rustfmt::skip]
pub const SERIES: [(&str, &str, &str); 6] = [
    ("CLOSE",  "Ref($close, {d}) / $close",             "$close / $close"),
    ("OPEN",   "Ref($open, {d}) / $close",              "$open / $close"),
    ("HIGH",   "Ref($high, {d}) / $close",              "$high / $close"),
    ("LOW",    "Ref($low, {d}) / $close",               "$low / $close"),
    ("VWAP",   "Ref($vwap, {d}) / $close",              "$vwap / $close"),
    ("VOLUME", "Ref($volume, {d}) / ($volume + 1e-12)", "$volume / ($volume + 1e-12)"),
];

/// The catalog is 360 features, as advertised.
const _: () = assert!(SERIES.len() * LAGS == 360);

/// Lowers the whole Alpha360 catalog through `ctx`, returning one
/// `(name, raw value)` entry per feature, in Qlib's column order: each series
/// in [`SERIES`] order, oldest lag (`59`) to newest (`0`).
///
/// Taking the context rather than building one lets the catalog share
/// subexpressions with any other factor lowered through it — `Ref($close, d)`
/// is exactly what Alpha158's `ROC{d}` is built from — and hash-consing makes
/// each distinct subexpression exactly one graph node. Even so this is the
/// most expensive catalog to run: the lags do not nest, so each of the 354
/// non-zero ones buffers its own window of the whole cross-section.
pub fn build_features_alpha360(
    b: &mut Builder<Instant, UnixTime>,
    ctx: &mut Context<1>,
) -> Vec<(String, ArrayPortHandle<f64, 1>)> {
    SERIES
        .iter()
        .flat_map(|&(name, lagged, today)| {
            (0..LAGS).rev().map(move |d| {
                let src = match d {
                    0 => today.to_string(),
                    d => lagged.replace("{d}", &d.to_string()),
                };
                (format!("{name}{d}"), src)
            })
        })
        .map(|(name, src)| {
            let wire = ctx
                .lower_str(b, &src)
                .unwrap_or_else(|e| panic!("{name}: {e}\n  {src}"));
            (name, wire)
        })
        .collect()
}
