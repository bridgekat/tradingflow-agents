//! Industry dummy exposures: one static `0/1` column per distinct industry
//! tag of the symbol list, named `IND_<tag>`. Stocks with an empty tag load
//! on no industry (only the always-on `COUNTRY` intercept). The columns are
//! stacked into the risk panel as-is — a dummy must stay `0/1`, so they
//! bypass the style finalizer.

use std::collections::BTreeSet;

use tradingflow::{
    data::{Array, Instant},
    graph::Builder,
    operators::array,
    ports::ArrayPortHandle,
    time::UnixTime,
};

/// Builds the industry dummy columns from the per-symbol tags, one
/// `(name, constant wire)` entry per distinct non-empty tag, in tag order.
pub fn build_features_industry(
    b: &mut Builder<Instant, UnixTime>,
    industries: &[String],
) -> Vec<(String, ArrayPortHandle<f64, 1>)> {
    let n = industries.len();
    let tags: BTreeSet<&String> = industries.iter().filter(|t| !t.is_empty()).collect();
    tags.into_iter()
        .map(|tag| {
            let column: Vec<f64> = industries
                .iter()
                .map(|t| if t == tag { 1.0 } else { 0.0 })
                .collect();
            (
                format!("IND_{tag}"),
                b.val(array::constant(Array::from_parts([n], column.into()))),
            )
        })
        .collect()
}
