//! The cross-sectional feature catalog: the alpha panel the mean predictor
//! regresses on, and the risk panel the covariance predictor reads as factor
//! exposures — both built by [`build_features`].
//!
//! **Every feature reaches a model on a normalized scale.** Neither model
//! defends itself against numerical blow-ups: both solve a cross-sectional
//! least-squares problem per period, where one outlier or one column carrying
//! a wildly different magnitude dominates the normal equations, and the single
//! scalar L2 term they regularize with only means the same thing across
//! features if those features are comparably scaled. So [`normalize`] finalizes
//! every entry of both panels to a winsorized, imputed, cross-sectionally
//! z-scored and clamped column: mean `0`, standard deviation `1`, bounded
//! magnitude, no infinities, no missing values.
//!
//! A catalog is still free to make a transform part of a feature's own
//! definition where the raw cross-section warrants it — the CICC fundamentals
//! rank themselves to `[0, 1]` percentiles, since a ratio whose denominator
//! crosses zero stays heavy-tailed under any purely linear rescaling. That is
//! about the *shape* of the cross-section; [`normalize`] then fixes its
//! location and scale, and the two compose without conflict.
//!
//! The risk panel additionally carries an always-on `COUNTRY` intercept
//! column (constant `1`) — the factor regression needs a market column
//! whatever else is selected, and with nothing else it degrades to a plain
//! single-factor market model — and, when selected, the [`industry`] dummies.
//! These are the only columns that bypass [`normalize`]: a `0/1` indicator is
//! already normalized, and z-scoring one would destroy the membership it
//! encodes.
//!
//! Fundamental building blocks are derived here from the raw carried report
//! fields in [`MarketData::fields`]. The market data store assets and incomes
//! debit-positive, liabilities, equity and expenses credit-negative; formulas
//! negate them where a positive magnitude is wanted.
//!
//! [`normalize`]: common::normalize
//! [`MarketData::fields`]: crate::data::MarketData::fields

mod alpha101;
mod alpha158;
mod alpha360;
mod barra;
mod cicc_fund;
mod cicc_pv;
mod common;
mod context;
mod industry;

pub use common::{AlphaFeatureSet, RiskFeatureSet, build_features};
