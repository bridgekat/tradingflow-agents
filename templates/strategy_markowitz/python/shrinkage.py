"""Windowed linear-shrinkage covariance predictor.

Strategy-local, self-contained copy assembled from the `tradingflow` Python
package (`predictor/_panel.py`, `predictor/variance/_base.py`,
`predictor/variance/_common.py` and `predictor/variance/shrinkage.py`). The
strategy loads it with `py_operator_module("shrinkage", params)`; edit freely
— the next run picks the change up with no install step.

The operator mirrors the Rust `Operator` trait method-for-method:
`init(inputs) -> state`, `reset(inputs, state) -> outputs` and
`compute(inputs, state, instant) -> outputs`. Inputs are
`(sample_signal, features, target, rebalance_signal, universe)` as owned
NumPy arrays; outputs are `(signal, covariance)` — an `[N, N]` matrix with
`NaN` outside the active block. Exceptions panic the node and poison the
graph, so recoverable conditions should be values rather than raises.

The file has three layers, top to bottom:

1. The **windowed panel harness** (`PanelState`, `PanelPredictor`): record
   sampled cross-sections into a bounded window, refit on a cadence, emit one
   prediction per rebalance, masking by universe and coverage.
2. The **covariance plumbing** (`VariancePanelState`, `covariance_predictor`):
   a covariance predictor fits from the target panel alone, so it never
   retains the feature window, and its prediction *is* the fitted matrix.
3. The **estimator** (`fit`, the shrinkage targets): a linear shrinkage
   `Σ = αF + (1 - α)S` of the NaN-robust sample covariance `S` toward a
   structured target `F`, with the intensity `α` estimated analytically by
   the Schäfer-Strimmer (2005) element-wise unbiased estimator (the three
   targets surveyed in Pantaleo et al. 2010, Section III.D).
"""

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from itertools import islice
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# The windowed panel harness.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PanelState:
    """Everything a windowed predictor carries between generations.

    This mirrors the Rust `Operator::State`: `init` is the only method that
    reads the operator, everything it needs is copied here, and the subclass
    hooks hang off the state too — so `reset` and `compute` are free functions
    of `(inputs, state)`, and every node's data is its own even when a shared
    operator instance is loaded from `__op__`. `slots=True` turns a stray
    attribute into an `AttributeError` rather than a silent cross-node alias.
    """

    #: `(x, y) -> params` over the training block, and
    #: `(features, params) -> predictions` over the active cross-section.
    fit: Callable
    predict: Callable
    target_offset: int
    refit_every: int
    max_periods: int | None
    min_periods: int | None
    universe_size: int | None
    #: Whether the fit needs a *window* of past feature cross-sections. When
    #: false the window is never accumulated, `fit` is handed `x=None`,
    #: coverage counts come from the target alone, and no stock is excluded
    #: for having unusable features — a model that never looks at them has no
    #: grounds to. On a wide panel that is the difference between megabytes
    #: and gigabytes of retained history.
    retain_features: bool
    #: Recorded `(N, F)` feature cross-sections, newest last. Stays empty
    #: unless `retain_features`.
    features: deque
    #: Recorded `(N,)` target cross-sections, newest last.
    target: deque
    #: The retained prediction, re-emitted on every non-rebalance tick.
    out: np.ndarray
    params: object = None
    fitted: bool = False
    rebalances: int = 0

    @staticmethod
    def empty(n: int) -> np.ndarray:
        """An all-`NaN` output for a cross-section of `n` stocks."""
        raise NotImplementedError

    @staticmethod
    def scatter(out: np.ndarray, mask: np.ndarray, values: np.ndarray) -> None:
        """Places the masked stocks' `values` into `out`."""
        raise NotImplementedError


def window(dq: deque, start: int, stop: int) -> list:
    """The `[start, stop)` slice of a deque, as a list of cross-sections."""
    return list(islice(dq, max(start, 0), max(stop, 0)))


class PanelPredictor:
    r"""Windowed panel predictor: record, refit on a cadence, predict.

    On each sampling tick one `(features, target)` cross-section pair is
    appended to a bounded window. On each rebalance tick the window is
    flattened into a training block — `features[i]` paired with
    `target[i + target_offset]` — the model is refit if the cadence is due, and
    a prediction is emitted for every stock that passes the mask.

    A stock is in the mask when it is in the universe, has at least
    `min_periods` valid observations in the window, and — for a model that
    reads features — has finite current features. Everything else stays at
    `NaN`, which is how downstream portfolios and metrics recognise a stock the
    model could not price.
    """

    #: The [`PanelState`] subclass this predictor builds, carrying the output
    #: shape, the scatter rule and whether features are read.
    state_type: type[PanelState] = PanelState

    def __init__(
        self,
        *,
        fit,
        predict,
        retain_features: bool = True,
        target_offset: int = 0,
        refit_every: int = 1,
        max_periods: int | None = None,
        min_periods: int | None = None,
        universe_size: int | None = None,
    ) -> None:
        assert target_offset >= 0, "target_offset must be non-negative"
        assert refit_every >= 1, "refit_every must be >= 1"
        assert max_periods is None or max_periods >= 1, "max_periods must be >= 1"
        assert min_periods is None or min_periods >= 1, "min_periods must be >= 1"

        self.fit = fit
        self.predict = predict
        self.retain_features = retain_features
        self.target_offset = int(target_offset)
        self.refit_every = int(refit_every)
        self.max_periods = max_periods
        self.min_periods = min_periods
        self.universe_size = universe_size

    def init(self, inputs) -> PanelState:
        *_, universe = inputs
        # A bounded deque does the trimming: forming `max_periods` pairs needs
        # `max_periods + target_offset` cross-sections once the forward offset
        # is accounted for, and the oldest fall off the left as they age out.
        maxlen = None if self.max_periods is None else self.max_periods + self.target_offset
        return self.state_type(
            fit=self.fit,
            predict=self.predict,
            retain_features=self.retain_features,
            target_offset=self.target_offset,
            refit_every=self.refit_every,
            max_periods=self.max_periods,
            min_periods=self.min_periods,
            universe_size=self.universe_size,
            features=deque(maxlen=maxlen),
            target=deque(maxlen=maxlen),
            out=self.state_type.empty(universe.shape[0]),
        )

    @staticmethod
    def reset(_, state: PanelState):
        return (False, state.out)

    @staticmethod
    def compute(inputs, state: PanelState, _):
        sample_signal, features, target, rebalance_signal, universe = inputs

        if sample_signal:
            state.target.append(target)
            if state.retain_features:
                state.features.append(features)

        if not rebalance_signal:
            return (False, state.out)

        n = universe.shape[0]
        # Emit on every rebalance whatever happens, so downstream metrics see
        # one prediction per period; an unfittable panel emits all-NaN.
        out = state.empty(n)
        if not state.target:
            state.out = out
            return (True, state.out)

        # Pairs run features[i] with target[i + target_offset]; the training
        # block is the last `n_use` of them, which — since the deques are
        # trimmed together — ends at the newest target.
        length = len(state.target)
        n_pair = max(0, length - state.target_offset)
        n_use = n_pair if state.max_periods is None else min(n_pair, state.max_periods)

        counts = np.zeros(n)
        x = y = None
        if n_use > 0:
            y = np.stack(window(state.target, length - n_use, length))  # (T, N)
            valid = np.isfinite(y)
            if state.retain_features:
                x = np.stack(window(state.features, n_pair - n_use, n_pair))  # (T, N, F)
                valid &= np.isfinite(x).all(axis=2)
            counts = valid.sum(axis=0)

        current = state.features[-1] if state.retain_features else None

        mask = universe > 0
        if state.universe_size is not None:
            assert int(mask.sum()) <= state.universe_size, (
                f"universe has {int(mask.sum())} nonzero entries, " f"exceeding universe_size={state.universe_size}"
            )
        if state.min_periods is not None:
            mask = mask & (counts >= state.min_periods)
        if current is not None:
            mask = mask & np.isfinite(current).all(axis=1)

        # Refit on the cadence, or whenever there is still nothing to predict
        # with; otherwise reuse the parameters from the last refit.
        refit = (not state.fitted) or (state.rebalances % state.refit_every == 0)
        state.rebalances += 1
        if refit and n_use > 0 and mask.any():
            state.params = state.fit(x[:, mask, :] if x is not None else None, y[:, mask])
            state.fitted = True

        if state.fitted and mask.any():
            active = current[mask] if current is not None else None
            state.scatter(out, mask, state.predict(active, state.params))
        state.out = out
        return (True, state.out)


# ---------------------------------------------------------------------------
# Covariance predictor plumbing.
# ---------------------------------------------------------------------------


def predict(features: np.ndarray | None, params: np.ndarray) -> np.ndarray:
    """Returns the fitted covariance block unchanged.

    `features` is always `None` here: a covariance predictor is never handed a
    feature cross-section, because it never asked for one.
    """
    return params


@dataclass(slots=True)
class VariancePanelState(PanelState):
    """A covariance predictor emits a matrix over the active stocks."""

    @staticmethod
    def empty(n: int) -> np.ndarray:
        return np.full((n, n), np.nan)

    @staticmethod
    def scatter(out: np.ndarray, mask: np.ndarray, values: np.ndarray) -> None:
        out[np.ix_(mask, mask)] = values


class VariancePredictor(PanelPredictor):
    type Inputs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    type Outputs = tuple[bool, np.ndarray]
    type Context = int
    type State = VariancePanelState

    state_type = VariancePanelState


def covariance_predictor(fit, *, retain_features: bool = False, **kwargs) -> VariancePredictor:
    """Wraps a `(T, N)` target panel estimator into a covariance predictor.

    Every estimator here ignores features and fits from the target panel
    alone, so `retain_features` defaults off and the harness skips the feature
    window entirely.
    """
    return VariancePredictor(
        fit=lambda x, y: fit(y),
        predict=predict,
        retain_features=retain_features,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# NaN-robust covariance building blocks.
# ---------------------------------------------------------------------------


def sample_covariance(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NaN-robust sample covariance using pairwise complete observations.

    Takes the `(T, N)` return panel and returns `(S, centered, finite)`: the
    `(N, N)` sample covariance, the mean-centered returns with non-finite
    entries zeroed (so they contribute nothing to sums), and the boolean mask
    of originally-finite entries.
    """
    mean = np.nanmean(y, axis=0)
    centered = y - mean
    finite = np.isfinite(centered)
    centered = np.where(finite, centered, 0.0)
    indicator = finite.astype(np.float64)
    counts = indicator.T @ indicator
    S = (centered.T @ centered) / np.maximum(counts - 1.0, 1.0)
    return S, centered, finite


def correlation_from_covariance(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample correlation matrix and standard deviations from a covariance.

    The diagonal is forced to 1 and entries are clipped to `[-1, 1]`.
    """
    diag = np.maximum(np.diag(S), 0.0)
    stds = np.sqrt(diag)
    stds_safe = np.where(stds > 0, stds, 1.0)
    C = S / np.outer(stds_safe, stds_safe)
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1.0, 1.0), stds


def single_index_covariance(y: np.ndarray) -> np.ndarray:
    r"""Single-index factor-model covariance estimator.

    Fits the factor model
    \(r_i(t) = \alpha_i + \beta_i f(t) + \epsilon_i(t)\) stock-by-stock
    against the equal-weighted cross-sectional mean return \(f(t)\),
    and returns

    \[
    \Sigma = \sigma_f^{2} \beta \beta^T + \mathrm{diag}(\sigma_\epsilon^{2}).
    \]

    All statistics are computed NaN-robustly from pairs of finite
    observations. Returns the zero matrix if the factor is never observable.
    """
    _, N = y.shape

    # Equal-weighted cross-sectional mean as the market-factor proxy.
    f = np.nanmean(y, axis=1)

    # Time-series centering.
    y_c = y - np.nanmean(y, axis=0)
    f_c = f - np.nanmean(f)

    # Keep only rows where the factor is observable.
    f_valid = np.isfinite(f_c)
    if not f_valid.any():
        return np.zeros((N, N), dtype=np.float64)
    y_c = y_c[f_valid]
    f_c = f_c[f_valid]

    # Per-stock finiteness (the factor is already finite on these rows).
    valid = np.isfinite(y_c)
    y_fill = np.where(valid, y_c, 0.0)
    f_mat = np.where(valid, f_c[:, None], 0.0)

    # OLS beta per stock using only pairs where y_i is observed.
    num = (y_fill * f_mat).sum(axis=0)
    den = (f_mat * f_mat).sum(axis=0)
    beta = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)

    # Residual (idiosyncratic) variances.
    resid = y_fill - beta[None, :] * f_mat
    counts = valid.sum(axis=0)
    resid_ss = (resid * resid).sum(axis=0)
    sigma_eps_sq = np.where(counts > 2, resid_ss / np.maximum(counts - 2.0, 1.0), 0.0)

    # Market-factor variance.
    sigma_f_sq = float((f_c * f_c).sum() / max(len(f_c) - 1, 1))

    F = sigma_f_sq * np.outer(beta, beta)
    F[np.diag_indices(N)] += sigma_eps_sq
    return F


def schafer_strimmer_alpha(
    S: np.ndarray,
    F: np.ndarray,
    centered: np.ndarray,
    finite: np.ndarray,
) -> tuple[float, int]:
    r"""Schäfer-Strimmer optimal linear-shrinkage intensity.

    Implements the analytic unbiased estimator of Schäfer & Strimmer
    (2005, *Statistical Applications in Genetics and Molecular Biology*),
    as used by Pantaleo et al. (2010, arXiv:1004.4272):

    \[
    \alpha^* = \frac{\sum_{i \ne j} \widehat{\mathrm{Var}}(s_{ij})}
                    {\sum_{i \ne j} (s_{ij} - f_{ij})^2}
    \]

    with the element-wise variance estimator computed pairwise over jointly
    observable timesteps. The sum is restricted to off-diagonal elements;
    diagonal (sample variance) elements are well-estimated and would
    otherwise dominate the numerator. Returns `(alpha, T_eff)` — the
    intensity clipped to `[0, 1]` and the number of rows with at least one
    finite observation.
    """
    N = S.shape[0]
    T_eff = int(finite.any(axis=1).sum())
    if T_eff < 2 or N < 2:
        return 1.0, T_eff

    # Pairwise counts of jointly-observable timesteps.
    indicator = finite.astype(np.float64)
    counts = indicator.T @ indicator  # (N, N)

    # w_ij(t) = z_i(t) z_j(t); rows with NaN are already zeroed in centered.
    sum_w = centered.T @ centered  # (N, N), Σ_t w_ij(t) over valid t
    centered_sq = centered * centered
    sum_w_sq = centered_sq.T @ centered_sq  # (N, N), Σ_t w_ij(t)^2 over valid t

    # Σ_t (w_ij(t) - w̄_ij)^2 = Σ_t w_ij(t)^2 - T_ij · w̄_ij^2
    # with w̄_ij = sum_w / T_ij (pairwise-mean).
    counts_safe = np.maximum(counts, 1.0)
    centered_sum_sq = sum_w_sq - sum_w * sum_w / counts_safe

    # V̂ar(s_ij) = T_ij / (T_ij - 1)^3 · centered_sum_sq.  Pairs with
    # fewer than 2 valid observations contribute nothing.
    dof_cube = np.maximum((counts - 1.0) ** 3, 1.0)
    var_s = np.where(counts >= 2, counts * centered_sum_sq / dof_cube, 0.0)

    off_diag = ~np.eye(N, dtype=bool)
    numerator = float(var_s[off_diag].sum())
    denominator = float(((S - F) ** 2)[off_diag].sum())

    if denominator < 1e-30:
        return 1.0, T_eff
    return float(np.clip(numerator / denominator, 0.0, 1.0)), T_eff


# ---------------------------------------------------------------------------
# The linear-shrinkage estimator.
# ---------------------------------------------------------------------------


class Target(IntEnum):
    r"""Shrinkage target selector — the three surveyed in Pantaleo et al.
    (2010), Section III.D.

    - `COMMON_COVARIANCE`: diagonal is the average sample variance,
      off-diagonal the average sample covariance.
    - `CONSTANT_CORRELATION`: diagonal is the sample variances, off-diagonal
      the average off-diagonal correlation times the outer product of the
      standard deviations.
    - `SINGLE_INDEX`: the single-index factor-model covariance.
    """

    COMMON_COVARIANCE = 1
    CONSTANT_CORRELATION = 2
    SINGLE_INDEX = 3


def common_covariance_target(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    n = s.shape[0]
    off = ~np.eye(n, dtype=bool)
    avg_var = float(np.mean(np.diag(s)))
    target = np.full((n, n), float(s[off].mean()) if n > 1 else avg_var)
    np.fill_diagonal(target, avg_var)
    return target


def constant_correlation_target(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    n = s.shape[0]
    corr, stds = correlation_from_covariance(s)
    off = ~np.eye(n, dtype=bool)
    r_bar = float(corr[off].mean()) if n > 1 else 1.0
    target = r_bar * np.outer(stds, stds)
    np.fill_diagonal(target, np.diag(s))
    return target


TARGETS = {
    Target.COMMON_COVARIANCE: common_covariance_target,
    Target.CONSTANT_CORRELATION: constant_correlation_target,
    Target.SINGLE_INDEX: lambda y, s: single_index_covariance(y),
}


def fit(y: np.ndarray, *, target: Target) -> np.ndarray:
    r"""Linear shrinkage \(\Sigma = \alpha F + (1 - \alpha) S\) of the sample
    covariance \(S\) toward a structured target \(F\), with the intensity
    \(\alpha\) estimated analytically by the Schäfer-Strimmer (2005)
    element-wise unbiased estimator."""
    s, centered, finite = sample_covariance(y)
    f = TARGETS[target](y, s)
    alpha, _ = schafer_strimmer_alpha(s, f, centered, finite)
    return alpha * f + (1.0 - alpha) * s


def build(*, target: int | Target = Target.COMMON_COVARIANCE, **kwargs) -> VariancePredictor:
    """Constructs a linear-shrinkage covariance predictor."""
    target = Target(target)
    return covariance_predictor(lambda y: fit(y, target=target), **kwargs)
