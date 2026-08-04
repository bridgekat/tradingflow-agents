"""Incremental ridge regression mean predictor.

A windowed mean predictor re-reads the whole training window and
re-factorizes it every rebalance. The incremental version instead keeps a
**sample pool** of sufficient statistics updated one sampling period at a
time, so the per-rebalance cost is independent of how far back history
reaches.

The design splits into a generic *harness* and a model *pool* — the same
shape as the other two operators (harness gates, masks and scatters; the
model is pure math):

* `MeanPredictor` — the harness. On **every sampling tick** it folds
  the pair that just became formable into the pool (`add_samples`, keyed by a
  monotone index) and drops the pairs that aged out of a rolling window
  (`remove_samples`); on **rebalance ticks** it refits on the cadence, asks
  the pool to `predict`, and emits through the universe/coverage mask.
  Folding eagerly rather than lazily is what bounds retention: each fold takes
  a single freshly-formable pair, so the harness never holds more than
  `target_offset + 1` cross-sections regardless of the window.
* `RLSMeanPool` — the model. It maintains the aggregate Gram, the per-stock
  valid-sample counts and the fitted parameters.

**Fit on all samples.** The pool fits on every stock with valid features; the
universe only selects which predictions are *emitted*. Whole-market fitting
needs just one aggregate Gram, folded a period at a time via `XᵀX` — a BLAS
GEMM with `O(N F)` memory traffic and no `(N, F, F)` per-stock array. Each
refit is then an `O(F³)` standardized-Gram solve, independent of both history
length and stock count. Per-stock counts are kept only for the `min_periods`
prediction filter.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np


def active_mask(universe: np.ndarray, universe_size: int | None) -> np.ndarray:
    """The stocks the universe marks active, checked against the sizing bound."""
    mask = universe > 0
    if universe_size is not None:
        assert int(mask.sum()) <= universe_size, (
            f"universe has {int(mask.sum())} nonzero entries, " f"exceeding universe_size={universe_size}"
        )
    return mask


#: Standardized features are clamped to this many pooled standard deviations
#: at prediction. The panel's winsorization is per-cross-section and the
#: scaler is pooled over history, so a regime-shifted day can standardize to
#: an astronomical z-score — which, pushed through the coefficients, overflows
#: downstream consumers (e.g. `expm1` in the portfolio's lognormal-to-linear
#: mapping). Inside the band the transform is the identity, so a sane
#: cross-section is untouched; a tail day contributes at most `Z_CLIP` per
#: feature, bounding the prediction at `Z_CLIP * ‖beta‖₁` around the
#: intercept.
Z_CLIP = 5.0

#: A feature value at or beyond this magnitude marks its whole row invalid,
#: exactly like a NaN — for the training pool and the prediction mask alike.
#: The panel's winsorization is per-cross-section, so a day where over 1% of
#: the section spikes (near-zero denominators in the ported catalogs) can
#: pass through values that are absurd as regression inputs: they would
#: dominate the pooled scaler, and their squares can overflow the Gram to
#: `inf` — which a later rolling-window down-date turns into a permanent
#: `NaN` (`inf - inf`). The limit sits far above every legitimate feature
#: scale while keeping `x²` sums comfortably inside `f64`.
FEATURE_LIMIT = 1e8


def valid_rows(x: np.ndarray) -> np.ndarray:
    """Rows of the feature cross-section that are usable: every entry finite
    and inside the [`FEATURE_LIMIT`] scale (a NaN or ±inf compares `False`)."""
    return (np.abs(x) < FEATURE_LIMIT).all(axis=1)


@dataclass(slots=True)
class LinearParams:
    """Fitted coefficients with the pooled standardization scaler."""

    beta: np.ndarray
    intercept: float
    x_mean: np.ndarray
    x_std: np.ndarray


def solve_standardized(
    n: float,
    sx: np.ndarray,
    sxx: np.ndarray,
    sy: float,
    sxy: np.ndarray,
    syy: float,
    alpha: float,
) -> LinearParams:
    """Solves pooled standardized OLS (`alpha=0`) or Ridge from raw moments.

    Reconstructs the pooled mean and standard deviation and the standardized
    Gram, then solves the normal equations `(ZᵀZ + α n I) β = Zᵀy` — the
    sample-size-invariant `(1/n)‖·‖² + α‖β‖²` penalty, reached from moments
    that can be updated incrementally.
    """
    f = sx.shape[0]
    if n <= 0:
        return LinearParams(np.zeros(f), 0.0, np.zeros(f), np.ones(f))

    # Pooled mean / population std, with a constant-column fallback of 1 so
    # constant features standardize to zero and contribute nothing.
    x_mean = sx / n
    x_std = np.sqrt(np.maximum(np.diag(sxx) / n - x_mean**2, 0.0))
    x_std = np.where(x_std > 0, x_std, 1.0)
    y_mean = sy / n
    y_std = float(np.sqrt(max(syy / n - y_mean**2, 0.0)))
    y_std = y_std if y_std > 0 else 1.0

    fallback = LinearParams(np.zeros(f), y_mean, x_mean, x_std)

    ztz = (sxx - n * np.outer(x_mean, x_mean)) / np.outer(x_std, x_std)
    zty = (sxy - n * x_mean * y_mean) / (x_std * y_std)
    try:
        beta = np.linalg.solve(ztz + alpha * n * np.eye(f), zty)
    except np.linalg.LinAlgError:
        return fallback
    if not np.all(np.isfinite(beta)):
        return fallback

    return LinearParams(y_std * beta, y_mean, x_mean, x_std)


@dataclass(slots=True)
class DayMoments:
    """One sampling period's contribution to the aggregate Gram.

    A rolling window has to subtract a period back out when it ages, and the
    obvious way is to keep that period's raw `(N, F)` cross-section and refold
    it with the sign flipped. Keeping the *contribution* instead is exactly
    equivalent — the fold is linear — and costs `O(F²)` rather than `O(N F)`,
    which on a wide cross-section is the difference between kilobytes and
    megabytes per retained period. It is also the more accurate of the two:
    subtracting the very values that were added cancels them exactly, where
    recomputing from the raw rows need not.
    """

    n: float
    sx: np.ndarray
    sxx: np.ndarray
    sy: float
    sxy: np.ndarray
    syy: float
    #: Which stocks this period counted toward, for the `min_periods` filter.
    valid: np.ndarray


class RLSMeanPool:
    """Pooled OLS / Ridge sample pool over a single aggregate Gram.

    Pure model math: folding, down-dating, refitting and predicting. Which
    stocks are worth *emitting* is the harness's concern.
    """

    def __init__(self, *, num_stocks: int, num_features: int, alpha: float) -> None:
        self.alpha = float(alpha)
        self.n = 0.0
        self.sx = np.zeros(num_features)
        self.sxx = np.zeros((num_features, num_features))
        self.sy = 0.0
        self.sxy = np.zeros(num_features)
        self.syy = 0.0
        #: Per-stock valid-sample counts, for the `min_periods` filter.
        self.counts = np.zeros(num_stocks)
        #: Per-period contributions retained for down-dating, keyed by index.
        self.days: dict[int, DayMoments] = {}
        self.params: LinearParams | None = None

    @staticmethod
    def contribution(x: np.ndarray, y: np.ndarray) -> DayMoments:
        """Reduces one period's cross-section to its moments."""
        valid = valid_rows(x) & np.isfinite(y)
        xr, yr = x[valid], y[valid]
        return DayMoments(
            n=float(xr.shape[0]),
            sx=xr.sum(axis=0),
            sxx=xr.T @ xr,
            sy=float(yr.sum()),
            sxy=xr.T @ yr,
            syy=float(yr @ yr),
            valid=valid,
        )

    def apply(self, day: DayMoments, sign: float) -> None:
        """Folds (`sign=+1`) or unfolds (`sign=-1`) a period's moments."""
        self.n += sign * day.n
        self.sx += sign * day.sx
        self.sxx += sign * day.sxx
        self.sy += sign * day.sy
        self.sxy += sign * day.sxy
        self.syy += sign * day.syy
        self.counts += sign * day.valid

    def add_samples(self, index: int, x: np.ndarray, y: np.ndarray) -> None:
        day = self.contribution(x, y)
        self.apply(day, 1.0)
        self.days[index] = day

    def remove_samples(self, index: int) -> None:
        self.apply(self.days.pop(index), -1.0)

    def fit(self) -> None:
        """Re-solves the pooled Ridge from the current aggregate moments."""
        if self.n > 0:
            self.params = solve_standardized(self.n, self.sx, self.sxx, self.sy, self.sxy, self.syy, self.alpha)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predictions for every stock (`NaN` rows where features are), with
        the standardized features clamped to the [`Z_CLIP`] band."""
        p = self.params
        if p is None:
            return np.full(features.shape[0], np.nan)
        z = np.clip((features - p.x_mean) / p.x_std, -Z_CLIP, Z_CLIP)
        return z @ p.beta + p.intercept


@dataclass(slots=True)
class MeanPredictorState:
    """Everything the incremental predictor carries between generations.

    `init` moves the whole build configuration in here so that `reset` and
    `compute` are free functions of `(inputs, state)` — mirroring the Rust
    `Operator`, and keeping every node's data its own even when a shared
    operator instance is loaded from `__op__`.
    """

    pool: RLSMeanPool
    target_offset: int
    refit_every: int
    #: The rolling training window in sampling periods, or `None` to expand.
    window: int | None
    min_periods: int | None
    universe_size: int | None
    #: The last `target_offset + 1` feature cross-sections — just enough to
    #: pair the oldest with the target that has now landed.
    features: deque = field(default_factory=deque)
    #: The retained prediction, re-emitted on every non-rebalance tick.
    out: np.ndarray | None = None
    added: int = 0
    removed: int = 0
    rebalances: int = 0


class MeanPredictor:
    type Inputs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    type Outputs = tuple[bool, np.ndarray]
    type Context = int
    type State = MeanPredictorState

    def __init__(
        self,
        *,
        alpha: float = 0.0,
        target_offset: int = 0,
        refit_every: int = 1,
        max_periods: int | None = None,
        min_periods: int | None = None,
        universe_size: int | None = None,
    ) -> None:
        assert alpha >= 0.0, "alpha must be non-negative"
        assert target_offset >= 0, "target_offset must be non-negative"
        assert refit_every >= 1, "refit_every must be >= 1"
        assert max_periods is None or max_periods >= 1, "max_periods must be >= 1"

        self.alpha = float(alpha)
        self.target_offset = int(target_offset)
        self.refit_every = int(refit_every)
        # The training window, as for the windowed predictors — here it selects
        # which pairs are down-dated out of the pool rather than which are
        # sliced out of a deque.
        self.window = None if max_periods is None else int(max_periods)
        self.min_periods = min_periods
        self.universe_size = universe_size

    def init(self, inputs: Inputs) -> State:
        _, features, _, _, universe = inputs
        return MeanPredictorState(
            pool=RLSMeanPool(
                num_stocks=universe.shape[0],
                num_features=features.shape[1],
                alpha=self.alpha,
            ),
            target_offset=self.target_offset,
            refit_every=self.refit_every,
            window=self.window,
            min_periods=self.min_periods,
            universe_size=self.universe_size,
            features=deque(maxlen=self.target_offset + 1),
            out=np.full(universe.shape[0], np.nan),
        )

    @staticmethod
    def reset(_: Inputs, state: State) -> Outputs:
        assert state.out is not None
        return (False, state.out)

    @staticmethod
    def compute(inputs: Inputs, state: State, _: Context) -> Outputs:
        assert state.out is not None
        sample_signal, features, target, rebalance_signal, universe = inputs

        if sample_signal:
            state.features.append(features)
            # A pair becomes formable once the deque spans the offset: its
            # oldest entry is the cross-section this target belongs to.
            if len(state.features) == state.target_offset + 1:
                state.pool.add_samples(state.added, state.features[0], target)
                state.added += 1

            # Rolling window: drop whatever has aged out. The pool down-dates
            # from its own retained copies, so this is independent of what the
            # harness still holds.
            if state.window is not None:
                start = max(0, state.added - state.window)
                for i in range(state.removed, start):
                    state.pool.remove_samples(i)
                state.removed = start

        if not rebalance_signal:
            return (False, state.out)

        if not state.features:
            state.out = np.full(universe.shape[0], np.nan)
            return (True, state.out)

        # Refit on the cadence, then emit through the mask: in the universe,
        # enough coverage, usable current features. Everything else stays
        # `NaN` — the predictor saying it has no opinion.
        refit = (state.rebalances % state.refit_every) == 0
        state.rebalances += 1
        if refit or state.pool.params is None:
            state.pool.fit()

        current = state.features[-1]
        mask = active_mask(universe, state.universe_size)
        if state.min_periods is not None:
            mask &= state.pool.counts >= state.min_periods
        mask &= valid_rows(current)

        out = np.full(universe.shape[0], np.nan)
        if mask.any():
            out[mask] = state.pool.predict(current)[mask]
        state.out = out
        return (True, state.out)


def build(*, alpha: float = 1.0, **kwargs) -> MeanPredictor:
    return MeanPredictor(alpha=alpha, **kwargs)
