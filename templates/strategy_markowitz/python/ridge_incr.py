"""Incremental (recursive-least-squares) pooled Ridge mean predictor.

Strategy-local, self-contained copy assembled from the `tradingflow` Python
package (`predictor/mean/_incremental.py`, plus `LinearParams` from
`predictor/mean/_base.py`). The strategy loads it with
`py_operator_module("ridge_incr", params)`; edit freely — the next run picks
the change up with no install step.

The operator mirrors the Rust `Operator` trait method-for-method:
`init(inputs) -> state`, `reset(inputs, state) -> outputs` and
`compute(inputs, state, instant) -> outputs`. Inputs are
`(sample_signal, features, target, rebalance_signal, universe)` as owned
NumPy arrays; outputs are `(signal, predictions)`, with `NaN` for stocks the
model could not price. Exceptions panic the node and poison the graph, so
recoverable conditions should be values rather than raises.

A windowed mean predictor re-reads the whole training window and
re-factorizes it every rebalance. The incremental version instead keeps a
**sample pool** of sufficient statistics updated one sampling period at a
time, so the per-rebalance cost is independent of how far back history
reaches.

The design splits into a generic *harness* and a model *pool*:

* `IncrementalMeanPredictor` — the harness. On **every sampling tick** it folds
  the pair that just became formable into the pool (`add_samples`, keyed by a
  monotone index) and drops the pairs that aged out of a rolling window
  (`remove_samples`); on **rebalance ticks** it asks the pool to `predict`.
  Folding eagerly rather than lazily is what bounds retention: each fold takes
  a single freshly-formable pair, so the harness never holds more than
  `target_offset + 1` cross-sections regardless of the window.
* `RlsMeanPool` — the model. It maintains the aggregate Gram, the per-stock
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


class LinearParams:
    """Fitted coefficients with the pooled standardization scaler."""

    __slots__ = ("beta", "intercept", "x_mean", "x_std")

    def __init__(self, beta, intercept, x_mean, x_std):
        self.beta = beta
        self.intercept = intercept
        self.x_mean = x_mean
        self.x_std = x_std


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


class RlsMeanPool:
    """Pooled OLS / Ridge sample pool over a single aggregate Gram."""

    def __init__(
        self,
        *,
        num_features: int,
        alpha: float,
        min_periods: int | None,
        universe_size: int | None,
        window: int | None,
    ) -> None:
        self.alpha = float(alpha)
        self.min_periods = min_periods
        self.universe_size = universe_size
        self.n = 0.0
        self.sx = np.zeros(num_features)
        self.sxx = np.zeros((num_features, num_features))
        self.sy = 0.0
        self.sxy = np.zeros(num_features)
        self.syy = 0.0
        #: Per-stock valid-sample counts, for the `min_periods` filter. Sized
        #: on the first fold, since the cross-section width is not known here.
        self.counts: np.ndarray | None = None
        #: Per-period contributions retained for down-dating, keyed by index —
        #: only when a rolling window is in force. An expanding window never
        #: subtracts anything, so it retains nothing at all.
        self.days: dict | None = {} if window is not None else None
        self.params: LinearParams | None = None
        self.fitted = False

    @staticmethod
    def contribution(x: np.ndarray, y: np.ndarray) -> DayMoments:
        """Reduces one period's cross-section to its moments."""
        valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
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
        if self.counts is None:
            self.counts = np.zeros(x.shape[0])
        day = self.contribution(x, y)
        self.apply(day, 1.0)
        if self.days is not None:
            self.days[index] = day

    def remove_samples(self, index: int) -> None:
        self.apply(self.days.pop(index), -1.0)

    def predict(self, universe: np.ndarray, features: np.ndarray, refit: bool) -> np.ndarray:
        if (refit or not self.fitted) and self.n > 0:
            self.params = solve_standardized(self.n, self.sx, self.sxx, self.sy, self.sxy, self.syy, self.alpha)
            self.fitted = True

        mask = universe > 0
        if self.universe_size is not None:
            assert int(mask.sum()) <= self.universe_size, (
                f"universe has {int(mask.sum())} nonzero entries, " f"exceeding universe_size={self.universe_size}"
            )
        if self.min_periods is not None and self.counts is not None:
            mask = mask & (self.counts >= self.min_periods)
        mask = mask & np.isfinite(features).all(axis=1)

        out = np.full(universe.shape[0], np.nan)
        if self.fitted and mask.any():
            p = self.params
            out[mask] = (features[mask] - p.x_mean) / p.x_std @ p.beta + p.intercept
        return out


@dataclass(slots=True)
class IncrementalState:
    """Everything the incremental predictor carries between generations.

    `init` moves the whole build configuration in here so that `reset` and
    `compute` are free functions of `(inputs, state)` — mirroring the Rust
    `Operator`, and keeping every node's data its own even when a shared
    operator instance is loaded from `__op__`.
    """

    pool: RlsMeanPool
    target_offset: int
    refit_every: int
    #: The rolling training window in sampling periods, or `None` to expand.
    window: int | None
    #: The last `target_offset + 1` feature cross-sections — just enough to
    #: pair the oldest with the target that has now landed.
    features: deque = field(default_factory=deque)
    out: np.ndarray | None = None
    added: int = 0
    removed: int = 0
    rebalances: int = 0


class IncrementalMeanPredictor:
    type Inputs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    type Outputs = tuple[bool, np.ndarray]
    type Context = int
    type State = IncrementalState

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

    def init(self, inputs) -> IncrementalState:
        _, features, _, _, universe = inputs
        return IncrementalState(
            pool=RlsMeanPool(
                num_features=features.shape[1],
                alpha=self.alpha,
                min_periods=self.min_periods,
                universe_size=self.universe_size,
                window=self.window,
            ),
            target_offset=self.target_offset,
            refit_every=self.refit_every,
            window=self.window,
            features=deque(maxlen=self.target_offset + 1),
            out=np.full(universe.shape[0], np.nan),
        )

    @staticmethod
    def reset(_, state: IncrementalState):
        return (False, state.out)

    @staticmethod
    def compute(inputs, state: IncrementalState, _):
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

        refit = (state.rebalances % state.refit_every) == 0
        state.rebalances += 1
        state.out = state.pool.predict(universe, state.features[-1], refit)
        return (True, state.out)


def build(*, alpha: float = 1.0, **kwargs) -> IncrementalMeanPredictor:
    """Constructs an incremental pooled Ridge mean predictor."""
    return IncrementalMeanPredictor(alpha=alpha, **kwargs)
