"""Windowed linear-shrinkage variance predictor.

Outputs are `(signal, risk)`, where `risk` is a `[factor_rank + 1, N]` factor
panel, `NaN` outside the active block: rows `[0, factor_rank)` are the factor
loadings `B.T` and the last row is the idiosyncratic std-dev `d`, so
`Sigma ~= B @ B.T + diag(d**2)` over the active stocks.

The file has two layers, top to bottom:

1. `VariancePredictor` - the windowed harness. Records sampled return
   cross-sections into a bounded window; on each rebalance, refit on the
   cadence over the universe/coverage mask and emit the fitted panel
   (retained between refits).
2. The **estimator** (`factor_fit`, the shrinkage targets): a linear
   shrinkage `Σ = αF + (1 - α)S` of the sample covariance `S` toward a
   structured target `F` (the three targets surveyed in Pantaleo et al.
   2010, Section III.D), with the intensity `α` estimated by the
   Schäfer-Strimmer (2005) element-wise unbiased estimator on a column
   subsample, and the top-`factor_rank` eigenpairs extracted by Lanczos
   iteration on the *implicit* `Σ` — matvecs against the `(T, m)` return
   panel — so nothing `O(m²·T)` or `O(m²)`-dense is ever formed for large
   active blocks.
"""

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh


def active_mask(universe: np.ndarray, universe_size: int | None) -> np.ndarray:
    """The stocks the universe marks active, checked against the sizing bound."""
    mask = universe > 0
    if universe_size is not None:
        assert int(mask.sum()) <= universe_size, (
            f"universe has {int(mask.sum())} nonzero entries, " f"exceeding universe_size={universe_size}"
        )
    return mask


# ---------------------------------------------------------------------------
# The implicit linear-shrinkage estimator.
# ---------------------------------------------------------------------------
#
# Everything below works against the `(T, m)` active return panel without
# forming any dense `(m, m)` matrix for large `m`:
#
# - The sample covariance appears only through matvecs
#   `S @ x = Z.T @ (Z @ x) / dof` against the centered, zero-filled,
#   per-stock variance-corrected panel `Z`.
# - Every shrinkage target is rank-<=1-plus-diagonal-plus-constant, so its
#   matvec and diagonal are `O(m)`.
# - The shrinkage intensity is the Schäfer-Strimmer estimator evaluated
#   exactly on a random *column subsample* — the intensity is a single global
#   scalar, so a few hundred columns estimate the two pair-sums it needs.
# - The top eigenpairs of the implicit `Sigma = alpha F + (1 - alpha) S`
#   come from Lanczos (`scipy.sparse.linalg.eigsh` on a `LinearOperator`).
#
# Approximations versus the textbook dense estimator, all deliberate:
# normalization uses a common `dof = T_eff - 1` with per-stock variance
# correction `sqrt(dof / (c_i - 1))` instead of pairwise-complete counts, and
# the intensity is a subsample estimate. Both disappear as coverage fills in,
# and the factor-model truncation downstream dwarfs either.


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


@dataclass(slots=True)
class ImplicitTarget:
    r"""A shrinkage target in `a * u u^T + diag(g) + c * 1 1^T` form.

    Every surveyed target fits this shape, so its matvec, diagonal and
    arbitrary columns are all `O(m)` per vector.
    """

    a: float
    u: np.ndarray
    g: np.ndarray
    c: float

    def matvec(self, x: np.ndarray) -> np.ndarray:
        return self.a * self.u * (self.u @ x) + self.g * x + self.c * x.sum()

    def diagonal(self) -> np.ndarray:
        return self.a * self.u * self.u + self.g

    def columns(self, idx: np.ndarray) -> np.ndarray:
        """The dense columns at `idx`, for the subsampled intensity."""
        cols = self.a * np.outer(self.u, self.u[idx]) + self.c
        cols[idx, np.arange(len(idx))] += self.g[idx]
        return cols


def build_target(
    target: Target,
    z: np.ndarray,
    finite: np.ndarray,
    counts: np.ndarray,
    dof: float,
) -> ImplicitTarget:
    """Builds the implicit shrinkage target from the centered panel.

    `z` is the centered, zero-filled, variance-corrected panel; `finite` its
    observation mask; `counts` the per-stock observation counts; `dof` the
    common degrees of freedom. All statistics are `O(T * m)`.
    """
    m = z.shape[1]
    diag_s = (z * z).sum(axis=0) / dof

    if target == Target.SINGLE_INDEX:
        # Equal-weighted cross-sectional mean return as the market factor,
        # with per-stock OLS betas over jointly-observed days.
        row_counts = finite.sum(axis=1)
        f = np.where(row_counts > 0, z.sum(axis=1) / np.maximum(row_counts, 1), 0.0)
        f_masked = np.where(finite, f[:, None], 0.0)
        num = (z * f_masked).sum(axis=0)
        den = (f_masked * f_masked).sum(axis=0)
        beta = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)
        resid_ss = np.maximum((z * z).sum(axis=0) - 2.0 * beta * num + beta * beta * den, 0.0)
        sigma_eps_sq = np.where(counts > 2, resid_ss / np.maximum(counts - 2.0, 1.0), diag_s)
        nf = int((row_counts > 0).sum())
        sigma_f_sq = float((f * f).sum() / max(nf - 1, 1))
        return ImplicitTarget(a=sigma_f_sq, u=beta, g=sigma_eps_sq, c=0.0)

    if target == Target.COMMON_COVARIANCE:
        avg_var = float(diag_s.mean())
        ones_proj = z.sum(axis=1)  # Z @ 1
        total = float(ones_proj @ ones_proj) / dof  # 1^T S 1
        c_off = (total - diag_s.sum()) / max(m * m - m, 1)
        return ImplicitTarget(a=0.0, u=np.zeros(m), g=np.full(m, avg_var - c_off), c=c_off)

    if target == Target.CONSTANT_CORRELATION:
        stds = np.sqrt(np.maximum(diag_s, 0.0))
        inv = np.where(stds > 0, 1.0 / np.where(stds > 0, stds, 1.0), 0.0)
        corr_proj = z @ inv  # Z @ D^-1 1
        total = float(corr_proj @ corr_proj) / dof  # 1^T Corr 1 (approx.)
        m_pos = int((stds > 0).sum())
        r_bar = (total - m_pos) / max(m_pos * m_pos - m_pos, 1)
        r_bar = float(np.clip(r_bar, -1.0, 1.0))
        return ImplicitTarget(a=r_bar, u=stds, g=diag_s - r_bar * stds * stds, c=0.0)

    raise ValueError(f"unknown shrinkage target {target!r}")


def subsample_alpha(
    z: np.ndarray,
    finite: np.ndarray,
    f_target: ImplicitTarget,
    idx: np.ndarray,
) -> float:
    r"""Schäfer-Strimmer optimal intensity, evaluated on the columns `idx`.

    Implements the analytic unbiased estimator of Schäfer & Strimmer (2005),
    as used by Pantaleo et al. (2010): the ratio of the summed element-wise
    variances of the sample covariances to the summed squared distances from
    the target, both restricted to off-diagonal pairs `(i, j in idx)` — the
    intensity is one global scalar, so a column subsample estimates it. The
    pairwise-complete count formulas of the dense estimator are kept exactly
    on the sampled columns. Returns the intensity clipped to `[0, 1]`.
    """
    m = z.shape[1]
    ks = len(idx)
    if m < 2 or ks < 1:
        return 1.0

    ind = finite.astype(np.float64)
    counts = ind.T @ ind[:, idx]  # (m, ks) pairwise counts
    sum_w = z.T @ z[:, idx]  # per-pair sums of w_ij(t)
    z_sq = z * z
    sum_w_sq = z_sq.T @ z_sq[:, idx]  # per-pair sums of w_ij(t)^2

    counts_safe = np.maximum(counts, 1.0)
    centered_sum_sq = sum_w_sq - sum_w * sum_w / counts_safe
    dof_cube = np.maximum((counts - 1.0) ** 3, 1.0)
    var_s = np.where(counts >= 2, counts * centered_sum_sq / dof_cube, 0.0)

    s_cols = sum_w / np.maximum(counts - 1.0, 1.0)
    f_cols = f_target.columns(idx)

    off = np.ones((m, ks), dtype=bool)
    off[idx, np.arange(ks)] = False
    numerator = float(var_s[off].sum())
    denominator = float(((s_cols - f_cols) ** 2)[off].sum())
    if denominator < 1e-30:
        return 1.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def factor_fit(
    y: np.ndarray,
    *,
    target: Target,
    factor_rank: int,
    alpha_subsample: int = 512,
    seed: int = 0,
) -> np.ndarray:
    r"""Fits the shrunk covariance and returns its `(factor_rank + 1, m)` split.

    Rows `[0, factor_rank)` are `B.T` (unused rows zero when the active block
    is smaller than the rank) and the last row is the idiosyncratic std-dev
    `d`, so `Sigma ~= B B.T + diag(d^2)` with the diagonal matched exactly.
    """
    m = y.shape[1]
    rank = int(factor_rank)
    out = np.zeros((rank + 1, m))

    finite = np.isfinite(y)
    counts = finite.sum(axis=0)
    t_eff = int(finite.any(axis=1).sum())
    if t_eff < 2 or m < 1:
        return out

    mean = np.where(counts > 0, np.where(finite, y, 0.0).sum(axis=0) / np.maximum(counts, 1), 0.0)
    centered = np.where(finite, y - mean, 0.0)
    dof = float(max(t_eff - 1, 1))

    # Per-stock variance correction: a common `dof` normalization would
    # understate the variance of sparsely-observed stocks (zeros contribute
    # nothing), which an optimizer would read as low risk. Scaling stock `i`
    # by `sqrt(dof / (c_i - 1))` makes the diagonal exact and corrects pairs
    # approximately.
    scale = np.sqrt(dof / np.maximum(counts - 1.0, 1.0))
    z = centered * scale

    f_target = build_target(Target(target), z, finite, counts, dof)
    rng = np.random.default_rng(seed)
    ks = min(m, int(alpha_subsample))
    idx = rng.choice(m, size=ks, replace=False) if ks < m else np.arange(m)
    alpha = subsample_alpha(z, finite, f_target, idx)

    def matvec(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(m)
        return alpha * f_target.matvec(x) + (1.0 - alpha) * (z.T @ (z @ x)) / dof

    diag_sigma = alpha * f_target.diagonal() + (1.0 - alpha) * (z * z).sum(axis=0) / dof

    # `eigsh` needs `k < m`; a single-stock block has no cross-section to
    # factor, so its variance is all idiosyncratic.
    r_eff = min(rank, m - 1)
    if r_eff >= 1:
        op = LinearOperator(shape=(m, m), matvec=matvec, dtype=np.float64)  # type: ignore
        ev, vecs = eigsh(op, k=r_eff, which="LA", tol=1e-6)  # type: ignore
        order = np.argsort(ev)[::-1]
        ev, vecs = ev[order], vecs[:, order]
        loadings = vecs * np.sqrt(np.maximum(ev, 0.0))  # (m, r_eff)
        d_sq = np.maximum(diag_sigma - (loadings * loadings).sum(axis=1), 0.0)
        out[:r_eff, :] = loadings.T
    else:
        d_sq = diag_sigma

    out[rank, :] = np.sqrt(d_sq)
    return out


# ---------------------------------------------------------------------------
# The windowed harness.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VariancePredictorState:
    """Everything the variance predictor carries between generations.

    This mirrors the Rust `Operator::State`: `init` is the only method that
    reads the operator, everything it needs is copied here — so `reset` and
    `compute` are free functions of `(inputs, state)`, and every node's data
    is its own even when a shared operator instance is loaded from `__op__`.
    `slots=True` turns a stray attribute into an `AttributeError` rather than
    a silent cross-node alias.
    """

    #: `(T, m) active return block -> (rank + 1, m) factor panel`.
    fit: Callable
    rank: int
    refit_every: int
    min_periods: int | None
    universe_size: int | None
    #: Recorded `(N,)` return cross-sections, newest last; the bounded deque
    #: does the `max_periods` trimming.
    target: deque
    #: The retained factor panel, re-emitted between refits.
    out: np.ndarray
    fitted: bool = False
    rebalances: int = 0


class VariancePredictor:
    r"""Windowed variance predictor: record, refit on a cadence, emit.

    On each sampling tick one return cross-section is appended to a bounded
    window. On each rebalance tick the model is refit if the cadence is due —
    over the stocks that are in the universe and have at least `min_periods`
    valid observations in the window — and the fitted factor panel is emitted,
    `NaN` outside the fitted block. Between refits the last panel is
    re-emitted unchanged.
    """

    type Inputs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    type Outputs = tuple[bool, np.ndarray]
    type Context = int
    type State = VariancePredictorState

    def __init__(
        self,
        *,
        fit,
        factor_rank: int,
        refit_every: int = 1,
        max_periods: int | None = None,
        min_periods: int | None = None,
        universe_size: int | None = None,
    ) -> None:
        assert factor_rank >= 1, "factor_rank must be >= 1"
        assert refit_every >= 1, "refit_every must be >= 1"
        assert max_periods is None or max_periods >= 1, "max_periods must be >= 1"
        assert min_periods is None or min_periods >= 1, "min_periods must be >= 1"

        self.fit = fit
        self.factor_rank = int(factor_rank)
        self.refit_every = int(refit_every)
        self.max_periods = max_periods
        self.min_periods = min_periods
        self.universe_size = universe_size

    def init(self, inputs) -> VariancePredictorState:
        *_, universe = inputs
        return VariancePredictorState(
            fit=self.fit,
            rank=self.factor_rank,
            refit_every=self.refit_every,
            min_periods=self.min_periods,
            universe_size=self.universe_size,
            target=deque(maxlen=self.max_periods),
            out=np.full((self.factor_rank + 1, universe.shape[0]), np.nan),
        )

    @staticmethod
    def reset(_, state: VariancePredictorState):
        return (False, state.out)

    @staticmethod
    def compute(inputs, state: VariancePredictorState, _):
        sample_signal, _features, target, rebalance_signal, universe = inputs

        if sample_signal:
            state.target.append(target)

        if not rebalance_signal:
            return (False, state.out)

        refit = (not state.fitted) or (state.rebalances % state.refit_every == 0)
        state.rebalances += 1
        if refit and state.target:
            y = np.stack(tuple(state.target))  # (T, N)
            mask = active_mask(universe, state.universe_size)
            if state.min_periods is not None:
                mask &= np.isfinite(y).sum(axis=0) >= state.min_periods
            if mask.any():
                out = np.full_like(state.out, np.nan)
                out[:, mask] = state.fit(y[:, mask])
                state.out = out
                state.fitted = True
        return (True, state.out)


def build(
    *,
    target: int | Target = Target.COMMON_COVARIANCE,
    factor_rank: int = 20,
    alpha_subsample: int = 512,
    seed: int = 0,
    **kwargs,
) -> VariancePredictor:
    target = Target(target)
    rank = int(factor_rank)
    return VariancePredictor(
        fit=lambda y: factor_fit(
            y,
            target=target,
            factor_rank=rank,
            alpha_subsample=alpha_subsample,
            seed=seed,
        ),
        factor_rank=rank,
        **kwargs,
    )
