"""Markowitz mean-variance portfolio, solved by CVXPY on a factor-model risk.

Strategy-local, self-contained copy assembled from the `tradingflow` Python
package (`portfolio/_base.py`, `portfolio/_factor.py`,
`portfolio/mean_variance/_modes.py` and
`portfolio/mean_variance/markowitz.py`), trimmed to the mean-variance shape
this strategy uses. The strategy loads it with
`py_operator_module("markowitz", params)`; edit freely — the next run picks
the change up with no install step.

The operator mirrors the Rust `Operator` trait method-for-method:
`init(inputs) -> state`, `reset(inputs, state) -> outputs` and
`compute(inputs, state, instant) -> outputs`. Inputs are
`(rebalance_signal, universe, mu, sigma)` as owned NumPy arrays; outputs are
`(signal, weights)` — a length-`N` book, zero outside the active set.
Exceptions panic the node and poison the graph, so recoverable conditions
should be values rather than raises.

The file has three layers, top to bottom:

1. The **portfolio harness** (`PortfolioState`, `Portfolio`): gate on the
   rebalance signal, mask down to the stocks worth optimizing over, map
   lognormal moments to linear ones, hand the active sub-problem to a solver,
   scatter the answer back and retain it.
2. The **factor-model risk** (`factor_decompose`, `assign_slots`,
   `factor_params_at`): a low-rank-plus-diagonal split of the covariance,
   `Sigma ~= B B^T + diag(d^2)`, scattered into fixed solver slots. A full
   `(M, M)` covariance parameter makes the CVXPY problem un-warm-startable
   (the DPP canonicalization is `O(M^3)`); the factor form is genuinely DPP,
   and stable slot assignment keeps the cached warm start aligned for names
   that persist across rebalances.
3. The **problem** (`Mode`, `build_solver`, `solve`): the four Markowitz
   modes over long-only (optional) and budget constraints, built once at
   `init` and re-solved with fresh parameter values each rebalance.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable

import numpy as np

try:
    from scipy.sparse.linalg import eigsh as _eigsh
except Exception:  # pragma: no cover - scipy always present in the cvxpy venv
    _eigsh = None

# ---------------------------------------------------------------------------
# The portfolio harness.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PortfolioState:
    """Everything a portfolio carries between generations.

    This mirrors the Rust `Operator::State`: `init` moves the build
    configuration in here, so `reset` and `compute` are free functions of
    `(inputs, state)` and nothing varies on the operator instance — which
    matters because a module binding `__op__` shares one instance across every
    node that loads it. `slots=True` makes a stray attribute an
    `AttributeError` rather than a silent cross-node alias.
    """

    #: `(state, active, universe, previous, mu, sigma) -> weights` over the
    #: active subset.
    solve: Callable
    logarithmic: bool
    max_universe_size: int
    #: The retained weights, re-emitted on every non-rebalance tick and handed
    #: to the next solve as its warm start.
    weights: np.ndarray
    #: Whatever `init_solver` built.
    solver: object = None

    @staticmethod
    def moments(inputs):
        """Unpacks `(universe, mu, sigma)` from the mean-variance inputs."""
        _, universe, mu, sigma = inputs
        return universe, mu, sigma


def to_linear(mu: np.ndarray | None, sigma: np.ndarray | None):
    r"""Maps lognormal moments to linear-return moments.

    For log returns `r` with mean `m` and covariance `S`, the linear return
    `e^r - 1` has

        mu_lin[i]    = exp(m[i] + S[i, i] / 2) - 1
        Sigma_lin[i, j] = (1 + mu_lin[i]) (1 + mu_lin[j]) (exp(S[i, j]) - 1)

    with the specialisations that fall out when one moment is absent: no
    covariance means the variance term drops from the drift, and no mean means
    the drift is the variance term alone.
    """
    if sigma is None:
        return np.expm1(mu), None

    drift = 0.5 * np.diag(sigma)
    mu_lin = np.expm1(drift if mu is None else mu + drift)
    factor = 1.0 + mu_lin
    sigma_lin = np.outer(factor, factor) * np.expm1(sigma)
    return (None if mu is None else mu_lin), sigma_lin


class MeanVariancePortfolio:
    """Gate, mask, convert, solve, scatter, retain."""

    type Inputs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    type Outputs = tuple[bool, np.ndarray]
    type Context = int
    type State = PortfolioState

    def __init__(
        self,
        *,
        solve,
        init_solver=None,
        max_universe_size: int | None = None,
        logarithmic: bool = True,
    ) -> None:
        assert max_universe_size is None or max_universe_size >= 1
        self.solve = solve
        self.init_solver = init_solver
        self.max_universe_size = max_universe_size
        self.logarithmic = logarithmic

    def init(self, inputs) -> PortfolioState:
        universe = inputs[1]
        n = universe.shape[0]
        m = n if self.max_universe_size is None else self.max_universe_size
        return PortfolioState(
            solve=self.solve,
            logarithmic=self.logarithmic,
            max_universe_size=m,
            weights=np.zeros(n),
            solver=None if self.init_solver is None else self.init_solver(m),
        )

    @staticmethod
    def reset(_, state: PortfolioState):
        return (False, state.weights)

    @staticmethod
    def compute(inputs, state: PortfolioState, _):
        if not inputs[0]:
            return (False, state.weights)

        universe, mu, sigma = state.moments(inputs)

        # A stock is worth optimizing over when it is in the universe and every
        # moment the portfolio depends on is finite for it. A NaN prediction is
        # the predictor saying it has no opinion, not a zero one.
        mask = (universe > 0) & np.isfinite(mu) & np.isfinite(np.diag(sigma))

        weights = np.zeros(universe.shape[0])
        if mask.any():
            active = int(mask.sum())
            if active > state.max_universe_size:
                raise ValueError(
                    f"active universe size {active} exceeds " f"max_universe_size {state.max_universe_size}"
                )

            sub_mu = mu[mask]
            sub_sigma = sigma[np.ix_(mask, mask)]
            if not np.all(np.isfinite(sub_sigma)):
                # A finite diagonal is not enough: an off-diagonal NaN would
                # silently poison the whole solve.
                raise ValueError("active covariance block contains non-finite entries")

            if state.logarithmic:
                sub_mu, sub_sigma = to_linear(sub_mu, sub_sigma)

            weights[mask] = state.solve(
                state,
                np.nonzero(mask)[0],
                universe[mask],
                state.weights[mask],
                sub_mu,
                sub_sigma,
            )

        state.weights = weights
        return (True, state.weights)


# ---------------------------------------------------------------------------
# Low-rank-plus-diagonal (factor-model) risk, for a warm-startable problem.
# ---------------------------------------------------------------------------


def factor_decompose(sigma: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-`rank` SVD low-rank + idiosyncratic-diagonal split of a PSD `sigma`.

    Returns `(B, d)`: `B` the `n x r_eff` loadings (`r_eff = min(rank, n)`) and `d`
    the length-`n` idiosyncratic **std-dev** (sqrt of the residual diagonal), so
    `sigma ~= B @ B.T + diag(d**2)` with the diagonal matched exactly.
    """
    n = sigma.shape[0]
    r_eff = min(int(rank), n)
    if _eigsh is not None and 1 <= r_eff < n:
        try:
            ev, V = _eigsh(sigma, k=r_eff, which="LA")  # top-r_eff (PSD => largest)
        except Exception:
            ev, V = np.linalg.eigh(sigma)
            ev, V = ev[-r_eff:], V[:, -r_eff:]
    else:
        ev, V = np.linalg.eigh(sigma)
        ev, V = ev[-r_eff:], V[:, -r_eff:]
    B = V * np.sqrt(np.maximum(ev, 0.0))  # n x r_eff
    d2 = np.maximum(np.diag(sigma) - np.einsum("ij,ij->i", B, B), 0.0)
    return B, np.sqrt(d2)


def assign_slots(slot_of: dict, active_idx, max_size: int) -> np.ndarray:
    """Stable mapping from active stocks to fixed solver slots `[0, max_size)`.

    `slot_of` (global stock index -> slot) is carried in the operator state and
    mutated in place: stocks that are still active keep their slot, departed stocks
    free theirs, and new entrants take the lowest free slots.  Keeping continuing
    names in the same slot is what lets cvxpy's cached primal+dual warm start stay
    aligned across rebalances (the variable `.value` seed is ignored on re-solves).

    Returns the slot of each active stock, in `active_idx` order.
    """
    active = [int(g) for g in active_idx]
    active_set = set(active)
    for g in [g for g in slot_of if g not in active_set]:  # free departed slots
        del slot_of[g]
    used = set(slot_of.values())
    free = (s for s in range(int(max_size)) if s not in used)
    for g in active:
        if g not in slot_of:
            slot_of[g] = next(free)
    return np.fromiter((slot_of[g] for g in active), dtype=np.intp, count=len(active))


def factor_params_at(
    sigma: np.ndarray, slots: np.ndarray, max_size: int, rank: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose `sigma` and scatter into the fixed solver size at `slots`.

    Returns `(F, d, active)` for the persistent problem's parameters: `F` is
    `rank x max_size` (`B.T` placed at `slots`, zero elsewhere and in unused factor
    rows), `d` is `max_size` (idiosyncratic std-dev at `slots`), and `active` is the
    0/1 slot mask.  The caller scatters `mu` (and reads weights back) at the same
    `slots`.
    """
    B, d_sub = factor_decompose(sigma, rank)
    r_eff = B.shape[1]
    m, r = int(max_size), int(rank)
    F = np.zeros((r, m))
    F[:r_eff, slots] = B.T
    d = np.zeros(m)
    d[slots] = d_sub
    active = np.zeros(m)
    active[slots] = 1.0
    return F, d, active


# ---------------------------------------------------------------------------
# The Markowitz problem.
# ---------------------------------------------------------------------------


class Mode(IntEnum):
    r"""Markowitz optimization mode.

    All modes share the long-only (optional) and budget constraints
    ``1^T x = 1``, ``x >= 0``.  The ``bound`` parameter's meaning is
    mode-dependent.

    - ``MIN_VARIANCE_GIVEN_RETURN``: minimize ``x^T Sigma x`` s.t. ``mu^T x >= bound``.
    - ``MAX_RETURN_GIVEN_STD_DEV``: maximize ``mu^T x`` s.t. ``sqrt(x^T Sigma x) <= bound``.
    - ``MIN_MEAN_VARIANCE``: maximize ``mu^T x - bound * x^T Sigma x``.
    - ``MIN_MEAN_STD_DEV``: maximize ``mu^T x - bound * sqrt(x^T Sigma x)``.
    """

    MIN_VARIANCE_GIVEN_RETURN = 1
    MAX_RETURN_GIVEN_STD_DEV = 2
    MIN_MEAN_VARIANCE = 3
    MIN_MEAN_STD_DEV = 4


def build_solver(
    max_universe_size: int,
    factor_rank: int,
    mode: Mode,
    bound: float,
    long_only: bool,
    full_position: bool,
) -> dict:
    """Builds the fixed-size DPP problem once, at `init`."""
    import cvxpy as cp

    m, r = int(max_universe_size), int(factor_rank)
    x = cp.Variable(m)
    factors = cp.Parameter((r, m))  # B.T, padded
    idio = cp.Parameter(m)  # idiosyncratic std-dev, padded
    mu = cp.Parameter(m)
    active = cp.Parameter(m, nonneg=True)

    constraints = [cp.multiply(1.0 - active, x) == 0]  # pin inactive weights to 0
    if long_only:
        constraints.append(x >= 0)
    constraints.append(cp.sum(x) == 1 if full_position else cp.sum(x) <= 1)

    variance = cp.sum_squares(factors @ x) + cp.sum_squares(cp.multiply(idio, x))
    expected = mu @ x
    match mode:
        case Mode.MIN_VARIANCE_GIVEN_RETURN:
            objective = cp.Minimize(variance)
            constraints.append(expected >= bound)
        case Mode.MAX_RETURN_GIVEN_STD_DEV:
            objective = cp.Maximize(expected)
            constraints.append(variance <= bound * bound)
        case Mode.MIN_MEAN_VARIANCE:
            objective = cp.Maximize(expected - bound * variance)
        case Mode.MIN_MEAN_STD_DEV:
            deviation = cp.norm(cp.hstack([factors @ x, cp.multiply(idio, x)]))
            objective = cp.Maximize(expected - bound * deviation)

    return {
        "problem": cp.Problem(objective, constraints),
        "x": x,
        "factors": factors,
        "idio": idio,
        "mu": mu,
        "active": active,
        "size": m,
        "rank": r,
        "slots": {},
    }


def solve(
    handle: dict,
    active_indices: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    long_only: bool,
) -> np.ndarray:
    """Maps the active stocks to stable slots, sets the parameters, re-solves.

    The variable is never seeded directly: CVXPY re-solves from its own cached
    solution, and the stable slot assignment is what keeps that cache aligned
    for the names that persist across the rebalance.
    """
    import cvxpy as cp

    n = len(mu)
    slots = assign_slots(handle["slots"], active_indices, handle["size"])
    factors, idio, active = factor_params_at(sigma, slots, handle["size"], handle["rank"])
    padded = np.zeros(handle["size"])
    padded[slots] = mu

    handle["factors"].value = factors
    handle["idio"].value = idio
    handle["mu"].value = padded
    handle["active"].value = active

    try:
        handle["problem"].solve(solver=cp.SCS, warm_start=True)
    except cp.SolverError:
        return np.full(n, 1.0 / n)
    if handle["x"].value is None:
        return np.full(n, 1.0 / n)

    weights = np.asarray(handle["x"].value[slots], dtype=np.float64)
    return np.maximum(weights, 0.0) if long_only else weights


def build(
    *,
    mode: Mode | int = Mode.MIN_MEAN_VARIANCE,
    bound: float,
    long_only: bool = True,
    full_position: bool = True,
    factor_rank: int = 20,
    **kwargs,
) -> MeanVariancePortfolio:
    """Constructs a Markowitz mean-variance portfolio."""
    mode = Mode(mode)
    bound = float(bound)
    return MeanVariancePortfolio(
        init_solver=lambda m: build_solver(
            m, factor_rank, mode, bound, long_only, full_position
        ),
        solve=lambda state, active, universe, previous, mu, sigma: solve(
            state.solver, active, mu, sigma, long_only
        ),
        **kwargs,
    )
