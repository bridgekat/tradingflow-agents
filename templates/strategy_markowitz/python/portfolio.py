"""Markowitz mean-variance portfolio, solved by CVXPY on a factor-model risk.

Consuming the factor panel rather than a dense `[N, N]` covariance keeps the
Rust <-> Python boundary cheap, and hands the solver exactly the
low-rank-plus-diagonal split its DPP problem wants — a full `(M, M)`
covariance parameter would make the CVXPY problem non-warm-startable (its
canonicalization is `O(M^3)`), and each sweep variant would repeat the
eigendecomposition the predictor now does once.

The file has three layers, top to bottom:

1. The **portfolio harness** (`PortfolioState`, `Portfolio`): gate on the
   rebalance signal, mask down to the stocks worth optimizing over, map
   lognormal moments to linear ones, hand the active sub-problem to a solver,
   scatter the answer back and retain it.
2. The **slot plumbing** (`assign_slots`, `factor_params_at`): scatter the
   active block's loadings into fixed solver slots; stable slot assignment
   keeps the cached warm start aligned for names that persist across
   rebalances.
3. The **problem** (`Mode`, `build_solver`, `solve`): the four Markowitz
   modes over long-only (optional) and budget constraints, built once at
   `init` and re-solved with fresh parameter values each rebalance.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

#: A predicted per-period return at or beyond this magnitude is treated as
#: the predictor malfunctioning rather than as an opinion, and the stock is
#: masked out of the solve exactly like a NaN. For log moments this also
#: keeps `to_linear_factor`'s `expm1` far away from overflow.
MU_LIMIT = 1.0

# ---------------------------------------------------------------------------
# Slot plumbing: scatter the factor-form risk into the fixed solver size.
# ---------------------------------------------------------------------------


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
    bt: np.ndarray, d_sub: np.ndarray, slots: np.ndarray, max_size: int, rank: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scatter the active factor-form risk into the fixed solver size at `slots`.

    Returns `(F, d, active)` for the persistent problem's parameters: `F` is
    `rank x max_size` (the `(rank, active)` loadings `bt` placed at `slots`,
    zero elsewhere), `d` is `max_size` (idiosyncratic std-dev at `slots`),
    and `active` is the 0/1 slot mask. The caller scatters `mu` (and reads
    weights back) at the same `slots`.
    """
    m, r = int(max_size), int(rank)
    assert bt.shape[0] == r, f"risk panel rank {bt.shape[0]} != solver factor_rank {r}"
    F = np.zeros((r, m))
    F[:, slots] = bt
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

    constraints: list[Any] = [cp.multiply(1.0 - active, x) == 0]  # pin inactive weights to 0
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
        "long_only": long_only,
        "slots": {},
    }


def solve(
    handle: dict,
    active_indices: np.ndarray,
    mu: np.ndarray,
    bt: np.ndarray,
    d_sub: np.ndarray,
) -> np.ndarray:
    """Maps the active stocks to stable slots, sets the parameters, re-solves.

    `bt` and `d_sub` are the active block's loadings and idiosyncratic
    std-dev. The variable is never seeded directly: CVXPY re-solves from its
    own cached solution, and the stable slot assignment is what keeps that
    cache aligned for the names that persist across the rebalance.
    """
    import cvxpy as cp

    n = len(mu)
    slots = assign_slots(handle["slots"], active_indices, handle["size"])
    factors, idio, active = factor_params_at(bt, d_sub, slots, handle["size"], handle["rank"])
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
    return np.maximum(weights, 0.0) if handle["long_only"] else weights


# ---------------------------------------------------------------------------
# The portfolio harness.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MeanVariancePortfolioState:
    """Everything a portfolio carries between generations.

    This mirrors the Rust `Operator::State`: `init` moves the build
    configuration in here, so `reset` and `compute` are free functions of
    `(inputs, state)` and nothing varies on the operator instance — which
    matters because a module binding `__op__` shares one instance across every
    node that loads it. `slots=True` makes a stray attribute an
    `AttributeError` rather than a silent cross-node alias.
    """

    logarithmic: bool
    max_universe_size: int
    #: The retained weights, re-emitted on every non-rebalance tick. The
    #: solver's warm start lives in `solver` (CVXPY's cached solution).
    weights: np.ndarray
    #: The persistent CVXPY problem built by `build_solver`.
    solver: dict


def to_linear_factor(mu: np.ndarray, bt: np.ndarray, d: np.ndarray):
    r"""Maps lognormal moments in factor form to linear-return moments.

    For log returns `r` with mean `m` and covariance `S`, the linear return
    `e^r - 1` has

        mu_lin[i]       = exp(m[i] + S[i, i] / 2) - 1
        Sigma_lin[i, j] = (1 + mu_lin[i]) (1 + mu_lin[j]) (exp(S[i, j]) - 1)

    The drift and the outer scaling are exact in factor form (`S[i, i]` and
    the row/column scale are per-stock); the elementwise `exp(S[i, j]) - 1`
    is taken to first order, `exp(S) - 1 ~= S`, which for daily log-return
    covariances (entries `~1e-4`) is far inside the estimation error — and is
    what keeps the low-rank-plus-diagonal form closed under the mapping.
    """
    var = (bt * bt).sum(axis=0) + d * d
    mu_lin = np.expm1(mu + 0.5 * var)
    scale = 1.0 + mu_lin
    return mu_lin, bt * scale[None, :], d * scale


class MeanVariancePortfolio:
    type Inputs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    type Outputs = tuple[bool, np.ndarray]
    type Context = int
    type State = MeanVariancePortfolioState

    def __init__(
        self,
        *,
        mode: "Mode | int",
        bound: float,
        long_only: bool = True,
        full_position: bool = True,
        factor_rank: int = 20,
        max_universe_size: int | None = None,
        logarithmic: bool = True,
    ) -> None:
        assert max_universe_size is None or max_universe_size >= 1
        self.mode = Mode(mode)
        self.bound = float(bound)
        self.long_only = long_only
        self.full_position = full_position
        self.factor_rank = int(factor_rank)
        self.max_universe_size = max_universe_size
        self.logarithmic = logarithmic

    def init(self, inputs: Inputs) -> State:
        universe = inputs[1]
        n = universe.shape[0]
        m = n if self.max_universe_size is None else self.max_universe_size
        return MeanVariancePortfolioState(
            logarithmic=self.logarithmic,
            max_universe_size=m,
            weights=np.zeros(n),
            solver=build_solver(m, self.factor_rank, self.mode, self.bound, self.long_only, self.full_position),
        )

    @staticmethod
    def reset(_: Inputs, state: State) -> Outputs:
        return (False, state.weights)

    @staticmethod
    def compute(inputs: Inputs, state: State, _: Context) -> Outputs:
        rebalance_signal, universe, mu, risk = inputs
        if not rebalance_signal:
            return (False, state.weights)

        # A stock is worth optimizing over when it is in the universe and every
        # moment the portfolio depends on is finite for it. A NaN prediction is
        # the predictor saying it has no opinion, not a zero one — and a finite
        # but absurd one ([`MU_LIMIT`]) counts as no opinion too. The risk
        # panel's last row is the idiosyncratic std-dev, its scatter mask.
        mask = (universe > 0) & np.isfinite(mu) & (np.abs(mu) < MU_LIMIT) & np.isfinite(risk[-1])

        weights = np.zeros(universe.shape[0])
        if mask.any():
            active = int(mask.sum())
            if active > state.max_universe_size:
                raise ValueError(
                    f"active universe size {active} exceeds " f"max_universe_size {state.max_universe_size}"
                )

            sub_mu = mu[mask]
            sub_bt = risk[:-1, mask]  # (rank, active) loadings
            sub_d = risk[-1, mask]
            if not (np.all(np.isfinite(sub_bt)) and np.all(np.isfinite(sub_d))):
                # A finite std-dev row is not enough: a NaN loading would
                # silently poison the whole solve.
                raise ValueError("active risk block contains non-finite entries")

            if state.logarithmic:
                sub_mu, sub_bt, sub_d = to_linear_factor(sub_mu, sub_bt, sub_d)

            weights[mask] = solve(state.solver, np.nonzero(mask)[0], sub_mu, sub_bt, sub_d)

        state.weights = weights
        return (True, state.weights)


def build(*, mode: Mode | int = Mode.MIN_MEAN_VARIANCE, **kwargs) -> MeanVariancePortfolio:
    return MeanVariancePortfolio(mode=Mode(mode), **kwargs)
