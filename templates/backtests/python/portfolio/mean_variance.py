from dataclasses import dataclass
import sys
import numpy as np
import cvxpy as cp


@dataclass(slots=True)
class MeanVarianceSolver:
    """A Markowitz mean-variance portfolio optimizer using CVXPY DPP.
    It solves the convex optimization problem:

    ```
    max_w { μᵀ w - δ wᵀ Σ w }
    ```

    where `Σ = X F Xᵀ + D`, `F` is symmetric positive semi-definite, `X` is
    (narrow) rectangular, and `D` is diagonal with non-negative entries.
    `δ` is the risk aversion parameter.
    """

    n: int
    k: int
    w: cp.Variable
    max: cp.Parameter
    bench: cp.Parameter
    mu: cp.Parameter
    rank: cp.Parameter  # any matrix `Z` such that `X F Xᵀ = Z Zᵀ`
    diag: cp.Parameter  # square root of diagonal of `D`
    prev: cp.Parameter  # previous weights, for the turnover penalty
    problem: cp.Problem

    def __init__(
        self,
        n: int,
        k: int,
        benchmark_relative: bool,
        risk_aversion: float,
        long_only: bool,
        full_position: bool,
        turnover_penalty: float = 0.0,
    ) -> None:
        """Initializes the optimizer for `n` slots and rank-`k` risk factor model."""

        self.n = n
        self.k = k
        self.w = cp.Variable(n)
        self.max = cp.Parameter(n, nonneg=True)
        self.bench = cp.Parameter(n)
        self.mu = cp.Parameter(n)
        self.rank = cp.Parameter((n, k))
        self.diag = cp.Parameter(n)
        self.prev = cp.Parameter(n)

        constraints: list[cp.Constraint] = []
        constraints.append(
            cp.abs(self.w) <= self.max
        )  # max position size & active mask
        constraints.append(cp.norm1(self.w) <= 1.0)  # no leverage
        if long_only:
            constraints.append(self.w >= 0.0)  # no short-selling
        if full_position:
            constraints.append(self.w.sum() == 1.0)  # fully invested

        if benchmark_relative:
            active_w = self.w - self.bench
            active_returns = self.mu.T @ active_w
            tracking_error = cp.sum_squares(self.rank.T @ active_w) + cp.sum_squares(
                cp.multiply(self.diag, active_w)
            )
            objective_expr = active_returns - risk_aversion * tracking_error
        else:
            returns = self.mu.T @ self.w
            variance = cp.sum_squares(self.rank.T @ self.w) + cp.sum_squares(
                cp.multiply(self.diag, self.w)
            )
            objective_expr = returns - risk_aversion * variance

        if turnover_penalty > 0.0:
            objective_expr = objective_expr - turnover_penalty * cp.norm1(
                self.w - self.prev
            )

        self.problem = cp.Problem(cp.Maximize(objective_expr), constraints)

    def solve(
        self,
        max: np.ndarray,
        bench: np.ndarray,
        mu: np.ndarray,
        exposures: np.ndarray,  # X
        covariance: np.ndarray,  # F
        specific: np.ndarray,  # diagonal of D
        prev: np.ndarray | None = None,  # previous weights (turnover penalty)
    ) -> np.ndarray | None:
        """Solves the optimization problem and returns the optimal weights."""

        assert max.shape == (self.n,) and np.isfinite(max).all()
        assert bench.shape == (self.n,) and np.isfinite(bench).all()
        assert mu.shape == (self.n,) and np.isfinite(mu).all()
        assert exposures.shape == (self.n, self.k) and np.isfinite(exposures).all()
        assert covariance.shape == (self.k, self.k) and np.isfinite(covariance).all()
        assert specific.shape == (self.n,) and np.isfinite(specific).all()
        if prev is None:
            prev = np.zeros((self.n,))
        assert prev.shape == (self.n,) and np.isfinite(prev).all()

        try:
            lam, s = np.linalg.eigh(covariance)  # F = S Λ S⁻¹ = S Λ Sᵀ
        except np.linalg.LinAlgError:
            print("portfolio: eigendecomposition did not converge", file=sys.stderr)
            return None

        rank = exposures @ s @ np.diag(np.sqrt(np.maximum(lam, 0.0)))
        diag = np.sqrt(np.maximum(specific, 0.0))

        self.max.value = max
        self.bench.value = bench
        self.mu.value = mu
        self.rank.value = rank
        self.diag.value = diag
        self.prev.value = prev

        try:
            self.problem.solve(solver=cp.SCS, warm_start=True)
            if self.w.value is not None:
                return self.w.value
            else:
                print("portfolio: no solution", file=sys.stderr)
                return None

        except cp.SolverError:
            print("portfolio: solver failed", file=sys.stderr)
            return None


@dataclass(slots=True)
class SlottedMeanVarianceSolver:
    """Restricts a size-`n` Markowitz problem to `m` solver slots.

    Wraps [`MeanVarianceSolver`]: gathers the active stocks into stable slots,
    scatters their moments into the fixed-size problem parameters —
    an unoccupied slot keeps `max = 0`, masking it out of the solve —
    and scatters the slot weights back onto the full axis. Keeping a
    continuing stock in the same slot across calls is what lets CVXPY/SCS
    cached primal-dual warm-start stay aligned across rebalances, so the
    solve benefits from the previous solution instead of restarting cold.
    """

    # global[global_mask] <-> slots[indices], slot_mask = bitset(indices)
    global_mask: np.ndarray
    slot_mask: np.ndarray
    indices: np.ndarray
    max_weight: float
    inner: MeanVarianceSolver

    def __init__(
        self,
        n: int,
        m: int,
        k: int,
        benchmark_relative: bool,
        risk_aversion: float,
        long_only: bool,
        full_position: bool,
        max_weight: float = 1.0,
        turnover_penalty: float = 0.0,
    ) -> None:
        """Initializes the wrapped optimizer for `n` assets, `m` slots and
        rank-`k` risk factor model.
        """

        self.global_mask = np.zeros((n,), dtype=bool)
        self.slot_mask = np.zeros((m,), dtype=bool)
        self.indices = np.zeros((0,), dtype=np.intp)
        self.max_weight = max_weight
        self.inner = MeanVarianceSolver(
            n=m,
            k=k,
            benchmark_relative=benchmark_relative,
            risk_aversion=risk_aversion,
            long_only=long_only,
            full_position=full_position,
            turnover_penalty=turnover_penalty,
        )

    def update_mask(self, new_global_mask: np.ndarray) -> None:
        """Updates masks and indices to the new active set."""

        new_slot_mask = np.zeros_like(self.slot_mask)
        new_indices = np.zeros((new_global_mask.sum(),), dtype=np.intp)

        prev_indices_keep = new_global_mask[self.global_mask]
        new_indices_keep = self.global_mask[new_global_mask]
        new_indices_alloc = ~new_indices_keep

        kept_indices = new_indices[new_indices_keep] = self.indices[prev_indices_keep]
        new_slot_mask[kept_indices] = True

        (free_list,) = np.nonzero(~new_slot_mask)
        alloc_indices = new_indices[new_indices_alloc] = free_list[
            : new_indices_alloc.sum()
        ]
        new_slot_mask[alloc_indices] = True

        self.global_mask = new_global_mask
        self.slot_mask = new_slot_mask
        self.indices = new_indices

    def solve(
        self,
        mask: np.ndarray,
        bench: np.ndarray,
        mu: np.ndarray,
        exposures: np.ndarray,  # X
        covariance: np.ndarray,  # F
        specific: np.ndarray,  # diagonal of D
        prev: np.ndarray | None = None,  # previous weights (turnover penalty)
    ) -> np.ndarray | None:
        """Solves over the stocks selected by `active` and returns the optimal
        weights on the full axis (zero off `active`), or `None` when there is
        nothing to solve or no solution.
        """

        m, k = self.inner.n, self.inner.k
        count = mask.sum()
        if count > m:
            raise ValueError(
                f"portfolio: {count} active stocks exceed universe size {m}"
            )
        if count == 0:
            print("portfolio: no active stocks to solve", file=sys.stderr)
            return None  # avoid infeasible problem, e.g. full position with no stocks

        self.update_mask(mask)
        # Per-name cap, relaxed towards equal weight when few names are
        # active so `sum(w) == 1` stays feasible with some slack.
        cap = max(self.max_weight, min(1.0, 2.0 / count))
        max_ = self.slot_mask.astype(np.float64) * cap
        bench_ = np.zeros((m,))
        bench_[self.indices] = bench[self.global_mask]
        mu_ = np.zeros((m,))
        mu_[self.indices] = mu[self.global_mask]
        exposures_ = np.zeros((m, k))
        exposures_[self.indices] = exposures[self.global_mask]
        specific_ = np.zeros((m,))
        specific_[self.indices] = specific[self.global_mask]
        prev_ = np.zeros((m,))
        if prev is not None:
            prev_[self.indices] = prev[self.global_mask]

        weights_ = self.inner.solve(
            max=max_,
            bench=bench_,
            mu=mu_,
            exposures=exposures_,
            covariance=covariance,
            specific=specific_,
            prev=prev_,
        )
        if weights_ is None:
            return None

        weights = np.zeros_like(bench)
        weights[self.global_mask] = weights_[self.indices]
        return weights


@dataclass(slots=True)
class PortfolioState:
    solver: SlottedMeanVarianceSolver
    out_weights: np.ndarray  # what the trader receives (possibly vol-scaled)
    raw_weights: np.ndarray  # the optimizer's own book, the turnover anchor
    vol_target: float  # annualized ex-ante vol target (0 = off)
    vol_periods: float  # periods per year for annualizing the risk forecast


class Portfolio:
    """Mean-variance book with a SEPARATE covariance for the vol-target scaler.

    Inputs carry two covariance forecasts over the same exposures `X`:

    - `(covariance, specific)` — used by the optimizer's risk term only.
    - `(scaler_covariance, scaler_specific)` — used by the `vol_target`
      scaler only.

    One matrix was doing two jobs that want opposite things. The optimizer
    wants a slow factor covariance: how orthogonal a feature is to the risk
    model is a property of `F`, and a fast `F` re-estimates the short-run
    turnover/volatility block from the last quarter, at which point the risk
    model prices the very structure an attention alpha tilts on and the
    optimizer hedges the signal away. The scaler wants a fast covariance: it
    is a pure volatility forecast, and a 250-day EWMA of a variance spike is
    still elevated two years later. When the two are wired to the same risk
    model this reduces exactly to the single-covariance build.
    """

    type Inputs = tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
    type Outputs = np.ndarray
    type Context = int
    type State = PortfolioState

    def __init__(
        self,
        universe_size: int,
        benchmark_relative: bool,
        risk_aversion: float,
        long_only: bool,
        full_position: bool,
        max_weight: float = 1.0,
        turnover_penalty: float = 0.0,
        vol_target: float = 0.0,
        vol_periods: float = 252.0,
    ) -> None:
        assert universe_size > 0, "portfolio: universe_size must be positive"
        assert risk_aversion > 0.0, "portfolio: risk_aversion must be positive"
        assert 0.0 < max_weight <= 1.0, "portfolio: max_weight must be in (0, 1]"
        assert turnover_penalty >= 0.0, "portfolio: turnover_penalty must be >= 0"
        assert vol_target >= 0.0, "portfolio: vol_target must be >= 0"

        self.universe_size = universe_size
        self.benchmark_relative = benchmark_relative
        self.risk_aversion = risk_aversion
        self.long_only = long_only
        self.full_position = full_position
        self.max_weight = max_weight
        self.turnover_penalty = turnover_penalty
        # Annualized ex-ante volatility target: when positive, the solved book
        # is scaled by `min(1, target / sqrt(w' Sigma w * periods))`, holding
        # the remainder in cash. The optimizer's own (unscaled) book stays the
        # turnover-penalty anchor, so the scaling is a downstream linear map
        # exactly like `--max-gross`.
        self.vol_target = vol_target
        self.vol_periods = vol_periods

    def init(self, inputs: Inputs) -> State:
        (
            rebalance_signal,
            universe,
            mu,
            exposures,
            covariance,
            specific,
            scaler_covariance,
            scaler_specific,
        ) = inputs
        n, k = exposures.shape
        assert universe.shape == (n,)
        assert mu.shape == (n,)
        assert covariance.shape == (k, k)
        assert specific.shape == (n,)
        assert scaler_covariance.shape == (k, k)
        assert scaler_specific.shape == (n,)

        return PortfolioState(
            solver=SlottedMeanVarianceSolver(
                n=n,
                m=min(self.universe_size, n),
                k=k,
                benchmark_relative=self.benchmark_relative,
                risk_aversion=self.risk_aversion,
                long_only=self.long_only,
                full_position=self.full_position,
                max_weight=self.max_weight,
                turnover_penalty=self.turnover_penalty,
            ),
            out_weights=np.zeros((n,)),
            raw_weights=np.zeros((n,)),
            vol_target=self.vol_target,
            vol_periods=self.vol_periods,
        )

    @staticmethod
    def reset(_: Inputs, state: State) -> Outputs:
        return state.out_weights

    @staticmethod
    def compute(inputs: Inputs, state: State, _: Context) -> Outputs:
        (
            rebalance_signal,
            universe,
            mu,
            exposures,
            covariance,
            specific,
            scaler_covariance,
            scaler_specific,
        ) = inputs
        n, k = exposures.shape
        assert universe.shape == (n,)
        assert mu.shape == (n,)
        assert covariance.shape == (k, k)
        assert specific.shape == (n,)
        assert scaler_covariance.shape == (k, k)
        assert scaler_specific.shape == (n,)

        if rebalance_signal:
            valid = (
                np.isfinite(mu)
                & np.isfinite(exposures).all(axis=1)
                & np.isfinite(specific)
                & np.isfinite(scaler_specific)
            )
            mask = valid & (universe > 0.0)
            weights = state.solver.solve(
                mask, universe, mu, exposures, covariance, specific,
                prev=state.raw_weights,
            )
            if weights is not None:
                state.raw_weights = weights
                if state.vol_target > 0.0:
                    # Ex-ante variance of the solved book under the SCALER's
                    # covariance forecast: w' (X F_s X' + D_s) w. The
                    # optimizer never sees F_s / D_s, so the book composition
                    # is set entirely by the slow model and only its size is
                    # set by the fast one.
                    xw = exposures[mask].T @ weights[mask]
                    var = float(
                        xw @ scaler_covariance @ xw
                        + np.sum(scaler_specific[mask] * weights[mask] ** 2)
                    )
                    vol = np.sqrt(max(var, 0.0) * state.vol_periods)
                    scale = 1.0 if vol <= 0.0 else min(1.0, state.vol_target / vol)
                    state.out_weights = weights * scale
                else:
                    state.out_weights = weights

        return state.out_weights


def build(**kwargs) -> Portfolio:
    return Portfolio(**kwargs)
