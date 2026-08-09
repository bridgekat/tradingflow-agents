from dataclasses import dataclass
from collections import deque
import sys
import numpy as np


@dataclass(slots=True)
class FactorModelRegressor:
    """An incremental factor model for covariance (risk) prediction.
    It assumes the target `r` is a vector process following (at each period):

        r = B f + ε

    for some predictable matrix process `B`, adapted vector process `f`, and
    adapted vector process `ε` with uncorrelated components (conditioned
    on past periods):

        E[ε] = 0
        Cov[ε] = diag(σ²_0, ..., σ²_{n - 1})
        Cov[f, ε] = 0

    where `σ²` is again a predictable vector process. Here we omit the
    time subscripts and other notational details: for example, `E[ε]` actually
    denotes `E[ε_t | 𝓕_{t - 1}]`.

    Each period, the matrix `B` is provided as input, and `f` is estimated by
    solving the (unregularized) least-squares problem:

        min_f { ‖B f - r‖² }

    then `σ²` is estimated by taking exponentially-weighted moving average over
    the residual `B f - r`. Similarly, we estimate `Cov[f]` by taking
    exponentially-weighted moving average over the outer product `f fᵀ`
    (i.e. ignoring the factor risk premium or alpha `E[f]` - which is close to
    0 compared to the second moment, and much harder to estimate reliably).

    The model assumptions directly leads to:

        Cov[r] = B Cov[f] Bᵀ + diag(σ²_0, ..., σ²_{n - 1})

    produced in three components: the exposures `B`, the factor covariances
    `Cov[f]`, and the specific variances `σ²`.
    """

    n: int
    k: int
    outer_lambda: float
    res_lambda: float

    count: int
    sum_outer: np.ndarray
    sum_outer_w: float
    sum_res: np.ndarray
    sum_res_w: np.ndarray

    covariance: np.ndarray
    specific: np.ndarray

    def __init__(
        self,
        n: int,
        k: int,
        outer_lambda: float,
        res_lambda: float,
    ) -> None:
        """Initializes the factor model for `n` assets and `k` factors."""

        self.n = n
        self.k = k
        self.outer_lambda = outer_lambda
        self.res_lambda = res_lambda

        self.count = 0
        self.sum_outer = np.zeros((k, k))
        self.sum_outer_w = 0.0
        self.sum_res = np.zeros((n,))
        self.sum_res_w = np.zeros((n,))

        self.covariance = np.zeros((k, k))
        self.specific = np.full((n,), np.nan)

    def add_cross_section(self, mask: np.ndarray, b: np.ndarray, r: np.ndarray) -> None:
        """Adds a cross-section of `(b[mask], r[mask])` to the accumulator."""

        assert mask.shape == (self.n,) and mask.dtype == np.bool_
        assert b.shape == (self.n, self.k) and np.isfinite(b[mask]).all()
        assert r.shape == (self.n,) and np.isfinite(r[mask]).all()

        try:
            pinv = np.linalg.pinv(b[mask])  # (Bᵀ B)⁻¹ Bᵀ
        except np.linalg.LinAlgError:
            print("risk_model: SVD did not converge", file=sys.stderr)
            return

        f = pinv @ r[mask]
        q = np.einsum("ij,ji->i", b[mask], pinv)  # leverage: diagonal of B (Bᵀ B)⁻¹ Bᵀ
        outer = np.outer(f, f)
        resid = np.zeros((self.n,))
        resid[mask] = np.square(b[mask] @ f - r[mask]) / np.maximum(1.0 - q, 1e-3)

        self.count += 1
        self.sum_outer = self.sum_outer * self.outer_lambda + outer
        self.sum_outer_w = self.sum_outer_w * self.outer_lambda + 1.0
        self.sum_res = self.sum_res * self.res_lambda + resid
        self.sum_res_w = self.sum_res_w * self.res_lambda + mask.astype(np.float64)

    def fit(self, mask: np.ndarray) -> None:
        """Calculates and record model parameters."""

        if self.count == 0:
            print("risk_model: no samples to fit", file=sys.stderr)
            return

        self.covariance = self.sum_outer / self.sum_outer_w
        self.specific[mask] = self.sum_res[mask] / self.sum_res_w[mask]

    def predict(self, mask: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts the components of the covariance matrix."""

        assert mask.shape == (self.n,) and mask.dtype == np.bool_
        assert b.shape == (self.n, self.k) and np.isfinite(b[mask]).all()

        return b, self.covariance, self.specific


@dataclass(slots=True)
class RiskModelState:
    target_offset: int
    min_periods: int

    # The last `target_offset + 1` feature cross-sections — just enough to
    # pair the oldest with the target that has now landed.
    pending: deque[np.ndarray]
    count: np.ndarray
    factor_model: FactorModelRegressor

    # Output covariance matrix is `Σ = X F Xᵀ + D`:
    out_exposures: np.ndarray  # X
    out_covariance: np.ndarray  # F
    out_specific: np.ndarray  # diagonal of D


class RiskModel:
    type Inputs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    type Outputs = tuple[np.ndarray, np.ndarray, np.ndarray]
    type Context = int
    type State = RiskModelState

    def __init__(
        self,
        universe_size: int,
        target_offset: int,
        min_periods: int,
        covariance_halflife: float,
        specific_halflife: float,
    ) -> None:
        assert universe_size > 0, "risk_model: universe_size must be positive"
        assert target_offset >= 0, "risk_model: target_offset must be non-negative"
        assert min_periods > 0, "risk_model: min_periods must be positive"
        assert covariance_halflife > 0.0, "risk_model: covariance_halflife must be positive"
        assert specific_halflife > 0.0, "risk_model: specific_halflife must be positive"

        self.universe_size = universe_size
        self.target_offset = target_offset
        self.min_periods = min_periods
        self.covariance_halflife = covariance_halflife
        self.specific_halflife = specific_halflife

    def init(self, inputs: Inputs) -> State:
        sample_signal, features, target, rebalance_signal, universe = inputs
        n, k = features.shape
        assert target.shape == (n,)

        return RiskModelState(
            target_offset=self.target_offset,
            min_periods=self.min_periods,
            pending=deque(maxlen=self.target_offset + 1),
            count=np.zeros((n,), dtype=np.int32),
            factor_model=FactorModelRegressor(
                n=n,
                k=k,
                outer_lambda=np.exp2(-1.0 / self.covariance_halflife),
                res_lambda=np.exp2(-1.0 / self.specific_halflife),
            ),
            out_exposures=np.zeros((n, k)),
            out_covariance=np.zeros((k, k)),
            out_specific=np.full((n,), np.nan),
        )

    @staticmethod
    def reset(_: Inputs, state: State) -> Outputs:
        return state.out_exposures, state.out_covariance, state.out_specific

    @staticmethod
    def compute(inputs: Inputs, state: State, _: Context) -> Outputs:
        sample_signal, features, target, rebalance_signal, universe = inputs
        n, k = features.shape
        assert target.shape == (n,)

        if sample_signal:
            state.pending.append(features)
            if len(state.pending) == state.target_offset + 1:
                # Adds one pair to the training set (cross-sectional).
                past_features = state.pending.popleft()
                valid = np.isfinite(past_features).all(axis=1) & np.isfinite(target)
                state.count += valid.astype(np.int32)
                state.factor_model.add_cross_section(valid, past_features, target)

        if rebalance_signal:
            # Predict `target_offset` periods ahead using the most recent features.
            valid = np.isfinite(features).all(axis=1) & (state.count >= state.min_periods)
            state.factor_model.fit(valid)
            state.out_exposures, state.out_covariance, state.out_specific = state.factor_model.predict(valid, features)

        return state.out_exposures, state.out_covariance, state.out_specific


def build(**kwargs) -> RiskModel:
    return RiskModel(**kwargs)
