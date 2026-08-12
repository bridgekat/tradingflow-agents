from dataclasses import dataclass
import numpy as np

from .estimator import ShrinkageEstimator
from .target import SHRINKAGE_TARGETS


@dataclass(slots=True)
class RiskModelState:
    target_offset: int
    min_periods: int

    count: np.ndarray
    estimator: ShrinkageEstimator

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
        rank: int,
        window: int,
        target: str = "diagonal",
        intensity: float | None = None,
    ) -> None:
        assert universe_size > 0, "risk_model: universe_size must be positive"
        assert target_offset >= 0, "risk_model: target_offset must be non-negative"
        assert min_periods > 0, "risk_model: min_periods must be positive"
        assert (
            covariance_halflife > 0.0
        ), "risk_model: covariance_halflife must be positive"
        assert specific_halflife > 0.0, "risk_model: specific_halflife must be positive"
        assert rank > 0, "risk_model: rank must be positive"
        assert window > 0, "risk_model: window must be positive"
        assert (
            target in SHRINKAGE_TARGETS
        ), f"risk_model: unknown shrinkage target {target!r}"
        assert (
            intensity is None or 0.0 <= intensity <= 1.0
        ), "risk_model: intensity must be in [0, 1]"

        self.universe_size = universe_size
        self.target_offset = target_offset
        self.min_periods = min_periods
        self.covariance_halflife = covariance_halflife
        self.specific_halflife = specific_halflife
        self.rank = rank
        self.window = window
        self.target = target
        self.intensity = intensity

    def init(self, inputs: Inputs) -> State:
        sample_signal, features, target, rebalance_signal, universe = inputs
        n, k = features.shape
        assert target.shape == (n,)
        assert universe.shape == (n,)

        rank = min(self.rank, n)
        return RiskModelState(
            target_offset=self.target_offset,
            min_periods=self.min_periods,
            count=np.zeros((n,), dtype=np.int32),
            estimator=ShrinkageEstimator(
                n=n,
                rank=rank,
                window=self.window,
                outer_lambda=np.exp2(-1.0 / self.covariance_halflife),
                var_lambda=np.exp2(-1.0 / self.specific_halflife),
                target=SHRINKAGE_TARGETS[self.target](),
                intensity=self.intensity,
            ),
            out_exposures=np.zeros((n, rank)),
            out_covariance=np.eye(rank),
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
        assert universe.shape == (n,)

        if sample_signal:
            # Adds one cross-section to the training set.
            valid = np.isfinite(target)
            state.count += valid.astype(np.int32)
            state.estimator.add_cross_section(valid, target)

        if rebalance_signal:
            # Predict any periods ahead.
            valid = state.count >= state.min_periods
            mask = valid & (universe > 0.0)
            state.estimator.fit(mask)
            state.out_exposures, state.out_covariance, state.out_specific = (
                state.estimator.predict(mask, features)
            )

        return state.out_exposures, state.out_covariance, state.out_specific


def build(**kwargs) -> RiskModel:
    return RiskModel(**kwargs)
