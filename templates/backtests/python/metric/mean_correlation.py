from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class MeanCorrelationState:
    out: float


class MeanCorrelation:
    """The mean off-diagonal correlation a factor risk model
    `Σ = X F Xᵀ + D` predicts across the cross-section.
    """

    type Inputs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    type Outputs = float
    type Context = int
    type State = MeanCorrelationState

    def init(self, inputs: Inputs) -> State:
        sample_signal, universe, exposures, covariance, specific = inputs
        n, k = exposures.shape
        assert universe.shape == (n,)
        assert covariance.shape == (k, k)
        assert specific.shape == (n,)

        return MeanCorrelationState(out=np.nan)

    @staticmethod
    def reset(_: Inputs, state: State) -> Outputs:
        return state.out

    @staticmethod
    def compute(inputs: Inputs, state: State, _: Context) -> Outputs:
        sample_signal, universe, exposures, covariance, specific = inputs

        if sample_signal:
            valid = np.isfinite(exposures).all(axis=1) & np.isfinite(specific)
            mask = valid & (universe > 0.0)
            m, x, d = mask.sum(), exposures[mask], specific[mask]
            if m > 1:
                factor_var = np.einsum("ij,ji->i", x @ covariance, x.T)
                total_var = factor_var + d
                m = (total_var > 0.0).sum()
                if m > 1:
                    s = np.where(total_var > 0.0, 1.0 / np.sqrt(total_var), 0.0)
                    xs = x.T @ s
                    total = xs @ (covariance @ xs) + d @ np.square(s)
                    state.out = (total - m) / (m * (m - 1))
                else:
                    state.out = np.nan
            else:
                state.out = np.nan

        return state.out


def build(**kwargs) -> MeanCorrelation:
    return MeanCorrelation(**kwargs)
