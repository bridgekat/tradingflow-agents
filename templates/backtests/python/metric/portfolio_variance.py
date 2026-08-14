from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class PortfolioVarianceState:
    out: float


class PortfolioVariance:
    """The variance of portfolio return a factor risk model
    `Σ = X F Xᵀ + D` predicts, for a given weights vector `w`: `wᵀ Σ w`.
    """

    type Inputs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    type Outputs = float
    type Context = int
    type State = PortfolioVarianceState

    def init(self, inputs: Inputs) -> State:
        sample_signal, weights, exposures, covariance, specific = inputs
        n, k = exposures.shape
        assert weights.shape == (n,)
        assert covariance.shape == (k, k)
        assert specific.shape == (n,)

        return PortfolioVarianceState(out=np.nan)

    @staticmethod
    def reset(_: Inputs, state: State) -> Outputs:
        return state.out

    @staticmethod
    def compute(inputs: Inputs, state: State, _: Context) -> Outputs:
        sample_signal, weights, exposures, covariance, specific = inputs

        if sample_signal:
            mask = weights != 0.0
            m, w, x, d = mask.sum(), weights[mask], exposures[mask], specific[mask]
            if m > 0:
                t = x.T @ w
                state.out = t @ (covariance @ t) + np.dot(np.square(w), d)
            else:
                state.out = 0.0

        return state.out


def build(**kwargs) -> PortfolioVariance:
    return PortfolioVariance(**kwargs)
