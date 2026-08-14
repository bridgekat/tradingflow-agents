from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class FactorShareState:
    out: float


class FactorShare:
    """The fraction of predicted variance a factor risk model
    `Σ = X F Xᵀ + D` carries in its factors rather than its specific
    diagonal: `Σᵢ (X F Xᵀ)ᵢᵢ / Σᵢ ((X F Xᵀ)ᵢᵢ + dᵢ)`.
    """

    type Inputs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    type Outputs = float
    type Context = int
    type State = FactorShareState

    def init(self, inputs: Inputs) -> State:
        sample_signal, universe, exposures, covariance, specific = inputs
        n, k = exposures.shape
        assert universe.shape == (n,)
        assert covariance.shape == (k, k)
        assert specific.shape == (n,)

        return FactorShareState(out=np.nan)

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
            if m > 0:
                factor_var = np.einsum("ij,ji->i", x @ covariance, x.T)
                factor = factor_var.sum()
                total = (factor_var + d).sum()
                state.out = factor / total if total > 0.0 else np.nan
            else:
                state.out = np.nan

        return state.out


def build(**kwargs) -> FactorShare:
    return FactorShare(**kwargs)
