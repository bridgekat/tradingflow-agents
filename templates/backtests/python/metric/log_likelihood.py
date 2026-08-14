from dataclasses import dataclass
import numpy as np


def log_likelihood(x: np.ndarray, f: np.ndarray, d: np.ndarray, v: np.ndarray) -> float:
    """Returns `log f(v)` where `f(v) = 1 / √det(2π Σ) exp(-1/2 vᵀ Σ⁻¹ v)`,
    `Σ = X F Xᵀ + D` and `D` is diagonal, at a cost of `O(k² n)`:

    ```
    log f(v)
    = log (1 / √det(2π Σ)) + log exp(-1/2 vᵀ Σ⁻¹ v)
    = -1/2 log det(2π Σ) - 1/2 vᵀ Σ⁻¹ v
    = -1/2 (n log(2π) + log det Σ + vᵀ Σ⁻¹ v)
    ```

    The determinant can be computed efficiently in a similar way as how
    [`portfolio.minimum_variance.inv_matvec`] computes `Σ⁻¹ v`:

    ```
    log det Σ
    = log det(X F Xᵀ + D)
    = log det D + log det(I + D⁻¹ X F Xᵀ)
    = log det D + log det(I + Xᵀ D⁻¹ X F)    (Weinstein–Aronszajn identity)
    = log det D + log det(I + X̄ᵀ X̄ F)    (where Rᵀ R = D⁻¹ and X̄ = R X)
    ```
    """

    n, k = x.shape
    r = 1.0 / np.sqrt(d)
    rv = r * v
    rx = x * r[:, None]
    cap = np.eye(k) + (rx.T @ rx) @ f
    y = np.linalg.solve(cap, rx.T @ rv)
    sign, logdet_cap = np.linalg.slogdet(cap)
    if sign <= 0.0:  # matrix has eigenvalues in [1, +∞) if `F` is PSD
        return np.nan
    logdet = np.log(d).sum() + logdet_cap
    quad = v @ (r * (rv - rx @ (f @ y)))
    return -0.5 * (n * np.log(2.0 * np.pi) + logdet + quad)


@dataclass(slots=True)
class LogLikelihoodState:
    specific_floor: float
    out: float


class LogLikelihood:
    """The log-likelihood of a return vector `r` under a factor risk model
    `Σ = X F Xᵀ + D`, assuming `r ~ N(0, Σ)` (normal distribution),
    divided by the number of assets `m`.
    """

    type Inputs = tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]
    type Outputs = float
    type Context = int
    type State = LogLikelihoodState

    def __init__(self, specific_floor: float = 1e-8) -> None:
        assert specific_floor > 0.0, "log_likelihood: specific_floor must be positive"

        self.specific_floor = specific_floor

    def init(self, inputs: Inputs) -> State:
        sample_signal, returns, universe, exposures, covariance, specific = inputs
        n, k = exposures.shape
        assert returns.shape == (n,)
        assert universe.shape == (n,)
        assert covariance.shape == (k, k)
        assert specific.shape == (n,)

        return LogLikelihoodState(
            specific_floor=self.specific_floor,
            out=np.nan,
        )

    @staticmethod
    def reset(_: Inputs, state: State) -> Outputs:
        return state.out

    @staticmethod
    def compute(inputs: Inputs, state: State, _: Context) -> Outputs:
        sample_signal, returns, universe, exposures, covariance, specific = inputs

        if sample_signal:
            valid = (
                np.isfinite(returns)
                & np.isfinite(exposures).all(axis=1)
                & np.isfinite(specific)
            )
            mask = valid & (universe > 0.0)
            m, r, x, d = mask.sum(), returns[mask], exposures[mask], specific[mask]
            if m > 0:
                ll = log_likelihood(
                    x, covariance, np.maximum(d, state.specific_floor), r
                )
                state.out = ll / m
            else:
                state.out = np.nan

        return state.out


def build(**kwargs) -> LogLikelihood:
    return LogLikelihood(**kwargs)
