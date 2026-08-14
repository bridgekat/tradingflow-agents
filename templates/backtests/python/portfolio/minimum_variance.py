from dataclasses import dataclass
import sys
import numpy as np


def inv_matvec(
    x: np.ndarray, f: np.ndarray, d: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Returns `Σ⁻¹ v` where `Σ = X F Xᵀ + D` and `D` is diagonal,
    at a cost of `O(k² n)`, using a variant of the Woodbury identity:

    ```
    (X F Xᵀ + D)⁻¹ v
    = D⁻¹ v - D⁻¹ X F (I + Xᵀ D⁻¹ X F)⁻¹ Xᵀ D⁻¹ v
    = Rᵀ R v - Rᵀ X̄ F (I + X̄ᵀ X̄ F)⁻¹ X̄ᵀ R v    (where Rᵀ R = D⁻¹ and X̄ = R X)
    ```

    If `F` is PSD, it is guaranteed that `I + X̄ᵀ X̄ F` is invertible:
    for any `v ∈ Ker(I + X̄ᵀ X̄ F)`,

    ```
    (I + X̄ᵀ X̄ F) v = 0
    ⇒ X̄ᵀ X̄ F v = -v
    ⇒ vᵀ Fᵀ X̄ᵀ X̄ F v = -vᵀ Fᵀ v
    ⇒ vᵀ Fᵀ v ≤ 0
    ⇒ vᵀ Fᵀ v = 0    and    v ∈ Ker(F)
    ```

    which gives `(I + X̄ᵀ X̄ F) v = v + 0 = v`, so `v = 0`.

    Similarly, we can show that all eigenvalues of `I + X̄ᵀ X̄ F` are in
    `[1, +∞)`. The solve's condition number is therefore bounded by
    the largest eigenvalues of `F` and `X̄ᵀ X̄ = Xᵀ D⁻¹ X`.
    """

    n, k = x.shape
    r = 1.0 / np.sqrt(d)
    rv = r * v
    rx = x * r[:, None]
    cap = np.eye(k) + (rx.T @ rx) @ f
    y = np.linalg.solve(cap, rx.T @ rv)
    return r * (rv - rx @ (f @ y))


@dataclass(slots=True)
class MinimumVarianceSolver:
    """An efficient global-minimum-variance (GMV) portfolio optimizer for
    risk model analysis. It solves the convex optimization problem:

    ```
    min_w { wᵀ Σ w }
    ```

    where `Σ = X F Xᵀ + D`, `F` is symmetric positive semi-definite, `X` is
    (narrow) rectangular, and `D` is diagonal with non-negative entries.

    Under the constraint `𝟙ᵀ w = 1`, we can write a perturbation function
    `F` such that the original problem is `min_w F(w, 0)`:

    ```
    F(w, λ) := wᵀ Σ w - 𝕀_{𝟙ᵀ w = 1 + λ}
    ```

    which corresponds to the Lagrangian

    ```
    L(w, λ*) := wᵀ Σ w - λ* (𝟙ᵀ w - 1)
    ```

    At optimality, the partial derivatives of `L` with respect to `w` and `λ*`
    both vanish, which gives the unique closed-form solution assuming `Σ` is
    invertible:

    ```
    w = Σ⁻¹ 𝟙 / (𝟙ᵀ Σ⁻¹ 𝟙)
    ```

    Setting a floor on the diagonal of `D` ensures that `Σ` is invertible,
    and improves numerical stability of the solution.
    """

    specific_floor: float

    def __init__(self, specific_floor: float) -> None:
        """Initializes the optimizer."""

        self.specific_floor = specific_floor

    def solve(
        self,
        exposures: np.ndarray,  # X
        covariance: np.ndarray,  # F
        specific: np.ndarray,  # diagonal of D
    ) -> np.ndarray | None:
        """Solves the optimization problem and returns the optimal weights."""

        n, k = exposures.shape
        assert exposures.shape == (n, k) and np.isfinite(exposures).all()
        assert covariance.shape == (k, k) and np.isfinite(covariance).all()
        assert specific.shape == (n,) and np.isfinite(specific).all()

        try:
            num = inv_matvec(
                exposures,
                covariance,
                np.maximum(specific, self.specific_floor),
                np.ones(n),
            )
        except np.linalg.LinAlgError:
            print("portfolio: matrix is ill-conditioned", file=sys.stderr)
            return None

        denom = num.sum()
        return num / denom if denom > 0.0 else None


@dataclass(slots=True)
class PortfolioState:
    solver: MinimumVarianceSolver
    out_weights: np.ndarray


class Portfolio:
    type Inputs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    type Outputs = np.ndarray
    type Context = int
    type State = PortfolioState

    def __init__(self, specific_floor: float = 1e-8) -> None:
        assert specific_floor > 0.0, "portfolio: specific_floor must be positive"

        self.specific_floor = specific_floor

    def init(self, inputs: Inputs) -> State:
        rebalance_signal, universe, exposures, covariance, specific = inputs
        n, k = exposures.shape
        assert universe.shape == (n,)
        assert covariance.shape == (k, k)
        assert specific.shape == (n,)

        return PortfolioState(
            solver=MinimumVarianceSolver(
                specific_floor=self.specific_floor,
            ),
            out_weights=np.zeros((n,)),
        )

    @staticmethod
    def reset(_: Inputs, state: State) -> Outputs:
        return state.out_weights

    @staticmethod
    def compute(inputs: Inputs, state: State, _: Context) -> Outputs:
        rebalance_signal, universe, exposures, covariance, specific = inputs
        n, k = exposures.shape
        assert universe.shape == (n,)
        assert covariance.shape == (k, k)
        assert specific.shape == (n,)

        if rebalance_signal:
            valid = np.isfinite(exposures).all(axis=1) & np.isfinite(specific)
            mask = valid & (universe > 0.0)
            if not mask.any():
                print("portfolio: no active stocks to solve", file=sys.stderr)
            else:
                w = state.solver.solve(exposures[mask], covariance, specific[mask])
                if w is not None:
                    weights = np.zeros_like(state.out_weights)
                    weights[mask] = w
                    state.out_weights = weights

        return state.out_weights


def build(**kwargs) -> Portfolio:
    return Portfolio(**kwargs)
