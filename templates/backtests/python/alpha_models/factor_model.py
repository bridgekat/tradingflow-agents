from dataclasses import dataclass
from collections import deque
import sys
import numpy as np


@dataclass(slots=True)
class FactorModelRegressor:
    """An incremental factor model for mean (alpha) prediction.
    It assumes the target `r` is a vector process following (at each period):

    ```
    r = B f + ε
    ```

    for some predictable matrix process `B`, adapted vector process `f`, and
    adapted vector process `ε` with uncorrelated components (conditioned
    on past periods):

    ```
    E[ε] = 0
    Cov[ε] = diag(σ²_0, ..., σ²_{n - 1})
    Cov[f, ε] = 0
    ```

    where `σ²` is again a predictable vector process. Here we omit the
    time subscripts and other notational details: for example, `E[ε]` actually
    denotes `E[ε_t | 𝓕_{t - 1}]`.

    Each period, the matrix `B` is provided as input, and `f` is estimated by
    solving the L2-regularized least-squares problem:

    ```
    min_f { (1 / m) ‖X f - r‖² + δ ‖f‖² }
    ```

    where `m` is the number of rows in `X` and `r`, and `δ` is the
    regularization parameter. Then `E[f]` is estimated by taking
    exponentially-weighted moving average over the per-period estimates.

    The model assumptions directly leads to:

    ```
    E[r] = B E[f]
    ```

    which is produced directly.
    """

    n: int
    k: int
    delta: float
    f_lambda: float

    sum_f_w: float
    sum_f_xw: np.ndarray

    premium: np.ndarray

    def __init__(self, n: int, k: int, delta: float, f_lambda: float) -> None:
        """Initializes the factor model for `n` assets and `k` factors."""

        self.n = n
        self.k = k
        self.delta = delta
        self.f_lambda = f_lambda

        self.sum_f_w = 0.0
        self.sum_f_xw = np.zeros(k)

        self.premium = np.zeros(k)

    def add_cross_section(self, mask: np.ndarray, b: np.ndarray, r: np.ndarray) -> None:
        """Adds a cross-section of `(b[mask], r[mask])` to the accumulator."""

        bm, rm = b[mask], r[mask]
        assert mask.shape == (self.n,) and mask.dtype == np.bool_
        assert b.shape == (self.n, self.k) and np.isfinite(bm).all()
        assert r.shape == (self.n,) and np.isfinite(rm).all()

        if not mask.any():
            return

        try:
            f = np.linalg.solve(
                bm.T @ bm
                + mask.sum() * self.delta * np.eye(self.k),  # invertible with δ > 0
                bm.T @ rm,
            )
        except np.linalg.LinAlgError:
            print("alpha_model: matrix is ill-conditioned", file=sys.stderr)
            return

        self.sum_f_w = self.sum_f_w * self.f_lambda + 1.0
        self.sum_f_xw = self.sum_f_xw * self.f_lambda + f

    def fit(self, mask: np.ndarray) -> None:
        """Calculates and record model parameters."""

        if self.sum_f_w == 0.0:
            print("alpha_model: no samples to fit", file=sys.stderr)
            return

        self.premium = self.sum_f_xw / self.sum_f_w

    def predict(self, mask: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Predicts the mean vector."""

        assert mask.shape == (self.n,) and mask.dtype == np.bool_
        assert b.shape == (self.n, self.k) and np.isfinite(b[mask]).all()

        return b @ self.premium


@dataclass(slots=True)
class AlphaModelState:
    target_offset: int
    min_periods: int

    # Optional column subset of the feature panel (indices into the panel's
    # column axis); `None` keeps every column.
    keep: np.ndarray | None

    # The last `target_offset + 1` feature cross-sections — just enough to
    # pair the oldest with the target that has now landed.
    pending: deque[np.ndarray]
    count: np.ndarray
    factor_model: FactorModelRegressor
    out_mu: np.ndarray


class AlphaModel:
    type Inputs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    type Outputs = np.ndarray
    type Context = int
    type State = AlphaModelState

    def __init__(
        self,
        universe_size: int,
        target_offset: int,
        min_periods: int,
        ridge_l2: float,
        premium_halflife: float,
        keep: list[int] | None = None,
    ) -> None:
        assert universe_size > 0, "alpha_model: universe_size must be positive"
        assert target_offset > 0, "alpha_model: target_offset must be positive"
        assert min_periods > 0, "alpha_model: min_periods must be positive"
        assert ridge_l2 > 0.0, "alpha_model: ridge_l2 must be positive"
        assert premium_halflife > 0.0, "alpha_model: premium_halflife must be positive"

        self.universe_size = universe_size
        self.target_offset = target_offset
        self.min_periods = min_periods
        self.ridge_l2 = ridge_l2
        self.premium_halflife = premium_halflife
        self.keep = None if not keep else np.asarray(keep, dtype=np.intp)

    def init(self, inputs: Inputs) -> State:
        sample_signal, features, target, rebalance_signal, universe = inputs
        n, k = features.shape
        assert target.shape == (n,)
        assert universe.shape == (n,)

        if self.keep is not None:
            assert (0 <= self.keep).all() and (self.keep < k).all()
            k = len(self.keep)

        return AlphaModelState(
            target_offset=self.target_offset,
            min_periods=self.min_periods,
            keep=self.keep,
            pending=deque(maxlen=self.target_offset + 1),
            count=np.zeros((n,), dtype=np.int32),
            factor_model=FactorModelRegressor(
                n=n,
                k=k,
                delta=self.ridge_l2,
                f_lambda=np.exp2(-1.0 / self.premium_halflife),
            ),
            out_mu=np.full((n,), np.nan),
        )

    @staticmethod
    def reset(_: Inputs, state: State) -> Outputs:
        return state.out_mu

    @staticmethod
    def compute(inputs: Inputs, state: State, _: Context) -> Outputs:
        sample_signal, features, target, rebalance_signal, universe = inputs
        n, k = features.shape
        assert target.shape == (n,)
        assert universe.shape == (n,)

        if state.keep is not None:
            features = features[:, state.keep]

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
            valid = np.isfinite(features).all(axis=1) & (
                state.count >= state.min_periods
            )
            mask = valid & (universe > 0.0)
            state.factor_model.fit(mask)
            state.out_mu = state.factor_model.predict(mask, features)

        return state.out_mu


def build(**kwargs) -> AlphaModel:
    return AlphaModel(**kwargs)
