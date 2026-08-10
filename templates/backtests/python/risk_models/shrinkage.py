from dataclasses import dataclass
import sys
import numpy as np
import scipy.sparse.linalg


@dataclass(slots=True)
class ShrinkageEstimator:
    """A windowed shrinkage estimator for covariance (risk) prediction.
    It estimates the covariance matrix of the target `r` directly from its
    historical sample covariance, and regularizes the estimate by shrinking
    it towards a structured target (Ledoit-Wolf 2003, Schäfer & Strimmer 2005):

    ```
    Σ* := (1 - δ) S + δ T
    ```

    where the random matrix `S` is a sample covariance estimator for `Cov[r]`,
    random matrix `T` is some chosen shrinkage target, and `δ ∈ [0, 1]` is
    some chosen shrinkage intensity.

    For now, we take `T` to be the diagonal of `S`. Shrinking towards `T` keeps
    the sample variances, only the correlations are shrunk towards zero.

    The matrix `S` is first estimated by taking exponentially-weighted moving
    average over the outer product `r rᵀ`, and over the last `window`
    cross-sections. This estimator deliberately ignores the assets' risk premium
    `E[r]` - which is close to 0 compared to the second moment, and much harder
    to estimate reliably. The sample correlation matrix `ρ` is then computed as
    `ρ := S / (σ σᵀ)` where `σ²` is the diagonal of `S`. Then, the per-asset
    variances `σ²` are re-estimated by taking exponentially-weighted moving
    average over the squared returns `r²` using a different decay rate, and
    `S` is refined via `S := ρ * (σ σᵀ)` where `σ²` is now the new estimate.

    > The variances of stock returns can spike quickly, requiring a faster
    > decay; the correlations between stock returns are slower-moving and
    > noisier, so a slower decay is preferable.

    When no fixed intensity is given, `δ` is estimated from the data,
    following Schäfer & Strimmer: the ratio of the total estimation variance
    of the off-diagonal correlations `ρ` to their total squared magnitude,

    ```
    δ* = Σ_{i≠j} Var[ρ_ij] / Σ_{i≠j} ρ²_ij
    ```

    clamped to `[0, 1]`. Here `Var[ρ_ij]` is estimated as well: larger means
    more uncertainty and noise, which shrinks `S` harder towards `T`.
    As the sample grows, `Var[ρ_ij]` decreases and `δ*` converges to 0.

    Finally, the shrunk estimate `Σ*` is truncated to a specified `rank`,
    preserving only the largest positive eigenpairs. This reduces the burden
    of downstream Markowitz optimization, while also putting the covariance
    matrix into a factor-model form `Σ* = X F Xᵀ + D`. This approximation
    preserves per-asset variances; correlations beyond the kept eigenpairs
    are lost.

    > The whole process is implemented matrix-free whenever possible, never
    > forming the full `n × n` matrices (where `n` is the number of assets).
    > It does need `t × n` matrices for the last `t` cross-sections.
    > This is intentional, given that `t` is typically smaller (e.g. 250)
    > compared to `n` (e.g. 6000).
    """

    n: int
    rank: int
    window: int
    outer_lambda: float
    var_lambda: float
    intensity: float | None

    # Ring buffer of the last `window` cross-sections. `buf_pos` is the next
    # slot to overwrite, `buf_len` the number of filled slots.
    buf_len: int
    buf_pos: int
    buf_w: np.ndarray
    buf_r: np.ndarray

    exposures: np.ndarray
    covariance: np.ndarray
    specific: np.ndarray

    def __init__(
        self,
        n: int,
        rank: int,
        window: int,
        outer_lambda: float,
        var_lambda: float,
        intensity: float | None,
    ) -> None:
        """Initializes the shrinkage estimator for `n` assets over a rolling
        `window` of cross-sections, truncated to `rank` eigenpairs.
        """

        self.n = n
        self.rank = rank
        self.window = window
        self.outer_lambda = outer_lambda
        self.var_lambda = var_lambda
        self.intensity = intensity

        self.buf_len = 0
        self.buf_pos = 0
        self.buf_w = np.zeros((window, n))
        self.buf_r = np.zeros((window, n))

        self.exposures = np.zeros((n, rank))
        self.covariance = np.eye(rank)
        self.specific = np.full((n,), np.nan)

    def add_cross_section(self, mask: np.ndarray, r: np.ndarray) -> None:
        """Adds a cross-section of `r[mask]` to the window, evicting the
        oldest one when full.
        """

        assert mask.shape == (self.n,) and mask.dtype == np.bool_
        assert r.shape == (self.n,) and np.isfinite(r[mask]).all()

        if not mask.any():
            return

        self.buf_w[self.buf_pos] = mask
        self.buf_r[self.buf_pos] = np.where(mask, r, 0.0)
        self.buf_pos = (self.buf_pos + 1) % self.window
        self.buf_len = min(self.buf_len + 1, self.window)

    def fit(self, mask: np.ndarray) -> None:
        """Calculates and record model parameters."""

        t = self.buf_len
        ws, rs = self.buf_w[:t], self.buf_r[:t]
        ages = (self.buf_pos - 1 - np.arange(t)) % self.window

        # Drop assets never observed inside the window (avoid division by 0).
        mask = mask & (ws.sum(axis=0) > 0.0)

        if not mask.any():
            print("risk_model: no samples to fit", file=sys.stderr)
            return

        # Restrict the window to the active columns only.
        m = mask.sum()
        ws = ws[:, mask]
        rs = rs[:, mask]

        # Per-sample weights from sample ages (newest slot has age 0).
        outer_ds = np.pow(self.outer_lambda, ages)
        var_ds = np.pow(self.var_lambda, ages)

        # We have `count := Σ w wᵀ` and `ssd := Σ r rᵀ` where sums are over `t`.
        # Apply decay to get weighted sums `Σ d w wᵀ` and `Σ d r rᵀ`.
        hrs = np.sqrt(outer_ds)[:, None] * rs  # `d r rᵀ = (√d r) (√d r)ᵀ`

        # The pairwise-complete weighted sample covariance estimator is
        # `S := Σ d r rᵀ / Σ d w wᵀ`. However, this matrix is not necessarily
        # positive semi-definite and harder to work with, so we instead use
        # `S := Σ d r rᵀ / √(Σ d w²) √(Σ d w²)ᵀ`. In particular, each pair's
        # sample count is now approximated by the geometric mean of the two
        # assets' sample counts, which is always an overestimate of the true
        # pairwise sample count, leading to an underestimate of the sample
        # covariance. In both cases, the diagonal of `S` is the same:
        # `σ² := Σ d r² / Σ d w²`, and the sample correlation estimator
        # is then `ρ := S / (σ σᵀ)`, which in the latter case rewrites to
        # `ρ := Σ d r rᵀ / √(Σ d r²) √(Σ d r²)ᵀ = Σ u uᵀ` where
        # `u := √d r / √(Σ d r²) = normalize(√d r)`.
        #
        # Whether this is a true underestimation of pairwise correlation
        # depends on interpretation: consider two assets observed to be almost
        # perfectly correlated whenever both are traded, but one has been
        # trading for a long time, while the other is recently listed.
        # Pairwise-complete correlation would be near 1, but the
        # geometric-mean correlation can be much lower, giving a perceived
        # opportunity for diversification, potentially underestimating risk.
        # But it can correct itself: as time passes, if the correlation
        # persists, the geometric-mean correlation will converge to 1 as well,
        # and exponential weighting makes this faster.
        norm_hrs = np.linalg.norm(hrs, axis=0)
        us = hrs / np.where(norm_hrs > 0.0, norm_hrs, 1.0)  # `u = normalize(√d r)`

        # `is_narrow` marks the small regime where `m × m` products are cheaper
        # than their `window × window` or matrix-free counterparts.
        is_narrow = m <= max(t, 512) or 4 * min(self.rank, m) >= m

        if self.intensity is not None:
            delta = self.intensity
        else:
            # For shrinkage intensity, we assume that `r rᵀ` is i.i.d. across
            # `t`, and use weighted sample means to estimate its mean and
            # variance:
            #
            # - `E[r_i r_j] ≈ Σ d (r_i r_j) / Σ d`
            # - `Var[r_i r_j] ≈ Σ d² (r_i r_j - E[r_i r_j])² / Σ d²`
            #
            # Using these, we derive an estimator for `Var[ρ_ij]`:
            #
            # ```
            # Var[ρ_ij]
            # = Var[Σ u_i u_j]
            # = Var[Σ d r_i r_j / √(Σ d r²_i) √(Σ d r²_j)]
            # ≈ Var[Σ d r_i r_j] / (Σ d r²_i) (Σ d r²_j)    (delta method omitting cross-terms!)
            # = (Σ d²) Var[r_i r_j] / (Σ d r²_i) (Σ d r²_j)    (i.i.d. assumption)
            # ≈ Σ d² (r_i r_j - E[r_i r_j])² / (Σ d r²_i) (Σ d r²_j)    (variance estimator)
            # ≈ Σ d² (r_i r_j - Σ d (r_i r_j) / Σ d)² / (Σ d r²_i) (Σ d r²_j)    (mean estimator)
            # = Σ (u_i u_j - ρ_ij · d / Σ d)²    (rewrites)
            # ```
            #
            # In matrix notation, the matrix of variances is
            # `Var[ρ] ≈ Σ (u uᵀ - ρ · d / Σ d)² = Σ (A - 2 B + C)`, where:
            #
            # - `A := Σ (u uᵀ)² = Σ u² u²ᵀ`
            # - `B := Σ (u uᵀ) · ρ · d / Σ d = ρ · Σ (u uᵀ) · d / Σ d`
            # - `C := Σ (ρ · d / Σ d)² = ρ² · Σ (d / Σ d)²`
            #
            # The numerator of shrinkage intensity is then obtained by
            # left- and right-multiplying `Var[ρ]` by the all-ones vector, then
            # subtracting the trace. The denominator is similarly obtained from
            # `ρ²`. The intensity is then clamped to `[0, 1]`.
            u2s = np.square(us)
            rho_diag = u2s.sum(axis=0)
            ds_unit = outer_ds / outer_ds.sum()  # `d / Σ d`
            ds_unit_ss = np.square(ds_unit).sum()  # `Σ (d / Σ d)²`
            a = np.square(u2s.sum(axis=1)).sum()  # sum over `A`
            a_diag = np.square(u2s).sum()  # diagonal sum over `A`
            b_diag = np.dot(ds_unit @ u2s, rho_diag)  # diagonal sum over `B`
            d_diag = np.square(rho_diag).sum()  # diagonal sum over `ρ²`
            c_diag = d_diag * ds_unit_ss  # diagonal sum over `C`

            # Summation over `B` and `ρ²` both require summation of the shape
            # `Σ_stij u_si u_sj u_ti u_tj`, which can be done in either
            # `O(m² t)` or `O(t² m)` depending on which axis is smaller.
            if is_narrow:
                rho = us.T @ us
                b = (rho * (us.T @ (us * ds_unit[:, None]))).sum()  # sum over `B`
                d = np.square(rho).sum()  # sum over `ρ²`
                c = d * ds_unit_ss  # sum over `C`
            else:
                g2 = np.square(us @ us.T)
                b = (g2 @ ds_unit).sum()  # sum over `B`
                d = g2.sum()  # sum over `ρ²`
                c = d * ds_unit_ss  # sum over `C`

            num = (a - a_diag) - 2.0 * (b - b_diag) + (c - c_diag)
            denom = d - d_diag
            delta = np.clip(num / (denom if denom > 0.0 else 1.0), 0.0, 1.0)

        # Per-asset variances under their own (faster) decay refine the
        # diagonal: `Σ* = (1 - δ) ρ · (σ σᵀ) + δ diag(σ²)`.
        dws = var_ds @ ws
        dr2s = var_ds @ np.square(rs)
        sigma2 = dr2s / np.where(dws > 0.0, dws, 1.0)
        sigma = np.sqrt(sigma2)

        # Truncate the shrunk covariance matrix to the specified rank.
        k = min(self.rank, m)
        if is_narrow:
            try:
                shrunk = (1.0 - delta) * ((us.T @ us) * np.outer(sigma, sigma))
                shrunk[np.diag_indices(m)] += delta * sigma2
                lam, vec = np.linalg.eigh(shrunk)  # ascending eigenvalues
                lam, vec = lam[-k:], vec[:, -k:]

            except np.linalg.LinAlgError:
                print(
                    "risk_model: eigendecomposition did not converge",
                    file=sys.stderr,
                )
                return

        else:
            try:
                # `Σ*` is only needed through matrix-vector products
                # `Σ* z = (1 - δ) σ · Uᵀ U (σ · z) + δ σ² · z`, each
                # `O(t m)` right-to-left.
                def matvec(z: np.ndarray) -> np.ndarray:
                    shrunk = sigma * (us.T @ (us @ (sigma * z)))
                    target = sigma2 * z
                    return (1.0 - delta) * shrunk + delta * target

                # Create the matrix-free linear operator for `Σ*`.
                op = scipy.sparse.linalg.LinearOperator(
                    shape=(m, m), matvec=matvec, dtype=np.float64  # type: ignore
                )

                # Fixed start vector keeps runs deterministic.
                lam, vec = scipy.sparse.linalg.eigsh(
                    op, k=k, which="LA", v0=np.full(m, 1.0 / np.sqrt(m))
                )

            except scipy.sparse.linalg.ArpackError:
                print(
                    "risk_model: sparse eigendecomposition did not converge",
                    file=sys.stderr,
                )
                return

        lam, vec = lam[::-1], vec[:, ::-1]  # descending eigenvalues
        kept = int((lam > 0.0).sum())
        self.exposures.fill(0.0)
        self.covariance.fill(0.0)
        self.specific.fill(np.nan)
        self.exposures[mask, :kept] = vec[:, :kept]
        self.covariance[:kept, :kept] = np.diag(lam[:kept])

        # The diagonal left unexplained by the kept eigenpairs becomes the
        # specific variance, preserving per-asset variances.
        sigma2_kept = np.square(vec[:, :kept]) @ lam[:kept]
        self.specific[mask] = np.maximum(sigma2 - sigma2_kept, 0.0)

    def predict(
        self, mask: np.ndarray, b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts the components of the covariance matrix."""

        assert mask.shape == (self.n,) and mask.dtype == np.bool_

        return self.exposures, self.covariance, self.specific


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
        shrinkage: float | None = None,
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
            shrinkage is None or 0.0 <= shrinkage <= 1.0
        ), "risk_model: shrinkage must be in [0, 1]"

        self.universe_size = universe_size
        self.target_offset = target_offset
        self.min_periods = min_periods
        self.covariance_halflife = covariance_halflife
        self.specific_halflife = specific_halflife
        self.rank = rank
        self.window = window
        self.shrinkage = shrinkage

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
                intensity=self.shrinkage,
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
