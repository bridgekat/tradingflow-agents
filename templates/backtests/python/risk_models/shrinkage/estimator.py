from dataclasses import dataclass
import sys
import numpy as np
import scipy.sparse.linalg

from .target import Target


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

    # The sample covariance

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

    # The shrinkage target

    Four classical shrinkage targets `T` are available, written via the sample
    correlations `ρ` and refined standard deviations `σ` defined above:

    - `diagonal`: `T := diag(σ²)` - correlations shrink towards zero,
      per-asset variances are kept.
    - `constant-correlation` (Ledoit-Wolf 2004): `T := r̄ σ σᵀ` off the
      diagonal, `σ²` on the diagonal, with `r̄` the mean off-diagonal sample
      correlation - correlations shrink towards their common mean, variances
      are kept.
    - `common-covariance` (Schäfer-Strimmer target C): `T := c̄` off the
      diagonal, `v̄` on the diagonal, the mean sample covariance and mean
      sample variance - everything shrinks towards its cross-sectional mean.
    - `single-index` (Ledoit-Wolf 2003): `T := (q σ) (q σ)ᵀ` off the
      diagonal, `σ²` on the diagonal, with `q` the asset-market sample
      correlations - correlations shrink towards a one-factor market model,
      variances are kept.

    Every target is rank-one plus diagonal and positive semi-definite, so
    applying one never requires a dense `n × n` matrix.

    # The shrinkage intensity

    When no fixed intensity is given, `δ` is estimated from the data,
    following Schäfer & Strimmer: the ratio of the total estimation variance
    of the off-diagonal correlations `ρ` to their total squared distance
    from the target's correlations `R`,

    ```
    δ* = Σ_{i≠j} Var[ρ_ij] / Σ_{i≠j} (ρ_ij - R_ij)²
    ```

    clamped to `[0, 1]`. Here `Var[ρ_ij]` is estimated as well: larger means
    more uncertainty and noise, which shrinks `S` harder towards `T`.
    As the sample grows, `Var[ρ_ij]` decreases and `δ*` converges to 0.

    # The low-rank approximation

    Finally, the shrunk estimate `Σ*` is truncated to a specified `rank`,
    preserving only the largest positive eigenpairs. The diagonal is then
    corrected to preserve the per-asset variances. This reduces the burden
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
    target: Target
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
        target: Target,
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
        self.target = target
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
        hrs = np.sqrt(outer_ds)[:, None] * rs  # `√d r`
        norm_hrs = np.linalg.norm(hrs, axis=0)
        us = hrs / np.where(norm_hrs > 0.0, norm_hrs, 1.0)  # `u = normalize(√d r)`

        # Per-asset variances under their own (faster) decay refine the
        # variance component: `Σ* = (1 - δ) ρ · (σ σᵀ) + δ diag(σ²)`.
        dws = var_ds @ ws
        dr2s = var_ds @ np.square(rs)
        sigma2 = dr2s / np.where(dws > 0.0, dws, 1.0)
        sigma = np.sqrt(sigma2)

        # Build the shrinkage target and compute the shrinkage intensity.
        self.target.build(outer_ds, ws, us, sigma)
        sigma2_target = self.target.diag()

        if self.intensity is not None:
            delta = self.intensity
        else:
            delta = self.target.intensity(outer_ds, ws, us, sigma)

        # Truncate the shrunk covariance matrix to the specified rank.
        k = min(self.rank, m)

        # Dispatch on the size and rank of the problem: if the requested rank
        # is large enough, use dense eigendecomposition; otherwise, use sparse
        # eigendecomposition.
        if 4 * k >= m:
            try:
                cov = (us.T @ us) * np.outer(sigma, sigma)
                target = self.target.dense()
                lam, vec = np.linalg.eigh((1.0 - delta) * cov + delta * target)
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
                    cov = sigma * (us.T @ (us @ (sigma * z)))
                    target = self.target.matvec(z)
                    return (1.0 - delta) * cov + delta * target

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
        # specific variance, preserving the diagonal of the shrunk matrix.
        sigma2_shrunk = (1.0 - delta) * sigma2 + delta * sigma2_target
        sigma2_kept = np.square(vec[:, :kept]) @ lam[:kept]
        self.specific[mask] = np.maximum(sigma2_shrunk - sigma2_kept, 0.0)

    def predict(
        self, mask: np.ndarray, b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts the components of the covariance matrix."""

        assert mask.shape == (self.n,) and mask.dtype == np.bool_

        return self.exposures, self.covariance, self.specific
