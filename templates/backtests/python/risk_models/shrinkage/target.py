from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


class Target(ABC):
    """An implicit shrinkage target for covariance (risk) prediction."""

    @abstractmethod
    def intensity(
        self, ds: np.ndarray, ws: np.ndarray, us: np.ndarray, sigma: np.ndarray
    ) -> float:
        """Returns an optimized shrinkage intensity `δ` for the given sample
        correlation matrix `ρ := Σ_t u_t u_tᵀ = Uᵀ U` and per-asset
        standard deviations `σ`.
        """
        pass

    @abstractmethod
    def build(
        self, ds: np.ndarray, ws: np.ndarray, us: np.ndarray, sigma: np.ndarray
    ) -> None:
        """Builds an internal representation for the shrinkage target matrix,
        given sample correlation matrix `ρ := Σ_t u_t u_tᵀ = Uᵀ U`
        and per-asset standard deviations `σ`.
        """
        pass

    @abstractmethod
    def dense(self) -> np.ndarray:
        """Returns the shrinkage target matrix `T` as a dense array."""
        pass

    @abstractmethod
    def diag(self) -> np.ndarray:
        """Returns the diagonal of the shrinkage target matrix `T`."""
        pass

    @abstractmethod
    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Returns `T x` where `T` is the shrinkage target matrix."""
        pass


def _correlation_noise(ds: np.ndarray, us: np.ndarray) -> tuple[float, float]:
    """Returns `Σ_{i≠j} Var[ρ_ij]` and `Σ_{i≠j} ρ²_ij` - the total estimation
    variance and total squared magnitude of the off-diagonal sample
    correlations. These are the two shared ingredients of every target's
    shrinkage intensity.

    All summations below are over the last `t` cross-sections, with the
    `t`-axis subscripts omitted.

    For the estimation of the estimation variances, we assume that `r rᵀ` is
    i.i.d. across `t`, and use weighted sample means to estimate its mean and
    variance:

    - `E[r_i r_j] ≈ Σ d (r_i r_j) / Σ d`
    - `Var[r_i r_j] ≈ Σ d² (r_i r_j - E[r_i r_j])² / Σ d²`

    Using these, we derive an estimator for `Var[ρ_ij]`:

    ```
    Var[ρ_ij]
    = Var[Σ u_i u_j]
    = Var[Σ d r_i r_j / √(Σ d r²_i) √(Σ d r²_j)]
    ≈ Var[Σ d r_i r_j] / (Σ d r²_i) (Σ d r²_j)    (delta method omitting cross-terms!)
    = (Σ d²) Var[r_i r_j] / (Σ d r²_i) (Σ d r²_j)    (i.i.d. assumption)
    ≈ Σ d² (r_i r_j - E[r_i r_j])² / (Σ d r²_i) (Σ d r²_j)    (variance estimator)
    ≈ Σ d² (r_i r_j - Σ d (r_i r_j) / Σ d)² / (Σ d r²_i) (Σ d r²_j)    (mean estimator)
    = Σ (u_i u_j - ρ_ij · d / Σ d)²    (rewrites)
    ```

    In matrix notation, the matrix of variances is
    `Var[ρ] ≈ Σ (u uᵀ - ρ · d / Σ d)² = Σ (A - 2 B + C)`, where:

    - `A := Σ (u uᵀ)² = Σ u² u²ᵀ`
    - `B := Σ (u uᵀ) · ρ · d / Σ d = ρ · Σ (u uᵀ) · d / Σ d`
    - `C := Σ (ρ · d / Σ d)² = ρ² · Σ (d / Σ d)²`

    the total estimation variance is then obtained by left- and right-
    multiplying `Var[ρ]` by the all-ones vector, then subtracting the trace;
    the total squared magnitude similarly from `ρ²`.
    """

    t, m = us.shape
    u2s = np.square(us)
    rho_diag = u2s.sum(axis=0)
    ds_unit = ds / ds.sum()  # `d / Σ d`
    ds_unit_ss = np.square(ds_unit).sum()  # `Σ (d / Σ d)²`
    a = np.square(u2s.sum(axis=1)).sum()  # sum over `A`
    a_diag = np.square(u2s).sum()  # diagonal sum over `A`
    b_diag = np.dot(ds_unit @ u2s, rho_diag)  # diagonal sum over `B`
    d_diag = np.square(rho_diag).sum()  # diagonal sum over `ρ²`
    c_diag = d_diag * ds_unit_ss  # diagonal sum over `C`

    # Summation over `B` and `ρ²` both require summation of the shape
    # `Σ_stij u_si u_sj u_ti u_tj`, which can be done in either
    # `O(m² t)` or `O(t² m)` depending on which axis is smaller.
    if m <= t:
        rho = us.T @ us  # `O(m² t)`
        b = (rho * (us.T @ (us * ds_unit[:, None]))).sum()  # sum over `B`
        d = np.square(rho).sum()  # sum over `ρ²`
        c = d * ds_unit_ss  # sum over `C`
    else:
        g2 = np.square(us @ us.T)  # `O(t² m)`
        b = (g2 @ ds_unit).sum()  # sum over `B`
        d = g2.sum()  # sum over `ρ²`
        c = d * ds_unit_ss  # sum over `C`

    noise = (a - a_diag) - 2.0 * (b - b_diag) + (c - c_diag)
    return float(noise), float(d - d_diag)


def _clip_intensity(noise: float, distance: float) -> float:
    """Returns the clamped shrinkage intensity `noise / distance`."""

    return float(np.clip(noise / (distance if distance > 0.0 else 1.0), 0.0, 1.0))


@dataclass(slots=True)
class TargetDiagonal(Target):
    """A shrinkage target that is the diagonal of the sample covariance matrix
    `S`, preserving per-asset variances while shrinking correlations to 0.
    """

    sigma2: np.ndarray = field(default_factory=lambda: np.array([]))

    def intensity(
        self, ds: np.ndarray, ws: np.ndarray, us: np.ndarray, sigma: np.ndarray
    ) -> float:
        # The target has zero off-diagonal correlations: the distance to it
        # is the magnitude of the off-diagonal sample correlations itself.
        noise, rho2 = _correlation_noise(ds, us)
        return _clip_intensity(noise, rho2)

    def build(
        self, ds: np.ndarray, ws: np.ndarray, us: np.ndarray, sigma: np.ndarray
    ) -> None:
        self.sigma2 = np.square(sigma)

    def dense(self) -> np.ndarray:
        return np.diag(self.sigma2)

    def diag(self) -> np.ndarray:
        return self.sigma2

    def matvec(self, x: np.ndarray) -> np.ndarray:
        return self.sigma2 * x


@dataclass(slots=True)
class TargetConstantCorrelation(Target):
    """A shrinkage target with all pairwise correlations equal to their
    cross-sectional mean `r̄` (Ledoit-Wolf 2004): `T := r̄ σ σᵀ` off the
    diagonal, `σ²` on the diagonal, preserving per-asset variances while
    shrinking correlations towards their common mean. Always positive
    semi-definite: `𝟙ᵀρ𝟙 ≥ 0` forces `r̄ ≥ -1/(m - 1)`, which is exactly
    the PSD condition for a constant-correlation matrix.

    The estimation noise of `r̄` itself is ignored in the intensity: it is
    an average over `m (m - 1)` entries, so its `Cov[ρ_ij, r̄]` corrections
    are of relative order `O(1/m)`.
    """

    rbar: float = 0.0
    sigma: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma2: np.ndarray = field(default_factory=lambda: np.array([]))

    def intensity(
        self, ds: np.ndarray, ws: np.ndarray, us: np.ndarray, sigma: np.ndarray
    ) -> float:
        # Distance to the mean correlation, by the variance-around-the-mean
        # identity: `Σ_{i≠j} (ρ - r̄)² = Σ_{i≠j} ρ² - r̄² m (m - 1)`.
        t, m = us.shape
        noise, rho2 = _correlation_noise(ds, us)
        return _clip_intensity(noise, rho2 - np.square(self.rbar) * m * (m - 1))

    def build(
        self, ds: np.ndarray, ws: np.ndarray, us: np.ndarray, sigma: np.ndarray
    ) -> None:
        # Mean off-diagonal correlation: `Σ_{i≠j} ρ_ij = ‖U 𝟙‖² - tr ρ`.
        t, m = us.shape
        rho_diag = np.square(us).sum(axis=0)
        off = np.square(us.sum(axis=1)).sum() - rho_diag.sum()
        self.rbar = float(off / (m * (m - 1))) if m > 1 else 0.0
        self.sigma = sigma
        self.sigma2 = np.square(sigma)

    def dense(self) -> np.ndarray:
        target = self.rbar * np.outer(self.sigma, self.sigma)
        np.fill_diagonal(target, self.sigma2)
        return target

    def diag(self) -> np.ndarray:
        return self.sigma2

    def matvec(self, x: np.ndarray) -> np.ndarray:
        rank_one = self.rbar * self.sigma * np.dot(self.sigma, x)
        return rank_one + (1.0 - self.rbar) * self.sigma2 * x


@dataclass(slots=True)
class TargetCommonCovariance(TargetConstantCorrelation):
    """A shrinkage target with all entries equal to their cross-sectional
    means (Schäfer-Strimmer target C): `T := c̄` off the diagonal and `v̄` on
    the diagonal, the mean sample covariance and mean sample variance. Unlike
    the other targets, this also shrinks the per-asset variances towards
    their mean. Positive semi-definite: `𝟙ᵀS𝟙 ≥ 0` forces
    `c̄ ≥ -v̄/(m - 1)`, and `c̄ ≤ v̄` by AM-GM.

    The intensity is inherited from the constant-correlation target: it is
    estimated in correlation space over the off-diagonal entries only.
    Compared to Schäfer & Strimmer's joint intensity this omits the `m`
    diagonal terms against the `m (m - 1)` off-diagonal ones - an `O(1/m)`
    approximation that avoids variance-of-variance machinery - and measures
    the distance up to the dispersion of variances.
    """

    vbar: float = 0.0
    cbar: float = 0.0

    def build(
        self, ds: np.ndarray, ws: np.ndarray, us: np.ndarray, sigma: np.ndarray
    ) -> None:
        super().build(ds, ws, us, sigma)
        # Mean off-diagonal covariance: `Σ_{i≠j} S_ij = σᵀρσ - Σ ρ_ii σ²`
        # with `σᵀρσ = ‖U σ‖²`; mean variance is the mean of `σ²`.
        t, m = us.shape
        rho_diag = np.square(us).sum(axis=0)
        off = np.square(us @ sigma).sum() - np.dot(rho_diag, self.sigma2)
        self.cbar = float(off / (m * (m - 1))) if m > 1 else 0.0
        self.vbar = float(self.sigma2.mean())

    def dense(self) -> np.ndarray:
        target = np.full((self.sigma2.size, self.sigma2.size), self.cbar)
        np.fill_diagonal(target, self.vbar)
        return target

    def diag(self) -> np.ndarray:
        return np.full_like(self.sigma2, self.vbar)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        return self.cbar * x.sum() + (self.vbar - self.cbar) * x


@dataclass(slots=True)
class TargetSingleIndex(Target):
    """A shrinkage target implied by a one-factor market model
    (Ledoit-Wolf 2003): `T := (q σ) (q σ)ᵀ + diag((1 - q²) σ²)`, where `q_i`
    is the sample correlation of asset `i` against the "market" - here the
    equal-weighted portfolio of standardized returns, whose loadings are the
    normalized row sums of `ρ`: `q := Uᵀ (U 𝟙) / ‖U 𝟙‖ = ρ 𝟙 / √(𝟙ᵀρ𝟙)`.
    Off the diagonal `T_ij = q_i q_j σ_i σ_j`, and the diagonal is `σ²`:
    per-asset variances are preserved while correlations shrink towards the
    market structure `ρ_ij → q_i q_j`. `|q_i| ≤ 1` by Cauchy-Schwarz, so `T`
    is positive semi-definite.

    Ledoit-Wolf regress on an external market index; the standardized
    equal-weighted portfolio is the self-contained, scale-invariant analogue
    computable from the window alone. As with the other estimated targets,
    the `Cov[ρ_ij, q_i q_j]` corrections to the intensity are ignored.
    """

    q: np.ndarray = field(default_factory=lambda: np.array([]))
    qsigma: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma2: np.ndarray = field(default_factory=lambda: np.array([]))

    def intensity(
        self, ds: np.ndarray, ws: np.ndarray, us: np.ndarray, sigma: np.ndarray
    ) -> float:
        # Distance to the rank-one correlations:
        # `Σ_{i≠j} (ρ - q qᵀ)² = Σ_{i≠j} ρ²
        #     - 2 (qᵀρq - Σ q² ρ_ii) + ((Σ q²)² - Σ q⁴)`
        # with `qᵀρq = ‖U q‖²`.
        noise, rho2 = _correlation_noise(ds, us)
        rho_diag = np.square(us).sum(axis=0)
        q2 = np.square(self.q)
        cross = np.square(us @ self.q).sum() - np.dot(q2, rho_diag)
        target_sq = np.square(q2.sum()) - np.square(q2).sum()
        return _clip_intensity(noise, rho2 - 2.0 * cross + target_sq)

    def build(
        self, ds: np.ndarray, ws: np.ndarray, us: np.ndarray, sigma: np.ndarray
    ) -> None:
        ones = us.sum(axis=1)  # `U 𝟙`
        norm_ones = np.linalg.norm(ones)
        self.q = us.T @ (ones / (norm_ones if norm_ones > 0.0 else 1.0))
        self.qsigma = self.q * sigma
        self.sigma2 = np.square(sigma)

    def dense(self) -> np.ndarray:
        target = np.outer(self.qsigma, self.qsigma)
        np.fill_diagonal(target, self.sigma2)
        return target

    def diag(self) -> np.ndarray:
        return self.sigma2

    def matvec(self, x: np.ndarray) -> np.ndarray:
        rank_one = self.qsigma * np.dot(self.qsigma, x)
        return rank_one + (self.sigma2 - np.square(self.qsigma)) * x


SHRINKAGE_TARGETS: dict[str, type[Target]] = {
    "diagonal": TargetDiagonal,
    "constant-correlation": TargetConstantCorrelation,
    "common-covariance": TargetCommonCovariance,
    "single-index": TargetSingleIndex,
}
