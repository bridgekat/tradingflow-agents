"""Cross-sectional neutralization of the alpha panel against the risk panel.

Each trading day the operator regresses every alpha column on the risk
exposure matrix `B` (the same panel the covariance predictor consumes: the
Barra styles plus the COUNTRY intercept) across the *tradable* cross-section,
and emits the residuals:

```
a_resid = a - B (BᵀB)⁻¹ Bᵀ a
```

The residual is the part of the feature the risk model cannot express, i.e.
the part a mean-variance optimizer can actually spend: the spanned part buys
factor exposure that the same covariance matrix immediately charges for.
Scoring `IC(a_resid, forward return)` is therefore the screen that asks
whether a feature's predictive power survives the risk model, rather than
whether it is large.

Notes on the details that matter:

* The regression runs over the rows in the universe with a complete risk row.
  Restricting to the universe is deliberate — the projection must be the one
  the optimizer performs, and its cross-section is the tradable one.
* Missing alpha entries are imputed at `0` inside the fit (the panel is
  cross-sectionally z-scored, so `0` is the cross-sectional mean, the
  least-informative value) and restored to `NaN` in the output, so a missing
  value never becomes a fabricated residual.
* Everything is contemporaneous: the projection at date `t` uses only date-`t`
  exposures. The lag against the forward return is applied downstream.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class NeutralizeState:
    out: np.ndarray


class Neutralize:
    type Inputs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    type Outputs = np.ndarray
    type Context = int
    type State = NeutralizeState

    def __init__(
        self, rcond: float = 1e-6, scope: str = "universe", project: bool = True
    ) -> None:
        assert scope in ("universe", "market"), f"neutralize: bad scope {scope!r}"
        self.rcond = rcond
        self.scope = scope
        # `project=False` applies the row mask and nothing else: the control
        # arm that isolates the projection from the change of cross-section
        # (with `scope="universe"`, masking alone already restricts whatever
        # consumes the panel to the tradable names).
        self.project = project

    def init(self, inputs: Inputs) -> State:
        _signal, alpha, risk, universe = inputs
        n, ka = alpha.shape
        assert risk.shape[0] == n
        assert universe.shape == (n,)
        return NeutralizeState(out=np.full((n, ka), np.nan))

    @staticmethod
    def reset(_: Inputs, state: State) -> Outputs:
        return state.out

    def compute(self, inputs: Inputs, state: State, _: Context) -> Outputs:
        signal, alpha, risk, universe = inputs
        n, ka = alpha.shape
        kr = risk.shape[1]

        if not signal:
            return state.out

        rows = np.isfinite(risk).all(axis=1)
        if self.scope == "universe":
            rows &= universe > 0.0
        out = np.full((n, ka), np.nan)
        if rows.sum() > kr + 10:
            a = alpha[rows]
            if not self.project:
                out[rows] = a
            else:
                b = risk[rows]
                miss = ~np.isfinite(a)
                filled = np.where(miss, 0.0, a)
                coef, *_ = np.linalg.lstsq(b, filled, rcond=self.rcond)
                resid = filled - b @ coef
                resid[miss] = np.nan
                out[rows] = resid
        state.out = out
        return state.out


def build(**kwargs) -> Neutralize:
    return Neutralize(**kwargs)
