# Validation

Everything under `templates/` is agent-mutable: a submission may rewrite the
features, the models, the universe construction, even the rebalance calendar.
Everything here is not. The split is not about which code is generic — it is
about which code produces the **number**.

A strategy's entire output is a weights time series. A submission decides *what
to hold*; nothing it writes decides *how that is scored*.

```
[sandboxed run]                                       [trusted]
baseline + submission → run on val panels → weights → score → stats
```

## `score`

Reads submitted weights and simulates trading them.

It loads no submitted code. The crate does not build `tradingflow`'s `python`
feature, so an embedded interpreter is not available to be reached even by
accident, and it compiles no Rust from a submission. Its whole input surface is
a weights CSV, the market panels and its own command line — which is what makes
its output evidence rather than a claim.

```bash
cargo run --release -- \
  --data-dir ../../data/a_shares_crawler/panels \
  --weights books/candidate.csv \
  --start 2024-01-01 --end 2025-01-01 \
  --output runs/candidate.csv
```

Writes the NAV curves as CSV and a `<stem>_summary.json` beside them. See the
module docs in `score/src/main.rs` for the full contract; the essentials:

**The weights format.** One CSV per portfolio, `date,symbol,weight`, rows
ascending by date. The portfolio's label is the file stem. A book is a complete
cross-section — a symbol absent from a date is held at zero, not carried over
from the last date it appeared on — and the residual `1 - Σw` is cash.

**What the scorer fixes, and why.** The scoring window (so a book cannot select
the slice that flatters it); delayed execution, filling a book dated `D` at the
next trading day's quotes; the fee schedule; the `Σ|w|` leverage limit; and the
cap-weighted benchmark, traded through the same costs on the scorer's own
calendar so neither side of the comparison is chosen by the submission.

## Intended gate order

Mechanical checks first, so a reviewer's attention is spent only on what
survives them:

1. **Replicate** — apply the submission's patch to a pristine baseline, build,
   run on the val panels. Produces the weights to be scored.
2. **Point-in-time consistency** — re-run with data truncated at date *D* and
   compare weights on all overlapping dates. Any difference means the strategy
   used information from after the date it acted on. Not yet built; it is the
   strongest anti-lookahead tool available and does not depend on review.
3. **Plausibility** — a new feature with mean IC > ~0.15 or |t| > 20 is a bug,
   not an edge.
4. **Literal grep** — `\d{6}\.(SZ|SH)` and date constants in the diff.
5. **Adversarial review** — of *intent*, on what is left: is the feature the
   thing its rationale claims? The validator should read the code before seeing
   the score, and default to rejection under uncertainty.

Submissions should carry a falsifiable pre-registration rather than a narrative:
which regime the strategy should work in, what turnover it implies, where it
should fail. A rationale generated to fit results reliably fails to predict the
shape of its own.
