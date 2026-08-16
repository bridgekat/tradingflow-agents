# TradingFlow Agents

A multi-objective autoresearch loop over trading strategies built on
[TradingFlow](https://github.com/bridgekat/tradingflow): agents search for
strategies that earn well *and* are not volatile, treating return and
volatility as two axes rather than collapsing them into one number.

```bash
git submodule update --init --recursive
uv sync
```

Then, from a Claude Code session at the repository root:

```
Read autoresearch/orchestrator.md and run a round of the search.
```

## Layout

| Path | What it is |
| ---- | ---------- |
| `autoresearch/` | The search: orchestrator and explorer instructions, and the shared ledger |
| `templates/backtests/` | The strategy harness agents copy and modify — features, models, optimizers, and three backtest binaries |
| `validation/` | A scorer that recomputes results from submitted weights alone, loading no strategy code |
| `data/` | Crawlers and the Parquet market panels they export |
| `tradingflow/` | The backtesting framework, as a submodule |

## The backtests

`templates/backtests` builds three binaries, in increasing order of cost. The
cheap ones are screens; running them first is what makes the expensive one
worth its time.

| Binary | What it measures |
| ------ | ---------------- |
| `features_base` | Each feature's information coefficient — whether it predicts the next day's cross-section at all |
| `risk_model_compare` | Covariance forecast quality: log-likelihood, realized GMV volatility, bias ratio |
| `strategy_base` | The whole strategy: features → alpha model → risk model → mean-variance optimizer → simulated trading |

They embed a Python interpreter for the model operators, so run them with the
virtualenv active and `OPENBLAS_NUM_THREADS=1` set:

```bash
source .venv/Scripts/activate
export OPENBLAS_NUM_THREADS=1
cd templates/backtests && cargo run --release --bin strategy_base -- \
    --start 2015-01-01 --end 2023-01-01 --universe-size 300
```

Trading is fractional and costed: every fill pays
`max(|amount| × rate, --fee-min)`, defaulting to 0.05% on buys and 0.15% on
sells — the A-share stamp duty on top of the commission, matching Qlib's
exchange defaults. The cap-weighted index baseline is traded through the same
schedule, so both sides of the comparison are net of the same frictions.
Passing `--fee-rate-buy 0 --fee-rate-sell 0` recovers a frictionless run, which
is how to read a strategy's gross edge against what its turnover costs to
harvest.

## The objective

Return up, volatility down, neither traded against the other. A result is
**good** when at most 10% of everything else in the ledger dominates it, and an
agent is scored on the fraction of its own submissions that are good — so
selectivity pays and volume does not.

The threshold, rather than a strict Pareto rank, exists because the two axes
are not equally measurable. Over a few years the standard error on annualized
return is comparable to the spread between strategies, while volatility's is a
couple of percent relative. A lucky run can dominate a great deal of honest
work without being a better strategy; tolerating a few dominators keeps real
results on the board when a fluke lands above them.

See `autoresearch/README.md` for the search itself.

## Caveats

- There is no sandbox. Agents run with your user's permissions.
- The 2023-01-01 cutoff that keeps later data held out is a convention the
  instructions state, not something enforced.
- `validation/score` exists and works, but nothing currently requires a
  submission to pass through it.
