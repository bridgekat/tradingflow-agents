# Templates

Starting points for agent autoresearch experiments on the A-shares dataset.

The Parquet panels these templates read are produced by
`data/a_shares_crawler/export_parquet.py`, which merges the per-symbol CSV
histories in `data/a_shares_crawler/a_shares_history/` (written by
[`a-shares-crawler`](https://github.com/bridgekat/a-shares-crawler)) into one
long-format Parquet table per kind under `data/a_shares_crawler/panels/`, in
the layout the `tradingflow` `source::panel::parquet` source expects — plus
`symbol_list.parquet` (`symbol`, `name`, `industry`) from which strategies
build their symbol axis. Report kinds are re-keyed to look-ahead-safe
effective dates; see the module docstring for details.

```bash
python data/a_shares_crawler/export_parquet.py  # defaults to <script dir> -> <script dir>/panels/
```

## `strategy_markowitz/`

A Ridge + Markowitz strategy over the exported Parquet panels: the symbol
axis spans the whole market (~5900 symbols, 1991–today), the tradable subset
is the cap-weighted top `--universe-size` (default 3800; `0` = every symbol)
by circulating market cap, re-selected on a rebalance calendar of
`--rebalance-every` (default 30) calendar days from `--start`.
`--risk-aversion` takes a comma-separated list and runs one Markowitz
portfolio per value against shared predictors — the parameter-sweep pattern —
reported alongside the cap-weighted index.

Build and run (the `python` feature embeds the interpreter; point PyO3 at the
repository venv when building):

```bash
cd templates/strategy_markowitz
cargo run --release -- --help  # all options
```

- On Windows, `python3XY.dll` must be on `PATH`.
- Set environment variable `OPENBLAS_NUM_THREADS=1` to disable OpenBLAS's
  internal parallelism, which is not thread-safe.

The crate is split along its extension points:

| Module | Contents — and what to extend there |
| --- | --- |
| `src/data.rs` | Parquet panels → cross-sectional wires (signaled-or-NaN daily prices, carried fundamentals, annualized flows). New columns/tables join here. |
| `src/features/` | The factor catalogs and the model-ready `(N, F)` panel, selected by `--features`. `basic.rs` is hand-wired: **a new factor is one `add("NAME", handle)` line**, and its seed catalog demonstrates one factor per idiom (rolling price/return stats, size, balance-sheet and TTM-flow valuation, liquidity). `alpha101.rs`, `alpha158.rs` and `alpha360.rs` are ported catalogs: **a new factor there is one formula string**, lowered through a shared `expr::Context`. |
| `src/universe.rs` | Cap-weighted top-`k` universe (positive = active), doubling as the benchmark index weights. Alternative screens slot in as operators of the same shape. |
| `src/quotes.rs` | The synthetic quote book: `close ∓ 0.01`, `±inf` outside the ±10% price-limit band off the previous known close (or when suspended), delisting (`flag = false`, marked out of NAV) after >20 consecutive quoteless trading days. |
| `src/main.rs` | The experiment itself: wiring data → features → universe → quotes → predictors → the optimizer sweep → traders, and which parameters to sweep. |

The strategy's Python operators — the incremental Ridge mean predictor
(`python/mean.py`), the shrinkage variance predictor (`python/variance.py`)
and the Markowitz optimizer (`python/portfolio.py`) — are self-contained
single files assembled from the corresponding `tradingflow` Python modules,
harnesses included: `main` appends the `python/` directory to the embedded
interpreter's `sys.path`, so `py_operator_module("<module>", params)`
resolves them like installed packages, and the example depends on no
built-in TradingFlow Python operators. The whole modeling pipeline is
therefore local source code — fits, shrinkage targets, constraints and
windowing are all natural points for experimentation, picked up on the next
run with no install step. Each file's docstring documents its operator
contract and layout.

Performance notes: the risk crosses the Rust <-> Python boundary as a
`[factor_rank + 1, N]` **factor panel** (loadings + idiosyncratic std-dev)
rather than a dense `[N, N]` covariance — the boundary copies every array on
every generation, so a ~280 MB market-wide matrix per sweep variant is the
difference between hours and minutes. The shrinkage estimator itself is
`O(N·T)` (implicit matvecs + a column-subsampled Schäfer-Strimmer
intensity), the eigendecomposition happens once in the predictor rather than
once per variant, and the per-delta CVXPY solves release the GIL so a sweep
parallelizes across the worker pool. A 20-year backtest over the **whole
market** (`--universe-size 0`) with an 8-value risk-aversion sweep completes
in ~3.5 minutes; the default top-800 single-delta run in ~30 seconds.
