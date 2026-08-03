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

```console
$ .venv/Scripts/python data/a_shares_crawler/export_parquet.py   # defaults to <script dir> -> <script dir>/panels/
```

## `strategy_markowitz/`

The `strategy_markowitz_panel` example from the `tradingflow` crate, adapted
onto the exported Parquet panels: the symbol axis spans the whole market
(~5900 symbols, 1991–today), the tradable subset is selected through the
`universe` vector (`--symbols`/`--industry`), and the covariance predictor and
optimizer rebalance every `--rebalance-every` (default 21) trading days since
a market-wide covariance cross-section is too large to emit daily.

Build and run (the `python` feature embeds the interpreter; point PyO3 at the
repository venv when building):

```console
$ cd templates/strategy_markowitz
$ cargo build --release
$ OPENBLAS_NUM_THREADS=1 \
  ./target/release/strategy_markowitz --start 2015-01-01
$ ./target/release/strategy_markowitz --help   # all options
```

- On Windows, `python3XY.dll` must be on `PATH`.
- `OPENBLAS_NUM_THREADS=1` disables OpenBLAS's internal parallelism, which is
  not thread-safe under the engine's worker threads.

The strategy's Python operators — the incremental Ridge mean predictor
(`python/ridge_incr.py`), the shrinkage covariance predictor
(`python/shrinkage.py`) and the Markowitz optimizer (`python/markowitz.py`) —
are self-contained single files assembled from the corresponding `tradingflow`
Python modules, harnesses included: `main` appends the `python/` directory to
the embedded interpreter's `sys.path`, so `py_operator_module("<module>",
params)` resolves them like installed packages, and the example depends on no
built-in TradingFlow Python operators. The whole modeling pipeline is
therefore local source code — fits, shrinkage targets, constraints and
windowing are all natural points for experimentation, picked up on the next
run with no install step. Each file's docstring documents its operator
contract and layout.
