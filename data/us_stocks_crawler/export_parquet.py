#!/usr/bin/env python3
"""Create long-format Parquet panels from `us-stocks-crawler` CSV data.

Reads the per-symbol CSV histories written by the `us-stocks-crawler` package
(`<data-dir>/us_stocks_history/<symbol>.<kind>.csv`) and writes one long-format
Parquet table per kind, in the layout the `tradingflow` panel source expects:
one timestamp column (sorted ascending), a dictionary-encoded `symbol` index
column, and value columns. The output matches the `a_shares_crawler` export
next door, so strategies can consume either market with the same wiring.

Two families of kinds are handled differently:

* **Event kinds** (`daily_prices`, `dividends`): the CSV's `date` is the
  market event time (trading day, ex-dividend day), so rows are only globally
  sorted by `(date, symbol)` and written as-is.

* **Report kinds** (`equity_structures`, `balance_sheets`,
  `income_statements`, `cash_flow_statements`): the CSV's `date` is the
  report's *period end*, which is not when the market learns of it. For
  look-ahead-safe backtesting each row is re-keyed to its **effective date**:

      effective = max(report_date, notice_date)

  Unlike the A-shares crawler, `us-stocks-crawler` keys every period to the
  *earliest* SEC filing that disclosed it, so `notice_date` usually is the
  first publication and can be believed. The exception is a period whose
  first appearance in XBRL is as a *comparative* in a later filing -- periods
  from before the company's XBRL history begins (the SEC mandate phased in
  around 2009, and newly listed companies republish pre-IPO years). Such a
  notice says nothing about when the market first saw the figures, so a
  notice later than the period's regulatory filing deadline is discarded and
  the period instead becomes visible at `report_date + lag`. As in the
  A-shares export, each lag is the deadline of the slowest filer class the
  SEC allows, so genuinely delinquent filings are the only rows placed
  slightly before their real publication:

  - `--quarterly-lag-days` (default 50) covers periods filed with a 10-Q:
    due 40 days after the period end for accelerated and large accelerated
    filers and 45 days for non-accelerated ones (Exchange Act Rule 13a-13),
    plus the 5-day Rule 12b-25 extension within which a late filing still
    counts as timely. It applies to every duration under about ten months,
    year-to-date interims included, since those ship inside the same 10-Q
    as their quarter.
  - `--annual-lag-days` (default 135) covers annual periods: a 10-K is due
    60/75/90 days by filer class plus a 15-day Rule 12b-25 extension, and a
    foreign private issuer's 20-F is due four months plus the same 15 days
    -- the outer envelope. It is also used for `equity_structures` and
    `balance_sheets`, whose rows do not record a period start and so cannot
    be told apart by duration.

  Each lag is both the plausibility cut and the fallback, and the annual
  value doubles as an upper bound the cut must respect anyway: a domestic
  comparative resurfaces no sooner than the next quarter's 10-Q, roughly
  130 days after the restated period's end, so a materially larger cut
  would start believing restatements. Two consequences are accepted and
  worth knowing. Domestic pre-XBRL annuals fall back to the foreign
  deadline, firing up to ~45 days after the market actually saw them
  (conservative). And before fiscal 2012 the 20-F deadline was six months,
  so the few foreign annuals from that era fire at the fallback slightly
  before their real filing.

  Rows are then sorted by effective date per symbol, and **superseded
  reports are dropped**: walking a symbol's rows in effective order, a row
  is kept only if its period end strictly advances the running maximum
  (a period overtaken before it was noticed would walk the panel's carried
  point-in-time state backwards) *and* its `(report_year,
  report_day_of_year)` key does not regress (`annualize` asserts exactly
  this monotonicity, so a violating row would abort the run downstream; a
  quarter whose year-to-date period was never tagged in XBRL can produce
  one).

  The emitted `date` column *is* the effective date (the panel source fires
  on it); the period end is kept as `report_date`, along with `report_year` /
  `report_day_of_year` (1-based) value columns for YTD annualization. US
  fiscal years are not calendar years (Apple's ends in late September), so
  for the duration kinds these come from the recorded period start:
  `report_year` is the year the fiscal period started and
  `report_day_of_year` counts days since that start -- exactly the
  reset-and-difference keys `annualize` expects. For `equity_structures` and
  `balance_sheets`, which are instants and never annualized, the calendar
  year and day of the period end are used. The rare transition period filed
  when a company moves its fiscal year end may annualize incorrectly, since
  two fiscal periods can then start in one calendar year.

All timestamp-like columns are written as Parquet `date32`; the panel source
converts them on read. Value columns keep their CSV dtypes (`float64`, with
`error` as `bool` and share counts as `int64`), and `symbol` is
dictionary-encoded.

The symbol list (`<data-dir>/symbol_list.csv`) is also exported, as
`symbol_list.parquet` with three columns: `symbol` (string), `name` (string)
and `industry` (dictionary-encoded string). Strategies build their symbol
axis (`data::Schema`) from this file, and the panel source panics on any
label missing from its schema, so every symbol appearing in a panel must be
in the list; per-symbol histories of unlisted symbols are skipped with a
warning to preserve this invariant.

Usage (defaults resolve against this script's own directory):

    python data/us_stocks_crawler/export_parquet.py \
        [--crawler-data path/to/us-stocks-crawler/data] \
        [--out path/to/output/dir]
"""

import argparse
import sys
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

EVENT_KINDS = ["daily_prices", "dividends"]
REPORT_KINDS = [
    "equity_structures",
    "balance_sheets",
    "income_statements",
    "cash_flow_statements",
]
"""Report kinds whose rows record the period start alongside the period end."""
DURATION_KINDS = ["income_statements", "cash_flow_statements"]

DATE_COLUMNS = ["date", "start_date", "notice_date", "report_date"]
"""A duration longer than this many days is taken to be an annual period."""
ANNUAL_DAYS = 300
ROW_GROUP_SIZE = 1 << 18


def export_symbol_list(crawler_data: Path, out_dir: Path) -> set[str]:
    """Writes `symbol_list.parquet` and returns the set of listed symbols.

    Keeps only the columns a strategy needs to build its symbol axis and
    group by sector: `symbol` and `name` as plain strings, `industry`
    dictionary-encoded (it has few distinct values).
    """
    df = pd.read_csv(crawler_data / "symbol_list.csv", dtype=str)
    df = df.sort_values("symbol", ignore_index=True)
    if df["symbol"].isna().any() or df["symbol"].duplicated().any():
        raise ValueError("symbol_list.csv: null or duplicate symbol")
    table = pa.table(
        {
            "symbol": pa.array(df["symbol"], pa.string()),
            "name": pa.array(df["symbols.name"], pa.string()),
            "industry": pc.cast(
                pa.array(df["symbols.industry"], pa.string()),
                pa.dictionary(pa.int32(), pa.string()),
            ),
        }
    )
    out_path = out_dir / "symbol_list.parquet"
    pq.write_table(table, out_path, compression="zstd")
    print(f"symbol_list: {len(table)} symbols -> {out_path}")
    return set(df["symbol"])


def iter_symbol_csvs(history_dir: Path, kind: str):
    """`(symbol, csv_path)` for every per-symbol history of `kind`, sorted."""
    suffix = f".{kind}.csv"
    for path in tqdm(sorted(history_dir.glob(f"*{suffix}"))):
        yield path.name.removesuffix(suffix), path


def read_history(path: Path, has_notice: bool, has_duration: bool) -> pd.DataFrame:
    dates = ["date"]
    if has_duration:
        dates.append("start_date")
    if has_notice:
        dates.append("notice_date")
    df = pd.read_csv(path, parse_dates=dates)
    if df["date"].isna().any():
        raise ValueError(f"{path}: null in date column")
    if has_duration and df["start_date"].isna().any():
        raise ValueError(f"{path}: null in start_date column")
    return df


def prepare_report(
    df: pd.DataFrame,
    has_duration: bool,
    quarterly_lag_days: int,
    annual_lag_days: int,
) -> pd.DataFrame:
    """Re-key one symbol's reports to effective dates, dropping superseded ones."""
    report_date = df["date"]
    notice_date = df["notice_date"]

    # One disclosure lag per period kind, serving as both the plausibility cut
    # on the notice and the fallback when it fails. The crawler keys each
    # period to the earliest filing disclosing it, so a notice past the
    # deadline means the period's first XBRL appearance was as a comparative
    # in a later filing — it says nothing about when the market first saw the
    # figures, exactly as uninformative as a missing one, and treated the same.
    if has_duration:
        start_date = df["start_date"]
        duration_days = (report_date - start_date).dt.days + 1
        is_annual = duration_days > ANNUAL_DAYS
        # US fiscal years are not calendar years, so YTD accumulation resets
        # at the recorded period start, not on January the 1st.
        report_year = start_date.dt.year
        report_day = duration_days
    else:
        # Instants record no period start (and are never annualized); assume
        # the slower annual deadline, and key years to the calendar.
        is_annual = pd.Series(True, index=df.index)
        report_year = report_date.dt.year
        report_day = report_date.dt.dayofyear

    lag = pd.to_timedelta(np.where(is_annual, annual_lag_days, quarterly_lag_days), unit="D")
    deadline = report_date + lag
    usable = notice_date.notna() & (notice_date <= deadline)
    effective = np.where(usable, np.maximum(report_date, notice_date), deadline)

    # Stable (effective, report_date) order, so simultaneous publications
    # keep ascending periods and the high-water walk below is well-defined.
    order = np.lexsort((report_date.to_numpy(), effective))
    df = df.iloc[order].reset_index(drop=True)
    report_date = df["date"]
    effective_date = pd.Series(effective[order])
    report_year = report_year.iloc[order].reset_index(drop=True)
    report_day = report_day.iloc[order].reset_index(drop=True)

    # A report fires only if its period strictly advances the newest period
    # already published — anything else was overtaken by a later period
    # before it was noticed, and firing it would regress the carried panel —
    # and if its (year, day) annualization key does not walk backwards, which
    # `annualize` asserts. The keys normally advance with the period, but a
    # quarter tagged without its year-to-date period restarts the day count
    # mid-year and must be dropped.
    high_water = report_date.cummax().shift(1)
    keep = high_water.isna() | (report_date > high_water)
    year_day = report_year * 1000 + report_day
    year_day_high_water = year_day.cummax().shift(1)
    keep &= year_day_high_water.isna() | (year_day >= year_day_high_water)

    out = pd.DataFrame()
    out["date"] = effective_date
    out["report_date"] = report_date
    out["notice_date"] = df["notice_date"]
    out["report_year"] = report_year.astype(np.int32)
    out["report_day_of_year"] = report_day.astype(np.int32)
    passthrough = ("date", "notice_date", "start_date")
    out = pd.concat([out] + [df[c] for c in df.columns if c not in passthrough], axis=1)
    if "start_date" in df.columns:
        out.insert(2, "start_date", df["start_date"])
    return out[keep.to_numpy()]


def export_kind(
    history_dir: Path,
    out_dir: Path,
    kind: str,
    known_symbols: set[str],
    quarterly_lag_days: int,
    annual_lag_days: int,
) -> None:
    is_report = kind in REPORT_KINDS
    has_duration = kind in DURATION_KINDS
    has_notice = kind != "daily_prices"

    frames = []
    symbols = []
    unlisted = []
    num_input_rows = 0
    for symbol, path in iter_symbol_csvs(history_dir, kind):
        # The panel source panics on labels missing from the strategy's
        # schema, which is built from the symbol list — skip such histories.
        if symbol not in known_symbols:
            unlisted.append(symbol)
            continue
        df = read_history(path, has_notice, has_duration)
        if df.empty:
            continue
        num_input_rows += len(df)
        if is_report:
            df = prepare_report(df, has_duration, quarterly_lag_days, annual_lag_days)
        if not df.empty:
            frames.append(df)
            symbols.extend([symbol] * len(df))

    if unlisted:
        print(
            f"{kind}: skipped {len(unlisted)} symbols not in the symbol list: "
            + ", ".join(unlisted[:10])
            + ("..." if len(unlisted) > 10 else ""),
            file=sys.stderr,
        )
    if not frames:
        print(f"{kind}: no rows, skipped", file=sys.stderr)
        return

    table = pd.concat(frames, ignore_index=True)
    table.insert(1, "symbol", pd.Categorical(symbols))

    # Global stable sort: the panel source streams the table in timestamp
    # order. Within a timestamp, symbols ascend; within a symbol, report
    # periods ascend (so the latest period wins the cross-section cell).
    keys = [table["symbol"].cat.codes.to_numpy(), table["date"].to_numpy()]
    if is_report:
        keys.insert(0, table["report_date"].to_numpy())
    table = table.iloc[np.lexsort(tuple(keys))].reset_index(drop=True)

    arrow = pa.Table.from_pandas(table, preserve_index=False)
    for name in DATE_COLUMNS:
        if name in table.columns:
            i = arrow.schema.get_field_index(name)
            column = pc.cast(arrow.column(i), pa.date32())
            arrow = arrow.set_column(i, pa.field(name, pa.date32()), column)

    # Pandas categoricals convert to `dictionary<int16, large_string>`; pin
    # the canonical `dictionary<int32, string>` instead — `large_string`
    # labels are not accepted by the panel source's index resolution.
    symbol_type = pa.dictionary(pa.int32(), pa.string())
    i = arrow.schema.get_field_index("symbol")
    column = pc.cast(arrow.column(i), symbol_type)
    arrow = arrow.set_column(i, pa.field("symbol", symbol_type), column)

    out_path = out_dir / f"{kind}.parquet"
    pq.write_table(arrow, out_path, row_group_size=ROW_GROUP_SIZE, compression="zstd")
    dropped = num_input_rows - len(table)
    note = f" ({dropped} superseded rows dropped)" if is_report else ""
    print(f"{kind}: {len(table)} rows -> {out_path}{note}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export us-stocks-crawler CSV data as long-format Parquet panels.")
    parser.add_argument(
        "--crawler-data",
        type=Path,
        default=Path(__file__).resolve().parent,
        metavar="DIR",
        help="the crawler's data directory, containing us_stocks_history/ and "
        "symbol_list.csv (default: this script's directory)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "panels",
        metavar="DIR",
        help="output directory for the Parquet files (default: panels/ next to this script)",
    )
    parser.add_argument(
        "--quarterly-lag-days",
        type=int,
        default=50,
        metavar="DAYS",
        help="disclosure lag for a period filed with a 10-Q (45-day "
        "non-accelerated deadline + 5-day Rule 12b-25 extension): a notice "
        "later than this is discarded, and the period instead becomes "
        "visible this long after its end (default: 50)",
    )
    parser.add_argument(
        "--annual-lag-days",
        type=int,
        default=135,
        metavar="DAYS",
        help="the same, for annual periods (four-month 20-F deadline + "
        "15-day Rule 12b-25 extension, covering all 10-K classes too), and "
        "for the instant kinds whose rows cannot reveal their period "
        "(default: 135)",
    )
    parser.add_argument(
        "--kinds",
        nargs="*",
        default=EVENT_KINDS + REPORT_KINDS,
        choices=EVENT_KINDS + REPORT_KINDS,
        metavar="KIND",
        help="subset of kinds to export (default: all)",
    )
    args = parser.parse_args()

    history_dir = args.crawler_data / "us_stocks_history"
    if not history_dir.is_dir():
        parser.error(f"{history_dir} is not a directory")
    args.out.mkdir(parents=True, exist_ok=True)

    known_symbols = export_symbol_list(args.crawler_data, args.out)
    for kind in args.kinds:
        export_kind(
            history_dir,
            args.out,
            kind,
            known_symbols,
            args.quarterly_lag_days,
            args.annual_lag_days,
        )


if __name__ == "__main__":
    main()
