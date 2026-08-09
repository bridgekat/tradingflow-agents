"""Plots the CSV output of any example in this directory.

The script infers the shape of the file rather than knowing about any
particular example:

* the **time column** is the first column named like a date (`date`, `time`,
  `timestamp`, ...), or else the first column, parsed as datetimes when it
  parses and left as-is when it does not;
* every remaining **numeric** column is a curve to draw;
* every remaining **non-numeric** column is a key, so a *long-format* file
  (`date,symbol,close,...`) draws one curve per key value per numeric column.

So a wide file such as the MACD example's `date,nav` draws its columns
together on one axes, and a long file such as the indicator example's
`date,symbol,close,ma_fast,...` splits into one panel per symbol — the
default is to facet when there is a key column *and* more than one numeric
column, since curves for different symbols rarely share a scale. Override it
with `--facet` or `--no-facet` either way, and use `--normalize` to rebase
curves of different magnitudes onto one axes.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# Column names taken as the time axis, in preference order, case-insensitive.
TIME_NAMES = ("date", "datetime", "time", "timestamp", "instant")

# Curve colors, cycled per numeric column so a column keeps its color across
# panels.
PALETTE = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:cyan",
    "black",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", type=Path, help="CSV file to plot")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="image to write (default: show in interactive window)",
    )
    p.add_argument(
        "--time-column",
        default=None,
        help="name of the time column (default: inferred)",
    )
    p.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="numeric columns to draw (default: all)",
    )
    p.add_argument(
        "--keys",
        nargs="*",
        default=None,
        help="key values to keep, e.g. two symbols (default: all)",
    )
    p.add_argument(
        "--band",
        action="append",
        default=[],
        metavar="LOWER:UPPER",
        help="shade between two columns instead of drawing them as lines; repeatable",
    )
    p.add_argument(
        "--start", default=None, help="first timestamp to plot, e.g. 2018-01-01"
    )
    p.add_argument(
        "--end", default=None, help="last timestamp to plot, e.g. 2020-12-31"
    )
    facet = p.add_mutually_exclusive_group()
    facet.add_argument(
        "--facet",
        dest="facet",
        action="store_true",
        default=None,
        help="one panel per key value",
    )
    facet.add_argument(
        "--no-facet", dest="facet", action="store_false", help="all curves on one axes"
    )
    p.add_argument(
        "--normalize",
        action="store_true",
        help="rebase each curve to 1.0 at its first finite value",
    )
    p.add_argument(
        "--log-scale", action="store_true", help="plot the value axis on a log scale"
    )
    p.add_argument(
        "--title", default=None, help="figure title (default: the file name)"
    )
    p.add_argument("--width", type=float, default=12.0, help="figure width in inches")
    p.add_argument(
        "--height", type=float, default=3.4, help="per-panel height in inches"
    )
    return p.parse_args()


def load(path: Path) -> pd.DataFrame:
    """Reads the CSV, or exits with a hint if it is not there."""
    if not path.exists():
        sys.exit(f"{path} not found")
    frame = pd.read_csv(path)
    if frame.empty:
        sys.exit(f"{path} has no rows")
    return frame


def split_roles(
    frame: pd.DataFrame, time_column: str | None
) -> tuple[str, list[str], list[str]]:
    """Splits the columns into `(time, keys, values)`.

    The time column is parsed to datetimes in place when it parses; numeric
    columns become curves and the rest become keys.
    """
    if time_column is not None:
        if time_column not in frame.columns:
            sys.exit(
                f"no column named {time_column!r}; found: {', '.join(frame.columns)}"
            )
        time = time_column
    else:
        named = [c for c in frame.columns if c.strip().lower() in TIME_NAMES]
        time = named[0] if named else frame.columns[0]

    # A time column that does not parse as dates is left alone: an integer
    # tick index plots perfectly well as an x-axis.
    try:
        frame[time] = pd.to_datetime(frame[time])
    except (ValueError, TypeError):
        pass

    rest = [c for c in frame.columns if c != time]
    values = [c for c in rest if pd.api.types.is_numeric_dtype(frame[c])]
    keys = [c for c in rest if c not in values]
    if not values:
        sys.exit(f"no numeric columns to plot; found: {', '.join(rest) or '(none)'}")
    return time, keys, values


def restrict(
    frame: pd.DataFrame, time: str, start: str | None, end: str | None
) -> pd.DataFrame:
    """Restricts to a time range, in whatever type the time column holds."""
    if start is None and end is None:
        return frame
    cast = pd.Timestamp if pd.api.types.is_datetime64_any_dtype(frame[time]) else float
    if start is not None:
        frame = frame[frame[time] >= cast(start)]
    if end is not None:
        frame = frame[frame[time] <= cast(end)]
    if frame.empty:
        sys.exit("no rows left after restricting the time range")
    return frame


def parse_bands(specs: list[str], values: list[str]) -> list[tuple[str, str]]:
    """Parses `LOWER:UPPER` band specifications, checking both columns exist."""
    bands = []
    for spec in specs:
        lower, _, upper = spec.partition(":")
        if not upper:
            sys.exit(f"--band expects LOWER:UPPER, got {spec!r}")
        for column in (lower, upper):
            if column not in values:
                sys.exit(
                    f"--band names {column!r}, which is not a numeric column: {', '.join(values)}"
                )
        bands.append((lower, upper))
    return bands


def normalize(series: pd.Series) -> pd.Series:
    """Rebases a curve to 1.0 at its first finite, non-zero value."""
    finite = series[series.notna() & (series != 0.0)]
    return series if finite.empty else series / finite.iloc[0]


def draw(
    ax: Axes,
    frame: pd.DataFrame,
    time: str,
    values: list[str],
    bands: list[tuple[str, str]],
    label_prefix: str,
    color_of: dict[str, str],
    rebase: bool,
) -> None:
    """Draws one group's curves (and bands) onto `ax`."""
    frame = frame.sort_values(time)
    x = frame[time]
    banded = {c for pair in bands for c in pair}

    for lower, upper in bands:
        lo, hi = frame[lower], frame[upper]
        if rebase:
            lo, hi = normalize(lo), normalize(hi)
        ax.fill_between(
            x,
            lo,
            hi,
            color=color_of[upper],
            alpha=0.12,
            linewidth=0,
            label=f"{label_prefix}{lower}-{upper}",
        )
        ax.plot(x, lo, color=color_of[upper], linewidth=0.6, alpha=0.45)
        ax.plot(x, hi, color=color_of[upper], linewidth=0.6, alpha=0.45)

    for column in values:
        if column in banded:
            continue
        y = normalize(frame[column]) if rebase else frame[column]
        ax.plot(
            x, y, color=color_of[column], linewidth=0.6, label=f"{label_prefix}{column}"
        )


def main() -> None:
    args = parse_args()
    frame = load(args.input)
    time, keys, values = split_roles(frame, args.time_column)
    frame = restrict(frame, time, args.start, args.end)

    if args.columns is not None:
        unknown = [c for c in args.columns if c not in values]
        if unknown:
            sys.exit(f"not numeric columns in {args.input}: {', '.join(unknown)}")
        values = args.columns

    # Keys are filtered on the first key column, which is the one long-format
    # panel files use for the instrument.
    key = keys[0] if keys else None
    if args.keys is not None:
        if key is None:
            sys.exit(f"--keys given, but {args.input} has no key column")
        present = set(frame[key])
        unknown = [k for k in args.keys if k not in present]
        if unknown:
            sys.exit(f"key values not present in {key!r}: {', '.join(unknown)}")
        frame = frame[frame[key].isin(args.keys)]

    bands = parse_bands(args.band, values)
    color_of = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(values)}

    # Facet when there is something to separate and the curves would not share
    # a scale, unless the choice was made explicitly.
    facet = (
        args.facet if args.facet is not None else (key is not None and len(values) > 1)
    )
    if facet and key is None:
        sys.exit(f"--facet given, but {args.input} has no key column")
    groups = list(frame.groupby(key, sort=False)) if facet else [(None, frame)]

    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(args.width, args.height * len(groups)),
        sharex=True,
        squeeze=False,
    )
    for ax, (name, group) in zip(axes[:, 0], groups):
        if facet:
            draw(ax, group, time, values, bands, "", color_of, args.normalize)
            ax.set_title(str(name), fontsize=10, loc="left")
        elif key is not None:
            # Not faceting, but there are keys: every group draws the same
            # columns onto one axes, so the palette has to separate the groups
            # rather than the columns, or the curves come out indistinguishable.
            for i, (sub_name, sub) in enumerate(group.groupby(key, sort=False)):
                shifted = {
                    c: PALETTE[(i * len(values) + j) % len(PALETTE)]
                    for j, c in enumerate(values)
                }
                draw(
                    ax,
                    sub,
                    time,
                    values,
                    bands,
                    f"{sub_name} ",
                    shifted,
                    args.normalize,
                )
        else:
            draw(ax, group, time, values, bands, "", color_of, args.normalize)
        if args.log_scale:
            ax.set_yscale("log")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.margins(x=0.01)

    axes[-1, 0].set_xlabel(time)
    fig.tight_layout()

    # Reserve the header in inches rather than figure fractions, so the title
    # and the legend keep the same physical spacing however many panels the
    # file happens to produce.
    title_in, legend_in = 0.30, 0.34
    header = (title_in + legend_in) / fig.get_figheight()
    fig.subplots_adjust(top=1.0 - header)
    fig.suptitle(args.title or args.input.name, y=1.0, va="top", fontsize=11)

    # One shared legend when every panel draws the same curves, a per-panel
    # legend when the curves are per-key and would not transfer. A file with
    # very many curves (e.g. the Alpha101 example's 101 columns) gets no
    # legend at all — select fewer with --columns or --keys instead.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if len(labels) > 24:
        print(f"{len(labels)} curves: legend omitted; select fewer with --columns or --keys")
    elif facet or key is None:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0 - title_in / fig.get_figheight()),
            ncol=min(len(labels), 8),
            fontsize=8,
            frameon=False,
        )
    else:
        axes[0, 0].legend(fontsize=8, ncol=2, frameon=False)

    if args.output is None:
        plt.show()
    else:
        output = args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
