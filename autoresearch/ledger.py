#!/usr/bin/env python3
"""The shared ledger: every attempt anyone has made, and how it scores.

A *point* is one backtest variant — one (return, volatility) observation. A
*submission* is a set of points produced together, usually a small parameter
sweep, plus the hypothesis behind them.

Scoring is by Pareto dominance in (return, volatility): higher return is
better, lower volatility is better, and neither is traded against the other.

    front fraction  = how many of the ledger's other points dominate this one,
                      as a fraction of them
    good            = front fraction at or below GOOD_THRESHOLD
    agent score     = that agent's good points / that agent's total points

The band near the front is deliberate rather than a strict Pareto rank. Return
over a few years is estimated badly — the standard error is comparable to the
spread between strategies — so a lucky run can land far up and to the left and
dominate a great many honest points without being a better strategy. Demanding
rank 1 would hand the front to whoever got the best draw. Tolerating a small
fraction of dominators keeps genuinely good work on the board when a fluke
lands above it.

Volatility, by contrast, is estimated precisely: its standard error is a few
percent relative over the same window. The two axes are not equally
trustworthy, and that asymmetry is worth remembering whenever a result looks
like an improvement in return alone.

Scoring an agent by its *fraction* of good points, rather than its count, is
what makes selectivity pay. Submitting a sweep you do not believe in dilutes
your own score whether or not the points land well.

Entries are one JSON file per submission under `ledger/`, so concurrent agents
never contend on a shared file.

Usage:

    ledger.py submit --agent NAME --nav PATH --hypothesis TEXT --prediction TEXT
    ledger.py note --agent NAME --finding TEXT --evidence TEXT
    ledger.py report [--agent NAME]
    ledger.py log [--agent NAME]
    ledger.py front
    ledger.py check --ret 0.09 --vol 0.18

Not every result is a point. An idea that was screened and rejected, a gain
that turned out to be in-sample, a mechanism that explains why a whole family
of strategies fails — these change what the next agent should try, and they
have no (return, volatility) to record. `note` writes them to the same ledger
so they survive the deletion of a run directory; they carry no points and so
never touch anyone's score. `log` replays both kinds in order, which is how
you find out what has already been tried before you try it again.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ENTRIES = ROOT / "ledger"

# A point is good when at most this fraction of the other points dominate it.
GOOD_THRESHOLD = 0.10

# Margin required before one point counts as beating another. Zero means plain
# Pareto dominance. Raise `EPS_RET` to demand that a return advantage clear
# some of its own estimation error before it counts — a few percentage points
# is defensible on a multi-year window, and it makes the front markedly less
# sensitive to lucky draws.
EPS_RET = 0.0
EPS_VOL = 0.0

# Volatility bands the explorers are assigned to, as (name, low, high).
BANDS = [("low", 0.0, 0.12), ("mid", 0.12, 0.18), ("high", 0.18, 10.0)]

TRADING_DAYS = 252.0


# ---------------------------------------------------------------------------
# Statistics, computed from the artifacts rather than from anyone's report
# ---------------------------------------------------------------------------


def nav_stats(nav: pd.Series) -> dict:
    """Return, volatility, Sharpe and max drawdown of one daily NAV curve.

    Matches what `templates/backtests` prints, so a submission's numbers agree
    with what the agent watched go by on its terminal.
    """
    s = pd.to_numeric(nav, errors="coerce").dropna()
    s = s[s > 0]
    if len(s) < 10:
        raise SystemExit(f"NAV column has only {len(s)} usable points")
    years = len(s) / TRADING_DAYS
    log_rets = np.log(s / s.shift(1)).dropna()
    sd = float(log_rets.std(ddof=0))
    peak = s.cummax()
    return {
        "ret": float((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1),
        "vol": sd * TRADING_DAYS**0.5,
        "sharpe": float(log_rets.mean()) / sd * TRADING_DAYS**0.5 if sd > 0 else float("nan"),
        "mdd": float((s / peak - 1).min()),
        "days": len(s),
        "start": str(s.index[0]),
        "end": str(s.index[-1]),
    }


def turnover(weights_csv: Path, years: float) -> float | None:
    """Annualized two-way turnover from a `date,symbol,weight` book file."""
    if not weights_csv.exists():
        return None
    w = pd.read_csv(weights_csv)
    if w.empty:
        return None
    book = w.pivot_table(index="date", columns="symbol", values="weight", aggfunc="sum")
    book = book.sort_index().fillna(0.0)
    # The opening book is a full purchase from cash; the rest are changes.
    total = float(book.iloc[0].abs().sum() + book.diff().iloc[1:].abs().sum().sum())
    return total / years if years > 0 else None


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


@dataclass
class Point:
    agent: str
    submission: str
    label: str
    ret: float
    vol: float
    sharpe: float
    mdd: float
    turnover: float | None
    hypothesis: str


def load_entries() -> list[dict]:
    """Every ledger entry, submissions and notes alike, oldest first."""
    entries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(ENTRIES.glob("*.json"))
    ]
    entries.sort(key=lambda e: e.get("at", ""))
    return entries


def load_points() -> list[Point]:
    points = []
    for path in sorted(ENTRIES.glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        for p in entry.get("points", []):
            points.append(
                Point(
                    agent=entry["agent"],
                    submission=entry["id"],
                    label=p["label"],
                    ret=p["ret"],
                    vol=p["vol"],
                    sharpe=p.get("sharpe", float("nan")),
                    mdd=p.get("mdd", float("nan")),
                    turnover=p.get("turnover"),
                    hypothesis=entry.get("hypothesis", ""),
                )
            )
    return points


def dominates(a: Point | tuple, b: Point | tuple) -> bool:
    """Whether `a` beats `b`: at least as good on both axes, and meaningfully
    better on one."""
    a_ret, a_vol = (a.ret, a.vol) if isinstance(a, Point) else a
    b_ret, b_vol = (b.ret, b.vol) if isinstance(b, Point) else b
    if a_ret < b_ret or a_vol > b_vol:
        return False
    return (a_ret - b_ret) > EPS_RET or (b_vol - a_vol) > EPS_VOL


def front_fraction(point, others: list[Point]) -> float:
    """The fraction of `others` that dominate `point`. Zero against an empty
    ledger: the first point on the board has nothing above it."""
    rivals = [o for o in others if o is not point]
    if not rivals:
        return 0.0
    return sum(dominates(o, point) for o in rivals) / len(rivals)


def band_of(vol: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= vol < hi:
            return name
    return "high"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_submit(args) -> None:
    nav_path = Path(args.nav)
    nav = pd.read_csv(nav_path, index_col=0)
    exclude = set(args.exclude or []) | {"index"}
    columns = [
        c for c in nav.columns
        if c not in exclude and not c.startswith("index_")
    ]
    if not columns:
        raise SystemExit(f"{nav_path}: no submittable columns (found {list(nav.columns)})")

    points = []
    for label in columns:
        stats = nav_stats(nav[label])
        years = stats["days"] / TRADING_DAYS
        weights = nav_path.with_name(f"{nav_path.stem}_weights_{label}.csv")
        points.append({"label": label, **stats, "turnover": turnover(weights, years)})

    entry = {
        "id": f"{args.agent}-{uuid.uuid4().hex[:8]}",
        "agent": args.agent,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hypothesis": args.hypothesis,
        "prediction": args.prediction,
        "code": args.code or "",
        "nav": str(nav_path),
        "points": points,
    }
    ENTRIES.mkdir(parents=True, exist_ok=True)
    out = ENTRIES / f"{entry['id']}.json"
    out.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")

    # Report how the new points land, so the agent sees the consequence.
    everything = load_points()
    print(f"recorded {len(points)} point(s) to {out.relative_to(ROOT.parent)}\n")
    print(f"{'variant':<28}{'return':>9}{'vol':>8}{'sharpe':>8}{'turn':>7}{'front':>8}  verdict")
    for p in points:
        me = (p["ret"], p["vol"])
        frac = front_fraction(me, everything)
        turn = f"{p['turnover']:.1f}x" if p["turnover"] is not None else "-"
        good = "GOOD" if frac <= GOOD_THRESHOLD else ""
        print(
            f"{p['label']:<28}{p['ret']*100:>+8.2f}%{p['vol']*100:>7.2f}%"
            f"{p['sharpe']:>8.3f}{turn:>7}{frac*100:>7.0f}%  {good}"
        )
    print()
    _print_agent_scores(everything, highlight=args.agent)


def cmd_note(args) -> None:
    """Record a finding that has no points: a rejected idea, a failed
    replication, a mechanism. Notes never affect scores."""
    entry = {
        "id": f"{args.agent}-note-{uuid.uuid4().hex[:8]}",
        "agent": args.agent,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "note",
        "finding": args.finding,
        "evidence": args.evidence,
        "code": args.code or "",
        "points": [],
    }
    ENTRIES.mkdir(parents=True, exist_ok=True)
    out = ENTRIES / f"{entry['id']}.json"
    out.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    print(f"recorded note {entry['id']} to {out.relative_to(ROOT.parent)}")
    print("notes carry no points and do not affect your score.")


def cmd_log(args) -> None:
    entries = load_entries()
    if args.agent:
        entries = [e for e in entries if e["agent"] == args.agent]
    if not entries:
        print("nothing recorded yet")
        return
    if args.limit:
        entries = entries[-args.limit:]
    everything = load_points()

    for e in entries:
        when = e.get("at", "")[:16].replace("T", " ")
        if e.get("kind") == "note" or not e.get("points"):
            print(f"[{when}] {e['agent']} -- NOTE")
            _print_field("finding", e.get("finding", ""))
            _print_field("evidence", e.get("evidence", ""))
        else:
            pts = e["points"]
            good = sum(
                front_fraction((p["ret"], p["vol"]), everything) <= GOOD_THRESHOLD
                for p in pts
            )
            vols = [p["vol"] for p in pts]
            print(f"[{when}] {e['agent']} -- {len(pts)} point(s), {good} good, "
                  f"vol {min(vols)*100:.1f}-{max(vols)*100:.1f}%")
            _print_field("hypothesis", e.get("hypothesis", ""))
            _print_field("prediction", e.get("prediction", ""))
        if e.get("code"):
            _print_field("code", e["code"])
        print()


def _print_field(name: str, text: str) -> None:
    if not text:
        return
    body = textwrap.fill(
        text, width=88, initial_indent="", subsequent_indent=" " * 14
    )
    print(f"  {name + ':':<12}{body}")


def cmd_report(args) -> None:
    points = load_points()
    if not points:
        print("ledger is empty")
        return
    print(f"{len(points)} point(s) from "
          f"{len({p.submission for p in points})} submission(s), "
          f"{len({p.agent for p in points})} agent(s)\n")

    for name, lo, hi in BANDS:
        in_band = [p for p in points if lo <= p.vol < hi]
        if not in_band:
            continue
        good = [p for p in in_band if front_fraction(p, points) <= GOOD_THRESHOLD]
        print(f"--- band {name} (vol {lo*100:.0f}-{hi*100:.0f}%): "
              f"{len(in_band)} point(s), {len(good)} good ---")
        rows = sorted(in_band, key=lambda p: front_fraction(p, points))[: args.top]
        _print_points(rows, points)
        print()
    _print_agent_scores(points, highlight=args.agent)


def cmd_front(args) -> None:
    points = load_points()
    if not points:
        print("ledger is empty")
        return
    good = [p for p in points if front_fraction(p, points) <= GOOD_THRESHOLD]
    good.sort(key=lambda p: p.vol)
    print(f"{len(good)} good point(s) of {len(points)}, by volatility\n")
    _print_points(good, points)


def cmd_check(args) -> None:
    points = load_points()
    frac = front_fraction((args.ret, args.vol), points)
    verdict = "GOOD" if frac <= GOOD_THRESHOLD else "not good"
    print(f"return {args.ret*100:+.2f}%, vol {args.vol*100:.2f}% "
          f"-> front fraction {frac*100:.0f}% ({verdict}) "
          f"against {len(points)} existing point(s), band {band_of(args.vol)}")


def _print_points(rows: list[Point], everything: list[Point]) -> None:
    print(f"{'agent':<22}{'variant':<26}{'return':>9}{'vol':>8}{'sharpe':>8}{'turn':>7}{'front':>8}")
    for p in rows:
        turn = f"{p.turnover:.1f}x" if p.turnover is not None else "-"
        print(
            f"{p.agent:<22}{p.label:<26}{p.ret*100:>+8.2f}%{p.vol*100:>7.2f}%"
            f"{p.sharpe:>8.3f}{turn:>7}{front_fraction(p, everything)*100:>7.0f}%"
        )


def _print_agent_scores(points: list[Point], highlight: str | None = None) -> None:
    # Notes count towards nobody's score, but an agent whose whole round was a
    # well-evidenced negative result has earned a line here all the same.
    notes: dict[str, int] = {}
    for e in load_entries():
        if e.get("kind") == "note" or not e.get("points"):
            notes[e["agent"]] = notes.get(e["agent"], 0) + 1

    agents = sorted({p.agent for p in points} | set(notes))
    print(f"{'agent':<22}{'points':>8}{'good':>7}{'score':>8}{'notes':>7}")
    for a in agents:
        mine = [p for p in points if p.agent == a]
        n = f"{notes.get(a, 0):>7}" if notes.get(a) else " " * 7
        mark = "  <-- you" if a == highlight else ""
        if not mine:
            print(f"{a:<22}{0:>8}{'-':>7}{'-':>8}{n}{mark}")
            continue
        good = sum(front_fraction(p, points) <= GOOD_THRESHOLD for p in mine)
        print(f"{a:<22}{len(mine):>8}{good:>7}{good/len(mine)*100:>7.0f}%{n}{mark}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="record a run's NAV curves as a submission")
    s.add_argument("--agent", required=True, help="your agent name")
    s.add_argument("--nav", required=True, help="the NAV CSV written by strategy_base")
    s.add_argument("--hypothesis", required=True, help="what you changed and why it should work")
    s.add_argument("--prediction", required=True, help="what you expected before running")
    s.add_argument("--code", help="path to the run directory holding the code")
    s.add_argument("--exclude", nargs="*", help="NAV columns to leave out")
    s.set_defaults(func=cmd_submit)

    n = sub.add_parser("note", help="record a finding that has no points")
    n.add_argument("--agent", required=True, help="your agent name")
    n.add_argument("--finding", required=True, help="what you concluded, in one or two sentences")
    n.add_argument("--evidence", required=True, help="the numbers behind it")
    n.add_argument("--code", help="path to the run directory holding the evidence")
    n.set_defaults(func=cmd_note)

    lg = sub.add_parser("log", help="what has been tried, in order, with reasons")
    lg.add_argument("--agent", help="only this agent's entries")
    lg.add_argument("--limit", type=int, help="only the most recent N entries")
    lg.set_defaults(func=cmd_log)

    r = sub.add_parser("report", help="the whole ledger, by band")
    r.add_argument("--agent", help="highlight this agent in the score table")
    r.add_argument("--top", type=int, default=8, help="rows per band (default: %(default)s)")
    r.set_defaults(func=cmd_report)

    f = sub.add_parser("front", help="the good points, by volatility")
    f.set_defaults(func=cmd_front)

    c = sub.add_parser("check", help="how a hypothetical point would score")
    c.add_argument("--ret", type=float, required=True, help="annualized return, e.g. 0.09")
    c.add_argument("--vol", type=float, required=True, help="annualized volatility, e.g. 0.18")
    c.set_defaults(func=cmd_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
