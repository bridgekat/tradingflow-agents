# Autoresearch

A multi-objective search for trading strategies: several explorer agents work
in parallel on different parts of the return/volatility frontier, and everything
anyone finds goes into one shared ledger.

## Starting it

From a Claude Code session at the repository root:

```
Read autoresearch/orchestrator.md and run a round of the search.
```

That session becomes the orchestrator. It reads the ledger, decides what to
explore, and spawns explorers as subagents — each of which reads
`autoresearch/explorer.md` and works in its own directory.

## The objective

Two axes, neither traded against the other: **return up, volatility down**.

A point is **good** when at most 10% of the other points in the ledger dominate
it. An agent's score is the fraction of its own submissions that are good — so
selectivity pays and volume does not.

The band near the front rather than a strict Pareto rank is deliberate. Return
over a few years carries a standard error comparable to the spread between
strategies, so a lucky run can dominate a lot of honest work without being a
better strategy. Tolerating a few dominators keeps real results on the board
when a fluke lands above them. Volatility, by contrast, is estimated an order of
magnitude more precisely — the two axes are not equally trustworthy, and both
sets of instructions say so.

## Layout

| Path | What it is |
| ---- | ---------- |
| `orchestrator.md` | Instructions for the orchestrating session |
| `explorer.md` | Instructions for each explorer subagent |
| `ledger.py` | Ledger tool: `submit`, `report`, `front`, `check` |
| `ledger/` | One JSON file per submission — no shared file, no write races |
| `../runs/` | Explorer working directories (repo root, so the crate's relative dependency resolves) |

## The ledger

Points are computed from the artifacts a run leaves behind, not transcribed by
the agent that produced them:

```bash
python autoresearch/ledger.py submit --agent NAME --nav runs/.../attempt.csv \
    --hypothesis "..." --prediction "..."
python autoresearch/ledger.py report          # everything, by volatility band
python autoresearch/ledger.py front           # just the good points
python autoresearch/ledger.py check --ret 0.09 --vol 0.18   # score a hypothetical
```

Thresholds and bands are constants at the top of `ledger.py`. `EPS_RET` is
worth knowing about: raise it to demand that a return advantage clear some of
its own estimation error before it counts as beating anything.

## Rules the agents work under

- The window is **2015-01-01 to 2023-01-01**. Everything after is held out.
- Each explorer builds and runs in its own directory; nothing shared is edited.
- Two or three explorers at a time — backtests saturate the machine.
- No external data.

## Prototype status

This is a working skeleton, not a finished system. It has no validator, no
out-of-sample gate, and no protection against an agent that games the ledger —
the instructions ask for good faith rather than enforcing it. `validation/`
holds a scorer that recomputes results from weights alone, which is where that
enforcement would attach when it is wanted.
