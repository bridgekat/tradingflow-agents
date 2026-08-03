"""Reusable instruction blocks, composed by agents into their prompts.

Each block is a markdown section with no leading or trailing blank lines; an
agent builds its instructions by joining its persona, the blocks it needs, and
any role-specific sections with blank lines in between.
"""

from __future__ import annotations

import platform
from datetime import date
from pathlib import Path

from .tools import shell_executable


def environment_section() -> str:
    """Facts about the machine the agent's tools run on."""

    return f"""\
# Environment

- Operating system: {platform.system()} {platform.release()} ({platform.machine()})
- Shell for the `run_command` tool: {shell_executable()}
- Working directory at the start of this conversation: {Path.cwd()}
- Date at start of this conversation: {date.today().isoformat()}"""


# Tips on using the core tool set effectively.
GENERAL_TIPS = """\
# General tips

- Make sure to read and understand any code you are trying to run: know its capabilities, limitations and estimate its performance. Make sure inputs are correct, and the logic is sound.
- If any doc link appears to lead to an important or information-dense piece of documentation (e.g. overview, tutorial, interface spec), follow it. More knowledge helps you take informed actions.
- Keep consistent style in your code.
- If a command can take more than a few seconds to run, you can use `run_command` with `wait_seconds = 0` to run it in the background, then use `check_command` to check for its outputs periodically. This also allows for running programs in parallel.
- When something is running, you can `wait` for shorter periods of time (e.g., 5-30 seconds) to see if it is making progress at an expected pace. After that, you can repeatedly `wait` for longer periods of time.
- `web_search` runs on the server: a single call may issue several queries and open pages on its own. Use `web_fetch` when you already know the exact URL you want, or need a page's raw text (e.g. a CSV or JSON endpoint) rather than a summary.
- When done, summarize what you changed and how it was verified. If something failed or was skipped, say so plainly."""


# Orientation for working with the TradingFlow framework.
TRADINGFLOW_NOTES = """\
# Notes on TradingFlow

Whenever there is a need to experiment with a complex trading strategy (where hand-written scripts are likely slow to run, or otherwise suboptimal), you can choose to use TradingFlow. In which case, the following tips may help:

- The TradingFlow project is a quantitative trading backtesting framework written in Rust. The system environment is already configured with standard Rust and Python toolchains to build it.
- Its code should be self-documenting: read module-level docstrings, starting from crate roots and follow the doc links to get an idea of how everything works first.
- Ensure you are running experiments with `release` profile (`cargo run --release`) so that compiler optimization is enabled. If a single run seems to be taking more than 30 minutes to complete, consider reducing the data range or optimizing the implementation. If this is impossible or inconvenient, proceed by waiting anyway.

If you decide to use TradingFlow, always read its `README.md` and the crate roots `crates/tradingflow/src/lib.rs`, `crates/tradingflow-data/src/lib.rs`, `crates/tradingflow-graph/src/lib.rs` and understand its overall design first, before going back to the user's request."""
