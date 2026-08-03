"""Command-line interface: one-shot mode and an interactive REPL."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from agents import Agent
from agents.exceptions import AgentsException
from dotenv import load_dotenv
from openai import OpenAIError

from ..core import tools
from ..core.console import run_with_output
from .agent import build_agent

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="general-research",
        description="A simple general-purpose quant research and coding agent (OpenAI Agents SDK + DeepSeek).",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        help="run a single task non-interactively instead of starting the REPL",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
        help="model name (default: %(default)s; DeepSeek serves the Responses API for"
        " deepseek-v4-flash only — deepseek-v4-pro is expected in early August 2026)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="run shell commands without asking for approval",
    )
    return parser.parse_args()


async def _oneshot(agent: Agent, prompt: str) -> None:
    await run_with_output(agent, [{"role": "user", "content": prompt}])


async def _repl(agent: Agent, prog: str) -> None:
    print(f"{prog} — type a task, '/exit' to quit, '/clear' to reset history.")
    items: list = []

    while True:
        try:
            user = input("\n> ").strip()
            print()
        except EOFError:
            break

        if not user:
            continue

        if user == "/exit":
            break

        if user == "/clear":
            items = []
            print("(history cleared)")
            print()
            continue

        items.append({"role": "user", "content": user})
        try:
            items = await run_with_output(agent, items)
        except (AgentsException, OpenAIError) as e:
            # Drop the failed turn from history and keep the REPL alive.
            items.pop()
            print(f"\nerror: {e}", file=sys.stderr)


def main() -> None:
    load_dotenv()  # before _parse_args: argument defaults read the environment
    args = _parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("error: DEEPSEEK_API_KEY is not set (see .env.example).")
    tools.AUTO_APPROVE = args.yes

    agent = build_agent(args.base_url, api_key, args.model)
    try:
        if args.prompt:
            asyncio.run(_oneshot(agent, args.prompt))
        else:
            asyncio.run(_repl(agent, "general-research"))
    except KeyboardInterrupt:
        print()
    except (AgentsException, OpenAIError) as e:
        # The REPL recovers per turn; this catches one-shot runs and anything
        # raised outside a turn (e.g. a model the endpoint refuses to serve).
        sys.exit(f"\nerror: {e}")


if __name__ == "__main__":
    main()
