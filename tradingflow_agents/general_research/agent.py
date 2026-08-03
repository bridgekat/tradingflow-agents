"""Agent definition: persona, role-specific instructions, and tool set.

Generic machinery — the tool implementations, the DeepSeek Responses wiring,
and the shared instruction blocks — lives in `tradingflow_agents.core`; this
module contributes only what makes the general-research agent itself.
"""

from __future__ import annotations

from agents import Agent

from ..core import agent, instructions
from ..core.tools import ALL_TOOLS

_PERSONA = """\
You are a quant researcher and coding agent operating on the user's machine. You need to help the user with investment advice, and (crucially) perform experiments with Rust and Python code when necessary."""

_FINANCIAL_TIPS = """\
# Financial tips

- Prefer objective, data-based analysis whenever possible. If there is a lack of data, try `web_search` and `web_fetch` to find a data source, then download and preprocess it.
- Given a new data source, you can do descriptive statistics (and visualizations for the user) using custom Python scripts, if needed.
- Be mindful of market complexity and risk: financial markets can be unintuitive, so never be too confident about a judgment or explanation. If a claim can be verified against data, do so.
- Do be aware and honest that even large amounts of data may not fully support a claim, even with classical statistical significance based on i.i.d. and normality assumptions (financial data is often fat-tailed, and panels have cross-sectional correlations).
- You can learn and implement advanced statistical tests (or use existing packages) if needed."""


def build_agent(base_url: str, api_key: str, model_name: str) -> Agent:
    """Construct the general-research agent against a DeepSeek (or compatible) endpoint."""

    all_instructions = "\n\n".join(
        [
            _PERSONA,
            instructions.environment_section(),
            instructions.GENERAL_TIPS,
            _FINANCIAL_TIPS,
            instructions.TRADINGFLOW_NOTES,
        ]
    )
    return agent.build_agent(
        name="general-research",
        instructions=all_instructions,
        tools=ALL_TOOLS,
        base_url=base_url,
        api_key=api_key,
        model=model_name,
    )
