"""Agent construction and execution: model wiring against the Responses API.

DeepSeek implements the OpenAI Responses API (for `deepseek-v4-flash`), so the
Agents SDK talks to it through `OpenAIResponsesModel` — the SDK's primary
surface — rather than the older Chat Completions adapter. This buys native
reasoning items (thinking text streams as `response.reasoning_text.delta`
instead of being smuggled through a non-standard `reasoning_content` field)
and the server-side `web_search` tool, which the Chat Completions layer could
not offer.

DeepSeek's implementation is stateless: `previous_response_id`, `conversation`
and `store` are unsupported, so the full item list is resent every turn — which
is exactly what the SDK does by default here. Reasoning items are echoed back
as plain text (`encrypted_content` is never populated), and the server rebuilds
a `web_search_call`'s results from its id when the item is passed back verbatim.
"""

from __future__ import annotations

from agents import Agent, ModelSettings, OpenAIResponsesModel, RunResultStreaming, Runner, Tool, set_tracing_disabled
from openai import AsyncOpenAI
from openai.types import Reasoning


def build_agent(
    name: str,
    instructions: str,
    tools: list[Tool],
    *,
    base_url: str,
    api_key: str,
    model: str,
) -> Agent:
    """Construct an agent against a DeepSeek (or compatible) endpoint."""

    # The SDK's tracing exporter uploads to the OpenAI platform, which we are
    # not using; disable it so it does not warn about a missing OpenAI key.
    set_tracing_disabled(True)
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    agent = Agent(
        name=name,
        instructions=instructions,
        model=OpenAIResponsesModel(model=model, openai_client=client),
        model_settings=ModelSettings(reasoning=Reasoning(effort="high")),
        tools=tools,
    )
    return agent


def run_agent(agent: Agent, items: list) -> RunResultStreaming:
    """Execute one run to completion, return the new history."""

    result = Runner.run_streamed(agent, input=items, max_turns=None)
    return result
