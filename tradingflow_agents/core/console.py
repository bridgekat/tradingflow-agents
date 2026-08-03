"""Streamed run output: assistant text and the model's thinking are printed as
they arrive, and tool calls / results are shown as single dim lines so the
agent loop is observable."""

from __future__ import annotations

from agents import Agent
from openai.types.responses import (
    ResponseFunctionWebSearch,
    ResponseInputItemParam,
    ResponseReasoningTextDeltaEvent,
    ResponseTextDeltaEvent,
)

from .agent import run_agent


def _short(text: str, limit: int = 200) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _clean_url(url: str) -> str:
    # DeepSeek tags the URLs of hosted-search steps with the call id.
    return url.split("#ws_call_id=")[0]


def _describe_search(action: object) -> str:
    """One-line description of a step the server-side web search took."""

    match getattr(action, "type", ""):
        case "open_page":
            return f"open {_clean_url(getattr(action, 'url', ''))}"
        case "find":
            return f"find {getattr(action, 'pattern', '')!r} in {_clean_url(getattr(action, 'url', ''))}"
        case _:
            # `queries` carries the call id alongside the real queries; older
            # payloads use the singular `query` instead.
            queries = [q for q in (getattr(action, "queries", None) or []) if not q.startswith("ws_call_id=")]
            if not queries and (query := getattr(action, "query", None)):
                queries = [query]
            return "search " + " | ".join(queries)


async def run_with_output(agent: Agent, items: list) -> list[ResponseInputItemParam]:
    """Run the agent once, streaming everything to the terminal; return the new history."""

    _DIM = "\x1b[2m"
    _RESET = "\x1b[0m"

    result = run_agent(agent, items)
    # Which kind of delta is mid-stream, so a switch between them (or any other
    # event) can close it off: "text" for the answer, "thinking" for reasoning.
    streaming: str | None = None

    def end_deltas() -> None:
        nonlocal streaming
        if streaming is None:
            return
        if streaming == "thinking":
            print(_RESET, end="", flush=True)
        print("\n", flush=True)
        streaming = None

    async for event in result.stream_events():
        if event.type == "raw_response_event":
            if isinstance(event.data, ResponseTextDeltaEvent):
                if streaming != "text":
                    end_deltas()
                    streaming = "text"
                print(event.data.delta, end="", flush=True)
            elif isinstance(event.data, ResponseReasoningTextDeltaEvent):
                if streaming != "thinking":
                    end_deltas()
                    print(f"{_DIM}[thinking] ", end="", flush=True)
                    streaming = "thinking"
                print(event.data.delta, end="", flush=True)
        else:
            end_deltas()

            if event.type == "run_item_stream_event":
                item = event.item
                if item.type == "message_output_item":
                    raw = item.raw_item
                    for content in raw.content:
                        if content.type == "output_text":
                            pass  # already printed from the text deltas above
                        elif content.type == "refusal":
                            print(content.refusal, flush=True)

                elif item.type == "reasoning_item":
                    pass  # already printed from the reasoning deltas above

                elif item.type == "tool_call_item":
                    raw = item.raw_item
                    if isinstance(raw, ResponseFunctionWebSearch):
                        # Hosted: it ran on DeepSeek's side, so there is no
                        # matching tool_call_output_item to print afterwards.
                        print(f"\n{_DIM}[tool] web_search {_short(_describe_search(raw.action))}{_RESET}", flush=True)
                    else:
                        name = getattr(raw, "name", "?")
                        args = getattr(raw, "arguments", "")
                        print(f"\n{_DIM}[tool] {name} {_short(args)}{_RESET}", flush=True)

                elif item.type == "tool_call_output_item":
                    print(f"{_DIM}[tool] ->", flush=True)
                    print(f"{_DIM}{_short(item.output)}{_RESET}", flush=True)

                else:
                    print(f"{_DIM}[other] {item.type}{_RESET}", flush=True)

            elif event.type == "agent_updated_stream_event":
                print(f"{_DIM}[agent] {event.new_agent.name}{_RESET}", flush=True)

    print()
    return result.to_input_list()
