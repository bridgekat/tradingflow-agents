# TradingFlow Agents

LLM agents for experimenting with trading strategies on [TradingFlow](https://github.com/bridgekat/tradingflow).

```bash
git submodule update --init --recursive
uv sync
```

Configuration comes from flags or environment (a `.env` in the working
directory is loaded automatically):

| Variable            | Default                    |                        |
| ------------------- | -------------------------- | ---------------------- |
| `DEEPSEEK_API_KEY`  | — (required)               | API key                |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | Responses API endpoint |
| `DEEPSEEK_MODEL`    | `deepseek-v4-flash`        | the only model DeepSeek serves over Responses |

`deepseek-v4-flash` is a thinking model (the harness asks for `effort="high"`).
`deepseek-v4-pro` is Responses-capable from early August 2026 and rejected with
a 400 until then.

## The `general-research` agent

A simple general-purpose quant research and coding agent built on the
[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/), pointed at
the DeepSeek API. Run it from the directory you want it to work in (tools
resolve relative paths against the current working directory):

```bash
general-research                                       # interactive REPL
general-research -p "add a --verbose flag to cli.py"   # one-shot task
general-research -y                                    # don't ask before running shell commands
```

In the REPL: `/exit` to leave, `/clear` to reset conversation history.

## Adding a new agent

An agent is a package under `tradingflow_agents/` with two small modules:

- `agent.py` — a `build_agent(base_url, api_key, model_name)` factory that
  composes instructions (persona and role-specific sections, plus shared
  blocks from `core.instructions`) and picks tools, handing both to
  `core.agent.build_agent`;
- `cli.py` — the entry point: argument parsing and a one-shot or REPL loop,
  streaming output through `core.bash.run_with_output`.

Register the executable in `pyproject.toml` under `[project.scripts]`.

## Transport: the Responses API

DeepSeek now implements the **Responses API**, so the Agents SDK is wired to it
through `OpenAIResponsesModel` (the SDK's primary surface) instead of the older
`OpenAIChatCompletionsModel` adapter. Over Chat Completions this gains:

- **Native reasoning items.** Thinking text arrives as
  `response.reasoning_text.delta` events rather than a non-standard
  `reasoning_content` field, and is streamed live as dim `[thinking]` output.
- **Server-side web search.** `web_search` is a hosted tool DeepSeek runs
  itself — see below.

Caveats of DeepSeek's implementation, all handled in `core/agent.py`:

| Behaviour | Consequence |
| --------- | ----------- |
| Stateless — `previous_response_id`, `conversation`, `store` unsupported | The whole item list is resent each turn (the SDK's default here) |
| `reasoning.summary` accepted but never produced | Ask for `effort` only; read the reasoning item's plain-text content |
| `truncation` unsupported | Overflowing the context window is a 400, not a silent trim |
| Only `deepseek-v4-flash` is served | `deepseek-v4-pro` returns 400 until early August 2026 |

## Tool set

All agents share the tool implementations in `core/tools.py`:

| Tool            | Purpose                                                    |
| --------------- | ---------------------------------------------------------- |
| `wait`          | Wait for a specified duration or keypress                  |
| `run_command`   | Shell command (asks for approval unless `-y`); commands outliving `wait_seconds` continue as background jobs |
| `check_command` | Status + incremental output of a background job            |
| `kill_command`  | Kill a background job (and its process tree)               |
| `list_dir`      | List directory entries                                     |
| `read_file`     | Read a file with line numbers (paged via `offset`/`limit`) |
| `write_file`    | Create/overwrite a file                                    |
| `edit_file`     | Exact-string replacement (unique match required)           |
| `glob`          | Find files by glob (`rg --files`, .gitignore-aware)        |
| `grep`          | Regex search via ripgrep (.gitignore-aware, context lines) |
| `web_fetch`     | Fetch a page as plain text (paged via `offset`)            |
| `web_search`    | Web search, run server-side by DeepSeek (`WebSearchTool`)  |

Every tool but `web_search` runs in-process and returns errors as strings,
so the model can observe and recover; outputs are truncated at fixed caps to
protect the context window.

### On the built-in tools

`web_search` is the SDK's hosted `WebSearchTool`, which DeepSeek's Responses
API implements: one call may issue several queries and open pages server-side,
and the results are restored from the call id when the item is passed back, so
they never occupy the harness's context. It replaces the previous keyless
DuckDuckGo (`ddgs`) function tool.

The SDK's other hosted tools — `FileSearchTool`, `CodeInterpreterTool`,
`ComputerTool` — are *not* usable: DeepSeek serves `function` and `web_search`
only and silently ignores other tool types. `web_fetch` therefore stays a local
function tool; it also does something hosted search does not, namely retrieve a
specific URL verbatim (e.g. a CSV or JSON endpoint).

## Caveats

- There is no sandbox: an agent can touch anything your user account can.
  Shell commands require interactive approval unless you pass `-y`.
- SDK tracing is disabled (it would try to upload traces to the OpenAI
  platform, which this harness does not use).
- `web_search` is billed and executed by DeepSeek: the queries, and the pages it
  chooses to open, are its server's traffic rather than this machine's.
