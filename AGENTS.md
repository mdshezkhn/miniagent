# miniagent — a coding agent you fully own

Source: D:/projects/mini-agent · Python 3.12 · dependencies: `openai`, `httpx` (+ `fastapi`/`uvicorn` for the GUI)

## The whole idea (read this first)

Every coding agent — Claude Code, Links, OpenHands, this one — is the same loop:

    1. context (system prompt + conversation)  ->  LLM
    2. LLM answers with text  OR  tool calls
    3. your code runs the tools, appends results
    4. repeat until the LLM stops calling tools

This project is that loop, ~500 commented lines, nothing hidden.

## Layout

    miniagent/
      providers.py  which brain to use (BigModel/OpenRouter/NVIDIA/Gemini/Groq/DeepSeek/custom)
      llm.py        streaming chat client (works with any OpenAI-compatible API)
      tools.py      the agent's hands: bash, read/write file, list, search, web_fetch, ask_user (+ safety gates)
      agent.py      THE LOOP + system prompt + context compaction
      cli.py        REPL & one-shot mode, `providers` and `selftest` subcommands
      gui.py        browser UI (FastAPI + SSE + one embedded page, approvals become buttons)
      store.py      conversations as JSON under .miniagent/sessions/ — survive restarts
      config.py     tiny .env loader

## Quick start

    # 1) put one or more API keys in .env (copy from .env.example)
    # 2) create the venv and install
    uv venv -p 3.12 .venv
    uv pip install -e .            # CLI only (1 dependency)
    uv pip install -e ".[gui]"     # + web GUI (fastapi, uvicorn)

    # 3) verify tools work without spending a token
    miniagent selftest
    miniagent providers

    # 4) talk to it
    miniagent                          # REPL (uses first provider whose key is set)
    miniagent --provider groq          # pick the brain
    miniagent --model gemini-2.5-flash --provider gemini
    miniagent "explain the structure of this repo"   # one-shot

    # 5) or run the GUI — chat in your browser
    miniagent gui                      # -> http://127.0.0.1:7860
    miniagent gui --port 7861 --host 127.0.0.1

    Conversations persist across restarts (JSON under .miniagent/sessions/,
    newest is auto-resumed in both REPL and GUI; /new starts a fresh one).

## Adding a tool (the classic extension)

`web_fetch` (v0.3) is the worked example — read it in tools.py end to end:

1. Add a JSON-schema dict to `TOOL_SCHEMAS` in `tools.py`
2. Add a handler (`_web_fetch`) + an entry in `_HANDLERS`
3. Mention it in the system prompt in `agent.py`
4. Add a hermetic check to `cmd_selftest` in `cli.py` (local test server)
That's it — the loop doesn't change. That's the whole trick.

Gotchas learned the hard way (all fixed in code, commented where they bite):
- httpx/urllib honor HTTP_PROXY env vars — localhost fetches die in the
  proxy; bypass it for local/private hosts (`trust_env=not is_local`)
- NVIDIA NIM model entitlements ROTATE: yesterday's model can 404 tomorrow
  ("Function not found for account") — re-probe GET /v1/models, pick another
- sorting session files by NAME breaks when two saves land in the same
  second; sort by mtime (store.py)

## Learning path from here

- read `agent.py` first (the loop), then `tools.py` (safety), then `llm.py` (streaming)
- the GUI lesson: `gui.py` changed ZERO lines of the loop — it only attaches
  callbacks (`llm.on_text`, `agent.on_event`, ToolContext `confirm`/`ask`).
  Good architecture = UI is an adapter, not a rewrite. Also study gui.py's
  SSE + worker-thread + approval-bridge pattern; it's what every web agent UI does.
- v0.4 adds the big-assistant features, each mapped to its product:
  web_search + cite-your-sources (Perplexity: retrieve -> rank -> cited answer),
  remember tool + .miniagent/memory.md injected at session start (ChatGPT memory),
  AGENTS.md auto-loaded into the system prompt (Claude Code's CLAUDE.md).
  The loop file (agent.py) still barely changed: build_system_prompt() + one tool each.
- exercises: per-tool approval rules · token counting & smarter compaction ·
  multi-session picker (list .miniagent/sessions/, switch chats)
- then study a production agent for the advanced patterns (aider / OpenHands / opencode)
