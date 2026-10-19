# miniagent — a coding agent you fully own

Source: D:/projects/mini-agent · Python 3.12 · single dependency: `openai`

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
      tools.py      the agent's hands: bash, read/write file, list, search, ask_user (+ safety gates)
      agent.py      THE LOOP + system prompt + context compaction
      cli.py        REPL & one-shot mode, `providers` and `selftest` subcommands
      config.py     tiny .env loader

## Quick start

    # 1) put one or more API keys in .env (copy from .env.example)
    # 2) create the venv and install
    uv venv -p 3.12 .venv
    uv pip install -e .

    # 3) verify tools work without spending a token
    miniagent selftest
    miniagent providers

    # 4) talk to it
    miniagent                          # REPL (uses first provider whose key is set)
    miniagent --provider groq          # pick the brain
    miniagent --model gemini-2.5-flash --provider gemini
    miniagent "explain the structure of this repo"   # one-shot

## Adding a tool (the classic extension)

1. Add a JSON-schema dict to `TOOL_SCHEMAS` in `tools.py`
2. Add an `elif name == "your_tool":` branch in `execute()`
3. Mention it in the system prompt in `agent.py`
That's it — the loop doesn't change. That's the whole trick.

## Learning path from here

- read `agent.py` first (the loop), then `tools.py` (safety), then `llm.py` (streaming)
- exercises: add a `web_fetch` tool · add per-tool approval rules · add token counting & smarter compaction
- then study a production agent for the advanced patterns (aider / OpenHands / opencode)
