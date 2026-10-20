"""miniagent — a minimal from-scratch coding agent.

Six files that matter:
    providers.py  which LLM brain to talk to
    llm.py        how to talk to it (streaming, OpenAI-compatible)
    tools.py      what the agent can do (its hands)
    agent.py      the loop (its brain-stem)  <-- start reading here
    cli.py        how the human drives it (terminal)
    gui.py        ...or the browser (web UI)
    store.py      conversations as JSON files (survive restarts)
"""

__version__ = "0.4.0"
