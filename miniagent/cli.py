"""CLI — how the human drives the agent.

    miniagent                      interactive REPL (conversation persists)
    miniagent "do a thing"         one-shot task, then exit
    miniagent gui                  local web GUI (browser chat interface)
    miniagent providers            which brains have API keys configured
    miniagent selftest             exercise all tools locally, zero tokens

Flags:
    --provider bigmodel|openrouter|nvidia|gemini|groq|deepseek
    --model <any-model-id>         overrides the provider default
    --base-url / --api-key         fully custom OpenAI-compatible endpoint
    --auto / -y                    skip confirmation prompts (hard blocklist stays)
    --max-turns N                  runaway guard
    --host / --port                gui mode only (defaults 127.0.0.1:7860)
"""

import argparse
import os
import sys
import tempfile

from . import __version__, providers
from .agent import Agent
from .config import load_env
from .llm import LLM


# ---------------------------------------------------------------------- #
#  Human-interaction callbacks injected into the tools                   #
# ---------------------------------------------------------------------- #

def console_confirm(prompt: str) -> bool:
    try:
        answer = input(f"{prompt}\n  approve? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def console_ask(question: str) -> str:
    try:
        return input(f"\n[agent asks] {question}\n> ").strip()
    except EOFError:
        return ""


# ---------------------------------------------------------------------- #
#  Subcommands that don't need an LLM (and cost nothing)                 #
# ---------------------------------------------------------------------- #

def cmd_providers() -> int:
    print(f"{'id':<12} {'key':<5} {'default model':<34} provider")
    print("-" * 78)
    for pid, name, model, has_key in providers.status_table():
        print(f"{pid:<12} {'yes' if has_key else '-':<5} {model:<34} {name}")
    print("-" * 78)
    print("Set a key in .env (see .env.example) or pass --api-key.")
    print("The agent auto-picks the first provider with a key, in order:")
    print("  " + " > ".join(providers.PREFERENCE_ORDER))
    return 0


def cmd_selftest() -> int:
    """Exercise every tool in a temp sandbox — verifies the agent's hands
    without spending a single token."""
    sandbox = tempfile.mkdtemp(prefix="miniagent-selftest-")
    ctx_agrees = type("Ctx", (), {})  # placeholder, real ctx below
    from . import tools
    ctx = tools.ToolContext(cwd=sandbox, confirm=lambda p: True, ask=lambda q: "42")

    results = []

    def check(label, cond):
        results.append((label, bool(cond)))

    # write -> read -> list round-trip
    r = tools.execute("write_file",
                      f'{{"path": "hello.txt", "content": "line1\\nline2\\n"}}', ctx)
    check("write_file", "created" in r)
    r = tools.execute("read_file", '{"path": "hello.txt"}', ctx)
    check("read_file", "1->line1" in r)
    r = tools.execute("list_dir", "{}", ctx)
    check("list_dir", "hello.txt" in r)

    # search finds what we wrote
    r = tools.execute("search_files", '{"pattern": "line2", "glob": "*.txt"}', ctx)
    check("search_files", "hello.txt:2:line2" in r)

    # bash round-trip (echo works in both cmd.exe and sh)
    r = tools.execute("run_bash", '{"command": "echo selftest-ok"}', ctx)
    check("run_bash", "selftest-ok" in r)

    # safety: hard blocklist must refuse, unknown tool must error politely
    r = tools.execute("run_bash", '{"command": "rm -rf /"}', ctx)
    check("safety-blocklist", r.startswith("Error"))
    r = tools.execute("teleport", "{}", ctx)
    check("unknown-tool", r.startswith("Error"))
    r = tools.execute("read_file", '{"path": "nope.txt"}', ctx)
    check("missing-file", r.startswith("Error"))

    # schema sanity: every handler has a schema and vice versa
    names = {t["function"]["name"] for t in tools.TOOL_SCHEMAS}
    check("schema-handler parity", names == set(tools._HANDLERS))

    width = max(len(l) for l, _ in results) + 2
    for label, ok in results:
        print(f"{'PASS' if ok else 'FAIL'}  {label:<{width}}")
    failed = [l for l, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed"
          + (f" | FAILURES: {failed}" if failed else " | all good."))
    return 1 if failed else 0


# ---------------------------------------------------------------------- #
#  Main                                                                   #
# ---------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="miniagent",
        description="A minimal from-scratch coding agent (Claude Code-style loop).",
    )
    ap.add_argument("task", nargs="*", help="task text (omit to enter the REPL)")
    ap.add_argument("--provider", "-p", help="bigmodel|openrouter|nvidia|gemini|groq|deepseek")
    ap.add_argument("--model", "-m", help="model id (overrides provider default)")
    ap.add_argument("--base-url", help="custom OpenAI-compatible endpoint")
    ap.add_argument("--api-key", help="API key (else taken from .env / environment)")
    ap.add_argument("--auto", "-y", action="store_true",
                    help="skip approval prompts (hard safety blocklist still applies)")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--host", default="127.0.0.1", help="gui mode: bind address")
    ap.add_argument("--port", type=int, default=7860, help="gui mode: port")
    ap.add_argument("--version", action="version", version=f"miniagent {__version__}")
    return ap


def main() -> int:
    load_env()
    args = build_parser().parse_args()

    sub = args.task[0] if args.task else ""
    if sub == "providers":
        return cmd_providers()
    if sub == "selftest":
        return cmd_selftest()

    if sub == "gui":
        try:
            from . import gui
        except ImportError:
            raise SystemExit(
                "GUI extras not installed. Run:  uv pip install -e '.[gui]'"
            )
        gui.serve(host=args.host, port=args.port)
        return 0

    llm = LLM(provider=args.provider, model=args.model,
              base_url=args.base_url, api_key=args.api_key)
    agent = Agent(llm, cwd=os.getcwd(),
                  confirm=console_confirm, ask=console_ask,
                  auto=args.auto, max_turns=args.max_turns)

    print(f"miniagent v{__version__} | brain: {llm.label} / {llm.model} | "
          f"dir: {os.getcwd()} | mode: {'auto' if args.auto else 'confirm'}")

    if args.task:  # one-shot
        try:
            agent.run(" ".join(args.task))
        except KeyboardInterrupt:
            print("\n(interrupted)")
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            # Friendly hints for the errors you'll actually hit.
            hint = ""
            text = str(e)
            if "402" in text or "credit" in text.lower():
                hint = " -> the provider account is out of credits; top up or switch brains (--provider)"
            elif "401" in text or "api key" in text.lower():
                hint = " -> check the API key in .env"
            elif "404" in text and "model" in text.lower():
                hint = " -> this model id isn't available to your account; try another (--model)"
            print(f"\n[error] {type(e).__name__}: {text[:300]}{hint}")
            return 1
        return 0

    # ---------------- REPL: a loop around the loop ---------------- #
    history: list[dict] = []
    print("Type a task, or /help. /new clears the conversation.\n")
    while True:
        try:
            task = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not task:
            continue
        if task in ("/exit", "/quit"):
            return 0
        if task == "/new":
            history.clear()
            print("(conversation cleared)\n")
            continue
        if task == "/help":
            print("commands: /new (clear conversation) /exit  — anything else is a task\n")
            continue
        try:
            agent.run(task, history=history)
        except KeyboardInterrupt:
            print("\n(interrupted)\n")
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001 — REPL must survive anything
            print(f"\n[error] {type(e).__name__}: {e}\n")
        print()


if __name__ == "__main__":
    sys.exit(main())
