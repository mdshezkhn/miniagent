"""THE AGENT LOOP — the heart of everything.

If you read one file in this project, read this one. Claude Code, OpenHands,
aider, Links (the assistant you're chatting with right now) — all of them
wrap this same while-loop in more polish:

    while True:
        message = llm.chat(system_prompt + conversation, tool_schemas)
        if message has no tool_calls:  -> done, that's the answer
        for each tool_call:            -> execute, append result, loop

Everything else here is production hygiene:
  * max_turns      — a runaway model can't loop forever
  * _compact       — context is finite; old tool outputs get squeezed
  * the system prompt — identity + rules; where 'safety' actually lives
"""

from . import tools
from .llm import LLM

SYSTEM_PROMPT = """\
You are mini-agent, a coding assistant with tool access running in a shell \
on the user's own computer. You solve tasks step by step.

# How you work
- Each turn you may call tools, observe their results, and continue — \
until you can give a final answer, which ends the turn.
- Tool results are plain text. An "Error: ..." result means: diagnose it \
and try something different. Never repeat the exact same failing call.
- Some actions require user approval. If the user refuses, adapt or ask \
what they'd prefer.

# Tools
- run_bash: run a shell command (builds, git, installs, quick analysis).
- read_file / write_file / list_dir / search_files: explore and edit the \
project in the working directory. Read a file before you edit it.
- web_search + web_fetch: search the live web, then read the best hits. \
When your answer relies on the web, LIST THE SOURCE URLS at the end.
- remember: save a durable fact (preferences, conventions, gotchas) to \
long-term memory for future sessions.
- ask_user: only when a human decision is genuinely needed.

# Rules
1. Explore first: list/read the relevant files before proposing changes.
2. Prefer reversible actions; never delete or force-push without asking.
3. Keep going autonomously on clear tasks; do not ask permission for \
every step.
4. Be concise. When finished, summarize what you did and what changed.
"""

MAX_TURNS = 30                 # runaway-loop guard
COMPACT_THRESHOLD = 120_000    # chars of conversation before compaction
COMPACT_TOOL_KEEP = 1_500      # old tool outputs squeezed to this length

# Caps for injected context (the agent-harness skill's advice: keep the
# system prompt minimal; big guidance lives in files and is loaded clipped).
AGENTS_MD_CAP = 6_000
MEMORY_CAP = 4_000


def build_system_prompt(cwd: str) -> str:
    """Claude-Code-style context assembly: static identity + project md +
    long-term memory, each capped. (Claude Code reads CLAUDE.md/AGENTS.md the
    same way; ChatGPT injects stored memories at query time.)"""
    from pathlib import Path

    parts = [SYSTEM_PROMPT]

    agents_md = Path(cwd) / "AGENTS.md"
    if agents_md.is_file():
        try:
            text = agents_md.read_text(encoding="utf-8", errors="replace")
            if len(text) > AGENTS_MD_CAP:
                text = text[:AGENTS_MD_CAP] + "\n... [AGENTS.md clipped]"
            parts.append("# Project context (AGENTS.md — instructions for you)\n" + text)
        except OSError:
            pass

    from . import store
    mem = store.load_memory(base=cwd)
    if mem:
        if len(mem) > MEMORY_CAP:
            mem = mem[:MEMORY_CAP] + "\n... [memory clipped]"
        parts.append("# Long-term memory from earlier sessions\n" + mem)

    return "\n\n".join(parts)


class Agent:
    def __init__(self, llm: LLM, cwd: str, confirm, ask,
                 auto: bool = False, max_turns: int = MAX_TURNS,
                 on_event=None):
        """on_event(dict): optional callback for tool activity —
        {"type": "tool_start"|"tool_result", "name": ..., ...}.
        Attached by the GUI; absent means CLI (printed instead).
        THE LOOP ITSELF NEVER CHANGES: same class powers both UIs."""
        self.llm = llm
        self.max_turns = max_turns
        self.on_event = on_event
        # Dynamic system prompt: identity + AGENTS.md + long-term memory.
        self.system_prompt = build_system_prompt(cwd)
        self.ctx = tools.ToolContext(cwd=cwd,
                                     confirm=(lambda _prompt: True) if auto else confirm,
                                     ask=ask)

    def _emit(self, event: dict) -> None:
        """Report tool activity: to the GUI if attached, else to stdout."""
        if self.on_event:
            self.on_event(event)
            return
        if event["type"] == "tool_start":
            args = event["args"]
            preview = args if len(args) <= 160 else args[:157] + "..."
            print(f"[tool] {event['name']} {preview}")
        else:
            first = event["result"].splitlines()[0] if event["result"] else ""
            print(f"       -> {first[:150]}")

    # ------------------------------------------------------------------ #

    def run(self, task: str, history: list[dict] | None = None) -> str:
        """Run one task to completion. Returns the final text answer.

        `history` lets the CLI keep one conversation across many tasks
        (that's all a REPL is: a loop around this loop).
        """
        messages = history if history is not None else []
        if not messages:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": task})

        for _turn in range(self.max_turns):
            self._maybe_compact(messages)

            # -- 1) ask the brain, possibly getting tool calls back
            reply = self.llm.chat(messages, tools.TOOL_SCHEMAS)
            messages.append(reply)

            # -- 2) no tool calls => the model is done talking
            if not reply.get("tool_calls"):
                return reply.get("content") or ""

            # -- 3) execute each requested tool, feed results back
            for call in reply["tool_calls"]:
                name = call["function"]["name"]
                raw_args = call["function"]["arguments"]

                self._emit({"type": "tool_start", "name": name, "args": raw_args})
                result = tools.execute(name, raw_args, self.ctx)
                messages.append({"role": "tool",
                                 "tool_call_id": call["id"],
                                 "content": result})
                self._emit({"type": "tool_result", "name": name, "result": result})

        return "(stopped: reached max turns without a final answer)"

    # ------------------------------------------------------------------ #

    @staticmethod
    def _maybe_compact(messages: list[dict]) -> None:
        """Keep the conversation under COMPACT_THRESHOLD chars.

        Naive but effective: squeeze OLD tool outputs (they're usually the
        bulk) to their head, keeping recent ones intact. Real agents do
        smarter things (summarization, embedding recall, sliding windows) —
        this is the honest v1.
        """
        total = sum(len(str(m.get("content") or "")) for m in messages)
        if total <= COMPACT_THRESHOLD:
            return

        for m in messages[:-6]:  # never touch the most recent messages
            if m.get("role") == "tool" and len(m.get("content") or "") > COMPACT_TOOL_KEEP:
                m["content"] = m["content"][:COMPACT_TOOL_KEEP] + "\n... [compacted]"
