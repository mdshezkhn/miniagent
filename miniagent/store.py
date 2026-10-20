"""Session persistence — conversations as plain JSON files.

The exact idea production agents use (Links stores one JSON per chat under
data/conversations/<uuid>.json): the message list IS the conversation, so
persisting it is just dumping it to disk after each task and loading the
newest file on start.

Layout:  <cwd>/.miniagent/sessions/<timestamp>-<id>.json
         {"saved_at": ..., "message_count": ..., "messages": [...]}
"""

import json
import time
import uuid
from pathlib import Path


def sessions_dir(base: str | Path | None = None) -> Path:
    d = Path(base or Path.cwd()) / ".miniagent" / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(history: list[dict], path: str | Path | None = None,
         base: str | Path | None = None) -> Path:
    """Write the conversation. Reuses `path` once a session has a file, so a
    long chat keeps updating ONE file instead of spawning one per task.
    (Re-serializing the whole list each save is O(n^2) over a session's life
    — irrelevant at mini-agent scale, and honest about it.)"""
    if path is not None:
        p = Path(path)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        p = sessions_dir(base) / f"{stamp}-{uuid.uuid4().hex[:8]}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "message_count": len(history),
        "messages": history,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def latest(base: str | Path | None = None) -> tuple[list[dict], Path | None]:
    """Load the most recent session. Returns ([], None) if none exist."""
    files = sorted(sessions_dir(base).glob("*.json"),
                   key=lambda f: f.stat().st_mtime)  # mtime, NOT name:
    if not files:                                     # same-second saves tie
        return [], None                               # on the timestamp prefix
    newest = files[-1]
    try:
        data = json.loads(newest.read_text(encoding="utf-8"))
        return list(data.get("messages", [])), newest
    except (json.JSONDecodeError, OSError):
        return [], None  # a corrupt file shouldn't brick the agent


# ---- long-term memory (ChatGPT-style: facts that outlive a session) ---- #

def memory_path(base: str | Path | None = None) -> Path:
    return Path(base or Path.cwd()) / ".miniagent" / "memory.md"


def load_memory(base: str | Path | None = None) -> str:
    try:
        return memory_path(base).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def append_memory(fact: str, base: str | Path | None = None) -> Path:
    p = memory_path(base)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("# miniagent long-term memory\n", encoding="utf-8")
    with p.open("a", encoding="utf-8") as f:
        f.write(f"- {time.strftime('%Y-%m-%d')} {fact}\n")
    return p
