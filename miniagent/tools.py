"""Tools — the agent's hands.

An agent is only as capable as its tools. Note the design choices, they
are the ones Claude Code / OpenHands make too:

  * FEW, ORTHOGONAL TOOLS. Six tools cover most coding work. The model
    picks tools from their JSON-schema descriptions — clear descriptions
    matter more than quantity.
  * RESULTS ARE ALWAYS STRINGS. Including errors ("Error: ..."). The
    agent reads the error and self-corrects on the next turn — that is
    THE core agent superpower, so never raise exceptions past execute().
  * OUTPUT CAPS. LLM context is finite; a tool that dumps 5MB will wreck
    the loop. `clip()` keeps head+tail, the way real agents do.
  * SAFETY GATES. Two layers: a hard blocklist for catastrophic commands,
    and a soft confirm() for anything that mutates state. --auto/--yolo
    skips the soft layer (CI mode), never the hard one.
"""

import fnmatch
import json
import os
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path

MAX_OUTPUT = 20_000   # chars per tool result fed back to the model
CONFIRMABLE_ACTIONS = ("run_bash", "write_file")

# Commands that are refused outright, whatever the model says.
DANGEROUS_PATTERNS = [
    r"rm\s+-rf\s+[/~]",            # rm -rf / or ~
    r"\bmkfs(\.\w+)?\b",           # format filesystem
    r"dd\s+if=.*of=/dev/",         # raw device writes
    r":\(\)\s*\{.*\}\s*;\s*:",     # fork bomb
    r"\bshutdown\b|\breboot\b",    # power actions
    r"\bdel\s+/[sqf]\b",           # windows force-delete
    r"\bformat\s+[a-z]:",          # windows format
    r"git\s+push\s+(-f|--force)\s+origin\s+(main|master)\b",
]
_DANGEROUS = [re.compile(p, re.IGNORECASE) for p in DANGEROUS_PATTERNS]

# Directories never entered while searching.
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build", ".idea"}


class ToolContext:
    """Everything tools need to know about the outside world.

    confirm/ask are injected by the CLI, so tools stay UI-independent —
    swap input() for a GUI dialog later without touching this file.
    """

    def __init__(self, cwd: str, confirm, ask):
        self.cwd = cwd
        self.confirm = confirm      # confirm(prompt: str) -> bool
        self.ask = ask              # ask(question: str) -> str


def clip(text: str, limit: int = MAX_OUTPUT) -> str:
    """Cap tool output, keeping head and tail (the middle matters least)."""
    if len(text) <= limit:
        return text
    head, tail = limit * 3 // 4, limit // 4
    return (text[:head]
            + f"\n... [{len(text) - head - tail} characters truncated] ...\n"
            + text[-tail:])


def _resolve(path: str, ctx: ToolContext) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(ctx.cwd) / p


# ====================================================================== #
#  TOOL SCHEMAS — what the model sees. Descriptions are prompts!         #
# ====================================================================== #

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": ("Execute a shell command and return exit code, stdout and "
                            "stderr. Use for builds, git, package installs, counting, "
                            "one-off analysis. Prefer non-destructive commands; risky "
                            "commands require user approval."),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command line to run"},
                    "timeout": {"type": "integer", "description": "Seconds to wait (default 60, max 600)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": ("Read a text file, returning lines numbered like '12->content'. "
                            "Use offset/limit for large files."),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, relative to working dir or absolute"},
                    "offset": {"type": "integer", "description": "1-based line to start from"},
                    "limit": {"type": "integer", "description": "Max lines to return (default 2000)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": ("Create or overwrite a text file. Parent directories are "
                            "created automatically. Overwriting an existing file "
                            "requires user approval."),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string", "description": "Full file content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List one directory (non-recursive): names, type, sizes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default '.')"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": ("Search file contents with a regular expression, recursively. "
                            "Skips .git/node_modules/etc. Returns file:line:match, max 200 hits. "
                            "Example pattern: 'def\\s+main'"),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Python regular expression"},
                    "path": {"type": "string", "description": "Directory to search (default '.')"},
                    "glob": {"type": "string", "description": "Filename filter, e.g. '*.py' (default '*')"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": ("Download a web page and return readable text (docs, APIs, "
                            "articles). HTML is stripped to text; JSON/XML/text return "
                            "raw; binary content reports metadata only. Output is capped."),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL, http:// or https://"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": ("Search the live web (DuckDuckGo) and get top results as "
                            "numbered title + URL + snippet. Use for current events, "
                            "library docs, error messages; follow up with web_fetch on "
                            "the best hits. Cite the URLs you actually used."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Max results (default 6)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": ("Save a durable fact to long-term memory "
                            "(.miniagent/memory.md), loaded in every future session. "
                            "Use for stable user preferences, project conventions and "
                            "hard-won gotchas — NOT per-task details."),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "One self-contained sentence"},
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": ("Ask the human a question when only they can decide: "
                            "ambiguity, missing parameter, or before a risky action. "
                            "Returns their answer as text."),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                },
                "required": ["question"],
            },
        },
    },
]

_TOOL_NAMES = {t["function"]["name"] for t in TOOL_SCHEMAS}


# ====================================================================== #
#  DISPATCH — the single entry point the agent loop calls                #
# ====================================================================== #

def execute(name: str, arguments_json: str, ctx: ToolContext) -> str:
    """Run one tool call. ALWAYS returns a string (errors included) so the
    model can read them and adapt — never let an exception escape."""
    try:
        args = json.loads(arguments_json or "{}")
    except json.JSONDecodeError as e:
        return f"Error: arguments were not valid JSON ({e})."

    if name not in _TOOL_NAMES:
        return f"Error: unknown tool '{name}'. Available: {', '.join(sorted(_TOOL_NAMES))}"

    try:
        return clip(_HANDLERS[name](args, ctx))
    except Exception as e:  # noqa: BLE001 — tools must never crash the loop
        return f"Error: {type(e).__name__}: {e}"


# ====================================================================== #
#  IMPLEMENTATIONS                                                       #
# ====================================================================== #

def _run_bash(args: dict, ctx: ToolContext) -> str:
    command = args["command"]
    timeout = min(int(args.get("timeout") or 60), 600)

    if any(p.search(command) for p in _DANGEROUS):
        return ("Error: command matched the hard safety blocklist and was refused. "
                "If you truly need it, the user can run it themselves.")

    if not ctx.confirm(f"  run command: {command}"):
        return "Refused by user. Ask what to do differently, or continue another way."

    try:
        proc = subprocess.run(
            command, shell=True, cwd=ctx.cwd,
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command exceeded {timeout}s timeout (not killed; avoid long-running processes)."

    out = [f"exit code: {proc.returncode}"]
    if proc.stdout:
        out.append("--- stdout ---\n" + proc.stdout.rstrip())
    if proc.stderr:
        out.append("--- stderr ---\n" + proc.stderr.rstrip())
    return "\n".join(out)


def _read_file(args: dict, ctx: ToolContext) -> str:
    path = _resolve(args["path"], ctx)
    if not path.is_file():
        return f"Error: no such file: {path}"
    offset = max(int(args.get("offset") or 1), 1)
    limit = int(args.get("limit") or 2000)

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    window = lines[offset - 1: offset - 1 + limit]
    numbered = "\n".join(f"{offset + i}->{line}" for i, line in enumerate(window))
    header = f"{path} (lines {offset}-{offset + len(window) - 1} of {len(lines)})\n"
    return header + (numbered or "(empty range)")


def _write_file(args: dict, ctx: ToolContext) -> str:
    path = _resolve(args["path"], ctx)
    existed = path.exists()

    if existed and not ctx.confirm(f"  overwrite file: {path}"):
        return "Refused by user. Propose a different path or ask how to proceed."

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args["content"], encoding="utf-8")
    verb = "overwrote" if existed else "created"
    return f"{verb} {path} ({len(args['content'])} chars)"


def _list_dir(args: dict, ctx: ToolContext) -> str:
    path = _resolve(args.get("path") or ".", ctx)
    if not path.is_dir():
        return f"Error: no such directory: {path}"

    rows = []
    for entry in sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name.lower())):
        if entry.is_dir():
            rows.append(f"{entry.name}/")
        else:
            rows.append(f"{entry.name} ({entry.stat().st_size:,} B)")
    return "\n".join(rows) or "(empty directory)"


def _search_files(args: dict, ctx: ToolContext) -> str:
    try:
        rx = re.compile(args["pattern"])
    except re.error as e:
        return f"Error: invalid regex: {e}"

    root = _resolve(args.get("path") or ".", ctx)
    glob = args.get("glob") or "*"
    hits: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in fnmatch.filter(filenames, glob):
            fpath = Path(dirpath) / fname
            if fpath.stat().st_size > 2_000_000:
                continue  # skip huge/binary-ish files
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{fpath}:{lineno}:{line.strip()[:300]}")
                    if len(hits) >= 200:
                        return "\n".join(hits) + "\n... (hit cap of 200 reached; narrow the pattern)"
    return "\n".join(hits) or f"no matches for /{args['pattern']}/ under {root}"


def _ask_user(args: dict, ctx: ToolContext) -> str:
    answer = ctx.ask(args["question"])
    return f"user answered: {answer}" if answer else "(user gave no answer)"


# ---- web_fetch --------------------------------------------------------- #
# Uses httpx — already in the venv because the openai SDK depends on it,
# so we add ZERO new dependencies. It handles gzip and redirects, which
# urllib makes painful.
# NOTE (SSRF): a deployed agent should block private/localhost URLs; for a
# local personal agent fetching your own dev servers is a feature, so we
# allow it. Read-only tool => no approval needed (same policy as read_file).

class _TextExtract(HTMLParser):
    """stdlib HTML -> text: keeps visible words, drops script/style/svg/head."""
    SKIP = {"script", "style", "noscript", "svg", "head", "template"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def _html_to_text(html: str) -> str:
    p = _TextExtract()
    p.feed(html)
    return "\n".join(p.parts)


def _web_fetch(args: dict, ctx: ToolContext) -> str:
    import httpx  # local import: keeps module import cheap for CLI users

    url = args["url"]
    if not re.match(r"^https?://", url):
        return "Error: url must start with http:// or https://"

    # THE PROXY GOTCHA (bit us in testing): httpx reads HTTP_PROXY/HTTPS_PROXY
    # env vars by default, so a local dev server gets fetched *through* the
    # proxy and dies. Correct behavior = what NO_PROXY does: never proxy
    # local/private hosts, proxy everything else.
    from urllib.parse import urlparse
    host = (urlparse(url).hostname or "").lower()
    is_local = (host in ("localhost", "127.0.0.1", "::1")
                or host.startswith(("192.168.", "10.", "169.254.")))

    try:
        r = httpx.get(url, follow_redirects=True, timeout=20,
                      trust_env=not is_local,   # ignore proxy env for local
                      headers={"User-Agent": "Mozilla/5.0 (miniagent/0.3; +local agent)"})
    except httpx.HTTPError as e:
        return (f"Error: fetch failed ({type(e).__name__}: {e}). "
                "The site may be unreachable from this network.")

    ctype = r.headers.get("content-type", "").lower()
    if "html" in ctype:
        body, kind = _html_to_text(r.text), "readable text"
    elif any(t in ctype for t in ("json", "text/", "xml", "javascript")):
        body, kind = r.text, "raw"
    else:
        return (f"{url} -> {r.status_code} {ctype} — binary/undisplayable "
                f"({len(r.content):,} bytes); download it with run_bash instead.")

    header = f"{url} -> {r.status_code} {ctype} ({kind}, {len(body):,} chars)"
    return header + "\n\n" + body


# ---- web_search ------------------------------------------------------- #
# The Perplexity feature: live retrieval before synthesis. No API key —
# DuckDuckGo's HTML endpoints. Goes THROUGH the system proxy (external),
# unlike web_fetch's local bypass. Fragile by nature (scraping); a real
# deployment swaps this for a search API (Brave/Tavily/SearXNG) — same shape.

_SEARCH_URL = "https://html.duckduckgo.com/html/"     # patched in selftest
_SEARCH_URL_FALLBACK = "https://lite.duckduckgo.com/lite/"


def _strip_tags(s: str) -> str:
    from html import unescape
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\s+", " ", unescape(s)).strip()


def _parse_search(page: str) -> list[tuple[str, str, str]]:
    """Extract (title, url, snippet) triples from a DDG results page."""
    from urllib.parse import unquote
    pairs = []
    for m in re.finditer(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                         page, re.S):
        url, title = m.group(1), _strip_tags(m.group(2))
        um = re.search(r"[?&]uddg=([^&]+)", url)   # DDG redirect: /l/?uddg=<enc>
        if um:
            url = unquote(um.group(1))
        pairs.append((title, url))
    snips = [_strip_tags(s) for s in re.findall(
        r'class="result__snippet"[^>]*>(.*?)</a>', page, re.S)]
    return [(t, u, snips[i] if i < len(snips) else "")
            for i, (t, u) in enumerate(pairs)]


def _web_search(args: dict, ctx: ToolContext) -> str:
    import httpx

    query = args["query"]
    want = min(int(args.get("max_results") or 6), 10)

    last = ""
    for endpoint in (_SEARCH_URL, _SEARCH_URL_FALLBACK):
        try:
            r = httpx.get(endpoint, params={"q": query}, timeout=20,
                          follow_redirects=True,
                          headers={"User-Agent": "Mozilla/5.0 (miniagent/0.4)"})
        except httpx.HTTPError as e:
            last = f"{type(e).__name__}: {e}"
            continue
        if r.status_code != 200:
            last = f"HTTP {r.status_code}"
            continue
        hits = _parse_search(r.text)[:want]
        if hits:
            lines = [f"web results for '{query}' ({len(hits)})"]
            for i, (title, url, snip) in enumerate(hits, 1):
                lines.append(f"[{i}] {title}\n    {url}" + (f"\n    {snip}" if snip else ""))
            return "\n".join(lines)
        if "anomaly" in r.text.lower() or "botnet" in r.text.lower():
            last = "DDG flagged the client as a bot"
            continue
        last = "no results parsed"

    return (f"Error: web search failed ({last}). Retry once, then answer from "
            "your own knowledge and say it may be outdated.")


# ---- remember (ChatGPT-style persistent memory) ------------------------ #

def _remember(args: dict, ctx: ToolContext) -> str:
    from . import store
    fact = args["fact"].strip()
    if not fact:
        return "Error: empty fact."
    path = store.append_memory(fact, base=ctx.cwd)  # memory lives with the project
    return f"remembered to {path} (will load in future sessions): {fact}"


_HANDLERS = {
    "run_bash": _run_bash,
    "read_file": _read_file,
    "write_file": _write_file,
    "list_dir": _list_dir,
    "search_files": _search_files,
    "web_fetch": _web_fetch,
    "web_search": _web_search,
    "remember": _remember,
    "ask_user": _ask_user,
}
