"""GUI — a local web interface for the agent (ChatGPT-style, in your browser).

    miniagent gui                 # -> http://127.0.0.1:7860

Why a web GUI and not Tkinter/PyQt? Because it's what real agents do
(OpenHands, Open WebUI), it needs no build step, and — the architecture
lesson — the agent loop did not change ONE line for this. The GUI is just
three adapters bolted onto existing hooks:

    llm.on_text      -> SSE "text" events     (was: print)
    agent.on_event   -> SSE tool events       (was: print)
    ToolContext.confirm/ask -> Approve/Deny buttons & input boxes in the
                               browser (was: input() in the terminal)

How the tricky parts work (all standard patterns, worth studying):

  * SSE (Server-Sent Events): one long-lived HTTP response where the server
    pushes "data: {...}\n\n" lines; the browser reads them live. Simpler
    than WebSockets and all we need (one-directional push).
  * The agent loop is synchronous, so it runs in a worker THREAD; events
    flow to the HTTP response through a queue.Queue.
  * Approvals cross threads: confirm() in the worker thread emits an
    "approval" event and blocks on a threading.Event until the browser
    POSTs /api/approve — the human clicks a button mid-task, the loop
    unblocks. This is exactly how Claude Code's permission prompts work.
"""

import json
import os
import queue
import threading
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from . import providers, store
from .agent import Agent
from .config import load_env
from .llm import LLM

app = FastAPI(title="miniagent")

# ---------------------------------------------------------------------- #
#  Session state — single local user, one conversation at a time          #
# ---------------------------------------------------------------------- #

class Session:
    def __init__(self):
        self.history: list[dict] = []          # the live conversation
        self.busy = threading.Lock()           # one task at a time
        self.events: queue.Queue | None = None # worker -> browser pipe
        self.pending: dict[str, dict] = {}     # approval ids awaiting clicks
        self.path = None                       # session JSON file on disk


session = Session()


def _push(event: dict) -> None:
    """Send an event to the browser (no-op if no stream is attached)."""
    if session.events is not None:
        session.events.put(event)


def _confirm(prompt: str) -> bool:
    """ToolContext.confirm — blocks the agent until the browser answers."""
    pid = uuid.uuid4().hex[:12]
    slot = {"done": threading.Event(), "value": None}
    session.pending[pid] = slot
    _push({"type": "approval", "id": pid, "prompt": prompt})
    answered = slot["done"].wait(timeout=300)  # 5 min to click
    session.pending.pop(pid, None)
    return bool(answered and slot["value"])


def _ask(question: str) -> str:
    """ToolContext.ask — same dance, but the answer is free text."""
    pid = uuid.uuid4().hex[:12]
    slot = {"done": threading.Event(), "value": ""}
    session.pending[pid] = slot
    _push({"type": "ask", "id": pid, "question": question})
    answered = slot["done"].wait(timeout=600)
    session.pending.pop(pid, None)
    return slot["value"] if answered else ""


# ---------------------------------------------------------------------- #
#  Endpoints                                                              #
# ---------------------------------------------------------------------- #

@app.get("/api/providers")
async def api_providers():
    return [{"id": pid, "name": name, "model": model, "key": has_key}
            for pid, name, model, has_key in providers.status_table()]


@app.post("/api/new")
async def api_new():
    if session.busy.locked():
        return JSONResponse({"error": "agent is busy"}, status_code=409)
    session.history.clear()
    session.path = None      # next save starts a fresh file
    return {"ok": True}


@app.get("/api/history")
async def api_history():
    """Replay a saved conversation as the same event shapes the live stream
    uses — so the browser re-renders last session with ONE code path.
    (The message list is the source of truth; the UI is derived from it.)"""
    events: list[dict] = []
    names: dict[str, str] = {}                 # tool_call_id -> tool name
    for m in session.history:
        role = m.get("role")
        if role == "user":
            events.append({"type": "user", "text": m.get("content") or ""})
        elif role == "assistant":
            if m.get("content"):
                events.append({"type": "text", "delta": m["content"]})
            for tc in m.get("tool_calls") or []:
                names[tc["id"]] = tc["function"]["name"]
                events.append({"type": "tool_start",
                               "name": tc["function"]["name"],
                               "args": tc["function"]["arguments"]})
        elif role == "tool":
            events.append({"type": "tool_result",
                           "name": names.get(m.get("tool_call_id"), "tool"),
                           "result": m.get("content") or ""})
    return events


@app.post("/api/approve")
async def api_approve(req: Request):
    """Browser answers an approval / ask card. value: true|false|<text>."""
    body = await req.json()
    slot = session.pending.get(body.get("id", ""))
    if not slot:
        return JSONResponse({"error": "unknown or expired id"}, status_code=404)
    slot["value"] = body.get("value")
    slot["done"].set()
    return {"ok": True}


@app.post("/api/chat")
async def api_chat(req: Request):
    body = await req.json()
    message = (body.get("message") or "").strip()
    if not message:
        return JSONResponse({"error": "empty message"}, status_code=400)
    if not session.busy.acquire(blocking=False):
        return JSONResponse({"error": "agent is busy — wait for the current task"},
                            status_code=409)

    q: queue.Queue = queue.Queue()
    session.events = q

    def worker():
        try:
            llm = LLM(provider=body.get("provider") or None,
                      model=body.get("model") or None,
                      on_text=lambda d: _push({"type": "text", "delta": d}))
            agent = Agent(llm, cwd=os.getcwd(), confirm=_confirm, ask=_ask,
                          auto=bool(body.get("auto")),
                          on_event=lambda e: _push(e))
            _push({"type": "start", "brain": f"{llm.label} / {llm.model}"})
            agent.run(message, history=session.history)
        except SystemExit as e:                       # e.g. missing API key
            _push({"type": "error", "message": str(e)})
        except Exception as e:  # noqa: BLE001 — surface anything to the UI
            _push({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            session.events = None
            try:
                session.path = store.save(session.history, session.path)
            except OSError as e:               # disk full, permissions, ...
                _push({"type": "error", "message": f"session not saved: {e}"})
            session.busy.release()
            q.put({"type": "done"})

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            try:
                item = q.get(timeout=1.0)
            except queue.Empty:
                yield ": keep-alive\n\n"      # SSE comment = harmless heartbeat
                continue
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            if item.get("type") == "done":
                break

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------- #
#  The page — one file, no build step, vanilla JS                         #
# ---------------------------------------------------------------------- #

@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE


# ---- PWA: install "MoGo" from Chrome (Add to Home screen / Install) ---- #
# Chrome requires: manifest + icons + service worker + secure context
# (HTTPS tunnel, or http://127.0.0.1 in Termux on the phone itself).

ICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512">
<rect width="512" height="512" rx="96" fill="#0f1115"/>
<rect x="28" y="28" width="456" height="456" rx="72" fill="#171a21"
      stroke="#4f8cff" stroke-width="8"/>
<text x="256" y="352" font-family="Arial, sans-serif" font-size="280"
      font-weight="bold" fill="#4f8cff" text-anchor="middle">M</text>
</svg>"""


@app.get("/icon.svg")
async def icon():
    from fastapi.responses import Response
    return Response(ICON_SVG, media_type="image/svg+xml")


@app.get("/manifest.json")
async def manifest():
    return {
        "name": "MoGa — miniagent", "short_name": "MoGa",
        "start_url": "/", "display": "standalone",
        "background_color": "#0f1115", "theme_color": "#0f1115",
        "description": "Your personal coding agent, everywhere — voice in, voice out.",
        "icons": [
            {"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml",
             "purpose": "any maskable"},
        ],
    }


SW_JS = """// MoGa service worker: offline shell + never cache the API/SSE stream.
const SHELL = ['/', '/icon.svg', '/manifest.json'];
self.addEventListener('install', e => {
  e.waitUntil(caches.open('mogo-v1').then(c => c.addAll(SHELL)));
  self.skipWaiting();
});
self.addEventListener('activate', e => self.clients.claim());
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (url.pathname.startsWith('/api/')) return;            // always live
  e.respondWith(
    caches.match(e.request).then(hit => hit || fetch(e.request))
  );
});
"""


@app.get("/sw.js")
async def sw():
    from fastapi.responses import Response
    return Response(SW_JS, media_type="application/javascript")


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MoGa</title>
<link rel="manifest" href="/manifest.json">
<meta name="theme-color" content="#0f1115">
<link rel="icon" href="/icon.svg" type="image/svg+xml">
<script>
if ('serviceWorker' in navigator)   // installable as "MoGa" on Android/iOS
  navigator.serviceWorker.register('/sw.js').catch(()=>{});
</script>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --line:#262b36; --text:#e6e9ef;
          --dim:#8b93a5; --accent:#4f8cff; --green:#3fb970; --red:#e0555f; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.5 system-ui, "Segoe UI", sans-serif; height:100vh;
         display:flex; flex-direction:column; }
  header { display:flex; gap:10px; align-items:center; padding:10px 16px;
           background:var(--panel); border-bottom:1px solid var(--line); }
  header h1 { font-size:15px; margin:0 12px 0 0; color:var(--accent); }
  header select, header input, header label {
         background:var(--bg); color:var(--text); border:1px solid var(--line);
         border-radius:6px; padding:5px 8px; font-size:12.5px; }
  header input#model { width:210px; }
  header .grow { flex:1; }
  header button { background:transparent; color:var(--dim); border:1px solid var(--line);
         border-radius:6px; padding:5px 10px; cursor:pointer; font-size:12.5px; }
  header button:hover { color:var(--text); }
  #chat { flex:1; overflow-y:auto; padding:20px; }
  .msg { max-width:860px; margin:0 auto 14px; }
  .user .bubble { background:#1d2b45; border-radius:10px; padding:10px 14px;
                  white-space:pre-wrap; display:inline-block; }
  .assistant .who { color:var(--accent); font-size:11px; letter-spacing:.4px;
                    text-transform:uppercase; margin-bottom:4px; }
  .assistant .bubble { white-space:pre-wrap; }
  .tool { max-width:860px; margin:6px auto; background:var(--panel);
          border:1px solid var(--line); border-left:3px solid var(--dim);
          border-radius:8px; overflow:hidden; }
  .tool.ok  { border-left-color:var(--green); }
  .tool.err { border-left-color:var(--red); }
  .tool summary { cursor:pointer; padding:7px 12px; font:12px ui-monospace,Consolas,monospace;
                  color:var(--dim); display:flex; gap:8px; align-items:center; }
  .tool summary b { color:var(--text); }
  .tool pre { margin:0; padding:10px 12px; border-top:1px solid var(--line);
              font:11.5px/1.45 ui-monospace,Consolas,monospace; color:var(--dim);
              white-space:pre-wrap; word-break:break-word; max-height:260px; overflow-y:auto; }
  .card { max-width:860px; margin:10px auto; background:var(--panel);
          border:1px solid var(--accent); border-radius:8px; padding:12px 14px; }
  .card .q { margin-bottom:10px; white-space:pre-wrap; }
  .card button { border:0; border-radius:6px; padding:6px 16px; cursor:pointer;
                 font-size:13px; margin-right:8px; }
  .card .yes { background:var(--green); color:#08130c; }
  .card .no  { background:var(--red); color:#fff; }
  .card input[type=text] { flex:1; background:var(--bg); color:var(--text);
         border:1px solid var(--line); border-radius:6px; padding:6px 10px; }
  .card .row { display:flex; gap:8px; }
  .err { max-width:860px; margin:8px auto; color:var(--red);
         font:12.5px ui-monospace,Consolas,monospace; white-space:pre-wrap; }
  #inputbar { display:flex; gap:10px; padding:14px 20px; background:var(--panel);
              border-top:1px solid var(--line); }
  #inputbar textarea { flex:1; resize:none; background:var(--bg); color:var(--text);
              border:1px solid var(--line); border-radius:8px; padding:10px 12px;
              font:14px/1.4 inherit; height:52px; }
  #send { background:var(--accent); color:#fff; border:0; border-radius:8px;
          padding:0 22px; cursor:pointer; font-size:14px; }
  #send:disabled { opacity:.45; cursor:default; }
  #mic { background:var(--panel); color:var(--text); border:1px solid var(--line);
         border-radius:8px; font-size:18px; cursor:pointer; padding:0 12px; }
  .acts { margin-top:8px; display:flex; gap:8px; }
  .acts button { background:var(--panel); color:var(--dim); border:1px solid var(--line);
         border-radius:6px; padding:3px 10px; font-size:12px; cursor:pointer; }
  .acts button:hover { color:var(--text); }
  .status { max-width:860px; margin:0 auto; color:var(--dim); font-size:12px;
            padding:2px 0; min-height:18px; }
  /* phone-size screens (agent accessed from a phone on the LAN) */
  @media (max-width: 640px) {
    header { flex-wrap: wrap; padding: 8px 10px; }
    header input#model { width: 140px; flex: 1; }
    #chat { padding: 12px 10px; }
    .msg, .tool, .card, .err { max-width: 100%; }
    #inputbar { padding: 10px; }
    #inputbar textarea { height: 64px; font-size: 16px; }  /* stops iOS zoom */
  }
</style>
</head>
<body>
<header>
  <h1>miniagent</h1>
  <select id="provider" title="brain"></select>
  <input id="model" placeholder="model (default)">
  <label title="skip approval prompts"><input type="checkbox" id="auto"> auto</label>
  <select id="tlang" title="translate target language">
    <option value="English">EN</option><option value="Chinese (Simplified)">中文</option>
    <option value="Hindi">हिं</option><option value="Arabic">عربي</option>
    <option value="Spanish">ES</option>
  </select>
  <div class="grow"></div>
  <button id="new">+ new chat</button>
</header>
<div id="chat"></div>
<div class="status" id="status"></div>
<div id="inputbar">
  <button id="mic" title="voice input (tap to dictate)">🎙</button>
  <textarea id="msg" placeholder="Give the agent a task...  (Enter to send, Shift+Enter for newline)"></textarea>
  <button id="send">Send</button>
</div>
<script>
const chat = document.getElementById('chat');
const statusEl = document.getElementById('status');
const sendBtn = document.getElementById('send');
let currentText = null;     // open text div inside current assistant bubble
let currentBubble = null;   // current assistant bubble container

/* ---------- rendering ---------- */
function scroll() { chat.scrollTop = chat.scrollHeight; }

function addUser(text) {
  const m = el('div', 'msg user'); const b = el('div', 'bubble');
  b.textContent = text; m.appendChild(b); chat.appendChild(m);
  currentText = null; currentBubble = null; scroll();
}

function ensureAssistant() {   // create the bubble lazily, once per turn
  if (!currentBubble) {
    const m = el('div', 'msg assistant');
    m.appendChild(el('div', 'who', 'agent'));
    currentBubble = el('div', 'bubble'); m.appendChild(currentBubble);
    chat.appendChild(m);
  }
  return currentBubble;
}

function addTextDelta(d) {
  const bubble = ensureAssistant();
  if (!currentText) { currentText = el('div'); bubble.appendChild(currentText); }
  currentText.textContent += d; scroll();
}

function addTool(name, args) {
  currentText = null;                     // text after this starts a new block
  const t = el('details', 'tool'); t.id = 't' + Math.random().toString(36).slice(2);
  const s = el('summary'); s.innerHTML = '<b>' + name + '</b> <span class="args"></span>';
  s.querySelector('.args').textContent = args.length > 120 ? args.slice(0,117)+'...' : args;
  t.appendChild(s); chat.appendChild(t); scroll();
  return t;
}

function fillTool(t, result) {
  if (result.startsWith('Error')) t.classList.add('err');
  else if (result.startsWith('Refused')) t.classList.add('err');
  else t.classList.add('ok');
  const pre = el('pre'); pre.textContent = result.length > 4000
      ? result.slice(0,4000) + '\\n... (truncated)' : result;
  t.appendChild(pre); scroll();
}

function addApproval(ev) {
  currentText = null;
  const c = el('div', 'card');
  c.appendChild(el('div', 'q', 'Approval needed: ' + ev.prompt));
  const row = el('div', 'row');
  const yes = el('button', 'yes', 'Approve'), no = el('button', 'no', 'Deny');
  row.append(yes, no); c.appendChild(row); chat.appendChild(c); scroll();
  const answer = v => { yes.disabled = no.disabled = true;
                        c.style.opacity = .55;
                        fetch('/api/approve', {method:'POST',
                          headers:{'Content-Type':'application/json'},
                          body: JSON.stringify({id: ev.id, value: v})}); };
  yes.onclick = () => answer(true); no.onclick = () => answer(false);
}

function addAsk(ev) {
  currentText = null;
  const c = el('div', 'card');
  c.appendChild(el('div', 'q', 'Agent asks: ' + ev.question));
  const row = el('div', 'row');
  const inp = document.createElement('input'); inp.type = 'text';
  const ok = el('button', 'yes', 'Answer');
  row.append(inp, ok); c.appendChild(row); chat.appendChild(c); scroll();
  inp.focus();
  const answer = () => { ok.disabled = true; c.style.opacity = .55;
    fetch('/api/approve', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({id: ev.id, value: inp.value})}); };
  ok.onclick = answer; inp.onkeydown = e => { if (e.key === 'Enter') answer(); };
}

/* ---------- hands-free: voice input (STT), read-aloud + translate (TTS) ---------- */
/* All via the browser's built-in Web Speech API — no server deps, works in
   Chrome on Android. Translation uses the agent's own brain as translator. */
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null, listening = false;
const micBtn = document.getElementById('mic');
micBtn.onclick = () => {
  if (!SR) { alert('Speech recognition is not supported in this browser.'); return; }
  if (listening) { rec.stop(); return; }
  rec = new SR();
  rec.lang = document.getElementById('tlang').value.startsWith('Chinese') ? 'zh-CN'
           : document.getElementById('tlang').value === 'Hindi' ? 'hi-IN'
           : document.getElementById('tlang').value === 'Arabic' ? 'ar-SA'
           : document.getElementById('tlang').value === 'Spanish' ? 'es-ES' : 'en-US';
  rec.interimResults = true;
  rec.onresult = e => {
    let t = '';
    for (const r of e.results) t += r[0].transcript;
    document.getElementById('msg').value = t;
  };
  rec.onend = () => { listening = false; micBtn.textContent = '🎙'; };
  rec.onerror = () => { listening = false; micBtn.textContent = '🎙'; };
  rec.start(); listening = true; micBtn.textContent = '⏺';
};

function addActionRow(bubble) {           // appears under each finished answer
  const row = el('div', 'acts');
  const speak = el('button', '', '🔊 read');
  const tr = el('button', '', '🌐 translate');
  speak.onclick = () => {
    const u = new SpeechSynthesisUtterance(bubble.innerText);
    speechSynthesis.cancel();
    speechSynthesis.speak(u);            // browser TTS reads the answer aloud
  };
  tr.onclick = () => {
    const lang = document.getElementById('tlang').value;
    document.getElementById('msg').value =
      'Translate your previous answer into ' + lang + '. Reply with ONLY the translation.';
    send();                              // agent translates via its own LLM
  };
  row.append(speak, tr);
  bubble.appendChild(row);
}

function el(clsOrTag, cls, text) {
  if (cls !== undefined && text !== undefined) {           // el(tag, cls, text)
    const e = document.createElement(clsOrTag); e.className = cls;
    e.textContent = text; return e;
  }
  if (cls !== undefined) {                                  // el(tag, cls)
    const e = document.createElement(clsOrTag); e.className = cls; return e;
  }
  return document.createElement(clsOrTag);                  // el(tag)
}

/* ---------- SSE over fetch (POST + stream, EventSource can't POST) ---------- */
async function send() {
  const msg = document.getElementById('msg').value.trim();
  if (!msg) return;
  document.getElementById('msg').value = '';
  addUser(msg);
  sendBtn.disabled = true; statusEl.textContent = 'thinking...';
  let openTool = null;
  try {
    const res = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        message: msg,
        provider: document.getElementById('provider').value,
        model: document.getElementById('model').value.trim(),
        auto: document.getElementById('auto').checked,
      }),
    });
    if (!res.ok) { const e = await res.json(); throw new Error(e.error || res.status); }
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = '';
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let i;
      while ((i = buf.indexOf('\\n\\n')) >= 0) {
        const frame = buf.slice(0, i); buf = buf.slice(i + 2);
        if (!frame.startsWith('data: ')) continue;         // heartbeats etc.
        const ev = JSON.parse(frame.slice(6));
        if (ev.type === 'start')  statusEl.textContent = 'brain: ' + ev.brain;
        if (ev.type === 'text')   addTextDelta(ev.delta);
        if (ev.type === 'tool_start') { statusEl.textContent = 'running ' + ev.name + '...';
                                        openTool = addTool(ev.name, ev.args); }
        if (ev.type === 'tool_result') { if (openTool) fillTool(openTool, ev.result); }
        if (ev.type === 'approval') addApproval(ev);
        if (ev.type === 'ask')      addAsk(ev);
        if (ev.type === 'error')    chat.appendChild(el('div', 'err', ev.message));
        if (ev.type === 'done')     { statusEl.textContent = '';
                                      if (currentBubble) addActionRow(currentBubble); }
      }
    }
  } catch (e) {
    chat.appendChild(el('div', 'err', String(e)));
    statusEl.textContent = '';
  } finally {
    sendBtn.disabled = false;
    document.getElementById('msg').focus();
  }
}

document.getElementById('send').onclick = send;
document.getElementById('msg').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
document.getElementById('new').onclick = async () => {
  await fetch('/api/new', {method: 'POST'}); chat.innerHTML = ''; statusEl.textContent = '';
};

/* ---------- provider dropdown ---------- */
fetch('/api/providers').then(r => r.json()).then(list => {
  const sel = document.getElementById('provider');
  list.forEach(p => {
    const o = document.createElement('option');
    o.value = p.id; o.textContent = p.id + (p.key ? ' *' : '');
    sel.appendChild(o);
  });
  sel.onchange = () => {
    const p = list.find(x => x.id === sel.value);
    document.getElementById('model').value = '';
    document.getElementById('model').placeholder = 'default: ' + p.model;
  };
  sel.onchange();
});

/* ---------- history replay: last session re-rendered on page load ---------- */
async function loadHistory() {
  const evs = await (await fetch('/api/history')).json();
  let openTool = null;
  for (const ev of evs) {
    if (ev.type === 'user') addUser(ev.text);
    else if (ev.type === 'text') addTextDelta(ev.delta);
    else if (ev.type === 'tool_start') openTool = addTool(ev.name, ev.args);
    else if (ev.type === 'tool_result') { if (openTool) fillTool(openTool, ev.result); openTool = null; }
  }
  if (evs.length) scroll();
}
loadHistory();
</script>
</body>
</html>
"""


def _install_basic_auth(token: str) -> None:
    """One shared password (HTTP Basic). Browser asks once, then remembers.
    Minimal but honest protection when exposing the GUI beyond localhost —
    this agent can read/write files and run commands, so a public URL
    MUST NOT be naked."""
    import base64
    import secrets as _secrets

    @app.middleware("http")
    async def _auth(request, call_next):
        header = request.headers.get("authorization", "")
        given = ""
        try:
            scheme, cred = header.split(None, 1)
            if scheme.lower() == "basic":
                given = base64.b64decode(cred).decode("utf-8", "replace").split(":", 1)[-1]
        except Exception:
            given = ""
        if not _secrets.compare_digest(given, token):
            from fastapi.responses import JSONResponse
            return JSONResponse({"error": "unauthorized"}, status_code=401,
                                headers={"WWW-Authenticate": 'Basic realm="miniagent"'})
        return await call_next(request)

    print("auth: ON (password = your --token; the browser will ask once)")


def serve(host: str = "127.0.0.1", port: int = 7860, token: str | None = None) -> None:
    load_env()
    # Resume the newest saved conversation (store.py) — survives restarts.
    session.history, session.path = store.latest()
    if session.history:
        print(f"(resumed session: {len(session.history)} messages from {session.path.name})")
    if token or host not in ("127.0.0.1", "localhost"):
        _install_basic_auth(token or "miniagent")  # non-localhost without token: safe default
    print(f"miniagent GUI -> http://{host}:{port}   (Ctrl+C to stop)")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    serve()
