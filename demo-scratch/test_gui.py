"""Test miniagent's GUI server exactly like a browser would:
SSE chat stream + approval-button flow. Run while `miniagent gui` is up."""
import json
import urllib.request

BASE = "http://127.0.0.1:7861"


def post(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=300)


def stream_chat(payload):
    """Yield SSE events from POST /api/chat, answering approvals on the way."""
    res = post("/api/chat", payload)
    buf = b""
    while True:
        chunk = res.read(256)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            frame = frame.decode("utf-8", "replace")
            if not frame.startswith("data: "):
                continue
            ev = json.loads(frame[6:])
            if ev["type"] == "approval":
                print(f"  [browser] approval card shown -> clicking Approve")
                post("/api/approve", {"id": ev["id"], "value": True})
            yield ev


def run_case(label, payload, expect_substrings):
    print(f"\n=== {label} ===")
    ok = set()
    for ev in stream_chat(payload):
        t = ev["type"]
        if t == "text":
            print("  [text]", ev["delta"][:80].replace("\n", " "), end="\n" if ev["delta"].endswith(("\n", " ")) else "")
        elif t in ("start", "done", "error"):
            print(f"  [{t}]", ev.get("brain") or ev.get("message", ""))
        elif t == "tool_result":
            head = ev["result"].splitlines()[0][:100]
            print(f"  [tool_result {ev['name']}] {head}")
            for s in expect_substrings:
                if s.lower() in ev["result"].lower():
                    ok.add(s)
    missing = [s for s in expect_substrings if s not in ok]
    print(f"  -> {'PASS' if not missing else 'FAIL (missing: ' + str(missing) + ')'}")
    return not missing


# Case 1: auto mode, full loop (write + read a file through tools)
ok1 = run_case(
    "auto-mode chat via NVIDIA",
    {"message": "Create gui-test.txt with content 'gui works' then read it back to verify. One short sentence answer.",
     "provider": "nvidia", "model": "nvidia/nemotron-3-super-120b-a12b", "auto": True},
    ["created", "gui works"],
)

# Case 2: approval flow — a bash command needs a button click mid-task
ok2 = run_case(
    "approval flow (confirm mode)",
    {"message": "Run the shell command: echo approval-flow-works — then tell me its stdout.",
     "provider": "nvidia", "model": "nvidia/nemotron-3-super-120b-a12b", "auto": False},
    ["approval-flow-works"],
)

print("\nRESULT:", "ALL PASS" if ok1 and ok2 else "FAILURES PRESENT")
