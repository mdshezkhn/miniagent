"""LLM client — how the agent talks to its brain.

One class, one important method: chat(messages, tools) -> assistant message.

It uses the official `openai` SDK pointed at a different base_url per
provider (see providers.py). The interesting part is STREAMING WITH TOOL
CALLS: text arrives in small chunks (printed live), and tool calls arrive
as function-name/argument fragments that must be stitched back together —
`tool_calls` below is that stitching.
"""

import time

from openai import OpenAI

from .providers import resolve


class LLM:
    def __init__(self, provider=None, model=None, base_url=None, api_key=None,
                 on_text=None):
        """on_text(delta): optional callback receiving streamed text as it
        arrives. When given (GUI mode), nothing is printed to stdout — the
        UI decides how to show it. This is the only hook a GUI needs."""
        self.on_text = on_text
        self.label, base_url, key, model = resolve(provider, model, base_url, api_key)
        if not key:
            raise SystemExit(
                f"No API key for {self.label}. Put it in .env or pass --api-key.\n"
                f"Run 'miniagent providers' to see which keys are configured."
            )
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=key, timeout=120)

    # ------------------------------------------------------------------ #

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """One round-trip. Returns an assistant message as a plain dict:

            {"role": "assistant", "content": "text or None",
             "tool_calls": [{"id": ..., "function": {"name": ..., "arguments": ...}}]}

        (tool_calls absent when the model just talks.)
        Retries transient failures with exponential backoff — real agents
        always do this, because rate limits DO happen mid-task.
        """
        delays = (0, 2, 6)
        for attempt, delay in enumerate(delays):
            if delay:
                time.sleep(delay)
            try:
                return self._chat_once(messages, tools)
            except Exception as e:  # noqa: BLE001 — classify below
                name = type(e).__name__
                transient = any(t in name for t in (
                    "RateLimit", "APIConnection", "APITimeout", "InternalServerError",
                    "ServiceUnavailable", "Overloaded",
                ))
                if transient and attempt < len(delays) - 1:
                    print(f"[llm] transient {name}, retrying...")
                    continue
                raise

    # ------------------------------------------------------------------ #

    def _chat_once(self, messages, tools):
        kwargs = dict(model=self.model, messages=messages, stream=True)
        if tools:
            kwargs["tools"] = tools

        stream = self.client.chat.completions.create(**kwargs)

        text_parts: list[str] = []
        calls: dict[int, dict] = {}  # index -> {"id","name","arguments"} being stitched

        for chunk in stream:
            if not chunk.choices:
                continue  # some providers send a final usage-only chunk
            delta = chunk.choices[0].delta
            if delta is None:
                continue

            # 1) plain text: stream it — to the GUI if attached, else stdout
            if delta.content:
                text_parts.append(delta.content)
                if self.on_text:
                    self.on_text(delta.content)
                else:
                    print(delta.content, end="", flush=True)

            # 2) tool calls arrive as fragments; reassemble by index
            for tc in (delta.tool_calls or []):
                slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                fn = tc.function
                if fn:
                    if fn.name:
                        slot["name"] += fn.name
                    if fn.arguments:
                        slot["arguments"] += fn.arguments

        if text_parts and not self.on_text:
            print()  # newline after streamed text (stdout mode only)

        message: dict = {"role": "assistant",
                         "content": "".join(text_parts) or None}
        if calls:
            message["tool_calls"] = [
                {"id": t["id"] or f"call_{i}", "type": "function",
                 "function": {"name": t["name"], "arguments": t["arguments"]}}
                for i, t in sorted(calls.items())
            ]
        return message
