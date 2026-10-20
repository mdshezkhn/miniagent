"""Provider registry — the multi-brain trick.

All of these providers expose an OpenAI-compatible /chat/completions API,
so ONE client class (llm.py) works with every one of them. You just swap
base_url + api_key + model name. This is exactly how Claude Code and
opencode support many models with one codebase.

Adding a provider = one entry in PROVIDERS. Nothing else changes.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    name: str        # human-readable label
    base_url: str    # OpenAI-compatible endpoint root
    env_key: str     # environment variable holding the API key
    default_model: str  # sensible default with good tool-calling support
    notes: str = ""


PROVIDERS: dict[str, Provider] = {
    "bigmodel": Provider(
        name="BigModel (Zhipu GLM)",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        env_key="BIGMODEL_API_KEY",
        default_model="glm-4.6",
        notes="glm-4-flash is free if you want zero-cost testing",
    ),
    "openrouter": Provider(
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        env_key="OPENROUTER_API_KEY",
        default_model="deepseek/deepseek-chat",
        notes="one key -> hundreds of models; browse openrouter.ai/models",
    ),
    "nvidia": Provider(
        name="NVIDIA NIM",
        base_url="https://integrate.api.nvidia.com/v1",
        env_key="NVIDIA_API_KEY",
        default_model="nvidia/nemotron-3.5-lightning-30b-a3b",
        notes="key starts with nvapi- (build.nvidia.com); WARNING: model "
              "entitlements ROTATE — a model that works today may 404 tomorrow "
              "('Function not found for account'). On 404 re-probe GET /v1/models; "
              "known-good alternate: nvidia/nemotron-3-ultra-550b-a55b",
    ),
    "gemini": Provider(
        name="Google Gemini (OpenAI-compat endpoint)",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        env_key="GEMINI_API_KEY",
        default_model="gemini-2.5-flash",
        notes="generous free tier at aistudio.google.com",
    ),
    "groq": Provider(
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        env_key="GROQ_API_KEY",
        default_model="llama-3.3-70b-versatile",
        notes="extremely fast inference",
    ),
    "deepseek": Provider(
        name="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        env_key="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
        notes="cheap and strong tool-calling",
    ),
}

# Which provider to try first when --provider is not given.
# (nvidia first: it currently has a working key + credits; openrouter's
# key is valid but the account is out of credits — reorder as yours change)
PREFERENCE_ORDER = ["nvidia", "bigmodel", "groq", "gemini", "openrouter", "deepseek"]


def default_provider() -> str:
    """Pick the first provider whose API key is present in the environment."""
    for pid in PREFERENCE_ORDER:
        if os.environ.get(PROVIDERS[pid].env_key):
            return pid
    return PREFERENCE_ORDER[0]  # nothing set -> caller will show a helpful error


def resolve(provider_id: str | None, model: str | None,
            base_url: str | None, api_key: str | None) -> tuple[str, str, str, str]:
    """Turn CLI flags into a concrete (label, base_url, api_key, model) tuple.

    Priority: explicit flags > provider defaults > first key found in env.
    Supports fully custom endpoints: pass --base-url (any OpenAI-compatible
    server: vLLM, Ollama, LM Studio, llama.cpp server, an internal proxy...).
    """
    pid = (provider_id or "").lower()
    if pid and pid not in PROVIDERS and not base_url:
        # Treat unknown --provider values as "custom" with a guessable URL.
        raise SystemExit(
            f"Unknown provider '{pid}'. Known: {', '.join(PROVIDERS)} "
            f"(or use --base-url for a custom OpenAI-compatible endpoint)."
        )

    if base_url:  # fully custom endpoint
        key = api_key or os.environ.get("CUSTOM_API_KEY", "EMPTY")
        return "custom", base_url, key, model or os.environ.get("CUSTOM_MODEL", "gpt-4o-mini")

    p = PROVIDERS[pid or default_provider()]
    key = api_key or os.environ.get(p.env_key, "")
    return p.name, p.base_url, key, model or p.default_model


def status_table() -> list[tuple[str, str, str, bool]]:
    """(id, name, default_model, key_present) — used by `miniagent providers`."""
    return [
        (pid, p.name, p.default_model, bool(os.environ.get(p.env_key)))
        for pid, p in PROVIDERS.items()
    ]
