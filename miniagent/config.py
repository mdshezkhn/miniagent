"""Config: a 15-line .env loader.

Why hand-rolled? So the project has exactly ONE dependency (openai).
Real agents do the same thing with python-dotenv — feel free to swap.
"""

import os
from pathlib import Path


def load_env() -> None:
    """Load KEY=VALUE pairs from .env (project dir, then cwd).

    Never overrides variables that are already set in the environment —
    explicit env vars always win over the file. That's the standard
    convention (same as python-dotenv).
    """
    candidates = [
        Path(__file__).resolve().parent.parent / ".env",  # project dir
        Path.cwd() / ".env",                              # wherever you run from
    ]
    seen = set()
    for env_file in candidates:
        if env_file in seen or not env_file.is_file():
            continue
        seen.add(env_file)
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
