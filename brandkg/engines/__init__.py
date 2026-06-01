"""LLM engine adapters.

Each engine shells out to a locally-installed agent CLI that authenticates with
the user's SUBSCRIPTION (OAuth), so no API key is required:

  - claude : `claude -p <prompt> --output-format json`   (Claude Max/Pro plan)
  - codex  : `codex exec --skip-git-repo-check <prompt>`  (ChatGPT Pro/Plus plan)

Select with the BRANDKG_ENGINE env var or config (default: claude).
"""
from __future__ import annotations
import os
from .base import Engine
from .claude_engine import ClaudeEngine
from .codex_engine import CodexEngine

_REGISTRY = {"claude": ClaudeEngine, "codex": CodexEngine}


def get_engine(name: str | None = None) -> Engine:
    name = (name or os.getenv("BRANDKG_ENGINE", "claude")).lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown engine '{name}'. Options: {list(_REGISTRY)}")
    return _REGISTRY[name]()
