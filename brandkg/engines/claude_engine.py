"""Claude engine: shells out to the `claude` CLI in headless print mode.

Uses the user's Claude subscription (OAuth in ~/.claude/.credentials.json) — no
ANTHROPIC_API_KEY required. Verified mechanism:
    claude -p "<prompt>" --output-format json
returns a JSON envelope whose `result` field holds the model's text.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
from .base import Engine


class ClaudeEngine(Engine):
    name = "claude"

    def __init__(self):
        self.bin = shutil.which("claude") or "claude"
        self.model = os.getenv("BRANDKG_CLAUDE_MODEL")  # optional, e.g. claude-sonnet-4-6

    def complete(self, prompt: str, *, timeout: int = 300) -> str:
        cmd = [self.bin, "-p", prompt, "--output-format", "json"]
        if self.model:
            cmd += ["--model", self.model]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL)
        if proc.returncode != 0:
            raise RuntimeError(f"claude CLI failed (exit {proc.returncode}): {proc.stderr[:400]}")
        out = proc.stdout.strip()
        # envelope: {"type":"result","result":"<text>",...}
        try:
            env = json.loads(out)
            if isinstance(env, dict) and "result" in env:
                return env["result"]
        except Exception:
            pass
        return out
