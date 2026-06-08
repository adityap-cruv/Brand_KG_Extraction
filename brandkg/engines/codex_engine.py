"""Codex engine: shells out to the `codex exec` CLI headlessly.

Uses the user's ChatGPT subscription (OAuth in ~/.codex/auth.json) — no
OPENAI_API_KEY required. Verified mechanism:
    codex exec --skip-git-repo-check "<prompt>"
prints a transcript; the model's reply is the block after the final `codex`
marker. We parse that out.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import tempfile
from .base import Engine


class CodexEngine(Engine):
    name = "codex"

    def __init__(self):
        self.bin = shutil.which("codex") or "codex"
        self.model = os.getenv("BRANDKG_CODEX_MODEL")  # optional

    def _complete_once(self, prompt: str, *, timeout: int = 300) -> str:
        out_path = None
        cmd = [self.bin, "exec", "--skip-git-repo-check", "--ephemeral"]
        if self.model:
            cmd += ["--model", self.model]
        with tempfile.NamedTemporaryFile(prefix="brandkg-codex-", suffix=".txt", delete=False) as f:
            out_path = f.name
        cmd += ["--output-last-message", out_path, "-"]
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                  timeout=timeout)
            if proc.returncode != 0:
                raise RuntimeError(f"codex CLI failed (exit {proc.returncode}): "
                                   f"stderr={proc.stderr[-1200:]!r} stdout={proc.stdout[-800:]!r}")
            try:
                reply = open(out_path, encoding="utf-8").read().strip()
            except OSError:
                reply = ""
            return reply or _parse_codex_transcript(proc.stdout)
        finally:
            if out_path:
                try:
                    os.unlink(out_path)
                except OSError:
                    pass


def _parse_codex_transcript(out: str) -> str:
    """Extract the assistant reply from `codex exec` transcript output.

    The transcript ends with:
        codex
        <reply...>
        tokens used
        <n>
    We return everything between the last 'codex' marker and 'tokens used'.
    """
    lines = out.splitlines()
    # find last standalone 'codex' line
    start = None
    for i, ln in enumerate(lines):
        if ln.strip() == "codex":
            start = i + 1
    if start is None:
        return out.strip()
    end = len(lines)
    for j in range(start, len(lines)):
        if lines[j].strip() == "tokens used":
            end = j
            break
    return "\n".join(lines[start:end]).strip()
