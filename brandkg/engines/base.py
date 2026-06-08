"""Engine interface + shared JSON-extraction helpers."""
from __future__ import annotations
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod

log = logging.getLogger("brandkg.engine")


def _retries() -> int:
    return int(os.getenv("BRANDKG_ENGINE_RETRIES", "3"))


def _retry_base() -> float:
    return float(os.getenv("BRANDKG_ENGINE_RETRY_BASE", "5"))


class Engine(ABC):
    name: str = "base"

    @abstractmethod
    def _complete_once(self, prompt: str, *, timeout: int = 300) -> str:
        """Run the agent CLI once and return its text output (or raise)."""
        raise NotImplementedError

    def complete(self, prompt: str, *, timeout: int = 300) -> str:
        """Run the agent CLI with retry + exponential backoff.

        Transient CLI failures (non-zero exit, rate limits, timeouts) are common
        under concurrency; we retry instead of dropping the call. Tune with
        BRANDKG_ENGINE_RETRIES (default 3) and BRANDKG_ENGINE_RETRY_BASE (seconds, 5).
        """
        attempts = max(1, _retries() + 1)
        last_exc = None
        for i in range(attempts):
            try:
                return self._complete_once(prompt, timeout=timeout)
            except Exception as e:  # noqa: BLE001 — retry any CLI failure
                last_exc = e
                log.warning("engine '%s' call failed (attempt %d/%d): %s",
                            self.name, i + 1, attempts, str(e)[:400])
                if i < attempts - 1:
                    wait = min(_retry_base() * (3 ** i), 60.0)
                    log.warning("retrying engine '%s' in %.0fs", self.name, wait)
                    time.sleep(wait)
        log.error("engine '%s' exhausted %d attempts; giving up", self.name, attempts)
        raise last_exc

    def complete_json(self, prompt: str, *, timeout: int = 300) -> dict | list:
        """Run complete() and parse the first JSON object/array from the output.

        The prompt should instruct the model to return only JSON; this is a
        tolerant parser for the common 'wrapped in prose / fenced' cases.
        """
        raw = self.complete(prompt, timeout=timeout)
        return extract_json(raw)


def extract_json(text: str):
    """Pull the first valid JSON object or array out of a model's text reply."""
    text = text.strip()
    # strip ```json fences
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    # fast path
    try:
        return json.loads(text)
    except Exception:
        pass
    # find the first balanced { } or [ ] block
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    chunk = text[start:i + 1]
                    try:
                        return json.loads(chunk)
                    except Exception:
                        break
    raise ValueError(f"No valid JSON found in engine output:\n{text[:500]}")
