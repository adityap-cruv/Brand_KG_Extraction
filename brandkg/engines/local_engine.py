"""Local engine: talks to a self-hosted, OpenAI-compatible chat endpoint.

Points at a local model (e.g. an Ollama / vLLM server exposed over ngrok) via the
OpenAI-compatible `/v1/chat/completions` API. No commercial subscription or real
API key is required — any non-empty `api_key` string is accepted by the server.

Configure with env vars (see config/settings.example.env):
    BRANDKG_LOCAL_BASE_URL   full base url incl. /v1 (e.g. https://xxxx.ngrok-free.app/v1)
    BRANDKG_LOCAL_MODEL      model name to request   (e.g. qwen3.6:35b)
    BRANDKG_LOCAL_API_KEY    any non-empty string    (default: "ollama")
    BRANDKG_LOCAL_MAX_TOKENS max output tokens (default 16384). A reasoning model
                             spends most of these on its hidden "thinking"; too low
                             a cap means thinking exhausts the budget and `content`
                             comes back EMPTY (finish_reason=length).
    BRANDKG_LOCAL_MIN_TIMEOUT floor (seconds) applied to every call (default 600).
                             qwen3.6 is a slow reasoning model — callers that pass a
                             short timeout (e.g. 120s) would otherwise time out before
                             it finishes thinking.
    BRANDKG_LOCAL_JSON_MODE  "1" requests response_format=json_object on JSON calls (default on)

This model (qwen3.6) ALWAYS emits a long chain-of-thought into a separate
`reasoning_content` field and only then writes the answer to `content`; the
thinking cannot be disabled from the client. So this engine: (1) gives a big
max_tokens + timeout floor so thinking can finish, (2) treats an empty/whitespace
`content` as a failure and retries (it means thinking ran out of room), (3) requests
JSON mode on extraction calls, and (4) retries on JSON *parse* failures (the base
only retries network errors). Any param the server rejects (HTTP 4xx) is stripped
automatically, so this stays compatible with plain OpenAI servers too.
"""
from __future__ import annotations
import logging
import os
from .base import Engine, extract_json, _retries, _retry_base
import time

log = logging.getLogger("brandkg.engine")

DEFAULT_BASE_URL = "https://d180-103-91-222-46.ngrok-free.app/v1"
DEFAULT_MODEL = "qwen3.6:35b"


def _flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off", "")


class LocalEngine(Engine):
    name = "local"

    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "The 'openai' package is required for the local engine. "
                "Install it with: pip install openai"
            ) from e

        self.base_url = os.getenv("BRANDKG_LOCAL_BASE_URL", DEFAULT_BASE_URL)
        self.model = os.getenv("BRANDKG_LOCAL_MODEL", DEFAULT_MODEL)
        api_key = os.getenv("BRANDKG_LOCAL_API_KEY", "ollama") or "ollama"
        self.max_tokens = int(os.getenv("BRANDKG_LOCAL_MAX_TOKENS", "16384"))
        self.min_timeout = int(os.getenv("BRANDKG_LOCAL_MIN_TIMEOUT", "600"))
        self.json_mode = _flag("BRANDKG_LOCAL_JSON_MODE", "1")
        # base.Engine.complete() already retries with backoff, so keep the SDK's
        # own retry count low to avoid compounding waits under concurrency.
        self._client = OpenAI(api_key=api_key, base_url=self.base_url, max_retries=1)

    def _create(self, prompt: str, *, timeout: int, json_mode: bool) -> str:
        # qwen3.6 is a slow reasoning model; never let a caller's short timeout cut
        # it off before it finishes thinking and writes the answer to `content`.
        timeout = max(int(timeout), self.min_timeout)
        kwargs = dict(model=self.model, messages=[{"role": "user", "content": prompt}],
                      stream=False, timeout=float(timeout))
        if self.max_tokens > 0:
            kwargs["max_tokens"] = self.max_tokens
        if json_mode and self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            # Distinguish "server rejects an optional param" (HTTP 4xx) from a
            # transient gateway/connection error (ngrok 3004, timeouts). Only the
            # former is worth stripping params for; the latter must propagate so
            # the outer retry handles it with backoff instead of hammering.
            status = (getattr(e, "status_code", None)
                      or getattr(getattr(e, "response", None), "status_code", None))
            has_optional = kwargs.get("response_format") is not None or "max_tokens" in kwargs
            if has_optional and status in (400, 404, 422):
                if kwargs.get("response_format") is not None:
                    self.json_mode = False  # stop sending it for the rest of the run
                log.warning("local engine: server rejected optional params (status %s); "
                            "retrying without response_format/max_tokens", status)
                for k in ("response_format", "max_tokens"):
                    kwargs.pop(k, None)
                resp = self._client.chat.completions.create(**kwargs)
            else:
                raise
        content = resp.choices[0].message.content
        # Empty `content` with finish_reason=length means the reasoning chain
        # consumed the whole token budget before any answer was written. Raise so
        # the retry layer re-rolls (a fresh attempt usually finishes in budget).
        if not (content and content.strip()):
            finish = resp.choices[0].finish_reason
            raise RuntimeError(
                f"local engine returned empty content (finish_reason={finish}); "
                "thinking likely exhausted max_tokens — raise BRANDKG_LOCAL_MAX_TOKENS")
        return content

    def _complete_once(self, prompt: str, *, timeout: int = 300) -> str:
        return self._create(prompt, timeout=timeout, json_mode=False)

    def complete_json(self, prompt: str, *, timeout: int = 300):
        """JSON completion that retries on BOTH network and *parse* failures.

        base.complete_json only retries transient network errors; but a small
        local model frequently emits structurally invalid JSON (misplaced braces,
        leaked reasoning) that a fresh attempt fixes. Requesting JSON mode plus
        re-rolling on a parse error makes extraction far more reliable.
        """
        attempts = max(1, _retries() + 1)
        last_exc = None
        for i in range(attempts):
            try:
                raw = self._create(prompt, timeout=timeout, json_mode=True)
                return extract_json(raw)
            except Exception as e:  # noqa: BLE001 — retry network AND parse failures
                last_exc = e
                log.warning("local engine JSON call failed (attempt %d/%d): %s",
                            i + 1, attempts, str(e)[:300])
                if i < attempts - 1:
                    time.sleep(min(_retry_base() * (3 ** i), 60.0))
        raise last_exc
