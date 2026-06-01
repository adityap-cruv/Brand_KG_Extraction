"""Engine interface + shared JSON-extraction helpers."""
from __future__ import annotations
import json
import re
from abc import ABC, abstractmethod


class Engine(ABC):
    name: str = "base"

    @abstractmethod
    def complete(self, prompt: str, *, timeout: int = 300) -> str:
        """Run the agent CLI headlessly and return its text output."""
        raise NotImplementedError

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
