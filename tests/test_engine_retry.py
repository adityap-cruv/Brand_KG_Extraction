import pytest

from brandkg.engines.base import Engine


def test_complete_retries_then_succeeds(monkeypatch):
    monkeypatch.setenv("BRANDKG_ENGINE_RETRY_BASE", "0")
    monkeypatch.setenv("BRANDKG_ENGINE_RETRIES", "3")

    class Flaky(Engine):
        name = "flaky"

        def __init__(self):
            self.calls = 0

        def _complete_once(self, prompt, *, timeout=300):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("transient boom")
            return "ok"

    e = Flaky()
    assert e.complete("p") == "ok"
    assert e.calls == 3   # failed twice, succeeded on third


def test_complete_raises_after_exhausting(monkeypatch):
    monkeypatch.setenv("BRANDKG_ENGINE_RETRY_BASE", "0")
    monkeypatch.setenv("BRANDKG_ENGINE_RETRIES", "2")

    class Always(Engine):
        name = "always"

        def __init__(self):
            self.calls = 0

        def _complete_once(self, prompt, *, timeout=300):
            self.calls += 1
            raise RuntimeError("permanent")

    e = Always()
    with pytest.raises(RuntimeError, match="permanent"):
        e.complete("p")
    assert e.calls == 3   # retries (2) + 1
