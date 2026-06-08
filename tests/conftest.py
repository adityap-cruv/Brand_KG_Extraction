"""Shared test fixtures: a scripted fake engine and a Neo4j availability gate."""
import sys
from pathlib import Path

import pytest

# make the repo importable as `brandkg`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeEngine:
    """Engine stub. `complete()` returns the next scripted reply each call."""
    name = "fake"

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def complete(self, prompt, *, timeout=300):
        self.calls.append(prompt)
        if not self._replies:
            return '{"action":"answer","text":"<<no more scripted replies>>"}'
        return self._replies.pop(0)

    def complete_json(self, prompt, *, timeout=300):
        from brandkg.engines.base import extract_json
        return extract_json(self.complete(prompt, timeout=timeout))


@pytest.fixture
def fake_engine():
    return FakeEngine


@pytest.fixture(autouse=True)
def _no_bench_file_logging(monkeypatch):
    """Keep the benchmark runner from attaching a FileHandler during tests."""
    try:
        from brandkg.graphrag_bench import runner
        monkeypatch.setattr(runner, "_setup_logging", lambda outdir: None, raising=False)
    except Exception:
        pass


def _neo4j_session_or_skip():
    """Return an open Neo4j session, or skip the test if no DB is reachable."""
    from brandkg.graphrag_bench import graph_generic as gg
    try:
        drv = gg.driver()
        s = drv.session()
        s.run("RETURN 1").single()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Neo4j not reachable: {e}")
    return drv, s


@pytest.fixture
def neo4j_session():
    drv, s = _neo4j_session_or_skip()
    yield s
    s.close()
    drv.close()
