import json

from brandkg.query import run_loop, ABSTAIN


def _eng(replies):
    from tests.conftest import FakeEngine
    return FakeEngine(replies)


def test_loop_runs_tool_then_answers():
    calls = []

    def search(session, action):
        calls.append(action)
        return [{"name": "Ada"}]

    engine = _eng([
        json.dumps({"action": "search", "term": "ada"}),
        json.dumps({"action": "answer", "text": "It is **Ada**."}),
    ])
    answer, contexts = run_loop(engine, session=None, system="SYS",
                                question="who?", tools={"search": search})
    assert answer == "It is **Ada**."
    assert len(contexts) == 1
    assert "Ada" in contexts[0]
    assert calls == [{"action": "search", "term": "ada"}]


def test_loop_runs_batched_tools_then_answers():
    calls = []

    def search(session, action):
        calls.append(action["term"])
        return [{"name": action["term"]}]

    engine = _eng([
        json.dumps({"action": "batch", "calls": [
            {"action": "search", "term": "ada"},
            {"action": "search", "term": "engine"},
        ]}),
        json.dumps({"action": "answer", "text": "Ada built the Engine."}),
    ])
    answer, contexts = run_loop(engine, session=None, system="SYS",
                                question="who?", tools={"search": search})
    assert answer == "Ada built the Engine."
    assert calls == ["ada", "engine"]
    assert len(contexts) == 1
    assert "engine" in contexts[0]


def test_loop_handles_unparseable_then_answers():
    engine = _eng(["not json at all",
                   json.dumps({"action": "answer", "text": "ok"})])
    answer, contexts = run_loop(engine, session=None, system="S",
                                question="q", tools={})
    assert answer == "ok"
    assert contexts == []


def test_loop_unknown_action_is_reported_not_crash():
    engine = _eng([json.dumps({"action": "bogus"}),
                   json.dumps({"action": "answer", "text": "done"})])
    answer, contexts = run_loop(engine, session=None, system="S",
                                question="q", tools={"search": lambda s, a: []})
    assert answer == "done"
    assert "unknown action" in contexts[0]


def test_loop_exhausts_steps_returns_abstain():
    engine = _eng([json.dumps({"action": "search", "term": "x"})] * 10)
    answer, contexts = run_loop(engine, session=None, system="S", question="q",
                                tools={"search": lambda s, a: []}, max_steps=3)
    assert answer == ABSTAIN
    assert len(contexts) == 3


def test_answer_wires_brand_tools(monkeypatch):
    import brandkg.query as q

    captured = {}

    def fake_run_loop(engine, session, system, question, tools, *a, **k):
        captured["tools"] = set(tools)
        captured["system"] = system
        return "FINAL", ["ctx"]

    class _Sess:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _Drv:
        def session(self): return _Sess()
        def close(self): pass

    monkeypatch.setattr(q, "run_loop", fake_run_loop)
    monkeypatch.setattr(q.graph, "driver", lambda: _Drv())
    monkeypatch.setattr(q, "get_engine", lambda name: object())

    assert q.answer("who leads?") == "FINAL"
    assert captured["tools"] == {"search", "bucket", "neighbors", "tier"}
    assert "buckets" in captured["system"].lower() or "bucket" in captured["system"].lower()
