import pytest

from brandkg.graphrag_bench import graph_generic as gg


def test_nk_normalizes_whitespace_and_case():
    assert gg._nk("  Ada   Lovelace ") == "ada lovelace"
    assert gg._nk(None) == ""


def test_rel_name_sanitizes():
    assert gg._rel_name("works for") == "WORKS_FOR"
    assert gg._rel_name("") == "RELATES_TO"
    assert gg._rel_name("a-b/c") == "A_B_C"


# --- integration (skipped without Neo4j) ---
def test_load_and_corpus_isolation(neo4j_session):
    s = neo4j_session
    payloads = [{
        "source_doc": "d1",
        "entities": [{"name": "Ada", "type": "person", "description": "math",
                      "aliases": ["Ada L."]},
                     {"name": "Engine", "type": "machine", "description": "analytical"}],
        "relationships": [{"source": "Ada", "target": "Engine",
                           "type": "works on", "description": "designed it"}],
    }]
    gg.wipe_corpus(s, "T-A")
    gg.wipe_corpus(s, "T-B")
    # same entity name in a different corpus must not collide
    gg.load(s, [{"source_doc": "d2", "entities": [{"name": "Ada", "type": "person"}],
                 "relationships": []}], "T-B")
    te, tr = gg.load(s, payloads, "T-A")
    assert te == 2 and tr == 1

    a_rows = s.run("MATCH (n:Entity {corpus_name:'T-A'}) RETURN n.name AS n").data()
    assert {r["n"] for r in a_rows} == {"Ada", "Engine"}
    b_rows = s.run("MATCH (n:Entity {corpus_name:'T-B'}) RETURN count(n) AS c").single()["c"]
    assert b_rows == 1  # isolated

    gg.wipe_corpus(s, "T-A")
    gg.wipe_corpus(s, "T-B")


def test_generic_answer_with_context_uses_loop(monkeypatch):
    from brandkg.graphrag_bench import query_generic as qgen

    # stub the loop to capture the tools + corpus binding, avoid Neo4j/engine
    captured = {}

    def fake_run_loop(engine, session, system, question, tools, *a, **k):
        captured["tools"] = set(tools)
        # exercise the bound search tool against a fake session
        captured["search_result"] = tools["search"](session, {"term": "x"})
        return "ANS", ["ctx-row"]

    class _Sess:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def run(self, *a, **k):
            class _R:
                def data(self_inner): return [{"name": "Ada"}]
            return _R()

    class _Drv:
        def session(self): return _Sess()
        def close(self): pass

    monkeypatch.setattr(qgen, "run_loop", fake_run_loop)
    monkeypatch.setattr(qgen, "driver", lambda: _Drv())
    monkeypatch.setattr(qgen, "get_engine", lambda name: object())

    ans, ctx = qgen.answer_with_context("who?", "Novel-1")
    assert ans == "ANS" and ctx == ["ctx-row"]
    assert captured["tools"] == {"search", "neighbors", "paths"}
    assert captured["search_result"] == [{"name": "Ada"}]
