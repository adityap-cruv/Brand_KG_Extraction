import json

from brandkg import config


def test_schema_default_is_brand():
    assert config.schema()["brand_name"] == "Genuin"


def test_schema_explicit_file(tmp_path, monkeypatch):
    # explicit filename argument loads that file from CONFIG_DIR
    s = config.schema("generic_schema.json")
    assert s.get("mode") == "generic"


def test_schema_env_override(monkeypatch):
    monkeypatch.setenv("BRANDKG_SCHEMA", "generic_schema.json")
    assert config.schema().get("mode") == "generic"


def test_chunks_cover_text_with_overlap():
    from brandkg.graphrag_bench.runner import _chunks
    text = "abcdefghij" * 100  # 1000 chars
    chunks = _chunks(text, size=300, overlap=50)
    assert "".join(c[:250] for c in chunks)  # sanity: non-empty
    # every original index is covered by some chunk
    joined = "".join(chunks)
    assert len(joined) >= len(text)
    assert chunks[0].startswith("abc")
    # overlap: end of chunk0 reappears at start of chunk1
    assert chunks[1][:50] == chunks[0][-50:]


def test_bench_paths():
    from brandkg.graphrag_bench.runner import _bench_paths
    cp, qp = _bench_paths("/repo", "novel")
    assert str(cp).endswith("Datasets/Corpus/novel.parquet")
    assert str(qp).endswith("Datasets/Questions/novel_questions.parquet")


def test_predictions_path_is_mode_specific(tmp_path):
    from brandkg.graphrag_bench.runner import _predictions_path
    assert _predictions_path(tmp_path, "novel", "hybrid").name == "predictions_novel.json"
    assert _predictions_path(tmp_path, "novel", "kg").name == "predictions_novel_kg.json"
    assert _predictions_path(tmp_path, "novel", "llm").name == "predictions_novel_llm.json"


def test_run_writes_prediction_records(tmp_path, monkeypatch):
    import pandas as pd
    from brandkg.graphrag_bench import runner

    repo = tmp_path / "repo"
    (repo / "Datasets/Corpus").mkdir(parents=True)
    (repo / "Datasets/Questions").mkdir(parents=True)
    pd.DataFrame([{"corpus_name": "C1", "context": "Ada built the Engine."}]).to_parquet(
        repo / "Datasets/Corpus/novel.parquet")
    pd.DataFrame([{"id": "q1", "source": "C1", "question": "who?",
                   "answer": "Ada", "question_type": "Fact Retrieval",
                   "evidence": "Ada built it.", "evidence_triple": "(a,b,c)"}]).to_parquet(
        repo / "Datasets/Questions/novel_questions.parquet")

    # stub build + query so no engine/Neo4j is needed
    monkeypatch.setattr(runner, "build_corpus", lambda *a, **k: (2, 1, 1))
    monkeypatch.setattr(runner, "answer_with_context",
                        lambda q, corpus, session=None: ("Ada.", ["row1", "row2"]))

    class _Sess:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _Drv:
        def session(self): return _Sess()
        def close(self): pass

    monkeypatch.setattr(runner.gg, "driver", lambda: _Drv())
    monkeypatch.setattr(runner.config, "ROOT", tmp_path)

    out_path = runner.run(subset="novel", bench_repo=str(repo))
    data = json.loads(out_path.read_text())
    assert len(data) == 1
    rec = data[0]
    assert set(rec) == {"id", "question", "source", "context", "evidence",
                        "question_type", "generated_answer", "ground_truth"}
    assert rec["context"] == ["row1", "row2"]
    assert rec["generated_answer"] == "Ada."
    assert rec["ground_truth"] == "Ada"


def test_run_llm_mode_skips_kg_and_neo4j(tmp_path, monkeypatch):
    import pandas as pd
    from brandkg.graphrag_bench import runner

    repo = tmp_path / "repo"
    (repo / "Datasets/Corpus").mkdir(parents=True)
    (repo / "Datasets/Questions").mkdir(parents=True)
    pd.DataFrame([{"corpus_name": "C1", "context": "Ada built the Engine."}]).to_parquet(
        repo / "Datasets/Corpus/novel.parquet")
    pd.DataFrame([{"id": "q1", "source": "C1", "question": "who built the Engine?",
                   "answer": "Ada", "question_type": "Fact Retrieval",
                   "evidence": "Ada built it."}]).to_parquet(
        repo / "Datasets/Questions/novel_questions.parquet")

    called = {"build": False, "driver": False}
    monkeypatch.setattr(runner, "build_corpus",
                        lambda *a, **k: called.__setitem__("build", True))
    monkeypatch.setattr(runner.gg, "driver",
                        lambda: called.__setitem__("driver", True))

    class Eng:
        def complete(self, prompt, *, timeout=300):
            return "Ada built the Engine."

    monkeypatch.setattr(runner, "get_engine", lambda name: Eng())
    monkeypatch.setattr(runner.config, "ROOT", tmp_path)

    out_path = runner.run(subset="novel", bench_repo=str(repo), answer_source="llm")
    data = json.loads(out_path.read_text())
    assert out_path.name == "predictions_novel_llm.json"
    assert data[0]["generated_answer"] == "Ada built the Engine."
    assert called == {"build": False, "driver": False}


def test_cli_dispatches_graphrag_bench(monkeypatch):
    from brandkg import cli
    called = {}
    import brandkg.graphrag_bench.runner as r
    monkeypatch.setattr(r, "main", lambda argv: called.update({"argv": argv}) or 0)
    rc = cli.main(["graphrag-bench", "--subset", "novel", "--sample", "3"])
    assert rc == 0
    assert called["argv"] == ["--subset", "novel", "--sample", "3"]


def test_cli_dispatches_graphrag_eval(monkeypatch):
    from brandkg import cli
    called = {}
    import brandkg.graphrag_bench.evaluate as ev
    monkeypatch.setattr(ev, "main", lambda argv: called.update({"argv": argv}) or 0)
    rc = cli.main(["graphrag-eval", "--subset", "novel"])
    assert rc == 0
    assert called["argv"] == ["--subset", "novel"]


def test_answer_corpus_preserves_order_under_concurrency(monkeypatch):
    from brandkg.graphrag_bench import runner

    monkeypatch.setattr(runner, "answer_with_context",
                        lambda question, corpus, session=None: (f"ans:{question}", [question]))

    class _Sess:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _Drv:
        def session(self): return _Sess()
        def close(self): pass

    qs = [{"id": f"q{i}", "question": f"Q{i}", "question_type": "Fact Retrieval",
           "evidence": "e", "answer": "a"} for i in range(25)]
    results = runner.answer_corpus(_Drv(), "C1", qs, workers=8)
    assert len(results) == 25
    assert [r["id"] for r in results] == [f"q{i}" for i in range(25)]  # order preserved
    assert results[3]["generated_answer"] == "ans:Q3"
    assert results[3]["context"] == ["Q3"]
    assert set(results[0]) == {"id", "question", "source", "context", "evidence",
                               "question_type", "generated_answer", "ground_truth"}


def test_answer_corpus_kg_mode_does_not_use_text(monkeypatch):
    from brandkg.graphrag_bench import runner

    monkeypatch.setattr(runner, "answer_with_context",
                        lambda question, corpus, session=None: ("KG", ["kgctx"]))

    class _Sess:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _Drv:
        def session(self): return _Sess()
        def close(self): pass

    called = {"text": False}

    def text_answer(question):
        called["text"] = True
        return "TEXT", ["textctx"]

    qs = [{"id": "q1", "question": "Q1", "question_type": "Fact Retrieval", "evidence": "e", "answer": "a"}]
    res = runner.answer_corpus(_Drv(), "C1", qs, workers=1,
                               answer_source="kg", text_answer_fn=text_answer)
    assert res[0]["generated_answer"] == "KG"
    assert res[0]["context"] == ["kgctx"]
    assert called["text"] is False


def test_answer_corpus_llm_mode_uses_text_only(monkeypatch):
    from brandkg.graphrag_bench import runner

    called = {"kg": False}

    def kg_answer(question, corpus, session=None):
        called["kg"] = True
        return "KG", []

    monkeypatch.setattr(runner, "answer_with_context", kg_answer)

    class _Drv:
        pass

    qs = [{"id": "q1", "question": "Q1", "question_type": "Fact Retrieval", "evidence": "e", "answer": "a"}]
    res = runner.answer_corpus(_Drv(), "C1", qs, workers=1,
                               answer_source="llm",
                               text_answer_fn=lambda q: ("TEXT", ["textctx"]))
    assert res[0]["generated_answer"] == "TEXT"
    assert res[0]["context"] == ["textctx"]
    assert called["kg"] is False


def test_answer_corpus_drops_failed_questions(monkeypatch):
    from brandkg.graphrag_bench import runner

    def flaky(question, corpus, session=None):
        if question == "Q1":
            raise RuntimeError("boom")
        return ("ok", [])

    monkeypatch.setattr(runner, "answer_with_context", flaky)

    class _Sess:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _Drv:
        def session(self): return _Sess()
        def close(self): pass

    qs = [{"id": "q0", "question": "Q0", "question_type": "Fact Retrieval", "evidence": "e", "answer": "a"},
          {"id": "q1", "question": "Q1", "question_type": "Fact Retrieval", "evidence": "e", "answer": "a"}]
    results = runner.answer_corpus(_Drv(), "C1", qs, workers=2)
    # the failed question is DROPPED (not saved as empty) so it retries next run
    assert [r["id"] for r in results] == ["q0"]


def test_build_corpus_reuses_existing_kg(monkeypatch):
    from brandkg.graphrag_bench import runner
    monkeypatch.setattr(runner.gg, "corpus_entity_count", lambda s, c: 42)
    monkeypatch.setattr(runner.gg, "ensure_index", lambda s: None)
    called = {"extract": False}
    monkeypatch.setattr(runner, "extract_blob",
                        lambda *a, **k: called.__setitem__("extract", True))
    te, tr, nc = runner.build_corpus(object(), {}, object(), "C1", "some text", force=False)
    assert (te, tr, nc) == (42, 0, 0)        # reused count, nothing rebuilt
    assert called["extract"] is False        # extraction skipped


def test_run_resumes_from_existing_predictions(tmp_path, monkeypatch):
    import pandas as pd
    from brandkg.graphrag_bench import runner

    repo = tmp_path / "repo"
    (repo / "Datasets/Corpus").mkdir(parents=True)
    (repo / "Datasets/Questions").mkdir(parents=True)
    pd.DataFrame([{"corpus_name": "C1", "context": "ctx"}]).to_parquet(
        repo / "Datasets/Corpus/novel.parquet")
    pd.DataFrame([
        {"id": "q1", "source": "C1", "question": "Q1", "answer": "a1",
         "question_type": "Fact Retrieval", "evidence": "e"},
        {"id": "q2", "source": "C1", "question": "Q2", "answer": "a2",
         "question_type": "Fact Retrieval", "evidence": "e"},
    ]).to_parquet(repo / "Datasets/Questions/novel_questions.parquet")

    # pre-existing predictions: q1 already answered (must be preserved, not redone)
    base = tmp_path / "bench" / "graphrag_bench"
    base.mkdir(parents=True)
    (base / "predictions_novel.json").write_text(json.dumps([
        {"id": "q1", "question": "Q1", "source": "C1", "context": ["old"], "evidence": "e",
         "question_type": "Fact Retrieval", "generated_answer": "OLD-A1", "ground_truth": "a1"}]))

    monkeypatch.setattr(runner, "build_corpus", lambda *a, **k: (5, 0, 0))
    answered = []

    def fake_answer(question, corpus, session=None):
        answered.append(question)
        return ("NEW", ["c"])

    monkeypatch.setattr(runner, "answer_with_context", fake_answer)

    class _Sess:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _Drv:
        def session(self): return _Sess()
        def close(self): pass

    monkeypatch.setattr(runner.gg, "driver", lambda: _Drv())
    monkeypatch.setattr(runner.config, "ROOT", tmp_path)

    out_path = runner.run(subset="novel", bench_repo=str(repo))
    data = {r["id"]: r for r in json.loads(out_path.read_text())}
    assert set(data) == {"q1", "q2"}                       # both present
    assert data["q1"]["generated_answer"] == "OLD-A1"      # preserved, not re-answered
    assert data["q2"]["generated_answer"] == "NEW"         # newly answered
    assert answered == ["Q2"]                              # only q2 was answered


def test_run_purges_poisoned_records(tmp_path, monkeypatch):
    import pandas as pd
    from brandkg.graphrag_bench import runner

    repo = tmp_path / "repo"
    (repo / "Datasets/Corpus").mkdir(parents=True)
    (repo / "Datasets/Questions").mkdir(parents=True)
    pd.DataFrame([{"corpus_name": "C1", "context": "ctx"}]).to_parquet(
        repo / "Datasets/Corpus/novel.parquet")
    pd.DataFrame([{"id": "q1", "source": "C1", "question": "Q1", "answer": "a1",
                   "question_type": "Fact Retrieval", "evidence": "e"}]).to_parquet(
        repo / "Datasets/Questions/novel_questions.parquet")

    # poisoned: q1 saved earlier with an EMPTY answer
    base = tmp_path / "bench" / "graphrag_bench"
    base.mkdir(parents=True)
    (base / "predictions_novel.json").write_text(json.dumps([
        {"id": "q1", "question": "Q1", "source": "C1", "context": [], "evidence": "e",
         "question_type": "Fact Retrieval", "generated_answer": "", "ground_truth": "a1"}]))

    monkeypatch.setattr(runner, "build_corpus", lambda *a, **k: (5, 0, 0))
    answered = []

    def fake_answer(question, corpus, session=None):
        answered.append(question)
        return ("GOOD", ["c"])

    monkeypatch.setattr(runner, "answer_with_context", fake_answer)

    class _Sess:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _Drv:
        def session(self): return _Sess()
        def close(self): pass

    monkeypatch.setattr(runner.gg, "driver", lambda: _Drv())
    monkeypatch.setattr(runner.config, "ROOT", tmp_path)

    out_path = runner.run(subset="novel", bench_repo=str(repo))
    data = {r["id"]: r for r in json.loads(out_path.read_text())}
    assert answered == ["Q1"]                        # poisoned record retried
    assert data["q1"]["generated_answer"] == "GOOD"  # replaced with a real answer


def test_stratified_sample_covers_all_types():
    from brandkg.graphrag_bench.runner import _stratified_sample
    # dataset-style ordering: all of one type first, then the next
    qs = ([{"id": f"f{i}", "question_type": "Fact Retrieval"} for i in range(10)]
          + [{"id": f"c{i}", "question_type": "Complex Reasoning"} for i in range(10)]
          + [{"id": f"s{i}", "question_type": "Contextual Summarize"} for i in range(10)]
          + [{"id": f"g{i}", "question_type": "Creative Generation"} for i in range(10)])
    out = _stratified_sample(qs, 2)
    from collections import Counter
    counts = Counter(q["question_type"] for q in out)
    assert counts == {"Fact Retrieval": 2, "Complex Reasoning": 2,
                      "Contextual Summarize": 2, "Creative Generation": 2}
    assert len(out) == 8  # 2 per type, all 4 types covered (vs head(2) = only Fact Retrieval)


def test_run_tolerates_empty_predictions_file(tmp_path, monkeypatch):
    import pandas as pd
    from brandkg.graphrag_bench import runner

    repo = tmp_path / "repo"
    (repo / "Datasets/Corpus").mkdir(parents=True)
    (repo / "Datasets/Questions").mkdir(parents=True)
    pd.DataFrame([{"corpus_name": "C1", "context": "ctx"}]).to_parquet(
        repo / "Datasets/Corpus/novel.parquet")
    pd.DataFrame([{"id": "q1", "source": "C1", "question": "Q1", "answer": "a1",
                   "question_type": "Fact Retrieval", "evidence": "e"}]).to_parquet(
        repo / "Datasets/Questions/novel_questions.parquet")

    # pre-existing EMPTY predictions file (the bug that crashed json.loads)
    base = tmp_path / "bench" / "graphrag_bench"
    base.mkdir(parents=True)
    (base / "predictions_novel.json").write_text("")   # empty -> must not crash

    monkeypatch.setattr(runner, "build_corpus", lambda *a, **k: (5, 0, 0))
    monkeypatch.setattr(runner, "answer_with_context",
                        lambda q, corpus, session=None: ("A1", ["c"]))

    class _Sess:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _Drv:
        def session(self): return _Sess()
        def close(self): pass

    monkeypatch.setattr(runner.gg, "driver", lambda: _Drv())
    monkeypatch.setattr(runner.config, "ROOT", tmp_path)

    out_path = runner.run(subset="novel", bench_repo=str(repo))   # should NOT raise
    data = json.loads(out_path.read_text())
    assert len(data) == 1 and data[0]["generated_answer"] == "A1"
