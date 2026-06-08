from brandkg.graphrag_bench.text_fallback import rank_passages, answer_from_text


def test_rank_passages_finds_relevant_span():
    text = "filler text here. " * 40 + "The capital of Haiaina is Taza, a walled city. " + "more filler. " * 40
    ps = rank_passages(text, "What is the capital of Haiaina?", k=1, win=80)
    assert ps and "Taza" in ps[0]


def test_rank_passages_empty_when_no_overlap():
    assert rank_passages("totally unrelated content about cooking", "quantum chromodynamics?", k=2) == []


def test_answer_from_text_injects_passages_and_answers():
    class Eng:
        def complete(self, prompt, *, timeout=300):
            assert "Taza" in prompt          # the retrieved passage was put in the prompt
            return "Taza is the capital of Haiaina."
    text = "junk " * 30 + "Taza is the capital of Haiaina. " + "junk " * 30
    ans, ctx = answer_from_text(Eng(), "capital of Haiaina?", text)
    assert ans == "Taza is the capital of Haiaina."
    assert ctx and ctx[0].startswith("[text-fallback]")


def test_answer_corpus_uses_fallback_only_when_kg_abstains(monkeypatch):
    from brandkg.graphrag_bench import runner
    from brandkg.query import ABSTAIN

    class _Sess:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _Drv:
        def session(self): return _Sess()
        def close(self): pass

    # case 1: KG abstains -> fallback answer used
    monkeypatch.setattr(runner, "answer_with_context",
                        lambda question, corpus, session=None: (ABSTAIN, []))
    qs = [{"id": "q1", "question": "who?", "question_type": "Fact Retrieval", "evidence": "e", "answer": "a"}]
    res = runner.answer_corpus(_Drv(), "C1", qs, workers=1,
                               fallback_fn=lambda q: ("FROM BOOK", ["[text-fallback] p"]))
    assert res[0]["generated_answer"] == "FROM BOOK"
    assert res[0]["context"] == ["[text-fallback] p"]

    # case 2: KG answers -> fallback NOT called
    monkeypatch.setattr(runner, "answer_with_context",
                        lambda question, corpus, session=None: ("KG ANSWER", ["kgctx"]))
    called = {"fb": False}

    def fb(q):
        called["fb"] = True
        return ("x", [])

    res = runner.answer_corpus(_Drv(), "C1", qs, workers=1, fallback_fn=fb)
    assert res[0]["generated_answer"] == "KG ANSWER"
    assert called["fb"] is False
