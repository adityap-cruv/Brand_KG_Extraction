import json

from brandkg.graphrag_bench.evaluate import build_eval_cmd, TASKS, _mode_suffix, add_average_summary


def test_build_eval_cmd_argv():
    cmd = build_eval_cmd(
        python="/venv/bin/python", module="Evaluation.generation_eval",
        predictions="/abs/predictions_novel.json", output="/abs/results_novel.json",
        model="gpt-4o-mini", base_url="https://api.openai.com/v1",
        embedding_model="BAAI/bge-large-en-v1.5")
    assert cmd[:4] == ["/venv/bin/python", "-m", "Evaluation.generation_eval", "--mode"]
    assert "API" in cmd
    assert "--data_file" in cmd and "/abs/predictions_novel.json" in cmd
    assert "--output_file" in cmd and "/abs/results_novel.json" in cmd
    assert "--model" in cmd and "gpt-4o-mini" in cmd


def test_tasks_cover_generation_and_retrieval():
    assert TASKS["generation"][0] == "Evaluation.generation_eval"
    assert TASKS["retrieval"][0] == "Evaluation.retrieval_eval"


def test_mode_suffix_preserves_hybrid_compatibility():
    assert _mode_suffix("hybrid") == ""
    assert _mode_suffix("kg") == "_kg"
    assert _mode_suffix("llm") == "_llm"


def test_add_average_summary(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({
        "Fact Retrieval": {"rouge_score": 0.2, "answer_correctness": 0.6},
        "Contextual Summarize": {"average_scores": {"answer_correctness": 0.8, "coverage_score": 0.4}},
    }))
    summary = add_average_summary(path)
    data = json.loads(path.read_text())
    assert summary["average"] == 0.5
    assert data["_summary"]["average"] == 0.5
    assert data["_summary"]["by_question_type_average"] == {
        "Fact Retrieval": 0.4,
        "Contextual Summarize": 0.6000000000000001,
    }
