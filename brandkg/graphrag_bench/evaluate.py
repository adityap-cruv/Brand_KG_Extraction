"""Run GraphRAG-Bench's own generation evaluator over our predictions file.

The evaluator needs torch/BGE/langchain-openai + an OpenAI key (eval-only), so it
runs inside the cloned benchmark repo using that repo's interpreter — it is NOT
installed into the light product venv.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from .. import config


# task -> the benchmark eval module + default output filename suffix
TASKS = {
    "generation": ("Evaluation.generation_eval", "results"),    # ROUGE-L, Answer Correctness, Coverage, Faithfulness
    "retrieval":  ("Evaluation.retrieval_eval",  "retrieval"),  # Context Relevancy, Evidence Recall
}
ANSWER_SOURCES = ("kg", "llm", "hybrid")


def _mode_suffix(answer_source: str) -> str:
    return "" if answer_source == "hybrid" else f"_{answer_source}"


def build_eval_cmd(python, module, predictions, output, model,
                   base_url, embedding_model):
    return [
        python, "-m", module,
        "--mode", "API",
        "--model", model,
        "--base_url", base_url,
        "--embedding_model", embedding_model,
        "--data_file", str(predictions),
        "--output_file", str(output),
    ]


def _numeric_scores(result_block):
    if not isinstance(result_block, dict):
        return []
    metrics = result_block.get("average_scores", result_block)
    if not isinstance(metrics, dict):
        return []
    scores = []
    for value in metrics.values():
        if isinstance(value, (int, float)) and not math.isnan(value):
            scores.append(float(value))
    return scores


def add_average_summary(path: Path):
    """Append leaderboard-style averages to a GraphRAG-Bench result JSON file."""
    data = json.loads(path.read_text() or "{}")
    if not isinstance(data, dict):
        return None
    per_type = {}
    all_scores = []
    for question_type, result_block in data.items():
        if str(question_type).startswith("_"):
            continue
        scores = _numeric_scores(result_block)
        if not scores:
            continue
        per_type[question_type] = sum(scores) / len(scores)
        all_scores.extend(scores)
    if not all_scores:
        return None
    summary = {
        "average": sum(all_scores) / len(all_scores),
        "by_question_type_average": per_type,
    }
    data["_summary"] = summary
    path.write_text(json.dumps(data, indent=2))
    return summary


def run(subset="novel", task="generation", predictions=None, output=None, model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        embedding_model="BAAI/bge-large-en-v1.5",
        bench_repo=None, bench_python=None, answer_source="hybrid"):
    if task not in TASKS:
        raise SystemExit(f"unknown task '{task}'. Choose from {list(TASKS)}")
    if answer_source not in ANSWER_SOURCES:
        raise SystemExit(f"unknown answer source '{answer_source}'. Choose from {ANSWER_SOURCES}")
    module, suffix = TASKS[task]
    repo = Path(bench_repo or config.BENCH_REPO or "")
    if not repo.exists():
        raise SystemExit("set --bench-repo or BRANDKG_BENCH_REPO to the cloned GraphRAG-Benchmark path")
    if not os.getenv("LLM_API_KEY"):
        raise SystemExit("LLM_API_KEY (OpenAI, eval-only) is not set")
    base = Path(config.ROOT) / "bench" / "graphrag_bench"
    mode_suffix = _mode_suffix(answer_source)
    preds = Path(predictions or base / f"predictions_{subset}{mode_suffix}.json").resolve()
    if not preds.exists():
        raise SystemExit(f"predictions file not found: {preds} (run `graphrag-bench` first)")
    out = Path(output or base / f"{suffix}_{subset}{mode_suffix}.json").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    python = bench_python or config.BENCH_PYTHON or sys.executable
    cmd = build_eval_cmd(python, module, preds, out, model, base_url, embedding_model)
    print(f"running ({task}):", " ".join(cmd))
    print("cwd:", repo)
    rc = subprocess.call(cmd, cwd=str(repo))
    if rc == 0:
        summary = add_average_summary(out)
        print(f"{task} eval complete -> {out}")
        if summary:
            print(f"overall average: {summary['average']:.4f}")
    return rc


def main(argv=None):
    p = argparse.ArgumentParser(prog="run.py graphrag-eval")
    p.add_argument("--subset", required=True,
                   help="benchmark subset name, e.g. novel, medical, or a custom parquet subset like brand_ref")
    p.add_argument("--task", default="generation", choices=list(TASKS),
                   help="generation = ROUGE-L/Correctness/Coverage/Faithfulness; retrieval = Context Relevancy/Evidence Recall")
    p.add_argument("--predictions", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--answer-source", default="hybrid", choices=ANSWER_SOURCES,
                   help="which prediction file to evaluate when --predictions is not provided")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--base_url", default="https://api.openai.com/v1")
    p.add_argument("--embedding_model", default="BAAI/bge-large-en-v1.5")
    p.add_argument("--bench-repo", default=None)
    p.add_argument("--bench-python", default=None)
    a = p.parse_args(argv)
    return run(subset=a.subset, task=a.task, predictions=a.predictions, output=a.output,
               model=a.model, base_url=a.base_url, embedding_model=a.embedding_model,
               bench_repo=a.bench_repo, bench_python=a.bench_python,
               answer_source=a.answer_source)
