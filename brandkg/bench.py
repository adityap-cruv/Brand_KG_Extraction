"""Benchmark the KG on a question set — query AND judge both use the configured
engine (claude/codex) via OAuth. No API key anywhere.

For each question:
  1. answer it with the GraphRAG-style query (engine synthesizes from the KG)
  2. judge correctness 0/1/2 against the gold answer (engine is the judge)
  3. compute fact recall = fraction of gold_facts present (substring + engine fallback)

Writes bench/results.json + bench/RESULTS.md.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

from . import config, query
from .engines import get_engine
from .engines.base import extract_json

QA_PATH = Path(__file__).resolve().parent.parent / "bench" / "qa_set.json"
OUT_DIR = QA_PATH.parent

JUDGE_ANSWER = """You judge whether a system's answer is consistent with the gold answer.
Be GENEROUS: different wording, ordering, or extra context is fine; partial lists are fine when the gold lists examples. What matters is the same factual truth.
Scores: 2 = captures the same core facts as gold; 1 = partially right (some facts, but misses or contradicts an essential one); 0 = wrong/off-topic/no contact with gold facts.
Output STRICT JSON only: {"score": <0|1|2>, "reasoning": "<one short sentence>"}

Question: %(q)s
Gold answer: %(gold)s
System answer: %(ans)s"""

JUDGE_ABSTAIN = """You judge whether a system correctly ABSTAINED from a question whose answer is NOT in the source.
Scores: 2 = clearly says it doesn't know / not in the graph; 1 = hedged/vague without committing; 0 = confidently asserts a specific answer (hallucination).
Output STRICT JSON only: {"score": <0|1|2>, "reasoning": "<one short sentence>"}

Question: %(q)s
System answer: %(ans)s"""

FACT_CHECK = """Does the system answer contain or paraphrase this fact? Answer about meaning, not exact words.
Fact: %(fact)s
System answer: %(ans)s
Output STRICT JSON only: {"present": true|false}"""


def _judge(engine, prompt) -> dict:
    """Always returns a dict. The engine sometimes replies with a bare value
    (e.g. `true` or `2`) instead of the requested JSON object — normalize those."""
    try:
        raw = engine.complete(prompt, timeout=120)
        out = extract_json(raw)
    except Exception as e:
        return {"score": 0, "reasoning": f"judge error: {e}", "present": False}
    if isinstance(out, dict):
        return out
    if isinstance(out, bool):
        return {"present": out, "score": 2 if out else 0}
    if isinstance(out, (int, float)):
        return {"score": int(out), "present": bool(out)}
    return {"score": 0, "present": False, "reasoning": f"unparsed judge output: {str(out)[:80]}"}


def _fact_recall(engine, answer, gold_facts) -> dict:
    if not gold_facts:
        return {"matched": 0, "total": 0, "score": None, "missing": []}
    low = answer.lower()
    matched, missing = [], []
    for f in gold_facts:
        if f.lower() in low:
            matched.append(f)
        else:
            missing.append(f)
    # engine fallback for paraphrased facts
    still = []
    for f in list(missing):
        v = _judge(engine, FACT_CHECK % {"fact": f, "ans": answer})
        if v.get("present"):
            matched.append(f)
        else:
            still.append(f)
    return {"matched": len(matched), "total": len(gold_facts),
            "score": round(len(matched) / len(gold_facts), 3), "missing": still}


def run(ids=None) -> dict:
    qa = json.loads(QA_PATH.read_text())
    if ids:
        qa = [q for q in qa if q["id"] in ids]
    engine = get_engine(config.ENGINE)
    print(f"bench: {len(qa)} questions | engine={engine.name} (query + judge, OAuth) | brand={config.schema()['brand_name']}")
    results = []
    t_all = time.time()
    for q in qa:
        t0 = time.time()
        ans = query.answer(q["question"])
        lat = round(time.time() - t0, 1)
        if q.get("expected_outcome") == "abstain":
            verdict = _judge(engine, JUDGE_ABSTAIN % {"q": q["question"], "ans": ans})
            recall = {"matched": 0, "total": 0, "score": None, "missing": []}
        else:
            verdict = _judge(engine, JUDGE_ANSWER % {"q": q["question"], "gold": q["gold_answer"], "ans": ans})
            recall = _fact_recall(engine, ans, q.get("gold_facts", []))
        score = max(0, min(2, int(verdict.get("score", 0))))
        results.append({"id": q["id"], "question": q["question"],
                        "expected_outcome": q.get("expected_outcome"), "gold_answer": q.get("gold_answer"),
                        "answer": ans, "latency": lat, "score": score,
                        "normalized": score / 2, "reasoning": verdict.get("reasoning", ""),
                        "fact_recall": recall})
        rs = recall["score"]
        print(f"  Q{q['id']:>2} [{q.get('expected_outcome'):7}] score={score}/2 "
              f"recall={rs if rs is not None else '-'} {lat}s")
    wall = round(time.time() - t_all, 1)
    _write(results, engine.name, wall)
    return {"results": results}


def _write(results, engine_name, wall):
    corr = sum(r["normalized"] for r in results) / len(results)
    recs = [r["fact_recall"]["score"] for r in results if r["fact_recall"]["score"] is not None]
    recall = sum(recs) / len(recs) if recs else 0
    lat = sum(r["latency"] for r in results) / len(results)
    full = [r["id"] for r in results if r["score"] == 2]
    part = [r["id"] for r in results if r["score"] == 1]
    zero = [r["id"] for r in results if r["score"] == 0]

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=str))

    L = [f"# Brand KG Benchmark Results",
         "",
         f"Generated via engine **{engine_name}** (query + judge, OAuth — no API key)   ·   Wall-clock: {wall}s",
         "",
         "## Summary",
         "",
         "| System | Correctness | Fact recall | Avg latency (s) | Fully correct | Partially correct | Incorrect |",
         "|---|---|---|---|---|---|---|",
         f"| brand-kg | {corr:.3f} | {recall:.3f} | {lat:.2f} | {len(full)}/{len(results)} | {len(part)}/{len(results)} | {len(zero)}/{len(results)} |",
         "",
         f"- **Fully correct (2/2):** {', '.join('Q'+str(i) for i in full) or '—'}",
         f"- **Partially correct (1/2):** {', '.join('Q'+str(i) for i in part) or '—'}",
         f"- **Incorrect (0/2):** {', '.join('Q'+str(i) for i in zero) or '—'}",
         "",
         "## Per-question detail",
         ""]
    for r in results:
        rs = r["fact_recall"]["score"]
        L.append(f"### Q{r['id']}: {r['question']}")
        L.append("")
        L.append("| System | Correctness | Fact recall | Latency | Answer |")
        L.append("|---|---|---|---|---|")
        ans = r["answer"].replace("\n", " ").replace("|", "\\|")[:240]
        L.append(f"| brand-kg | {r['normalized']:.2f} | {rs if rs is not None else '—'} | {r['latency']}s | {ans} |")
        L.append("")
    (OUT_DIR / "RESULTS.md").write_text("\n".join(L))
    print(f"\nCorrectness {corr:.3f} | Fact recall {recall:.3f} | Avg latency {lat:.2f}s")
    print(f"Fully correct {len(full)}/{len(results)} | wrote bench/results.json + bench/RESULTS.md")
