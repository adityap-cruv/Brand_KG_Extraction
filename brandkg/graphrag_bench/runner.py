"""Run the brandkg pipeline over a GraphRAG-Bench subset and emit predictions JSON.

For each corpus: chunk its text -> extract entities/rels per chunk via the engine
(concurrently) -> load into Neo4j (namespaced by corpus_name) -> answer each of the
corpus's questions with retrieved context -> write the benchmark's prediction format.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .. import config
from ..engines import get_engine
from ..extract_entities import extract_blob
from ..query import ABSTAIN
from . import graph_generic as gg
from .query_generic import answer_fast_with_context, answer_with_context
from .text_fallback import answer_from_text

log = logging.getLogger("brandkg.bench")
ANSWER_SOURCES = ("kg", "llm", "hybrid")
KG_QUERY_MODES = ("fast", "agentic")


def _setup_logging(outdir: Path):
    """Log to console AND bench/graphrag_bench/run.log so failures are diagnosable."""
    root = logging.getLogger("brandkg")
    if any(isinstance(h, logging.FileHandler) for h in root.handlers):
        return
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = logging.FileHandler(outdir / "run.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


def _chunks(text: str, size: int = 7000, overlap: int = 400):
    text = text or ""
    if len(text) <= size:
        return [text] if text else []
    out, i, n = [], 0, len(text)
    step = max(1, size - overlap)
    while i < n:
        out.append(text[i:i + size])
        i += step
    return out


def _bench_paths(repo, subset):
    repo = Path(repo)
    return (repo / f"Datasets/Corpus/{subset}.parquet",
            repo / f"Datasets/Questions/{subset}_questions.parquet")


def _stratified_sample(questions, n):
    """Take up to n questions PER question_type, so all types are represented.

    The dataset lists questions grouped by type, so a plain head(n) would return
    only the first type (e.g. all 'Fact Retrieval'). Stratifying guarantees the
    sample exercises every metric (Coverage, Faithfulness, etc.)."""
    by_type, out = {}, []
    for q in questions:
        t = q.get("question_type", "?")
        if by_type.get(t, 0) < n:
            by_type[t] = by_type.get(t, 0) + 1
            out.append(q)
    return out


def _answer_question(drv, corpus_name, q, answer_source, text_answer_fn=None, kg_query_mode="agentic"):
    """Answer ONE question in its own Neo4j session (driver is thread-safe; sessions
    are not, so each worker opens its own).

    answer_source:
    - kg: use only the corpus knowledge graph.
    - llm: use only retrieved passages from the source dataset document.
    - hybrid: try the KG first, then use dataset text only if the KG abstains.

    Returns a prediction record on success, or None on failure (so the question
    retries next run instead of being checkpointed as a permanent empty answer).
    """
    ans, ctx = ABSTAIN, []
    if answer_source in {"kg", "hybrid"}:
        try:
            with drv.session() as s:
                kg_answer_fn = answer_fast_with_context if kg_query_mode == "fast" else answer_with_context
                try:
                    ans, ctx = kg_answer_fn(
                        q["question"], corpus_name, session=s, question_type=q.get("question_type"))
                except TypeError:
                    ans, ctx = kg_answer_fn(q["question"], corpus_name, session=s)
        except Exception as e:  # noqa: BLE001 — one bad question must not abort the corpus
            log.warning("question %s/%s KG answer failed (not saved, will retry next run): %s",
                        corpus_name, q.get("id"), str(e)[:300])
            return None
    if answer_source == "llm" or (answer_source == "hybrid" and ans == ABSTAIN):
        if text_answer_fn is None:
            log.warning("question %s/%s has no source-text answer function", corpus_name, q.get("id"))
            return None
        try:
            try:
                ans, ctx = text_answer_fn(q["question"], question_type=q.get("question_type"))
            except TypeError:
                ans, ctx = text_answer_fn(q["question"])
        except Exception as e:  # noqa: BLE001 — one bad fallback must not abort the corpus
            log.warning("question %s/%s text answer failed (not saved, will retry next run): %s",
                        corpus_name, q.get("id"), str(e)[:300])
            return None
    return {
        "id": q["id"],
        "question": q["question"],
        "source": corpus_name,
        "context": ctx,
        "evidence": q.get("evidence"),
        "question_type": q["question_type"],
        "generated_answer": ans,
        "ground_truth": q.get("answer"),
    }


def answer_corpus(drv, corpus_name, questions, workers, answer_source="hybrid",
                  text_answer_fn=None, fallback_fn=None, kg_query_mode="agentic"):
    """Answer a corpus's questions concurrently; returns only successful records
    (failed ones are dropped so they retry next run), preserving input order.
    `text_answer_fn(question) -> (answer, contexts)` is used for llm mode, and
    when hybrid mode needs dataset-text fallback."""
    if answer_source not in ANSWER_SOURCES:
        raise ValueError(f"unknown answer_source '{answer_source}'. Choose from {ANSWER_SOURCES}")
    if kg_query_mode not in KG_QUERY_MODES:
        raise ValueError(f"unknown kg_query_mode '{kg_query_mode}'. Choose from {KG_QUERY_MODES}")
    if text_answer_fn is None and fallback_fn is not None:
        text_answer_fn = fallback_fn
    results = [None] * len(questions)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_answer_question, drv, corpus_name, q, answer_source,
                          text_answer_fn, kg_query_mode): i
                for i, q in enumerate(questions)}
        done = 0
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
            done += 1
            if done % 10 == 0 or done == len(questions):
                log.info("[%s] answered %d/%d", corpus_name, done, len(questions))
    ok = [r for r in results if r is not None]
    if len(ok) < len(questions):
        log.warning("[%s] %d/%d questions failed and were skipped (will retry next run)",
                    corpus_name, len(questions) - len(ok), len(questions))
    return ok


def build_corpus(engine, schema, session, corpus_name, context, force=False):
    """Chunk + extract + load one corpus. Returns (entities, rels, n_chunks).

    Resume: if the corpus already has a KG in Neo4j and force is False, reuse it
    (skip the expensive extraction) instead of rebuilding.
    """
    if not force:
        existing = gg.corpus_entity_count(session, corpus_name)
        if existing > 0:
            gg.ensure_index(session)
            log.info("[%s] reusing existing KG (%d entities) — skipping extraction",
                     corpus_name, existing)
            return existing, 0, 0
    gg.wipe_corpus(session, corpus_name)
    chunks = _chunks(context)
    payloads = []
    failed = 0
    with ThreadPoolExecutor(max_workers=config.CONCURRENCY) as ex:
        futs = {ex.submit(extract_blob, engine, schema, f"{corpus_name}#{i}", c): i
                for i, c in enumerate(chunks)}
        for fut in as_completed(futs):
            try:
                payloads.append(fut.result())
            except Exception as e:  # noqa: BLE001 — one bad chunk must not abort the corpus
                failed += 1
                log.warning("[%s] chunk extract failed: %s", corpus_name, str(e)[:300])
    if failed:
        log.warning("[%s] %d/%d chunks failed to extract", corpus_name, failed, len(chunks))
    te, tr = gg.load(session, payloads, corpus_name)
    gg.ensure_index(session)
    return te, tr, len(chunks)


def _predictions_path(outdir: Path, subset: str, answer_source: str) -> Path:
    if answer_source == "hybrid":
        return outdir / f"predictions_{subset}.json"
    return outdir / f"predictions_{subset}_{answer_source}.json"


def _default_kg_query_mode():
    return os.getenv("BRANDKG_KG_QUERY_MODE", "agentic").lower()


def run(subset, sample=None, corpus_limit=None, bench_repo=None, force=False,
        answer_source="hybrid", kg_query_mode=None):
    if answer_source not in ANSWER_SOURCES:
        raise SystemExit(f"unknown answer source '{answer_source}'. Choose from {ANSWER_SOURCES}")
    kg_query_mode = (kg_query_mode or _default_kg_query_mode()).lower()
    if kg_query_mode not in KG_QUERY_MODES:
        raise SystemExit(f"unknown KG query mode '{kg_query_mode}'. Choose from {KG_QUERY_MODES}")
    repo = bench_repo or config.BENCH_REPO
    if not repo:
        raise SystemExit("set --bench-repo or BRANDKG_BENCH_REPO to the cloned GraphRAG-Benchmark path")
    cpath, qpath = _bench_paths(repo, subset)
    if not cpath.exists() or not qpath.exists():
        raise SystemExit(f"benchmark parquet not found under {repo} (looked for {cpath.name}, {qpath.name})")

    schema = config.schema("generic_schema.json")
    engine = get_engine(config.ENGINE)
    corpora = pd.read_parquet(cpath).to_dict("records")
    if corpus_limit:
        corpora = corpora[:corpus_limit]
    qdf = pd.read_parquet(qpath)

    outdir = Path(config.ROOT) / "bench" / "graphrag_bench"
    outdir.mkdir(parents=True, exist_ok=True)
    _setup_logging(outdir)
    outpath = _predictions_path(outdir, subset, answer_source)

    # --- resume: load prior predictions, keyed by (source, id) ---
    # Only keep records with a real answer; drop empty/failed ones so they are
    # retried this run instead of being treated as permanently done.
    records = {}
    if outpath.exists() and not force:
        try:
            raw = json.loads(outpath.read_text() or "[]")
        except (json.JSONDecodeError, ValueError):
            log.warning("%s is empty or not valid JSON — starting fresh", outpath.name)
            raw = []
        kept = [r for r in raw if r.get("generated_answer")]
        for r in kept:
            records[(r["source"], r["id"])] = r
        dropped = len(raw) - len(kept)
        if raw:
            log.info("resume: %d answers loaded from %s%s", len(kept), outpath.name,
                     f" ({dropped} empty/failed dropped — will retry)" if dropped else "")

    def _save():
        outpath.write_text(json.dumps(list(records.values()), indent=2, ensure_ascii=False))

    def _process_corpus(c, session=None):
        name = c["corpus_name"]
        if answer_source == "llm":
            log.info("[%s] answer_source=llm — skipping KG build/reuse", name)
        else:
            te, tr, nc = build_corpus(engine, schema, session, name, c["context"], force)
            if nc:
                log.info("[%s] %d chunks -> %d entities, %d rels", name, nc, te, tr)
        qs = qdf[qdf["source"] == name].to_dict("records")
        if sample:
            qs = _stratified_sample(qs, sample)
        todo = [q for q in qs if (name, q["id"]) not in records]
        if not todo:
            log.info("[%s] all %d questions already answered — skipping", name, len(qs))
            return
        log.info("[%s] answering %d/%d questions (concurrency=%d, answer_source=%s, kg_query_mode=%s)",
                 name, len(todo), len(qs), config.CONCURRENCY, answer_source, kg_query_mode)
        text_fn = None
        if answer_source in {"llm", "hybrid"}:
            ctext = c["context"]
            text_fn = lambda question, question_type=None, _t=ctext: answer_from_text(
                engine, question, _t, question_type=question_type)
        for rec in answer_corpus(drv, name, todo, config.CONCURRENCY,
                                 answer_source=answer_source, text_answer_fn=text_fn,
                                 kg_query_mode=kg_query_mode):
            records[(name, rec["id"])] = rec
        _save()   # checkpoint after each corpus

    drv = None if answer_source == "llm" else gg.driver()
    try:
        if drv is None:
            for c in corpora:
                _process_corpus(c)
        else:
            with drv.session() as s:
                for c in corpora:
                    _process_corpus(c, s)
    finally:
        if drv is not None:
            drv.close()

    _save()
    log.info("wrote %d predictions -> %s", len(records), outpath)
    return outpath


def main(argv=None):
    p = argparse.ArgumentParser(prog="run.py graphrag-bench")
    p.add_argument("--subset", required=True,
                   help="benchmark subset name, e.g. novel, medical, or a custom parquet subset like brand_ref")
    p.add_argument("--sample", type=int, default=None,
                   help="max questions PER question_type per corpus (stratified so all 4 types are covered)")
    p.add_argument("--corpus-limit", type=int, default=None, help="max corpora to build")
    p.add_argument("--bench-repo", default=None, help="path to cloned GraphRAG-Benchmark (or BRANDKG_BENCH_REPO)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--answer-source", default="hybrid", choices=ANSWER_SOURCES,
                   help="kg = answer from KG only with multi-hop graph tools; llm = answer from dataset text only; hybrid = KG first, dataset text if KG abstains")
    p.add_argument("--kg-query-mode", default="fast", choices=KG_QUERY_MODES,
                   help="fast = one Codex call after deterministic multi-hop KG retrieval; agentic = slower tool loop")
    p.add_argument("--no-text-fallback", action="store_true",
                   help="deprecated alias for --answer-source kg")
    a = p.parse_args(argv)
    answer_source = "kg" if a.no_text_fallback else a.answer_source
    run(subset=a.subset, sample=a.sample, corpus_limit=a.corpus_limit,
        bench_repo=a.bench_repo, force=a.force, answer_source=answer_source,
        kg_query_mode=a.kg_query_mode)
    return 0
