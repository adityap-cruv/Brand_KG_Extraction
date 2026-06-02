# RAGAS Benchmark for brand-kg-builder — Design

**Date:** 2026-06-02
**Status:** Approved (design phase)

## Goal

Replace the homegrown LLM-as-judge benchmark (`brandkg/bench.py`) with a
**recognized, externally-defensible** RAG evaluation that:

1. Uses the **real method companies use** — RAGAS, the de-facto industry-standard
   RAG eval framework (also used by a peer-reviewed Agentic Graph-RAG system over
   a private KG: Frontiers in Medicine 2025, PMC12748213).
2. **Builds the benchmark dataset from our own brand documents** automatically
   (RAGAS knowledge-graph testset generation), producing an *auditable* gold set
   of single-hop + multi-hop questions.
3. Is **not self-graded**: the judge is a *different vendor* (OpenAI `gpt-4o-mini`)
   than the model that answers (Claude/Codex via the product's CLI engine).

## Key decisions (from brainstorming)

- **Judge LLM:** OpenAI `gpt-4o-mini`, **benchmark-only**. The product's `build`
  and `query` paths stay 100% API-key-free (CLI/OAuth). Key lives in a separate
  `BENCHMARK_OPENAI_KEY` env var to make "eval-only" explicit.
- **Embeddings:** OpenAI `text-embedding-3-small` (cheap; avoids a heavy local
  `torch`/sentence-transformers dependency). Swappable to local later.
- **Gold-set input:** the **source docs** (`data/extracted/*.txt`), NOT the Neo4j
  graph. RAGAS builds its own throwaway internal KG from the docs only to generate
  good multi-hop questions. The Neo4j brand KG is the system *under test*.
- **Entity-linking eval:** deferred to phase 2 (build core RAGAS first).
- **Old `bench.py`:** left in place (superseded, not deleted).

## The two knowledge graphs (critical mental model)

```
data/extracted/*.txt  (SOURCE DOCS)
   │                         │
   ▼                         ▼
RAGAS internal KG       YOUR brand KG (Neo4j, graph.py)
(throwaway, makes Qs)        │ (system under test)
   │                         │
   ▼                         ▼
GOLD QA SET  ─────►  query.py answers each Q by traversing Neo4j
(question + reference,       │
 ground truth = docs)        ▼
                       RAGAS judges KG-derived answer vs doc-derived gold
```

Because gold answers come from the docs and answers come from the KG, the
benchmark measures **how faithfully the KG preserved the source documents.**

## Components

### A. Gold-set generator — `brandkg/ragas_gen.py`
- Load `data/extracted/*.txt` as LangChain `Document`s.
- `TestsetGenerator` (LLM=`gpt-4o-mini`, embeddings=`text-embedding-3-small`),
  default single-hop + multi-hop mix.
- Write `bench/ragas_testset.json` — human-readable, git-committable (auditable).
- CLI: `python run.py ragas-gen [--size N]`.

### B. System adapter — `answer_with_context()` in `brandkg/query.py`
- Runs the existing agentic loop unchanged, but **records each tool result**
  (search/bucket/neighbors/tier rows) and returns `(answer, retrieved_contexts)`
  where `retrieved_contexts` is a list of stringified graph rows.
- Existing `answer()` is untouched; this is a thin instrumented sibling that
  shares the loop via a small refactor (the loop yields its context).

### C. Evaluator — `brandkg/ragas_bench.py`
- Read `bench/ragas_testset.json`.
- For each row: `answer_with_context(question)` → assemble RAGAS sample
  (`user_input`, `retrieved_contexts`, `response`, `reference`).
- Metrics: **Faithfulness, ResponseRelevancy (answer relevancy),
  LLMContextPrecisionWithReference, LLMContextRecall**.
- Write `bench/RAGAS_RESULTS.md` (summary + per-question) and
  `bench/ragas_results.json`.
- CLI: `python run.py ragas-bench`.

### D. (Phase 2) Entity-linking eval
- Deterministic precision/recall/F1: KG entities the gold answer needs vs. entities
  the traversal pulled. No LLM. Most reproducible graph-specific number.

## Metric methods (what each computes)

| Metric | Method | Needs gold? |
|---|---|---|
| Faithfulness | decompose answer → claims; fraction inferable from context | no |
| Answer relevancy | reverse-generate questions from answer; cosine-sim to original | no |
| Context precision | rank-aware: are gold-relevant chunks ranked high | yes |
| Context recall | decompose gold → claims; fraction present in retrieved context | yes |

Context precision/recall are the reference-anchored, hardest-to-dismiss numbers.

## Dependencies & install
- New `requirements-bench.txt` (kept separate so the product install stays light):
  `ragas`, `langchain-openai`, `datasets`, `pandas`.
- Pin RAGAS to a known-good line (`ragas>=0.2,<0.3`) — its API shifted across
  versions; code targets the 0.2 testset + evaluate API.

## Out of scope (YAGNI)
- Public-dataset leaderboards (we keep the brand corpus by design).
- Local/Ollama judge (chosen against; API judge is more credible + simpler).
- Touching `build`/`query` product behavior or the existing `bench.py`.

## Success criteria
- `python run.py ragas-gen` produces an inspectable gold set from Genuin docs.
- `python run.py ragas-bench` runs the KG-RAG over it and writes a results report
  with the four metrics, judged by a different vendor than the answerer.
