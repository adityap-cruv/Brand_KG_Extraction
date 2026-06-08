# GraphRAG-Bench Integration for brand-kg-builder — Design

**Date:** 2026-06-02
**Status:** Approved (design phase)

## Goal

Make the `brandkg` pipeline evaluable on **GraphRAG-Bench**
([repo](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark), ICLR'26), so we can
get a **leaderboard-comparable** score for our graph-RAG pipeline against published
baselines (LightRAG, HippoRAG2, fast-graphrag, DIGIMON, Microsoft GraphRAG) on the
*same fixed corpus with the same standardized evaluator*.

The benchmark's workflow we are plugging into:

```
their fixed corpus (novel/medical parquet)
   │  (we plug our pipeline in here)
   ▼
brandkg build  → Neo4j graph (our pipeline = system under test)
   ▼
brandkg query  → (answer, retrieved_context) per benchmark question
   ▼
predictions JSON (their exact format)
   ▼
their Evaluation/generation_eval.py  (judge = gpt-4o-mini, a different vendor)
   ▼
leaderboard-comparable scores
```

## Guiding principle

The benchmark must measure our **actual** pipeline — LLM-driven extraction → Neo4j
graph → agentic query loop — not a reimplementation. The design therefore **reuses
the real query loop and engine layer** and keeps the brand pipeline **100%
untouched** (no regression risk to the Genuin product). The benchmark gets a
*generic schema* path alongside the existing brand path.

## Key decisions (from brainstorming)

- **Schema:** brandkg's fixed 7-bucket BRAND schema does not fit novels / medical
  text. We add a **generic schema mode** (open entity/relation extraction, no Brand
  root, single `:Entity` label). The Genuin **brand schema stays the default**;
  generic mode is opt-in for the benchmark only.
- **Pipeline fidelity:** refactor (behavior-preserving) the agentic query loop out
  of `query.py` so both the brand toolset and a generic toolset drive the *same*
  loop. Guarantees we benchmark the real pipeline, and yields `answer_with_context()`
  (also satisfies the prior RAGAS spec's need).
- **Per-corpus isolation:** Neo4j Community Edition allows only one database, so we
  namespace every node/relationship with a `corpus_name` property and filter all
  generic query tools by it. The `novel` subset has 20 corpora; `medical` has 1.
- **Judge / eval:** use the benchmark's standard path — `gpt-4o-mini` via an
  OpenAI key (eval-only) + local `BAAI/bge-large-en-v1.5` embeddings. This matches
  the RAGAS spec's "different-vendor judge" principle and keeps scores comparable to
  the public leaderboard. The product `build`/`query` paths stay API-key-free.
- **Eval environment:** the evaluator pulls in `torch` / BGE / `langchain-openai`.
  Run it **inside the cloned benchmark repo's own environment**, NOT the light
  product venv. The benchmark repo is referenced by path (`BRANDKG_BENCH_REPO`),
  **not vendored** into this repo.
- **Run scope:** default to a **small sample** (`--subset novel --corpus-limit 1
  --sample 30`) to validate the full loop cheaply before any large run. Each query
  shells out to the `claude`/`codex` CLI up to `MAX_STEPS` times, so full-corpus
  runs are expensive (~4000 questions across both subsets).

## The benchmark contract (verified against the repo)

**Dataset** (`Datasets/Corpus/*.parquet`, `Datasets/Questions/*.parquet`):
- Corpus row: `{corpus_name, context}` — `context` is one large document string
  (novel corpora range ~93K–450K chars).
- Question row: `{id, source, question, answer, question_type, evidence,
  evidence_triple}`. `source` == a `corpus_name`.
- Question types: `Fact Retrieval`, `Complex Reasoning`, `Contextual Summarize`,
  `Creative Generation`. Counts: novel 2010 Qs / 20 corpora; medical 2062 Qs / 1.

**Prediction record we must emit** (one JSON list per subset):
```json
{
  "id": "...",
  "question": "...",
  "source": "<corpus_name>",
  "context": ["<retrieved row 1>", "..."],
  "evidence": "...",
  "question_type": "<one of the 4>",
  "generated_answer": "...",
  "ground_truth": "<gold answer>"
}
```
(`generation_eval.py` reads `question_type`, `generated_answer`, `ground_truth`,
`context`.)

**Evaluator** (`Evaluation/generation_eval.py`), metrics per question type:
| Question type | Metrics |
|---|---|
| Fact Retrieval | ROUGE-L, Answer Correctness |
| Complex Reasoning | ROUGE-L, Answer Correctness |
| Contextual Summarize | Answer Correctness, Coverage |
| Creative Generation | Answer Correctness, Coverage, Faithfulness |

Run: `python -m Evaluation.generation_eval --mode API --model gpt-4o-mini
--base_url https://api.openai.com/v1 --embedding_model BAAI/bge-large-en-v1.5
--data_file <predictions.json> --output_file <results.json>` with `LLM_API_KEY` set.

## Components

### 1. Shared agentic loop — refactor `brandkg/query.py`
Extract `run_loop(engine, system, tools, question, max_steps, verbose) ->
(answer, contexts)`:
- `tools`: dict mapping `action_name -> fn(session, action_dict) -> result`.
- The loop drives engine → parse JSON action → run tool → feed result back →
  until an `answer` action or `MAX_STEPS`. It **records each tool result string**
  into `contexts`.
- `answer(question)` (brand, public) is preserved exactly: it builds the brand
  system prompt + the 4 brand tools (`search/bucket/neighbors/tier`) and calls
  `run_loop`, returning only the answer text.
- New `answer_with_context(question) -> (answer, contexts)` exposes the recorded
  contexts. Generic mode uses the same `run_loop`.

### 2. Generic schema — `config/generic_schema.json`
Selected via `BRANDKG_SCHEMA` env (default: `brand_schema.json`). Generic extraction
asks for open entities `{name, type, description, aliases}` and relationships
`{source, target, type, description}` — no buckets, no brand root, no tiers.
`extract_entities._build_prompt` branches on schema mode to emit the generic output
shape; the graphiti skill body is still injected as guidance.

### 3. Generic graph builder — `brandkg/graphrag_bench/graph_generic.py`
Scaffold-free, parameterized by `corpus_name`:
- `wipe_corpus(session, corpus)` — delete that corpus's nodes/rels (idempotent rebuild).
- `load(session, payloads, corpus)` — MERGE `:Entity {name_key, corpus}` nodes (key
  = `(corpus, name_key)`), set `name/type/description/aliases/sources`, MERGE typed
  relationships between same-corpus entities. No single-home reconcile, no brand
  invariant, no people retier/hierarchy.
- `ensure_index(session)` — full-text index over `Entity(name, description)`.

### 4. Generic query tools — `brandkg/graphrag_bench/query_generic.py`
- `search(session, corpus, term, limit)` — entities matching name/description/alias
  within the corpus.
- `neighbors(session, corpus, name, limit)` — 1-hop relationships of a named entity
  within the corpus.
- `system_prompt(corpus_hint)` — generic instructions + the two tools + `answer`.
- `answer_with_context(question, corpus) -> (answer, contexts)` — builds tools bound
  to `corpus`, calls the shared `run_loop`.

### 5. Benchmark runner — `brandkg/graphrag_bench/runner.py` (+ CLI)
`python run.py graphrag-bench --subset {novel,medical} [--sample N]
[--corpus-limit K] [--bench-repo PATH] [--force]`:
1. Resolve bench repo path (`--bench-repo` or `BRANDKG_BENCH_REPO`); load the
   subset's corpus + questions parquet via `pandas`.
2. For each corpus (limited by `--corpus-limit`):
   a. Chunk `context` into ~6–8K-char windows with small overlap.
   b. Extract entities/relationships per chunk concurrently via the engine
      (reuse `extract_entities` machinery, generic prompt). Checkpoint per
      `(corpus, chunk_hash)` so re-runs resume cheaply.
   c. `wipe_corpus` then `load` all chunk payloads tagged with `corpus_name`.
3. Group questions by `source`; for each question with that corpus,
   `answer_with_context(question, corpus)`, sampling first `--sample` per corpus.
4. Write `bench/graphrag_bench/predictions_<subset>.json` in the contract format.

### 6. Eval wrapper — `python run.py graphrag-eval`
Thin wrapper that shells `python -m Evaluation.generation_eval ...` **with the
benchmark repo as cwd and its own interpreter** (`BRANDKG_BENCH_PYTHON`, default the
repo's venv python), passing our predictions file and writing
`bench/graphrag_bench/results_<subset>.json`. Requires `LLM_API_KEY` (OpenAI). Also
documents the manual command for users who prefer to run eval directly.

### 7. Config / wiring — `brandkg/config.py`
Add: `SCHEMA_FILE` (`BRANDKG_SCHEMA`, default `brand_schema.json`), `BENCH_REPO`
(`BRANDKG_BENCH_REPO`), `BENCH_PYTHON` (`BRANDKG_BENCH_PYTHON`). `schema()` reads the
selected file. Neo4j config reused as-is.

## Data flow

```
parquet corpus ─chunk→ extracted chunks ─engine→ generic entity JSON
      │                                                     │
      └──────────────── questions parquet                   ▼
                              │                  graph_generic.load(corpus)
                              ▼                            (Neo4j)
              answer_with_context(q, corpus) ──run_loop──→ search/neighbors
                              │                                  │
                              ▼                                  ▼
                 predictions_<subset>.json  ◄── (answer, contexts)
                              │
                              ▼
       Evaluation/generation_eval.py (gpt-4o-mini judge, in bench repo venv)
                              │
                              ▼
                    results_<subset>.json (4 metrics)
```

## Error handling

- **Engine extraction failure on a chunk:** log + skip that chunk (partial graph is
  acceptable; a single bad chunk must not abort the corpus). Recorded in run summary.
- **Unparseable query action:** existing loop behavior — re-prompt for valid JSON,
  bounded by `MAX_STEPS`; fall back to the abstain sentence.
- **Empty retrieval / no graph hit:** `answer_with_context` returns the abstain
  answer and an empty `context` list; the prediction is still emitted (judge will
  score it).
- **Missing bench repo / parquet:** fail fast with an actionable message naming the
  expected path and `BRANDKG_BENCH_REPO`.
- **Missing `LLM_API_KEY` at eval:** fail fast before invoking the evaluator.

## Testing

- **Unit:** `graph_generic.load` produces expected nodes/rels and corpus isolation
  (two corpora with same entity name do not collide); generic prompt builder emits
  the generic shape; chunker covers the full text with overlap.
- **Loop refactor regression:** `answer()` output for a brand question is unchanged
  vs. pre-refactor (golden transcript with a stubbed engine).
- **Integration (sample):** `graphrag-bench --subset novel --corpus-limit 1
  --sample 3` end-to-end produces a well-formed predictions file with the required
  fields and non-empty contexts for at least one question.
- **Eval smoke:** predictions file is accepted by `generation_eval.py` schema
  (field presence) — run on the 3-sample file once a judge key is available.

## Scope (YAGNI)

- **Phase 1 (this work):** generation_eval (4 leaderboard metrics) + emitting the
  contexts that retrieval_eval needs.
- **Phase 1 optional:** wire `retrieval_eval` (free — we already emit contexts).
- **Deferred (phase 2):** `indexing_eval` — requires exporting the Neo4j graph to a
  supported on-disk format (GraphML). Not needed for headline leaderboard numbers.
- **Out of scope:** vendoring the benchmark repo or its heavy deps; changing
  product `build`/`query`/`bench` behavior; touching the brand schema.

## Success criteria

- `python run.py graphrag-bench --subset novel --corpus-limit 1 --sample 3` builds a
  per-corpus generic graph and writes a contract-valid predictions file.
- `python run.py graphrag-eval` runs the benchmark's own evaluator over that file and
  writes a results JSON with the four generation metrics, judged by gpt-4o-mini.
- The brand `build`/`query`/`bench` commands behave identically to before (no
  regression).
