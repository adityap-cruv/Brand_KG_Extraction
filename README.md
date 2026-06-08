# brand-kg-builder

Build an **incremental brand knowledge graph** from a folder of documents into
local Neo4j, then query it — using your **Claude or Codex subscription** as the
LLM engine. **No API key required.**

The "engine" shells out to a locally-installed agent CLI in headless mode, which
authenticates with your subscription OAuth:

| Engine | Command used | Auth |
|---|---|---|
| `claude` | `claude -p "<prompt>" --output-format json` | Claude Max/Pro (`~/.claude/.credentials.json`) |
| `codex`  | `codex exec --skip-git-repo-check "<prompt>"` | ChatGPT Pro/Plus (`~/.codex/auth.json`) |

Select with `BRANDKG_ENGINE=claude` or `codex`.

> Why this works when `graphiti-core` couldn't: a Python library can't use your
> subscription (it calls an API endpoint needing an API key). But the agent CLIs
> *can* run headlessly on your subscription, and this repo shells out to them —
> the same mechanism the assistant used to build the graph interactively.

## Fixed schema (generalized for any brand)

```
Brand (root)
 ├─OFFERS──────────→ Product
 ├─HAS_PERSON──────→ People        (+ RoleTier hierarchy, REPORTS_TO)
 ├─OPERATES_AS─────→ Business
 ├─BUILT_WITH──────→ Engineering
 ├─COMMUNICATES_VIA→ Marketing
 ├─SERVES──────────→ Audience
 └─HAS_PARTNERSHIP→ Partnerships
(bucket)-[:CONTAINS]->(entity)   every entity in exactly ONE bucket
```
Brand connects to exactly the 7 buckets; every entity is ≤2 hops from Brand.
Only node labels: `Brand`, the 7 buckets, `RoleTier`, `TextUnit`. Everything
brand-specific lives in `config/brand_schema.json` — swap it for a new brand.

## Setup

```bash
pip install -r requirements.txt          # python deps
# have the claude and/or codex CLI installed + logged into your subscription
cp config/settings.example.env config/settings.env   # edit BRANDKG_SOURCE_DIR etc.
docker compose -f docker/docker-compose.yml up -d     # Neo4j on 7475/7688
```

## Use

```bash
python run.py build                 # docs -> KG (engine extracts entities)
python run.py query "Who are the leaders?"
python run.py stats                 # graph metrics + invariant checks
python run.py wipe                  # clear the graph
```

Neo4j Browser: http://localhost:7475  (neo4j / brandkg2026)

## How `build` works
1. **extract text** — docx/pptx/md/txt → `data/extracted/*.txt` (+ hash manifest, incremental)
2. **engine extracts entities** — the agent CLI returns bucketed entities+relationships JSON per doc (concurrent)
3. **scaffold** — create Brand + 7 buckets + RoleTiers
4. **load** — MERGE entities into buckets (single-home), typed relationships, brand-invariant guard
5. **retier + hierarchy** — derive seniority from role titles; build REPORTS_TO within affiliation
6. **text layer** — chunk source text into TextUnit nodes + full-text indexes (GraphRAG fallback)
7. **verify** — assert only-allowed labels, brand has exactly 7 edges, 0 orphans

Re-running `build` is incremental: unchanged docs are skipped; new facts MERGE in.

## Switching brands
Edit `config/brand_schema.json` (`brand_name`), point `BRANDKG_SOURCE_DIR` at the
new brand's documents, `python run.py wipe && python run.py build`.

## Benchmark on GraphRAG-Bench

Score this pipeline against the public [GraphRAG-Bench](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)
leaderboard (LightRAG, HippoRAG2, etc.) on its fixed novel/medical corpus, using a
generic (non-brand) schema so extraction fits arbitrary documents. Your KG build +
agentic query loop are the system under test; the benchmark's own gpt-4o-mini judge
scores your answers against its gold QA set.

```bash
# 1. clone the benchmark repo separately (it has heavy eval deps: torch, BGE, ragas)
git clone https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.git
export BRANDKG_BENCH_REPO=/abs/path/to/GraphRAG-Benchmark

# 2. build a small sample + emit predictions (uses your claude/codex engine; no API key)
#    Answer-source modes:
#      kg     = answer only from the KG using multi-hop graph tools
#      llm    = answer only from retrieved GraphRAG-Bench source text passages
#      hybrid = try KG first, then source text only if the KG abstains
python run.py graphrag-bench --subset novel --corpus-limit 1 --sample 30 --answer-source kg
python run.py graphrag-bench --subset novel --corpus-limit 1 --sample 30 --answer-source llm
python run.py graphrag-bench --subset novel --corpus-limit 1 --sample 30 --answer-source hybrid

# 3. score with the benchmark's own gpt-4o-mini judge (eval-only OpenAI key).
#    Install the benchmark's requirements into ITS OWN venv first so torch/BGE stay
#    out of this light product venv:
#      python -m venv /path/bench-venv
#      /path/bench-venv/bin/pip install -r $BRANDKG_BENCH_REPO/requirements.txt
export BRANDKG_BENCH_PYTHON=/path/bench-venv/bin/python
export LLM_API_KEY=sk-...      # OpenAI, benchmark judge ONLY
python run.py graphrag-eval --subset novel --answer-source kg
```

Outputs: `bench/graphrag_bench/predictions_<subset>_kg.json`,
`predictions_<subset>_llm.json`, or `predictions_<subset>.json` for hybrid, plus
matching `results_...json` files from `graphrag-eval`. The generation evaluator
reports the four GraphRAG-Bench question groups: Fact Retrieval, Complex Reasoning,
Contextual Summarize, and Creative Generation. Each result file also includes
`_summary.average` and `_summary.by_question_type_average` for leaderboard-style
comparison across the benchmark factors.

The benchmark uses `config/generic_schema.json` (open entity/relation extraction, no
`Brand` root); the brand `build`/`query` commands are unaffected. Start small with
`--corpus-limit`/`--sample` — each question drives the agentic query loop, which
shells out to your engine CLI multiple times.
