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
