# GraphRAG-Bench Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `brandkg` graph-RAG pipeline evaluable on GraphRAG-Bench by adding a generic-schema build+query path that emits the benchmark's prediction format, then a wrapper to run the benchmark's own evaluator.

**Architecture:** Refactor the agentic query loop out of `query.py` so both the existing brand toolset and a new generic toolset drive the *same* loop (the pipeline being benchmarked stays faithful to the product). Add a self-contained `brandkg/graphrag_bench/` package: a scaffold-free generic graph builder (nodes namespaced by `corpus_name`), generic query tools, a runner that chunks each benchmark corpus → extracts via the engine → loads into Neo4j → answers questions with retrieved context → writes predictions JSON, and a thin wrapper that shells the benchmark repo's `generation_eval` (gpt-4o-mini judge) in the repo's own environment. The brand pipeline is untouched.

**Tech Stack:** Python 3.12, Neo4j (existing), pandas/pyarrow (installed), the existing `claude`/`codex` CLI engine layer, pytest (added as dev dep). The benchmark repo (cloned separately) supplies the corpus/questions parquet and the evaluator.

> **Session note:** git is disabled in this session. Each task ends with a `git commit` step for completeness; if running here, skip the commit and just keep changes in the working tree.

---

## File Structure

**New files:**
- `config/generic_schema.json` — generic (open-extraction) schema; a safe superset of the brand schema keys so any module imports cleanly.
- `brandkg/graphrag_bench/__init__.py` — package marker.
- `brandkg/graphrag_bench/graph_generic.py` — scaffold-free Neo4j loader; nodes `:Entity {corpus_name,...}`, typed rels, corpus-scoped; `driver/wipe_corpus/load/ensure_index` + pure helpers `_nk/_rel_name`.
- `brandkg/graphrag_bench/query_generic.py` — `search`/`neighbors` tools scoped by corpus + `SYSTEM` prompt + `answer_with_context(question, corpus, session=None)` driving the shared loop.
- `brandkg/graphrag_bench/runner.py` — `_chunks`, `_bench_paths`, `build_corpus`, `run`, `main(argv)`; writes `bench/graphrag_bench/predictions_<subset>.json`.
- `brandkg/graphrag_bench/evaluate.py` — `build_eval_cmd` (pure) + `run`/`main(argv)`; shells `Evaluation.generation_eval` in the bench repo.
- `requirements-dev.txt` — `pytest`.
- `tests/conftest.py`, `tests/test_query_loop.py`, `tests/test_generic_extract.py`, `tests/test_graph_generic.py`, `tests/test_runner.py`, `tests/test_eval_cmd.py`.

**Modified files:**
- `brandkg/query.py` — extract `run_loop(...)` + `ABSTAIN`; `answer()` becomes a thin wrapper (behavior preserved); add `answer_with_context()`.
- `brandkg/extract_entities.py` — add generic branch to `_build_prompt`; add reusable `extract_blob(engine, schema, doc, text)`.
- `brandkg/config.py` — `schema(file=None)` accepts explicit file / `BRANDKG_SCHEMA` env; add `BENCH_REPO`, `BENCH_PYTHON`, expose `ROOT`.
- `brandkg/cli.py` — dispatch `graphrag-bench` and `graphrag-eval` to the new package; update docstring.

---

## Task 1: Dev test harness

**Files:**
- Create: `requirements-dev.txt`
- Create: `tests/conftest.py`

- [ ] **Step 1: Add pytest dev requirement**

Create `requirements-dev.txt`:
```
pytest>=8.0
```

- [ ] **Step 2: Install it**

Run: `./venv/bin/pip install -r requirements-dev.txt`
Expected: pytest installs; `./venv/bin/pytest --version` prints a version.

- [ ] **Step 3: Create shared test fixtures**

Create `tests/conftest.py`:
```python
"""Shared test fixtures: a scripted fake engine and a Neo4j availability gate."""
import sys
from pathlib import Path

import pytest

# make the repo importable as `brandkg`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeEngine:
    """Engine stub. `complete()` returns the next scripted reply each call."""
    name = "fake"

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def complete(self, prompt, *, timeout=300):
        self.calls.append(prompt)
        if not self._replies:
            return '{"action":"answer","text":"<<no more scripted replies>>"}'
        return self._replies.pop(0)

    def complete_json(self, prompt, *, timeout=300):
        from brandkg.engines.base import extract_json
        return extract_json(self.complete(prompt, timeout=timeout))


@pytest.fixture
def fake_engine():
    return FakeEngine


def _neo4j_session_or_skip():
    """Return an open Neo4j session, or skip the test if no DB is reachable."""
    from brandkg.graphrag_bench import graph_generic as gg
    try:
        drv = gg.driver()
        s = drv.session()
        s.run("RETURN 1").single()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Neo4j not reachable: {e}")
    return drv, s


@pytest.fixture
def neo4j_session():
    drv, s = _neo4j_session_or_skip()
    yield s
    s.close()
    drv.close()
```

- [ ] **Step 4: Verify collection works**

Run: `./venv/bin/pytest tests/ -q`
Expected: `no tests ran` (0 collected) with exit code 5, and **no import/collection errors**.

- [ ] **Step 5: Commit**

```bash
git add requirements-dev.txt tests/conftest.py
git commit -m "test: add pytest harness with fake engine + neo4j gate"
```

---

## Task 2: Extract the agentic loop from `query.py` (behavior-preserving)

**Files:**
- Modify: `brandkg/query.py`
- Test: `tests/test_query_loop.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_query_loop.py`:
```python
import json

from brandkg.query import run_loop, ABSTAIN


def _eng(replies):
    from tests.conftest import FakeEngine
    return FakeEngine(replies)


def test_loop_runs_tool_then_answers():
    calls = []

    def search(session, action):
        calls.append(action)
        return [{"name": "Ada"}]

    engine = _eng([
        json.dumps({"action": "search", "term": "ada"}),
        json.dumps({"action": "answer", "text": "It is **Ada**."}),
    ])
    answer, contexts = run_loop(engine, session=None, system="SYS",
                                question="who?", tools={"search": search})
    assert answer == "It is **Ada**."
    assert len(contexts) == 1               # one tool result recorded
    assert "Ada" in contexts[0]
    assert calls == [{"action": "search", "term": "ada"}]


def test_loop_handles_unparseable_then_answers():
    engine = _eng(["not json at all",
                   json.dumps({"action": "answer", "text": "ok"})])
    answer, contexts = run_loop(engine, session=None, system="S",
                                question="q", tools={})
    assert answer == "ok"
    assert contexts == []


def test_loop_unknown_action_is_reported_not_crash():
    engine = _eng([json.dumps({"action": "bogus"}),
                   json.dumps({"action": "answer", "text": "done"})])
    answer, contexts = run_loop(engine, session=None, system="S",
                                question="q", tools={"search": lambda s, a: []})
    assert answer == "done"
    assert "unknown action" in contexts[0]


def test_loop_exhausts_steps_returns_abstain():
    engine = _eng([json.dumps({"action": "search", "term": "x"})] * 10)
    answer, contexts = run_loop(engine, session=None, system="S", question="q",
                                tools={"search": lambda s, a: []}, max_steps=3)
    assert answer == ABSTAIN
    assert len(contexts) == 3               # one per step, no answer reached
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_query_loop.py -q`
Expected: FAIL — `ImportError: cannot import name 'run_loop'` (and `ABSTAIN`).

- [ ] **Step 3: Refactor `query.py` to add `run_loop` + `ABSTAIN`**

In `brandkg/query.py`, add after the `MAX_STEPS = 8` line:
```python
ABSTAIN = "I don't have enough information to answer this from the knowledge graph."
```

Add this function just above the existing `def answer(`:
```python
def run_loop(engine, session, system, question, tools,
             max_steps=MAX_STEPS, verbose=False, abstain=ABSTAIN):
    """Drive the agentic loop. `tools` maps action name -> fn(session, action) -> result.

    Returns (answer_text, contexts) where contexts is the list of stringified tool
    results the model saw (used as retrieved-context for benchmarking/RAGAS).
    """
    transcript = f"{system}\n\nQUESTION: {question}\n\nYour first action (JSON only):"
    final = abstain
    contexts = []
    for step in range(max_steps):
        raw = engine.complete(transcript, timeout=180)
        try:
            action = extract_json(raw)
        except Exception:
            transcript += "\n\n[your reply could not be parsed as JSON; reply with ONE JSON action]"
            continue
        if isinstance(action, list) and action:
            action = action[0]
        act = (action.get("action") or "").lower() if isinstance(action, dict) else ""
        if verbose:
            print(f"[step {step+1}] {action}")
        if act == "answer":
            final = action.get("text", final)
            break
        fn = tools.get(act)
        if fn is None:
            result = {"error": f"unknown action '{act}'. Use {list(tools)}|answer."}
        else:
            result = fn(session, action)
        blob = json.dumps(result, default=str)[:6000]
        contexts.append(blob)
        transcript += (f"\n\nACTION: {json.dumps(action)}\nRESULT: {blob}\n\n"
                       f"Next action (JSON only — use 'answer' when ready):")
    return final, contexts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_query_loop.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Rewrite `answer()` to use the loop + add `answer_with_context()`**

Replace the entire existing `def answer(question: str, verbose: bool = False) -> str:` body (lines ~119-149) with:
```python
def _brand_tools():
    return {
        "search": lambda s, a: _search(s, a.get("term", "")),
        "bucket": lambda s, a: _bucket(s, a.get("name", "")),
        "neighbors": lambda s, a: _neighbors(s, a.get("name", "")),
        "tier": lambda s, a: _tier(s, a.get("tier", "")),
    }


def _brand_system() -> str:
    return SYSTEM % {"brand": BRAND, "buckets": ", ".join(BUCKETS),
                     "skill": skill_name(), "skillbody": skill_body()}


def answer_with_context(question: str, verbose: bool = False):
    """Brand answer plus the retrieved-context rows the model saw."""
    engine = get_engine(config.ENGINE)
    drv = graph.driver()
    try:
        with drv.session() as s:
            return run_loop(engine, s, _brand_system(), question,
                            _brand_tools(), MAX_STEPS, verbose)
    finally:
        drv.close()


def answer(question: str, verbose: bool = False) -> str:
    final, _ = answer_with_context(question, verbose)
    return final
```

- [ ] **Step 6: Add a regression test that `answer()` wires brand tools into the loop**

Append to `tests/test_query_loop.py`:
```python
def test_answer_wires_brand_tools(monkeypatch):
    import brandkg.query as q

    captured = {}

    def fake_run_loop(engine, session, system, question, tools, *a, **k):
        captured["tools"] = set(tools)
        captured["system"] = system
        return "FINAL", ["ctx"]

    class _Sess:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _Drv:
        def session(self): return _Sess()
        def close(self): pass

    monkeypatch.setattr(q, "run_loop", fake_run_loop)
    monkeypatch.setattr(q.graph, "driver", lambda: _Drv())
    monkeypatch.setattr(q, "get_engine", lambda name: object())

    assert q.answer("who leads?") == "FINAL"
    assert captured["tools"] == {"search", "bucket", "neighbors", "tier"}
    assert "buckets" in captured["system"].lower() or "bucket" in captured["system"].lower()
```

- [ ] **Step 7: Run the full query-loop test file**

Run: `./venv/bin/pytest tests/test_query_loop.py -q`
Expected: PASS (5 passed).

- [ ] **Step 8: Commit**

```bash
git add brandkg/query.py tests/test_query_loop.py
git commit -m "refactor: extract run_loop from query.answer; add answer_with_context"
```

---

## Task 3: Config — selectable schema + benchmark paths

**Files:**
- Modify: `brandkg/config.py`
- Test: `tests/test_runner.py` (config portion)

- [ ] **Step 1: Write the failing test**

Create `tests/test_runner.py` with the config test first:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_runner.py -q`
Expected: FAIL — `config.schema("generic_schema.json")` raises `TypeError` (no arg) or FileNotFoundError; `mode` missing.

- [ ] **Step 3: Update `config.py`**

In `brandkg/config.py`, replace the existing `def schema()` with:
```python
def schema(file: str | None = None) -> dict:
    """Load a schema JSON from CONFIG_DIR.

    Precedence: explicit `file` arg > BRANDKG_SCHEMA env > brand_schema.json.
    """
    name = file or os.getenv("BRANDKG_SCHEMA", "brand_schema.json")
    return json.loads((CONFIG_DIR / name).read_text())
```

Add near the other `env(...)` constants at the bottom of the file:
```python
BENCH_REPO = env("BRANDKG_BENCH_REPO", "")
BENCH_PYTHON = env("BRANDKG_BENCH_PYTHON", "")
```
(`ROOT` is already defined at the top of the module.)

- [ ] **Step 4: Create the generic schema file**

Create `config/generic_schema.json`:
```json
{
  "mode": "generic",
  "brand_name": "GenericCorpus",
  "brand_aliases": [],
  "brand_attrs": { "description": "Generic corpus graph for GraphRAG-Bench." },
  "buckets": [],
  "brand_to_bucket_rel": {},
  "membership_rel": "CONTAINS",
  "seniority_tiers": [],
  "allowed_cross_child_rels": []
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_runner.py -q`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add brandkg/config.py config/generic_schema.json tests/test_runner.py
git commit -m "feat: selectable schema (generic_schema.json) + bench repo config"
```

---

## Task 4: Generic extraction prompt + reusable `extract_blob`

**Files:**
- Modify: `brandkg/extract_entities.py`
- Test: `tests/test_generic_extract.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_generic_extract.py`:
```python
import json

from brandkg import config
from brandkg.extract_entities import _build_prompt, extract_blob


def test_generic_prompt_has_no_brand_buckets():
    schema = config.schema("generic_schema.json")
    p = _build_prompt(schema, "doc1", "Ada Lovelace wrote the first algorithm.")
    assert "7 buckets" not in p
    assert "entities" in p and "relationships" in p
    assert "Ada Lovelace" in p


def test_brand_prompt_still_has_buckets():
    schema = config.schema("brand_schema.json")
    p = _build_prompt(schema, "doc1", "some text")
    assert "7 buckets" in p


def test_extract_blob_parses_engine_json():
    from tests.conftest import FakeEngine
    payload = {"entities": [{"name": "Ada", "type": "person"}], "relationships": []}
    engine = FakeEngine([json.dumps(payload)])
    out = extract_blob(engine, config.schema("generic_schema.json"), "doc1", "text")
    assert out["entities"][0]["name"] == "Ada"
    assert out["source_doc"] == "doc1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_generic_extract.py -q`
Expected: FAIL — `ImportError: cannot import name 'extract_blob'`.

- [ ] **Step 3: Add the generic template + branch + `extract_blob`**

In `brandkg/extract_entities.py`, add this template constant after `PROMPT_TEMPLATE`:
```python
GENERIC_PROMPT_TEMPLATE = """You are running the `{skill_name}` skill. Follow its guidance below as the
authoritative method for extracting a knowledge graph from text.

================= GRAPHITI SKILL (source of truth) =================
{skill}
===================================================================

Apply that skill to ONE document. Extract ALL salient, specific entities (people,
places, organizations, concepts, objects, events) and the relationships between
them. Use open, descriptive types — there are no fixed categories.

Pruning rules from the skill: prefer full canonical names; merge obvious aliases;
drop pure boilerplate; every relationship's source and target MUST also appear in
entities.

Output ONLY valid JSON (no prose, no markdown fences) in EXACTLY this shape:
{{"source_doc": "{doc}",
  "entities": [{{"name": "...", "type": "<short word>", "aliases": [],
                "description": "1-2 sentences grounded in the text"}}],
  "relationships": [{{"source": "...", "target": "...", "type": "<verb phrase>",
                      "description": "..."}}]}}

DOCUMENT TEXT:
\"\"\"
{text}
\"\"\"
"""
```

Replace the existing `def _build_prompt(...)` with:
```python
def _build_prompt(schema: dict, doc: str, text: str) -> str:
    if schema.get("mode") == "generic":
        return GENERIC_PROMPT_TEMPLATE.format(
            skill_name=skill_name(), skill=skill_body(),
            doc=doc, text=text[:24000])
    return PROMPT_TEMPLATE.format(
        skill_name=skill_name(),
        skill=skill_body(),
        brand=schema["brand_name"],
        buckets="\n".join(f"  - {b}" for b in schema["buckets"]),
        tiers=", ".join(schema["seniority_tiers"]),
        rels=", ".join(schema["allowed_cross_child_rels"]),
        doc=doc, text=text[:24000],
    )
```

Add this reusable function (used by the benchmark runner) after `_build_prompt`:
```python
def extract_blob(engine, schema: dict, doc: str, text: str, timeout: int = 420) -> dict:
    """Extract one text blob -> validated payload dict (no file IO). Raises on bad shape."""
    payload = engine.complete_json(_build_prompt(schema, doc, text), timeout=timeout)
    if not isinstance(payload, dict) or "entities" not in payload:
        raise ValueError("engine returned unexpected shape (no 'entities')")
    payload.setdefault("source_doc", doc)
    return payload
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_generic_extract.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add brandkg/extract_entities.py tests/test_generic_extract.py
git commit -m "feat: generic extraction prompt + reusable extract_blob"
```

---

## Task 5: Generic graph builder

**Files:**
- Create: `brandkg/graphrag_bench/__init__.py`
- Create: `brandkg/graphrag_bench/graph_generic.py`
- Test: `tests/test_graph_generic.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_graph_generic.py`:
```python
import pytest

from brandkg.graphrag_bench import graph_generic as gg


def test_nk_normalizes_whitespace_and_case():
    assert gg._nk("  Ada   Lovelace ") == "ada lovelace"
    assert gg._nk(None) == ""


def test_rel_name_sanitizes():
    assert gg._rel_name("works for") == "WORKS_FOR"
    assert gg._rel_name("") == "RELATES_TO"
    assert gg._rel_name("a-b/c") == "A_B_C"


# --- integration (skipped without Neo4j) ---
def test_load_and_corpus_isolation(neo4j_session):
    s = neo4j_session
    payloads = [{
        "source_doc": "d1",
        "entities": [{"name": "Ada", "type": "person", "description": "math",
                      "aliases": ["Ada L."]},
                     {"name": "Engine", "type": "machine", "description": "analytical"}],
        "relationships": [{"source": "Ada", "target": "Engine",
                           "type": "works on", "description": "designed it"}],
    }]
    gg.wipe_corpus(s, "T-A")
    gg.wipe_corpus(s, "T-B")
    # same entity name in a different corpus must not collide
    gg.load(s, [{"source_doc": "d2", "entities": [{"name": "Ada", "type": "person"}],
                 "relationships": []}], "T-B")
    te, tr = gg.load(s, payloads, "T-A")
    assert te == 2 and tr == 1

    a_rows = s.run("MATCH (n:Entity {corpus_name:'T-A'}) RETURN n.name AS n").data()
    assert {r["n"] for r in a_rows} == {"Ada", "Engine"}
    b_rows = s.run("MATCH (n:Entity {corpus_name:'T-B'}) RETURN count(n) AS c").single()["c"]
    assert b_rows == 1  # isolated

    gg.wipe_corpus(s, "T-A")
    gg.wipe_corpus(s, "T-B")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_graph_generic.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'brandkg.graphrag_bench'`.

- [ ] **Step 3: Create the package + builder**

Create `brandkg/graphrag_bench/__init__.py`:
```python
"""GraphRAG-Bench integration: generic-schema build + query + benchmark runner."""
```

Create `brandkg/graphrag_bench/graph_generic.py`:
```python
"""Scaffold-free Neo4j loader for the benchmark.

No Brand root, no buckets, no tiers. Every node/relationship is namespaced by a
`corpus_name` property so many benchmark corpora coexist in one (Community) database.
Nodes carry a single `:Entity` label; the descriptive kind is a `type` property.
"""
from __future__ import annotations
import re

from neo4j import GraphDatabase

from .. import config


def driver():
    return GraphDatabase.driver(config.NEO4J_URI, auth=config.NEO4J_AUTH)


def _nk(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _rel_name(t: str) -> str:
    r = re.sub(r"[^A-Za-z0-9]+", "_", (t or "RELATES_TO").upper()).strip("_")
    return r or "RELATES_TO"


_NODE = """
MERGE (e:Entity {corpus_name:$corpus, name_key:$key})
ON CREATE SET e.name=$name, e.aliases=[], e.sources=[], e.layer=2
SET e.name=coalesce(e.name,$name), e.type=$type,
    e.description = CASE WHEN $desc='' THEN e.description
        WHEN e.description IS NULL OR size($desc)>size(coalesce(e.description,'')) THEN $desc
        ELSE e.description END,
    e.aliases = coalesce(e.aliases,[]) + [a IN $aliases WHERE NOT a IN coalesce(e.aliases,[])],
    e.sources = coalesce(e.sources,[]) + [x IN [$doc] WHERE NOT x IN coalesce(e.sources,[])]
"""
_REL = """
MATCH (a:Entity {corpus_name:$corpus, name_key:$sk}),
      (b:Entity {corpus_name:$corpus, name_key:$tk})
MERGE (a)-[r:`__REL__`]->(b)
ON CREATE SET r.sources=[]
SET r.description=CASE WHEN $desc='' THEN r.description
        WHEN r.description IS NULL OR size($desc)>size(coalesce(r.description,'')) THEN $desc
        ELSE r.description END,
    r.sources=coalesce(r.sources,[]) + [x IN [$doc] WHERE NOT x IN coalesce(r.sources,[])]
"""


def ensure_index(s):
    try:
        s.run("CREATE FULLTEXT INDEX genericEntity IF NOT EXISTS "
              "FOR (n:Entity) ON EACH [n.name, n.description]")
    except Exception:
        pass


def wipe_corpus(s, corpus: str):
    s.run("MATCH (n:Entity {corpus_name:$c}) DETACH DELETE n", c=corpus)


def load(s, payloads, corpus: str):
    """MERGE entities + relationships for one corpus. Returns (entities, rels)."""
    seen = set()
    te = tr = 0
    for p in payloads:
        doc = p.get("source_doc", "doc")
        for e in p.get("entities", []):
            name = (e.get("name") or "").strip()
            nk = _nk(name)
            if not nk:
                continue
            aliases = sorted({a.strip() for a in e.get("aliases", []) if a and a.strip()})
            s.run(_NODE, corpus=corpus, key=nk, name=name,
                  type=(e.get("type") or "").strip(),
                  desc=(e.get("description") or "").strip(), aliases=aliases, doc=doc)
            if nk not in seen:
                seen.add(nk)
                te += 1
        for r in p.get("relationships", []):
            sk, tk = _nk(r.get("source")), _nk(r.get("target"))
            if not sk or not tk:
                continue
            s.run(_REL.replace("__REL__", _rel_name(r.get("type"))),
                  corpus=corpus, sk=sk, tk=tk,
                  desc=(r.get("description") or "").strip(), doc=doc)
            tr += 1
    return te, tr
```

- [ ] **Step 4: Run tests**

Run: `./venv/bin/pytest tests/test_graph_generic.py -q`
Expected: PASS — 2 unit tests pass; the integration test PASSES if Neo4j is up (`docker compose -f docker/docker-compose.yml up -d`), otherwise SKIPS. No failures.

- [ ] **Step 5: Commit**

```bash
git add brandkg/graphrag_bench/__init__.py brandkg/graphrag_bench/graph_generic.py tests/test_graph_generic.py
git commit -m "feat: generic corpus-scoped Neo4j graph builder"
```

---

## Task 6: Generic query tools + `answer_with_context`

**Files:**
- Create: `brandkg/graphrag_bench/query_generic.py`
- Test: `tests/test_graph_generic.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_graph_generic.py`:
```python
def test_generic_answer_with_context_uses_loop(monkeypatch):
    from brandkg.graphrag_bench import query_generic as qgen

    # stub the loop to capture the tools + corpus binding, avoid Neo4j/engine
    captured = {}

    def fake_run_loop(engine, session, system, question, tools, *a, **k):
        captured["tools"] = set(tools)
        # exercise the bound search tool against a fake session
        captured["search_result"] = tools["search"](session, {"term": "x"})
        return "ANS", ["ctx-row"]

    class _Sess:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def run(self, *a, **k):
            class _R:
                def data(self_inner): return [{"name": "Ada"}]
            return _R()

    class _Drv:
        def session(self): return _Sess()
        def close(self): pass

    monkeypatch.setattr(qgen, "run_loop", fake_run_loop)
    monkeypatch.setattr(qgen, "driver", lambda: _Drv())
    monkeypatch.setattr(qgen, "get_engine", lambda name: object())

    ans, ctx = qgen.answer_with_context("who?", "Novel-1")
    assert ans == "ANS" and ctx == ["ctx-row"]
    assert captured["tools"] == {"search", "neighbors"}
    assert captured["search_result"] == [{"name": "Ada"}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_graph_generic.py::test_generic_answer_with_context_uses_loop -q`
Expected: FAIL — `ModuleNotFoundError`/`cannot import name` for `query_generic`.

- [ ] **Step 3: Create `query_generic.py`**

Create `brandkg/graphrag_bench/query_generic.py`:
```python
"""Generic agentic query over a corpus-scoped graph. Reuses query.run_loop so the
benchmarked pipeline is identical to the product's, only the toolset differs."""
from __future__ import annotations

from .. import config
from ..engines import get_engine
from ..query import run_loop, ABSTAIN
from .graph_generic import driver


def _search(s, corpus, term, limit=25):
    t = (term or "").lower()
    rows = s.run(
        "MATCH (n:Entity {corpus_name:$c}) WHERE "
        "  toLower(n.name) CONTAINS $t OR toLower(coalesce(n.description,'')) CONTAINS $t "
        "  OR any(a IN coalesce(n.aliases,[]) WHERE toLower(a) CONTAINS $t) "
        "RETURN n.name AS name, n.type AS type, "
        "substring(coalesce(n.description,''),0,200) AS description "
        "LIMIT $lim", c=corpus, t=t, lim=limit)
    return [r.data() for r in rows]


def _neighbors(s, corpus, name, limit=50):
    rows = s.run(
        "MATCH (a:Entity {corpus_name:$c}) WHERE toLower(a.name)=toLower($n) "
        "MATCH (a)-[r]-(b:Entity {corpus_name:$c}) "
        "RETURN a.name AS entity, type(r) AS rel, "
        "CASE WHEN startNode(r)=a THEN 'out' ELSE 'in' END AS dir, "
        "b.name AS other, coalesce(r.description,'') AS description "
        "LIMIT $lim", c=corpus, n=name, lim=limit)
    return [r.data() for r in rows]


SYSTEM = """You answer a question by EXPLORING a knowledge graph built from one document.

GRAPH SHAPE
- Nodes are entities with: name, type, description, aliases.
- Entities are linked by typed, described relationships.

YOU HAVE THESE READ-ONLY TOOLS. Respond with ONE JSON object per turn, nothing else:
- {"action":"search","term":"<text>"}      find entities matching text in name/description/alias
- {"action":"neighbors","name":"<Entity>"} a node's relationships (in and out)
- {"action":"answer","text":"<final answer>"} when you can answer

RULES
- Explore as many steps as needed; you'll be given each tool result, then asked for the next action.
- Base the final answer ONLY on what the tools return. Do not use outside knowledge.
- If the graph genuinely lacks the answer, answer EXACTLY:
  "%(abstain)s"
- Prefer specific facts; cite entity names in **bold**.
""" % {"abstain": ABSTAIN}


def _tools(corpus):
    return {
        "search": lambda s, a: _search(s, corpus, a.get("term", "")),
        "neighbors": lambda s, a: _neighbors(s, corpus, a.get("name", "")),
    }


def answer_with_context(question: str, corpus: str, session=None, verbose: bool = False):
    """Return (answer, contexts) for a question scoped to one corpus.

    If `session` is provided it is reused (no driver churn across many questions);
    otherwise a driver/session is opened and closed for this call.
    """
    engine = get_engine(config.ENGINE)
    if session is not None:
        return run_loop(engine, session, SYSTEM, question, _tools(corpus), verbose=verbose)
    drv = driver()
    try:
        with drv.session() as s:
            return run_loop(engine, s, SYSTEM, question, _tools(corpus), verbose=verbose)
    finally:
        drv.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_graph_generic.py -q`
Expected: PASS (unit + this new test; integration test skips without Neo4j).

- [ ] **Step 5: Commit**

```bash
git add brandkg/graphrag_bench/query_generic.py tests/test_graph_generic.py
git commit -m "feat: generic corpus-scoped query tools over the shared loop"
```

---

## Task 7: Benchmark runner (chunk → build → answer → predictions)

**Files:**
- Create: `brandkg/graphrag_bench/runner.py`
- Test: `tests/test_runner.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_runner.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/pytest tests/test_runner.py -q`
Expected: FAIL — `ModuleNotFoundError: ...runner` (import inside tests).

- [ ] **Step 3: Create `runner.py`**

Create `brandkg/graphrag_bench/runner.py`:
```python
"""Run the brandkg pipeline over a GraphRAG-Bench subset and emit predictions JSON.

For each corpus: chunk its text -> extract entities/rels per chunk via the engine
(concurrently) -> load into Neo4j (namespaced by corpus_name) -> answer each of the
corpus's questions with retrieved context -> write the benchmark's prediction format.
"""
from __future__ import annotations
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .. import config
from ..engines import get_engine
from ..extract_entities import extract_blob
from . import graph_generic as gg
from .query_generic import answer_with_context


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


def build_corpus(engine, schema, session, corpus_name, context, force=False):
    """Chunk + extract + load one corpus. Returns (entities, rels, n_chunks)."""
    gg.wipe_corpus(session, corpus_name)
    chunks = _chunks(context)
    payloads = []
    with ThreadPoolExecutor(max_workers=config.CONCURRENCY) as ex:
        futs = {ex.submit(extract_blob, engine, schema, f"{corpus_name}#{i}", c): i
                for i, c in enumerate(chunks)}
        for fut in as_completed(futs):
            try:
                payloads.append(fut.result())
            except Exception as e:  # noqa: BLE001 — one bad chunk must not abort the corpus
                print(f"  chunk extract failed: {str(e)[:160]}")
    te, tr = gg.load(session, payloads, corpus_name)
    gg.ensure_index(session)
    return te, tr, len(chunks)


def run(subset, sample=None, corpus_limit=None, bench_repo=None, force=False):
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

    out = []
    drv = gg.driver()
    try:
        with drv.session() as s:
            for c in corpora:
                name = c["corpus_name"]
                te, tr, nc = build_corpus(engine, schema, s, name, c["context"], force)
                print(f"[{name}] {nc} chunks -> {te} entities, {tr} rels")
                qs = qdf[qdf["source"] == name].to_dict("records")
                if sample:
                    qs = qs[:sample]
                print(f"[{name}] answering {len(qs)} questions")
                for q in qs:
                    ans, ctx = answer_with_context(q["question"], name, session=s)
                    out.append({
                        "id": q["id"],
                        "question": q["question"],
                        "source": name,
                        "context": ctx,
                        "evidence": q.get("evidence"),
                        "question_type": q["question_type"],
                        "generated_answer": ans,
                        "ground_truth": q.get("answer"),
                    })
    finally:
        drv.close()

    outdir = Path(config.ROOT) / "bench" / "graphrag_bench"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"predictions_{subset}.json"
    outpath.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {len(out)} predictions -> {outpath}")
    return outpath


def main(argv=None):
    p = argparse.ArgumentParser(prog="run.py graphrag-bench")
    p.add_argument("--subset", required=True, choices=["novel", "medical"])
    p.add_argument("--sample", type=int, default=None, help="max questions per corpus")
    p.add_argument("--corpus-limit", type=int, default=None, help="max corpora to build")
    p.add_argument("--bench-repo", default=None, help="path to cloned GraphRAG-Benchmark (or BRANDKG_BENCH_REPO)")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    run(subset=a.subset, sample=a.sample, corpus_limit=a.corpus_limit,
        bench_repo=a.bench_repo, force=a.force)
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/bin/pytest tests/test_runner.py -q`
Expected: PASS (6 passed total in this file).

- [ ] **Step 5: Commit**

```bash
git add brandkg/graphrag_bench/runner.py tests/test_runner.py
git commit -m "feat: GraphRAG-Bench runner -> predictions JSON"
```

---

## Task 8: Eval wrapper (shell the benchmark's own evaluator)

**Files:**
- Create: `brandkg/graphrag_bench/evaluate.py`
- Test: `tests/test_eval_cmd.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_cmd.py`:
```python
from brandkg.graphrag_bench.evaluate import build_eval_cmd


def test_build_eval_cmd_argv():
    cmd = build_eval_cmd(
        python="/venv/bin/python", subset="novel",
        predictions="/abs/predictions_novel.json", output="/abs/results_novel.json",
        model="gpt-4o-mini", base_url="https://api.openai.com/v1",
        embedding_model="BAAI/bge-large-en-v1.5")
    assert cmd[:4] == ["/venv/bin/python", "-m", "Evaluation.generation_eval", "--mode"]
    assert "API" in cmd
    assert "--data_file" in cmd and "/abs/predictions_novel.json" in cmd
    assert "--output_file" in cmd and "/abs/results_novel.json" in cmd
    assert "--model" in cmd and "gpt-4o-mini" in cmd
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_eval_cmd.py -q`
Expected: FAIL — `ModuleNotFoundError`/`cannot import name 'build_eval_cmd'`.

- [ ] **Step 3: Create `evaluate.py`**

Create `brandkg/graphrag_bench/evaluate.py`:
```python
"""Run GraphRAG-Bench's own generation evaluator over our predictions file.

The evaluator needs torch/BGE/langchain-openai + an OpenAI key (eval-only), so it
runs inside the cloned benchmark repo using that repo's interpreter — it is NOT
installed into the light product venv.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

from .. import config


def build_eval_cmd(python, subset, predictions, output, model,
                   base_url, embedding_model):
    return [
        python, "-m", "Evaluation.generation_eval",
        "--mode", "API",
        "--model", model,
        "--base_url", base_url,
        "--embedding_model", embedding_model,
        "--data_file", str(predictions),
        "--output_file", str(output),
    ]


def run(subset="novel", predictions=None, output=None, model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        embedding_model="BAAI/bge-large-en-v1.5",
        bench_repo=None, bench_python=None):
    repo = Path(bench_repo or config.BENCH_REPO or "")
    if not repo.exists():
        raise SystemExit("set --bench-repo or BRANDKG_BENCH_REPO to the cloned GraphRAG-Benchmark path")
    if not os.getenv("LLM_API_KEY"):
        raise SystemExit("LLM_API_KEY (OpenAI, eval-only) is not set")
    base = Path(config.ROOT) / "bench" / "graphrag_bench"
    preds = Path(predictions or base / f"predictions_{subset}.json").resolve()
    if not preds.exists():
        raise SystemExit(f"predictions file not found: {preds} (run `graphrag-bench` first)")
    out = Path(output or base / f"results_{subset}.json").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    python = bench_python or config.BENCH_PYTHON or sys.executable
    cmd = build_eval_cmd(python, subset, preds, out, model, base_url, embedding_model)
    print("running:", " ".join(cmd))
    print("cwd:", repo)
    rc = subprocess.call(cmd, cwd=str(repo))
    if rc == 0:
        print(f"eval complete -> {out}")
    return rc


def main(argv=None):
    p = argparse.ArgumentParser(prog="run.py graphrag-eval")
    p.add_argument("--subset", required=True, choices=["novel", "medical"])
    p.add_argument("--predictions", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--base_url", default="https://api.openai.com/v1")
    p.add_argument("--embedding_model", default="BAAI/bge-large-en-v1.5")
    p.add_argument("--bench-repo", default=None)
    p.add_argument("--bench-python", default=None)
    a = p.parse_args(argv)
    return run(subset=a.subset, predictions=a.predictions, output=a.output,
               model=a.model, base_url=a.base_url, embedding_model=a.embedding_model,
               bench_repo=a.bench_repo, bench_python=a.bench_python)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_eval_cmd.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add brandkg/graphrag_bench/evaluate.py tests/test_eval_cmd.py
git commit -m "feat: graphrag-eval wrapper around benchmark generation_eval"
```

---

## Task 9: CLI wiring

**Files:**
- Modify: `brandkg/cli.py`
- Test: `tests/test_runner.py` (append a dispatch test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_runner.py`:
```python
def test_cli_dispatches_graphrag_bench(monkeypatch):
    from brandkg import cli
    called = {}
    import brandkg.graphrag_bench.runner as r
    monkeypatch.setattr(r, "main", lambda argv: called.setdefault("argv", argv) or 0)
    rc = cli.main(["graphrag-bench", "--subset", "novel", "--sample", "3"])
    assert rc == 0
    assert called["argv"] == ["--subset", "novel", "--sample", "3"]


def test_cli_dispatches_graphrag_eval(monkeypatch):
    from brandkg import cli
    called = {}
    import brandkg.graphrag_bench.evaluate as ev
    monkeypatch.setattr(ev, "main", lambda argv: called.setdefault("argv", argv) or 0)
    rc = cli.main(["graphrag-eval", "--subset", "novel"])
    assert rc == 0
    assert called["argv"] == ["--subset", "novel"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/pytest tests/test_runner.py -k graphrag -q`
Expected: FAIL — `cli.main(["graphrag-bench", ...])` prints usage and returns 1 (unknown command), so `rc == 0` assertion fails.

- [ ] **Step 3: Add dispatch in `cli.py`**

In `brandkg/cli.py`, inside `main()`, add these branches before the final `else:`:
```python
    elif cmd == "graphrag-bench":
        from .graphrag_bench import runner
        return runner.main(rest)
    elif cmd == "graphrag-eval":
        from .graphrag_bench import evaluate
        return evaluate.main(rest)
```

Also update the module docstring command list (after the `bench` line) to include:
```
  graphrag-bench     build a GraphRAG-Bench subset with the generic pipeline ->
                     predictions JSON (--subset novel|medical [--sample N --corpus-limit K])
  graphrag-eval      run GraphRAG-Bench's generation_eval over the predictions
                     (needs the bench repo + LLM_API_KEY for the gpt-4o-mini judge)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/bin/pytest tests/test_runner.py -k graphrag -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the whole suite**

Run: `./venv/bin/pytest tests/ -q`
Expected: all PASS; Neo4j integration test passes or skips. No failures.

- [ ] **Step 6: Commit**

```bash
git add brandkg/cli.py tests/test_runner.py
git commit -m "feat: wire graphrag-bench + graphrag-eval CLI commands"
```

---

## Task 10: Docs + setup notes

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a GraphRAG-Bench section to `README.md`**

Append this section to `README.md`:
```markdown
## Benchmark on GraphRAG-Bench

Score this pipeline against the public GraphRAG-Bench leaderboard (LightRAG,
HippoRAG2, etc.) on its fixed novel/medical corpus, using a generic (non-brand)
schema so the extraction fits arbitrary documents.

```bash
# 1. clone the benchmark repo (separately; it has heavy eval deps)
git clone https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.git
export BRANDKG_BENCH_REPO=/abs/path/to/GraphRAG-Benchmark

# 2. build a small sample + emit predictions (uses your claude/codex engine)
python run.py graphrag-bench --subset novel --corpus-limit 1 --sample 30

# 3. score with the benchmark's own gpt-4o-mini judge (eval-only OpenAI key)
#    install the benchmark's requirements into ITS OWN venv first:
#      python -m venv /path/bench-venv && /path/bench-venv/bin/pip install -r $BRANDKG_BENCH_REPO/requirements.txt
export BRANDKG_BENCH_PYTHON=/path/bench-venv/bin/python
export LLM_API_KEY=sk-...      # OpenAI, benchmark judge only
python run.py graphrag-eval --subset novel
```

Outputs: `bench/graphrag_bench/predictions_<subset>.json` and
`bench/graphrag_bench/results_<subset>.json`. The brand `build`/`query` commands
are unaffected; the benchmark uses `config/generic_schema.json`.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: GraphRAG-Bench benchmarking instructions"
```

---

## Self-Review

**Spec coverage:**
- Generalize schema → Task 3 (`generic_schema.json`, selectable) + Task 4 (generic prompt). ✓
- Reuse real loop → Task 2 (`run_loop`, brand `answer()` preserved). ✓
- `answer_with_context` (brand + generic) → Task 2 + Task 6. ✓
- Per-corpus isolation via `corpus_name` → Task 5 (load/wipe/tools all scoped) + isolation test. ✓
- Generic graph builder, query tools, runner, eval wrapper, CLI, config → Tasks 5–9. ✓
- Prediction format exactly matches the contract `{id,question,source,context,evidence,question_type,generated_answer,ground_truth}` → Task 7 test asserts the field set. ✓
- Eval in bench repo's own env via `BENCH_PYTHON`, OpenAI judge, fail-fast on missing key/repo → Task 8. ✓
- Sample-first default documented → Task 10. ✓
- Deferred: `indexing_eval` (phase 2), `retrieval_eval` (optional) — intentionally not tasked. ✓
- Brand pipeline untouched → only `query.py` refactor (regression test in Task 2 Step 6) + additive config/extract branches. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full content. ✓

**Type/name consistency:** `run_loop(engine, session, system, question, tools, max_steps, verbose, abstain)` used identically in Tasks 2, 6. `extract_blob(engine, schema, doc, text)` defined in Task 4, called in Task 7. `gg.driver/wipe_corpus/load/ensure_index` defined in Task 5, used in Tasks 6, 7. Node property is `corpus_name` everywhere (builder, tools, tests). `answer_with_context(question, corpus, session=None)` consistent between Task 6 def and Task 7 call. `build_eval_cmd` signature matches between Task 8 def and test. ✓
