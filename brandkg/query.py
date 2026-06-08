"""Answer questions by letting CLAUDE explore the knowledge graph itself.

No text-matching heuristics, no per-topic handlers, no fixed slice. Claude is
given a small set of read-only graph tools and runs an agentic loop:

    Claude -> {"action": "<tool>", ...}   (it decides what to look up)
    Python -> runs it on Neo4j, returns the result
    ... repeats ...
    Claude -> {"action": "answer", "text": "..."}   (when it can answer)

This generalizes to ANY question — leadership, products, partnerships, metrics —
because Claude drives the exploration. The engine is the configured agent CLI
(claude/codex) over the user's subscription; no API key.
"""
from __future__ import annotations
import json

from . import config, graph
from .engines import get_engine
from .engines.base import extract_json
from .skill import skill_body, skill_name

SCHEMA = config.schema()
BRAND = SCHEMA["brand_name"]
BUCKETS = SCHEMA["buckets"]
MAX_STEPS = 8
ABSTAIN = "I don't have enough information to answer this from the knowledge graph."


# ---------------------------------------------------------------- read-only tools
def _search(s, term, limit=25):
    """Find data nodes whose name/aliases/description/role_title/tier match a term."""
    t = (term or "").lower()
    rows = s.run(
        "MATCH (n) WHERE n.layer=2 AND ("
        "  toLower(n.name) CONTAINS $t OR toLower(coalesce(n.description,'')) CONTAINS $t "
        "  OR toLower(coalesce(n.role_title,'')) CONTAINS $t "
        "  OR toLower(coalesce(n.seniority_tier,'')) CONTAINS $t "
        "  OR any(a IN coalesce(n.aliases,[]) WHERE toLower(a) CONTAINS $t)) "
        "RETURN n.name AS name, labels(n)[0] AS bucket, n.type AS type, "
        "n.role_title AS role_title, n.seniority_tier AS tier, "
        "substring(coalesce(n.description,''),0,200) AS description "
        "LIMIT $lim", t=t, lim=limit)
    return [r.data() for r in rows]


def _bucket(s, name, limit=60):
    """List the data nodes contained in one of the 7 buckets."""
    if name not in BUCKETS:
        return {"error": f"unknown bucket '{name}'. Buckets: {BUCKETS}"}
    rows = s.run(
        f"MATCH (:`{name}` {{kind:'bucket'}})-[:CONTAINS]->(n) "
        "RETURN n.name AS name, n.type AS type, n.role_title AS role_title, "
        "n.seniority_tier AS tier, substring(coalesce(n.description,''),0,160) AS description "
        f"LIMIT {limit}")
    return [r.data() for r in rows]


def _neighbors(s, name, limit=50):
    """All 1-hop relationships of a named entity (in and out)."""
    rows = s.run(
        "MATCH (a) WHERE a.layer=2 AND toLower(a.name)=toLower($n) "
        "MATCH (a)-[r]-(b) WHERE b.layer=2 OR b:RoleTier "
        "RETURN a.name AS entity, type(r) AS rel, "
        "CASE WHEN startNode(r)=a THEN 'out' ELSE 'in' END AS dir, "
        "coalesce(b.name, b.tier) AS other, labels(b)[0] AS other_label "
        f"LIMIT {limit}", n=name)
    return [r.data() for r in rows]


def _tier(s, tier, limit=60):
    """List people at a given seniority tier (FOUNDER, C_SUITE, VP, ...)."""
    rows = s.run(
        "MATCH (p:People {layer:2}) WHERE toLower(coalesce(p.seniority_tier,''))=toLower($t) "
        "RETURN p.name AS name, p.role_title AS role_title, p.seniority_tier AS tier "
        f"LIMIT {limit}", t=tier)
    return [r.data() for r in rows]


def _run_tool(s, action: dict):
    a = (action.get("action") or "").lower()
    if a == "search":
        return _search(s, action.get("term", ""))
    if a == "bucket":
        return _bucket(s, action.get("name", ""))
    if a == "neighbors":
        return _neighbors(s, action.get("name", ""))
    if a == "tier":
        return _tier(s, action.get("tier", ""))
    return {"error": f"unknown action '{a}'. Use search|bucket|neighbors|tier|answer."}


SYSTEM = """You answer a question about the brand "%(brand)s" by EXPLORING its knowledge graph.

GRAPH SHAPE
- One Brand root connects to 7 category buckets: %(buckets)s
- Every data node lives in exactly one bucket and has: name, type, description, aliases, plus
  People also have role_title and seniority_tier (FOUNDER > C_SUITE > SVP > VP > DIRECTOR > MANAGER > IC > ADVISOR > OTHER).
- Nodes are linked by typed relationships (e.g. CONTAINS, REPORTS_TO, AT_TIER, USES, POWERS, PARTNERS_WITH, OFFERS).

YOU HAVE THESE READ-ONLY TOOLS. Respond with ONE JSON object per turn, nothing else:
- {"action":"search","term":"<text>"}        find nodes matching text in name/description/role/tier/alias
- {"action":"bucket","name":"<Bucket>"}      list everything in a bucket (one of the 7)
- {"action":"neighbors","name":"<Entity>"}   a node's relationships
- {"action":"tier","tier":"<TIER>"}          people at a seniority tier (e.g. FOUNDER, C_SUITE)
- {"action":"answer","text":"<final answer>"} when you can answer

RULES
- Explore as many steps as needed (you'll be given each tool result, then asked for the next action).
- Base the final answer ONLY on what the tools return. Do not use outside knowledge.
- If after exploring the graph genuinely lacks the answer, answer EXACTLY:
  "I don't have enough information to answer this from the knowledge graph."
- Prefer specific facts; cite entity names in **bold**.

Follow the `%(skill)s` skill's guidance where relevant:
%(skillbody)s
"""


def run_loop(engine, session, system, question, tools,
             max_steps=MAX_STEPS, verbose=False, abstain=ABSTAIN):
    """Drive the agentic loop. `tools` maps action name -> fn(session, action) -> result.

    Returns (answer_text, contexts) where contexts is the list of stringified tool
    results the model saw (used as retrieved-context for benchmarking/RAGAS).
    """
    transcript = f"{system}\n\nQUESTION: {question}\n\nYour first action (JSON only):"
    final = abstain
    contexts = []

    def _run_one(action):
        act = (action.get("action") or "").lower() if isinstance(action, dict) else ""
        if act == "answer":
            return act, action.get("text", final)
        fn = tools.get(act)
        if fn is None:
            return act, {"error": f"unknown action '{act}'. Use {list(tools)}|batch|answer."}
        return act, fn(session, action)

    for step in range(max_steps):
        raw = engine.complete(transcript, timeout=180)
        try:
            action = extract_json(raw)
        except Exception:
            transcript += "\n\n[your reply could not be parsed as JSON; reply with ONE JSON action]"
            continue
        if verbose:
            print(f"[step {step+1}] {action}")

        if isinstance(action, dict) and (action.get("action") or "").lower() == "batch":
            calls = action.get("calls", [])
        elif isinstance(action, list):
            calls = action
        else:
            calls = [action]

        results = []
        for call in calls[:8]:
            act, result = _run_one(call)
            if act == "answer":
                final = result
                break
            results.append({"action": call, "result": result})
        if final != abstain:
            break

        blob = json.dumps(results[0]["result"] if len(results) == 1 else results,
                          default=str)[:12000]
        contexts.append(blob)
        transcript += (f"\n\nACTION: {json.dumps(action)}\nRESULT: {blob}\n\n"
                       f"Next action (JSON only — use 'answer' when ready):")
    return final, contexts


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
