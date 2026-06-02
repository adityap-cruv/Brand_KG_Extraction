"""GraphRAG-style local search where CLAUDE drives retrieval.

Reimplemented from scratch (no external reference). Flow:
  1. Introspect the live Neo4j schema (labels, relationship types, sample props).
  2. CLAUDE generates a Cypher query from the question + schema (engine/OAuth).
  3. Execute it -> seed rows. (Fallback to keyword search if it errors/empties.)
  4. Expand: pull 1-hop relationships around the seed entities.
  5. CLAUDE synthesizes the final answer from seeds + relationships.

Everything LLM is done via the configured engine (claude/codex) — no API key.
This is the "completely engine-driven" retrieval path; contrast with query.py
(deterministic keyword routing).
"""
from __future__ import annotations
import json
import re

from . import config, graph
from .engines import get_engine
from .skill import skill_body, skill_name

SCHEMA = config.schema()
BRAND = SCHEMA["brand_name"]
BUCKETS = SCHEMA["buckets"]

CYPHER_SYSTEM = """You translate a question into ONE Neo4j Cypher query that finds the most relevant nodes to answer it.

GRAPH SCHEMA
------------
Brand root: (:Brand {name})  -- the brand itself; do NOT search it for facts.
Seven category buckets (each a node with kind:'bucket'): %(buckets)s
Data nodes carry exactly ONE bucket label and have layer=2, with properties:
  name, type, description, aliases (list), sources (list)
People data nodes also have: role_title, seniority_tier, affiliation.
Relationships between data nodes are typed verbs, e.g.:
  %(rels)s
Membership: (bucket)-[:CONTAINS]->(dataNode). People: (p)-[:REPORTS_TO]->(p), (p)-[:AT_TIER]->(:RoleTier).

RULES
- Output ONLY the Cypher, no prose, no markdown fences.
- Facts live in the `description` property — search name, aliases, AND description.
- Use case-insensitive matching: toLower(n.property) CONTAINS toLower("...")
- To gather related context, you MAY match a node and its 1-hop neighbours.
- Always RETURN n.name, labels(n)[0] AS bucket, n.type, n.description. LIMIT 25.
- Cast a WIDE net: if the question is about a capability/feature/platform, search
  descriptions for the key terms, not just names. Prefer returning too much over nothing.

EXAMPLES
Q: "What platforms is the SDK compatible with?"
MATCH (n) WHERE n.layer=2 AND (toLower(n.name) CONTAINS "sdk"
  OR toLower(coalesce(n.description,"")) CONTAINS "sdk"
  OR toLower(coalesce(n.description,"")) CONTAINS "ios"
  OR toLower(coalesce(n.description,"")) CONTAINS "android"
  OR toLower(coalesce(n.description,"")) CONTAINS "flutter"
  OR toLower(coalesce(n.description,"")) CONTAINS "react")
RETURN n.name, labels(n)[0] AS bucket, n.type, n.description LIMIT 25

Q: "Who founded the company?"
MATCH (p:People {layer:2}) WHERE toLower(coalesce(p.role_title,"")) CONTAINS "found"
  OR toLower(coalesce(p.description,"")) CONTAINS "found" OR p.seniority_tier="FOUNDER"
RETURN p.name, labels(p)[0] AS bucket, p.type, p.description LIMIT 25
"""

ANSWER_SYSTEM = """Answer the question using ONLY the provided knowledge-graph context (entities + their relationships).
- Use ONLY facts in the context; never use outside knowledge.
- If the context genuinely lacks the answer, reply exactly:
  "I don't have enough information to answer this from the knowledge graph."
- Be concise and specific; cite entity names in bold."""


def _engine():
    return get_engine(config.ENGINE)


def _strip_fences(c: str) -> str:
    c = c.strip()
    if c.startswith("```"):
        nl = c.find("\n")
        c = c[nl + 1:] if nl != -1 else c[3:]
        if c.rstrip().endswith("```"):
            c = c.rstrip()[:-3]
    return c.strip()


def _keywords(q):
    stop = set("is the a an of in on at to was were are do does did when where who what which how "
               "there any it its that this with for and or not from by be been has have had can could "
               "will would should may might about tell me please".split())
    bw = {BRAND.lower()} | {w for a in [BRAND] + SCHEMA.get("brand_aliases", []) for w in a.lower().split()}
    return [w for w in re.findall(r"[a-zA-Z0-9]+", q.lower()) if w not in stop and w not in bw][:6]


def _gen_cypher(engine, question: str) -> str:
    sysmsg = CYPHER_SYSTEM % {"buckets": ", ".join(BUCKETS),
                             "rels": ", ".join(SCHEMA["allowed_cross_child_rels"][:18])}
    prompt = f"{sysmsg}\n\nQuestion: {question}\n\nCypher:"
    return _strip_fences(engine.complete(prompt, timeout=120))


def _keyword_fallback(s, kws):
    if not kws:
        return []
    cond = " OR ".join(
        f"toLower(n.name) CONTAINS '{k}' OR toLower(coalesce(n.description,'')) CONTAINS '{k}'"
        for k in kws)
    return [r.data() for r in s.run(
        f"MATCH (n) WHERE n.layer=2 AND ({cond}) "
        "RETURN n.name AS name, labels(n)[0] AS bucket, n.type AS type, n.description AS description LIMIT 20")]


def _expand(s, names):
    if not names:
        return []
    return [r.data() for r in s.run(
        "MATCH (a)-[r]-(b) WHERE a.layer=2 AND a.name IN $names AND b.layer=2 "
        "RETURN a.name AS source, type(r) AS rel, b.name AS target LIMIT 40", names=names)]


def answer(question: str, verbose: bool = False) -> str:
    engine = _engine()
    cypher = _gen_cypher(engine, question)
    if verbose:
        print(f"[cypher]\n{cypher}\n")
    drv = graph.driver()
    seeds = []
    with drv.session() as s:
        # run Claude's Cypher; tolerate errors
        try:
            seeds = [r.data() for r in s.run(cypher)]
        except Exception as e:
            if verbose:
                print(f"[cypher error] {e}")
        # normalize/extract names from whatever columns came back
        def names_of(rows):
            out = set()
            for row in rows:
                for k, v in row.items():
                    if isinstance(v, str) and (k.endswith("name") or k == "name"):
                        out.add(v)
            return list(out)
        # fallback if Claude's query returned nothing
        if not seeds:
            seeds = _keyword_fallback(s, _keywords(question))
            if verbose:
                print(f"[fallback] {len(seeds)} rows")
        names = names_of(seeds)[:18]
        rels = _expand(s, names)
    drv.close()
    if verbose:
        print(f"[seeds] {len(seeds)}  [rels] {len(rels)}")
    ctx = (f"You are running the `{skill_name()}` skill; follow its query guidance.\n\n"
           f"===== GRAPHITI SKILL (source of truth) =====\n{skill_body()}\n"
           f"=============================================\n\n"
           f"Question: {question}\n\n## Entities ({len(seeds)})\n{json.dumps(seeds, indent=2, default=str)}\n\n"
           f"## Relationships ({len(rels)})\n{json.dumps(rels, indent=2, default=str)}\n\n"
           f"{ANSWER_SYSTEM}\n\nAnswer using ONLY the context above.")
    return engine.complete(ctx, timeout=300)
