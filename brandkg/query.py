"""GraphRAG-style Q&A over the brand KG, with the configured engine doing synthesis.

Brand-agnostic: brand/buckets/tiers come from config. Retrieval = intent routing
(leadership->senior tiers, bucket words->that bucket) + full-text entity search +
1-hop relationship expansion. The engine (claude/codex)
synthesizes the final answer from the retrieved context — no API key.
"""
from __future__ import annotations
import json

from . import config, graph
from .engines import get_engine
from .skill import skill_body, skill_name

SCHEMA = config.schema()
BRAND = SCHEMA["brand_name"]
BUCKETS = SCHEMA["buckets"]
SENIOR = [t for t in SCHEMA["seniority_tiers"] if t not in ("IC", "ADVISOR", "OTHER")]
BRAND_WORDS = {BRAND.lower()} | {w for a in [BRAND] + SCHEMA.get("brand_aliases", []) for w in a.lower().split()}

INTENT = {
    "People": ["leader", "leadership", "executive", "ceo", "cto", "cfo", "cmo", "founder",
               "who runs", "who leads", "management", "head of", "staff", "employee"],
    "Product": ["product", "offer", "service", "feature", "app", "tool", "platform", "capabilit"],
    "Partnerships": ["partner", "partnership", "alliance", "sponsor"],
    "Engineering": ["sdk", "api", "tech", "stack", "infrastructure", "engineering", "built", "architecture", "compatible", "integrate"],
    "Marketing": ["campaign", "marketing", "channel", "activation", "advertis"],
    "Audience": ["audience", "segment", "customer", "user", "metric", "reach", "competitor"],
    "Business": ["revenue", "funding", "business", "operation", "legal", "financ", "headquarter", "founded", "office"],
}
LEADER = ["leader", "leadership", "executive", "who runs", "who leads", "management", "ceo", "cto", "cfo", "cmo", "founder", "head of"]

ANSWER_SYSTEM = f"""You answer questions about the brand "{BRAND}" using ONLY the provided knowledge-graph context (entities, relationships, source excerpts).
- Use ONLY facts in the context; never use outside knowledge.
- If the context lacks the answer, reply exactly: "I don't have enough information to answer this from the knowledge graph."
- Cite entity names in bold. Keep factual answers brief; list for "what/which/who" questions."""


def _keywords(q):
    import re
    stop = set("is the a an of in on at to was were are do does did when where who what which how there any it its that this with for and or not from by be been has have had can could will would should may might about tell me please mentioned".split())
    return [w for w in re.findall(r"[a-zA-Z0-9]+", q.lower()) if w not in stop and w not in BRAND_WORDS][:6]


def _route(q):
    ql = q.lower()
    return [b for b, hints in INTENT.items() if any(h in ql for h in hints)]


def answer(question: str) -> str:
    """Dispatch to the configured retrieval method (graphrag = Claude-generated
    Cypher; keyword = deterministic routing)."""
    if config.QUERY_METHOD == "graphrag":
        from . import query_graphrag
        return query_graphrag.answer(question)
    return answer_keyword(question)


def answer_keyword(question: str) -> str:
    engine = get_engine(config.ENGINE)
    drv = graph.driver()
    with drv.session() as s:
        routes, ql = _route(question), question.lower()
        seeds = []
        if "People" in routes and any(w in ql for w in LEADER):
            seeds += [r.data() for r in s.run(
                "MATCH (p:People {layer:2}) WHERE p.seniority_tier IN $t "
                "RETURN p.name AS name, 'People' AS bucket, p.role_title AS role_title, p.description AS description LIMIT 25", t=SENIOR)]
        for b in routes:
            if b != "People":
                seeds += [r.data() for r in s.run(
                    f"MATCH (n:`{b}` {{layer:2}}) RETURN n.name AS name, '{b}' AS bucket, n.type AS type, n.description AS description LIMIT 20")]
        kws = _keywords(question)
        if kws:
            terms = " OR ".join(kws)
            try:
                seeds += [r.data() for r in s.run(
                    "CALL db.index.fulltext.queryNodes('entityFulltext',$q) YIELD node,score WHERE node.layer=2 "
                    "RETURN node.name AS name, labels(node)[0] AS bucket, node.type AS type, node.description AS description ORDER BY score DESC LIMIT 15", q=terms)]
            except Exception:
                pass
        seen, uniq = set(), []
        for r in seeds:
            if r.get("name") and r["name"] not in seen:
                seen.add(r["name"]); uniq.append(r)
        seeds = uniq[:30]
        names = [r["name"] for r in seeds][:18]
        rels = [r.data() for r in s.run(
            "MATCH (a)-[r]-(b) WHERE a.layer=2 AND a.name IN $names AND b.layer=2 "
            "RETURN a.name AS source, type(r) AS rel, b.name AS target LIMIT 40", names=names)] if names else []
    drv.close()
    ctx = (f"You are running the `{skill_name()}` skill; follow its query guidance.\n\n"
           f"===== GRAPHITI SKILL (source of truth) =====\n{skill_body()}\n"
           f"=============================================\n\n"
           f"Question: {question}\n\n## Entities ({len(seeds)})\n{json.dumps(seeds, indent=2, default=str)}\n\n"
           f"## Relationships ({len(rels)})\n{json.dumps(rels, indent=2, default=str)}\n\n"
           f"{ANSWER_SYSTEM}\n\nAnswer using ONLY the context above.")
    return engine.complete(ctx, timeout=300)
