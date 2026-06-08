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


def corpus_entity_count(s, corpus: str) -> int:
    """How many entities exist for a corpus (0 = not built yet). Used for resume."""
    return s.run("MATCH (n:Entity {corpus_name:$c}) RETURN count(n) AS c", c=corpus).single()["c"]


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
