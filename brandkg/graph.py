"""Neo4j graph construction: scaffold, bucketed load, people hierarchy, retier,
text layer, verification. Ported from the validated reference implementation and
parameterized entirely by config/brand_schema.json (brand-agnostic).

Node-type policy: the ONLY labels are Brand + the 7 bucket names + RoleTier. Data
nodes carry only their bucket label; descriptive type is a property. Brand connects
to exactly the 7 bucket nodes — nothing else. (No raw-text/TextUnit layer.)
"""
from __future__ import annotations
import json
import re
from pathlib import Path

from neo4j import GraphDatabase

from . import config

SCHEMA = config.schema()
BUCKETS = SCHEMA["buckets"]
TIERS = SCHEMA["seniority_tiers"]
ALLOWED = set(SCHEMA["allowed_cross_child_rels"])
BRAND = SCHEMA["brand_name"]
BRAND_KEYS = {BRAND.lower()} | {a.lower() for a in SCHEMA.get("brand_aliases", [])}
BUCKET_PRIORITY = ["People", "Partnerships", "Product", "Engineering", "Marketing", "Audience", "Business"]
RANK = {b: i for i, b in enumerate(BUCKET_PRIORITY)}
TIER_RANK = {t: i for i, t in enumerate(TIERS, 1)}

_NON_PERSON = re.compile(r"(?i)\b(role|team|console|admin|viewer|editor|developer|owner|user|"
                         r"persona|account|group|dept|department|committee|board|panel|squad|pod)\b")
_TITLE_RULES = [
    (r"\b(founder|co[- ]?founder|founding partner)\b", "FOUNDER"),
    (r"\b(ceo|cto|cfo|cmo|coo|cio|ciso|chief|president|chairman|chairperson)\b", "C_SUITE"),
    (r"\b(head of|head,|global head|general manager|gm)\b", "C_SUITE"),
    (r"\b(svp|evp|executive vice president|senior vice president)\b", "SVP"),
    (r"\b(vp|vice president|chief of staff)\b", "VP"),
    (r"\b(director|group director)\b", "DIRECTOR"),
    (r"\b(principal|staff)\b", "DIRECTOR"),
    (r"\b(manager|senior manager|lead|team lead)\b", "MANAGER"),
    (r"\b(advisor|adviser|board member|independent director)\b", "ADVISOR"),
    (r"\b(engineer|developer|analyst|designer|associate|intern|sde|qa|writer|scientist|"
     r"architect|specialist|coordinator|representative|devops|hr)\b", "IC"),
]


def driver():
    return GraphDatabase.driver(config.NEO4J_URI, auth=config.NEO4J_AUTH)


def _nk(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _is_person(name):
    n = (name or "").strip()
    if not n or _NON_PERSON.search(n) or re.search(r"\d", n):
        return False
    return n[0].isupper() and 1 <= len(n.split()) <= 4


# ---------------------------------------------------------------- scaffold
def build_scaffold(s):
    s.run("CREATE CONSTRAINT brand_name IF NOT EXISTS FOR (b:Brand) REQUIRE b.name IS UNIQUE")
    s.run("CREATE CONSTRAINT tier_name IF NOT EXISTS FOR (t:RoleTier) REQUIRE t.tier IS UNIQUE")
    for b in BUCKETS:
        s.run(f"CREATE CONSTRAINT key_{b.lower()} IF NOT EXISTS FOR (n:`{b}`) REQUIRE n.name_key IS UNIQUE")
    s.run("MERGE (b:Brand {name:$n}) SET b.description=$d, b.is_root=true, b.load_count=coalesce(b.load_count,0)",
          n=BRAND, d=SCHEMA["brand_attrs"].get("description", ""))
    for b in BUCKETS:
        rel = SCHEMA["brand_to_bucket_rel"][b]
        s.run(f"MATCH (br:Brand {{name:$n}}) MERGE (c:`{b}` {{name:$b, kind:'bucket'}}) "
              f"SET c.layer=1 MERGE (br)-[:`{rel}`]->(c)", n=BRAND, b=b)
    for rank, t in enumerate(TIERS, 1):
        s.run("MERGE (t:RoleTier {tier:$t}) SET t.rank=$r", t=t, r=rank)


# ---------------------------------------------------------------- load
def _valid_bucket(b):
    return b if b in BUCKETS else "Business"


def _resolve_buckets(s, payloads):
    proposed = {}
    for p in payloads:
        for e in p.get("entities", []):
            nk = _nk(e.get("name"))
            if not nk or nk in BRAND_KEYS:
                continue
            b = _valid_bucket(e.get("bucket", ""))
            if b == "People" and not _is_person(e.get("name", "")):
                b = "Business"
            if nk not in proposed or RANK[b] < RANK[proposed[nk]]:
                proposed[nk] = b
    existing = {}
    for b in BUCKETS:
        for r in s.run(f"MATCH (n:`{b}` {{layer:2}}) RETURN n.name_key AS k"):
            existing[r["k"]] = b
    return {nk: existing.get(nk, b) for nk, b in proposed.items()}


_NODE = """
MATCH (cat:`__BK__` {name:$bucket, kind:'bucket'})
MERGE (e:`__BK__` {name_key:$key})
ON CREATE SET e.name=$name, e.created_at=$ts, e.sources=[], e.aliases=[], e.layer=2
SET e.name=coalesce(e.name,$name), e.type=$type,
    e.description = CASE WHEN $desc='' THEN e.description
        WHEN e.description IS NULL OR size($desc)>size(coalesce(e.description,'')) THEN $desc
        ELSE e.description END,
    e.aliases = coalesce(e.aliases,[]) + [a IN $aliases WHERE NOT a IN coalesce(e.aliases,[])],
    e.sources = coalesce(e.sources,[]) + [x IN [$doc] WHERE NOT x IN coalesce(e.sources,[])],
    e.updated_at=$ts
MERGE (cat)-[:CONTAINS]->(e)
"""
_PEOPLE = """
MATCH (e:`People` {name_key:$key})
SET e.seniority_tier=CASE WHEN $tier='' THEN e.seniority_tier ELSE $tier END,
    e.affiliation=CASE WHEN $aff='' THEN e.affiliation ELSE $aff END,
    e.role_title=CASE WHEN $role='' THEN e.role_title ELSE $role END
WITH e OPTIONAL MATCH (t:RoleTier {tier:$tier})
FOREACH (_ IN CASE WHEN t IS NULL THEN [] ELSE [1] END | MERGE (e)-[:AT_TIER]->(t))
"""
_REL = """
MATCH (a:`__SB__` {name_key:$sk}), (b:`__TB__` {name_key:$tk})
MERGE (a)-[x:`__REL__`]->(b)
ON CREATE SET x.created_at=$ts, x.sources=[]
SET x.description=CASE WHEN $desc='' THEN x.description
        WHEN x.description IS NULL OR size($desc)>size(coalesce(x.description,'')) THEN $desc
        ELSE x.description END,
    x.sources=coalesce(x.sources,[]) + [s IN [$doc] WHERE NOT s IN coalesce(x.sources,[])],
    x.updated_at=$ts
"""


def load_entities(s, payloads, ts):
    bucket_of = _resolve_buckets(s, payloads)
    te = tr = 0
    for p in payloads:
        doc = p.get("source_doc", "doc")
        for e in p.get("entities", []):
            nk = _nk(e.get("name")); name = (e.get("name") or "").strip()
            if not name or nk in BRAND_KEYS:
                continue
            bucket = bucket_of.get(nk, _valid_bucket(e.get("bucket", "")))
            aliases = sorted({a.strip() for a in e.get("aliases", []) if a and a.strip()})
            s.run(_NODE.replace("__BK__", bucket), key=nk, name=name, type=e.get("type", ""),
                  bucket=bucket, desc=(e.get("description") or "").strip(), aliases=aliases, doc=doc, ts=ts)
            if bucket == "People":
                tier = e.get("seniority_tier", "") or ""
                tier = tier if tier in TIERS else ("OTHER" if tier else "")
                s.run(_PEOPLE, key=nk, tier=tier, aff=(e.get("affiliation") or "").strip(),
                      role=(e.get("role_title") or "").strip())
            te += 1
        for r in p.get("relationships", []):
            sk, tk = _nk(r.get("source")), _nk(r.get("target"))
            if not sk or not tk or sk in BRAND_KEYS or tk in BRAND_KEYS:
                continue
            sb, tb = bucket_of.get(sk), bucket_of.get(tk)
            if not sb or not tb:
                continue
            rel = re.sub(r"[^A-Za-z0-9]+", "_", (r.get("type") or "RELATES_TO").upper()).strip("_") or "RELATES_TO"
            if rel not in ALLOWED:
                rel = "RELATES_TO"
            s.run(_REL.replace("__SB__", sb).replace("__TB__", tb).replace("__REL__", rel),
                  sk=sk, tk=tk, desc=(r.get("description") or "").strip(), doc=doc, ts=ts)
            tr += 1
    # single-home reconcile
    for row in list(s.run("MATCH (c)-[:CONTAINS]->(e) WHERE e.layer=2 "
                          "WITH e, collect(c.name) AS bs WHERE size(bs)>1 RETURN e.name_key AS k, bs")):
        primary = sorted(row["bs"], key=lambda b: RANK.get(b, 99))[0]
        s.run("MATCH (c)-[r:CONTAINS]->(e {name_key:$k}) WHERE c.name<>$p DELETE r", k=row["k"], p=primary)
        s.run("MATCH (e {name_key:$k}) SET e.bucket=$p", k=row["k"], p=primary)
    # brand invariant guard
    s.run("MATCH (b:Brand)-[r]-(x) WHERE x.kind<>'bucket' OR x.kind IS NULL DELETE r")
    return te, tr


# ---------------------------------------------------------------- retier + hierarchy
def retier_people(s):
    changed = 0
    for p in list(s.run("MATCH (p:People {layer:2}) RETURN p.name_key AS k, p.role_title AS role, p.seniority_tier AS tier")):
        title = (p["role"] or "").lower()
        new = next((tier for pat, tier in _TITLE_RULES if re.search(pat, title)), None)
        if not new:
            continue
        old = p["tier"]
        if old is None or old == "OTHER" or TIER_RANK.get(new, 99) < TIER_RANK.get(old, 99):
            if new != old:
                s.run("MATCH (p:People {name_key:$k}) SET p.seniority_tier=$t "
                      "WITH p MATCH (rt:RoleTier {tier:$t}) "
                      "OPTIONAL MATCH (p)-[o:AT_TIER]->() DELETE o MERGE (p)-[:AT_TIER]->(rt)",
                      k=p["k"], t=new)
                changed += 1
    return changed


def build_hierarchy(s):
    s.run("MATCH (p:People {layer:2}) WHERE p.seniority_tier IS NOT NULL "
          "MATCH (t:RoleTier {tier:p.seniority_tier}) MERGE (p)-[:AT_TIER]->(t)")
    people = list(s.run("MATCH (p:People {layer:2}) WHERE p.affiliation IS NOT NULL "
                        "AND p.seniority_tier IS NOT NULL RETURN p.name_key AS k, p.affiliation AS aff, p.seniority_tier AS tier"))
    by_aff = {}
    for r in people:
        by_aff.setdefault(r["aff"], []).append((r["k"], r["tier"]))
    for aff, members in by_aff.items():
        tier_of = dict(members)
        for k, tier in members:
            cands = [mk for mk, mt in members if TIER_RANK.get(mt, 99) < TIER_RANK.get(tier, 99)]
            if not cands:
                continue
            mgr = sorted(cands, key=lambda mk: TIER_RANK.get(tier_of[mk], 99))[-1]
            s.run("MATCH (p:People {name_key:$pk}),(m:People {name_key:$mk}) "
                  "WHERE NOT (p)-[:REPORTS_TO]->(:People) AND p<>m "
                  "MERGE (p)-[r:REPORTS_TO]->(m) ON CREATE SET r.inferred=true", pk=k, mk=mgr)


# ---------------------------------------------------------------- entity search index
def ensure_entity_index(s):
    """Full-text index over entity name+description for query retrieval.

    (No TextUnit/raw-text layer is created — the graph holds only Brand, the 7
    buckets, and RoleTier. Queries retrieve from entities + relationships.)
    """
    bucket_list = "|".join(BUCKETS)
    try:
        s.run(f"CREATE FULLTEXT INDEX entityFulltext IF NOT EXISTS "
              f"FOR (n:{bucket_list}) ON EACH [n.name, n.description]")
    except Exception:
        pass


# ---------------------------------------------------------------- verify
def verify(s) -> dict:
    one = lambda q: s.run(q).single()[0]
    labels = sorted(r["label"] for r in s.run("CALL db.labels() YIELD label RETURN label"))
    allowed = set(["Brand", "RoleTier"] + BUCKETS)
    return {
        "labels": labels,
        "unexpected_labels": [l for l in labels if l not in allowed],
        "brand_edges": one("MATCH (:Brand)-[r]-() RETURN count(r)"),
        "entities": one("MATCH (n) WHERE n.layer=2 RETURN count(n)"),
        "rels": one("MATCH (a)-[r]->(b) WHERE a.layer=2 AND b.layer=2 RETURN count(r)"),
        "orphans": one("MATCH (e) WHERE e.layer=2 AND NOT (e)<-[:CONTAINS]-() RETURN count(e)"),
        "beyond_2_hops": one("MATCH (b:Brand),(e) WHERE e.layer=2 AND NOT (b)-[*..2]-(e) RETURN count(e)"),
        "per_bucket": {b: one(f"MATCH (n:`{b}` {{layer:2}}) RETURN count(n)") for b in BUCKETS},
    }


def wipe(s):
    s.run("MATCH (n) DETACH DELETE n")
    for c in list(s.run("SHOW CONSTRAINTS YIELD name RETURN name")):
        s.run(f"DROP CONSTRAINT `{c['name']}` IF EXISTS")
