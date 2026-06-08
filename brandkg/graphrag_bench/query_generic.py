"""Generic agentic query over a corpus-scoped graph.

The benchmark KG mode is deliberately graph-only: the model can search nodes,
inspect neighbors, and traverse multi-hop paths, but it never receives source
document text. If normal exploration exhausts the step budget, we still ask for a
best-effort answer from the graph observations rather than saving the product
query loop's abstain text as a benchmark answer.
"""
from __future__ import annotations
import json
import re

from .. import config
from ..engines import get_engine
from ..query import run_loop, ABSTAIN
from .graph_generic import driver

KG_MAX_STEPS = 10
_STOP = set(
    "the a an of to in and or is are was were be been being by for with on at from as that this "
    "these those which who whom whose what when where how why during according narrative story "
    "novel within context events described into out over under about does did has have had its their "
    "his her him she he they them between among through after before because explain summarize".split()
)


def _search(s, corpus, term, limit=25):
    t = (term or "").lower()
    return s.run(
        "MATCH (n:Entity {corpus_name:$c}) WHERE "
        "  toLower(n.name) CONTAINS $t OR toLower(coalesce(n.description,'')) CONTAINS $t "
        "  OR any(a IN coalesce(n.aliases,[]) WHERE toLower(a) CONTAINS $t) "
        "RETURN n.name AS name, n.type AS type, "
        "substring(coalesce(n.description,''),0,200) AS description "
        "LIMIT $lim", c=corpus, t=t, lim=limit).data()


def _keywords(question, limit=12):
    words = []
    seen = set()
    for word in re.findall(r"[A-Za-z][A-Za-z0-9'-]+", question or ""):
        w = word.lower().strip("'")
        if len(w) <= 3 or w in _STOP or w in seen:
            continue
        seen.add(w)
        words.append(w)
        if len(words) >= limit:
            break
    return words


def _seed_nodes(s, corpus, question, limit=10):
    terms = _keywords(question)
    if not terms:
        return []
    return s.run(
        "MATCH (n:Entity {corpus_name:$c}) "
        "WITH n, $terms AS terms "
        "WITH n, reduce(score=0, term IN terms | score "
        "  + CASE WHEN toLower(n.name) CONTAINS term THEN 6 ELSE 0 END "
        "  + CASE WHEN toLower(coalesce(n.description,'')) CONTAINS term THEN 2 ELSE 0 END "
        "  + CASE WHEN any(a IN coalesce(n.aliases,[]) WHERE toLower(a) CONTAINS term) THEN 5 ELSE 0 END"
        ") AS score "
        "WHERE score > 0 "
        "RETURN n.name_key AS key, n.name AS name, n.type AS type, score, "
        "substring(coalesce(n.description,''),0,360) AS description "
        "ORDER BY score DESC, n.name ASC "
        "LIMIT $lim", c=corpus, terms=terms, lim=limit).data()


def _neighbors(s, corpus, name, limit=50):
    return s.run(
        "MATCH (a:Entity {corpus_name:$c}) WHERE toLower(a.name)=toLower($n) "
        "MATCH (a)-[r]-(b:Entity {corpus_name:$c}) "
        "RETURN a.name AS entity, type(r) AS rel, "
        "CASE WHEN startNode(r)=a THEN 'out' ELSE 'in' END AS dir, "
        "b.name AS other, b.type AS other_type, "
        "substring(coalesce(b.description,''),0,240) AS other_description, "
        "coalesce(r.description,'') AS description "
        "LIMIT $lim", c=corpus, n=name, lim=limit).data()


def _neighbor_rows(s, corpus, keys, limit=80):
    if not keys:
        return []
    return s.run(
        "MATCH (a:Entity {corpus_name:$c})-[r]-(b:Entity {corpus_name:$c}) "
        "WHERE a.name_key IN $keys "
        "RETURN a.name AS entity, type(r) AS rel, "
        "CASE WHEN startNode(r)=a THEN 'out' ELSE 'in' END AS dir, "
        "b.name AS other, b.type AS other_type, "
        "substring(coalesce(b.description,''),0,240) AS other_description, "
        "coalesce(r.description,'') AS description "
        "LIMIT $lim", c=corpus, keys=keys, lim=limit).data()


def _format_path(record):
    nodes = record.get("nodes") or []
    rels = record.get("rels") or []
    parts = []
    for i, node in enumerate(nodes):
        label = node.get("name", "")
        ntype = node.get("type")
        desc = node.get("description")
        if ntype:
            label = f"{label} ({ntype})"
        if desc:
            label = f"{label}: {desc}"
        parts.append(label)
        if i < len(rels):
            rel = rels[i]
            rdesc = rel.get("description")
            edge = rel.get("type", "RELATES_TO")
            parts.append(f"-[{edge}: {rdesc}]-" if rdesc else f"-[{edge}]-")
    return " ".join(parts)


def _bounded_int(value, default, low, high):
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = default
    return max(low, min(n, high))


def _paths(s, corpus, start, end="", depth=3, limit=15):
    """Return multi-hop paths from a named start entity, optionally toward another term."""
    depth = _bounded_int(depth, 3, 1, 4)
    limit = _bounded_int(limit, 15, 1, 30)
    end = (end or "").lower()
    if end:
        query = (
            f"MATCH p=(a:Entity {{corpus_name:$c}})-[*1..{depth}]-(b:Entity {{corpus_name:$c}}) "
            "WHERE toLower(a.name)=toLower($start) AND ("
            "  toLower(b.name) CONTAINS $end OR "
            "  toLower(coalesce(b.description,'')) CONTAINS $end OR "
            "  any(alias IN coalesce(b.aliases,[]) WHERE toLower(alias) CONTAINS $end)) "
            "RETURN [n IN nodes(p) | {name:n.name, type:n.type, "
            "  description:substring(coalesce(n.description,''),0,160)}] AS nodes, "
            "[r IN relationships(p) | {type:type(r), "
            "  description:substring(coalesce(r.description,''),0,180)}] AS rels "
            "LIMIT $lim"
        )
        rows = s.run(query, c=corpus, start=start, end=end, lim=limit).data()
    else:
        query = (
            f"MATCH p=(a:Entity {{corpus_name:$c}})-[*1..{depth}]-(b:Entity {{corpus_name:$c}}) "
            "WHERE toLower(a.name)=toLower($start) "
            "RETURN [n IN nodes(p) | {name:n.name, type:n.type, "
            "  description:substring(coalesce(n.description,''),0,160)}] AS nodes, "
            "[r IN relationships(p) | {type:type(r), "
            "  description:substring(coalesce(r.description,''),0,180)}] AS rels "
            "LIMIT $lim"
        )
        rows = s.run(query, c=corpus, start=start, lim=limit).data()
    return [_format_path(r) for r in rows]


def _seed_paths(s, corpus, keys, depth=4, limit=40):
    if len(keys) < 2:
        return []
    depth = _bounded_int(depth, 4, 2, 4)
    limit = _bounded_int(limit, 40, 1, 80)
    query = (
        f"MATCH p=(a:Entity {{corpus_name:$c}})-[*2..{depth}]-(b:Entity {{corpus_name:$c}}) "
        "WHERE a.name_key IN $keys AND b.name_key IN $keys AND a.name_key < b.name_key "
        "RETURN [n IN nodes(p) | {name:n.name, type:n.type, "
        "  description:substring(coalesce(n.description,''),0,140)}] AS nodes, "
        "[r IN relationships(p) | {type:type(r), "
        "  description:substring(coalesce(r.description,''),0,160)}] AS rels "
        "LIMIT $lim"
    )
    return [_format_path(r) for r in s.run(query, c=corpus, keys=keys, lim=limit).data()]


SYSTEM = """You answer a question by EXPLORING a knowledge graph built from one document.

GRAPH SHAPE
- Nodes are entities with: name, type, description, aliases.
- Entities are linked by typed, described relationships.

YOU HAVE THESE READ-ONLY TOOLS. Respond with ONE JSON object per turn, nothing else:
- {"action":"search","term":"<text>"}                    find entities matching text in name/description/alias
- {"action":"neighbors","name":"<Entity>"}                inspect one-hop relationships
- {"action":"paths","start":"<Entity>","end":"<term>"}    inspect multi-hop paths up to 4 hops; omit end to expand outward
- {"action":"batch","calls":[<tool action>, ...]}         run up to 8 search/neighbors/paths calls in one turn
- {"action":"answer","text":"<final answer>"}             when you can answer

RULES
- Use multi-hop reasoning. Search for important entities from the question, inspect their neighbors,
  and use the paths tool when the answer requires connecting people, events, places, concepts, causes,
  evidence, or outcomes across more than one relationship.
- Prefer batch actions: in your first turn search several likely entity/event terms together; in
  later turns inspect neighbors or paths for several promising entities together. This reduces
  repeated tool rounds while keeping your exploration agentic.
- Base the final answer ONLY on what the tools return. Do not use outside knowledge.
- ANSWER FORMAT: write like a human reference answer, not like graph debug output. Do not
  mention node names, edge labels, relationship chains, search terms, observations, or the
  knowledge graph itself. Preserve the exact names, places, dates, objects, and phrasing
  that appear in the question when they are supported by the graph.
- Give enough detail for benchmark overlap and coverage. Fact Retrieval should be a direct
  answer, usually one full sentence; if the question asks for multiple facts, include all
  requested facts in that sentence or in two compact sentences. Complex Reasoning should
  state the conclusion first and then explain the supporting cause/evidence chain in 3-5
  connected sentences. Contextual Summarize should cover every requested entity, location,
  event, role, cause, and outcome in 4-7 sentences. Creative Generation should satisfy the
  requested form or voice while staying factual, usually one cohesive paragraph of 4-7
  sentences.
- Avoid overly terse answers unless the question asks for a single name/title/place. Avoid
  generic filler, bullets, markdown, and preambles such as "According to the knowledge graph."
- Do not answer with an abstain/refusal sentence. If the graph evidence is incomplete, provide the
  most defensible answer supported by the graph observations you have.
"""


def _answer_guidance(question_type: str | None) -> str:
    qt = (question_type or "").strip()
    if qt == "Fact Retrieval":
        return (
            "Give the requested fact directly in natural prose. Use one complete sentence for "
            "a single fact; if the question asks for several details, include all of them in "
            "one or two compact sentences. Keep exact entity/place/title wording from the "
            "question when supported, and avoid graph terms."
        )
    if qt == "Complex Reasoning":
        return (
            "Give a reference-style reasoning answer in 3-5 connected sentences. Start with "
            "the answer/conclusion, then explain the causal or evidential chain using the "
            "important entities and events. Do not expose graph relationship labels or tool "
            "steps; write the reasoning as ordinary prose."
        )
    if qt == "Contextual Summarize":
        return (
            "Give a complete 4-7 sentence summary that covers every facet requested in the "
            "question: entities, roles, places, events, causes, comparisons, and outcomes. "
            "Favor coverage over brevity, but do not invent unsupported details."
        )
    if qt == "Creative Generation":
        return (
            "Write one grounded paragraph of 4-7 sentences in the requested form or voice "
            "(for example diary entry, report, letter, or first person). Preserve the key "
            "facts, names, relationships, constraints, and outcomes, while using natural "
            "creative prose rather than bullet points or graph terminology."
        )
    return ("Give a complete answer with enough detail to match the question scope, while using "
            "only the graph observations.")


SYNTHESIZE_PROMPT = """You must answer using ONLY these knowledge graph observations.
Do not use source-document text or outside knowledge.

Benchmark question type:
%(question_type)s

Answer style:
%(answer_guidance)s

Question:
%(question)s

Knowledge graph observations:
%(contexts)s

Write the final answer in the requested style, matching the scope and wording of the
question as closely as the observations allow. Do not include preamble, caveats,
markdown, bullets, graph/tool terminology, or any sentence saying that the knowledge graph lacks information.
If evidence is partial, give the most defensible answer from the observations.
Answer:"""


FAST_KG_PROMPT = """Answer using ONLY the retrieved knowledge graph evidence below.
The evidence was selected by keyword seed nodes, one-hop neighbors, and multi-hop
paths between seed nodes. Do not use source-document text or outside knowledge.

Benchmark question type:
%(question_type)s

Answer style:
%(answer_guidance)s

Question:
%(question)s

Knowledge graph evidence:
%(contexts)s

Write the final answer in the requested style, matching the scope and wording of the
question as closely as the evidence allows. Do not include preamble, caveats,
markdown, bullets, graph/tool terminology, or any sentence saying that the knowledge graph lacks information.
If evidence is partial, give the most defensible answer from the graph evidence.
Answer:"""


def _tools(corpus):
    return {
        "search": lambda s, a: _search(s, corpus, a.get("term", "")),
        "neighbors": lambda s, a: _neighbors(s, corpus, a.get("name", "")),
        "paths": lambda s, a: _paths(s, corpus, a.get("start", ""),
                                     a.get("end", ""), a.get("depth", 3)),
    }


def _synthesize_best_effort(engine, question, contexts, question_type=None):
    prompt = SYNTHESIZE_PROMPT % {
        "question_type": question_type or "Unspecified",
        "answer_guidance": _answer_guidance(question_type),
        "question": question,
        "contexts": "\n---\n".join(contexts[-12:]) or "No graph observations were returned.",
    }
    return engine.complete(prompt, timeout=120).strip()


def retrieve_fast_context(session, corpus, question):
    seeds = _seed_nodes(session, corpus, question)
    keys = [r["key"] for r in seeds if r.get("key")]
    neighbors = _neighbor_rows(session, corpus, keys[:8])
    paths = _seed_paths(session, corpus, keys[:8])
    contexts = []
    if seeds:
        contexts.append("[seed_nodes] " + json.dumps(seeds, default=str))
    if neighbors:
        contexts.append("[neighbors] " + json.dumps(neighbors, default=str))
    if paths:
        contexts.append("[multi_hop_paths] " + json.dumps(paths, default=str))
    return contexts


def answer_fast_with_context(question: str, corpus: str, session=None, verbose: bool = False,
                             question_type: str | None = None):
    engine = get_engine(config.ENGINE)

    def _answer(s):
        contexts = retrieve_fast_context(s, corpus, question)
        prompt = FAST_KG_PROMPT % {
            "question_type": question_type or "Unspecified",
            "answer_guidance": _answer_guidance(question_type),
            "question": question,
            "contexts": "\n---\n".join(contexts) or "No graph observations were returned.",
        }
        if verbose:
            print(f"[fast-kg] contexts={len(contexts)}")
        return engine.complete(prompt, timeout=180).strip(), contexts

    if session is not None:
        return _answer(session)
    drv = driver()
    try:
        with drv.session() as s:
            return _answer(s)
    finally:
        drv.close()


def _looks_like_abstain(text):
    t = (text or "").lower()
    return (
        not text
        or text == ABSTAIN
        or "don't have enough information" in t
        or "do not have enough information" in t
        or "cannot answer" in t
        or "can't answer" in t
        or "lacks the information" in t
    )


def answer_with_context(question: str, corpus: str, session=None, verbose: bool = False,
                        question_type: str | None = None):
    """Return (answer, contexts) for a question scoped to one corpus.

    If `session` is provided it is reused (no driver churn across many questions);
    otherwise a driver/session is opened and closed for this call.
    """
    engine = get_engine(config.ENGINE)
    task_question = (
        f"QUESTION TYPE: {question_type or 'Unspecified'}\n"
        f"ANSWER STYLE: {_answer_guidance(question_type)}\n"
        f"QUESTION: {question}"
    )
    if session is not None:
        ans, ctx = run_loop(engine, session, SYSTEM, task_question, _tools(corpus),
                            max_steps=KG_MAX_STEPS, verbose=verbose)
        if _looks_like_abstain(ans):
            ans = _synthesize_best_effort(engine, question, ctx, question_type)
        return ans, ctx
    drv = driver()
    try:
        with drv.session() as s:
            ans, ctx = run_loop(engine, s, SYSTEM, task_question, _tools(corpus),
                                max_steps=KG_MAX_STEPS, verbose=verbose)
            if _looks_like_abstain(ans):
                ans = _synthesize_best_effort(engine, question, ctx, question_type)
            return ans, ctx
    finally:
        drv.close()
