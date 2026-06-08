"""Text fallback for the benchmark: when the KG can't answer a question, retrieve
the most relevant passages from the corpus's source document and answer from them.

This makes the pipeline a HYBRID (GraphRAG + text retrieval) — the same design the
leaderboard systems (LightRAG, HippoRAG) use. It only triggers when the graph query
abstains, so the KG stays the primary path and the text is a safety net.

Retrieval is keyword/overlap based (no embeddings, no API key) to stay light.
"""
from __future__ import annotations
import re

from ..query import ABSTAIN

_STOP = set(
    "the a an of to in and or is are was were be been being by for with on at from as that this "
    "these those which who whom whose what when where how why during according narrative story "
    "novel within context events described into out over under about".split()
)


def _keywords(q):
    return [w for w in re.findall(r"[A-Za-z][A-Za-z'-]+", (q or "").lower())
            if w not in _STOP and len(w) > 3]


def rank_passages(text, question, k=4, win=800):
    """Return up to k non-overlapping ~win-char passages from text, ranked by how many
    of the question's keywords they contain."""
    text = text or ""
    terms = _keywords(question)
    if not terms or not text:
        return []
    step = max(1, win // 2)
    chunks = [(i, text[i:i + win]) for i in range(0, len(text), step)]
    scored = sorted(chunks, key=lambda c: -sum(c[1].lower().count(t) for t in terms))
    picked, used = [], []
    for i, c in scored:
        if sum(c.lower().count(t) for t in terms) == 0:
            break
        if any(abs(i - j) < win for j in used):
            continue
        used.append(i)
        picked.append(re.sub(r"\s+", " ", c).strip())
        if len(picked) >= k:
            break
    return picked


def _answer_guidance(question_type: str | None) -> str:
    qt = (question_type or "").strip()
    if qt == "Fact Retrieval":
        return (
            "Give the requested fact directly in natural prose. Use one complete sentence for "
            "a single fact; if the question asks for several details, include all of them in "
            "one or two compact sentences. Keep exact entity/place/title wording from the "
            "question when supported."
        )
    if qt == "Complex Reasoning":
        return (
            "Give a reference-style reasoning answer in 3-5 connected sentences. Start with "
            "the answer/conclusion, then explain the causal or evidential chain using the "
            "important entities and events from the passages."
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
            "creative prose."
        )
    return "Give a complete answer with enough detail to match the question scope."


FALLBACK_PROMPT = """Answer the question using ONLY the passages below, taken from a source document.
Do not use outside knowledge.

Benchmark question type:
%(question_type)s

Answer style:
%(answer_guidance)s

Write like a human reference answer and match the scope and wording of the question
as closely as the passages allow. Do not include preamble, caveats, markdown, or
bullets. If the passages genuinely do not contain the answer, reply EXACTLY:
"%(abstain)s"

PASSAGES:
%(passages)s

QUESTION: %(question)s
ANSWER:"""


def answer_from_text(engine, question, corpus_text, k=6, question_type=None):
    """Retrieve passages for the question and answer from them. Returns (answer, contexts).
    Returns (ABSTAIN, ...) if no relevant passage or the model can't answer."""
    passages = rank_passages(corpus_text, question, k=k)
    if not passages:
        return ABSTAIN, []
    prompt = FALLBACK_PROMPT % {"abstain": ABSTAIN,
                                "question_type": question_type or "Unspecified",
                                "answer_guidance": _answer_guidance(question_type),
                                "passages": "\n---\n".join(passages),
                                "question": question}
    ans = engine.complete(prompt, timeout=120).strip()
    contexts = ["[text-fallback] " + p[:500] for p in passages]
    return ans, contexts
