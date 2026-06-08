"""Step 2: the LLM engine extracts bucketed entities+relationships per document.

This is the "engine" step — instead of a human or an API key, we shell out to the
configured agent CLI (claude -p / codex exec) which authenticates with the user's
subscription. Each data/extracted/*.txt -> data/entities/*.json.

Runs files concurrently (thread pool) up to BRANDKG_CONCURRENCY.
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config
from .engines import get_engine
from .skill import skill_body, skill_name

log = logging.getLogger("brandkg.extract")

EXTRACTED = config.EXTRACTED_DIR
ENTITIES = config.ENTITIES_DIR

# The extraction contract is driven by the graphiti SKILL.md (Approach 1: skill
# integration). The skill body is prepended verbatim; below it we bind the skill's
# rules to this brand's concrete schema and the exact JSON output shape the loader
# expects, then hand over the document text.
PROMPT_TEMPLATE = """You are running the `{skill_name}` skill. Follow its guidance below as the
authoritative method for extracting a brand knowledge graph.

================= GRAPHITI SKILL (source of truth) =================
{skill}
===================================================================

Now apply that skill to ONE document for the brand "{brand}".

This brand's FIXED 7 buckets (every entity belongs to exactly one):
{buckets}

People entities also include: "seniority_tier" (one of {tiers}), "role_title", "affiliation".
Relationship "type" must be one of: {rels}

Per the skill's pruning rules: People = real named humans only (no roles/teams/
placeholders); no slide-fragments or generic abstractions; prefer full canonical
names; never create the brand "{brand}" as an entity or as a relationship endpoint;
every relationship source/target must appear in entities.

Output ONLY valid JSON (no prose, no markdown fences) in EXACTLY this shape:
{{"source_doc": "{doc}",
  "entities": [{{"name": "...", "bucket": "<one of 7>", "type": "<short word>",
                "aliases": [], "description": "1-2 sentences from the text"}}],
  "relationships": [{{"source": "...", "target": "...", "type": "<allowed verb>",
                      "description": "..."}}]}}

DOCUMENT TEXT:
\"\"\"
{text}
\"\"\"
"""

GENERIC_PROMPT_TEMPLATE = """You are running the `{skill_name}` skill. Follow its guidance below as the
authoritative method for extracting a knowledge graph from text.

================= GRAPHITI SKILL (source of truth) =================
{skill}
===================================================================

Apply that skill to ONE document. Extract ALL salient, specific entities (people,
places, organizations, concepts, objects, events) and the relationships between
them. Use open, descriptive types — there are no fixed categories.

CAPTURE CONCRETE FACTS (important — downstream questions ask for specific details):
- In each entity's description, record its concrete ATTRIBUTES exactly as stated in the text:
  materials, measurements, numbers, dates, titles/roles, nicknames, locations, what it is made
  of or used for, who did what to whom, and any other specific factual detail.
- Create a relationship for every specific factual connection the text states (e.g. who commanded
  what, what is made of what, who lived where, who wrote/published what, who reported to whom),
  and put the precise fact in the relationship's description.
- Do not lose a stated fact just because it is minor — single concrete details are exactly what
  gets queried later.

Pruning rules from the skill: prefer full canonical names; merge obvious aliases;
drop pure boilerplate; every relationship's source and target MUST also appear in
entities.

Output ONLY valid JSON (no prose, no markdown fences) in EXACTLY this shape:
{{"source_doc": "{doc}",
  "entities": [{{"name": "...", "type": "<short word>", "aliases": [],
                "description": "1-2 sentences with the concrete facts/attributes from the text"}}],
  "relationships": [{{"source": "...", "target": "...", "type": "<verb phrase>",
                      "description": "the specific fact stated in the text"}}]}}

DOCUMENT TEXT:
\"\"\"
{text}
\"\"\"
"""


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


def extract_blob(engine, schema: dict, doc: str, text: str, timeout: int = 420) -> dict:
    """Extract one text blob -> validated payload dict (no file IO).

    The engine layer already retries hard CLI failures; here we additionally retry
    the *degraded* case where the call succeeds (exit 0) but the model returns
    something without an 'entities' key (seen intermittently under rate pressure).
    Raises after exhausting retries.
    """
    attempts = max(1, int(os.getenv("BRANDKG_ENGINE_RETRIES", "3")) + 1)
    prompt = _build_prompt(schema, doc, text)
    for i in range(attempts):
        payload = engine.complete_json(prompt, timeout=timeout)
        if isinstance(payload, dict) and "entities" in payload:
            payload.setdefault("source_doc", doc)
            return payload
        log.warning("extraction for %s returned unexpected shape (attempt %d/%d): %s",
                    doc, i + 1, attempts, str(payload)[:200])
        if i < attempts - 1:
            time.sleep(min(float(os.getenv("BRANDKG_ENGINE_RETRY_BASE", "5")) * (3 ** i), 60.0))
    raise ValueError("engine returned unexpected shape (no 'entities') after retries")


def _extract_one(engine, schema, txt_path: Path, skill_sig: str, force: bool) -> tuple[str, int, int, str | None]:
    doc = txt_path.stem
    for suf in ("__docx", "__pptx", "__md", "__txt"):
        doc = doc.replace(suf, "")
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    out = ENTITIES / f"{txt_path.stem}.json"
    src_hash = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()

    # CHECKPOINT: skip if we already extracted this exact text successfully.
    # The output records the source hash + prompt signature; if both match, the
    # engine is not called again (so a crash mid-build resumes cheaply).
    if out.exists() and not force:
        try:
            prev = json.loads(out.read_text())
            if prev.get("_src_hash") == src_hash and prev.get("_skill_sig") == skill_sig and "entities" in prev:
                return txt_path.name, len(prev.get("entities", [])), len(prev.get("relationships", [])), "skip"
        except Exception:
            pass  # unreadable/old format -> re-extract

    prompt = _build_prompt(schema, doc, text)
    try:
        payload = engine.complete_json(prompt, timeout=420)
        if not isinstance(payload, dict) or "entities" not in payload:
            return txt_path.name, 0, 0, "engine returned unexpected shape"
        payload.setdefault("source_doc", doc)
        payload["_src_hash"] = src_hash          # checkpoint markers
        payload["_skill_sig"] = skill_sig
        out.write_text(json.dumps(payload, indent=2))
        return txt_path.name, len(payload.get("entities", [])), len(payload.get("relationships", [])), None
    except Exception as e:
        return txt_path.name, 0, 0, str(e)[:200]


def run(only: list[str] | None = None, force: bool = False) -> dict:
    ENTITIES.mkdir(parents=True, exist_ok=True)
    schema = config.schema()
    engine = get_engine(config.ENGINE)
    # skill signature: if the skill text changes, all docs re-extract automatically
    skill_sig = hashlib.sha256(skill_body().encode("utf-8", "ignore")).hexdigest()[:16]
    files = sorted(EXTRACTED.glob("*.txt"))
    if only:
        files = [f for f in files if f.name in only or f.stem in only]
    print(f"extract_entities: {len(files)} docs via engine='{engine.name}' "
          f"(concurrency={config.CONCURRENCY}, force={force})")
    ok = fail = skipped = te = tr = 0
    with ThreadPoolExecutor(max_workers=config.CONCURRENCY) as ex:
        futs = {ex.submit(_extract_one, engine, schema, f, skill_sig, force): f for f in files}
        for fut in as_completed(futs):
            name, ne, nr, err = fut.result()
            if err == "skip":
                skipped += 1; te += ne; tr += nr
                print(f"  SKIP {name} (already extracted, unchanged): {ne} entities, {nr} rels")
            elif err:
                fail += 1
                print(f"  FAIL {name}: {err}")
            else:
                ok += 1; te += ne; tr += nr
                print(f"  OK {name}: {ne} entities, {nr} rels")
    print(f"extract_entities: {ok} extracted, {skipped} skipped(checkpoint), {fail} failed; "
          f"{te} entities, {tr} rels total")
    return {"ok": ok, "skipped": skipped, "failed": fail, "entities": te, "relationships": tr}


if __name__ == "__main__":
    run()
