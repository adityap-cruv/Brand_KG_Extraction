"""Step 2: the LLM engine extracts bucketed entities+relationships per document.

This is the "engine" step — instead of a human or an API key, we shell out to the
configured agent CLI (claude -p / codex exec) which authenticates with the user's
subscription. Each data/extracted/*.txt -> data/entities/*.json.

Runs files concurrently (thread pool) up to BRANDKG_CONCURRENCY.
"""
from __future__ import annotations
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config
from .engines import get_engine
from .skill import skill_body, skill_name

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


def _build_prompt(schema: dict, doc: str, text: str) -> str:
    return PROMPT_TEMPLATE.format(
        skill_name=skill_name(),
        skill=skill_body(),
        brand=schema["brand_name"],
        buckets="\n".join(f"  - {b}" for b in schema["buckets"]),
        tiers=", ".join(schema["seniority_tiers"]),
        rels=", ".join(schema["allowed_cross_child_rels"]),
        doc=doc, text=text[:24000],   # cap to keep prompt size sane
    )


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
