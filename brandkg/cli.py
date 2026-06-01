"""brand-kg CLI.

Commands:
  build              full pipeline: extract text -> engine extracts entities ->
                     scaffold -> load -> retier -> hierarchy -> text layer -> verify
  query "<q>"        answer a question against the KG (engine synthesizes)
  stats              print graph statistics
  verify             check schema invariants
  wipe               delete all graph data + constraints

The LLM engine (claude/codex) is chosen by BRANDKG_ENGINE and authenticates with
the user's subscription — no API key.
"""
from __future__ import annotations
import sys

from . import config, graph
from .engines import get_engine


def _stamp(s):
    row = s.run("MATCH (b:Brand) SET b.load_count=coalesce(b.load_count,0)+1 RETURN b.load_count AS n").single()
    return f"load-{(row['n'] if row else 1):04d}"


def cmd_build(force: bool = False):
    import json
    from . import extract_text, extract_entities
    print(f"== engine: {config.ENGINE} | brand: {config.schema()['brand_name']} ==")
    print("\n[1/7] extract text"); extract_text.run()
    print("\n[2/7] engine extracts entities (checkpointed — unchanged docs skip)")
    extract_entities.run(force=force)
    payloads = [json.loads(p.read_text()) for p in sorted(config.ENTITIES_DIR.glob("*.json"))]
    drv = graph.driver()
    with drv.session() as s:
        print("\n[3/7] scaffold (Brand + 7 buckets + tiers)"); graph.build_scaffold(s)
        print("[4/7] load entities + relationships")
        te, tr = graph.load_entities(s, payloads, _stamp(s))
        print(f"      {te} entities, {tr} relationships")
        print("[5/7] retier people from titles"); n = graph.retier_people(s); print(f"      retiered {n}")
        print("      build hierarchy"); graph.build_hierarchy(s)
        print("[6/7] text layer"); ntu = graph.ingest_textunits(s); print(f"      {ntu} TextUnits")
        print("[7/7] verify"); v = graph.verify(s)
    drv.close()
    print("\n== RESULT ==")
    print(f"  labels: {v['labels']}")
    print(f"  unexpected labels: {v['unexpected_labels']}")
    print(f"  brand edges: {v['brand_edges']} (must be {len(config.schema()['buckets'])})")
    print(f"  entities: {v['entities']} | rels: {v['rels']} | orphans: {v['orphans']} | beyond-2-hops: {v['beyond_2_hops']}")
    print(f"  per bucket: {v['per_bucket']}")


def cmd_query(q):
    from . import query
    print(query.answer(q))


def cmd_stats():
    drv = graph.driver()
    with drv.session() as s:
        v = graph.verify(s)
    drv.close()
    for k, val in v.items():
        print(f"{k}: {val}")


def cmd_verify():
    cmd_stats()


def cmd_wipe():
    drv = graph.driver()
    with drv.session() as s:
        graph.wipe(s)
    drv.close()
    print("graph wiped (nodes + constraints)")


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print(__doc__); return 1
    cmd, rest = argv[0], argv[1:]
    if cmd == "build":
        cmd_build(force="--force" in rest)
    elif cmd == "query":
        if not rest:
            print('usage: query "<question>"'); return 1
        cmd_query(" ".join(rest))
    elif cmd == "stats":
        cmd_stats()
    elif cmd == "verify":
        cmd_verify()
    elif cmd == "wipe":
        cmd_wipe()
    else:
        print(__doc__); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
