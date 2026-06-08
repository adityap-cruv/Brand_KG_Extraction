"""Export the generic Neo4j graphs to GraphML and run GraphRAG-Bench's indexing_eval.

indexing_eval (`--framework graphml`) reads *.graphml files with igraph and reports
graph-structure metrics (num_nodes/edges, density, clustering coefficient, connected
components, degree distribution). It needs NO LLM/API key — it's purely structural.

We keep our KG in Neo4j, so we first export each corpus's subgraph to a .graphml file,
then point indexing_eval at that directory (run with the bench repo's interpreter).
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from .. import config
from . import graph_generic as gg


def _to_graphml(nodes, edges) -> str:
    """Build a GraphML string igraph can read.

    nodes: iterable of (name_key, name, type); edges: iterable of (src_key, type, tgt_key).
    Edges whose endpoints aren't in `nodes` are skipped.
    """
    idmap = {k: f"n{i}" for i, (k, _, _) in enumerate(nodes)}
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
           '<key id="name" for="node" attr.name="name" attr.type="string"/>',
           '<key id="ntype" for="node" attr.name="type" attr.type="string"/>',
           '<key id="etype" for="edge" attr.name="type" attr.type="string"/>',
           '<graph edgedefault="directed">']
    for k, name, typ in nodes:
        out.append(f'<node id="{idmap[k]}">'
                   f'<data key="name">{escape(name or "")}</data>'
                   f'<data key="ntype">{escape(typ or "")}</data></node>')
    for s, t, o in edges:
        if s in idmap and o in idmap:
            out.append(f'<edge source="{idmap[s]}" target="{idmap[o]}">'
                       f'<data key="etype">{escape(t or "")}</data></edge>')
    out.append('</graph>')
    out.append('</graphml>')
    return "\n".join(out)


def export_graphml(session, corpus, path) -> tuple[int, int]:
    """Export one corpus's :Entity subgraph from Neo4j to a GraphML file."""
    nodes = [(r["k"], r["name"], r["type"]) for r in session.run(
        "MATCH (n:Entity {corpus_name:$c}) "
        "RETURN n.name_key AS k, n.name AS name, n.type AS type", c=corpus)]
    edges = [(r["s"], r["t"], r["o"]) for r in session.run(
        "MATCH (a:Entity {corpus_name:$c})-[r]->(b:Entity {corpus_name:$c}) "
        "RETURN a.name_key AS s, type(r) AS t, b.name_key AS o", c=corpus)]
    Path(path).write_text(_to_graphml(nodes, edges), encoding="utf-8")
    return len(nodes), len(edges)


def export_all(outdir) -> int:
    """Export every corpus currently in Neo4j to outdir/<corpus>.graphml. Returns count."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    drv = gg.driver()
    n = 0
    try:
        with drv.session() as s:
            corpora = [r["c"] for r in s.run(
                "MATCH (n:Entity) RETURN DISTINCT n.corpus_name AS c ORDER BY c")]
            for c in corpora:
                safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in c)
                nn, ne = export_graphml(s, c, outdir / f"{safe}.graphml")
                print(f"  exported {c}: {nn} nodes, {ne} edges")
                n += 1
    finally:
        drv.close()
    return n


def build_index_cmd(python, base_path, output):
    return [python, "-m", "Evaluation.indexing_eval",
            "--framework", "graphml", "--base_path", str(base_path), "--output", str(output)]


def run(subset="novel", bench_repo=None, bench_python=None, output=None):
    repo = Path(bench_repo or config.BENCH_REPO or "")
    if not repo.exists():
        raise SystemExit("set --bench-repo or BRANDKG_BENCH_REPO to the cloned GraphRAG-Benchmark path")
    base = Path(config.ROOT) / "bench" / "graphrag_bench"
    gdir = base / f"graphml_{subset}"
    print(f"exporting Neo4j graphs to GraphML in {gdir} ...")
    n = export_all(gdir)
    if not n:
        raise SystemExit("no graphs in Neo4j to export — build a subset first")
    out = Path(output or base / f"indexing_{subset}.txt").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    python = bench_python or config.BENCH_PYTHON or sys.executable
    cmd = build_index_cmd(python, gdir.resolve(), out)
    print("running:", " ".join(cmd))
    print("cwd:", repo)
    rc = subprocess.call(cmd, cwd=str(repo))
    if rc == 0:
        print(f"indexing eval complete -> {out}")
    return rc


def main(argv=None):
    p = argparse.ArgumentParser(prog="run.py graphrag-index-eval")
    p.add_argument("--subset", default="novel", choices=["novel", "medical"])
    p.add_argument("--bench-repo", default=None)
    p.add_argument("--bench-python", default=None)
    p.add_argument("--output", default=None)
    a = p.parse_args(argv)
    return run(subset=a.subset, bench_repo=a.bench_repo,
               bench_python=a.bench_python, output=a.output)
