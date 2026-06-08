from brandkg.graphrag_bench.indexing import _to_graphml, build_index_cmd


def test_to_graphml_structure_and_escaping():
    nodes = [("ada", "Ada & Co", "person"), ("eng", "Engine", "machine")]
    edges = [("ada", "WORKS_ON", "eng"), ("ada", "REL", "missing")]  # 2nd edge endpoint absent
    xml = _to_graphml(nodes, edges)
    assert xml.startswith("<?xml")
    assert "<graphml" in xml and "</graphml>" in xml
    assert xml.count("<node ") == 2
    assert xml.count("<edge ") == 1            # the missing-endpoint edge is skipped
    assert "Ada &amp; Co" in xml               # XML-escaped
    assert "WORKS_ON" in xml


def test_build_index_cmd_argv():
    cmd = build_index_cmd("/venv/bin/python", "/abs/graphml_novel", "/abs/indexing_novel.txt")
    assert cmd[:3] == ["/venv/bin/python", "-m", "Evaluation.indexing_eval"]
    assert "--framework" in cmd and "graphml" in cmd
    assert "--base_path" in cmd and "/abs/graphml_novel" in cmd
    assert "--output" in cmd and "/abs/indexing_novel.txt" in cmd


def test_cli_dispatches_index_eval(monkeypatch):
    from brandkg import cli
    import brandkg.graphrag_bench.indexing as idx
    called = {}
    monkeypatch.setattr(idx, "main", lambda argv: called.update({"argv": argv}) or 0)
    rc = cli.main(["graphrag-index-eval", "--subset", "novel"])
    assert rc == 0
    assert called["argv"] == ["--subset", "novel"]
