import json

from brandkg import config
from brandkg.extract_entities import _build_prompt, extract_blob


def test_generic_prompt_has_no_brand_buckets():
    schema = config.schema("generic_schema.json")
    p = _build_prompt(schema, "doc1", "Ada Lovelace wrote the first algorithm.")
    assert "7 buckets" not in p
    assert "entities" in p and "relationships" in p
    assert "Ada Lovelace" in p


def test_brand_prompt_still_has_buckets():
    schema = config.schema("brand_schema.json")
    p = _build_prompt(schema, "doc1", "some text")
    assert "7 buckets" in p


def test_extract_blob_parses_engine_json():
    from tests.conftest import FakeEngine
    payload = {"entities": [{"name": "Ada", "type": "person"}], "relationships": []}
    engine = FakeEngine([json.dumps(payload)])
    out = extract_blob(engine, config.schema("generic_schema.json"), "doc1", "text")
    assert out["entities"][0]["name"] == "Ada"
    assert out["source_doc"] == "doc1"


def test_extract_blob_retries_on_missing_entities(monkeypatch):
    monkeypatch.setenv("BRANDKG_ENGINE_RETRY_BASE", "0")
    monkeypatch.setenv("BRANDKG_ENGINE_RETRIES", "3")
    from brandkg import config
    from brandkg.extract_entities import extract_blob

    class Eng:
        def __init__(self):
            self.n = 0

        def complete_json(self, prompt, *, timeout=300):
            self.n += 1
            if self.n < 2:
                return {"oops": True}    # missing 'entities' -> should retry
            return {"entities": [{"name": "A"}], "relationships": []}

    e = Eng()
    out = extract_blob(e, config.schema("generic_schema.json"), "doc", "text")
    assert out["entities"][0]["name"] == "A"
    assert e.n == 2   # retried once after the bad shape
