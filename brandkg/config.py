"""Central config: loads settings.env + config/brand_schema.json."""
from __future__ import annotations
import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
ENTITIES_DIR = DATA_DIR / "entities"

# load settings.env if present (falls back to settings.example.env)
if load_dotenv:
    for cand in (CONFIG_DIR / "settings.env", CONFIG_DIR / "settings.example.env"):
        if cand.exists():
            load_dotenv(cand)
            break


def schema(file: str | None = None) -> dict:
    """Load a schema JSON from CONFIG_DIR.

    Precedence: explicit `file` arg > BRANDKG_SCHEMA env > brand_schema.json.
    """
    name = file or os.getenv("BRANDKG_SCHEMA", "brand_schema.json")
    return json.loads((CONFIG_DIR / name).read_text())


def env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


NEO4J_URI = env("BRANDKG_NEO4J_URI", "bolt://localhost:7688")
NEO4J_AUTH = (env("BRANDKG_NEO4J_USER", "neo4j"), env("BRANDKG_NEO4J_PASSWORD", "brandkg2026"))
SOURCE_DIR = env("BRANDKG_SOURCE_DIR", "")
ENGINE = env("BRANDKG_ENGINE", "claude")
# Engine used for ANSWER generation (graphrag-bench). Defaults to ENGINE, but can be
# set independently — e.g. ENGINE=local for cheap KG building, ANSWER_ENGINE=openai
# for higher-quality commercial answers.
ANSWER_ENGINE = env("BRANDKG_ANSWER_ENGINE", ENGINE)
CONCURRENCY = int(env("BRANDKG_CONCURRENCY", "6"))
BENCH_REPO = env("BRANDKG_BENCH_REPO", "")
BENCH_PYTHON = env("BRANDKG_BENCH_PYTHON", "")
