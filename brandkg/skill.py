"""Load the graphiti skill as the single source of truth for engine behavior.

Approach 1 (skill integration): instead of carrying its own inline prompts, the
repo reads ~/.claude/skills/graphiti/SKILL.md and injects its guidance into every
engine prompt. So the build/query behavior always matches the installed skill —
edit the skill, and this repo follows it.

Override the path with BRANDKG_SKILL_PATH.
"""
from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path

DEFAULT_SKILL = Path.home() / ".claude" / "skills" / "graphiti" / "SKILL.md"


def skill_path() -> Path:
    return Path(os.getenv("BRANDKG_SKILL_PATH", str(DEFAULT_SKILL)))


@lru_cache(maxsize=1)
def skill_body() -> str:
    """Return the skill's markdown body with YAML frontmatter stripped."""
    p = skill_path()
    if not p.exists():
        raise FileNotFoundError(
            f"graphiti skill not found at {p}. Install it at "
            "~/.claude/skills/graphiti/SKILL.md or set BRANDKG_SKILL_PATH.")
    text = p.read_text(encoding="utf-8")
    if text.startswith("---"):
        # drop the first frontmatter block
        end = text.find("\n---", 3)
        if end != -1:
            nl = text.find("\n", end + 1)
            text = text[nl + 1:] if nl != -1 else ""
    return text.strip()


def skill_name() -> str:
    return skill_path().parent.name
