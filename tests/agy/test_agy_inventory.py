"""docs/AGENTS_AND_SKILLS.md is generated; this test fails when it is stale (run `python -m agy_studio inventory --write`)."""

from __future__ import annotations

from pathlib import Path

from agy_studio.agents.roles import ROLES
from agy_studio.inventory import DOC_REL_PATH, render_inventory

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_inventory_doc_is_up_to_date() -> None:
    expected = render_inventory(REPO_ROOT)
    committed = (REPO_ROOT / DOC_REL_PATH).read_text(encoding="utf-8")
    assert committed == expected, "docs/AGENTS_AND_SKILLS.md is stale — run `python -m agy_studio inventory --write`"


def test_inventory_covers_every_role_adapter_and_customization() -> None:
    text = render_inventory(REPO_ROOT)
    for name in ROLES:
        assert f"`{name}`" in text
    import yaml

    models = yaml.safe_load((REPO_ROOT / "agy_studio" / "registry.yaml").read_text())["models"]
    for m in models:
        if m.get("enabled", True):
            assert f"`{m['id']}`" in text, m["id"]
    for item in ("casting-agent", "dialogue-agent", "autonomous-director", "prompt-videos", "zero-touch-editor-mode"):
        assert f"`{item}`" in text, item
    assert "ingest_media" in text and "render_timeline" in text  # MCP tools picked up
    assert "FakeVideoAdapter" not in text and "AGY_FAKE_GENERATION" not in text  # the test double is not a product feature
