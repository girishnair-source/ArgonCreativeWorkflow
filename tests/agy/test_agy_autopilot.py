"""Autopilot mode: approve G1 once, then script -> Resolve timeline with zero further human input.

Before this test existed, ``control_mode=autopilot`` only auto-passed G4 (and only when voices were already locked):
G2 asked the reference-sheet question and waited, G3 and G5 opened gates and waited, and locking an environment at G2
always built a 3D set + Blender export. A real run stalled inside the first minute.
"""

from __future__ import annotations

from pathlib import Path

from agy_studio.orchestrator import Studio
from agy_studio.resolve_bridge import FakeTransport
from agy_studio.set3d import store
from agy_studio.state import SceneStage
from tests.agy.test_agy_acceptance import seed_two_scene_fixture


def _awaiting_gates(orch: Studio, sid: str) -> list[str]:
    sc = orch.state.load().scenes[sid]
    return sorted(g for g, status in sc.gates.items() if status == "awaiting")


def test_autopilot_runs_g2_to_g6_without_human_and_skips_3d_set(tmp_path: Path) -> None:
    project, registry = seed_two_scene_fixture(tmp_path / "proj_autopilot")
    transport = FakeTransport()
    orch = Studio(project, registry=registry, fake=True, transport=transport)
    orch.set_control_mode("autopilot")

    # The ONLY human action in autopilot: approve the Global Bible once.
    orch.open_g1()
    g1_text = orch.respond("G1", None, "approve all")  # G1 approval itself kicks the scheduler
    rep = orch.advance("SC01")  # idempotent: nothing left to do if G1 already ran the scene through
    text = g1_text + "\n" + rep.text()

    sc = orch.state.scene("SC01")
    assert sc.stage == SceneStage.HANDED_OFF, (sc.stage, text)  # arrived at G6 with no further human input
    assert _awaiting_gates(orch, "SC01") == [], text  # nothing left waiting for a person
    assert not [q for q in orch.questions.open() if q.scene_id == "SC01"], text
    assert "auto-approved (autopilot mode" in text

    # Every clip has a selected take and the cut is in the editor.
    clips = [c for c in project.iter_all_clips() if c.scene_id == "SC01"]
    assert clips and all(c.selected_take_id for c in clips)
    assert any("SC01" in name for name in transport.timelines()), transport.timelines()

    # No 3D set was built (no human to walk it); it stays available on demand.
    assert store.list_sets(project) == []

    # Audit trail: the G2/G3/G5 packets still exist as review folders even though nobody had to look at them.
    review = project.path("scenes", "SC01", "review")
    assert review.exists() and any(review.iterdir()), list(project.path("scenes", "SC01").iterdir())


def test_director_mode_still_waits_at_g2(tmp_path: Path) -> None:
    """Regression guard: the human-in-the-loop modes are unchanged."""
    project, registry = seed_two_scene_fixture(tmp_path / "proj_director")
    orch = Studio(project, registry=registry, fake=True, transport=FakeTransport())
    orch.open_g1()
    orch.respond("G1", None, "approve all")
    orch.advance("SC01")
    assert orch.state.scene("SC01").stage in (SceneStage.AWAITING_G2, SceneStage.AWAITING_G3)
    assert orch.state.scene("SC01").stage != SceneStage.HANDED_OFF
