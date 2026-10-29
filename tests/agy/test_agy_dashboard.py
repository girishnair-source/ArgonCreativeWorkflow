"""Live dashboard: DASHBOARD.md after every engine step, JSON model, local web page (read-only, project-jailed)."""

from __future__ import annotations

from pathlib import Path

from starlette.testclient import TestClient

from agy_studio.dashboard import DASHBOARD_NAME, collect_dashboard, render_markdown, write_dashboard
from agy_studio.dashboard_web import build_app
from agy_studio.orchestrator import Studio
from agy_studio.resolve_bridge import FakeTransport
from agy_studio.state import SceneStage
from agy_studio.testing import build_demo_project


def test_dashboard_on_fresh_project_tells_you_what_to_do(tmp_path: Path) -> None:
    project = build_demo_project(tmp_path / "p")
    d = collect_dashboard(project)
    assert d["project"]["id"] == project.project_id
    assert any("No scene has started" in r["text"] for r in d["risks"])
    md = render_markdown(d)
    assert "## Status" in md and "no scene started" in md and "## Things that may go wrong" in md


def test_engine_rewrites_dashboard_after_each_step_and_it_reflects_the_run(tmp_path: Path) -> None:
    from tests.agy.test_agy_acceptance import seed_two_scene_fixture

    project, registry = seed_two_scene_fixture(tmp_path / "proj_dash")
    orch = Studio(project, registry=registry, fake=True, transport=FakeTransport())
    orch.set_control_mode("autopilot")
    orch.open_g1()
    orch.respond("G1", None, "approve all")
    orch.advance("SC01")
    assert orch.state.scene("SC01").stage == SceneStage.HANDED_OFF

    md_path = project.root / DASHBOARD_NAME
    assert md_path.is_file()  # written by the engine, not by the test
    md = md_path.read_text()
    assert "| SC01 | `handed_off` |" in md
    assert "delivered to Resolve" in md
    assert "autopilot" in md and "Latest decisions" in md
    d = collect_dashboard(project)
    assert d["outputs"]["takes_completed"] >= 1 and d["outputs"]["takes_selected"]
    assert d["time"]["per_role"]  # agent calls accounted
    assert d["eta"]["per_scene"]["SC01"]["seconds"] == 0  # nothing left for a handed-off scene
    assert d["bibles"]["global_context_bible"]
    # explicit write + json round trip
    assert write_dashboard(project) == md_path


def test_dashboard_lists_qa_failures_with_prompt_and_references(tmp_path: Path) -> None:
    from agy_studio.models import DispatchStatus, Take, TakeStatus

    from agy_studio.decomposer import decompose_scene
    from tests.agy.test_agy_acceptance import seed_two_scene_fixture

    project, registry = seed_two_scene_fixture(tmp_path / "proj_qa")
    decompose_scene(project, registry, "SC01")
    clip = next(iter(project.iter_all_clips()))
    tdir = project.take_dir(f"{clip.id}_T09")
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "payload.json").write_text('{"bundle": {"prompt": "Sparky waves at the camera", "negative_prompt": "text, blur", '
                                       '"reference_images": ["library/packs/X/front.png"], "first_frame": "scenes/SC01/boards/SH01.png"}}')
    take = Take(id=f"{clip.id}_T09", clip_id=clip.id, model="omni", provider_model="gemini-omni-1.5-flash",
                payload_hash="h", dispatch_status=DispatchStatus.FAILED, status=TakeStatus.REJECTED,
                error={"message": "RAI filtered"})
    project.save_take(take)
    clip.takes = [*clip.takes, take.id]
    project.save_clip(clip)
    d = collect_dashboard(project)
    fails = [t for t in d["qa_failures"] if t["take_id"] == take.id]
    assert fails and fails[0]["prompt"] == "Sparky waves at the camera"
    assert fails[0]["reference_images"] == ["library/packs/X/front.png"]
    md = render_markdown(d)
    assert "Sparky waves at the camera" in md and "front.png" in md and "RAI filtered" in md


def test_web_dashboard_serves_page_json_and_only_project_files(tmp_path: Path) -> None:
    project = build_demo_project(tmp_path / "p")
    (project.root / "note.md").write_text("hello")
    outside = tmp_path / "secret.txt"
    outside.write_text("nope")
    client = TestClient(build_app(project))
    assert client.get("/").status_code == 200 and "AGY Studio" in client.get("/").text
    data = client.get("/api/dashboard").json()
    assert data["project"]["id"] == project.project_id and "risks" in data
    assert client.get("/file/note.md").status_code == 200
    assert client.get("/file/../secret.txt").status_code in (403, 404)
    assert client.get("/file/%2e%2e/secret.txt").status_code in (403, 404)
    assert client.get("/file/project.yaml").status_code == 200  # yaml allowed, inside the project
    assert client.get("/file/does_not_exist.png").status_code == 404
