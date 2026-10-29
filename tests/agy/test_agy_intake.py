"""Minute-one intake (bootstrap_intake.py, stdlib only) -> init --from-intake -> G2 auto-import of the Director's sheets."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from agy_studio import media
from agy_studio.cli import run as cli_run
from agy_studio.orchestrator import Studio
from agy_studio.project import Project
from agy_studio.state import SceneStage

REPO = Path(__file__).resolve().parents[2]


def test_bootstrap_intake_then_init_from_intake(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    (ws / "productions").mkdir(parents=True)
    script = tmp_path / "scriptV1.txt"
    script.write_text("INT. CANYON - DAY\n\nSPARKY\nHello there!\n")
    sheet = tmp_path / "Sparky.png"
    media.write_png(sheet, width=32, height=48, rgb=(200, 120, 40))

    # bootstrap_intake.py runs with a bare interpreter from the WORKSPACE (no venv, no deps)
    proc = subprocess.run([sys.executable, str(REPO / "bootstrap_intake.py"), "SPARKY_EP5", "--title",
                           "Sonic Heroes: The Canyon Echo Rescue", "--mode", "autopilot", "--format", "episode_16_9",
                           "--language", "en", "--script", str(script), "--ref", f"Sparky={sheet}",
                           "--primary-model", "veo", "--secondary-model", "omni", "--no-3d", "--takes-per-clip", "1", "--workspace", str(ws)],
                          cwd=ws, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    # it must not import agy_studio / third-party packages
    src = (REPO / "bootstrap_intake.py").read_text()
    assert "import agy_studio" not in src and "from agy_studio" not in src and "pydantic" not in src
    assert "import yaml" not in src

    proj_dir = ws / "productions" / "SPARKY_EP5"
    intake = yaml.safe_load((proj_dir / "intake.yaml").read_text())
    assert intake["control_mode"] == "autopilot" and intake["aspect_ratio"] == "16:9"
    assert (proj_dir / "script" / "script_v1.txt").read_text().startswith("INT. CANYON")
    assert (proj_dir / "library" / "imports" / "Sparky" / "Sparky.png").is_file()
    assert intake["model_policy"] == {"video_model_primary": "veo", "video_model_secondary": "omni"}

    out = cli_run(["--workspace", str(ws), "init", "SPARKY_EP5", "--from-intake"], fake=True)
    assert "mode=autopilot" in out and "format=episode_16_9" in out and "16:9" in out and "lang=en" in out
    assert "Sparky (1)" in out
    proj = Project(proj_dir)
    st = proj.config.settings
    assert st.control_mode == "autopilot" and st.aspect_ratio == "16:9" and st.width == 1920
    assert st.video_model_primary == "veo" and st.video_model_secondary == "omni" and st.takes_per_clip == 1
    assert proj.config.title == "Sonic Heroes: The Canyon Echo Rescue" and proj.config.language == "en"
    # CLI flags still override the intake
    ws2 = tmp_path / "ws2"
    (ws2 / "productions" / "SPARKY_EP5").mkdir(parents=True)
    (ws2 / "productions" / "SPARKY_EP5" / "intake.yaml").write_text((proj_dir / "intake.yaml").read_text())
    out2 = cli_run(["--workspace", str(ws2), "init", "SPARKY_EP5", "--from-intake", "--mode", "director"], fake=True)
    assert "mode=director" in out2


def test_g2_uses_directors_imported_sheets_before_asking_or_generating(tmp_path: Path) -> None:
    from agy_studio.testing import build_demo_project

    project = build_demo_project(tmp_path / "productions", lock=False)  # every pack still proposed -> G2 pending
    orch = Studio(project, fake=True)
    pending = [i for i in orch.design_delta("SC01") if "_STATE_" in i]
    assert pending, orch.design_delta("SC01")
    item = pending[0]
    ent = project.entity_state_index()[item][0]
    name = (getattr(ent, "display_name", None) or ent.id.removeprefix("CHAR_")).replace("_", " ")
    folder = project.path("library", "imports", name)
    folder.mkdir(parents=True)
    media.write_png(folder / "a_front.png", width=32, height=48, rgb=(10, 200, 30))
    media.write_png(folder / "b_three_quarter.png", width=32, height=48, rgb=(30, 10, 200))
    lines = orch.auto_import_reference_sheets("SC01")
    assert any(line.startswith(f"{item}: using the Director's 2 reference image(s)") for line in lines), lines
    manifest = orch._design_manifest("SC01")
    assert "front" in manifest[item] and project.abs(manifest[item]["front"]).is_file()
    assert "imports" in manifest[item]["front"]  # the Director's file, not an AI candidate
    # second call is a no-op (already staged)
    assert orch.auto_import_reference_sheets("SC01") == []
    # and the G2 handler itself picks it up without asking a reference-sheet question about this item
    orch.open_g1()
    orch.respond("G1", None, "approve all")
    rep = orch.advance("SC01")
    assert not any(item in str(q.text) for q in orch.questions.open()), rep.text()
