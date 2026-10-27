"""set3d — grey-box 3D set: schema, designer (fake/manual), viewer API, render upload, style pass -> cam: angle."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from agy_studio import media
from agy_studio.cli import run as cli_run
from agy_studio.registry import default_registry
from agy_studio.set3d import designer, store
from agy_studio.set3d.schema import Primitive, SetCamera, SetLayout, layout_json_schema
from agy_studio.set3d.server import build_app
from agy_studio.set3d.stylepass import stylize_camera

ENV = "ENV_TERRACE"


def _png_b64(tmp: Path, name: str, w: int = 16, h: int = 28) -> str:
    p = tmp / name
    media.write_png(p, width=w, height=h, rgb=(90, 90, 90))
    return base64.b64encode(p.read_bytes()).decode()


# ---------------------------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------------------------
def test_schema_validates_scale_and_ids():
    with pytest.raises(ValueError):
        Primitive(id="wall", size=(0, 1, 1))
    with pytest.raises(ValueError):
        Primitive(id="1bad", size=(1, 1, 1))
    with pytest.raises(ValueError):
        SetCamera(name="SH01", position=(0, 1, 0), look_at=(0, 1, 0))
    cam = SetCamera(name="SH01", position=(0, 0.4, 6), look_at=(0, 1.5, 0), focal_mm=24, aspect_ratio="9:16")
    assert cam.horizontal_fov_deg() == pytest.approx(73.7, abs=0.2)
    assert cam.vertical_fov_deg() > cam.horizontal_fov_deg()  # portrait frame
    assert "ground-level 24 mm wide" in cam.describe() and "looking up" in cam.describe()
    lay = SetLayout(environment_id=ENV, primitives=[Primitive(id="a"), Primitive(id="b")], cameras=[cam])
    with pytest.raises(ValueError):
        SetLayout(environment_id=ENV, primitives=[Primitive(id="a"), Primitive(id="a")])
    schema = layout_json_schema()
    assert "cameras" not in schema["properties"] and "primitives" in schema["properties"]
    # locked cameras cannot be moved or removed
    cam.locked = True
    lay.upsert_camera(cam)
    with pytest.raises(ValueError):
        lay.upsert_camera(cam.model_copy(update={"focal_mm": 50.0}))
    with pytest.raises(ValueError):
        lay.remove_camera("SH01")


# ---------------------------------------------------------------------------------------------
# designer
# ---------------------------------------------------------------------------------------------
def test_fake_designer_builds_and_links_locked_environment(demo_project):
    lay = designer.build_set(demo_project, ENV, fake=True)
    assert lay.primitives and lay.mannequins, "fake layout has blocks and mannequins"
    assert {m.character_id for m in lay.mannequins} <= {"CHAR_RISHI", "CHAR_MOHNISH"}
    assert store.has_set(demo_project, ENV)
    env = demo_project.load_environment(ENV)
    assert env.set3d_uri == f"bible/sets/{ENV}/set.json"
    assert env.approval_status.value == "locked"  # still locked; the link was recorded as a Craft decision
    rows = [json.loads(l) for l in demo_project.decisions_path.read_text().splitlines() if l.strip()]
    assert any(r["target_artifact_id"] == ENV and r["payload"].get("set3d_uri") for r in rows)
    calls = list((demo_project.agent_calls_dir).rglob("*.json")) if demo_project.agent_calls_dir.exists() else []
    assert any("set_designer" in p.name or "set_designer" in p.read_text() for p in calls)


def test_rebuild_keeps_cameras_and_versions(demo_project):
    designer.build_set(demo_project, ENV, fake=True)
    lay = store.load_set(demo_project, ENV)
    lay.upsert_camera(SetCamera(name="SH07", position=(1, 1.4, 4), look_at=(0, 1.2, 0)))
    store.save_set(demo_project, lay, bump=False)
    lay2 = designer.build_set(demo_project, ENV, fake=True, steer="more clutter")
    assert lay2.camera("SH07") is not None and lay2.camera("WIDE") is not None
    assert lay2.version == 2
    assert (store.set_dir(demo_project, ENV) / "set_history").exists()
    lay3 = designer.build_set(demo_project, ENV, fake=True, keep_cameras=False)
    assert lay3.camera("SH07") is None


def test_manual_designer_validates(demo_project):
    manual = {"environment_id": "WRONG", "primitives": [{"id": "floor", "size": [8, 0.1, 8], "structural": True}],
              "notes": "AGY authored"}
    lay = designer.build_set(demo_project, ENV, mode="manual", manual_output=manual)
    assert lay.environment_id == ENV and lay.notes == "AGY authored"
    with pytest.raises(Exception, match="no primitives|validation"):
        designer.build_set(demo_project, ENV, mode="manual", manual_output={"primitives": []})


# ---------------------------------------------------------------------------------------------
# viewer API
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def client(demo_project):
    designer.build_set(demo_project, ENV, fake=True)
    return TestClient(build_app(demo_project))


def test_api_sets_and_layout(client):
    r = client.get("/api/sets").json()
    assert r["project_id"] == "DEMO_S01" and r["sets"][0]["env_id"] == ENV and r["sets"][0]["has_still"]
    lay = client.get(f"/api/set/{ENV}").json()
    assert lay["environment_id"] == ENV and lay["primitives"]
    assert client.get("/api/set/ENV_NOPE").status_code == 404
    assert client.get("/").status_code == 200 and "three" in client.get("/").text
    assert client.get("/static/vendor/three.module.min.js").status_code == 200
    assert client.get(f"/api/set/{ENV}/still").status_code == 200


def test_api_camera_roundtrip_and_render_upload(client, demo_project, tmp_path):
    cam = {"name": "SH03", "position": [2, 1.5, 5], "look_at": [0, 1.2, 0], "focal_mm": 40, "aspect_ratio": "9:16",
           "note": "over the parapet"}
    assert client.post(f"/api/set/{ENV}/cameras", json=cam).status_code == 200
    assert client.post(f"/api/set/{ENV}/cameras", json={**cam, "name": "bad name!"}).status_code == 400
    names = {c["name"] for c in client.get(f"/api/set/{ENV}/cameras").json()}
    assert {"WIDE", "SH03"} <= names
    # render upload
    assert client.get(f"/api/set/{ENV}/render/SH03").status_code == 404
    body = {"blockout_png_b64": _png_b64(tmp_path, "b.png"), "depth_png_b64": _png_b64(tmp_path, "d.png"),
            "width": 16, "height": 28, "visible_mannequins": ["MQ_CHAR_RISHI"]}
    r = client.post(f"/api/set/{ENV}/render/SH03", json=body)
    assert r.status_code == 200, r.text
    rdir = demo_project.abs(r.json()["render_dir"])
    assert (rdir / "blockout.png").is_file() and (rdir / "depth.png").is_file() and (rdir / "camera.json").is_file()
    assert client.post(f"/api/set/{ENV}/render/NOPE", json=body).status_code == 404
    assert client.post(f"/api/set/{ENV}/render/SH03", json={"blockout_png_b64": "aGVsbG8="}).status_code == 400
    m = client.get(f"/api/set/{ENV}/render/SH03").json()
    assert m["camera"]["name"] == "SH03" and m["visible_mannequins"] == ["MQ_CHAR_RISHI"]
    img = client.get(m["blockout"])
    assert img.status_code == 200 and img.headers["content-type"].startswith("image/png")
    assert client.get("/files/../../etc/passwd").status_code in (400, 404)
    assert client.get("/files/project.yaml").status_code == 403  # only images / json
    # camera carries last_render; delete is refused once locked
    cams = {c["name"]: c for c in client.get(f"/api/set/{ENV}/cameras").json()}
    assert cams["SH03"]["last_render"] == r.json()["render_dir"]
    assert client.delete(f"/api/set/{ENV}/cameras/WIDE").status_code == 200
    assert client.delete(f"/api/set/{ENV}/cameras/WIDE").status_code == 404


def test_api_put_layout_keeps_cameras(client):
    lay = client.get(f"/api/set/{ENV}").json()
    lay["primitives"].append({"id": "crate", "label": "crate", "position": [1, 0.3, 1], "size": [0.6, 0.6, 0.6]})
    lay["cameras"] = []  # the viewer never owns cameras
    assert client.put(f"/api/set/{ENV}", json=lay).status_code == 200
    after = client.get(f"/api/set/{ENV}").json()
    assert any(p["id"] == "crate" for p in after["primitives"]) and after["cameras"]
    bad = dict(after)
    bad["primitives"] = [{"id": "x", "size": [0, 0, 0]}]
    assert client.put(f"/api/set/{ENV}", json=bad).status_code == 400


# ---------------------------------------------------------------------------------------------
# style pass -> environment angle -> spatial anchor
# ---------------------------------------------------------------------------------------------
def test_stylize_registers_cam_angle(client, demo_project, tmp_path):
    cam = {"name": "SH03", "position": [2, 1.5, 5], "look_at": [0, 1.2, 0], "focal_mm": 40, "aspect_ratio": "9:16"}
    client.post(f"/api/set/{ENV}/cameras", json=cam)
    adapter = default_registry().adapter("nano_banana_pro", project=demo_project, fake=True)
    with pytest.raises(FileNotFoundError, match="not been rendered"):
        stylize_camera(demo_project, ENV, "SH03", image_adapter=adapter)
    client.post(f"/api/set/{ENV}/render/SH03", json={"blockout_png_b64": _png_b64(tmp_path, "b.png"),
                                                    "width": 16, "height": 28,
                                                    "visible_mannequins": ["MQ_CHAR_RISHI"]})
    res = stylize_camera(demo_project, ENV, "SH03", image_adapter=adapter, steer="wetter floor")
    assert demo_project.abs(res["styled"]).is_file()
    assert res["library"].startswith(f"library/environments/{ENV}/") and res["library"].endswith("cam_SH03.png")
    assert "wetter floor" in res["prompt"] and "EMPTY of people" in res["prompt"]
    env = demo_project.load_environment(ENV)
    assert env.derived_angles.cameras["SH03"] == res["library"]
    assert env.angle_uri("cam:SH03") == res["library"]
    assert env.angle_uri("cam:NOPE") is None
    assert env.approval_status.value == "locked"
    assert store.load_set(demo_project, ENV).camera("SH03").locked
    # the styled reference is a real file the linter's L6 "exists on disk" check will accept
    assert demo_project.abs(env.angle_uri("cam:SH03")).is_file()
    with pytest.raises(ValueError, match="no camera"):
        stylize_camera(demo_project, ENV, "SH99", image_adapter=adapter)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def test_cli_set3d(demo_project, productions_dir):
    ws = str(productions_dir.parent)
    out = cli_run(["--workspace", ws, "--project", "DEMO_S01", "--fake", "set3d", "status"])
    assert "No 3D sets yet" in out
    out = cli_run(["--workspace", ws, "--project", "DEMO_S01", "--fake", "set3d", "build", "env_terrace"])
    assert "3D set for ENV_TERRACE ready" in out and "set3d serve ENV_TERRACE" in out
    out = cli_run(["--workspace", ws, "--project", "DEMO_S01", "--fake", "set3d", "cameras", ENV])
    assert "WIDE" in out
    out = cli_run(["--workspace", ws, "--project", "DEMO_S01", "--fake", "set3d", "show", ENV])
    assert json.loads(out)["environment_id"] == ENV
    with pytest.raises(FileNotFoundError):
        cli_run(["--workspace", ws, "--project", "DEMO_S01", "--fake", "set3d", "serve", "ENV_STAIRWELL"])


@pytest.mark.parametrize("mode", ["director", "hybrid", "autopilot"])
def test_g2_never_auto_builds_set3d_in_any_mode(unlocked_project, mode):
    """3D world building is opt-in: locking an environment at G2 must not spend an LLM pass + Blender export."""
    from agy_studio.orchestrator import Studio

    s = Studio(unlocked_project, fake=True)
    s.set_control_mode(mode)
    env = unlocked_project.load_environment(ENV)
    msg = s._promote_design("SC01", ENV, {"geometry": env.geometry_still}, "D000001")
    assert not store.has_set(unlocked_project, ENV)
    assert f"set3d build {ENV}" in msg and f"set3d open-blender {ENV}" in msg
    assert not unlocked_project.abs(f"bible/sets/{ENV}").exists()


def test_user_built_set3d_is_injected_into_scene_context(unlocked_project):
    """When the Director DOES ask for the set, it is exposed to the agents exactly as before."""
    from agy_studio.agents.runner import scene_context
    from agy_studio.orchestrator import Studio
    from agy_studio.working_set import build_working_set

    s = Studio(unlocked_project, fake=True)
    designer.build_set(unlocked_project, ENV, fake=True)  # == `set3d build ENV_TERRACE`
    env = unlocked_project.load_environment(ENV)
    msg = s._promote_design("SC01", ENV, {"geometry": env.geometry_still}, "D000001")
    assert f"set3d open-blender {ENV}" in msg
    assert store.has_set(unlocked_project, ENV)
    ws = build_working_set(unlocked_project, "SC01")
    ctx = scene_context(unlocked_project, ws)
    env_rec = ctx["resolved_refs"]["environment"]["record"]
    assert "set3d_summary" in env_rec
    assert "cam:WIDE" in env_rec["set3d_summary"]["saved_cameras"]
    assert env_rec["set3d_summary"]["mannequins"]


# ---------------------------------------------------------------------------------------------
# Blender launcher: the Director can actually SEE the world (on demand), and it fails fast without Blender
# ---------------------------------------------------------------------------------------------
def _fake_blender(tmp_path: Path, *, fail: bool = False) -> Path:
    """Stand-in Blender: records its argv, and when run with --save-and-render writes the files the real one would."""
    import stat

    exe = tmp_path / "bin" / "blender"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text(
        "#!/bin/sh\n"
        "echo \"$@\" > \"$(dirname \"$0\")/argv.txt\"\n"
        + ("echo 'Error: GPU not found' 1>&2; exit 3\n" if fail else "")
        + "for a in \"$@\"; do case \"$a\" in *.py) script=\"$a\";; esac; done\n"
        "dir=$(dirname \"$script\")\n"
        "case \"$*\" in *--save-and-render*)\n"
        "  touch \"$dir\"/*.blend 2>/dev/null || : ; for b in \"$dir\"/*.py; do :; done\n"
        "  env=$(basename \"$dir\"); : ; mkdir -p \"$dir/cam_renders_9_16\"; printf x > \"$dir/cam_renders_9_16/WIDE_3d_clean.png\"\n"
        "  touch \"$dir/ENV_TERRACE.blend\"\n"
        "  ;; esac\n"
        "exit 0\n"
    )
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return exe


def test_open_blender_render_produces_blend_and_camera_blockouts(demo_project, tmp_path, monkeypatch):
    from agy_studio.set3d.blender_export import open_in_blender

    designer.build_set(demo_project, ENV, fake=True)
    exe = _fake_blender(tmp_path)
    monkeypatch.setenv("BLENDER_BIN", str(exe))
    res = open_in_blender(demo_project, ENV, render=True)
    argv = (tmp_path / "bin" / "argv.txt").read_text()
    assert "-b" in argv and "--python" in argv and "--save-and-render" in argv
    assert res["blend_exists"] and res["rendered"] == [f"bible/sets/{ENV}/blender/cam_renders_9_16/WIDE_3d_clean.png"]
    assert res["launched_gui"] is False


def test_open_blender_gui_launches_without_blocking(demo_project, tmp_path, monkeypatch):
    from agy_studio.set3d.blender_export import open_in_blender

    designer.build_set(demo_project, ENV, fake=True)
    monkeypatch.setenv("BLENDER_BIN", str(_fake_blender(tmp_path)))
    res = open_in_blender(demo_project, ENV)
    assert res["launched_gui"] is True and "-b" not in res["command"].split()
    assert demo_project.abs(res["blender_script"]).is_file()


def test_open_blender_fails_fast_when_blender_missing(demo_project, tmp_path, monkeypatch):
    from agy_studio.set3d import blender_export as be

    designer.build_set(demo_project, ENV, fake=True)
    monkeypatch.setenv("BLENDER_BIN", str(tmp_path / "nope"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(be, "BLENDER_CANDIDATES", ())
    with pytest.raises(be.BlenderNotFoundError, match="brew install --cask blender"):
        be.open_in_blender(demo_project, ENV)


def test_open_blender_render_error_is_reported_not_hidden(demo_project, tmp_path, monkeypatch):
    from agy_studio.set3d.blender_export import open_in_blender

    designer.build_set(demo_project, ENV, fake=True)
    monkeypatch.setenv("BLENDER_BIN", str(_fake_blender(tmp_path, fail=True)))
    with pytest.raises(RuntimeError, match="GPU not found"):
        open_in_blender(demo_project, ENV, render=True)


def test_generated_blender_script_has_no_foreign_production_paths(demo_project):
    from agy_studio.set3d.blender_export import export_blender_scene

    designer.build_set(demo_project, ENV, fake=True)
    res = export_blender_scene(demo_project, ENV)
    script = demo_project.abs(res["blender_script"]).read_text()
    assert "ENV_RISHIS_HOUSE" not in script and "CHAR_MOTHER_STATE_DEFAULT" not in script
