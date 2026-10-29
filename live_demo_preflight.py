#!/usr/bin/env python3
"""AGY Studio v5 — Script-Agnostic Pre-Flight Validator & Project Bootstrapper.

Run before going on stage (or let Antigravity run it automatically on Turn 1):
    .venv/bin/python live_demo_preflight.py
    .venv/bin/python live_demo_preflight.py --init-project MY_SHOW --script path/to/script.fountain
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRODUCTIONS_DIR = ROOT / "productions"
sys.path.insert(0, str(ROOT))

from agy_studio.adapters import auth  # noqa: E402 - stdlib-only; also prepends every usual gcloud/ffmpeg dir to PATH

auth.ensure_tool_dirs_on_path()


def check_binary(name: str, explicit_path: str | None = None) -> tuple[bool, str]:
    target = explicit_path or shutil.which(name)
    if not target or not Path(target).exists():
        return False, f"MISSING ({name})"
    return True, str(target)


def check_vertex_auth() -> tuple[bool, str]:
    """Actually mint a token + resolve the project (the exact calls Gate G4/G5 make). Blocking when it fails."""
    try:
        status = auth.require_vertex_auth()
    except auth.AuthError as exc:
        return False, str(exc)
    return True, status.summary()


def run_preflight() -> bool:
    print("=" * 72)
    print("🎬 AGY STUDIO v5 — SCRIPT-AGNOSTIC HARNESS PRE-FLIGHT CHECK")
    print("=" * 72)
    all_ok = True

    checks = [
        ("Python .venv", str(ROOT / ".venv/bin/python")),
        ("FFmpeg (Audio/Video/ContactSheet)", shutil.which("ffmpeg")),
        ("Google Cloud CLI (Vertex/Omni/TTS)", auth.find_gcloud(required=False)),
        ("Blender 5.2 (3D Set & Camera Engine)", "/Applications/Blender.app/Contents/MacOS/Blender"),
    ]
    for label, path in checks:
        ok, detail = check_binary(label, path)
        optional = label.startswith("Blender")
        status = "✅ READY" if ok else ("⚠️ OPTIONAL/MISSING" if optional else "❌ REQUIRED/MISSING")
        print(f"  {status:<20} {label:<34} -> {detail}")
        if not optional and not ok:
            all_ok = False

    # The real G4/G5 blocker: can gcloud mint a Vertex token right now? (No offline fallback exists — by design.)
    auth_ok, auth_detail = check_vertex_auth()
    if auth_ok:
        print(f"  {'✅ READY':<20} {'Google Cloud Sign-In (gcloud token)':<34} -> {auth_detail}")
    else:
        all_ok = False
        print(f"  {'❌ BLOCKING':<20} {'Google Cloud Sign-In (gcloud token)':<34} -> FAILED")
        print("-" * 72)
        print(auth_detail)
        print("-" * 72)
        print("STOP: fix the sign-in above before starting any gate. AGY Studio fails fast and never substitutes\n"
              "offline voices or still-image videos — that would waste time and hide this exact problem.")
        print("=" * 72)
        return False

    from agy_studio.generic_bible import (
        CameraRig33,
        CineAperture,
        CineCameraBody,
        CineLens,
        validate_workspace_layout,
    )
    from agy_studio.harness_v5 import (
        DraftFinalizeController,
        GeminiTTS38Line,
        KeyframePinSpec,
        SceneSlate,
        SlateController,
        StateTokenGuard,
        TimelineGitRepo,
        VisualContinuityCritic,
        WindowedRetakeSpec,
        build_3x3_contact_sheet,
    )
    from agy_studio.registry import Registry

    reg = Registry()
    omni = reg.get("omni")
    layout_check = validate_workspace_layout(ROOT)
    print("-" * 72)
    print("🔒 AGY v5 HARNESS CAPABILITIES & ANTI-HALLUCINATION GUARDS:")
    print(f"  ✅ Active GCP Project        : {auth.vertex_project()} (VERTEX_PROJECT_ID / gcloud config)")
    print(f"  ✅ Two-Tier Video Engine     : {omni.provider_model} ({omni.draft_resolution} Draft -> {omni.finalize_resolution} Finalize, reuse_prompt={omni.supports.reuse_prompt_on_finalize})")
    print(f"  ✅ Native Voice Engine       : {omni.native_voice_model} (Preserves [acoustic delivery tags] + native script)")
    print(f"  ✅ Optical & Rig Matrix      : {len(CineCameraBody)} Camera Bodies × {len(CineLens)} Lenses × {len(CineAperture)} Apertures × {len(CameraRig33)} Rigs")
    print("  ✅ Visual Continuity Critic  : 3×3 Contact Sheet + 6 Failure Modes (Identity, Lighting, Prop, Wardrobe, Eyeline, Scale)")
    print("  ✅ NLE Slate & Git Timeline  : 5-Track SceneSlate (-14 LUFS) + StateTokenGuard + TimelineGitRepo (branch/commit/PR/undo)")
    print(f"  ✅ Shareable Bundle Layout   : {'VALID' if layout_check['valid'] else 'INCOMPLETE'} ({len(layout_check['present_shareable_paths'])} core paths verified)")
    print("=" * 72)
    return all_ok


def bootstrap_demo_project(project_id: str, script_path: Path | None) -> Path:
    proj_dir = PRODUCTIONS_DIR / project_id
    proj_dir.mkdir(parents=True, exist_ok=True)
    script_dir = proj_dir / "script"
    script_dir.mkdir(parents=True, exist_ok=True)
    target_script = script_dir / "script_v1.fountain"
    if script_path and script_path.exists():
        shutil.copy2(script_path, target_script)
        print(f"[Bootstrap] Copied script -> {target_script}")
    return proj_dir


def run_quick_auth_check() -> bool:
    """Seconds, not minutes: only the thing most likely to need the Director — can gcloud mint a Vertex token?"""
    ok, detail = check_vertex_auth()
    print(("✅ READY   " if ok else "❌ BLOCKING ") + "Google Cloud Sign-In (gcloud token) -> " + (detail if ok else "FAILED"))
    if not ok:
        print(detail)
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AGY Studio v5 Pre-Flight & Bootstrapper")
    parser.add_argument("--init-project", type=str, default=None, help="Optional new project ID to bootstrap")
    parser.add_argument("--script", type=Path, default=None, help="Optional path to .fountain script")
    parser.add_argument("--quick", action="store_true", help="only the gcloud token / project check (no venv needed)")
    args = parser.parse_args()

    if args.quick:
        sys.exit(0 if run_quick_auth_check() else 1)
    ok = run_preflight()
    if args.init_project:
        bootstrap_demo_project(args.init_project, args.script)
    sys.exit(0 if ok else 1)
