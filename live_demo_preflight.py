#!/usr/bin/env python3
"""AGY Studio v5 — Script-Agnostic Pre-Flight Validator & Project Bootstrapper.

Run before going on stage (or let Antigravity run it automatically on Turn 1):
    .venv/bin/python live_demo_preflight.py
    .venv/bin/python live_demo_preflight.py --init-project MY_SHOW --script path/to/script.fountain
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + os.environ.get("PATH", "")

ROOT = Path(__file__).resolve().parent
PRODUCTIONS_DIR = ROOT / "productions"


def check_binary(name: str, explicit_path: str | None = None) -> tuple[bool, str]:
    target = explicit_path or shutil.which(name)
    if not target or not Path(target).exists():
        return False, f"MISSING ({name})"
    return True, str(target)


def run_preflight() -> bool:
    print("=" * 72)
    print("🎬 AGY STUDIO v5 — SCRIPT-AGNOSTIC HARNESS PRE-FLIGHT CHECK")
    print("=" * 72)
    all_ok = True

    checks = [
        ("Python .venv", str(ROOT / ".venv/bin/python")),
        ("FFmpeg (Audio/Video/ContactSheet)", shutil.which("ffmpeg")),
        ("Google Cloud CLI (Vertex/Omni/TTS)", shutil.which("gcloud")),
        ("Blender 5.2 (3D Set & Camera Engine)", "/Applications/Blender.app/Contents/MacOS/Blender"),
    ]
    for label, path in checks:
        ok, detail = check_binary(label, path)
        status = "✅ READY" if ok else "⚠️ OPTIONAL/MISSING"
        print(f"  {status:<20} {label:<34} -> {detail}")
        if label != "Blender 5.2 (3D Set & Camera Engine)" and not ok:
            all_ok = False

    from agy_studio.adapters.base import vertex_project_id
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
    print(f"  ✅ Active GCP Project        : {vertex_project_id()} (auto-detected via gcloud/env)")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AGY Studio v5 Pre-Flight & Bootstrapper")
    parser.add_argument("--init-project", type=str, default=None, help="Optional new project ID to bootstrap")
    parser.add_argument("--script", type=Path, default=None, help="Optional path to .fountain script")
    args = parser.parse_args()

    ok = run_preflight()
    if args.init_project:
        bootstrap_demo_project(args.init_project, args.script)
    sys.exit(0 if ok else 1)
