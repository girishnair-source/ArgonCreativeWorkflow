#!/usr/bin/env python3
"""Minute-one intake — captures the Director's inputs BEFORE the environment exists.

Standard library only: works with any ``python3`` straight after ``git clone`` while ``.venv`` / pip / gcloud are
still being set up in the background. It writes everything the engine needs to start into the production folder:

    productions/<PROJECT_ID>/
        intake.yaml                 mode, format, aspect, language, model policy, 3D opt-in, scope, title
        script/script_v1.<ext>      the screenplay / story brief (copied in)
        library/imports/<Name>/..   character / location reference sheets (copied in, named by character)

Then, once ``live_demo_preflight.py`` is green:

    .venv/bin/python -m agy_studio init <PROJECT_ID> --from-intake
    .venv/bin/python -m agy_studio --project <PROJECT_ID> ingest productions/<PROJECT_ID>/script/script_v1.<ext>
    .venv/bin/python -m agy_studio --project <PROJECT_ID> g1

Example:
    python3 bootstrap_intake.py SPARKY_EP5 --title "Sonic Heroes: The Canyon Echo Rescue" \\
        --mode autopilot --format episode_16_9 --language en --script scriptV1.txt \\
        --ref Sparky=chars/Sparky.jpg --ref Scratchy=chars/Scratchy.jpg --primary-model omni --no-3d
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODES = ("director", "hybrid", "autopilot")
FORMATS = {"microdrama_9_16": "9:16", "episode_16_9": "16:9", "feature_film": "16:9", "marketing_spot": "9:16"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
SCRIPT_EXTS = {".txt", ".fountain", ".md", ".fdx", ".pdf"}


def _yaml_scalar(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return str(v)
    return json.dumps(str(v), ensure_ascii=False)  # JSON string literals are valid YAML scalars


def _emit_yaml(data: dict, indent: int = 0) -> str:
    pad = "  " * indent
    out: list[str] = []
    for k, v in data.items():
        if isinstance(v, dict):
            out.append(f"{pad}{k}:")
            out.append(_emit_yaml(v, indent + 1) if v else f"{pad}  {{}}")
        elif isinstance(v, list):
            out.append(f"{pad}{k}:" + (" []" if not v else ""))
            for item in v:
                if isinstance(item, dict):
                    body = _emit_yaml(item, indent + 2).splitlines()
                    out.append(f"{pad}  - " + body[0].strip())
                    out.extend(body[1:])
                else:
                    out.append(f"{pad}  - {_yaml_scalar(item)}")
        else:
            out.append(f"{pad}{k}: {_yaml_scalar(v)}")
    return "\n".join(line for line in out if line is not None)


def _safe_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    if not slug:
        raise SystemExit(f"invalid name {name!r}")
    return slug


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AGY Studio minute-one intake (no dependencies)")
    ap.add_argument("project_id", help="e.g. SPARKY_EP5 (A-Z, 0-9, _)")
    ap.add_argument("--title", required=True)
    ap.add_argument("--mode", choices=MODES, default="director")
    ap.add_argument("--format", dest="content_format", choices=sorted(FORMATS), default="microdrama_9_16")
    ap.add_argument("--aspect", choices=("9:16", "16:9"), default=None, help="defaults from --format")
    ap.add_argument("--language", default="hi", help="primary dialogue language code / name (hi, en, ta ...)")
    ap.add_argument("--script", type=Path, required=True, help="screenplay or story brief file")
    ap.add_argument("--ref", action="append", default=[], metavar="NAME=path",
                    help="existing reference sheet for a character or location; repeatable, several per name allowed")
    ap.add_argument("--primary-model", default=None, help="video engine registry id (omni | veo)")
    ap.add_argument("--secondary-model", default=None, help="omni | veo | none")
    ap.add_argument("--set3d", dest="set3d", action="store_true", help="the Director wants a 3D blocking set")
    ap.add_argument("--no-3d", dest="set3d", action="store_false")
    ap.set_defaults(set3d=False)
    ap.add_argument("--scope", default="pilot_scene", choices=("pilot_scene", "full_script"))
    ap.add_argument("--takes-per-clip", type=int, default=None)
    ap.add_argument("--workspace", type=Path, default=None,
                    help="folder that holds productions/ (default: this repository checkout)")
    a = ap.parse_args(argv)
    productions = (a.workspace.expanduser().resolve() if a.workspace else ROOT) / "productions"

    pid = a.project_id.strip()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{1,31}", pid):
        raise SystemExit("project_id must be UPPER_SNAKE (A-Z, 0-9, _), e.g. SPARKY_EP5")
    script = a.script.expanduser().resolve()
    if not script.is_file():
        raise SystemExit(f"script not found: {script}")
    if script.suffix.lower() not in SCRIPT_EXTS:
        print(f"warning: unusual script extension {script.suffix}; continuing", file=sys.stderr)

    proj = productions / pid
    (proj / "script").mkdir(parents=True, exist_ok=True)
    script_dest = proj / "script" / f"script_v1{script.suffix.lower() or '.txt'}"
    if not script_dest.exists():
        shutil.copy2(script, script_dest)

    refs: dict[str, list[str]] = {}
    for item in a.ref:
        name, _, path = item.partition("=")
        if not name or not path:
            raise SystemExit(f"--ref expects NAME=path, got {item!r}")
        src = Path(path).expanduser().resolve()
        if not src.is_file() or src.suffix.lower() not in IMAGE_EXTS:
            raise SystemExit(f"--ref {name}: {path} is not an image file ({sorted(IMAGE_EXTS)})")
        folder = proj / "library" / "imports" / _safe_name(name)
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / src.name
        if not dest.exists():
            shutil.copy2(src, dest)
        refs.setdefault(name.strip(), []).append(str(dest.relative_to(proj)))

    aspect = a.aspect or FORMATS[a.content_format]
    intake = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_id": pid,
        "title": a.title,
        "control_mode": a.mode,
        "content_format": a.content_format,
        "aspect_ratio": aspect,
        "language": a.language,
        "scope": a.scope,
        "set3d_requested": bool(a.set3d),
        "script": str(script_dest.relative_to(proj)),
        "script_source": str(script),
        "reference_sheets": refs,
        "model_policy": {"video_model_primary": a.primary_model, "video_model_secondary": a.secondary_model},
        "takes_per_clip": a.takes_per_clip,
    }
    (proj / "intake.yaml").write_text(_emit_yaml(intake) + "\n", encoding="utf-8")

    print(f"Intake captured for {pid} at {proj}")
    print(f"  script : {script_dest.relative_to(proj)}")
    print(f"  refs   : {sum(len(v) for v in refs.values())} file(s) for {', '.join(refs) or 'nobody'}")
    print(f"  mode   : {a.mode} · {a.content_format} · {aspect} · lang={a.language} · 3D={'yes' if a.set3d else 'no'}")
    print("Next (once live_demo_preflight.py is green):")
    print(f"  .venv/bin/python -m agy_studio init {pid} --from-intake")
    print(f"  .venv/bin/python -m agy_studio --project {pid} ingest productions/{pid}/{script_dest.relative_to(proj)}")
    print(f"  .venv/bin/python -m agy_studio --project {pid} g1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
