"""Persistence for set3d: ``productions/<P>/bible/sets/<ENV>/`` is the system of record.

    bible/sets/<ENV>/set.json                 the SetLayout (versioned history in set_history/)
    bible/sets/<ENV>/renders/<CAM>/<ts>/      blockout.png, depth.png, camera.json  (immutable)
    bible/sets/<ENV>/styled/<CAM>/            style-pass candidates before promotion to library/
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..fsio import atomic_write_text, compact_ts, next_version_path, validate_id
from ..project import Project
from .schema import CAMERA_NAME_RX, SetLayout

SET_FILE = "set.json"


class SetNotFoundError(FileNotFoundError):
    pass


def sets_dir(project: Project) -> Path:
    return project.path("bible", "sets")


def set_dir(project: Project, env_id: str) -> Path:
    return sets_dir(project) / validate_id(env_id, "environment id")


def set_path(project: Project, env_id: str) -> Path:
    return set_dir(project, env_id) / SET_FILE


def has_set(project: Project, env_id: str) -> bool:
    return set_path(project, env_id).is_file()


def list_sets(project: Project) -> list[str]:
    root = sets_dir(project)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / SET_FILE).is_file())


def load_set(project: Project, env_id: str) -> SetLayout:
    path = set_path(project, env_id)
    if not path.is_file():
        raise SetNotFoundError(f"{env_id} has no 3D set yet — run `set3d build {env_id}` first")
    return SetLayout.model_validate_json(path.read_text(encoding="utf-8"))


def save_set(project: Project, layout: SetLayout, *, bump: bool = True) -> Path:
    """Atomic save; the previous file is kept in ``set_history/`` so an edit never loses a layout."""
    path = set_path(project, layout.environment_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        hist = path.parent / "set_history"
        hist.mkdir(exist_ok=True)
        prev = next_version_path(hist, "set", ".json")
        prev.write_bytes(path.read_bytes())
        if bump:
            layout.version = max(layout.version, _read_version(path) + 1)
    atomic_write_text(path, layout.model_dump_json(indent=2))
    env = project.load_environment(layout.environment_id)
    rel = project.rel(path)
    if env.set3d_uri != rel:
        env.set3d_uri = rel
        project.save_environment(env, confirm_decision_id=craft_decision(
            project, env.id, "annotate", {"set3d_uri": rel}, f"3D set linked to {env.id}"))
    return path


def craft_decision(project: Project, target: str, action: str, payload: dict[str, Any], rationale: str,
                   *, user: str | None = None) -> str:
    """Record the Craft-tier decision that authorises an additive set3d edit on a (possibly locked) Environment."""
    from ..decisions import DecisionLog  # noqa: PLC0415 - avoid import cycle at module load

    d = DecisionLog(project).record(tier="craft", target_artifact_id=target, action=action, payload=payload,
                                    rationale=rationale, gate="G2", user=user or "agy")
    return d.id


def _read_version(path: Path) -> int:
    m = re.search(r'"version":\s*(\d+)', path.read_text(encoding="utf-8"))
    return int(m.group(1)) if m else 1


def camera_name(name: str) -> str:
    if not CAMERA_NAME_RX.fullmatch(name or ""):
        raise ValueError(f"camera name {name!r}: letters, digits, _ or -, max 32, starting with a letter")
    return name


def new_render_dir(project: Project, env_id: str, cam_name: str) -> Path:
    d = set_dir(project, env_id) / "renders" / camera_name(cam_name) / compact_ts()
    n = 0
    base = d
    while d.exists():  # two renders inside one second
        n += 1
        d = base.with_name(f"{base.name}_{n}")
    d.mkdir(parents=True)
    return d


def latest_render_dir(project: Project, env_id: str, cam_name: str) -> Path | None:
    root = set_dir(project, env_id) / "renders" / camera_name(cam_name)
    if not root.is_dir():
        return None
    dirs = sorted((p for p in root.iterdir() if p.is_dir() and (p / "blockout.png").is_file()), key=lambda p: p.name)
    return dirs[-1] if dirs else None


def styled_dir(project: Project, env_id: str, cam_name: str) -> Path:
    d = set_dir(project, env_id) / "styled" / camera_name(cam_name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def render_manifest(render_dir: Path) -> dict[str, Any]:
    import json

    p = render_dir / "camera.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
