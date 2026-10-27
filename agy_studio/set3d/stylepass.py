"""Style pass (set3d): repaint a blockout render in the production's locked style and promote it.

Input: the latest browser render of a saved camera (``blockout.png`` + ``depth.png`` + ``camera.json``), the
Environment's locked geometry still and the StyleLock. Output: a styled still promoted append-only into
``library/environments/<ENV>/vN/cam_<NAME>.png`` and registered as ``Environment.derived_angles.cameras[NAME]`` so
that a shot with ``environment_angle = "cam:<NAME>"`` resolves it as its ``spatial_anchor``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..fsio import generated_by, next_version_path
from ..project import Project
from .schema import SetCamera, SetLayout
from .store import craft_decision, latest_render_dir, load_set, render_manifest, save_set, styled_dir

AGENT = "agy.set3d.stylepass"


def _visible_mannequins(layout: SetLayout, manifest: dict[str, Any]) -> list[str]:
    names = manifest.get("visible_mannequins")
    if isinstance(names, list):
        ids = {str(n) for n in names}
        return [m.character_id for m in layout.mannequins if m.id in ids or m.character_id in ids]
    return [m.character_id for m in layout.mannequins if m.visible]


def style_prompt(project: Project, layout: SetLayout, cam: SetCamera, manifest: dict[str, Any], *,
                 steer: str = "", with_characters: bool = False) -> str:
    env = project.load_environment(layout.environment_id)
    style = project.load_style_lock()
    lines = [
        f"Generate a new photographic production still of this location from a NEW 3D camera angle ({cam.name}: {cam.note or cam.describe()}).",
        "Reference Image 1 is the 3D grey-box perspective blockout showing the EXACT new camera angle, lens perspective, horizon line, and foreground/background spatial arrangement.",
        "Reference Image 2 shows the room's locked visual design from a wide front angle — use Image 2 ONLY for the surface materials, wall paint, furniture design, props, and lighting atmosphere.",
        f"CRITICAL: DO NOT copy the front wide camera angle of Reference Image 2! You MUST move and rotate the camera inside the 3D room to match the exact perspective, framing, and focal length of Reference Image 1 ({cam.describe()}; {cam.note}).",
        "LOCATION: " + env.canonical_description.strip(),
    ]
    mats = [f"{p.label or p.id}: {p.material_note}" for p in layout.primitives if p.material_note]
    if mats:
        lines.append("MATERIALS of the blocks (match them): " + "; ".join(mats[:40]) + ".")
    if layout.lights:
        lines.append("LIGHT SOURCES (physical only): " + "; ".join(
            f"{l.label or l.id} ({l.kind})" for l in layout.lights) + ".")
    if env.geometry_still:
        lines.append("Reference image 2 is the locked geometry still of this location: match its surfaces, colours, "
                     "weathering, props and atmosphere exactly so both images read as the same real place.")
    if style:
        lines.append(f"STYLE LOCK: medium {style.medium}; lens language {style.lens_language}."
                     + (f" Grade: {style.lut_or_grade_ref}." if style.lut_or_grade_ref else ""))
        if style.donts:
            lines.append("Never: " + "; ".join(style.donts) + ".")
    visible = _visible_mannequins(layout, manifest)
    if visible and not with_characters:
        lines.append("The grey human figures are stand-ins for blocking only: render the location EMPTY of people — "
                     "remove the figures and show what is behind them.")
    elif visible:
        lines.append("The grey human figures mark where characters stand; keep their positions and heights but render "
                     "them as anonymous, out-of-focus silhouettes with no visible faces.")
    lines.append("No text, no watermark, no captions, no lens flare added, no people unless stated.")
    if steer.strip():
        lines.append("EDITOR DIRECTION: " + steer.strip())
    return " ".join(lines)


def stylize_camera(project: Project, env_id: str, cam_name: str, *, image_adapter: Any, steer: str = "",
                   with_characters: bool = False, promote: bool = True, user: str | None = None) -> dict[str, Any]:
    """Run the style pass for one saved camera. Returns ``{"styled": rel, "library": rel | None, "prompt": str}``."""
    layout = load_set(project, env_id)
    cam = layout.camera(cam_name)
    if cam is None:
        raise ValueError(f"{env_id} has no camera {cam_name!r}; saved cameras: "
                         f"{', '.join(c.name for c in layout.cameras) or 'none'}")
    rdir = latest_render_dir(project, env_id, cam_name)
    if rdir is None:
        raise FileNotFoundError(f"camera {cam_name} has not been rendered yet — open the viewer and press "
                                "'Render reference' on it")
    manifest = render_manifest(rdir)
    blockout = rdir / "blockout.png"
    refs: list[str] = [project.rel(blockout)]
    env = project.load_environment(env_id)
    if env.geometry_still:
        refs.append(env.geometry_still)
    prompt = style_prompt(project, layout, cam, manifest, steer=steer, with_characters=with_characters)
    out_dir = styled_dir(project, env_id, cam_name)
    dest = next_version_path(out_dir, "styled", ".png")
    abs_refs = [str(project.abs(r)) for r in refs]
    image_adapter.generate(prompt, dest, reference_images=abs_refs, aspect_ratio=_image_aspect(cam.aspect_ratio))
    meta = {
        "camera": cam.model_dump(mode="json"), "render_dir": project.rel(rdir), "prompt": prompt,
        "reference_images": refs, "model": getattr(image_adapter, "provider_model", lambda: None)(),
        "generated_by": generated_by(AGENT, model=getattr(image_adapter, "model_id", None)).model_dump(mode="json"),
    }
    dest.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    result: dict[str, Any] = {"styled": project.rel(dest), "library": None, "prompt": prompt}
    if promote:
        result["library"] = promote_styled(project, env_id, cam_name, dest, user=user)
    return result


def promote_styled(project: Project, env_id: str, cam_name: str, styled: Path, *, user: str | None = None) -> str:
    """Append-only promotion into library/environments/<ENV>/vN/ and registration on the Environment."""
    env = project.load_environment(env_id)
    prov = generated_by(AGENT, notes=f"camera={cam_name}; approved_by={user or 'agy'}")
    rels = project.promote_library_files(("environments", env_id), {f"cam_{cam_name}.png": styled}, provenance=prov)
    rel = next(iter(rels.values()))
    cams = dict(env.derived_angles.cameras)
    cams[cam_name] = rel
    env.derived_angles = type(env.derived_angles).model_validate({**env.derived_angles.model_dump(), "cameras": cams})
    decision_id = craft_decision(project, env_id, "lock", {"camera": cam_name, "uri": rel},
                                 f"set3d camera {cam_name} styled and locked as angle cam:{cam_name}", user=user)
    project.save_environment(env, confirm_decision_id=decision_id)
    layout = load_set(project, env_id)
    cam = layout.camera(cam_name)
    if cam is not None:
        cam.locked = True
        layout.upsert_camera(cam)
        save_set(project, layout, bump=False)
    return rel


def _image_aspect(aspect: str) -> str:
    return aspect if aspect in ("9:16", "16:9", "1:1", "4:3") else "16:9"
