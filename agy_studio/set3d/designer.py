"""Set Designer (set3d) — authors the grey-box ``SetLayout`` for one Environment.

Three modes, mirroring the rest of the studio:

* ``llm``    — Gemini (registry kind ``llm``) with the role prompt, the Environment record, the StyleLock, the
              characters active in the location and — when locked — the geometry still as an image part.
* ``manual`` — AGY authored the layout itself (YAML/JSON file) — validated exactly like an LLM answer.
* ``fake``   — a deterministic room derived from the environment id (tests / dry runs, no paid calls).

Every call is logged through ``agent_log`` with the exact prompt and the validated output.
"""

from __future__ import annotations

import colorsys
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Literal

from ..adapters.base import AdapterError, fake_mode, image_mime
from ..agent_log import log_agent_call
from ..agents.roles import get_role
from ..fsio import generated_by, stable_hash
from ..models import Entity, Environment
from ..project import Project
from .schema import Mannequin, Primitive, SetCamera, SetLayout, SetLight, layout_json_schema
from .store import load_set, save_set

DesignMode = Literal["llm", "manual"]
AGENT = "agy.set_designer"
DEFAULT_HEIGHTS = {"child": 1.25, "teen": 1.55, "adult": 1.72}


# ---------------------------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------------------------
def _characters_for(project: Project, env: Environment) -> list[dict[str, Any]]:
    """Characters whose entity-states are active in scenes that take place in this environment."""
    active_states: set[str] = set()
    scoped = False
    try:
        for sc in project.list_scenes():
            if sc.environment_id == env.id:
                scoped = True
                active_states.update(sc.entity_states_active)
    except Exception:  # noqa: BLE001 - scenes may not exist yet
        pass
    out: list[dict[str, Any]] = []
    for ent in project.list_entities():
        if ent.type.value != "character":
            continue
        states = [s for s in ent.states if s.id in active_states] if scoped else list(ent.states)
        if scoped and not states:
            continue
        st = states[0] if states else None
        out.append({
            "character_id": ent.id,
            "name": ent.display_name or ent.id,
            "description": ent.canonical_description,
            "wardrobe_state": st.id if st else None,
            "wardrobe": st.delta_description if st else "",
        })
    return out


def design_context(project: Project, env: Environment) -> dict[str, Any]:
    style = project.load_style_lock()
    return {
        "environment": env.model_dump(mode="json", exclude={"generated_by"}),
        "style_lock": style.model_dump(mode="json", include={"medium", "aspect_ratio", "lens_language"}) if style
        else None,
        "characters": _characters_for(project, env),
        "project": {"aspect_ratio": project.config.settings.aspect_ratio, "fps": project.config.settings.fps},
    }


def _task_text(env: Environment, context: dict[str, Any], steer: str) -> str:
    parts = [
        f"Build the grey-box 3D set for environment {env.id}.",
        "CANONICAL DESCRIPTION:\n" + env.canonical_description.strip(),
    ]
    if env.axis_of_action:
        parts.append("AXIS OF ACTION (text): " + env.axis_of_action)
    if env.lighting_setups:
        parts.append("LIGHTING SETUPS:\n" + "\n".join(f"- {ls.id}: {ls.description}" for ls in env.lighting_setups))
    if context["characters"]:
        parts.append("CHARACTERS IN THIS LOCATION (one mannequin each):\n" + "\n".join(
            f"- {c['character_id']} ({c['name']}): {c['description']} | wardrobe: {c['wardrobe'] or 'n/a'}"
            for c in context["characters"]))
    if context["style_lock"]:
        parts.append("STYLE LOCK: " + json.dumps(context["style_lock"], ensure_ascii=False))
    if env.geometry_still:
        parts.append("A geometry still of this location is attached: reproduce its layout faithfully.")
    if steer.strip():
        parts.append("EDITOR DIRECTION: " + steer.strip())
    parts.append("Return the SetLayout JSON (environment_id must be exactly " + env.id + ").")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------------------------
# fake layout (deterministic)
# ---------------------------------------------------------------------------------------------
def _hue_color(seed: str, s: float = 0.25, v: float = 0.55) -> str:
    h = int(hashlib.sha256(seed.encode()).hexdigest()[:4], 16) / 65535.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def fake_layout(env: Environment, characters: list[dict[str, Any]]) -> SetLayout:
    """A plausible 8 × 3 × 10 m room with one opening, two landmarks, a practical light and mannequins."""
    outdoor = any(w in env.canonical_description.lower() for w in ("terrace", "roof", "street", "outdoor", "garden"))
    w, h, d = (14.0, 6.0, 14.0) if outdoor else (8.0, 3.0, 10.0)
    prims = [
        Primitive(id="floor", kind="box", label="floor", position=(0, -0.05, 0), size=(w, 0.1, d),
                  color="#6e6e6e", material_note="worn concrete", structural=True),
    ]
    wall_h = 1.1 if outdoor else h
    wall_label = "parapet" if outdoor else "wall"
    for name, pos, size in (
        ("back", (0, wall_h / 2, -d / 2), (w, wall_h, 0.2)),
        ("left", (-w / 2, wall_h / 2, 0), (0.2, wall_h, d)),
        ("right", (w / 2, wall_h / 2, 0), (0.2, wall_h, d)),
    ):
        prims.append(Primitive(id=f"{wall_label}_{name}", label=f"{wall_label} {name}", position=pos, size=size,
                               color="#8a8580", material_note="plastered brick", structural=True))
    if outdoor:
        prims.append(Primitive(id="water_tank", kind="cylinder", label="water tank", position=(-w / 2 + 1.5, 0.75,
                               -d / 2 + 1.5), size=(1.2, 1.5, 1.2), color="#3a3a3a", material_note="black plastic"))
        prims.append(Primitive(id="stair_head", label="stairwell head room", position=(w / 2 - 1.6, 1.25, d / 2 - 1.5),
                               size=(2.4, 2.5, 2.6), color="#9c948c", material_note="painted brick", structural=True))
        prims.append(Primitive(id="stair_door", label="stairwell door", position=(w / 2 - 2.81, 1.05, d / 2 - 1.5),
                               size=(0.05, 2.1, 0.9), color="#4b3a2a", material_note="old teak door"))
        lights = [SetLight(id="sun", kind="sun", position=(-20, 15, 10), target=(0, 0, 0), color="#ffd9a8",
                           intensity=3.0, label="late sun"),
                  SetLight(id="street_lamp", kind="point", position=(w / 2 + 2, 6, -2), color="#ffb257",
                           intensity=40.0, label="sodium street lamp")]
    else:
        prims.append(Primitive(id="door", label="door opening", position=(w / 2 - 0.1, 1.05, d / 2 - 1.2),
                               size=(0.05, 2.1, 0.9), color="#4b3a2a", material_note="painted door"))
        prims.append(Primitive(id="table", label="table", position=(0, 0.375, -1.0), size=(1.6, 0.75, 0.8),
                               color="#6b4f35", material_note="dark wood"))
        prims.append(Primitive(id="sofa", label="sofa", position=(-2.2, 0.4, 1.0), size=(2.0, 0.8, 0.9),
                               rotation_y_deg=90, color="#4d5d6b", material_note="worn fabric"))
        lights = [SetLight(id="bulb", kind="point", position=(0, h - 0.3, 0), color="#fff1d6", intensity=25.0,
                           label="bare tungsten bulb"),
                  SetLight(id="window", kind="area", position=(-w / 2 + 0.1, 1.6, -2), target=(0, 1.2, -2),
                           color="#cfe2ff", intensity=8.0, label="daylight through window")]
    mannequins = []
    for i, c in enumerate(characters[:6]):
        mannequins.append(Mannequin(
            id=f"MQ_{c['character_id']}", character_id=c["character_id"], label=c.get("name") or c["character_id"],
            height_m=DEFAULT_HEIGHTS["adult"] - 0.04 * i, position=(-1.0 + 1.6 * i, 0.0, 0.5),
            facing_deg=180.0 if i % 2 else 0.0, wardrobe_color=_hue_color(c["character_id"]),
        ))
    axis = ((-1.0, 0.0, 0.5), (0.6, 0.0, 0.5)) if len(mannequins) >= 2 else None
    return SetLayout(
        environment_id=env.id, title=env.id.replace("_", " ").title(), bounds=(w, h, d),
        floor_color="#6e6e6e", sky_color="#1c2230" if not outdoor else "#2b3a55",
        axis_of_action=axis, primitives=prims, lights=lights, mannequins=mannequins,
        cameras=[SetCamera(name="WIDE", position=(0.0, 1.6, d / 2 - 0.5), look_at=(0.0, 1.2, -1.0), focal_mm=24,
                           aspect_ratio="9:16", note="establishing wide from the open side")],
        notes="Fake deterministic layout for dry runs: open fourth wall on +Z, landmarks along the back.",
        source_still=env.geometry_still,
    )


# ---------------------------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------------------------
def _llm_layout(project: Project, env: Environment, system: str, task: str, context: dict[str, Any],
                registry: Any, temperature: float) -> tuple[dict[str, Any], str]:
    from ..adapters.base import gemini_generate_json, gemini_part_from_file  # noqa: PLC0415
    from ..registry import default_registry  # noqa: PLC0415

    reg = registry or default_registry()
    spec = reg.first_of_kind("llm")
    adapter = reg.adapter(spec.id, project=project, fake=False)
    contents: list[Any] = [task + "\n\nCONTEXT JSON:\n" + json.dumps(context, ensure_ascii=False, indent=1)]
    if env.geometry_still:
        still = project.abs(env.geometry_still)
        if still.is_file():
            contents.append(gemini_part_from_file(still, image_mime(still)))
    raw = gemini_generate_json(adapter.model_chain(), spec.location, system, contents,
                               schema=None, temperature=temperature, client=adapter.client)
    return raw, adapter.provider_model()


def _coerce(raw: Any, env: Environment, source_still: str | None) -> SetLayout:
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise AdapterError("set designer returned no JSON object")
    data = dict(raw)
    data["environment_id"] = env.id
    data.pop("generated_by", None)
    data.pop("version", None)
    data.setdefault("cameras", [])
    data["source_still"] = source_still
    for p in data.get("primitives") or []:
        if isinstance(p, dict) and isinstance(p.get("size"), (list, tuple)) and len(p["size"]) == 3:
            p["size"] = [max(float(x), 0.02) for x in p["size"]]
    for l in data.get("lights") or []:
        if isinstance(l, dict) and l.get("position") is None:
            l["position"] = [0.0, 2.5, 0.0]
    try:
        layout = SetLayout.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        raise AdapterError(f"set layout failed validation: {exc}") from exc
    if not layout.primitives:
        raise AdapterError("set layout has no primitives")
    return layout


# ---------------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------------
def build_set(project: Project, env_id: str, *, mode: DesignMode = "llm", manual_output: Any = None,
              steer: str = "", fake: bool | None = None, registry: Any = None, temperature: float = 0.3,
              keep_cameras: bool = True) -> SetLayout:
    """Author (or re-author) the grey-box set for ``env_id`` and save it as the current layout.

    Existing cameras are carried over when ``keep_cameras`` is true so a rebuilt set never loses the editor's
    framing work.
    """
    env = project.load_environment(env_id)
    role = get_role("set_designer")
    context = design_context(project, env)
    task = _task_text(env, context, steer)
    system = role.load_prompt()
    use_fake = fake_mode() if fake is None else bool(fake)
    started = time.monotonic()
    model_name: str | None = None
    error: str | None = None
    layout: SetLayout | None = None
    mode_used = mode
    try:
        if mode == "manual":
            if manual_output is None:
                raise AdapterError("manual mode needs the layout AGY authored (--manual-file)")
            layout = _coerce(manual_output, env, env.geometry_still)
            model_name = "manual"
        elif use_fake:
            layout = fake_layout(env, context["characters"])
            model_name = "fake"
            mode_used = "llm"
        else:
            raw, model_name = _llm_layout(project, env, system, task, context, registry, temperature)
            layout = _coerce(raw, env, env.geometry_still)
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        raise
    finally:
        log_agent_call(project, agent=AGENT, scene_id=None, working_set_hash=None, task=task,
                       inputs={"environment_id": env_id, "steer": steer, "context_hash": stable_hash(context),
                               "mode": mode_used, "fake": use_fake},
                       outputs={"layout": layout.model_dump(mode="json") if layout else None},
                       model=model_name, mode=mode_used, prompt=f"=== SYSTEM ===\n{system}\n\n=== USER ===\n{task}",
                       duration_s=round(time.monotonic() - started, 3), error=error)
    assert layout is not None
    if keep_cameras:
        try:
            prev = load_set(project, env_id)
            names = {c.name for c in layout.cameras}
            layout.cameras = layout.cameras + [c for c in prev.cameras if c.name not in names]
        except FileNotFoundError:
            pass
    layout.generated_by = generated_by(AGENT, model=model_name, notes=f"mode={mode_used}").model_dump(mode="json")
    save_set(project, layout)
    return layout


def manual_file_layout(path: str | Path) -> Any:
    import yaml  # noqa: PLC0415

    p = Path(path).expanduser()
    return yaml.safe_load(p.read_text(encoding="utf-8"))
