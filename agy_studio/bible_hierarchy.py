"""Two-Layer Bible Hierarchy (`GlobalContextBible` & `SceneBible`) + Strict 3-Tier Reference Resolver.

Implements the core Bible architecture from the AGY Studio v4 blueprint:
1. **STEP 1 / GATE 1 — ``GLOBAL_KNOWLEDGE / CONTEXT_BIBLE``** (``bible/global_context_bible.yaml`` +
   ``bible/GLOBAL_BIBLE.md``):
   - Single compact knowledge index holding all Scenes, initial Scene-to-Shot Breakdown, Base Character Sheets,
     Base Location Sheets, Props, and Locked Character Voices so AGY never gets bogged down by context bloat.
   - Once approved at Gate 1, base entries in ``GlobalContextBible`` remain **strictly intact** (scene-level overrides
     never mutate the Global Bible; only additive promotions of newly approved voices/props/states are appended).
2. **GATE 2 — ``SCENE BIBLE``** (``scenes/<SC>/scene_bible.yaml``):
   - Created per scene when entering Scene $N$.
   - Automatically checks the Global Library first to **reuse objects/states generated in Scene 1..N-1** (saving $$$
     and time).
   - Stages only **scene-specific deltas** for Gate 2 approval:
     * Character with costume / condition (e.g. injured hero, hero with bruise on face, post-shower damp hair).
     * Conditional location variant *"if needed"* (e.g. hero's bedroom at night, room during sunset).
     * Scene-specific prop states (e.g. sealed parcel vs. torn-open parcel).
3. **GATE 3 — Strict 3-Tier Reference Lookup** (:func:`resolve_reference`):
   - When AGY creates storyboards or keyframes for a shot, it resolves every character, prop, and location in strict
     order:
     1. **Tier 1**: Check ``SceneBible`` (``scenes/<SC>/scene_bible.yaml``)
     2. **Tier 2**: Check ``GlobalContextBible`` (``bible/global_context_bible.yaml``)
     3. **Tier 3**: If still unable to find a reference, open a blocking ``Question`` to **ask the user** (never
        hallucinate an unanchored character or location).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from .continuity import all_scene_states, entity_id_of_state
from .decisions import QuestionLog
from .fsio import generated_by, read_yaml, save_versioned, validate_id
from .models import ApprovalStatus, EntityType, GeneratedBy, Model
from .project import Project

AGENT = "agy.bible_hierarchy"
GLOBAL_BIBLE_REL = "bible/global_context_bible.yaml"
GLOBAL_BIBLE_MD_REL = "bible/GLOBAL_BIBLE.md"


# ---------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------
class ShotMapEntry(Model):
    shot_id: str
    beat_summary: str = ""
    characters: list[str] = Field(default_factory=list)
    environment_id: str | None = None
    camera_angle: str | None = None
    duration_s: float = 4.0


class GlobalCharacterEntry(Model):
    id: str
    name: str
    tier: str = "principal"
    canonical_description: str = ""
    base_state_id: str | None = None
    base_sheet_uri: str | None = None
    turnaround_uris: dict[str, str] = Field(default_factory=dict)
    voice_profile_id: str | None = None
    voice_profile_uri: str | None = None
    voice_anchor_uri: str | None = None
    approval_status: ApprovalStatus = ApprovalStatus.PROPOSED


class GlobalLocationEntry(Model):
    id: str
    canonical_description: str = ""
    base_sheet_uri: str | None = None
    set3d_uri: str | None = None
    blender_dir_uri: str | None = None
    cameras: dict[str, str] = Field(default_factory=dict)
    approval_status: ApprovalStatus = ApprovalStatus.PROPOSED


class GlobalPropEntry(Model):
    id: str
    name: str
    canonical_description: str = ""
    base_state_id: str | None = None
    base_sheet_uri: str | None = None
    state_sheets: dict[str, str] = Field(default_factory=dict)
    approval_status: ApprovalStatus = ApprovalStatus.PROPOSED


class GlobalContextBible(Model):
    """``bible/global_context_bible.yaml`` — The single Global Knowledge & Context Bible locked at Gate 1."""

    project_id: str
    title: str
    aspect_ratio: str = "9:16"
    language: str = "hi"
    control_mode: str = "director"
    content_format: str = "microdrama_9_16"
    style_lock_uri: str = "bible/style_lock.yaml"
    characters: dict[str, GlobalCharacterEntry] = Field(default_factory=dict)
    locations: dict[str, GlobalLocationEntry] = Field(default_factory=dict)
    props: dict[str, GlobalPropEntry] = Field(default_factory=dict)
    scene_shot_map: dict[str, list[ShotMapEntry]] = Field(default_factory=dict)
    reusable_states_library: dict[str, str] = Field(default_factory=dict)  # state_id -> primary library image URI
    approval_status: ApprovalStatus = ApprovalStatus.PROPOSED
    version: int = 1
    generated_by: GeneratedBy | None = None


class SceneCharacterDelta(Model):
    character_id: str
    state_id: str
    costume_or_condition: str = ""
    sheet_uri: str | None = None
    turnaround_uris: dict[str, str] = Field(default_factory=dict)
    reused_from_global: bool = False
    approval_status: ApprovalStatus = ApprovalStatus.PROPOSED


class SceneEnvironmentVariant(Model):
    environment_id: str
    variant_id: str  # e.g. ENV_RISHIS_ROOM_DEFAULT or ENV_RISHIS_ROOM_NIGHT
    time_of_day_or_mood: str = "default"
    if_needed_override: bool = False  # True when a special night/sunset/weather plate was generated for this scene
    sheet_uri: str | None = None
    cameras: dict[str, str] = Field(default_factory=dict)
    reused_from_global: bool = True
    approval_status: ApprovalStatus = ApprovalStatus.PROPOSED


class ScenePropDelta(Model):
    prop_id: str
    state_id: str
    description: str = ""
    sheet_uri: str | None = None
    reused_from_global: bool = False
    approval_status: ApprovalStatus = ApprovalStatus.PROPOSED


class SceneBible(Model):
    """``scenes/<SC>/scene_bible.yaml`` — Per-scene mutable Bible created at Gate 2."""

    scene_id: str
    slug: str = ""
    global_bible_ref: str = GLOBAL_BIBLE_REL
    character_overrides: dict[str, SceneCharacterDelta] = Field(default_factory=dict)
    environment_variant: SceneEnvironmentVariant | None = None
    scene_props: dict[str, ScenePropDelta] = Field(default_factory=dict)
    reused_asset_ids: list[str] = Field(default_factory=list)
    newly_generated_asset_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    approval_status: ApprovalStatus = ApprovalStatus.PROPOSED
    version: int = 1
    generated_by: GeneratedBy | None = None

    @property
    def character_states(self) -> dict[str, SceneCharacterDelta]:
        return self.character_overrides

    @property
    def angle_3d_references(self) -> dict[str, str]:
        return dict(self.environment_variant.cameras) if self.environment_variant else {}


class ResolvedReference(Model):
    """Result of the 3-tier reference lookup (SceneBible -> GlobalContextBible -> AskUser)."""

    query_id: str
    scene_id: str
    source_tier: Literal["scene_bible", "global_bible", "ask_user"]
    resolved_id: str | None = None
    sheet_uri: str | None = None
    extra_uris: dict[str, str] = Field(default_factory=dict)
    description: str = ""
    question_id: str | None = None


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------
def _primary_pack_uri(pack: Any) -> str | None:
    t = pack.turnaround
    for cand in (t.front, pack.face_closeup, t.three_quarter, pack.single_ref, t.profile, t.rear):
        if cand:
            return str(cand)
    return None


def _turnaround_dict(pack: Any) -> dict[str, str]:
    t = pack.turnaround
    raw = {
        "front": t.front,
        "three_quarter": t.three_quarter,
        "profile": t.profile,
        "rear": t.rear,
        "face_closeup": pack.face_closeup,
        "single_ref": pack.single_ref,
    }
    return {k: str(v) for k, v in raw.items() if v}


def global_bible_path(project: Project) -> Path:
    return project.abs(GLOBAL_BIBLE_REL)


def scene_bible_path(project: Project, scene_id: str) -> Path:
    validate_id(scene_id, "scene id")
    return project.scene_dir(scene_id) / "scene_bible.yaml"


# ---------------------------------------------------------------------------------------------
# Global Context Bible (Step 1 / Gate 1)
# ---------------------------------------------------------------------------------------------
def build_global_context_bible(
    project: Project,
    *,
    control_mode: Any = None,
    content_format: Any = None,
    save: bool = True,
) -> GlobalContextBible:
    """Build or refresh ``bible/global_context_bible.yaml`` and ``bible/GLOBAL_BIBLE.md`` from project records."""
    cfg = project.config
    settings = cfg.settings
    if control_mode is not None or content_format is not None:
        updates: dict[str, Any] = {}
        if control_mode is not None:
            updates["control_mode"] = str(getattr(control_mode, "value", control_mode))
        if content_format is not None:
            updates["content_format"] = str(getattr(content_format, "value", content_format))
        settings = settings.model_copy(update=updates)
        cfg.settings = settings
        if save:
            project.save_config(cfg)

    characters: dict[str, GlobalCharacterEntry] = {}
    props: dict[str, GlobalPropEntry] = {}
    reusable_states: dict[str, str] = {}

    casting_map = (project.casting().get("voice_profiles") or {}) if hasattr(project, "casting") else {}

    for ent in project.list_entities():
        state_sheets: dict[str, str] = {}
        base_state_id: str | None = None
        base_sheet: str | None = None
        base_turnarounds: dict[str, str] = {}

        for st in ent.states:
            uri = _primary_pack_uri(st.reference_pack)
            if uri:
                state_sheets[st.id] = uri
                if st.approval_status == ApprovalStatus.LOCKED or str(uri).startswith("library/"):
                    reusable_states[st.id] = uri
            if base_state_id is None:
                base_state_id = st.id
                base_sheet = uri
                base_turnarounds = _turnaround_dict(st.reference_pack)

        ent_name = getattr(ent, "display_name", None) or getattr(ent, "name", None) or ent.id
        if ent.type == EntityType.CHARACTER:
            vp = project.load_voice_profile(ent.id)
            c_entry = casting_map.get(ent.id) if isinstance(casting_map, dict) else None
            lib_v = c_entry.get("library_version") if isinstance(c_entry, dict) else None
            vp_uri = (
                project.rel(project.voice_profile_path(ent.id, int(lib_v)))
                if (lib_v and hasattr(project, "voice_profile_path"))
                else f"bible/voice_profiles/{ent.id}.yaml"
            )
            characters[ent.id] = GlobalCharacterEntry(
                id=ent.id,
                name=ent_name,
                tier=ent.tier.value,
                canonical_description=ent.canonical_description,
                base_state_id=base_state_id,
                base_sheet_uri=base_sheet,
                turnaround_uris=base_turnarounds,
                voice_profile_id=vp.id if vp else f"VOICE_{ent.id}",
                voice_profile_uri=vp_uri,
                voice_anchor_uri=(vp.reference_audio or vp.anchor_wav) if vp else None,
                approval_status=ent.approval_status,
            )
        else:
            props[ent.id] = GlobalPropEntry(
                id=ent.id,
                name=ent_name,
                canonical_description=ent.canonical_description,
                base_state_id=base_state_id,
                base_sheet_uri=base_sheet,
                state_sheets=state_sheets,
                approval_status=ent.approval_status,
            )

    locations: dict[str, GlobalLocationEntry] = {}
    for env in project.list_environments():
        cams = dict(env.derived_angles.cameras or {})
        blender_rel = f"bible/sets/{env.id}/blender"
        locations[env.id] = GlobalLocationEntry(
            id=env.id,
            canonical_description=env.canonical_description,
            base_sheet_uri=env.geometry_still,
            set3d_uri=env.set3d_uri,
            blender_dir_uri=blender_rel if project.abs(blender_rel).is_dir() else None,
            cameras=cams,
            approval_status=env.approval_status,
        )

    scene_shot_map: dict[str, list[ShotMapEntry]] = {}
    for sc in project.list_scenes():
        shots = project.list_shots(sc.id)
        entries: list[ShotMapEntry] = []
        if shots:
            for sh in shots:
                summary_txt = getattr(sh, "intention", "") or getattr(sh, "description", "") or sh.spec.visual.subject_and_action
                chars_in_shot = list(getattr(sh, "characters_in_frame", None) or sc.entity_states_active)
                entries.append(
                    ShotMapEntry(
                        shot_id=sh.id,
                        beat_summary=summary_txt,
                        characters=chars_in_shot,
                        environment_id=sc.environment_id,
                        camera_angle=sh.environment_angle,
                        duration_s=sh.duration_s,
                    )
                )
        else:
            # Synthesize initial Gate 1 shot breakdown from scene beats so Gate 1 always has a shot map
            for idx, beat in enumerate(sc.beats, start=1):
                est_dur = getattr(beat, "estimated_duration_s", None) or 4.0
                entries.append(
                    ShotMapEntry(
                        shot_id=f"{sc.id}_SH{idx:02d}",
                        beat_summary=beat.summary,
                        characters=list(sc.entity_states_active),
                        environment_id=sc.environment_id,
                        camera_angle="cam:WIDE_MASTER" if idx == 1 else "cam:OTS_A_TO_B",
                        duration_s=max(3.0, float(est_dur)),
                    )
                )
        scene_shot_map[sc.id] = entries

    g1_passed = False
    try:
        from .state import StateStore

        g1_passed = StateStore(project).load().gates.get("G1") == "passed"
    except Exception:  # noqa: BLE001
        pass

    gb = GlobalContextBible(
        project_id=cfg.project_id,
        title=cfg.title,
        aspect_ratio=settings.aspect_ratio,
        language=cfg.language,
        control_mode=getattr(settings, "control_mode", "director"),
        content_format=getattr(settings, "content_format", "microdrama_9_16"),
        characters=characters,
        locations=locations,
        props=props,
        scene_shot_map=scene_shot_map,
        reusable_states_library=reusable_states,
        approval_status=ApprovalStatus.LOCKED if g1_passed else ApprovalStatus.PROPOSED,
        generated_by=generated_by(AGENT),
    )
    if save:
        save_versioned(global_bible_path(project), gb)
        _write_global_bible_markdown(project, gb)
    return gb


def load_global_context_bible(project: Project) -> GlobalContextBible:
    path = global_bible_path(project)
    if path.is_file():
        return GlobalContextBible.model_validate(read_yaml(path))
    return build_global_context_bible(project, save=True)


def _write_global_bible_markdown(project: Project, gb: GlobalContextBible) -> Path:
    style = project.load_style_lock()
    lines: list[str] = [
        f"# Global Knowledge & Context Bible — `{gb.project_id}` ({gb.title})",
        "",
        f"- **Format**: `{gb.content_format}` (`{gb.aspect_ratio}`) · **Language**: `{gb.language}` · **Control Mode**: `{gb.control_mode}`",
        f"- **Gate 1 Status**: `{gb.approval_status.value}`",
        "",
    ]
    if style is not None:
        lines += [
            "## 0. Visual StyleLock & Cinema Optical Package",
            "",
            f"- **Medium & Aesthetic**: {style.medium} (`{style.aspect_ratio}`, `{style.approval_status.value}`)",
            f"- **Camera & Lens Language**: {style.lens_language}",
            f"- **Color Grade / LUT Reference**: {style.lut_or_grade_ref or 'natural cinematic grade'}",
            f"- **Reference Films**: {', '.join(style.reference_films) if style.reference_films else 'none'}",
            f"- **Creative Don'ts**: {', '.join(style.donts) if style.donts else 'none'}",
            f"- **Global Negative Constraints**: {', '.join(style.global_negative) if style.global_negative else 'none'}",
            "",
        ]
    lines += [
        "## 1. Canonical Cast & Voice Registry",
        "",
        "| Character ID | Name | Tier | Base Reference Sheet | Voice Profile | Voice Anchor |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for cid, ch in gb.characters.items():
        lines.append(
            f"| `{cid}` | {ch.name} | `{ch.tier}` | `{ch.base_sheet_uri or 'pending'}` | "
            f"`{ch.voice_profile_id or 'pending'}` | `{ch.voice_anchor_uri or 'set from first approved take'}` |"
        )
    lines.append("")
    for cid, ch in gb.characters.items():
        vp = project.load_voice_profile(cid)
        vp_desc = vp.description if vp else "Voice profile pending."
        lines += [
            f"### `{cid}` — {ch.name} (`{ch.tier}`, `{ch.approval_status.value}`)",
            f"- **Canonical Visual & Wardrobe Description**: {ch.canonical_description}",
            f"- **Voice Timbre & Delivery (`{ch.voice_profile_id}`)**: {vp_desc}",
            f"- **Base Reference Sheet**: `{ch.base_sheet_uri or 'pending at Gate G2'}`",
            "",
        ]

    lines += [
        "## 2. Canonical Locations & 3D Virtual Sets",
        "",
        "| Environment ID | Base Geometry Sheet | 3D Set (`set3d`) | Locked `cam:` Angles |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for eid, loc in gb.locations.items():
        cams = ", ".join(f"`cam:{k}`" for k in loc.cameras) if loc.cameras else "none"
        lines.append(f"| `{eid}` | `{loc.base_sheet_uri or 'pending'}` | `{loc.set3d_uri or 'on-demand at G2/G3'}` | {cams} |")
    lines.append("")
    for eid, loc in gb.locations.items():
        lines += [
            f"### `{eid}` (`{loc.approval_status.value}`)",
            f"- **Architectural & Lighting Description**: {loc.canonical_description}",
            f"- **Master Plate**: `{loc.base_sheet_uri or 'pending'}` · **3D Set**: `{loc.set3d_uri or 'pending'}`",
            "",
        ]

    lines += [
        "## 3. Canonical Hero Props",
        "",
        "| Prop ID | Name | Base Reference Sheet | States Available |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for pid, pr in gb.props.items():
        states = ", ".join(f"`{k}`" for k in pr.state_sheets) if pr.state_sheets else f"`{pr.base_state_id}`"
        lines.append(f"| `{pid}` | {pr.name} | `{pr.base_sheet_uri or 'pending'}` | {states} |")
    lines.append("")
    for pid, pr in gb.props.items():
        lines += [
            f"### `{pid}` — {pr.name} (`{pr.approval_status.value}`)",
            f"- **Canonical Prop Description**: {pr.canonical_description}",
            "",
        ]

    lines += [
        "## 4. Scene-to-Shot Breakdown Map",
        "",
    ]
    for sid, shots in gb.scene_shot_map.items():
        lines.append(f"### `{sid}` ({len(shots)} shots)")
        for sh in shots:
            lines.append(f"- **`{sh.shot_id}`** (`{sh.duration_s:g}s`, `{sh.camera_angle or 'default'}`): {sh.beat_summary}")
        lines.append("")

    md_path = project.abs(GLOBAL_BIBLE_MD_REL)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return md_path


# ---------------------------------------------------------------------------------------------
# Scene Bible (Gate 2 — Scene-Wise Delta + Cross-Scene Reuse)
# ---------------------------------------------------------------------------------------------
def build_scene_bible(
    project: Project,
    scene_id: str,
    *,
    environment_time_of_day: str | None = None,
    environment_override_sheet: str | None = None,
    save: bool = True,
) -> SceneBible:
    """Create or update ``scenes/<SC>/scene_bible.yaml`` without ever mutating ``GlobalContextBible``."""
    validate_id(scene_id, "scene id")
    gb = load_global_context_bible(project)
    scene = project.load_scene(scene_id)
    state_index = project.entity_state_index()
    all_ent_ids = [e.id for e in project.list_entities()]

    char_overrides: dict[str, SceneCharacterDelta] = {}
    scene_props: dict[str, ScenePropDelta] = {}
    reused_ids: list[str] = []
    new_ids: list[str] = []

    for state_id in all_scene_states(scene):
        ent_id = entity_id_of_state(state_id, all_ent_ids) or state_id.split("_STATE_", 1)[0]
        hit = state_index.get(state_id)
        if hit is not None:
            ent, st = hit
            sheet_uri = _primary_pack_uri(st.reference_pack)
            turnarounds = _turnaround_dict(st.reference_pack)
            reused = bool(
                state_id in gb.reusable_states_library
                or (sheet_uri and str(sheet_uri).startswith("library/"))
            )
            if reused:
                reused_ids.append(state_id)
            else:
                new_ids.append(state_id)
            st_desc = getattr(st, "delta_description", "") or getattr(st, "description", "")
            if ent.type == EntityType.CHARACTER:
                char_overrides[ent.id] = SceneCharacterDelta(
                    character_id=ent.id,
                    state_id=state_id,
                    costume_or_condition=st_desc,
                    sheet_uri=sheet_uri or gb.characters.get(ent.id, GlobalCharacterEntry(id=ent.id, name=ent.id)).base_sheet_uri,
                    turnaround_uris=turnarounds,
                    reused_from_global=reused,
                    approval_status=st.approval_status,
                )
            else:
                scene_props[ent.id] = ScenePropDelta(
                    prop_id=ent.id,
                    state_id=state_id,
                    description=st_desc,
                    sheet_uri=sheet_uri,
                    reused_from_global=reused,
                    approval_status=st.approval_status,
                )
        else:
            new_ids.append(state_id)
            if ent_id.startswith("PROP_"):
                scene_props[ent_id] = ScenePropDelta(
                    prop_id=ent_id,
                    state_id=state_id,
                    description=f"Scene {scene_id} state {state_id}",
                    sheet_uri=None,
                    reused_from_global=False,
                    approval_status=ApprovalStatus.PROPOSED,
                )
            else:
                char_overrides[ent_id] = SceneCharacterDelta(
                    character_id=ent_id,
                    state_id=state_id,
                    costume_or_condition=f"Scene {scene_id} look {state_id}",
                    sheet_uri=None,
                    reused_from_global=False,
                    approval_status=ApprovalStatus.PROPOSED,
                )

    env_variant: SceneEnvironmentVariant | None = None
    if scene.environment_id:
        eid = scene.environment_id
        gloc = gb.locations.get(eid)
        tod = environment_time_of_day or (scene.lighting_setup_id or "default")
        if_needed = bool(environment_override_sheet)
        sheet = environment_override_sheet or (gloc.base_sheet_uri if gloc else None)
        cams = dict(gloc.cameras) if gloc else {}
        if not if_needed and sheet:
            reused_ids.append(eid)
        elif if_needed:
            new_ids.append(f"{eid}_{tod.upper()}")
        env_variant = SceneEnvironmentVariant(
            environment_id=eid,
            variant_id=f"{eid}_{tod.upper()}" if if_needed else eid,
            time_of_day_or_mood=tod,
            if_needed_override=if_needed,
            sheet_uri=sheet,
            cameras=cams,
            reused_from_global=not if_needed,
            approval_status=ApprovalStatus.LOCKED if (gloc and gloc.approval_status == ApprovalStatus.LOCKED and not if_needed) else ApprovalStatus.PROPOSED,
        )

    sb = SceneBible(
        scene_id=scene_id,
        slug=scene.slug,
        character_overrides=char_overrides,
        environment_variant=env_variant,
        scene_props=scene_props,
        reused_asset_ids=list(dict.fromkeys(reused_ids)),
        newly_generated_asset_ids=list(dict.fromkeys(new_ids)),
        approval_status=ApprovalStatus.LOCKED if not new_ids else ApprovalStatus.PROPOSED,
        generated_by=generated_by(AGENT),
    )
    if save:
        save_versioned(scene_bible_path(project, scene_id), sb)
        _write_scene_bible_markdown(project, sb)
    return sb


def _write_scene_bible_markdown(project: Project, sb: SceneBible) -> Path:
    lines: list[str] = [
        f"# Scene Bible — `{sb.scene_id}` ({sb.slug})",
        "",
        f"- **Gate 2 Status**: `{sb.approval_status.value}`",
        f"- **Reused From Global Bible (`$0` Token Cost)**: {', '.join(f'`{x}`' for x in sb.reused_asset_ids) if sb.reused_asset_ids else 'none'}",
        f"- **Scene-Specific Deltas / New Assets**: {', '.join(f'`{x}`' for x in sb.newly_generated_asset_ids) if sb.newly_generated_asset_ids else 'none'}",
        "",
        "## 1. Scene Character States",
        "",
        "| Character ID | Active State ID | Costume / Condition | Reference Sheet | Reused? |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for cid, ch in sb.character_overrides.items():
        lines.append(
            f"| `{cid}` | `{ch.state_id}` | {ch.costume_or_condition or 'default'} | "
            f"`{ch.sheet_uri or 'pending'}` | `{'Yes' if ch.reused_from_global else 'New Delta'}` |"
        )
    lines += [
        "",
        "## 2. Scene Location & 3D Camera Angles",
        "",
    ]
    if sb.environment_variant:
        ev = sb.environment_variant
        cams = ", ".join(f"`cam:{k}`" for k in ev.cameras) if ev.cameras else "none"
        lines += [
            f"- **Environment**: `{ev.environment_id}` (Variant: `{ev.variant_id}`, Mood/Time: `{ev.time_of_day_or_mood}`)",
            f"- **Master Plate**: `{ev.sheet_uri or 'pending'}` · **Reused From Global**: `{'Yes' if ev.reused_from_global else 'Scene Override'}`",
            f"- **Available 3D Camera Angles**: {cams}",
            "",
        ]
    lines += [
        "## 3. Scene Props",
        "",
        "| Prop ID | Active State ID | Description | Reference Sheet | Reused? |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for pid, pr in sb.scene_props.items():
        lines.append(
            f"| `{pid}` | `{pr.state_id}` | {pr.description or 'default'} | "
            f"`{pr.sheet_uri or 'pending'}` | `{'Yes' if pr.reused_from_global else 'New Delta'}` |"
        )
    md_path = project.scene_dir(sb.scene_id) / "SCENE_BIBLE.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return md_path


def load_scene_bible(project: Project, scene_id: str) -> SceneBible:
    path = scene_bible_path(project, scene_id)
    if path.is_file():
        return SceneBible.model_validate(read_yaml(path))
    return build_scene_bible(project, scene_id, save=True)


# ---------------------------------------------------------------------------------------------
# Strict 3-Tier Reference Resolver (1. Scene Bible -> 2. Global Bible -> 3. Ask User)
# ---------------------------------------------------------------------------------------------
def resolve_reference(
    project: Project,
    scene_id: str,
    query_id: str,
    *,
    ask_on_missing: bool = True,
) -> ResolvedReference:
    """Resolve a character, state, prop, or environment reference using the strict 3-tier hierarchy:
    1. First check ``SceneBible`` (``scenes/<SC>/scene_bible.yaml``)
    2. Second check ``GlobalContextBible`` (``bible/global_context_bible.yaml``)
    3. Third, if still missing, open a blocking ``Question`` to ask the user.
    """
    validate_id(scene_id, "scene id")
    qid = query_id.strip().upper()
    sb = load_scene_bible(project, scene_id)
    gb = load_global_context_bible(project)

    # --- TIER 1: Check Scene Bible ---
    if qid in sb.character_overrides:
        ch_delta = sb.character_overrides[qid]
        if ch_delta.sheet_uri:
            return ResolvedReference(
                query_id=qid,
                scene_id=scene_id,
                source_tier="scene_bible",
                resolved_id=ch_delta.state_id,
                sheet_uri=ch_delta.sheet_uri,
                extra_uris=dict(ch_delta.turnaround_uris),
                description=ch_delta.costume_or_condition,
            )
    for ch_delta in sb.character_overrides.values():
        if ch_delta.state_id == qid and ch_delta.sheet_uri:
            return ResolvedReference(
                query_id=qid,
                scene_id=scene_id,
                source_tier="scene_bible",
                resolved_id=ch_delta.state_id,
                sheet_uri=ch_delta.sheet_uri,
                extra_uris=dict(ch_delta.turnaround_uris),
                description=ch_delta.costume_or_condition,
            )
    if qid in sb.scene_props:
        pr_delta = sb.scene_props[qid]
        if pr_delta.sheet_uri:
            return ResolvedReference(
                query_id=qid,
                scene_id=scene_id,
                source_tier="scene_bible",
                resolved_id=pr_delta.state_id,
                sheet_uri=pr_delta.sheet_uri,
                description=pr_delta.description,
            )
    for pr_delta in sb.scene_props.values():
        if pr_delta.state_id == qid and pr_delta.sheet_uri:
            return ResolvedReference(
                query_id=qid,
                scene_id=scene_id,
                source_tier="scene_bible",
                resolved_id=pr_delta.state_id,
                sheet_uri=pr_delta.sheet_uri,
                description=pr_delta.description,
            )
    if sb.environment_variant and qid in (sb.environment_variant.environment_id, sb.environment_variant.variant_id):
        ev = sb.environment_variant
        if ev.if_needed_override and ev.sheet_uri:
            return ResolvedReference(
                query_id=qid,
                scene_id=scene_id,
                source_tier="scene_bible",
                resolved_id=ev.variant_id,
                sheet_uri=ev.sheet_uri,
                extra_uris=dict(ev.cameras),
                description=ev.time_of_day_or_mood,
            )

    # --- TIER 2: Fallback to Global Context Bible (Intact / Immutable) ---
    base_entity_id = qid.split("_STATE_", 1)[0] if "_STATE_" in qid else qid
    if base_entity_id in gb.characters:
        gch = gb.characters[base_entity_id]
        sheet = gb.reusable_states_library.get(qid) or gch.base_sheet_uri
        if sheet:
            return ResolvedReference(
                query_id=qid,
                scene_id=scene_id,
                source_tier="global_bible",
                resolved_id=gch.base_state_id or gch.id,
                sheet_uri=sheet,
                extra_uris=dict(gch.turnaround_uris),
                description=gch.canonical_description,
            )
    if base_entity_id in gb.props:
        gpr = gb.props[base_entity_id]
        sheet = gpr.state_sheets.get(qid) or gpr.base_sheet_uri
        if sheet:
            return ResolvedReference(
                query_id=qid,
                scene_id=scene_id,
                source_tier="global_bible",
                resolved_id=qid if qid in gpr.state_sheets else (gpr.base_state_id or gpr.id),
                sheet_uri=sheet,
                extra_uris=dict(gpr.state_sheets),
                description=gpr.canonical_description,
            )
    if qid in gb.locations:
        gloc = gb.locations[qid]
        if gloc.base_sheet_uri:
            return ResolvedReference(
                query_id=qid,
                scene_id=scene_id,
                source_tier="global_bible",
                resolved_id=gloc.id,
                sheet_uri=gloc.base_sheet_uri,
                extra_uris=dict(gloc.cameras),
                description=gloc.canonical_description,
            )

    # --- TIER 3: Unable to find in Scene Bible or Global Bible -> Ask User ---
    question_id: str | None = None
    if ask_on_missing:
        qlog = QuestionLog(project)
        q = qlog.ask(
            f"Reference for {qid} in {scene_id} was not found in either the Scene Bible or Global Bible. "
            f"Would you like to provide an existing reference image path or have AGY generate a new sheet?",
            scene_id=scene_id,
            stage="reference_lookup",
            blocks=[f"{scene_id}:{qid}"],
            options=[
                f"Generate a new reference sheet for {qid} in {scene_id} Scene Bible",
                f"Import an existing reference image file for {qid}",
            ],
        )
        question_id = q.id
    return ResolvedReference(
        query_id=qid,
        scene_id=scene_id,
        source_tier="ask_user",
        resolved_id=None,
        sheet_uri=None,
        description=f"Missing reference for {qid}; user input requested.",
        question_id=question_id,
    )


def promote_scene_assets_to_global_library(project: Project, scene_id: str) -> list[str]:
    """Promote newly approved Scene Bible assets into ``GlobalContextBible.reusable_states_library``
    without overwriting any existing base Global Bible sheets, so Scene N+1 can reuse them at $0 cost.
    """
    sb = load_scene_bible(project, scene_id)
    gb = load_global_context_bible(project)
    promoted: list[str] = []

    for ch in sb.character_overrides.values():
        if ch.sheet_uri and ch.state_id not in gb.reusable_states_library:
            gb.reusable_states_library[ch.state_id] = ch.sheet_uri
            promoted.append(ch.state_id)
    for pr in sb.scene_props.values():
        if pr.sheet_uri and pr.state_id not in gb.reusable_states_library:
            gb.reusable_states_library[pr.state_id] = pr.sheet_uri
            promoted.append(pr.state_id)
            if pr.prop_id in gb.props and pr.state_id not in gb.props[pr.prop_id].state_sheets:
                gb.props[pr.prop_id].state_sheets[pr.state_id] = pr.sheet_uri

    if promoted:
        gb.version += 1
        gb.generated_by = generated_by(AGENT)
        save_versioned(global_bible_path(project), gb)
        _write_global_bible_markdown(project, gb)
    return promoted
