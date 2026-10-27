"""Generic & Scalable Multi-Format Production Bible Engine (`production_bible.py`).

Scales seamlessly across:
- Cartoons (2D Cel-Shaded Animation, 3D Stylized Animation, Anime, Stop-Motion)
- Short 20-Minute Episodes (Multi-Act, 15-30 Scenes, 150-250+ Shots per episode)
- Feature Movies (Multi-Reel, Multi-Act theatrical 16:9 or 2.39:1 widescreen)
- Vertical Microdramas (1-2 Minute 9:16 mobile episodes)

Maintains strict Character, Voice, and Visual Style Consistency across episodes and locations via:
1. `GlobalStyleLock`: Medium-aware positive rendering rules and negative anti-drift locks.
2. `CastingPreflightGate`: Integration with `casting_agent.py` to reuse existing character and
   location reference sheets before spending generation tokens.
3. `DialogueContinuityGate`: Integration with `dialogue_agent.py` for human word-count pacing,
   immutable character voice prints, and single-speaker lip-lock isolation.
4. `SceneBoundaryReset`: Automatically resets previous-shot image conditioning at scene/location
   transitions so lighting and backgrounds never bleed between locations.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from casting_agent import CastingPreflightReport, preflight_reference_sheet_check
from dialogue_agent import (
    CharacterVoiceProfile,
    DialoguePacingReport,
    SceneDialogueLine,
    compile_speaker_isolation_prompt,
    validate_dialogue_pacing,
)
from schemas import (
    AnchorSheetSpec,
    AspectRatio,
    CharacterCastSpec,
    CompiledShotPrompt,
    HeroPropSpec,
    LocationScoutSpec,
    ShotAudioSpec,
    ShotBreakdownSpec,
    StrictContract,
)


def _file_fingerprint(path: Path) -> str:
    """Content hash so a re-shot keyframe saved under the same filename is detected."""
    return hashlib.sha1(path.read_bytes()).hexdigest()[:16]


@dataclass
class DialogueLine:
    speaker: str
    voice_profile: str
    delivery: str
    line_hinglish: str
    language: str = "Hindi"


@dataclass
class I2VShotDirection:
    """One beat of motion direction layered on top of a locked start keyframe."""

    shot_id: str
    intent: str
    camera_motion: str
    subject_action: str
    performance_physics: str
    scene_motion_timing: str
    sound_design: str
    continuity_end_state: str
    dialogue: list[DialogueLine] = field(default_factory=list)
    extra_negatives: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PromptLocks:
    """Episode-level prompt locks (style / continuity / language / negatives)."""

    style: str = (
        "Photorealistic live-action cinema, 24fps natural motion blur, true-to-life skin and "
        "fabric texture, subtle film grain. Single continuous take with no cuts, no transitions."
    )
    continuity: str = (
        "Keep the exact faces, hairstyles, clothing, props, location geometry and lighting of the first "
        "frame unchanged for the entire shot. No new characters enter."
    )
    continuity_ingredient: str = (
        "Begin the shot exactly on the composition of the attached reference image: same camera angle, same framing, "
        "same placement of every person and object. Keep exact faces, hairstyles, clothing, props, and lighting unchanged."
    )
    language: str = (
        "On-camera speakers' lips move in sync with their words. No subtitles, captions or text on screen."
    )
    global_negative: str = (
        "floating UI, notification icons floating in the air, holographic HUD, on-screen text, subtitles, captions, "
        "watermark, logo, wardrobe change, different clothes, duplicate person, extra limbs, distorted hands, "
        "jump cut, scene change, black frames"
    )


def compile_i2v_prompt(
    direction: I2VShotDirection,
    mode: str = "image-to-video",
    locks: PromptLocks | None = None,
) -> str:
    """Compile a structured Image-to-Video direction into the Filmmaker framework prompt string."""
    lk = locks or PromptLocks()
    parts: list[str] = [
        f"{direction.camera_motion} as {direction.subject_action}.",
        direction.performance_physics,
        direction.scene_motion_timing,
    ]
    if direction.dialogue:
        lines = []
        for d in direction.dialogue:
            lang = getattr(d, "language", "Hindi") or "Hindi"
            lines.append(
                f'{d.speaker} ({d.voice_profile}, {d.delivery}) says in {lang}: "{d.line_hinglish}"'
            )
        parts.append("Dialogue: " + " Then ".join(lines) + ".")
        parts.append(lk.language)
    else:
        parts.append("No dialogue, no voice-over, no on-screen text.")
    parts.append(f"Sound: {direction.sound_design}")
    parts.append(f"End state: {direction.continuity_end_state}")
    parts.append(lk.continuity_ingredient if mode == "reference-to-video" else lk.continuity)
    parts.append(lk.style)
    return " ".join(p.strip() for p in parts if p and p.strip())


def compile_negative_prompt(direction: I2VShotDirection, locks: PromptLocks | None = None) -> str:
    base = (locks or PromptLocks()).global_negative
    extras = ", ".join(direction.extra_negatives)
    return f"{base}, {extras}" if extras else base



class VisualMedium(StrEnum):
    """Visual medium governing character rendering, shaders, optics, and negative locks."""

    LIVE_ACTION_CINEMA = "live_action_cinema"
    CARTOON_2D_CEL = "cartoon_2d_cel"
    CARTOON_3D_STYLIZED = "cartoon_3d_stylized"
    ANIME_2D = "anime_2d"
    STOP_MOTION_CLAY = "stop_motion_clay"


class ProductionScale(StrEnum):
    """Project scale and runtime format."""

    VERTICAL_MICRODRAMA = "vertical_microdrama"    # 1-3 min, 9:16
    SHORT_EPISODE_20MIN = "short_episode_20min"    # 15-25 min, 15-30 scenes, 150-220 shots
    FEATURE_MOVIE = "feature_movie"                # 60-150 min multi-act film
    ANIMATED_SERIES = "animated_series"            # 5-22 min cartoon / anime episode


MEDIUM_DEFAULT_LOCKS: dict[VisualMedium, dict[str, str]] = {
    VisualMedium.LIVE_ACTION_CINEMA: {
        "positive": (
            "Photorealistic live-action 35mm cinema photography, ARRI Alexa Mini LF sensor color science, "
            "natural human skin micro-texture, physically accurate lighting and shallow depth of field."
        ),
        "negative": (
            "cartoon, 2D illustration, anime, 3D CGI plastic skin, Pixar style, cel shading, "
            "video game render, floating UI, text overlays, watermark, face morphing"
        ),
    },
    VisualMedium.CARTOON_2D_CEL: {
        "positive": (
            "High-craft 2D cel-shaded cartoon animation, consistent clean ink linework with uniform line weight, "
            "hand-painted gouache/watercolor background plates, expressive 2D character key-posing on 2s, "
            "flat cel shadow shapes with locked character model-sheet proportions."
        ),
        "negative": (
            "photorealistic live-action human actor, real skin pores, 3D CGI volumetric render, "
            "inconsistent line thickness, off-model facial proportions, camera lens flare, text overlays, watermark"
        ),
    },
    VisualMedium.CARTOON_3D_STYLIZED: {
        "positive": (
            "Feature-grade 3D stylized animated film render, appealing sculpted character proportions, "
            "soft subsurface scattering on stylized skin, tactile fabric weave shaders, cinematic global illumination, "
            "locked 3D character rig geometry across all angles."
        ),
        "negative": (
            "live-action real photography, flat 2D cel outlines, uncanny valley realistic skin pores, "
            "low-poly game engine artifacts, off-rig face stretching, text overlays, watermark"
        ),
    },
    VisualMedium.ANIME_2D: {
        "positive": (
            "Theatrical 2D Japanese anime cinematography, crisp sakuga keyframe linework, multi-plane painted skies "
            "and architectural backgrounds, expressive anime eye highlights, dramatic cel rim-lighting."
        ),
        "negative": (
            "live-action photography, western 3D CGI cartoon, photorealistic skin texture, "
            "inconsistent eye geometry, text subtitles burned in, watermark"
        ),
    },
    VisualMedium.STOP_MOTION_CLAY: {
        "positive": (
            "Tactile handcrafted stop-motion miniature animation, subtle clay/armature surface texture, "
            "miniature studio practical lighting, authentic 24fps stop-motion frame cadence."
        ),
        "negative": (
            "smooth 60fps CGI interpolation, 2D flat cartoon, live-action full-scale human actors, "
            "digital text overlays, watermark"
        ),
    },
}


class GlobalStyleLock(StrictContract):
    """Show-wide visual style bible ensuring every episode and location shares one cohesive look."""

    medium: VisualMedium = VisualMedium.LIVE_ACTION_CINEMA
    scale: ProductionScale = ProductionScale.SHORT_EPISODE_20MIN
    aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE
    width: Annotated[int, Field(ge=256, le=3840)] = 1920
    height: Annotated[int, Field(ge=256, le=3840)] = 1080
    fps: Annotated[float, Field(ge=12.0, le=60.0)] = 24.0
    primary_image_model: str = Field(
        default="gemini-3-pro-image",
        description="Primary keyframe/sheet generator: Nano Banana Pro (supports up to 14 reference images).",
    )
    max_image_references: Annotated[int, Field(ge=1, le=14)] = 14
    primary_video_model: str = Field(
        default="gemini-omni-1.1-flash",
        description="Primary video generator: Gemini Omni 1.1 (supports up to 10 ref images + 3 ref videos, 1080p/4K, extend_video).",
    )
    fallback_video_model: str = Field(
        default="veo-3.1-generate-001",
        description="Fallback start-frame video generator when Omni 1.1 is unavailable.",
    )
    max_video_references: Annotated[int, Field(ge=1, le=10)] = 10
    max_video_clip_references: Annotated[int, Field(ge=0, le=3)] = 3
    color_palette_and_lut_lock: str = Field(
        default="Consistent show-wide color palette and contrast curve across all episodes.",
        description="Global color script, film stock, or cartoon palette lock.",
    )
    lens_or_rendering_signature: str = Field(
        default="Cooke S4 prime lens package with natural optical roll-off.",
        description="Camera optics (live action) or brush/shader engine signature (cartoon/3D).",
    )
    custom_positive_style_tokens: str = ""
    custom_negative_style_tokens: str = ""

    def compile_positive_style_header(self) -> str:
        base = MEDIUM_DEFAULT_LOCKS[self.medium]["positive"]
        extra = f" {self.custom_positive_style_tokens.strip()}" if self.custom_positive_style_tokens.strip() else ""
        return (
            f"GLOBAL STYLE BIBLE LOCK ({self.medium.value.upper()} | {self.aspect_ratio.value}): "
            f"{base} Color & Optics Lock: {self.color_palette_and_lut_lock} "
            f"{self.lens_or_rendering_signature}{extra}"
        )

    def compile_negative_style_footer(self) -> str:
        base = MEDIUM_DEFAULT_LOCKS[self.medium]["negative"]
        extra = f", {self.custom_negative_style_tokens.strip()}" if self.custom_negative_style_tokens.strip() else ""
        return f"{base}{extra}"


class ScalableShotSpec(StrictContract):
    """Generic shot specification supporting any episode/scene/shot numbering (1 to 9,999+)."""

    shot_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{3,64}$")]
    shot_number: Annotated[int, Field(ge=1, le=9999)]
    duration_seconds: Annotated[int, Field(ge=2, le=16)] = 8
    character_ids: list[str] = Field(default_factory=list)
    wardrobe_state_by_character: dict[str, str] = Field(default_factory=dict)
    prop_ids: list[str] = Field(default_factory=list)
    camera_framing_and_motion: str
    action_beat_prompt: str
    end_frame_hold_state: str = Field(
        default="Subject settles into a clean, stable pose for the final 0.5s cut point."
    )
    inherit_previous_shot_in_scene: bool = True
    dialogue_line: SceneDialogueLine | None = None
    ambient_and_foley_prompt: str = "Clean natural room tone and subtle cloth/movement foley."
    music_cue: str = "Subtle underscore ducked beneath any spoken dialogue."


class SceneSpec(StrictContract):
    """A single location/time-of-day scene within an episode (a 20-min episode typically has 15-30 scenes)."""

    scene_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{2,64}$")]
    scene_number: Annotated[int, Field(ge=1, le=999)]
    title: str
    location_id: Annotated[str, Field(pattern=r"^LOC_[A-Z0-9_]+$")]
    time_of_day_override: str | None = None
    scene_lighting_lock: str = Field(
        default="",
        description="Optional scene-specific lighting override (e.g. power outage, sunset).",
    )
    shots: list[ScalableShotSpec] = Field(min_length=1)


class EpisodeSpec(StrictContract):
    """An episode or film reel composed of multiple ordered scenes."""

    episode_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{2,64}$")]
    episode_number: Annotated[int, Field(ge=1, le=999)]
    title: str
    target_runtime_seconds: Annotated[int, Field(ge=15, le=10800)] = 1200  # Default 20 mins (1200s)
    scenes: list[SceneSpec] = Field(min_length=1)


class ProjectBibleSpec(StrictContract):
    """Top-level multi-episode show, movie, or cartoon production bible."""

    project_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{2,64}$")]
    title: str
    style_lock: GlobalStyleLock
    characters: dict[str, CharacterCastSpec]
    voice_profiles: dict[str, CharacterVoiceProfile] = Field(default_factory=dict)
    locations: dict[str, LocationScoutSpec]
    props: dict[str, HeroPropSpec] = Field(default_factory=dict)
    episodes: list[EpisodeSpec] = Field(min_length=1)


class CompiledScalableShot(StrictContract):
    """Fully compiled, continuity-locked shot payload ready for Nano Banana Pro, Gemini Omni 1.1 & DaVinci Resolve."""

    project_id: str
    episode_id: str
    scene_id: str
    shot_id: str
    resolve_bin_path: str
    aspect_ratio: AspectRatio
    width: int
    height: int
    fps: float
    duration_seconds: int
    duration_frames: int
    continuity_seed: int
    image_model: str = "gemini-3-pro-image"
    video_model: str = "gemini-omni-1.1-flash"
    fallback_video_model: str = "veo-3.1-generate-001"
    reference_image_paths: list[str] = Field(
        default_factory=list,
        description="Up to 14 reference image paths for Nano Banana Pro (gemini-3-pro-image).",
    )
    omni_video_reference_paths: list[str] = Field(
        default_factory=list,
        description="Up to 10 reference image paths for Gemini Omni 1.1 (gemini-omni-1.1-flash).",
    )
    fallback_3_reference_paths: list[str] = Field(
        default_factory=list,
        description="Top 3 reference image paths when falling back to 3-reference local tools.",
    )
    positive_prompt: str
    negative_prompt: str
    dialogue_pacing_report: DialoguePacingReport | None = None


def derive_continuity_seed(project_id: str, medium: VisualMedium, character_ids: list[str], location_id: str) -> int:
    """Derive a deterministic 31-bit seed from the project, medium, cast, and location."""
    key = f"{project_id}:{medium.value}:{','.join(sorted(character_ids))}:{location_id}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def preflight_project_continuity(
    project: ProjectBibleSpec,
    episode_id: str | None = None,
) -> dict[str, Any]:
    """Run both the Casting Reference Sheet Gate and Dialogue Pacing Gate across the project/episode."""
    needed_chars: list[str] = []
    needed_locs: list[str] = []
    dialogue_reports: list[dict[str, Any]] = []
    pacing_failures: list[str] = []

    target_episodes = [
        ep for ep in project.episodes if episode_id is None or ep.episode_id == episode_id
    ]

    for ep in target_episodes:
        for sc in ep.scenes:
            if sc.location_id not in needed_locs:
                needed_locs.append(sc.location_id)
            for sh in sc.shots:
                for cid in sh.character_ids:
                    if cid not in needed_chars:
                        needed_chars.append(cid)
                if sh.dialogue_line is not None:
                    vp = project.voice_profiles.get(sh.dialogue_line.intended_speaker_id)
                    rep = validate_dialogue_pacing(sh.dialogue_line, vp)
                    dialogue_reports.append(rep.model_dump())
                    if not rep.is_human_paced:
                        pacing_failures.append(f"{sh.shot_id}: {rep.recommendation}")

    casting_report: CastingPreflightReport = preflight_reference_sheet_check(
        character_ids=needed_chars,
        location_ids=needed_locs,
        custom_characters=project.characters,
        custom_locations=project.locations,
    )

    return {
        "project_id": project.project_id,
        "medium": project.style_lock.medium.value,
        "scale": project.style_lock.scale.value,
        "casting_reference_gate": casting_report.model_dump(),
        "dialogue_pacing_gate": {
            "total_spoken_shots": len(dialogue_reports),
            "all_human_paced": len(pacing_failures) == 0,
            "pacing_failures": pacing_failures,
            "reports": dialogue_reports,
        },
    }


def compile_episode_shots(
    project: ProjectBibleSpec,
    episode_id: str,
    reference_paths_by_id: dict[str, str] | None = None,
    shot_keyframes_by_id: dict[str, str] | None = None,
) -> list[CompiledScalableShot]:
    """Compile all scenes and shots in an episode with strict cross-scene and cross-location locks.

    Collects up to 14 reference images per shot for Nano Banana Pro (`gemini-3-pro-image`) and
    up to 10 reference images per shot for Gemini Omni 1.1 (`gemini-omni-1.1-flash`).
    """
    ep = next((e for e in project.episodes if e.episode_id == episode_id), None)
    if ep is None:
        raise ValueError(f"Episode '{episode_id}' not found in project '{project.project_id}'.")

    ref_map = reference_paths_by_id or {}
    kf_map = shot_keyframes_by_id or {}
    style = project.style_lock
    compiled_list: list[CompiledScalableShot] = []

    for scene in ep.scenes:
        loc = project.locations[scene.location_id]
        # Reset previous-shot conditioning at the start of every new scene/location
        # so lighting and architecture from a previous scene never bleed into the new scene!
        scene_prev_keyframes: list[str] = []

        for shot in scene.shots:
            char_clauses: list[str] = []
            ref_candidates: list[str] = []

            def _add_ref(path_val: str | None) -> None:
                if path_val and path_val not in ref_candidates:
                    ref_candidates.append(path_val)

            # 1. Character reference sheets (turnarounds, expressions, wardrobe variants)
            for cid in shot.character_ids:
                cspec = project.characters[cid]
                w_key = shot.wardrobe_state_by_character.get(cid)
                if w_key and w_key in cspec.wardrobe_by_state:
                    w_desc = cspec.wardrobe_by_state[w_key]
                else:
                    w_desc = next(iter(cspec.wardrobe_by_state.values()), "")
                char_clauses.append(f"[{cspec.name} ({cid})]: {cspec.immutable_physical_prompt} {w_desc}")

                for key in (cid, cspec.anchor_sheet_id):
                    _add_ref(ref_map.get(key))
                for k, v in ref_map.items():
                    if k.startswith(f"{cid}_"):
                        _add_ref(v)

            # 2. Scene location reference plates (wide, reverse angle, lighting key)
            for key in (loc.location_id, loc.anchor_plate_id):
                _add_ref(ref_map.get(key))
            for k, v in ref_map.items():
                if k.startswith(f"{loc.location_id}_"):
                    _add_ref(v)

            # 3. Previous shot keyframes within the same scene (most recent first)
            if shot.inherit_previous_shot_in_scene:
                for prev_kf in reversed(scene_prev_keyframes):
                    _add_ref(prev_kf)

            # 4. Hero prop reference sheets
            for pid in shot.prop_ids:
                _add_ref(ref_map.get(pid))
                if pid in project.props:
                    _add_ref(ref_map.get(project.props[pid].anchor_sheet_id))
                for k, v in ref_map.items():
                    if k.startswith(f"{pid}_"):
                        _add_ref(v)

            # 5. Global Style Bible reference sheet
            for style_key in ("STYLE_BIBLE", project.project_id):
                _add_ref(ref_map.get(style_key))

            nano_banana_refs = ref_candidates[: style.max_image_references]  # Up to 14 refs
            omni_video_refs = ref_candidates[: style.max_video_references]   # Up to 10 refs
            fallback_3_refs = ref_candidates[:3]                             # Top 3 refs

            prop_clauses = [project.props[pid].visual_prompt for pid in shot.prop_ids if pid in project.props]
            lighting = scene.scene_lighting_lock or loc.lighting_and_lens_prompt

            dialogue_report: DialoguePacingReport | None = None
            positive_dlg = "SILENT BEAT: All visible characters keep their mouths closed; no spoken dialogue."
            negative_dlg = "unwanted speech, random lip movement, gibberish talking"

            if shot.dialogue_line is not None:
                vp = project.voice_profiles.get(
                    shot.dialogue_line.intended_speaker_id,
                    CharacterVoiceProfile(
                        character_id=shot.dialogue_line.intended_speaker_id,
                        speaker_name=shot.dialogue_line.intended_speaker_id,
                        vocal_gender_age_tag="adult speaker",
                        pitch_band="natural mid-range",
                        timbre_and_texture="clear conversational resonance",
                        accent_and_dialect="natural delivery",
                    ),
                )
                dialogue_report = validate_dialogue_pacing(shot.dialogue_line, vp)
                iso = compile_speaker_isolation_prompt(shot.dialogue_line, vp)
                positive_dlg = iso["positive_dialogue_prompt"]
                negative_dlg = iso["negative_dialogue_prompt"]

            pos_parts = [
                style.compile_positive_style_header(),
                f"SCENE LOCATION LOCK ({loc.name} - {loc.setting_type} {scene.time_of_day_override or loc.time_of_day}): "
                f"{loc.architectural_prompt} Lighting: {lighting}",
                f"CHARACTER & WARDROBE CONTINUITY LOCK: {' ; '.join(char_clauses) if char_clauses else 'Environment only (no characters in frame)'}.",
            ]
            if prop_clauses:
                pos_parts.append(f"HERO PROPS LOCK: {' ; '.join(prop_clauses)}.")
            pos_parts.extend(
                [
                    f"CAMERA & FRAMING: {shot.camera_framing_and_motion}",
                    f"ACTION BEAT: {shot.action_beat_prompt}",
                    positive_dlg,
                    f"AUDIO / FOLEY BED: {shot.ambient_and_foley_prompt}",
                    f"END STATE: {shot.end_frame_hold_state}",
                ]
            )

            neg_prompt = f"{style.compile_negative_style_footer()}, {negative_dlg}"
            duration_frames = int(round(shot.duration_seconds * style.fps))

            compiled_list.append(
                CompiledScalableShot(
                    project_id=project.project_id,
                    episode_id=ep.episode_id,
                    scene_id=scene.scene_id,
                    shot_id=shot.shot_id,
                    resolve_bin_path=f"Video/{ep.episode_id}/{scene.scene_id}",
                    aspect_ratio=style.aspect_ratio,
                    width=style.width,
                    height=style.height,
                    fps=style.fps,
                    duration_seconds=shot.duration_seconds,
                    duration_frames=duration_frames,
                    continuity_seed=derive_continuity_seed(
                        project.project_id, style.medium, shot.character_ids, scene.location_id
                    ),
                    image_model=style.primary_image_model,
                    video_model=style.primary_video_model,
                    fallback_video_model=style.fallback_video_model,
                    reference_image_paths=nano_banana_refs,
                    omni_video_reference_paths=omni_video_refs,
                    fallback_3_reference_paths=fallback_3_refs,
                    positive_prompt=" ".join(pos_parts),
                    negative_prompt=neg_prompt,
                    dialogue_pacing_report=dialogue_report,
                )
            )

            if shot.shot_id in kf_map:
                scene_prev_keyframes.append(kf_map[shot.shot_id])

    return compiled_list
