"""Tier-0 AGY Generic Filmmaking Bible, Multi-Shot Commercial Prompting & Canonical Folder Structure.

Implements the top-level ``GENERIC BIBLE`` domain intelligence from the AGY Studio v4 architecture:
1. **Operating / Control Modes** (:class:`ControlMode`):
   - ``director``: Ultimate human control at every gate (G1 -> G6).
   - ``hybrid``: Stops at Vision/Storyboard/Video/Cut gates; auto-passes G2/G4 when no new states or voices exist.
   - ``autopilot``: Seamless execution from Gate 1 approval straight to Gate 6 DaVinci Resolve cut using QA self-healing.
2. **Content Formats** (:class:`ContentFormat`):
   - ``microdrama_9_16`` (Vertical 9:16 series)
   - ``episode_16_9`` (20-min / episodic series)
   - ``feature_film`` (Long-form narrative cinema)
   - ``marketing_spot`` (Multi-shot commercial / brand film using time-coded beat escalation)
3. **7-Layer Visual + 4-Layer Audio + Multi-Shot Time-Coded Prompt Compiler** (incorporating best practices from
   ``replicate-prompt-videos``):
   - 7 Visual Layers: Subject, Context, Action, Style, Camera, Composition, Ambiance.
   - 4 Audio Layers: Dialogue (colon syntax + ``(no subtitles)``), Ambient sound (explicit to prevent studio-audience
     hallucinations), Sound effects, Music.
   - Multi-Shot Time-Coded Escalation (``(0-3s) Macro/Wide -> (3-7s) Glide/Medium -> (7-12s) Hard cut to Slow-Mo
     Detail -> (12-15s) Seamless pull-out to Hero Frame``).
4. **Locked Canonical Folder Structure**:
   - Explicitly separates **Shareable Solution Files** (code, `.agents/`, skills, rules, MCP server, setup script)
     from **Per-Production Data** (`productions/<PROJECT_ID>/`) so teams know exactly what to share.
"""

from __future__ import annotations

import shutil
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field

from .models import Model


class ControlMode(StrEnum):
    """How much human gate intervention vs. autonomous execution the team wants."""

    DIRECTOR = "director"      # Ultimate control: stops at G1, G2, G3, G4, G5, G6
    HYBRID = "hybrid"          # Balanced: stops at G1, G3, G5, G6; auto-passes G2/G4 when reusing Global Bible
    AUTOPILOT = "autopilot"    # Seamless: stops at G1 (Global Bible) and G6 (Resolve Cut); G2-G5 run autonomously


class ContentFormat(StrEnum):
    """Target production format preset."""

    MICRODRAMA_9_16 = "microdrama_9_16"
    EPISODE_16_9 = "episode_16_9"
    FEATURE_FILM = "feature_film"
    MARKETING_SPOT = "marketing_spot"
    COMMERCIAL_AD = "commercial_ad"


class TimeCodedBeat(Model):
    """One time-coded beat inside a multi-shot commercial or dynamic microdrama clip."""

    start_s: float = 0.0
    end_s: float = 3.0
    shot_type: str = "medium shot"
    camera_motion: str = "static camera"
    transition: str | None = None  # e.g. "Hard cut to", "Seamless pull-out to", "Whip pan to"
    subject_action: str = ""
    lighting_ambiance: str = ""
    audio_cue: str = ""

    def format_beat(self) -> str:
        span = f"({self.start_s:g}-{self.end_s:g}s)"
        prefix = f"{self.transition} " if self.transition else ""
        parts = [
            f"{prefix}{self.shot_type}".strip(),
            self.camera_motion.strip(),
            self.subject_action.strip(),
            self.lighting_ambiance.strip(),
            self.audio_cue.strip(),
        ]
        body = ", ".join(p for p in parts if p)
        return f"{span} {body}."


class SevenLayerVisualSpec(Model):
    """The 7 mandatory visual layers from the Generic Bible / prompt-videos skill."""

    subject: str
    context: str
    action: str
    style: str
    camera: str
    composition: str
    ambiance: str
    time_coded_beats: list[TimeCodedBeat] = Field(default_factory=list)
    reference_tags: list[str] = Field(default_factory=list)  # e.g. ["[Image1] character sheet", "[Image2] location"]


class FourLayerAudioSpec(Model):
    """The 4 mandatory audio layers (prevents hallucinated background laughter or baked-in subtitles)."""

    dialogue: str = "No spoken dialogue; all visible characters keep mouths 100% CLOSED."
    ambient_sound: str = "Natural room tone and authentic environmental ambiance."
    sound_effects: str = "Subtle realistic foley matching subject movement."
    music: str = "No background music."
    no_subtitles: bool = True


# ---------------------------------------------------------------------------------------------
# Virtual Cinema Optical Glass (6 Camera Bodies x 11 Lenses x 3 Apertures) & 33 Camera Rigs
# ---------------------------------------------------------------------------------------------
class CineCameraBody(StrEnum):
    """The 6 Virtual Cinema Camera Bodies."""

    ARRI_ALEXA_35 = "ARRI Alexa 35 (Super 35, 17-stop dynamic range, organic filmic highlight rolloff)"
    RED_V_RAPTOR = "RED V-Raptor 8K VV (VistaVision large format, razor-sharp modern contrast)"
    SONY_VENICE_2 = "Sony Venice 2 8K (Full-Frame dual base ISO, clean shadow detail & natural skin tones)"
    PANAVISION_DXL2 = "Panavision Millennium DXL2 (Large format, warm classical Hollywood color science)"
    IMAX_MSM_9802 = "IMAX MSM 9802 65mm 15-perf (Ultra-wide visceral immersion & zero distortion)"
    ARRIFLEX_435_35MM = "Arriflex 435 35mm Film Camera (Kodak Vision3 500T 5219 halation & organic photochemical grain)"


class CineLens(StrEnum):
    """The 11 Cinema Optical Glass Lenses."""

    ARRI_SIGNATURE_PRIME = "ARRI Signature Prime (ultra-clean modern glass, creamy bokeh, zero chromatic aberration)"
    ZEISS_SUPREME_PRIME = "Zeiss Supreme Prime (crisp micro-contrast, gentle focus falloff)"
    COOKE_S4_PRIME = "Cooke S4/i Prime (warm 'Cooke Look' skin rendition, velvety shadow rolloff)"
    PANAVISION_PRIMO = "Panavision Primo Prime (classic cinematic warmth, subtle golden edge flare)"
    HAWK_V_LITE_ANAMORPHIC = "Hawk V-Lite 2x Anamorphic (oval bokeh, horizontal blue streak flares, widescreen compression)"
    ATLAS_ORION_ANAMORPHIC = "Atlas Orion 2x Anamorphic (modern vintage anamorphic character, signature cyan flares)"
    CANON_K35_VINTAGE = "Canon K35 Vintage Prime (1970s low-contrast glowing highlights, warm pastel flares)"
    HELIOS_44_2 = "Helios 44-2 58mm Rehoused (swirly expressive background bokeh, dreamy subject isolation)"
    PETZVAL_85_ART = "Lomography Petzval 85mm Art (tack-sharp center with dramatic radial field curvature)"
    LAOWA_24MM_PROBE = "Laowa 24mm f/14 2x Macro Probe (bug-eye extreme close-up perspective through tight spaces)"
    ANGENIEUX_OPTIMO_ZOOM = "Angenieux Optimo 24-290mm Cinema Zoom (rich theatrical compression & fluid focal transitions)"


class CineAperture(StrEnum):
    """The 3 Canonical Cinema T-Stop / f-Stop Depth-of-Field Tiers."""

    SHALLOW_F1_4 = "f/1.4 (wide-open ultra-shallow depth of field, intimate subject isolation, melted background bokeh)"
    BALANCED_F4 = "f/4.0 (natural narrative depth of field, crisp subject with legible spatial environment)"
    DEEP_F11 = "f/11 (deep focus staging, foreground and background subjects simultaneously tack-sharp)"


class CameraRig33(StrEnum):
    """The 33 Physical Camera Rigs & Movement Primitives."""

    STATIC_TRIPOD = "static locked-off fluid-head tripod"
    SLOW_DOLLY_IN = "slow motorized dolly push-in on precision track"
    SLOW_DOLLY_OUT = "slow motorized dolly pull-out revealing environment"
    TRUCK_LEFT = "lateral dolly truck left parallel to subject"
    TRUCK_RIGHT = "lateral dolly truck right parallel to subject"
    PEDESTAL_UP = "vertical column pedestal rise up"
    PEDESTAL_DOWN = "vertical column pedestal sink down"
    PAN_LEFT = "smooth fluid-head pan left"
    PAN_RIGHT = "smooth fluid-head pan right"
    TILT_UP = "smooth fluid-head tilt up reveal"
    TILT_DOWN = "smooth fluid-head tilt down reveal"
    WHIP_PAN = "high-velocity kinetic whip pan with motion blur"
    WHIP_TILT = "high-velocity vertical whip tilt"
    STEADICAM_FOLLOW = "floating Steadicam following behind subject at walking pace"
    STEADICAM_LEAD = "floating Steadicam leading backwards in front of walking subject"
    STEADICAM_ORBIT_180 = "smooth 180-degree Steadicam arc orbit around subject"
    ORBIT_360_HERO = "continuous 360-degree circular dolly orbit around hero subject"
    HANDHELD_INTIMATE = "organic shoulder-mounted documentary handheld with subtle breathing micro-motion"
    HANDHELD_CHAOTIC = "urgent kinetic handheld combat/chase camera with visceral shake"
    SNORRICAM_BODY_MOUNT = "body-mounted SnorriCam locked to actor's chest (subject static, background spinning)"
    CRANE_JIB_SWEEP_UP = "sweeping Technocrane jib rise from ground level to high-angle overview"
    CRANE_JIB_SWEEP_DOWN = "sweeping Technocrane jib descent from high ceiling down to intimate close-up"
    DUTCH_ANGLE_TENSION = "canted 18-degree Dutch tilt rig for psychological unease"
    LOW_ANGLE_HERO = "floor-level low-angle hi-hat rig looking up at subject"
    HIGH_ANGLE_VULNERABLE = "overhead high-angle menace boom looking down at subject"
    BIRDS_EYE_TOP_DOWN = "90-degree vertical overhead God's-eye rig straight down"
    WORMS_EYE_GROUND = "recessed worm's-eye ground plate looking straight up"
    DOLLY_ZOOM_VERTIGO = "synchronized zolly (dolly-in while zooming-out) Vertigo perspective warp"
    RACK_FOCUS_FG_TO_BG = "precision follow-focus rack from foreground silhouette to background speaker"
    RACK_FOCUS_BG_TO_FG = "precision follow-focus rack from background action to foreground reaction"
    PROBE_MACRO_GLIDE = "motorized motion-control macro probe gliding through tabletop objects"
    FPV_DRONE_DIVE = "acrobatic cinewhoop FPV drone seamless interior-to-exterior thread"
    CABLE_CAM_FLYTHROUGH = "suspended wire-cam high-speed overhead tracking shot"


class CineOpticalSpec(Model):
    """Structured optical glass & rig package injected into Keyframe and Video prompts."""

    camera_body: CineCameraBody = CineCameraBody.ARRI_ALEXA_35
    lens: CineLens = CineLens.COOKE_S4_PRIME
    focal_length_mm: int = 35
    aperture: CineAperture = CineAperture.BALANCED_F4
    rig: CameraRig33 = CameraRig33.STATIC_TRIPOD

    def format_optical_line(self) -> str:
        return (
            f"Shot on {self.camera_body.value} paired with {self.focal_length_mm}mm {self.lens.value} "
            f"at {self.aperture.value}, mounted on a {self.rig.value}."
        )



# ---------------------------------------------------------------------------------------------
# Multi-Shot Commercial & Escalation Templates (from replicate-prompt-videos)
# ---------------------------------------------------------------------------------------------
TRANSITION_VOCABULARY: tuple[str, ...] = (
    "Hard cut to",
    "Seamless morph into",
    "Whip pan to",
    "Snap cut to",
    "Seamless pull-out to",
)

COMMERCIAL_ESCALATION_TEMPLATE: list[TimeCodedBeat] = [
    TimeCodedBeat(
        start_s=0.0,
        end_s=3.0,
        shot_type="Macro / establishing shot",
        camera_motion="shallow depth of field, gentle drift",
        transition=None,
        subject_action="hero subject or product introduced in tactile environment",
        lighting_ambiance="warm directional key light with floating atmospheric detail",
        audio_cue="soft ambient room tone and subtle opening musical texture",
    ),
    TimeCodedBeat(
        start_s=3.0,
        end_s=7.0,
        shot_type="Medium-close interaction shot",
        camera_motion="camera glides closer",
        transition=None,
        subject_action="character interacts physically with the hero object",
        lighting_ambiance="natural skin and surface reflections",
        audio_cue="crisp tactile foley of fabric and object contact",
    ),
    TimeCodedBeat(
        start_s=7.0,
        end_s=12.0,
        shot_type="Extreme close-up / slow-motion detail",
        camera_motion="orbiting macro lens",
        transition="Hard cut to",
        subject_action="peak kinetic or emotional micro-action in slow motion",
        lighting_ambiance="high-contrast rim light against deep background",
        audio_cue="signature crisp sound effect synchronized to the action",
    ),
    TimeCodedBeat(
        start_s=12.0,
        end_s=15.0,
        shot_type="Hero frame wide-medium shot",
        camera_motion="controlled dolly pull-out",
        transition="Seamless pull-out to",
        subject_action="subject or product centered in clean iconic silhouette",
        lighting_ambiance="volumetric lighting and clean brand/cinematic palette",
        audio_cue="resonant finish and elegant silence",
    ),
]


def compile_generic_bible_prompt(
    visual: SevenLayerVisualSpec | None = None,
    audio: FourLayerAudioSpec | None = None,
    *,
    optics: CineOpticalSpec | None = None,
    single_speaker_id: str | None = None,
    off_screen: bool = False,
    scene_and_theme: str = "",
    subject_and_character: str = "",
    action_and_physics: str = "",
    camera_and_framing: str = "",
    lighting_and_color: str = "",
    audio_and_dialogue: str = "",
    style_and_negatives: str = "",
    beats: list[TimeCodedBeat] | None = None,
    compact_4_layer: bool = False,
) -> str:
    """Compile a deterministic, hallucination-resistant video prompt adhering to the Generic Bible."""
    if visual is None:
        visual = SevenLayerVisualSpec(
            subject=subject_and_character or scene_and_theme,
            context=scene_and_theme,
            action=action_and_physics,
            style=style_and_negatives or "photorealistic cinematic",
            camera=camera_and_framing,
            composition=camera_and_framing,
            ambiance=lighting_and_color,
            time_coded_beats=list(beats or []),
        )
    if audio is None:
        audio = FourLayerAudioSpec(
            dialogue=audio_and_dialogue or "none",
            ambient_sound=audio_and_dialogue or "room tone",
            sound_effects=audio_and_dialogue or "subtle foley",
            music="cinematic underscore",
        )
    if compact_4_layer:
        lines = [
            f"Subject: {visual.subject}",
            f"Context: {visual.context}",
            f"Action: {visual.action}",
            f"Camera: {visual.camera}",
        ]
    else:
        lines = [
            f"Subject: {visual.subject}",
            f"Context: {visual.context}",
            f"Action: {visual.action}",
            f"Style: {visual.style}",
            f"Camera: {visual.camera}",
            f"Composition: {visual.composition}",
            f"Ambiance: {visual.ambiance}",
        ]
    if optics is not None:
        lines.append(f"Optical Glass & Camera Rig: {optics.format_optical_line()}")
    if visual.reference_tags:
        lines.append("Reference Mapping: " + "; ".join(visual.reference_tags))
    if visual.time_coded_beats:
        lines.append("Multi-Shot Time-Coded Direction:")
        for beat in visual.time_coded_beats:
            lines.append(f"  {beat.format_beat()}")

    # Single-speaker lip-sync & anti-hallucination guardrails
    if off_screen or not single_speaker_id:
        lip_rule = "LIP-SYNC RULE: Off-screen audio / non-dialogue beat — ALL visible characters keep mouths 100% CLOSED."
    else:
        lip_rule = (
            f"LIP-SYNC RULE: ONLY {single_speaker_id} speaks and lip-syncs; "
            "all other visible characters keep mouths 100% CLOSED."
        )
    lines.append(lip_rule)

    # 4-layer audio specification
    lines.append(f"Audio Layer 1 (Dialogue): {audio.dialogue}")
    lines.append(f"Audio Layer 2 (Ambient Sound): {audio.ambient_sound}")
    lines.append(f"Audio Layer 3 (Sound Effects): {audio.sound_effects}")
    lines.append(f"Audio Layer 4 (Music): {audio.music}")
    if audio.no_subtitles:
        lines.append("(no subtitles) No subtitles. No on-screen text overlays or watermarks.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Locked Canonical Folder Structure (Shareable Solution vs. Per-Production Data)
# ---------------------------------------------------------------------------------------------
SHAREABLE_SOLUTION_PATHS: tuple[str, ...] = (
    ".gitignore",
    "GEMINI.md",
    "GENERIC_BIBLE.md",
    "STARTER_PROMPT.md",
    "README.md",
    "FOLDER_STRUCTURE.md",
    "EDITOR_HANDOFF_GUIDE.md",
    "LIVE_DEMO_RUNBOOK.md",
    "live_demo_preflight.py",
    "pyproject.toml",
    "requirements.txt",
    "setup_studio_env.sh",
    "mcp_config.json",
    "resolve_mcp_server.py",
    "resolve_sync_trigger.py",
    "vertex_veo_director.py",
    "casting_agent.py",
    "dialogue_agent.py",
    "pro_audio_director.py",
    "production_bible.py",
    "production_ledger.py",
    "schemas.py",
    "new_project_template.py",
    ".agents",
    "agy_studio",
    "tests",
)

PRODUCTION_SUBDIRS: tuple[str, ...] = (
    "bible",
    "bible/entities",
    "bible/environments",
    "bible/sets",
    "library",
    "library/characters",
    "library/props",
    "library/environments",
    "library/packs",
    "library/voices",
    "scenes",
    "script",
    "post",
    "reviews",
    "logs",
)


def validate_workspace_layout(workspace_root: Path) -> dict[str, Any]:
    """Verify that the workspace adheres to the locked shareable + production folder structure."""
    root = Path(workspace_root).resolve()
    present: list[str] = []
    missing: list[str] = []
    for rel in SHAREABLE_SOLUTION_PATHS:
        if (root / rel).exists():
            present.append(rel)
        else:
            missing.append(rel)
    return {
        "workspace_root": str(root),
        "valid": len(missing) == 0,
        "present_shareable_paths": present,
        "missing_shareable_paths": missing,
    }


def export_shareable_bundle(workspace_root: Path, dest_dir: Path, *, include_sample_project: str | None = None) -> Path:
    """Create a clean, portable copy of the AGY Studio solution ready to share with another team."""
    root = Path(workspace_root).resolve()
    dest = Path(dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".venv", ".DS_Store", "*.lock")

    for rel in SHAREABLE_SOLUTION_PATHS:
        src = root / rel
        if not src.exists():
            continue
        target = dest / rel
        if src.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target, ignore=ignore)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)

    prod_root = dest / "productions"
    prod_root.mkdir(parents=True, exist_ok=True)
    (prod_root / ".gitkeep").write_text("", encoding="utf-8")
    if include_sample_project:
        sample_src = root / "productions" / include_sample_project
        if sample_src.is_dir():
            sample_dst = prod_root / include_sample_project
            if sample_dst.exists():
                shutil.rmtree(sample_dst)
            shutil.copytree(sample_src, sample_dst, ignore=ignore)
    return dest
