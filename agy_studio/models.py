"""AGY Film Studio data model — BUILD_SPEC_v3 §4 (plus §3 project config and §12 orchestration state).

Every record is a strict Pydantic v2 model (``extra="forbid"``) so typos in hand-edited YAML fail loudly.
Entity / state / environment / voice IDs are UPPERCASE_ALNUM (``ID_PATTERN``) because they are the ONLY way a
prompt may reference an entity (linter rule L3) and they are also used as folder names inside the project
(see ``fsio.validate_id`` — IDs are validated before they ever touch the filesystem).
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ID_PATTERN = r"^[A-Z][A-Z0-9_]{0,95}$"
_ID_RE = re.compile(ID_PATTERN)
PREFERENCE_SCOPE_PATTERN = r"^(project|(character|scene|agent|user):[A-Za-z0-9_.\-]{1,96})$"
_SCOPE_RE = re.compile(PREFERENCE_SCOPE_PATTERN)

VISUAL_FIELDS: tuple[str, ...] = (
    "subject_and_action",
    "camera",
    "lens_and_aspect",
    "lighting",
    "palette",
    "composition",
    "atmosphere",
    "mood",
    "attribution",
)


def _check_id(value: str, what: str = "id") -> str:
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise ValueError(f"{what} {value!r} must match {ID_PATTERN} (UPPERCASE_ALNUM, underscores)")
    return value


class Model(BaseModel):
    """Base for every on-disk record."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, populate_by_name=True)


# ---------------------------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------------------------
class ApprovalStatus(StrEnum):
    PROPOSED = "proposed"
    LOCKED = "locked"
    STALE = "stale"


class EntityType(StrEnum):
    CHARACTER = "character"
    PROP = "prop"
    VEHICLE = "vehicle"
    CREATURE = "creature"


class Tier(StrEnum):
    PRINCIPAL = "principal"
    SUPPORTING = "supporting"
    INCIDENTAL = "incidental"


class SceneStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    HANDED_OFF = "handed_off"
    HUMAN_APPROVED = "human_approved"
    DONE = "done"
    NEEDS_REVIEW = "needs_review"  # §4.7: previous scene exit_state changed


class Ownership(StrEnum):
    AGY = "agy"
    HUMAN = "human"


class RiskTier(StrEnum):
    LOW = "low"
    MED = "med"
    HIGH = "high"


class AudioMode(StrEnum):
    NATIVE_AV = "native_av"
    VIDEO_ONLY = "video_only"


class ClipStatus(StrEnum):
    PENDING = "pending"
    PLANNED = "planned"
    LINTED = "linted"
    GENERATING = "generating"
    GENERATED = "generated"
    RANKED = "ranked"
    SELECTED = "selected"
    STALE = "stale"
    NEEDS_VOICE_REVIEW = "needs_voice_review"  # §4.7 voice re-anchor: QA re-scored, never regenerated
    NEEDS_RELINT = "needs_relint"  # §4.7 ShotSpec changed


class TakeStatus(StrEnum):
    CANDIDATE = "candidate"
    SELECTED = "selected"
    REJECTED = "rejected"


class DispatchStatus(StrEnum):
    PLANNED = "planned"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    FILTERED = "filtered"


class Recommendation(StrEnum):
    ACCEPT = "accept"
    REVIEW = "review"
    REJECT = "reject"


class DecisionTier(StrEnum):
    VISION = "vision"
    CRAFT = "craft"
    POLISH = "polish"


class DecisionSource(StrEnum):
    CHAT = "chat"
    RESOLVE = "resolve"


class DecisionAction(StrEnum):
    CHOOSE = "choose"
    ANNOTATE = "annotate"
    STEER = "steer"
    OVERRIDE = "override"
    LOCK = "lock"
    REQUEST_ALTERNATIVES = "request_alternatives"
    REJECT = "reject"
    ANSWER_QUESTION = "answer_question"
    APPROVE = "approve"


class GateId(StrEnum):
    G1 = "G1"  # Bible + state timeline + StyleLock + voice profiles + schedule (Vision)
    G2 = "G2"  # new packs & geometry -> library (Vision; auto-pass if nothing new)
    G3 = "G3"  # boards (Craft)
    G4 = "G4"  # audio_spec per clip + cue plan (Craft)
    G5 = "G5"  # takes (Craft; auto-select on low risk) + voice-anchor prompt
    G6 = "G6"  # human's edited SC_vN approved (Resolve APPROVED marker / chat approve)
    G7 = "G7"  # picture lock (Vision)


class SceneStage(StrEnum):
    """Per-scene state machine stages (§11 scene loop). ``awaiting_*`` stages are human gates."""

    PENDING = "pending"
    WORKING_SET = "working_set"
    DESIGN = "design"
    AWAITING_G2 = "awaiting_G2"
    DIRECTION = "direction"
    AWAITING_G3 = "awaiting_G3"
    DECOMPOSITION = "decomposition"
    AUDIO_PLAN = "audio_plan"
    AWAITING_G4 = "awaiting_G4"
    LINT = "lint"
    GENERATING = "generating"
    QA = "qa"
    AWAITING_G5 = "awaiting_G5"
    ASSEMBLY = "assembly"
    HANDED_OFF = "handed_off"  # == awaiting G6 (human edits in Resolve)
    DONE = "done"


class GlobalStage(StrEnum):
    PREPRODUCTION = "preproduction"
    AWAITING_G1 = "awaiting_G1"
    SCENE_LOOP = "scene_loop"
    POST = "post"
    AWAITING_G7 = "awaiting_G7"
    DELIVERY = "delivery"
    DELIVERED = "delivered"


class ModelKind(StrEnum):
    VIDEO_AV = "video_av"
    MUSIC = "music"
    SFX = "sfx"
    TTS = "tts"
    IMAGE = "image"
    IMAGE_EDIT = "image_edit"
    VIDEO_EDIT = "video_edit"
    TRANSCRIBE = "transcribe"
    SPEAKER_SIMILARITY = "speaker_similarity"
    LLM = "llm"
    VISION_JUDGE = "vision_judge"


class MarkerColor(StrEnum):
    """§10.3 marker semantics."""

    RED = "Red"  # QA reject / needs regen
    YELLOW = "Yellow"  # review / high stitch risk / state transition / voice-drift warning
    BLUE = "Blue"  # gate decision required
    GREEN = "Green"  # locked
    PURPLE = "Purple"  # continuity note
    ORANGE = "Orange"  # AGY_REGEN_READY


# ---------------------------------------------------------------------------------------------
# Provenance (§3: every file carries a generated_by block)
# ---------------------------------------------------------------------------------------------
class GeneratedBy(Model):
    agent: str
    model: str | None = None
    model_version: str | None = None
    seed: int | None = None
    working_set_hash: str | None = None
    ts: str
    tool_version: str = "3.0.0"
    notes: str | None = None


# ---------------------------------------------------------------------------------------------
# §4.1 Style Lock
# ---------------------------------------------------------------------------------------------
class StyleLock(Model):
    medium: str
    aspect_ratio: str
    lens_language: str
    lut_or_grade_ref: str | None = None
    reference_films: list[str] = Field(default_factory=list)
    donts: list[str] = Field(default_factory=list)
    global_negative: list[str] = Field(default_factory=list)  # visual + audio negatives
    approval_status: ApprovalStatus = ApprovalStatus.PROPOSED
    version: int = 1
    derived_from: str | None = None
    generated_by: GeneratedBy | None = None


# ---------------------------------------------------------------------------------------------
# §4.2 Entities / Entity-States / Reference Packs
# ---------------------------------------------------------------------------------------------
class Turnaround(Model):
    front: str | None = None
    three_quarter: str | None = None
    profile: str | None = None
    rear: str | None = None


class ReferencePack(Model):
    """Rendered in NEUTRAL studio lighting. URIs are project-relative POSIX paths into ``library/``.

    Tier requirements (linter L5): principal = full turnaround + face_closeup; supporting = front,
    three_quarter, face_closeup; incidental = single_ref (or text_only=True when it has no line and
    <= 2 s screen time — Open Decision 2).
    """

    turnaround: Turnaround = Field(default_factory=Turnaround)
    face_closeup: str | None = None
    expression_sheet: str | None = None
    identity_embedding: str | None = None
    single_ref: str | None = None
    text_only: bool = False
    generated_by: GeneratedBy | None = None
    approved_by: str | None = None
    approved_at: str | None = None

    def image_uris(self) -> list[str]:
        """All conditioning image URIs in a stable, most-useful-first order."""
        t = self.turnaround
        ordered = [self.face_closeup, t.front, t.three_quarter, t.profile, t.rear, self.expression_sheet, self.single_ref]
        out: list[str] = []
        for uri in ordered:
            if uri and uri not in out:
                out.append(uri)
        return out


class EntityState(Model):
    id: str
    derived_from: str | None = None
    delta_description: str = ""
    state_specific_negative: list[str] = Field(default_factory=list)
    reference_pack: ReferencePack = Field(default_factory=ReferencePack)
    scenes_active: list[str] = Field(default_factory=list)
    approval_status: ApprovalStatus = ApprovalStatus.PROPOSED
    version: int = 1

    @field_validator("id")
    @classmethod
    def _v_id(cls, v: str) -> str:
        return _check_id(v, "EntityState.id")


class Entity(Model):
    id: str
    type: EntityType
    tier: Tier
    canonical_description: str
    display_name: str | None = None
    voice_profile_id: str | None = None
    states: list[EntityState] = Field(default_factory=list)
    approval_status: ApprovalStatus = ApprovalStatus.PROPOSED
    version: int = 1
    derived_from: str | None = None
    generated_by: GeneratedBy | None = None

    @field_validator("id")
    @classmethod
    def _v_id(cls, v: str) -> str:
        return _check_id(v, "Entity.id")

    @model_validator(mode="after")
    def _v_states(self) -> "Entity":
        for st in self.states:
            if not st.id.startswith(self.id + "_"):
                raise ValueError(f"EntityState {st.id} must be prefixed with its entity id {self.id}_")
        return self

    def state(self, state_id: str) -> EntityState | None:
        return next((s for s in self.states if s.id == state_id), None)


# ---------------------------------------------------------------------------------------------
# §4.3 Voice Profile
# ---------------------------------------------------------------------------------------------
class VoiceProfile(Model):
    id: str  # e.g. VOICE_HERO_01
    character_id: str
    description: str  # deterministic, physical voice description
    language: str | None = None  # e.g. "Hindi — colloquial Mumbai, Devanagari script"
    state_variants: dict[str, str] = Field(default_factory=dict)  # EntityState.id -> delta
    reference_audio: str | None = None  # only if registry[model].supports.voice_reference_audio
    anchor_take: str | None = None  # first human-approved speaking take; set once, immutable
    anchor_wav: str | None = None  # dialogue extracted from anchor_take
    approval_status: ApprovalStatus = ApprovalStatus.PROPOSED
    version: int = 1
    derived_from: str | None = None
    generated_by: GeneratedBy | None = None

    @field_validator("id", "character_id")
    @classmethod
    def _v_ids(cls, v: str) -> str:
        return _check_id(v, "VoiceProfile id")

    def voice_for_state(self, state_id: str | None) -> str:
        variant = self.state_variants.get(state_id or "", "")
        return f"{self.description}; {variant}" if variant else self.description


# ---------------------------------------------------------------------------------------------
# §4.4 Environments
# ---------------------------------------------------------------------------------------------
class LightingSetup(Model):
    id: str
    description: str  # physical light sources only
    key_frame: str | None = None


class DerivedAngles(Model):
    reverse: str | None = None
    angle_45L: str | None = None
    angle_45R: str | None = None
    wide_establishing: str | None = None
    inserts: dict[str, str] = Field(default_factory=dict)
    cameras: dict[str, str] = Field(default_factory=dict)  # set3d camera name -> styled reference still (library/)

    def get(self, name: str) -> str | None:
        if name in ("reverse", "angle_45L", "angle_45R", "wide_establishing"):
            return getattr(self, name)
        if name.startswith("insert:"):
            return self.inserts.get(name.split(":", 1)[1])
        if name.startswith("cam:"):
            return self.cameras.get(name.split(":", 1)[1])
        return None


class Environment(Model):
    id: str
    canonical_description: str
    geometry_candidates: list[str] = Field(default_factory=list, max_length=4)
    geometry_still: str | None = None  # locked at G2
    set3d_uri: str | None = None  # bible/sets/<ENV>/set.json — the grey-box 3D set (set3d)
    axis_of_action: str = ""
    derived_angles: DerivedAngles = Field(default_factory=DerivedAngles)
    lighting_setups: list[LightingSetup] = Field(default_factory=list)
    room_tone: str = ""  # native-audio ambience descriptor
    approval_status: ApprovalStatus = ApprovalStatus.PROPOSED
    version: int = 1
    derived_from: str | None = None
    generated_by: GeneratedBy | None = None

    @field_validator("id")
    @classmethod
    def _v_id(cls, v: str) -> str:
        return _check_id(v, "Environment.id")

    def angle_uri(self, angle: str | None) -> str | None:
        if not angle or angle == "geometry":
            return self.geometry_still
        return self.derived_angles.get(angle)


# ---------------------------------------------------------------------------------------------
# §4.5 Scene / Breakdown / Working Set
# ---------------------------------------------------------------------------------------------
class SceneDialogueLine(Model):
    line_id: str  # e.g. SC01_L001
    character_id: str
    text: str  # verbatim from script (for Hindi: Devanagari)
    parenthetical: str | None = None
    transliteration: str | None = None  # phonetic romanisation (prevents accent drift on Hindi lines)
    language: str | None = None
    off_screen: bool = False  # speaker not in frame (phone / V.O. / O.S.)

    @field_validator("line_id", "character_id")
    @classmethod
    def _v_ids(cls, v: str) -> str:
        return _check_id(v, "dialogue id")


class Beat(Model):
    beat_id: str
    summary: str
    performance_states: dict[str, str] = Field(default_factory=dict)  # character_id -> performance state


class MidSceneTransition(Model):
    """An entity changes state from a given shot/clip onward (linter L4)."""

    entity_id: str
    from_state: str
    to_state: str
    at_shot_id: str
    at_clip_index: int = 1
    note: str = ""


class SceneBreakdown(Model):
    estimated_duration_s: float = 0.0
    n_entity_states: int = 0
    n_simultaneous_characters: int = 0
    physical_interaction: bool = False
    dialogue_density: float = 0.0  # words per second of screen time
    action_density: float = 0.0  # physical actions per 10 s
    n_state_transitions: int = 0
    has_crowd: bool = False
    camera_complexity: RiskTier = RiskTier.LOW
    new_entities_introduced: list[str] = Field(default_factory=list)
    new_speaking_characters: list[str] = Field(default_factory=list)
    estimated_clips: int = 0
    estimated_cost: float = 0.0
    risk_tier: RiskTier = RiskTier.MED
    recommended_model: str = "omni"  # registry id: omni | veo
    audio_mode: AudioMode = AudioMode.NATIVE_AV
    rationale: str = ""


class SceneResolve(Model):
    timeline_name: str | None = None
    bin_path: str | None = None
    handoff_fingerprint: str | None = None
    ownership: Ownership = Ownership.AGY
    markers_enabled: bool = True  # Open Decision 7: Orange AGY_REGEN_READY on human-owned timelines


class Scene(Model):
    id: str  # e.g. SC01
    slug: str
    order: int = 0
    episode: int | None = None
    script_text: str = ""
    page_range: str | None = None
    beats: list[Beat] = Field(default_factory=list)
    performance_states: dict[str, str] = Field(default_factory=dict)
    entity_states_active: list[str] = Field(default_factory=list)
    environment_id: str | None = None
    lighting_setup_id: str | None = None
    time_of_day: str | None = None
    mid_scene_transitions: list[MidSceneTransition] = Field(default_factory=list)
    dialogue: list[SceneDialogueLine] = Field(default_factory=list)
    breakdown: SceneBreakdown | None = None
    entry_state: dict[str, Any] = Field(default_factory=dict)
    exit_state: dict[str, Any] = Field(default_factory=dict)
    creative_shots: list[str] = Field(default_factory=list)  # ordered CreativeShot ids
    status: SceneStatus = SceneStatus.PENDING
    resolve: SceneResolve = Field(default_factory=SceneResolve)
    model_override: str | None = None  # chat: `model SC12 omni|veo`
    version: int = 1
    generated_by: GeneratedBy | None = None

    @field_validator("id")
    @classmethod
    def _v_id(cls, v: str) -> str:
        return _check_id(v, "Scene.id")

    def line(self, line_id: str) -> SceneDialogueLine | None:
        return next((ln for ln in self.dialogue if ln.line_id == line_id), None)


class RefPointer(Model):
    """Pointer from a working set into the Global Asset Library (never a copy)."""

    id: str
    ref: str  # project-relative path of the record (yaml) in bible/ or library/
    version: int = 1
    approval_status: ApprovalStatus = ApprovalStatus.PROPOSED


class PinnedRef(Model):
    path: str  # project-relative (copied into scenes/<SC>/pinned/ on pin)
    why: str
    pinned_at: str
    decision_id: str | None = None


class Preference(Model):
    id: str
    scope: str  # project | character:<id> | scene:<id> | agent:<role> | user:<id>
    text: str
    weight: float = 1.0
    source_decision_ids: list[str] = Field(default_factory=list)
    created_at: str
    active: bool = True

    @field_validator("scope")
    @classmethod
    def _v_scope(cls, v: str) -> str:
        if not _SCOPE_RE.match(v):
            raise ValueError(f"preference scope {v!r} must match {PREFERENCE_SCOPE_PATTERN}")
        return v


class SceneWorkingSet(Model):
    """THE ONLY CONTEXT AN AGENT SEES FOR SCENE N (§4.5)."""

    scene_id: str
    style_lock: RefPointer
    entity_states: list[RefPointer] = Field(default_factory=list)
    voice_profiles: list[RefPointer] = Field(default_factory=list)
    environment: RefPointer | None = None
    scene: Scene
    prev_scene_exit_state: dict[str, Any] = Field(default_factory=dict)
    next_scene_entry_needs: dict[str, Any] = Field(default_factory=dict)
    story_so_far: list[str] = Field(default_factory=list, max_length=5)
    scene_preferences: list[Preference] = Field(default_factory=list)
    pinned_extra_refs: list[PinnedRef] = Field(default_factory=list)
    feedback_history: list[dict[str, Any]] = Field(default_factory=list)
    working_set_hash: str | None = None
    generated_by: GeneratedBy | None = None


# ---------------------------------------------------------------------------------------------
# §4.6 CreativeShot -> GenClip -> Take
# ---------------------------------------------------------------------------------------------
class VisualSpec(Model):
    """9-element visual IR in FIXED order (linter L10)."""

    subject_and_action: str = ""  # ONE sentence, ONE physical verb, entity IDs only; speech is not an action
    camera: str = ""
    lens_and_aspect: str = ""
    lighting: str = ""  # physical sources; NO color words (L7)
    palette: str = ""  # NO light sources (L7)
    composition: str = ""
    atmosphere: str = ""
    mood: str = ""
    attribution: str = ""  # LAST


class AudioLine(Model):
    character_id: str
    text: str  # verbatim from scene.dialogue (L14)
    line_id: str | None = None
    delivery: str = ""
    voice: str = ""  # VoiceProfile.description (+ state variant) (L15)
    transliteration: str | None = None


class AudioSpec(Model):
    dialogue: list[AudioLine] = Field(default_factory=list)
    off_screen: list[AudioLine] = Field(default_factory=list)
    ambience: str = ""
    diegetic_sfx: list[str] = Field(default_factory=list)
    music: str = "none"  # ALWAYS none (L17 auto-fixes anything else)

    def all_lines(self) -> list[AudioLine]:
        return [*self.dialogue, *self.off_screen]


class ShotSpec(Model):
    visual: VisualSpec = Field(default_factory=VisualSpec)
    audio_spec: AudioSpec = Field(default_factory=AudioSpec)
    negative: list[str] = Field(default_factory=list)


class Conditioning(Model):
    reference_uris: list[str] = Field(default_factory=list)
    reference_entity_states: list[str] = Field(default_factory=list)  # which state packs are attached (L6)
    spatial_anchor: str | None = None  # environment geometry still / derived angle
    first_frame_anchor: str | None = None  # last frame of preceding clip's approved take (L9)
    first_frame_anchor_take_id: str | None = None
    last_frame_target: str | None = None
    identity_embeddings: list[str] = Field(default_factory=list)
    voice_reference_audio: list[str] = Field(default_factory=list)


class ModelTarget(Model):
    preferred: str = "omni"
    fallback: str | None = "veo"


class PromptBundle(Model):
    """Model-ready serialisation of one GenClip (Prompt Engineer output; adapters add provider fields).

    All media paths are project-relative POSIX paths. ``reference_legend`` maps each entity-state /
    environment id to the 1-based indices of its images inside ``reference_images``.
    """

    model_id: str  # registry id (omni | veo)
    clip_id: str | None = None
    prompt: str
    negative_prompt: str | None = None  # None when the model has no negative field (negatives folded into prompt)
    reference_images: list[str] = Field(default_factory=list)
    reference_legend: dict[str, list[int]] = Field(default_factory=dict)
    first_frame: str | None = None
    last_frame: str | None = None
    voice_reference_audio: list[str] = Field(default_factory=list)
    duration_s: float
    aspect_ratio: str = "9:16"
    resolution: str = "1080p"


class GenerationPayload(Model):
    """Exactly what is sent to a provider. ``payload_hash`` dedupes dispatches across kills/resumes."""

    model_id: str
    provider_model: str
    bundle: PromptBundle
    seed: int | None = None
    payload_hash: str
    extra: dict[str, Any] = Field(default_factory=dict)


class LintFinding(Model):
    rule: str  # L1..L18
    severity: Literal["error", "warn", "info"]
    action: Literal["auto_fixed", "bounce", "fail", "warn", "fallback"]
    message: str
    field: str | None = None
    suggestion: str | None = None


class LintReport(Model):
    ts: str
    passed: bool
    blocking: bool
    bounce_to_decomposer: bool = False
    findings: list[LintFinding] = Field(default_factory=list)
    auto_fixes: list[str] = Field(default_factory=list)
    model_used: str | None = None  # after L12 fallback
    spec_hash: str | None = None


class CreativeShot(Model):
    id: str  # e.g. SC12_SH04
    scene_id: str
    order: int = 0
    intention: str = ""
    coverage_type: str = "single"
    duration_s: float = 0.0  # creative duration
    board_frame: str | None = None
    environment_angle: str | None = None  # geometry | reverse | angle_45L | angle_45R | wide_establishing | insert:<k>
    camera_side: str | None = None  # left|right of axis_of_action (L13)
    beat_ids: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)  # atomic physical actions in order (decomposer input)
    lines: list[str] = Field(default_factory=list)  # dialogue line ids covered by this shot, in order
    line_action_index: dict[str, int] = Field(default_factory=dict)  # line_id -> index in actions it is spoken during
    multibeat_ok: bool = False  # human override (chat decision): in-clip multi-beat staging allowed -> L2 warns only
    continuous: bool = True  # consecutive clips of this shot are one continuous camera take (anchored first frames)
    spec: ShotSpec = Field(default_factory=ShotSpec)
    gen_clips: list[str] = Field(default_factory=list)
    approval_status: ApprovalStatus = ApprovalStatus.PROPOSED
    version: int = 1
    generated_by: GeneratedBy | None = None

    @field_validator("id", "scene_id")
    @classmethod
    def _v_ids(cls, v: str) -> str:
        return _check_id(v, "CreativeShot id")


class GenClip(Model):
    id: str  # e.g. SC12_SH04_C02
    shot_id: str
    scene_id: str
    sequence_index: int  # 1-based within the shot
    duration_s: float  # <= registry[model].max_duration_s (L8)
    spec: ShotSpec = Field(default_factory=ShotSpec)
    lines: list[str] = Field(default_factory=list)  # dialogue lines fully contained in this clip
    performance_state: str = ""
    conditioning: Conditioning = Field(default_factory=Conditioning)
    model_target: ModelTarget = Field(default_factory=ModelTarget)
    stitch_risk: RiskTier = RiskTier.LOW
    anchor_from_clip_id: str | None = None  # preceding clip whose approved take's last frame anchors this one
    environment_angle: str | None = None
    camera_side: str | None = None
    takes: list[str] = Field(default_factory=list)
    selected_take_id: str | None = None
    status: ClipStatus = ClipStatus.PENDING
    lint_report: LintReport | None = None
    flags: list[str] = Field(default_factory=list)  # e.g. merged_short_clip, recommend_cutaway
    version: int = 1
    generated_by: GeneratedBy | None = None

    @field_validator("id", "shot_id", "scene_id")
    @classmethod
    def _v_ids(cls, v: str) -> str:
        return _check_id(v, "GenClip id")


class QAReport(Model):
    """§7.2 post-generation QA. Scores are 0..1 (None = not measured)."""

    take_id: str | None = None
    identity_similarity: dict[str, float] = Field(default_factory=dict)  # state id -> score vs LIBRARY pack
    anchor_match: float | None = None
    spatial_match: float | None = None
    spatial_continuity: float | None = None
    motion_continuity: float | None = None
    prompt_adherence: float | None = None
    performance_match: float | None = None
    artifact_flags: list[str] = Field(default_factory=list)
    transcript: str | None = None
    transcript_match: float | None = None  # 1 - WER vs expected lines
    speaker_similarity: dict[str, float | None] = Field(default_factory=dict)  # null until anchor set
    music_present: bool = False
    unwanted_audio_flags: list[str] = Field(default_factory=list)
    ambience_continuity: float | None = None
    delivery_match: float | None = None
    composite: float | None = None
    recommendation: Recommendation = Recommendation.REVIEW
    hard_reject_reasons: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    generated_by: GeneratedBy | None = None


class Take(Model):
    id: str  # e.g. SC12_SH04_C02_T01
    clip_id: str
    model: str  # registry id (omni | veo)
    provider_model: str | None = None  # concrete API model name actually used
    model_version: str | None = None
    seed: int | None = None
    payload_hash: str
    video_uri: str | None = None  # native AV take.mp4 (project-relative)
    first_frame_uri: str | None = None
    last_frame_uri: str | None = None
    dialogue_wav_uri: str | None = None
    transcript: str | None = None
    qa: QAReport | None = None
    rank: int | None = None
    status: TakeStatus = TakeStatus.CANDIDATE
    dispatch_status: DispatchStatus = DispatchStatus.PLANNED
    operation_name: str | None = None  # provider operation / interaction id (resume after kill)
    anchors_downstream: list[str] = Field(default_factory=list)
    cost_estimate: float = 0.0
    error: dict[str, Any] | None = None
    imported_from: str | None = None  # Phase 0 wrap: legacy clip path
    created_at: str | None = None
    generated_by: GeneratedBy | None = None

    @field_validator("id", "clip_id")
    @classmethod
    def _v_ids(cls, v: str) -> str:
        return _check_id(v, "Take id")


# ---------------------------------------------------------------------------------------------
# §4.8 Decisions, Preferences, Questions
# ---------------------------------------------------------------------------------------------
class Decision(Model):
    id: str
    ts: str
    user: str = "director"
    tier: DecisionTier
    gate: str | None = None
    target_artifact_id: str
    source: DecisionSource = DecisionSource.CHAT
    action: DecisionAction
    payload: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class Question(Model):
    id: str
    ts: str
    text: str
    scene_id: str | None = None
    stage: str | None = None
    blocks: list[str] = Field(default_factory=list)
    options: list[str] = Field(default_factory=list)
    status: Literal["open", "answered"] = "open"
    answer: str | None = None
    answered_at: str | None = None
    decision_id: str | None = None


# ---------------------------------------------------------------------------------------------
# §10.4 Resolve handoff record
# ---------------------------------------------------------------------------------------------
class ResolveHandoff(Model):
    scene_id: str
    timeline_name: str
    version: int = 1
    otio_path: str | None = None
    fingerprint: str | None = None
    ts: str
    ownership: Ownership = Ownership.AGY
    bin_path: str | None = None
    markers_enabled: bool = True
    orange_marker_opt_in: bool = True
    legacy: bool = False  # Phase 0: pre-v3 episode timeline adopted as human-owned
    agy_marker_names: list[str] = Field(default_factory=list)  # AGY-authored markers (excluded from fingerprint)
    history: list[dict[str, Any]] = Field(default_factory=list)
    generated_by: GeneratedBy | None = None


# ---------------------------------------------------------------------------------------------
# §3 project.yaml
# ---------------------------------------------------------------------------------------------
class QAThresholds(Model):
    composite_accept: float = 0.75
    auto_select_composite_min: float = 0.75
    transcript_match_min: float = 0.80
    speaker_similarity_min: float = 0.70
    speaker_similarity_warn: float = 0.70
    speaker_similarity_reject: float = 0.55
    identity_similarity_min: float = 0.70
    ambience_continuity_min: float = 0.60


class OpenDecisions(Model):
    """§15 — defaults per spec; changed only through chat decisions."""

    identity_conditioning: Literal["native_refs", "lora"] = "native_refs"
    incidental_conditioning: Literal["single_still_if_line_or_gt2s", "text_only"] = "single_still_if_line_or_gt2s"
    stem_separation: bool = False
    voice_reference_audio: bool = True  # on where the model supports it
    default_model_tossup: str | None = None  # filled from pilot results
    roundtrip_transport: Literal["db_polling", "otio_export"] = "db_polling"
    orange_marker_default: bool = True
    confirmed_by_human: bool = False


class ProjectSettings(Model):
    fps: int = 24
    width: int = 1080
    height: int = 1920
    aspect_ratio: str = "9:16"
    takes_per_clip: int = 2
    speech_pace_wps: float = 2.2  # L18 words / second (human Hindi pace: 8-14 words per 8 s shot)
    speech_margin_s: float = 0.5  # L18 margin
    speech_pre_roll_s: float = 0.6  # decomposer duration estimate: breath cushion before a line
    speech_post_roll_s: float = 0.6  # decomposer duration estimate: breath cushion after a line
    min_action_s: float = 2.5  # decomposer duration estimate for a silent physical action
    min_clip_s: float = 2.0  # §8 clip < 2 s -> merge with neighbour
    split_every_line: bool = True  # §8 clip boundary at every line boundary
    single_speaker_clips: bool = True  # workspace rule: only the intended speaker moves their lips per clip
    lint_allowed_words: list[str] = Field(default_factory=list)  # human-approved exemptions from L1
    lint_extra_acronyms: list[str] = Field(default_factory=list)  # extra all-caps tokens L3 must accept
    polish_timeout_s: int = 3600
    auto_select_low_risk: bool = True
    qa: QAThresholds = Field(default_factory=QAThresholds)
    budget_cap_multiplier: float = 1.25  # per-scene cap = estimated_cost * multiplier
    resolve_project_name: str | None = None
    resolve_db_path: str | None = None
    resolve_poll_interval_s: int = 60
    max_agent_rounds: int = 3
    control_mode: Literal["director", "hybrid", "autopilot"] = "director"
    content_format: Literal["microdrama_9_16", "episode_16_9", "feature_film", "marketing_spot"] = "microdrama_9_16"
    open_decisions: OpenDecisions = Field(default_factory=OpenDecisions)


class ProjectConfig(Model):
    project_id: str
    title: str
    script_version: int = 1
    language: str = "hi"
    model_pins: dict[str, str] = Field(default_factory=dict)  # registry id -> provider model / version
    settings: ProjectSettings = Field(default_factory=ProjectSettings)
    created_at: str | None = None
    generated_by: GeneratedBy | None = None

    @field_validator("project_id")
    @classmethod
    def _v_id(cls, v: str) -> str:
        if not isinstance(v, str) or not re.match(r"^[A-Za-z][A-Za-z0-9_]{0,95}$", v):
            raise ValueError(f"project_id {v!r} must match ^[A-Za-z][A-Za-z0-9_]{{0,95}}$")
        return v


# ---------------------------------------------------------------------------------------------
# §12 orchestration state (state.json)
# ---------------------------------------------------------------------------------------------
class StageTransition(Model):
    ts: str
    from_stage: str
    to_stage: str
    reason: str = ""
    decision_id: str | None = None


class SceneState(Model):
    scene_id: str
    stage: SceneStage = SceneStage.PENDING
    gates: dict[str, str] = Field(default_factory=dict)  # G2..G6 -> pending|awaiting|passed|auto_passed
    awaiting: str | None = None  # e.g. "awaiting:G3"
    blocked_by: list[str] = Field(default_factory=list)  # question ids / pack locks / voice anchors
    paused: bool = False
    budget_cap: float | None = None
    budget_spent: float = 0.0
    agent_rounds: dict[str, int] = Field(default_factory=dict)
    history: list[StageTransition] = Field(default_factory=list)
    updated_at: str | None = None


class ProjectState(Model):
    project_id: str
    global_stage: GlobalStage = GlobalStage.PREPRODUCTION
    gates: dict[str, str] = Field(default_factory=dict)  # G1, G7
    scenes: dict[str, SceneState] = Field(default_factory=dict)
    schedule: list[str] = Field(default_factory=list)  # [pilot] + ordered scene ids
    pilot_scene_id: str | None = None
    dispatched_payloads: dict[str, str] = Field(default_factory=dict)  # payload_hash -> take id (dedupe)
    pack_locks: dict[str, str] = Field(default_factory=dict)  # entity_state / voice id -> owning scene id
    pending_confirms: dict[str, dict[str, Any]] = Field(default_factory=dict)  # artifact id -> fan-out shown
    regen_queue: list[dict[str, Any]] = Field(default_factory=list)  # {clip_id, steer, decision_id, source, ts}
    resume_cursor: dict[str, Any] = Field(default_factory=dict)
    paused: bool = False
    budget_spent_total: float = 0.0
    write_seq: int = 0
    updated_at: str | None = None
