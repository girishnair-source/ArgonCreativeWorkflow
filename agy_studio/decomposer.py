"""Shot Decomposer — CreativeShot -> GenClips (BUILD_SPEC_v3 §6 "Shot Decomposer", §8 decomposition rule).

Deterministic and data-driven (acceptance criterion 10): every limit comes from the model capability registry
(``max_duration_s``, ``allowed_durations_s``, ``max_speakers_per_clip``, ``max_reference_images``) or from
``ProjectSettings`` (speech pace, breath cushions, minimum action / clip durations, ``split_every_line``,
``single_speaker_clips``). Changing any of them changes the decomposition with no code change.

Rules (§8):

* a clip boundary at every action boundary and (``split_every_line``) at every dialogue-line boundary; a line is
  never split; lines attach to actions through ``shot.line_action_index`` (default: the last action);
* duration = snap(max(min_action_s, pre_roll + words / pace + post_roll)) onto the registry's allowed durations,
  never above ``max_duration_s``; a silent action longer than the maximum becomes continuation clips; a single line
  that cannot fit the maximum is flagged ``line_exceeds_max_duration`` (L18 then bounces it);
* a clip whose content is shorter than ``min_clip_s`` is merged with a neighbour of the same action when the
  duration / speaker limits allow (flag ``merged_short_clip``), else padded and flagged ``recommend_cutaway``;
* each clip's ``subject_and_action`` is its ONE action; the other visual fields are inherited from the shot spec;
* ``audio_spec`` is filled VERBATIM from ``scene.dialogue`` (voice = locked profile + active-state variant);
  scripted off-screen lines — and lines whose speaker is not visible in the shot — go to ``off_screen``;
* conditioning = the active-state packs of every entity referenced in the clip's visual text plus its on-screen
  speakers (principal > supporting > incidental, >= 1 image per pack, capped by ``max_reference_images`` including
  the spatial anchor), the environment angle as spatial anchor, voice reference audio where supported;
* consecutive clips of a ``continuous`` shot anchor on the previous clip (first frame = its selected take's last
  frame, attached later by :func:`link_anchor_frame`);
* re-decomposition never reuses the id of a clip that has takes: old clips become ``stale`` (never deleted).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from . import continuity
from .agent_log import log_agent_call
from .fsio import generated_by, read_record, validate_id
from .models import (
    VISUAL_FIELDS,
    AudioLine,
    AudioMode,
    AudioSpec,
    ClipStatus,
    Conditioning,
    CreativeShot,
    Entity,
    EntityState,
    Environment,
    GenClip,
    ModelTarget,
    ProjectSettings,
    RiskTier,
    Scene,
    SceneDialogueLine,
    ShotSpec,
    StyleLock,
    Tier,
    VisualSpec,
    VoiceProfile,
)
from .project import Project, parse_artifact_id
from .project import clip_id as make_clip_id
from .registry import ModelSpec, Registry, UnknownModelError

AGENT_NAME = "Shot Decomposer"
PROVENANCE_AGENT = "agy.shot_decomposer"
DEFAULT_MODEL_ID = "omni"
AUDIO_NEGATIVES: tuple[str, ...] = ("background music", "score", "narration")
REQUIRED_PACK_ROLES: dict[Tier, tuple[str, ...]] = {
    Tier.PRINCIPAL: ("front", "three_quarter", "profile", "rear", "face_closeup"),
    Tier.SUPPORTING: ("front", "three_quarter", "face_closeup"),
    Tier.INCIDENTAL: ("single_ref",),
}
TIER_RANK: dict[Tier, int] = {Tier.PRINCIPAL: 0, Tier.SUPPORTING: 1, Tier.INCIDENTAL: 2}
TEXT_ONLY_MAX_S = 2.0  # Open Decision 2: incidental with no line and <= 2 s may be text-only
CAMERA_ACRONYMS: frozenset[str] = frozenset({
    "CU", "ECU", "XCU", "BCU", "MCU", "MCS", "MS", "MWS", "MLS", "WS", "EWS", "XWS", "LS", "ELS", "XLS", "FS",
    "OTS", "POV", "DOF", "CCTV", "FPV", "DSLR", "SLR", "ND", "LED", "HMI", "IR", "TV", "LUT",
})
_HIGH_MOTION_RX = re.compile(
    r"\b(?:fall|falls|falling|fell|fallen|stumbl\w*|trip|trips|tripp\w*|jump\w*|leap\w*|run|runs|running|ran|"
    r"sprint\w*|spin|spins|spinning|spun|tumbl\w*|collaps\w*|dive|dives|diving|dove|lung\w*|throw\w*|threw|"
    r"crash\w*|fight\w*|fought|punch\w*|kick\w*|tackl\w*|slam\w*|flail\w*|thrash\w*|roll\w*|flip\w*)\b",
    re.IGNORECASE,
)
_EPS = 1e-6


# ---------------------------------------------------------------------------------------------
# Shared helpers (also used by linter.py and prompt_engineer.py)
# ---------------------------------------------------------------------------------------------
def merge_unique(*groups: Iterable[str] | None) -> list[str]:
    """Concatenate string groups, dropping blanks and case-insensitive duplicates (first spelling wins)."""
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group or ():
            text = (item or "").strip()
            key = text.casefold()
            if text and key not in seen:
                seen.add(key)
                out.append(text)
    return out


def effective_max_speakers(registry: Registry, model_id: str, settings: ProjectSettings) -> int:
    """Registry speaker limit, further capped at 1 when the workspace single-speaker rule is on."""
    limit = int(registry.max_speakers_per_clip(model_id))
    if settings.single_speaker_clips:
        limit = min(limit, 1)
    return max(0, limit)


def video_model_ids(registry: Registry) -> list[str]:
    return [m.id for m in registry.video_models()]


_NO_FALLBACK = frozenset({"", "none", "null", "off"})


def resolve_model_target(registry: Registry, scene: Scene, model_id: str | None = None,
                         settings: Any = None) -> ModelTarget:
    """Pick the preferred + fallback video engine for a scene.

    Preferred: ``model_id`` > ``scene.model_override`` > ``breakdown.recommended_model`` >
    ``settings.video_model_primary`` (project.yaml) > ``omni`` > first enabled video model.
    Fallback: ``settings.video_model_secondary`` when set (``"none"`` disables engine fallback), else the other
    enabled video model. Both must be enabled ``video_av`` registry ids.
    """
    video = video_model_ids(registry)
    if not video:
        raise UnknownModelError("no enabled video_av model in the registry")
    if model_id is not None and model_id not in video:
        raise ValueError(f"model {model_id!r} is not an enabled video model ({video})")
    primary_setting = (getattr(settings, "video_model_primary", None) or "").strip() or None
    if primary_setting and primary_setting not in video:
        raise ValueError(f"settings.video_model_primary={primary_setting!r} is not an enabled video model ({video})")
    candidates = [model_id, scene.model_override, scene.breakdown.recommended_model if scene.breakdown else None,
                  primary_setting, DEFAULT_MODEL_ID, video[0]]
    preferred = next(m for m in candidates if m and m in video)
    secondary = getattr(settings, "video_model_secondary", None)
    if secondary is not None:
        sec = str(secondary).strip()
        if sec.lower() in _NO_FALLBACK:
            return ModelTarget(preferred=preferred, fallback=None)
        if sec not in video:
            raise ValueError(f"settings.video_model_secondary={sec!r} is not an enabled video model ({video})")
        return ModelTarget(preferred=preferred, fallback=None if sec == preferred else sec)
    fallback = next((m for m in video if m != preferred), None)
    return ModelTarget(preferred=preferred, fallback=fallback)


def snap_duration(spec: ModelSpec, seconds: float) -> float:
    """Smallest registry-allowed duration >= ``seconds`` that is <= ``max_duration_s`` (the largest if none fits)."""
    hi = spec.max_duration_s
    allowed = sorted(float(d) for d in (spec.allowed_durations_s or []) if hi is None or d <= hi + _EPS)
    if allowed:
        return next((d for d in allowed if d + _EPS >= seconds), allowed[-1])
    lo = float(spec.min_duration_s or 0.0)
    value = math.ceil(max(seconds, lo) * 2 - _EPS) / 2.0  # half-second granularity
    return float(min(value, hi)) if hi is not None else float(value)


def duration_problems(spec: ModelSpec, duration_s: float) -> list[str]:
    """Why ``duration_s`` is not acceptable for ``spec`` (empty list = acceptable). Used by L8 and L12."""
    lo, hi = spec.min_duration_s, spec.max_duration_s
    problems: list[str] = []
    if lo is not None and duration_s < lo - _EPS:
        problems.append(f"{duration_s:g}s is below {spec.id} min_duration_s {lo:g}s")
    if hi is not None and duration_s > hi + _EPS:
        problems.append(f"{duration_s:g}s exceeds {spec.id} max_duration_s {hi:g}s")
    if not problems and spec.allowed_durations_s:
        allowed = [float(d) for d in spec.allowed_durations_s if hi is None or d <= hi + _EPS]
        if allowed and not any(abs(duration_s - d) <= 1e-3 for d in allowed):
            problems.append(f"{duration_s:g}s is not an allowed {spec.id} duration {allowed}")
    return problems


def line_seconds(text: str, settings: ProjectSettings) -> float:
    """Human-paced on-screen time for one line: pre-roll breath + words / pace + post-roll breath."""
    return (settings.speech_pre_roll_s + continuity.count_words(text) / settings.speech_pace_wps
            + settings.speech_post_roll_s)


def scene_audio_mode(scene: Scene) -> AudioMode:
    return scene.breakdown.audio_mode if scene.breakdown else AudioMode.NATIVE_AV


def required_negatives(style: StyleLock | None, states: Iterable[EntityState], audio_mode: AudioMode) -> list[str]:
    """L11 floor: style global negatives + state-specific negatives + audio negatives for native-AV clips."""
    groups: list[Iterable[str]] = [style.global_negative if style else []]
    groups.extend(st.state_specific_negative for st in states)
    if audio_mode == AudioMode.NATIVE_AV:
        groups.append(AUDIO_NEGATIVES)
    return merge_unique(*groups)


def scene_shot_order(project: Project, scene: Scene) -> list[str]:
    """``scene.creative_shots`` first, then any other saved shots of the scene by (order, id)."""
    order = list(dict.fromkeys(scene.creative_shots))
    for shot in project.list_shots(scene.id):
        if shot.id not in order:
            order.append(shot.id)
    return order


def state_owner_index(entities: Mapping[str, Entity]) -> dict[str, str]:
    return {st.id: ent.id for ent in entities.values() for st in ent.states}


def active_state_map(scene: Scene, shot_order: list[str], shot_id: str, clip_index: int, *,
                     state_owner: Mapping[str, str], entity_ids: Iterable[str]) -> dict[str, str]:
    """entity id -> its active state at (shot, clip_index), applying mid-scene transitions."""
    ids = list(entity_ids)
    out: dict[str, str] = {}
    for st in continuity.active_states_at(scene, shot_order, shot_id, clip_index):
        owner = state_owner.get(st) or continuity.entity_id_of_state(st, ids)
        if owner and owner not in out:
            out[owner] = st
    return out


def load_voice_profiles(project: Project, character_ids: Iterable[str] = ()) -> dict[str, VoiceProfile]:
    """character id -> current VoiceProfile (casting.yaml proposals or locked library versions)."""
    ids = list(dict.fromkeys([*project.casting().get("voice_profiles", {}).keys(), *character_ids]))
    out: dict[str, VoiceProfile] = {}
    for cid in ids:
        try:
            profile = project.load_voice_profile(validate_id(str(cid), "character id"))
        except (FileNotFoundError, ValueError):
            continue
        if profile is not None:
            out[cid] = profile
    return out


@dataclass
class VisualRefs:
    """Classification of every UPPERCASE token in a VisualSpec (L3/L4 input, conditioning input)."""

    tokens: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)  # referenced entity ids (bare or via a state token)
    explicit_states: dict[str, str] = field(default_factory=dict)  # entity id -> explicitly named state id
    conflicting_states: list[str] = field(default_factory=list)  # a second, different state token of one entity
    environments: list[str] = field(default_factory=list)
    voices: list[str] = field(default_factory=list)
    lighting: list[str] = field(default_factory=list)
    acronyms: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)


def visual_text(visual: VisualSpec) -> str:
    return "\n".join(getattr(visual, name) or "" for name in VISUAL_FIELDS)


def scan_visual_refs(visual: VisualSpec, *, entities: Mapping[str, Entity], state_owner: Mapping[str, str],
                     environments: Iterable[str] = (), voice_ids: Iterable[str] = (),
                     lighting_ids: Iterable[str] = (), acronyms: Iterable[str] = ()) -> VisualRefs:
    env_ids, voices, lights = set(environments), set(voice_ids), set(lighting_ids)
    acr = {a.upper() for a in acronyms}
    refs = VisualRefs()
    for tok in continuity.entity_tokens(visual_text(visual)):
        refs.tokens.append(tok)
        if tok in state_owner:
            owner = state_owner[tok]
            prev = refs.explicit_states.get(owner)
            if prev and prev != tok:
                refs.conflicting_states.append(tok)
            else:
                refs.explicit_states[owner] = tok
            if owner not in refs.entities:
                refs.entities.append(owner)
        elif tok in entities:
            if tok not in refs.entities:
                refs.entities.append(tok)
        elif tok in env_ids:
            refs.environments.append(tok)
        elif tok in voices:
            refs.voices.append(tok)
        elif tok in lights:
            refs.lighting.append(tok)
        elif tok in acr:
            refs.acronyms.append(tok)
        else:
            refs.unknown.append(tok)
    return refs


def allocate_reference_images(groups: Sequence[tuple[str, Sequence[str]]], cap: int, *,
                              anchor: str | None = None) -> tuple[dict[str, list[str]], bool]:
    """Pick at most ``cap`` images: one per pack first (``groups`` already in priority order), then the spatial
    ``anchor``, then the remaining pack images round-robin. Returns (state id -> chosen uris, anchor_included)."""
    cap = max(0, int(cap))
    chosen: dict[str, list[str]] = {sid: [] for sid, _ in groups}
    used: set[str] = set()
    count = 0
    for sid, imgs in groups:
        if count >= cap:
            break
        first = next((u for u in imgs if u and u not in used), None)
        if first:
            chosen[sid].append(first)
            used.add(first)
            count += 1
    anchor_included = False
    if anchor and count < cap and anchor not in used:
        anchor_included = True
        used.add(anchor)
        count += 1
    pending = {sid: [u for u in imgs if u and u not in used] for sid, imgs in groups}
    while count < cap and any(pending.values()):
        for sid, _ in groups:
            if count >= cap:
                break
            queue = pending[sid]
            while queue and queue[0] in used:
                queue.pop(0)
            if queue:
                uri = queue.pop(0)
                chosen[sid].append(uri)
                used.add(uri)
                count += 1
    return chosen, anchor_included


# ---------------------------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------------------------
@dataclass
class _Segment:
    action_index: int
    action: str
    line_ids: list[str] = field(default_factory=list)
    content_s: float = 0.0
    planned_s: float = 0.0
    continues_action: bool = False
    flags: list[str] = field(default_factory=list)


@dataclass
class _Ctx:
    project: Project
    registry: Registry
    scene: Scene
    shot: CreativeShot
    shot_order: list[str]
    settings: ProjectSettings
    style: StyleLock | None
    entities: dict[str, Entity]
    states: dict[str, tuple[Entity, EntityState]]
    state_owner: dict[str, str]
    environments: dict[str, Environment]
    environment: Environment | None
    voices: dict[str, VoiceProfile]
    lines: dict[str, SceneDialogueLine]
    target: ModelTarget
    spec: ModelSpec
    max_d: float
    max_speakers: int
    audio_mode: AudioMode

    @classmethod
    def build(cls, project: Project, registry: Registry, shot: CreativeShot, model_id: str | None) -> "_Ctx":
        scene = project.load_scene(shot.scene_id)
        entities = {e.id: e for e in project.list_entities()}
        environments = {e.id: e for e in project.list_environments()}
        settings = project.config.settings
        target = resolve_model_target(registry, scene, model_id, settings=settings)
        spec = registry.get(target.preferred)
        if spec.max_duration_s is None:
            raise ValueError(f"model {spec.id} has no max_duration_s in the registry")
        return cls(
            project=project, registry=registry, scene=scene, shot=shot,
            shot_order=scene_shot_order(project, scene), settings=settings, style=project.load_style_lock(),
            entities=entities, states={st.id: (e, st) for e in entities.values() for st in e.states},
            state_owner=state_owner_index(entities), environments=environments,
            environment=environments.get(scene.environment_id) if scene.environment_id else None,
            voices=load_voice_profiles(project, [ln.character_id for ln in scene.dialogue]),
            lines={ln.line_id: ln for ln in scene.dialogue}, target=target, spec=spec,
            max_d=float(spec.max_duration_s),
            max_speakers=effective_max_speakers(registry, target.preferred, settings),
            audio_mode=scene_audio_mode(scene),
        )

    def speakers(self, line_ids: Sequence[str]) -> list[str]:
        return list(dict.fromkeys(self.lines[lid].character_id for lid in line_ids))

    def lines_seconds(self, line_ids: Sequence[str]) -> float:
        return sum(line_seconds(self.lines[lid].text, self.settings) for lid in line_ids)

    def refs(self, visual: VisualSpec) -> VisualRefs:
        lighting = [ls.id for env in self.environments.values() for ls in env.lighting_setups]
        return scan_visual_refs(visual, entities=self.entities, state_owner=self.state_owner,
                                environments=self.environments, voice_ids=[v.id for v in self.voices.values()],
                                lighting_ids=lighting, acronyms=[*CAMERA_ACRONYMS, *self.settings.lint_extra_acronyms])


def _group_lines(ctx: _Ctx, line_ids: list[str]) -> list[list[str]]:
    if not line_ids:
        return []
    if ctx.settings.split_every_line:
        return [[lid] for lid in line_ids]
    groups: list[list[str]] = []
    current: list[str] = []
    for lid in line_ids:
        candidate = [*current, lid]
        if current and (len(ctx.speakers(candidate)) > ctx.max_speakers
                        or ctx.lines_seconds(candidate) > ctx.max_d + _EPS):
            groups.append(current)
            current = [lid]
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups


def _merge_short(ctx: _Ctx, segments: list[_Segment]) -> list[_Segment]:
    out = list(segments)
    i = 0
    while i < len(out):
        seg = out[i]
        if seg.line_ids and seg.content_s < ctx.settings.min_clip_s - _EPS:
            target = None
            for j in (i - 1, i + 1):
                if not 0 <= j < len(out):
                    continue
                nb = out[j]
                if nb.action_index != seg.action_index or not nb.line_ids:
                    continue
                merged = [*nb.line_ids, *seg.line_ids] if j < i else [*seg.line_ids, *nb.line_ids]
                if len(ctx.speakers(merged)) <= ctx.max_speakers and ctx.lines_seconds(merged) <= ctx.max_d + _EPS:
                    target = j
                    break
            if target is not None:
                nb = out[target]
                nb.line_ids = [*nb.line_ids, *seg.line_ids] if target < i else [*seg.line_ids, *nb.line_ids]
                nb.content_s = ctx.lines_seconds(nb.line_ids)
                nb.flags = merge_unique(nb.flags, seg.flags, ["merged_short_clip"])
                if target > i:
                    nb.continues_action = seg.continues_action
                del out[i]
                continue
            seg.flags = merge_unique(seg.flags, ["recommend_cutaway"])
        i += 1
    return out


def _plan_segments(ctx: _Ctx) -> list[_Segment]:
    s = ctx.settings
    actions = [a.strip() for a in ctx.shot.actions if a and a.strip()]
    if not actions:
        actions = [(ctx.shot.spec.visual.subject_and_action or "").strip()]
    n_actions = len(actions)
    attached: dict[int, list[str]] = {i: [] for i in range(n_actions)}
    for lid in dict.fromkeys(ctx.shot.lines):
        if lid not in ctx.lines:
            raise ValueError(f"{ctx.shot.id} lists dialogue line {lid} which is not in {ctx.scene.id}.dialogue")
        idx = int(ctx.shot.line_action_index.get(lid, n_actions - 1))
        attached[min(max(idx, 0), n_actions - 1)].append(lid)

    segments: list[_Segment] = []
    for i, action in enumerate(actions):
        groups = _group_lines(ctx, attached[i])
        if not groups:
            segments.append(_Segment(i, action, content_s=s.min_action_s))
        for g, lids in enumerate(groups):
            segments.append(_Segment(i, action, list(lids), content_s=ctx.lines_seconds(lids), continues_action=g > 0))
    segments = _merge_short(ctx, segments)

    for seg in segments:
        if seg.line_ids:
            seg.planned_s = max(s.min_action_s, s.min_clip_s, seg.content_s)
            if seg.content_s > ctx.max_d + _EPS:
                seg.flags = merge_unique(seg.flags, ["line_exceeds_max_duration"])
    silent = [seg for seg in segments if not seg.line_ids]
    if silent:
        dialogue_total = sum(snap_duration(ctx.spec, seg.planned_s) for seg in segments if seg.line_ids)
        share = max(0.0, float(ctx.shot.duration_s or 0.0) - dialogue_total) / len(silent)
        for seg in silent:
            seg.planned_s = max(s.min_action_s, share)
            if seg.planned_s < s.min_clip_s - _EPS:
                seg.planned_s = s.min_clip_s
                seg.flags = merge_unique(seg.flags, ["recommend_cutaway"])

    out: list[_Segment] = []
    for seg in segments:
        if seg.line_ids or seg.planned_s <= ctx.max_d + _EPS:
            out.append(seg)
            continue
        parts = math.ceil(seg.planned_s / ctx.max_d - _EPS)
        each = seg.planned_s / parts
        for k in range(parts):
            out.append(_Segment(seg.action_index, seg.action, [], content_s=each, planned_s=each,
                                continues_action=seg.continues_action or k > 0,
                                flags=merge_unique(seg.flags, ["continuation"] if k else [])))
    return out


def _has_takes(project: Project, clip: GenClip) -> bool:
    if clip.takes or clip.selected_take_id:
        return True
    takes_dir = project.clip_dir(clip.id) / "takes"
    return takes_dir.is_dir() and any(takes_dir.iterdir())


def _allocate_ids(project: Project, shot_id: str, n: int) -> tuple[list[str], list[GenClip], dict[str, GenClip]]:
    """New clip ids (never the id of a clip with takes or of a stale clip) + live clips to retire + old clips."""
    existing = {c.id: c for c in project.list_clips(shot_id)}
    reserved: set[int] = set()
    for c in existing.values():
        if c.status == ClipStatus.STALE or _has_takes(project, c):
            reserved.add(int(c.id.rsplit("_C", 1)[1]))
    clips_dir = project.shot_dir(shot_id) / "clips"
    if clips_dir.is_dir():  # orphan clip folders that still hold takes are reserved too
        pattern = re.compile(re.escape(shot_id) + r"_C(\d{2,3})")
        for d in clips_dir.iterdir():
            m = pattern.fullmatch(d.name)
            if m and d.name not in existing and (d / "takes").is_dir() and any((d / "takes").iterdir()):
                reserved.add(int(m.group(1)))
    numbers: list[int] = []
    n_ = 1
    while len(numbers) < n:
        if n_ not in reserved:
            numbers.append(n_)
        n_ += 1
    ids = [make_clip_id(shot_id, x) for x in numbers]
    retired = [c for c in existing.values() if c.status != ClipStatus.STALE and c.id not in ids]
    return ids, retired, existing


def _performance(ctx: _Ctx, referenced: list[str]) -> str:
    perf = dict(ctx.scene.performance_states)
    first_beat = next((b for bid in ctx.shot.beat_ids for b in ctx.scene.beats if b.beat_id == bid), None)
    if first_beat:
        perf.update(first_beat.performance_states)
    return "; ".join(f"{eid}: {perf[eid]}" for eid in referenced if perf.get(eid))


def _stitch_risk(seg: _Segment, anchored: bool, transition: bool) -> RiskTier:
    score = 0
    if anchored:
        score += 1
        if transition:
            score += 2
        if _HIGH_MOTION_RX.search(seg.action or ""):
            score += 1
    if "line_exceeds_max_duration" in seg.flags:
        score += 1
    return RiskTier.HIGH if score >= 3 else RiskTier.MED if score >= 1 else RiskTier.LOW


def _conditioning(ctx: _Ctx, ref_states: list[tuple[Entity, EntityState]], lines: list[AudioLine]) -> Conditioning:
    supports = ctx.spec.supports
    cap = supports.max_reference_images if supports.reference_to_video else 0
    ordered = [pair for _, pair in sorted(enumerate(ref_states), key=lambda t: (TIER_RANK.get(t[1][0].tier, 9), t[0]))]
    groups = [(st.id, st.reference_pack.image_uris()) for _, st in ordered]
    anchor = ctx.environment.angle_uri(ctx.shot.environment_angle) if ctx.environment else None
    chosen, _ = allocate_reference_images(groups, cap, anchor=anchor)
    attached = [sid for sid, _ in groups if chosen.get(sid)]
    voice_audio: list[str] = []
    if supports.voice_reference_audio and ctx.settings.open_decisions.voice_reference_audio:
        for line in lines:
            vp = ctx.voices.get(line.character_id)
            if vp and vp.reference_audio and vp.reference_audio not in voice_audio:
                voice_audio.append(vp.reference_audio)
    return Conditioning(
        reference_uris=[u for sid, _ in groups for u in chosen.get(sid, [])],
        reference_entity_states=attached,
        spatial_anchor=anchor,
        identity_embeddings=[st.reference_pack.identity_embedding for _, st in ordered
                             if st.id in attached and st.reference_pack.identity_embedding],
        voice_reference_audio=voice_audio,
    )


def _build_clips(ctx: _Ctx, segments: list[_Segment], ids: list[str], previous: dict[str, GenClip]) -> list[GenClip]:
    clips: list[GenClip] = []
    prev_active: dict[str, str] | None = None
    for k, (seg, cid) in enumerate(zip(segments, ids), start=1):
        visual = ctx.shot.spec.visual.model_copy(deep=True)
        if seg.action:
            visual.subject_and_action = seg.action
        refs = ctx.refs(visual)
        active = active_state_map(ctx.scene, ctx.shot_order, ctx.shot.id, k, state_owner=ctx.state_owner,
                                  entity_ids=ctx.entities)
        flags = merge_unique(seg.flags, ["continues_action"] if seg.continues_action else [])
        dialogue: list[AudioLine] = []
        off_screen: list[AudioLine] = []
        for lid in seg.line_ids:
            ln = ctx.lines[lid]
            vp = ctx.voices.get(ln.character_id)
            audio = AudioLine(character_id=ln.character_id, text=ln.text, line_id=ln.line_id,
                              delivery=(ln.parenthetical or "").strip(),
                              voice=vp.voice_for_state(active.get(ln.character_id)) if vp else "",
                              transliteration=ln.transliteration)
            if ln.off_screen or ln.character_id not in refs.entities:
                off_screen.append(audio)
                if not ln.off_screen:
                    flags.append(f"speaker_out_of_frame:{lid}")
            else:
                dialogue.append(audio)
        referenced = list(refs.entities)
        for audio in dialogue:
            if audio.character_id in ctx.entities and audio.character_id not in referenced:
                referenced.append(audio.character_id)
        ref_states: list[tuple[Entity, EntityState]] = []
        for eid in referenced:
            sid = refs.explicit_states.get(eid) or active.get(eid)
            if sid and sid in ctx.states:
                ref_states.append(ctx.states[sid])
        anchored = bool(ctx.shot.continuous and k > 1)
        transition = prev_active is not None and any(active.get(e) != prev_active.get(e) for e in referenced)
        shot_audio = ctx.shot.spec.audio_spec
        ambience = (shot_audio.ambience or "").strip() or (ctx.environment.room_tone if ctx.environment else "")
        clips.append(GenClip(
            id=cid, shot_id=ctx.shot.id, scene_id=ctx.scene.id, sequence_index=k,
            duration_s=snap_duration(ctx.spec, seg.planned_s),
            spec=ShotSpec(
                visual=visual,
                audio_spec=AudioSpec(dialogue=dialogue, off_screen=off_screen, ambience=ambience,
                                     diegetic_sfx=list(shot_audio.diegetic_sfx), music="none"),
                negative=merge_unique(ctx.shot.spec.negative,
                                      required_negatives(ctx.style, [st for _, st in ref_states], ctx.audio_mode)),
            ),
            lines=list(seg.line_ids),
            performance_state=_performance(ctx, referenced),
            conditioning=_conditioning(ctx, ref_states, [*dialogue, *off_screen]),
            model_target=ctx.target.model_copy(),
            stitch_risk=_stitch_risk(seg, anchored, transition),
            anchor_from_clip_id=ids[k - 2] if anchored else None,
            environment_angle=ctx.shot.environment_angle,
            camera_side=ctx.shot.camera_side,
            status=ClipStatus.PENDING,
            flags=flags,
            version=previous[cid].version + 1 if cid in previous else 1,
            generated_by=generated_by(PROVENANCE_AGENT, model=ctx.target.preferred,
                                      notes=f"decomposed from {ctx.shot.id} v{ctx.shot.version}"),
        ))
        prev_active = active
    return clips


def scene_working_set_hash(project: Project, scene_id: str) -> str | None:
    """``working_set_hash`` of ``scenes/<SC>/working_set.yaml`` (None before a working set exists)."""
    path = project.scene_dir(scene_id) / "working_set.yaml"
    if not path.exists():
        return None
    data = read_record(path) or {}
    return data.get("working_set_hash") if isinstance(data, dict) else None


def decompose_shot(project: Project, registry: Registry, shot_id: str, *, model_id: str | None = None,
                   save: bool = True) -> list[GenClip]:
    """Decompose one CreativeShot into GenClips. ``save=False`` is a dry run (nothing is written)."""
    ids = parse_artifact_id(validate_id(shot_id, "shot id"))
    if ids["shot"] != shot_id:
        raise ValueError(f"{shot_id!r} is not a shot id (expected SC12_SH04)")
    ctx = _Ctx.build(project, registry, project.load_shot(shot_id), model_id)
    segments = _plan_segments(ctx)
    clip_ids, retired, previous = _allocate_ids(project, shot_id, len(segments))
    clips = _build_clips(ctx, segments, clip_ids, previous)
    if not save:
        return clips
    for old in retired:
        stale = old.model_copy(deep=True)
        stale.status = ClipStatus.STALE
        stale.flags = merge_unique(stale.flags, ["superseded_by_redecomposition"])
        project.save_clip(stale)
    for clip in clips:
        project.save_clip(clip)
    shot = ctx.shot.model_copy(deep=True)
    shot.gen_clips = [c.id for c in clips]
    project.save_shot(shot)
    spec = ctx.spec
    log_agent_call(
        project, agent=AGENT_NAME, scene_id=ctx.scene.id, working_set_hash=scene_working_set_hash(project, ctx.scene.id),
        task=f"decompose {shot_id}", mode="deterministic",
        inputs={"shot_id": shot_id, "shot_version": ctx.shot.version, "model_target": ctx.target.model_dump(),
                "registry": {"max_duration_s": spec.max_duration_s, "allowed_durations_s": spec.allowed_durations_s,
                             "max_speakers_per_clip": spec.supports.max_speakers_per_clip,
                             "max_reference_images": spec.supports.max_reference_images},
                "effective_max_speakers": ctx.max_speakers,
                "settings": ctx.settings.model_dump(include={
                    "speech_pace_wps", "speech_pre_roll_s", "speech_post_roll_s", "min_action_s", "min_clip_s",
                    "split_every_line", "single_speaker_clips"})},
        outputs={"clips": [{"id": c.id, "duration_s": c.duration_s, "lines": c.lines, "flags": c.flags,
                            "anchor_from_clip_id": c.anchor_from_clip_id, "stitch_risk": c.stitch_risk.value}
                           for c in clips],
                 "stale": [c.id for c in retired]},
    )
    return clips


def decompose_scene(project: Project, registry: Registry, scene_id: str, **kw) -> dict[str, list[GenClip]]:
    """Decompose every saved shot of the scene, in scene order."""
    scene = project.load_scene(validate_id(scene_id, "scene id"))
    out: dict[str, list[GenClip]] = {}
    for sid in scene_shot_order(project, scene):
        if (project.shot_dir(sid) / "shot.yaml").exists():
            out[sid] = decompose_shot(project, registry, sid, **kw)
    return out


def plan_anchor_waves(clips: Iterable[GenClip]) -> list[list[str]]:
    """Dispatch waves: wave 0 = unanchored clips (or anchored on a clip outside the set); wave n = clips anchored
    on a wave n-1 clip. Stale clips are ignored; anchor cycles raise ValueError."""
    active = [c for c in clips if c.status != ClipStatus.STALE]
    by_id = {c.id: c for c in active}
    depth: dict[str, int] = {}
    for clip in active:
        chain: list[str] = []
        cur: str | None = clip.id
        while cur is not None and cur not in depth:
            if cur in chain:
                raise ValueError(f"anchor cycle through {cur}")
            chain.append(cur)
            anchor = by_id[cur].anchor_from_clip_id
            cur = anchor if anchor in by_id else None
        base = depth[cur] + 1 if cur is not None else 0
        for offset, cid in enumerate(reversed(chain)):
            depth[cid] = base + offset
    waves: list[list[str]] = [[] for _ in range(max(depth.values(), default=-1) + 1)]
    for clip in active:
        waves[depth[clip.id]].append(clip.id)
    return waves


def link_anchor_frame(project: Project, clip: GenClip) -> GenClip:
    """Copy of ``clip`` whose first-frame anchor is the last frame of its anchor clip's SELECTED take (unchanged
    copy when the clip is unanchored or the anchor has no selected take with a last frame yet)."""
    out = clip.model_copy(deep=True)
    if not clip.anchor_from_clip_id:
        return out
    anchor = project.load_clip(clip.anchor_from_clip_id)
    if not anchor.selected_take_id:
        return out
    take = project.load_take(anchor.selected_take_id)
    if take.last_frame_uri:
        out.conditioning.first_frame_anchor = take.last_frame_uri
        out.conditioning.first_frame_anchor_take_id = take.id
    return out
