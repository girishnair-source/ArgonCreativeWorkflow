"""Editor — scene assembly into a hand-written OpenTimelineIO cut (BUILD_SPEC_v3 §6 Post, §10.2, §10.3).

``assemble_scene`` lays the selected take of every active clip end-to-end on **V1** (native AV: picture plus its
dialogue / ambience, whose audio is linked on **A1–A2**), puts the next-best take of each clip as a DISABLED
alternate on **V2** (or the shot's board frame when there is no alternate), the scene's SFX-supplement stems
(``scenes/<SC>/audio/sfx_supplement/*.wav``) on **A3–A4**, leaves **A5–A6** empty (reserved for stem separation /
the human) and lays the music cues (``scenes/<SC>/audio/cues/*.wav``) on **A7–A8**. Positions are frames at the
project fps. The cut is written as ``scenes/<SC>/assembly/scene_cut_v<n>.otio`` — plain OTIO JSON written by hand
(``Timeline.1 / Stack.1 / Track.1 / Clip.1 / Gap.1 / ExternalReference.1 / Marker.2``; no new dependency) — as an
immutable artifact (``next_version_path`` + ``write_new``). The full ``SceneCut`` record rides along in the
timeline metadata (``metadata.agy.scene_cut``) so the bridge and the round-trip can reload it.

Takes live at ``takes/<TAKE_ID>/take.mp4``; Resolve names clips after their file, so every take that goes to Resolve
is exposed under its own name through a hard link ``scenes/<SC>/assembly/media/<TAKE_ID>.mp4`` (a copy only when a
link is impossible). Take folders are never touched.

``assemble_master`` writes ``post/master_v<n>.otio`` from HUMAN-APPROVED scenes only (G6 passed / scene done): one
entry per scene pointing at the human's handed-off timeline, which Resolve nests unchanged (§10.4.8).

The ``handoff_snapshot.json`` helpers live here too (assembly-folder records shared by the bridge and round-trip).
"""

from __future__ import annotations

import os
import re
import shutil
import wave
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import Field

from .continuity import entity_id_of_state, speakers_in_clip
from .fsio import (
    ImmutableArtifactError,
    PathSecurityError,
    generated_by,
    next_version_path,
    now_iso,
    read_json,
    read_record,
    save_versioned,
    sha256_file,
    validate_id,
    write_new,
)
from .models import (
    ClipStatus,
    CreativeShot,
    DispatchStatus,
    GeneratedBy,
    GenClip,
    MarkerColor,
    Model,
    ProjectSettings,
    Recommendation,
    RiskTier,
    Scene,
    SceneStatus,
    Take,
    TakeStatus,
)
from .project import Project, parse_artifact_id
from .resolve_db import TimelineSnapshot
from .state import StateStore

AGENT = "agy.editor"
TRACK_LAYOUT: dict[str, str] = {
    "V1": "picture - native AV clips (selected takes)",
    "V2": "alternates / boards (disabled)",
    "A1": "native audio of V1 (linked; dialogue + ambience)",
    "A2": "native audio pair partner",
    "A3": "SFX supplement",
    "A4": "SFX supplement (overlaps)",
    "A5": "reserved - stem separation / human",
    "A6": "reserved - stem separation / human",
    "A7": "music cues",
    "A8": "music cues (overlaps)",
}
VIDEO_TRACKS = ("V1", "V2")
AUDIO_TRACKS = tuple(f"A{i}" for i in range(1, 9))
OTIO_COLORS = {
    MarkerColor.RED: "RED", MarkerColor.YELLOW: "YELLOW", MarkerColor.BLUE: "BLUE",
    MarkerColor.GREEN: "GREEN", MarkerColor.PURPLE: "PURPLE", MarkerColor.ORANGE: "ORANGE",
}
_MARKER_PRIORITY = (MarkerColor.RED, MarkerColor.YELLOW, MarkerColor.PURPLE, MarkerColor.BLUE, MarkerColor.GREEN,
                    MarkerColor.ORANGE)
TAKE_ID_RE = re.compile(r"(SC\d{2,4}_SH\d{2,3}_C\d{2,3}_T\d{2,3})")
CLIP_PREFIX_RE = re.compile(r"^(SC\d{2,4}_SH\d{2,3}_C\d{2,3})")
HANDOFF_SNAPSHOT = "handoff_snapshot.json"
STILL_EXTS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})
AUDIO_EXTS = frozenset({".wav"})


# ---------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------
class CutMarker(Model):
    frame: int = 0  # frames from the timeline start
    color: MarkerColor = MarkerColor.YELLOW
    name: str
    note: str = ""
    duration: int = 1
    custom_data: str = ""


class CutClip(Model):
    key: str  # unique within the cut, e.g. "V1:SC01_SH01_C01"
    track: str  # V1 | V2 | A1..A8
    kind: Literal["av", "video", "audio", "still"]
    role: Literal["picture", "native_audio", "alternate", "board", "sfx", "music"]
    name: str  # file name as Resolve shows it
    media: str  # project-relative POSIX path of the file Resolve imports
    source_media: str | None = None  # original take media when ``media`` is a staged link
    scene_id: str
    shot_id: str | None = None
    clip_id: str | None = None
    take_id: str | None = None
    source_in: int = 0
    duration: int
    record_in: int  # frames from the timeline start
    enabled: bool = True
    linked_to: str | None = None  # native audio -> key of its V1 picture clip
    metadata: dict[str, str] = Field(default_factory=dict)  # Scene, Shot, Take, Comments (§10.3)
    keywords: list[str] = Field(default_factory=list)

    @property
    def record_out(self) -> int:
        return self.record_in + self.duration

    @property
    def source_out(self) -> int:
        return self.source_in + self.duration


class SceneCut(Model):
    scene_id: str
    slug: str = ""
    version: int = 0
    fps: int = 24
    width: int = 1080
    height: int = 1920
    start_frame: int = 86400  # 01:00:00:00
    duration: int = 0
    clips: list[CutClip] = Field(default_factory=list)
    markers: list[CutMarker] = Field(default_factory=list)  # timeline markers (§10.3 colours)
    tracks: dict[str, str] = Field(default_factory=lambda: dict(TRACK_LAYOUT))
    otio_path: str | None = None  # project-relative
    timeline_base: str = ""  # Resolve timeline base name; the bridge adds _v<n>
    warnings: list[str] = Field(default_factory=list)
    generated_by: GeneratedBy | None = None

    def track(self, name: str) -> list[CutClip]:
        return sorted((c for c in self.clips if c.track == name), key=lambda c: (c.record_in, c.key))

    def picture(self) -> list[CutClip]:
        return self.track("V1")


class MasterEntry(Model):
    scene_id: str
    timeline_name: str  # the human's handed-off timeline (nested unchanged)
    record_in: int
    duration: int
    scene_cut: str | None = None  # project-relative path of the scene's latest AGY cut (reference only)


class MasterCut(Model):
    version: int
    name: str  # MASTER_v<n> (the bridge still picks the next free Resolve name)
    fps: int = 24
    width: int = 1080
    height: int = 1920
    start_frame: int = 86400
    duration: int = 0
    entries: list[MasterEntry] = Field(default_factory=list)
    otio_path: str | None = None
    warnings: list[str] = Field(default_factory=list)
    generated_by: GeneratedBy | None = None


# ---------------------------------------------------------------------------------------------
# Media helpers
# ---------------------------------------------------------------------------------------------
def take_id_of(text: str | None) -> str | None:
    """First fully-qualified take id inside a clip name / file name / path."""
    m = TAKE_ID_RE.search(text or "")
    return m.group(1) if m else None


def _frames(seconds: float | None, fps: int) -> int:
    try:
        value = float(seconds or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    return max(1, int(round(value * fps)))


def media_duration_s(path: Path) -> float | None:
    """Duration of a media file: WAV via the stdlib, others via ffprobe when installed; None if unknown."""
    path = Path(path)
    if not path.is_file():
        return None
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as w:
                rate = w.getframerate()
                return (w.getnframes() / rate) if rate else None
        except (wave.Error, EOFError, OSError):
            return None
    if path.suffix.lower() in STILL_EXTS or not shutil.which("ffprobe"):
        return None
    try:
        from . import media  # lazy: numpy import only when probing real media

        value = media.duration_s(path)
    except Exception:  # noqa: BLE001 - placeholder / unreadable media: the caller falls back to the planned length
        return None
    return value if value and value > 0 else None


def take_media_path(project: Project, take: Take | None) -> Path | None:
    if take is None or not take.video_uri:
        return None
    try:
        path = project.abs(take.video_uri)
    except (PathSecurityError, ValueError):
        return None
    return path if path.is_file() else None


def _same_file(a: Path, b: Path) -> bool:
    try:
        if os.path.samefile(a, b):
            return True
        return a.stat().st_size == b.stat().st_size and sha256_file(a) == sha256_file(b)
    except OSError:
        return False


def stage_take_media(project: Project, take: Take | None) -> Path | None:
    """The take's media under a Resolve-friendly name ``<TAKE_ID>.<ext>`` (hard link; copy as fallback).

    Returns the original path when its file name already carries the take id. Never overwrites anything.
    """
    src = take_media_path(project, take)
    if src is None or take is None:
        return None
    if take.id in src.name:
        return src
    scene = parse_artifact_id(take.id)["scene"]
    dest_dir = project.path("scenes", scene, "assembly", "media")
    suffix = src.suffix.lower() or ".mp4"
    dest = dest_dir / f"{take.id}{suffix}"
    if dest.exists():
        if _same_file(src, dest):
            return dest
        dest = dest_dir / f"{take.id}_{sha256_file(src)[:8]}{suffix}"
        if dest.exists():
            if _same_file(src, dest):
                return dest
            raise ImmutableArtifactError(f"refusing to overwrite staged media {dest}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
    except OSError:
        tmp = dest_dir / f".{dest.name}.tmp"
        shutil.copy2(src, tmp)
        if dest.exists():
            raise ImmutableArtifactError(f"refusing to overwrite staged media {dest}")
        os.replace(tmp, dest)
    return dest


# ---------------------------------------------------------------------------------------------
# Clip / take selection
# ---------------------------------------------------------------------------------------------
def active_clips(project: Project, scene_id: str) -> list[tuple[CreativeShot, GenClip]]:
    """(shot, clip) pairs of the current decomposition in shot / clip order (stale clips excluded)."""
    scene = project.load_scene(scene_id)
    shots = project.list_shots(scene_id)
    order = {sid: i for i, sid in enumerate(scene.creative_shots)}
    shots.sort(key=lambda s: (order.get(s.id, len(order)), s.order, s.id))
    out: list[tuple[CreativeShot, GenClip]] = []
    for shot in shots:
        clips = project.list_clips(shot.id)
        if shot.gen_clips:
            by_id = {c.id: c for c in clips}
            ordered = [by_id[cid] for cid in shot.gen_clips if cid in by_id]
        else:
            ordered = [c for c in clips if c.status != ClipStatus.STALE]
        out.extend((shot, c) for c in ordered)
    return out


def selected_take(project: Project, clip: GenClip) -> Take | None:
    if clip.selected_take_id:
        try:
            return project.load_take(clip.selected_take_id)
        except FileNotFoundError:
            pass
    chosen = [t for t in project.list_takes(clip.id) if t.status == TakeStatus.SELECTED]
    return chosen[0] if chosen else None


def next_best_take(project: Project, clip: GenClip, selected: Take | None) -> Take | None:
    """Best remaining take (rank, then QA composite) that has media — the V2 alternate."""
    candidates = [
        t for t in project.list_takes(clip.id)
        if (selected is None or t.id != selected.id) and t.status != TakeStatus.REJECTED
        and t.dispatch_status not in (DispatchStatus.FAILED, DispatchStatus.FILTERED)
        and take_media_path(project, t) is not None
    ]
    if not candidates:
        return None

    def key(t: Take) -> tuple:
        composite = t.qa.composite if t.qa is not None and t.qa.composite is not None else -1.0
        return (t.rank if t.rank is not None else 10 ** 6, -composite, t.id)

    return sorted(candidates, key=key)[0]


def anchor_fanout(clips: Iterable[GenClip]) -> dict[str, int]:
    """clip id -> number of clips anchored on it, transitively (first-frame anchor chains)."""
    clips = list(clips)
    children: dict[str, list[str]] = {}
    for c in clips:
        if c.anchor_from_clip_id:
            children.setdefault(c.anchor_from_clip_id, []).append(c.id)
    out: dict[str, int] = {}
    for c in clips:
        seen: set[str] = set()
        stack = list(children.get(c.id, []))
        while stack:
            cid = stack.pop()
            if cid in seen:
                continue
            seen.add(cid)
            stack.extend(children.get(cid, []))
        out[c.id] = len(seen)
    return out


# ---------------------------------------------------------------------------------------------
# §10.3 metadata, keywords, markers
# ---------------------------------------------------------------------------------------------
def _comments(clip: GenClip | None, take: Take) -> str:
    parts: list[str] = []
    qa = take.qa
    if qa is not None:
        parts.append(f"QA {qa.composite:.2f} {qa.recommendation.value}" if qa.composite is not None
                     else f"QA {qa.recommendation.value}")
        if qa.hard_reject_reasons:
            parts.append("reject: " + ", ".join(qa.hard_reject_reasons))
        sims = [f"{k} {v:.2f}" for k, v in sorted(qa.speaker_similarity.items()) if v is not None]
        parts.append("speaker-sim: " + (", ".join(sims) if sims else "n/a"))
        if qa.transcript_match is not None:
            parts.append(f"transcript match {qa.transcript_match:.2f}")
    if take.transcript:
        parts.append(f"transcript: {take.transcript[:160]}")
    if clip is not None:
        parts.append(f"stitch_risk: {clip.stitch_risk.value}")
    return " | ".join(parts)


def take_metadata(project: Project, take: Take, *, clip: GenClip | None = None, shot: CreativeShot | None = None,
                  fanout: dict[str, int] | None = None,
                  entity_ids: Iterable[str] | None = None) -> tuple[dict[str, str], list[str]]:
    """Resolve clip metadata ``Scene, Shot, Take, Comments`` + keywords (§10.3) for one take."""
    del shot  # shot-level data is already carried by the clip id
    ids = parse_artifact_id(take.id)
    if clip is None:
        try:
            clip = project.load_clip(take.clip_id)
        except FileNotFoundError:
            clip = None
    short = {k: (v.split("_")[-1] if v else "") for k, v in ids.items()}
    meta = {
        "Scene": ids["scene"] or "",
        "Shot": "_".join(x for x in (short["shot"], short["clip"]) if x),
        "Take": short["take"] or take.id,
        "Comments": _comments(clip, take),
    }
    keywords = [take.clip_id, take.id]
    if clip is not None:
        known = list(entity_ids) if entity_ids is not None else [e.id for e in project.list_entities()]
        keywords.append(f"risk:{clip.stitch_risk.value}")
        keywords.append(f"anchor_fanout:{(fanout or {}).get(clip.id, 0)}")
        for state_id in clip.conditioning.reference_entity_states:
            ent = entity_id_of_state(state_id, known) or state_id.split("_STATE_")[0]
            keywords.append(f"entity:{ent}")
        keywords.extend(f"speakers:{s}" for s in speakers_in_clip(clip))
    keywords.append(f"model:{take.model}")
    if take.seed is not None:
        keywords.append(f"seed:{take.seed}")
    return meta, list(dict.fromkeys(k for k in keywords if k))


def _clip_flags(scene: Scene, shot: CreativeShot, clip: GenClip, take: Take, settings: ProjectSettings,
                selected_by_clip: dict[str, str | None]) -> list[tuple[MarkerColor, str, str]]:
    out: list[tuple[MarkerColor, str, str]] = []
    qa = take.qa
    if qa is not None and (qa.recommendation == Recommendation.REJECT or qa.hard_reject_reasons):
        out.append((MarkerColor.RED, "QA_REJECT",
                    "QA reject: " + (", ".join(qa.hard_reject_reasons) or "composite below threshold")))
    if clip.stitch_risk == RiskTier.HIGH:
        out.append((MarkerColor.YELLOW, "HIGH_STITCH_RISK", "high stitch risk at this cut"))
    for tr in scene.mid_scene_transitions:
        if tr.at_shot_id == shot.id and int(tr.at_clip_index) == int(clip.sequence_index):
            out.append((MarkerColor.YELLOW, "STATE_TRANSITION",
                        f"{tr.entity_id}: {tr.from_state} -> {tr.to_state}" + (f" ({tr.note})" if tr.note else "")))
    if qa is not None:
        low = [f"{k} {v:.2f}" for k, v in sorted(qa.speaker_similarity.items())
               if v is not None and v < settings.qa.speaker_similarity_min]
        if low:
            out.append((MarkerColor.YELLOW, "VOICE_DRIFT", "voice-drift warning: " + ", ".join(low)))
        if qa.composite is not None and qa.recommendation == Recommendation.REVIEW:
            out.append((MarkerColor.YELLOW, "REVIEW", f"QA asks for a review (composite {qa.composite:.2f})"))
    anchor_take = clip.conditioning.first_frame_anchor_take_id
    anchor_sel = selected_by_clip.get(clip.anchor_from_clip_id or "")
    if clip.anchor_from_clip_id and anchor_take and anchor_sel and anchor_sel != anchor_take:
        out.append((MarkerColor.PURPLE, "ANCHOR_CONTINUITY",
                    f"generated from {anchor_take}'s last frame but {anchor_sel} is the selected take"))
    if "recommend_cutaway" in clip.flags:
        out.append((MarkerColor.PURPLE, "CONTINUITY_NOTE", "the decomposer recommends a cutaway here"))
    return out


def _merge_flags(flags: list[tuple[MarkerColor, str, str]], clip_id: str, frame: int) -> CutMarker:
    """Resolve allows one marker per frame: one marker per clip, highest-priority colour wins."""
    ranked = sorted(flags, key=lambda f: _MARKER_PRIORITY.index(f[0]))
    color, label, _ = ranked[0]
    return CutMarker(frame=frame, color=color, name=f"{label} {clip_id}", note=" | ".join(f[2] for f in ranked),
                     duration=1, custom_data=f"agy:{clip_id}")


# ---------------------------------------------------------------------------------------------
# OTIO JSON (hand-written)
# ---------------------------------------------------------------------------------------------
def _rt(value: float, rate: float) -> dict[str, Any]:
    return {"OTIO_SCHEMA": "RationalTime.1", "rate": float(rate), "value": float(value)}


def _tr(start: float, duration: float, rate: float) -> dict[str, Any]:
    return {"OTIO_SCHEMA": "TimeRange.1", "duration": _rt(duration, rate), "start_time": _rt(start, rate)}


def _otio_marker(m: CutMarker, rate: float) -> dict[str, Any]:
    return {"OTIO_SCHEMA": "Marker.2", "color": OTIO_COLORS.get(m.color, "YELLOW"), "comment": m.note,
            "marked_range": _tr(m.frame, max(1, m.duration), rate),
            "metadata": {"agy": {"custom_data": m.custom_data, "resolve_color": m.color.value}}, "name": m.name}


def _gap(duration: int, rate: float) -> dict[str, Any]:
    return {"OTIO_SCHEMA": "Gap.1", "effects": [], "enabled": True, "markers": [], "metadata": {}, "name": "",
            "source_range": _tr(0, duration, rate)}


def _otio_clip(project: Project, c: CutClip, rate: float) -> dict[str, Any]:
    ref = {"OTIO_SCHEMA": "ExternalReference.1", "available_range": None,
           "metadata": {"agy": {"rel": c.media, "source_media": c.source_media}}, "name": c.name,
           "target_url": project.abs(c.media).as_uri()}
    meta = {"key": c.key, "role": c.role, "kind": c.kind, "scene_id": c.scene_id, "shot_id": c.shot_id,
            "clip_id": c.clip_id, "take_id": c.take_id, "linked_to": c.linked_to, "metadata": c.metadata,
            "keywords": c.keywords}
    return {"OTIO_SCHEMA": "Clip.1", "effects": [], "enabled": bool(c.enabled), "markers": [],
            "media_reference": ref, "metadata": {"agy": meta}, "name": c.name,
            "source_range": _tr(c.source_in, c.duration, rate)}


def _otio_track(project: Project, name: str, clips: list[CutClip], rate: float) -> dict[str, Any]:
    children: list[dict[str, Any]] = []
    cursor = 0
    for c in sorted(clips, key=lambda x: (x.record_in, x.key)):
        if c.record_in < cursor:
            raise ValueError(f"track {name}: {c.key} overlaps the previous clip")
        if c.record_in > cursor:
            children.append(_gap(c.record_in - cursor, rate))
        children.append(_otio_clip(project, c, rate))
        cursor = c.record_out
    return {"OTIO_SCHEMA": "Track.1", "children": children, "effects": [], "enabled": True,
            "kind": "Video" if name.startswith("V") else "Audio", "markers": [],
            "metadata": {"agy": {"role": TRACK_LAYOUT.get(name, "")}}, "name": name, "source_range": None}


def scene_cut_to_otio(project: Project, cut: SceneCut) -> dict[str, Any]:
    rate = float(cut.fps)
    tracks = [_otio_track(project, name, cut.track(name), rate) for name in (*VIDEO_TRACKS, *AUDIO_TRACKS)]
    return {
        "OTIO_SCHEMA": "Timeline.1",
        "metadata": {"agy": {
            "kind": "scene_cut", "scene_id": cut.scene_id, "version": cut.version, "track_layout": cut.tracks,
            "scene_cut": cut.model_dump(mode="json"),
            "generated_by": cut.generated_by.model_dump(mode="json") if cut.generated_by else None,
        }},
        "name": f"{cut.scene_id}_cut_v{cut.version}",
        "global_start_time": _rt(cut.start_frame, rate),
        "tracks": {"OTIO_SCHEMA": "Stack.1", "children": tracks, "effects": [], "enabled": True,
                   "markers": [_otio_marker(m, rate) for m in cut.markers], "metadata": {}, "name": "tracks",
                   "source_range": None},
    }


def _contained(project: Project, path: str | Path) -> Path:
    p = Path(path)
    p = p if p.is_absolute() else project.abs(str(path))
    return project.abs(project.rel(p))  # raises PathSecurityError outside the project


def load_scene_cut(project: Project, path: str | Path) -> SceneCut:
    data = read_json(_contained(project, path))
    return SceneCut.model_validate(data["metadata"]["agy"]["scene_cut"])


def latest_scene_cut(project: Project, scene_id: str) -> Path | None:
    directory = project.path("scenes", validate_id(scene_id, "scene id"), "assembly")
    best, best_v = None, 0
    if directory.is_dir():
        for p in directory.glob("scene_cut_v*.otio"):
            m = re.fullmatch(r"scene_cut_v(\d+)\.otio", p.name)
            if m and int(m.group(1)) > best_v:
                best, best_v = p, int(m.group(1))
    return best


# ---------------------------------------------------------------------------------------------
# Scene assembly
# ---------------------------------------------------------------------------------------------
def _board_still(project: Project, shot: CreativeShot) -> Path | None:
    if not shot.board_frame:
        return None
    try:
        p = project.abs(shot.board_frame)
    except (PathSecurityError, ValueError):
        return None
    return p if p.is_file() and p.suffix.lower() in STILL_EXTS else None


def _audio_files(project: Project, scene_id: str, sub: str) -> list[Path]:
    d = project.path("scenes", scene_id, "audio", sub)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir()
                  if p.is_file() and not p.is_symlink() and p.suffix.lower() in AUDIO_EXTS and not p.name.startswith("."))


def _cue_plan(project: Project, scene_id: str) -> dict[str, dict[str, Any]]:
    p = project.path("scenes", scene_id, "audio", "cue_plan.yaml")
    if not p.is_file():
        return {}
    data = read_record(p) or {}
    out: dict[str, dict[str, Any]] = {}
    for cue in data.get("cues") or []:
        if isinstance(cue, dict) and cue.get("cue_id"):
            out[str(cue["cue_id"])] = cue
    return out


def _place_on_pair(placed: dict[str, list[tuple[int, int]]], pair: tuple[str, str], start: int,
                   duration: int) -> tuple[str, int]:
    """First track of the pair that is free at [start, start+duration); overflow goes after the last clip."""
    for track in pair:
        spans = placed.setdefault(track, [])
        if all(start >= e or start + duration <= s for s, e in spans):
            spans.append((start, start + duration))
            return track, start
    track = pair[-1]
    start = max([e for _, e in placed[track]] + [start])
    placed[track].append((start, start + duration))
    return track, start


def assemble_scene(project: Project, scene_id: str) -> SceneCut:
    """Selected takes in shot / clip order -> ``SceneCut`` + ``scenes/<SC>/assembly/scene_cut_v<n>.otio``."""
    validate_id(scene_id, "scene id")
    scene = project.load_scene(scene_id)
    settings = project.config.settings
    fps = int(settings.fps or 24)
    pairs = active_clips(project, scene_id)
    fanout = anchor_fanout(c for _, c in pairs)
    selected_by_clip: dict[str, str | None] = {c.id: c.selected_take_id for _, c in pairs}
    entity_ids = [e.id for e in project.list_entities()]
    clips: list[CutClip] = []
    markers: list[CutMarker] = []
    warnings: list[str] = []
    rec = 0
    for shot, clip in pairs:
        take = selected_take(project, clip)
        if take is None:
            warnings.append(f"{clip.id}: no selected take yet - left out of the cut")
            continue
        media = stage_take_media(project, take)
        if media is None:
            warnings.append(f"{clip.id}: selected take {take.id} has no media file - left out of the cut")
            continue
        selected_by_clip[clip.id] = take.id
        dur = _frames(media_duration_s(media) or clip.duration_s, fps)
        meta, kws = take_metadata(project, take, clip=clip, shot=shot, fanout=fanout, entity_ids=entity_ids)
        common = {"scene_id": scene_id, "shot_id": shot.id, "clip_id": clip.id, "source_in": 0, "duration": dur,
                  "record_in": rec, "metadata": meta, "keywords": kws, "take_id": take.id,
                  "source_media": take.video_uri, "name": media.name, "media": project.rel(media)}
        picture = CutClip(key=f"V1:{clip.id}", track="V1", kind="av", role="picture", **common)
        clips.append(picture)
        clips.append(CutClip(key=f"A1:{clip.id}", track="A1", kind="audio", role="native_audio",
                             linked_to=picture.key, **common))
        alt = next_best_take(project, clip, take)
        alt_media = stage_take_media(project, alt) if alt is not None else None
        if alt is not None and alt_media is not None:
            alt_dur = min(dur, _frames(media_duration_s(alt_media) or clip.duration_s, fps))
            alt_meta, alt_kws = take_metadata(project, alt, clip=clip, shot=shot, fanout=fanout, entity_ids=entity_ids)
            clips.append(CutClip(key=f"V2:{clip.id}", track="V2", kind="video", role="alternate", name=alt_media.name,
                                 media=project.rel(alt_media), source_media=alt.video_uri, scene_id=scene_id,
                                 shot_id=shot.id, clip_id=clip.id, take_id=alt.id, source_in=0, duration=alt_dur,
                                 record_in=rec, enabled=False, metadata=alt_meta, keywords=alt_kws))
        else:
            board = _board_still(project, shot)
            if board is not None:
                clips.append(CutClip(key=f"V2:{clip.id}", track="V2", kind="still", role="board", name=board.name,
                                     media=project.rel(board), scene_id=scene_id, shot_id=shot.id, clip_id=clip.id,
                                     source_in=0, duration=dur, record_in=rec, enabled=False,
                                     metadata={"Scene": scene_id, "Comments": f"board frame for {clip.id}"},
                                     keywords=[clip.id, "board"]))
        flags = _clip_flags(scene, shot, clip, take, settings, selected_by_clip)
        if flags:
            markers.append(_merge_flags(flags, clip.id, rec))
        rec += dur
    pictures = [c for c in clips if c.track == "V1"]
    if not pictures:
        raise ValueError(f"{scene_id}: no clip has a selected take with media yet - nothing to assemble")
    total = rec
    placed: dict[str, list[tuple[int, int]]] = {}
    clip_rec = {c.clip_id: c.record_in for c in pictures}
    for p in _audio_files(project, scene_id, "sfx_supplement"):  # A3-A4: anchored on <CLIP_ID>_*.wav
        m = CLIP_PREFIX_RE.match(p.stem)
        anchor = m.group(1) if m else None
        if anchor and anchor not in clip_rec:
            warnings.append(f"SFX {p.name}: {anchor} is not in the cut - placed at the start")
        dur = _frames(media_duration_s(p) or 2.0, fps)
        track, start = _place_on_pair(placed, ("A3", "A4"), clip_rec.get(anchor, 0) if anchor else 0, dur)
        clips.append(CutClip(key=f"{track}:{p.stem}", track=track, kind="audio", role="sfx", name=p.name,
                             media=project.rel(p), scene_id=scene_id, clip_id=anchor, source_in=0, duration=dur,
                             record_in=start, metadata={"Scene": scene_id, "Comments": "SFX supplement (hard effect)"},
                             keywords=["sfx_supplement", *([anchor] if anchor else [])]))
    plan = _cue_plan(project, scene_id)
    cursor = 0
    for p in _audio_files(project, scene_id, "cues"):  # A7-A8: cue_plan start_s, else back to back
        entry = plan.get(p.stem)
        dur = _frames(media_duration_s(p) or (entry or {}).get("duration_s") or total / fps, fps)
        start = int(round(float((entry or {}).get("start_s") or 0) * fps)) if entry else cursor
        if start >= total:
            warnings.append(f"music cue {p.name} starts after the picture ends - not placed")
            continue
        dur = max(1, min(dur, total - start))
        track, start = _place_on_pair(placed, ("A7", "A8"), start, dur)
        if entry is None:
            cursor = start + dur
        clips.append(CutClip(key=f"{track}:{p.stem}", track=track, kind="audio", role="music", name=p.name,
                             media=project.rel(p), scene_id=scene_id, source_in=0, duration=dur, record_in=start,
                             metadata={"Scene": scene_id, "Comments": f"music cue {p.stem}"},
                             keywords=["music", p.stem]))
    out_dir = project.path("scenes", scene_id, "assembly")
    path = next_version_path(out_dir, "scene_cut", ".otio")
    version = int(re.search(r"_v(\d+)\.otio$", path.name).group(1))  # type: ignore[union-attr]
    cut = SceneCut(scene_id=scene_id, slug=scene.slug, version=version, fps=fps, width=int(settings.width),
                   height=int(settings.height), start_frame=fps * 3600, duration=total, clips=clips, markers=markers,
                   otio_path=project.rel(path), timeline_base=scene_id, warnings=warnings,
                   generated_by=generated_by(AGENT, notes=f"{len(pictures)} picture clips, {total} frames"))
    write_new(path, scene_cut_to_otio(project, cut))
    return cut


# ---------------------------------------------------------------------------------------------
# handoff_snapshot.json (assembly-folder record shared by the bridge and the round-trip)
# ---------------------------------------------------------------------------------------------
def handoff_snapshot_path(project: Project, scene_id: str) -> Path:
    return project.path("scenes", validate_id(scene_id, "scene id"), "assembly", HANDOFF_SNAPSHOT)


def read_handoff_snapshot(project: Project, scene_id: str) -> dict[str, Any] | None:
    p = handoff_snapshot_path(project, scene_id)
    if not p.is_file():
        return None
    data = read_json(p)
    if not isinstance(data, dict) or not isinstance(data.get("timelines"), dict):
        return None
    return data


def snapshot_entry(project: Project, scene_id: str, timeline_name: str) -> TimelineSnapshot | None:
    """The recorded (handoff-time) snapshot of one timeline, or None when absent / still pending."""
    data = read_handoff_snapshot(project, scene_id) or {}
    entry = (data.get("timelines") or {}).get(timeline_name)
    if not isinstance(entry, dict) or entry.get("pending") or not entry.get("snapshot"):
        return None
    return TimelineSnapshot.model_validate(entry["snapshot"])


def record_timeline_snapshot(project: Project, scene_id: str, timeline_name: str, snap: TimelineSnapshot | None,
                             *, agent: str = AGENT, extra: dict[str, Any] | None = None) -> Path:
    """Add / replace one timeline's baseline in ``handoff_snapshot.json`` (previous file archived)."""
    path = handoff_snapshot_path(project, scene_id)
    data = read_handoff_snapshot(project, scene_id) or {"scene_id": scene_id, "timelines": {}}
    entry: dict[str, Any] = {"timeline_name": timeline_name, "recorded_at": now_iso(), "pending": snap is None,
                             "fingerprint": snap.fingerprint if snap is not None else None,
                             "snapshot": snap.model_dump(mode="json") if snap is not None else None}
    entry.update(extra or {})
    data["timelines"][timeline_name] = entry
    data["generated_by"] = generated_by(agent).model_dump(mode="json")
    return save_versioned(path, data)


# ---------------------------------------------------------------------------------------------
# Master (human-approved scenes only)
# ---------------------------------------------------------------------------------------------
def is_scene_approved(project: Project, scene_id: str, state: StateStore | None = None) -> bool:
    """G6 passed (Resolve APPROVED marker / chat approve) or the scene is already approved / done."""
    scene = project.load_scene(scene_id)
    if scene.status in (SceneStatus.HUMAN_APPROVED, SceneStatus.DONE):
        return True
    try:
        return (state or StateStore(project)).gate_status(scene_id, "G6") == "passed"
    except Exception:  # noqa: BLE001 - unreadable state never counts as an approval
        return False


def _snapshot_duration(snap: TimelineSnapshot | None) -> int:
    if snap is None or not snap.items:
        return 0
    base = snap.base_frame()
    return max(0, int(round(max(float(i.start) + float(i.duration) for i in snap.items) - base)))


def _timeline_duration(project: Project, scene_id: str, timeline_name: str, db: Any = None) -> int:
    if db is not None:
        try:
            duration = _snapshot_duration(db.timeline(timeline_name))
        except Exception:  # noqa: BLE001 - live DB unreadable: fall back to the recorded snapshot
            duration = 0
        if duration:
            return duration
    duration = _snapshot_duration(snapshot_entry(project, scene_id, timeline_name))
    if duration:
        return duration
    cut_path = latest_scene_cut(project, scene_id)
    return load_scene_cut(project, cut_path).duration if cut_path else 0


def master_to_otio(project: Project, mc: MasterCut) -> dict[str, Any]:
    rate = float(mc.fps)
    children = []
    for e in mc.entries:
        if e.scene_cut:
            ref: dict[str, Any] = {"OTIO_SCHEMA": "ExternalReference.1", "available_range": None,
                                   "metadata": {"agy": {"rel": e.scene_cut}}, "name": e.timeline_name,
                                   "target_url": project.abs(e.scene_cut).as_uri()}
        else:
            ref = {"OTIO_SCHEMA": "MissingReference.1", "available_range": None, "metadata": {},
                   "name": e.timeline_name}
        children.append({"OTIO_SCHEMA": "Clip.1", "effects": [], "enabled": True, "markers": [],
                         "media_reference": ref,
                         "metadata": {"agy": {"scene_id": e.scene_id, "resolve_timeline": e.timeline_name,
                                              "nested": True}},
                         "name": e.timeline_name, "source_range": _tr(0, e.duration, rate)})
    track = {"OTIO_SCHEMA": "Track.1", "children": children, "effects": [], "enabled": True, "kind": "Video",
             "markers": [], "metadata": {"agy": {"role": "nested human-approved scene timelines (picture + audio)"}},
             "name": "V1", "source_range": None}
    return {"OTIO_SCHEMA": "Timeline.1",
            "metadata": {"agy": {"kind": "master", "master_cut": mc.model_dump(mode="json")}},
            "name": mc.name, "global_start_time": _rt(mc.start_frame, rate),
            "tracks": {"OTIO_SCHEMA": "Stack.1", "children": [track], "effects": [], "enabled": True, "markers": [],
                       "metadata": {}, "name": "tracks", "source_range": None}}


def load_master_cut(project: Project, path: str | Path) -> MasterCut:
    data = read_json(_contained(project, path))
    return MasterCut.model_validate(data["metadata"]["agy"]["master_cut"])


def assemble_master(project: Project, scene_ids: Iterable[str], *, db: Any = None) -> Path:
    """``post/master_v<n>.otio`` nesting the human's approved scene timelines, in the given order."""
    ids = list(dict.fromkeys(validate_id(s, "scene id") for s in scene_ids))
    if not ids:
        raise ValueError("no scenes were given for the master")
    state = StateStore(project)
    unapproved = [s for s in ids if not is_scene_approved(project, s, state)]
    if unapproved:
        raise ValueError(f"The master only uses scenes you approved; not approved yet: {', '.join(unapproved)} "
                         f"(add an APPROVED marker in Resolve or say `approve {unapproved[0]}`).")
    settings = project.config.settings
    fps = int(settings.fps or 24)
    entries: list[MasterEntry] = []
    warnings: list[str] = []
    rec = 0
    for sid in ids:
        handoff = project.load_handoff(sid)
        if handoff is None or not handoff.timeline_name:
            raise ValueError(f"{sid} has no Resolve timeline yet - nothing to nest in the master")
        duration = _timeline_duration(project, sid, handoff.timeline_name, db)
        if duration <= 0:
            warnings.append(f"{handoff.timeline_name}: length unknown - 1 s placeholder in the OTIO (Resolve nests "
                            f"the real timeline)")
            duration = fps
        cut_path = latest_scene_cut(project, sid)
        entries.append(MasterEntry(scene_id=sid, timeline_name=handoff.timeline_name, record_in=rec,
                                   duration=duration, scene_cut=project.rel(cut_path) if cut_path else None))
        rec += duration
    path = next_version_path(project.path("post"), "master", ".otio")
    version = int(re.search(r"_v(\d+)\.otio$", path.name).group(1))  # type: ignore[union-attr]
    mc = MasterCut(version=version, name=f"MASTER_v{version}", fps=fps, width=int(settings.width),
                   height=int(settings.height), start_frame=fps * 3600, duration=rec, entries=entries,
                   otio_path=project.rel(path), warnings=warnings,
                   generated_by=generated_by(AGENT, notes=f"master from {', '.join(ids)}"))
    write_new(path, master_to_otio(project, mc))
    return path
