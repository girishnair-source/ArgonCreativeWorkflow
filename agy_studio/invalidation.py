"""Anchor graph & invalidation — BUILD_SPEC_v3 §4.7 (Workstream A).

| Edge                                   | On upstream change                                              |
|----------------------------------------|-----------------------------------------------------------------|
| GenClip -> previous Take.last_frame    | anchor descendants -> ``stale`` (ask before regenerating)        |
| GenClip -> Environment.derived_angle   | clips on that angle -> ``stale``                                 |
| GenClip -> EntityState.reference_pack  | clips using the pack -> ``stale``; Vision-tier confirm           |
| GenClip -> VoiceProfile.anchor_take    | that character's speaking clips -> ``needs_voice_review``; confirm|
| Scene -> Scene.exit_state              | next scene's working set -> ``needs_review``                     |
| CreativeShot -> ShotSpec               | child clips -> ``needs_relint``                                  |

:func:`apply_invalidation` only ever rewrites the clip/scene records listed in the fan-out (acceptance 6), never
downgrades a stronger status (stale > needs_relint > needs_voice_review), and logs ``logs/invalidation.jsonl``.
Fan-outs with ``requires_confirm`` need the id of a recorded Vision-tier confirm decision.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from pydantic import Field

from .continuity import speakers_in_clip
from .decisions import DecisionLog
from .fsio import append_jsonl, locked, now_iso, read_record, validate_id
from .models import ClipStatus, CreativeShot, DecisionAction, DecisionTier, GenClip, Model, Scene, SceneStatus, Take
from .project import Project, parse_artifact_id
from .state import ConfirmRequiredError, StateStore

FanOutAction = Literal["stale", "needs_voice_review", "needs_relint", "needs_review"]
FanOutKind = Literal["anchor", "environment_angle", "pack", "voice", "exit_state", "shot_spec"]
_STRENGTH: dict[ClipStatus, int] = {ClipStatus.STALE: 3, ClipStatus.NEEDS_RELINT: 2, ClipStatus.NEEDS_VOICE_REVIEW: 1}
_ACTION_STATUS: dict[str, ClipStatus] = {
    "stale": ClipStatus.STALE, "needs_relint": ClipStatus.NEEDS_RELINT,
    "needs_voice_review": ClipStatus.NEEDS_VOICE_REVIEW,
}
_NOT_A_CONFIRM = frozenset({DecisionAction.REJECT, DecisionAction.REQUEST_ALTERNATIVES})


class FanOut(Model):
    root: str
    kind: FanOutKind
    clips: list[str] = Field(default_factory=list)
    shots: list[str] = Field(default_factory=list)
    scenes: list[str] = Field(default_factory=list)
    takes: list[str] = Field(default_factory=list)
    action: FanOutAction
    requires_confirm: bool = False
    summary: str = ""


# ---------------------------------------------------------------------------------------------
# Tolerant, path-safe loaders (clips are found even before their shot.yaml exists)
# ---------------------------------------------------------------------------------------------
def _is_id(name: str) -> bool:
    try:
        validate_id(name)
    except ValueError:
        return False
    return True


def _child_dirs(base: Path, canonical: Callable[[str], Path]) -> list[Path]:
    """Sub-directories whose name is a valid id AND that sit exactly where ``canonical(name)`` says."""
    if not base.is_dir():
        return []
    out = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or not _is_id(d.name):
            continue
        try:
            if canonical(d.name) == d.resolve():
                out.append(d)
        except ValueError:  # malformed id / path escaping the project
            continue
    return out


def scene_clip_paths(project: Project, scene_id: str) -> list[Path]:
    paths = []
    for shot_dir in _child_dirs(project.scene_dir(scene_id) / "shots", project.shot_dir):
        for clip_dir in _child_dirs(shot_dir / "clips", project.clip_dir):
            if (clip_dir / "clip.yaml").is_file():
                paths.append(clip_dir / "clip.yaml")
    return paths


def load_scene_clips(project: Project, scene_id: str) -> list[GenClip]:
    clips = []
    for p in scene_clip_paths(project, scene_id):
        try:
            clip = GenClip.model_validate(read_record(p))
        except (OSError, ValueError):
            continue
        if clip.id == p.parent.name:
            clips.append(clip)
    return sorted(clips, key=lambda c: (c.shot_id, c.sequence_index, c.id))


def load_scene_shots(project: Project, scene_id: str) -> list[CreativeShot]:
    shots = []
    for shot_dir in _child_dirs(project.scene_dir(scene_id) / "shots", project.shot_dir):
        if (shot_dir / "shot.yaml").is_file():
            try:
                shots.append(CreativeShot.model_validate(read_record(shot_dir / "shot.yaml")))
            except (OSError, ValueError):
                continue
    return sorted(shots, key=lambda s: (s.order, s.id))


def load_clip_takes(project: Project, clip_id: str) -> list[Take]:
    takes = []
    for take_dir in _child_dirs(project.clip_dir(clip_id) / "takes", project.take_dir):
        if (take_dir / "take.yaml").is_file():
            try:
                takes.append(Take.model_validate(read_record(take_dir / "take.yaml")))
            except (OSError, ValueError):
                continue
    return sorted(takes, key=lambda t: t.id)


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _voice_character(project: Project, voice_id: str) -> str | None:
    for char, entry in (project.casting().get("voice_profiles") or {}).items():
        if not isinstance(entry, dict):
            continue
        if entry.get("voice_profile_id") == voice_id or (entry.get("profile") or {}).get("id") == voice_id:
            return str(char)
    return voice_id[len("VOICE_"):] if voice_id.startswith("VOICE_CHAR_") else None


def confirm_target(fanout: FanOut) -> str:
    """The artifact id the director types after ``confirm``."""
    return fanout.root.split(":", 1)[0] if fanout.kind == "environment_angle" else fanout.root


class _hybrid:
    """``AnchorGraph.build(project)`` (new graph) and ``graph.build(project)`` (rebuild in place) both work."""

    def __init__(self, func: Callable[..., Any]) -> None:
        self.func = func

    def __get__(self, obj: Any, cls: type) -> Callable[[Project], Any]:
        target = obj if obj is not None else cls()
        return lambda project: self.func(target, project)


# ---------------------------------------------------------------------------------------------
# Anchor graph
# ---------------------------------------------------------------------------------------------
class AnchorGraph:
    def __init__(self, project: Project | None = None) -> None:
        self.project: Project | None = None
        self.scenes: list[Scene] = []
        self.shots: dict[str, CreativeShot] = {}
        self.clips: dict[str, GenClip] = {}
        self.takes: dict[str, list[Take]] = {}
        self.children: dict[str, set[str]] = {}
        if project is not None:
            self.build(project)

    @_hybrid
    def build(self, project: Project) -> "AnchorGraph":
        self.project = project
        self.scenes = project.list_scenes()
        self.shots, self.clips, self.takes = {}, {}, {}
        for sc in self.scenes:
            self.shots.update({s.id: s for s in load_scene_shots(project, sc.id)})
            self.clips.update({c.id: c for c in load_scene_clips(project, sc.id)})
        self.takes = {cid: load_clip_takes(project, cid) for cid in self.clips}
        self.children = {cid: set() for cid in self.clips}
        for cid, clip in self.clips.items():
            parents = [clip.anchor_from_clip_id] if clip.anchor_from_clip_id else []
            anchor_take = clip.conditioning.first_frame_anchor_take_id
            if anchor_take:
                try:
                    parents.append(parse_artifact_id(anchor_take)["clip"])
                except ValueError:
                    pass
            for parent in parents:
                if parent in self.children and parent != cid:
                    self.children[parent].add(cid)
        for cid, takes in self.takes.items():
            for take in takes:
                for child in take.anchors_downstream:
                    if child in self.clips and child != cid:
                        self.children[cid].add(child)
        return self

    def _ordered(self, clip_ids: Iterable[str]) -> list[str]:
        def key(c: str) -> tuple[str, int, str]:
            clip = self.clips.get(c)
            return (clip.shot_id, clip.sequence_index, c) if clip else ("", 0, c)
        return sorted(set(clip_ids), key=key)

    def anchor_descendants(self, clip_id: str) -> list[str]:
        """Every clip whose first frame depends (transitively) on ``clip_id``'s approved take."""
        seen: set[str] = set()
        queue = deque(self.children.get(clip_id, ()))
        while queue:
            c = queue.popleft()
            if c in seen or c == clip_id:
                continue
            seen.add(c)
            queue.extend(self.children.get(c, ()))
        return self._ordered(seen)

    def _make(self, root: str, kind: str, clips: list[str], action: str, requires_confirm: bool, *,
              scenes: list[str] | None = None) -> FanOut:
        shots = list(dict.fromkeys(self.clips[c].shot_id for c in clips if c in self.clips))
        if scenes is None:
            scenes = list(dict.fromkeys(self.clips[c].scene_id for c in clips if c in self.clips))
        takes = [t.id for c in clips for t in self.takes.get(c, [])] if action in ("stale", "needs_voice_review") else []
        fo = FanOut(root=root, kind=kind, clips=clips, shots=shots, scenes=scenes, takes=takes, action=action,
                    requires_confirm=requires_confirm)  # type: ignore[arg-type]
        fo.summary = describe_fanout(fo)
        return fo

    def _require_project(self) -> Project:
        if self.project is None:
            raise RuntimeError("AnchorGraph.build(project) first")
        return self.project

    def fanout(self, artifact_id: str) -> FanOut:
        aid = str(artifact_id or "").strip()
        try:
            ids: dict[str, str | None] | None = parse_artifact_id(aid)
        except ValueError:
            ids = None
        if ids:
            if ids["clip"]:  # a clip regenerated or a different take selected (take ids map to their clip)
                return self._make(aid, "anchor", self.anchor_descendants(str(ids["clip"])), "stale", False)
            if ids["shot"]:
                clips = self._ordered(c for c, clip in self.clips.items() if clip.shot_id == aid)
                return self._make(aid, "shot_spec", clips, "needs_relint", False)
            nxt = self._next_scene(aid)
            return self._make(aid, "exit_state", [], "needs_review", False, scenes=[nxt] if nxt else [])
        project = self._require_project()
        if aid.startswith("VOICE_"):
            char = _voice_character(project, aid)
            if char:
                return self.fanout_voice(char)
        if aid.startswith("ENV_"):
            env_id, _, angle = aid.partition(":")
            return self.fanout_environment(env_id, angle or None)
        index = project.entity_state_index()
        if aid in index:
            return self.fanout_pack(aid)
        states = sorted(sid for sid, (ent, _st) in index.items() if ent.id == aid)
        if states:
            clips = self._ordered(c for c, clip in self.clips.items()
                                  if set(states) & set(clip.conditioning.reference_entity_states))
            return self._make(aid, "pack", clips, "stale", True)
        raise ValueError(f"I don't know what depends on {aid!r} (expected a scene/shot/clip/take, entity state, "
                         "entity, VOICE_<character> or ENV_<location>[:angle] id)")

    def fanout_pack(self, state_id: str) -> FanOut:
        validate_id(state_id, "entity state id")
        clips = self._ordered(c for c, clip in self.clips.items() if state_id in clip.conditioning.reference_entity_states)
        return self._make(state_id, "pack", clips, "stale", True)

    def fanout_environment(self, env_id: str, angle: str | None = None) -> FanOut:
        validate_id(env_id, "environment id")
        scenes = {s.id for s in self.scenes if s.environment_id == env_id}

        def clip_angle(clip: GenClip) -> str:
            shot = self.shots.get(clip.shot_id)
            return clip.environment_angle or (shot.environment_angle if shot else None) or "geometry"

        whole = angle in (None, "", "geometry")  # the geometry still is the root of every derived angle
        clips = self._ordered(c for c, clip in self.clips.items()
                              if clip.scene_id in scenes and (whole or clip_angle(clip) == angle))
        return self._make(f"{env_id}:{angle}" if angle else env_id, "environment_angle", clips, "stale", True)

    def fanout_voice(self, character_id: str) -> FanOut:
        validate_id(character_id, "character id")
        project = self._require_project()
        vp = project.load_voice_profile(character_id)
        clips = self._ordered(c for c, clip in self.clips.items() if character_id in speakers_in_clip(clip))
        return self._make(vp.id if vp else f"VOICE_{character_id}", "voice", clips, "needs_voice_review", True)

    def _next_scene(self, scene_id: str) -> str | None:
        ids = [s.id for s in self.scenes]
        if scene_id not in ids:
            return None
        i = ids.index(scene_id)
        return ids[i + 1] if i + 1 < len(ids) else None


# ---------------------------------------------------------------------------------------------
# Describe / confirm / apply
# ---------------------------------------------------------------------------------------------
def describe_fanout(fanout: FanOut) -> str:
    """Plain-language fan-out shown to the director BEFORE any confirm (§9.3, acceptance 4)."""
    n, root = len(fanout.clips), fanout.root
    where = f" in {_plural(len(fanout.scenes), 'scene')} ({', '.join(fanout.scenes)})" if n and fanout.scenes else ""
    listing = (": " + ", ".join(fanout.clips[:12]) + (f" and {n - 12} more" if n > 12 else "")) if n else ""
    if fanout.kind == "anchor":
        text = (f"Regenerating {root} changes the first frame of {_plural(n, 'later clip')}{where}{listing}. "
                "They will be marked stale; I'll ask you before regenerating them." if n else
                f"Regenerating {root} affects no other clip — nothing starts from its last frame.")
    elif fanout.kind == "pack":
        text = (f"Changing the {root} reference pack affects {_plural(n, 'clip')}{where}{listing}. "
                "They will be marked stale and regenerated with the new pack." if n else
                f"Changing the {root} reference pack affects no generated clip yet.")
    elif fanout.kind == "environment_angle":
        text = (f"Changing {root} affects {_plural(n, 'clip')} filmed on it{where}{listing}. They will be marked stale."
                if n else f"Changing {root} affects no generated clip yet.")
    elif fanout.kind == "voice":
        text = (f"Re-anchoring {root} means every clip where this character speaks ({_plural(n, 'clip')}, "
                f"{_plural(len(fanout.takes), 'take')}) gets its voice re-checked against the new anchor{listing}. "
                "Nothing is regenerated.")
    elif fanout.kind == "shot_spec":
        text = f"Changing the shot spec of {root} sends {_plural(n, 'clip')} back through the pre-flight check{listing}."
    else:
        text = (f"Changing how {root} ends means {', '.join(fanout.scenes)} must be reviewed again "
                "(its working set is rebuilt)." if fanout.scenes else f"Changing how {root} ends affects no later scene.")
    if fanout.requires_confirm:
        text += f" This is a Vision-tier change — reply `confirm {confirm_target(fanout)}` to go ahead."
    return text


def request_confirm(state: StateStore, fanout: FanOut) -> str:
    """Register the pending confirm (fan-out shown) and return the text to show the director."""
    state.request_confirm(confirm_target(fanout), action=f"invalidate:{fanout.kind}", fanout=fanout.model_dump(mode="json"))
    return describe_fanout(fanout)


def _confirm_targets(project: Project, fanout: FanOut) -> set[str]:
    targets = {fanout.root, confirm_target(fanout)}
    if fanout.kind == "voice":
        char = _voice_character(project, fanout.root)
        if char:
            targets.add(f"VOICE_{char}")
    return targets


def _check_confirm(project: Project, state: StateStore | None, fanout: FanOut, decision_id: str | None) -> None:
    ask = f"reply `confirm {confirm_target(fanout)}` after reading the fan-out: {describe_fanout(fanout)}"
    if not decision_id:
        raise ConfirmRequiredError(f"{fanout.root} is a Vision-tier change — {ask}")
    validate_id(decision_id, "decision id")
    d = DecisionLog(project).get(decision_id)
    if d is None:
        raise ConfirmRequiredError(f"decision {decision_id} is not on record — {ask}")
    if d.tier != DecisionTier.VISION or d.action in _NOT_A_CONFIRM:
        raise ConfirmRequiredError(f"decision {decision_id} is not a Vision-tier confirm — {ask}")
    allowed = _confirm_targets(project, fanout)
    if d.target_artifact_id not in allowed and (d.payload or {}).get("confirm") not in allowed:
        raise ConfirmRequiredError(f"decision {decision_id} confirms {d.target_artifact_id}, not {fanout.root} — {ask}")
    if state is not None:
        for t in sorted(allowed):
            if state.pending_confirm(t) is not None:
                state.consume_confirm(t)


def _mark_clip(project: Project, clip_id: str, status: ClipStatus) -> bool:
    path = project.clip_dir(clip_id) / "clip.yaml"
    if not path.is_file():
        return False
    with locked(path):
        clip = GenClip.model_validate(read_record(path))
        if _STRENGTH.get(clip.status, 0) >= _STRENGTH[status]:
            return False  # never downgrade (stale > needs_relint > needs_voice_review)
        clip.status = status
        project.save_clip(clip)
    return True


def _mark_scene_for_review(project: Project, scene_id: str) -> bool:
    if not (project.scene_dir(scene_id) / "working_set.yaml").is_file():
        return False  # no working set built yet: it will be built fresh, nothing to review
    path = project.path("bible", "scenes", f"{scene_id}.yaml")
    with locked(path):
        scene = project.load_scene(scene_id)
        if scene.status == SceneStatus.NEEDS_REVIEW:
            return False
        scene.status = SceneStatus.NEEDS_REVIEW
        project.save_scene(scene)
    return True


def apply_invalidation(project: Project, state: StateStore | None, fanout: FanOut, *, reason: str,
                       decision_id: str | None = None) -> list[str]:
    """Mark exactly the fan-out's clips/scenes; returns the ids whose record actually changed."""
    reason = " ".join(str(reason or "").split())
    if not reason:
        raise ValueError("an invalidation needs a reason (it is logged)")
    for cid in fanout.clips:
        if parse_artifact_id(cid)["clip"] != cid:
            raise ValueError(f"{cid!r} in the fan-out is not a clip id")
    for sid in fanout.scenes:
        validate_id(sid, "scene id")
    if fanout.requires_confirm:
        _check_confirm(project, state, fanout, decision_id)
    elif decision_id is not None:
        validate_id(decision_id, "decision id")
    changed: list[str] = []
    status = _ACTION_STATUS.get(fanout.action)
    if status is not None:
        changed += [cid for cid in fanout.clips if _mark_clip(project, cid, status)]
    elif fanout.action == "needs_review":
        changed += [sid for sid in fanout.scenes if _mark_scene_for_review(project, sid)]
    append_jsonl(project.path("logs", "invalidation.jsonl"), {
        "ts": now_iso(), "root": fanout.root, "kind": fanout.kind, "action": fanout.action, "reason": reason,
        "decision_id": decision_id, "changed": changed, "fanout": fanout.model_dump(mode="json"),
    })
    return changed
