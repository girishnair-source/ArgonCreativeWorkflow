"""Scene working sets — BUILD_SPEC_v3 §4.5 (Workstream A). THE ONLY CONTEXT AN AGENT SEES FOR SCENE N.

:func:`build_working_set` assembles pointers (never copies) into the Bible / Global Asset Library for exactly the
entity-states, voices and location the scene uses, the full scene record, the neighbouring scenes' exit/entry
continuity, a ≤5-line story-so-far built from EARLIER scenes' beat summaries only (no other scene's dialogue ever
enters), the preferences relevant to this scene, pinned references and this scene's feedback history.
``working_set_hash`` is a deterministic content hash: identical inputs always give the identical hash, and any
change (e.g. a new scene preference) changes it (acceptance 5).
"""

from __future__ import annotations

import contextlib
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .continuity import all_scene_states, entity_id_of_state
from .decisions import DecisionLog, PreferenceStore, decision_text, decision_touches_scene, preference_is_relevant
from .fsio import (
    ImmutableArtifactError,
    PathSecurityError,
    append_jsonl,
    generated_by,
    locked,
    now_iso,
    read_jsonl,
    read_record,
    save_versioned,
    stable_hash,
    validate_id,
)
from .invalidation import _voice_character
from .models import ApprovalStatus, Decision, PinnedRef, Preference, RefPointer, Scene, SceneWorkingSet
from .project import Project

STORY_LINES = 5
FEEDBACK_LIMIT = 40
PIN_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".tif", ".tiff", ".mp4", ".mov", ".m4v",
                            ".wav", ".mp3", ".m4a", ".aac", ".flac", ".pdf", ".txt", ".md"})
PIN_MAX_BYTES = 512 * 1024 * 1024


def _flat(text: Any) -> str:
    return " ".join(str(text or "").split())


def working_set_path(project: Project, scene_id: str) -> Path:
    return project.scene_dir(scene_id) / "working_set.yaml"


def pins_path(project: Project, scene_id: str) -> Path:
    return project.scene_dir(scene_id) / "pinned" / "pins.jsonl"


def working_set_hash(ws: SceneWorkingSet) -> str:
    return stable_hash(ws.model_dump(mode="json", exclude={"working_set_hash", "generated_by"}), length=16)


# ---------------------------------------------------------------------------------------------
# Pointers into the Bible / library
# ---------------------------------------------------------------------------------------------
def _library_pack_ref(project: Project, state_id: str, pack: Any) -> str | None:
    pat = re.compile(rf"^library/packs/{re.escape(state_id)}/v(\d+)/")
    for uri in [*pack.image_uris(), pack.identity_embedding]:
        m = pat.match(uri or "")
        if m:
            rel = f"library/packs/{state_id}/v{m.group(1)}/pack.yaml"
            if project.abs(rel).is_file():
                return rel
    return None


def _entity_pointers(project: Project, scene: Scene) -> list[RefPointer]:
    index = project.entity_state_index()
    out = []
    for sid in all_scene_states(scene):
        hit = index.get(sid)
        if hit is None:  # not designed yet: the designers' delta
            entity = sid.split("_STATE_", 1)[0]
            out.append(RefPointer(id=sid, ref=f"bible/entities/{entity}.yaml", version=0,
                                  approval_status=ApprovalStatus.PROPOSED))
            continue
        ent, st = hit
        lib = _library_pack_ref(project, sid, st.reference_pack) if st.approval_status == ApprovalStatus.LOCKED else None
        out.append(RefPointer(id=sid, ref=lib or f"bible/entities/{ent.id}.yaml", version=st.version,
                              approval_status=st.approval_status))
    return out


def _voice_pointers(project: Project, scene: Scene) -> list[RefPointer]:
    casting = project.casting().get("voice_profiles") or {}
    out = []
    for char in dict.fromkeys(ln.character_id for ln in scene.dialogue):
        entry = casting.get(char)
        vp = project.load_voice_profile(char) if isinstance(entry, dict) and entry else None
        if vp is None:
            out.append(RefPointer(id=f"VOICE_{char}", ref="bible/casting.yaml", version=0,
                                  approval_status=ApprovalStatus.PROPOSED))
            continue
        lib_v = entry.get("library_version")
        ref = project.rel(project.voice_profile_path(char, int(lib_v))) if lib_v else "bible/casting.yaml"
        out.append(RefPointer(id=vp.id, ref=ref, version=vp.version, approval_status=vp.approval_status))
    return out


def _environment_pointer(project: Project, scene: Scene) -> RefPointer | None:
    if not scene.environment_id:
        return None
    ref = f"bible/environments/{validate_id(scene.environment_id, 'environment id')}.yaml"
    try:
        env = project.load_environment(scene.environment_id)
    except FileNotFoundError:
        return RefPointer(id=scene.environment_id, ref=ref, version=0, approval_status=ApprovalStatus.PROPOSED)
    return RefPointer(id=env.id, ref=ref, version=env.version, approval_status=env.approval_status)


def _style_pointer(project: Project) -> RefPointer:
    style = project.load_style_lock()
    return RefPointer(id="STYLE_LOCK", ref="bible/style_lock.yaml", version=style.version if style else 0,
                      approval_status=style.approval_status if style else ApprovalStatus.PROPOSED)


def _story_line(scene: Scene) -> str:
    beats = "; ".join(_flat(b.summary) for b in scene.beats if _flat(b.summary))
    return (f"{scene.id} ({scene.slug}): {beats}" if beats else f"{scene.id} ({scene.slug})")[:240]


def _scene_characters(project: Project, scene: Scene) -> list[str]:
    entity_ids = [e.id for e in project.list_entities()]
    chars = [entity_id_of_state(s, entity_ids) or s.split("_STATE_", 1)[0] for s in all_scene_states(scene)]
    chars += [ln.character_id for ln in scene.dialogue]
    return list(dict.fromkeys(chars))


def _preferences(project: Project, preferences: Any) -> list[Preference]:
    if preferences is None:
        return PreferenceStore(project).list()
    if isinstance(preferences, PreferenceStore):
        return preferences.list()
    return [p if isinstance(p, Preference) else Preference.model_validate(p) for p in preferences]


def _decisions(project: Project, decisions: Any, scene_id: str) -> list[Decision]:
    if decisions is None:
        return DecisionLog(project).for_scene(scene_id)
    if isinstance(decisions, DecisionLog):
        return decisions.for_scene(scene_id)
    items = [d if isinstance(d, Decision) else Decision.model_validate(d) for d in decisions]
    return [d for d in items if decision_touches_scene(d, scene_id)]


def _feedback(d: Decision) -> dict[str, Any]:
    row = {"decision_id": d.id, "ts": d.ts, "tier": d.tier.value, "gate": d.gate, "action": d.action.value,
           "target": d.target_artifact_id, "source": d.source.value, "text": decision_text(d) or None}
    return {k: v for k, v in row.items() if v is not None}


# ---------------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------------
def build_working_set(project: Project, scene_id: str, *, preferences: Any = None, decisions: Any = None,
                      save: bool = True) -> SceneWorkingSet:
    validate_id(scene_id, "scene id")
    scene = project.load_scene(scene_id)
    scenes = project.list_scenes()
    ids = [s.id for s in scenes]
    pos = ids.index(scene_id) if scene_id in ids else None
    prev = scenes[pos - 1] if pos else None
    nxt = scenes[pos + 1] if pos is not None and pos + 1 < len(scenes) else None
    chars = _scene_characters(project, scene)
    prefs = sorted((p for p in _preferences(project, preferences) if p.active and (
        preference_is_relevant(p, scene_id=scene_id, character_ids=chars) or p.scope.startswith(("agent:", "user:")))),
        key=lambda p: p.id)
    pins = []
    for row in read_jsonl(pins_path(project, scene_id)):
        try:
            pins.append(PinnedRef.model_validate(row))
        except ValueError:
            continue
    ws = SceneWorkingSet(
        scene_id=scene_id,
        style_lock=_style_pointer(project),
        entity_states=_entity_pointers(project, scene),
        voice_profiles=_voice_pointers(project, scene),
        environment=_environment_pointer(project, scene),
        scene=scene,
        prev_scene_exit_state=dict(prev.exit_state) if prev else {},
        next_scene_entry_needs=({"scene_id": nxt.id, "entry_state": dict(nxt.entry_state),
                                 "entity_states_active": list(nxt.entity_states_active)} if nxt else {}),
        story_so_far=[_story_line(s) for s in scenes[:pos or 0]][-STORY_LINES:],
        scene_preferences=prefs,
        pinned_extra_refs=pins,
        feedback_history=[_feedback(d) for d in _decisions(project, decisions, scene_id)][-FEEDBACK_LIMIT:],
    )
    ws.working_set_hash = working_set_hash(ws)
    ws.generated_by = generated_by("agy.working_set", working_set_hash=ws.working_set_hash)
    if save:
        path = working_set_path(project, scene_id)
        with locked(path):
            current = None
            if path.is_file():
                with contextlib.suppress(OSError, ValueError, AttributeError):
                    current = read_record(path).get("working_set_hash")
            if current != ws.working_set_hash:  # unchanged content is not rewritten (no _history churn)
                save_versioned(path, ws)
    return ws


def load_working_set(project: Project, scene_id: str) -> SceneWorkingSet | None:
    path = working_set_path(project, scene_id)
    return SceneWorkingSet.model_validate(read_record(path)) if path.is_file() else None


def resolve_refs(project: Project, ws: SceneWorkingSet) -> dict[str, Any]:
    """Load the records the pointers name; ``warnings`` lists anything missing or changed since the build."""
    warnings: list[str] = []
    out: dict[str, Any] = {"style_lock": project.load_style_lock(), "entity_states": {}, "voice_profiles": {},
                           "environment": None, "pinned": [], "warnings": warnings}
    if out["style_lock"] is None:
        warnings.append("there is no style lock yet")
    elif out["style_lock"].version != ws.style_lock.version:
        warnings.append("the style lock changed since this working set was built")
    index = project.entity_state_index()
    for ptr in ws.entity_states:
        hit = index.get(ptr.id)
        if hit is None:
            warnings.append(f"{ptr.id} is not in the bible yet")
            continue
        ent, st = hit
        images = []
        for uri in st.reference_pack.image_uris():
            try:
                images.append(project.abs(uri))
            except PathSecurityError:
                warnings.append(f"{ptr.id}: ignored unsafe pack path {uri!r}")
        out["entity_states"][ptr.id] = {"entity": ent, "state": st, "images": images}
        lib = _library_pack_ref(project, ptr.id, st.reference_pack) if st.approval_status == ApprovalStatus.LOCKED else None
        if st.version != ptr.version or (ptr.ref.startswith("library/") and lib != ptr.ref):
            warnings.append(f"{ptr.id} changed since this working set was built")
    for ptr in ws.voice_profiles:
        char = _voice_character(project, ptr.id)
        vp = project.load_voice_profile(char) if char else None
        if vp is None:
            warnings.append(f"{ptr.id} has no voice profile yet")
            continue
        out["voice_profiles"][ptr.id] = vp
        if vp.version != ptr.version:
            warnings.append(f"{ptr.id} changed since this working set was built")
    if ws.environment is not None:
        try:
            env = project.load_environment(ws.environment.id)
        except FileNotFoundError:
            warnings.append(f"{ws.environment.id} is not in the bible yet")
        else:
            out["environment"] = env
            if env.version != ws.environment.version:
                warnings.append(f"{env.id} changed since this working set was built")
    for pin in ws.pinned_extra_refs:
        try:
            out["pinned"].append(project.abs(pin.path))
        except PathSecurityError:
            warnings.append(f"ignored unsafe pinned path {pin.path!r}")
    return out


def _ptr_text(p: RefPointer) -> str:
    return f"{p.id} — not designed yet" if p.version == 0 else f"{p.id} v{p.version} ({p.approval_status})"


def _kv(d: dict[str, Any]) -> str:
    return "; ".join(f"{k}: {v}" for k, v in d.items())


def show_working_set(project: Project, scene_id: str) -> str:
    """Plain-language view for the chat command ``show working_set SC12`` (rebuilds, saving only on change)."""
    ws = build_working_set(project, scene_id, save=True)
    sc = ws.scene
    nxt = ws.next_scene_entry_needs
    lines = [
        f"Working set — {sc.id} {sc.slug} (hash {ws.working_set_hash}). This is everything the scene's agents see.",
        f"Style lock: {_ptr_text(ws.style_lock)}",
        "Characters & props: " + (", ".join(_ptr_text(p) for p in ws.entity_states) or "none"),
        "Voices: " + (", ".join(_ptr_text(p) for p in ws.voice_profiles) or "none"),
        "Location: " + (_ptr_text(ws.environment) if ws.environment else "none")
        + (f" · lighting {sc.lighting_setup_id}" if sc.lighting_setup_id else ""),
        "Previous scene ends with: " + (_kv(ws.prev_scene_exit_state) or "— (first scene)"),
        "Next scene needs: " + ((f"{nxt.get('scene_id')} starts with " + (_kv(nxt.get("entry_state") or {})
                                 or ", ".join(nxt.get("entity_states_active") or []) or "nothing specific"))
                                if nxt else "— (last scene)"),
        "Story so far:" + ("" if ws.story_so_far else " — (this is the first scene)"),
        *[f"  • {s}" for s in ws.story_so_far],
        f"Dialogue ({len(sc.dialogue)} lines):",
        *[f"  {ln.line_id} {ln.character_id}{' (off-screen)' if ln.off_screen else ''}: {ln.text}"
          + (f" — {ln.transliteration}" if ln.transliteration else "") for ln in sc.dialogue],
        f"Standing notes ({len(ws.scene_preferences)}):",
        *[f"  • [{p.scope}] {p.text}" for p in ws.scene_preferences],
        f"Pinned references ({len(ws.pinned_extra_refs)}):",
        *[f"  • {project.abs(p.path)} — {p.why}" for p in ws.pinned_extra_refs],
        f"Feedback so far ({len(ws.feedback_history)} decisions):",
        *[f"  • {f['decision_id']} {f['action']} {f['target']}" + (f" — {f['text']}" if f.get("text") else "")
          for f in ws.feedback_history[-5:]],
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Pinning (user-supplied files are COPIED into the project, never referenced in place)
# ---------------------------------------------------------------------------------------------
def _unique_dest(directory: Path, stem: str, ext: str) -> Path:
    candidate, n = directory / f"{stem}{ext}", 1
    while candidate.exists():
        n += 1
        candidate = directory / f"{stem}_{n}{ext}"
    return candidate


def _copy_new(src: Path, dest: Path, max_bytes: int) -> None:
    """Streaming, atomic, never-overwrite copy of a regular file (local stand-in for a shared fsio.copy_new)."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    src_fd = os.open(src, flags)
    try:
        info = os.fstat(src_fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{src} is not a regular file")
        if info.st_size > max_bytes:
            raise ValueError(f"{src} is larger than the {max_bytes // (1024 * 1024)} MB pin limit")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=str(dest.parent))
        try:
            total = 0
            with os.fdopen(tmp_fd, "wb") as out:
                while block := os.read(src_fd, 1 << 20):
                    total += len(block)
                    if total > max_bytes:
                        raise ValueError(f"{src} grew past the pin size limit while copying")
                    out.write(block)
                out.flush()
                os.fsync(out.fileno())
            try:
                os.link(tmp, dest)  # atomic and never replaces an existing file
            except FileExistsError:
                raise ImmutableArtifactError(f"refusing to overwrite {dest}") from None
            except OSError:  # filesystem without hard links
                if dest.exists():
                    raise ImmutableArtifactError(f"refusing to overwrite {dest}") from None
                os.replace(tmp, dest)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)  # our own temp file only
    finally:
        os.close(src_fd)


def pin(project: Project, scene_id: str, src_path: str | Path, why: str, *,
        decision_id: str | None = None) -> PinnedRef:
    """Copy an extra reference into ``scenes/<SC>/pinned/`` and add it to the scene's working set."""
    validate_id(scene_id, "scene id")
    project.load_scene(scene_id)  # the scene must exist
    why = _flat(why)
    if not why or len(why) > 500:
        raise ValueError("say in a sentence (max 500 characters) why this reference matters")
    if decision_id is not None:
        validate_id(decision_id, "decision id")
    src = Path(src_path).expanduser()
    try:
        real = src.resolve(strict=True)
    except (OSError, RuntimeError):
        raise FileNotFoundError(f"{src_path} does not exist") from None
    ext = real.suffix.lower()
    if ext not in PIN_EXTENSIONS:
        raise ValueError(f"{ext or 'files without an extension'} can't be pinned "
                         f"(allowed: {', '.join(sorted(PIN_EXTENSIONS))})")
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", src.stem).strip("._-")[:80] or "ref"
    dest_dir = project.scene_dir(scene_id) / "pinned"
    with locked(pins_path(project, scene_id)):  # serialise name allocation
        dest = _unique_dest(dest_dir, stem, ext)
        _copy_new(real, dest, PIN_MAX_BYTES)
    ref = PinnedRef(path=project.rel(dest), why=why, pinned_at=now_iso(), decision_id=decision_id)
    append_jsonl(pins_path(project, scene_id), ref)
    return ref
