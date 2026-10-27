"""Project folder — the durable system of record (BUILD_SPEC_v3 §3).

``Project`` owns the folder layout, bible records, the append-only Global Asset Library and the
shot/clip/take tree. Locked records are never edited in place: edits require a Vision-tier confirm
decision id and produce a new version (previous content archived under ``_history/``). Library assets are
promoted into new ``v<n>`` folders and are never overwritten.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Iterable, TypeVar

from pydantic import BaseModel

from .fsio import (
    ImmutableArtifactError,
    PathSecurityError,
    atomic_write_yaml,
    from_rel,
    generated_by,
    now_iso,
    read_record,
    save_versioned,
    safe_join,
    to_rel,
    validate_id,
    write_new,
)
from .models import (
    ApprovalStatus,
    CreativeShot,
    Entity,
    Environment,
    GeneratedBy,
    GenClip,
    ProjectConfig,
    ReferencePack,
    ResolveHandoff,
    Scene,
    StyleLock,
    Take,
    Turnaround,
    VoiceProfile,
)

T = TypeVar("T", bound=BaseModel)

PRODUCTIONS_DIR_NAME = "productions"
_ARTIFACT_RE = re.compile(
    r"^(?P<scene>SC\d{2,4})(?:_(?P<shot>SH\d{2,3}))?(?:_(?P<clip>C\d{2,3}))?(?:_(?P<take>T\d{2,3}))?$"
)
PACK_ROLES = ("front", "three_quarter", "profile", "rear", "face_closeup", "expression_sheet", "single_ref", "identity_embedding")


class LockedAssetError(PermissionError):
    """Editing a locked asset requires an explicit Vision-tier `confirm <artifact_id>` decision."""


def parse_artifact_id(artifact_id: str) -> dict[str, str | None]:
    """Split ``SC12_SH04_C02_T01`` into scene/shot/clip/take ids (each fully qualified)."""
    m = _ARTIFACT_RE.match(artifact_id or "")
    if not m:
        raise ValueError(f"{artifact_id!r} is not a scene/shot/clip/take id (expected SC12[_SH04[_C02[_T01]]])")
    scene = m.group("scene")
    shot = f"{scene}_{m.group('shot')}" if m.group("shot") else None
    clip = f"{shot}_{m.group('clip')}" if shot and m.group("clip") else None
    take = f"{clip}_{m.group('take')}" if clip and m.group("take") else None
    return {"scene": scene, "shot": shot, "clip": clip, "take": take}


def scene_id(n: int) -> str:
    return f"SC{n:02d}"


def shot_id(scene: str, n: int) -> str:
    return f"{scene}_SH{n:02d}"


def clip_id(shot: str, n: int) -> str:
    return f"{shot}_C{n:02d}"


def take_id(clip: str, n: int) -> str:
    return f"{clip}_T{n:02d}"


class Project:
    """Filesystem-backed project. All paths are resolved and confined to ``root``."""

    SCAFFOLD_DIRS = (
        "script",
        "bible/entities",
        "bible/environments",
        "bible/scenes",
        "library/packs",
        "library/environments",
        "library/voices",
        "library/score/themes",
        "library/sound/palette",
        "scenes",
        "post",
        "logs/agent_calls",
        "scripts",
    )

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        if not (self.root / "project.yaml").exists():
            raise FileNotFoundError(f"{self.root} is not an AGY project (missing project.yaml)")
        self._config: ProjectConfig | None = None

    # ------------------------------------------------------------------------------------------
    # Creation / discovery
    # ------------------------------------------------------------------------------------------
    @classmethod
    def create(cls, productions_dir: Path, project_id: str, title: str, *, language: str = "hi",
               config: ProjectConfig | None = None) -> "Project":
        validate_id(project_id, "project_id")
        root = safe_join(Path(productions_dir), project_id)
        if (root / "project.yaml").exists():
            raise FileExistsError(f"project {project_id} already exists at {root}")
        for rel in cls.SCAFFOLD_DIRS:
            (root / rel).mkdir(parents=True, exist_ok=True)
        cfg = config or ProjectConfig(project_id=project_id, title=title, language=language)
        cfg.created_at = cfg.created_at or now_iso()
        cfg.generated_by = cfg.generated_by or generated_by("agy.project.create")
        atomic_write_yaml(root / "project.yaml", cfg)
        for name in ("decisions.jsonl", "questions.jsonl"):
            (root / name).touch(exist_ok=True)
        atomic_write_yaml(root / "preferences.yaml", {"preferences": [], "generated_by": generated_by("agy.decisions")})
        atomic_write_yaml(root / "bible" / "casting.yaml", {
            "voice_profiles": {}, "score_themes": [], "sound_palette": [],
            "generated_by": generated_by("agy.project.create"),
        })
        atomic_write_yaml(root / "bible" / "state_timeline.yaml", {
            "entries": [], "generated_by": generated_by("agy.project.create"),
        })
        return cls(root)

    @classmethod
    def open(cls, root: Path) -> "Project":
        return cls(root)

    @staticmethod
    def productions_dir(workspace: Path) -> Path:
        return Path(workspace).resolve() / PRODUCTIONS_DIR_NAME

    # ------------------------------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------------------------------
    def path(self, *parts: str | Path) -> Path:
        return safe_join(self.root, *parts)

    def rel(self, path: Path | str) -> str:
        return to_rel(self.root, path)

    def abs(self, rel: str) -> Path:
        return from_rel(self.root, rel)

    @property
    def project_id(self) -> str:
        return self.config.project_id

    @property
    def state_path(self) -> Path:
        return self.path("state.json")

    @property
    def decisions_path(self) -> Path:
        return self.path("decisions.jsonl")

    @property
    def questions_path(self) -> Path:
        return self.path("questions.jsonl")

    @property
    def preferences_path(self) -> Path:
        return self.path("preferences.yaml")

    @property
    def agent_calls_dir(self) -> Path:
        return self.path("logs", "agent_calls")

    @property
    def bridge_log_path(self) -> Path:
        return self.path("logs", "resolve_bridge.log")

    def scene_dir(self, scene: str) -> Path:
        return self.path("scenes", validate_id(scene, "scene id"))

    def shot_dir(self, shot: str) -> Path:
        ids = parse_artifact_id(shot)
        return self.path("scenes", ids["scene"], "shots", validate_id(shot, "shot id"))

    def clip_dir(self, clip: str) -> Path:
        ids = parse_artifact_id(clip)
        return self.path("scenes", ids["scene"], "shots", ids["shot"], "clips", validate_id(clip, "clip id"))

    def take_dir(self, take: str) -> Path:
        ids = parse_artifact_id(take)
        return self.path("scenes", ids["scene"], "shots", ids["shot"], "clips", ids["clip"], "takes",
                         validate_id(take, "take id"))

    def review_dir(self, scene: str, gate: str, ts: str) -> Path:
        return self.path("scenes", validate_id(scene, "scene id"), "review", f"{gate}_{ts}")

    # ------------------------------------------------------------------------------------------
    # Generic record IO
    # ------------------------------------------------------------------------------------------
    def _load(self, model: type[T], path: Path) -> T:
        return model.model_validate(read_record(path))

    def _load_opt(self, model: type[T], path: Path) -> T | None:
        return self._load(model, path) if path.exists() else None

    def _save_guarded(self, path: Path, record: BaseModel, *, confirm_decision_id: str | None,
                      artifact_id: str) -> Path:
        """Versioned save that refuses to mutate a locked record without a Vision-tier confirm."""
        existing = read_record(path) if path.exists() else None
        if existing is not None and existing.get("approval_status") == ApprovalStatus.LOCKED.value:
            new_payload = record.model_dump(mode="json")
            unchanged = {k: v for k, v in new_payload.items() if k not in ("generated_by",)} == {
                k: v for k, v in existing.items() if k not in ("generated_by",)
            }
            if not unchanged:
                if not confirm_decision_id:
                    raise LockedAssetError(
                        f"{artifact_id} is locked; editing it needs `confirm {artifact_id}` (Vision tier) after fan-out"
                    )
                old_version = int(existing.get("version", 1))
                if getattr(record, "version", None) is not None and record.version <= old_version:
                    record.version = old_version + 1
                if hasattr(record, "derived_from"):
                    record.derived_from = f"{artifact_id}@v{old_version}"
        return save_versioned(path, record)

    # ------------------------------------------------------------------------------------------
    # project.yaml
    # ------------------------------------------------------------------------------------------
    @property
    def config(self) -> ProjectConfig:
        if self._config is None:
            self._config = self._load(ProjectConfig, self.root / "project.yaml")
        return self._config

    def save_config(self, cfg: ProjectConfig) -> None:
        save_versioned(self.root / "project.yaml", cfg)
        self._config = cfg

    # ------------------------------------------------------------------------------------------
    # Bible: style lock, entities, environments, scenes, voices
    # ------------------------------------------------------------------------------------------
    def load_style_lock(self) -> StyleLock | None:
        return self._load_opt(StyleLock, self.path("bible", "style_lock.yaml"))

    def save_style_lock(self, style: StyleLock, *, confirm_decision_id: str | None = None) -> Path:
        return self._save_guarded(self.path("bible", "style_lock.yaml"), style,
                                  confirm_decision_id=confirm_decision_id, artifact_id="STYLE_LOCK")

    def load_entity(self, entity_id: str) -> Entity:
        return self._load(Entity, self.path("bible", "entities", f"{validate_id(entity_id, 'entity id')}.yaml"))

    def save_entity(self, entity: Entity, *, confirm_decision_id: str | None = None) -> Path:
        return self._save_guarded(self.path("bible", "entities", f"{validate_id(entity.id, 'entity id')}.yaml"),
                                  entity, confirm_decision_id=confirm_decision_id, artifact_id=entity.id)

    def list_entities(self) -> list[Entity]:
        return [self._load(Entity, p) for p in sorted(self.path("bible", "entities").glob("*.yaml"))]

    def entity_state_index(self) -> dict[str, tuple[Entity, object]]:
        """state id -> (entity, EntityState)."""
        out: dict[str, tuple[Entity, object]] = {}
        for ent in self.list_entities():
            for st in ent.states:
                out[st.id] = (ent, st)
        return out

    def load_environment(self, env_id: str) -> Environment:
        return self._load(Environment, self.path("bible", "environments", f"{validate_id(env_id, 'env id')}.yaml"))

    def save_environment(self, env: Environment, *, confirm_decision_id: str | None = None) -> Path:
        return self._save_guarded(self.path("bible", "environments", f"{validate_id(env.id, 'env id')}.yaml"),
                                  env, confirm_decision_id=confirm_decision_id, artifact_id=env.id)

    def list_environments(self) -> list[Environment]:
        return [self._load(Environment, p) for p in sorted(self.path("bible", "environments").glob("*.yaml"))]

    def load_scene(self, scene: str) -> Scene:
        return self._load(Scene, self.path("bible", "scenes", f"{validate_id(scene, 'scene id')}.yaml"))

    def save_scene(self, scene: Scene) -> Path:
        return save_versioned(self.path("bible", "scenes", f"{validate_id(scene.id, 'scene id')}.yaml"), scene)

    def list_scenes(self) -> list[Scene]:
        scenes = [self._load(Scene, p) for p in self.path("bible", "scenes").glob("*.yaml")]
        return sorted(scenes, key=lambda s: (s.order, s.id))

    def casting(self) -> dict:
        p = self.path("bible", "casting.yaml")
        return read_record(p) if p.exists() else {"voice_profiles": {}, "score_themes": [], "sound_palette": []}

    def save_casting(self, data: dict) -> Path:
        data = dict(data)
        data["generated_by"] = generated_by("agy.casting").model_dump(mode="json")
        return save_versioned(self.path("bible", "casting.yaml"), data)

    def voice_profile_path(self, character_id: str, version: int) -> Path:
        validate_id(character_id, "character id")
        return self.path("library", "voices", character_id, f"v{int(version)}", "voice_profile.yaml")

    def load_voice_profile(self, character_id: str) -> VoiceProfile | None:
        """Current voice profile: proposed ones live in casting.yaml, locked ones in library/voices/<id>/v<n>/."""
        entry = self.casting().get("voice_profiles", {}).get(character_id)
        if not entry:
            return None
        if entry.get("library_version"):
            return self._load(VoiceProfile, self.voice_profile_path(character_id, entry["library_version"]))
        return VoiceProfile.model_validate(entry["profile"])

    def save_voice_profile(self, profile: VoiceProfile, *, confirm_decision_id: str | None = None) -> Path:
        """Proposed profiles are stored in casting.yaml; locked profiles are promoted to a new library version."""
        validate_id(profile.character_id, "character id")
        casting = self.casting()
        entry = casting.setdefault("voice_profiles", {}).get(profile.character_id, {})
        current_v = entry.get("library_version")
        if profile.approval_status != ApprovalStatus.LOCKED and not current_v:
            entry = {"profile": profile.model_dump(mode="json")}
            casting["voice_profiles"][profile.character_id] = entry
            return self.save_casting(casting)
        if current_v:
            existing = self._load(VoiceProfile, self.voice_profile_path(profile.character_id, current_v))
            if existing.model_dump(exclude={"generated_by"}) == profile.model_dump(exclude={"generated_by"}):
                return self.voice_profile_path(profile.character_id, current_v)
            # Setting the anchor for the first time is the one allowed evolution without confirm (§4.3).
            first_anchor = existing.anchor_take is None and profile.anchor_take is not None and \
                existing.model_dump(exclude={"anchor_take", "anchor_wav", "generated_by", "version", "derived_from"}) == \
                profile.model_dump(exclude={"anchor_take", "anchor_wav", "generated_by", "version", "derived_from"})
            if not first_anchor and not confirm_decision_id:
                raise LockedAssetError(
                    f"{profile.id} is locked; changing it (incl. re-anchoring) needs `confirm {profile.id}` (Vision tier)")
            profile.version = existing.version + 1
            profile.derived_from = f"{profile.id}@v{existing.version}"
        new_v = (current_v or 0) + 1
        profile.approval_status = ApprovalStatus.LOCKED
        target = self.voice_profile_path(profile.character_id, new_v)
        write_new(target, profile)
        casting["voice_profiles"][profile.character_id] = {"library_version": new_v, "voice_profile_id": profile.id}
        self.save_casting(casting)
        return target

    # ------------------------------------------------------------------------------------------
    # Global Asset Library (append-only)
    # ------------------------------------------------------------------------------------------
    def _next_library_version_dir(self, *parts: str) -> Path:
        base = self.path("library", *parts)
        base.mkdir(parents=True, exist_ok=True)
        versions = [int(p.name[1:]) for p in base.iterdir() if p.is_dir() and re.fullmatch(r"v\d+", p.name)]
        target = base / f"v{(max(versions) if versions else 0) + 1}"
        target.mkdir(parents=False, exist_ok=False)
        return target

    def promote_pack(self, entity_state_id: str, files: dict[str, Path], *, approved_by: str,
                     provenance: GeneratedBy) -> ReferencePack:
        """Copy approved pack images into ``library/packs/<STATE_ID>/v<n>/`` and return the ReferencePack."""
        validate_id(entity_state_id, "entity state id")
        unknown = set(files) - set(PACK_ROLES)
        if unknown:
            raise ValueError(f"unknown pack roles {sorted(unknown)}; allowed {PACK_ROLES}")
        vdir = self._next_library_version_dir("packs", entity_state_id)
        rels: dict[str, str] = {}
        for role, src in files.items():
            src = Path(src)
            if not src.is_file():
                raise FileNotFoundError(src)
            dest = vdir / f"{role}{src.suffix.lower()}"
            if dest.exists():
                raise ImmutableArtifactError(dest)
            shutil.copy2(src, dest)
            rels[role] = self.rel(dest)
        pack = ReferencePack(
            turnaround=Turnaround(front=rels.get("front"), three_quarter=rels.get("three_quarter"),
                                  profile=rels.get("profile"), rear=rels.get("rear")),
            face_closeup=rels.get("face_closeup"),
            expression_sheet=rels.get("expression_sheet"),
            single_ref=rels.get("single_ref"),
            identity_embedding=rels.get("identity_embedding"),
            generated_by=provenance,
            approved_by=approved_by,
            approved_at=now_iso(),
        )
        write_new(vdir / "pack.yaml", pack)
        return pack

    def promote_library_files(self, category: tuple[str, ...], files: dict[str, Path], *,
                              provenance: GeneratedBy, manifest_name: str = "manifest.yaml") -> dict[str, str]:
        """Generic append-only promotion (environments, score themes, sound palette, voice anchors)."""
        for part in category:
            if part not in ("environments", "voices", "score", "themes", "sound", "palette") and not re.fullmatch(
                    r"[A-Z][A-Z0-9_]*", part):
                raise PathSecurityError(f"illegal library category component {part!r}")
        vdir = self._next_library_version_dir(*category)
        rels: dict[str, str] = {}
        for name, src in files.items():
            if not re.fullmatch(r"[A-Za-z0-9_.\-]{1,96}", name) or name.startswith("."):
                raise PathSecurityError(f"illegal library file name {name!r}")
            dest = vdir / name
            shutil.copy2(Path(src), dest)
            rels[name] = self.rel(dest)
        write_new(vdir / manifest_name, {"files": rels, "generated_by": provenance.model_dump(mode="json")})
        return rels

    # ------------------------------------------------------------------------------------------
    # Shots / clips / takes
    # ------------------------------------------------------------------------------------------
    def load_shot(self, shot: str) -> CreativeShot:
        return self._load(CreativeShot, self.shot_dir(shot) / "shot.yaml")

    def save_shot(self, shot: CreativeShot) -> Path:
        return save_versioned(self.shot_dir(shot.id) / "shot.yaml", shot)

    def list_shots(self, scene: str) -> list[CreativeShot]:
        base = self.scene_dir(scene) / "shots"
        shots = [self._load(CreativeShot, p) for p in base.glob("*/shot.yaml")] if base.exists() else []
        return sorted(shots, key=lambda s: (s.order, s.id))

    def load_clip(self, clip: str) -> GenClip:
        return self._load(GenClip, self.clip_dir(clip) / "clip.yaml")

    def save_clip(self, clip: GenClip) -> Path:
        return save_versioned(self.clip_dir(clip.id) / "clip.yaml", clip)

    def list_clips(self, shot: str) -> list[GenClip]:
        base = self.shot_dir(shot) / "clips"
        clips = [self._load(GenClip, p) for p in base.glob("*/clip.yaml")] if base.exists() else []
        return sorted(clips, key=lambda c: (c.sequence_index, c.id))

    def list_scene_clips(self, scene: str) -> list[GenClip]:
        out: list[GenClip] = []
        for shot in self.list_shots(scene):
            out.extend(self.list_clips(shot.id))
        return out

    def load_take(self, take: str) -> Take:
        return self._load(Take, self.take_dir(take) / "take.yaml")

    def save_take(self, take: Take) -> Path:
        """take.yaml is the take's provenance/status record; its media files are immutable."""
        return save_versioned(self.take_dir(take.id) / "take.yaml", take)

    def list_takes(self, clip: str) -> list[Take]:
        base = self.clip_dir(clip) / "takes"
        takes = [self._load(Take, p) for p in base.glob("*/take.yaml")] if base.exists() else []
        return sorted(takes, key=lambda t: t.id)

    def next_take_id(self, clip: str) -> str:
        existing = {t.id for t in self.list_takes(clip)}
        n = 1
        while take_id(clip, n) in existing or self.take_dir(take_id(clip, n)).exists():
            n += 1
        return take_id(clip, n)

    # ------------------------------------------------------------------------------------------
    # Resolve handoff
    # ------------------------------------------------------------------------------------------
    def handoff_path(self, scene: str) -> Path:
        return self.scene_dir(scene) / "assembly" / "resolve_handoff.yaml"

    def load_handoff(self, scene: str) -> ResolveHandoff | None:
        return self._load_opt(ResolveHandoff, self.handoff_path(scene))

    def save_handoff(self, handoff: ResolveHandoff) -> Path:
        return save_versioned(self.handoff_path(handoff.scene_id), handoff)

    # ------------------------------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------------------------------
    def iter_all_clips(self) -> Iterable[GenClip]:
        for scene in self.list_scenes():
            yield from self.list_scene_clips(scene.id)


def find_project(workspace: Path, project_id: str | None = None) -> Project:
    """Locate a project under ``<workspace>/productions``; with no id, the single/most recent project."""
    prod = Project.productions_dir(workspace)
    if project_id:
        return Project(safe_join(prod, validate_id(project_id, "project_id")))
    candidates = sorted((p for p in prod.glob("*/project.yaml")), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no AGY projects under {prod}")
    return Project(candidates[0].parent)
