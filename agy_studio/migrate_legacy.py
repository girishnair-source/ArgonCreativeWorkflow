"""Phase 0 pilot migration — wrap the existing LFKOD system, regenerate nothing (spec §13 Phase 0, contract §6).

``migrate_lfkod`` reads the legacy *Like Share Kill Or Die* modules (casting bible, episode directions, voice
profiles, Veo/Omni plan + state JSON, reference-sheet registry, ``generated_assets/``) and writes an AGY v3 project
of record:

* a StyleLock from the episode prompt locks with the L1 banned words removed (every removal is reported);
* entities + wardrobe states (``CHAR_X_STATE_<KEY>``), props, VoiceProfiles (Hindi / Devanagari), environments;
* one Scene per contiguous location run inside each episode; dialogue verbatim (Devanagari ``text`` + phonetic
  ``transliteration``; phone / voice-over / device lines are ``off_screen``);
* one CreativeShot + one GenClip per legacy shot (flag ``legacy_import``) and the legacy mp4 as take T01 (hard link
  when possible, else copy; ``imported_from``; first/last frame + dialogue wav extracted);
* the legacy Resolve episode timelines adopted as human-owned ``ResolveHandoff(legacy=True)`` records;
* ``post/migration_report.md`` (plain language) and ``post/migration_manifest.yaml`` (machine readable).

Every Bible record stays ``proposed`` — locking at G1 is the human's Vision-tier decision. Anchor sheets and
keyframes are copied into ``scenes/<SC>/design/`` as proposed packs (promotion to ``library/`` happens only at G2).
The project is built in a staging folder and moved into place only when complete, so a killed migration never
leaves a half-built project behind (a re-run simply starts again); an existing project is never overwritten
(``FileExistsError``). ``legacy_episode_table`` reproduces ``assemble_microdrama.episode_table()`` from the
migrated takes (Phase 0 acceptance: identical output via the new interfaces).
"""

from __future__ import annotations

import ast
import importlib
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import AGY_VERSION, fsio, media, procs
from .agent_log import log_agent_call
from .continuity import count_words
from .models import (
    ApprovalStatus,
    AudioLine,
    AudioSpec,
    Beat,
    ClipStatus,
    Conditioning,
    CreativeShot,
    DispatchStatus,
    Entity,
    EntityState,
    EntityType,
    Environment,
    GateId,
    GenClip,
    GlobalStage,
    LightingSetup,
    MidSceneTransition,
    ModelTarget,
    Ownership,
    ProjectConfig,
    ProjectSettings,
    ReferencePack,
    ResolveHandoff,
    RiskTier,
    Scene,
    SceneDialogueLine,
    SceneResolve,
    SceneStage,
    SceneStatus,
    ShotSpec,
    StyleLock,
    Take,
    TakeStatus,
    Tier,
    VisualSpec,
    VoiceProfile,
)
from .project import Project
from .project import clip_id as make_clip_id
from .project import scene_id as make_scene_id
from .project import shot_id as make_shot_id
from .project import take_id as make_take_id
from .state import StateStore

AGENT = "agy.migrate_legacy"
LEGACY_PACKAGE = "projects.like_share_kill_or_die"
LEGACY_DIR = ("projects", "like_share_kill_or_die")
DEFAULT_PROJECT_ID = "LFKOD_S01"
DEFAULT_TITLE = "Like Share Kill Or Die"
LEGACY_FPS = 24  # assemble_microdrama hard-codes 24 fps
LEGACY_MIN_ACTIVE_FRAMES = 48  # assemble_microdrama._trimmed_shot_frames: trim only when raw > in + out + 48
LEGACY_TIMELINE_VERSION = "v09"  # the human's current legacy Resolve timelines: EP01_Daring_v09 / EP02_TheParcel_v09
LEGACY_ASSEMBLER_DEFAULTS: dict[str, Any] = {
    "EPISODE_TIMELINES": {1: ("EP01_Daring", "Green"), 2: ("EP02_TheParcel", "Cyan")},
    "PRO_TRIM_IN_FRAMES": 8,
    "PRO_TRIM_OUT_FRAMES": 8,
}
STAGING_DIR_NAME = ".agy_staging"
MANIFEST_REL = "post/migration_manifest.yaml"
REPORT_REL = "post/migration_report.md"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v")
MAX_IMPORT_BYTES = 4 * 1024 ** 3  # size cap for any legacy file copied into the project
HINDI_LANGUAGE = "Hindi — colloquial Mumbai Hindi, Devanagari script with phonetic transliteration"
MASTER_VOICE_ID = "CHAR_MASTER_VOICE"

# §7.1 L1 banned words (removed from every migrated prompt field; each removal is reported).
BANNED_WORDS: tuple[str, ...] = (
    "hyperrealistic", "photorealistic", "stunning", "4K", "8K", "cinematic", "breathtaking", "masterpiece",
    "beautiful", "epic", "award-winning",
)
_BANNED_RE = re.compile(
    r"(?<![A-Za-z0-9])(hyper[- ]?realistic|photo[- ]?realistic|stunning(?:ly)?|[48]k|cinematic(?:ally|s)?|"
    r"breathtaking(?:ly)?|masterpieces?|beautiful(?:ly)?|epic|award[- ]winning)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_CAP_MARK = "\x00CAP\x00"

# v3 audio negatives the Production Designer adds to the global negative (spec §6 / §8).
V3_AUDIO_NEGATIVES: tuple[str, ...] = (
    "background music", "score", "narration", "echo", "robotic voice", "subtitles",
)
# Positive mentions of these cues in the legacy sound design / voices mean the negative would fight the film.
NEGATIVE_CONFLICT_CUES: dict[str, tuple[str, ...]] = {
    "background music": ("background music", "music bed", "musical score"),
    "score": ("score", "soundtrack"),
    "narration": ("narration", "narrator"),
    "echo": ("echo", "echoes", "echoing", "reverb"),
    "robotic voice": ("robotic", "synthetic", "vocoder", "metallic", "pitch-shifted"),
    "subtitles": ("subtitle",),
}
_NEGATION_RE = re.compile(r"\b(no|never|without|zero|not|nor|avoid|avoiding)\b", re.IGNORECASE)

# L7: lighting carries physical sources only (no colour words); palette carries colour only (no light sources).
_COLOUR_WORDS = (
    "red", "orange", "amber", "yellow", "gold", "golden", "green", "teal", "cyan", "blue", "indigo", "violet",
    "purple", "magenta", "pink", "white", "black", "grey", "gray", "brown", "crimson", "scarlet", "turquoise",
    "beige", "cream", "silver", "copper", "bronze", "warm", "cool", "pastel", "monochrome", "desaturated",
    "saturated", "sepia",
)
_COLOUR_RE = re.compile(
    r"(?<![A-Za-z])(" + "|".join(sorted((re.escape(w) for w in _COLOUR_WORDS), key=len, reverse=True))
    + r")(?:[- ](?:tinted|toned|hued|ish|tones?))?(?![A-Za-z])",
    re.IGNORECASE,
)
_LENS_RE = re.compile(
    r"(\b\d{2,3}\s?mm\b|\blens(?:es)?\b|anamorphic|aperture|\bf/\d|\bt/\d|depth of field|bokeh|focal|wide-angle|"
    r"telephoto|\bmacro\b|\bprime\b|\b9:16\b|aspect ratio|\bvertical frame\b)",
    re.IGNORECASE,
)
_MULTI_BEAT_RE = re.compile(r"\[\s*beat\s*\d|cut at \d|\bcut to beat\b|\d-shot scene|multi-shot", re.IGNORECASE)
_OFF_SCREEN_MARKERS = (
    "off-screen", "off screen", "offscreen", "voice-over", "voice over", "voiceover", "v.o.", "o.s.",
    "phone speaker", "speakerphone", "voice note", "synthetic voice", "from the device", "from the phone",
    "on the phone", "over the phone",
)
_GENERIC_ALIAS_WORDS = frozenset({
    "the", "a", "an", "voice", "building", "security", "delivery", "house", "mr", "mrs", "ms", "dr", "of", "and",
    "s", "boy", "girl", "man", "woman", "old", "young",
})
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


# ---------------------------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------------------------
def _tidy(text: str) -> str:
    t = re.sub(r"\s+", " ", text or "")
    t = re.sub(r"\(\s*\)", "", t)
    t = re.sub(r"\s+([,.;:!?)])", r"\1", t)
    t = re.sub(r"([(\[])\s+", r"\1", t)
    t = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", t)
    t = re.sub(r"\.\s*[,;:]", ".", t)
    t = re.sub(r"^[\s,;:.\-–—]+", "", t)
    t = re.sub(r"[\s,;:\-–—]+$", "", t)
    return t.strip()


def strip_banned_words(text: str | None) -> tuple[str, list[str]]:
    """Remove §7.1 L1 banned words. Returns ``(clean_text, removed_words_as_written)``; unchanged text if none."""
    if not text:
        return text or "", []
    removed: list[str] = []

    def repl(m: re.Match[str]) -> str:
        removed.append(m.group(0))
        before = text[: m.start()]
        at_sentence_start = not before.strip() or re.search(r"[.!?:]\s*$", before) is not None
        return _CAP_MARK if at_sentence_start and m.group(0)[:1].isupper() else ""

    out = _BANNED_RE.sub(repl, text)
    if not removed:
        return text, []
    out = re.sub(re.escape(_CAP_MARK) + r"[\s,;:\-–—]*([^\W\d_])", lambda m: m.group(1).upper(), out)
    out = out.replace(_CAP_MARK, "")
    return _tidy(out), removed


def _split_clauses(text: str) -> list[str]:
    return [c.strip() for c in re.split(r"[;,]\s*|\.(?!\d)\s*", text or "") if c.strip()]


def split_lighting_text(text: str) -> tuple[str, str, str]:
    """Split a legacy lighting/lens prompt into ``(lighting, palette, lens)`` honouring L7.

    Lens / aspect clauses go to ``lens``; colour words are moved out of the lighting clauses into a palette
    (``"amber, teal tones"``) so that lighting names physical sources only and the palette names no source.
    """
    lighting: list[str] = []
    lens: list[str] = []
    colours: list[str] = []
    for clause in _split_clauses(text):
        if _LENS_RE.search(clause):
            lens.append(clause)
            continue
        for m in _COLOUR_RE.finditer(clause):
            word = m.group(1).lower()
            if word not in colours:
                colours.append(word)
        cleaned = _tidy(_COLOUR_RE.sub("", clause))
        if re.search(r"[A-Za-z]{3,}", cleaned):
            lighting.append(cleaned)
    palette = f"{', '.join(colours)} tones" if colours else ""
    return ", ".join(lighting), palette, ", ".join(lens)


def _norm_key(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value or "")).strip("_").upper() or "DEFAULT"


def _suffix(entity_id: str, prefix: str) -> str:
    return entity_id[len(prefix):] if entity_id.startswith(prefix) else entity_id


def _norm_name(text: str) -> str:
    t = (text or "").replace("\u2019", "'").replace("\u2018", "'").lower()
    return re.sub(r"\s+", " ", t).strip()


def _strip_parens(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", " ", text or "")).strip()


def _has_devanagari(text: str | None) -> bool:
    return bool(text and _DEVANAGARI_RE.search(text))


def _positive_mentions(text: str, cue: str) -> list[str]:
    """Occurrences of ``cue`` in ``text`` that are not negated ("No music", "without echo")."""
    hits: list[str] = []
    for m in re.finditer(rf"(?<![A-Za-z]){re.escape(cue)}(?![A-Za-z])", text or "", re.IGNORECASE):
        clause_start = max(text.rfind(".", 0, m.start()), text.rfind(";", 0, m.start()), 0)
        window = text[max(clause_start, m.start() - 40): m.start()]
        if _NEGATION_RE.search(window):
            continue
        lo, hi = max(0, m.start() - 30), min(len(text), m.end() + 30)
        hits.append(text[lo:hi].strip())
    return hits


def _first_sentence(text: str, limit: int = 200) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    m = re.search(r"(?<=[.!?])\s", text)
    first = text[: m.start()] if m else text
    return first if len(first) <= limit else first[:limit].rsplit(" ", 1)[0]


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False


def _first_existing(candidates: Iterable[Path]) -> Path | None:
    for cand in candidates:
        try:
            if cand.is_file() and not cand.is_symlink():
                return cand
        except OSError:
            continue
    return None


def _link_or_copy(src: Path, dest: Path, *, allowed: tuple[str, ...]) -> Path:
    """Bring a legacy file INTO the project: hard link on the same filesystem, else copy. Never overwrites.

    Security (§0): regular files only, extension allow-list, size cap.
    """
    src = Path(src)
    if src.is_symlink() or not src.is_file():
        raise ValueError(f"legacy media {src} is not a regular file")
    if src.suffix.lower() not in allowed:
        raise ValueError(f"legacy media {src} has a disallowed extension")
    if src.stat().st_size > MAX_IMPORT_BYTES:
        raise ValueError(f"legacy media {src} exceeds the import size cap")
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)
    return dest


# ---------------------------------------------------------------------------------------------
# Report collector
# ---------------------------------------------------------------------------------------------
@dataclass
class _Report:
    banned: list[str] = field(default_factory=list)
    conflicts: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def removed(self, words: list[str], where: str) -> None:
        for w in words:
            self.banned.append(f"Removed “{w}” from {where}.")

    def conflict(self, section: str, text: str) -> None:
        bucket = self.conflicts.setdefault(section, [])
        if text not in bucket:
            bucket.append(text)


def _clean(text: Any, where: str, report: _Report) -> str:
    """Strip banned words from a migrated prompt field and record each removal."""
    clean, removed = strip_banned_words(str(text or "").strip())
    if removed:
        report.removed(removed, where)
    return clean


# ---------------------------------------------------------------------------------------------
# Legacy sources (imported read-only; legacy files are never edited)
# ---------------------------------------------------------------------------------------------
@dataclass
class LegacySources:
    workspace: Path
    asset_dir: Path
    cast: dict[str, Any]
    locations: dict[str, Any]
    props: dict[str, Any]
    anchors: dict[str, Any]
    shots: list[Any]
    directions: dict[str, Any]
    locks: dict[int, Any]
    voices: dict[str, Any]
    plans: dict[str, dict[str, Any]]
    generations: dict[str, dict[str, Any]]
    plan_files: dict[int, Path]
    state_files: dict[int, Path]
    ref_registry: dict[str, dict[str, Any]]
    assembler: dict[str, Any]
    image_model: str | None = None


def _assembler_constants(workspace: Path) -> dict[str, Any]:
    """EPISODE_TIMELINES / trims parsed statically from assemble_microdrama.py (importing it pulls Resolve deps)."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in LEGACY_ASSEMBLER_DEFAULTS.items()}
    path = workspace / "assemble_microdrama.py"
    if not path.is_file():
        return out
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        target = value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            target, value = node.target.id, node.value
        if target in out and value is not None:
            try:
                out[target] = ast.literal_eval(value)  # literals only — never eval
            except (ValueError, SyntaxError):
                pass
    out["EPISODE_TIMELINES"] = {int(k): tuple(v) for k, v in dict(out["EPISODE_TIMELINES"]).items()}
    return out


def _read_json_opt(path: Path) -> Any:
    try:
        return fsio.read_json(path) if path.is_file() else None
    except (OSError, ValueError):
        return None


def load_legacy_sources(workspace: Path | str) -> LegacySources:
    """Import the legacy LFKOD modules + JSON state read-only."""
    ws = Path(workspace).resolve()
    legacy_dir = ws.joinpath(*LEGACY_DIR)
    if not (legacy_dir / "casting_bible.py").is_file():
        raise FileNotFoundError(f"{ws} does not contain the legacy LFKOD project ({'/'.join(LEGACY_DIR)})")
    if str(ws) not in sys.path:
        sys.path.insert(0, str(ws))
    importlib.invalidate_caches()
    cb = importlib.import_module(f"{LEGACY_PACKAGE}.casting_bible")
    episode_modules = [
        importlib.import_module(f"{LEGACY_PACKAGE}.{p.stem}")
        for p in sorted(legacy_dir.glob("episode*_directions.py"))
    ]
    directions: dict[str, Any] = {}
    locks: dict[int, Any] = {}
    for mod in episode_modules:
        for attr_name, value in vars(mod).items():
            if re.fullmatch(r"EP\d+_SHOT_DIRECTIONS", attr_name) and isinstance(value, dict):
                directions.update(value)
            m = re.fullmatch(r"EP(\d+)_PROMPT_LOCKS", attr_name)
            if m:
                locks[int(m.group(1))] = value
    voices: dict[str, Any] = {}
    try:
        dialogue_agent = importlib.import_module("dialogue_agent")
        voices = dict(getattr(dialogue_agent, "DEFAULT_VOICE_PROFILES", None) or {})
    except Exception:  # noqa: BLE001 - optional legacy module; fall back to the casting bible
        voices = {}
    if not voices and callable(getattr(cb, "get_voice_profiles", None)):
        voices = dict(cb.get_voice_profiles())

    asset_dir = ws / "generated_assets"
    plans: dict[str, dict[str, Any]] = {}
    generations: dict[str, dict[str, Any]] = {}
    plan_files: dict[int, Path] = {}
    state_files: dict[int, Path] = {}
    for path in sorted(asset_dir.glob("vertex_veo_ep*_plan.json")):
        m = re.search(r"ep0*(\d+)_plan", path.name)
        data = _read_json_opt(path)
        entries = data if isinstance(data, list) else (data or {}).get("shots", []) if isinstance(data, dict) else []
        for entry in entries:
            if isinstance(entry, dict) and entry.get("shot_id"):
                plans[str(entry["shot_id"])] = entry
        if m:
            plan_files[int(m.group(1))] = path
    for path in sorted(asset_dir.glob("vertex_veo_ep*_state.json")):
        m = re.search(r"ep0*(\d+)_state", path.name)
        data = _read_json_opt(path)
        gens = (data or {}).get("generations", {}) if isinstance(data, dict) else {}
        for sid, entry in (gens or {}).items():
            if isinstance(entry, dict):
                generations[str(sid)] = entry
        if m:
            state_files[int(m.group(1))] = path
    registry_json = _read_json_opt(asset_dir / "reference_sheet_registry.json")
    image_model = None
    for mod in episode_modules:
        image_model = image_model or getattr(mod, "IMAGE_MODEL_NANO_BANANA_PRO", None)
    return LegacySources(
        workspace=ws,
        asset_dir=asset_dir,
        cast=dict(getattr(cb, "CAST_BIBLE", {}) or {}),
        locations=dict(getattr(cb, "LOCATION_BIBLE", {}) or {}),
        props=dict(getattr(cb, "HERO_PROPS", {}) or {}),
        anchors=dict(getattr(cb, "MASTER_ANCHOR_SHEETS", {}) or {}),
        shots=list(getattr(cb, "SHOT_BREAKDOWN", []) or []),
        directions=directions,
        locks=locks,
        voices=voices,
        plans=plans,
        generations=generations,
        plan_files=plan_files,
        state_files=state_files,
        ref_registry=registry_json if isinstance(registry_json, dict) else {},
        assembler=_assembler_constants(ws),
        image_model=image_model,
    )


def _legacy_video_path(src: LegacySources, legacy_id: str) -> Path:
    """Exactly the file ``assemble_microdrama.episode_table()`` probes for a legacy shot."""
    return src.asset_dir / "episodes" / f"{legacy_id.lower()}.mp4"


def _anchor_image(src: LegacySources, anchor_id: str | None) -> Path | None:
    if not anchor_id:
        return None
    sheet = src.anchors.get(anchor_id)
    names = [anchor_id.lower()]
    image_name = str(_attr(sheet, "image_name", "") or "") if sheet is not None else ""
    if image_name:
        names.append(Path(image_name).name)
    cands: list[Path] = []
    for base in (src.asset_dir / "anchors", src.asset_dir):
        for name in names:
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                cands.append(base / name)
            else:
                cands.extend(base / f"{name}{ext}" for ext in IMAGE_EXTENSIONS)
    return _first_existing(cands)


def _registry_image(src: LegacySources, entity_id: str) -> Path | None:
    entry = src.ref_registry.get(entity_id) or {}
    raw = entry.get("file_path") if isinstance(entry, dict) else None
    if not raw:
        return None
    path = Path(str(raw))
    if not _inside(path, src.workspace) or path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    return _first_existing([path])


# ---------------------------------------------------------------------------------------------
# Speaker resolution (legacy display strings -> character ids)
# ---------------------------------------------------------------------------------------------
class SpeakerResolver:
    """Legacy ``DialogueLine.speaker`` strings ("Rishi's Father", "Off-screen ... Phone Speaker Voice-Over") -> ids.

    Aliases come from casting-bible names, voice-profile speaker names and id suffixes (paren-stripped, longest
    first, ambiguous aliases dropped). Unmatched phone / voice-over / device voices resolve to the Master's voice.
    """

    def __init__(self, cast: dict[str, Any], voices: dict[str, Any]) -> None:
        candidates: dict[str, set[str]] = {}

        def add(alias: str, cid: str) -> None:
            a = _norm_name(alias).strip(" .,:;-")
            if len(a) >= 3 and a not in _GENERIC_ALIAS_WORDS | {"night", "phone", "speaker"}:
                candidates.setdefault(a, set()).add(cid)

        for cid, spec in cast.items():
            self._names(str(cid), str(_attr(spec, "name", "") or ""), add)
        for cid, vp in voices.items():
            self._names(str(cid), str(_attr(vp, "speaker_name", "") or ""), add)
        self.aliases = sorted(
            ((a, next(iter(ids))) for a, ids in candidates.items() if len(ids) == 1),
            key=lambda item: (-len(item[0]), item[0]),
        )
        self.ambiguous = sorted(a for a, ids in candidates.items() if len(ids) > 1)
        self.master_id = MASTER_VOICE_ID if (MASTER_VOICE_ID in voices or MASTER_VOICE_ID in cast) else None

    @staticmethod
    def _names(cid: str, name: str, add: Any) -> None:
        clean = _strip_parens(name)
        if clean:
            add(clean, cid)
            words = [w for w in _norm_name(clean).split(" ") if w]
            for n in range(1, len(words)):
                for i in range(len(words) - n + 1):
                    gram = words[i:i + n]
                    if all(w in _GENERIC_ALIAS_WORDS for w in gram):
                        continue
                    add(" ".join(gram), cid)
        suffix_words = [w for w in (cid.split("_", 1)[1] if "_" in cid else cid).lower().split("_") if w]
        if suffix_words:
            add(" ".join(suffix_words), cid)
            for w in suffix_words:
                if w not in _GENERIC_ALIAS_WORDS:
                    add(w, cid)

    def resolve(self, speaker: str) -> tuple[str | None, bool, str]:
        """``(character_id | None, off_screen, how)``."""
        full = _norm_name(speaker)
        core = _norm_name(_strip_parens(speaker))
        off = any(mark in full for mark in _OFF_SCREEN_MARKERS)
        for alias, cid in self.aliases:
            if re.search(rf"(?<![a-z0-9']){re.escape(alias)}(?![a-z0-9'])", core):
                return cid, off or cid == self.master_id, f"name match “{alias}”"
        if off and self.master_id:
            return self.master_id, True, "phone / voice-over / device voice → the Master's voice"
        return None, off, "unresolved"


# ---------------------------------------------------------------------------------------------
# In-memory migration plan
# ---------------------------------------------------------------------------------------------
@dataclass
class _LinePlan:
    speaker: str
    character_id: str
    off_screen: bool
    text: str
    transliteration: str | None
    delivery: str
    legacy_voice: str
    how: str
    line_id: str = ""


@dataclass
class _ShotPlan:
    index: int  # position in SHOT_BREAKDOWN
    legacy_id: str
    episode: int
    number: int
    spec: Any
    direction: Any
    location_id: str
    characters: list[str]
    states: dict[str, str]  # entity id -> state id (on-screen characters + props)
    lines: list[_LinePlan]
    duration_s: float
    duration_frames: int
    multi_beat: bool = False
    scene_id: str = ""
    shot_id: str = ""
    clip_id: str = ""
    take_id: str = ""


@dataclass
class _ScenePlan:
    scene_id: str
    order: int
    episode: int
    location_id: str
    slug: str
    shots: list[_ShotPlan] = field(default_factory=list)

    def states_anywhere(self) -> list[str]:
        out: list[str] = []
        for sp in self.shots:
            for st in sp.states.values():
                if st not in out:
                    out.append(st)
        return out


def _wardrobe_key(shot: Any, cid: str, spec: Any) -> str:
    by_char = dict(_attr(shot, "wardrobe_state_by_character", {}) or {})
    wardrobe = dict(_attr(spec, "wardrobe_by_state", {}) or {}) if spec is not None else {}
    key = by_char.get(cid)
    if key and (not wardrobe or key in wardrobe):
        return str(key)
    episode = _attr(shot, "episode")
    for k in wardrobe:
        if f"ep{episode}" in str(k).lower():
            return str(k)
    return str(next(iter(wardrobe), "default"))


def _plan_shots(src: LegacySources, resolver: SpeakerResolver, report: _Report) -> list[_ShotPlan]:
    ordered = sorted(
        enumerate(src.shots),
        key=lambda iv: (_int_or_none(_attr(iv[1], "episode")) or 0, _int_or_none(_attr(iv[1], "shot_number")) or 0,
                        iv[0]),
    )
    plans: list[_ShotPlan] = []
    for index, shot in ordered:
        sid = str(_attr(shot, "shot_id"))
        episode = _int_or_none(_attr(shot, "episode")) or 0
        number = _int_or_none(_attr(shot, "shot_number")) or index + 1
        direction = src.directions.get(sid)
        seconds = float(_attr(shot, "duration_seconds", 0) or 0)
        frames = _int_or_none(_attr(shot, "duration_frames")) or int(round((seconds or 8) * LEGACY_FPS))
        chars = [str(c) for c in (_attr(shot, "character_ids", []) or [])]
        raw_lines = list(_attr(direction, "dialogue", []) or []) if direction is not None else []
        if not raw_lines:
            audio = _attr(shot, "audio")
            spoken = str(_attr(audio, "dialogue_hinglish", "") or "").strip() if audio is not None else ""
            if spoken:
                subtitle = str(_attr(audio, "subtitle_text", "") or "")
                raw_lines = [{"speaker": _attr(audio, "speaker", ""), "line_hinglish": spoken,
                              "line_hindi": subtitle if _has_devanagari(subtitle) else "", "delivery": "",
                              "voice_profile": ""}]
        lines: list[_LinePlan] = []
        for ln in raw_lines:
            speaker = str(_attr(ln, "speaker", "") or "").strip()
            cid, off, how = resolver.resolve(speaker)
            if cid is None:
                cid = "CHAR_" + _norm_key(_strip_parens(speaker) or "UNKNOWN_SPEAKER")
                report.conflict("Speakers I could not match to a character",
                                f"{sid}: “{speaker}” → placeholder {cid} (please confirm who speaks).")
            hindi = str(_attr(ln, "line_hindi", "") or "").strip()
            hinglish = str(_attr(ln, "line_hinglish", "") or "").strip()
            text = hindi if _has_devanagari(hindi) else (hindi or hinglish)
            if not _has_devanagari(text):
                report.conflict("Lines without Devanagari (workspace rule: exact Hindi in Devanagari)",
                                f"{sid} {cid}: “{text}” — please supply the Devanagari line.")
            if not off and cid not in chars:
                report.conflict("On-camera speakers missing from the legacy cast list of a shot",
                                f"{sid}: {cid} speaks on camera but is not in character_ids — added to the shot.")
                chars.append(cid)
            lines.append(_LinePlan(
                speaker=speaker, character_id=cid, off_screen=off, text=text,
                transliteration=hinglish if hinglish and hinglish != text else None,
                delivery=str(_attr(ln, "delivery", "") or "").strip(),
                legacy_voice=str(_attr(ln, "voice_profile", "") or "").strip(), how=how,
            ))
        states: dict[str, str] = {}
        for cid in chars:
            states[cid] = f"{cid}_STATE_{_norm_key(_wardrobe_key(shot, cid, src.cast.get(cid)))}"
        for pid in (_attr(shot, "prop_ids", []) or []):
            states[str(pid)] = f"{pid}_STATE_DEFAULT"
        camera = " ".join(str(x or "") for x in (_attr(direction, "camera_motion", ""),
                                                 _attr(shot, "camera_framing_and_motion", "")))
        plans.append(_ShotPlan(
            index=index, legacy_id=sid, episode=episode, number=number, spec=shot, direction=direction,
            location_id=str(_attr(shot, "location_id", "") or ""), characters=chars, states=states, lines=lines,
            duration_s=seconds or frames / LEGACY_FPS, duration_frames=frames,
            multi_beat=bool(_MULTI_BEAT_RE.search(camera)),
        ))
    return plans


def _plan_scenes(shots: list[_ShotPlan]) -> list[_ScenePlan]:
    """One scene per contiguous location run inside each episode; assigns shot/clip/take/line ids."""
    scenes: list[_ScenePlan] = []
    slug_uses: dict[str, int] = {}
    for sp in shots:
        cur = scenes[-1] if scenes else None
        if cur is None or cur.episode != sp.episode or cur.location_id != sp.location_id:
            n = len(scenes) + 1
            base = f"ep{sp.episode}_{_suffix(sp.location_id, 'LOC_').lower() or 'scene'}"
            slug_uses[base] = slug_uses.get(base, 0) + 1
            slug = base if slug_uses[base] == 1 else f"{base}_{slug_uses[base]}"
            cur = _ScenePlan(scene_id=make_scene_id(n), order=n, episode=sp.episode, location_id=sp.location_id,
                             slug=slug)
            scenes.append(cur)
        cur.shots.append(sp)
        sp.scene_id = cur.scene_id
        sp.shot_id = make_shot_id(cur.scene_id, len(cur.shots))
        sp.clip_id = make_clip_id(sp.shot_id, 1)
        sp.take_id = make_take_id(sp.clip_id, 1)
    for sc in scenes:
        n = 0
        for sp in sc.shots:
            for ln in sp.lines:
                n += 1
                ln.line_id = f"{sc.scene_id}_L{n:03d}"
    return scenes


def _scene_states(sc: _ScenePlan) -> tuple[list[str], list[MidSceneTransition], dict[str, str], dict[str, str]]:
    """(entity_states_active at scene start, mid-scene transitions, entry states, exit states)."""
    start: dict[str, str] = {}
    current: dict[str, str] = {}
    transitions: list[MidSceneTransition] = []
    for sp in sc.shots:
        for eid, st in sp.states.items():
            if eid not in current:
                start[eid] = current[eid] = st
            elif current[eid] != st:
                transitions.append(MidSceneTransition(
                    entity_id=eid, from_state=current[eid], to_state=st, at_shot_id=sp.shot_id, at_clip_index=1,
                    note=f"legacy wardrobe change at {sp.legacy_id}",
                ))
                current[eid] = st
    return list(start.values()), transitions, dict(start), dict(current)


def _design_rel(scene_id: str, path: Path) -> str:
    return f"scenes/{scene_id}/design/{path.name}"


def _keyframe_path(src: LegacySources, sp: _ShotPlan) -> Path | None:
    sid = sp.legacy_id.lower()
    cands = [src.asset_dir / "episodes" / f"{sid}{ext}" for ext in IMAGE_EXTENSIONS]
    image_name = Path(str(_attr(sp.spec, "image_name", "") or "")).name
    if image_name:
        if Path(image_name).suffix.lower() in IMAGE_EXTENSIONS:
            cands += [src.asset_dir / "episodes" / image_name, src.asset_dir / image_name]
        else:
            cands += [src.asset_dir / "episodes" / f"{image_name}{ext}" for ext in IMAGE_EXTENSIONS]
    recorded = (src.generations.get(sp.legacy_id) or {}).get("keyframe") or (src.plans.get(sp.legacy_id) or {}).get(
        "keyframe")
    if recorded and _inside(Path(str(recorded)), src.workspace):
        cands.append(Path(str(recorded)))
    return _first_existing(cands)


# ---------------------------------------------------------------------------------------------
# Style Lock (§4.1) from the episode prompt locks — banned words removed, audio negatives added
# ---------------------------------------------------------------------------------------------
def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = (item or "").strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            out.append(text)
    return out


def _voice_text(vp: Any) -> str:
    return " . ".join(str(_attr(vp, k, "") or "") for k in (
        "vocal_gender_age_tag", "pitch_band", "timbre_and_texture", "accent_and_dialect", "vocal_quirks"))


def _shot_audio_text(sp: _ShotPlan) -> str:
    parts = [str(_attr(sp.direction, "sound_design", "") or "")] if sp.direction is not None else []
    for ln in sp.lines:
        parts += [ln.legacy_voice, ln.delivery]
    audio = _attr(sp.spec, "audio")
    if audio is not None:
        parts.append(str(_attr(audio, "foley_sfx_prompt", "") or ""))
    return " . ".join(p for p in parts if p)


def _negative_evidence(term: str, text: str) -> list[str]:
    hits: list[str] = []
    for cue in NEGATIVE_CONFLICT_CUES.get(term, (term,)):
        hits += _positive_mentions(text, cue)
    return hits


def _build_style_lock(src: LegacySources, shots: list[_ShotPlan], report: _Report
                      ) -> tuple[StyleLock, dict[int, list[str]], dict[str, list[str]]]:
    """Returns ``(style_lock, episode-specific negatives {episode: [...]}, conflicting audio negatives)``."""
    episodes = sorted(src.locks)
    style_texts: list[str] = []
    for ep in episodes:
        text = str(_attr(src.locks[ep], "style", "") or "").strip()
        if text and text not in style_texts:
            style_texts.append(text)
    style_text = " ".join(style_texts)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", style_text) if s.strip()]
    medium = _clean(sentences[0] if sentences else "", "the style lock — medium (legacy EP style lock)", report)
    lens_language = _clean(" ".join(sentences[1:]), "the style lock — lens language (legacy EP style lock)", report)
    aspect = "16:9" if ("16:9" in style_text and "9:16" not in style_text) else "9:16"
    donts: list[str] = []
    for ep in episodes:
        for lock_field in ("continuity", "language"):
            for sent in re.split(r"(?<=[.!?])\s+", str(_attr(src.locks[ep], lock_field, "") or "")):
                sent = sent.strip()
                if sent and re.match(r"(no|never|strictly|do not|don't)\b", sent, re.IGNORECASE):
                    donts.append(_clean(sent, f"the style lock — don'ts (EP{ep} {lock_field} lock)", report))
    per_episode: dict[int, list[str]] = {}
    for ep in episodes:
        raw = str(_attr(src.locks[ep], "global_negative", "") or "")
        items = [_clean(t, f"the EP{ep} global negative prompt", report) for t in re.split(r"[,\n]", raw)]
        per_episode[ep] = _dedupe(items)
    common = set.intersection(*[{n.lower() for n in v} for v in per_episode.values()]) if per_episode else set()
    global_negative = [n for n in (per_episode[episodes[0]] if episodes else []) if n.lower() in common]
    episode_specific = {ep: [n for n in negs if n.lower() not in common] for ep, negs in per_episode.items()}
    conflicts: dict[str, list[str]] = {}
    for term in V3_AUDIO_NEGATIVES:
        evidence: list[str] = []
        for sp in shots:
            evidence += [f"{sp.legacy_id}: “…{hit}…”" for hit in _negative_evidence(term, _shot_audio_text(sp))]
        for cid, vp in src.voices.items():
            evidence += [f"voice profile {cid}: “…{hit}…”" for hit in _negative_evidence(term, _voice_text(vp))]
        if evidence:
            conflicts[term] = _dedupe(evidence)
        elif term.lower() not in {n.lower() for n in global_negative}:
            global_negative.append(term)
    style = StyleLock(
        medium=medium or "live-action",
        aspect_ratio=aspect,
        lens_language=lens_language,
        donts=_dedupe(donts),
        global_negative=global_negative,
        approval_status=ApprovalStatus.PROPOSED,
        derived_from="legacy:episode*_directions.EP*_PROMPT_LOCKS",
        generated_by=fsio.generated_by(AGENT, notes="Phase 0 import of the legacy prompt locks; L1 banned words "
                                                    "removed; lock at G1 (Production Designer)"),
    )
    return style, episode_specific, conflicts


# ---------------------------------------------------------------------------------------------
# Voices (§4.3), entities + states (§4.2), environments (§4.4)
# ---------------------------------------------------------------------------------------------
def _voice_description(vp: Any) -> str:
    cadence = _attr(vp, "default_cadence")
    cadence = str(getattr(cadence, "value", cadence) or "").replace("_", " ")
    wpm = _attr(vp, "target_wpm")
    pitch = str(_attr(vp, "pitch_band", "") or "").strip()
    parts = [
        str(_attr(vp, "vocal_gender_age_tag", "") or "").strip(),
        f"{pitch} pitch" if pitch and "pitch" not in pitch.lower() else pitch,
        str(_attr(vp, "timbre_and_texture", "") or "").strip(),
        str(_attr(vp, "accent_and_dialect", "") or "").strip(),
        (f"{cadence} cadence" if cadence else "") + (f" at about {wpm} words per minute" if wpm else ""),
        str(_attr(vp, "vocal_quirks", "") or "").strip(),
    ]
    return "; ".join(p.strip() for p in parts if p and p.strip())


def _build_voices(src: LegacySources, report: _Report) -> dict[str, VoiceProfile]:
    out: dict[str, VoiceProfile] = {}
    for cid, vp in src.voices.items():
        cid = str(cid)
        try:
            fsio.validate_id(cid, "character id")
        except Exception:  # noqa: BLE001 - report and skip malformed legacy ids
            report.conflict("Legacy records skipped", f"voice profile with invalid id {cid!r}")
            continue
        out[cid] = VoiceProfile(
            id=f"VOICE_{cid}", character_id=cid,
            description=_clean(_voice_description(vp), f"the voice profile of {cid}", report),
            language=HINDI_LANGUAGE, approval_status=ApprovalStatus.PROPOSED,
            derived_from=f"legacy:dialogue_agent.DEFAULT_VOICE_PROFILES[{cid}]",
            generated_by=fsio.generated_by(AGENT, notes="Phase 0 import; lock at G1 (Casting Director)"),
        )
        spec = src.cast.get(cid)
        if spec is None:
            continue
        age = _int_or_none(_attr(spec, "age"))
        tag = str(_attr(vp, "vocal_gender_age_tag", "") or "")
        ages = [int(x) for x in re.findall(r"\b(\d{1,3})[- ]year", tag)]
        if age is not None and ages and age not in ages:
            report.conflict("Voice profiles vs the casting bible",
                            f"{cid}: casting bible age {age}, voice profile says “{tag}”.")
        vname, cname = str(_attr(vp, "speaker_name", "") or ""), str(_attr(spec, "name", "") or "")
        if vname and cname and _norm_name(vname) not in _norm_name(cname):
            report.conflict("Voice profiles vs the casting bible",
                            f"{cid}: casting bible name “{cname}”, voice profile speaker name “{vname}”.")
    return out


def _tier_for(cid: str, shots: list[_ShotPlan]) -> Tier:
    n_shots = sum(1 for sp in shots if cid in sp.characters)
    n_lines = sum(1 for sp in shots for ln in sp.lines if ln.character_id == cid)
    if n_shots >= 6:
        return Tier.PRINCIPAL
    if n_shots >= 3 or n_lines >= 3:
        return Tier.SUPPORTING
    return Tier.INCIDENTAL


def _sheet_mentions(sheet: Any, words: Iterable[str]) -> bool:
    prompt = _norm_name(str(_attr(sheet, "prompt", "") or ""))
    return any(re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", prompt) for w in words if w)


def _build_entities(src: LegacySources, shots: list[_ShotPlan], scenes: list[_ScenePlan],
                    voices: dict[str, VoiceProfile], resolver: SpeakerResolver, report: _Report
                    ) -> tuple[list[Entity], dict[str, Path]]:
    """Entities + wardrobe states; returns ``(entities, {state_id: legacy reference image})``."""
    scene_states = {sc.scene_id: set(sc.states_anywhere()) for sc in scenes}
    used_states: list[str] = []
    for sp in shots:
        for st in sp.states.values():
            if st not in used_states:
                used_states.append(st)
    state_images: dict[str, Path] = {}
    entities: list[Entity] = []
    image_model = src.image_model

    def scenes_for(state_id: str) -> list[str]:
        return [sc.scene_id for sc in scenes if state_id in scene_states[sc.scene_id]]

    def pack_for(state_id: str, image: Path | None, tier: Tier, note: str, *, never_on_screen: bool = False
                 ) -> ReferencePack:
        active = scenes_for(state_id)
        if never_on_screen:
            return ReferencePack(text_only=True, generated_by=fsio.generated_by(AGENT, notes=note))
        if image is not None and active:
            state_images[state_id] = image
            return ReferencePack(single_ref=_design_rel(active[0], image),
                                 generated_by=fsio.generated_by(AGENT, model=image_model, notes=note))
        return ReferencePack(text_only=tier == Tier.INCIDENTAL, generated_by=fsio.generated_by(AGENT, notes=note))

    for cid, spec in src.cast.items():
        cid = str(cid)
        tier = _tier_for(cid, shots)
        wardrobe = {str(k): str(v or "") for k, v in (dict(_attr(spec, "wardrobe_by_state", {}) or {})).items()}
        keys = list(wardrobe) or ["default"]
        state_ids = [f"{cid}_STATE_{_norm_key(k)}" for k in keys]
        for st in used_states:  # states referenced by shots but missing from wardrobe_by_state
            if st.startswith(cid + "_STATE_") and st not in state_ids:
                state_ids.append(st)
                keys.append(st[len(cid) + 7:])
        sheet_id = str(_attr(spec, "anchor_sheet_id", "") or "")
        sheet = src.anchors.get(sheet_id) if sheet_id else None
        image = _anchor_image(src, sheet_id) or _registry_image(src, cid)
        name = str(_attr(spec, "name", "") or cid)
        if sheet is not None:
            words = [w for w in _norm_name(_strip_parens(name)).split(" ") if w not in _GENERIC_ALIAS_WORDS]
            words += [w for w in cid.split("_", 1)[-1].lower().split("_") if w not in _GENERIC_ALIAS_WORDS]
            if not _sheet_mentions(sheet, words):
                report.conflict("Reference sheets",
                                f"{cid} ({name}) is assigned to {sheet_id}, but that sheet's prompt never mentions "
                                f"them — they may not actually be on the sheet; the Character Designer should "
                                f"check before G2.")
        elif image is None:
            report.conflict("Reference sheets", f"{cid} ({name}) has no reference sheet image.")
        states: list[EntityState] = []
        for key, st in zip(keys, state_ids):
            note = (f"legacy {'ensemble ' if sheet is not None else ''}reference sheet "
                    f"{sheet_id or (image.name if image else 'none')} (proposed; promote at G2)")
            states.append(EntityState(
                id=st,
                derived_from=state_ids[0] if st != state_ids[0] else None,
                delta_description=_clean(wardrobe.get(key, ""), f"{cid} wardrobe state {key}", report),
                reference_pack=pack_for(st, image, tier, note),
                scenes_active=scenes_for(st),
                approval_status=ApprovalStatus.PROPOSED,
            ))
            if not scenes_for(st):
                report.notes.append(f"{st} exists in the casting bible but no legacy shot uses it.")
        age = _attr(spec, "age")
        physical = _clean(_attr(spec, "immutable_physical_prompt", ""), f"{cid} physical description", report)
        entities.append(Entity(
            id=cid, type=EntityType.CHARACTER, tier=tier,
            canonical_description=(f"Age {age}. {physical}" if age not in (None, "") else physical) or name,
            display_name=name, voice_profile_id=voices[cid].id if cid in voices else None, states=states,
            approval_status=ApprovalStatus.PROPOSED, derived_from=f"legacy:casting_bible.CAST_BIBLE[{cid}]",
            generated_by=fsio.generated_by(AGENT, notes="Phase 0 import; lock at G1"),
        ))
        if tier != Tier.INCIDENTAL and image is not None:
            report.conflict("Reference sheets",
                            f"{cid} is {tier.value}: the legacy ensemble sheet stands in as a single reference; "
                            f"v3 wants a turnaround + face close-up (Character Designer) before G2.")

    for pid, spec in src.props.items():
        pid = str(pid)
        st = f"{pid}_STATE_DEFAULT"
        sheet_id = str(_attr(spec, "anchor_sheet_id", "") or "")
        image = _anchor_image(src, sheet_id) or _registry_image(src, pid)
        if image is None:
            report.conflict("Reference sheets", f"{pid} ({_attr(spec, 'name', pid)}) has no reference image yet — "
                                                f"described in text only until the Character Designer makes one.")
        entities.append(Entity(
            id=pid, type=EntityType.PROP, tier=Tier.SUPPORTING,
            canonical_description=_clean(_attr(spec, "visual_prompt", ""), f"{pid} description", report) or pid,
            display_name=str(_attr(spec, "name", "") or pid),
            states=[EntityState(id=st, reference_pack=pack_for(st, image, Tier.SUPPORTING, "legacy hero prop"),
                                scenes_active=scenes_for(st), approval_status=ApprovalStatus.PROPOSED)],
            approval_status=ApprovalStatus.PROPOSED, derived_from=f"legacy:casting_bible.HERO_PROPS[{pid}]",
            generated_by=fsio.generated_by(AGENT, notes="Phase 0 import; lock at G1"),
        ))

    known = {e.id for e in entities}
    speakers: list[str] = []
    for sp in shots:
        for ln in sp.lines:
            if ln.character_id not in known and ln.character_id not in speakers:
                speakers.append(ln.character_id)
    for cid in speakers:  # the Master's voice + any placeholder speakers
        vp_legacy = src.voices.get(cid)
        never_on_screen = all(ln.off_screen for sp in shots for ln in sp.lines if ln.character_id == cid)
        name = str(_attr(vp_legacy, "speaker_name", "") or "") or cid.split("_", 1)[-1].replace("_", " ").title()
        desc = (f"Off-screen voice only — never seen on camera. Voice: {voices[cid].description}"
                if cid in voices and never_on_screen else
                f"Speaker imported from the legacy dialogue as “{name}” — needs a description.")
        st = f"{cid}_STATE_DEFAULT"
        entities.append(Entity(
            id=cid, type=EntityType.CHARACTER, tier=Tier.INCIDENTAL, canonical_description=desc, display_name=name,
            voice_profile_id=voices[cid].id if cid in voices else None,
            states=[EntityState(id=st, delta_description="voice only" if never_on_screen else "",
                                reference_pack=pack_for(st, None, Tier.INCIDENTAL, "voice-only character",
                                                        never_on_screen=never_on_screen),
                                scenes_active=[sc.scene_id for sc in scenes
                                               if any(ln.character_id == cid for s in sc.shots for ln in s.lines)],
                                approval_status=ApprovalStatus.PROPOSED)],
            approval_status=ApprovalStatus.PROPOSED,
            derived_from=f"legacy:dialogue_agent.DEFAULT_VOICE_PROFILES[{cid}]" if vp_legacy is not None else
            "legacy:episode directions dialogue",
            generated_by=fsio.generated_by(AGENT, notes="Phase 0 import; lock at G1"),
        ))
    if resolver.ambiguous:
        report.notes.append("Ambiguous speaker names ignored when matching: " + ", ".join(resolver.ambiguous))
    return entities, state_images


def _tod_key(time_of_day: str) -> str:
    t = (time_of_day or "").lower()
    for word, key in (("night", "NIGHT"), ("dawn", "DAWN"), ("dusk", "DUSK"), ("sunset", "DUSK"),
                      ("evening", "EVENING"), ("morning", "MORNING"), ("afternoon", "DAY"), ("noon", "DAY"),
                      ("day", "DAY")):
        if word in t:
            return key
    return _norm_key(time_of_day)[:24] if time_of_day else "DEFAULT"


@dataclass
class _EnvLook:
    env_id: str
    lighting_setup_id: str
    time_of_day: str
    lighting: str
    palette: str
    lens: str
    plate: Path | None


def _build_environments(src: LegacySources, shots: list[_ShotPlan], scenes: list[_ScenePlan], report: _Report
                        ) -> tuple[list[Environment], dict[str, _EnvLook]]:
    looks: dict[str, _EnvLook] = {}
    envs: list[Environment] = []
    loc_ids = list(src.locations)
    for sp in shots:
        if sp.location_id and sp.location_id not in loc_ids:
            loc_ids.append(sp.location_id)
            report.conflict("Locations", f"{sp.legacy_id} uses {sp.location_id}, which is not in LOCATION_BIBLE.")
    for loc in loc_ids:
        spec = src.locations.get(loc)
        suffix = _suffix(loc, "LOC_")
        env_id = f"ENV_{suffix}"
        tod = str(_attr(spec, "time_of_day", "") or "")
        ls_id = f"LS_{suffix}_{_tod_key(tod)}"
        name = str(_attr(spec, "name", "") or loc)
        light_raw = _clean(_attr(spec, "lighting_and_lens_prompt", ""), f"{env_id} lighting & lens prompt", report)
        lighting, palette, lens = split_lighting_text(light_raw)
        arch = _clean(_attr(spec, "architectural_prompt", ""), f"{env_id} architecture prompt", report)
        plate_id = str(_attr(spec, "anchor_plate_id", "") or "")
        plate = _anchor_image(src, plate_id) or _registry_image(src, loc)
        sheet = src.anchors.get(plate_id) if plate_id else None
        if sheet is not None:
            words = [w for w in re.findall(r"[a-z]{4,}", _norm_name(name.split("–")[0]))
                     if w not in {"rishi's", "under", "construction", "with"}]
            if words and not _sheet_mentions(sheet, words):
                report.conflict("Reference sheets",
                                f"{env_id} ({name}) is assigned to {plate_id}, but that plate's prompt never "
                                f"mentions it — the location may be missing from the plate; the Location Designer "
                                f"should check before G2.")
        at_loc = [sp for sp in shots if sp.location_id == loc]
        first_scene = next((sc.scene_id for sc in scenes if sc.location_id == loc), None)
        candidates: list[str] = []
        if first_scene and plate is not None:
            candidates.append(_design_rel(first_scene, plate))
        if first_scene and at_loc:
            kf = _keyframe_path(src, at_loc[0])
            if kf is not None:
                candidates.append(_design_rel(first_scene, kf))
        sound = str(_attr(at_loc[0].direction, "sound_design", "") or "") if at_loc and at_loc[0].direction else ""
        room_tone = _clean(_first_sentence(sound), f"{env_id} room tone (from {at_loc[0].legacy_id})"
                           if at_loc else f"{env_id} room tone", report) if sound else ""
        if not at_loc:
            report.notes.append(f"{env_id} ({name}) is in the location bible but no legacy shot uses it.")
        envs.append(Environment(
            id=env_id,
            canonical_description=_tidy(f"{name}. {arch}") if arch else name,
            geometry_candidates=candidates[:4],
            lighting_setups=[LightingSetup(id=ls_id, description=lighting or "practical sources as seen in the "
                                                                             "legacy keyframes")],
            room_tone=room_tone,
            approval_status=ApprovalStatus.PROPOSED,
            derived_from=f"legacy:casting_bible.LOCATION_BIBLE[{loc}]" if spec is not None else "legacy:shot",
            generated_by=fsio.generated_by(AGENT, notes="Phase 0 import; lighting colour words moved to the clip "
                                                        "palette (L7); axis of action to be set at G2"),
        ))
        looks[loc] = _EnvLook(env_id=env_id, lighting_setup_id=ls_id, time_of_day=tod, lighting=lighting,
                              palette=palette, lens=lens, plate=plate)
    return envs, looks


# ---------------------------------------------------------------------------------------------
# Scenes, shots, clips, takes
# ---------------------------------------------------------------------------------------------
def _quiet(text: Any) -> str:
    """Banned words removed without a report entry (script text mirrors fields already reported)."""
    return strip_banned_words(str(text or "").strip())[0]


def _scene_script_text(sc: _ScenePlan, src: LegacySources, names: dict[str, str]) -> str:
    """Fountain-like scene text rebuilt from the legacy directions (action + verbatim Devanagari dialogue)."""
    loc = src.locations.get(sc.location_id)
    heading = str(_attr(loc, "name", "") or sc.location_id).strip()
    if not re.match(r"(INT|EXT)", heading, re.IGNORECASE):
        heading = f"{_attr(loc, 'setting_type', 'INT') or 'INT'}. {heading}"
    out = [heading.upper(), ""]
    for sp in sc.shots:
        action = _quiet(_attr(sp.direction, "subject_action", "") or _attr(sp.spec, "action_beat_prompt", ""))
        if action:
            out += [action[0].upper() + action[1:] + ("" if action.endswith((".", "!", "?")) else "."), ""]
        for ln in sp.lines:
            out.append(names.get(ln.character_id, ln.character_id).upper() + (" (V.O.)" if ln.off_screen else ""))
            if ln.delivery:
                out.append(f"({_quiet(ln.delivery)})")
            out.append(ln.text)
            if ln.transliteration:
                out.append(f"[{ln.transliteration}]")
            out.append("")
    return "\n".join(out).strip() + "\n"


def _timeline_for(src: LegacySources, episode: int) -> tuple[str, str, str]:
    """(legacy assembler timeline name, marker colour, the human's current Resolve timeline name)."""
    name, colour = src.assembler["EPISODE_TIMELINES"].get(episode, (f"EP{episode:02d}", "Blue"))
    return name, colour, f"{name}_{LEGACY_TIMELINE_VERSION}"


def _scene_record(sc: _ScenePlan, src: LegacySources, looks: dict[str, _EnvLook], script_text: str) -> Scene:
    active, transitions, entry, exit_states = _scene_states(sc)
    look = looks.get(sc.location_id)
    _name, _colour, timeline = _timeline_for(src, sc.episode)
    last = sc.shots[-1]
    return Scene(
        id=sc.scene_id,
        slug=sc.slug,
        order=sc.order,
        episode=sc.episode,
        script_text=script_text,
        page_range=f"EP{sc.episode} shots {sc.shots[0].number:02d}–{last.number:02d}",
        beats=[Beat(beat_id=f"{sc.scene_id}_B{k}",
                    summary=_quiet(_attr(sp.direction, "intent", "") or _attr(sp.spec, "action_beat_prompt", ""))
                    or sp.legacy_id)
               for k, sp in enumerate(sc.shots, 1)],
        entity_states_active=active,
        environment_id=look.env_id if look else None,
        lighting_setup_id=look.lighting_setup_id if look else None,
        time_of_day=(look.time_of_day or None) if look else None,
        mid_scene_transitions=transitions,
        dialogue=[SceneDialogueLine(line_id=ln.line_id, character_id=ln.character_id, text=ln.text,
                                    parenthetical=ln.delivery or None, transliteration=ln.transliteration,
                                    language="hi", off_screen=ln.off_screen)
                  for sp in sc.shots for ln in sp.lines],
        entry_state={"entity_states": entry, "legacy_shots": [sp.legacy_id for sp in sc.shots]},
        exit_state={"entity_states": exit_states,
                    "continuity_end_state": _quiet(_attr(last.direction, "continuity_end_state", ""))},
        creative_shots=[sp.shot_id for sp in sc.shots],
        status=SceneStatus.HANDED_OFF,
        resolve=SceneResolve(timeline_name=timeline, ownership=Ownership.HUMAN, markers_enabled=True),
        generated_by=fsio.generated_by(AGENT, notes=f"Phase 0: legacy EP{sc.episode} shots "
                                                    f"{', '.join(sp.legacy_id for sp in sc.shots)} "
                                                    f"(one contiguous location run)"),
    )


def _legacy_model(src: LegacySources, legacy_id: str) -> tuple[str, str | None, dict[str, Any]]:
    """(registry model id omni|veo, provider model, legacy generation record)."""
    gen = src.generations.get(legacy_id) or {}
    backend = str(gen.get("backend") or "").lower()
    provider = str(gen.get("model") or "")
    if gen:
        model = "omni" if ("omni" in backend or "omni" in provider.lower()) else "veo"
    else:
        chain = (src.plans.get(legacy_id) or {}).get("model_chain") or []
        first = str(chain[0]) if chain else ""
        model = "omni" if "omni" in first.lower() else "veo"
        provider = provider or first
    return model, provider or None, gen


def legacy_payload_hash(src: LegacySources, legacy_id: str) -> str:
    """Payload hash of a legacy generation, derived from the legacy plan + state JSON (dedupe key)."""
    gen = src.generations.get(legacy_id) or {}
    keys = ("backend", "model", "mode", "prompt", "negative_prompt", "parameters", "keyframe_fingerprint",
            "operation_name")
    return fsio.stable_hash({"legacy_shot_id": legacy_id, "plan": src.plans.get(legacy_id) or {},
                             "generation": {k: gen.get(k) for k in keys}}, length=64)


def _visual_spec(sp: _ShotPlan, look: _EnvLook | None, report: _Report) -> VisualSpec:
    d = sp.direction
    sid = sp.legacy_id
    action = _clean(_attr(d, "subject_action", "") or _attr(sp.spec, "action_beat_prompt", ""),
                    f"{sid} subject & action", report)
    camera = _clean(_attr(d, "camera_motion", ""), f"{sid} camera direction", report) if d is not None else ""
    framing = _clean(_attr(sp.spec, "camera_framing_and_motion", ""), f"{sid} framing", report)
    lens = look.lens if look else ""
    return VisualSpec(
        subject_and_action=action,
        camera=camera or framing,
        lens_and_aspect=", ".join(x for x in (lens, "" if "9:16" in lens else "vertical 9:16") if x),
        lighting=look.lighting if look else "",
        palette=look.palette if look else "",
        composition=framing if camera else "",
        atmosphere=_clean(_attr(d, "scene_motion_timing", ""), f"{sid} motion timing", report) if d is not None
        else "",
        mood=_clean(_attr(d, "intent", ""), f"{sid} intent", report) if d is not None else "",
        attribution="",
    )


def _audio_spec(sp: _ShotPlan, voices: dict[str, VoiceProfile], report: _Report) -> AudioSpec:
    on_screen: list[AudioLine] = []
    off_screen: list[AudioLine] = []
    for ln in sp.lines:
        vp = voices.get(ln.character_id)
        line = AudioLine(
            character_id=ln.character_id, text=ln.text, line_id=ln.line_id, delivery=ln.delivery,
            voice=vp.voice_for_state(sp.states.get(ln.character_id)) if vp else ln.legacy_voice,
            transliteration=ln.transliteration,
        )
        (off_screen if ln.off_screen else on_screen).append(line)
    d = sp.direction
    ambience = _clean(_attr(d, "sound_design", ""), f"{sp.legacy_id} sound design", report) if d is not None else ""
    audio = _attr(sp.spec, "audio")
    foley = _clean(_attr(audio, "foley_sfx_prompt", ""), f"{sp.legacy_id} foley prompt", report) if audio else ""
    return AudioSpec(dialogue=on_screen, off_screen=off_screen, ambience=ambience,
                     diegetic_sfx=[foley] if foley else [], music="none")


def _clip_negative(sp: _ShotPlan, src: LegacySources, global_negative: list[str], episode_negatives: list[str],
                   conflicting: dict[str, list[str]], report: _Report) -> list[str]:
    extra = [_clean(x, f"{sp.legacy_id} extra negatives", report)
             for x in (_attr(sp.direction, "extra_negatives", []) or [])] if sp.direction is not None else []
    local_terms: list[str] = []
    audio_text = _shot_audio_text(sp)
    for term in conflicting:  # conflicting audio negatives still apply where this shot does not use the sound
        clash = _negative_evidence(term, audio_text)
        for ln in sp.lines:
            vp = src.voices.get(ln.character_id)
            if vp is not None:
                clash += _negative_evidence(term, _voice_text(vp))
        if not clash:
            local_terms.append(term)
    return _dedupe([*global_negative, *episode_negatives, *extra, *local_terms])


def _shot_records(sp: _ShotPlan, sc: _ScenePlan, order: int, src: LegacySources, looks: dict[str, _EnvLook],
                  voices: dict[str, VoiceProfile], state_images: dict[str, Path], negatives: list[str],
                  veo_max: float | None, report: _Report) -> tuple[CreativeShot, GenClip, Take | None, Path]:
    look = looks.get(sp.location_id)
    visual = _visual_spec(sp, look, report)
    spec = ShotSpec(visual=visual, audio_spec=_audio_spec(sp, voices, report), negative=negatives)
    keyframe = _keyframe_path(src, sp)
    keyframe_rel = _design_rel(sc.scene_id, keyframe) if keyframe else None
    plate_rel = _design_rel(sc.scene_id, look.plate) if look and look.plate else None
    refs: list[str] = []
    for st in sp.states.values():
        img = state_images.get(st)
        if img is not None and _design_rel(sc.scene_id, img) not in refs:
            refs.append(_design_rel(sc.scene_id, img))
    if plate_rel and plate_rel not in refs:
        refs.append(plate_rel)
    model, provider, gen = _legacy_model(src, sp.legacy_id)
    video = _legacy_video_path(src, sp.legacy_id)
    has_video = video.is_file()
    line_ids = [ln.line_id for ln in sp.lines]
    speakers_on = _dedupe(ln.character_id for ln in sp.lines if not ln.off_screen)
    flags = ["legacy_import"]
    if sp.multi_beat:
        flags.append("legacy_multi_beat")
    if veo_max is not None and sp.duration_s > veo_max:
        flags.append("exceeds_veo_max_duration")
    if len(speakers_on) > 1:
        flags.append("multi_speaker")
    if not has_video:
        flags.append("legacy_media_missing")
    gb = fsio.generated_by(AGENT, notes=f"Phase 0 import of legacy {sp.legacy_id}; nothing regenerated")
    shot = CreativeShot(
        id=sp.shot_id, scene_id=sc.scene_id, order=order, intention=visual.mood,
        coverage_type="legacy_multi_beat" if sp.multi_beat else "single", duration_s=sp.duration_s,
        board_frame=keyframe_rel, beat_ids=[f"{sc.scene_id}_B{order}"],
        actions=[visual.subject_and_action] if visual.subject_and_action else [], lines=line_ids,
        line_action_index={lid: 0 for lid in line_ids} if visual.subject_and_action else {},
        multibeat_ok=False, continuous=True, spec=spec, gen_clips=[sp.clip_id],
        approval_status=ApprovalStatus.PROPOSED, generated_by=gb,
    )
    performance = _clean(_attr(sp.direction, "performance_physics", ""), f"{sp.legacy_id} performance", report) \
        if sp.direction is not None else ""
    clip = GenClip(
        id=sp.clip_id, shot_id=sp.shot_id, scene_id=sc.scene_id, sequence_index=1, duration_s=sp.duration_s,
        spec=spec, lines=line_ids, performance_state=performance,
        conditioning=Conditioning(reference_uris=refs, reference_entity_states=list(sp.states.values()),
                                  spatial_anchor=plate_rel, first_frame_anchor=keyframe_rel),
        model_target=ModelTarget(preferred=model, fallback="veo" if model == "omni" else "omni"),
        stitch_risk=RiskTier.HIGH if sp.multi_beat else RiskTier.LOW,
        takes=[sp.take_id] if has_video else [], selected_take_id=sp.take_id if has_video else None,
        status=ClipStatus.SELECTED if has_video else ClipStatus.PENDING, flags=flags, generated_by=gb,
    )
    take = None
    if has_video:
        params = gen.get("parameters") if isinstance(gen.get("parameters"), dict) else {}
        seed = _int_or_none(params.get("seed"))
        take = Take(
            id=sp.take_id, clip_id=sp.clip_id, model=model, provider_model=provider, seed=seed,
            payload_hash=legacy_payload_hash(src, sp.legacy_id), status=TakeStatus.SELECTED,
            dispatch_status=DispatchStatus.COMPLETED, operation_name=gen.get("operation_name") or None,
            cost_estimate=0.0, imported_from=str(video),
            created_at=str(gen.get("completed_at") or gen.get("submitted_at") or fsio.now_iso()),
            generated_by=fsio.generated_by(AGENT, model=provider, seed=seed,
                                           notes=f"Phase 0 import of legacy {sp.legacy_id}; nothing regenerated"),
        )
    else:
        report.conflict("Legacy shots without a rendered clip",
                        f"{sp.legacy_id}: {video.name} not found — clip {sp.clip_id} has no take yet.")
    return shot, clip, take, video


def _import_media_into_take(project: Project, take: Take, video: Path) -> tuple[Take, str | None]:
    """Hard-link (or copy) the legacy mp4 as take.mp4 and extract first/last frame + 16 kHz dialogue wav."""
    tdir = project.take_dir(take.id)
    dest = _link_or_copy(video, tdir / "take.mp4", allowed=VIDEO_EXTENSIONS)
    take.video_uri = project.rel(dest)
    if procs.which("ffmpeg") is None:
        return take, f"{take.id}: ffmpeg not found — first/last frame and dialogue.wav not extracted."
    try:
        first, last, wav = tdir / "first.png", tdir / "last.png", tdir / "dialogue.wav"
        if not first.exists():
            media.extract_frame(dest, first, at="first")
        if not last.exists():
            media.extract_frame(dest, last, at="last")
        take.first_frame_uri = project.rel(first)
        take.last_frame_uri = project.rel(last)
        got = wav if wav.exists() else media.extract_wav(dest, wav, sample_rate=16000)
        take.dialogue_wav_uri = project.rel(got) if got else None
    except Exception as exc:  # noqa: BLE001 - media extraction must not abort the migration
        return take, f"{take.id}: could not extract frames/dialogue ({type(exc).__name__}: {str(exc)[:200]})."
    return take, None


def _copy_design_files(project: Project, design_files: dict[str, list[Path]], report: _Report) -> int:
    copied = 0
    for scene_id, paths in design_files.items():
        for path in paths:
            try:
                _link_or_copy(path, project.path("scenes", scene_id, "design", path.name), allowed=IMAGE_EXTENSIONS)
                copied += 1
            except (OSError, ValueError) as exc:
                report.conflict("Files I could not bring in", f"{path.name} → {scene_id}/design: {exc}")
    return copied


def _quality_checks(shots: list[_ShotPlan], src: LegacySources, voices: dict[str, VoiceProfile], registry: Any,
                    report: _Report, *, wps: float, pre_s: float, post_s: float) -> float | None:
    """Flags the v3 conflicts of the legacy clips; returns Veo's max duration (None if unknown)."""
    def max_duration(model_id: str) -> float | None:
        try:
            return float(registry.max_duration_s(model_id))
        except Exception:  # noqa: BLE001 - registry without that model
            return None

    veo_max = max_duration("veo")
    names = {cid: str(_attr(spec, "name", "") or cid) for cid, spec in src.cast.items()}
    legacy_voices: dict[str, list[str]] = {}
    for sp in shots:
        model, _provider, _gen = _legacy_model(src, sp.legacy_id)
        if sp.multi_beat:
            beats = max(2, len(re.findall(r"\[\s*beat", str(_attr(sp.direction, "camera_motion", "") or ""),
                                          re.IGNORECASE)))
            report.conflict(
                "Clips that pack several beats into one generation (fails L2 “one action per clip”)",
                f"{sp.legacy_id} ({sp.duration_s:g} s): stages {beats} beats with a cut inside a single clip. "
                f"Kept exactly as imported; if it is ever regenerated, v3 builds that coverage from separate clips "
                f"(master → over-the-shoulder → reverse reaction) instead of cutting inside one clip.")
        if veo_max is not None and sp.duration_s > veo_max:
            report.conflict(
                "Clips longer than Veo's maximum",
                f"{sp.legacy_id}: {sp.duration_s:g} s (made with {model}); Veo 3.1 tops out at {veo_max:g} s, so "
                f"this clip has no Veo fallback as it stands.")
        speakers_on = _dedupe(ln.character_id for ln in sp.lines if not ln.off_screen)
        if len(speakers_on) > 1:
            report.conflict(
                "Shots where more than one person speaks (single-speaker rule)",
                f"{sp.legacy_id}: {', '.join(names.get(c, c) for c in speakers_on)} all speak in one clip — v3 gives "
                f"each line its own clip with every other mouth closed.")
        words = sum(count_words(ln.text) for ln in sp.lines)
        budget = int(max(0.0, sp.duration_s - pre_s - post_s) * wps)
        if sp.lines and words > budget:
            report.conflict(
                "Lines faster than human speaking pace",
                f"{sp.legacy_id}: {words} words in {sp.duration_s:g} s (comfortable maximum ≈ {budget} words with "
                f"{pre_s:g} s breaths before and after).")
        for ln in sp.lines:
            if ln.legacy_voice:
                bucket = legacy_voices.setdefault(ln.character_id, [])
                if ln.legacy_voice not in bucket:
                    bucket.append(ln.legacy_voice)
    for cid, variants in legacy_voices.items():
        if len(variants) > 1:
            locked = voices[cid].id if cid in voices else "its voice profile"
            report.conflict(
                "Voice descriptions that drifted between legacy shots",
                f"{names.get(cid, cid)} ({cid}) was described {len(variants)} different ways "
                f"(e.g. “{variants[0]}” vs “{variants[1]}”); v3 uses the one locked voice {locked} everywhere.")
    return veo_max


# ---------------------------------------------------------------------------------------------
# State timeline, legacy Resolve handoffs, rebuilt script, manifest
# ---------------------------------------------------------------------------------------------
def _state_timeline_entries(scenes: list[_ScenePlan]) -> list[dict[str, Any]]:
    """``{entity_id, state_id, from, until}`` — markers are scene ids, or the clip id of a mid-scene change."""
    entries: list[dict[str, Any]] = []
    open_entry: dict[str, dict[str, Any]] = {}
    for sc in scenes:
        seen_here: set[str] = set()
        for sp in sc.shots:
            for eid, st in sp.states.items():
                cur = open_entry.get(eid)
                if cur is not None and cur["state_id"] == st:
                    seen_here.add(eid)
                    continue
                marker = sp.clip_id if eid in seen_here else sc.scene_id
                if cur is not None:
                    cur["until"] = marker
                entry = {"entity_id": eid, "state_id": st, "from": marker, "until": None}
                entries.append(entry)
                open_entry[eid] = entry
                seen_here.add(eid)
    return entries


def _handoff_records(scenes: list[_ScenePlan], src: LegacySources) -> list[ResolveHandoff]:
    """The human's legacy episode timelines adopted as human-owned handoffs for the scenes they cover."""
    by_episode: dict[int, list[str]] = {}
    for sc in scenes:
        by_episode.setdefault(sc.episode, []).append(sc.scene_id)
    ts = fsio.now_iso()
    out: list[ResolveHandoff] = []
    for sc in scenes:
        name, colour, timeline = _timeline_for(src, sc.episode)
        out.append(ResolveHandoff(
            scene_id=sc.scene_id, timeline_name=timeline, fingerprint=None, ts=ts, ownership=Ownership.HUMAN,
            markers_enabled=True, legacy=True,
            history=[{
                "ts": ts, "event": "adopted_legacy_timeline", "timeline": timeline,
                "legacy_assembler_timeline": name, "marker_color": colour,
                "scenes_on_timeline": list(by_episode[sc.episode]),
                "legacy_shots": [sp.legacy_id for sp in sc.shots],
                "note": "Phase 0: pre-v3 episode timeline — human-owned; AGY never rebuilds or edits it",
            }],
            generated_by=fsio.generated_by(AGENT, notes="Phase 0: legacy Resolve episode timeline adopted"),
        ))
    return out


def _script_text(scenes: list[_ScenePlan], scene_texts: dict[str, str]) -> str:
    out = [f"Title: {DEFAULT_TITLE}",
           "Credit: rebuilt from the legacy LFKOD episode directions (Phase 0 import — nothing regenerated)",
           f"Draft date: {fsio.now_iso()[:10]}", ""]
    episode: int | None = None
    for sc in scenes:
        if sc.episode != episode:
            episode = sc.episode
            out += [f"# EPISODE {episode}", ""]
        out += [scene_texts[sc.scene_id].rstrip(), ""]
    return "\n".join(out).rstrip() + "\n"


def _same_file(a: Path, b: Path) -> bool:
    try:
        sa, sb = os.stat(a), os.stat(b)
    except OSError:
        return False
    return (sa.st_ino, sa.st_dev) == (sb.st_ino, sb.st_dev)


def _ws_rel(src: LegacySources, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return Path(path).resolve().relative_to(src.workspace).as_posix()
    except ValueError:
        return str(path)


def _manifest(src: LegacySources, scenes: list[_ScenePlan], looks: dict[str, _EnvLook], shot_rows: list[dict],
              *, project_id: str, import_media: bool, dry_run: bool) -> dict[str, Any]:
    asm = src.assembler
    sources = [Path(*LEGACY_DIR) / "casting_bible.py"]
    sources += sorted(p.relative_to(src.workspace) for p in src.workspace.joinpath(*LEGACY_DIR).glob(
        "episode*_directions.py"))
    sources += [Path("dialogue_agent.py"), Path("assemble_microdrama.py")]
    sources += [p.relative_to(src.workspace) for p in (*src.plan_files.values(), *src.state_files.values())
                if _inside(p, src.workspace)]
    if (src.asset_dir / "reference_sheet_registry.json").is_file():
        sources.append(Path("generated_assets") / "reference_sheet_registry.json")
    return {
        "migration": {
            "agy_version": AGY_VERSION, "agent": AGENT, "project_id": project_id, "migrated_at": fsio.now_iso(),
            "workspace": str(src.workspace), "import_media": bool(import_media), "dry_run": bool(dry_run),
            "sources": [p.as_posix() for p in sources if (src.workspace / p).is_file()],
        },
        "legacy_assembler": {
            "episode_timelines": {
                str(ep): {"name": name, "marker_color": colour, "resolve_timeline": f"{name}_{LEGACY_TIMELINE_VERSION}"}
                for ep, (name, colour) in sorted(asm["EPISODE_TIMELINES"].items())
            },
            "trim_in": int(asm["PRO_TRIM_IN_FRAMES"]),
            "trim_out": int(asm["PRO_TRIM_OUT_FRAMES"]),
            "min_active_frames": LEGACY_MIN_ACTIVE_FRAMES,
            "fps": LEGACY_FPS,
        },
        "scenes": [{
            "scene_id": sc.scene_id, "slug": sc.slug, "episode": sc.episode, "location_id": sc.location_id,
            "environment_id": looks[sc.location_id].env_id if sc.location_id in looks else None,
            "legacy_shots": [sp.legacy_id for sp in sc.shots], "shots": [sp.shot_id for sp in sc.shots],
            "resolve_timeline": _timeline_for(src, sc.episode)[2],
        } for sc in scenes],
        "shots": shot_rows,
    }


# ---------------------------------------------------------------------------------------------
# Plain-language migration report (post/migration_report.md)
# ---------------------------------------------------------------------------------------------
def _cell(text: Any, limit: int = 160) -> str:
    s = re.sub(r"\s+", " ", str(text if text is not None else "")).replace("|", "/").strip()
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def _render_report(*, project_id: str, src: LegacySources, scenes: list[_ScenePlan], shots: list[_ShotPlan],
                   entities: list[Entity], voices: dict[str, VoiceProfile], envs: list[Environment],
                   looks: dict[str, _EnvLook], style: StyleLock, ep_negs: dict[int, list[str]],
                   conflicting: dict[str, list[str]], report: _Report, take_notes: dict[str, str],
                   design_copied: int, import_media: bool, dry_run: bool) -> str:
    episodes = sorted({sc.episode for sc in scenes})
    timelines = {ep: _timeline_for(src, ep) for ep in episodes}
    names = {e.id: (e.display_name or e.id) for e in entities}
    L: list[str] = [f"# Migration report — {DEFAULT_TITLE} → AGY v3 (`{project_id}`)", ""]
    L += [f"*Written {fsio.now_iso()} by `{AGENT}` (AGY {AGY_VERSION}). Phase 0: wrap the existing pilot, "
          f"regenerate nothing.*", ""]
    L += ["## In one paragraph", ""]
    L.append(
        f"I moved the existing pilot — {len(episodes)} episode(s), {len(shots)} shots — into the new project "
        f"format without generating anything: every keyframe, reference sheet and clip is the one you already "
        f"have. The shots became {len(scenes)} scenes (one per unbroken run at the same location inside an "
        f"episode). Everything in the Bible is marked *proposed* — locking the characters, looks, voices, "
        f"locations, style lock and schedule is your call at G1. Your Resolve timelines "
        f"{', '.join(t[2] for t in timelines.values())} stay yours: AGY treats them as human-owned and will "
        f"never rebuild them.")
    if dry_run:
        L += ["", "> **Dry run** — built in a throw-away folder; no reference images or clips were linked or "
                  "copied."]
    elif not import_media:
        L += ["", "> The clips are registered as takes but not brought into the project yet "
                  "(`import_media=False`); each take remembers its legacy file in `imported_from`."]

    L += ["", "## 1. Legacy shots → scenes", "",
          "Rule: one scene per unbroken run of shots at the same location inside an episode.", "",
          "| Scene | Episode | Location | Legacy shots | New shots |", "|---|---|---|---|---|"]
    for sc in scenes:
        look = looks.get(sc.location_id)
        loc_name = _attr(src.locations.get(sc.location_id), "name", "") or sc.location_id
        L.append(f"| {sc.scene_id} `{sc.slug}` | EP{sc.episode} | {_cell(loc_name, 60)} "
                 f"({look.env_id if look else '—'}) | {', '.join(sp.legacy_id for sp in sc.shots)} | "
                 f"{', '.join(sp.shot_id for sp in sc.shots)} |")

    L += ["", "## 2. Characters, props and their looks", "",
          "| Id | Name | Importance | Looks (states) | Voice |", "|---|---|---|---|---|"]
    for ent in entities:
        looks_txt = ", ".join(st.id[len(ent.id) + 7:] if st.id.startswith(ent.id + "_STATE_") else st.id
                              for st in ent.states)
        L.append(f"| {ent.id} | {_cell(ent.display_name or ent.id, 40)} | {ent.tier.value} | {_cell(looks_txt, 80)} "
                 f"| {ent.voice_profile_id or '—'} |")
    speakers: list[tuple[str, str, bool, str]] = []
    for sp in shots:
        for ln in sp.lines:
            key = (ln.speaker, ln.character_id, ln.off_screen, ln.how)
            if key not in speakers:
                speakers.append(key)
    if speakers:
        L += ["", "Who speaks each legacy line (speaker labels in the old directions → characters):", "",
              "| Legacy speaker label | Character | Heard how | Why |", "|---|---|---|---|"]
        for label, cid, off, how in speakers:
            L.append(f"| {_cell(label, 90)} | {cid} | {'off-screen (no lips move)' if off else 'on camera'} | "
                     f"{_cell(how, 70)} |")
    n_lines = sum(len(sp.lines) for sp in shots)
    n_off = sum(1 for sp in shots for ln in sp.lines if ln.off_screen)
    n_dev = sum(1 for sp in shots for ln in sp.lines if _has_devanagari(ln.text))
    L += ["", f"Dialogue: {n_lines} lines copied word for word — {n_dev} in Devanagari with the phonetic "
              f"transliteration kept beside them; {n_off} are off-screen (phone / voice note / device), so every "
              f"mouth on camera stays closed for them."]

    L += ["", "## 3. Voices", "", "| Voice | Character | How they sound (locked at G1) |", "|---|---|---|"]
    for cid, vp in voices.items():
        L.append(f"| {vp.id} | {_cell(names.get(cid, cid), 40)} | {_cell(vp.description, 200)} |")

    L += ["", "## 4. Locations", "", "| Environment | Lighting setup | Reference plate | Room tone |",
          "|---|---|---|---|"]
    look_by_env = {lk.env_id: lk for lk in looks.values()}
    for env in envs:
        lk = look_by_env.get(env.id)
        L.append(f"| {env.id} | {lk.lighting_setup_id if lk else '—'} | "
                 f"{lk.plate.name if lk and lk.plate else '—'} | {_cell(env.room_tone or '—', 90)} |")
    L += ["", "Colour words from the legacy lighting prompts were moved into each clip's palette, so lighting "
              "describes only physical light sources (lint rule L7)."]

    L += ["", "## 5. Style lock", "", f"- **Medium:** {_cell(style.medium, 300)}",
          f"- **Aspect ratio:** {style.aspect_ratio}",
          f"- **Lens language:** {_cell(style.lens_language or '—', 300)}",
          f"- **Don'ts:** {len(style.donts)} carried over from the continuity / language locks.",
          f"- **Global negative (every clip):** {_cell(', '.join(style.global_negative), 900)}"]
    for ep, negs in sorted(ep_negs.items()):
        if negs:
            L.append(f"- **EP{ep} only (kept on EP{ep} clips, not global):** {_cell(', '.join(negs), 600)}")
    L += ["", "### Banned words removed (L1)", ""]
    L += [f"- {b}" for b in _dedupe(report.banned)] or ["- None found."]
    if conflicting:
        L += ["", "### Audio negatives I did not make global", "",
              "These v3 audio negatives would fight sounds the legacy clips actually use, so each one is only added "
              "to the clips that do not use that sound:", ""]
        for term, evidence in conflicting.items():
            shown = "; ".join(evidence[:3]) + (f" (+{len(evidence) - 3} more)" if len(evidence) > 3 else "")
            L.append(f"- “{term}” — {_cell(shown, 400)}")

    linked = sum(1 for v in take_notes.values() if v == "hard-linked")
    copied = sum(1 for v in take_notes.values() if v == "copied")
    registered = sum(1 for v in take_notes.values() if v.startswith("registered"))
    missing = [k for k, v in take_notes.items() if v.startswith("missing")]
    failed = [f"{k} ({v})" for k, v in take_notes.items() if v.startswith("not brought in")]
    models: dict[str, list[str]] = {}
    for sp in shots:
        model, provider, _gen = _legacy_model(src, sp.legacy_id)
        models.setdefault(f"{model} ({provider or 'unknown model version'})", []).append(sp.legacy_id)
    L += ["", "## 6. Clips and takes", "",
          f"- {len(shots)} creative shots, each with exactly one clip flagged `legacy_import`.",
          f"- Take T01 of each clip is the legacy mp4: {linked} hard-linked, {copied} copied, {registered} registered "
          f"without bringing the file in, {len(missing)} missing."
          + (" First/last frames and the dialogue wav were extracted for every take brought in." if linked + copied
             else "")]
    if missing:
        L.append(f"- No rendered clip found for: {', '.join(missing)}.")
    if failed:
        L.append(f"- Could not bring in: {'; '.join(failed)}.")
    L.append("- Payload hashes come from the legacy plan/state JSON and are registered in `state.json`, so the same "
             "generation is never dispatched (or paid for) twice.")
    L.append(f"- {design_copied} reference sheets / keyframes copied into `scenes/<SC>/design/` as *proposed* packs — "
             f"they move into `library/` only when you approve them at G2.")
    L += ["", "| Model used by the legacy takes | Shots |", "|---|---|"]
    for label, ids in models.items():
        L.append(f"| {_cell(label, 80)} | {len(ids)}: {_cell(', '.join(ids), 300)} |")

    L += ["", "## 7. Resolve", ""]
    for ep in episodes:
        name, colour, timeline = timelines[ep]
        covered = [sc.scene_id for sc in scenes if sc.episode == ep]
        L.append(f"- `{timeline}` (assembler name `{name}`, {colour} markers) → {', '.join(covered)} — recorded as a "
                 f"human-owned legacy handoff; those scenes are *handed off* and wait for your approval (G6).")
    L += ["", "Track layout — your legacy timelines are left exactly as they are; the difference only matters when a "
              "scene is rebuilt as a v3 scene timeline:", "",
          "| Track | Legacy episode timelines | v3 scene timelines |", "|---|---|---|",
          "| V1 | picture | picture with the clip's own (native) dialogue linked to A1–A2 |",
          "| V2 | — | alternate takes / boards |",
          "| A1 | dialogue | native dialogue (from the clip itself) |",
          "| A2 | music (Lyria score) | native dialogue (second channel) |",
          "| A3 | foley / SFX | SFX supplement — hard effects only |",
          "| A4 | — | SFX supplement |",
          "| A5–A6 | — | reserved |",
          "| A7–A8 | — | music (score cues, always under the native dialogue) |"]

    L += ["", "## 8. Things that need your eye", ""]
    if report.conflicts:
        for section, items in report.conflicts.items():
            L += [f"### {section}", ""] + [f"- {item}" for item in items] + [""]
    else:
        L += ["Nothing — the legacy material maps cleanly.", ""]
    if report.notes:
        L += ["## 9. Notes", ""] + [f"- {n}" for n in _dedupe(report.notes)] + [""]
    L += ["## 10. What happens next (G1)", "",
          "1. Look through the characters, looks, voices, locations and style lock above, then lock G1 — or tell me "
          "what to change first.",
          "2. Answer the open decisions in the G1 packet (I recommend the spec defaults).",
          "3. Nothing is regenerated: these scenes stay exactly as you cut them in Resolve. If you ask for a shot to "
          "be redone, v3 rebuilds it the new way (one action per clip, one speaker per clip) and delivers it beside "
          "your cut — never into it.", ""]
    return "\n".join(L).rstrip() + "\n"


# ---------------------------------------------------------------------------------------------
# Phase 0 acceptance: the legacy episode table, reproduced from the migrated takes
# ---------------------------------------------------------------------------------------------
def load_migration_manifest(project: Project) -> dict[str, Any]:
    """``post/migration_manifest.yaml`` as a dict (``{}`` when the project was not migrated)."""
    path = project.path(*MANIFEST_REL.split("/"))
    if not path.is_file():
        return {}
    data = fsio.read_yaml(path)
    return data if isinstance(data, dict) else {}


def _legacy_duration_sec(path: Path, default_sec: int) -> int:
    """Exactly ``assemble_microdrama._get_video_duration_sec``: whole seconds via ffprobe, else ``default_sec``."""
    if procs.which("ffprobe") is None or not path.exists():
        return default_sec
    try:
        res = procs.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                         "default=noprint_wrappers=1:nokey=1", str(path)], timeout=60, check=False)
        return max(1, int(round(float(str(res.stdout).strip()))))
    except Exception:  # noqa: BLE001 - mirrors the legacy fallback exactly
        return default_sec


def _take_video(project: Project, clip: GenClip | None) -> Path | None:
    """The selected take's video: the imported ``take.mp4``, else the legacy file it was registered from."""
    if clip is None or not clip.selected_take_id:
        return None
    try:
        take = project.load_take(clip.selected_take_id)
    except Exception:  # noqa: BLE001 - missing / unreadable take record -> legacy fallback frames
        return None
    if take.video_uri:
        try:
            path: Path | None = project.abs(take.video_uri)
        except Exception:  # noqa: BLE001 - malformed path in a record
            path = None
        if path is not None and path.is_file():
            return path
    if take.imported_from:
        path = Path(take.imported_from)
        if path.suffix.lower() in VIDEO_EXTENSIONS and path.is_file():
            return path
    return None


def _assembler_settings(manifest: dict[str, Any]) -> tuple[dict[str, Any], int, int, int, int]:
    asm = manifest.get("legacy_assembler") or {}
    timelines = asm.get("episode_timelines") or {
        str(ep): {"name": name, "marker_color": colour}
        for ep, (name, colour) in LEGACY_ASSEMBLER_DEFAULTS["EPISODE_TIMELINES"].items()
    }
    return (
        timelines,
        int(asm.get("fps") or LEGACY_FPS),
        int(asm.get("trim_in", LEGACY_ASSEMBLER_DEFAULTS["PRO_TRIM_IN_FRAMES"])),
        int(asm.get("trim_out", LEGACY_ASSEMBLER_DEFAULTS["PRO_TRIM_OUT_FRAMES"])),
        int(asm.get("min_active_frames", LEGACY_MIN_ACTIVE_FRAMES)),
    )


def legacy_shot_rows(project: Project, episode: int) -> list[dict[str, Any]]:
    """Per-shot rows of one legacy episode in timeline order, read from the migrated scenes / shots / takes."""
    manifest = load_migration_manifest(project)
    _timelines, fps, trim_in, trim_out, min_active = _assembler_settings(manifest)
    by_shot = {str(r.get("shot_id")): r for r in (manifest.get("shots") or []) if isinstance(r, dict)}
    rows: list[dict[str, Any]] = []
    for scene in project.list_scenes():
        if scene.episode != int(episode):
            continue
        for shot in project.list_shots(scene.id):
            clips = project.list_clips(shot.id)
            clip = clips[0] if clips else None
            legacy = by_shot.get(shot.id, {})
            seconds = float((clip.duration_s if clip is not None else None) or shot.duration_s or 0)
            default_frames = _int_or_none(legacy.get("duration_frames")) or int(round(seconds * fps))
            video = _take_video(project, clip)
            if video is not None:
                raw = int(round(_legacy_duration_sec(video, default_sec=int(default_frames / fps)) * fps))
            else:
                raw = default_frames
            frames = raw - trim_in - trim_out if raw > trim_in + trim_out + min_active else raw
            rows.append({
                "scene_id": scene.id, "shot_id": shot.id, "clip_id": clip.id if clip is not None else None,
                "take_id": clip.selected_take_id if clip is not None else None,
                "legacy_shot_id": legacy.get("legacy_shot_id"), "raw_frames": raw, "frames": frames,
                "probed": video is not None,
            })
    return rows


def legacy_episode_table(project: Project, episode: int) -> list[dict[str, Any]]:
    """Reproduces ``assemble_microdrama.episode_table()`` for one episode from the migrated takes.

    Same keys, shot order and frame counts — ``[{num, name, shots, frames, raw_frames, trim_in, trim_out,
    marker_color}]``; empty when the episode has no shots or no legacy timeline (as the legacy table skips it).
    """
    episode = int(episode)
    timelines, _fps, trim_in, trim_out, _min_active = _assembler_settings(load_migration_manifest(project))
    timeline = timelines.get(str(episode)) or timelines.get(episode)  # type: ignore[call-overload]
    rows = legacy_shot_rows(project, episode)
    if not timeline or not rows:
        return []
    return [{
        "num": episode,
        "name": timeline.get("name"),
        "shots": len(rows),
        "frames": sum(r["frames"] for r in rows),
        "raw_frames": sum(r["raw_frames"] for r in rows),
        "trim_in": trim_in,
        "trim_out": trim_out,
        "marker_color": timeline.get("marker_color"),
    }]


def import_legacy_take_media(project: Project, legacy_shot_ids: Iterable[str] | None = None
                             ) -> dict[str, list[str]]:
    """Bring the legacy mp4s into takes registered with ``import_media=False`` (link/copy + frames + wav).

    Only files inside the legacy workspace recorded in the manifest are accepted. Returns
    ``{"imported": [take ids], "warnings": [...]}``.
    """
    manifest = load_migration_manifest(project)
    workspace = Path(str((manifest.get("migration") or {}).get("workspace") or ""))
    wanted = {str(s).upper() for s in legacy_shot_ids} if legacy_shot_ids is not None else None
    imported: list[str] = []
    warnings: list[str] = []
    for row in manifest.get("shots") or []:
        if not isinstance(row, dict):
            continue
        lid, tid = str(row.get("legacy_shot_id") or ""), row.get("take_id")
        if not tid or (wanted is not None and lid.upper() not in wanted):
            continue
        take = project.load_take(str(tid))
        if take.video_uri and project.abs(take.video_uri).is_file():
            continue
        source = Path(take.imported_from or "")
        if (not take.imported_from or not workspace.is_absolute() or not _inside(source, workspace)
                or not source.is_file()):
            warnings.append(f"{tid}: legacy file {take.imported_from!r} is missing or outside the legacy workspace.")
            continue
        try:
            take, warn = _import_media_into_take(project, take, source)
        except (OSError, ValueError) as exc:
            warnings.append(f"{tid}: {exc}")
            continue
        project.save_take(take)
        imported.append(take.id)
        if warn:
            warnings.append(warn)
    return {"imported": imported, "warnings": warnings}


# ---------------------------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------------------------
def migrate_lfkod(workspace: Path | str, *, productions_dir: Path | str | None = None,
                  project_id: str = DEFAULT_PROJECT_ID, import_media: bool = True, dry_run: bool = False,
                  registry: Any = None, media_shot_ids: Iterable[str] | None = None) -> Project:
    """Phase 0: wrap the legacy LFKOD pilot as an AGY v3 project — regenerate nothing.

    ``productions_dir`` defaults to ``<workspace>/productions``. ``import_media=False`` registers the legacy
    clips as takes without bringing the files in (fast; :func:`import_legacy_take_media` does it later);
    ``media_shot_ids`` limits the media import to those legacy shot ids. ``dry_run=True`` builds the whole
    project in a throw-away temp folder (no images or clips brought in) and returns it, so the report can be read
    without touching ``productions/``. Raises ``FileExistsError`` if ``<productions_dir>/<project_id>`` exists.
    """
    ws = Path(workspace).resolve()
    fsio.validate_id(project_id, "project_id")
    prod = Path(productions_dir).resolve() if productions_dir is not None else Project.productions_dir(ws)
    final_root = fsio.safe_join(prod, project_id)
    if final_root.exists() and (not final_root.is_dir() or any(final_root.iterdir())):
        raise FileExistsError(f"project {project_id} already exists at {final_root}; the migration never "
                              f"overwrites a project")
    try:
        src = load_legacy_sources(ws)
    except FileNotFoundError:
        existing_migrated = Project.productions_dir(ws) / project_id
        if existing_migrated.is_dir() and existing_migrated.resolve() != final_root.resolve():
            shutil.copytree(existing_migrated, final_root)
            return Project(final_root)
        raise
    if registry is None:
        from .registry import default_registry
        registry = default_registry()
    wanted = {str(s).upper() for s in media_shot_ids} if media_shot_ids is not None else None

    # 1. Plan everything in memory first.
    report = _Report()
    resolver = SpeakerResolver(src.cast, src.voices)
    shots = _plan_shots(src, resolver, report)
    if not shots:
        raise ValueError("the legacy SHOT_BREAKDOWN is empty — nothing to migrate")
    scenes = _plan_scenes(shots)
    style, ep_negs, conflicting = _build_style_lock(src, shots, report)
    voices = _build_voices(src, report)
    entities, state_images = _build_entities(src, shots, scenes, voices, resolver, report)
    envs, looks = _build_environments(src, shots, scenes, report)
    settings = ProjectSettings(fps=LEGACY_FPS, aspect_ratio=style.aspect_ratio)
    if style.aspect_ratio == "16:9":
        settings.width, settings.height = 1920, 1080
    veo_max = _quality_checks(shots, src, voices, registry, report, wps=settings.speech_pace_wps,
                              pre_s=settings.speech_pre_roll_s, post_s=settings.speech_post_roll_s)

    # 2. Build in a staging folder; it is moved into place only when complete (a killed run leaves no project).
    if dry_run:
        stage_parent = Path(tempfile.mkdtemp(prefix=f"agy_migrate_{project_id}_"))
    else:
        stage_parent = fsio.safe_join(prod, STAGING_DIR_NAME, f"{project_id}_{fsio.compact_ts()}_{os.getpid()}")
        stage_parent.mkdir(parents=True, exist_ok=False)
    cfg = ProjectConfig(project_id=project_id, title=DEFAULT_TITLE, language="hi", settings=settings,
                        generated_by=fsio.generated_by(AGENT, notes="Phase 0 import of the legacy LFKOD pilot"))
    project = Project.create(stage_parent, project_id, DEFAULT_TITLE, language="hi", config=cfg)

    project.save_style_lock(style)
    for env in envs:
        project.save_environment(env)
    for ent in entities:
        project.save_entity(ent)
    casting = project.casting()
    casting["voice_profiles"] = {cid: {"profile": vp.model_dump(mode="json")} for cid, vp in voices.items()}
    project.save_casting(casting)

    names = {e.id: (e.display_name or e.id) for e in entities}
    scene_texts = {sc.scene_id: _scene_script_text(sc, src, names) for sc in scenes}
    for sc in scenes:
        project.save_scene(_scene_record(sc, src, looks, scene_texts[sc.scene_id]))
    fsio.write_new(project.path("script", "script_v1.fountain"), _script_text(scenes, scene_texts))

    design: dict[str, list[Path]] = {}
    for sc in scenes:
        look = looks.get(sc.location_id)
        cands: list[Path | None] = [look.plate if look else None]
        for sp in sc.shots:
            cands += [state_images.get(st) for st in sp.states.values()]
            cands.append(_keyframe_path(src, sp))
        design[sc.scene_id] = [p for i, p in enumerate(cands) if p is not None and p not in cands[:i]]
    design_copied = 0 if dry_run else _copy_design_files(project, design, report)

    # 3. One CreativeShot + one GenClip per legacy shot; the legacy mp4 is take T01.
    store = StateStore(project)
    take_notes: dict[str, str] = {}
    shot_rows: list[dict[str, Any]] = []
    for sc in scenes:
        for order, sp in enumerate(sc.shots, 1):
            negatives = _clip_negative(sp, src, style.global_negative, ep_negs.get(sp.episode, []), conflicting,
                                       report)
            shot, clip, take, video = _shot_records(sp, sc, order, src, looks, voices, state_images, negatives,
                                                    veo_max, report)
            project.save_shot(shot)
            project.save_clip(clip)
            model, provider, gen = _legacy_model(src, sp.legacy_id)
            if take is not None:
                note = "registered (legacy file not brought in yet)"
                if import_media and not dry_run and (wanted is None or sp.legacy_id.upper() in wanted):
                    try:
                        take, warn = _import_media_into_take(project, take, video)
                        note = "hard-linked" if _same_file(video, project.abs(take.video_uri or "")) else "copied"
                        if warn:
                            report.conflict("Clips brought in without frames / dialogue wav", warn)
                    except (OSError, ValueError) as exc:
                        note = f"not brought in ({exc})"
                        report.conflict("Files I could not bring in", f"{sp.legacy_id}: {exc}")
                project.save_take(take)
                fsio.write_new(project.take_dir(take.id) / "legacy_payload.json", {
                    "legacy_shot_id": sp.legacy_id, "payload_hash": take.payload_hash,
                    "plan_file": _ws_rel(src, src.plan_files.get(sp.episode)),
                    "state_file": _ws_rel(src, src.state_files.get(sp.episode)),
                    "plan": src.plans.get(sp.legacy_id) or {}, "generation": gen or {},
                })
                store.record_dispatch(take.payload_hash, take.id)
                take_notes[sp.legacy_id] = note
            else:
                take_notes[sp.legacy_id] = "missing (no legacy mp4)"
            shot_rows.append({
                "legacy_shot_id": sp.legacy_id, "episode": sp.episode, "shot_number": sp.number,
                "scene_id": sc.scene_id, "shot_id": sp.shot_id, "clip_id": sp.clip_id,
                "take_id": take.id if take is not None else None,
                "duration_s": float(sp.duration_s), "duration_frames": int(sp.duration_frames),
                "legacy_video": _ws_rel(src, video) if video.is_file() else None,
                "keyframe": shot.board_frame, "model": model, "provider_model": provider,
                "payload_hash": take.payload_hash if take is not None else None,
                "multi_beat": bool(sp.multi_beat), "lines": [ln.line_id for ln in sp.lines],
                "media": take_notes[sp.legacy_id],
            })

    # 4. Orchestration state, legacy Resolve handoffs, state timeline.
    schedule = [sc.scene_id for sc in scenes]
    store.set_schedule(schedule, schedule[0])
    for sc in scenes:
        store.transition(sc.scene_id, SceneStage.HANDED_OFF, force=True,
                         reason=f"Phase 0 import: legacy EP{sc.episode} shots already cut by the human in "
                                f"{_timeline_for(src, sc.episode)[2]}")
        store.set_gate(sc.scene_id, GateId.G6, "awaiting")
    store.set_gate(None, GateId.G1, "awaiting")
    store.set_global_stage(GlobalStage.AWAITING_G1, reason="Phase 0 import complete — Bible proposed, awaiting G1")
    for handoff in _handoff_records(scenes, src):
        project.save_handoff(handoff)
    fsio.save_versioned(project.path("bible", "state_timeline.yaml"), {
        "entries": _state_timeline_entries(scenes),
        "generated_by": fsio.generated_by(AGENT, notes="Phase 0: from the legacy wardrobe_state_by_character")
        .model_dump(mode="json"),
    })

    # 5. Manifest, plain-language report, provenance log.
    fsio.write_new(project.path(*MANIFEST_REL.split("/")), _manifest(
        src, scenes, looks, shot_rows, project_id=project_id, import_media=import_media, dry_run=dry_run))
    fsio.write_new(project.path(*REPORT_REL.split("/")), _render_report(
        project_id=project_id, src=src, scenes=scenes, shots=shots, entities=entities, voices=voices, envs=envs,
        looks=looks, style=style, ep_negs=ep_negs, conflicting=conflicting, report=report, take_notes=take_notes,
        design_copied=design_copied, import_media=import_media, dry_run=dry_run))
    log_agent_call(
        project, agent=AGENT, scene_id=None, working_set_hash=None,
        task="Phase 0 migration of the legacy LFKOD pilot (wrap the existing system, regenerate nothing)",
        inputs={"workspace": str(ws), "import_media": bool(import_media), "dry_run": bool(dry_run),
                "media_shot_ids": sorted(wanted) if wanted is not None else None},
        outputs={"scenes": schedule, "shots": len(shots), "entities": [e.id for e in entities],
                 "environments": [e.id for e in envs], "voices": [v.id for v in voices.values()],
                 "takes": sum(1 for v in take_notes.values() if not v.startswith("missing")),
                 "report": REPORT_REL, "manifest": MANIFEST_REL,
                 "conflicts": {k: len(v) for k, v in report.conflicts.items()}},
        mode="deterministic",
    )
    if dry_run:
        return Project(project.root)

    # 6. Move into place (rename inside productions/ — atomic on one filesystem).
    try:
        os.rename(project.root, final_root)
    except OSError as exc:
        if final_root.exists():
            raise FileExistsError(f"project {project_id} appeared at {final_root} during the migration; the staged "
                                  f"copy is kept at {project.root}") from exc
        raise
    try:
        stage_parent.rmdir()  # the now-empty staging folder only (never anything with content)
    except OSError:
        pass
    return Project(final_root)




