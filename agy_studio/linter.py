"""Pre-flight Linter — deterministic rules L1–L18 (BUILD_SPEC_v3 §7.1), run before EVERY dispatch.

:func:`lint_clip` returns a FIXED COPY of the clip plus a :class:`~agy_studio.models.LintReport`;
``report.spec_hash`` is :func:`fsio.clip_lint_hash` of the fixed copy (computed after auto-fixes), so the Shot
Generator can refuse to dispatch anything that changed after lint (acceptance criterion 1).

| Rule | Check | Action |
|---|---|---|
| L1  | banned words (hyperrealistic, photorealistic, stunning, 4K, 8K, cinematic, ...) | auto-strip, log |
| L2  | > 1 physical verb in ``subject_and_action`` (speech verbs excluded) | bounce to Shot Decomposer |
| L3  | every ``[A-Z][A-Z0-9_]+`` token resolves to a registered Entity / State / Environment / Voice | fail |
| L4  | referenced state is active at this clip index (mid-scene transitions applied) | fail |
| L5  | packs locked; tier requirements met | fail |
| L6  | conditioning holds every referenced entity's pack + the spatial anchor | fail |
| L7  | ``lighting`` has no colour words; ``palette`` has no light sources | fail + rewrite suggestion |
| L8  | ``duration_s`` within the model's registry limits | bounce |
| L9  | first-frame anchor = last frame of the preceding clip's approved (selected) take | fail |
| L10 | all 9 visual fields present, non-empty, ordered | fail |
| L11 | negatives ⊇ global ∪ state-specific (+ music / narration negatives for native AV) | auto-fix |
| L12 | a required feature is unsupported by the model | fallback or fail |
| L13 | camera side vs ``axis_of_action`` | warn |
| L14 | dialogue verbatim from ``scene.dialogue``; every line exactly once across the shot's clips | fail |
| L15 | every speaker has a locked VoiceProfile; ``voice`` = description (+ state variant) | fail |
| L16 | speakers per clip ≤ effective registry max (single-speaker workspace rule) | bounce |
| L17 | ``music == none``; no music / score terms anywhere in ``audio_spec`` | auto-fix + log |
| L18 | words ÷ pace ≤ ``duration_s`` − margin | bounce |

With ``apply_fixes=False`` nothing is modified and every auto-fixable problem is reported as a blocking failure
(the fix is in ``suggestion``), so ``passed`` always describes the returned clip exactly.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from . import continuity
from .agent_log import log_agent_call
from .decomposer import (
    CAMERA_ACRONYMS,
    REQUIRED_PACK_ROLES,
    TEXT_ONLY_MAX_S,
    VisualRefs,
    active_state_map,
    duration_problems,
    effective_max_speakers,
    load_voice_profiles,
    merge_unique,
    required_negatives,
    scan_visual_refs,
    scene_audio_mode,
    scene_shot_order,
    scene_working_set_hash,
    state_owner_index,
)
from .fsio import clip_lint_hash, now_iso, validate_id
from .models import (
    VISUAL_FIELDS,
    ApprovalStatus,
    AudioMode,
    ClipStatus,
    CreativeShot,
    Entity,
    EntityState,
    EntityType,
    Environment,
    GenClip,
    LintFinding,
    LintReport,
    ModelKind,
    ProjectSettings,
    ReferencePack,
    Scene,
    SceneDialogueLine,
    StyleLock,
    TakeStatus,
    Tier,
    VoiceProfile,
)
from .project import Project, parse_artifact_id
from .registry import ModelSpec, Registry, UnknownModelError

AGENT_NAME = "Pre-flight Linter"
PROVENANCE_AGENT = "agy.preflight_linter"
RULE_TITLES: dict[str, str] = {
    "L1": "banned words",
    "L2": "one physical verb per clip",
    "L3": "registered ids only",
    "L4": "active entity-states",
    "L5": "locked, complete reference packs",
    "L6": "conditioning complete",
    "L7": "lighting / palette separation",
    "L8": "duration within model limits",
    "L9": "first-frame anchor",
    "L10": "nine visual fields",
    "L11": "negatives",
    "L12": "model features",
    "L13": "180-degree axis",
    "L14": "verbatim dialogue, each line once",
    "L15": "locked voice profiles",
    "L16": "speakers per clip",
    "L17": "no music",
    "L18": "speech fits the clip",
}
SEVERITY_BY_ACTION: dict[str, str] = {
    "auto_fixed": "info", "warn": "warn", "fallback": "warn", "fail": "error", "bounce": "error",
}
BLOCKING_ACTIONS = frozenset({"fail", "bounce"})
_EPS = 1e-6

# ---------------------------------------------------------------------------------------------
# L1 banned words
# ---------------------------------------------------------------------------------------------
_BANNED_ALTS: tuple[str, ...] = (
    r"hyper[\s-]?realis(?:tic(?:ally)?|m)",
    r"photo[\s-]?realis(?:tic(?:ally)?|m)",
    r"stunning(?:ly)?",
    r"[48]k",
    r"cinematic(?:ally|s)?",
    r"breath[\s-]?taking(?:ly)?",
    r"masterpieces?",
    r"beautiful(?:ly)?",
    r"epic(?:ally|s)?",
    r"award[\s-]?winning",
)
_BANNED_GROUP = "|".join(_BANNED_ALTS)
BANNED_RX = re.compile(r"\b(?:" + _BANNED_GROUP + r")\b", re.IGNORECASE)
_STRIP_RX = re.compile(
    r"\b(?P<pre>(?:very|truly|really|absolutely|incredibly|super|so|most|ultra|quite)\s+)?(?P<word>"
    + _BANNED_GROUP + r")\b",
    re.IGNORECASE,
)
_TAIL_RX = re.compile(r"[\s,;:.!?\-–—]*")
_AN_RX = re.compile(r"\b([Aa])n\s+(?=[bcdfgjklmnpqrstvwxyz])", re.IGNORECASE)
_DANGLING_WORDS = frozenset({
    "a", "an", "the", "and", "or", "with", "in", "of", "like", "to", "for", "at", "by", "as", "very", "so",
    "truly", "really", "absolutely", "most", "quite", "into", "onto",
})


def _norm_word(text: str) -> str:
    return re.sub(r"[\s\-]+", "", (text or "").casefold())


def _allowed_set(allowed: Iterable[str] | None) -> set[str]:
    return {_norm_word(w) for w in (allowed or ()) if w and w.strip()}


def find_banned_words(text: str, allowed: Iterable[str] = ()) -> list[str]:
    """Banned words in ``text`` as written (human-approved ``lint_allowed_words`` are exempt)."""
    ok = _allowed_set(allowed)
    return [m.group(0) for m in BANNED_RX.finditer(text or "") if _norm_word(m.group(0)) not in ok]


def tidy_text(text: str) -> str:
    """Collapse whitespace and repair punctuation left behind after words were removed."""
    t = re.sub(r"\s+", " ", text or "")
    t = re.sub(r"\(\s*\)", " ", t)
    t = re.sub(r"\s+([,;:.!?)])", r"\1", t)
    t = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", t)
    t = re.sub(r"[,;:]+\s*([.!?])", r"\1", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"^[\s,;:.!?\-–—]+", "", t)
    return re.sub(r"[\s,;:\-–—]+$", "", t)


def _drop_dangling_tail(text: str) -> str:
    m = re.search(r"[.!?]+$", text)
    end = m.group(0) if m else ""
    body = text[: m.start()] if m else text
    words = body.rstrip(" ,;:-–—").split()
    while words and words[-1].casefold() in _DANGLING_WORDS:
        words.pop()
    body = " ".join(words).rstrip(" ,;:-–—")
    return f"{body}{end}" if body else ""


def strip_banned(text: str, allowed: Iterable[str] = ()) -> tuple[str, list[str]]:
    """Remove banned words (with a leading intensifier such as "very") -> (clean text, removed words).

    ``"Cinematic wide shot, stunning 4K detail"`` -> ``"wide shot, detail"``; ``"lit like a masterpiece."`` ->
    ``"lit."``. Text without banned words is returned unchanged.
    """
    source = text or ""
    ok = _allowed_set(allowed)
    pieces: list[str] = []
    removed: list[str] = []
    pos = 0
    dangling = False
    for m in _STRIP_RX.finditer(source):
        if _norm_word(m.group("word")) in ok:
            continue
        pieces.append(source[pos:m.start()])
        pieces.append(" ")
        pos = m.end()
        removed.append(m.group("word"))
        dangling = _TAIL_RX.fullmatch(source[pos:]) is not None
    if not removed:
        return source, []
    pieces.append(source[pos:])
    clean = tidy_text("".join(pieces))
    if dangling:
        clean = _drop_dangling_tail(clean)
    clean = _AN_RX.sub(lambda mm: mm.group(1) + " ", clean)
    return clean, removed


# ---------------------------------------------------------------------------------------------
# L17 music terms
# ---------------------------------------------------------------------------------------------
MUSIC_TERMS_RX = re.compile(
    r"\b(?:music(?:al)?|scor(?:e|es|ed|ing)|soundtracks?|songs?|melod(?:y|ies|ic)|tunes?|jingles?|"
    r"instrumentals?|orchestra(?:l|s)?|symphon(?:y|ies|ic)|pianos?|guitars?|violins?|"
    r"synth(?:s|esizers?|wave|pop)?|basslines?|underscor(?:e|es|ing)|leitmotifs?)\b",
    re.IGNORECASE,
)


def strip_music_terms(text: str) -> tuple[str, list[str]]:
    """Drop every comma / semicolon clause that mentions music -> (clean text, dropped clauses)."""
    source = text or ""
    if not MUSIC_TERMS_RX.search(source):
        return source, []
    kept: list[str] = []
    dropped: list[str] = []
    for clause in re.split(r"[,;]", source):
        c = clause.strip()
        if not c:
            continue
        (dropped if MUSIC_TERMS_RX.search(c) else kept).append(c)
    return tidy_text(", ".join(kept)), dropped


# ---------------------------------------------------------------------------------------------
# L7 colour words / light sources
# ---------------------------------------------------------------------------------------------
COLOR_RX = re.compile(
    r"\b(?:red|orange|yellow|green|blue|purple|violet|pink|magenta|cyan|teal|turquoise|amber|gold(?:en)?|"
    r"crimson|scarlet|maroon|indigo|lavender|beige|ochre|emerald|azure|navy|sepia|white|gr[ae]y|brown|"
    r"reddish|bluish|purplish|pinkish|greenish|yellowish|orangey|gr[ae]yish)(?:s|-?tinted|-?toned|-?ish)?\b",
    re.IGNORECASE,
)  # "silver" is deliberately absent: a silver reflector is a light-shaping tool, not a colour cue
LIGHT_SOURCE_RX = re.compile(
    r"\b(?:lamps?|lamp-?posts?|street-?lamps?|street-?lights?|bulbs?|lanterns?|candles?|candle-?light|"
    r"torch(?:es|light)?|flash-?lights?|head-?lights?|spot-?lights?|flood-?lights?|sun-?light|moon-?light|"
    r"day-?light|fire-?light|sunlit|moonlit|back-?light(?:s|ing)?|key[\s-]?lights?|fill[\s-]?lights?|"
    r"rim[\s-]?lights?|kickers?|practicals?|fluorescents?|tungsten|halogen|sodium(?:[\s-]vapou?r)?|leds?|"
    r"hmis?|softbox(?:es)?|strobes?|neon\s+(?:signs?|tubes?|lights?)|"
    r"(?:phone|tv|television|monitor|laptop)\s+screens?|light\s+sources?|sun(?!-)|moon(?!-))\b",
    re.IGNORECASE,
)


def _unique_matches(rx: re.Pattern[str], text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in rx.finditer(text or ""):
        key = m.group(0).casefold()
        if key not in seen:
            seen.add(key)
            out.append(m.group(0))
    return out


# ---------------------------------------------------------------------------------------------
# L13 axis of action
# ---------------------------------------------------------------------------------------------
_AXIS_SIDE_RX = re.compile(
    r"camera\s+(?:stays|must\s+stay|remains|keeps|keep|stay|is\s+kept)\s+(?:on\s+)?(?:the\s+)?"
    r"(?P<side>[a-z][a-z\-]*(?:\s+[a-z][a-z\-]*)?)\s+(?:side|of\s+the\s+(?:line|axis))",
    re.IGNORECASE,
)
_COMPASS_OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}


def axis_side(axis_of_action: str) -> str | None:
    """The side the camera must stay on, parsed from ``Environment.axis_of_action`` (None if unstated)."""
    m = _AXIS_SIDE_RX.search(axis_of_action or "")
    return m.group("side").strip().casefold() if m else None


# ---------------------------------------------------------------------------------------------
# L2 physical verbs
# ---------------------------------------------------------------------------------------------
_PHYSICAL_BASE = frozenset("""
approach bang beckon bend bolt bounce bow brace burst catch chase clamber clap climb collapse crash crawl creep
crouch dance dart dash descend dig dive dodge drag drop enter escape exit fall flail flee flick fling flinch flip
follow gesture grab hit hop hug hunch hurl jab jerk jog jump kick kiss kneel knock launch leap lean lie lift lunge
lurch march mount nod pivot plunge point pound pour pull punch push raise reach recline recoil retreat rise roll
rub run rush scramble scratch shake shiver shove shrug shuffle sidestep sit skid skip slam slap slide slip slouch
slump snatch sneak spin sprint squat stagger stand step stomp stoop stretch stride strike stumble swat sway swing
swipe teeter throw tiptoe topple toss tremble trip tumble turn twirl twist vault walk wander wave wipe wobble
wrestle yank
""".split())
_AMBIGUOUS_BASE = frozenset("""
race pace circle cross limp pass open close shut lower lock button place type film record photograph play check
smoke duck block tackle hammer elbow steady tap
""".split())  # the bare form is usually a noun -> only inflected forms count
_STATIC_BASE = frozenset("hold wear carry clutch grip grasp cradle clasp perch".split())
_POSTURE_BASE = frozenset("sit stand kneel crouch squat lie lean recline perch hunch slouch slump".split())
_SPEECH_BASE = frozenset("""
say speak shout yell scream whisper mutter murmur call cry laugh giggle sing hum sigh gasp groan moan sob chuckle
exclaim reply answer ask tell talk chat mumble stammer stutter cheer bark growl hiss snarl roar shriek squeal wail
howl grunt chant pray recite announce declare plead beg order warn
""".split())
_GAZE_BASE = frozenset("""
look stare glance gaze watch see notice listen hear peer squint glare gape eye observe scan study read smile grin
frown wince blink smirk sneer pout beam scowl grimace
""".split())  # gaze, perception and facial expression are performance, not physical actions
_LEXICONS: tuple[tuple[str, frozenset[str]], ...] = (
    ("speech", _SPEECH_BASE), ("gaze", _GAZE_BASE), ("static", _STATIC_BASE), ("posture", _POSTURE_BASE),
    ("ambiguous", _AMBIGUOUS_BASE), ("physical", _PHYSICAL_BASE),
)
_IRREGULAR: dict[str, str] = {
    "ran": "run", "fell": "fall", "fallen": "fall", "threw": "throw", "thrown": "throw", "rose": "rise",
    "risen": "rise", "sat": "sit", "stood": "stand", "knelt": "kneel", "leapt": "leap", "crept": "creep",
    "swung": "swing", "struck": "strike", "spun": "spin", "slid": "slide", "caught": "catch", "dug": "dig",
    "dove": "dive", "bent": "bend", "shook": "shake", "shaken": "shake", "strode": "stride", "flung": "fling",
    "fled": "flee", "lay": "lie", "lain": "lie", "leant": "lean", "snuck": "sneak", "held": "hold", "wore": "wear",
    "worn": "wear", "said": "say", "spoke": "speak", "spoken": "speak", "told": "tell", "sang": "sing",
    "sung": "sing", "saw": "see", "seen": "see", "heard": "hear",
}
_DETERMINERS = frozenset("""
a an the his her their its my your our this that these those each every another one two three some any no whose
which what
""".split())
_SUBJECTISH = frozenset("""
he she they it who we i you both all together and or then but while before after until as when to is are was were
be been being start starts started begin begins began try tries tried keep keeps kept continue continues continued
again also still just almost nearly finally suddenly immediately instantly
""".split()) | frozenset({",", ";"})
_POSTURE_PARTICLES = frozenset({"up", "down", "back", "upright", "forward"})
_TOKEN_RX = re.compile(r"[A-Za-z0-9_]+(?:['’][A-Za-z]+)?|[,;]")


def _verb_bases(tok: str) -> list[tuple[str, str]]:
    """Candidate (base, form) analyses of a lowercase token, most literal first."""
    out: list[tuple[str, str]] = []
    if tok in _IRREGULAR:
        out.append((_IRREGULAR[tok], "irregular"))
    out.append((tok, "bare"))
    n = len(tok)
    if n > 3 and tok.endswith("ies"):
        out.append((tok[:-3] + "y", "s"))
    if n > 3 and tok.endswith("es"):
        out.append((tok[:-2], "s"))
    if n > 2 and tok.endswith("s") and not tok.endswith("ss"):
        out.append((tok[:-1], "s"))
    if n > 4 and tok.endswith("ied"):
        out.append((tok[:-3] + "y", "ed"))
    if n > 3 and tok.endswith("ed"):
        out.append((tok[:-2], "ed"))
        out.append((tok[:-1], "ed"))
        if n > 4 and tok[-3] == tok[-4]:
            out.append((tok[:-3], "ed"))
    if n > 4 and tok.endswith("ying"):
        out.append((tok[:-4] + "ie", "ing"))
    if n > 4 and tok.endswith("ing"):
        out.append((tok[:-3], "ing"))
        out.append((tok[:-3] + "e", "ing"))
        if n > 5 and tok[-4] == tok[-5]:
            out.append((tok[:-4], "ing"))
    return out


def _classify(tok: str) -> tuple[str | None, str]:
    for base, form in _verb_bases(tok):
        for kind, lexicon in _LEXICONS:
            if base in lexicon:
                return kind, form
    return None, "bare"


def _subjectish(prev: str) -> bool:
    """True when ``prev`` can directly precede a finite verb (entity id, pronoun, coordinator, adverb, ...)."""
    if not prev:
        return False
    if prev in (",", ";"):
        return True
    if continuity.ENTITY_TOKEN_RE.fullmatch(prev):
        return True
    low = prev.casefold()
    if low in _SUBJECTISH:
        return True
    if prev.isalpha() and prev.islower() and len(prev) >= 5 and low.endswith("ly"):
        return True
    return prev.isalpha() and prev[0].isupper() and prev[1:].islower() and low not in _DETERMINERS


def physical_verbs(text: str) -> list[str]:
    """Physical-action verbs in ``text`` (L2). Speech, gaze / facial expression and holding verbs never count;
    postures count only as a change ("sits down", "stands up"); a verb must follow an entity id, pronoun,
    coordinator, adverb or clause break, so nouns ("a step", "concrete steps") are not counted."""
    tokens = _TOKEN_RX.findall(text or "")
    found: list[str] = []
    for i, tok in enumerate(tokens):
        if not (tok.isalpha() and tok.islower()):
            continue
        kind, form = _classify(tok)
        if kind not in ("physical", "ambiguous", "posture"):
            continue
        if not _subjectish(tokens[i - 1] if i else ""):
            continue
        if kind == "ambiguous" and form == "bare":
            continue
        if kind == "posture":
            nxt = tokens[i + 1].casefold() if i + 1 < len(tokens) else ""
            if form == "ing" or nxt not in _POSTURE_PARTICLES:
                continue
        found.append(tok)
    return found


# ---------------------------------------------------------------------------------------------
# L5 / L12 helpers
# ---------------------------------------------------------------------------------------------
def _role_uri(pack: ReferencePack, role: str) -> str | None:
    if role in ("front", "three_quarter", "profile", "rear"):
        return getattr(pack.turnaround, role)
    return getattr(pack, role, None)


def missing_pack_roles(pack: ReferencePack, tier: Tier, entity_type: EntityType = EntityType.CHARACTER) -> list[str]:
    """Pack roles the tier requires that are absent (incidental: any single still is enough)."""
    if tier == Tier.INCIDENTAL:
        return [] if (pack.single_ref or pack.image_uris()) else ["single_ref"]
    if entity_type == EntityType.PROP:
        roles = ("front", "three_quarter") if tier == Tier.PRINCIPAL else ("single_ref",)
        return [role for role in roles if not _role_uri(pack, role)]
    return [role for role in REQUIRED_PACK_ROLES.get(tier, ("single_ref",)) if not _role_uri(pack, role)]


_FEATURE_LABELS: dict[str, str] = {
    "native_audio": "native audio",
    "dialogue_in_prompt": "dialogue in the prompt",
    "reference_to_video": "reference images",
    "multi_subject_refs": "multi-subject references",
    "first_frame": "first-frame conditioning",
    "last_frame": "last-frame conditioning",
    "refs_with_first_frame": "reference images together with a first frame",
}


def feature_problems(spec: ModelSpec, needs: Mapping[str, Any], settings: ProjectSettings | None = None) -> list[str]:
    """Why ``spec`` cannot render a clip with these ``needs`` (empty list = it can). Durations are not checked."""
    problems: list[str] = []
    if not spec.enabled or spec.kind != ModelKind.VIDEO_AV:
        problems.append(f"{spec.id} is not an enabled video model")
    for feature, label in _FEATURE_LABELS.items():
        if needs.get(feature) and not getattr(spec.supports, feature, False):
            problems.append(f"no {label}")
    n_images = int(needs.get("n_reference_images") or 0)
    cap = int(spec.supports.max_reference_images)
    if needs.get("reference_to_video") and spec.supports.reference_to_video and n_images > cap:
        problems.append(f"needs {n_images} reference images (one per pack + spatial anchor) but "
                        f"max_reference_images is {cap}")
    if settings is not None and spec.aspect_ratios and settings.aspect_ratio not in spec.aspect_ratios:
        problems.append(f"aspect ratio {settings.aspect_ratio} is not offered")
    return problems


def _finding(rule: str, action: str, message: str, *, where: str | None = None,
             suggestion: str | None = None) -> LintFinding:
    return LintFinding(rule=rule, severity=SEVERITY_BY_ACTION[action], action=action, message=message,
                       field=where, suggestion=suggestion)


def _rule_no(rule: str) -> int:
    digits = rule[1:]
    return int(digits) if digits.isdigit() else 99


# ---------------------------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------------------------
@dataclass
class LintContext:
    """Everything the linter needs for one shot (loaded once, shared by every clip of the shot)."""

    project: Project
    registry: Registry
    scene: Scene
    shot: CreativeShot
    shot_order: list[str]
    style: StyleLock | None
    entities: dict[str, Entity]
    states: dict[str, EntityState]
    state_owner: dict[str, str]
    environments: dict[str, Environment]
    voices: dict[str, VoiceProfile]  # character id -> current VoiceProfile
    settings: ProjectSettings
    shot_clips: list[GenClip] = field(default_factory=list)
    lines: dict[str, SceneDialogueLine] = field(default_factory=dict)

    @classmethod
    def build(cls, project: Project, registry: Registry, scene_id: str, shot_id: str) -> "LintContext":
        validate_id(scene_id, "scene id")
        ids = parse_artifact_id(validate_id(shot_id, "shot id"))
        if ids["shot"] != shot_id or ids["scene"] != scene_id:
            raise ValueError(f"{shot_id!r} is not a shot of scene {scene_id!r} (expected {scene_id}_SH01)")
        scene = project.load_scene(scene_id)
        entities = {e.id: e for e in project.list_entities()}
        characters = [e.id for e in entities.values() if e.type == EntityType.CHARACTER]
        return cls(
            project=project,
            registry=registry,
            scene=scene,
            shot=project.load_shot(shot_id),
            shot_order=scene_shot_order(project, scene),
            style=project.load_style_lock(),
            entities=entities,
            states={st.id: st for ent in entities.values() for st in ent.states},
            state_owner=state_owner_index(entities),
            environments={e.id: e for e in project.list_environments()},
            voices=load_voice_profiles(project, [*characters, *(ln.character_id for ln in scene.dialogue)]),
            settings=project.config.settings,
            shot_clips=project.list_clips(shot_id),
            lines={ln.line_id: ln for ln in scene.dialogue},
        )

    @property
    def environment(self) -> Environment | None:
        env_id = self.scene.environment_id
        return self.environments.get(env_id) if env_id else None

    @property
    def voice_ids(self) -> set[str]:
        ids = {vp.id for vp in self.voices.values()}
        ids.update(e.voice_profile_id for e in self.entities.values() if e.voice_profile_id)
        return ids

    @property
    def lighting_ids(self) -> set[str]:
        return {ls.id for env in self.environments.values() for ls in env.lighting_setups}

    @property
    def acronyms(self) -> set[str]:
        extra = {a.strip().upper() for a in self.settings.lint_extra_acronyms if a and a.strip()}
        return set(CAMERA_ACRONYMS) | extra


@dataclass
class _Facts:
    refs: VisualRefs
    active: dict[str, str]  # entity id -> state active at this clip
    active_set: set[str]
    referenced: list[str]  # entities visible in the clip (visual tokens + on-screen speakers)
    resolved: dict[str, str | None]  # entity id -> state used for it (explicit token, else active)
    speakers: list[str]  # on- and off-screen
    on_screen: list[str]


# ---------------------------------------------------------------------------------------------
# Per-clip linter
# ---------------------------------------------------------------------------------------------
class _ClipLinter:
    def __init__(self, ctx: LintContext, clip: GenClip, *, model_id: str | None, apply_fixes: bool) -> None:
        self.ctx = ctx
        self.clip = clip.model_copy(deep=True)
        self.apply_fixes = apply_fixes
        self.requested = model_id
        self.findings: list[LintFinding] = []
        self.auto_fixes: list[str] = []
        self.model_used: str | None = None
        self.spec: ModelSpec | None = None

    # -- plumbing ----------------------------------------------------------------------------
    def add(self, rule: str, action: str, message: str, *, where: str | None = None,
            suggestion: str | None = None) -> None:
        self.findings.append(_finding(rule, action, message, where=where, suggestion=suggestion))

    def fixable(self, rule: str, message: str, *, where: str, apply: Callable[[], Any],
                suggestion: str | None = None) -> None:
        """Auto-fix (info finding + auto_fixes log) or, with fixes disabled, a blocking failure with the fix."""
        if self.apply_fixes:
            apply()
            self.add(rule, "auto_fixed", message, where=where)
            self.auto_fixes.append(f"{rule} {where}: {message}")
        else:
            self.add(rule, "fail", f"{message} (auto-fix disabled)", where=where, suggestion=suggestion)

    def exists(self, rel: str | None) -> bool:
        if not rel:
            return False
        try:
            return self.ctx.project.abs(rel).is_file()
        except (ValueError, OSError):  # PathSecurityError is a ValueError
            return False

    def text_only_ok(self, owner: Entity, state: EntityState, facts: _Facts) -> bool:
        """Open Decision 2: an incidental with a text-only pack, no line and <= 2 s needs no image."""
        if owner.tier != Tier.INCIDENTAL or not state.reference_pack.text_only:
            return False
        if self.ctx.settings.open_decisions.incidental_conditioning == "text_only":
            return True
        return owner.id not in facts.speakers and float(self.clip.duration_s) <= TEXT_ONLY_MAX_S + _EPS

    def run(self) -> tuple[GenClip, list[LintFinding], list[str], str | None]:
        self.l1_banned_words()
        self.l17_music()
        facts = self.facts()
        self.l12_select_model(facts)
        self.l11_negatives(facts)
        self.l2_physical_verbs()
        self.l3_registered_ids(facts)
        self.l4_active_states(facts)
        self.l5_packs(facts)
        self.l6_conditioning(facts)
        self.l7_lighting_palette()
        self.l8_duration()
        self.l9_first_frame()
        self.l10_visual_fields()
        self.l13_axis()
        self.l14_dialogue()
        self.l15_voices(facts)
        self.l16_speakers(facts)
        self.l18_speech_time()
        return self.clip, self.findings, self.auto_fixes, self.model_used

    def facts(self) -> _Facts:
        ctx, clip = self.ctx, self.clip
        refs = scan_visual_refs(clip.spec.visual, entities=ctx.entities, state_owner=ctx.state_owner,
                                environments=ctx.environments, voice_ids=ctx.voice_ids,
                                lighting_ids=ctx.lighting_ids, acronyms=ctx.acronyms)
        active = active_state_map(ctx.scene, ctx.shot_order, clip.shot_id, clip.sequence_index,
                                  state_owner=ctx.state_owner, entity_ids=ctx.entities)
        active_set = set(continuity.active_states_at(ctx.scene, ctx.shot_order, clip.shot_id, clip.sequence_index))
        on_screen = continuity.on_screen_speakers(clip)
        referenced = list(refs.entities)
        for cid in on_screen:
            if cid in ctx.entities and cid not in referenced:
                referenced.append(cid)
        resolved = {eid: refs.explicit_states.get(eid) or active.get(eid) for eid in referenced}
        return _Facts(refs=refs, active=active, active_set=active_set, referenced=referenced, resolved=resolved,
                      speakers=continuity.speakers_in_clip(clip), on_screen=on_screen)

    # -- L1 ----------------------------------------------------------------------------------
    def _strip(self, where: str, text: str, allowed: Iterable[str], setter: Callable[[str], Any]) -> None:
        clean, removed = strip_banned(text, allowed)
        if removed:
            words = ", ".join(dict.fromkeys(removed))
            self.fixable("L1", f"stripped banned word(s) {words}", where=where, apply=lambda: setter(clean),
                         suggestion=clean)

    def l1_banned_words(self) -> None:
        allowed = list(self.ctx.settings.lint_allowed_words)
        clip = self.clip
        visual = clip.spec.visual
        audio = clip.spec.audio_spec
        for name in VISUAL_FIELDS:
            self._strip(f"spec.visual.{name}", getattr(visual, name) or "", allowed,
                        lambda value, n=name: setattr(visual, n, value))
        self._strip("performance_state", clip.performance_state or "", allowed,
                    lambda value: setattr(clip, "performance_state", value))
        self._strip("spec.audio_spec.ambience", audio.ambience or "", allowed,
                    lambda value: setattr(audio, "ambience", value))
        for group in ("dialogue", "off_screen"):
            for i, line in enumerate(getattr(audio, group)):
                self._strip(f"spec.audio_spec.{group}[{i}].delivery", line.delivery or "", allowed,
                            lambda value, ln=line: setattr(ln, "delivery", value))
        cleaned: list[str] = []
        removed: list[str] = []
        for item in audio.diegetic_sfx:
            clean, words = strip_banned(item or "", allowed)
            removed.extend(words)
            if clean:
                cleaned.append(clean)
        if removed:
            words_text = ", ".join(dict.fromkeys(removed))
            self.fixable("L1", f"stripped banned word(s) {words_text}", where="spec.audio_spec.diegetic_sfx",
                         apply=lambda: setattr(audio, "diegetic_sfx", cleaned), suggestion="; ".join(cleaned))

    # -- L17 ---------------------------------------------------------------------------------
    def l17_music(self) -> None:
        audio = self.clip.spec.audio_spec
        if (audio.music or "").strip().casefold() != "none":
            was = audio.music
            self.fixable("L17", f"music must be none (was {was!r}); the score is laid in assembly",
                         where="spec.audio_spec.music", apply=lambda: setattr(audio, "music", "none"),
                         suggestion="none")
        clean, dropped = strip_music_terms(audio.ambience)
        if dropped:
            gone = "; ".join(dropped)
            self.fixable("L17", f"removed music term(s) from ambience: {gone}", where="spec.audio_spec.ambience",
                         apply=lambda: setattr(audio, "ambience", clean), suggestion=clean)
        musical = [s for s in audio.diegetic_sfx if MUSIC_TERMS_RX.search(s or "")]
        if musical:
            kept = [s for s in audio.diegetic_sfx if not MUSIC_TERMS_RX.search(s or "")]
            gone = ", ".join(musical)
            self.fixable("L17", f"removed music sound effect(s): {gone}", where="spec.audio_spec.diegetic_sfx",
                         apply=lambda: setattr(audio, "diegetic_sfx", kept), suggestion="; ".join(kept))
        for group in ("dialogue", "off_screen"):
            for i, line in enumerate(getattr(audio, group)):
                clean_d, dropped_d = strip_music_terms(line.delivery)
                if dropped_d:
                    gone = "; ".join(dropped_d)
                    self.fixable("L17", f"removed music term(s) from delivery: {gone}",
                                 where=f"spec.audio_spec.{group}[{i}].delivery",
                                 apply=lambda ln=line, c=clean_d: setattr(ln, "delivery", c), suggestion=clean_d)

    # -- L12 ---------------------------------------------------------------------------------
    def needs(self, facts: _Facts) -> dict[str, Any]:
        ctx, clip = self.ctx, self.clip
        cond = clip.conditioning
        packs = [sid for sid in dict.fromkeys(cond.reference_entity_states) if sid in ctx.states]
        for eid in facts.referenced:
            sid = facts.resolved.get(eid)
            state = ctx.states.get(sid) if sid else None
            if state is not None and sid not in packs and not self.text_only_ok(ctx.entities[eid], state, facts):
                packs.append(sid)
        anchor = bool(cond.spatial_anchor)
        native = scene_audio_mode(ctx.scene) == AudioMode.NATIVE_AV
        first = bool(clip.anchor_from_clip_id or cond.first_frame_anchor)
        refs = bool(packs) or anchor
        return {
            "native_audio": native,
            "dialogue_in_prompt": native and bool(clip.spec.audio_spec.all_lines()),
            "reference_to_video": refs,
            "multi_subject_refs": len(packs) > 1,
            "first_frame": first,
            "last_frame": bool(cond.last_frame_target),
            "refs_with_first_frame": first and refs,
            "n_reference_images": len(packs) + (1 if anchor else 0),
        }

    def l12_select_model(self, facts: _Facts) -> None:
        ctx, clip = self.ctx, self.clip
        requested = self.requested or clip.model_target.preferred
        candidates = [requested]
        fallback = clip.model_target.fallback
        if self.requested is None and fallback and fallback != requested:
            candidates.append(fallback)  # an explicit model_id (re-lint for a fallback) is checked alone
        needs = self.needs(facts)
        reasons: dict[str, list[str]] = {}
        for i, mid in enumerate(candidates):
            try:
                spec = ctx.registry.get(mid)
            except UnknownModelError:
                reasons[mid] = [f"{mid} is not in the model registry"]
                continue
            problems = feature_problems(spec, needs, ctx.settings)
            if i:
                problems.extend(duration_problems(spec, float(clip.duration_s)))
            if not problems:
                self.model_used, self.spec = mid, spec
                if i:
                    why = "; ".join(reasons.get(requested, []))
                    self.add("L12", "fallback", f"{requested} cannot render this clip ({why}) — using fallback {mid}",
                             where="model_target")
                return
            reasons[mid] = problems
        self.model_used = requested
        try:
            self.spec = ctx.registry.get(requested)
        except UnknownModelError:
            self.spec = None
        detail = " | ".join(f"{m}: " + "; ".join(p) for m, p in reasons.items())
        self.add("L12", "fail", f"no available model can render this clip — {detail}", where="model_target")

    # -- L11 ---------------------------------------------------------------------------------
    def l11_negatives(self, facts: _Facts) -> None:
        ctx, clip = self.ctx, self.clip
        sids = merge_unique([s for s in facts.resolved.values() if s], clip.conditioning.reference_entity_states)
        states = [ctx.states[s] for s in sids if s in ctx.states]
        required = required_negatives(ctx.style, states, scene_audio_mode(ctx.scene))
        have = {n.strip().casefold() for n in clip.spec.negative}
        missing = [n for n in required if n.casefold() not in have]
        if missing:
            fixed = merge_unique(clip.spec.negative, missing)
            spec = clip.spec
            listed = ", ".join(missing)
            self.fixable("L11", f"added missing negative(s): {listed}", where="spec.negative",
                         apply=lambda: setattr(spec, "negative", fixed), suggestion=", ".join(fixed))

    # -- L2 ----------------------------------------------------------------------------------
    def l2_physical_verbs(self) -> None:
        verbs = physical_verbs(self.clip.spec.visual.subject_and_action)
        if len(verbs) <= 1:
            return
        listed = ", ".join(verbs)
        msg = f"subject_and_action has {len(verbs)} physical verbs ({listed}); a clip carries ONE atomic action"
        if self.ctx.shot.multibeat_ok:
            self.add("L2", "warn", msg + " (multi-beat staging approved for this shot)",
                     where="spec.visual.subject_and_action")
        else:
            self.add("L2", "bounce", msg + " — the Shot Decomposer must split it",
                     where="spec.visual.subject_and_action")

    # -- L3 ----------------------------------------------------------------------------------
    def l3_registered_ids(self, facts: _Facts) -> None:
        ctx, clip = self.ctx, self.clip
        audio = clip.spec.audio_spec
        unknown = list(facts.refs.unknown)
        known = {*ctx.entities, *ctx.states, *ctx.environments, *ctx.voice_ids, *ctx.lighting_ids, *ctx.acronyms}
        extra = "\n".join([clip.performance_state or "", audio.ambience or "", *audio.diegetic_sfx,
                           *(ln.delivery or "" for ln in audio.all_lines())])
        for tok in continuity.entity_tokens(extra):
            if tok not in known and tok not in unknown:
                unknown.append(tok)
        if unknown:
            listed = ", ".join(unknown)
            self.add("L3", "fail", f"unregistered id(s) {listed} — prompts may only name registered Entity / "
                     "EntityState / Environment / Voice ids (register the asset, rephrase, or add camera "
                     "acronyms to lint_extra_acronyms)", where="spec")
        strangers = list(dict.fromkeys(ln.character_id for ln in audio.all_lines()
                                       if ln.character_id not in ctx.entities))
        if strangers:
            listed = ", ".join(strangers)
            self.add("L3", "fail", f"speaker id(s) {listed} are not registered entities", where="spec.audio_spec")

    # -- L4 ----------------------------------------------------------------------------------
    def l4_active_states(self, facts: _Facts) -> None:
        ctx, clip = self.ctx, self.clip
        at = f"{clip.shot_id} clip {clip.sequence_index}"
        active = ", ".join(sorted(facts.active_set)) or "none"
        named = list(dict.fromkeys([*facts.refs.explicit_states.values(), *facts.refs.conflicting_states]))
        for sid in named:
            if sid not in facts.active_set:
                self.add("L4", "fail", f"{sid} is not active at {at} (active: {active})", where="spec.visual")
        for sid in dict.fromkeys(facts.refs.conflicting_states):
            owner = ctx.state_owner.get(sid, "?")
            self.add("L4", "fail", f"{sid} conflicts with another state of {owner} named in the same clip",
                     where="spec.visual")
        for eid in facts.referenced:
            if not facts.resolved.get(eid):
                self.add("L4", "fail", f"{eid} is in the clip but has no active state at {at} — add it to "
                         "scene.entity_states_active or a mid-scene transition", where="spec.visual")
        for sid in dict.fromkeys(clip.conditioning.reference_entity_states):
            if sid in ctx.states and sid not in facts.active_set:
                self.add("L4", "fail", f"conditioning attaches the {sid} pack, which is not active at {at}",
                         where="conditioning.reference_entity_states")

    # -- L5 ----------------------------------------------------------------------------------
    def l5_packs(self, facts: _Facts) -> None:
        ctx, clip = self.ctx, self.clip
        needed = merge_unique([s for s in facts.resolved.values() if s], clip.conditioning.reference_entity_states)
        for sid in needed:
            state = ctx.states.get(sid)
            owner = ctx.entities.get(ctx.state_owner.get(sid, ""))
            if state is None or owner is None:
                continue  # unknown ids are reported by L3 / L6
            if state.approval_status != ApprovalStatus.LOCKED:
                status = ApprovalStatus(state.approval_status).value
                self.add("L5", "fail", f"{sid} reference pack is {status}, not locked — approve the pack first",
                         where="conditioning")
            if self.text_only_ok(owner, state, facts):
                continue
            missing = missing_pack_roles(state.reference_pack, owner.tier, owner.type)
            if missing:
                listed = ", ".join(missing)
                self.add("L5", "fail", f"{sid} pack lacks {listed} required for a {Tier(owner.tier).value} "
                         f"{EntityType(owner.type).value}", where="conditioning")
            absent = [u for u in state.reference_pack.image_uris() if not self.exists(u)]
            if absent:
                listed = ", ".join(absent)
                self.add("L5", "fail", f"{sid} pack image(s) missing on disk: {listed}", where="conditioning")

    # -- L6 ----------------------------------------------------------------------------------
    def l6_conditioning(self, facts: _Facts) -> None:
        ctx, clip = self.ctx, self.clip
        cond = clip.conditioning
        attached = set(cond.reference_entity_states)
        uris = set(cond.reference_uris)
        for sid in dict.fromkeys(cond.reference_entity_states):
            if sid not in ctx.states:
                self.add("L6", "fail", f"conditioning names unknown entity-state {sid}",
                         where="conditioning.reference_entity_states")
        for eid in facts.referenced:
            sid = facts.resolved.get(eid)
            if not sid or sid not in ctx.states:
                continue  # L4 reports entities without an active state
            state = ctx.states[sid]
            if self.text_only_ok(ctx.entities[eid], state, facts):
                continue
            if sid not in attached:
                self.add("L6", "fail", f"{eid} is in the clip but its {sid} pack is not attached",
                         where="conditioning.reference_entity_states")
            elif not uris.intersection(state.reference_pack.image_uris()):
                self.add("L6", "fail", f"{sid} is attached but none of its pack images are in reference_uris",
                         where="conditioning.reference_uris")
        missing = [u for u in cond.reference_uris if not self.exists(u)]
        if missing:
            listed = ", ".join(missing)
            self.add("L6", "fail", f"reference image(s) missing on disk: {listed}", where="conditioning.reference_uris")
        env = ctx.environment
        env_id = env.id if env is not None else (facts.refs.environments[0] if facts.refs.environments else None)
        if env_id is None:
            return
        if not cond.spatial_anchor:
            expected = env.angle_uri(clip.environment_angle) if env is not None else None
            angle = clip.environment_angle or "geometry"
            self.add("L6", "fail", f"no spatial anchor: {env_id} is referenced, so its {angle} still must be attached",
                     where="conditioning.spatial_anchor", suggestion=expected)
        elif not self.exists(cond.spatial_anchor):
            self.add("L6", "fail", f"spatial anchor {cond.spatial_anchor} is missing on disk",
                     where="conditioning.spatial_anchor")

    # -- L7 ----------------------------------------------------------------------------------
    def l7_lighting_palette(self) -> None:
        visual = self.clip.spec.visual
        colours = _unique_matches(COLOR_RX, visual.lighting)
        if colours:
            listed = ", ".join(colours)
            self.add("L7", "fail", f"lighting names colour(s) {listed}; lighting describes physical sources only "
                     "(colour belongs in palette)", where="spec.visual.lighting",
                     suggestion=tidy_text(COLOR_RX.sub(" ", visual.lighting)))
        sources = _unique_matches(LIGHT_SOURCE_RX, visual.palette)
        if sources:
            listed = ", ".join(sources)
            self.add("L7", "fail", f"palette names light source(s) {listed}; light sources belong in lighting",
                     where="spec.visual.palette", suggestion=tidy_text(LIGHT_SOURCE_RX.sub(" ", visual.palette)))

    # -- L8 ----------------------------------------------------------------------------------
    def l8_duration(self) -> None:
        if self.spec is None:
            return
        for problem in duration_problems(self.spec, float(self.clip.duration_s)):
            self.add("L8", "bounce", f"{problem} — the Shot Decomposer must re-split or re-snap this clip",
                     where="duration_s")

    # -- L9 ----------------------------------------------------------------------------------
    def l9_first_frame(self) -> None:
        ctx, clip = self.ctx, self.clip
        cond = clip.conditioning
        anchor_id = clip.anchor_from_clip_id
        where = "conditioning.first_frame_anchor"
        if not anchor_id:
            if cond.first_frame_anchor:
                self.add("L9", "warn", "first_frame_anchor is set but anchor_from_clip_id is empty, so its "
                         "provenance cannot be checked", where=where)
            return
        try:
            anchor = ctx.project.load_clip(validate_id(anchor_id, "clip id"))
        except (FileNotFoundError, ValueError) as exc:
            self.add("L9", "fail", f"anchor clip {anchor_id} cannot be loaded ({type(exc).__name__})",
                     where="anchor_from_clip_id")
            return
        if anchor.status == ClipStatus.STALE:
            self.add("L9", "fail", f"anchor clip {anchor_id} is stale — re-decompose the shot",
                     where="anchor_from_clip_id")
            return
        if anchor.shot_id != clip.shot_id or anchor.sequence_index != clip.sequence_index - 1:
            self.add("L9", "warn", f"anchor clip {anchor_id} is not the clip immediately before {clip.id}",
                     where="anchor_from_clip_id")
        if not anchor.selected_take_id:
            self.add("L9", "fail", f"waiting for an approved take of {anchor_id}: its last frame becomes this "
                     f"clip's first frame (generate and select {anchor_id} first)", where=where)
            return
        try:
            take = ctx.project.load_take(validate_id(anchor.selected_take_id, "take id"))
        except (FileNotFoundError, ValueError):
            self.add("L9", "fail", f"selected take {anchor.selected_take_id} of {anchor_id} cannot be loaded",
                     where=where)
            return
        if take.status != TakeStatus.SELECTED:
            status = TakeStatus(take.status).value
            self.add("L9", "fail", f"{take.id} is {status}, not an approved (selected) take", where=where)
            return
        if not take.last_frame_uri:
            self.add("L9", "fail", f"{take.id} has no extracted last frame yet", where=where)
            return
        if cond.first_frame_anchor != take.last_frame_uri or cond.first_frame_anchor_take_id != take.id:
            found = cond.first_frame_anchor or "nothing"
            self.add("L9", "fail", f"first_frame_anchor must be {take.last_frame_uri} (last frame of {take.id}); "
                     f"found {found} — link it with decomposer.link_anchor_frame", where=where,
                     suggestion=take.last_frame_uri)
            return
        if not self.exists(take.last_frame_uri):
            self.add("L9", "fail", f"anchor frame {take.last_frame_uri} is missing on disk", where=where)

    # -- L10 ---------------------------------------------------------------------------------
    def l10_visual_fields(self) -> None:
        visual = self.clip.spec.visual
        order = tuple(type(visual).model_fields)
        if order[: len(VISUAL_FIELDS)] != VISUAL_FIELDS:
            self.add("L10", "fail", "visual fields are not in the fixed 9-field order", where="spec.visual")
        empty = [name for name in VISUAL_FIELDS if not (getattr(visual, name, "") or "").strip()]
        if empty:
            listed = ", ".join(empty)
            self.add("L10", "fail", f"empty visual field(s): {listed} — all 9 fields are required "
                     "(subject_and_action ... attribution)", where="spec.visual")

    # -- L13 ---------------------------------------------------------------------------------
    def l13_axis(self) -> None:
        env = self.ctx.environment
        if env is None or not (env.axis_of_action or "").strip():
            return
        side = axis_side(env.axis_of_action)
        if not side:
            return
        cam_side = (self.clip.camera_side or "").strip().casefold()
        if not cam_side:
            self.add("L13", "warn", f"camera_side is not set; {env.id} keeps the camera on the {side} side of the "
                     "axis of action", where="camera_side")
        elif side not in cam_side and cam_side not in side:
            self.add("L13", "warn", f"camera_side {self.clip.camera_side!r} may cross the line: {env.id} keeps the "
                     f"camera on the {side} side ({env.axis_of_action})", where="camera_side")
        opposite = _COMPASS_OPPOSITE.get(side)
        if opposite and re.search(r"\b" + opposite + r"\b", self.clip.spec.visual.camera or "", re.IGNORECASE):
            self.add("L13", "warn", f"camera text mentions the {opposite} side but {env.id} keeps the camera "
                     f"{side} of the line", where="spec.visual.camera")

    # -- L14 (clip part; the shot part is in lint_shot) -----------------------------------------
    def l14_dialogue(self) -> None:
        ctx, clip = self.ctx, self.clip
        audio = clip.spec.audio_spec
        seen: list[str] = []
        for group, lines in (("dialogue", audio.dialogue), ("off_screen", audio.off_screen)):
            for i, line in enumerate(lines):
                path = f"spec.audio_spec.{group}[{i}]"
                if not line.line_id:
                    self.add("L14", "fail", f"{path} has no line_id; every spoken line must come verbatim from "
                             "scene.dialogue", where=path)
                    continue
                src = ctx.lines.get(line.line_id)
                if src is None:
                    self.add("L14", "fail", f"{line.line_id} is not in {ctx.scene.id}.dialogue", where=path)
                    continue
                seen.append(line.line_id)
                if line.text != src.text:
                    self.add("L14", "fail", f"{line.line_id} is not verbatim: expected \u201c{src.text}\u201d, got "
                             f"\u201c{line.text}\u201d", where=f"{path}.text", suggestion=src.text)
                if line.character_id != src.character_id:
                    self.add("L14", "fail", f"{line.line_id} belongs to {src.character_id}, not {line.character_id}",
                             where=f"{path}.character_id")
                if group == "dialogue" and src.off_screen:
                    self.add("L14", "fail", f"{line.line_id} is scripted off-screen but sits in on-screen dialogue "
                             "(every on-screen mouth must stay closed for it)", where=path)
                if src.transliteration and line.transliteration != src.transliteration:
                    self.add("L14", "warn", f"{line.line_id} transliteration differs from scene.dialogue",
                             where=f"{path}.transliteration", suggestion=src.transliteration)
        counts = Counter(seen)
        repeated = [lid for lid, n in counts.items() if n > 1]
        if repeated:
            listed = ", ".join(repeated)
            self.add("L14", "fail", f"line(s) {listed} are spoken more than once in this clip", where="spec.audio_spec")
        declared = list(clip.lines)
        doubled = [lid for lid, n in Counter(declared).items() if n > 1]
        if doubled:
            listed = ", ".join(doubled)
            self.add("L14", "fail", f"clip.lines lists {listed} more than once", where="lines")
        silent = [lid for lid in dict.fromkeys(declared) if lid not in counts]
        if silent:
            listed = ", ".join(silent)
            self.add("L14", "fail", f"clip.lines lists {listed} but audio_spec does not speak it", where="lines")
        undeclared = [lid for lid in counts if lid not in declared]
        if undeclared:
            listed = ", ".join(undeclared)
            self.add("L14", "fail", f"audio_spec speaks {listed}, which is missing from clip.lines", where="lines")

    # -- L15 ---------------------------------------------------------------------------------
    def l15_voices(self, facts: _Facts) -> None:
        ctx, clip = self.ctx, self.clip
        audio = clip.spec.audio_spec
        for group, lines in (("dialogue", audio.dialogue), ("off_screen", audio.off_screen)):
            for i, line in enumerate(lines):
                path = f"spec.audio_spec.{group}[{i}].voice"
                vp = ctx.voices.get(line.character_id)
                if vp is None:
                    self.add("L15", "fail", f"{line.character_id} speaks but has no VoiceProfile (cast the voice "
                             "at G1)", where=path)
                    continue
                if vp.approval_status != ApprovalStatus.LOCKED:
                    status = ApprovalStatus(vp.approval_status).value
                    self.add("L15", "fail", f"{vp.id} for {line.character_id} is {status}, not locked", where=path)
                state = facts.active.get(line.character_id)
                expected = vp.voice_for_state(state)
                if not (line.voice or "").strip():
                    self.add("L15", "fail", f"voice for {line.line_id or line.character_id} is empty; it must carry "
                             f"{vp.id}'s description", where=path, suggestion=expected)
                elif line.voice.strip() != expected.strip():
                    self.add("L15", "fail", f"voice for {line.line_id or line.character_id} does not match {vp.id} "
                             f"(state {state or 'base'})", where=path, suggestion=expected)

    # -- L16 ---------------------------------------------------------------------------------
    def l16_speakers(self, facts: _Facts) -> None:
        if self.spec is None or self.model_used is None:
            return
        limit = effective_max_speakers(self.ctx.registry, self.model_used, self.ctx.settings)
        if len(facts.speakers) > limit:
            listed = ", ".join(facts.speakers)
            rule = " (workspace single-speaker rule)" if self.ctx.settings.single_speaker_clips else ""
            self.add("L16", "bounce", f"{len(facts.speakers)} speakers ({listed}) exceed {limit} per clip for "
                     f"{self.model_used}{rule} — split into singles / reaction cutaways", where="lines")

    # -- L18 ---------------------------------------------------------------------------------
    def l18_speech_time(self) -> None:
        s = self.ctx.settings
        texts = continuity.clip_line_texts(self.clip)
        if not texts:
            return
        speech = continuity.estimate_speech_s(texts, s.speech_pace_wps)
        duration = float(self.clip.duration_s)
        budget = duration - s.speech_margin_s
        if speech > budget + _EPS:
            words = sum(continuity.count_words(t) for t in texts)
            self.add("L18", "bounce", f"{words} words need about {speech:.1f}s at {s.speech_pace_wps:g} words/s but "
                     f"only {max(budget, 0.0):.1f}s fit in a {duration:g}s clip ({s.speech_margin_s:g}s margin) — "
                     "extend the clip or split the line", where="duration_s")


# ---------------------------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------------------------
def _build_report(clip: GenClip, findings: list[LintFinding], auto_fixes: list[str],
                  model_used: str | None) -> LintReport:
    ordered = sorted(findings, key=lambda f: _rule_no(f.rule))
    blocking = any(f.action in BLOCKING_ACTIONS for f in ordered)
    return LintReport(
        ts=now_iso(),
        passed=not blocking,
        blocking=blocking,
        bounce_to_decomposer=any(f.action == "bounce" for f in ordered),
        findings=ordered,
        auto_fixes=list(auto_fixes),
        model_used=model_used,
        spec_hash=clip_lint_hash(clip),
    )


def _apply_report(clip: GenClip, report: LintReport) -> GenClip:
    """Attach the report; PENDING -> LINTED on pass, LINTED -> PENDING on failure. Other statuses are left alone
    (a NEEDS_RELINT clip with a selected take stays NEEDS_RELINT so the orchestrator still regenerates it)."""
    clip.lint_report = report
    if report.passed:
        if clip.status == ClipStatus.PENDING or (clip.status == ClipStatus.NEEDS_RELINT and not clip.selected_take_id):
            clip.status = ClipStatus.LINTED
    elif clip.status == ClipStatus.LINTED:
        clip.status = ClipStatus.PENDING
    return clip


def lint_clip(ctx: LintContext, clip: GenClip, *, model_id: str | None = None,
              apply_fixes: bool = True) -> tuple[GenClip, LintReport]:
    """Lint one clip -> (fixed copy carrying the report, report). Never writes anything."""
    if clip.shot_id != ctx.shot.id:
        raise ValueError(f"{clip.id} belongs to {clip.shot_id}, not to the context shot {ctx.shot.id}")
    fixed, findings, auto_fixes, model_used = _ClipLinter(ctx, clip, model_id=model_id, apply_fixes=apply_fixes).run()
    report = _build_report(fixed, findings, auto_fixes, model_used)
    return _apply_report(fixed, report), report


def _shot_level_l14(ctx: LintContext, staged: list[tuple[GenClip, list[LintFinding], list[str], str | None]]) -> None:
    owners: dict[str, list[str]] = {}
    for clip, _findings, _fixes, _model in staged:
        for lid in dict.fromkeys(clip.lines):
            owners.setdefault(lid, []).append(clip.id)
    for clip, findings, _fixes, _model in staged:
        for lid in dict.fromkeys(clip.lines):
            if len(owners[lid]) > 1:
                where = ", ".join(owners[lid])
                findings.append(_finding("L14", "fail", f"{lid} is spoken in {len(owners[lid])} clips ({where}); "
                                         "every line appears exactly once per shot", where="lines"))
    missing = [lid for lid in dict.fromkeys(ctx.shot.lines) if lid not in owners]
    if missing:
        listed = ", ".join(missing)
        message = f"line(s) {listed} of {ctx.shot.id} are not in any clip — re-decompose the shot"
        for _clip, findings, _fixes, _model in staged:
            findings.append(_finding("L14", "fail", message, where="lines"))


def lint_shot(project: Project, registry: Registry, shot_id: str, *, apply_fixes: bool = True,
              save: bool = True) -> list[tuple[GenClip, LintReport]]:
    """Lint every live (non-stale) clip of a shot, including the shot-level L14 check (each line exactly once).

    ``save=True`` stores each fixed clip (with its ``lint_report`` and status) and logs the agent call.
    """
    ids = parse_artifact_id(validate_id(shot_id, "shot id"))
    if ids["shot"] != shot_id:
        raise ValueError(f"{shot_id!r} is not a shot id (expected SC12_SH04)")
    ctx = LintContext.build(project, registry, str(ids["scene"]), shot_id)
    clips = [c for c in ctx.shot_clips if c.status != ClipStatus.STALE]
    staged = [_ClipLinter(ctx, c, model_id=None, apply_fixes=apply_fixes).run() for c in clips]
    _shot_level_l14(ctx, staged)
    results: list[tuple[GenClip, LintReport]] = []
    for fixed, findings, auto_fixes, model_used in staged:
        report = _build_report(fixed, findings, auto_fixes, model_used)
        results.append((_apply_report(fixed, report), report))
    if save:
        for fixed, _report in results:
            project.save_clip(fixed)
        log_agent_call(
            project, agent=AGENT_NAME, scene_id=ctx.scene.id,
            working_set_hash=scene_working_set_hash(project, ctx.scene.id),
            task=f"lint {shot_id}", mode="deterministic",
            inputs={"shot_id": shot_id, "apply_fixes": apply_fixes,
                    "clips": {c.id: clip_lint_hash(c) for c in clips}},
            outputs={"reports": {fixed.id: {
                "passed": report.passed, "blocking": report.blocking,
                "bounce_to_decomposer": report.bounce_to_decomposer, "model_used": report.model_used,
                "spec_hash": report.spec_hash, "auto_fixes": report.auto_fixes,
                "findings": [f.model_dump(mode="json") for f in report.findings],
            } for fixed, report in results}},
        )
    return results


def lint_scene(project: Project, registry: Registry, scene_id: str, **kw: Any) -> dict[str, LintReport]:
    """Lint every saved shot of the scene (scene order) -> clip id -> report."""
    scene = project.load_scene(validate_id(scene_id, "scene id"))
    out: dict[str, LintReport] = {}
    for sid in scene_shot_order(project, scene):
        if (project.shot_dir(sid) / "shot.yaml").exists():
            for clip, report in lint_shot(project, registry, sid, **kw):
                out[clip.id] = report
    return out


def summarize(reports: Mapping[str, LintReport] | Iterable[Any]) -> dict[str, Any]:
    """Pass rate and rule counts for a set of reports (dict clip id -> report, reports, or (clip, report))."""
    items: list[tuple[str | None, LintReport]] = []
    if isinstance(reports, Mapping):
        items = [(str(k), v) for k, v in reports.items()]
    else:
        for entry in reports:
            if isinstance(entry, tuple):
                clip, report = entry
                items.append((getattr(clip, "id", None), report))
            else:
                items.append((None, entry))
    n = len(items)
    passed = sum(1 for _, r in items if r.passed)
    rules = Counter(f.rule for _, r in items for f in r.findings)
    errors = Counter(f.rule for _, r in items for f in r.findings if f.severity == "error")
    return {
        "clips": n,
        "passed": passed,
        "failed": n - passed,
        "pass_rate": round(passed / n, 4) if n else 1.0,
        "blocking": sum(1 for _, r in items if r.blocking),
        "bounced": sum(1 for _, r in items if r.bounce_to_decomposer),
        "auto_fixes": sum(len(r.auto_fixes) for _, r in items),
        "rule_counts": dict(sorted(rules.items(), key=lambda kv: _rule_no(kv[0]))),
        "error_rule_counts": dict(sorted(errors.items(), key=lambda kv: _rule_no(kv[0]))),
        "failing_clips": [cid for cid, r in items if cid and not r.passed],
    }
