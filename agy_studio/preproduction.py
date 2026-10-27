"""Pre-production — BUILD_SPEC_v3 §6 Story department (Script Analyst, Breakdown Analyst) and §11 (G1 packet).

* ``ingest_script``  — fountain-like parsing into Bible ``Scene`` records with VERBATIM dialogue (exact Hindi in
  Devanagari + optional phonetic transliteration), ``SCnn`` / ``SCnn_Lnnn`` ids, ``CHAR_<ASCII NAME>`` speakers and
  off-screen detection for (V.O.) / (O.S.) / (O.C.) / (PHONE) cues. The script file is copied INTO the project
  (``script/script_v<n>.<ext>``, immutable) — it is never referenced in place.
* ``compute_breakdown`` — deterministic §4.5 SceneBreakdown (densities, speakers, transitions, crowd, camera
  complexity, newly introduced entities/speakers, clip/cost estimate from the registry, risk tier, model
  recommendation with rationale, audio mode).
* ``build_schedule`` — ``[pilot] + scenes ordered by first introduction of reused entities/speakers`` (then the rest,
  which only reuse locked assets and can run in parallel).
* ``g1_packet_items`` — the G1 presentation (Bible + state timeline + StyleLock + voice profiles + schedule) incl.
  the §15 Open Decisions in plain filmmaking language with the spec defaults recommended.

Nothing here locks anything: G1 is the human's Vision-tier decision.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from pydantic import Field

from .continuity import all_scene_states, count_words, entity_id_of_state
from .fsio import generated_by, read_record, validate_id, write_new
from .models import (
    AudioMode,
    Model,
    OpenDecisions,
    RiskTier,
    Scene,
    SceneBreakdown,
    SceneDialogueLine,
    SceneStatus,
)
from .project import Project, scene_id as make_scene_id

AGENT = "agy.preproduction"
SCRIPT_EXTENSIONS = (".fountain", ".txt")
MAX_SCRIPT_BYTES = 5 * 1024 * 1024
MAX_WORDS_PER_BREATH = 14  # workspace rule: 8–14 words per 8 s shot

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_HEADING_RE = re.compile(r"^(?:INT\.?/EXT|EXT\.?/INT|INT/EXT|I/E|INT|EXT|EST)[.\s]", re.IGNORECASE)
_SCENE_N_RE = re.compile(r"^(?:SCENE|दृश्य)\s*(\d+)\b[\s:.\-–—]*(.*)$", re.IGNORECASE)
_FORCED_HEADING_RE = re.compile(r"^\.(?=[A-Za-z0-9])")
_TRANSITION_RE = re.compile(r"^(?:[A-Z][A-Z .\-]*TO:|FADE (?:IN|OUT)[:.]?|CUT TO BLACK\.?|THE END\.?)$")
_CUE_NAME_RE = re.compile(r"^[A-Z0-9][A-Z0-9 .'’\-&]{0,47}$")
_EXT_RE = re.compile(r"\(([^()]*)\)\s*$")
_TRANSLIT_PREFIX_RE = re.compile(r"^(?:translit|transliteration)\s*:\s*(.*)$", re.IGNORECASE)
_BRACKET_LINE_RE = re.compile(r"^\[(?!\[)(.*)\]$")
_TRAILING_BRACKET_RE = re.compile(r"^(.*?)\s*\[(?!\[)([^\[\]]+)\]\s*$")
_NOTE_RE = re.compile(r"\[\[.*?\]\]", re.DOTALL)
_TITLE_KV_RE = re.compile(r"^[A-Za-z][A-Za-z ]{0,30}:\s*\S")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+")

OFF_SCREEN_EXTENSIONS = frozenset({"VO", "OS", "OC", "PHONE", "ONPHONE", "OFF", "OFFSCREEN", "FILTERED"})
CONTINUED_EXTENSIONS = frozenset({"CONTD", "CONT", "CONTINUED", "CONTINUING"})

_PHYSICAL_RE = re.compile(
    r"\b(grab\w*|push\w*|shove\w*|hug\w*|embrac\w*|punch\w*|slap\w*|kick\w*|fight\w*|wrestl\w*|tackl\w*|"
    r"drag\w*|carr(?:y|ies|ied)|catch\w*|caught|lift\w*|pull\w*|kiss\w*|strangl\w*|chok\w*|shak\w* hands|"
    r"hands? (?:it|him|her|the) |snatch\w*|grip\w*|seiz\w*)\b", re.IGNORECASE)
_CROWD_RE = re.compile(r"\b(crowd\w*|mob|audience|bystanders|passers?-by|onlookers|dozens of|hundreds of|"
                       r"people gather\w*|packed with people)\b|भीड़", re.IGNORECASE)
_CAMERA_RE = re.compile(r"\b(tracking|tracks with|crane|drone|aerial|whip[- ]pan|orbit\w*|360|steadicam|dolly|"
                        r"zoom\w*|rack focus|pov|slow[- ]motion|time[- ]lapse|chase\w*|sprint\w*)\b", re.IGNORECASE)
_DIEGETIC_RE = re.compile(
    r"\b(ring\w*|buzz\w*|vibrat\w*|knock\w*|honk\w*|siren\w*|alarm\w*|crash\w*|bang\w*|thud\w*|slam\w*|"
    r"shout\w*|scream\w*|footsteps|beep\w*|notification\w*|whistl\w*|doorbell|engine\w*|traffic|rain|thunder\w*|"
    r"radio|television|tv|applause|laugh\w*|clatter\w*|sizzl\w*|chime\w*|ding\w*)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------------------------
class ScriptIngestResult(Model):
    scenes: list[Scene] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)  # plain-language questions for G1
    new_characters: list[str] = Field(default_factory=list)  # CHAR_* ids not yet in the Bible
    characters: dict[str, str] = Field(default_factory=dict)  # every speaking id -> display name (from the cue)
    new_environments: dict[str, dict[str, Any]] = Field(default_factory=dict)  # ENV_* proposals (Location Designer)
    scene_environment_hints: dict[str, str] = Field(default_factory=dict)  # scene id -> proposed ENV_* id
    script_rel: str | None = None  # project-relative copy of the ingested script
    script_version: int | None = None


# ---------------------------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------------------------
def ascii_id_part(name: str) -> str:
    """``"Rishi's Father"`` -> ``RISHIS_FATHER`` (ASCII upper, underscores)."""
    cleaned = re.sub(r"['’`]", "", name or "")
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", cleaned).strip("_").upper()
    return re.sub(r"_+", "_", cleaned)


def character_id_for(name: str) -> str:
    part = ascii_id_part(name) or "UNNAMED"
    return validate_id(f"CHAR_{part}"[:96].rstrip("_"), "character id")


def _slugify(text: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return slug[:48].strip("_") or fallback


def _is_heading(line: str) -> bool:
    s = line.strip()
    return bool(s) and (bool(_HEADING_RE.match(s)) or bool(_SCENE_N_RE.match(s)) or bool(_FORCED_HEADING_RE.match(s)))


def _parse_heading(line: str) -> dict[str, Any]:
    s = re.sub(r"\s*#[^#]*#\s*$", "", line.strip())  # fountain scene numbers "#12#"
    m = _SCENE_N_RE.match(s)
    number = None
    if m:
        number = int(m.group(1))
        s = m.group(2).strip()
    if _FORCED_HEADING_RE.match(s):
        s = s[1:].strip()
    setting = None
    sm = re.match(r"^(INT\.?/EXT|EXT\.?/INT|INT/EXT|I/E|INT|EXT|EST)[.\s]+(.*)$", s, re.IGNORECASE)
    rest = s
    if sm:
        setting = sm.group(1).upper().rstrip(".")
        rest = sm.group(2).strip()
    parts = [p.strip() for p in re.split(r"\s+[-–—]\s+", rest) if p.strip()]
    location = parts[0] if parts else rest
    time_of_day = parts[-1].lower() if len(parts) > 1 else None
    return {"raw": line.strip(), "number": number, "setting": setting, "location": location,
            "time_of_day": time_of_day}


def _split_cue(line: str) -> tuple[str, list[str], bool] | None:
    """Return (name, normalized extensions, dual) if ``line`` looks like an ALL-CAPS character cue."""
    s = line.strip()
    dual = False
    if s.endswith("^"):
        dual, s = True, s[:-1].rstrip()
    forced = s.startswith("@")
    if forced:
        s = s[1:].strip()
    exts: list[str] = []
    while True:
        m = _EXT_RE.search(s)
        if not m:
            break
        exts.insert(0, m.group(1))
        s = s[: m.start()].rstrip()
    name = s.strip()
    if not name or _DEVANAGARI_RE.search(name) or name.endswith(":"):
        return None
    if not forced:
        if not _CUE_NAME_RE.match(name) or not re.search(r"[A-Z]", name) or name != name.upper():
            return None
    norm = [re.sub(r"[^A-Z]", "", e.upper()) for e in exts]
    return name, norm, dual


def _strip_notes(text: str) -> str:
    return _NOTE_RE.sub("", text)


def parse_fountain(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Split a fountain-like script into raw scenes: ``{heading, lines, blocks}``.

    ``blocks`` are ``{"kind": "action", "text": ...}`` or ``{"kind": "dialogue", "name", "extensions", "dual",
    "parentheticals", "text_lines", "translit_lines"}``. Returns (scenes, warnings).
    """
    lines = _strip_notes(text.replace("\r\n", "\n").replace("\r", "\n")).split("\n")
    scenes: list[dict[str, Any]] = []
    warnings: list[str] = []
    preamble: list[str] = []
    current: dict[str, Any] | None = None
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        s = raw.strip()
        if _is_heading(s):
            current = {"heading": _parse_heading(s), "lines": [raw], "blocks": []}
            scenes.append(current)
            i += 1
            continue
        if current is None:
            if s:
                preamble.append(s)
            i += 1
            continue
        prev_blank = i == 0 or not lines[i - 1].strip() or _is_heading(lines[i - 1])
        next_line = lines[i + 1].strip() if i + 1 < n else ""
        cue = _split_cue(s) if (s and prev_blank and next_line and not _TRANSITION_RE.match(s)) else None
        if cue is not None:
            name, exts, dual = cue
            block = {"kind": "dialogue", "name": name, "extensions": exts, "dual": dual, "parentheticals": [],
                     "text_lines": [], "translit_lines": [], "line_no": i + 1}
            current["lines"].append(raw)
            i += 1
            while i < n and lines[i].strip():
                d = lines[i].strip()
                current["lines"].append(lines[i])
                tm = _TRANSLIT_PREFIX_RE.match(d)
                bm = _BRACKET_LINE_RE.match(d)
                if tm:
                    block["translit_lines"].append(tm.group(1).strip())
                elif bm:
                    block["translit_lines"].append(bm.group(1).strip())
                elif d.startswith("(") and d.endswith(")"):
                    block["parentheticals"].append(d[1:-1].strip())
                else:
                    tb = _TRAILING_BRACKET_RE.match(d)
                    if tb and tb.group(1).strip():
                        block["text_lines"].append(tb.group(1).strip())
                        block["translit_lines"].append(tb.group(2).strip())
                    else:
                        block["text_lines"].append(d)
                i += 1
            current["blocks"].append(block)
            continue
        current["lines"].append(raw)
        if s and not _TRANSITION_RE.match(s):
            current["blocks"].append({"kind": "action", "text": s, "line_no": i + 1})
        i += 1
    meaningful = [p for p in preamble if not _TITLE_KV_RE.match(p)]
    if meaningful and scenes:
        warnings.append(f"Text before the first scene heading was not used: {meaningful[0][:80]!r}")
    if not scenes and preamble:
        current = {"heading": {"raw": "", "number": None, "setting": None, "location": "", "time_of_day": None},
                   "lines": [], "blocks": []}
        warnings.append("The script has no scene headings (INT./EXT. or SCENE n); it was read as ONE scene.")
        reparsed, _ = parse_fountain("SCENE 1\n\n" + text)
        return reparsed, warnings
    return scenes, warnings


def _action_sentences(text: str) -> list[str]:
    out: list[str] = []
    for part in _SENTENCE_SPLIT_RE.split(text or ""):
        part = part.strip()
        if part and re.search(r"[A-Za-z\u0900-\u097F]", part):
            out.append(part)
    return out


def scene_action_lines(scene: Scene) -> list[str]:
    """Physical action sentences of a scene (from its verbatim ``script_text``)."""
    if not scene.script_text.strip():
        return []
    text = scene.script_text if _is_heading(scene.script_text.strip().split("\n", 1)[0]) else \
        "SCENE 1\n" + scene.script_text
    raw_scenes, _ = parse_fountain(text)
    out: list[str] = []
    for rs in raw_scenes:
        for block in rs["blocks"]:
            if block["kind"] == "action":
                out.extend(_action_sentences(block["text"]))
    return out


# ---------------------------------------------------------------------------------------------
# ingest_script
# ---------------------------------------------------------------------------------------------
def _copy_script_into_project(project: Project, script_path: Path) -> tuple[Path, int, str]:
    src = Path(script_path)
    if src.is_symlink() or not src.is_file():
        raise ValueError(f"script must be a regular file (not a symlink/directory): {src}")
    ext = src.suffix.lower()
    if ext not in SCRIPT_EXTENSIONS:
        raise ValueError(f"script extension {ext!r} not allowed; use one of {SCRIPT_EXTENSIONS}")
    size = src.stat().st_size
    if size > MAX_SCRIPT_BYTES:
        raise ValueError(f"script is {size} bytes; the limit is {MAX_SCRIPT_BYTES}")
    data = src.read_bytes()
    text = data.decode("utf-8")  # raises UnicodeDecodeError for non-UTF-8 input
    script_dir = project.path("script")
    script_dir.mkdir(parents=True, exist_ok=True)
    existing = [int(m.group(1)) for p in script_dir.iterdir()
                if (m := re.fullmatch(r"script_v(\d+)\.[A-Za-z0-9]+", p.name))]
    version = (max(existing) if existing else 0) + 1
    dest = project.path("script", f"script_v{version}{ext}")
    write_new(dest, data)
    return dest, version, text


def _existing_scene_numbers(project: Project) -> tuple[int, int]:
    numbers, orders = [0], [0]
    for sc in project.list_scenes():
        m = re.fullmatch(r"SC(\d+)", sc.id)
        if m:
            numbers.append(int(m.group(1)))
        orders.append(sc.order)
    return max(numbers), max(orders)


def ingest_script(project: Project, script_path: Path | str, *, episode: int | None = None) -> ScriptIngestResult:
    """Parse a fountain-like Hindi script into pending Bible scenes (verbatim dialogue) and save them."""
    dest, version, text = _copy_script_into_project(project, Path(script_path))
    raw_scenes, warnings = parse_fountain(text)
    ambiguities: list[str] = list(warnings)
    entities = {e.id: e for e in project.list_entities()}
    by_display = {(e.display_name or "").strip().lower(): e.id for e in entities.values() if e.display_name}
    environments = {e.id for e in project.list_environments()}
    last_number, last_order = _existing_scene_numbers(project)

    result = ScriptIngestResult(script_rel=project.rel(dest), script_version=version)
    prov = generated_by(f"{AGENT}.ingest_script", notes=f"parsed from {project.rel(dest)}")
    for idx, rs in enumerate(raw_scenes, start=1):
        sid = make_scene_id(last_number + idx)
        heading = rs["heading"]
        dialogue: list[SceneDialogueLine] = []
        on_screen_chars: list[str] = []
        for block in rs["blocks"]:
            if block["kind"] != "dialogue":
                continue
            name = block["name"]
            names = [p.strip() for p in re.split(r"\s*&\s*|\s+AND\s+", name) if p.strip()]
            if len(names) > 1 or block["dual"]:
                ambiguities.append(
                    f"{sid}: '{name}' speak together — the single-speaker rule needs one speaker per line; "
                    f"the line was given to {names[0] if names else name}. Please split it if needed.")
            primary = names[0] if names else name
            char_id = by_display.get(primary.strip().lower()) or character_id_for(primary)
            result.characters.setdefault(char_id, primary.title())
            exts = block["extensions"]
            off_screen = any(e in OFF_SCREEN_EXTENSIONS for e in exts)
            unknown = [e for e in exts if e and e not in OFF_SCREEN_EXTENSIONS and e not in CONTINUED_EXTENSIONS]
            if unknown:
                ambiguities.append(f"{sid}: cue '{name}' has an unknown extension {unknown} — kept as a delivery "
                                   "note; is the speaker on screen?")
            text_lines = block["text_lines"]
            if not text_lines:
                ambiguities.append(f"{sid}: cue '{name}' (script line {block['line_no']}) has no dialogue text.")
                continue
            line_id = f"{sid}_L{len(dialogue) + 1:03d}"
            line_text = " ".join(text_lines)
            parenthetical = "; ".join(p for p in block["parentheticals"] if p) or None
            if unknown:
                parenthetical = "; ".join([*(parenthetical and [parenthetical] or []), *[u.lower() for u in unknown]])
            translit = " ".join(block["translit_lines"]) or None
            proj_lang = (project.config.language or "hi").strip().lower()
            has_devanagari = bool(_DEVANAGARI_RE.search(line_text))
            if proj_lang in ("hi", "mr", "hindi", "marathi") and not has_devanagari:
                ambiguities.append(
                    f"{line_id}: the line has no Devanagari text — the exact dialogue must be written in "
                    "its native script (Devanagari, with transliteration alongside) so Omni renders it clearly.")
            words = count_words(line_text)
            if words > MAX_WORDS_PER_BREATH:
                ambiguities.append(
                    f"{line_id}: {words} words — longer than one breath at human pace (8–14 words per 8 s). It is "
                    "kept verbatim (a line is never split inside a clip); consider breaking it into two lines.")
            dialogue.append(SceneDialogueLine(
                line_id=line_id, character_id=char_id, text=line_text, parenthetical=parenthetical,
                transliteration=translit, language="hi" if has_devanagari else proj_lang, off_screen=off_screen))
            if not off_screen and char_id not in on_screen_chars:
                on_screen_chars.append(char_id)

        env_id = None
        location = heading.get("location") or ""
        if location:
            proposed = f"ENV_{ascii_id_part(location)}"[:96].rstrip("_")
            if proposed != "ENV_" and re.fullmatch(r"[A-Z][A-Z0-9_]*", proposed):
                if proposed in environments:
                    env_id = proposed
                else:
                    result.new_environments.setdefault(proposed, {
                        "name": location, "setting": heading.get("setting"), "first_scene": sid})
                result.scene_environment_hints[sid] = proposed
        states_active: list[str] = []
        for cid in on_screen_chars:
            ent = entities.get(cid)
            if ent and ent.states:
                default = next((s.id for s in ent.states if s.id.endswith("_STATE_DEFAULT")), ent.states[0].id)
                states_active.append(default)
        slug_src = "_".join(p for p in (location, heading.get("time_of_day") or "") if p)
        scene = Scene(
            id=sid,
            slug=_slugify(slug_src, f"scene_{last_number + idx}"),
            order=last_order + idx,
            episode=episode,
            script_text="\n".join(rs["lines"]).strip(),
            page_range=None,
            entity_states_active=states_active,
            environment_id=env_id,
            time_of_day=heading.get("time_of_day"),
            dialogue=dialogue,
            status=SceneStatus.PENDING,
            generated_by=prov,
        )
        project.save_scene(scene)
        result.scenes.append(scene)

    seen_new: list[str] = []
    for cid in result.characters:
        if cid not in entities and cid not in seen_new:
            seen_new.append(cid)
    result.new_characters = seen_new
    result.ambiguities = list(dict.fromkeys(ambiguities))
    try:
        cfg = project.config
        if cfg.script_version != version:
            project.save_config(cfg.model_copy(update={"script_version": version}))
    except Exception:  # noqa: BLE001 — the script version is informative; never fail the ingest on it
        pass
    return result


# ---------------------------------------------------------------------------------------------
# compute_breakdown (§4.5, deterministic)
# ---------------------------------------------------------------------------------------------
def _entity_of_state(state_id: str, entity_ids: Iterable[str]) -> str | None:
    owner = entity_id_of_state(state_id, entity_ids)
    if owner:
        return owner
    if "_STATE_" in state_id:
        return state_id.split("_STATE_", 1)[0]
    return None


def scene_entities(scene: Scene, entity_ids: Iterable[str] = ()) -> tuple[list[str], list[str]]:
    """(visual entities on screen, speakers) of a scene. Off-screen speakers are speakers, not visuals."""
    ids = list(entity_ids)
    visual: list[str] = []
    for st in all_scene_states(scene):
        owner = _entity_of_state(st, ids)
        if owner and owner not in visual:
            visual.append(owner)
    for tr in scene.mid_scene_transitions:
        if tr.entity_id not in visual:
            visual.append(tr.entity_id)
    speakers: list[str] = []
    for ln in scene.dialogue:
        if ln.character_id not in speakers:
            speakers.append(ln.character_id)
        if not ln.off_screen and ln.character_id not in visual:
            visual.append(ln.character_id)
    return visual, speakers


def _scene_clip_info(project: Project, scene: Scene) -> tuple[list[str], float, int]:
    """(shot actions, total creative shot duration, number of live clips) from the shot tree, if any."""
    actions: list[str] = []
    total = 0.0
    clips = 0
    try:
        shots = project.list_shots(scene.id)
    except Exception:  # noqa: BLE001
        shots = []
    for shot in shots:
        actions.extend(a for a in shot.actions if a.strip())
        total += float(shot.duration_s or 0.0)
        try:
            clips += sum(1 for c in project.list_clips(shot.id) if c.status.value != "stale")
        except Exception:  # noqa: BLE001
            pass
    return actions, total, clips


def _camera_complexity(text: str, n_chars: int, n_shots_actions: int) -> RiskTier:
    hits = {m.group(1).lower() for m in _CAMERA_RE.finditer(text)}
    if len(hits) >= 2 or (n_chars >= 5 and hits):
        return RiskTier.HIGH
    if hits or n_chars >= 4 or n_shots_actions >= 12:
        return RiskTier.MED
    return RiskTier.LOW


def _recommend_model(registry: Any, scene: Scene, *, refs_needed: int, longest_line_clip_s: float,
                     artefact_prone: bool, open_decisions: OpenDecisions) -> tuple[str, list[str]]:
    video = {m.id: m for m in registry.video_models()}
    if not video:
        return "omni", ["no enabled video model in the registry; defaulting to omni"]
    if scene.model_override and scene.model_override in video:
        return scene.model_override, [f"the human chose {scene.model_override} for this scene (override)"]
    if "omni" not in video or "veo" not in video:
        only = "omni" if "omni" in video else sorted(video)[0]
        return only, [f"{only} is the only enabled video model"]
    omni, veo = video["omni"], video["veo"]
    o_score, v_score, why = 0, 0, []
    veo_refs = int(veo.supports.max_reference_images or 0)
    if refs_needed > veo_refs:
        o_score += 2
        why.append(f"needs about {refs_needed} reference images; Veo accepts {veo_refs}, Omni "
                   f"{int(omni.supports.max_reference_images or 0)}")
    if veo.max_duration_s and longest_line_clip_s > float(veo.max_duration_s):
        o_score += 2
        why.append(f"the longest line needs ~{longest_line_clip_s:.1f} s in one clip; Veo's max is "
                   f"{float(veo.max_duration_s or 0):g} s, Omni's {float(omni.max_duration_s or 0):g} s")
    if artefact_prone and veo.supports.negative_prompt and not omni.supports.negative_prompt:
        v_score += 1
        why.append("crowd/contact action benefits from Veo's separate negative-prompt field")
    if omni.cost_per_s < veo.cost_per_s:
        o_score += 1
        why.append(f"Omni is cheaper per second (${omni.cost_per_s:.2f} vs ${veo.cost_per_s:.2f})")
    elif veo.cost_per_s < omni.cost_per_s:
        v_score += 1
        why.append(f"Veo is cheaper per second (${veo.cost_per_s:.2f} vs ${omni.cost_per_s:.2f})")
    if o_score == v_score:
        pick = open_decisions.default_model_tossup if open_decisions.default_model_tossup in video else "omni"
        why.append(f"toss-up between the models — using {pick} (Open Decision 5: decided from pilot results)")
        return pick, why
    return ("omni" if o_score > v_score else "veo"), why


def compute_breakdown(project: Project, registry: Any, scene: Scene) -> SceneBreakdown:
    """Deterministic §4.5 SceneBreakdown for ``scene`` (no LLM, no clock, no randomness)."""
    settings = project.config.settings
    entities = {e.id: e for e in project.list_entities()}
    visual, speakers = scene_entities(scene, entities)
    earlier = [s for s in project.list_scenes() if s.id != scene.id and (s.order, s.id) < (scene.order, scene.id)]
    seen_visual: set[str] = set()
    seen_speakers: set[str] = set()
    seen_states: set[str] = set()
    for prev in earlier:
        pv, ps = scene_entities(prev, entities)
        seen_visual.update(pv)
        seen_speakers.update(ps)
        seen_states.update(all_scene_states(prev))
    new_entities = [e for e in visual if e not in seen_visual]
    new_speakers = [c for c in speakers if c not in seen_speakers]
    scene_states = all_scene_states(scene)
    new_states = [s for s in scene_states if s not in seen_states]

    shot_actions, shot_total_s, live_clips = _scene_clip_info(project, scene)
    script_actions = scene_action_lines(scene)
    actions = shot_actions or script_actions
    action_text = " ".join([*script_actions, *shot_actions])
    n_actions = len(actions)
    n_lines = len(scene.dialogue)
    words = sum(count_words(ln.text) for ln in scene.dialogue)
    cushions = settings.speech_pre_roll_s + settings.speech_post_roll_s
    speech_s = words / settings.speech_pace_wps + n_lines * cushions
    estimated = shot_total_s if shot_total_s > 0 else speech_s + n_actions * settings.min_action_s
    estimated = round(max(estimated, settings.min_clip_s), 1)

    characters_on_screen = [e for e in visual if (entities.get(e) is None and e.startswith("CHAR_"))
                            or (entities.get(e) is not None and entities[e].type.value in ("character", "creature"))]
    n_chars = len(characters_on_screen)
    physical = bool(_PHYSICAL_RE.search(action_text))
    crowd = bool(_CROWD_RE.search(action_text)) or n_chars >= 5
    camera = _camera_complexity(action_text, n_chars, n_actions)
    diegetic = bool(_DIEGETIC_RE.search(action_text))
    estimated_clips = live_clips or max(1, n_actions + n_lines)
    dialogue_density = round(words / estimated, 3) if estimated else 0.0
    action_density = round(n_actions / estimated * 10.0, 3) if estimated else 0.0

    longest_line = max((count_words(ln.text) / settings.speech_pace_wps + cushions for ln in scene.dialogue),
                       default=0.0)
    refs_needed = len(scene_states) + (1 if scene.environment_id else 0)
    model_id, model_why = _recommend_model(
        registry, scene, refs_needed=refs_needed, longest_line_clip_s=longest_line,
        artefact_prone=crowd or physical, open_decisions=settings.open_decisions)
    try:
        spec = registry.get(model_id)
        clip_s = spec.snap_duration(estimated / estimated_clips)
        cost = spec.estimate_cost(duration_s=clip_s, calls=estimated_clips * settings.takes_per_clip)
    except Exception:  # noqa: BLE001 — unknown model in a hand-edited registry: cost unknown, never crash
        clip_s, cost = estimated / estimated_clips, 0.0

    points = 0
    reasons: list[str] = []
    if n_chars >= 5:
        points += 2
        reasons.append(f"{n_chars} characters on screen")
    elif n_chars >= 3:
        points += 1
        reasons.append(f"{n_chars} characters on screen")
    if physical:
        points += 1
        reasons.append("physical contact between characters")
    if crowd:
        points += 2
        reasons.append("crowd")
    if scene.mid_scene_transitions:
        points += 1
        reasons.append(f"{len(scene.mid_scene_transitions)} costume/state change(s) mid-scene")
    if camera == RiskTier.MED:
        points += 1
        reasons.append("moving camera")
    elif camera == RiskTier.HIGH:
        points += 2
        reasons.append("complex camera work")
    if dialogue_density >= 0.7 * settings.speech_pace_wps:
        points += 1
        reasons.append("dialogue-heavy")
    if len(new_speakers) >= 2:
        points += 1
        reasons.append(f"{len(new_speakers)} new voices to cast")
    risk = RiskTier.LOW if points <= 1 else (RiskTier.MED if points <= 3 else RiskTier.HIGH)
    audio_mode = AudioMode.NATIVE_AV if (n_lines or diegetic) else AudioMode.VIDEO_ONLY

    rationale = (
        f"~{estimated:.0f} s of screen time: {n_lines} line(s) / {words} words at "
        f"{settings.speech_pace_wps:g} words/s with {cushions:.1f} s breath cushions, {n_actions} physical "
        f"action(s). About {estimated_clips} clip(s) (one action or one line each) of ~{clip_s:g} s on "
        f"{model_id}, {settings.takes_per_clip} takes each ≈ ${cost:.2f}. "
        f"Risk {risk.value}: {', '.join(reasons) if reasons else 'simple staging'}. "
        f"Model {model_id}: {'; '.join(model_why)}. "
        f"Audio {audio_mode.value}: "
        + ("native dialogue and sound generated with the picture." if audio_mode == AudioMode.NATIVE_AV
           else "no dialogue or meaningful on-screen sound — picture only.")
        + (f" New in this scene: {', '.join(new_entities)}." if new_entities else "")
        + (f" New voices: {', '.join(new_speakers)}." if new_speakers else "")
        + (f" New looks/states: {', '.join(new_states)}." if new_states else "")
    )
    return SceneBreakdown(
        estimated_duration_s=estimated,
        n_entity_states=len(scene_states),
        n_simultaneous_characters=n_chars,
        physical_interaction=physical,
        dialogue_density=dialogue_density,
        action_density=action_density,
        n_state_transitions=len(scene.mid_scene_transitions),
        has_crowd=crowd,
        camera_complexity=camera,
        new_entities_introduced=new_entities,
        new_speaking_characters=new_speakers,
        estimated_clips=estimated_clips,
        estimated_cost=round(float(cost), 2),
        risk_tier=risk,
        recommended_model=model_id,
        audio_mode=audio_mode,
        rationale=rationale,
    )


def breakdown_all(project: Project, registry: Any, *, save: bool = True) -> dict[str, SceneBreakdown]:
    """Compute (and optionally store on the scene records) the breakdown of every scene."""
    out: dict[str, SceneBreakdown] = {}
    for scene in project.list_scenes():
        bd = compute_breakdown(project, registry, scene)
        out[scene.id] = bd
        if save and scene.breakdown != bd:
            scene.breakdown = bd
            project.save_scene(scene)
    return out


# ---------------------------------------------------------------------------------------------
# build_schedule (§11)
# ---------------------------------------------------------------------------------------------
_RISK_PENALTY = {RiskTier.LOW: 0, RiskTier.MED: 0, RiskTier.HIGH: 20}


def build_schedule(project: Project, scenes: Iterable[Scene | str] | None = None, *, registry: Any = None,
                   save: bool = False) -> tuple[list[str], str | None]:
    """``[pilot] + scenes by first introduction of reused entities/speakers``, then the parallel rest.

    The pilot is the scene whose entities and voices are reused by the most other scenes, excluding high-risk
    scenes when possible (so the packs and voice anchors it locks pay off soonest). Pure unless ``save=True``
    (which records the schedule in state.json — normally done when the human passes G1).
    """
    loaded: list[Scene] = []
    for sc in (project.list_scenes() if scenes is None else scenes):
        loaded.append(project.load_scene(sc) if isinstance(sc, str) else sc)
    if not loaded:
        return [], None
    loaded.sort(key=lambda s: (s.order, s.id))
    entities = {e.id: e for e in project.list_entities()}
    reg = registry
    risk: dict[str, RiskTier] = {}
    members: dict[str, set[str]] = {}
    speakers_of: dict[str, set[str]] = {}
    for sc in loaded:
        if sc.breakdown is None:
            if reg is None:
                from .registry import default_registry

                reg = default_registry()
            risk[sc.id] = compute_breakdown(project, reg, sc).risk_tier
        else:
            risk[sc.id] = sc.breakdown.risk_tier
        visual, speakers = scene_entities(sc, entities)
        members[sc.id] = set(visual) | {f"VOICE:{c}" for c in speakers}
        speakers_of[sc.id] = set(speakers)
    appearances: dict[str, int] = {}
    for ids in members.values():
        for item in ids:
            appearances[item] = appearances.get(item, 0) + 1
    reused = {k for k, v in appearances.items() if v >= 2}
    order_index = {sc.id: i for i, sc in enumerate(loaded)}

    def reuse_value(sid: str, pool: set[str]) -> int:
        return sum(appearances[item] - 1 for item in members[sid] & pool)

    def pilot_key(sc: Scene) -> tuple:
        value = reuse_value(sc.id, reused) + 2 * len({f"VOICE:{c}" for c in speakers_of[sc.id]} & reused)
        return (-(value - _RISK_PENALTY[risk[sc.id]]), 0 if risk[sc.id] == RiskTier.MED else 1, order_index[sc.id])

    pilot = min(loaded, key=pilot_key).id
    schedule = [pilot]
    introduced = set(members[pilot])
    remaining = [sc.id for sc in loaded if sc.id != pilot]
    while remaining:
        best = None
        best_key = None
        for sid in remaining:
            fresh = (members[sid] & reused) - introduced
            key = (-reuse_value(sid, fresh), order_index[sid])
            if fresh and (best_key is None or key < best_key):
                best, best_key = sid, key
        if best is None:  # everything left only reuses introduced assets -> parallel, story order
            schedule.extend(remaining)
            break
        schedule.append(best)
        introduced |= members[best]
        remaining.remove(best)
    if save:
        from .state import StateStore

        StateStore(project).set_schedule(schedule, pilot)
    return schedule, pilot


# ---------------------------------------------------------------------------------------------
# G1 packet (§11) incl. §15 Open Decisions in plain filmmaking language
# ---------------------------------------------------------------------------------------------
OPEN_DECISION_TEXT: tuple[dict[str, Any], ...] = (
    {"number": 1, "id": "identity_conditioning",
     "question": "How do we keep every actor's face and body identical from shot to shot?",
     "options": [("native_refs", "Show the video model the approved character reference sheets in every shot — "
                                 "no training, works today"),
                 ("lora", "Train a custom face model for each principal character first — slower and costs extra")],
     "recommended": "native_refs"},
    {"number": 2, "id": "incidental_conditioning",
     "question": "Small one-off characters (e.g. a delivery rider): do they get a reference picture?",
     "options": [("single_still_if_line_or_gt2s", "Give a single reference still to anyone who speaks or is on "
                                                  "screen for more than 2 seconds; others are described in words"),
                 ("text_only", "Describe all small characters in words only (cheaper, less consistent)")],
     "recommended": "single_still_if_line_or_gt2s"},
    {"number": 3, "id": "stem_separation",
     "question": "Should we split each generated clip's sound into separate dialogue and effects tracks?",
     "options": [(False, "No — keep the dialogue and natural sound exactly as generated with the picture"),
                 (True, "Yes — run stem separation on every clip (can add audible artefacts)")],
     "recommended": False},
    {"number": 4, "id": "voice_reference_audio",
     "question": "Once a character's voice is approved, should we play that voice sample to the video model "
                 "(where it supports it) so the voice stays identical?",
     "options": [(True, "Yes — send the approved voice sample wherever the model accepts it"),
                 (False, "No — rely on the written voice description only")],
     "recommended": True},
    {"number": 5, "id": "default_model_tossup",
     "question": "When both video models suit a scene equally well, which one should we use?",
     "options": [(None, "Decide after the pilot scene, based on which model kept faces and voices more stable"),
                 ("omni", "Always Gemini Omni"), ("veo", "Always Veo 3.1")],
     "recommended": None},
    {"number": 6, "id": "roundtrip_transport",
     "question": "How should AGY notice the changes you make in DaVinci Resolve?",
     "options": [("db_polling", "AGY quietly reads Resolve's project database every minute — no clicks or "
                                "exports needed from you"),
                 ("otio_export", "You export an OTIO/FCPXML file whenever you want AGY to look")],
     "recommended": "db_polling"},
    {"number": 7, "id": "orange_marker_default",
     "question": "When a regenerated shot is ready beside a timeline you own, may AGY drop an orange "
                 "'AGY_REGEN_READY' marker on your timeline?",
     "options": [(True, "Yes — add the orange marker (you can switch it off for any scene)"),
                 (False, "No — just put the new take in the /REGEN/ bin and tell me in chat")],
     "recommended": True},
)


def open_decision_items(project: Project) -> list[dict[str, Any]]:
    current = project.config.settings.open_decisions
    items = []
    for spec in OPEN_DECISION_TEXT:
        value = getattr(current, spec["id"])
        items.append({
            "kind": "open_decision",
            "id": spec["id"],
            "number": spec["number"],
            "title": f"Open decision {spec['number']}",
            "question": spec["question"],
            "options": [{"value": v, "label": label, "recommended": v == spec["recommended"]}
                        for v, label in spec["options"]],
            "recommended": spec["recommended"],
            "recommended_label": next(label for v, label in spec["options"] if v == spec["recommended"]),
            "current": value,
            "confirmed_by_human": current.confirmed_by_human,
        })
    return items


def _state_timeline(project: Project) -> list[dict[str, Any]]:
    path = project.path("bible", "state_timeline.yaml")
    if not path.exists():
        return []
    data = read_record(path) or {}
    return list(data.get("entries", []))


def g1_packet_items(project: Project) -> list[dict[str, Any]]:
    """Everything the human reviews at G1, in presentation order (plain language; nothing is locked here)."""
    items: list[dict[str, Any]] = []
    style = project.load_style_lock()
    if style is not None:
        items.append({
            "kind": "style_lock", "id": "STYLE_LOCK", "title": "Look of the film (Style Lock)",
            "artifact": "bible/style_lock.yaml", "approval_status": style.approval_status.value,
            "summary": f"{style.medium} · {style.aspect_ratio} · {style.lens_language}",
            "global_negative": list(style.global_negative), "donts": list(style.donts),
        })
    else:
        items.append({"kind": "style_lock", "id": "STYLE_LOCK", "title": "Look of the film (Style Lock)",
                      "approval_status": "missing", "summary": "No Style Lock proposed yet."})
    for ent in project.list_entities():
        items.append({
            "kind": "entity", "id": ent.id, "title": ent.display_name or ent.id, "entity_type": ent.type.value,
            "tier": ent.tier.value, "approval_status": ent.approval_status.value,
            "artifact": f"bible/entities/{ent.id}.yaml", "summary": ent.canonical_description,
            "states": [{"id": st.id, "look": st.delta_description, "scenes": list(st.scenes_active)}
                       for st in ent.states],
            "voice_profile_id": ent.voice_profile_id,
        })
    for env in project.list_environments():
        items.append({
            "kind": "environment", "id": env.id, "title": env.id, "approval_status": env.approval_status.value,
            "artifact": f"bible/environments/{env.id}.yaml", "summary": env.canonical_description,
            "lighting": [ls.description for ls in env.lighting_setups], "room_tone": env.room_tone,
        })
    for char_id in sorted((project.casting().get("voice_profiles") or {})):
        profile = project.load_voice_profile(char_id)
        if profile is None:
            continue
        items.append({
            "kind": "voice_profile", "id": profile.id, "character_id": char_id,
            "title": f"Voice of {char_id}", "approval_status": profile.approval_status.value,
            "summary": profile.description, "language": profile.language,
            "state_variants": dict(profile.state_variants),
        })
    scenes = project.list_scenes()
    items.append({
        "kind": "scenes", "id": "SCENES", "title": f"Scene list ({len(scenes)} scenes)",
        "scenes": [{"id": sc.id, "slug": sc.slug, "episode": sc.episode, "lines": len(sc.dialogue),
                    "environment_id": sc.environment_id, "status": sc.status.value,
                    "risk": sc.breakdown.risk_tier.value if sc.breakdown else None,
                    "model": sc.breakdown.recommended_model if sc.breakdown else None,
                    "estimated_cost": sc.breakdown.estimated_cost if sc.breakdown else None}
                   for sc in scenes],
    })
    items.append({"kind": "state_timeline", "id": "STATE_TIMELINE", "title": "Who looks how, in which scene",
                  "artifact": "bible/state_timeline.yaml", "entries": _state_timeline(project)})
    from .state import StateStore

    state = StateStore(project).load()
    schedule, pilot = list(state.schedule), state.pilot_scene_id
    proposed = False
    if not schedule and scenes:
        schedule, pilot = build_schedule(project, scenes)
        proposed = True
    items.append({
        "kind": "schedule", "id": "SCHEDULE", "title": "Production order",
        "pilot": pilot, "schedule": schedule, "proposed": proposed,
        "summary": (f"Pilot {pilot} first, then {', '.join(schedule[1:])}" if pilot and len(schedule) > 1
                    else (f"Pilot {pilot}" if pilot else "No scenes yet")),
    })
    items.extend(open_decision_items(project))
    from .fsio import read_jsonl

    open_q = [q for q in read_jsonl(project.questions_path) if q.get("status", "open") == "open"]
    if open_q:
        items.append({"kind": "open_questions", "id": "QUESTIONS", "title": "Questions waiting for you",
                      "questions": [{"id": q.get("id"), "text": q.get("text"), "scene_id": q.get("scene_id")}
                                    for q in open_q]})
    return items
