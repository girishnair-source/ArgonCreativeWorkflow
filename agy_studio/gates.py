"""Human gates G1–G7 — BUILD_SPEC_v3 §9 (Workstream A).

* :func:`render_gate` implements the §9.3 chat presentation contract: header, review assets (absolute paths +
  Resolve bin), per-item summary, expected reply format, defaults (silence behaviour) and voice-anchor prompts.
* :func:`parse_gate_response` turns the director's reply (``C01:T2, C03:regen "slower delivery"``, ``approve all``,
  ``SH02:alternatives 3 "differ in lens"``, ``anchor CHAR_RISHI=SC01_SH01_C01_T02`` …) into a :class:`GateResponse`
  with a one-line interpretation. Short ids resolve against the packet's item ids by suffix.
* :class:`GateManager` persists packets, records one :class:`Decision` per choice and sets the gate status in
  ``state.json``. It never advances scene stages. Vision and Craft gates never auto-proceed: :meth:`auto_pass` is
  only allowed for G2 (nothing new) and G5 (low-risk auto-select), and refuses when its guards fail.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import Field, field_validator

from .continuity import all_scene_states, speakers_in_clip
from .decisions import DecisionLog
from .fsio import (
    PathSecurityError,
    append_jsonl,
    compact_ts,
    generated_by,
    now_iso,
    read_jsonl,
    read_yaml,
    safe_join,
    save_versioned,
    validate_id,
)
from .invalidation import load_scene_clips
from .models import ApprovalStatus, ClipStatus, Decision, DecisionAction, DecisionTier, GateId, Model, RiskTier
from .project import Project, parse_artifact_id
from .state import GLOBAL_GATES, StateStore

GATE_TIERS: dict[GateId, DecisionTier] = {
    GateId.G1: DecisionTier.VISION,
    GateId.G2: DecisionTier.VISION,
    GateId.G3: DecisionTier.CRAFT,
    GateId.G4: DecisionTier.CRAFT,
    GateId.G5: DecisionTier.CRAFT,
    GateId.G6: DecisionTier.CRAFT,
    GateId.G7: DecisionTier.VISION,
}
AUTO_PASS_GATES = frozenset({GateId.G2, GateId.G5})
GATE_TITLES: dict[GateId, str] = {
    GateId.G1: "Bible, style lock & voices",
    GateId.G2: "Packs & geometry",
    GateId.G3: "Storyboards",
    GateId.G4: "Audio plan",
    GateId.G5: "Take selection",
    GateId.G6: "Scene cut approval",
    GateId.G7: "Picture lock",
}
GATE_NOUNS: dict[GateId, str] = {
    GateId.G1: "item", GateId.G2: "asset", GateId.G3: "shot", GateId.G4: "clip",
    GateId.G5: "clip", GateId.G6: "timeline", GateId.G7: "scene",
}
NEVER_AUTO = "Vision and Craft gates never auto-proceed — nothing moves until you reply."
_DEFAULTS: dict[GateId, str] = {
    GateId.G1: "`approve all` locks the bible, style lock and voice profiles exactly as shown.",
    GateId.G2: "`approve all` locks every proposed pack and geometry still into the library.",
    GateId.G3: "`approve all` accepts every board as shown.",
    GateId.G4: "`approve all` accepts every delivery, ambience and cue as planned.",
    GateId.G5: "`approve all` takes the recommended take for every clip.",
    GateId.G6: "`approve all` (or the APPROVED marker in Resolve) marks your cut of this scene as approved.",
    GateId.G7: "`approve all` locks picture for the whole film.",
}
_TEMPLATES: dict[GateId, tuple[str, ...]] = {
    GateId.G5: ("{id}:T2", "{id}:T1", '{id}:regen "slower delivery, less head movement"', "{id}:ok"),
    GateId.G3: ("{id}:ok", '{id}:alternatives 3 "differ in lens"', '{id}:regen "tighter framing"'),
    GateId.G4: ("{id}:ok", '{id}:regen "softer delivery, more breath between words"'),
}
_DEFAULT_TEMPLATES = ("{id}:ok", '{id}:reject "what to change"')
LEGEND = ("How to read this: QA = overall quality score 0–1 (higher is better) · transcript ✓ = the spoken words "
          "match the script · speaker-sim = how close the voice is to the character's anchor voice (— = no anchor "
          "yet) · anchor fan-out = later clips that start from this take's last frame.")
_MEDIA_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".mov", ".m4v", ".wav", ".mp3", ".m4a"})


# ---------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------
def _single_line_id(value: str) -> str:
    value = str(value or "").strip()
    if not value or len(value) > 160 or any(ch.isspace() for ch in value):
        raise ValueError(f"item id {value!r} must be a single token of 1-160 characters")
    return value


class GateItem(Model):
    item_id: str
    recommendation: str | None = None
    qa_composite: float | None = None
    transcript_ok: bool | None = None
    speaker_sim: float | dict[str, float | None] | None = None
    flags: list[str] = Field(default_factory=list)
    anchor_fanout: int | None = None
    options: list[str] = Field(default_factory=list)
    summary: str = ""
    media: list[str] = Field(default_factory=list)  # project-relative or absolute; rendered absolute

    @field_validator("item_id")
    @classmethod
    def _v_item(cls, v: str) -> str:
        return _single_line_id(v)


class GatePacket(Model):
    gate: GateId
    scene_id: str | None = None
    title: str = ""  # empty -> GATE_TITLES[gate]
    risk: str | None = None
    model: str | None = None
    blocks: list[str] = Field(default_factory=list)
    review_dir: str | None = None
    resolve_bin: str | None = None
    items: list[GateItem] = Field(default_factory=list)
    expected_format: str = ""
    defaults: str = ""
    voice_anchor_prompts: list[str] = Field(default_factory=list)  # character ids (or ready-made sentences)
    notes: list[str] = Field(default_factory=list)

    @field_validator("scene_id")
    @classmethod
    def _v_scene(cls, v: str | None) -> str | None:
        return validate_id(v, "scene id") if v is not None else None


class ItemChoice(Model):
    item_id: str
    action: Literal["choose", "ok", "regen", "reject", "alternatives"]
    value: str | None = None
    steer: str | None = None


class GateResponse(Model):
    gate: GateId
    approve_all: bool = False
    reject_all: bool = False
    choices: list[ItemChoice] = Field(default_factory=list)
    anchors: dict[str, str] = Field(default_factory=dict)  # character_id -> take_id
    free_text: str = ""
    interpretation: str = ""
    unresolved: list[str] = Field(default_factory=list)  # item references that matched no packet item


# ---------------------------------------------------------------------------------------------
# Rendering (§9.3)
# ---------------------------------------------------------------------------------------------
def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _flat(text: str) -> str:
    return " ".join(str(text or "").split())


def item_noun(packet: GatePacket) -> str:
    """'take' / 'clip' / 'shot' / 'scene' when every item id is that kind of artifact, else the gate default."""
    levels: set[str | None] = set()
    for it in packet.items:
        try:
            ids = parse_artifact_id(it.item_id)
        except ValueError:
            levels.add(None)
            continue
        levels.add("take" if ids["take"] else "clip" if ids["clip"] else "shot" if ids["shot"] else "scene")
    if len(levels) == 1 and None not in levels:
        return str(next(iter(levels)))
    return GATE_NOUNS[packet.gate]


def render_header(packet: GatePacket) -> str:
    parts = [f"GATE {packet.gate.value}"]
    if packet.scene_id:
        parts.append(packet.scene_id)
    parts.append(_flat(packet.title) or GATE_TITLES[packet.gate])
    if packet.items:
        parts.append(_plural(len(packet.items), item_noun(packet)))
    if packet.risk:
        parts.append(f"risk: {_flat(packet.risk)}")
    if packet.model:
        parts.append(f"model: {_flat(packet.model)}")
    if packet.blocks:
        parts.append("blocks: " + ", ".join(_flat(b) for b in packet.blocks))
    return "[" + " · ".join(parts) + "]"


def short_ids(item_ids: Iterable[str]) -> dict[str, str]:
    """Shortest unique ``_``-token suffix per id (``SC01_SH02_C01`` -> ``C01`` when unambiguous)."""
    ids = list(dict.fromkeys(item_ids))
    out: dict[str, str] = {}
    for iid in ids:
        toks = iid.split("_")
        out[iid] = iid
        for k in range(1, len(toks) + 1):
            cand = "_".join(toks[-k:])
            if sum(1 for o in ids if o == cand or o.endswith("_" + cand)) == 1:
                out[iid] = cand
                break
    return out


def _abs_path(p: str, root: Path | None) -> str:
    path = Path(p)
    if path.is_absolute():
        return str(path.resolve())
    if root is None:
        return p
    try:
        return str(safe_join(root, p))
    except PathSecurityError:
        return p


def _sim_text(sim: float | dict[str, float | None]) -> str | None:
    if isinstance(sim, dict):
        parts = [f"{c} {v:.2f}" if v is not None else f"{c} —" for c, v in sim.items()]
        return ("speaker-sim " + ", ".join(parts)) if parts else None
    return f"speaker-sim {sim:.2f}"


def render_item_line(item: GateItem, gate: GateId) -> str:
    bits: list[str] = []
    if item.recommendation:
        bits.append(f"recommend {item.recommendation}")
    if item.qa_composite is not None:
        bits.append(f"QA {item.qa_composite:.2f}")
    if item.transcript_ok is not None:
        bits.append("transcript " + ("✓" if item.transcript_ok else "✗"))
    if item.speaker_sim is not None:
        sim = _sim_text(item.speaker_sim)
        if sim:
            bits.append(sim)
    if item.flags:
        bits.append("flags: " + ", ".join(_flat(f) for f in item.flags))
    elif gate == GateId.G5:
        bits.append("flags: none")
    if item.anchor_fanout is not None:
        bits.append(f"anchor fan-out: {_plural(item.anchor_fanout, 'clip')}")
    return item.item_id + (" — " + " · ".join(bits) if bits else "")


def _review_lines(packet: GatePacket, root: Path | None, listed: set[str]) -> list[str]:
    lines: list[str] = []
    folder: Path | None = None
    if packet.review_dir:
        shown = _abs_path(packet.review_dir, root)
        lines.append(f"Review folder: {shown.rstrip('/')}/")
        candidate = Path(shown)
        inside = root is None or str(candidate).startswith(str(root) + "/")
        if candidate.is_absolute() and inside and candidate.is_dir():
            folder = candidate
    if packet.resolve_bin:
        lines.append(f"Resolve bin: {_flat(packet.resolve_bin)}")
    if folder is not None:
        for p in sorted(folder.iterdir()):
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in _MEDIA_EXTS and str(p) not in listed:
                lines.append(f"  {p}")
    return lines


def _reply_lines(packet: GatePacket) -> list[str]:
    if packet.expected_format.strip():
        lines = [f"  {ln}" for ln in packet.expected_format.strip().splitlines()]
    elif packet.items:
        shorts = short_ids(it.item_id for it in packet.items)
        templates = _TEMPLATES.get(packet.gate, _DEFAULT_TEMPLATES)
        n = min(len(packet.items), len(templates))
        lines = ["  " + ", ".join(templates[i].format(id=shorts[packet.items[i].item_id]) for i in range(n))]
    else:
        lines = ['  approve all   — or —   reject all "what to change"']
    where = "this scene" if packet.scene_id else "the whole film"
    lines.append(f"  Add any free-text note after it — I keep it as a standing note for {where}.")
    if packet.items:
        lines.append("  Or simply: approve all · reject all")
    return lines


def render_gate(packet: GatePacket, *, project_root: Path | str | None = None) -> str:
    """Chat text for a gate (§9.3). Relative media paths are shown absolute when ``project_root`` is given."""
    root = Path(project_root).resolve() if project_root else None
    out = [render_header(packet)]
    item_blocks: list[str] = []
    listed: set[str] = set()
    for it in packet.items:
        item_blocks.append("• " + render_item_line(it, packet.gate))
        if it.summary.strip():
            item_blocks.append(f"    {_flat(it.summary)}")
        if it.options:
            item_blocks.append("    options: " + " · ".join(_flat(o) for o in it.options))
        for m in it.media:
            shown = _abs_path(m, root)
            listed.add(shown)
            item_blocks.append(f"    {shown}")
    review = _review_lines(packet, root, listed)
    if review:
        out += ["", *review]
    if item_blocks:
        noun = item_noun(packet)
        out += ["", (noun if len(packet.items) == 1 else noun + "s").capitalize() + ":", *item_blocks]
        if any(it.qa_composite is not None or it.transcript_ok is not None or it.speaker_sim is not None
               or it.anchor_fanout is not None for it in packet.items):
            out += ["", LEGEND]
    if packet.voice_anchor_prompts:
        out += ["", "Voice anchors (first time these characters speak):"]
        for p in packet.voice_anchor_prompts:
            p = _flat(p)
            if re.fullmatch(r"CHAR_[A-Z0-9_]+", p):
                out += [f"• Pick {p}'s voice anchor — all future takes will be scored against it.",
                        f"  Reply with: anchor {p}=<take id>"]
            else:
                out.append(f"• {p}")
    out += ["", "How to reply:", *_reply_lines(packet)]
    defaults = _flat(packet.defaults) or _DEFAULTS[packet.gate]
    out += ["", "Defaults:", f"  {defaults}"]
    if NEVER_AUTO not in defaults:
        out.append(f"  {NEVER_AUTO}")
    if packet.notes:
        out += ["", "Notes:", *[f"• {_flat(n)}" for n in packet.notes]]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------------------------
_QUOTE_TRANSLATION = str.maketrans({  # 1 char -> 1 char, so positions are preserved
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"', "\u00ab": '"', "\u00bb": '"',
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u00a0": " ", "\u202f": " ", "\u2007": " ", "\u2009": " ",
})
_TOKEN = r"[A-Za-z][A-Za-z0-9_]*"
_B = r"(?=$|[\s.!,;:)\"'])"
_LEAD = r"(?:(?:and|then|so|please|just|otherwise|else|yes|ok|okay)\s+)*"
_ANCHOR_RE = re.compile(
    rf"^\s*(?:set\s+)?(?:the\s+)?(?:voice\s+)?anchor(?:\s*:\s*|\s+)(?:for\s+)?(?P<char>{_TOKEN})"
    r"\s*(?:=|:|->|→|\s+to\s+|\s+is\s+|\s+)\s*(?P<take>[A-Za-z0-9_]+)(?P<rest>.*)$", re.I | re.S)
_REJECT_ALL_RE = re.compile(rf"^\s*{_LEAD}reject\s+(?:all|everything|them\s+all|all\s+of\s+them)\b", re.I)
_APPROVE_ALL_RE = re.compile(
    rf"^\s*{_LEAD}(?:(?:approve|accept|ok|okay|keep)\s+(?:all|everything|the\s+rest|rest|them\s+all|"
    r"as\s+(?:is|recommended))|all\s+(?:ok|okay|good|approved|fine|great))\b", re.I)
_PICK_FOR_RE = re.compile(
    rf"^\s*(?:choose|pick|select|use|take)\s+(?:t|take\s*#?\s*)0*(?P<n>\d{{1,3}})\s+(?:for|on|in)\s+"
    rf"(?P<item>{_TOKEN})(?P<rest>.*)$", re.I | re.S)
_CMD_FIRST_RE = re.compile(
    r"^\s*(?P<verb>regenerate|regen|re-do|redo|retry|reshoot|reject|drop|alternatives|alternative|alts|alt|"
    rf"variations|choose|pick|select|use|approve|accept|okay|ok|keep)\s+(?P<item>{_TOKEN})(?P<rest>.*)$", re.I | re.S)
_ITEM_FIRST_RE = re.compile(
    rf"^\s*(?P<item>{_TOKEN})(?:\s*(?:->|→|:|=|–|—|-)\s*|\s+|$)(?P<cmd>.*)$", re.S)
_CMD_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(rf"^(?:t|take\s*#?\s*)0*(?P<n>\d{{1,3}}){_B}", re.I), "choose"),
    (re.compile(r"^#?0*(?P<n>\d{1,2})\s*$"), "choose"),
    (re.compile(rf"^(?:regenerate|regen|re-do|redo|retry|reshoot|again){_B}", re.I), "regen"),
    (re.compile(rf"^(?:rejected|reject|drop|nope|no|kill|bin){_B}", re.I), "reject"),
    (re.compile(rf"^(?:alternatives|alternative|alts|alt|variations|variants){_B}\s*(?:x\s*)?(?P<n>\d{{1,2}})?",
                re.I), "alternatives"),
    (re.compile(rf"^(?:okay|okey|ok|approved|approve|accepted|accept|good|keep|fine|yes|lgtm|✓|✔|👍){_B}",
                re.I), "ok"),
    # "C02: choose T2", "OD3: choose on", "SH01: choose A1" — the value (take or option) is read by _emit
    (re.compile(rf"^(?:choose|pick|select|use){_B}", re.I), "choose"),
)
_VERB_ACTION = {
    "regenerate": "regen", "regen": "regen", "re-do": "regen", "redo": "regen", "retry": "regen",
    "reshoot": "regen", "reject": "reject", "drop": "reject", "alternatives": "alternatives",
    "alternative": "alternatives", "alts": "alternatives", "alt": "alternatives", "variations": "alternatives",
    "choose": "choose", "pick": "choose", "select": "choose", "use": "choose", "approve": "ok", "accept": "ok",
    "okay": "ok", "ok": "ok", "keep": "ok",
}
_CONNECTORS = frozenset({"with", "for", "to", "please", "that", "and", "differ in", "differing in", "varying"})
_FILLER_WORDS = frozenset({"ok", "okay", "thanks", "thank", "you", "done", "yes", "cool", "great", "ty", "👍", "nice",
                           "perfect", "good", "fine", "sure", "lgtm", "please"})
_LEADING_CONNECTOR_RE = re.compile(
    r"^\s*(?:and|but|except(?:\s+for)?|apart\s+from|other\s+than|also|plus|then)\s+(?=\S)", re.I)


def _quote_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, ch in enumerate(text):
        if ch == '"':
            if start is None:
                start = i
            else:
                spans.append((start, i + 1))
                start = None
    if start is not None:  # unbalanced quote: the rest is quoted
        spans.append((start, len(text)))
    return spans


def _mask(text: str) -> str:
    """Same-length copy with every quoted span blanked, so markers inside quotes are never matched."""
    chars = list(text)
    for a, b in _quote_spans(text):
        chars[a:b] = " " * (b - a)
    return "".join(chars)


def _split_segments(text: str) -> list[str]:
    out, buf, in_q = [], [], False
    for ch in text:
        if ch == '"':
            in_q = not in_q
        if ch in ",;\n" and not in_q:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf))
    return [s.strip() for s in out if s.strip()]


def _norm_token(token: str) -> str:
    parts = []
    for part in token.upper().split("_"):
        m = re.fullmatch(r"(SC|SH|C|T)(\d+)", part)
        parts.append(f"{m.group(1)}{m.group(2).zfill(2)}" if m else part)
    return "_".join(parts)


def _looks_like_id(token: str) -> bool:
    return bool(re.fullmatch(r"(?:SC|SH|C|T)\d{1,4}(?:_[A-Za-z0-9]+)*", token, re.I)
                or re.fullmatch(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+|[A-Z]+\d+[A-Z0-9]*", token))


def _suffix_match(tok: str, item_ids: list[str]) -> str | None:
    cands = {i for i in item_ids if i.upper() == tok or i.upper().endswith("_" + tok)}
    return next(iter(cands)) if len(cands) == 1 else None


def _is_clip_id(item_id: str) -> bool:
    try:
        ids = parse_artifact_id(item_id)
    except ValueError:
        return False
    return ids["clip"] == item_id


def _is_clip_ref(token: str) -> bool:
    """Short or full clip/take reference (``C03``, ``SH02_C01``, ``SC01_SH02_C01_T2``): clips choose takes."""
    return bool(re.fullmatch(r"(?:SC\d+_)?(?:SH\d+_)?C\d+(?:_T\d+)?|T\d+", _norm_token(token)))


def _take_value(item_id: str, n: int) -> str:
    """Fully-qualified take id for clip items (``SC01_SH02_C01_T02``), else ``T02``."""
    return f"{item_id}_T{n:02d}" if _is_clip_id(item_id) else f"T{n:02d}"


def _resolve(token: str, item_ids: list[str]) -> tuple[str | None, str | None]:
    """(item id, implied take id) for a short or full reference; (None, None) when it matches nothing."""
    tok = _norm_token(token)
    if not item_ids:
        if not _looks_like_id(token):
            return None, None
        try:
            ids = parse_artifact_id(tok)
        except ValueError:
            return tok, None
        return (ids["clip"], ids["take"]) if ids["take"] else (tok, None)
    hit = _suffix_match(tok, item_ids)
    if hit:
        return hit, None
    m = re.fullmatch(r"(?P<base>.+)_T(?P<n>\d{1,3})", tok)
    if m:
        base = _suffix_match(m.group("base"), item_ids)
        if base:
            return base, _take_value(base, int(m.group("n")))
    return None, None


def _resolve_take(value: str, item_ids: list[str]) -> str | None:
    tok = _norm_token(value)
    try:
        ids = parse_artifact_id(tok)
        if ids["take"]:
            return ids["take"]
    except ValueError:
        pass
    _, take = _resolve(value, item_ids)
    return take if take and parse_artifact_id_safe(take) else None


def parse_artifact_id_safe(value: str) -> bool:
    try:
        return bool(parse_artifact_id(value)["take"])
    except ValueError:
        return False


def _norm_char(raw: str) -> str | None:
    c = raw.upper()
    if c.startswith("VOICE_"):
        c = c[len("VOICE_"):]
    if not c.startswith("CHAR_"):
        c = "CHAR_" + c
    try:
        return validate_id(c, "character id")
    except ValueError:
        return None


def _extract_steer(tail: str) -> tuple[str | None, str]:
    """Steer text after a regen/reject/alternatives command: the first quoted span, else the rest of the sentence."""
    spans = _quote_spans(tail)
    if spans:
        a, b = spans[0]
        quoted = _flat(tail[a:b].strip('"'))
        pre = _flat(tail[:a].strip(" :-–—,"))
        steer = quoted if not pre or pre.lower() in _CONNECTORS else _flat(f"{pre} {quoted}")
        return steer or None, tail[b:]
    if re.match(r"^\s*[.!?]", tail):
        return None, tail
    body = tail.lstrip(" :-–—")
    m = re.search(r"[.!?](?:\s|$)", body)
    steer, rest = (body[:m.start()], body[m.end():]) if m else (body, "")
    return _flat(steer) or None, rest


class _Ctx:
    def __init__(self, item_ids: Iterable[str]) -> None:
        self.item_ids = list(dict.fromkeys(item_ids))
        self.choices: list[ItemChoice] = []
        self.anchors: dict[str, str] = {}
        self.approve_all = False
        self.reject_all = False
        self.unresolved: list[str] = []
        self.free: list[tuple[str, str]] = []
        self.sep = ""

    def add_free(self, text: str) -> None:
        text = _flat(text).lstrip(".,;:-–— ")
        if len(text) >= 2 and text[0] == text[-1] == '"':
            text = text[1:-1].strip()
        words = re.findall(r"[^\W_]+|👍", text.lower())
        if words and not all(w in _FILLER_WORDS for w in words):  # pleasantries alone are not a note
            self.free.append((self.sep, text))
            self.sep = " "

    def free_text(self) -> str:
        out = ""
        for sep, text in self.free:
            out = text if not out else out + (sep or " ") + text
        return out


def _continue(ctx: _Ctx, rest: str, depth: int) -> None:
    rest = rest.lstrip(" \t.!?,;:-–—")
    if rest.strip():
        _parse_segment(ctx, rest, depth + 1)


def _option_value(tail: str) -> tuple[str, str] | None:
    """(option, rest) after a choose verb on a NON-clip item ("OD3: choose on", "SH01: choose A1").

    The option must be the whole instruction — one word or one quoted phrase, optionally followed by filler or by
    the next item reference — so a longer sentence stays a note and is never mistaken for a choice."""
    spans = _quote_spans(tail)
    if spans and not tail[:spans[0][0]].strip(" \t:=-–—"):
        a, b = spans[0]
        value, rest = _flat(tail[a:b].strip('"')), tail[b:]
    else:
        m = re.match(r"^\s*(?::|=)?\s*'?([^\W_][\w\-]*)'?(?=$|[\s.!,;:)])", _mask(tail))
        if not m:
            return None
        value, rest = m.group(1), tail[m.end():]
    words = re.findall(r"[^\W_]+|👍", rest.lower())
    nxt = re.match(r"^[\s.!?,;:\-–—]*(?P<tok>[A-Za-z][A-Za-z0-9_]*)", rest)
    if not value or (words and not all(w in _FILLER_WORDS for w in words)
                     and not (nxt and _looks_like_id(nxt.group("tok")))):
        return None
    return value, rest


def _emit(ctx: _Ctx, token: str, action: str, value: Any, tail: str, depth: int) -> bool:
    item, implied = _resolve(token, ctx.item_ids)
    if item is None and not _looks_like_id(token):
        return False
    if action == "ok" and implied:
        action, value = "choose", implied
    if action == "choose" and value is None:
        masked_tail = _mask(tail)
        m = re.match(r"^\s*(?::|=)?\s*(?:t|take\s*#?\s*)0*(\d{1,3})(?=$|[\s.!,;:)])", masked_tail, re.I)
        clip_like = _is_clip_ref(item if item is not None else token)
        if not m and clip_like:  # "C02: pick 2" — a bare number only when it is the whole instruction
            m = re.match(r"^\s*(?::|=)?\s*#?0*(\d{1,3})\s*(?=$|[.!,;:)])", masked_tail)
        option = None if (m or implied or clip_like) else _option_value(tail)
        if m:
            value, tail = int(m.group(1)), tail[m.end():]
        elif implied:
            value = implied
        elif option:
            value, tail = option  # open decision / alternative label: the option text is the value
        else:
            return False
    if action == "alternatives" and value is None:
        m = re.match(r"^\s*(?::|=)?\s*(?:x\s*)?(\d{1,2})\b", _mask(tail))
        if m:
            value, tail = str(int(m.group(1))), tail[m.end():]
    steer = None
    if action in ("regen", "reject", "alternatives"):
        steer, tail = _extract_steer(tail)
    if item is None:
        ctx.unresolved.append(token)
    else:
        if action == "choose" and not isinstance(value, str):
            value = _take_value(item, int(value))
        ctx.choices.append(ItemChoice(item_id=item, action=action, value=value, steer=steer))  # type: ignore[arg-type]
    _continue(ctx, tail, depth)
    return True


def _parse_command(cmd: str) -> tuple[str, Any, str] | None:
    masked = _mask(cmd)
    for regex, action in _CMD_PATTERNS:
        m = regex.match(masked)
        if m:
            n = m.groupdict().get("n")
            value: Any = None
            if action == "choose":
                value = int(n) if n else None  # verb form ("choose …"): _emit reads the take or option
            elif action == "alternatives" and n:
                value = str(int(n))
            return action, value, cmd[m.end():]
    return None


def _parse_segment(ctx: _Ctx, seg: str, depth: int = 0) -> None:
    seg = seg.strip()
    if not seg:
        return
    if depth > 24:
        ctx.add_free(seg)
        return
    masked = _mask(seg)
    m = _LEADING_CONNECTOR_RE.match(masked)
    if m:  # "except C03:regen …", "but keep …": the connector itself carries no instruction
        _parse_segment(ctx, seg[m.end():], depth + 1)
        return
    m = _ANCHOR_RE.match(masked)
    if m:
        char, take = _norm_char(m.group("char")), _resolve_take(m.group("take"), ctx.item_ids)
        if char and take:
            ctx.anchors[char] = take
        else:
            ctx.unresolved.append(f"anchor {m.group('char')}={m.group('take')}")
        _continue(ctx, seg[m.start("rest"):], depth)
        return
    for regex, flag in ((_REJECT_ALL_RE, "reject_all"), (_APPROVE_ALL_RE, "approve_all")):
        m = regex.match(masked)
        if m:
            setattr(ctx, flag, True)
            _continue(ctx, seg[m.end():], depth)
            return
    m = _PICK_FOR_RE.match(masked)
    if m and _emit(ctx, m.group("item"), "choose", int(m.group("n")), seg[m.start("rest"):], depth):
        return
    m = _CMD_FIRST_RE.match(masked)
    if m and _emit(ctx, m.group("item"), _VERB_ACTION[m.group("verb").lower()], None, seg[m.start("rest"):], depth):
        return
    m = _ITEM_FIRST_RE.match(masked)
    if m:
        cmd = seg[m.start("cmd"):]
        parsed = _parse_command(cmd) if cmd.strip() else None
        if parsed and _emit(ctx, m.group("item"), parsed[0], parsed[1], parsed[2], depth):
            return
        if not cmd.strip() and _resolve(m.group("item"), ctx.item_ids)[1]:  # bare take reference -> choose it
            if _emit(ctx, m.group("item"), "choose", None, "", depth):
                return
    ctx.add_free(seg)


def interpret(response: GateResponse, item_ids: Iterable[str] = ()) -> str:
    """One plain-language line confirming how the reply was understood (§9.3.4)."""
    ids = list(item_ids)
    shorts = short_ids([*ids, *[c.item_id for c in response.choices if c.item_id not in ids]])
    parts: list[str] = []
    if response.reject_all:
        parts.append("reject everything — nothing moves forward")
    if response.approve_all:
        parts.append("approve everything as recommended" + (" except what's listed" if response.choices else ""))
    for c in response.choices:
        name = shorts.get(c.item_id, c.item_id)
        steer = f' — "{c.steer}"' if c.steer else ""
        if c.action == "choose":
            tm = re.search(r"(?:^|_)(T\d{2,3})$", str(c.value or ""))
            parts.append(f"{name}: take {tm.group(1)}" if tm else f"{name}: choose {c.value}")
        elif c.action == "ok":
            parts.append(f"{name}: keep the recommendation")
        elif c.action == "regen":
            parts.append(f"{name}: regenerate{steer}")
        elif c.action == "reject":
            parts.append(f"{name}: reject{steer}")
        else:
            parts.append(f"{name}: {c.value or 'a few'} alternatives" + (f' differing in "{c.steer}"' if c.steer else ""))
    for char, take in response.anchors.items():
        parts.append(f"voice anchor {char} = {take}")
    if response.free_text:
        parts.append(f'standing note: "{response.free_text}"')
    if response.unresolved:
        hint = f" (use one of: {', '.join(short_ids(ids).values())})" if ids else ""
        parts.append("couldn't match " + ", ".join(response.unresolved) + hint)
    if not parts:
        return "I couldn't find an instruction in that reply — nothing is recorded yet."
    return _flat("Understood: " + " · ".join(parts) + ".")


def parse_gate_response(gate: GateId | str, text: str, *, item_ids: Iterable[str] = ()) -> GateResponse:
    gate = GateId(gate)
    ctx = _Ctx(item_ids)
    for seg in _split_segments((text or "").translate(_QUOTE_TRANSLATION)):
        ctx.sep = ", " if ctx.free else ""
        _parse_segment(ctx, seg)
    last: dict[str, ItemChoice] = {}
    for c in ctx.choices:  # a later instruction for the same item wins
        last.pop(c.item_id, None)
        last[c.item_id] = c
    resp = GateResponse(gate=gate, approve_all=ctx.approve_all, reject_all=ctx.reject_all,
                        choices=list(last.values()), anchors=ctx.anchors, free_text=ctx.free_text(),
                        unresolved=list(dict.fromkeys(ctx.unresolved)))
    resp.interpretation = interpret(resp, ctx.item_ids)
    return resp


# ---------------------------------------------------------------------------------------------
# Gate manager
# ---------------------------------------------------------------------------------------------
class GateManager:
    """Opens and closes gates. Persists every packet, records one Decision per choice, sets gate status.

    It never advances scene stages (the orchestrator does that after reading the gate status).
    """

    def __init__(self, project: Project, state: StateStore | None = None,
                 decisions: DecisionLog | None = None) -> None:
        self.project = project
        self.state = state if state is not None else StateStore(project)
        self.decisions = decisions if decisions is not None else DecisionLog(project)
        self.log_path = project.path("logs", "gates.jsonl")

    @staticmethod
    def _scope(gate: GateId | str, scene_id: str | None) -> GateId:
        gate = GateId(gate)
        if gate in GLOBAL_GATES:
            if scene_id is not None:
                raise ValueError(f"{gate.value} covers the whole film; pass scene_id=None")
        else:
            if not scene_id:
                raise ValueError(f"{gate.value} is a scene gate; pass a scene id")
            validate_id(scene_id, "scene id")
        return gate

    def _normalized(self, packet: GatePacket) -> GatePacket:
        """Copy with review/media paths stored project-relative (refuses paths outside the project)."""
        pkt = packet.model_copy(deep=True)
        if pkt.review_dir:
            pkt.review_dir = self.project.rel(pkt.review_dir)
        for it in pkt.items:
            it.media = [self.project.rel(m) for m in it.media]
        return pkt

    def open(self, packet: GatePacket) -> str:
        gate = self._scope(packet.gate, packet.scene_id)
        pkt = self._normalized(packet)
        ts = compact_ts()
        if pkt.review_dir:
            folder = self.project.abs(pkt.review_dir)
        elif pkt.scene_id:
            folder = self.project.review_dir(pkt.scene_id, gate.value, ts)
        else:
            folder = self.project.path("logs", "gates", f"{gate.value}_{ts}")
        record_path = folder / "gate.yaml"
        text = render_gate(pkt, project_root=self.project.root)
        opened_at = now_iso()
        save_versioned(record_path, {
            "gate": gate.value, "scene_id": pkt.scene_id, "tier": GATE_TIERS[gate].value, "status": "awaiting",
            "opened_at": opened_at, "packet": pkt.model_dump(mode="json"), "rendered": text,
            "generated_by": generated_by("agy.gates").model_dump(mode="json"),
        })
        self.state.set_gate(pkt.scene_id, gate, "awaiting")
        append_jsonl(self.log_path, {
            "event": "open", "ts": opened_at, "gate": gate.value, "scene_id": pkt.scene_id,
            "record": self.project.rel(record_path), "title": pkt.title or GATE_TITLES[gate],
            "items": [it.item_id for it in pkt.items],
        })
        return text

    def _latest_open(self, gate: GateId, scene_id: str | None) -> dict[str, Any] | None:
        latest = None
        for row in read_jsonl(self.log_path):
            if row.get("event") == "open" and row.get("gate") == gate.value and row.get("scene_id") == scene_id:
                latest = row
        return latest

    def latest_packet(self, gate: GateId | str, scene_id: str | None) -> GatePacket | None:
        gate = self._scope(gate, scene_id)
        row = self._latest_open(gate, scene_id)
        if not row or not isinstance(row.get("record"), str):
            return None
        try:
            return GatePacket.model_validate(read_yaml(self.project.abs(row["record"]))["packet"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def respond(self, gate: GateId | str, scene_id: str | None, text: str, *, user: str = "director",
                source: str = "chat") -> tuple[GateResponse, list[Decision]]:
        """Parse a chat reply against the open packet's items, then :meth:`close`."""
        packet = self.latest_packet(gate, scene_id)
        response = parse_gate_response(gate, text, item_ids=[it.item_id for it in packet.items] if packet else ())
        return response, self.close(gate, scene_id, response, user=user, source=source)

    def close(self, gate: GateId | str, scene_id: str | None, response: GateResponse, *, user: str = "director",
              source: str = "chat") -> list[Decision]:
        gate = self._scope(gate, scene_id)
        if GateId(response.gate) != gate:
            raise ValueError(f"that reply is for {GateId(response.gate).value}, not {gate.value}")
        tier = GATE_TIERS[gate]
        packet = self.latest_packet(gate, scene_id)
        recs = {it.item_id: it.recommendation for it in packet.items} if packet else {}
        target = scene_id or "PROJECT"
        free = response.free_text.strip()
        out: list[Decision] = []

        def record(action: DecisionAction, artifact: str, payload: dict[str, Any],
                   tier_: DecisionTier = tier) -> Decision:
            # rationale stays empty: decision_text() falls back to it and must not turn into a preference
            d = self.decisions.record(tier=tier_, target_artifact_id=artifact, action=action,
                                      payload={k: v for k, v in payload.items() if v is not None},
                                      gate=gate, source=source, user=user)
            out.append(d)
            return d

        if response.reject_all:
            record(DecisionAction.REJECT, target, {"all": True, "text": free or None})
        chosen = [c.item_id for c in response.choices]
        if response.approve_all:  # items with their own instruction in this reply are excluded from "all"
            record(DecisionAction.APPROVE, target, {
                "all": True, "except": chosen or None,
                "recommendations": {k: v for k, v in recs.items() if v and k not in chosen}})
        for c in response.choices:
            if c.action == "choose":
                take = c.value if c.value and parse_artifact_id_safe(c.value) else None
                record(DecisionAction.CHOOSE, c.item_id, {"command": "choose", "value": c.value, "take_id": take})
            elif c.action == "ok":
                record(DecisionAction.APPROVE, c.item_id, {"command": "ok", "recommendation": recs.get(c.item_id)})
            elif c.action == "regen":
                d = record(DecisionAction.STEER, c.item_id, {"command": "regen", "steer": c.steer})
                if _is_clip_id(c.item_id):
                    self.state.enqueue_regen(c.item_id, steer=c.steer or "", decision_id=d.id, source=str(source))
            elif c.action == "reject":
                record(DecisionAction.REJECT, c.item_id, {"command": "reject", "steer": c.steer})
            else:
                n = int(c.value) if c.value and str(c.value).isdigit() else None
                record(DecisionAction.REQUEST_ALTERNATIVES, c.item_id,
                       {"command": "alternatives", "n": n, "differ_in": c.steer})
        for char, take in response.anchors.items():
            validate_id(char, "character id")
            if not parse_artifact_id_safe(take):
                raise ValueError(f"voice anchor for {char} must be a take id, got {take!r}")
            vp = self.project.load_voice_profile(char)
            record(DecisionAction.CHOOSE, vp.id if vp else f"VOICE_{char}",
                   {"command": "anchor", "character_id": char, "anchor_take": take}, DecisionTier.VISION)
        if free and not response.reject_all:
            record(DecisionAction.STEER, target, {"text": free})
        status = self._status_after(response, list(recs))
        if status:
            self.state.set_gate(scene_id, gate, status)
        append_jsonl(self.log_path, {
            "event": "close", "ts": now_iso(), "gate": gate.value, "scene_id": scene_id,
            "status": status or self.state.gate_status(scene_id, gate), "decision_ids": [d.id for d in out],
            "interpretation": response.interpretation,
        })
        return out

    @staticmethod
    def _status_after(response: GateResponse, known: list[str]) -> str | None:
        """rejected (anything sent back) / passed (everything approved or chosen) / None (leave as is)."""
        if response.reject_all or any(c.action in ("regen", "reject", "alternatives") for c in response.choices):
            return "rejected"
        if response.approve_all:
            return "passed"
        covered = {c.item_id for c in response.choices}
        if known:
            return "passed" if set(known) <= covered else None
        return "passed" if covered else None

    def auto_pass_blockers(self, gate: GateId | str, scene_id: str, *,
                           clip_ids: Iterable[str] | None = None) -> list[str]:
        """Plain-language reasons why ``gate`` may not auto-pass for ``scene_id`` (empty list = allowed).

        G5 checks the speakers of the clips being auto-selected: ``clip_ids`` when given, else the scene's
        currently selected clips, else every clip, else the scene's dialogue (nothing generated yet)."""
        g = GateId(gate)
        scene = self.project.load_scene(scene_id)
        problems: list[str] = []
        if g == GateId.G2:
            index = self.project.entity_state_index()
            for sid in all_scene_states(scene):
                hit = index.get(sid)
                if hit is None:
                    problems.append(f"{sid} has no reference pack yet")
                elif getattr(hit[1], "approval_status", None) != ApprovalStatus.LOCKED:
                    problems.append(f"{sid} is new (its pack is not locked)")
            if scene.environment_id:
                try:
                    env = self.project.load_environment(scene.environment_id)
                except FileNotFoundError:
                    problems.append(f"{scene.environment_id} has no geometry yet")
                else:
                    if env.approval_status != ApprovalStatus.LOCKED:
                        problems.append(f"{scene.environment_id} is new (its geometry is not locked)")
        elif g == GateId.G5:
            bd = scene.breakdown
            if bd is None:
                problems.append("there is no scene breakdown, so the risk is unknown")
            elif bd.risk_tier != RiskTier.LOW:
                problems.append(f"the scene is {bd.risk_tier.value} risk")
            clips = load_scene_clips(self.project, scene_id)
            if clip_ids is not None:
                wanted = {validate_id(c, "clip id") for c in clip_ids}
                scope = [c for c in clips if c.id in wanted]
            else:
                scope = [c for c in clips if c.status == ClipStatus.SELECTED or c.selected_take_id] or clips
            speakers = list(dict.fromkeys(ch for clip in scope for ch in speakers_in_clip(clip)))
            if not scope and clip_ids is None:
                speakers = list(dict.fromkeys(ln.character_id for ln in scene.dialogue))
            for char in speakers:
                vp = self.project.load_voice_profile(char)
                if vp is not None and not vp.anchor_take:  # same rule as the orchestrator's anchor prompt
                    problems.append(f"{char} speaks for the first time — the director picks the voice anchor")
        return problems

    def auto_pass(self, gate: GateId | str, scene_id: str, *, reason: str,
                  clip_ids: Iterable[str] | None = None) -> Decision:
        """ONLY G2 (nothing new) or G5 (low-risk auto-select); every other gate raises ValueError.

        For G5 pass the auto-selected ``clip_ids`` so only their speakers need a voice anchor."""
        g = GateId(gate)
        if g not in AUTO_PASS_GATES:
            raise ValueError(f"{g.value} is a {GATE_TIERS[g].value} gate — it never auto-proceeds; "
                             "the director must reply")
        self._scope(g, scene_id)
        reason = _flat(reason)
        if not reason:
            raise ValueError("auto-pass needs a reason (it is recorded as a decision)")
        clip_list = list(clip_ids) if clip_ids is not None else None
        blockers = self.auto_pass_blockers(g, scene_id, clip_ids=clip_list)
        if blockers:
            raise ValueError(f"{g.value} for {scene_id} can't auto-pass: " + "; ".join(blockers))
        payload: dict[str, Any] = {"auto": True}
        if clip_list:
            payload["clips"] = clip_list
        d = self.decisions.record(tier=GATE_TIERS[g], target_artifact_id=scene_id, action=DecisionAction.APPROVE,
                                  payload=payload, rationale=reason, gate=g, user="agy")
        self.state.set_gate(scene_id, g, "auto_passed")
        append_jsonl(self.log_path, {"event": "auto_pass", "ts": d.ts, "gate": g.value, "scene_id": scene_id,
                                     "status": "auto_passed", "decision_ids": [d.id], "reason": reason})
        return d

    def pending(self) -> list[dict[str, Any]]:
        """Gates waiting for the director, oldest first."""
        st = self.state.load()
        waiting: list[tuple[str, str | None]] = [(g, None) for g, s in st.gates.items() if s == "awaiting"]
        waiting += [(g, sid) for sid, sc in st.scenes.items() for g, s in sc.gates.items() if s == "awaiting"]
        out = []
        for g, sid in waiting:
            gate = GateId(g)
            row = self._latest_open(gate, sid) or {}
            out.append({"gate": gate.value, "scene_id": sid, "tier": GATE_TIERS[gate].value,
                        "title": row.get("title") or GATE_TITLES[gate], "opened_at": row.get("ts"),
                        "record": row.get("record"), "items": len(row.get("items") or [])})
        return sorted(out, key=lambda r: (r["opened_at"] or "", r["scene_id"] or "", r["gate"]))
