"""Prompt Engineer — GenClip -> model-ready PromptBundle / GenerationPayload (BUILD_SPEC_v3 §6 "Prompt Engineer").

Deterministic serialisation of ONE (linted) GenClip for ONE registry model. The bundle mirrors the clip: durations,
lines, speakers and conditioning are never changed — those belong to the Shot Decomposer and the Pre-flight Linter.

Prompt layout (paragraphs separated by blank lines):

1. **Visual** — the nine ShotSpec fields in the FIXED order (L10): subject_and_action, camera, lens_and_aspect,
   lighting, palette, composition, atmosphere, mood, attribution. ``prompt_style: prose`` -> one sentence per field;
   ``structured`` -> ``Label: value`` lines. Entity / state ids become display names, lighting-setup ids their
   physical description, environment ids a place name.
2. **Reference legend** — which 1-based ``reference_images`` index shows whom (also stored as
   ``PromptBundle.reference_legend``), derived-state deltas (e.g. the INJURED look) and a plain description for any
   visible entity that has no image in the bundle.
3. **Performance** — ``clip.performance_state``.
4. **Audio** — :func:`render_audio_block` with the model's ``audio_prompt_syntax``: the exact Devanagari line in
   quotes + phonetic transliteration + the locked voice description + delivery (``line``); ``off_screen`` for unseen
   speakers; ``listeners_closed`` for every visible character who is not speaking (all of them during off-screen
   lines and silent clips); ``ambience``; ``sfx``; ``music_none`` (always — music is laid in assembly).
5. **Negatives** — the model's negative field when ``supports.negative_prompt``, otherwise ``Avoid: ...``.

L1 banned words are stripped from every generated sentence; the verbatim dialogue and its transliteration are
inserted untouched (the linter's L14 guards them). When a clip carries more reference images than the target model
accepts (e.g. a fallback model with a smaller ``max_reference_images``), the images are re-prioritised with the
decomposer's allocator: one image per pack first, then the spatial anchor, then round-robin.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from . import continuity
from .decomposer import (
    active_state_map,
    allocate_reference_images,
    load_voice_profiles,
    merge_unique,
    scan_visual_refs,
    scene_shot_order,
    state_owner_index,
)
from .fsio import compute_payload_hash, validate_id
from .linter import MUSIC_TERMS_RX, strip_banned, strip_music_terms
from .models import (
    VISUAL_FIELDS,
    AudioLine,
    AudioSpec,
    Entity,
    EntityState,
    EntityType,
    Environment,
    GenClip,
    GenerationPayload,
    ModelKind,
    ProjectSettings,
    PromptBundle,
    Scene,
    VisualSpec,
    VoiceProfile,
)
from .project import Project
from .registry import ModelSpec, Registry

AGENT_NAME = "Prompt Engineer"
PROVENANCE_AGENT = "agy.prompt_engineer"
AUDIO_SYNTAX_KEYS: tuple[str, ...] = ("line", "off_screen", "listeners_closed", "ambience", "sfx", "music_none")
DEFAULT_AUDIO_SYNTAX: dict[str, str] = {  # only used when registry.yaml lacks (or mangles) a key
    "line": '{speaker} ({voice}) speaks {language}, {delivery}: "{text}"{phonetic}',
    "off_screen": 'Off-screen, {speaker} ({voice}) is heard {delivery}: "{text}"{phonetic}. '
                  "Every on-screen mouth stays closed.",
    "listeners_closed": "{names} keep their mouths fully closed and do not speak.",
    "ambience": "Ambient sound: {ambience}.",
    "sfx": "Sound effects: {sfx}.",
    "music_none": "No background music, no score, no narration, no subtitles.",
}
_REQUIRED_PLACEHOLDERS: dict[str, tuple[str, ...]] = {
    "line": ("text",), "off_screen": ("text",), "listeners_closed": ("names",), "ambience": ("ambience",),
    "sfx": ("sfx",), "music_none": (),
}
FIELD_LABELS: dict[str, str] = {
    "subject_and_action": "Subject and action",
    "camera": "Camera",
    "lens_and_aspect": "Lens and aspect",
    "lighting": "Lighting",
    "palette": "Palette",
    "composition": "Composition",
    "atmosphere": "Atmosphere",
    "mood": "Mood",
    "attribution": "Attribution",
}
LANGUAGE_NAMES: dict[str, str] = {
    "hi": "Hindi", "en": "English", "mr": "Marathi", "ta": "Tamil", "te": "Telugu", "bn": "Bengali",
    "gu": "Gujarati", "pa": "Punjabi", "ur": "Urdu", "kn": "Kannada", "ml": "Malayalam",
}
INDIAN_LANGUAGE_CODES = frozenset({"hi", "mr", "ta", "te", "bn", "gu", "pa", "ur", "kn", "ml"})
NATIVE_ARTICULATION = "natural native Indian articulation, zero English accent"
PHONETIC_FORMAT = " (phonetic transliteration: {translit})"
IDENTITY_NOTE = "Keep every face, hairstyle, outfit and prop identical to its reference images."
FIRST_FRAME_NOTE = "Open exactly on the supplied first frame and continue its motion seamlessly."
EXTRA_REFERENCE_KEY = "EXTRA_REFERENCE"
SPATIAL_ANCHOR_KEY = "SPATIAL_ANCHOR"
_ENTITY_PREFIXES: tuple[str, ...] = ("CHAR_", "PROP_", "VEHICLE_", "VEH_", "CREATURE_", "CRE_")
_PLACEHOLDER_RX = re.compile(r"\{([a-z_]+)\}")
_TEXT_SLOT = "\x00TEXT\x00"  # the verbatim dialogue is inserted after every clean-up pass
_PHONETIC_SLOT = "\x00PHON\x00"


# ---------------------------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------------------------
def fallback_display_name(entity_id: str) -> str:
    """``CHAR_SECURITY_GUARD`` -> ``Security Guard`` (used when an entity has no ``display_name``)."""
    base = entity_id or ""
    for prefix in _ENTITY_PREFIXES:
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    words = [w.capitalize() for w in base.split("_") if w]
    return " ".join(words) or entity_id


def display_name(entity: Entity | None, entity_id: str) -> str:
    if entity is not None and (entity.display_name or "").strip():
        return str(entity.display_name).strip()
    return fallback_display_name(entity_id)


def place_name(environment_id: str) -> str:
    """``ENV_ROOFTOP_TERRACE`` -> ``the rooftop terrace``."""
    base = environment_id[4:] if environment_id.startswith("ENV_") else environment_id
    words = [w.lower() for w in base.split("_") if w]
    return "the " + " ".join(words) if words else environment_id


def join_names(names: Sequence[str]) -> str:
    """``["A"]`` -> ``A``; ``["A", "B", "C"]`` -> ``A, B and C``."""
    items = [n for n in dict.fromkeys((n or "").strip() for n in names) if n]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def language_label(language: str | None, *, default: str = "hi") -> str:
    """Spoken-language label for the audio templates. Indian languages always carry the workspace's
    native-articulation mandate (natural native Indian articulation, zero English accent)."""
    raw = " ".join((language or "").split()) or default
    label = LANGUAGE_NAMES.get(raw.casefold(), raw)
    low = label.casefold()
    indian = raw.casefold() in INDIAN_LANGUAGE_CODES or any(
        LANGUAGE_NAMES[code].casefold() in low for code in INDIAN_LANGUAGE_CODES)
    if indian and "accent" not in low:
        label = f"{label}, {NATIVE_ARTICULATION}"
    return label


def readable(text: str | None, names: Mapping[str, str]) -> str:
    """Replace registered UPPERCASE ids with their readable names (unknown tokens and acronyms stay)."""
    return continuity.ENTITY_TOKEN_RE.sub(lambda m: names.get(m.group(0), m.group(0)), text or "")


def _plain(text: str | None) -> str:
    """Single-spaced, without trailing sentence punctuation (templates add their own)."""
    return re.sub(r"[\s.;,:]+$", "", " ".join((text or "").split()))


def _sentence(text: str | None) -> str:
    t = " ".join((text or "").split()).rstrip(" ,;:")
    if not t:
        return ""
    if t[0].isalpha() and t[0].islower():
        t = t[0].upper() + t[1:]
    if t[-1] not in ".!?":
        t += "."
    return t


def tidy_sentence(text: str) -> str:
    """Repair punctuation left by empty template values (``speaks Hindi, : `` -> ``speaks Hindi: ``)."""
    t = re.sub(r"\(\s*\)", " ", text or "")
    t = re.sub(r",\s*(?=[,:;.!?])", "", t)
    t = re.sub(r"\s+([,:;.!?)])", r"\1", t)
    t = re.sub(r"\(\s+", "(", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def render_template(template: str, values: Mapping[str, str]) -> str:
    """Fill ``{name}`` placeholders (unknown names -> empty). Registry text is data, so no ``str.format``."""
    return _PLACEHOLDER_RX.sub(lambda m: str(values.get(m.group(1), "")), template or "")


def audio_syntax(spec: ModelSpec) -> dict[str, str]:
    """The model's ``audio_prompt_syntax`` with defaults for missing keys (a template that lost its required
    placeholder — e.g. ``line`` without ``{text}`` — falls back to the default so no dialogue is ever dropped)."""
    out = dict(DEFAULT_AUDIO_SYNTAX)
    for key, template in (spec.audio_prompt_syntax or {}).items():
        if key not in out or not isinstance(template, str) or not template.strip():
            continue
        if all("{" + name + "}" in template for name in _REQUIRED_PLACEHOLDERS[key]):
            out[key] = template
    return out


def _index_label(indices: Sequence[int]) -> str:
    idx = sorted(indices)
    if not idx:
        return ""
    if len(idx) == 1:
        return str(idx[0])
    if idx == list(range(idx[0], idx[-1] + 1)):
        return f"{idx[0]}-{idx[-1]}"
    return ", ".join(str(i) for i in idx)


def readable_names(entities: Mapping[str, Entity], environments: Mapping[str, Environment] | None = None,
                   voices: Mapping[str, VoiceProfile] | None = None) -> dict[str, str]:
    """id -> readable text for every registered id a prompt may contain (entities, states, environments, lighting
    setups, voices)."""
    names: dict[str, str] = {}
    for eid, ent in entities.items():
        name = display_name(ent, eid)
        names[eid] = name
        for st in ent.states:
            names[st.id] = name
    for env_id, env in (environments or {}).items():
        names[env_id] = place_name(env_id)
        for ls in env.lighting_setups:
            if (ls.description or "").strip():
                names[ls.id] = _plain(ls.description)
    for vp in (voices or {}).values():
        owner = names.get(vp.character_id) or fallback_display_name(vp.character_id)
        names[vp.id] = f"{owner}'s voice"
    for eid, ent in entities.items():
        if ent.voice_profile_id and ent.voice_profile_id not in names:
            names[ent.voice_profile_id] = f"{names[eid]}'s voice"
    return names


# ---------------------------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------------------------
def render_visual_block(visual: VisualSpec, names: Mapping[str, str], *, prompt_style: str = "prose",
                        allowed_words: Iterable[str] = ()) -> str:
    """The nine visual fields in the fixed L10 order (empty fields are skipped; L10 fails them upstream)."""
    allowed = list(allowed_words)
    structured = (prompt_style or "").strip().casefold() == "structured"
    parts: list[str] = []
    for name in VISUAL_FIELDS:
        text = _sentence(strip_banned(readable(getattr(visual, name, "") or "", names), allowed)[0])
        if text:
            parts.append(f"{FIELD_LABELS[name]}: {text}" if structured else text)
    return ("\n" if structured else " ").join(parts)


def render_reference_legend(legend: Mapping[str, Sequence[int]], labels: Mapping[str, str]) -> str:
    """``Reference images: 1-4 show Rishi; 9 shows the terrace set layout.``"""
    parts: list[str] = []
    for key, indices in legend.items():
        if not indices:
            continue
        verb = "shows" if len(indices) == 1 else "show"
        parts.append(f"{_index_label(indices)} {verb} {labels.get(key, key)}")
    return "Reference images: " + "; ".join(parts) + "." if parts else ""


def render_audio_block(spec: ModelSpec, audio: AudioSpec, *, names: Mapping[str, str] | None = None,
                       visible: Sequence[str] = (), languages: Mapping[str, str] | None = None,
                       default_language: str = "Hindi", allowed_words: Iterable[str] = ()) -> str:
    """Serialise ``audio`` with ``spec.audio_prompt_syntax``.

    * ``line`` for every on-screen line and ``off_screen`` for every unseen speaker: speaker display name, the
      locked ``voice`` description, the ``language`` label, ``delivery``, the EXACT ``text`` in quotes and the
      phonetic transliteration;
    * ``listeners_closed`` for every ``visible`` character without an on-screen line (so during an off-screen /
      phone / narrator line, or a silent clip, every visible mouth stays closed);
    * ``ambience``, ``sfx`` (music clauses dropped) and ALWAYS ``music_none``.

    ``names`` maps ids to readable names, ``languages`` character id -> language label.
    """
    syntax = audio_syntax(spec)
    name_map = dict(names or {})
    langs = dict(languages or {})
    allowed = list(allowed_words)

    def value(text: str | None) -> str:
        return _plain(strip_banned(readable(text, name_map), allowed)[0])

    def finish(rendered: str) -> str:
        return strip_banned(tidy_sentence(rendered), allowed)[0]

    def speaker(character_id: str) -> str:
        return name_map.get(character_id) or fallback_display_name(character_id)

    def spoken(key: str, line: AudioLine) -> str:
        rendered = finish(render_template(syntax[key], {
            "speaker": speaker(line.character_id),
            "voice": value(line.voice),
            "language": value(langs.get(line.character_id) or default_language),
            "delivery": value(line.delivery),
            "text": _TEXT_SLOT,
            "phonetic": _PHONETIC_SLOT,
        }))
        translit = " ".join((line.transliteration or "").split())
        phonetic = PHONETIC_FORMAT.format(translit=translit) if translit else ""
        if phonetic and _PHONETIC_SLOT not in rendered:  # template without {phonetic}: keep the transliteration
            quoted = _TEXT_SLOT + '"'
            target = quoted if quoted in rendered else _TEXT_SLOT
            rendered = rendered.replace(target, target + _PHONETIC_SLOT, 1)
        return rendered.replace(_TEXT_SLOT, line.text).replace(_PHONETIC_SLOT, phonetic)

    parts: list[str] = [spoken("line", ln) for ln in audio.dialogue]
    parts.extend(spoken("off_screen", ln) for ln in audio.off_screen)
    talking = {ln.character_id for ln in audio.dialogue}
    listeners = [cid for cid in dict.fromkeys(visible) if cid not in talking]
    if listeners:
        names_text = join_names([speaker(cid) for cid in listeners])
        parts.append(finish(render_template(syntax["listeners_closed"], {"names": names_text})))
    ambience = value(strip_music_terms(audio.ambience or "")[0])
    if ambience:
        parts.append(finish(render_template(syntax["ambience"], {"ambience": ambience})))
    sfx = [value(item) for item in audio.diegetic_sfx if not MUSIC_TERMS_RX.search(item or "")]
    sfx = [item for item in sfx if item]
    if sfx:
        parts.append(finish(render_template(syntax["sfx"], {"sfx": ", ".join(sfx)})))
    parts.append(finish(render_template(syntax["music_none"], {})))
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------------------------
# Reference images
# ---------------------------------------------------------------------------------------------
def plan_reference_images(clip: GenClip, spec: ModelSpec, states: Mapping[str, EntityState], *,
                          environment_id: str | None = None) -> tuple[list[str], dict[str, list[int]]]:
    """(reference_images, legend). Pack images first (conditioning order = tier priority), the spatial anchor last.

    When the clip carries more images than ``spec.supports.max_reference_images`` (a fallback model with a smaller
    cap) they are re-prioritised with :func:`decomposer.allocate_reference_images` (one per pack, then the anchor,
    then round-robin). A model without reference support keeps the clip's images untouched — the adapter's bundle
    check then refuses it instead of silently generating without identity references.
    """
    cond = clip.conditioning
    pack_owner: dict[str, str] = {}
    for sid, st in states.items():
        for uri in st.reference_pack.image_uris():
            pack_owner.setdefault(uri, sid)
    groups: dict[str, list[str]] = {}
    placed: set[str] = set()
    for sid in dict.fromkeys(cond.reference_entity_states):
        st = states.get(sid)
        pack = set(st.reference_pack.image_uris()) if st is not None else set()
        uris = [u for u in dict.fromkeys(cond.reference_uris) if u in pack and u not in placed]
        if uris:
            groups.setdefault(sid, []).extend(uris)
            placed.update(uris)
    for uri in cond.reference_uris:
        if uri and uri not in placed:
            groups.setdefault(pack_owner.get(uri, EXTRA_REFERENCE_KEY), []).append(uri)
            placed.add(uri)
    anchor = cond.spatial_anchor if cond.spatial_anchor and cond.spatial_anchor not in placed else None
    cap = int(spec.supports.max_reference_images)
    total = sum(len(v) for v in groups.values()) + (1 if anchor else 0)
    if spec.supports.reference_to_video and total > cap:
        chosen, with_anchor = allocate_reference_images(list(groups.items()), cap, anchor=anchor)
        groups = {sid: list(chosen.get(sid, [])) for sid in groups}
        anchor = anchor if with_anchor else None
    images: list[str] = []
    legend: dict[str, list[int]] = {}
    for sid, uris in groups.items():
        for uri in uris:
            images.append(uri)
            legend.setdefault(sid, []).append(len(images))
    if anchor:
        images.append(anchor)
        legend.setdefault(environment_id or SPATIAL_ANCHOR_KEY, []).append(len(images))
    return images, legend


def _resolution(settings: ProjectSettings, spec: ModelSpec) -> str:
    short = min(int(settings.width), int(settings.height))
    target = f"{short}p"
    if not spec.resolutions or target in spec.resolutions:
        return target
    numeric = sorted((int(r[:-1]), r) for r in spec.resolutions if r.endswith("p") and r[:-1].isdigit())
    below = [r for n, r in numeric if n <= short]
    if below:
        return below[-1]
    return numeric[0][1] if numeric else spec.resolutions[0]


# ---------------------------------------------------------------------------------------------
# Bundle / payload
# ---------------------------------------------------------------------------------------------
@dataclass
class _View:
    """Project records one bundle needs (loaded once per call)."""

    scene: Scene
    entities: dict[str, Entity]
    states: dict[str, EntityState]
    state_owner: dict[str, str]
    environments: dict[str, Environment]
    voices: dict[str, VoiceProfile]
    settings: ProjectSettings
    names: dict[str, str]

    @classmethod
    def build(cls, project: Project, clip: GenClip) -> "_View":
        scene = project.load_scene(validate_id(clip.scene_id, "scene id"))
        entities = {e.id: e for e in project.list_entities()}
        environments = {e.id: e for e in project.list_environments()}
        characters = [e.id for e in entities.values() if e.type == EntityType.CHARACTER]
        speakers = [ln.character_id for ln in clip.spec.audio_spec.all_lines()]
        voices = load_voice_profiles(project, [*characters, *speakers])
        return cls(
            scene=scene,
            entities=entities,
            states={st.id: st for ent in entities.values() for st in ent.states},
            state_owner=state_owner_index(entities),
            environments=environments,
            voices=voices,
            settings=project.config.settings,
            names=readable_names(entities, environments, voices),
        )


def target_model_id(registry: Registry, clip: GenClip, model_id: str | None = None) -> str:
    """``model_id`` > the linter's ``model_used`` (L12 fallback) > ``clip.model_target.preferred``.

    Raises :class:`registry.UnknownModelError` for unknown ids and ValueError for non-video models.
    """
    if model_id:
        chosen = model_id
    elif clip.lint_report is not None and clip.lint_report.model_used:
        chosen = clip.lint_report.model_used
    else:
        chosen = clip.model_target.preferred
    spec = registry.get(chosen)
    if spec.kind != ModelKind.VIDEO_AV:
        raise ValueError(f"{chosen} is a {ModelKind(spec.kind).value} model, not a video model")
    return chosen


def provider_model_for(project: Project, registry: Registry, model_id: str) -> str:
    """Concrete provider model: ``project.config.model_pins`` override, else the registry value."""
    return project.config.model_pins.get(model_id) or registry.get(model_id).provider_model


def _continuity_notes(view: _View, referenced: Sequence[str], resolved: Mapping[str, str | None],
                      legend: Mapping[str, Sequence[int]]) -> list[str]:
    with_images = {view.state_owner.get(key) for key in legend}
    notes: list[str] = []
    for eid in referenced:
        ent = view.entities.get(eid)
        if ent is None:
            continue
        details: list[str] = []
        state = view.states.get(resolved.get(eid) or "")
        if state is not None and state.derived_from and (state.delta_description or "").strip():
            details.append(_plain(state.delta_description))
        if eid not in with_images and (ent.canonical_description or "").strip():
            details.append(_plain(ent.canonical_description))
        if details:
            detail = "; ".join(details)
            notes.append(f"{view.names.get(eid) or display_name(ent, eid)} ({detail})")
    return notes


def build_prompt_bundle(project: Project, registry: Registry, clip: GenClip, *,
                        model_id: str | None = None) -> PromptBundle:
    """Serialise ``clip`` for ``model_id`` (see the module docstring for the layout). Pure: writes nothing."""
    validate_id(clip.id, "clip id")
    model = target_model_id(registry, clip, model_id)
    spec = registry.get(model)
    view = _View.build(project, clip)
    settings = view.settings
    allowed = list(settings.lint_allowed_words)
    cond = clip.conditioning

    refs = scan_visual_refs(clip.spec.visual, entities=view.entities, state_owner=view.state_owner)
    shot_order = scene_shot_order(project, view.scene)
    active = active_state_map(view.scene, shot_order, clip.shot_id, clip.sequence_index,
                              state_owner=view.state_owner, entity_ids=view.entities)
    referenced = list(refs.entities)
    for cid in continuity.on_screen_speakers(clip):
        if cid in view.entities and cid not in referenced:
            referenced.append(cid)
    resolved = {eid: refs.explicit_states.get(eid) or active.get(eid) for eid in referenced}
    visible = [eid for eid in referenced if view.entities[eid].type == EntityType.CHARACTER]

    env_key = view.scene.environment_id or SPATIAL_ANCHOR_KEY
    images, legend = plan_reference_images(clip, spec, view.states, environment_id=env_key)
    labels = dict(view.names)
    labels[env_key] = (place_name(env_key) + " set layout") if env_key != SPATIAL_ANCHOR_KEY else "the set layout"
    labels[EXTRA_REFERENCE_KEY] = "an extra pinned reference"

    audio = clip.spec.audio_spec.model_copy(deep=True)
    languages: dict[str, str] = {}
    for line in audio.all_lines():
        vp = view.voices.get(line.character_id)
        if not (line.voice or "").strip() and vp is not None:  # the linter (L15) normally guarantees it
            line.voice = vp.voice_for_state(active.get(line.character_id))
        if line.character_id not in languages:
            source = view.scene.line(line.line_id) if line.line_id else None
            label_src = (vp.language if vp is not None and vp.language else None) or \
                (source.language if source is not None and source.language else None) or project.config.language
            languages[line.character_id] = language_label(label_src)

    style = (spec.prompt_style or "prose").strip().casefold()
    segments: list[str] = [render_visual_block(clip.spec.visual, view.names, prompt_style=style,
                                               allowed_words=allowed)]
    legend_text = render_reference_legend(legend, labels)
    if legend_text:
        segments.append(strip_banned(legend_text + " " + IDENTITY_NOTE, allowed)[0])
    notes = _continuity_notes(view, referenced, resolved, legend)
    if notes:
        segments.append(strip_banned("Continuity: " + "; ".join(notes) + ".", allowed)[0])
    if cond.first_frame_anchor:
        segments.append(FIRST_FRAME_NOTE)
    performance = _sentence(strip_banned(readable(clip.performance_state, view.names), allowed)[0])
    if performance:
        segments.append("Performance \u2014 " + performance)
    segments.append(render_audio_block(spec, audio, names=view.names, visible=visible, languages=languages,
                                       default_language=language_label(project.config.language),
                                       allowed_words=allowed))
    negatives = merge_unique([strip_banned(readable(n, view.names), allowed)[0] for n in clip.spec.negative])
    negative_prompt: str | None = None
    if negatives:
        if spec.supports.negative_prompt:
            negative_prompt = ", ".join(negatives)
        else:
            segments.append("Avoid: " + ", ".join(negatives) + ".")

    voice_audio = list(cond.voice_reference_audio) if (
        spec.supports.voice_reference_audio and settings.open_decisions.voice_reference_audio) else []
    for rel in [*images, cond.first_frame_anchor, cond.last_frame_target, *voice_audio]:
        if rel:
            project.abs(rel)  # PathSecurityError (a ValueError) for absolute or escaping paths
    return PromptBundle(
        model_id=model,
        clip_id=clip.id,
        prompt="\n\n".join(s for s in segments if s and s.strip()),
        negative_prompt=negative_prompt,
        reference_images=images,
        reference_legend=legend,
        first_frame=cond.first_frame_anchor,
        last_frame=cond.last_frame_target,
        voice_reference_audio=voice_audio,
        duration_s=float(clip.duration_s),
        aspect_ratio=settings.aspect_ratio,
        resolution=_resolution(settings, spec),
    )


def payload_for(project: Project, registry: Registry, clip: GenClip, *, model_id: str | None = None,
                seed: int | None = None) -> GenerationPayload:
    """Bundle + provider model (``model_pins`` override) + content-addressed ``payload_hash``.

    Identical to what ``adapters.base.VideoAdapter.build_payload`` produces for the same clip, model and seed, so the
    Shot Generator's dispatch dedupe (acceptance criterion 9) sees the same hash. ``extra`` is informational and not
    part of the hash.
    """
    bundle = build_prompt_bundle(project, registry, clip, model_id=model_id)
    provider = provider_model_for(project, registry, bundle.model_id)
    payload_hash = compute_payload_hash(project.root, model_id=bundle.model_id, provider_model=provider,
                                        bundle=bundle, seed=seed)
    return GenerationPayload(
        model_id=bundle.model_id,
        provider_model=provider,
        bundle=bundle,
        seed=seed,
        payload_hash=payload_hash,
        extra={
            "clip_id": clip.id,
            "scene_id": clip.scene_id,
            "speakers": continuity.speakers_in_clip(clip),
            "expected_text": " ".join(continuity.clip_line_texts(clip)),
        },
    )
