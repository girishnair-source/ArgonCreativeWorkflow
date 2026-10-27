"""Gate 4 Voice Discovery & Global Voice Bible Lock (`voice_audition.py`).

What Gate 4 locks is a **voice specification** — text — not audio:

* Gemini Omni / Veo generate the speech natively while rendering the clip; the only voice signal that reaches the
  video model is the per-line ``voice`` text that :mod:`agy_studio.prompt_engineer` fills from
  :attr:`VoiceProfile.description` (``registry.yaml`` has ``voice_reference_audio: false`` for both models).
* Therefore no audition WAV is synthesised here. (An earlier version wrote a 2.4 s sine tone as the "audition" and
  then used it as ``anchor_wav`` — the QA critic compared real Omni speech against a sine tone and hard-rejected every
  take as ``speaker_voice_mismatch``.) The real acoustic anchor is still the dialogue of the first approved speaking
  take (:meth:`Studio.set_voice_anchor`), extracted from genuine model output.

Workflow:

1. **Existing character in the Global Bible** (locked ``VoiceProfile``): reused automatically.
2. **New character**: a deterministic, physical voice spec (sex/age register, timbre, pitch band, pace, language and
   accent) is derived from the character's Bible description and staged as a ``proposed`` ``VoiceProfile`` for Gate 4.
3. On Gate 4 approval :func:`lock_approved_voices_to_global_bible` locks the profiles in
   ``bible/global_context_bible.yaml`` so every later scene reuses the identical voice spec.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .bible_hierarchy import build_global_context_bible, load_global_context_bible
from .fsio import generated_by, validate_id
from .models import ApprovalStatus, VoiceProfile
from .project import Project

AGENT = "agy.voice_audition"

_FEMALE_WORDS = ("female", "woman", "mother", "sister", "girl", "daughter", "wife", "aunt", "grandmother", "lady",
                 "she ", "her ")
_SYNTHETIC_WORDS = ("automated", "synthetic", "robot", "ai ", "sphere", "device", "speaker voice", "announcer")
_ELDER_WORDS = ("father", "older", "elderly", "grandfather", "uncle", "baritone", "deep", "stern", "veteran", "60s",
                "70s", "aged")
_CHILD_WORDS = ("child", "kid", "boy", "little", "young girl", "young boy", "teen", "school")


def _character_pitch_hz(character_id: str, description: str = "") -> float:
    """Deterministic fundamental pitch (Hz) from the character description / id — the centre of the voice's band."""
    desc = (description or "").lower()
    digest = int(hashlib.sha256(character_id.encode("utf-8")).hexdigest()[:4], 16)
    jitter = (digest % 25) - 12
    if any(w in desc for w in _CHILD_WORDS):
        return float(260 + jitter)
    if any(w in desc for w in _FEMALE_WORDS):
        return float(210 + jitter)
    if any(w in desc for w in _SYNTHETIC_WORDS):
        return float(165 + jitter)
    if any(w in desc for w in _ELDER_WORDS):
        return float(115 + jitter)
    return float(135 + jitter)


def voice_spec_from_description(character_id: str, name: str, description: str, language: str | None) -> str:
    """A compact, physical voice specification the video model can act on (no audio involved).

    Deterministic for a given description so re-runs lock the same spec; the Director can still edit it at Gate 4
    (``C01:regen "<delivery note>"``) and the Voice Director agent refines per-line delivery on top of it.
    """
    desc = (description or "").lower()
    pitch = _character_pitch_hz(character_id, description)
    if any(w in desc for w in _CHILD_WORDS):
        register, timbre = "child voice", "light, bright timbre"
    elif any(w in desc for w in _FEMALE_WORDS):
        register, timbre = "adult female voice", "warm mezzo timbre with clear diction"
    elif any(w in desc for w in _SYNTHETIC_WORDS):
        register, timbre = "neutral synthetic announcer voice", "even, slightly compressed timbre"
    elif any(w in desc for w in _ELDER_WORDS):
        register, timbre = "older adult male voice", "deep, grainy baritone"
    else:
        register, timbre = "adult male voice", "grounded medium-low timbre, slight chest resonance"
    pace = "measured, natural conversational pace"
    if any(w in desc for w in ("anxious", "nervous", "young", "energetic", "excited", "influencer", "streamer")):
        pace = "quick, energetic conversational pace"
    elif any(w in desc for w in ("calm", "stern", "authority", "officer", "doctor", "judge")):
        pace = "slow, deliberate, authoritative pace"
    lang = f"speaks {language}" if language else "speaks in the script's native language"
    return (f"{name}: {register}, {timbre}, fundamental pitch around {pitch:.0f} Hz, {pace}; {lang} with natural "
            f"native articulation; same voice in every shot — no accent drift, no pitch shift between takes.")


def prepare_scene_voices(project: Project, scene_id: str, *, fake: bool = False) -> dict[str, Any]:
    """Ensure every speaking character in ``scene_id`` has a VoiceProfile: reused from the Global Bible when locked,
    otherwise a proposed voice spec staged for Gate 4. **No audio is generated** (see module docstring).

    ``fake`` is accepted for signature compatibility; the function is pure bookkeeping in every mode.
    """
    validate_id(scene_id, "scene id")
    scene = project.load_scene(scene_id)
    gb = load_global_context_bible(project)
    speakers = list(dict.fromkeys(ln.character_id for ln in scene.dialogue))

    reused_voices: list[dict[str, Any]] = []
    new_specs: list[dict[str, Any]] = []

    for char_id in speakers:
        vp = project.load_voice_profile(char_id)
        g_char = gb.characters.get(char_id)

        # Case 1: voice spec already locked in the Global Bible -> reuse verbatim
        if vp is not None and vp.approval_status == ApprovalStatus.LOCKED:
            reused_voices.append(
                {
                    "character_id": char_id,
                    "voice_id": vp.id,
                    "voice_profile_id": vp.id,
                    "description": vp.description,
                    "anchor_take": vp.anchor_take,
                    "anchor_wav": vp.anchor_wav,
                    "reference_audio": vp.reference_audio,
                    "source": "global_bible",
                }
            )
            continue

        # Case 2: new character (or an unlocked profile) -> derive a physical voice spec from the Bible description
        if vp is None:
            ent = None
            try:
                ent = project.load_entity(char_id)
            except FileNotFoundError:
                pass
            ent_name = (getattr(ent, "display_name", None) or getattr(ent, "name", None) or ent.id) if ent else char_id
            canonical = getattr(ent, "canonical_description", "") if ent else ""
            if not canonical and g_char is not None:
                canonical = str(getattr(g_char, "description", "") or "")
            vp = VoiceProfile(
                id=f"VOICE_{char_id}",
                character_id=char_id,
                description=voice_spec_from_description(char_id, ent_name, canonical, project.config.language),
                language=project.config.language,
                approval_status=ApprovalStatus.PROPOSED,
                generated_by=generated_by(AGENT, notes=f"voice spec derived from Bible description for {scene_id}"),
            )
            project.save_voice_profile(vp)

        new_specs.append(
            {
                "character_id": char_id,
                "voice_id": vp.id,
                "voice_profile_id": vp.id,
                "description": vp.description,
                "pitch_hz": _character_pitch_hz(char_id, vp.description),
                "source": "new_voice_spec",
            }
        )

    build_global_context_bible(project, save=True)
    return {
        "scene_id": scene_id,
        "reused_from_global": reused_voices,
        "reused_from_global_bible": reused_voices,
        "auditioned": new_specs,  # name kept for callers; these are text specs, not audio auditions
        "new_voice_specs": new_specs,
        "auto_pass_eligible": len(new_specs) == 0,
    }


def lock_approved_voices_to_global_bible(
    project: Project,
    scene_id: str,
    *,
    decision_id: str | None = None,
    user: str = "director",
) -> list[str]:
    """Lock every speaking character's voice spec in the Global Bible so later scenes reuse it verbatim.

    Only the specification is locked here. ``anchor_take`` / ``anchor_wav`` stay empty until the first approved
    speaking take sets them from real model output (:meth:`Studio.set_voice_anchor`) — never from synthetic audio.
    """
    validate_id(scene_id, "scene id")
    scene = project.load_scene(scene_id)
    speakers = list(dict.fromkeys(ln.character_id for ln in scene.dialogue))
    locked_chars: list[str] = []

    for char_id in speakers:
        vp = project.load_voice_profile(char_id)
        if vp is None:
            prepare_scene_voices(project, scene_id)
            vp = project.load_voice_profile(char_id)
        if vp is None:
            continue
        was_locked = vp.approval_status == ApprovalStatus.LOCKED
        if was_locked:
            locked_chars.append(char_id)
            continue
        vp.approval_status = ApprovalStatus.LOCKED
        vp.version += 1
        vp.generated_by = generated_by(AGENT, notes=f"voice spec locked in Global Bible at G4 ({scene_id}) by {user}")
        project.save_voice_profile(vp)
        locked_chars.append(char_id)

    build_global_context_bible(project, save=True)
    return locked_chars
