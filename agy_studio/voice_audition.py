"""Gate 4 Voice Discovery, Audition & Global Voice Bible Lock (`voice_audition.py`).

Implements the Gate 4 workflow from the AGY Studio v4 blueprint:
- Once the Storyboard (Gate 3) is locked, AGY inspects every speaking character in the scene:
  1. **Existing character in Global Bible** (already has a locked ``VoiceProfile`` + ``reference_audio`` in
     ``library/voices/``): reuses the locked voice from the Global Bible automatically.
  2. **New character** (or character without a locked audition WAV): generates tone, voice timbre, pitch, and an
     audition WAV sample (``scenes/<SC>/audio/auditions/<CHAR_ID>_audition.wav``) for Gate 4 review.
- When Gate 4 is approved, :func:`lock_approved_voices_to_global_bible` promotes the audition WAV and ``VoiceProfile``
  into ``library/voices/<CHAR_ID>/v<N>/`` and updates ``bible/global_context_bible.yaml`` so all subsequent scenes
  reuse the exact same voice print.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from . import media
from .bible_hierarchy import build_global_context_bible, load_global_context_bible
from .fsio import generated_by, next_version_path, validate_id
from .models import ApprovalStatus, VoiceProfile
from .project import Project

AGENT = "agy.voice_audition"


def _character_pitch_hz(character_id: str, description: str = "") -> float:
    """Deterministic fundamental pitch from character description / id for audition synthesis."""
    desc = (description or "").lower()
    digest = int(hashlib.sha256(character_id.encode("utf-8")).hexdigest()[:4], 16)
    jitter = (digest % 25) - 12
    if any(w in desc for w in ("female", "woman", "mother", "sister", "girl", "ria", "mom")):
        return float(210 + jitter)
    if any(w in desc for w in ("automated", "synthetic", "robot", "ai", "sphere")):
        return float(165 + jitter)
    if any(w in desc for w in ("father", "older", "baritone", "deep", "stern")):
        return float(115 + jitter)
    return float(135 + jitter)


def prepare_scene_voices(project: Project, scene_id: str, *, fake: bool = False) -> dict[str, Any]:
    """Ensure every speaking character in ``scene_id`` either reuses a locked Global Bible voice or gets a generated
    audition sample + VoiceProfile staged for Gate 4 approval.
    """
    validate_id(scene_id, "scene id")
    scene = project.load_scene(scene_id)
    gb = load_global_context_bible(project)
    speakers = list(dict.fromkeys(ln.character_id for ln in scene.dialogue))

    reused_voices: list[dict[str, Any]] = []
    new_auditions: list[dict[str, Any]] = []

    audition_dir = project.scene_dir(scene_id) / "audio" / "auditions"
    audition_dir.mkdir(parents=True, exist_ok=True)

    for char_id in speakers:
        vp = project.load_voice_profile(char_id)
        g_char = gb.characters.get(char_id)

        # Case 1: Voice already locked in Global Bible with a valid reference audio file
        existing_ref = (vp.reference_audio or vp.anchor_wav) if vp else (g_char.voice_anchor_uri if g_char else None)
        if (
            vp is not None
            and vp.approval_status == ApprovalStatus.LOCKED
            and existing_ref
            and str(existing_ref).startswith("library/")
            and project.abs(existing_ref).is_file()
        ):
            reused_voices.append(
                {
                    "character_id": char_id,
                    "voice_id": vp.id,
                    "voice_profile_id": vp.id,
                    "description": vp.description,
                    "reference_audio": existing_ref,
                    "anchor_wav": existing_ref,
                    "source": "global_bible",
                }
            )
            continue

        # Case 2: New character or voice without a locked audition sample -> generate tone/pitch/audition WAV
        if vp is None:
            ent = None
            try:
                ent = project.load_entity(char_id)
            except FileNotFoundError:
                pass
            ent_name = (getattr(ent, "display_name", None) or getattr(ent, "name", None) or ent.id) if ent else char_id
            base_desc = (
                f"Natural cinematic voice for {ent_name} ({ent.canonical_description[:120]})"
                if ent
                else f"Clear, grounded vocal timbre for {char_id}, medium pitch, natural conversational pace"
            )
            vp = VoiceProfile(
                id=f"VOICE_{char_id}",
                character_id=char_id,
                description=base_desc,
                language=project.config.language,
                approval_status=ApprovalStatus.PROPOSED,
                generated_by=generated_by(AGENT),
            )

        freq = _character_pitch_hz(char_id, vp.description)
        wav_path = next_version_path(audition_dir, f"{char_id}_audition", ".wav")
        media.write_wav(wav_path, duration_s=2.4, freq_hz=freq)
        rel_wav = project.rel(wav_path)

        vp.reference_audio = rel_wav
        vp.generated_by = generated_by(AGENT, notes=f"audition staged in {scene_id} ({freq:.0f} Hz)")
        project.save_voice_profile(vp)

        new_auditions.append(
            {
                "character_id": char_id,
                "voice_id": vp.id,
                "voice_profile_id": vp.id,
                "description": vp.description,
                "pitch_hz": freq,
                "audition_wav": rel_wav,
                "source": "new_audition",
            }
        )

    build_global_context_bible(project, save=True)
    return {
        "scene_id": scene_id,
        "reused_from_global": reused_voices,
        "reused_from_global_bible": reused_voices,
        "auditioned": new_auditions,
        "new_auditions": new_auditions,
        "auto_pass_eligible": len(new_auditions) == 0,
    }


def lock_approved_voices_to_global_bible(
    project: Project,
    scene_id: str,
    *,
    decision_id: str | None = None,
    user: str = "director",
) -> list[str]:
    """Promote all speaking characters' approved voices in ``scene_id`` into ``library/voices/`` and lock them in
    ``GlobalContextBible`` so future scenes reuse them automatically.
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

        # Ensure a WAV exists to promote into library/voices/<CHAR_ID>/vN/
        src_wav: Path | None = None
        if vp.reference_audio and project.abs(vp.reference_audio).is_file():
            src_wav = project.abs(vp.reference_audio)
        elif vp.anchor_wav and project.abs(vp.anchor_wav).is_file():
            src_wav = project.abs(vp.anchor_wav)
        else:
            audition_dir = project.scene_dir(scene_id) / "audio" / "auditions"
            audition_dir.mkdir(parents=True, exist_ok=True)
            tmp_wav = audition_dir / f"{char_id}_audition_v1.wav"
            media.write_wav(tmp_wav, duration_s=2.4, freq_hz=_character_pitch_hz(char_id, vp.description))
            src_wav = tmp_wav

        if not (vp.reference_audio and str(vp.reference_audio).startswith("library/")):
            prov = generated_by(AGENT, notes=f"locked at Gate 4 ({scene_id}) by {user}")
            rels = project.promote_library_files(("voices", char_id), {"audition.wav": src_wav}, provenance=prov)
            vp.reference_audio = next(iter(rels.values()))
        if not vp.anchor_wav:
            vp.anchor_wav = vp.reference_audio

        was_locked = vp.approval_status == ApprovalStatus.LOCKED
        vp.approval_status = ApprovalStatus.LOCKED
        vp.version += 1
        vp.generated_by = generated_by(AGENT, notes=f"locked in Global Bible at G4 ({scene_id})")
        project.save_voice_profile(vp, confirm_decision_id=decision_id if was_locked else None)
        locked_chars.append(char_id)

    build_global_context_bible(project, save=True)
    return locked_chars
