"""ASR & audio-defect classifier adapter (BUILD_SPEC_v3 §5 ``gemini_asr``, §7.2 audio QA).

* ``transcribe(wav, *, language_hint="hi", expected_text=None)`` returns the verbatim spoken transcript (Devanagari
  for Hindi).
* ``classify_audio(wav)`` inspects the clip's native audio for spontaneous background music, unwanted narration,
  or extra voices beyond the declared speakers.

In fake mode, reads the ``fake_take.json`` sidecar written by :class:`~agy_studio.adapters.fake.FakeVideoAdapter`
so tests can inject wrong transcripts, spontaneous music, extra voices, or silent tracks deterministically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .. import media
from ..registry import ModelSpec
from .base import (
    AUDIO_EXTS,
    BaseAdapter,
    adapter_kwargs,
    check_kind,
    gemini_generate_json,
    gemini_part_from_file,
    resolve_media,
)
from .fake import read_fake_sidecar


class AsrAdapter(BaseAdapter):
    """Speech transcription and unwanted-audio detector (registry kind ``transcribe``)."""

    agent_name = "agy.adapter.asr"

    def transcribe(
        self,
        wav: str | Path,
        *,
        language_hint: str = "hi",
        expected_text: str | None = None,
    ) -> str:
        path = resolve_media(self.project, wav, exts=AUDIO_EXTS, what="audio wav")
        if self.fake:
            sidecar = read_fake_sidecar(path)
            if sidecar is not None:
                if sidecar.get("no_audio"):
                    return ""
                spoken = sidecar.get("spoken_text")
                if spoken is not None:
                    return str(spoken).strip()
            if expected_text is not None:
                return str(expected_text).strip()
            samples, _ = media.read_wav_mono(path)
            if samples.size == 0 or float(np.max(np.abs(samples))) < 1e-3:
                return ""
            return ""

        prompt = (
            f"Transcribe the exact spoken words in this audio clip (language hint: {language_hint}). "
            "Use authentic Devanagari script for Hindi dialogue. Return JSON: {\"transcript\": \"...\"}. "
            "If there is no speech, return {\"transcript\": \"\"}."
        )
        data = gemini_generate_json(
            self.model_chain(),
            self.spec.location,
            None,
            [gemini_part_from_file(path, "audio/wav"), prompt],
            schema={
                "type": "object",
                "properties": {"transcript": {"type": "string"}},
                "required": ["transcript"],
            },
            temperature=0.0,
            client=self.client,
        )
        return str(data.get("transcript", "")).strip()

    def classify_audio(self, wav: str | Path) -> dict[str, Any]:
        """Return ``{music_present, narration_present, extra_voice, n_speakers, notes}``."""
        path = resolve_media(self.project, wav, exts=AUDIO_EXTS, what="audio wav")
        if self.fake:
            sidecar = read_fake_sidecar(path)
            if sidecar is not None:
                if sidecar.get("no_audio"):
                    return {
                        "music_present": False,
                        "narration_present": False,
                        "extra_voice": False,
                        "n_speakers": 0,
                        "notes": "silent fake take",
                    }
                music = bool(sidecar.get("music", False))
                extra_voice = bool(sidecar.get("extra_voice", False))
                speakers = list(sidecar.get("speakers") or [])
                n_speakers = len(speakers) + (1 if extra_voice else 0)
                return {
                    "music_present": music,
                    "narration_present": False,
                    "extra_voice": extra_voice,
                    "n_speakers": n_speakers,
                    "notes": "from fake_take.json",
                }
            samples, _ = media.read_wav_mono(path)
            silent = samples.size == 0 or float(np.max(np.abs(samples))) < 1e-3
            return {
                "music_present": False,
                "narration_present": False,
                "extra_voice": False,
                "n_speakers": 0 if silent else 1,
                "notes": "heuristic mono wav check",
            }

        prompt = (
            "Inspect this film take's audio track and report whether it contains unwanted background music/score, "
            "unwanted voiceover narration, or extra unscripted speakers. "
            "Return JSON with keys: music_present (bool), narration_present (bool), extra_voice (bool), "
            "n_speakers (int), notes (string)."
        )
        data = gemini_generate_json(
            self.model_chain(),
            self.spec.location,
            None,
            [gemini_part_from_file(path, "audio/wav"), prompt],
            schema={
                "type": "object",
                "properties": {
                    "music_present": {"type": "boolean"},
                    "narration_present": {"type": "boolean"},
                    "extra_voice": {"type": "boolean"},
                    "n_speakers": {"type": "integer"},
                    "notes": {"type": "string"},
                },
                "required": ["music_present", "narration_present", "extra_voice", "n_speakers", "notes"],
            },
            temperature=0.0,
            client=self.client,
        )
        return {
            "music_present": bool(data.get("music_present", False)),
            "narration_present": bool(data.get("narration_present", False)),
            "extra_voice": bool(data.get("extra_voice", False)),
            "n_speakers": int(data.get("n_speakers", 0) or 0),
            "notes": str(data.get("notes", "")),
        }


def create_adapter(spec: ModelSpec, **kw: Any) -> AsrAdapter:
    check_kind(spec, "transcribe")
    return AsrAdapter(spec, **adapter_kwargs(kw))
