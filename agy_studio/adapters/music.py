"""Music cue adapter — wraps ``pro_audio_director.generate_lyria_music_stem`` (never edited, imported lazily).

In fake mode (``AGY_FAKE_GENERATION=1`` / ``fake=True``), writes a deterministic harmonic WAV via
:func:`agy_studio.media.write_tone_wav` with zero network access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import fsio, media
from ..registry import ModelSpec
from .base import BaseAdapter, adapter_kwargs, check_kind, staged_write


class MusicAdapter(BaseAdapter):
    """Music cue generator (registry kind ``music``)."""

    agent_name = "agy.adapter.music"

    def generate_cue(
        self,
        prompt: str,
        duration_s: float,
        dest: Path | str,
        *,
        negative_prompt: str = "vocals, singing, speech, spoken words, cheering, clapping",
    ) -> Path:
        """Generate an instrumental music cue WAV of ``duration_s`` seconds at ``dest`` (staged immutable write)."""
        dest = Path(dest)
        dur = max(0.5, float(duration_s or 10.0))
        if self.fake:
            u = int(fsio.stable_hash(["agy.fake.music", prompt], 8), 16) / float(0xFFFFFFFF)
            freq = round(130.81 * (2.0 ** u), 2)
            out = staged_write(
                dest,
                lambda staging: media.write_tone_wav(
                    staging,
                    duration_s=dur,
                    freq=freq,
                    sample_rate=16000,
                    amplitude=0.22,
                    harmonics=(1.0, 0.5, 0.25, 0.125),
                ),
            )
            self.last_result = {"prompt": prompt, "duration_s": dur, "fake": True, "freq_hz": freq}
            return out

        import pro_audio_director  # noqa: PLC0415 - legacy module, real path only

        out = staged_write(
            dest,
            lambda staging: pro_audio_director.generate_lyria_music_stem(
                prompt=prompt,
                target_wav=staging,
                duration_seconds=dur,
                negative_prompt=negative_prompt,
                force=True,
            ),
        )
        self.last_result = {"prompt": prompt, "duration_s": dur, "fake": False, "provider_model": self.provider_model()}
        return out


def create_adapter(spec: ModelSpec, **kw: Any) -> MusicAdapter:
    check_kind(spec, "music")
    return MusicAdapter(spec, **adapter_kwargs(kw))
