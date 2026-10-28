"""SFX supplement adapter — wraps ``pro_audio_director`` for hard effects native audio misses.

In fake mode (``fake=True``, unit tests only), writes a deterministic low-frequency percussion/foley
tone WAV via :func:`agy_studio.media.write_tone_wav` with zero network access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import fsio, media
from ..registry import ModelSpec
from .base import BaseAdapter, adapter_kwargs, check_kind, staged_write


class SfxAdapter(BaseAdapter):
    """Hard-effect SFX supplement generator (registry kind ``sfx``)."""

    agent_name = "agy.adapter.sfx"

    def generate(self, prompt: str, duration_s: float, dest: Path | str) -> Path:
        """Generate a hard-effect SFX stem WAV of ``duration_s`` seconds at ``dest`` (staged immutable write)."""
        dest = Path(dest)
        dur = max(0.5, min(float(self.spec.max_duration_s or 10.0), float(duration_s or 2.0)))
        if self.fake:
            u = int(fsio.stable_hash(["agy.fake.sfx", prompt], 8), 16) / float(0xFFFFFFFF)
            freq = round(90.0 + 180.0 * u, 2)
            out = staged_write(
                dest,
                lambda staging: media.write_tone_wav(
                    staging,
                    duration_s=dur,
                    freq=freq,
                    sample_rate=16000,
                    amplitude=0.35,
                    harmonics=(1.0, 0.75, 0.5, 0.25),
                ),
            )
            self.last_result = {"prompt": prompt, "duration_s": dur, "fake": True, "freq_hz": freq}
            return out

        import pro_audio_director  # noqa: PLC0415 - legacy module, real path only

        def _produce(staging: Path) -> Path:
            res = pro_audio_director.generate_omni_360p_sfx_batch(
                [(prompt, staging, max(1, int(round(dur))))],
                force=True,
            )
            return res.get(staging.name, staging)

        out = staged_write(dest, _produce)
        self.last_result = {"prompt": prompt, "duration_s": dur, "fake": False, "provider_model": self.provider_model()}
        return out


def create_adapter(spec: ModelSpec, **kw: Any) -> SfxAdapter:
    check_kind(spec, "sfx")
    return SfxAdapter(spec, **adapter_kwargs(kw))
