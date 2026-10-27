"""Local deterministic speaker-similarity adapter (BUILD_SPEC_v3 §5 ``speaker_sim_local``, §7.2 voice QA).

Computes a pitch-sensitive log-frequency spectral embedding over 16 kHz mono WAVs using numpy only (zero network).
Same-speaker takes (sharing fundamental frequency / harmonic structure) score near 1.0; swapped-voice takes with a
different fundamental frequency score well below the warning/reject thresholds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .. import media
from ..registry import ModelSpec
from .base import AUDIO_EXTS, BaseAdapter, adapter_kwargs, check_kind, resolve_media

N_BINS = 96
F_MIN_HZ = 80.0
F_MAX_HZ = 1200.0


class SpeakerSimAdapter(BaseAdapter):
    """Local numpy log-spectral voice embedding & cosine similarity (registry kind ``speaker_similarity``)."""

    agent_name = "agy.adapter.speaker_sim"

    def embed(self, wav: str | Path) -> np.ndarray:
        """Deterministic L2-normalised voice timbre embedding of shape ``(N_BINS,)``."""
        path = resolve_media(self.project, wav, exts=AUDIO_EXTS, what="voice wav")
        samples, sr = media.read_wav_mono(path)
        if samples.size < 64 or float(np.max(np.abs(samples))) < 1e-5:
            return np.zeros(N_BINS, dtype=np.float32)

        win = min(max(1024, int(sr * 0.12)), samples.size)
        hop = max(256, win // 2)
        window = np.hanning(win).astype(np.float32)
        frames: list[np.ndarray] = []
        for start in range(0, max(1, samples.size - win + 1), hop):
            seg = samples[start : start + win]
            if seg.size < win:
                seg = np.pad(seg, (0, win - seg.size))
            energy = float(np.mean(seg * seg))
            if energy > 1e-7:
                frames.append(np.abs(np.fft.rfft(seg * window)))
        if not frames:
            frames.append(np.abs(np.fft.rfft(samples[:win] * np.hanning(min(win, samples.size)))))

        mag = np.mean(np.stack(frames, axis=0), axis=0).astype(np.float64)
        freqs = np.fft.rfftfreq(win, d=1.0 / float(sr))
        edges = np.geomspace(F_MIN_HZ, min(F_MAX_HZ, sr * 0.45), N_BINS + 1)
        vec = np.zeros(N_BINS, dtype=np.float64)
        for i in range(N_BINS):
            lo, hi = edges[i], edges[i + 1]
            mask = (freqs >= lo) & (freqs < hi)
            if np.any(mask):
                vec[i] = float(np.max(mag[mask]))
            else:
                idx = int(np.argmin(np.abs(freqs - 0.5 * (lo + hi))))
                vec[i] = float(mag[idx])

        vec = np.log1p(np.maximum(0.0, vec))
        baseline = float(np.median(vec))
        vec = np.maximum(0.0, vec - baseline)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-9:
            return np.zeros(N_BINS, dtype=np.float32)
        return (vec / norm).astype(np.float32)

    def similarity(self, wav_a: str | Path, wav_b: str | Path) -> float:
        """Cosine similarity in ``[0.0, 1.0]`` between two WAV files."""
        emb_a = self.embed(wav_a)
        emb_b = self.embed(wav_b)
        na = float(np.linalg.norm(emb_a))
        nb = float(np.linalg.norm(emb_b))
        if na <= 1e-8 or nb <= 1e-8:
            return 0.0
        cos = float(np.dot(emb_a, emb_b) / (na * nb))
        return round(max(0.0, min(1.0, cos)), 4)


def create_adapter(spec: ModelSpec, **kw: Any) -> SpeakerSimAdapter:
    check_kind(spec, "speaker_similarity")
    return SpeakerSimAdapter(spec, **adapter_kwargs(kw))
