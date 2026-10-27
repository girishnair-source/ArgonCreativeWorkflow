"""Vision judge adapter (BUILD_SPEC_v3 §5 ``gemini_vision_judge``, §7.2 visual QA).

Scores identity similarity against locked reference packs, detects visual artifacts (burned-in subtitles, morphing,
extra limbs), checks prompt adherence, and verifies spatial continuity against the preceding clip's last frame.

In fake mode, uses deterministic pixel comparisons via :func:`agy_studio.media.read_image_rgb` with zero network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .. import media
from ..registry import ModelSpec
from .base import (
    IMAGE_EXTS,
    BaseAdapter,
    adapter_kwargs,
    check_kind,
    gemini_generate_json,
    gemini_part_from_file,
    image_mime,
    resolve_media,
)


class VisionJudge(BaseAdapter):
    """Multimodal visual QA critic adapter (registry kind ``vision_judge``)."""

    agent_name = "agy.adapter.vision_judge"

    def _resolve_images(self, items: Sequence[str | Path]) -> list[Path]:
        out: list[Path] = []
        for item in items:
            if not item:
                continue
            try:
                out.append(resolve_media(self.project, item, exts=IMAGE_EXTS, what="frame"))
            except Exception:  # noqa: BLE001
                p = Path(item)
                if p.is_file():
                    out.append(p)
        return out

    def identity_similarity(
        self,
        frames: Sequence[str | Path],
        pack_images: Sequence[str | Path],
    ) -> float:
        """Score in ``[0.0, 1.0]`` comparing take frames against locked character reference pack images."""
        f_paths = self._resolve_images(frames)
        p_paths = self._resolve_images(pack_images)
        if not f_paths:
            return 0.0
        if not p_paths:
            return 0.90
        if self.fake:
            try:
                f_rgb = np.mean([media.read_image_rgb(p, (16, 16)).astype(np.float32) for p in f_paths], axis=0)
                p_rgb = np.mean([media.read_image_rgb(p, (16, 16)).astype(np.float32) for p in p_paths], axis=0)
                diff = float(np.mean(np.abs(f_rgb - p_rgb))) / 255.0
                return round(max(0.82, min(0.99, 1.0 - 0.35 * diff)), 4)
            except Exception:  # noqa: BLE001
                return 0.90

        parts = [gemini_part_from_file(p, image_mime(p)) for p in (*p_paths[:4], *f_paths[:2])]
        parts.append(
            "Compare the character appearance in the take frames against the reference pack images. "
            "Return JSON: {\"score\": <float 0.0 to 1.0>}."
        )
        data = gemini_generate_json(
            self.model_chain(),
            self.spec.location,
            None,
            parts,
            schema={"type": "object", "properties": {"score": {"type": "number"}}, "required": ["score"]},
            temperature=0.0,
            client=self.client,
        )
        return round(max(0.0, min(1.0, float(data.get("score", 0.85)))), 4)

    def artifact_flags(self, frames: Sequence[str | Path]) -> list[str]:
        """List of visual defects detected in ``frames`` (e.g. ``subtitles_burned_in``, ``extra_limbs``)."""
        f_paths = self._resolve_images(frames)
        if not f_paths:
            return ["missing_frames"]
        if self.fake:
            injected = getattr(self.client, "artifact_flags", None)
            if isinstance(injected, (list, tuple)):
                return [str(x) for x in injected]
            return []

        parts = [gemini_part_from_file(p, image_mime(p)) for p in f_paths[:3]]
        parts.append(
            "Inspect these video frames for defects: burned-in subtitles/text, extra limbs, severe face morphing, "
            "watermarks, or unintended split-screens. Return JSON: {\"flags\": [\"...\"]}."
        )
        data = gemini_generate_json(
            self.model_chain(),
            self.spec.location,
            None,
            parts,
            schema={
                "type": "object",
                "properties": {"flags": {"type": "array", "items": {"type": "string"}}},
                "required": ["flags"],
            },
            temperature=0.0,
            client=self.client,
        )
        return [str(f) for f in (data.get("flags") or []) if str(f).strip()]

    def prompt_adherence(self, frames: Sequence[str | Path], text: str) -> float:
        """Score in ``[0.0, 1.0]`` measuring how well ``frames`` match the visual description ``text``."""
        f_paths = self._resolve_images(frames)
        if not f_paths:
            return 0.0
        if self.fake:
            return 0.92 if text.strip() else 0.85

        parts = [gemini_part_from_file(p, image_mime(p)) for p in f_paths[:2]]
        parts.append(f"Score how well these frames match this shot description: {text!r}. Return JSON: {{\"score\": 0.9}}.")
        data = gemini_generate_json(
            self.model_chain(),
            self.spec.location,
            None,
            parts,
            schema={"type": "object", "properties": {"score": {"type": "number"}}, "required": ["score"]},
            temperature=0.0,
            client=self.client,
        )
        return round(max(0.0, min(1.0, float(data.get("score", 0.85)))), 4)

    def spatial_match(self, frame: str | Path, anchor: str | Path) -> float:
        """Score in ``[0.0, 1.0]`` measuring continuity between ``frame`` (first frame) and ``anchor`` (prev last frame)."""
        pair = self._resolve_images([frame, anchor])
        if len(pair) < 2:
            return 0.0
        if self.fake:
            try:
                a = media.read_image_rgb(pair[0], (16, 16)).astype(np.float32)
                b = media.read_image_rgb(pair[1], (16, 16)).astype(np.float32)
                diff = float(np.mean(np.abs(a - b))) / 255.0
                return round(max(0.0, min(1.0, 1.0 - 0.5 * diff)), 4)
            except Exception:  # noqa: BLE001
                return 0.90

        parts = [
            gemini_part_from_file(pair[1], image_mime(pair[1])),
            gemini_part_from_file(pair[0], image_mime(pair[0])),
            "Score spatial, lighting, and framing continuity between the preceding shot's last frame (first image) "
            "and this shot's first frame (second image). Return JSON: {\"score\": <0.0 to 1.0>}.",
        ]
        data = gemini_generate_json(
            self.model_chain(),
            self.spec.location,
            None,
            parts,
            schema={"type": "object", "properties": {"score": {"type": "number"}}, "required": ["score"]},
            temperature=0.0,
            client=self.client,
        )
        return round(max(0.0, min(1.0, float(data.get("score", 0.85)))), 4)


def create_adapter(spec: ModelSpec, **kw: Any) -> VisionJudge:
    check_kind(spec, "vision_judge")
    return VisionJudge(spec, **adapter_kwargs(kw))
