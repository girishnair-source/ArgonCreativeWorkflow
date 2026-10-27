"""Nano Banana Pro image adapter (``gemini-3-pro-image-preview`` + fallbacks, up to 14 reference images).

Used for character turnaround sheets, environment geometry plates, derived camera angles, and shot storyboards.
In fake mode, renders a deterministic solid-colour PNG derived from the prompt and any reference images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .. import fsio, media
from ..registry import ModelSpec
from .base import (
    IMAGE_EXTS,
    AdapterError,
    AuthError,
    BaseAdapter,
    UnsupportedFeatureError,
    adapter_kwargs,
    check_kind,
    gemini_client,
    gemini_part_from_file,
    image_mime,
    is_auth_failure,
    is_safety_error,
    raise_if_auth_failure,
    resolve_media,
    safety_reprompt_text,
    scrub,
    short_error,
    staged_write,
)

_ASPECT_SIZES: dict[str, tuple[int, int]] = {
    "9:16": (72, 128),
    "16:9": (128, 72),
    "3:4": (72, 96),
    "4:3": (96, 72),
    "1:1": (96, 96),
}


class ImageAdapter(BaseAdapter):
    """Multi-reference image generator (registry kind ``image``)."""

    agent_name = "agy.adapter.image"

    def generate(
        self,
        prompt: str,
        dest: Path | str,
        *,
        reference_images: Sequence[str | Path] = (),
        aspect_ratio: str = "9:16",
    ) -> Path:
        dest = Path(dest)
        max_refs = int(self.spec.supports.max_reference_images or 14)
        if len(reference_images) > max_refs:
            raise UnsupportedFeatureError(
                f"{self.spec.id}: {len(reference_images)} reference images > max_reference_images {max_refs}"
            )
        ref_paths = [
            resolve_media(self.project, item, exts=IMAGE_EXTS, what="reference image")
            for item in reference_images
        ]
        if self.fake:
            w, h = _ASPECT_SIZES.get(aspect_ratio, (72, 128))
            rgb = self._fake_color(prompt, aspect_ratio, ref_paths)
            out = staged_write(dest, lambda staging: media.write_png(staging, width=w, height=h, rgb=rgb))
            self.last_result = {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "reference_images": [p.name for p in ref_paths],
                "fake": True,
                "rgb": rgb,
            }
            return out

        from google.genai import types  # noqa: PLC0415 - real path only

        client = self.client or gemini_client(self.spec.location)
        ref_parts: list[Any] = [gemini_part_from_file(p, image_mime(p)) for p in ref_paths]
        errors: list[str] = []

        def _produce(staging: Path) -> Path:
            for model in self.model_chain():
                active_prompt = prompt
                for safety_level in range(3):
                    try:
                        resp = client.models.generate_content(
                            model=model,
                            contents=[*ref_parts, active_prompt],
                            config=types.GenerateContentConfig(
                                response_modalities=["IMAGE"],
                                image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
                            ),
                        )
                        data = _extract_image_bytes(resp)
                        if data:
                            staging.write_bytes(data)
                            self.last_result = {
                                "provider_model": model,
                                "aspect_ratio": aspect_ratio,
                                "reference_images": [p.name for p in ref_paths],
                                "safety_level": safety_level,
                                "fake": False,
                            }
                            return staging
                        reason = f"{model}: response contained no image bytes (safety_level={safety_level})"
                        errors.append(reason)
                    except AuthError:
                        raise  # credentials: stop now — the next model / safety level would fail identically
                    except Exception as exc:  # noqa: BLE001
                        err_msg = short_error(exc)
                        if self.client is None and is_auth_failure(exc):
                            # ADC mismatch -> retry exactly once with the gcloud token; a broken gcloud raises AuthError
                            alt = gemini_client(self.spec.location, use_gcloud_token=True)
                            try:
                                resp = alt.models.generate_content(
                                    model=model,
                                    contents=[*ref_parts, active_prompt],
                                    config=types.GenerateContentConfig(
                                        response_modalities=["IMAGE"],
                                        image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
                                    ),
                                )
                                data = _extract_image_bytes(resp)
                                if data:
                                    staging.write_bytes(data)
                                    self.last_result = {
                                        "provider_model": model,
                                        "aspect_ratio": aspect_ratio,
                                        "reference_images": [p.name for p in ref_paths],
                                        "safety_level": safety_level,
                                        "fake": False,
                                    }
                                    return staging
                            except Exception as exc2:  # noqa: BLE001
                                raise_if_auth_failure(exc2, context=f"generating {dest.name} with {model}")
                                err_msg = short_error(exc2)
                        reason = f"{model}: {err_msg}"
                        errors.append(reason)
                        if not is_safety_error(err_msg) and "no image" not in err_msg.lower():
                            break
                    if safety_level < 2:
                        old_p = active_prompt
                        active_prompt, rewrites = safety_reprompt_text(prompt, level=safety_level + 1)
                        if self.project is not None:
                            fsio.append_jsonl(
                                self.project.path("logs", "safety_retries.jsonl"),
                                {
                                    "ts": fsio.now_iso(),
                                    "target": dest.name,
                                    "model": self.spec.id,
                                    "provider_model": model,
                                    "level": safety_level + 1,
                                    "reason": scrub(reason),
                                    "rewrites_applied": rewrites,
                                    "original_prompt": old_p,
                                    "rewritten_prompt": active_prompt,
                                },
                            )
            raise AdapterError("image generation failed across provider chain: " + "; ".join(errors))

        return staged_write(dest, _produce)

    @staticmethod
    def _fake_color(prompt: str, aspect_ratio: str, ref_paths: list[Path]) -> tuple[int, int, int]:
        digest = fsio.stable_hash(["agy.fake.image", prompt, aspect_ratio, [p.name for p in ref_paths]], 6)
        r = 40 + (int(digest[0:2], 16) % 180)
        g = 40 + (int(digest[2:4], 16) % 180)
        b = 40 + (int(digest[4:6], 16) % 180)
        if ref_paths:
            try:
                ref_rgb = media.read_image_rgb(ref_paths[0], (8, 8)).reshape(-1, 3).mean(axis=0)
                r = int(round(0.75 * float(ref_rgb[0]) + 0.25 * r))
                g = int(round(0.75 * float(ref_rgb[1]) + 0.25 * g))
                b = int(round(0.75 * float(ref_rgb[2]) + 0.25 * b))
            except Exception:  # noqa: BLE001
                pass
        return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def _extract_image_bytes(resp: Any) -> bytes | None:
    for cand in getattr(resp, "candidates", None) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None) if inline is not None else None
            if data:
                return bytes(data) if not isinstance(data, str) else __import__("base64").b64decode(data)
    return None


def create_adapter(spec: ModelSpec, **kw: Any) -> ImageAdapter:
    check_kind(spec, "image")
    return ImageAdapter(spec, **adapter_kwargs(kw))
