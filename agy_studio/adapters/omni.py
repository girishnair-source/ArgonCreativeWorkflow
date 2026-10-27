"""Gemini Omni 1.1 video adapter — wraps ``vertex_veo_director.GeminiOmniClient`` (never edited, imported lazily).

Request mapping (registry ``omni``: refs_with_first_frame true, max 9 refs, no negative-prompt field):

* first frame (``image_path``) + up to ``max_reference_images`` asset references -> ``reference_to_video``;
  first frame (+ optional last frame) without references -> ``image_to_video``;
  references only -> the first reference is the primary image (``reference_to_video``);
  neither -> ``text_to_video`` through the same enterprise interactions client the legacy pipeline uses.
* negatives are folded into the prompt (``Avoid: ...``) because Omni has no negative-prompt field.
* output goes to ``<VERTEX_GCS_BUCKET_ROOT>/agy/<PROJECT>/<payload_hash>/a<attempt>`` (inline delivery when GCS is
  refused), the take folder receives it via a staged immutable write.
* every provider call runs through :meth:`LegacyVideoAdapter.call_with_auth`: an HTTP 401 during a long poll forces a
  gcloud token refresh and retries once (the interaction-fetch 401 listed in registry ``known_failures``).

Provider fallbacks, RAI re-rolls, crash-safe dispatch/resume are inherited from :class:`VideoAdapter`.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from ..models import DispatchStatus, GenerationPayload, PromptBundle
from ..registry import ModelSpec
from .base import (
    AdapterError,
    LegacyVideoAdapter,
    PollResult,
    UnsupportedFeatureError,
    VideoAdapter,
    create_video_adapter,
    fold_negatives,
    short_error,
    vertex_project_id,
)

IN_PROGRESS = frozenset({"in_progress", "running", "pending", "queued", "processing", "created", ""})
FAILED_STATES = frozenset({"failed", "error", "cancelled", "canceled", "expired", "incomplete"})
_RAI_WORDS = ("safety", "rai", "responsible ai", "blocked", "prohibited", "policy", "filtered", "sensitive")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def interaction_contents(inter: Any) -> list[Any]:
    """Model-output content items of an Omni interaction (``steps[type=model_output].content`` or ``outputs``)."""
    contents: list[Any] = []
    for step in _get(inter, "steps", None) or []:
        if _get(step, "type", "") == "model_output":
            contents.extend(_get(step, "content", None) or [])
    if not contents:
        contents.extend(_get(inter, "outputs", None) or [])
    return contents


def interaction_usage(inter: Any) -> dict[str, Any] | None:
    """Plain-dict token usage of a finished Omni interaction (input/output/thought tokens), None when absent."""
    usage = _get(inter, "usage", None)
    if not usage:
        return None
    out: dict[str, Any] = {}
    for key in ("total_input_tokens", "total_output_tokens", "total_thought_tokens", "total_tokens", "total_cached_tokens"):
        val = _get(usage, key, None)
        if val is not None:
            try:
                out[key] = int(val)
            except (TypeError, ValueError):
                continue
    for key in ("input_tokens_by_modality", "output_tokens_by_modality"):
        items = _get(usage, key, None) or []
        try:
            by_mod = {str(_get(m, "modality", "?")): int(_get(m, "tokens", 0) or 0) for m in items}
        except (TypeError, ValueError):
            by_mod = {}
        if by_mod:
            out[key] = by_mod
    return out or None


def _looks_filtered(text: str) -> bool:
    low = text.lower()
    return any(word in low for word in _RAI_WORDS)


class OmniAdapter(LegacyVideoAdapter):
    """Real Omni wrapper. Pass ``client=`` (anything with ``start_generation`` / ``fetch_interaction``) to inject a
    test double; otherwise a ``GeminiOmniClient`` is built lazily on the first real call."""

    agent_name = "agy.adapter.omni"
    default_poll_interval_s = 15.0
    default_timeout_s = 1800.0

    def __init__(self, spec: ModelSpec, **kw: Any) -> None:
        super().__init__(spec, **kw)
        self._omni: Any = self.client if self._is_omni_client(self.client) else None

    @staticmethod
    def _is_omni_client(obj: Any) -> bool:
        return obj is not None and hasattr(obj, "start_generation") and hasattr(obj, "fetch_interaction")

    def omni_client(self) -> Any:
        if self._omni is None:
            legacy = self.legacy()
            location = self.spec.location or str(legacy.OMNI_LOCATION)
            self._omni = legacy.GeminiOmniClient(vertex_project_id(), self.token, location)
        return self._omni

    # -- validation ---------------------------------------------------------------------------
    def check_bundle(self, bundle: PromptBundle) -> None:
        super().check_bundle(bundle)
        if bundle.last_frame and bundle.reference_images:
            raise UnsupportedFeatureError(
                f"{self.spec.id}: a last-frame target cannot be combined with asset reference images in one Omni "
                "request (reference_to_video ignores the end image) - drop one of them and re-lint")
        if bundle.last_frame and not bundle.first_frame:
            raise UnsupportedFeatureError(f"{self.spec.id}: a last-frame target needs a first frame (image_to_video)")

    # -- provider primitives ------------------------------------------------------------------
    def submit(self, payload: GenerationPayload, *, provider_model: str | None = None, attempt: int = 0) -> str:
        bundle = payload.bundle
        self.check_bundle(bundle)
        model = provider_model or payload.provider_model
        prompt = fold_negatives(bundle.prompt, bundle.negative_prompt)  # Omni has no negative-prompt field
        first = self.media_path(bundle.first_frame)
        last = self.media_path(bundle.last_frame)
        limit = max(0, int(self.spec.supports.max_reference_images))
        refs = [p for p in (self.media_path(r) for r in bundle.reference_images[:limit]) if p is not None]
        duration = int(round(float(self.spec.snap_duration(bundle.duration_s))))
        gcs = self.gcs_prefix(payload, attempt)
        client = self.omni_client()

        def start(prefix: str | None) -> tuple[str, dict[str, Any]]:
            if first is None and not refs:
                return self._text_to_video(client, model, prompt, bundle, duration, prefix)
            primary = first if first is not None else refs[0]
            others = refs if first is not None else refs[1:]
            task = "reference_to_video" if others or (first is None and refs) else "image_to_video"
            return client.start_generation(
                model, prompt, primary,
                reference_image_paths=others or None,
                end_image_path=last if task == "image_to_video" else None,
                task=task,
                aspect_ratio=bundle.aspect_ratio,
                duration_seconds=duration,
                resolution=bundle.resolution,
                gcs_output_prefix=prefix,
            )

        try:
            op_id, params = self.call_with_auth(lambda: start(gcs))
        except Exception as exc:  # noqa: BLE001 - mirror the legacy pipeline: GCS refusal -> inline delivery
            message = str(exc).lower()
            if gcs and any(word in message for word in ("storage", "bucket", "gcs")):
                gcs = None
                op_id, params = self.call_with_auth(lambda: start(None))
            else:
                raise
        self.last_result = {"provider_model": model, "attempt": attempt, "params": _safe_params(params),
                            "gcs_output_prefix": gcs}
        if not op_id:
            raise AdapterError(f"{model}: interactions.create returned no interaction id")
        return str(op_id)

    def _text_to_video(self, client: Any, model: str, prompt: str, bundle: PromptBundle, duration: int,
                       gcs: str | None) -> tuple[str, dict[str, Any]]:
        """No image at all: ``text_to_video`` through the legacy client's enterprise genai client."""
        from google.genai import interactions  # noqa: PLC0415 - real path only

        resp_kwargs: dict[str, Any] = {"aspect_ratio": bundle.aspect_ratio, "duration": f"{max(3, min(10, duration))}s",
                                       "resolution": bundle.resolution}
        if gcs:
            resp_kwargs["delivery"] = "uri"
            resp_kwargs["gcs_uri"] = gcs.rstrip("/")
        inter = client._get_client().interactions.create(  # noqa: SLF001 - same call the legacy pipeline makes
            model=model,
            input=[{"type": "text", "text": prompt}],
            generation_config=interactions.GenerationConfig(video_config=interactions.VideoConfig(task="text_to_video")),
            response_format=interactions.VideoResponseFormat(**resp_kwargs),
            background=True,
        )
        return str(inter.id), {"task": "text_to_video", **resp_kwargs}

    def poll(self, operation_name: str) -> PollResult:
        client = self.omni_client()
        try:
            inter = self.call_with_auth(lambda: client.fetch_interaction(operation_name))
        except Exception as exc:  # noqa: BLE001 - transient: keep running, bounded by max_poll_errors
            return self.poll_failed(operation_name, exc)
        self.poll_ok(operation_name)
        status = str(_get(inter, "status", "") or "").lower()
        raw: dict[str, Any] = {"operation_name": operation_name, "provider_status": status}
        if status in IN_PROGRESS:
            return PollResult(status=DispatchStatus.RUNNING, raw=raw)
        try:
            usage = interaction_usage(inter)
        except Exception:  # noqa: BLE001 - usage is informational only, never fail a poll on it
            usage = None
        if usage:
            raw["usage"] = usage
        contents = interaction_contents(inter)
        if status == "completed" and contents:
            item = contents[0]
            uri = _get(item, "uri", None)
            data = _get(item, "data", None)
            if uri and str(uri).startswith("gs://"):
                return PollResult(status=DispatchStatus.COMPLETED, video_uri=str(uri), raw=raw)
            if data:
                raw["inline_data"] = data if isinstance(data, str) else base64.b64encode(bytes(data)).decode("ascii")
                return PollResult(status=DispatchStatus.COMPLETED, raw=raw)
            return PollResult(status=DispatchStatus.FAILED, raw=raw,
                              error={"stage": "poll", "message": "interaction completed without a video uri or data"})
        detail = short_error(_get(inter, "error", None) or inter, 400)
        filtered = _looks_filtered(detail)
        if status == "completed" and not contents:
            filtered = filtered or True  # completed with no output: Omni's silent safety drop
        result_status = DispatchStatus.FILTERED if filtered else DispatchStatus.FAILED
        return PollResult(status=result_status, raw=raw,
                          error={"stage": "poll", "provider_status": status or "unknown", "message": detail})


def _safe_params(params: Any) -> dict[str, Any]:
    """Provider parameters without inline media (paths only, never base64 bodies)."""
    if not isinstance(params, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[str(key)] = value if not isinstance(value, str) else value[:300]
        elif isinstance(value, (list, tuple)):
            out[str(key)] = [Path(str(v)).name for v in value][:20]
    return out


def create_adapter(spec: ModelSpec, **kw: Any) -> VideoAdapter:
    """Registry factory: :class:`FakeVideoAdapter` in fake mode, else :class:`OmniAdapter`."""
    return create_video_adapter(spec, OmniAdapter, **kw)
