"""Vertex AI Veo 3.1 video adapter — wraps ``vertex_veo_director.VertexVeoClient`` (never edited, imported lazily).

Request mapping (registry ``veo``: refs_with_first_frame false, max 3 refs, native negativePrompt):

* first frame (``image_path``) + optional last frame (``lastFrame``) -> ``predictLongRunning`` image-to-video;
* asset reference images (up to 3, when no first frame is set) -> ``referenceImages`` with ``referenceType="asset"``;
* text-only -> ``predictLongRunning`` with ``prompt`` only;
* ``negative_prompt`` is sent in ``parameters.negativePrompt``;
* every provider call runs through :meth:`LegacyVideoAdapter.call_with_auth` so HTTP 401 refreshes the gcloud token;
* provider fallbacks (``veo-3.1-generate-001`` -> ``veo-3.1-fast-generate-001``), RAI retries, and crash-safe
  dispatch/resume are inherited from :class:`VideoAdapter`.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from ..models import DispatchStatus, GenerationPayload, PromptBundle
from ..registry import ModelSpec
from .base import (
    AdapterError,
    AuthError,
    LegacyVideoAdapter,
    PollResult,
    UnsupportedFeatureError,
    VideoAdapter,
    b64_file,
    create_video_adapter,
    image_mime,
    run_coro_sync,
    short_error,
    vertex_project_id,
)

_RAI_WORDS = ("safety", "rai", "responsible ai", "blocked", "prohibited", "policy", "filtered", "sensitive")


def _looks_filtered(text: str) -> bool:
    low = text.lower()
    return any(word in low for word in _RAI_WORDS)


class VeoAdapter(LegacyVideoAdapter):
    """Real Veo 3.1 wrapper around ``vertex_veo_director.VertexVeoClient``."""

    agent_name = "agy.adapter.veo"
    default_poll_interval_s = 15.0
    default_timeout_s = 1800.0

    def __init__(self, spec: ModelSpec, **kw: Any) -> None:
        super().__init__(spec, **kw)
        self._veo: Any = self.client if self._is_veo_client(self.client) else None
        self._op_models: dict[str, str] = {}

    @staticmethod
    def _is_veo_client(obj: Any) -> bool:
        return obj is not None and (hasattr(obj, "start_image_to_video") or hasattr(obj, "fetch_operation"))

    def veo_client(self) -> Any:
        if self._veo is None:
            legacy = self.legacy()
            location = self.spec.location or str(legacy.VERTEX_LOCATION)
            self._veo = legacy.VertexVeoClient(vertex_project_id(), location, self.token)
        return self._veo

    def check_bundle(self, bundle: PromptBundle) -> None:
        super().check_bundle(bundle)
        if bundle.last_frame and not bundle.first_frame:
            raise UnsupportedFeatureError(f"{self.spec.id}: a last-frame target requires a first frame")

    def submit(self, payload: GenerationPayload, *, provider_model: str | None = None, attempt: int = 0) -> str:
        bundle = payload.bundle
        self.check_bundle(bundle)
        model = provider_model or payload.provider_model
        first = self.media_path(bundle.first_frame)
        last = self.media_path(bundle.last_frame)
        limit = max(0, int(self.spec.supports.max_reference_images))
        refs = [p for p in (self.media_path(r) for r in bundle.reference_images[:limit]) if p is not None]
        duration = int(round(float(self.spec.snap_duration(bundle.duration_s))))
        gcs = self.gcs_prefix(payload, attempt)
        negative = (bundle.negative_prompt or "").strip()
        client = self.veo_client()

        def _invoke() -> tuple[str, dict[str, Any]]:
            if (first is not None and last is None and not refs and bundle.aspect_ratio == "9:16"
                    and hasattr(client, "start_image_to_video")):
                res = client.start_image_to_video(
                    model=model,
                    prompt=bundle.prompt,
                    negative_prompt=negative,
                    image_path=first,
                    gcs_output_prefix=gcs,
                    resolution=bundle.resolution,
                    duration_seconds=duration,
                    seed=payload.seed,
                )
                return run_coro_sync(res) if hasattr(res, "__await__") else res
            instance: dict[str, Any] = {"prompt": bundle.prompt}
            if first is not None:
                instance["image"] = {"bytesBase64Encoded": b64_file(first), "mimeType": image_mime(first)}
            if last is not None:
                instance["lastFrame"] = {"bytesBase64Encoded": b64_file(last), "mimeType": image_mime(last)}
            if refs:
                instance["referenceImages"] = [
                    {
                        "image": {"bytesBase64Encoded": b64_file(r), "mimeType": image_mime(r)},
                        "referenceType": "asset",
                    }
                    for r in refs
                ]
            params: dict[str, Any] = {
                "aspectRatio": bundle.aspect_ratio,
                "durationSeconds": duration,
                "sampleCount": 1,
                "generateAudio": True,
                "personGeneration": "allow_adult",
            }
            if negative:
                params["negativePrompt"] = negative
            if bundle.resolution:
                params["resolution"] = bundle.resolution
            if gcs:
                params["storageUri"] = gcs.rstrip("/") + "/"
            if payload.seed is not None:
                params["seed"] = int(payload.seed)
            url = f"{client.models_base}/{model}:predictLongRunning"
            post = client._post(url, {"instances": [instance], "parameters": params})  # noqa: SLF001
            out = run_coro_sync(post) if hasattr(post, "__await__") else post
            op_name = str((out or {}).get("name", ""))
            if not op_name:
                raise AdapterError(f"{model}: predictLongRunning returned no operation name")
            return op_name, params

        op_name, params_used = self.call_with_auth(_invoke)
        if not op_name:
            raise AdapterError(f"{model}: predictLongRunning returned an empty operation name")
        self._op_models[op_name] = model
        self.last_result = {"provider_model": model, "attempt": attempt, "params": params_used, "gcs_output_prefix": gcs}
        return str(op_name)

    def _model_for_op(self, operation_name: str) -> str:
        if operation_name in self._op_models:
            return self._op_models[operation_name]
        if "/models/" in operation_name:
            tail = operation_name.split("/models/", 1)[1]
            model = tail.split("/", 1)[0].split(":", 1)[0]
            if model:
                return model
        return self.provider_model()

    def poll(self, operation_name: str) -> PollResult:
        client = self.veo_client()
        model = self._model_for_op(operation_name)

        def _fetch() -> dict[str, Any]:
            res = client.fetch_operation(model, operation_name)
            return run_coro_sync(res) if hasattr(res, "__await__") else dict(res)

        try:
            op = self.call_with_auth(_fetch)
        except AuthError:
            raise  # an expired / blocked sign-in is not transient: stop polling and tell the human
        except Exception as exc:  # noqa: BLE001 - transient poll error
            return self.poll_failed(operation_name, exc)
        self.poll_ok(operation_name)
        raw: dict[str, Any] = {"operation_name": operation_name, "done": bool(op.get("done"))}
        if not op.get("done"):
            return PollResult(status=DispatchStatus.RUNNING, raw=raw)
        if op.get("error"):
            err = op["error"]
            msg = short_error(err.get("message") if isinstance(err, dict) else err, 400)
            status = DispatchStatus.FILTERED if _looks_filtered(msg) else DispatchStatus.FAILED
            return PollResult(status=status, raw=raw, error={"stage": "poll", "message": msg})
        resp = op.get("response") or {}
        videos = resp.get("videos") or resp.get("generatedSamples") or []
        filtered_count = int(resp.get("raiMediaFilteredCount", 0) or 0)
        reasons = [str(r) for r in (resp.get("raiMediaFilteredReasons") or [])]
        if videos:
            first = videos[0] if isinstance(videos[0], dict) else {}
            inner = first.get("video") if isinstance(first.get("video"), dict) else first
            uri = inner.get("gcsUri") or inner.get("uri") or first.get("gcsUri") or first.get("uri")
            b64 = inner.get("bytesBase64Encoded") or first.get("bytesBase64Encoded")
            if uri and str(uri).startswith("gs://"):
                return PollResult(status=DispatchStatus.COMPLETED, video_uri=str(uri), raw=raw)
            if b64:
                raw["inline_data"] = b64 if isinstance(b64, str) else base64.b64encode(bytes(b64)).decode("ascii")
                return PollResult(status=DispatchStatus.COMPLETED, raw=raw)
        if filtered_count > 0 or reasons:
            return PollResult(
                status=DispatchStatus.FILTERED,
                raw=raw,
                error={
                    "stage": "poll",
                    "message": f"RAI filtered ({filtered_count}): {', '.join(reasons) or 'policy'}",
                    "rai_media_filtered_count": filtered_count,
                    "reasons": reasons,
                },
            )
        return PollResult(
            status=DispatchStatus.FAILED,
            raw=raw,
            error={"stage": "poll", "message": "operation completed without videos or RAI reasons"},
        )


def create_adapter(spec: ModelSpec, **kw: Any) -> VideoAdapter:
    """Registry factory: :class:`FakeVideoAdapter` in fake mode, else :class:`VeoAdapter`."""
    return create_video_adapter(spec, VeoAdapter, **kw)
