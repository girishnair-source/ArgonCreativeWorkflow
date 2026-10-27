"""Model Capability Registry loader (BUILD_SPEC_v3 §5).

``registry.yaml`` is data, not code: max durations, speaker limits, feature support, prompt syntax and cost live
there. The registry is hot-reloadable (mtime check on every access) and adapter modules are only importable
from the ``agy_studio.adapters.`` namespace (no arbitrary code loading from YAML).
"""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

from pydantic import Field

from .fsio import read_yaml
from .models import Model, ModelKind

DEFAULT_REGISTRY_PATH = Path(__file__).with_name("registry.yaml")
_ADAPTER_NAMESPACE = "agy_studio.adapters."


class ModelSupports(Model):
    reference_to_video: bool = False
    first_frame: bool = False
    last_frame: bool = False
    multi_subject_refs: bool = False
    negative_prompt: bool = False
    native_audio: bool = False
    dialogue_in_prompt: bool = False
    voice_reference_audio: bool = False
    max_speakers_per_clip: int = 0
    max_reference_images: int = 0
    max_reference_videos: int = 0
    extend_video: bool = False
    refs_with_first_frame: bool = False
    two_tier_draft_finalize: bool = False
    reuse_prompt_on_finalize: bool = False


class ModelSpec(Model):
    id: str
    kind: ModelKind
    enabled: bool = True
    provider_model: str
    provider_fallbacks: list[str] = Field(default_factory=list)
    location: str | None = None
    adapter: str
    min_duration_s: float | None = None
    max_duration_s: float | None = None
    allowed_durations_s: list[float] | None = None
    aspect_ratios: list[str] = Field(default_factory=list)
    resolutions: list[str] = Field(default_factory=list)
    draft_resolution: str | None = None
    finalize_resolution: str | None = None
    reuse_prompt_on_finalize: bool = False
    native_voice_model: str | None = None
    supports: ModelSupports = Field(default_factory=ModelSupports)
    prompt_style: str = "prose"
    audio_prompt_syntax: dict[str, str] = Field(default_factory=dict)
    cost_per_s: float = 0.0
    cost_per_call: float = 0.0
    cost_is_estimate: bool = False
    avg_latency_s: float = 0.0
    known_failures: list[str] = Field(default_factory=list)
    notes: str = ""

    def supports_feature(self, feature: str) -> bool:
        value = getattr(self.supports, feature, None)
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value > 0
        return False

    def snap_duration(self, seconds: float) -> float:
        """Smallest allowed duration >= ``seconds`` (or the max if none fits)."""
        if self.allowed_durations_s:
            for d in sorted(self.allowed_durations_s):
                if d + 1e-6 >= seconds:
                    return float(d)
            return float(max(self.allowed_durations_s))
        lo = self.min_duration_s or 0.0
        hi = self.max_duration_s or seconds
        return float(min(max(seconds, lo), hi))

    def estimate_cost(self, duration_s: float = 0.0, calls: int = 1) -> float:
        return round(self.cost_per_s * duration_s * calls + self.cost_per_call * calls, 4)


class RegistryFile(Model):
    version: int = 1
    models: list[ModelSpec]


class UnknownModelError(KeyError):
    pass


_MODEL_ALIASES: dict[str, str] = {
    "omni_1_5_flash": "omni",
    "gemini_tts_3_8": "omni",
}


class Registry:
    """Hot-reloadable view over ``registry.yaml``."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_REGISTRY_PATH
        self._lock = threading.Lock()
        self._mtime: float | None = None
        self._models: dict[str, ModelSpec] = {}
        self.reload(force=True)

    # -- loading -----------------------------------------------------------------------------
    def reload(self, force: bool = False) -> bool:
        """Reload if the file changed (or ``force``). Returns True when a reload happened."""
        mtime = self.path.stat().st_mtime
        with self._lock:
            if not force and self._mtime == mtime:
                return False
            parsed = RegistryFile.model_validate(read_yaml(self.path) or {"models": []})
            ids = [m.id for m in parsed.models]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate model ids in {self.path}: {ids}")
            for m in parsed.models:
                if not m.adapter.startswith(_ADAPTER_NAMESPACE):
                    raise ValueError(f"model {m.id}: adapter {m.adapter!r} must live under {_ADAPTER_NAMESPACE}")
            self._models = {m.id: m for m in parsed.models}
            self._mtime = mtime
            return True

    def _fresh(self) -> dict[str, ModelSpec]:
        self.reload()
        return self._models

    # -- queries -----------------------------------------------------------------------------
    def get(self, model_id: str) -> ModelSpec:
        models = self._fresh()
        resolved = _MODEL_ALIASES.get(model_id, model_id)
        if resolved not in models:
            raise UnknownModelError(f"model {model_id!r} not in registry ({sorted(models)})")
        return models[resolved]

    def has(self, model_id: str) -> bool:
        models = self._fresh()
        return model_id in models or _MODEL_ALIASES.get(model_id) in models

    def all(self) -> list[ModelSpec]:
        return list(self._fresh().values())

    def of_kind(self, kind: ModelKind | str, enabled_only: bool = True) -> list[ModelSpec]:
        kind = ModelKind(kind)
        return [m for m in self._fresh().values() if m.kind == kind and (m.enabled or not enabled_only)]

    def supports(self, model_id: str, feature: str) -> bool:
        return self.get(model_id).supports_feature(feature)

    def max_duration_s(self, model_id: str) -> float:
        value = self.get(model_id).max_duration_s
        if value is None:
            raise ValueError(f"model {model_id} has no max_duration_s")
        return float(value)

    def max_speakers_per_clip(self, model_id: str) -> int:
        return int(self.get(model_id).supports.max_speakers_per_clip)

    def video_models(self) -> list[ModelSpec]:
        return self.of_kind(ModelKind.VIDEO_AV)

    def first_of_kind(self, kind: ModelKind | str) -> ModelSpec:
        models = self.of_kind(kind)
        if not models:
            raise UnknownModelError(f"no enabled model of kind {kind}")
        return models[0]

    # -- adapters ----------------------------------------------------------------------------
    def adapter_module(self, model_id: str) -> ModuleType:
        dotted = self.get(model_id).adapter
        if not dotted.startswith(_ADAPTER_NAMESPACE):  # defence in depth (also checked on load)
            raise ValueError(f"refusing to import adapter outside {_ADAPTER_NAMESPACE}: {dotted}")
        return importlib.import_module(dotted)

    def adapter(self, model_id: str, **kwargs: Any) -> Any:
        """Instantiate the adapter for ``model_id`` via its module-level ``create_adapter(spec, **kw)``."""
        module = self.adapter_module(model_id)
        factory = getattr(module, "create_adapter", None)
        if factory is None:
            raise AttributeError(f"{module.__name__} has no create_adapter(spec, **kwargs)")
        return factory(self.get(model_id), **kwargs)


_DEFAULT: Registry | None = None


def default_registry() -> Registry:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Registry()
    return _DEFAULT
