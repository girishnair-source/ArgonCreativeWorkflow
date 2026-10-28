"""Adapter base layer (BUILD_SPEC_v3 §5 adapter interface; AGY_INTERFACES §4).

* :func:`resolve_fake` — the deterministic local backend is the unit-test double ONLY: ``fake=True`` is honoured
  under pytest and raises :class:`FakeModeUnavailable` anywhere else. There is no environment switch and no offline
  production mode.
* :class:`VideoAdapter` — the native audio+video interface used by the Shot Generator:
  ``build_payload`` / ``payload_from_bundle`` (content-addressed ``payload_hash``), ``submit`` / ``poll`` /
  ``download`` (provider specific), and the crash-safe ``dispatch`` / ``resume`` loops. ``dispatch`` persists the
  provider operation name through ``on_update`` BEFORE polling, so a killed process can always ``resume`` the
  in-flight operation instead of submitting the same work twice (acceptance criterion 9).
* :class:`LegacyVideoAdapter` — shared plumbing for the real Omni / Veo wrappers around ``vertex_veo_director``
  (lazy import, cached gcloud token with forced refresh on HTTP 401, bounded transient poll errors, GCS / inline
  downloads into the take folder).
* Shared helpers for the real adapters: staged immutable writes, sync wrappers for the legacy async clients,
  Gemini-on-Vertex JSON calls and secret-scrubbed error messages.

Provider SDKs and the legacy generators are imported lazily, only on real (non-fake) code paths.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import json
import os
import re
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic import Field

from .. import continuity, fsio, media
from ..models import (
    Conditioning,
    DispatchStatus,
    GenClip,
    GenerationPayload,
    Model,
    PromptBundle,
    ShotSpec,
    Take,
)
from ..registry import ModelSpec
from . import auth
from .auth import AuthError, is_auth_failure, raise_if_auth_failure

_TRUE = frozenset({"1", "true", "yes", "on"})

TERMINAL_STATUSES = frozenset({DispatchStatus.COMPLETED, DispatchStatus.FAILED, DispatchStatus.FILTERED})
IN_FLIGHT_STATUSES = frozenset({DispatchStatus.SUBMITTED, DispatchStatus.RUNNING})
DISPATCH_LOG_NAME = "dispatch.jsonl"
TAKE_VIDEO_NAME = "take.mp4"
DIALOGUE_WAV_NAME = "dialogue.wav"
PAYLOAD_NAME = "payload.json"


class FakeModeUnavailable(RuntimeError):
    """The deterministic test backend was requested outside the test-suite. There is no offline generation mode."""


def _under_pytest() -> bool:
    import sys  # noqa: PLC0415

    return "pytest" in sys.modules and bool(os.environ.get("PYTEST_CURRENT_TEST") or "pytest" in sys.argv[0])


def fake_mode() -> bool:
    """Always False: there is NO environment switch for offline generation (fail fast is the only mode).

    Kept so callers that defer to "the environment" keep working; the deterministic backend exists purely as the
    unit-test double and can only be requested programmatically (``fake=True``) from inside pytest.
    """
    return False


def resolve_fake(fake: bool | None) -> bool:
    """``fake=True`` is honoured ONLY under pytest; anywhere else it raises :class:`FakeModeUnavailable`."""
    if not fake:
        return False
    if not _under_pytest():
        raise FakeModeUnavailable(
            "Offline / fake generation is not available: AGY Studio only produces real Vertex AI output. "
            "If credentials are the problem, fix them (gcloud auth login) — do not look for a fallback.")
    return True


class AdapterError(RuntimeError):
    """Provider / adapter failure. Messages are scrubbed of credentials before they are raised or logged."""


class UnsupportedFeatureError(AdapterError):
    """The bundle asks for something ``registry.yaml`` says this model cannot do (linter L12 decides fallbacks)."""


class PollResult(Model):
    """One provider poll. ``raw`` carries provider details (e.g. ``operation_name``, inline video data)."""

    status: DispatchStatus
    video_uri: str | None = None
    error: dict[str, Any] | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------------------------
_SECRET_PATTERNS = (
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=\-]+"), "Bearer ***"),
    (re.compile(r"ya29\.[A-Za-z0-9._\-]+"), "ya29.***"),
    (re.compile(r"(?i)(api[_-]?key|access_token|token)=([^&\s]+)"), r"\1=***"),
)


def scrub(text: str) -> str:
    for pattern, repl in _SECRET_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def short_error(exc: Any, limit: int = 300) -> str:
    """One-line, credential-free description of an exception (or any object)."""
    text = f"{type(exc).__name__}: {exc}" if isinstance(exc, BaseException) else str(exc)
    return scrub(" ".join(text.split()))[:limit]


def exc_status(exc: BaseException) -> int | None:
    """Best-effort HTTP status of a provider exception (VertexError.status, genai errors' .code, ...)."""
    for attr in ("status", "code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value
    text = str(exc)
    if "UNAUTHENTICATED" in text or re.search(r"\b401\b", text):
        return 401
    return None


def run_coro_sync(coro: Any) -> Any:
    """Run a coroutine from sync code (works whether or not an event loop is already running)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def staged_write(dest: Path, produce: Callable[[Path], Any]) -> Path:
    """Produce a file in a private temp dir, then publish it to ``dest`` with ``fsio.write_new``.

    Media in take folders / ``library/`` are immutable, so a partially written provider download or ffmpeg output
    never appears at ``dest`` (and an existing ``dest`` is never overwritten).
    """
    dest = Path(dest)
    if dest.exists():
        raise fsio.ImmutableArtifactError(f"refusing to overwrite immutable artifact {dest}")
    with tempfile.TemporaryDirectory(prefix="agy_stage_") as tmp:
        staging = Path(tmp) / dest.name
        result = produce(staging)
        produced = Path(result) if isinstance(result, (str, Path)) else staging
        if not produced.is_file() or produced.stat().st_size == 0:
            raise AdapterError(f"no output was produced for {dest.name}")
        fsio.write_new(dest, produced.read_bytes())
    return dest


def write_inline_bytes(data: str | bytes | bytearray, dest: Path) -> Path:
    raw = base64.b64decode(data) if isinstance(data, str) else bytes(data)
    if not raw:
        raise AdapterError(f"provider returned empty inline data for {Path(dest).name}")
    return fsio.write_new(Path(dest), raw)


def image_mime(path: Path) -> str:
    suffix = Path(path).suffix.lower()
    return {".png": "image/png", ".webp": "image/webp"}.get(suffix, "image/jpeg")


def b64_file(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
AUDIO_EXTS = frozenset({".wav"})


def write_pcm_wav(path: Path, samples: Any, sample_rate: int) -> Path:
    """16-bit mono PCM WAV from a float array in [-1, 1] (pure Python/numpy, no ffmpeg)."""
    import wave  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    data = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = (data * 32767.0).astype("<i2")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return path


def resolve_media(project: Any, item: str | Path, *, exts: frozenset[str] | None = None, what: str = "media") -> Path:
    """Absolute path of an input file given as a project-relative POSIX path or an absolute path.

    With a project, absolute paths must stay inside the project root (user files are copied INTO the project first,
    never referenced in place). The file must exist, be a regular file and carry an allowed extension.
    """
    raw = Path(str(item))
    if project is not None:
        path = project.abs(project.rel(raw)) if raw.is_absolute() else project.abs(str(item))
    else:
        path = raw.resolve()
    if not path.is_file():
        raise AdapterError(f"{what} not found: {item}")
    if exts is not None and path.suffix.lower() not in exts:
        raise AdapterError(f"{what} {path.name}: extension not in {sorted(exts)}")
    return path


def fold_negatives(prompt: str, negative: str | None) -> str:
    """Serialise negatives into the prompt for models without a negative-prompt field (e.g. Omni)."""
    negative = (negative or "").strip()
    return f"{prompt.rstrip()}\n\nAvoid: {negative}" if negative else prompt


_CACHED_GCLOUD_PROJECT: str | None = None

# gcloud / ffmpeg live outside the PATH of GUI-launched processes (Antigravity, Resolve); auth.py knows every usual
# install dir (Homebrew bin, Homebrew's google-cloud-sdk/bin, Caskroom, ~/google-cloud-sdk, snap, Windows ...).
auth.ensure_tool_dirs_on_path()


def vertex_project_id() -> str:
    """Vertex project from ``VERTEX_PROJECT_ID`` / ``GOOGLE_CLOUD_PROJECT`` -> active gcloud config.

    There is deliberately NO hidden default project: an unset project raises :class:`AuthError` with the command
    that fixes it instead of billing a stranger's project and failing with a 403 minutes later.
    """
    global _CACHED_GCLOUD_PROJECT  # noqa: PLW0603
    for key in auth.PROJECT_ENV_KEYS:
        if os.environ.get(key, "").strip():
            return os.environ[key].strip()
    if _CACHED_GCLOUD_PROJECT:
        return _CACHED_GCLOUD_PROJECT
    _CACHED_GCLOUD_PROJECT = auth.vertex_project()
    return _CACHED_GCLOUD_PROJECT


def gemini_client(location: str | None = None, *, use_gcloud_token: bool = False) -> Any:
    """google-genai client on Vertex AI.

    ``use_gcloud_token=True`` authenticates with a fresh ``gcloud auth print-access-token`` (the same credential the
    video adapters use); otherwise Application Default Credentials. A broken gcloud sign-in raises
    :class:`AuthError` immediately — it is never swallowed.
    """
    try:
        import certifi  # noqa: PLC0415

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:  # pragma: no cover - certifi ships with httpx
        pass
    from google import genai  # noqa: PLC0415 - real path only

    project = vertex_project_id()
    if use_gcloud_token:
        from google.oauth2.credentials import Credentials  # noqa: PLC0415

        tok = auth.gcloud_access_token(auth.preferred_account())
        creds = Credentials(token=tok, quota_project_id=project)
        return genai.Client(vertexai=True, project=project, location=location or "global", credentials=creds)
    return genai.Client(vertexai=True, project=project, location=location or "global")


def parse_json_text(text: str | None) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw or "{}")
    return data if isinstance(data, dict) else {"items": data}


def gemini_generate_json(models: Sequence[str], location: str | None, system: str | None, contents: list[Any],
                         schema: dict[str, Any] | None = None, temperature: float = 0.2,
                         client: Any = None) -> dict[str, Any]:
    """Structured JSON generation with provider fallbacks (first model that answers wins).

    Credential / permission failures (gcloud broken, HTTP 401/403) are raised as :class:`AuthError` at once — the next
    model would fail identically, so there is nothing to gain from trying it.
    """
    from google.genai import types  # noqa: PLC0415 - real path only

    client = client or gemini_client(location)
    cfg: dict[str, Any] = {"response_mime_type": "application/json", "temperature": float(temperature)}
    if system:
        cfg["system_instruction"] = system
    if schema:
        cfg["response_json_schema"] = schema
    errors: list[str] = []
    for model in models:
        try:
            resp = client.models.generate_content(model=model, contents=contents,
                                                  config=types.GenerateContentConfig(**cfg))
            return parse_json_text(getattr(resp, "text", None))
        except AuthError:
            raise
        except Exception as exc:  # noqa: BLE001 - ADC 401/403 -> retry once with the gcloud token, else next model
            msg = short_error(exc)
            if is_auth_failure(exc):
                alt_client = gemini_client(location, use_gcloud_token=True)  # raises AuthError if gcloud is broken
                try:
                    resp = alt_client.models.generate_content(model=model, contents=contents,
                                                              config=types.GenerateContentConfig(**cfg))
                    return parse_json_text(getattr(resp, "text", None))
                except Exception as exc2:  # noqa: BLE001
                    raise_if_auth_failure(exc2, context=f"calling {model}")
                    msg = short_error(exc2)
            errors.append(f"{model}: {msg}")
    raise AdapterError("all provider models failed: " + "; ".join(errors))


def gemini_part_from_file(path: Path, mime_type: str) -> Any:
    from google.genai import types  # noqa: PLC0415 - real path only

    return types.Part.from_bytes(data=Path(path).read_bytes(), mime_type=mime_type)


def extract_dialogue_wav(project: Any, take: Take) -> Path | None:
    """``<take_dir>/dialogue.wav`` (16 kHz mono PCM) from the take's native audio; reused when it already exists.

    Returns None when the take has no video or the video has no audio stream. Written via a staged immutable write.
    """
    if not take.video_uri:
        return None
    out = project.take_dir(take.id) / DIALOGUE_WAV_NAME
    if out.is_file():
        return out
    video = project.abs(take.video_uri)
    if not video.is_file() or not media.has_audio(video):
        return None
    return staged_write(out, lambda staging: media.extract_wav(video, staging, sample_rate=16000, mono=True))


# ---------------------------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------------------------
class BaseAdapter:
    """Common constructor for every adapter: ``(spec, *, project=None, fake=None, client=None, registry=None)``."""

    agent_name = "agy.adapter"

    def __init__(self, spec: ModelSpec, *, project: Any = None, fake: bool | None = None, client: Any = None,
                 registry: Any = None) -> None:
        self.spec = spec
        self.project = project
        self.fake = resolve_fake(fake)
        self.client = client
        self.registry = registry
        self.last_result: dict[str, Any] = {}

    @property
    def model_id(self) -> str:
        return self.spec.id

    def supports(self, feature: str) -> bool:
        return self.spec.supports_feature(feature)

    def provider_model(self) -> str:
        """Concrete provider model: ``project.config.model_pins`` override, else the registry value."""
        pins = self.project.config.model_pins if self.project is not None else {}
        return pins.get(self.spec.id) or self.spec.provider_model

    def provider_fallbacks(self) -> list[str]:
        """Ordered provider fallbacks: ``project.settings.provider_fallbacks[<id>]`` when set (an empty list means
        "no provider fallback"), else the registry chain."""
        if self.project is not None:
            overrides = getattr(self.project.config.settings, "provider_fallbacks", None) or {}
            if self.spec.id in overrides:
                return [str(m) for m in overrides[self.spec.id]]
        return list(self.spec.provider_fallbacks)

    def model_chain(self) -> list[str]:
        chain = [self.provider_model()]
        for model in self.provider_fallbacks():
            if model not in chain:
                chain.append(model)
        return chain

    def _require_project(self) -> Any:
        if self.project is None:
            raise AdapterError(f"{type(self).__name__} needs project= for this operation")
        return self.project

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.spec.id} fake={self.fake}>"


def check_kind(spec: ModelSpec, *kinds: str) -> None:
    """Factories refuse registry entries of the wrong kind (a typo in registry.yaml fails loudly)."""
    if spec.kind.value not in kinds:
        raise AdapterError(f"model {spec.id}: adapter expects kind {' / '.join(kinds)}, registry says {spec.kind.value}")


_BASE_KW = ("project", "fake", "client", "registry")


def adapter_kwargs(kw: dict[str, Any]) -> dict[str, Any]:
    """The constructor kwargs every non-video adapter accepts.

    ``Registry.adapter(id, **kw)`` is called generically (e.g. for every model id with the same kwargs), so keys that
    only video adapters understand (``sleep``, ``token`` ...) are ignored here instead of raising.
    """
    return {k: kw[k] for k in _BASE_KW if k in kw}


_SAFETY_LEVEL1_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\bArvind Joshi\b", re.IGNORECASE), "Arvind", "public_figure_name->first_name_arvind"),
    (re.compile(r"\bRohit Sharma\b", re.IGNORECASE), "Rohit", "public_figure_name->first_name_rohit"),
    (re.compile(r"\bSuresh Kumar\b", re.IGNORECASE), "Suresh", "public_figure_name->first_name_suresh"),
    (re.compile(r"\bcrime journalist\b|\btrue-?crime\b", re.IGNORECASE), "documentary storyteller", "crime_journalist->documentary_storyteller"),
    (re.compile(r"\bdead on arrival\b", re.IGNORECASE), "critical condition upon arrival", "dead_on_arrival->critical_condition"),
    (re.compile(r"\b(?:dead|death|dying|died)\b", re.IGNORECASE), "still and silent", "mortality->still_and_silent"),
    (re.compile(r"\b(?:corpses?|laash|lash)\b|लाश", re.IGNORECASE), "motionless silhouette in dim shadow", "corpse->motionless_silhouette"),
    (re.compile(r"\b(?:murder\w*|homicid\w*|kill\w*|qaatil|qatil|hatya)\b|क़ातिल|कातिल|हत्या|क़त्ल|कत्ल", re.IGNORECASE), "dark family betrayal", "murder->dark_family_betrayal"),
    (re.compile(r"\b(?:suicid\w*|khudkushi|aatmhatya)\b|खुदकुशी|आत्महत्या", re.IGNORECASE), "tragic disappearance", "suicide->tragic_disappearance"),
    (re.compile(r"\b(?:poison\w*|pesticid\w*|organophosphate|zahar|zeher|toxic\w*)\b|ज़हर|जहर|कीटनाशक", re.IGNORECASE), "bitter herbal liquid in a steel tumbler", "poison->bitter_herbal_liquid"),
    (re.compile(r"\b(?:unconscious|behosh)\b|बेहोश", re.IGNORECASE), "deeply asleep in dim morning light", "unconscious->deeply_asleep"),
    (re.compile(r"\b(?:blue lips|honth neele)\b|होंठ नीले|नीले होंठ", re.IGNORECASE), "pale face in cold dawn light", "blue_lips->pale_face"),
    (re.compile(r"\b(?:cold hands|haath thande)\b|हाथ ठंडे|ठंडे हाथ", re.IGNORECASE), "still hands with mehndi patterns resting on the bedsheet", "cold_hands->still_mehndi_hands"),
    (re.compile(r"\b(?:blood\w*|bleed\w*|khoon)\b|खून", re.IGNORECASE), "dark crimson stain", "blood->crimson_stain"),
    (re.compile(r"\b(?:needle\w*|prick\w*|puncture\w*|stab\w*|slash\w*|wound\w*|cut\w*)\b", re.IGNORECASE), "small metallic pin sensor", "needle->metallic_pin_sensor"),
    (re.compile(r"\b(?:strangl\w*|chok\w*)\b", re.IGNORECASE), "overwhelming pressure", "strangle->pressure"),
    (re.compile(r"\b(?:marta hai|mar gayi|maari)\b|मरता है|मर गई|मारी", re.IGNORECASE), "खो जाता है", "hindi_mortality->kho_jaata_hai"),
)

_SAFETY_LEVEL2_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\b(?:scream\w*|shriek\w*|cheekh)\b|चीख", re.IGNORECASE), "urgent voice calling out in the courtyard", "scream->urgent_call"),
    (re.compile(r"\b(?:crime scene|forensic\w*|post-?mortem|autopsy)\b|पोस्टमार्टम", re.IGNORECASE), "official case file on the wooden desk", "forensics->case_file"),
    (re.compile(r"\b(?:suspect\w*|interrogat\w*|arrest\w*|giraftaar)\b|गिरफ़्तार|पूछताछ", re.IGNORECASE), "serious conversation in a dimly lit office", "interrogation->serious_conversation"),
    (re.compile(r"\b(?:victim\w*|traged\w*|tragic|trauma\w*)\b", re.IGNORECASE), "somber emotional moment", "tragedy->somber_moment"),
)

_SAFETY_ERR_WORDS = ("safety", "rai", "responsible ai", "blocked", "prohibited", "policy", "filtered", "sensitive", "harm", "violat")


def is_safety_error(text: str | None) -> bool:
    low = (text or "").lower()
    return any(w in low for w in _SAFETY_ERR_WORDS)


def _split_negative_tail(prompt: str) -> tuple[str, str]:
    """Split ``prompt`` into ``(positive_body, negative_tail)`` so negative lists are not rewritten."""
    for marker in ("\n\nAvoid:", "\nNEGATIVE:", "\nNegative:"):
        idx = prompt.find(marker)
        if idx != -1:
            return prompt[:idx], prompt[idx:]
    return prompt, ""


def safety_reprompt_text(prompt: str, level: int = 1) -> tuple[str, list[str]]:
    """Rewrite sensitive mortality/crime/poisoning phrases into safe atmospheric Bollywood cinema descriptions."""
    body, neg_tail = _split_negative_tail(prompt)
    out = body
    applied: list[str] = []
    rules = list(_SAFETY_LEVEL1_RULES)
    if level >= 2:
        rules.extend(_SAFETY_LEVEL2_RULES)
    for rx, replacement, tag in rules:
        if rx.search(out):
            out = rx.sub(replacement, out)
            applied.append(tag)
    if level >= 2:
        suffix = " Safe PG-13 theatrical Indian family drama cinematography, focusing on lighting, shadow, and subtle facial emotion."
        if suffix.strip() not in out:
            out = out.rstrip() + suffix
            applied.append("level2_pg13_framing_suffix")
    elif not applied and level >= 1:
        suffix = " Atmospheric Indian dramatic cinema framing with gentle lighting and natural emotional expressions."
        out = out.rstrip() + suffix
        applied.append("level1_atmospheric_framing_suffix")
    return out + neg_tail, applied


def safety_reprompt_payload(payload: GenerationPayload, level: int = 1) -> tuple[GenerationPayload, list[str]]:
    """Return a copy of ``payload`` with its prompt sanitized for safety retry level ``level`` (1 or 2)."""
    new_prompt, applied = safety_reprompt_text(payload.bundle.prompt, level=level)
    new_bundle = payload.bundle.model_copy(update={"prompt": new_prompt}, deep=True)
    new_payload = payload.model_copy(
        update={
            "bundle": new_bundle,
            "extra": {
                **dict(payload.extra or {}),
                "safety_reprompt_level": level,
                "safety_rewrites": applied,
            },
        },
        deep=True,
    )
    return new_payload, applied


class VideoAdapter(BaseAdapter, ABC):
    """Native audio+video generator (registry kind ``video_av``)."""

    agent_name = "agy.adapter.video"
    max_rai_retries = 3  # 1 original + up to 2 progressive safety re-prompt retries per provider model
    default_poll_interval_s = 15.0
    default_timeout_s = 1800.0

    def __init__(self, spec: ModelSpec, *, project: Any = None, fake: bool | None = None, client: Any = None,
                 registry: Any = None, sleep: Callable[[float], None] | None = None) -> None:
        super().__init__(spec, project=project, fake=fake, client=client, registry=registry)
        self._sleep = sleep or time.sleep

    # -- payloads --------------------------------------------------------------------------
    def _bundle_for(self, clip: GenClip, model_id: str) -> PromptBundle:
        """Seam: the Prompt Engineer (workstream B) serialises the clip. Monkeypatched in tests."""
        try:
            from .. import prompt_engineer  # noqa: PLC0415 - lazy: owned by another workstream
        except ImportError as exc:
            raise AdapterError("agy_studio.prompt_engineer is not available; cannot build a prompt bundle") from exc
        registry = self.registry
        if registry is None:
            from ..registry import default_registry  # noqa: PLC0415

            registry = default_registry()
        return prompt_engineer.build_prompt_bundle(self._require_project(), registry, clip, model_id=model_id)

    def build_payload(self, shot_spec: ShotSpec | None, conditioning: Conditioning | None, *, clip: GenClip,
                      seed: int | None = None) -> GenerationPayload:
        """Bundle the clip (with the given spec/conditioning) for this model and hash it."""
        update: dict[str, Any] = {}
        if shot_spec is not None and shot_spec != clip.spec:
            update["spec"] = shot_spec
        if conditioning is not None and conditioning != clip.conditioning:
            update["conditioning"] = conditioning
        if update:
            clip = clip.model_copy(update=update, deep=True)
        bundle = self._bundle_for(clip, self.spec.id)
        if bundle.clip_id is None:
            bundle = bundle.model_copy(update={"clip_id": clip.id})
        extra = {
            "clip_id": clip.id,
            "scene_id": clip.scene_id,
            "speakers": continuity.speakers_in_clip(clip),
            "expected_text": " ".join(continuity.clip_line_texts(clip)),
        }
        return self.payload_from_bundle(bundle, seed=seed, extra=extra)

    def payload_from_bundle(self, bundle: PromptBundle, *, seed: int | None = None,
                            extra: dict[str, Any] | None = None) -> GenerationPayload:
        """Pure: validate against the registry and compute the content-addressed ``payload_hash``.

        ``extra`` is informational (speakers, expected text) and is NOT part of the hash.
        """
        project = self._require_project()
        if bundle.model_id != self.spec.id:
            raise AdapterError(f"bundle was built for model {bundle.model_id!r}, not {self.spec.id!r}")
        self.check_bundle(bundle)
        provider_model = self.provider_model()
        payload_hash = fsio.compute_payload_hash(project.root, model_id=self.spec.id, provider_model=provider_model,
                                                 bundle=bundle, seed=seed)
        return GenerationPayload(model_id=self.spec.id, provider_model=provider_model, bundle=bundle, seed=seed,
                                 payload_hash=payload_hash, extra=dict(extra or {}))

    def check_bundle(self, bundle: PromptBundle) -> None:
        """Raise :class:`UnsupportedFeatureError` when the registry says this model cannot honour the bundle."""
        spec, sup = self.spec, self.spec.supports
        problems: list[str] = []
        if spec.max_duration_s is not None and bundle.duration_s > float(spec.max_duration_s) + 1e-6:
            problems.append(f"duration {bundle.duration_s}s > max_duration_s {spec.max_duration_s}s")
        if spec.aspect_ratios and bundle.aspect_ratio not in spec.aspect_ratios:
            problems.append(f"aspect ratio {bundle.aspect_ratio} not in {spec.aspect_ratios}")
        if spec.resolutions and bundle.resolution not in spec.resolutions:
            problems.append(f"resolution {bundle.resolution} not in {spec.resolutions}")
        if bundle.first_frame and not sup.first_frame:
            problems.append("first-frame conditioning is not supported")
        if bundle.last_frame and not sup.last_frame:
            problems.append("last-frame conditioning is not supported")
        if bundle.reference_images:
            if not sup.reference_to_video:
                problems.append("reference images are not supported")
            if len(bundle.reference_images) > sup.max_reference_images:
                problems.append(f"{len(bundle.reference_images)} reference images > max_reference_images "
                                f"{sup.max_reference_images}")
            if bundle.first_frame and not sup.refs_with_first_frame:
                problems.append("a first-frame image cannot be combined with asset reference images "
                                "(registry refs_with_first_frame: false) - re-lint so L12 picks a fallback")
        if bundle.voice_reference_audio and not sup.voice_reference_audio:
            problems.append("voice reference audio is not supported")
        if problems:
            raise UnsupportedFeatureError(f"{spec.id}: " + "; ".join(problems))

    def provider_chain(self, payload: GenerationPayload) -> list[str]:
        chain = [payload.provider_model]
        for model in self.provider_fallbacks():
            if model not in chain:
                chain.append(model)
        return chain

    def media_path(self, rel: str | None) -> Path | None:
        """Absolute path of a project-relative media reference (must exist)."""
        if not rel:
            return None
        path = self._require_project().abs(rel)
        if not path.is_file():
            raise AdapterError(f"conditioning media not found: {rel}")
        return path

    # -- provider primitives ----------------------------------------------------------------
    @abstractmethod
    def submit(self, payload: GenerationPayload, *, provider_model: str | None = None, attempt: int = 0) -> str:
        """Start one provider job; returns the operation / interaction name."""

    @abstractmethod
    def poll(self, operation_name: str) -> PollResult:
        """One non-blocking status check of a provider job."""

    @abstractmethod
    def download(self, result: PollResult, dest: Path) -> Path:
        """Write the finished video to ``dest`` (immutable; never overwrites)."""

    def _log_safety_reprompt(self, take: Take, *, provider_model: str, level: int, reason: str,
                             old_prompt: str, new_prompt: str, rewrites: list[str]) -> None:
        self.log_event(
            take,
            "safety_reprompt",
            provider_model=provider_model,
            level=level,
            reason=reason,
            rewrites=",".join(rewrites),
        )
        if self.project is not None:
            fsio.append_jsonl(
                self.project.path("logs", "safety_retries.jsonl"),
                {
                    "ts": fsio.now_iso(),
                    "take_id": take.id,
                    "clip_id": take.clip_id,
                    "model": self.spec.id,
                    "provider_model": provider_model,
                    "level": level,
                    "reason": scrub(reason),
                    "rewrites_applied": rewrites,
                    "original_prompt": old_prompt,
                    "rewritten_prompt": new_prompt,
                },
            )

    # -- dispatch / resume --------------------------------------------------------------------
    def dispatch(self, payload: GenerationPayload, *, take: Take, on_update: Callable[[Take], Any],
                 poll_interval_s: float = 10, timeout_s: float = 1800) -> Take:
        """Submit, persist the operation name via ``on_update`` BEFORE polling, poll, download.

        Provider fallbacks (registry ``provider_fallbacks``) are tried in order after a failure; RAI-filtered results
        are automatically re-prompted with ``safety_reprompt_payload`` and re-rolled up to ``max_rai_retries`` times
        per provider model. On timeout the take is returned still in flight (``submitted``/``running``) so
        :meth:`resume` can pick it up later — nothing is resubmitted.

        **Fail fast on credentials:** an :class:`AuthError` (gcloud token missing / expired / blocked by
        Context-Aware Access, HTTP 401/403) propagates immediately. It is never logged as a per-model failure, never
        re-rolled and never handed to the next provider model — every one of those would fail the same way.
        """
        take = take.model_copy(deep=True)
        deadline = time.monotonic() + float(timeout_s)
        attempt = 0
        last_error: dict[str, Any] | None = None
        for provider in self.provider_chain(payload):
            active_payload = payload
            for rai_round in range(max(1, int(self.max_rai_retries))):
                try:
                    operation = self.submit(active_payload, provider_model=provider, attempt=attempt)
                except (UnsupportedFeatureError, AuthError):
                    raise
                except Exception as exc:  # noqa: BLE001
                    raise_if_auth_failure(exc, context=f"submitting {take.id} to {provider}")
                    err_msg = short_error(exc)
                    if is_safety_error(err_msg) and rai_round + 1 < self.max_rai_retries and time.monotonic() < deadline:
                        old_p = active_payload.bundle.prompt
                        active_payload, rewrites = safety_reprompt_payload(payload, level=rai_round + 1)
                        self._log_safety_reprompt(
                            take,
                            provider_model=provider,
                            level=rai_round + 1,
                            reason=f"submit_filtered: {err_msg}",
                            old_prompt=old_p,
                            new_prompt=active_payload.bundle.prompt,
                            rewrites=rewrites,
                        )
                        continue
                    last_error = {"stage": "submit", "status": "failed", "provider_model": provider,
                                  "message": err_msg}
                    self.log_event(take, "submit_failed", **last_error)
                    break
                attempt += 1
                take.operation_name = operation
                take.provider_model = provider
                take.dispatch_status = DispatchStatus.SUBMITTED
                take.error = None
                self.log_event(take, "submitted", operation_name=operation, provider_model=provider, attempt=attempt,
                               safety_level=rai_round)
                on_update(take)  # persisted BEFORE polling (acceptance 9)
                take, result = self._await(take, on_update=on_update, poll_interval_s=poll_interval_s,
                                           deadline=deadline)
                if result.status not in TERMINAL_STATUSES:
                    return take  # timed out: still in flight, resume later
                if result.status == DispatchStatus.COMPLETED:
                    return self._finish(take, result, on_update)
                last_error = {**(result.error or {}), "status": result.status.value, "provider_model": provider,
                              "operation_name": operation}
                self.log_event(take, f"result_{result.status.value}", **_flat(last_error), **_usage_fields(result))
                retry_rai = (result.status == DispatchStatus.FILTERED and rai_round + 1 < self.max_rai_retries
                             and time.monotonic() < deadline)
                if not retry_rai:
                    break  # failed (or RAI retries exhausted) -> next provider model
                old_p = active_payload.bundle.prompt
                active_payload, rewrites = safety_reprompt_payload(payload, level=rai_round + 1)
                self._log_safety_reprompt(
                    take,
                    provider_model=provider,
                    level=rai_round + 1,
                    reason=str((result.error or {}).get("message") or "RAI filtered during generation"),
                    old_prompt=old_p,
                    new_prompt=active_payload.bundle.prompt,
                    rewrites=rewrites,
                )
        filtered = (last_error or {}).get("status") == DispatchStatus.FILTERED.value
        take.dispatch_status = DispatchStatus.FILTERED if filtered else DispatchStatus.FAILED
        take.error = last_error or {"message": "no provider model accepted the request"}
        self.log_event(take, "gave_up", status=take.dispatch_status.value)
        on_update(take)
        return take

    def resume(self, take: Take, *, on_update: Callable[[Take], Any], poll_interval_s: float = 10,
               timeout_s: float = 1800) -> Take:
        """Poll an existing provider operation to completion. NEVER resubmits."""
        if not take.operation_name:
            raise AdapterError(f"{take.id} has no operation_name to resume (it is never resubmitted)")
        take = take.model_copy(deep=True)
        if take.dispatch_status == DispatchStatus.COMPLETED and take.video_uri:
            return take
        if take.dispatch_status == DispatchStatus.PLANNED:
            take.dispatch_status = DispatchStatus.SUBMITTED
        self.log_event(take, "resume", operation_name=take.operation_name)
        take, result = self._await(take, on_update=on_update, poll_interval_s=poll_interval_s,
                                   deadline=time.monotonic() + float(timeout_s))
        if result.status not in TERMINAL_STATUSES:
            return take
        if result.status == DispatchStatus.COMPLETED:
            return self._finish(take, result, on_update)
        take.dispatch_status = result.status
        take.error = {**(result.error or {}), "status": result.status.value, "operation_name": take.operation_name,
                      "resumed": True}
        self.log_event(take, f"result_{result.status.value}", **_flat(take.error))
        on_update(take)
        return take

    def _await(self, take: Take, *, on_update: Callable[[Take], Any], poll_interval_s: float,
               deadline: float) -> tuple[Take, PollResult]:
        while True:
            result = self.poll(str(take.operation_name))
            if result.status in TERMINAL_STATUSES:
                return take, result
            if result.status == DispatchStatus.RUNNING and take.dispatch_status != DispatchStatus.RUNNING:
                take.dispatch_status = DispatchStatus.RUNNING
                on_update(take)
            if time.monotonic() >= deadline:
                self.log_event(take, "poll_timeout", operation_name=take.operation_name)
                return take, result
            self._sleep(max(0.0, float(poll_interval_s)))

    def _finish(self, take: Take, result: PollResult, on_update: Callable[[Take], Any]) -> Take:
        project = self._require_project()
        dest = project.take_dir(take.id) / TAKE_VIDEO_NAME
        try:
            path = dest if dest.is_file() and dest.stat().st_size > 0 else self.download(result, dest)
        except Exception as exc:  # noqa: BLE001 - keep in flight: resume re-polls and re-downloads, never resubmits
            take.error = {"stage": "download", "message": short_error(exc), "operation_name": take.operation_name}
            self.log_event(take, "download_failed", message=take.error["message"])
            if take.dispatch_status != DispatchStatus.RUNNING:
                take.dispatch_status = DispatchStatus.RUNNING
            on_update(take)
            return take
        take.video_uri = project.rel(path)
        take.dispatch_status = DispatchStatus.COMPLETED
        take.error = None
        self.log_event(take, "completed", video_uri=take.video_uri, provider_model=take.provider_model,
                       **_usage_fields(result))
        on_update(take)
        return take

    # -- post-processing -------------------------------------------------------------------
    def extract_dialogue(self, take: Take) -> Path | None:
        """``<take_dir>/dialogue.wav`` (16 kHz mono PCM) from the take's native audio; None without audio."""
        return extract_dialogue_wav(self._require_project(), take)

    def log_event(self, take: Take, event: str, **fields: Any) -> None:
        """Append-only dispatch audit trail in the take folder (no payload bodies, no credentials)."""
        if self.project is None:
            return
        record = {"ts": fsio.now_iso(), "take_id": take.id, "model": self.spec.id, "event": event,
                  **{k: (scrub(v) if isinstance(v, str) else v) for k, v in _flat(fields).items()}}
        fsio.append_jsonl(self.project.take_dir(take.id) / DISPATCH_LOG_NAME, record)


def _flat(values: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe shallow copy (nested values are stringified)."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            out[str(key)] = value
        elif isinstance(value, (list, tuple)) and all(isinstance(v, (str, int, float, bool)) for v in value):
            out[str(key)] = list(value)
        else:
            out[str(key)] = short_error(value, 500)
    return out


def _usage_fields(result: Any) -> dict[str, int]:
    """Flat ``usage_*`` token counts from a poll result's ``raw['usage']`` (empty when the provider reports none)."""
    raw = getattr(result, "raw", None)
    usage = raw.get("usage") if isinstance(raw, dict) else None
    if not isinstance(usage, dict):
        return {}
    out: dict[str, int] = {}
    for src, dst in (("total_input_tokens", "usage_input_tokens"), ("total_output_tokens", "usage_output_tokens"),
                     ("total_thought_tokens", "usage_thought_tokens"), ("total_tokens", "usage_total_tokens")):
        if isinstance(usage.get(src), int):
            out[dst] = usage[src]
    return out


class LegacyVideoAdapter(VideoAdapter):
    """Shared plumbing for the real wrappers around ``vertex_veo_director`` (never edited, imported lazily).

    * the gcloud access token comes from ``vertex_veo_director.GcloudToken`` (or ``client.token`` / ``token=``) and is
      force-refreshed + the call retried once on HTTP 401 — long polls outlive the 45-minute token cache,
    * transient poll errors keep the take ``running`` (bounded by ``max_poll_errors`` consecutive failures),
    * output goes to ``<VERTEX_GCS_BUCKET_ROOT>/agy/<PROJECT>/<payload_hash>/a<attempt>`` (or inline delivery) and is
      downloaded into the take folder with a staged immutable write.
    """

    agent_name = "agy.adapter.vertex"
    max_poll_errors = 8

    def __init__(self, spec: ModelSpec, *, project: Any = None, fake: bool | None = False, client: Any = None,
                 registry: Any = None, sleep: Callable[[float], None] | None = None, token: Any = None,
                 gcs_output_root: str | None = None, inline_delivery: bool = False) -> None:
        super().__init__(spec, project=project, fake=False, client=client, registry=registry, sleep=sleep)
        self._token = token if token is not None else getattr(client, "token", None)
        self.gcs_output_root = gcs_output_root
        self.inline_delivery = bool(inline_delivery)
        self._poll_errors: dict[str, int] = {}

    @staticmethod
    def legacy() -> Any:
        import vertex_veo_director  # noqa: PLC0415 - legacy generator, real path only (wrapped, never edited)

        return vertex_veo_director

    @property
    def token(self) -> Any:
        if self._token is None:
            self._token = self.legacy().GcloudToken()
        return self._token

    def refresh_token(self) -> None:
        """Force a fresh gcloud token; a broken sign-in is an :class:`AuthError` (complete message, no retry)."""
        try:
            self.token.get(force=True)
        except Exception as exc:  # noqa: BLE001
            raise auth.as_auth_error(exc, context="refreshing the gcloud access token") from exc

    def call_with_auth(self, fn: Callable[[], Any]) -> Any:
        """Run ``fn``; on HTTP 401 force a token refresh and retry exactly once.

        A failure to mint the token, or a second 401/403 after the refresh, is raised as :class:`AuthError` so the
        dispatch loop stops immediately instead of re-rolling seeds or trying the next provider model.
        """
        try:
            return fn()
        except AuthError:
            raise
        except Exception as exc:
            if exc_status(exc) == 403:
                raise auth.as_auth_error(exc, context="calling Vertex AI") from exc
            if exc_status(exc) != 401:
                if is_auth_failure(exc):
                    raise auth.as_auth_error(exc, context="calling Vertex AI") from exc
                raise
        self.refresh_token()
        try:
            return fn()
        except AuthError:
            raise
        except Exception as exc:
            raise_if_auth_failure(exc, context="calling Vertex AI after a token refresh")
            raise

    def gcs_prefix(self, payload: GenerationPayload, attempt: int) -> str | None:
        if self.inline_delivery:
            return None
        root = self.gcs_output_root or os.environ.get("VERTEX_GCS_BUCKET_ROOT") or ""
        if not root.strip():
            return None  # no bucket configured -> inline delivery (there is no default bucket, by design)
        project_id = self.project.project_id if self.project is not None else "ADHOC"
        return f"{root.rstrip('/')}/agy/{project_id}/{payload.payload_hash}/a{int(attempt)}"

    def poll_failed(self, operation_name: str, exc: BaseException) -> PollResult:
        count = self._poll_errors.get(operation_name, 0) + 1
        self._poll_errors[operation_name] = count
        error = {"stage": "poll", "message": short_error(exc), "consecutive_errors": count}
        if count >= self.max_poll_errors:
            return PollResult(status=DispatchStatus.FAILED, error=error, raw={"operation_name": operation_name})
        return PollResult(status=DispatchStatus.RUNNING, error={**error, "transient": True},
                          raw={"operation_name": operation_name})

    def poll_ok(self, operation_name: str) -> None:
        self._poll_errors.pop(operation_name, None)

    def download(self, result: PollResult, dest: Path) -> Path:
        dest = Path(dest)
        inline = result.raw.get("inline_data")
        if inline:
            return write_inline_bytes(inline, dest)
        uri = str(result.video_uri or "")
        if not uri.startswith("gs://"):
            raise AdapterError("the finished operation carries neither a gs:// URI nor inline video data")
        legacy, token = self.legacy(), self.token
        return staged_write(dest, lambda staging: run_coro_sync(legacy.download_gcs_object(uri, staging, token)))


def create_video_adapter(spec: ModelSpec, real_cls: type[VideoAdapter], *, project: Any = None,
                         fake: bool | None = None, client: Any = None, registry: Any = None,
                         **kw: Any) -> VideoAdapter:
    """Shared factory for ``omni`` / ``veo``: fake backend in fake mode, else the real wrapper."""
    from .fake import FakeVideoAdapter, FakeVideoBackend  # noqa: PLC0415 - avoid an import cycle

    if spec.kind.value != "video_av":
        raise AdapterError(f"{spec.id}: {real_cls.__name__} needs a video_av registry entry, got {spec.kind.value}")
    if resolve_fake(fake) or isinstance(client, FakeVideoBackend):
        return FakeVideoAdapter(spec, project=project, client=client, registry=registry, **kw)
    return real_cls(spec, project=project, client=client, registry=registry, **kw)
