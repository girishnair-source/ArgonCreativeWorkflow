"""Deterministic local generation backend (``AGY_FAKE_GENERATION=1`` / ``fake=True``) — never touches the network.

* :class:`FakeVideoBackend` plays the provider: it records every ``submit`` / ``poll`` (so tests can prove that nothing
  is ever dispatched twice), decides each job's outcome at submit time and can be told to misbehave:
  ``fail_models`` (provider rejects the request), ``fail_next`` / ``filter_next`` (job ends failed / RAI-filtered),
  ``inject_music`` / ``music_next`` (spontaneous score), ``extra_voice``, ``transcript_override`` (the clip "says"
  the wrong words), ``voice_swap`` / ``voice_tones`` (voice drift), ``no_audio`` and ``kill_after_submit`` (the process
  "dies" on the first poll after a submit -> :class:`SimulatedKill`).
* :class:`FakeVideoAdapter` is the :class:`~agy_studio.adapters.base.VideoAdapter` on top of it. Takes are rendered with
  :func:`agy_studio.media.make_test_clip`: the picture is the dominant colour of the first conditioning image (first
  frame, else first reference image) and the audio is a sine tone whose frequency is a stable hash of the speaking
  character (:func:`speaker_tone_hz`), so speaker-similarity QA is meaningful offline. A ``fake_take.json`` sidecar in
  the take folder records what the fake "performed" (speakers, spoken text, injected defects) for the fake ASR.

Jobs are also persisted under ``<project>/logs/fake_backend/`` so a resumed process can poll an operation that was
submitted by a process that died (acceptance criterion 9 in fake mode).
"""

from __future__ import annotations

import re
import threading
import uuid
from pathlib import Path
from typing import Any

from .. import fsio, media
from ..models import DispatchStatus, GenerationPayload
from ..registry import ModelSpec
from .base import AdapterError, PollResult, VideoAdapter, check_kind, staged_write

FAKE_SIDECAR_NAME = "fake_take.json"
FAKE_STORE_DIR = ("logs", "fake_backend")
AMBIENCE_TONE_HZ = 70.0  # "room tone" of clips without dialogue
_OP_RE = re.compile(r"^fakeop-[A-Za-z0-9_.\-]{1,200}$")
_PORTRAIT = (270, 480)
_LANDSCAPE = (480, 270)


class SimulatedKill(BaseException):
    """Raised by the fake backend to simulate the process dying right after a submit (tests only).

    Derives from BaseException so no ``except Exception`` in the pipeline can swallow it — exactly like a SIGKILL.
    """


def speaker_tone_hz(character_id: str) -> float:
    """Stable per-character fake 'voice' frequency, log-uniform in 110–440 Hz (content hash of the id)."""
    u = int(fsio.stable_hash(["agy.fake.voice", character_id], 8), 16) / float(0xFFFFFFFF)
    return round(110.0 * 2.0 ** (2.0 * u), 1)


def read_fake_sidecar(near: Path | str | None) -> dict[str, Any] | None:
    """The ``fake_take.json`` next to ``near`` (a take's dialogue.wav / take.mp4), or None."""
    if not near:
        return None
    sidecar = Path(near).parent / FAKE_SIDECAR_NAME
    if not sidecar.is_file():
        return None
    try:
        data = fsio.read_json(sidecar)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _no_sleep(_seconds: float) -> None:
    return None


class FakeVideoBackend:
    """In-memory (+ optional on-disk) fake provider. Thread-safe; every knob is a plain attribute."""

    def __init__(self, *, polls_until_done: int = 1) -> None:
        self._lock = threading.RLock()
        self.reset(polls_until_done=polls_until_done)

    def reset(self, *, polls_until_done: int = 1) -> "FakeVideoBackend":
        with self._lock:
            self.jobs: dict[str, dict[str, Any]] = {}
            self.submit_calls: list[dict[str, Any]] = []
            self.poll_calls: list[str] = []
            self.timeline: list[tuple[str, str]] = []
            self.fail_models: set[str] = set()
            self.submit_errors_next = 0
            self.fail_next = 0
            self.filter_next = 0
            self.inject_music = False
            self.music_next = 0
            self.extra_voice = False
            self.transcript_override: str | None = None
            self.voice_swap: dict[str, str] = {}
            self.voice_tones: dict[str, float] = {}
            self.no_audio = False
            self.kill_after_submit = False
            self.enforce_safety_filters = False
            self.polls_until_done = max(0, int(polls_until_done))
        return self

    # -- inspection helpers -------------------------------------------------------------------
    def submits_for(self, payload_hash: str) -> int:
        with self._lock:
            return sum(1 for c in self.submit_calls if c["payload_hash"] == payload_hash)

    def polls_for(self, operation_name: str) -> int:
        with self._lock:
            return sum(1 for op in self.poll_calls if op == operation_name)

    def note(self, event: str, ref: str) -> None:
        with self._lock:
            self.timeline.append((event, ref))

    # -- provider primitives ------------------------------------------------------------------
    def submit(self, payload: GenerationPayload, *, model_id: str, provider_model: str, attempt: int = 0,
               store_dir: Path | None = None, color: str | None = None) -> str:
        from .base import _SAFETY_LEVEL1_RULES, _split_negative_tail

        with self._lock:
            if provider_model in self.fail_models:
                self.timeline.append(("submit_rejected", provider_model))
                raise AdapterError(f"fake provider {provider_model} rejected the request (fail_models)")
            if self.submit_errors_next > 0:
                self.submit_errors_next -= 1
                self.timeline.append(("submit_rejected", provider_model))
                raise AdapterError(f"fake provider {provider_model} returned HTTP 500 on submit")
            outcome = "completed"
            pos_body, _ = _split_negative_tail(payload.bundle.prompt)
            if self.fail_next > 0:
                self.fail_next -= 1
                outcome = "failed"
            elif self.filter_next > 0:
                self.filter_next -= 1
                outcome = "filtered"
            elif self.enforce_safety_filters and any(rx.search(pos_body) for rx, _, _ in _SAFETY_LEVEL1_RULES):
                outcome = "filtered"
            music = bool(self.inject_music)
            if self.music_next > 0:
                self.music_next -= 1
                music = True
            extra = payload.extra or {}
            speakers = [str(s) for s in (extra.get("speakers") or [])]
            voice = self.voice_swap.get(speakers[0], speakers[0]) if speakers else None
            if voice:
                tone = float(self.voice_tones.get(voice) or speaker_tone_hz(voice))
            else:
                tone = AMBIENCE_TONE_HZ
            expected = str(extra.get("expected_text") or "")
            spoken = self.transcript_override if self.transcript_override is not None else expected
            op = f"fakeop-{model_id}-{payload.payload_hash[:16]}-a{int(attempt)}-{uuid.uuid4().hex[:8]}"
            job = {
                "operation_name": op,
                "model_id": model_id,
                "provider_model": provider_model,
                "payload_hash": payload.payload_hash,
                "attempt": int(attempt),
                "clip_id": extra.get("clip_id") or payload.bundle.clip_id,
                "duration_s": max(1.0, float(payload.bundle.duration_s)),
                "aspect_ratio": payload.bundle.aspect_ratio,
                "speakers": speakers,
                "voice_character": voice,
                "tone_hz": tone,
                "expected_text": expected,
                "spoken_text": spoken,
                "music": music,
                "extra_voice": bool(self.extra_voice),
                "no_audio": bool(self.no_audio),
                "outcome": outcome,
                "running_polls": self.polls_until_done,
                "color": color,
                "created_at": fsio.now_iso(),
                "polls": 0,
            }
            self.jobs[op] = job
            self.submit_calls.append({"operation_name": op, "model_id": model_id, "provider_model": provider_model,
                                      "payload_hash": payload.payload_hash, "attempt": int(attempt),
                                      "clip_id": job["clip_id"]})
            self.timeline.append(("submit", op))
        if store_dir is not None:
            fsio.write_new(Path(store_dir) / f"{op}.json", {k: v for k, v in job.items() if k != "polls"})
        return op

    def job(self, operation_name: str, *, store_dir: Path | None = None) -> dict[str, Any] | None:
        with self._lock:
            found = self.jobs.get(operation_name)
            if found is not None:
                return found
            if store_dir is None or not _OP_RE.match(operation_name or ""):
                return None
            path = Path(store_dir) / f"{operation_name}.json"
            if not path.is_file():
                return None
            data = fsio.read_json(path)
            if not isinstance(data, dict):
                return None
            data["polls"] = 0
            self.jobs[operation_name] = data
            return data

    def poll(self, operation_name: str, *, store_dir: Path | None = None) -> PollResult:
        with self._lock:
            job = self.job(operation_name, store_dir=store_dir)
            self.poll_calls.append(operation_name)
            self.timeline.append(("poll", operation_name))
            raw = {"operation_name": operation_name}
            if job is None:
                return PollResult(status=DispatchStatus.FAILED, raw=raw,
                                  error={"message": f"unknown fake operation {operation_name}"})
            if self.kill_after_submit and int(job.get("polls", 0)) == 0:
                raise SimulatedKill(f"simulated process kill right after submitting {operation_name}")
            job["polls"] = int(job.get("polls", 0)) + 1
            if job["polls"] <= int(job.get("running_polls", 0)):
                return PollResult(status=DispatchStatus.RUNNING, raw={**raw, "provider_status": "running"})
            outcome = job.get("outcome", "completed")
            if outcome == "failed":
                return PollResult(status=DispatchStatus.FAILED, raw=raw,
                                  error={"message": "fake provider: internal error while generating", "code": 500})
            if outcome == "filtered":
                return PollResult(status=DispatchStatus.FILTERED, raw=raw,
                                  error={"message": "fake provider: output blocked by the RAI filter",
                                         "rai_media_filtered_count": 1, "reasons": ["fake_rai_filter"]})
            return PollResult(status=DispatchStatus.COMPLETED, video_uri=f"fake://{operation_name}", raw=raw)


_DEFAULT_BACKEND: FakeVideoBackend | None = None
_DEFAULT_LOCK = threading.Lock()


def default_backend() -> FakeVideoBackend:
    """Process-wide fake provider shared by every FakeVideoAdapter that was not given ``client=``."""
    global _DEFAULT_BACKEND
    with _DEFAULT_LOCK:
        if _DEFAULT_BACKEND is None:
            _DEFAULT_BACKEND = FakeVideoBackend()
        return _DEFAULT_BACKEND


def reset_default_backend(*, polls_until_done: int = 1) -> FakeVideoBackend:
    return default_backend().reset(polls_until_done=polls_until_done)


class FakeVideoAdapter(VideoAdapter):
    """Video adapter backed by :class:`FakeVideoBackend` (no network, no sleeping)."""

    agent_name = "agy.adapter.fake_video"
    default_poll_interval_s = 0.0
    default_timeout_s = 120.0

    def __init__(self, spec: ModelSpec, *, project: Any = None, fake: bool | None = True, client: Any = None,
                 registry: Any = None, sleep: Any = None, **_real_only: Any) -> None:
        super().__init__(spec, project=project, fake=True, client=None, registry=registry, sleep=sleep or _no_sleep)
        self.backend: FakeVideoBackend = client if isinstance(client, FakeVideoBackend) else default_backend()
        self.client = self.backend

    def _store_dir(self) -> Path | None:
        return self.project.path(*FAKE_STORE_DIR) if self.project is not None else None

    def _conditioning_color(self, payload: GenerationPayload) -> str | None:
        """Dominant colour of the first conditioning image as an ffmpeg colour (``0xRRGGBB``)."""
        bundle = payload.bundle
        rel = bundle.first_frame or (bundle.reference_images[0] if bundle.reference_images else None)
        if not rel or self.project is None:
            return None
        try:
            rgb = media.read_image_rgb(self.project.abs(rel), (8, 8)).reshape(-1, 3).mean(axis=0)
        except Exception:  # noqa: BLE001 - a fake colour is cosmetic; fall back to the test pattern
            return None
        r, g, b = (int(max(0, min(255, round(float(c))))) for c in rgb)
        return f"0x{r:02X}{g:02X}{b:02X}"

    def submit(self, payload: GenerationPayload, *, provider_model: str | None = None, attempt: int = 0) -> str:
        return self.backend.submit(payload, model_id=self.spec.id, provider_model=provider_model or payload.provider_model,
                                   attempt=attempt, store_dir=self._store_dir(), color=self._conditioning_color(payload))

    def poll(self, operation_name: str) -> PollResult:
        return self.backend.poll(operation_name, store_dir=self._store_dir())

    def download(self, result: PollResult, dest: Path) -> Path:
        dest = Path(dest)
        op = str(result.raw.get("operation_name") or str(result.video_uri or "").removeprefix("fake://"))
        job = self.backend.job(op, store_dir=self._store_dir())
        if job is None:
            raise AdapterError(f"fake backend has no job {op!r} to download")
        size = _LANDSCAPE if job.get("aspect_ratio") == "16:9" else _PORTRAIT
        tone = None if job.get("no_audio") else float(job.get("tone_hz") or AMBIENCE_TONE_HZ)
        duration = float(job.get("duration_s") or 4.0)
        color = job.get("color")
        path = staged_write(dest, lambda staging: media.make_test_clip(staging, duration_s=duration, size=size, fps=24,
                                                                       tone_hz=tone, color=color))
        sidecar = dest.parent / FAKE_SIDECAR_NAME
        if not sidecar.exists():
            keys = ("operation_name", "model_id", "provider_model", "payload_hash", "clip_id", "speakers",
                    "voice_character", "tone_hz", "expected_text", "spoken_text", "music", "extra_voice", "no_audio",
                    "color", "duration_s")
            record = {k: job.get(k) for k in keys}
            record["generated_by"] = fsio.generated_by(self.agent_name, model=str(job.get("provider_model"))).model_dump(
                mode="json")
            fsio.write_new(sidecar, record)
        self.backend.note("download", op)
        return path


def create_adapter(spec: ModelSpec, **kw: Any) -> FakeVideoAdapter:
    """Registry factory (only used if a registry entry points here explicitly)."""
    check_kind(spec, "video_av")
    kw.pop("fake", None)
    return FakeVideoAdapter(spec, **kw)
