"""Shot Generator — BUILD_SPEC_v3 §6 Production (Shot Generator), §7.1 pre-flight lint gate, §12 crash-safe dispatch.

* **Pre-flight lint gate (acceptance criterion 1):** :func:`generate_clip` REFUSES to dispatch unless the clip carries
  a passing, non-blocking :class:`~agy_studio.models.LintReport` whose ``spec_hash`` equals ``fsio.clip_lint_hash(clip)``.
  Mutating a linted clip without re-linting raises :class:`LintGateError`.
* **Crash-safe deduplication (acceptance criterion 9):** before calling ``adapter.dispatch``, the payload hash is
  registered in ``state.json`` (``state.record_dispatch(payload_hash, take_id)``) and the operation name is persisted
  via ``on_update`` BEFORE polling. If the process is killed mid-poll and later resumed via :func:`resume_inflight` or
  another :func:`generate_clip` call with the same payload hash, the existing operation is polled to completion and is
  **never** submitted a second time.
* **Post-processing:** every completed take gets ``first.png``, ``last.png``, ``dialogue.wav`` (16 kHz mono PCM), and
  an ASR transcript stored on the take record.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import continuity, fsio, media
from .adapters.base import (
    DIALOGUE_WAV_NAME,
    IN_FLIGHT_STATUSES,
    PAYLOAD_NAME,
    TERMINAL_STATUSES,
    UnsupportedFeatureError,
    VideoAdapter,
    extract_dialogue_wav,
    staged_write,
)
from .models import ClipStatus, DispatchStatus, GenClip, GenerationPayload, Take, TakeStatus
from .project import Project
from .registry import Registry
from .state import StateStore

AGENT = "agy.shot_generator"
FIRST_FRAME_NAME = "first.png"
LAST_FRAME_NAME = "last.png"


class LintGateError(RuntimeError):
    """Raised when a clip is unlinted, has blocking lint findings, or was modified since its lint report."""


class ModelFallbackNeeded(RuntimeError):
    """Raised when the preferred model cannot honour or complete the clip and a fallback model is configured."""

    def __init__(self, model_id: str, reason: str = "") -> None:
        super().__init__(f"fallback to model {model_id!r} required" + (f": {reason}" if reason else ""))
        self.model_id = model_id
        self.reason = reason


def check_lint_gate(clip: GenClip) -> None:
    """Verify that ``clip`` passed pre-flight lint and has not been mutated since (acceptance criterion 1)."""
    rep = clip.lint_report
    if rep is None:
        raise LintGateError(f"{clip.id}: missing pre-flight lint_report — run lint before generating")
    if not rep.passed or rep.blocking:
        rules = ", ".join(f.rule for f in rep.findings if f.severity == "error") or "blocking findings"
        raise LintGateError(f"{clip.id}: lint_report did not pass ({rules}) — refusing to dispatch")
    current_hash = fsio.clip_lint_hash(clip)
    if rep.spec_hash != current_hash:
        raise LintGateError(
            f"{clip.id}: clip spec changed after lint (lint hash {rep.spec_hash} != current {current_hash}) — re-lint first"
        )


def _default_seed_base(clip_id: str) -> int:
    return int(fsio.stable_hash(["agy.seed", clip_id], 8), 16) % 1_000_000_000


def _ensure_clip_has_take(project: Project, clip_id: str, take_id: str) -> GenClip:
    clip = project.load_clip(clip_id)
    if take_id not in clip.takes:
        clip.takes = [*clip.takes, take_id]
        project.save_clip(clip)
    return clip


def _postprocess_take(
    project: Project,
    registry: Registry,
    clip: GenClip,
    take: Take,
    *,
    adapter: VideoAdapter | None = None,
    fake: bool | None = None,
) -> Take:
    """Extract first/last frames, 16 kHz dialogue WAV, and ASR transcript for a completed take (idempotent)."""
    if take.dispatch_status != DispatchStatus.COMPLETED or not take.video_uri:
        return take
    video = project.abs(take.video_uri)
    if not video.is_file():
        return take
    tdir = project.take_dir(take.id)
    changed = False

    first_png = tdir / FIRST_FRAME_NAME
    if not first_png.is_file():
        staged_write(first_png, lambda staging: media.extract_frame(video, staging, at="first", overwrite=True))
    first_rel = project.rel(first_png)
    if take.first_frame_uri != first_rel:
        take.first_frame_uri = first_rel
        changed = True

    last_png = tdir / LAST_FRAME_NAME
    if not last_png.is_file():
        staged_write(last_png, lambda staging: media.extract_frame(video, staging, at="last", overwrite=True))
    last_rel = project.rel(last_png)
    if take.last_frame_uri != last_rel:
        take.last_frame_uri = last_rel
        changed = True

    wav = adapter.extract_dialogue(take) if adapter is not None else extract_dialogue_wav(project, take)
    if wav is not None and wav.is_file():
        wav_rel = project.rel(wav)
        if take.dialogue_wav_uri != wav_rel:
            take.dialogue_wav_uri = wav_rel
            changed = True
        if take.transcript is None:
            expected = " ".join(continuity.clip_line_texts(clip))
            asr = registry.adapter("gemini_asr", project=project, fake=fake, registry=registry)
            take.transcript = asr.transcribe(wav, language_hint="hi", expected_text=expected)
            changed = True
    elif take.transcript is None:
        take.transcript = ""
        changed = True

    if changed:
        project.save_take(take)
    return take


def _load_or_recover_take(
    project: Project,
    clip: GenClip,
    take_id: str,
    payload: GenerationPayload,
) -> Take:
    try:
        return project.load_take(take_id)
    except FileNotFoundError:
        take = Take(
            id=take_id,
            clip_id=clip.id,
            model=payload.model_id,
            provider_model=payload.provider_model,
            seed=payload.seed,
            payload_hash=payload.payload_hash,
            dispatch_status=DispatchStatus.PLANNED,
            status=TakeStatus.CANDIDATE,
            generated_by=fsio.generated_by(AGENT, model=payload.provider_model, seed=payload.seed),
        )
        project.save_take(take)
        _ensure_clip_has_take(project, clip.id, take_id)
        return take


def generate_clip(
    project: Project,
    registry: Registry,
    state: StateStore,
    clip_id: str,
    *,
    n_takes: int | None = None,
    budget: Any = None,
    fake: bool | None = None,
    seed_base: int | None = None,
) -> list[Take]:
    """Generate ``n_takes`` takes for ``clip_id`` after enforcing the pre-flight lint gate and payload deduplication."""
    fsio.validate_id(clip_id, "clip id")
    clip = project.load_clip(clip_id)
    check_lint_gate(clip)

    model_id = clip.model_target.preferred or "omni"
    fallback_id = clip.model_target.fallback
    adapter: VideoAdapter = registry.adapter(model_id, project=project, fake=fake, registry=registry)
    count = max(1, int(n_takes if n_takes is not None else project.config.settings.takes_per_clip))
    base = int(seed_base) if seed_base is not None else _default_seed_base(clip.id)

    results: list[Take] = []
    for i in range(count):
        seed = base + i
        try:
            payload = adapter.build_payload(clip.spec, clip.conditioning, clip=clip, seed=seed)
        except UnsupportedFeatureError as exc:
            if fallback_id and fallback_id != model_id:
                raise ModelFallbackNeeded(fallback_id, str(exc)) from exc
            raise

        existing_id = state.dispatched_take(payload.payload_hash)
        if existing_id is not None:
            take = _load_or_recover_take(project, clip, existing_id, payload)
            _ensure_clip_has_take(project, clip.id, take.id)
            if take.dispatch_status == DispatchStatus.COMPLETED and take.video_uri:
                take = _postprocess_take(project, registry, clip, take, adapter=adapter, fake=fake)
                results.append(take)
                continue
            if take.operation_name:
                take = adapter.resume(
                    take,
                    on_update=project.save_take,
                    poll_interval_s=adapter.default_poll_interval_s,
                    timeout_s=adapter.default_timeout_s,
                )
                if take.dispatch_status == DispatchStatus.COMPLETED and take.video_uri:
                    take = _postprocess_take(project, registry, clip, take, adapter=adapter, fake=fake)
                results.append(take)
                continue
            # Kill occurred after record_dispatch but before submit wrote an operation_name: dispatch using this take.
            est = budget.estimate_clip(clip, model_id=model_id, takes=1) if budget is not None else 0.0
            if budget is not None:
                budget.require(clip.scene_id, est)
        else:
            est = budget.estimate_clip(clip, model_id=model_id, takes=1) if budget is not None else 0.0
            if budget is not None:
                budget.require(clip.scene_id, est)
            take_id = project.next_take_id(clip.id)
            tdir = project.take_dir(take_id)
            payload_path = tdir / PAYLOAD_NAME
            if not payload_path.exists():
                fsio.write_new(payload_path, payload.model_dump(mode="json"))
            take = Take(
                id=take_id,
                clip_id=clip.id,
                model=model_id,
                provider_model=payload.provider_model,
                seed=seed,
                payload_hash=payload.payload_hash,
                dispatch_status=DispatchStatus.PLANNED,
                status=TakeStatus.CANDIDATE,
                generated_by=fsio.generated_by(AGENT, model=payload.provider_model, seed=seed),
            )
            project.save_take(take)
            clip = _ensure_clip_has_take(project, clip.id, take_id)
            # Record payload_hash -> take_id BEFORE submitting so a kill never loses track of the take.
            state.record_dispatch(payload.payload_hash, take_id)

        take = adapter.dispatch(
            payload,
            take=take,
            on_update=project.save_take,
            poll_interval_s=adapter.default_poll_interval_s,
            timeout_s=adapter.default_timeout_s,
        )
        if budget is not None and take.operation_name:
            budget.charge(clip.scene_id, est, ref=take.id, note=f"{clip.id} ({model_id})")
        if take.dispatch_status == DispatchStatus.COMPLETED and take.video_uri:
            take = _postprocess_take(project, registry, clip, take, adapter=adapter, fake=fake)
        results.append(take)

    clip = project.load_clip(clip.id)
    for t in results:
        if t.id not in clip.takes:
            clip.takes.append(t.id)
    if any(t.dispatch_status == DispatchStatus.COMPLETED and t.video_uri for t in results):
        if clip.status in (ClipStatus.LINTED, ClipStatus.PENDING, ClipStatus.PLANNED, ClipStatus.STALE, ClipStatus.NEEDS_RELINT):
            clip.status = ClipStatus.GENERATED
        project.save_clip(clip)
        return results

    project.save_clip(clip)
    if results and all(t.dispatch_status in (DispatchStatus.FAILED, DispatchStatus.FILTERED) for t in results):
        if fallback_id and fallback_id != model_id:
            raise ModelFallbackNeeded(fallback_id, f"all {len(results)} take(s) on {model_id} failed or were filtered")
    return results


def resume_inflight(
    project: Project,
    registry: Registry,
    state: StateStore,
    *,
    fake: bool | None = None,
) -> list[Take]:
    """Resume every in-flight take across the project by polling its existing ``operation_name`` (never resubmits)."""
    seen_ids: set[str] = set()
    candidates: list[Take] = []
    for take_id in state.load().dispatched_payloads.values():
        try:
            t = project.load_take(take_id)
        except FileNotFoundError:
            continue
        seen_ids.add(t.id)
        candidates.append(t)

    for clip in project.iter_all_clips():
        for t in project.list_takes(clip.id):
            if t.id not in seen_ids:
                seen_ids.add(t.id)
                candidates.append(t)

    resumed: list[Take] = []
    for take in candidates:
        needs_resume = (
            take.operation_name is not None
            and (
                take.dispatch_status in IN_FLIGHT_STATUSES
                or (take.dispatch_status == DispatchStatus.PLANNED)
                or (take.dispatch_status == DispatchStatus.COMPLETED and not take.video_uri)
            )
        )
        if not needs_resume:
            continue
        clip = project.load_clip(take.clip_id)
        model_id = take.model or clip.model_target.preferred or "omni"
        adapter: VideoAdapter = registry.adapter(model_id, project=project, fake=fake, registry=registry)
        updated = adapter.resume(
            take,
            on_update=project.save_take,
            poll_interval_s=adapter.default_poll_interval_s,
            timeout_s=adapter.default_timeout_s,
        )
        if updated.dispatch_status == DispatchStatus.COMPLETED and updated.video_uri:
            updated = _postprocess_take(project, registry, clip, updated, adapter=adapter, fake=fake)
            clip = project.load_clip(clip.id)
            if updated.id not in clip.takes:
                clip.takes.append(updated.id)
            if clip.status in (ClipStatus.LINTED, ClipStatus.PENDING, ClipStatus.PLANNED, ClipStatus.STALE, ClipStatus.NEEDS_RELINT):
                clip.status = ClipStatus.GENERATED
            project.save_clip(clip)
        resumed.append(updated)
    return resumed
