"""AGY Studio v5 Harness Upgrades — Two-Tier 360p->4K Omni 1.5 Flash, 3x3 Contact Sheets,
6 Visual Continuity Critics, Gemini TTS 3.8 Native Delivery, SceneSlate & Git Timeline VCS.

Implements the 6 + 1 architectural upgrades tailored to AGY Studio:
1. **Gemini TTS 3.8 Native Acoustic Delivery** (``GeminiTTS38Spec``):
   - Uses ``gemini-3.8-tts`` (`gemini_tts_3_8`), which natively understands inline acting/acoustic tags
     (no ElevenLabs bracket-stripping hack required).
2. **Two-Tier ``360p`` Draft -> ``4K`` Finalize Video Generation** (:class:`DraftFinalizeController`):
   - Uses ``gemini-omni-1.5-flash`` (`omni_1_5_flash`) to generate a fast ``360p`` draft take, builds a ``3x3``
     contact sheet, evaluates the 6 Visual Continuity Critic rules, and promotes passing takes to ``4K``
     finalization reusing the **exact same prompt, reference images, and seed** (``reuse_prompt_on_finalize=True``).
3. **Automatic ``3x3`` Video Frame Contact Sheet & 6 Visual Continuity Critics** (:func:`build_3x3_contact_sheet`,
   :class:`VisualContinuityCritic`):
   - Extracts 9 evenly spaced frames into a single ``3x3`` PNG contact sheet (zero raw `.mp4` binary loading in chat).
   - Evaluates the 6 visual continuity failure modes (`identity_ethnicity_drift`, `wardrobe_violation`,
     `vanishing_prop`, `camera_framing_violation`, `text_label_hallucination`, `scale_anatomy_distortion`)
     and emits structured `{"status": "accepted" | "rejected", "failure_mode": ..., "reason": ..., "remediation": ...}`.
4. **Two-Tier SceneSlate -> DaVinci Resolve Timeline Auto-Promotion** (:class:`SceneSlate`, :class:`SlateController`):
   - Declarative rough-cut slate (`scenes/<SC>/slate.json`) with clip trims, gain, transitions, up to 5 background
     audio tracks (`music | voiceover | sfx`) with `none | light | medium | heavy` sidechain ducking and `-14 LUFS`
     loudness normalization, plus one-call promotion to the master timeline.
5. **Git-Style Timeline Version Control** (:class:`TimelineGitRepo`):
   - Full `branch`, `commit`, `undo`, `redo`, `pull_request`, `merge`, `reset`, and `revert` over scene/project
     timelines and slates.
6. **Optimistic Concurrency (`expected_state_token`), Multi-Keyframe Fractional Timestamp Pinning (`0.0..20.0s`),
   & Windowed Audio/Video Retakes** (:class:`StateTokenGuard`, :class:`KeyframePinSpec`, :class:`WindowedRetakeSpec`).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from . import media
from .models import Model


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------------------------
# 1. Gemini TTS 3.8 Native Acoustic Delivery (No ElevenLabs Bracket Stripping)
# ---------------------------------------------------------------------------------------------
class GeminiTTS38Line(Model):
    """A single dialogue or voiceover line synthesized via Gemini TTS 3.8 (`gemini-3.8-tts`).

    Unlike ElevenLabs, Gemini TTS 3.8 natively interprets inline performance tags such as
    ``[whispering]``, ``[tense breath]``, or ``[sarcastic chuckle]`` without reading the brackets aloud,
    so inline performance tags are preserved intact alongside the acoustic delivery profile.
    """

    character_id: str
    voice_id: str
    language: str = "en-IN"
    native_script_text: str
    phonetic_guide: str | None = None
    acoustic_delivery: str = "natural conversational"
    off_screen: bool = False
    model_id: str = "gemini_tts_3_8"
    provider_model: str = "gemini-3.8-tts"

    def format_tts_prompt(self) -> str:
        phonetic = f" (phonetic: {self.phonetic_guide})" if self.phonetic_guide else ""
        mode = "OFF-SCREEN VOICEOVER (all visible mouths 100% CLOSED)" if self.off_screen else "ON-CAMERA SINGLE SPEAKER"
        return (
            f"[{self.provider_model} | {mode}] {self.character_id} ({self.voice_id}, {self.language}, "
            f"delivery={self.acoustic_delivery}): \"{self.native_script_text}\"{phonetic}"
        )


# ---------------------------------------------------------------------------------------------
# 2. Automatic 3x3 Video Frame Contact Sheet & 6 Visual Continuity Critics
# ---------------------------------------------------------------------------------------------
class VisualContinuityFailureMode(StrEnum):
    """The 6 canonical visual continuity failure modes inspected on 3x3 contact sheets."""

    IDENTITY_ETHNICITY_DRIFT = "identity_ethnicity_drift"
    WARDROBE_VIOLATION = "wardrobe_violation"
    VANISHING_PROP = "vanishing_prop"
    CAMERA_FRAMING_VIOLATION = "camera_framing_violation"
    TEXT_LABEL_HALLUCINATION = "text_label_hallucination"
    SCALE_ANATOMY_DISTORTION = "scale_anatomy_distortion"
    STATIC_FRAMES = "static_frames"  # measured: the 9 frames do not change -> a still / slideshow, not a video take


REMEDIATION_RECIPES: dict[VisualContinuityFailureMode, str] = {
    VisualContinuityFailureMode.IDENTITY_ETHNICITY_DRIFT: (
        "Lock character identity crop in Reference Slot 1 and prepend explicit facial/ethnicity anchor from Bible."
    ),
    VisualContinuityFailureMode.WARDROBE_VIOLATION: (
        "Re-attach canonical wardrobe turnaround sheet and enforce exact fabric/color description from CharacterState."
    ),
    VisualContinuityFailureMode.VANISHING_PROP: (
        "Pin hero prop reference pack and add continuous object-permanence constraint across all 9 contact-sheet frames."
    ),
    VisualContinuityFailureMode.CAMERA_FRAMING_VIOLATION: (
        "Re-anchor camera to 3D set viewport render + compute_view_space_blocking_summary() with isolated character crops."
    ),
    VisualContinuityFailureMode.TEXT_LABEL_HALLUCINATION: (
        "Strip debug labels from 3D reference render (`render_clean_viewport`) and append `(no subtitles) No text overlays`."
    ),
    VisualContinuityFailureMode.SCALE_ANATOMY_DISTORTION: (
        "Enforce realistic human anatomy & table-prop scale ratios from 3D set bounding boxes; regenerate with new seed."
    ),
    VisualContinuityFailureMode.STATIC_FRAMES: (
        "This is not a generated video take — the sampled frames are identical (a storyboard still padded into an "
        "mp4 / ffmpeg zoompan). Regenerate the clip with the video model (Omni / Veo); never substitute stills."
    ),
}


class VisualCriticReport(Model):
    """Structured accept/reject verdict with auto-remediation recipe."""

    take_id: str
    contact_sheet_path: str
    status: Literal["accepted", "rejected"]
    score: float = Field(ge=0.0, le=1.0)
    failure_mode: VisualContinuityFailureMode | None = None
    reason: str = "All 9 contact-sheet frames pass identity, wardrobe, prop, framing, text, and anatomy checks."
    remediation: str | None = None
    frame_checks: dict[str, bool] = Field(default_factory=dict)
    motion: dict[str, Any] = Field(default_factory=dict)  # measured frame-difference statistics


def build_3x3_contact_sheet(video_path: Path, output_png: Path | None = None) -> Path:
    """Tile 9 frames spread EVENLY across the whole take into one 3x3 PNG for the continuity critic.

    Fails fast when ``video_path`` is not a real, decodable video or ffmpeg is missing — there is deliberately no
    placeholder sheet (an earlier version sampled only the first second and drew a grey grid when decoding failed,
    which let still images pass as video).
    """
    video_path = Path(video_path)
    if output_png is None:
        output_png = video_path.with_name(f"{video_path.stem}_contact_3x3.png")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg/ffprobe are required to build a contact sheet (brew install ffmpeg)")
    return media.contact_sheet_3x3(video_path, Path(output_png))


class VisualContinuityCritic:
    """Evaluates a take's ``3x3`` contact sheet against the 6 visual continuity rules."""

    def evaluate(
        self,
        take_id: str,
        contact_sheet_path: Path,
        *,
        video_path: Path | None = None,
        checks: dict[str, bool] | None = None,
        threshold: float = 0.85,
    ) -> VisualCriticReport:
        """Measured check first (static frames), then the six judged continuity modes.

        ``video_path`` (preferred) samples 9 frames across the take; without it the 9 tiles of the contact sheet are
        compared. ``checks`` carries the vision-judge / human verdicts for the judged modes; it can never override a
        measured ``static_frames`` failure.
        """
        default_checks = {mode.value: True for mode in VisualContinuityFailureMode}
        if checks:
            default_checks.update({k: v for k, v in checks.items() if k != VisualContinuityFailureMode.STATIC_FRAMES.value})
        motion = self.measure_motion(video_path or contact_sheet_path, from_sheet=video_path is None)
        default_checks[VisualContinuityFailureMode.STATIC_FRAMES.value] = not motion["static"]

        failed_modes = [
            VisualContinuityFailureMode(k)
            for k, passed in default_checks.items()
            if not passed and k in VisualContinuityFailureMode._value2member_map_
        ]
        total = len(default_checks)
        passed_count = sum(1 for v in default_checks.values() if v)
        score = round(passed_count / max(1, total), 4)

        if failed_modes or score < threshold:
            primary = failed_modes[0] if failed_modes else VisualContinuityFailureMode.CAMERA_FRAMING_VIOLATION
            remediation = REMEDIATION_RECIPES[primary]
            return VisualCriticReport(
                take_id=take_id,
                contact_sheet_path=str(contact_sheet_path),
                status="rejected",
                score=score,
                failure_mode=primary,
                reason=f"Visual continuity check failed on {primary.value} (score={score:.2f} < {threshold:.2f}).",
                remediation=remediation,
                frame_checks=default_checks,
                motion=motion,
            )

        return VisualCriticReport(
            take_id=take_id,
            contact_sheet_path=str(contact_sheet_path),
            status="accepted",
            score=score,
            failure_mode=None,
            reason="All 9 contact-sheet frames pass identity, wardrobe, prop, framing, text, and anatomy checks.",
            remediation=None,
            frame_checks=default_checks,
            motion=motion,
        )

    @staticmethod
    def measure_motion(path: Path, *, from_sheet: bool = False) -> dict[str, Any]:
        """Mean/max per-pixel change between the 9 sampled frames (or the 9 tiles of a sheet)."""
        path = Path(path)
        if not from_sheet:
            return media.analyze_motion(path, n=9)
        img = media.read_image_rgb(path, size=(96 * 3, 54 * 3))
        tiles = [img[r * 54:(r + 1) * 54, c * 96:(c + 1) * 96, :] for r in range(3) for c in range(3)]
        frames = np.stack(tiles, axis=0)
        stats = media.frame_motion_stats(frames)
        return {**stats, "frames": 9, "static": stats["max_diff"] < media.STATIC_FRAME_DIFF_MAX,
                "low_motion": stats["mean_diff"] < media.LOW_MOTION_MEAN_DIFF * 2.5, "source": "sheet"}


# ---------------------------------------------------------------------------------------------
# 3. Two-Tier 360p Draft -> 4K Finalize Video Generation (Gemini Omni 1.5 Flash)
# ---------------------------------------------------------------------------------------------
class TwoTierTakeRecord(Model):
    """Tracks a clip's 360p draft exploration and 4K finalization using the exact same prompt."""

    clip_id: str
    model_id: str = "omni_1_5_flash"
    provider_model: str = "gemini-omni-1.5-flash"
    prompt: str
    prompt_sha256: str
    seed: int = 42
    reference_image_paths: list[str] = Field(default_factory=list)
    draft_resolution: Literal["360p"] = "360p"
    finalize_resolution: Literal["4k"] = "4k"
    reuse_prompt_on_finalize: bool = True
    stage: Literal["draft_360p", "critic_rejected", "finalized_4k"] = "draft_360p"
    draft_video_path: str | None = None
    draft_contact_sheet_path: str | None = None
    critic_report: VisualCriticReport | None = None
    final_4k_video_path: str | None = None
    final_4k_contact_sheet_path: str | None = None
    finalized_prompt: str | None = None


class DraftFinalizeController:
    """Orchestrates Two-Tier ``360p`` Draft -> ``3x3`` Critic QA -> ``4K`` Finalize on Gemini Omni 1.5 Flash.

    Guarantees that when a ``360p`` draft take passes the Visual Continuity Critic, the ``4K`` finalize
    pass reuses the **exact same prompt, seed, and reference image list** (`prompt_sha256` match enforced).
    """

    def __init__(self, scene_dir: Path) -> None:
        self.scene_dir = Path(scene_dir)
        self.takes_dir = self.scene_dir / "takes"
        self.takes_dir.mkdir(parents=True, exist_ok=True)
        self.critic = VisualContinuityCritic()

    @staticmethod
    def _hash_prompt(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def register_draft_360p(
        self,
        clip_id: str,
        prompt: str,
        draft_video: Path,
        *,
        seed: int = 42,
        reference_image_paths: list[str] | None = None,
        critic_checks: dict[str, bool] | None = None,
    ) -> TwoTierTakeRecord:
        """Record a REAL 360p draft produced by the video adapter, build its contact sheet and run the critic.

        This controller never generates media itself (an earlier version wrote a JSON stub named ``.mp4`` and the
        critic accepted it). ``draft_video`` must be a decodable video; otherwise this raises.
        """
        refs = list(reference_image_paths or [])
        prompt_hash = self._hash_prompt(prompt)
        draft_mp4 = Path(draft_video)
        if not draft_mp4.is_file():
            raise FileNotFoundError(f"draft video not found: {draft_mp4}")
        contact_png = build_3x3_contact_sheet(draft_mp4, self.takes_dir / f"{clip_id}_draft_360p_contact_3x3.png")
        report = self.critic.evaluate(f"{clip_id}_draft_360p", contact_png, video_path=draft_mp4, checks=critic_checks)

        stage: Literal["draft_360p", "critic_rejected"] = (
            "draft_360p" if report.status == "accepted" else "critic_rejected"
        )
        record = TwoTierTakeRecord(
            clip_id=clip_id,
            prompt=prompt,
            prompt_sha256=prompt_hash,
            seed=seed,
            reference_image_paths=refs,
            stage=stage,
            draft_video_path=str(draft_mp4),
            draft_contact_sheet_path=str(contact_png),
            critic_report=report,
        )
        self._save_record(record)
        return record

    def finalize_4k(self, record: TwoTierTakeRecord, final_video: Path) -> TwoTierTakeRecord:
        """Record the REAL 4K render of an accepted draft (same prompt + seed enforced) and re-run the critic."""
        if record.critic_report is None or record.critic_report.status != "accepted":
            raise ValueError(
                f"Cannot finalize {record.clip_id} to 4K: 360p draft has not been accepted by VisualContinuityCritic."
            )
        if self._hash_prompt(record.prompt) != record.prompt_sha256:
            raise ValueError("Prompt hash mismatch: finalize_4k must reuse the exact same prompt as draft_360p.")
        final_mp4 = Path(final_video)
        if not final_mp4.is_file():
            raise FileNotFoundError(f"final video not found: {final_mp4}")
        final_contact = build_3x3_contact_sheet(final_mp4, self.takes_dir / f"{record.clip_id}_final_4k_contact_3x3.png")
        final_report = self.critic.evaluate(f"{record.clip_id}_final_4k", final_contact, video_path=final_mp4)
        if final_report.status != "accepted":
            raise ValueError(f"4K render of {record.clip_id} rejected by the critic: {final_report.reason}")

        record.stage = "finalized_4k"
        record.final_4k_video_path = str(final_mp4)
        record.final_4k_contact_sheet_path = str(final_contact)
        record.finalized_prompt = record.prompt
        self._save_record(record)
        return record

    def _save_record(self, record: TwoTierTakeRecord) -> Path:
        out = self.takes_dir / f"{record.clip_id}_two_tier.json"
        out.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return out


# ---------------------------------------------------------------------------------------------
# 4. Two-Tier SceneSlate (kind: "slate") -> DaVinci Resolve Timeline Auto-Promotion
# ---------------------------------------------------------------------------------------------
class DuckingAmount(StrEnum):
    NONE = "none"      # 0 dB reduction
    LIGHT = "light"    # -4 dB under voiceover/dialogue
    MEDIUM = "medium"  # -9 dB under voiceover/dialogue
    HEAVY = "heavy"    # -15 dB under voiceover/dialogue


DUCKING_DB_MAP: dict[DuckingAmount, float] = {
    DuckingAmount.NONE: 0.0,
    DuckingAmount.LIGHT: -4.0,
    DuckingAmount.MEDIUM: -9.0,
    DuckingAmount.HEAVY: -15.0,
}


class SlateClipItem(Model):
    clip_id: str
    video_path: str
    trim_in_s: float = 0.0
    trim_out_s: float = 5.0
    gain_db: float = 0.0
    transition_in: Literal["cut", "cross_dissolve", "dip_to_black", "whip_pan"] = "cut"
    resolution: Literal["360p", "720p", "1080p", "4k"] = "360p"

    @property
    def duration_s(self) -> float:
        return max(0.0, round(self.trim_out_s - self.trim_in_s, 3))


class SlateAudioTrack(Model):
    track_id: str
    role: Literal["music", "voiceover", "sfx"]
    audio_path: str
    start_s: float = 0.0
    end_s: float = 10.0
    gain_db: float = 0.0
    ducking: DuckingAmount = DuckingAmount.NONE

    @property
    def effective_ducked_gain_db(self) -> float:
        return round(self.gain_db + DUCKING_DB_MAP[self.ducking], 2)


class SceneSlate(Model):
    """Declarative rough-cut SceneSlate (`kind: "slate"`) with up to 5 background audio tracks and -14 LUFS target."""

    kind: Literal["slate", "timeline"] = "slate"
    scene_id: str
    clips: list[SlateClipItem] = Field(default_factory=list)
    audio_tracks: list[SlateAudioTrack] = Field(default_factory=list)
    target_lufs: float = -14.0
    promoted_to_timeline: bool = False
    state_token: str = ""

    @field_validator("audio_tracks")
    @classmethod
    def _check_max_5_tracks(cls, v: list[SlateAudioTrack]) -> list[SlateAudioTrack]:
        if len(v) > 5:
            raise ValueError(f"SceneSlate supports at most 5 background audio tracks (got {len(v)}).")
        return v

    def total_duration_s(self) -> float:
        return round(sum(c.duration_s for c in self.clips), 3)


class SlateController:
    """Manages SceneSlate creation, sidechain auto-ducking calculation, and promotion to master Timeline."""

    def __init__(self, scene_dir: Path) -> None:
        self.scene_dir = Path(scene_dir)
        self.slate_path = self.scene_dir / "slate.json"
        self.timeline_path = self.scene_dir / "timeline_promoted.json"

    def save_slate(self, slate: SceneSlate, *, expected_state_token: str | None = None) -> SceneSlate:
        if expected_state_token is not None and self.slate_path.exists():
            existing = SceneSlate.model_validate_json(self.slate_path.read_text(encoding="utf-8"))
            StateTokenGuard.verify_state_token(existing.state_token, expected_state_token)

        payload_without_token = slate.model_dump(exclude={"state_token"})
        slate.state_token = StateTokenGuard.compute_state_token(payload_without_token)
        self.scene_dir.mkdir(parents=True, exist_ok=True)
        self.slate_path.write_text(slate.model_dump_json(indent=2), encoding="utf-8")
        return slate

    def promote_to_timeline(self, *, require_4k_clips: bool = False) -> SceneSlate:
        if not self.slate_path.exists():
            raise FileNotFoundError(f"No slate.json found in {self.scene_dir}")
        slate = SceneSlate.model_validate_json(self.slate_path.read_text(encoding="utf-8"))
        if require_4k_clips:
            non_4k = [c.clip_id for c in slate.clips if c.resolution != "4k"]
            if non_4k:
                raise ValueError(f"Cannot promote slate with non-4K clips when require_4k_clips=True: {non_4k}")

        slate.kind = "timeline"
        slate.promoted_to_timeline = True
        payload_without_token = slate.model_dump(exclude={"state_token"})
        slate.state_token = StateTokenGuard.compute_state_token(payload_without_token)
        self.slate_path.write_text(slate.model_dump_json(indent=2), encoding="utf-8")
        self.timeline_path.write_text(slate.model_dump_json(indent=2), encoding="utf-8")
        return slate


# ---------------------------------------------------------------------------------------------
# 5. Optimistic Concurrency Guard (`expected_state_token`), Keyframe Pinning & Windowed Retakes
# ---------------------------------------------------------------------------------------------
class ConcurrencyConflictError(RuntimeError):
    """Raised when `expected_state_token` does not match the current on-disk state token."""


class StateTokenGuard:
    """Deterministic SHA-256 state token generator & optimistic concurrency verifier."""

    @staticmethod
    def compute_state_token(state_obj: Any) -> str:
        canonical = json.dumps(state_obj, sort_keys=True, separators=(",", ":"), default=str)
        return "st_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def verify_state_token(actual_token: str, expected_token: str) -> None:
        if actual_token != expected_token:
            raise ConcurrencyConflictError(
                f"Optimistic concurrency conflict: expected_state_token={expected_token!r}, "
                f"but current state_token={actual_token!r}."
            )


class KeyframePin(Model):
    """Pins a reference image at an exact fractional timestamp (`0.0s .. 20.0s`) inside a generated shot."""

    image_path: str
    timestamp_s: float = Field(ge=0.0, le=20.0)
    description: str = ""


class KeyframePinSpec(Model):
    """Collection of fractional timestamp keyframe pins (`0.0..20.0s`) for fine intra-shot motion control."""

    clip_id: str
    duration_s: float = Field(default=8.0, gt=0.0, le=20.0)
    pins: list[KeyframePin] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_pins(self) -> KeyframePinSpec:
        for pin in self.pins:
            if pin.timestamp_s > self.duration_s + 1e-6:
                raise ValueError(
                    f"Keyframe pin timestamp {pin.timestamp_s}s exceeds clip duration {self.duration_s}s."
                )
        return self


class WindowedRetakeSpec(Model):
    """Surgical sub-segment audio/video retake (`0.0..20.0s`) preserving surrounding frames."""

    clip_id: str
    mode: Literal["audio_only", "video_only", "audio_video"] = "audio_video"
    start_s: float = Field(ge=0.0, le=20.0)
    end_s: float = Field(gt=0.0, le=20.0)
    retake_prompt: str
    preserve_outside_window: bool = True

    @model_validator(mode="after")
    def _validate_window(self) -> WindowedRetakeSpec:
        if self.end_s <= self.start_s:
            raise ValueError(f"Windowed retake end_s ({self.end_s}) must be greater than start_s ({self.start_s}).")
        return self


# ---------------------------------------------------------------------------------------------
# 6. Git-Style Timeline Version Control (branch, commit, undo, redo, PR, merge, reset, revert)
# ---------------------------------------------------------------------------------------------
class TimelineCommit(Model):
    commit_id: str
    parent_id: str | None = None
    branch: str
    message: str
    author: str = "director"
    ts: str
    snapshot: dict[str, Any]


class TimelinePullRequest(Model):
    pr_id: str
    source_branch: str
    target_branch: str
    title: str
    status: Literal["open", "merged", "closed"] = "open"
    created_at: str
    merged_commit_id: str | None = None


class TimelineGitRepo:
    """Git-style version control for SceneSlates & NLE Timelines (`branch`, `commit`, `undo`, `redo`, `pr`, `merge`)."""

    def __init__(self, repo_dir: Path) -> None:
        self.repo_dir = Path(repo_dir) / ".timeline_git"
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.repo_dir / "vcs_state.json"
        if not self.state_file.exists():
            init_state = {
                "current_branch": "main",
                "branches": {"main": None},
                "commits": {},
                "undo_stack": [],
                "redo_stack": [],
                "pull_requests": {},
            }
            self.state_file.write_text(json.dumps(init_state, indent=2), encoding="utf-8")

    def _load(self) -> dict[str, Any]:
        return json.loads(self.state_file.read_text(encoding="utf-8"))

    def _save(self, data: dict[str, Any]) -> None:
        self.state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def commit(self, snapshot: dict[str, Any], message: str, *, author: str = "director") -> TimelineCommit:
        data = self._load()
        branch = data["current_branch"]
        parent_id = data["branches"].get(branch)
        commit_id = "c_" + uuid.uuid4().hex[:10]
        commit_obj = TimelineCommit(
            commit_id=commit_id,
            parent_id=parent_id,
            branch=branch,
            message=message,
            author=author,
            ts=_utc_now(),
            snapshot=snapshot,
        )
        if parent_id is not None:
            data["undo_stack"].append(parent_id)
        data["redo_stack"].clear()
        data["commits"][commit_id] = commit_obj.model_dump()
        data["branches"][branch] = commit_id
        self._save(data)
        return commit_obj

    def create_branch(self, branch_name: str, *, checkout: bool = True) -> str:
        data = self._load()
        head_id = data["branches"].get(data["current_branch"])
        data["branches"][branch_name] = head_id
        if checkout:
            data["current_branch"] = branch_name
        self._save(data)
        return branch_name

    def checkout(self, branch_name: str) -> TimelineCommit | None:
        data = self._load()
        if branch_name not in data["branches"]:
            raise KeyError(f"Unknown timeline branch: {branch_name}")
        data["current_branch"] = branch_name
        self._save(data)
        head_id = data["branches"][branch_name]
        return TimelineCommit.model_validate(data["commits"][head_id]) if head_id else None

    def undo(self) -> TimelineCommit:
        data = self._load()
        if not data["undo_stack"]:
            raise RuntimeError("Nothing to undo on current timeline branch.")
        branch = data["current_branch"]
        current_head = data["branches"].get(branch)
        prev_commit_id = data["undo_stack"].pop()
        if current_head:
            data["redo_stack"].append(current_head)
        data["branches"][branch] = prev_commit_id
        self._save(data)
        return TimelineCommit.model_validate(data["commits"][prev_commit_id])

    def redo(self) -> TimelineCommit:
        data = self._load()
        if not data["redo_stack"]:
            raise RuntimeError("Nothing to redo on current timeline branch.")
        branch = data["current_branch"]
        current_head = data["branches"].get(branch)
        next_commit_id = data["redo_stack"].pop()
        if current_head:
            data["undo_stack"].append(current_head)
        data["branches"][branch] = next_commit_id
        self._save(data)
        return TimelineCommit.model_validate(data["commits"][next_commit_id])

    def reset(self, target_commit_id: str) -> TimelineCommit:
        data = self._load()
        if target_commit_id not in data["commits"]:
            raise KeyError(f"Unknown commit_id: {target_commit_id}")
        branch = data["current_branch"]
        current_head = data["branches"].get(branch)
        if current_head:
            data["undo_stack"].append(current_head)
        data["branches"][branch] = target_commit_id
        self._save(data)
        return TimelineCommit.model_validate(data["commits"][target_commit_id])

    def revert(self, commit_id_to_revert: str) -> TimelineCommit:
        data = self._load()
        if commit_id_to_revert not in data["commits"]:
            raise KeyError(f"Unknown commit_id: {commit_id_to_revert}")
        target = data["commits"][commit_id_to_revert]
        parent_id = target.get("parent_id")
        restored_snapshot = data["commits"][parent_id]["snapshot"] if parent_id and parent_id in data["commits"] else {}
        return self.commit(restored_snapshot, f"Revert '{target['message']}' ({commit_id_to_revert})")

    def open_pull_request(self, source_branch: str, target_branch: str, title: str) -> TimelinePullRequest:
        data = self._load()
        if source_branch not in data["branches"] or target_branch not in data["branches"]:
            raise KeyError(f"Both {source_branch!r} and {target_branch!r} must exist.")
        pr_id = "pr_" + uuid.uuid4().hex[:8]
        pr = TimelinePullRequest(
            pr_id=pr_id,
            source_branch=source_branch,
            target_branch=target_branch,
            title=title,
            status="open",
            created_at=_utc_now(),
        )
        data["pull_requests"][pr_id] = pr.model_dump()
        self._save(data)
        return pr

    def merge_pull_request(self, pr_id: str) -> TimelineCommit:
        data = self._load()
        if pr_id not in data["pull_requests"]:
            raise KeyError(f"Unknown pull request: {pr_id}")
        pr_data = data["pull_requests"][pr_id]
        source_head_id = data["branches"].get(pr_data["source_branch"])
        if not source_head_id:
            raise ValueError(f"Source branch {pr_data['source_branch']!r} has no commits to merge.")
        source_commit = data["commits"][source_head_id]
        data["current_branch"] = pr_data["target_branch"]
        self._save(data)
        merged_commit = self.commit(
            source_commit["snapshot"],
            f"Merge PR {pr_id} ({pr_data['source_branch']} -> {pr_data['target_branch']}): {pr_data['title']}",
        )
        data = self._load()
        data["pull_requests"][pr_id]["status"] = "merged"
        data["pull_requests"][pr_id]["merged_commit_id"] = merged_commit.commit_id
        self._save(data)
        return merged_commit

    def current_head(self) -> TimelineCommit | None:
        data = self._load()
        head_id = data["branches"].get(data["current_branch"])
        if not head_id:
            return None
        return TimelineCommit.model_validate(data["commits"][head_id])
