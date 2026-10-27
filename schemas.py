"""Formal Pydantic V2 API contracts for the DaVinci Resolve 20 NLE MCP Server.

Enforces Spec-Driven Agentic Engineering (SDD):
- All request and response payloads inherit from `StrictContract` (`extra="forbid"`, `frozen=True`).
- Enforces DaVinci Resolve 20 Free Edition operational limits (max 3840x2160 / 2160x3840 UHD, <=60 fps).
- Enforces strict filesystem path sandboxing and HTTPS URL validation (anti-SSRF & anti-traversal).
"""

from __future__ import annotations

import ipaddress
import os
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

WORKSPACE_ROOT: Path = Path(__file__).resolve().parent
ASSET_DIR: Path = (WORKSPACE_ROOT / "generated_assets").resolve()
DEBUG_FRAMES_DIR: Path = (WORKSPACE_ROOT / "debug_frames").resolve()
RENDER_EXPORT_DIR: Path = (WORKSPACE_ROOT / "rendered_exports").resolve()

ALLOWED_MEDIA_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp4",
        ".mov",
        ".mkv",
        ".webm",
        ".wav",
        ".mp3",
        ".aac",
        ".flac",
        ".m4a",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".exr",
        ".srt",
        ".vtt",
        ".cube",
        ".drx",
    }
)


def resolve_safe_workspace_path(
    target_path: str | Path,
    allowed_root: Path = WORKSPACE_ROOT,
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve `target_path` and verify it remains strictly within `allowed_root`.

    Prevents directory traversal (`../`) and partial-prefix bypasses.
    """
    root_resolved = allowed_root.resolve()
    raw_path = Path(target_path)
    candidate = (
        raw_path.resolve() if raw_path.is_absolute() else (root_resolved / raw_path).resolve()
    )

    root_prefix = str(root_resolved) + os.sep
    if candidate != root_resolved and not str(candidate).startswith(root_prefix):
        raise ValueError(
            f"Path '{target_path}' escapes allowed directory boundary '{root_resolved}'."
        )

    if must_exist and not candidate.exists():
        raise ValueError(f"Path '{candidate}' does not exist.")

    return candidate


def validate_https_source_url(url: str) -> str:
    """Validate that a remote asset URL uses HTTPS and does not target private/loopback hosts."""
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() != "https":
        raise ValueError("Remote media URLs must use 'https://' scheme.")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Remote media URL must include a valid hostname.")

    lowered = hostname.lower()
    if lowered in {"localhost", "127.0.0.1", "::1", "0.0.0.0", "169.254.169.254"}:
        raise ValueError(f"Remote media URL host '{hostname}' is not permitted.")

    try:
        ip_obj = ipaddress.ip_address(lowered)
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_reserved
        ):
            raise ValueError(f"Remote media URL IP '{hostname}' is restricted.")
    except ValueError as exc:
        if "is restricted" in str(exc) or "is not permitted" in str(exc):
            raise

    return url.strip()


class StrictContract(BaseModel):
    """Base contract enforcing Spec-Driven Agentic Engineering rules."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class OperationStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"


class AspectRatio(StrEnum):
    LANDSCAPE = "16:9"
    PORTRAIT = "9:16"


class TrackType(StrEnum):
    VIDEO = "video"
    AUDIO = "audio"
    SUBTITLE = "subtitle"


class AudioTrackSubType(StrEnum):
    MONO = "mono"
    STEREO = "stereo"
    SURROUND_5_1 = "5.1"
    ADAPTIVE = "adaptive"


class MediaPlacementType(StrEnum):
    ALL = "all"  # Both video and audio
    VIDEO_ONLY = "video_only"  # DaVinci mediaType = 1 (B-roll / picture-only)
    AUDIO_ONLY = "audio_only"  # DaVinci mediaType = 2 (J/L cuts, SFX, music)


class CompositeMode(StrEnum):
    NORMAL = "Normal"
    ADD = "Add"
    MULTIPLY = "Multiply"
    SCREEN = "Screen"
    OVERLAY = "Overlay"
    SOFT_LIGHT = "SoftLight"
    DIFFERENCE = "Difference"


class MarkerColor(StrEnum):
    BLUE = "Blue"
    CYAN = "Cyan"
    GREEN = "Green"
    YELLOW = "Yellow"
    RED = "Red"
    PINK = "Pink"
    PURPLE = "Purple"
    FUCHSIA = "Fuchsia"
    ROSE = "Rose"
    LAVENDER = "Lavender"
    SKY = "Sky"
    MINT = "Mint"
    LEMON = "Lemon"
    SAND = "Sand"
    COCOA = "Cocoa"
    CREAM = "Cream"


# ============================================================================
# Domain 1: Media Ingest, Remote URL Download & Bin Management Contracts
# ============================================================================


class IngestMediaItem(StrictContract):
    source: Annotated[
        str,
        Field(
            min_length=1,
            max_length=2048,
            description="Local file path inside workspace OR remote https:// media URL.",
        ),
    ]
    target_bin: Annotated[
        str,
        Field(
            default="Master",
            pattern=r"^[a-zA-Z0-9_\-/ ]+$",
            description="Target Media Pool folder path (e.g., 'Video/Scene1', 'Audio/SFX').",
        ),
    ] = "Master"
    custom_name: (
        Annotated[
            str,
            Field(
                pattern=r"^[a-zA-Z0-9_\-\.]+$",
                description="Optional sanitized filename when downloading from a remote URL.",
            ),
        ]
        | None
    ) = None
    generation_id: (
        Annotated[
            str,
            Field(
                max_length=256,
                description="Optional generation/operation ID stored in clip metadata.",
            ),
        ]
        | None
    ) = None

    @field_validator("source")
    @classmethod
    def validate_source_url_or_path(cls, v: str) -> str:
        stripped = v.strip()
        if stripped.startswith(("http://", "https://")):
            return validate_https_source_url(stripped)
        resolved = resolve_safe_workspace_path(stripped, WORKSPACE_ROOT)
        if resolved.suffix.lower() not in ALLOWED_MEDIA_EXTENSIONS:
            raise ValueError(
                f"Unsupported media file extension '{resolved.suffix}'. "
                f"Allowed: {sorted(ALLOWED_MEDIA_EXTENSIONS)}"
            )
        return stripped


class IngestMediaRequest(StrictContract):
    items: Annotated[list[IngestMediaItem], Field(min_length=1, max_length=50)]


class IngestedClipMetadata(StrictContract):
    clip_name: str
    local_path: str
    bin_path: str
    duration_frames: int = Field(default=0, ge=0)
    fps: str | None = None
    resolution: str | None = None
    has_video: bool = True
    has_audio: bool = False
    generation_id: str | None = None


class IngestMediaResult(StrictContract):
    status: OperationStatus
    imported_clips: list[IngestedClipMetadata] = Field(default_factory=list)
    error_message: str | None = None


class ManageBinsRequest(StrictContract):
    action: Annotated[str, Field(pattern=r"^(create|list|switch|delete)$")]
    bin_path: Annotated[
        str,
        Field(
            default="Master",
            pattern=r"^[a-zA-Z0-9_\-/ ]+$",
            description="Bin path relative to root folder (e.g., 'Video/B-Roll' or 'Audio/Foley').",
        ),
    ] = "Master"


class ManageBinsResult(StrictContract):
    status: OperationStatus
    active_bin: str = "Master"
    bins: list[str] = Field(default_factory=list)
    clip_names_in_active_bin: list[str] = Field(default_factory=list)
    error_message: str | None = None


class CreateSubclipRequest(StrictContract):
    clip_name: Annotated[str, Field(min_length=1, max_length=256)]
    mark_in: Annotated[int, Field(ge=0, description="Source start frame (inclusive).")]
    mark_out: Annotated[int, Field(gt=0, description="Source end frame (inclusive).")]
    subclip_name: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_\- ]+$")] | None = None
    target_bin: Annotated[
        str,
        Field(default="Master", pattern=r"^[a-zA-Z0-9_\-/ ]+$"),
    ] = "Master"

    @model_validator(mode="after")
    def validate_frame_bounds(self) -> CreateSubclipRequest:
        if self.mark_out <= self.mark_in:
            raise ValueError("mark_out must be strictly greater than mark_in.")
        return self


class CreateSubclipResult(StrictContract):
    status: OperationStatus
    subclip_name: str | None = None
    duration_frames: int = 0
    bin_path: str = "Master"
    error_message: str | None = None


class SetClipMetadataRequest(StrictContract):
    clip_name: Annotated[str, Field(min_length=1, max_length=256)]
    scene: str | None = None
    shot: str | None = None
    take: str | None = None
    keywords: str | None = None
    comments: str | None = None
    clip_color: MarkerColor | None = None
    generation_id: str | None = None


class SetClipMetadataResult(StrictContract):
    status: OperationStatus
    clip_name: str
    updated_fields: dict[str, str] = Field(default_factory=dict)
    error_message: str | None = None


# ============================================================================
# Domain 2: Project & Timeline Contracts (Resolve 20 Free Edition Compliant)
# ============================================================================


class ProjectConfigRequest(StrictContract):
    project_name: Annotated[
        str,
        Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_\- ]+$"),
    ]
    width: Annotated[
        int,
        Field(
            default=1920,
            ge=256,
            le=3840,
            description="Timeline width in pixels (max 3840 UHD in DaVinci Resolve 20 Free Edition).",
        ),
    ] = 1920
    height: Annotated[
        int,
        Field(
            default=1080,
            ge=256,
            le=3840,
            description="Timeline height in pixels (max 3840 for vertical UHD in Resolve 20 Free Edition).",
        ),
    ] = 1080
    frame_rate: Annotated[
        str,
        Field(
            default="24",
            pattern=r"^(23\.976|24|25|29\.97|30|50|59\.94|60)$",
            description="Frame rate up to 60 fps (Resolve 20 Free Edition ceiling).",
        ),
    ] = "24"


class ProjectConfigResult(StrictContract):
    status: OperationStatus
    project_name: str | None = None
    width: int = 1920
    height: int = 1080
    frame_rate: str = "24"
    resolve_edition: str = "Free"
    error_message: str | None = None


class TimelineActionRequest(StrictContract):
    action: Annotated[str, Field(pattern=r"^(create|switch|duplicate)$")]
    timeline_name: Annotated[
        str,
        Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_\- ]+$"),
    ]
    new_name: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_\- ]+$")] | None = None
    start_timecode: Annotated[
        str,
        Field(default="01:00:00:00", pattern=r"^\d{2}:\d{2}:\d{2}[:;]\d{2}$"),
    ] = "01:00:00:00"


class TimelineActionResult(StrictContract):
    status: OperationStatus
    action: str
    timeline_name: str | None = None
    start_timecode: str = "01:00:00:00"
    total_timelines_in_project: int = 0
    error_message: str | None = None


class TrackManagementRequest(StrictContract):
    action: Annotated[
        str,
        Field(pattern=r"^(add|delete|rename|enable|disable|lock|unlock)$"),
    ]
    track_type: TrackType
    track_index: Annotated[int, Field(default=1, ge=1, le=64)] = 1
    audio_sub_type: AudioTrackSubType = AudioTrackSubType.STEREO
    track_name: Annotated[str, Field(max_length=64, pattern=r"^[a-zA-Z0-9_\- ]+$")] | None = None


class TrackManagementResult(StrictContract):
    status: OperationStatus
    action: str
    track_type: TrackType
    track_index: int
    track_count: int = 0
    track_name: str | None = None
    error_message: str | None = None


class TimelineGapInfo(StrictContract):
    track_type: TrackType
    track_index: int = Field(ge=1)
    gap_start_frame: int = Field(ge=0)
    gap_end_frame: int = Field(ge=0)
    duration_frames: int = Field(ge=1)


class TimelineClipDetail(StrictContract):
    item_index: int = Field(ge=0)
    name: str
    track_type: TrackType
    track_index: int = Field(ge=1)
    record_start_frame: int = Field(ge=0)
    record_end_frame: int = Field(ge=0)
    duration_frames: int = Field(ge=0)
    source_start_frame: int | None = None
    source_end_frame: int | None = None
    enabled: bool = True
    properties: dict[str, str | float | int] = Field(default_factory=dict)
    markers: dict[str, dict[str, Any]] = Field(default_factory=dict)


class TimelineTrackDetail(StrictContract):
    track_type: TrackType
    track_index: int = Field(ge=1)
    track_name: str
    enabled: bool = True
    locked: bool = False
    clips: list[TimelineClipDetail] = Field(default_factory=list)


class TimelineStructureResult(StrictContract):
    status: OperationStatus
    timeline_name: str | None = None
    start_frame: int = 0
    end_frame: int = 0
    duration_frames: int = 0
    frame_rate: str | None = None
    resolution: str | None = None
    current_timecode: str | None = None
    tracks: list[TimelineTrackDetail] = Field(default_factory=list)
    detected_gaps: list[TimelineGapInfo] = Field(default_factory=list)
    markers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    error_message: str | None = None


# ============================================================================
# Domain 3: Precision NLE Placement, Trims & Transform Contracts
# ============================================================================


class ClipPlacementSpec(StrictContract):
    clip_name: Annotated[
        str,
        Field(
            min_length=1,
            max_length=1024,
            description="Clip name in Media Pool OR local path inside workspace to auto-import.",
        ),
    ]
    track_index: Annotated[int, Field(default=1, ge=1, le=64)] = 1
    media_type: MediaPlacementType = MediaPlacementType.ALL
    start_frame: Annotated[int, Field(ge=0, description="Optional source in-point frame.")] | None = (
        None
    )
    end_frame: Annotated[int, Field(gt=0, description="Optional source out-point frame.")] | None = (
        None
    )
    record_frame: (
        Annotated[
            int,
            Field(
                ge=0,
                description="Exact timeline frame for 3-point/J-cut/L-cut/overlay placement. Omit to append at track end.",
            ),
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def check_source_range(self) -> ClipPlacementSpec:
        if (
            self.start_frame is not None
            and self.end_frame is not None
            and self.end_frame <= self.start_frame
        ):
            raise ValueError("end_frame must be greater than start_frame.")
        return self


class PlaceClipsRequest(StrictContract):
    timeline_name: (
        Annotated[
            str,
            Field(
                pattern=r"^[a-zA-Z0-9_\- ]+$",
                description="Optional timeline name to create or switch to before placing clips.",
            ),
        ]
        | None
    ) = None
    placements: Annotated[list[ClipPlacementSpec], Field(min_length=1, max_length=100)]


class PlacedClipInfo(StrictContract):
    clip_name: str
    track_index: int
    media_type: MediaPlacementType
    record_start_frame: int
    record_end_frame: int
    duration_frames: int


class PlaceClipsResult(StrictContract):
    status: OperationStatus
    timeline_name: str | None = None
    placed_clips: list[PlacedClipInfo] = Field(default_factory=list)
    total_timeline_frames: int = 0
    error_message: str | None = None


class TransformClipRequest(StrictContract):
    track_type: TrackType = TrackType.VIDEO
    track_index: Annotated[int, Field(default=1, ge=1, le=64)] = 1
    item_index: Annotated[int, Field(default=0, ge=0)] = 0
    pan: float | None = None
    tilt: float | None = None
    zoom_x: Annotated[float, Field(gt=0.0, le=100.0)] | None = None
    zoom_y: Annotated[float, Field(gt=0.0, le=100.0)] | None = None
    rotation_angle: Annotated[float, Field(ge=-360.0, le=360.0)] | None = None
    crop_left: Annotated[float, Field(ge=0.0, le=3840.0)] | None = None
    crop_right: Annotated[float, Field(ge=0.0, le=3840.0)] | None = None
    crop_top: Annotated[float, Field(ge=0.0, le=3840.0)] | None = None
    crop_bottom: Annotated[float, Field(ge=0.0, le=3840.0)] | None = None
    opacity: Annotated[float, Field(ge=0.0, le=100.0)] | None = None
    composite_mode: CompositeMode | None = None
    speed_percent: Annotated[float, Field(gt=1.0, le=1000.0)] | None = None
    stabilize: bool = False
    smart_reframe: bool = Field(
        default=False,
        description="Attempts Studio SmartReframe; automatically falls back to 9:16 center-crop zoom (1.778x) on Resolve 20 Free.",
    )


class TransformClipResult(StrictContract):
    status: OperationStatus
    clip_name: str | None = None
    applied_properties: dict[str, str | float | bool] = Field(default_factory=dict)
    fallback_notes: list[str] = Field(default_factory=list)
    error_message: str | None = None


class ManageTimelineItemRequest(StrictContract):
    action: Annotated[str, Field(pattern=r"^(delete|enable|disable|set_color)$")]
    track_type: TrackType = TrackType.VIDEO
    track_index: Annotated[int, Field(default=1, ge=1, le=64)] = 1
    item_index: Annotated[int, Field(default=0, ge=0)] = 0
    ripple_delete: bool = False
    clip_color: MarkerColor | None = None


class ManageTimelineItemResult(StrictContract):
    status: OperationStatus
    action: str
    clip_name: str | None = None
    total_timeline_frames: int = 0
    error_message: str | None = None


# ============================================================================
# Domain 4: Titles, Generators & Subtitles Contracts (Resolve 20 Free Aware)
# ============================================================================


class InsertTitleOrGeneratorRequest(StrictContract):
    element_type: Annotated[str, Field(pattern=r"^(title|generator|adjustment_clip)$")]
    template_name: Annotated[
        str,
        Field(
            default="Text+",
            min_length=1,
            max_length=128,
            description="E.g., 'Text+', 'Lower Third', 'Scroll', 'Solid Color', 'Adjustment Clip'.",
        ),
    ] = "Text+"
    text_content: Annotated[str, Field(max_length=500)] | None = None
    track_index: Annotated[int, Field(default=2, ge=1, le=64)] = 2
    duration_frames: Annotated[int, Field(default=120, ge=1, le=100000)] = 120


class InsertTitleOrGeneratorResult(StrictContract):
    status: OperationStatus
    element_type: str
    template_name: str
    track_index: int
    record_start_frame: int = 0
    record_end_frame: int = 0
    error_message: str | None = None


class SubtitleLineSpec(StrictContract):
    start_frame: Annotated[int, Field(ge=0)]
    end_frame: Annotated[int, Field(gt=0)]
    text: Annotated[str, Field(min_length=1, max_length=300)]

    @model_validator(mode="after")
    def validate_subtitle_bounds(self) -> SubtitleLineSpec:
        if self.end_frame <= self.start_frame:
            raise ValueError("Subtitle end_frame must be greater than start_frame.")
        return self


class ManageSubtitlesRequest(StrictContract):
    mode: Annotated[
        str,
        Field(
            pattern=r"^(import_srt|generate_srt_from_lines|auto_transcribe)$",
            description="'import_srt' or 'generate_srt_from_lines' work natively in Resolve 20 Free Edition.",
        ),
    ]
    srt_file_path: str | None = Field(
        default=None,
        description="Workspace-relative path to an .srt file when mode='import_srt'.",
    )
    subtitle_lines: list[SubtitleLineSpec] = Field(
        default_factory=list,
        description="Structured caption lines converted to .srt and imported onto the Subtitle/Video track.",
    )
    fps: Annotated[float, Field(default=24.0, gt=0.0, le=60.0)] = 24.0

    @field_validator("srt_file_path")
    @classmethod
    def validate_srt_path(cls, v: str | None) -> str | None:
        if v is None:
            return None
        resolved = resolve_safe_workspace_path(v, WORKSPACE_ROOT)
        if resolved.suffix.lower() not in {".srt", ".vtt"}:
            raise ValueError("Subtitle file must have .srt or .vtt extension.")
        return v


class ManageSubtitlesResult(StrictContract):
    status: OperationStatus
    mode_used: str
    srt_path: str | None = None
    captions_count: int = 0
    fallback_note: str | None = None
    error_message: str | None = None


# ============================================================================
# Domain 5: Color Grading, LUTs, ASC-CDL & Continuity Contracts
# ============================================================================


class ColorGradeRequest(StrictContract):
    track_index: Annotated[int, Field(default=1, ge=1, le=64)] = 1
    item_index: Annotated[int, Field(default=0, ge=0)] = 0
    node_index: Annotated[int, Field(default=1, ge=1, le=64)] = 1
    lut_path: str | None = Field(
        default=None,
        description="Optional path to a .cube LUT file.",
    )
    drx_grade_path: str | None = Field(
        default=None,
        description="Optional path to a .drx gallery still grade file.",
    )
    cdl_slope: Annotated[str, Field(pattern=r"^\d+(\.\d+)? \d+(\.\d+)? \d+(\.\d+)?$")] | None = (
        None
    )
    cdl_offset: (
        Annotated[str, Field(pattern=r"^-?\d+(\.\d+)? -?\d+(\.\d+)? -?\d+(\.\d+)?$")] | None
    ) = None
    cdl_power: Annotated[str, Field(pattern=r"^\d+(\.\d+)? \d+(\.\d+)? \d+(\.\d+)?$")] | None = (
        None
    )
    cdl_saturation: Annotated[str, Field(pattern=r"^\d+(\.\d+)?$")] | None = None
    copy_grade_from_item_index: Annotated[int, Field(ge=0)] | None = None
    reset_all_grades: bool = False


class ColorGradeResult(StrictContract):
    status: OperationStatus
    clip_name: str | None = None
    node_index: int = 1
    applied_operations: list[str] = Field(default_factory=list)
    error_message: str | None = None


# ============================================================================
# Domain 6: Fairlight Multi-Track Audio, Sync & Ducking Contracts
# ============================================================================


class PlaceSyncedAudioRequest(StrictContract):
    audio_clip_name: Annotated[
        str,
        Field(
            min_length=1,
            max_length=1024,
            description="Audio clip name in Media Pool OR local/HTTPS source to auto-ingest.",
        ),
    ]
    audio_role: Annotated[
        str,
        Field(
            default="dialogue",
            pattern=r"^(dialogue|foley_sfx|music_bed)$",
            description="Maps by default to A1 (dialogue), A2 (foley_sfx), or A3 (music_bed).",
        ),
    ] = "dialogue"
    audio_track_index: Annotated[int, Field(ge=1, le=64)] | None = None
    sync_to_video_track_index: Annotated[int, Field(default=1, ge=1, le=64)] = 1
    sync_to_video_item_index: (
        Annotated[
            int,
            Field(
                ge=0,
                description="If set, aligns the audio start frame with this video clip's record_start_frame.",
            ),
        ]
        | None
    ) = None
    frame_offset: int = Field(
        default=0,
        description="Frame offset relative to target sync point (negative for J-cut audio lead-in, positive for L-cut/delay).",
    )
    record_frame: Annotated[int, Field(ge=0)] | None = None
    source_start_frame: Annotated[int, Field(ge=0)] | None = None
    source_end_frame: Annotated[int, Field(gt=0)] | None = None


class PlaceSyncedAudioResult(StrictContract):
    status: OperationStatus
    audio_clip_name: str
    audio_role: str
    audio_track_index: int
    record_start_frame: int
    record_end_frame: int
    error_message: str | None = None


class ConfigureAudioClipRequest(StrictContract):
    audio_track_index: Annotated[int, Field(default=1, ge=1, le=64)] = 1
    item_index: Annotated[int, Field(default=0, ge=0)] = 0
    volume_db: Annotated[float, Field(ge=-96.0, le=24.0)] | None = None
    pan: Annotated[float, Field(ge=-100.0, le=100.0)] | None = None
    voice_isolation_enabled: bool | None = Field(
        default=None,
        description="Attempts Resolve Studio VoiceIsolation; returns graceful Free-edition note if unsupported.",
    )
    duck_against_dialogue_track: (
        Annotated[
            int,
            Field(
                ge=1,
                le=64,
                description="Free-Edition auto-ducking: lowers gain when dialogue clips are present on the specified audio track.",
            ),
        ]
        | None
    ) = None
    duck_reduction_db: Annotated[float, Field(default=-12.0, ge=-48.0, le=0.0)] = -12.0


class ConfigureAudioClipResult(StrictContract):
    status: OperationStatus
    clip_name: str | None = None
    audio_track_index: int
    applied_settings: dict[str, float | bool | str] = Field(default_factory=dict)
    fallback_notes: list[str] = Field(default_factory=list)
    error_message: str | None = None


# ============================================================================
# Domain 7: Visual Critic Inspection, Markers & Render Queue Contracts
# ============================================================================


class InspectFramesRequest(StrictContract):
    timecodes: list[str] = Field(
        default_factory=list,
        description="List of timecodes ('01:00:02:12') to sample. If empty, exports the current playhead frame.",
    )
    frame_numbers: list[int] = Field(
        default_factory=list,
        description="Optional list of timeline frame numbers to convert to timecode and export.",
    )
    output_dir: str = Field(
        default="./debug_frames",
        description="Workspace-relative output directory for exported PNG stills.",
    )
    filename_prefix: Annotated[
        str,
        Field(default="playhead_frame", pattern=r"^[a-zA-Z0-9_\-]+$"),
    ] = "playhead_frame"

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: str) -> str:
        resolve_safe_workspace_path(v, WORKSPACE_ROOT)
        return v

    @field_validator("timecodes")
    @classmethod
    def validate_timecode_strings(cls, v: list[str]) -> list[str]:
        import re

        pattern = re.compile(r"^\d{2}:\d{2}:\d{2}[:;]\d{2}$")
        for tc in v:
            if not pattern.match(tc):
                raise ValueError(f"Invalid SMPTE timecode format '{tc}' (expected HH:MM:SS:FF).")
        return v


class ExportedFrameInfo(StrictContract):
    timecode: str
    frame_path: str


class InspectFramesResult(StrictContract):
    status: OperationStatus
    exported_frames: list[ExportedFrameInfo] = Field(default_factory=list)
    frame_path: str | None = Field(
        default=None,
        description="Primary exported frame path (backward-compatible with single-frame callers).",
    )
    error_message: str | None = None


class MarkerRequest(StrictContract):
    action: Annotated[str, Field(pattern=r"^(add|delete_by_color|delete_at_frame|list)$")]
    target: Annotated[str, Field(default="timeline", pattern=r"^(timeline|clip)$")] = "timeline"
    frame_id: Annotated[int, Field(default=0, ge=0)] = 0
    color: MarkerColor = MarkerColor.RED
    name: Annotated[str, Field(default="Critic Note", max_length=128)] = "Critic Note"
    note: Annotated[str, Field(default="", max_length=1000)] = ""
    duration: Annotated[int, Field(default=1, ge=1, le=100000)] = 1
    custom_data: Annotated[
        str,
        Field(default="", max_length=512, description="Stores generation/operation IDs."),
    ] = ""
    track_type: TrackType = TrackType.VIDEO
    track_index: Annotated[int, Field(default=1, ge=1, le=64)] = 1
    item_index: Annotated[int, Field(default=0, ge=0)] = 0


class MarkerResult(StrictContract):
    status: OperationStatus
    action: str
    target: str
    markers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    error_message: str | None = None


class RenderTimelineRequest(StrictContract):
    preset_name: Annotated[str, Field(max_length=128)] | None = None
    target_dir: str = Field(
        default="./rendered_exports",
        description="Workspace-relative directory where rendered master will be written.",
    )
    custom_filename: Annotated[
        str,
        Field(default="master_export", pattern=r"^[a-zA-Z0-9_\-]+$"),
    ] = "master_export"
    format_extension: Annotated[str, Field(default="mp4", pattern=r"^(mp4|mov|wav)$")] = "mp4"
    video_codec: Annotated[str, Field(default="H264", pattern=r"^(H264|H265|ProRes422HQ)$")] = (
        "H264"
    )
    width: (
        Annotated[
            int,
            Field(ge=256, le=3840, description="Max 3840 UHD on Resolve 20 Free Edition."),
        ]
        | None
    ) = None
    height: (
        Annotated[
            int,
            Field(ge=256, le=3840, description="Max 3840 UHD on Resolve 20 Free Edition."),
        ]
        | None
    ) = None
    mark_in: Annotated[int, Field(ge=0)] | None = None
    mark_out: Annotated[int, Field(gt=0)] | None = None
    wait_for_completion: bool = True

    @field_validator("target_dir")
    @classmethod
    def validate_render_target_dir(cls, v: str) -> str:
        resolve_safe_workspace_path(v, WORKSPACE_ROOT)
        return v


class RenderTimelineResult(StrictContract):
    status: OperationStatus
    job_id: str | None = None
    output_file_path: str | None = None
    render_status: str | None = None
    completion_percentage: int = 0
    error_message: str | None = None


# ============================================================================
# Domain 8: Generic Casting, Location Scout & Shot Breakdown Contracts
# ============================================================================


class CharacterCastSpec(StrictContract):
    """Locked character identity managed by the Casting Director across all episodes."""

    character_id: Annotated[str, Field(pattern=r"^CHAR_[A-Z0-9_]+$")]
    name: str
    age: int = Field(ge=1, le=100)
    role_description: str
    immutable_physical_prompt: Annotated[
        str,
        Field(
            min_length=20,
            description="Exact facial geometry, skin tone, hair, facial hair, and build that never changes.",
        ),
    ]
    wardrobe_by_state: dict[str, str] = Field(
        description="Map of wardrobe_state_key -> exact clothing & accessory description."
    )
    anchor_sheet_id: str = Field(
        description="ID of the Master Cast Reference Sheet containing this character."
    )


class LocationScoutSpec(StrictContract):
    """Locked architectural & lighting specification managed by the Location Scout."""

    location_id: Annotated[str, Field(pattern=r"^LOC_[A-Z0-9_]+$")]
    name: str
    setting_type: Annotated[str, Field(pattern=r"^(EXT|INT)$")]
    time_of_day: str
    architectural_prompt: Annotated[
        str,
        Field(
            min_length=20,
            description="Exact walls, floor, props, background elements, and spatial layout.",
        ),
    ]
    lighting_and_lens_prompt: str
    anchor_plate_id: str = Field(
        description="ID of the Master Location Reference Plate for this setting."
    )


class HeroPropSpec(StrictContract):
    """Locked hero prop specification for cross-shot continuity."""

    prop_id: Annotated[str, Field(pattern=r"^PROP_[A-Z0-9_]+$")]
    name: str
    visual_prompt: str
    anchor_sheet_id: str = ""


class AnchorSheetSpec(StrictContract):
    """Specification for a Master Reference Sheet generated before any scene shots."""

    anchor_id: str
    image_name: str
    aspect_ratio: AspectRatio = AspectRatio.PORTRAIT
    prompt: str


class ShotAudioSpec(StrictContract):
    """Dialogue, Voiceover, and Foley SFX cue attached to a production shot."""

    speaker: str | None = None
    dialogue_hinglish: str | None = None
    subtitle_text: str | None = None
    j_cut_offset_frames: int = Field(
        default=0,
        description="Negative value starts dialogue before the visual cut (J-cut).",
    )
    foley_sfx_prompt: str
    music_bed_cue: str


class ShotBreakdownSpec(StrictContract):
    """Complete Spec-Driven definition for a single production shot (microdrama, 20-min episode, or film)."""

    shot_id: Annotated[str, Field(pattern=r"^[A-Z0-9_]{4,64}$")]
    episode: Annotated[int, Field(ge=1, le=999)]
    scene_id: str | None = None
    shot_number: Annotated[int, Field(ge=1, le=9999)]
    image_name: str
    duration_seconds: Annotated[int, Field(ge=2, le=16)] = 5
    duration_frames: Annotated[int, Field(ge=24, le=960)] = 120
    location_id: str
    character_ids: list[str]
    wardrobe_state_by_character: dict[str, str]
    prop_ids: list[str] = Field(default_factory=list)
    camera_framing_and_motion: str
    action_beat_prompt: str
    inherit_previous_shot_reference: bool = True
    audio: ShotAudioSpec


class CompiledShotPrompt(StrictContract):
    """Deterministic output compiled from a Casting & Location Bible for a shot."""

    shot_id: str
    episode: int
    image_name: str
    aspect_ratio: AspectRatio
    full_visual_prompt: str
    reference_anchor_ids: list[str]
    reference_image_paths: list[str] = Field(
        default_factory=list,
        description="Reference image paths passed to keyframe/video generation.",
    )
    duration_frames: int
    audio: ShotAudioSpec

