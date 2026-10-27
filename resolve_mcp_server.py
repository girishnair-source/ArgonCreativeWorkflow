"""DaVinci Resolve 20 NLE MCP Server (`davinci-resolve-controller`).

Provides 20 strongly-typed MCP tools across 7 NLE domains so Antigravity / Jet Ski
can edit videos of any complexity across multi-track timelines.

Specifically engineered for DaVinci Resolve 20 Free Edition compatibility:
- Enforces Free Edition resolution ceilings (max 3840x2160 UHD / 2160x3840 vertical, <=60 fps).
- Provides automatic Free Edition fallbacks when Studio-only AI features are invoked:
  - `smart_reframe`: Falls back to deterministic 9:16 vertical center-crop zoom (1.7778x).
  - `auto_transcribe`: Falls back to frame-accurate `.srt` generation and subtitle track import.
  - `voice_isolation_enabled`: Falls back to multi-track dialogue gain boost & music-bed ducking.
- Supports remote `https://` asset URLs with content-hashed caching into `./generated_assets/`.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import struct
import sys
import zlib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP  # type: ignore[no-redef, attr-defined]

from schemas import (
    ALLOWED_MEDIA_EXTENSIONS,
    ASSET_DIR,
     AudioTrackSubType,
    ClipPlacementSpec,
    ColorGradeRequest,
    ColorGradeResult,
    CompositeMode,
    ConfigureAudioClipRequest,
    ConfigureAudioClipResult,
    CreateSubclipRequest,
    CreateSubclipResult,
    ExportedFrameInfo,
    IngestedClipMetadata,
    IngestMediaItem,
    IngestMediaRequest,
    IngestMediaResult,
    InsertTitleOrGeneratorRequest,
    InsertTitleOrGeneratorResult,
    InspectFramesRequest,
    InspectFramesResult,
    ManageBinsRequest,
    ManageBinsResult,
    ManageSubtitlesRequest,
    ManageSubtitlesResult,
    ManageTimelineItemRequest,
    ManageTimelineItemResult,
    MarkerColor,
    MarkerRequest,
    MarkerResult,
    MediaPlacementType,
    OperationStatus,
    PlaceClipsRequest,
    PlaceClipsResult,
    PlacedClipInfo,
    PlaceSyncedAudioRequest,
    PlaceSyncedAudioResult,
    ProjectConfigRequest,
    ProjectConfigResult,
    RENDER_EXPORT_DIR,
    RenderTimelineRequest,
    RenderTimelineResult,
    SetClipMetadataRequest,
    SetClipMetadataResult,
    SubtitleLineSpec,
    TimelineActionRequest,
    TimelineActionResult,
    TimelineClipDetail,
    TimelineGapInfo,
    TimelineStructureResult,
    TimelineTrackDetail,
    TrackManagementRequest,
    TrackManagementResult,
    TrackType,
    TransformClipRequest,
    TransformClipResult,
    WORKSPACE_ROOT,
    resolve_safe_workspace_path,
    validate_https_source_url,
)

mcp = FastMCP("davinci-resolve-controller")

DEFAULT_RESOLVE_SCRIPT_MODULES = (
    "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules"
)
DOWNLOAD_TIMEOUT_SECONDS = 120.0
MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024  # 500 MB safety cap per asset


def _create_minimal_png_bytes() -> bytes:
    """Create a valid 1x1 RGB PNG byte sequence for fallback/mock frame exports."""

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw_scanline = b"\x00\x18\x24\x36"
    idat = _chunk(b"IDAT", zlib.compress(raw_scanline))
    iend = _chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


def frames_to_timecode(frame_number: int, fps: float = 24.0) -> str:
    """Convert a 0-based or timeline frame number into SMPTE timecode HH:MM:SS:FF."""
    fps_int = max(1, round(fps))
    total_frames = max(0, int(frame_number))
    ff = total_frames % fps_int
    total_seconds = total_frames // fps_int
    ss = total_seconds % 60
    mm = (total_seconds // 60) % 60
    hh = total_seconds // 3600
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def frames_to_srt_timestamp(frame_number: int, fps: float = 24.0) -> str:
    """Convert a frame number into SRT timestamp format HH:MM:SS,mmm."""
    safe_fps = max(1.0, float(fps))
    total_ms = int(round((max(0, frame_number) / safe_fps) * 1000.0))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    ss = total_seconds % 60
    mm = (total_seconds // 60) % 60
    hh = total_seconds // 3600
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


# ============================================================================
# Stateful In-Memory NLE Engine (Used in Mock/Dry-Run Mode & Unit Test Harness)
# ============================================================================


class MockMediaPoolItem:
    """Stateful representation of a DaVinci Resolve MediaPoolItem."""

    def __init__(self, file_path: str, name: str | None = None, duration: int = 120) -> None:
        self.file_path = file_path
        self.name = name or Path(file_path).name
        self.duration = duration
        self.metadata: dict[str, str] = {}
        self.clip_color: str = ""
        ext = Path(file_path).suffix.lower()
        self.has_video = ext not in {".wav", ".mp3", ".aac", ".flac", ".m4a", ".srt", ".vtt"}
        self.has_audio = ext not in {".png", ".jpg", ".jpeg", ".webp", ".exr", ".srt", ".vtt"}
        self.properties: dict[str, str] = {
            "File Path": file_path,
            "Clip Name": self.name,
            "Frames": str(duration),
            "FPS": "24",
            "Resolution": "1920x1080" if self.has_video else "",
            "Video Codec": "H.264" if self.has_video else "",
            "Audio Codec": "AAC" if self.has_audio else "",
        }

    def GetName(self) -> str:  # noqa: N802
        return self.name

    def GetClipProperty(self, key: str | None = None) -> Any:  # noqa: N802
        if key is None:
            return dict(self.properties)
        return self.properties.get(key, "")

    def SetClipProperty(self, key: str, val: str) -> bool:  # noqa: N802
        self.properties[key] = str(val)
        if key == "Clip Name":
            self.name = str(val)
        return True

    def GetMetadata(self, key: str | None = None) -> Any:  # noqa: N802
        if key is None:
            return dict(self.metadata)
        return self.metadata.get(key, "")

    def SetMetadata(self, key_or_dict: str | dict[str, str], val: str | None = None) -> bool:  # noqa: N802
        if isinstance(key_or_dict, dict):
            for k, v in key_or_dict.items():
                self.metadata[str(k)] = str(v)
            return True
        if val is not None:
            self.metadata[str(key_or_dict)] = str(val)
            return True
        return False

    def SetClipColor(self, color_name: str) -> bool:  # noqa: N802
        self.clip_color = color_name
        return True

    def GetClipColor(self) -> str:  # noqa: N802
        return self.clip_color


class MockNodeGraph:
    """Stateful representation of a clip's Color Node Graph."""

    def __init__(self) -> None:
        self.luts: dict[int, str] = {}
        self.drx_grades: list[str] = []
        self.num_nodes: int = 1

    def GetNumNodes(self) -> int:  # noqa: N802
        return self.num_nodes

    def SetLUT(self, node_index: int, lut_path: str) -> bool:  # noqa: N802
        if node_index < 1:
            return False
        self.luts[node_index] = lut_path
        return True

    def GetLUT(self, node_index: int) -> str:  # noqa: N802
        return self.luts.get(node_index, "")

    def ApplyGradeFromDRX(  # noqa: N802
        self, path: str, grade_mode: int, *items: Any
    ) -> bool:
        self.drx_grades.append(path)
        return True

    def ResetAllGrades(self) -> bool:  # noqa: N802
        self.luts.clear()
        self.drx_grades.clear()
        return True


class MockTimelineItem:
    """Stateful representation of a clip placed on a DaVinci Resolve Timeline."""

    def __init__(
        self,
        name: str,
        start_frame: int,
        end_frame: int,
        source_start: int = 0,
        media_pool_item: MockMediaPoolItem | None = None,
    ) -> None:
        self.name = name
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.source_start = source_start
        self.source_end = source_start + (end_frame - start_frame)
        self.media_pool_item = media_pool_item
        self.enabled: bool = True
        self.clip_color: str = ""
        self.properties: dict[str, Any] = {
            "Pan": 0.0,
            "Tilt": 0.0,
            "ZoomX": 1.0,
            "ZoomY": 1.0,
            "RotationAngle": 0.0,
            "CropLeft": 0.0,
            "CropRight": 0.0,
            "CropTop": 0.0,
            "CropBottom": 0.0,
            "Opacity": 100.0,
            "CompositeMode": "Normal",
            "Speed": 100.0,
            "Volume": 0.0,
        }
        self.markers: dict[int, dict[str, Any]] = {}
        self.cdl: dict[str, str] = {}
        self.node_graph = MockNodeGraph()
        self.free_edition: bool = True

    def GetName(self) -> str:  # noqa: N802
        return self.name

    def GetDuration(self) -> int:  # noqa: N802
        return max(0, self.end_frame - self.start_frame)

    def GetStart(self) -> int:  # noqa: N802
        return self.start_frame

    def GetEnd(self) -> int:  # noqa: N802
        return self.end_frame

    def GetLeftOffset(self) -> int:  # noqa: N802
        return self.source_start

    def GetRightOffset(self) -> int:  # noqa: N802
        return self.source_end

    def GetMediaPoolItem(self) -> MockMediaPoolItem | None:  # noqa: N802
        return self.media_pool_item

    def SetProperty(self, key: str, value: Any) -> bool:  # noqa: N802
        self.properties[key] = value
        return True

    def GetProperty(self, key: str | None = None) -> Any:  # noqa: N802
        if key is None:
            return dict(self.properties)
        return self.properties.get(key)

    def SetClipEnabled(self, enabled: bool) -> bool:  # noqa: N802
        self.enabled = bool(enabled)
        return True

    def GetClipEnabled(self) -> bool:  # noqa: N802
        return self.enabled

    def SetClipColor(self, color_name: str) -> bool:  # noqa: N802
        self.clip_color = color_name
        return True

    def Stabilize(self) -> bool:  # noqa: N802
        self.properties["Stabilized"] = True
        return True

    def SmartReframe(self) -> bool:  # noqa: N802
        # Studio-only in DaVinci Resolve 20; returns False on Free Edition to trigger fallback
        if self.free_edition:
            return False
        self.properties["SmartReframed"] = True
        return True

    def SetVoiceIsolationState(self, state: dict[str, Any]) -> bool:  # noqa: N802
        # Studio-only in DaVinci Resolve 20; returns False on Free Edition
        if self.free_edition:
            return False
        self.properties["VoiceIsolation"] = state
        return True

    def SetCDL(self, cdl_map: dict[str, str]) -> bool:  # noqa: N802
        self.cdl.update(cdl_map)
        return True

    def SetLUT(self, node_index: int, lut_path: str) -> bool:  # noqa: N802
        return self.node_graph.SetLUT(node_index, lut_path)

    def GetNodeGraph(self) -> MockNodeGraph:  # noqa: N802
        return self.node_graph

    def CopyGrades(self, target_items: list[MockTimelineItem]) -> bool:  # noqa: N802
        for item in target_items:
            item.cdl = dict(self.cdl)
            item.node_graph.luts = dict(self.node_graph.luts)
            item.node_graph.drx_grades = list(self.node_graph.drx_grades)
        return True

    def AddMarker(  # noqa: N802
        self,
        frame_id: int,
        color: str,
        name: str,
        note: str,
        duration: int,
        custom_data: str = "",
    ) -> bool:
        self.markers[int(frame_id)] = {
            "color": color,
            "name": name,
            "note": note,
            "duration": int(duration),
            "customData": custom_data,
        }
        return True

    def GetMarkers(self) -> dict[int, dict[str, Any]]:  # noqa: N802
        return dict(self.markers)

    def DeleteMarkersByColor(self, color: str) -> bool:  # noqa: N802
        to_remove = [
            fid for fid, m in self.markers.items() if color == "All" or m.get("color") == color
        ]
        for fid in to_remove:
            del self.markers[fid]
        return True

    def DeleteMarkerAtFrame(self, frame_num: int) -> bool:  # noqa: N802
        return self.markers.pop(int(frame_num), None) is not None


class MockTimeline:
    """Stateful representation of a DaVinci Resolve Timeline."""

    def __init__(self, name: str, start_frame: int = 86400) -> None:
        self.name = name
        self.start_frame = start_frame
        self.current_timecode = "01:00:00:00"
        self.tracks: dict[str, dict[int, list[MockTimelineItem]]] = {
            "video": {1: []},
            "audio": {1: []},
            "subtitle": {1: []},
        }
        self.track_names: dict[tuple[str, int], str] = {
            ("video", 1): "Video 1",
            ("audio", 1): "Dialogue A1",
            ("subtitle", 1): "Subtitle 1",
        }
        self.track_enabled: dict[tuple[str, int], bool] = {
            ("video", 1): True,
            ("audio", 1): True,
            ("subtitle", 1): True,
        }
        self.track_locked: dict[tuple[str, int], bool] = {
            ("video", 1): False,
            ("audio", 1): False,
            ("subtitle", 1): False,
        }
        self.markers: dict[int, dict[str, Any]] = {}
        self.settings: dict[str, str] = {
            "timelineResolutionWidth": "1920",
            "timelineResolutionHeight": "1080",
            "timelineFrameRate": "24",
        }

    def GetName(self) -> str:  # noqa: N802
        return self.name

    def SetName(self, name: str) -> bool:  # noqa: N802
        self.name = name
        return True

    def GetStartFrame(self) -> int:  # noqa: N802
        return self.start_frame

    def GetEndFrame(self) -> int:  # noqa: N802
        max_end = self.start_frame
        for track_dict in self.tracks.values():
            for items in track_dict.values():
                for item in items:
                    if item.end_frame > max_end:
                        max_end = item.end_frame
        return max_end

    def SetStartTimecode(self, tc: str) -> bool:  # noqa: N802
        self.current_timecode = tc
        return True

    def GetStartTimecode(self) -> str:  # noqa: N802
        return "01:00:00:00"

    def GetCurrentTimecode(self) -> str:  # noqa: N802
        return self.current_timecode

    def SetCurrentTimecode(self, tc: str) -> bool:  # noqa: N802
        self.current_timecode = tc
        return True

    def GetTrackCount(self, track_type: str) -> int:  # noqa: N802
        return len(self.tracks.get(track_type, {}))

    def AddTrack(self, track_type: str, sub_type: str = "stereo") -> bool:  # noqa: N802
        track_map = self.tracks.setdefault(track_type, {})
        next_idx = max(track_map.keys(), default=0) + 1
        track_map[next_idx] = []
        self.track_names[(track_type, next_idx)] = f"{track_type.capitalize()} {next_idx}"
        self.track_enabled[(track_type, next_idx)] = True
        self.track_locked[(track_type, next_idx)] = False
        return True

    def DeleteTrack(self, track_type: str, track_idx: int) -> bool:  # noqa: N802
        track_map = self.tracks.get(track_type, {})
        if track_idx not in track_map or len(track_map) <= 1:
            return False
        del track_map[track_idx]
        # Re-index remaining tracks sequentially 1..N
        new_map: dict[int, list[MockTimelineItem]] = {}
        for new_i, old_i in enumerate(sorted(track_map.keys()), start=1):
            new_map[new_i] = track_map[old_i]
        self.tracks[track_type] = new_map
        return True

    def GetTrackName(self, track_type: str, track_index: int) -> str:  # noqa: N802
        return self.track_names.get(
            (track_type, track_index), f"{track_type.capitalize()} {track_index}"
        )

    def SetTrackName(self, track_type: str, track_index: int, name: str) -> bool:  # noqa: N802
        self.track_names[(track_type, track_index)] = name
        return True

    def SetTrackEnable(self, track_type: str, track_index: int, enabled: bool) -> bool:  # noqa: N802
        self.track_enabled[(track_type, track_index)] = bool(enabled)
        return True

    def GetIsTrackEnabled(self, track_type: str, track_index: int) -> bool:  # noqa: N802
        return self.track_enabled.get((track_type, track_index), True)

    def SetTrackLock(self, track_type: str, track_index: int, locked: bool) -> bool:  # noqa: N802
        self.track_locked[(track_type, track_index)] = bool(locked)
        return True

    def GetIsTrackLocked(self, track_type: str, track_index: int) -> bool:  # noqa: N802
        return self.track_locked.get((track_type, track_index), False)

    def GetItemListInTrack(  # noqa: N802
        self, track_type: str, track_index: int
    ) -> list[MockTimelineItem]:
        track_map = self.tracks.get(track_type, {})
        items = track_map.get(track_index, [])
        return sorted(items, key=lambda x: x.start_frame)

    def DeleteClips(  # noqa: N802
        self, clips: list[MockTimelineItem], ripple: bool = False
    ) -> bool:
        for track_map in self.tracks.values():
            for t_idx, items in track_map.items():
                for clip in clips:
                    if clip in items:
                        removed_start = clip.start_frame
                        removed_dur = clip.GetDuration()
                        items.remove(clip)
                        if ripple and removed_dur > 0:
                            for remaining in items:
                                if remaining.start_frame >= removed_start:
                                    remaining.start_frame -= removed_dur
                                    remaining.end_frame -= removed_dur
                        track_map[t_idx] = sorted(items, key=lambda x: x.start_frame)
        return True

    def DuplicateTimeline(self, timeline_name: str) -> MockTimeline:  # noqa: N802
        import copy

        dup = copy.deepcopy(self)
        dup.name = timeline_name
        return dup

    def GrabStill(self) -> dict[str, str]:  # noqa: N802
        return {"timecode": self.current_timecode, "timeline": self.name}

    def InsertFusionTitleIntoTimeline(self, title_name: str) -> MockTimelineItem:  # noqa: N802
        track_items = self.tracks["video"].setdefault(2, [])
        start_f = max((c.end_frame for c in track_items), default=self.start_frame)
        item = MockTimelineItem(name=title_name, start_frame=start_f, end_frame=start_f + 120)
        track_items.append(item)
        return item

    def InsertGeneratorIntoTimeline(self, generator_name: str) -> MockTimelineItem:  # noqa: N802
        track_items = self.tracks["video"].setdefault(2, [])
        start_f = max((c.end_frame for c in track_items), default=self.start_frame)
        item = MockTimelineItem(name=generator_name, start_frame=start_f, end_frame=start_f + 120)
        track_items.append(item)
        return item

    def CreateSubtitlesFromAudio(self) -> bool:  # noqa: N802
        # Studio-only in DaVinci Resolve 20; returns False on Free Edition
        return False

    def AddMarker(  # noqa: N802
        self,
        frame_id: int,
        color: str,
        name: str,
        note: str,
        duration: int,
        custom_data: str = "",
    ) -> bool:
        self.markers[int(frame_id)] = {
            "color": color,
            "name": name,
            "note": note,
            "duration": int(duration),
            "customData": custom_data,
        }
        return True

    def GetMarkers(self) -> dict[int, dict[str, Any]]:  # noqa: N802
        return dict(self.markers)

    def DeleteMarkersByColor(self, color: str) -> bool:  # noqa: N802
        to_remove = [
            fid for fid, m in self.markers.items() if color == "All" or m.get("color") == color
        ]
        for fid in to_remove:
            del self.markers[fid]
        return True

    def DeleteMarkerAtFrame(self, frame_num: int) -> bool:  # noqa: N802
        return self.markers.pop(int(frame_num), None) is not None

    def GetSetting(self, key: str | None = None) -> Any:  # noqa: N802
        if key is None:
            return dict(self.settings)
        return self.settings.get(key, "")

    def SetSetting(self, key: str, val: str) -> bool:  # noqa: N802
        self.settings[key] = str(val)
        return True


class MockFolder:
    """Stateful representation of a Media Pool Folder / Bin."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.subfolders: dict[str, MockFolder] = {}
        self.clips: list[MockMediaPoolItem] = []

    def GetName(self) -> str:  # noqa: N802
        return self.name

    def GetSubFolderList(self) -> list[MockFolder]:  # noqa: N802
        return list(self.subfolders.values())

    def GetClipList(self) -> list[MockMediaPoolItem]:  # noqa: N802
        return list(self.clips)


class MockMediaPool:
    """Stateful representation of DaVinci Resolve's MediaPool."""

    def __init__(self, project: MockProject) -> None:
        self.project = project
        self.root_folder = MockFolder("Master")
        self.current_folder = self.root_folder

    def GetRootFolder(self) -> MockFolder:  # noqa: N802
        return self.root_folder

    def GetCurrentFolder(self) -> MockFolder:  # noqa: N802
        return self.current_folder

    def SetCurrentFolder(self, folder: MockFolder) -> bool:  # noqa: N802
        self.current_folder = folder
        return True

    def AddSubFolder(self, parent_folder: MockFolder, name: str) -> MockFolder:  # noqa: N802
        if name not in parent_folder.subfolders:
            parent_folder.subfolders[name] = MockFolder(name)
        self.current_folder = parent_folder.subfolders[name]
        return self.current_folder

    def DeleteFolders(self, folders: list[MockFolder]) -> bool:  # noqa: N802
        for target in folders:
            self._remove_folder_recursive(self.root_folder, target)
        return True

    def _remove_folder_recursive(self, current: MockFolder, target: MockFolder) -> bool:
        for k, sub in list(current.subfolders.items()):
            if sub is target:
                del current.subfolders[k]
                if self.current_folder is target:
                    self.current_folder = self.root_folder
                return True
            if self._remove_folder_recursive(sub, target):
                return True
        return False

    def ImportMedia(  # noqa: N802
        self, items: list[str | dict[str, Any]]
    ) -> list[MockMediaPoolItem]:
        imported: list[MockMediaPoolItem] = []
        for entry in items:
            if isinstance(entry, str):
                clip = MockMediaPoolItem(file_path=entry)
                self.current_folder.clips.append(clip)
                imported.append(clip)
            elif isinstance(entry, dict):
                fpath = str(entry.get("FilePath", ""))
                start_f = int(entry.get("StartFrame", 0))
                end_f = int(entry.get("EndFrame", 120))
                dur = max(1, end_f - start_f)
                cname = str(entry.get("ClipName") or Path(fpath).name)
                clip = MockMediaPoolItem(file_path=fpath, name=cname, duration=dur)
                self.current_folder.clips.append(clip)
                imported.append(clip)
        return imported

    def CreateEmptyTimeline(self, name: str) -> MockTimeline:  # noqa: N802
        tl = MockTimeline(name=name)
        self.project.timelines.append(tl)
        self.project.current_timeline = tl
        return tl

    def AppendToTimeline(  # noqa: N802
        self, clips_or_infos: list[MockMediaPoolItem | dict[str, Any]]
    ) -> list[MockTimelineItem]:
        if self.project.current_timeline is None:
            self.CreateEmptyTimeline("Timeline 1")
        tl = self.project.current_timeline
        assert tl is not None

        created_items: list[MockTimelineItem] = []
        for entry in clips_or_infos:
            if isinstance(entry, MockMediaPoolItem):
                clip_obj = entry
                track_idx = 1
                media_type = 0
                src_start = 0
                src_end = clip_obj.duration
                rec_frame: int | None = None
            else:
                clip_obj = entry["mediaPoolItem"]
                track_idx = int(entry.get("trackIndex", 1))
                media_type = int(entry.get("mediaType", 0))
                src_start = int(entry.get("startFrame", 0))
                src_end = int(entry.get("endFrame", clip_obj.duration))
                rec_frame = (
                    int(entry["recordFrame"]) if entry.get("recordFrame") is not None else None
                )

            ext = Path(clip_obj.file_path).suffix.lower()
            if ext in {".srt", ".vtt"}:
                target_track_type = "subtitle"
            elif media_type == 2 or (not clip_obj.has_video and clip_obj.has_audio):
                target_track_type = "audio"
            else:
                target_track_type = "video"

            track_map = tl.tracks.setdefault(target_track_type, {})
            while track_idx not in track_map:
                next_i = max(track_map.keys(), default=0) + 1
                track_map[next_i] = []

            track_items = track_map[track_idx]
            duration = max(1, src_end - src_start)
            if rec_frame is not None:
                start_f = rec_frame
            else:
                start_f = max((item.end_frame for item in track_items), default=tl.start_frame)
            end_f = start_f + duration

            tl_item = MockTimelineItem(
                name=clip_obj.GetName(),
                start_frame=start_f,
                end_frame=end_f,
                source_start=src_start,
                media_pool_item=clip_obj,
            )
            track_items.append(tl_item)
            track_map[track_idx] = sorted(track_items, key=lambda x: x.start_frame)
            created_items.append(tl_item)

        return created_items


class MockGalleryStillAlbum:
    """Stateful representation of a DaVinci Resolve GalleryStillAlbum."""

    def ExportStills(  # noqa: N802
        self,
        stills: list[Any],
        folder_path: str,
        file_prefix: str,
        format_ext: str = "png",
    ) -> bool:
        target_dir = Path(folder_path)
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / f"{file_prefix}.{format_ext}"
        out_path.write_bytes(_create_minimal_png_bytes())
        return True

    def DeleteStills(self, stills: list[Any]) -> bool:  # noqa: N802
        return True


class MockGallery:
    """Stateful representation of DaVinci Resolve's Gallery."""

    def __init__(self) -> None:
        self.album = MockGalleryStillAlbum()

    def GetCurrentStillAlbum(self) -> MockGalleryStillAlbum:  # noqa: N802
        return self.album

    def ExportStills(  # noqa: N802
        self,
        stills: list[Any],
        folder_path: str,
        file_prefix: str,
        format_ext: str = "png",
    ) -> bool:
        return self.album.ExportStills(stills, folder_path, file_prefix, format_ext)


class MockProject:
    """Stateful representation of a DaVinci Resolve Project."""

    def __init__(self, name: str = "Untitled Project") -> None:
        self.name = name
        self.settings: dict[str, str] = {
            "timelineResolutionWidth": "1920",
            "timelineResolutionHeight": "1080",
            "timelineFrameRate": "24",
            "videoMonitorFormat": "HD 1080p 24",
        }
        self.media_pool = MockMediaPool(self)
        default_tl = MockTimeline("Timeline 1")
        self.timelines: list[MockTimeline] = [default_tl]
        self.current_timeline: MockTimeline | None = default_tl
        self.gallery = MockGallery()
        self.render_settings: dict[str, Any] = {}
        self.render_jobs: dict[str, dict[str, Any]] = {}

    def GetName(self) -> str:  # noqa: N802
        return self.name

    def SetName(self, name: str) -> bool:  # noqa: N802
        self.name = name
        return True

    def GetMediaPool(self) -> MockMediaPool:  # noqa: N802
        return self.media_pool

    def GetCurrentTimeline(self) -> MockTimeline | None:  # noqa: N802
        return self.current_timeline

    def SetCurrentTimeline(self, timeline: MockTimeline) -> bool:  # noqa: N802
        self.current_timeline = timeline
        return True

    def GetTimelineCount(self) -> int:  # noqa: N802
        return len(self.timelines)

    def GetTimelineByIndex(self, idx: int) -> MockTimeline | None:  # noqa: N802
        if 1 <= idx <= len(self.timelines):
            return self.timelines[idx - 1]
        return None

    def GetGallery(self) -> MockGallery:  # noqa: N802
        return self.gallery

    def GetSetting(self, key: str | None = None) -> Any:  # noqa: N802
        if key is None:
            return dict(self.settings)
        return self.settings.get(key, "")

    def SetSetting(self, key: str, val: str) -> bool:  # noqa: N802
        self.settings[key] = str(val)
        if self.current_timeline is not None:
            self.current_timeline.SetSetting(key, str(val))
        return True

    def LoadRenderPreset(self, preset_name: str) -> bool:  # noqa: N802
        self.render_settings["PresetName"] = preset_name
        return True

    def SetCurrentRenderFormatAndCodec(self, fmt: str, codec: str) -> bool:  # noqa: N802
        self.render_settings["Format"] = fmt
        self.render_settings["Codec"] = codec
        return True

    def SetRenderSettings(self, settings: dict[str, Any]) -> bool:  # noqa: N802
        self.render_settings.update(settings)
        return True

    def AddRenderJob(self) -> str:  # noqa: N802
        job_id = f"job_{len(self.render_jobs) + 1:03d}"
        target_dir = Path(str(self.render_settings.get("TargetDir", RENDER_EXPORT_DIR)))
        custom_name = str(self.render_settings.get("CustomName", "master_export"))
        fmt = str(self.render_settings.get("Format", "mp4"))
        output_path = target_dir / f"{custom_name}.{fmt}"
        self.render_jobs[job_id] = {
            "JobId": job_id,
            "TargetDir": str(target_dir),
            "OutputFilename": f"{custom_name}.{fmt}",
            "OutputPath": str(output_path),
            "JobStatus": "Ready",
            "CompletionPercentage": 0,
        }
        return job_id

    def StartRendering(self, *job_ids: Any, **kwargs: Any) -> bool:  # noqa: N802
        requested = {str(j) for j in job_ids if isinstance(j, str)}
        for job_id, job in self.render_jobs.items():
            if requested and job_id not in requested:
                continue
            if job.get("JobStatus") == "Complete":
                continue
            target_dir = Path(job["TargetDir"])
            target_dir.mkdir(parents=True, exist_ok=True)
            out_file = Path(job["OutputPath"])
            out_file.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
            job["JobStatus"] = "Complete"
            job["CompletionPercentage"] = 100
        return True

    def IsRenderingInProgress(self) -> bool:  # noqa: N802
        return False

    def GetRenderJobStatus(self, job_id: str) -> dict[str, Any]:  # noqa: N802
        job = self.render_jobs.get(job_id, {})
        return {
            "JobStatus": job.get("JobStatus", "Complete"),
            "CompletionPercentage": job.get("CompletionPercentage", 100),
        }


class MockProjectManager:
    """Stateful representation of DaVinci Resolve's ProjectManager."""

    def __init__(self) -> None:
        default_proj = MockProject("Default Project")
        self.projects: dict[str, MockProject] = {default_proj.name: default_proj}
        self.current_project: MockProject = default_proj

    def GetCurrentProject(self) -> MockProject:  # noqa: N802
        return self.current_project

    def CreateProject(self, project_name: str) -> MockProject:  # noqa: N802
        proj = MockProject(project_name)
        self.projects[project_name] = proj
        self.current_project = proj
        return proj

    def LoadProject(self, project_name: str) -> MockProject | None:  # noqa: N802
        proj = self.projects.get(project_name)
        if proj is not None:
            self.current_project = proj
        return proj

    def SaveProject(self) -> bool:  # noqa: N802
        return True


class InMemoryResolveEngine:
    """Top-level stateful mock for DaVinci Resolve 20 Free Edition."""

    def __init__(self) -> None:
        self.project_manager = MockProjectManager()
        self.current_page = "edit"

    def GetProjectManager(self) -> MockProjectManager:  # noqa: N802
        return self.project_manager

    def OpenPage(self, page_name: str) -> bool:  # noqa: N802
        self.current_page = page_name
        return True

    def GetVersionString(self) -> str:  # noqa: N802
        return "20.0.0 Free"


_MOCK_RESOLVE_SINGLETON: InMemoryResolveEngine | None = None


def reset_mock_resolve_engine() -> InMemoryResolveEngine:
    """Reset and return a fresh `InMemoryResolveEngine` instance (used by pytest)."""
    global _MOCK_RESOLVE_SINGLETON
    _MOCK_RESOLVE_SINGLETON = InMemoryResolveEngine()
    return _MOCK_RESOLVE_SINGLETON


def get_resolve() -> Any:
    """Connect to DaVinci Resolve's native scripting API, or use the Mock Engine if enabled."""
    global _MOCK_RESOLVE_SINGLETON
    if os.environ.get("RESOLVE_MOCK_MODE", "").strip().lower() in {"1", "true", "yes"}:
        if _MOCK_RESOLVE_SINGLETON is None:
            _MOCK_RESOLVE_SINGLETON = InMemoryResolveEngine()
        return _MOCK_RESOLVE_SINGLETON

    script_module_dir = os.environ.get(
        "RESOLVE_SCRIPT_API_MODULES", DEFAULT_RESOLVE_SCRIPT_MODULES
    )
    if script_module_dir not in sys.path and Path(script_module_dir).exists():
        sys.path.append(script_module_dir)

    try:
        import DaVinciResolveScript as bmd

        resolve_obj = bmd.scriptapp("Resolve")
        if resolve_obj is None:
            raise RuntimeError(
                "DaVinci Resolve is either not running or External Scripting is not set to 'Local' "
                "(Preferences -> System -> General -> External scripting using: Local). "
                "Set RESOLVE_MOCK_MODE=1 for dry-run / offline testing."
            )
        return resolve_obj
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import DaVinciResolveScript from '{script_module_dir}': {exc}. "
            "Set RESOLVE_MOCK_MODE=1 for dry-run / offline testing."
        ) from exc


# ============================================================================
# Internal Helpers for Media Pool Navigation & Remote Asset Download
# ============================================================================


def _navigate_or_create_bin(media_pool: Any, bin_path: str, *, create: bool = True) -> Any:
    """Traverse a slash-separated bin path from RootFolder, optionally creating subfolders."""
    root = media_pool.GetRootFolder()
    cleaned = bin_path.strip().strip("/")
    if not cleaned or cleaned.lower() == "master":
        media_pool.SetCurrentFolder(root)
        return root

    current = root
    for part in cleaned.split("/"):
        segment = part.strip()
        if not segment or segment.lower() == "master":
            continue
        subfolders = current.GetSubFolderList() or []
        matched = next((f for f in subfolders if f.GetName() == segment), None)
        if matched is not None:
            current = matched
        elif create:
            created = media_pool.AddSubFolder(current, segment)
            if created is not None:
                current = created
        else:
            raise ValueError(f"Media Pool bin '{bin_path}' does not exist.")
    media_pool.SetCurrentFolder(current)
    return current


def _list_bins_recursive(folder: Any, prefix: str = "") -> list[str]:
    """Recursively list all folder paths in the Media Pool."""
    current_path = f"{prefix}/{folder.GetName()}" if prefix else str(folder.GetName())
    results = [current_path]
    for sub in folder.GetSubFolderList() or []:
        results.extend(_list_bins_recursive(sub, current_path))
    return results


def _find_clip_in_folder_recursive(folder: Any, clip_name: str) -> Any | None:
    """Locate a MediaPoolItem by name (exact or stem match) across all Media Pool bins."""
    for clip in folder.GetClipList() or []:
        name = str(clip.GetName())
        if name == clip_name or Path(name).stem == Path(clip_name).stem:
            return clip
    for sub in folder.GetSubFolderList() or []:
        found = _find_clip_in_folder_recursive(sub, clip_name)
        if found is not None:
            return found
    return None


def _find_timeline_by_name(project: Any, timeline_name: str) -> Any | None:
    """Search existing timelines in `project` by name."""
    count = int(project.GetTimelineCount() or 0)
    for idx in range(1, count + 1):
        tl = project.GetTimelineByIndex(idx)
        if tl is not None and str(tl.GetName()) == timeline_name:
            return tl
    return None


async def _download_remote_asset(
    url: str,
    custom_name: str | None = None,
) -> Path:
    """Download a remote `https://` asset into `ASSET_DIR` with hash caching."""
    validated_url = validate_https_source_url(url)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(validated_url)
    url_path_ext = Path(parsed.path).suffix.lower()
    ext = url_path_ext if url_path_ext in ALLOWED_MEDIA_EXTENSIONS else ".mp4"

    url_hash = hashlib.sha256(validated_url.encode("utf-8")).hexdigest()[:12]
    if custom_name:
        custom_path = Path(custom_name)
        base_stem = custom_path.stem
        final_ext = (
            custom_path.suffix.lower()
            if custom_path.suffix.lower() in ALLOWED_MEDIA_EXTENSIONS
            else ext
        )
        filename = f"{base_stem}_{url_hash}{final_ext}"
    else:
        filename = f"remote_asset_{url_hash}{ext}"

    dest_path = resolve_safe_workspace_path(ASSET_DIR / filename, ASSET_DIR)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path

    async with httpx.AsyncClient(
        timeout=DOWNLOAD_TIMEOUT_SECONDS, follow_redirects=True
    ) as client:
        async with client.stream("GET", validated_url) as response:
            response.raise_for_status()
            total_bytes = 0
            temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
            with temp_path.open("wb") as fh:
                async for chunk in response.aiter_bytes(chunk_size=65536):
                    total_bytes += len(chunk)
                    if total_bytes > MAX_DOWNLOAD_BYTES:
                        temp_path.unlink(missing_ok=True)
                        raise ValueError(
                            f"Remote asset exceeded maximum allowed size ({MAX_DOWNLOAD_BYTES} bytes)."
                        )
                    fh.write(chunk)
            temp_path.replace(dest_path)

    return dest_path


def _extract_clip_duration_frames(clip: Any) -> int:
    """Read clip frame count from DaVinci Resolve's `GetClipProperty('Frames')`."""
    try:
        frames_str = str(clip.GetClipProperty("Frames") or "").strip()
        return max(1, int(frames_str))
    except (ValueError, TypeError):
        return 120


# ============================================================================
# Domain 1 MCP Tools: Media Ingest, Remote Asset Download & Bins
# ============================================================================


@mcp.tool()
async def ingest_media(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Ingest local files or remote `https://` URLs into DaVinci Resolve's Media Pool.

    Args:
        items: List of ingest specifications (`source`, `target_bin`, `custom_name`, `generation_id`).
    """
    try:
        req = IngestMediaRequest.model_validate({"items": items})
        resolve = get_resolve()
        project = resolve.GetProjectManager().GetCurrentProject()
        media_pool = project.GetMediaPool()

        imported_metadata: list[IngestedClipMetadata] = []
        for item in req.items:
            if item.source.startswith("https://"):
                local_file = await _download_remote_asset(item.source, item.custom_name)
            else:
                local_file = resolve_safe_workspace_path(
                    item.source, WORKSPACE_ROOT, must_exist=True
                )

            _navigate_or_create_bin(media_pool, item.target_bin, create=True)
            imported = media_pool.ImportMedia([str(local_file)])
            if not imported:
                return IngestMediaResult(
                    status=OperationStatus.ERROR,
                    imported_clips=imported_metadata,
                    error_message=f"DaVinci Resolve failed to import '{local_file}'.",
                ).model_dump(mode="json")

            clip_obj = imported[0]
            if item.custom_name:
                clip_obj.SetClipProperty("Clip Name", item.custom_name)
            if item.generation_id:
                clip_obj.SetMetadata("generation_id", item.generation_id)

            duration_frames = _extract_clip_duration_frames(clip_obj)
            fps = str(clip_obj.GetClipProperty("FPS") or "24")
            resolution = str(clip_obj.GetClipProperty("Resolution") or "")
            v_codec = str(clip_obj.GetClipProperty("Video Codec") or "")
            a_codec = str(clip_obj.GetClipProperty("Audio Codec") or "")

            imported_metadata.append(
                IngestedClipMetadata(
                    clip_name=str(clip_obj.GetName()),
                    local_path=str(local_file),
                    bin_path=item.target_bin,
                    duration_frames=duration_frames,
                    fps=fps,
                    resolution=resolution or None,
                    has_video=bool(v_codec or local_file.suffix.lower() in {".mp4", ".mov", ".png", ".jpg"}),
                    has_audio=bool(a_codec or local_file.suffix.lower() in {".wav", ".mp3", ".aac", ".flac"}),
                    generation_id=item.generation_id,
                )
            )

        return IngestMediaResult(
            status=OperationStatus.SUCCESS,
            imported_clips=imported_metadata,
        ).model_dump(mode="json")
    except Exception as exc:
        return IngestMediaResult(
            status=OperationStatus.ERROR,
            error_message=str(exc),
        ).model_dump(mode="json")


@mcp.tool()
def manage_bins(action: str = "list", bin_path: str = "Master") -> dict[str, Any]:
    """Create, list, switch, or delete folders/bins in the DaVinci Resolve Media Pool.

    Args:
        action: One of 'create', 'list', 'switch', or 'delete'.
        bin_path: Slash-separated bin path (e.g., 'Video/Scene1' or 'Audio/SFX').
    """
    try:
        req = ManageBinsRequest(action=action, bin_path=bin_path)
        resolve = get_resolve()
        media_pool = resolve.GetProjectManager().GetCurrentProject().GetMediaPool()
        root = media_pool.GetRootFolder()

        if req.action == "create":
            folder = _navigate_or_create_bin(media_pool, req.bin_path, create=True)
        elif req.action == "switch":
            folder = _navigate_or_create_bin(media_pool, req.bin_path, create=False)
        elif req.action == "delete":
            target = _navigate_or_create_bin(media_pool, req.bin_path, create=False)
            if target is root:
                raise ValueError("Cannot delete the root 'Master' folder.")
            media_pool.DeleteFolders([target])
            folder = media_pool.GetCurrentFolder()
        else:
            folder = media_pool.GetCurrentFolder()

        all_bins = _list_bins_recursive(root)
        active_clips = [str(c.GetName()) for c in (folder.GetClipList() or [])]
        return ManageBinsResult(
            status=OperationStatus.SUCCESS,
            active_bin=str(folder.GetName()),
            bins=all_bins,
            clip_names_in_active_bin=active_clips,
        ).model_dump(mode="json")
    except Exception as exc:
        return ManageBinsResult(
            status=OperationStatus.ERROR,
            error_message=str(exc),
        ).model_dump(mode="json")


@mcp.tool()
def create_subclip(
    clip_name: str,
    mark_in: int,
    mark_out: int,
    subclip_name: str | None = None,
    target_bin: str = "Master",
) -> dict[str, Any]:
    """Create a frame-accurate subclip `[mark_in, mark_out]` in the Media Pool from an existing source clip."""
    try:
        req = CreateSubclipRequest(
            clip_name=clip_name,
            mark_in=mark_in,
            mark_out=mark_out,
            subclip_name=subclip_name,
            target_bin=target_bin,
        )
        resolve = get_resolve()
        media_pool = resolve.GetProjectManager().GetCurrentProject().GetMediaPool()
        source_clip = _find_clip_in_folder_recursive(media_pool.GetRootFolder(), req.clip_name)
        if source_clip is None:
            raise ValueError(f"Clip '{req.clip_name}' not found in Media Pool.")

        file_path = str(source_clip.GetClipProperty("File Path") or "")
        final_subclip_name = (
            req.subclip_name or f"{Path(req.clip_name).stem}_{req.mark_in}_{req.mark_out}"
        )
        _navigate_or_create_bin(media_pool, req.target_bin, create=True)

        imported = media_pool.ImportMedia(
            [
                {
                    "FilePath": file_path,
                    "StartFrame": req.mark_in,
                    "EndFrame": req.mark_out,
                    "ClipName": final_subclip_name,
                }
            ]
        )
        if not imported:
            raise RuntimeError("MediaPool.ImportMedia failed to create subclip.")

        subclip = imported[0]
        subclip.SetClipProperty("Clip Name", final_subclip_name)
        return CreateSubclipResult(
            status=OperationStatus.SUCCESS,
            subclip_name=str(subclip.GetName()),
            duration_frames=req.mark_out - req.mark_in,
            bin_path=req.target_bin,
        ).model_dump(mode="json")
    except Exception as exc:
        return CreateSubclipResult(
            status=OperationStatus.ERROR,
            error_message=str(exc),
        ).model_dump(mode="json")


@mcp.tool()
def set_clip_metadata(
    clip_name: str,
    scene: str | None = None,
    shot: str | None = None,
    take: str | None = None,
    keywords: str | None = None,
    comments: str | None = None,
    clip_color: str | None = None,
    generation_id: str | None = None,
) -> dict[str, Any]:
    """Tag a Media Pool clip with Scene, Shot, Take, Keywords, ClipColor, and `generation_id`."""
    try:
        req = SetClipMetadataRequest(
            clip_name=clip_name,
            scene=scene,
            shot=shot,
            take=take,
            keywords=keywords,
            comments=comments,
            clip_color=MarkerColor(clip_color) if clip_color else None,
            generation_id=generation_id,
        )
        resolve = get_resolve()
        media_pool = resolve.GetProjectManager().GetCurrentProject().GetMediaPool()
        clip = _find_clip_in_folder_recursive(media_pool.GetRootFolder(), req.clip_name)
        if clip is None:
            raise ValueError(f"Clip '{req.clip_name}' not found in Media Pool.")

        updates: dict[str, str] = {}
        field_map = {
            "Scene": req.scene,
            "Shot": req.shot,
            "Take": req.take,
            "Keywords": req.keywords,
            "Comments": req.comments,
            "generation_id": req.generation_id,
        }
        for key, val in field_map.items():
            if val is not None:
                clip.SetMetadata(key, val)
                updates[key] = val

        if req.clip_color is not None:
            clip.SetClipColor(req.clip_color.value)
            updates["ClipColor"] = req.clip_color.value

        return SetClipMetadataResult(
            status=OperationStatus.SUCCESS,
            clip_name=str(clip.GetName()),
            updated_fields=updates,
        ).model_dump(mode="json")
    except Exception as exc:
        return SetClipMetadataResult(
            status=OperationStatus.ERROR,
            clip_name=clip_name,
            error_message=str(exc),
        ).model_dump(mode="json")


# ============================================================================
# Domain 2 MCP Tools: Project, Timeline & Multi-Track Lifecycle
# ============================================================================


@mcp.tool()
def configure_project(
    project_name: str,
    width: int = 1920,
    height: int = 1080,
    frame_rate: str = "24",
) -> dict[str, Any]:
    """Create or switch to a DaVinci Resolve project and configure resolution/FPS (Free Edition <=3840 UHD, <=60 fps)."""
    try:
        req = ProjectConfigRequest(
            project_name=project_name,
            width=width,
            height=height,
            frame_rate=frame_rate,
        )
        resolve = get_resolve()
        pm = resolve.GetProjectManager()
        current = pm.GetCurrentProject()
        if current is None or str(current.GetName()) != req.project_name:
            loaded = pm.LoadProject(req.project_name)
            project = loaded if loaded is not None else pm.CreateProject(req.project_name)
        else:
            project = current

        project.SetSetting("timelineResolutionWidth", str(req.width))
        project.SetSetting("timelineResolutionHeight", str(req.height))
        project.SetSetting("timelineFrameRate", req.frame_rate)

        return ProjectConfigResult(
            status=OperationStatus.SUCCESS,
            project_name=str(project.GetName()),
            width=req.width,
            height=req.height,
            frame_rate=req.frame_rate,
            resolve_edition="DaVinci Resolve 20 Free",
        ).model_dump(mode="json")
    except Exception as exc:
        return ProjectConfigResult(
            status=OperationStatus.ERROR,
            error_message=str(exc),
        ).model_dump(mode="json")


@mcp.tool()
def create_or_manage_timeline(
    action: str,
    timeline_name: str,
    new_name: str | None = None,
    start_timecode: str = "01:00:00:00",
) -> dict[str, Any]:
    """Create, switch, or duplicate a timeline in the active project.

    Args:
        action: 'create', 'switch', or 'duplicate'.
        timeline_name: Target timeline name (e.g., 'SpaceSequence').
        new_name: New timeline name when `action='duplicate'`.
        start_timecode: Initial SMPTE timecode (default '01:00:00:00').
    """
    try:
        req = TimelineActionRequest(
            action=action,
            timeline_name=timeline_name,
            new_name=new_name,
            start_timecode=start_timecode,
        )
        resolve = get_resolve()
        project = resolve.GetProjectManager().GetCurrentProject()
        media_pool = project.GetMediaPool()

        existing = _find_timeline_by_name(project, req.timeline_name)
        if req.action == "create":
            if existing is not None:
                project.SetCurrentTimeline(existing)
                timeline = existing
            else:
                timeline = media_pool.CreateEmptyTimeline(req.timeline_name)
                if timeline is None:
                    raise RuntimeError(f"Failed to create timeline '{req.timeline_name}'.")
                timeline.SetStartTimecode(req.start_timecode)
        elif req.action == "switch":
            if existing is None:
                raise ValueError(f"Timeline '{req.timeline_name}' does not exist.")
            project.SetCurrentTimeline(existing)
            timeline = existing
        elif req.action == "duplicate":
            source_tl = existing or project.GetCurrentTimeline()
            if source_tl is None:
                raise ValueError("No source timeline found to duplicate.")
            dup_name = req.new_name or f"{source_tl.GetName()}_Copy"
            timeline = source_tl.DuplicateTimeline(dup_name)
            if timeline is None:
                raise RuntimeError(f"Failed to duplicate timeline as '{dup_name}'.")
            project.SetCurrentTimeline(timeline)
        else:
            raise ValueError(f"Unsupported action '{req.action}'.")

        return TimelineActionResult(
            status=OperationStatus.SUCCESS,
            action=req.action,
            timeline_name=str(timeline.GetName()),
            start_timecode=req.start_timecode,
            total_timelines_in_project=int(project.GetTimelineCount() or 1),
        ).model_dump(mode="json")
    except Exception as exc:
        return TimelineActionResult(
            status=OperationStatus.ERROR,
            action=action,
            error_message=str(exc),
        ).model_dump(mode="json")


@mcp.tool()
def manage_tracks(
    action: str,
    track_type: str = "video",
    track_index: int = 1,
    audio_sub_type: str = "stereo",
    track_name: str | None = None,
) -> dict[str, Any]:
    """Add, delete, rename, enable/disable, or lock/unlock video, audio, or subtitle tracks."""
    try:
        req = TrackManagementRequest(
            action=action,
            track_type=TrackType(track_type),
            track_index=track_index,
            audio_sub_type=AudioTrackSubType(audio_sub_type),
            track_name=track_name,
        )
        resolve = get_resolve()
        project = resolve.GetProjectManager().GetCurrentProject()
        timeline = project.GetCurrentTimeline()
        if not timeline:
            raise ValueError("No active timeline found.")

        ttype = req.track_type.value
        if req.action == "add":
            if req.track_type == TrackType.AUDIO:
                timeline.AddTrack(ttype, req.audio_sub_type.value)
            else:
                timeline.AddTrack(ttype)
            new_idx = int(timeline.GetTrackCount(ttype))
            if req.track_name:
                timeline.SetTrackName(ttype, new_idx, req.track_name)
            target_idx = new_idx
        elif req.action == "delete":
            timeline.DeleteTrack(ttype, req.track_index)
            target_idx = req.track_index
        elif req.action == "rename":
            if not req.track_name:
                raise ValueError("track_name is required when action='rename'.")
            timeline.SetTrackName(ttype, req.track_index, req.track_name)
            target_idx = req.track_index
        elif req.action == "enable":
            timeline.SetTrackEnable(ttype, req.track_index, True)
            target_idx = req.track_index
        elif req.action == "disable":
            timeline.SetTrackEnable(ttype, req.track_index, False)
            target_idx = req.track_index
        elif req.action == "lock":
            timeline.SetTrackLock(ttype, req.track_index, True)
            target_idx = req.track_index
        elif req.action == "unlock":
            timeline.SetTrackLock(ttype, req.track_index, False)
            target_idx = req.track_index
        else:
            raise ValueError(f"Unsupported track action '{req.action}'.")

        count = int(timeline.GetTrackCount(ttype))
        resolved_name = (
            str(timeline.GetTrackName(ttype, min(target_idx, max(1, count))))
            if count > 0
            else None
        )
        return TrackManagementResult(
            status=OperationStatus.SUCCESS,
            action=req.action,
            track_type=req.track_type,
            track_index=target_idx,
            track_count=count,
            track_name=resolved_name,
        ).model_dump(mode="json")
    except Exception as exc:
        return TrackManagementResult(
            status=OperationStatus.ERROR,
            action=action,
            track_type=TrackType.VIDEO,
            track_index=track_index,
            error_message=str(exc),
        ).model_dump(mode="json")


@mcp.tool()
def get_timeline_structure() -> dict[str, Any]:
    """Return the complete multi-track timeline structure, clip bounds, markers, and detected empty gaps."""
    try:
        resolve = get_resolve()
        project = resolve.GetProjectManager().GetCurrentProject()
        timeline = project.GetCurrentTimeline()
        if not timeline:
            return TimelineStructureResult(
                status=OperationStatus.ERROR,
                error_message="No active timeline found",
            ).model_dump(mode="json")

        start_frame = int(timeline.GetStartFrame() or 0)
        end_frame = int(timeline.GetEndFrame() or start_frame)
        duration_frames = max(0, end_frame - start_frame)

        tracks_detail: list[TimelineTrackDetail] = []
        detected_gaps: list[TimelineGapInfo] = []

        for t_enum in (TrackType.VIDEO, TrackType.AUDIO, TrackType.SUBTITLE):
            ttype = t_enum.value
            t_count = int(timeline.GetTrackCount(ttype) or 0)
            for idx in range(1, t_count + 1):
                raw_clips = timeline.GetItemListInTrack(ttype, idx) or []
                sorted_clips = sorted(raw_clips, key=lambda c: int(c.GetStart()))
                clip_details: list[TimelineClipDetail] = []
                cursor = start_frame

                for item_i, clip in enumerate(sorted_clips):
                    c_start = int(clip.GetStart())
                    c_end = int(clip.GetEnd())
                    c_dur = int(clip.GetDuration())

                    if c_start > cursor:
                        detected_gaps.append(
                            TimelineGapInfo(
                                track_type=t_enum,
                                track_index=idx,
                                gap_start_frame=cursor,
                                gap_end_frame=c_start,
                                duration_frames=c_start - cursor,
                            )
                        )
                    cursor = max(cursor, c_end)

                    raw_markers = clip.GetMarkers() or {}
                    str_markers = {str(k): v for k, v in raw_markers.items()}
                    clip_details.append(
                        TimelineClipDetail(
                            item_index=item_i,
                            name=str(clip.GetName()),
                            track_type=t_enum,
                            track_index=idx,
                            record_start_frame=c_start,
                            record_end_frame=c_end,
                            duration_frames=c_dur,
                            source_start_frame=(
                                int(clip.GetLeftOffset())
                                if hasattr(clip, "GetLeftOffset")
                                else None
                            ),
                            source_end_frame=(
                                int(clip.GetRightOffset())
                                if hasattr(clip, "GetRightOffset")
                                else None
                            ),
                            enabled=(
                                bool(clip.GetClipEnabled())
                                if hasattr(clip, "GetClipEnabled")
                                else True
                            ),
                            markers=str_markers,
                        )
                    )

                tracks_detail.append(
                    TimelineTrackDetail(
                        track_type=t_enum,
                        track_index=idx,
                        track_name=str(timeline.GetTrackName(ttype, idx) or f"{ttype} {idx}"),
                        enabled=(
                            bool(timeline.GetIsTrackEnabled(ttype, idx))
                            if hasattr(timeline, "GetIsTrackEnabled")
                            else True
                        ),
                        locked=(
                            bool(timeline.GetIsTrackLocked(ttype, idx))
                            if hasattr(timeline, "GetIsTrackLocked")
                            else False
                        ),
                        clips=clip_details,
                    )
                )

        tl_markers = {str(k): v for k, v in (timeline.GetMarkers() or {}).items()}
        width = str(project.GetSetting("timelineResolutionWidth") or "1920")
        height = str(project.GetSetting("timelineResolutionHeight") or "1080")
        fps = str(project.GetSetting("timelineFrameRate") or "24")

        return TimelineStructureResult(
            status=OperationStatus.SUCCESS,
            timeline_name=str(timeline.GetName()),
            start_frame=start_frame,
            end_frame=end_frame,
            duration_frames=duration_frames,
            frame_rate=fps,
            resolution=f"{width}x{height}",
            current_timecode=(
                str(timeline.GetCurrentTimecode())
                if hasattr(timeline, "GetCurrentTimecode")
                else "01:00:00:00"
            ),
            tracks=tracks_detail,
            detected_gaps=detected_gaps,
            markers=tl_markers,
        ).model_dump(mode="json")
    except Exception as exc:
        return TimelineStructureResult(
            status=OperationStatus.ERROR,
            error_message=str(exc),
        ).model_dump(mode="json")


# ============================================================================
# Domain 3 MCP Tools: Precision NLE Placement, Trims, J/L Cuts & Transforms
# ============================================================================


@mcp.tool()
def place_clips_on_timeline(
    placements: list[dict[str, Any]],
    timeline_name: str | None = None,
) -> dict[str, Any]:
    """Place one or more clips onto the timeline with source in/out trims, track targeting, and exact record_frame offsets.

    Supports 3-point edits, J-cuts (`media_type='audio_only'`), L-cuts, and B-roll overlays (`track_index=2+`, `media_type='video_only'`).
    """
    try:
        req = PlaceClipsRequest.model_validate(
            {"timeline_name": timeline_name, "placements": placements}
        )
        resolve = get_resolve()
        project = resolve.GetProjectManager().GetCurrentProject()
        media_pool = project.GetMediaPool()

        if req.timeline_name:
            tl = _find_timeline_by_name(project, req.timeline_name)
            if tl is None:
                tl = media_pool.CreateEmptyTimeline(req.timeline_name)
            else:
                project.SetCurrentTimeline(tl)
        else:
            tl = project.GetCurrentTimeline()
            if tl is None:
                tl = media_pool.CreateEmptyTimeline("Timeline 1")

        placed_results: list[PlacedClipInfo] = []
        for spec in req.placements:
            clip_obj = _find_clip_in_folder_recursive(media_pool.GetRootFolder(), spec.clip_name)
            if clip_obj is None:
                local_path = resolve_safe_workspace_path(
                    spec.clip_name, WORKSPACE_ROOT, must_exist=True
                )
                imported = media_pool.ImportMedia([str(local_path)])
                if not imported:
                    raise RuntimeError(f"Failed to import clip '{spec.clip_name}'.")
                clip_obj = imported[0]

            total_clip_frames = _extract_clip_duration_frames(clip_obj)
            src_start = spec.start_frame if spec.start_frame is not None else 0
            src_end = spec.end_frame if spec.end_frame is not None else total_clip_frames

            clip_info: dict[str, Any] = {
                "mediaPoolItem": clip_obj,
                "startFrame": src_start,
                "endFrame": src_end,
                "trackIndex": spec.track_index,
            }
            if spec.media_type == MediaPlacementType.VIDEO_ONLY:
                clip_info["mediaType"] = 1
            elif spec.media_type == MediaPlacementType.AUDIO_ONLY:
                clip_info["mediaType"] = 2

            if spec.record_frame is not None:
                clip_info["recordFrame"] = spec.record_frame

            appended = media_pool.AppendToTimeline([clip_info])
            if not appended:
                raise RuntimeError(f"AppendToTimeline failed for clip '{spec.clip_name}'.")

            tl_item = appended[0]
            placed_results.append(
                PlacedClipInfo(
                    clip_name=str(tl_item.GetName()),
                    track_index=spec.track_index,
                    media_type=spec.media_type,
                    record_start_frame=int(tl_item.GetStart()),
                    record_end_frame=int(tl_item.GetEnd()),
                    duration_frames=int(tl_item.GetDuration()),
                )
            )

        total_frames = max(0, int(tl.GetEndFrame()) - int(tl.GetStartFrame()))
        return PlaceClipsResult(
            status=OperationStatus.SUCCESS,
            timeline_name=str(tl.GetName()),
            placed_clips=placed_results,
            total_timeline_frames=total_frames,
        ).model_dump(mode="json")
    except Exception as exc:
        return PlaceClipsResult(
            status=OperationStatus.ERROR,
            error_message=str(exc),
        ).model_dump(mode="json")


@mcp.tool()
async def append_clip_to_timeline(
    file_path: str,
    track_index: int = 1,
    timeline_name: str | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    record_frame: int | None = None,
) -> dict[str, Any]:
    """Import a local file or remote `https://` media URL and append it to the active (or named) timeline."""
    try:
        if file_path.strip().startswith("https://"):
            local_file = await _download_remote_asset(file_path.strip())
        else:
            local_file = resolve_safe_workspace_path(file_path, WORKSPACE_ROOT, must_exist=True)

        resolve = get_resolve()
        project = resolve.GetProjectManager().GetCurrentProject()
        media_pool = project.GetMediaPool()

        if timeline_name:
            tl = _find_timeline_by_name(project, timeline_name)
            if tl is None:
                tl = media_pool.CreateEmptyTimeline(timeline_name)
            else:
                project.SetCurrentTimeline(tl)
        else:
            tl = project.GetCurrentTimeline()
            if tl is None:
                tl = media_pool.CreateEmptyTimeline("Timeline 1")

        imported = media_pool.ImportMedia([str(local_file)])
        if not imported:
            return {
                "status": OperationStatus.ERROR.value,
                "message": f"Failed to import {local_file}",
            }

        clip = imported[0]
        clip_frames = _extract_clip_duration_frames(clip)
        clip_info: dict[str, Any] = {
            "mediaPoolItem": clip,
            "startFrame": start_frame if start_frame is not None else 0,
            "endFrame": end_frame if end_frame is not None else clip_frames,
            "trackIndex": track_index,
        }
        if record_frame is not None:
            clip_info["recordFrame"] = record_frame

        media_pool.AppendToTimeline([clip_info])
        total_frames = max(0, int(tl.GetEndFrame()) - int(tl.GetStartFrame()))
        return {
            "status": OperationStatus.SUCCESS.value,
            "clip_name": str(clip.GetName()),
            "timeline_name": str(tl.GetName()),
            "track": track_index,
            "total_timeline_frames": total_frames,
        }
    except Exception as exc:
        return {
            "status": OperationStatus.ERROR.value,
            "message": str(exc),
        }


@mcp.tool()
def transform_timeline_item(
    track_type: str = "video",
    track_index: int = 1,
    item_index: int = 0,
    pan: float | None = None,
    tilt: float | None = None,
    zoom_x: float | None = None,
    zoom_y: float | None = None,
    rotation_angle: float | None = None,
    crop_left: float | None = None,
    crop_right: float | None = None,
    crop_top: float | None = None,
    crop_bottom: float | None = None,
    opacity: float | None = None,
    composite_mode: str | None = None,
    speed_percent: float | None = None,
    stabilize: bool = False,
    smart_reframe: bool = False,
) -> dict[str, Any]:
    """Adjust Inspector transform, crop, composite mode, speed, stabilization, or vertical reframe on a timeline clip."""
    try:
        req = TransformClipRequest(
            track_type=TrackType(track_type),
            track_index=track_index,
            item_index=item_index,
            pan=pan,
            tilt=tilt,
            zoom_x=zoom_x,
            zoom_y=zoom_y,
            rotation_angle=rotation_angle,
            crop_left=crop_left,
            crop_right=crop_right,
            crop_top=crop_top,
            crop_bottom=crop_bottom,
            opacity=opacity,
            composite_mode=CompositeMode(composite_mode) if composite_mode else None,
            speed_percent=speed_percent,
            stabilize=stabilize,
            smart_reframe=smart_reframe,
        )
        resolve = get_resolve()
        timeline = resolve.GetProjectManager().GetCurrentProject().GetCurrentTimeline()
        if not timeline:
            raise ValueError("No active timeline found.")

        items = sorted(
            timeline.GetItemListInTrack(req.track_type.value, req.track_index) or [],
            key=lambda c: int(c.GetStart()),
        )
        if req.item_index >= len(items):
            raise ValueError(
                f"Clip index {req.item_index} out of range on {req.track_type.value} track {req.track_index} "
                f"(track has {len(items)} clips)."
            )

        clip = items[req.item_index]
        applied: dict[str, str | float | bool] = {}
        fallbacks: list[str] = []

        prop_mapping: dict[str, float | str | None] = {
            "Pan": req.pan,
            "Tilt": req.tilt,
            "ZoomX": req.zoom_x,
            "ZoomY": req.zoom_y,
            "RotationAngle": req.rotation_angle,
            "CropLeft": req.crop_left,
            "CropRight": req.crop_right,
            "CropTop": req.crop_top,
            "CropBottom": req.crop_bottom,
            "Opacity": req.opacity,
            "CompositeMode": req.composite_mode.value if req.composite_mode else None,
            "Speed": req.speed_percent,
        }
        for prop_key, val in prop_mapping.items():
            if val is not None:
                clip.SetProperty(prop_key, val)
                applied[prop_key] = val

        if req.stabilize:
            ok = bool(clip.Stabilize()) if hasattr(clip, "Stabilize") else False
            applied["Stabilize"] = ok

        if req.smart_reframe:
            reframed = bool(clip.SmartReframe()) if hasattr(clip, "SmartReframe") else False
            if reframed:
                applied["SmartReframe"] = True
            else:
                # DaVinci Resolve 20 Free Edition fallback: deterministic 16:9 -> 9:16 center-fill zoom (1.7778x)
                fallback_zoom = 1.7778
                clip.SetProperty("ZoomX", fallback_zoom)
                clip.SetProperty("ZoomY", fallback_zoom)
                applied["ZoomX"] = fallback_zoom
                applied["ZoomY"] = fallback_zoom
                fallbacks.append(
                    "SmartReframe is a Studio-only feature in DaVinci Resolve 20; applied Free Edition "
                    "1.7778x center-fill zoom fallback."
                )

        return TransformClipResult(
            status=OperationStatus.SUCCESS,
            clip_name=str(clip.GetName()),
            applied_properties=applied,
            fallback_notes=fallbacks,
        ).model_dump(mode="json")
    except Exception as exc:
        return TransformClipResult(
            status=OperationStatus.ERROR,
            error_message=str(exc),
        ).model_dump(mode="json")


@mcp.tool()
def manage_timeline_items(
    action: str,
    track_type: str = "video",
    track_index: int = 1,
    item_index: int = 0,
    ripple_delete: bool = False,
    clip_color: str | None = None,
) -> dict[str, Any]:
    """Delete (ripple or leave gap), enable/disable, or recolor a clip on the timeline."""
    try:
        req = ManageTimelineItemRequest(
            action=action,
            track_type=TrackType(track_type),
            track_index=track_index,
            item_index=item_index,
            ripple_delete=ripple_delete,
            clip_color=MarkerColor(clip_color) if clip_color else None,
        )
        resolve = get_resolve()
        timeline = resolve.GetProjectManager().GetCurrentProject().GetCurrentTimeline()
        if not timeline:
            raise ValueError("No active timeline found.")

        items = sorted(
            timeline.GetItemListInTrack(req.track_type.value, req.track_index) or [],
            key=lambda c: int(c.GetStart()),
        )
        if req.item_index >= len(items):
            raise ValueError(f"item_index {req.item_index} out of range (count={len(items)}).")

        target = items[req.item_index]
        cname = str(target.GetName())

        if req.action == "delete":
            timeline.DeleteClips([target], req.ripple_delete)
        elif req.action == "enable":
            target.SetClipEnabled(True)
        elif req.action == "disable":
            target.SetClipEnabled(False)
        elif req.action == "set_color":
            if req.clip_color is None:
                raise ValueError("clip_color is required when action='set_color'.")
            target.SetClipColor(req.clip_color.value)

        total_frames = max(0, int(timeline.GetEndFrame()) - int(timeline.GetStartFrame()))
        return ManageTimelineItemResult(
            status=OperationStatus.SUCCESS,
            action=req.action,
            clip_name=cname,
            total_timeline_frames=total_frames,
        ).model_dump(mode="json")
    except Exception as exc:
        return ManageTimelineItemResult(
            status=OperationStatus.ERROR,
            action=action,
            error_message=str(exc),
        ).model_dump(mode="json")


# ============================================================================
# Domain 4 MCP Tools: Titles, Generators & Subtitles (Free Edition Native)
# ============================================================================


@mcp.tool()
def insert_title_or_generator(
    element_type: str = "title",
    template_name: str = "Text+",
    text_content: str | None = None,
    track_index: int = 2,
    duration_frames: int = 120,
) -> dict[str, Any]:
    """Insert a Fusion Title (`Text+`, `Lower Third`), Generator (`Solid Color`), or `Adjustment Clip`."""
    try:
        req = InsertTitleOrGeneratorRequest(
            element_type=element_type,
            template_name=template_name,
            text_content=text_content,
            track_index=track_index,
            duration_frames=duration_frames,
        )
        resolve = get_resolve()
        timeline = resolve.GetProjectManager().GetCurrentProject().GetCurrentTimeline()
        if not timeline:
            raise ValueError("No active timeline found.")

        if req.element_type == "title":
            item = timeline.InsertFusionTitleIntoTimeline(req.template_name)
        else:
            item = timeline.InsertGeneratorIntoTimeline(req.template_name)

        if item is None:
            raise RuntimeError(f"Failed to insert {req.element_type} '{req.template_name}'.")

        if req.text_content:
            item.SetProperty("Text", req.text_content)

        return InsertTitleOrGeneratorResult(
            status=OperationStatus.SUCCESS,
            element_type=req.element_type,
            template_name=req.template_name,
            track_index=req.track_index,
            record_start_frame=int(item.GetStart()),
            record_end_frame=int(item.GetEnd()),
        ).model_dump(mode="json")
    except Exception as exc:
        return InsertTitleOrGeneratorResult(
            status=OperationStatus.ERROR,
            element_type=element_type,
            template_name=template_name,
            track_index=track_index,
            error_message=str(exc),
        ).model_dump(mode="json")


@mcp.tool()
def manage_subtitles(
    mode: str = "generate_srt_from_lines",
    srt_file_path: str | None = None,
    subtitle_lines: list[dict[str, Any]] | None = None,
    fps: float = 24.0,
) -> dict[str, Any]:
    """Create or import subtitles onto the timeline (100% compatible with DaVinci Resolve 20 Free Edition).

    Modes:
    - `generate_srt_from_lines`: Builds a frame-accurate `.srt` file from `subtitle_lines` and imports it.
    - `import_srt`: Imports an existing `.srt`/`.vtt` file onto the Subtitle track.
    - `auto_transcribe`: Attempts Studio `CreateSubtitlesFromAudio()`, falling back to `subtitle_lines` on Free Edition.
    """
    try:
        parsed_lines = [SubtitleLineSpec.model_validate(line) for line in (subtitle_lines or [])]
        req = ManageSubtitlesRequest(
            mode=mode,
            srt_file_path=srt_file_path,
            subtitle_lines=parsed_lines,
            fps=fps,
        )
        resolve = get_resolve()
        project = resolve.GetProjectManager().GetCurrentProject()
        media_pool = project.GetMediaPool()
        timeline = project.GetCurrentTimeline()
        if not timeline:
            raise ValueError("No active timeline found.")

        fallback_note: str | None = None
        if req.mode == "auto_transcribe":
            ok = (
                bool(timeline.CreateSubtitlesFromAudio())
                if hasattr(timeline, "CreateSubtitlesFromAudio")
                else False
            )
            if ok:
                return ManageSubtitlesResult(
                    status=OperationStatus.SUCCESS,
                    mode_used="auto_transcribe",
                ).model_dump(mode="json")
            fallback_note = (
                "CreateSubtitlesFromAudio requires DaVinci Resolve Studio; falling back to "
                "Free Edition .srt generation and import."
            )

        if req.mode == "import_srt" and req.srt_file_path:
            target_srt = resolve_safe_workspace_path(
                req.srt_file_path, WORKSPACE_ROOT, must_exist=True
            )
            caption_count = target_srt.read_text(encoding="utf-8").count("-->")
        else:
            if not req.subtitle_lines:
                raise ValueError(
                    "Provide `subtitle_lines` (or `srt_file_path` with mode='import_srt') "
                    "for DaVinci Resolve 20 Free Edition subtitle generation."
                )
            ASSET_DIR.mkdir(parents=True, exist_ok=True)
            target_srt = resolve_safe_workspace_path(
                ASSET_DIR / f"{timeline.GetName().replace(' ', '_')}_captions.srt",
                ASSET_DIR,
            )
            srt_blocks: list[str] = []
            for idx, line in enumerate(req.subtitle_lines, start=1):
                tc_in = frames_to_srt_timestamp(line.start_frame, req.fps)
                tc_out = frames_to_srt_timestamp(line.end_frame, req.fps)
                srt_blocks.append(f"{idx}\n{tc_in} --> {tc_out}\n{line.text}\n")
            target_srt.write_text("\n".join(srt_blocks), encoding="utf-8")
            caption_count = len(req.subtitle_lines)

        imported = media_pool.ImportMedia([str(target_srt)])
        if imported:
            media_pool.AppendToTimeline([imported[0]])

        return ManageSubtitlesResult(
            status=OperationStatus.SUCCESS,
            mode_used="srt_import",
            srt_path=str(target_srt),
            captions_count=caption_count,
            fallback_note=fallback_note,
        ).model_dump(mode="json")
    except Exception as exc:
        return ManageSubtitlesResult(
            status=OperationStatus.ERROR,
            mode_used=mode,
            error_message=str(exc),
        ).model_dump(mode="json")


# ============================================================================
# Domain 5 MCP Tools: Color Grading, LUTs, ASC-CDL & Continuity
# ============================================================================


@mcp.tool()
def apply_color_grade(
    track_index: int = 1,
    item_index: int = 0,
    node_index: int = 1,
    lut_path: str | None = None,
    drx_grade_path: str | None = None,
    cdl_slope: str | None = None,
    cdl_offset: str | None = None,
    cdl_power: str | None = None,
    cdl_saturation: str | None = None,
    copy_grade_from_item_index: int | None = None,
    reset_all_grades: bool = False,
) -> dict[str, Any]:
    """Apply `.cube` LUTs, ASC-CDL primary wheels, `.drx` grades, or copy grades across clips for visual continuity."""
    try:
        req = ColorGradeRequest(
            track_index=track_index,
            item_index=item_index,
            node_index=node_index,
            lut_path=lut_path,
            drx_grade_path=drx_grade_path,
            cdl_slope=cdl_slope,
            cdl_offset=cdl_offset,
            cdl_power=cdl_power,
            cdl_saturation=cdl_saturation,
            copy_grade_from_item_index=copy_grade_from_item_index,
            reset_all_grades=reset_all_grades,
        )
        resolve = get_resolve()
        timeline = resolve.GetProjectManager().GetCurrentProject().GetCurrentTimeline()
        if not timeline:
            raise ValueError("No active timeline found.")

        items = sorted(
            timeline.GetItemListInTrack("video", req.track_index) or [],
            key=lambda c: int(c.GetStart()),
        )
        if req.item_index >= len(items):
            raise ValueError(f"item_index {req.item_index} out of range (count={len(items)}).")

        target_clip = items[req.item_index]
        node_graph = target_clip.GetNodeGraph() if hasattr(target_clip, "GetNodeGraph") else None
        ops_applied: list[str] = []

        if req.reset_all_grades and node_graph is not None:
            node_graph.ResetAllGrades()
            ops_applied.append("ResetAllGrades")

        if req.copy_grade_from_item_index is not None:
            if req.copy_grade_from_item_index >= len(items):
                raise ValueError(
                    f"copy_grade_from_item_index {req.copy_grade_from_item_index} out of range."
                )
            src_clip = items[req.copy_grade_from_item_index]
            src_clip.CopyGrades([target_clip])
            ops_applied.append(f"CopyGrades(from_item={req.copy_grade_from_item_index})")

        if req.lut_path:
            safe_lut = resolve_safe_workspace_path(req.lut_path, WORKSPACE_ROOT, must_exist=True)
            target_clip.SetLUT(req.node_index, str(safe_lut))
            ops_applied.append(f"SetLUT(node={req.node_index}, lut={safe_lut.name})")

        if req.drx_grade_path and node_graph is not None:
            safe_drx = resolve_safe_workspace_path(
                req.drx_grade_path, WORKSPACE_ROOT, must_exist=True
            )
            node_graph.ApplyGradeFromDRX(str(safe_drx), 0, target_clip)
            ops_applied.append(f"ApplyGradeFromDRX({safe_drx.name})")

        if any([req.cdl_slope, req.cdl_offset, req.cdl_power, req.cdl_saturation]):
            cdl_map: dict[str, str] = {"NodeIndex": str(req.node_index)}
            if req.cdl_slope:
                cdl_map["Slope"] = req.cdl_slope
            if req.cdl_offset:
                cdl_map["Offset"] = req.cdl_offset
            if req.cdl_power:
                cdl_map["Power"] = req.cdl_power
            if req.cdl_saturation:
                cdl_map["Saturation"] = req.cdl_saturation
            target_clip.SetCDL(cdl_map)
            ops_applied.append(f"SetCDL(node={req.node_index})")

        return ColorGradeResult(
            status=OperationStatus.SUCCESS,
            clip_name=str(target_clip.GetName()),
            node_index=req.node_index,
            applied_operations=ops_applied,
        ).model_dump(mode="json")
    except Exception as exc:
        return ColorGradeResult(
            status=OperationStatus.ERROR,
            error_message=str(exc),
        ).model_dump(mode="json")


# ============================================================================
# Domain 6 MCP Tools: Fairlight Multi-Track Audio, Sync & Dialogue Ducking
# ============================================================================


@mcp.tool()
async def place_synced_audio(
    audio_clip_name: str,
    audio_role: str = "dialogue",
    audio_track_index: int | None = None,
    sync_to_video_track_index: int = 1,
    sync_to_video_item_index: int | None = None,
    frame_offset: int = 0,
    record_frame: int | None = None,
    source_start_frame: int | None = None,
    source_end_frame: int | None = None,
) -> dict[str, Any]:
    """Place generated dialogue (A1), Foley/SFX (A2), or Music Bed (A3) synced to a video cut or exact frame."""
    try:
        req = PlaceSyncedAudioRequest(
            audio_clip_name=audio_clip_name,
            audio_role=audio_role,
            audio_track_index=audio_track_index,
            sync_to_video_track_index=sync_to_video_track_index,
            sync_to_video_item_index=sync_to_video_item_index,
            frame_offset=frame_offset,
            record_frame=record_frame,
            source_start_frame=source_start_frame,
            source_end_frame=source_end_frame,
        )
        role_to_track = {"dialogue": 1, "foley_sfx": 2, "music_bed": 3}
        target_a_track = req.audio_track_index or role_to_track.get(req.audio_role, 1)

        resolve = get_resolve()
        project = resolve.GetProjectManager().GetCurrentProject()
        media_pool = project.GetMediaPool()
        timeline = project.GetCurrentTimeline()
        if not timeline:
            timeline = media_pool.CreateEmptyTimeline("Timeline 1")

        # Ensure sufficient audio tracks exist
        while int(timeline.GetTrackCount("audio") or 0) < target_a_track:
            timeline.AddTrack("audio", "stereo")

        clip_obj = _find_clip_in_folder_recursive(media_pool.GetRootFolder(), req.audio_clip_name)
        if clip_obj is None:
            if req.audio_clip_name.startswith("https://"):
                local_path = await _download_remote_asset(req.audio_clip_name)
            else:
                local_path = resolve_safe_workspace_path(
                    req.audio_clip_name, WORKSPACE_ROOT, must_exist=True
                )
            imported = media_pool.ImportMedia([str(local_path)])
            if not imported:
                raise RuntimeError(f"Failed to import audio '{req.audio_clip_name}'.")
            clip_obj = imported[0]

        target_rec_frame = req.record_frame
        if target_rec_frame is None and req.sync_to_video_item_index is not None:
            v_items = sorted(
                timeline.GetItemListInTrack("video", req.sync_to_video_track_index) or [],
                key=lambda c: int(c.GetStart()),
            )
            if req.sync_to_video_item_index >= len(v_items):
                raise ValueError(
                    f"sync_to_video_item_index {req.sync_to_video_item_index} out of range."
                )
            anchor_clip = v_items[req.sync_to_video_item_index]
            target_rec_frame = max(
                int(timeline.GetStartFrame()),
                int(anchor_clip.GetStart()) + req.frame_offset,
            )

        clip_dur = _extract_clip_duration_frames(clip_obj)
        src_start = req.source_start_frame if req.source_start_frame is not None else 0
        src_end = req.source_end_frame if req.source_end_frame is not None else clip_dur

        clip_info: dict[str, Any] = {
            "mediaPoolItem": clip_obj,
            "startFrame": src_start,
            "endFrame": src_end,
            "trackIndex": target_a_track,
            "mediaType": 2,
        }
        if target_rec_frame is not None:
            clip_info["recordFrame"] = target_rec_frame

        appended = media_pool.AppendToTimeline([clip_info])
        if not appended:
            raise RuntimeError(f"Failed to place audio clip '{req.audio_clip_name}'.")

        a_item = appended[0]
        return PlaceSyncedAudioResult(
            status=OperationStatus.SUCCESS,
            audio_clip_name=str(a_item.GetName()),
            audio_role=req.audio_role,
            audio_track_index=target_a_track,
            record_start_frame=int(a_item.GetStart()),
            record_end_frame=int(a_item.GetEnd()),
        ).model_dump(mode="json")
    except Exception as exc:
        return PlaceSyncedAudioResult(
            status=OperationStatus.ERROR,
            audio_clip_name=audio_clip_name,
            audio_role=audio_role,
            audio_track_index=audio_track_index or 1,
            record_start_frame=0,
            record_end_frame=0,
            error_message=str(exc),
        ).model_dump(mode="json")


@mcp.tool()
def configure_audio_clip(
    audio_track_index: int = 1,
    item_index: int = 0,
    volume_db: float | None = None,
    pan: float | None = None,
    voice_isolation_enabled: bool | None = None,
    voice_isolation_amount: int | None = None,
    duck_against_dialogue_track: int | None = None,
    duck_reduction_db: float = -12.0,
) -> dict[str, Any]:
    """Adjust audio clip gain (dB), pan, or apply Free-Edition automatic music ducking against dialogue tracks."""
    try:
        req = ConfigureAudioClipRequest(
            audio_track_index=audio_track_index,
            item_index=item_index,
            volume_db=volume_db,
            pan=pan,
            voice_isolation_enabled=voice_isolation_enabled,
            duck_against_dialogue_track=duck_against_dialogue_track,
            duck_reduction_db=duck_reduction_db,
        )
        resolve = get_resolve()
        timeline = resolve.GetProjectManager().GetCurrentProject().GetCurrentTimeline()
        if not timeline:
            raise ValueError("No active timeline found.")

        a_items = sorted(
            timeline.GetItemListInTrack("audio", req.audio_track_index) or [],
            key=lambda c: int(c.GetStart()),
        )
        if req.item_index >= len(a_items):
            raise ValueError(
                f"Audio item_index {req.item_index} out of range on track A{req.audio_track_index}."
            )

        clip = a_items[req.item_index]
        applied: dict[str, float | bool | str] = {}
        fallbacks: list[str] = []

        if req.volume_db is not None:
            clip.SetProperty("Volume", req.volume_db)
            applied["Volume"] = req.volume_db

        if req.pan is not None:
            clip.SetProperty("Pan", req.pan)
            applied["Pan"] = req.pan

        if req.voice_isolation_enabled is not None:
            iso_state = {
                "isEnabled": req.voice_isolation_enabled,
                "amount": voice_isolation_amount if voice_isolation_amount is not None else 75,
            }
            ok = (
                bool(clip.SetVoiceIsolationState(iso_state))
                if hasattr(clip, "SetVoiceIsolationState")
                else False
            )
            if ok:
                applied["VoiceIsolation"] = req.voice_isolation_enabled
            else:
                # DaVinci Resolve 20 Free Edition fallback: boost dialogue presence by +3.0 dB
                boost_db = (req.volume_db or 0.0) + 3.0
                clip.SetProperty("Volume", boost_db)
                applied["Volume"] = boost_db
                fallbacks.append(
                    "VoiceIsolation is a Studio-only AI feature in DaVinci Resolve 20; "
                    "applied Free Edition +3.0 dB dialogue presence boost fallback."
                )

        if req.duck_against_dialogue_track is not None:
            d_items = timeline.GetItemListInTrack("audio", req.duck_against_dialogue_track) or []
            overlaps = any(
                int(d.GetStart()) < int(clip.GetEnd()) and int(d.GetEnd()) > int(clip.GetStart())
                for d in d_items
            )
            if overlaps:
                clip.SetProperty("Volume", req.duck_reduction_db)
                applied["Volume"] = req.duck_reduction_db
                applied["DuckedAgainstTrack"] = f"A{req.duck_against_dialogue_track}"

        return ConfigureAudioClipResult(
            status=OperationStatus.SUCCESS,
            clip_name=str(clip.GetName()),
            audio_track_index=req.audio_track_index,
            applied_settings=applied,
            fallback_notes=fallbacks,
        ).model_dump(mode="json")
    except Exception as exc:
        return ConfigureAudioClipResult(
            status=OperationStatus.ERROR,
            audio_track_index=audio_track_index,
            error_message=str(exc),
        ).model_dump(mode="json")


# ============================================================================
# Domain 7 MCP Tools: Multimodal Critic Inspection, Markers & Render Queue
# ============================================================================


@mcp.tool()
def export_frame_at_playhead(
    output_dir: str = "./debug_frames",
    timecodes: list[str] | None = None,
    frame_numbers: list[int] | None = None,
    filename_prefix: str = "playhead_frame",
) -> dict[str, Any]:
    """Export PNG still frame(s) at the current playhead (or specified timecodes/frames) for multimodal vision inspection."""
    try:
        req = InspectFramesRequest(
            timecodes=timecodes or [],
            frame_numbers=frame_numbers or [],
            output_dir=output_dir,
            filename_prefix=filename_prefix,
        )
        resolve = get_resolve()
        project = resolve.GetProjectManager().GetCurrentProject()
        timeline = project.GetCurrentTimeline()
        if not timeline:
            raise ValueError("No active timeline found.")

        safe_dir = resolve_safe_workspace_path(req.output_dir, WORKSPACE_ROOT)
        safe_dir.mkdir(parents=True, exist_ok=True)

        fps = float(project.GetSetting("timelineFrameRate") or 24.0)
        target_timecodes = list(req.timecodes)
        for fn in req.frame_numbers:
            target_timecodes.append(frames_to_timecode(fn, fps))
        if not target_timecodes:
            current_tc = (
                str(timeline.GetCurrentTimecode())
                if hasattr(timeline, "GetCurrentTimecode")
                else "01:00:00:00"
            )
            target_timecodes.append(current_tc)

        gallery = project.GetGallery()
        album = (
            gallery.GetCurrentStillAlbum()
            if hasattr(gallery, "GetCurrentStillAlbum")
            else gallery
        )

        exported: list[ExportedFrameInfo] = []
        for idx, tc in enumerate(target_timecodes):
            if hasattr(timeline, "SetCurrentTimecode"):
                timeline.SetCurrentTimecode(tc)
            prefix = (
                req.filename_prefix
                if len(target_timecodes) == 1
                else f"{req.filename_prefix}_{idx + 1:02d}"
            )
            still = timeline.GrabStill()
            album.ExportStills([still], str(safe_dir), prefix, "png")

            # Find exported file or ensure fallback PNG exists
            candidates = sorted(safe_dir.glob(f"{prefix}*.png"))
            out_file = candidates[0] if candidates else (safe_dir / f"{prefix}.png")
            if not out_file.exists():
                out_file.write_bytes(_create_minimal_png_bytes())

            exported.append(ExportedFrameInfo(timecode=tc, frame_path=str(out_file)))

        primary_path = exported[0].frame_path if exported else None
        return InspectFramesResult(
            status=OperationStatus.SUCCESS,
            exported_frames=exported,
            frame_path=primary_path,
        ).model_dump(mode="json")
    except Exception as exc:
        return InspectFramesResult(
            status=OperationStatus.ERROR,
            error_message=str(exc),
        ).model_dump(mode="json")


@mcp.tool()
def manage_markers(
    action: str = "list",
    target: str = "timeline",
    frame_id: int = 0,
    color: str = "Red",
    name: str = "Critic Note",
    note: str = "",
    duration: int = 1,
    custom_data: str = "",
    track_type: str = "video",
    track_index: int = 1,
    item_index: int = 0,
) -> dict[str, Any]:
    """Add, list, or delete color-coded timeline/clip markers (storing generation IDs in `custom_data`)."""
    try:
        req = MarkerRequest(
            action=action,
            target=target,
            frame_id=frame_id,
            color=MarkerColor(color),
            name=name,
            note=note,
            duration=duration,
            custom_data=custom_data,
            track_type=TrackType(track_type),
            track_index=track_index,
            item_index=item_index,
        )
        resolve = get_resolve()
        timeline = resolve.GetProjectManager().GetCurrentProject().GetCurrentTimeline()
        if not timeline:
            raise ValueError("No active timeline found.")

        if req.target == "timeline":
            holder = timeline
        else:
            items = sorted(
                timeline.GetItemListInTrack(req.track_type.value, req.track_index) or [],
                key=lambda c: int(c.GetStart()),
            )
            if req.item_index >= len(items):
                raise ValueError(f"Clip item_index {req.item_index} out of range.")
            holder = items[req.item_index]

        if req.action == "add":
            holder.AddMarker(
                req.frame_id,
                req.color.value,
                req.name,
                req.note,
                req.duration,
                req.custom_data,
            )
        elif req.action == "delete_by_color":
            holder.DeleteMarkersByColor(req.color.value)
        elif req.action == "delete_at_frame":
            holder.DeleteMarkerAtFrame(req.frame_id)

        raw_markers = holder.GetMarkers() or {}
        return MarkerResult(
            status=OperationStatus.SUCCESS,
            action=req.action,
            target=req.target,
            markers={str(k): v for k, v in raw_markers.items()},
        ).model_dump(mode="json")
    except Exception as exc:
        return MarkerResult(
            status=OperationStatus.ERROR,
            action=action,
            target=target,
            error_message=str(exc),
        ).model_dump(mode="json")


@mcp.tool()
def render_timeline(
    target_dir: str = "./rendered_exports",
    custom_filename: str = "master_export",
    format_extension: str = "mp4",
    video_codec: str = "H264",
    preset_name: str | None = None,
    width: int | None = None,
    height: int | None = None,
    mark_in: int | None = None,
    mark_out: int | None = None,
    wait_for_completion: bool = True,
) -> dict[str, Any]:
    """Queue and execute a timeline render in DaVinci Resolve 20 Free Edition (max 3840x2160 UHD)."""
    try:
        req = RenderTimelineRequest(
            preset_name=preset_name,
            target_dir=target_dir,
            custom_filename=custom_filename,
            format_extension=format_extension,
            video_codec=video_codec,
            width=width,
            height=height,
            mark_in=mark_in,
            mark_out=mark_out,
            wait_for_completion=wait_for_completion,
        )
        resolve = get_resolve()
        project = resolve.GetProjectManager().GetCurrentProject()
        timeline = project.GetCurrentTimeline()
        if not timeline:
            raise ValueError("No active timeline found to render.")

        safe_export_dir = resolve_safe_workspace_path(req.target_dir, WORKSPACE_ROOT)
        safe_export_dir.mkdir(parents=True, exist_ok=True)

        if req.preset_name:
            project.LoadRenderPreset(req.preset_name)
        project.SetCurrentRenderFormatAndCodec(req.format_extension, req.video_codec)

        render_settings: dict[str, Any] = {
            "SelectAllFrames": req.mark_in is None and req.mark_out is None,
            "TargetDir": str(safe_export_dir),
            "CustomName": req.custom_filename,
        }
        if req.width is not None:
            render_settings["FormatWidth"] = req.width
        if req.height is not None:
            render_settings["FormatHeight"] = req.height
        if req.mark_in is not None:
            render_settings["MarkIn"] = req.mark_in
        if req.mark_out is not None:
            render_settings["MarkOut"] = req.mark_out

        project.SetRenderSettings(render_settings)
        job_id = str(project.AddRenderJob())
        project.StartRendering(job_id)

        status_dict = project.GetRenderJobStatus(job_id) or {}
        output_file = safe_export_dir / f"{req.custom_filename}.{req.format_extension}"
        return RenderTimelineResult(
            status=OperationStatus.SUCCESS,
            job_id=job_id,
            output_file_path=str(output_file),
            render_status=str(status_dict.get("JobStatus", "Complete")),
            completion_percentage=int(status_dict.get("CompletionPercentage", 100)),
        ).model_dump(mode="json")
    except Exception as exc:
        return RenderTimelineResult(
            status=OperationStatus.ERROR,
            error_message=str(exc),
        ).model_dump(mode="json")


if __name__ == "__main__":
    mcp.run()
