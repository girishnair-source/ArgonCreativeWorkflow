"""Passive, read-only reader for DaVinci Resolve's project library (BUILD_SPEC_v3 §10.4.2, §10.5).

Resolve Free 21 has no external scripting, so AGY never *asks* Resolve anything. Reads are passive: the live
``Project.db`` (+ ``-wal`` / ``-shm``) is COPIED into a private temp dir (exactly the
``resolve_sync_trigger._db_snapshot`` pattern) and the copy is opened read-only (``file:...?mode=ro``,
``uri=True``). Only parameterized SELECTs run; identifiers are constants or validated names coming from
``sqlite_master`` / ``PRAGMA table_info``. The live file is never opened by SQLite.

Marker storage — verified on a read-only copy of the real Resolve 21.1.1.10 library (LikeFollowKillOrDie_S01)
    * Timeline markers live in ``BtLockableBlob`` rows (``DbType = 'Sm2SequenceLockableBlob'``) whose
      ``BlobOwner`` is the timeline's ``Sm2Sequence_id`` (== ``Sm2Timeline.Sequence``). ``BtLockableBlobMap_id``
      is one project-wide map row (no per-timeline indirection). Timelines without markers have no row.
    * ``FieldsBlob`` is a Qt ``QDataStream`` property map ``u32 version(=1) | u32 count | (QString key,
      QVariant value)*`` with the single key ``BlobData`` (QVariant type 12 = QByteArray).
    * ``BlobData`` (and every other Resolve 21 ``FieldsBlob``) is an envelope ``u32 BE tag (10001) |
      u32 BE n | payload[n]`` where ``payload = 0x81 + one zstd frame``.
    * The decompressed payload is protobuf: ``2: { 2: marker* }`` with
      ``marker = 1: {1: frame as zig-zag sint64, frames from the timeline start; empty when 0}`` and
      ``2: bytes = u32 BE 2 | u32 BE n | protobuf {1: {1: colour code, 3: note, 3: duration (text),
      3: name, 6: custom data}}``. Colour codes seen: Green = 8, Cyan = 4 (we read them as ``2 ** (i + 1)`` over
      Resolve's colour list — a hypothesis consistent with both observations).
    * Clip markers would be in ``Sm2TiItem.MarkersBA`` (NULL everywhere in the real DB) — decoded with the same
      pipeline. ``Sm2TiItemLockableBlob`` rows (``BlobOwner`` = item id) are scanned for the strict marker layout
      only (they mostly carry grades / clip data and are otherwise not part of the fingerprint).

Timeline fingerprint (§10.4.2)
    sha256 over the ORDERED tuple of every timeline item ``(track type, track index, clip name, source in/out,
    record in/out, enabled/flags, media path, media ref, retime-map hash, item markers)`` plus the timeline markers.
    * UI-state columns (playhead/zoom in ``Sm2Sequence.UIElementsState``, ``UiMemento``, thumbnails, render
      caches, change/save timestamps, lock ids, the uninitialised ``CurrentSelectorIdx``) are never hashed.
    * AGY-authored markers are excluded by NAME: every marker whose name/strings start with ``AGY_REGEN_READY``
      plus any name passed in ``exclude_marker_names`` (``ResolveHandoff.agy_marker_names``).
    * Fail closed: a marker blob that can not be decoded is hashed RAW and is never excluded by name.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import re
import shutil
import sqlite3
import struct
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import quote

from pydantic import Field

from .fsio import canonical_json, now_iso
from .models import Model

try:  # Python 3.14+: stdlib zstd (Resolve 21 compresses every FieldsBlob payload with zstd)
    from compression import zstd as _zstd
except ImportError:  # Python < 3.14: fallback to system libzstd via ctypes if present
    _zstd = None
    try:
        import ctypes
        import ctypes.util

        _lib_candidates = [
            "/opt/homebrew/lib/libzstd.dylib",
            "/opt/homebrew/lib/libzstd.1.dylib",
            "/usr/local/lib/libzstd.dylib",
            ctypes.util.find_library("zstd") or "",
        ]
        _libzstd = None
        for _cand in _lib_candidates:
            if _cand and Path(_cand).exists() or (_cand and not _cand.startswith("/")):
                try:
                    _libzstd = ctypes.CDLL(_cand)
                    break
                except OSError:
                    continue
        if _libzstd is not None:
            _libzstd.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
            _libzstd.ZSTD_compressBound.restype = ctypes.c_size_t
            _libzstd.ZSTD_compress.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int,
            ]
            _libzstd.ZSTD_compress.restype = ctypes.c_size_t
            _libzstd.ZSTD_decompress.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t,
            ]
            _libzstd.ZSTD_decompress.restype = ctypes.c_size_t
            _libzstd.ZSTD_isError.argtypes = [ctypes.c_size_t]
            _libzstd.ZSTD_isError.restype = ctypes.c_uint

            class _CtypesZstd:
                @staticmethod
                def compress(data: bytes, level: int = 3) -> bytes:
                    bound = _libzstd.ZSTD_compressBound(len(data))
                    buf = ctypes.create_string_buffer(bound)
                    n = _libzstd.ZSTD_compress(buf, bound, data, len(data), level)
                    if _libzstd.ZSTD_isError(n):
                        raise ValueError("zstd compress error")
                    return buf.raw[:n]

                @staticmethod
                def decompress(data: bytes) -> bytes:
                    for cap in (max(4096, len(data) * 8), 256 * 1024, 4 * 1024 * 1024):
                        buf = ctypes.create_string_buffer(cap)
                        n = _libzstd.ZSTD_decompress(buf, cap, data, len(data))
                        if not _libzstd.ZSTD_isError(n):
                            return buf.raw[:n]
                    raise ValueError("zstd decompress error")

            _zstd = _CtypesZstd()
    except Exception:  # pragma: no cover
        _zstd = None

RESOLVE_SUPPORT_DIR = Path.home() / "Library" / "Application Support" / "Blackmagic Design" / "DaVinci Resolve"
RESOLVE_PROJECTS_DIR = (RESOLVE_SUPPORT_DIR / "Resolve Project Library" / "Resolve Projects" / "Users" / "guest"
                        / "Projects")
RECEIPTS_BIN = "_AGY_RECEIPTS"
AGY_MARKER_PREFIX = "AGY_REGEN_READY"
FINGERPRINT_VERSION = "agyfp2"
ZSTD_AVAILABLE = _zstd is not None
KNOWN_MARKER_LINKAGE = "direct:BlobOwner"
TRACK_TYPES = {0: "video", 1: "audio", 2: "subtitle"}
_TRACK_ORDER = {"video": 0, "audio": 1, "subtitle": 2}
RESOLVE_MARKER_COLORS = ("Blue", "Cyan", "Green", "Yellow", "Red", "Pink", "Purple", "Fuchsia", "Rose", "Lavender",
                         "Sky", "Mint", "Lemon", "Sand", "Cocoa", "Cream")
RESOLVE_COLOR_CODES = {c: 1 << (i + 1) for i, c in enumerate(RESOLVE_MARKER_COLORS)}  # observed: Cyan 4, Green 8
_CODE_TO_COLOR = {v: k for k, v in RESOLVE_COLOR_CODES.items()}
_COLOR_LOOKUP = {c.lower(): c for c in (*RESOLVE_MARKER_COLORS, "Orange")}
# Never read by the fingerprint (documented for reviewers and asserted by tests): pure UI / cache / bookkeeping.
UI_STATE_COLUMNS = frozenset({
    "UiMemento", "UIElementsState", "CurPlayheadPosition", "Folded", "Thumbnail", "ThumbnailDirtyFlag",
    "RenderCacheBA", "AuxRenderCacheBA", "LastChangedTime", "LastRenderedTime", "DbSavedTime", "ModTimeInSecs",
    "IsMarkedForCaching", "LockStatus", "BlobLockSysId", "LockSysId", "VersionTableLockSysId", "CompTableLockSysId",
    "CurrentSelectorIdx",  # holds uninitialised garbage in the real DB (1714761776, -10854554, ...)
})
_ITEM_COLUMNS = ("Sm2TiItem_id", "Name", "Start", "Duration", "In", "Flags", "MediaFilePath", "MediaRef",
                 "CurrentSelectorIdx", "MediaTimemapBA", "MarkersBA", "Sm2TiTrack_id")
_TRACK_LINK_TABLES = ("Sm2Sequence_Sm2TiTrack", "Sm2SequenceContainer_Sm2TiTrack")
_IDENT_RE = re.compile(r"^[A-Za-z0-9_:]{1,128}$")
_PROJECT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,127}$")
_SYS_COL_RE = re.compile(r"(?i)sys|lock")
_SKIP_MAP_COLS = frozenset({"DbType", "DbIndex", "DbPropertyName"})
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_QT_BYTEARRAY = 12
_MAX_DECODED = 64 << 20
_MAX_ITEM_BLOB = 256 << 10


def default_db_path(resolve_project_name: str) -> Path:
    """``Project.db`` of a Resolve project in the local (guest) project library."""
    if (not isinstance(resolve_project_name, str) or not _PROJECT_NAME_RE.match(resolve_project_name)
            or ".." in resolve_project_name):
        raise ValueError(f"invalid Resolve project name {resolve_project_name!r}")
    return RESOLVE_PROJECTS_DIR / resolve_project_name / "Project.db"


# ---------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------
class MarkerRow(Model):
    scope: str = "timeline"  # timeline | item
    frame: float | None = None  # timeline markers: frames from the timeline start
    name: str = ""
    note: str = ""
    color: str | None = None
    color_code: int | None = None
    duration: float | None = None
    custom_data: str | None = None
    strings: list[str] = Field(default_factory=list)  # every readable string of the marker message (evidence)
    fields_hash: str | None = None  # hash of the full decoded message (any field change is detected)
    raw_hash: str | None = None  # set when the blob could not be decoded (fail closed)
    decoded: bool = True
    layout: str | None = None  # resolve21 | generic | raw | agy_fake


class ItemRow(Model):
    track_type: str
    track_index: int
    name: str = ""
    start: float = 0  # record in (absolute timeline frames)
    duration: float = 0
    source_in: float = 0
    enabled: bool | None = None  # None = unknown (DB: carried inside ``flags``)
    flags: int | None = None
    media_path: str | None = None
    media_ref: str | None = None
    selector_idx: int | None = None  # evidence only (never hashed)
    retime_hash: str | None = None
    item_id: str | None = None
    markers: list[MarkerRow] = Field(default_factory=list)

    @property
    def record_out(self) -> float:
        return self.start + self.duration

    @property
    def source_out(self) -> float:
        return self.source_in + self.duration


class TimelineSnapshot(Model):
    name: str
    timeline_id: str | None = None
    sequence_ids: list[str] = Field(default_factory=list)
    start_frame: float | None = None
    fps: float | None = None
    items: list[ItemRow] = Field(default_factory=list)
    markers: list[MarkerRow] = Field(default_factory=list)
    markers_decoded: bool = True
    marker_linkage: str | None = None
    fingerprint: str | None = None
    read_at: str
    db_mtime: float | None = None

    def fingerprint_excluding(self, exclude_marker_names: Iterable[str] = ()) -> str:
        return compute_fingerprint(self.items, self.markers, exclude_marker_names=exclude_marker_names)

    def track_items(self, track_type: str = "video", track_index: int = 1) -> list[ItemRow]:
        return sorted((i for i in self.items if i.track_type == track_type and i.track_index == track_index),
                      key=lambda i: (float(i.start), float(i.source_in), i.name or ""))

    def base_frame(self) -> float:
        """Frame the timeline starts at (Resolve default 01:00:00:00); approximated by the first item if unknown."""
        if self.start_frame is not None:
            return float(self.start_frame)
        starts = [float(i.start) for i in self.items]
        return min(starts) if starts else 0.0

    def all_markers(self) -> list[MarkerRow]:
        out = list(self.markers)
        for it in self.items:
            out.extend(it.markers)
        return out


# ---------------------------------------------------------------------------------------------
# Fingerprint (shared with the in-memory FakeTransport so both compute identical hashes)
# ---------------------------------------------------------------------------------------------
def _norm(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        f = float(x)
        if not math.isfinite(f):
            return str(f)
        return int(f) if f.is_integer() else round(f, 6)
    return x


def marker_excluded(marker: MarkerRow, exclude: Iterable[str] = ()) -> bool:
    """AGY-authored marker?  Undecoded (raw) markers are NEVER excluded (fail closed)."""
    if not marker.decoded:
        return False
    names = set(exclude or ())
    for text in (marker.name, *marker.strings):
        if text and (text in names or text.startswith(AGY_MARKER_PREFIX)):
            return True
    return False


def _marker_key(m: MarkerRow) -> list[Any]:
    if not m.decoded:
        return ["raw", m.scope, m.raw_hash]
    return ["m", m.scope, _norm(m.frame), m.name, m.note, m.color, m.color_code, _norm(m.duration), m.custom_data,
            m.fields_hash]


def _item_sort_key(it: ItemRow) -> tuple:
    return (_TRACK_ORDER.get(it.track_type, 9), it.track_type, it.track_index, float(it.start or 0),
            float(it.source_in or 0), it.name or "", it.media_path or "", it.item_id or "")


def compute_fingerprint(items: Iterable[ItemRow], markers: Iterable[MarkerRow], *,
                        exclude_marker_names: Iterable[str] = ()) -> str:
    exclude = set(exclude_marker_names or ())
    rows = []
    for it in sorted(items, key=_item_sort_key):
        rows.append([
            it.track_type, it.track_index, it.name,
            _norm(it.source_in), _norm(float(it.source_in) + float(it.duration)),
            _norm(it.start), _norm(float(it.start) + float(it.duration)),
            it.enabled, it.flags, it.media_path, it.media_ref, it.retime_hash,
            sorted(canonical_json(_marker_key(m)) for m in it.markers if not marker_excluded(m, exclude)),
        ])
    payload = {
        "v": FINGERPRINT_VERSION,
        "items": rows,
        "markers": sorted(canonical_json(_marker_key(m)) for m in markers if not marker_excluded(m, exclude)),
    }
    return f"{FINGERPRINT_VERSION}:{hashlib.sha256(canonical_json(payload).encode('utf-8')).hexdigest()}"


# ---------------------------------------------------------------------------------------------
# Resolve 21 blob envelopes: Qt property map, ``u32 tag | u32 n | 0x81 + zstd`` framing
# ---------------------------------------------------------------------------------------------
def _zstd_decompress(data: bytes) -> bytes | None:
    if _zstd is None:
        return None
    try:
        out = _zstd.decompress(data)
    except Exception:  # noqa: BLE001 - corrupt / truncated frame -> caller fails closed
        return None
    return out if len(out) <= _MAX_DECODED else None


def _unframe(blob: bytes) -> tuple[bool, bytes | None]:
    """``(recognised, payload)`` for ``u32 BE tag | u32 BE n | payload[n]`` (n == len - 8)."""
    if len(blob) < 9:
        return False, None
    _tag, n = struct.unpack(">II", blob[:8])
    if n != len(blob) - 8:
        return False, None
    payload = blob[8:]
    if payload[:1] == b"\x81" and payload[1:5] == _ZSTD_MAGIC:
        return True, _zstd_decompress(payload[1:])
    if payload[:4] == _ZSTD_MAGIC:
        return True, _zstd_decompress(payload)
    return True, payload


def _qt_value(blob: bytes, pos: int, vtype: int) -> tuple[Any, int]:
    fixed = {1: 1, 2: 4, 3: 4, 4: 8, 5: 8, 6: 8, 7: 2, 38: 4, 33: 2, 34: 1, 36: 2, 37: 1}
    if vtype in fixed:
        size = fixed[vtype]
        if pos + size > len(blob):
            raise ValueError("truncated QVariant")
        return blob[pos:pos + size], pos + size
    if vtype in (10, _QT_BYTEARRAY):
        (ln,) = struct.unpack(">I", blob[pos:pos + 4])
        pos += 4
        if ln == 0xFFFFFFFF:
            return (None, pos)
        if pos + ln > len(blob):
            raise ValueError("truncated QVariant payload")
        chunk = blob[pos:pos + ln]
        return (chunk.decode("utf-16-be") if vtype == 10 else chunk), pos + ln
    raise ValueError(f"unsupported QVariant type {vtype}")


def qt_property_map(blob: bytes) -> list[tuple[str, int, Any]] | None:
    """Parse a Qt ``QDataStream`` property map ``u32 version(1) | u32 count | (QString key, QVariant)*``."""
    try:
        if len(blob) < 8:
            return None
        version, count = struct.unpack(">II", blob[:8])
        if version != 1 or not 0 < count <= 4096:
            return None
        pos = 8
        out: list[tuple[str, int, Any]] = []
        for _ in range(count):
            (klen,) = struct.unpack(">I", blob[pos:pos + 4])
            pos += 4
            if klen == 0xFFFFFFFF or klen % 2 or klen > 4096 or pos + klen > len(blob):
                return None
            key = blob[pos:pos + klen].decode("utf-16-be")
            pos += klen
            (vtype,) = struct.unpack(">I", blob[pos:pos + 4])
            pos += 5  # type + isNull flag
            value, pos = _qt_value(blob, pos, vtype)
            out.append((key, vtype, value))
        return out if pos == len(blob) else None
    except (struct.error, UnicodeDecodeError, ValueError):
        return None


def unwrap_blob(blob: bytes, depth: int = 0) -> tuple[list[bytes], bool]:
    """Resolve blob -> protobuf payload candidates. ``ok`` is False when a recognised envelope could not be
    decoded (zstd unavailable / corrupt frame); callers then fail closed."""
    if depth > 4:
        return [], False
    qt = qt_property_map(blob)
    if qt is not None:
        payloads: list[bytes] = []
        ok = True
        for _key, vtype, value in qt:
            if vtype == _QT_BYTEARRAY and value:
                sub, sub_ok = unwrap_blob(bytes(value), depth + 1)
                payloads.extend(sub)
                ok = ok and sub_ok
        return payloads, ok
    recognised, payload = _unframe(blob)
    if recognised:
        return ([payload], True) if payload is not None else ([], False)
    return [blob], True


# ---------------------------------------------------------------------------------------------
# Generic protobuf wire-format decoding (no schema) + marker extraction
# ---------------------------------------------------------------------------------------------
def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("truncated varint")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7
        if shift >= 70:
            raise ValueError("varint too long")


def _as_text(chunk: bytes) -> str | None:
    if not chunk:
        return ""
    try:
        text = chunk.decode("utf-8")
    except UnicodeDecodeError:
        return None
    for ch in text:
        if ch in "\t\n\r":
            continue
        if not ch.isprintable():
            return None
    return text


def _parse_message(buf: bytes, depth: int = 0) -> list[tuple[str, int, Any]] | None:
    """Parse ``buf`` as a protobuf message; None if it is not a clean message (fail closed upstream)."""
    if depth > 12 or len(buf) > _MAX_DECODED:
        return None
    out: list[tuple[str, int, Any]] = []
    pos = 0
    n = len(buf)
    try:
        while pos < n:
            key, pos = _varint(buf, pos)
            fno, wt = key >> 3, key & 7
            if fno < 1 or fno > (1 << 29) - 1:
                return None
            if wt == 0:
                val, pos = _varint(buf, pos)
                out.append(("v", fno, val))
            elif wt == 1:
                if pos + 8 > n:
                    return None
                out.append(("f64", fno, buf[pos:pos + 8]))
                pos += 8
            elif wt == 2:
                ln, pos = _varint(buf, pos)
                if ln > n - pos:
                    return None
                out.append(_classify_chunk(fno, buf[pos:pos + ln], depth))
                pos += ln
            elif wt == 5:
                if pos + 4 > n:
                    return None
                out.append(("f32", fno, buf[pos:pos + 4]))
                pos += 4
            else:
                return None
    except ValueError:
        return None
    return out


def _classify_chunk(fno: int, chunk: bytes, depth: int) -> tuple[str, int, Any]:
    sub = _parse_message(chunk, depth + 1) if chunk else None
    if sub and chunk[0] < 0x20:  # message keys for low field numbers are control bytes: prefer the message
        return ("m", fno, sub)
    text = _as_text(chunk)
    if text is not None:
        return ("s", fno, text)
    if sub:
        return ("m", fno, sub)
    if len(chunk) >= 9:  # Resolve frames some sub-messages as ``u32 BE version | u32 BE n | protobuf``
        _ver, ln = struct.unpack(">II", chunk[:8])
        if ln == len(chunk) - 8:
            inner = _parse_message(chunk[8:], depth + 1)
            if inner:
                return ("m", fno, inner)
    return ("b", fno, chunk.hex())


def _canon(children: list[tuple[str, int, Any]]) -> list[Any]:
    out = []
    for kind, fno, val in children:
        if kind == "m":
            out.append([kind, fno, _canon(val)])
        elif isinstance(val, (bytes, bytearray)):
            out.append([kind, fno, bytes(val).hex()])
        else:
            out.append([kind, fno, val])
    return out


def _is_marker_like(children: list[tuple[str, int, Any]]) -> bool:
    has_str = any(k == "s" and str(v).strip() for k, _, v in children)
    has_num = any(k in ("v", "f64", "f32") for k, _, _ in children)
    return has_str and has_num


def _marker_messages(root: list[tuple[str, int, Any]]) -> list[list[tuple[str, int, Any]]]:
    def walk(children: list[tuple[str, int, Any]]) -> list[list[tuple[str, int, Any]]]:
        found: list[list[tuple[str, int, Any]]] = []
        for kind, _fno, val in children:
            if kind != "m":
                continue
            inner = walk(val)
            if inner:
                found.extend(inner)
            elif _is_marker_like(val):
                found.append(val)
        return found

    result = walk(root)
    if not result and _is_marker_like(root):
        result = [root]
    return result


def _num_value(kind: str, val: Any) -> float | None:
    if kind == "v":
        return float(val)
    raw = bytes(val)
    if kind == "f64":
        d = struct.unpack("<d", raw)[0]
        if math.isfinite(d) and abs(d) < 1e12:
            return d
        return float(int.from_bytes(raw, "little"))
    if kind == "f32":
        f = struct.unpack("<f", raw)[0]
        if math.isfinite(f) and abs(f) < 1e9:
            return f
        return float(int.from_bytes(raw, "little"))
    return None


def _sha(data: bytes | str, length: int = 24) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:length]


def _unzigzag(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


def _zigzag(n: int) -> int:
    return (n << 1) ^ (n >> 63)


def color_from_code(code: int | None) -> str | None:
    return _CODE_TO_COLOR.get(int(code)) if code is not None else None


def _split_marker_texts(texts: list[str]) -> tuple[str, float | None, str]:
    """Observed field-3 order: ``[note, duration, name]``."""
    def as_num(t: str | None) -> float | None:
        return float(t) if t is not None and re.fullmatch(r"\s*\d+(\.\d+)?\s*", t) else None

    if len(texts) >= 3:
        return texts[0], as_num(texts[1]), texts[-1]
    if len(texts) == 2:
        if as_num(texts[0]) is not None:
            return "", as_num(texts[0]), texts[1]
        if as_num(texts[1]) is not None:
            return texts[0], as_num(texts[1]), ""
        return texts[0], None, texts[1]
    return "", None, texts[0] if texts else ""


def _resolve_marker(children: list[tuple[str, int, Any]], scope: str, *, strict: bool = False) -> MarkerRow | None:
    """One marker in the Resolve 21 layout (see module docstring), else None."""
    data_wrap = next((v for k, f, v in children if f == 2 and k == "m"), None)
    if data_wrap is None:
        return None
    data = next((v for k, f, v in data_wrap if f == 1 and k == "m"), None)
    if data is None:
        return None
    texts = [str(v) for k, f, v in data if f == 3 and k == "s"]
    if not texts:
        return None
    pos_part = next(((k, v) for k, f, v in children if f == 1), None)
    frame = 0
    if pos_part is None:
        if strict:
            return None
    else:
        kind, val = pos_part
        if kind == "m":
            frame = _unzigzag(next((x for kk, ff, x in val if ff == 1 and kk == "v"), 0))
        elif kind == "v":
            frame = _unzigzag(val)
        elif not (kind == "s" and val == ""):
            return None
    code = next((v for k, f, v in data if f == 1 and k == "v"), None)
    custom = next((str(v) for k, f, v in data if f == 6 and k == "s"), None)
    note, duration, name = _split_marker_texts(texts)
    return MarkerRow(scope=scope, frame=float(frame), name=name, note=note, color=color_from_code(code),
                     color_code=code, duration=duration, custom_data=custom,
                     strings=texts + ([custom] if custom else []),
                     fields_hash=_sha(canonical_json(_canon(children))), decoded=True, layout="resolve21")


def _interpret_marker(children: list[tuple[str, int, Any]], scope: str) -> MarkerRow:
    """Generic fallback for unknown layouts (strings + numbers)."""
    strings = [str(v) for k, _, v in children if k == "s"]
    nums = [(k, v) for k, _, v in children if k in ("v", "f64", "f32")]
    color = None
    rest: list[str] = []
    for s in strings:
        key = s.strip().lower()
        if color is None and key in _COLOR_LOOKUP:
            color = _COLOR_LOOKUP[key]
        else:
            rest.append(s)
    frame = _num_value(*nums[0]) if nums else None
    duration = _num_value(*nums[1]) if len(nums) > 1 else None
    return MarkerRow(scope=scope, frame=frame, name=rest[0] if rest else "", note=rest[1] if len(rest) > 1 else "",
                     color=color, duration=duration, custom_data=rest[2] if len(rest) > 2 else None,
                     strings=strings, fields_hash=_sha(canonical_json(_canon(children))), decoded=True,
                     layout="generic")


def _extract_markers(tree: list[tuple[str, int, Any]], scope: str, *, strict: bool = False) -> list[MarkerRow]:
    found: list[MarkerRow] = []

    def walk(children: list[tuple[str, int, Any]]) -> None:
        for kind, _fno, val in children:
            if kind != "m":
                continue
            row = _resolve_marker(val, scope, strict=strict)
            if row is not None:
                found.append(row)
                continue
            walk(val)

    walk(tree)
    if found or strict:
        return found
    return [_interpret_marker(msg, scope) for msg in _marker_messages(tree)]


def _parse_payload(payload: bytes) -> list[tuple[str, int, Any]] | None:
    tree = _parse_message(payload)
    if tree is None and len(payload) > 1 and payload[0] in (0x00, 0x01, 0x80):
        tree = _parse_message(payload[1:])
    return tree


_ASCII_RUN = re.compile(rb"[\x20-\x7e]{4,}")
_UTF16_RUN = re.compile(rb"(?:\x00[\x20-\x7e]){4,}|(?:[\x20-\x7e]\x00){4,}")


def strings_in_blob(blob: bytes, limit: int = 40) -> list[str]:
    """Readable ASCII and UTF-16 runs (evidence for undecodable blobs)."""
    out: list[str] = []
    for m in _ASCII_RUN.finditer(blob):
        out.append(m.group(0).decode("ascii"))
    for m in _UTF16_RUN.finditer(blob):
        raw = m.group(0)
        text = raw.decode("utf-16-be" if raw[0] == 0 else "utf-16-le", errors="ignore")
        if text and text not in out:
            out.append(text)
    return out[:limit]


def decode_marker_blob(blob: bytes | None, *, scope: str = "timeline") -> tuple[list[MarkerRow], bool]:
    """Decode one marker blob. Returns ``(markers, decoded_ok)``; undecodable blobs yield one RAW marker row."""
    if not blob:
        return [], True
    blob = bytes(blob)
    payloads, ok = unwrap_blob(blob)
    markers: list[MarkerRow] = []
    if ok:
        for payload in payloads:
            tree = _parse_payload(payload)
            if tree is None:
                ok = False
                break
            markers.extend(_extract_markers(tree, scope))
    if ok:
        return markers, True
    strings = strings_in_blob(blob)
    return [MarkerRow(scope=scope, decoded=False, raw_hash=_sha(blob, 64), strings=strings,
                      name=strings[0] if strings else "", layout="raw")], False


def strict_markers_in_blob(blob: bytes | None, *, scope: str = "item") -> list[MarkerRow]:
    """Markers in the exact Resolve 21 layout only (used for ``Sm2TiItemLockableBlob`` rows); never fails."""
    if not blob or len(blob) > _MAX_ITEM_BLOB:
        return []
    payloads, ok = unwrap_blob(bytes(blob))
    if not ok:
        return []
    out: list[MarkerRow] = []
    for payload in payloads:
        tree = _parse_payload(payload)
        if tree:
            out.extend(_extract_markers(tree, scope, strict=True))
    return out


# ---- encoder (synthetic fixtures / tests reproduce the exact on-disk layout) -------------------
def _enc_varint(n: int) -> bytes:
    if n < 0:
        raise ValueError("varint must be >= 0")
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _enc_len(fno: int, data: bytes) -> bytes:
    return _enc_varint((fno << 3) | 2) + _enc_varint(len(data)) + data


def _enc_uint(fno: int, n: int) -> bytes:
    return _enc_varint(fno << 3) + _enc_varint(n)


def encode_marker_blob(markers: Iterable[dict[str, Any]], *, compress: bool = True) -> bytes:
    """Timeline-marker ``BtLockableBlob.FieldsBlob`` in the layout observed in Resolve 21 (fixtures / tests).

    Each marker dict: ``frame`` (from timeline start), ``color`` (Resolve colour name), ``name``, ``note``,
    ``duration`` (frames), ``custom_data``.
    """
    entries = []
    for m in markers:
        frame = int(m.get("frame") or 0)
        pos = _enc_uint(1, _zigzag(frame)) if frame else b""
        code = RESOLVE_COLOR_CODES.get(str(m.get("color") or "Blue"), RESOLVE_COLOR_CODES["Blue"])
        data = (_enc_uint(1, code) + _enc_len(3, str(m.get("note") or "").encode("utf-8"))
                + _enc_len(3, str(int(m.get("duration") or 1)).encode("utf-8"))
                + _enc_len(3, str(m.get("name") or "").encode("utf-8")))
        if m.get("custom_data"):
            data += _enc_len(6, str(m["custom_data"]).encode("utf-8"))
        inner = _enc_len(1, data)
        framed = struct.pack(">II", 2, len(inner)) + inner
        entries.append(_enc_len(2, _enc_len(1, pos) + _enc_len(2, framed)))
    root = _enc_len(2, b"".join(entries))
    if compress:
        if _zstd is None:
            raise RuntimeError("zstd is not available in this interpreter (Python >= 3.14 required)")
        payload = b"\x81" + _zstd.compress(root)
    else:
        payload = b"\x01" + root
    blob_data = struct.pack(">II", 10001, len(payload)) + payload
    key = "BlobData".encode("utf-16-be")
    return (struct.pack(">II", 1, 1) + struct.pack(">I", len(key)) + key + struct.pack(">I", _QT_BYTEARRAY)
            + b"\x00" + struct.pack(">I", len(blob_data)) + blob_data)


# ---------------------------------------------------------------------------------------------
# SQLite helpers (read-only copies, validated identifiers, parameterized values)
# ---------------------------------------------------------------------------------------------
def _q(name: str) -> str:
    if not isinstance(name, str) or not _IDENT_RE.match(name):
        raise ValueError(f"unsafe SQL identifier {name!r}")
    return '"' + name + '"'


def _bytes(v: Any) -> bytes:
    if v is None:
        return b""
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v)
    if isinstance(v, str):
        return v.encode("utf-8")
    return str(v).encode("utf-8")


def _num(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (bytes, bytearray, memoryview)):
        try:
            v = bytes(v).decode("ascii")
        except UnicodeDecodeError:
            return None
    try:
        return float(str(v).strip())
    except ValueError:
        return None


def _int(v: Any) -> int | None:
    f = _num(v)
    if f is None or not math.isfinite(f):
        return None
    return int(f)


def _text(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v).decode("utf-8", errors="replace")
    return str(v)


def _le_doubles(b: bytes) -> tuple[float, ...] | None:
    if len(b) < 8 or len(b) % 8:
        return None
    vals = struct.unpack("<" + "d" * (len(b) // 8), b)
    return vals if all(math.isfinite(v) for v in vals) else None


def _has_table(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?", (table,)).fetchone() is not None


def _cols(conn: sqlite3.Connection, table: str) -> list[str]:
    if not _has_table(conn, table):
        return []
    return [r[1] for r in conn.execute(f"PRAGMA table_info({_q(table)})")]


def _rows(conn: sqlite3.Connection, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    cur = conn.execute(sql, tuple(params))
    names = [d[0] for d in cur.description or ()]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def _placeholders(n: int) -> str:
    return ",".join("?" for _ in range(n))


def _chunks(seq: list[Any], size: int = 500) -> Iterator[list[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ---------------------------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------------------------
class ResolveDB:
    """Read-only view of a Resolve ``Project.db`` (always through a temp COPY)."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path).expanduser()
        self._linkage_cache: dict[tuple[float, int], tuple[str, str | None]] = {}

    # ---- snapshot -------------------------------------------------------------------------
    def exists(self) -> bool:
        return self.db_path.is_file()

    def mtime(self) -> float | None:
        return self.db_path.stat().st_mtime if self.db_path.exists() else None

    def _signature(self) -> tuple:
        sig = []
        for p in (self.db_path, self.db_path.with_name(self.db_path.name + "-wal")):
            sig.append((p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else None)
        return tuple(sig)

    @contextlib.contextmanager
    def snapshot(self) -> Iterator[sqlite3.Connection]:
        """Copy Project.db (+ -wal/-shm) to a private temp dir and yield a READ-ONLY connection on the copy."""
        if not self.db_path.is_file():
            raise FileNotFoundError(f"Resolve project database not found: {self.db_path}")
        with tempfile.TemporaryDirectory(prefix="agy_resolve_db_") as tmp:
            snap = None
            for attempt in range(4):  # retry when Resolve writes while we copy
                target = Path(tmp) / f"a{attempt}"
                target.mkdir()
                snap = target / "Project.db"
                before = self._signature()
                shutil.copy2(self.db_path, snap)
                for side in ("-wal", "-shm"):
                    src = self.db_path.with_name(self.db_path.name + side)
                    if src.exists():
                        shutil.copy2(src, snap.with_name(snap.name + side))
                if self._signature() == before:
                    break
                time.sleep(0.2)
            conn = sqlite3.connect(f"file:{quote(str(snap))}?mode=ro", uri=True)
            try:
                conn.execute("PRAGMA query_only = ON")
                yield conn
            finally:
                conn.close()

    @contextlib.contextmanager
    def _use(self, conn: sqlite3.Connection | None) -> Iterator[sqlite3.Connection]:
        if conn is not None:
            yield conn
            return
        with self.snapshot() as fresh:
            yield fresh

    # ---- timelines ------------------------------------------------------------------------
    def timelines(self, *, conn: sqlite3.Connection | None = None) -> list[str]:
        with self._use(conn) as c:
            if not _has_table(c, "Sm2Timeline"):
                return []
            return sorted({_text(r[0]) or "" for r in c.execute(f"SELECT {_q('Name')} FROM {_q('Sm2Timeline')}")}
                          - {""})

    def timeline(self, name: str, *, conn: sqlite3.Connection | None = None) -> TimelineSnapshot | None:
        with self._use(conn) as c:
            return self._timeline(c, name)

    def snapshots(self, names: Iterable[str], *, conn: sqlite3.Connection | None = None
                  ) -> dict[str, TimelineSnapshot | None]:
        names = list(dict.fromkeys(names))
        if not names:
            return {}
        with self._use(conn) as c:
            return {n: self._timeline(c, n) for n in names}

    def fingerprint(self, name: str, *, exclude_marker_names: Iterable[str] = (),
                    conn: sqlite3.Connection | None = None) -> str | None:
        snap = self.timeline(name, conn=conn)
        return snap.fingerprint_excluding(exclude_marker_names) if snap else None

    def marker_linkage_mode(self, *, conn: sqlite3.Connection | None = None) -> str:
        """Project-wide marker linkage (``direct:BlobOwner`` on Resolve 21) / ``global`` / ``none``."""
        with self._use(conn) as c:
            return self._linkage(c)[0]

    def _timeline(self, c: sqlite3.Connection, name: str) -> TimelineSnapshot | None:
        if not _has_table(c, "Sm2Timeline"):
            return None
        tcols = _cols(c, "Sm2Timeline")
        sel = [x for x in ("Sm2Timeline_id", "Sequence", "FieldsBlob") if x in tcols]
        tl_rows = _rows(c, f"SELECT {', '.join(_q(x) for x in sel)} FROM {_q('Sm2Timeline')} WHERE {_q('Name')} = ? "
                           f"ORDER BY {_q('Sm2Timeline_id')}", (name,))
        if not tl_rows:
            return None
        items: list[ItemRow] = []
        markers: list[MarkerRow] = []
        decoded_all = True
        linkage = None
        seq_ids: list[str] = []
        start_frame: float | None = None
        fps: float | None = None
        for tl in tl_rows:
            tid = tl.get("Sm2Timeline_id")
            seq_rows = self._sequence_rows(c, tid, tl.get("Sequence"))
            sids = [str(s["Sm2Sequence_id"]) for s in seq_rows if s.get("Sm2Sequence_id") is not None]
            seq_ids.extend(sids)
            sf, f = self._seq_timing(seq_rows)
            start_frame = start_frame if start_frame is not None else sf
            fps = fps if fps is not None else f
            tracks = self._tracks(c, sids)
            its, ok_items = self._items(c, tracks)
            items.extend(its)
            mks, linkage, ok_markers = self._timeline_markers(c, tl, seq_rows)
            markers.extend(mks)
            decoded_all = decoded_all and ok_items and ok_markers
        snap = TimelineSnapshot(
            name=name, timeline_id=str(tl_rows[0].get("Sm2Timeline_id")), sequence_ids=seq_ids,
            start_frame=start_frame, fps=fps, items=items, markers=markers, markers_decoded=decoded_all,
            marker_linkage=linkage, read_at=now_iso(), db_mtime=self.mtime(),
        )
        snap.fingerprint = compute_fingerprint(items, markers)
        return snap

    @staticmethod
    def _seq_timing(seq_rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
        """(start frame, fps) from ``Sm2Sequence.MediaExtents`` (LE doubles: start s, length s) + ``FrameRate``."""
        for s in seq_rows:
            fr = _le_doubles(_bytes(s.get("FrameRate")))
            fps = fr[0] if fr and 1.0 <= fr[0] <= 240.0 else None
            ext = _le_doubles(_bytes(s.get("MediaExtents")))
            if fps and ext and 0 <= ext[0] < 864000:
                return float(round(ext[0] * fps)), fps
            if fps:
                return None, fps
        return None, None

    def _sequence_rows(self, c: sqlite3.Connection, timeline_id: Any, seq_ref: Any) -> list[dict[str, Any]]:
        scols = _cols(c, "Sm2Sequence")
        if not scols:
            return []
        want = [x for x in ("Sm2Sequence_id", "UniqueSequenceId", "BlobLockSysId", "LockSysId",
                            "VersionTableLockSysId", "FieldsBlob", "MediaExtents", "FrameRate") if x in scols]
        out = _rows(c, f"SELECT {', '.join(_q(x) for x in want)} FROM {_q('Sm2Sequence')} "
                       f"WHERE {_q('Sm2Timeline_id')} = ? ORDER BY {_q('Sm2Sequence_id')}", (timeline_id,)) \
            if "Sm2Timeline_id" in scols else []
        known = {r.get("Sm2Sequence_id") for r in out}
        if seq_ref is not None and seq_ref not in known:
            out += _rows(c, f"SELECT {', '.join(_q(x) for x in want)} FROM {_q('Sm2Sequence')} "
                            f"WHERE {_q('Sm2Sequence_id')} = ?", (seq_ref,))
        return out

    def _tracks(self, c: sqlite3.Connection, seq_ids: list[str]) -> dict[str, tuple[str, int]]:
        trcols = _cols(c, "Sm2TiTrack")
        if not trcols or not seq_ids:
            return {}
        conds = [f"{_q(col)} IN ({_placeholders(len(seq_ids))})" for col in ("Sequence", "Sm2Sequence_id")
                 if col in trcols]
        if not conds:
            return {}
        params: list[Any] = []
        for _ in conds:
            params.extend(seq_ids)
        rows = _rows(c, f"SELECT {_q('Sm2TiTrack_id')} AS tid, {_q('Type')} AS ttype, rowid AS rid "
                        f"FROM {_q('Sm2TiTrack')} WHERE {' OR '.join(conds)}", params)
        order: dict[str, int | None] = {}
        link_tables = [(t, _cols(c, t)) for t in _TRACK_LINK_TABLES]
        for r in rows:
            order[r["tid"]] = None
            for link, lcols in link_tables:
                if "DbAssociate" in lcols and "DbIndex" in lcols:
                    hit = c.execute(f"SELECT MIN({_q('DbIndex')}) FROM {_q(link)} WHERE {_q('DbAssociate')} = ?",
                                    (r["tid"],)).fetchone()
                    if hit and hit[0] is not None:
                        order[r["tid"]] = _int(hit[0])
                        break
        by_type: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            code = _int(r["ttype"])
            ttype = TRACK_TYPES.get(code if code is not None else -1, f"other{_text(r['ttype'])}")
            by_type.setdefault(ttype, []).append(r)
        out: dict[str, tuple[str, int]] = {}
        for ttype, trs in by_type.items():
            trs.sort(key=lambda r: (order[r["tid"]] is None, order[r["tid"]] or 0, r["rid"]))
            for idx, r in enumerate(trs, start=1):
                out[r["tid"]] = (ttype, idx)
        return out

    def _items(self, c: sqlite3.Connection, tracks: dict[str, tuple[str, int]]) -> tuple[list[ItemRow], bool]:
        icols = _cols(c, "Sm2TiItem")
        if not icols or not tracks or "Sm2TiTrack_id" not in icols:
            return [], True
        select = ", ".join(f"{_q(col)} AS {_q('c_' + col)}" if col in icols else f"NULL AS {_q('c_' + col)}"
                           for col in _ITEM_COLUMNS)
        tids = list(tracks)
        rows = _rows(c, f"SELECT {select} FROM {_q('Sm2TiItem')} WHERE {_q('Sm2TiTrack_id')} "
                        f"IN ({_placeholders(len(tids))})", tids)
        blob_markers = self._item_blob_markers(c, [_text(r["c_Sm2TiItem_id"]) for r in rows
                                                   if r["c_Sm2TiItem_id"] is not None])
        items: list[ItemRow] = []
        ok_all = True
        for r in rows:
            ttype, tidx = tracks[r["c_Sm2TiTrack_id"]]
            mk, ok = decode_marker_blob(_bytes(r["c_MarkersBA"]), scope="item")
            ok_all = ok_all and ok
            iid = _text(r["c_Sm2TiItem_id"])
            mk = mk + blob_markers.get(iid or "", [])
            timemap = _bytes(r["c_MediaTimemapBA"])
            items.append(ItemRow(
                track_type=ttype, track_index=tidx, name=_text(r["c_Name"]) or "",
                start=_num(r["c_Start"]) or 0.0, duration=_num(r["c_Duration"]) or 0.0,
                source_in=_num(r["c_In"]) or 0.0, enabled=None, flags=_int(r["c_Flags"]),
                media_path=_text(r["c_MediaFilePath"]), media_ref=_text(r["c_MediaRef"]),
                selector_idx=_int(r["c_CurrentSelectorIdx"]), retime_hash=_sha(timemap) if timemap else None,
                item_id=iid, markers=mk,
            ))
        return items, ok_all

    def _item_blob_markers(self, c: sqlite3.Connection, item_ids: list[str | None]) -> dict[str, list[MarkerRow]]:
        bcols = _cols(c, "BtLockableBlob")
        ids = [i for i in item_ids if i]
        if not ids or not {"BlobOwner", "FieldsBlob"} <= set(bcols):
            return {}
        out: dict[str, list[MarkerRow]] = {}
        id_sel = _q("BtLockableBlob_id") if "BtLockableBlob_id" in bcols else "rowid"
        for chunk in _chunks(ids):
            for owner, _bid, blob in c.execute(
                    f"SELECT {_q('BlobOwner')}, {id_sel}, {_q('FieldsBlob')} FROM {_q('BtLockableBlob')} "
                    f"WHERE {_q('BlobOwner')} IN ({_placeholders(len(chunk))}) ORDER BY {id_sel}", chunk):
                found = strict_markers_in_blob(_bytes(blob), scope="item")
                if found:
                    out.setdefault(str(owner), []).extend(found)
        return out

    # ---- marker linkage ---------------------------------------------------------------------
    @staticmethod
    def _owner_keys(tl: dict[str, Any], seq_rows: list[dict[str, Any]]) -> tuple[set[Any], set[Any]]:
        str_keys: set[Any] = set()
        sys_keys: set[Any] = set()
        for v in (tl.get("Sm2Timeline_id"), *[s.get("Sm2Sequence_id") for s in seq_rows],
                  *[s.get("UniqueSequenceId") for s in seq_rows]):
            if isinstance(v, str) and len(v) >= 6:
                str_keys.add(v)
        for s in seq_rows:
            for col in ("BlobLockSysId", "LockSysId", "VersionTableLockSysId"):
                v = s.get(col)
                if v not in (None, "", 0, b""):
                    sys_keys.add(v)
        return str_keys, sys_keys

    def _blobs_by_method(self, c: sqlite3.Connection, method: str, tl: dict[str, Any],
                         seq_rows: list[dict[str, Any]]) -> list[tuple[str, bytes]]:
        bcols = _cols(c, "BtLockableBlob")
        if "FieldsBlob" not in bcols:
            return []
        id_sel = _q("BtLockableBlob_id") if "BtLockableBlob_id" in bcols else "rowid"
        str_keys, sys_keys = self._owner_keys(tl, seq_rows)
        found: dict[str, bytes] = {}

        def grab(col: str, values: Iterable[Any]) -> None:
            for v in values:
                for rid, blob in c.execute(f"SELECT {id_sel}, {_q('FieldsBlob')} FROM {_q('BtLockableBlob')} "
                                           f"WHERE {_q(col)} = ?", (v,)):
                    found[str(rid)] = _bytes(blob)

        kind, _, col = method.partition(":")
        if kind == "direct" and col in bcols:
            grab(col, str_keys | (sys_keys if _SYS_COL_RE.search(col) else set()))
        elif kind == "map" and _has_table(c, "BtLockableBlobMap"):
            mcols = _cols(c, "BtLockableBlobMap")
            if col in mcols:
                values: set[Any] = set()
                keys = str_keys | (sys_keys if _SYS_COL_RE.search(col) else set())
                for k in keys:
                    for row in _rows(c, f"SELECT * FROM {_q('BtLockableBlobMap')} WHERE {_q(col)} = ?", (k,)):
                        for cc, vv in row.items():
                            if cc == col or cc in _SKIP_MAP_COLS or vv in (None, ""):
                                continue
                            if isinstance(vv, str) and len(vv) >= 6:
                                values.add(vv)
                            elif isinstance(vv, int) and _SYS_COL_RE.search(cc + "id") and vv > 0:
                                values.add(vv)
                for bc in bcols:
                    if bc != "FieldsBlob" and _IDENT_RE.match(bc):
                        grab(bc, values)
        elif kind == "reverse":
            owner_blobs = [_bytes(s.get("FieldsBlob")) for s in seq_rows] + [_bytes(tl.get("FieldsBlob"))]
            owner_blobs = [b for b in owner_blobs if b]
            if owner_blobs and "BtLockableBlob_id" in bcols:
                for rid, blob in c.execute(f"SELECT {_q('BtLockableBlob_id')}, {_q('FieldsBlob')} "
                                           f"FROM {_q('BtLockableBlob')}"):
                    if isinstance(rid, str) and len(rid) >= 8 and any(rid.encode() in ob for ob in owner_blobs):
                        found[rid] = _bytes(blob)
        elif kind == "embedded":
            for k in str_keys:
                for rid, blob in c.execute(f"SELECT {id_sel}, {_q('FieldsBlob')} FROM {_q('BtLockableBlob')} "
                                           f"WHERE instr({_q('FieldsBlob')}, ?) > 0", (k.encode("utf-8"),)):
                    found[str(rid)] = _bytes(blob)
        elif kind == "global":
            for rid, blob in c.execute(f"SELECT {id_sel}, {_q('FieldsBlob')} FROM {_q('BtLockableBlob')}"):
                b = _bytes(blob)
                mk, ok = decode_marker_blob(b)
                if ok and mk:
                    found[str(rid)] = b
        return sorted(found.items())

    def _candidate_methods(self, c: sqlite3.Connection) -> list[str]:
        bcols = [x for x in _cols(c, "BtLockableBlob") if x != "FieldsBlob"]
        methods = [KNOWN_MARKER_LINKAGE] if "BlobOwner" in bcols else []
        methods += [f"direct:{x}" for x in bcols if f"direct:{x}" not in methods and x not in UI_STATE_COLUMNS]
        methods += [f"map:{x}" for x in _cols(c, "BtLockableBlobMap") if x not in _SKIP_MAP_COLS]
        methods += ["reverse", "embedded"]
        return methods

    def _linkage(self, c: sqlite3.Connection) -> tuple[str, str | None]:
        try:
            st = self.db_path.stat()
            cache_key = (st.st_mtime, st.st_size)
        except OSError:
            cache_key = (0.0, 0)
        if cache_key in self._linkage_cache:
            return self._linkage_cache[cache_key]
        result: tuple[str, str | None] = ("none", None)
        if "FieldsBlob" in _cols(c, "BtLockableBlob") and _has_table(c, "Sm2Timeline"):
            tcols = _cols(c, "Sm2Timeline")
            sel = [x for x in ("Sm2Timeline_id", "Sequence", "FieldsBlob", "Name") if x in tcols]
            timelines = _rows(c, f"SELECT {', '.join(_q(x) for x in sel)} FROM {_q('Sm2Timeline')} "
                                 f"ORDER BY {_q('Sm2Timeline_id')} LIMIT 200")
            for method in self._candidate_methods(c):
                for tl in timelines:
                    seq_rows = self._sequence_rows(c, tl.get("Sm2Timeline_id"), tl.get("Sequence"))
                    blobs = self._blobs_by_method(c, method, tl, seq_rows)
                    if any(decode_marker_blob(b)[0] for _, b in blobs):
                        result = (method, _text(tl.get("Name")))
                        break
                if result[0] != "none":
                    break
            if result[0] == "none" and self._blobs_by_method(c, "global", {}, []):
                result = ("global", None)
        self._linkage_cache[cache_key] = result
        return result

    def _timeline_markers(self, c: sqlite3.Connection, tl: dict[str, Any], seq_rows: list[dict[str, Any]]
                          ) -> tuple[list[MarkerRow], str, bool]:
        method, _ = self._linkage(c)
        if method == "none":
            # No marker blob anywhere yet: still read the verified Resolve 21 linkage (the first marker a human
            # adds lands there), so a fresh project cannot hide markers from the fingerprint.
            method = KNOWN_MARKER_LINKAGE if "BlobOwner" in _cols(c, "BtLockableBlob") else "none"
            if method == "none":
                return [], "none", True
        markers: list[MarkerRow] = []
        ok_all = True
        for _rid, blob in self._blobs_by_method(c, method, tl, seq_rows):
            mk, ok = decode_marker_blob(blob, scope="timeline")
            markers.extend(mk)
            ok_all = ok_all and ok
        return markers, method, ok_all

    # ---- media pool ---------------------------------------------------------------------------
    def _folder_paths(self, c: sqlite3.Connection) -> list[tuple[str, list[str]]]:
        fcols = _cols(c, "Sm2MpFolder")
        if not {"Sm2MpFolder_id", "Name"} <= set(fcols):
            return []
        parent_col = "MpFolder" if "MpFolder" in fcols else ("Sm2MpFolder_Owner_id" if "Sm2MpFolder_Owner_id" in fcols
                                                             else None)
        sel = f"{_q('Sm2MpFolder_id')}, {_q('Name')}" + (f", {_q(parent_col)}" if parent_col else ", NULL")
        rows = [(r[0], _text(r[1]) or "", r[2]) for r in c.execute(f"SELECT {sel} FROM {_q('Sm2MpFolder')}")]
        by_id = {fid: (name, parent) for fid, name, parent in rows}
        out: list[tuple[str, list[str]]] = []
        for fid, _name, _parent in rows:
            segs: list[str] = []
            cur = fid
            seen = set()
            while cur in by_id and cur not in seen:
                seen.add(cur)
                name, parent = by_id[cur]
                if parent in (None, "") or parent not in by_id:
                    break  # ``cur`` is the media pool root ("Master") - not part of the path
                segs.append(name)
                cur = parent
            if segs:
                out.append((str(fid), list(reversed(segs))))
        return out

    def bins(self, *, conn: sqlite3.Connection | None = None) -> list[str]:
        with self._use(conn) as c:
            return sorted({"/" + "/".join(segs) for _fid, segs in self._folder_paths(c)})

    def receipts(self, prefix: str = RECEIPTS_BIN, *, conn: sqlite3.Connection | None = None
                 ) -> dict[str, list[str]]:
        """``/<prefix>/<run_id>/<field>`` sub-bins written by the AGY_Bridge Lua script, keyed by run id."""
        with self._use(conn) as c:
            out: dict[str, set[str]] = {}
            for _fid, segs in self._folder_paths(c):
                if len(segs) >= 2 and segs[0] == prefix:
                    out.setdefault(segs[1], set())
                    if len(segs) >= 3:
                        out[segs[1]].add(segs[2])
            return {k: sorted(v) for k, v in sorted(out.items())}

    # ---- diagnostics ---------------------------------------------------------------------------
    def describe(self, timeline_name: str | None = None) -> dict[str, Any]:
        """Schema / linkage summary used by the integrator (names and counts only, no media content)."""
        with self.snapshot() as c:
            tables = {t: _has_table(c, t) for t in ("Sm2Timeline", "Sm2Sequence", "Sm2TiTrack", "Sm2TiItem",
                                                     "Sm2MpFolder", "BtLockableBlob", "BtLockableBlobMap",
                                                     *_TRACK_LINK_TABLES)}
            method, example = self._linkage(c)
            info: dict[str, Any] = {
                "tables": tables,
                "zstd_available": ZSTD_AVAILABLE,
                "BtLockableBlob_columns": _cols(c, "BtLockableBlob"),
                "n_timelines": len(self.timelines(conn=c)),
                "marker_linkage": method,
                "marker_linkage_example_timeline": example,
            }
            if timeline_name:
                snap = self._timeline(c, timeline_name)
                if snap:
                    info["timeline"] = {
                        "name": snap.name, "items": len(snap.items), "start_frame": snap.start_frame,
                        "fps": snap.fps, "tracks": sorted({f"{i.track_type}{i.track_index}" for i in snap.items}),
                        "markers": len(snap.markers), "markers_decoded": snap.markers_decoded,
                        "marker_samples": [{"name": m.name, "note": m.note[:60], "color": m.color,
                                            "color_code": m.color_code, "frame": m.frame, "duration": m.duration,
                                            "layout": m.layout} for m in snap.markers[:12]],
                        "fingerprint": snap.fingerprint,
                    }
            return info
