"""Resolve Bridge — BUILD_SPEC_v3 §10 (non-destructive bins, timelines, metadata, markers, ownership & preflight).

Key guarantees:
* **Additive only (§10.4.4):** :func:`assert_lua_additive` statically rejects any Lua script containing delete, move,
  relink, replace, or rename calls. Receipt bins under ``/_AGY_RECEIPTS/<run_id>/`` are never deleted.
* **No clobbering (§10.4.1, §10.4.5, acceptance criterion 8):** once a timeline is handed off to the human (or adopted
  from the legacy pipeline), AGY never writes clips or edits to it. :meth:`ResolveBridge.preflight` raises
  :class:`ClobberError` if any operation targets an existing or human-owned timeline.
* **Post-handoff regens (§10.4.6, acceptance criterion 8):** :meth:`ResolveBridge.deliver_regen` places new takes into
  ``/REGEN/<SC_ID>/`` and optionally adds an Orange ``AGY_REGEN_READY`` marker whose name is recorded in
  ``ResolveHandoff.agy_marker_names`` (excluded from the timeline fingerprint), so the human timeline's fingerprint
  remains bit-for-bit identical.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from pydantic import Field

from . import editor, fsio
from .models import (
    MarkerColor,
    Model,
    Ownership,
    ResolveHandoff,
    Take,
)
from .procs import run
from .project import Project, parse_artifact_id
from .resolve_db import (
    AGY_MARKER_PREFIX,
     RECEIPTS_BIN,
    RESOLVE_COLOR_CODES,
    RESOLVE_SUPPORT_DIR,
    ItemRow,
    MarkerRow,
    TimelineSnapshot,
    compute_fingerprint,
)

AGENT = "agy.resolve_bridge"
UTILITY_SCRIPTS_DIR = RESOLVE_SUPPORT_DIR / "Fusion" / "Scripts" / "Utility"
BRIDGE_SCRIPT_NAME = "AGY_Bridge.lua"

_FORBIDDEN_LUA_RE = re.compile(
    r"\b("
    r"DeleteClips|DeleteTimelines|DeleteFolder|DeleteSubFolder|DeleteSubFolders|"
    r"DeleteMarker|DeleteMarkersByColor|DeleteMarkInOut|"
    r"MoveClips|MoveSubFolder|RelinkClips|ReplaceClip|"
    r"ImportIntoTimeline|SetName|"
    r"os\.remove|os\.execute|io\.open|io\.popen"
    r")\b"
)


class ClobberError(PermissionError):
    """Raised when an operation would overwrite or modify an existing or human-owned Resolve timeline."""


class BridgeOp(Model):
    """One additive Resolve operation."""

    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


class Transport(Protocol):
    def execute(self, ops: Sequence[BridgeOp], *, run_id: str) -> dict[str, Any]: ...


def assert_lua_additive(lua_source: str) -> None:
    """Static guard (§10.4.4): reject any Lua source that calls destructive Resolve or OS primitives."""
    m = _FORBIDDEN_LUA_RE.search(lua_source or "")
    if m:
        raise ClobberError(f"Forbidden non-additive Lua call detected: {m.group(1)!r}")


def _marker_row(
    *,
    frame: float,
    color: str,
    name: str,
    note: str = "",
    duration: float = 1.0,
    custom_data: str = "",
    scope: str = "timeline",
) -> MarkerRow:
    c_norm = color.strip().capitalize() if color else "Yellow"
    code = RESOLVE_COLOR_CODES.get(c_norm, 0)
    strings = [s for s in (name, note, custom_data) if s]
    payload_str = fsio.canonical_json(
        {"frame": frame, "color": c_norm, "name": name, "note": note, "duration": duration, "custom": custom_data}
    )
    f_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()[:16]
    return MarkerRow(
        scope=scope,
        frame=float(frame),
        name=name,
        note=note,
        color=c_norm,
        color_code=code,
        duration=float(duration),
        custom_data=custom_data,
        strings=strings,
        fields_hash=f_hash,
        decoded=True,
        layout="agy_fake",
    )


def _snapshot_from_cut(timeline_name: str, cut: editor.SceneCut) -> TimelineSnapshot:
    base = float(cut.start_frame)
    items: list[ItemRow] = []
    for c in cut.clips:
        t_type = "video" if c.track.startswith("V") else "audio"
        t_idx = int(c.track[1:])
        items.append(
            ItemRow(
                track_type=t_type,
                track_index=t_idx,
                name=c.name,
                start=base + float(c.record_in),
                duration=float(c.duration),
                source_in=float(c.source_in),
                enabled=bool(c.enabled),
                flags=1 if c.enabled else 0,
                media_path=c.media,
                media_ref=c.take_id or c.clip_id or c.key,
                item_id=c.key,
                markers=[],
            )
        )
    markers = [
        _marker_row(
            frame= float(m.frame),
            color=m.color.value.capitalize(),
            name=m.name,
            note=m.note,
            duration=float(m.duration),
            custom_data=m.custom_data,
        )
        for m in cut.markers
    ]
    fp = compute_fingerprint(items, markers)
    return TimelineSnapshot(
        name=timeline_name,
        timeline_id=f"tl:{timeline_name}",
        sequence_ids=[f"seq:{timeline_name}"],
        start_frame=base,
        fps=float(cut.fps),
        items=items,
        markers=markers,
        markers_decoded=True,
        marker_linkage="agy_fake",
        fingerprint=fp,
        read_at=fsio.now_iso(),
    )


# ---------------------------------------------------------------------------------------------
# FakeTransport (in-memory Resolve + ResolveDB interface for offline tests)
# ---------------------------------------------------------------------------------------------
class FakeTransport:
    """In-memory Resolve simulator + read-only DB façade for offline testing."""

    def __init__(self) -> None:
        self._bins: dict[str, list[dict[str, Any]]] = {"/": []}
        self._timelines: dict[str, TimelineSnapshot] = {}
        self._receipts: list[str] = []
        self.executed_runs: list[tuple[str, list[BridgeOp]]] = []
        self.db = self

    # -- ResolveDB-compatible read methods ---------------------------------------------------
    def timelines(self) -> list[str]:
        return sorted(self._timelines.keys())

    def timeline(self, name: str) -> TimelineSnapshot | None:
        snap = self._timelines.get(name)
        return snap.model_copy(deep=True) if snap is not None else None

    def fingerprint(self, name: str, *, exclude_marker_names: Iterable[str] = ()) -> str | None:
        snap = self._timelines.get(name)
        if snap is None:
            return None
        return snap.fingerprint_excluding(exclude_marker_names)

    def bins(self) -> list[str]:
        return sorted(self._bins.keys())

    def bin_items(self, bin_path: str) -> list[dict[str, Any]]:
        norm = "/" + "/".join(p for p in str(bin_path).split("/") if p)
        return [dict(x) for x in self._bins.get(norm, [])]

    def receipts(self, prefix: str = RECEIPTS_BIN) -> list[str]:
        return [r for r in self._receipts if prefix in r]

    # -- Transport execution -----------------------------------------------------------------
    def execute(self, ops: Sequence[BridgeOp], *, run_id: str) -> dict[str, Any]:
        op_list = list(ops)
        lua = render_lua(op_list, run_id=run_id)
        assert_lua_additive(lua)
        self.executed_runs.append((run_id, op_list))

        for op in op_list:
            k = op.kind
            p = op.params
            if k == "ensure_bin":
                bpath = "/" + "/".join(x for x in str(p.get("path", "")).split("/") if x)
                self._bins.setdefault(bpath, [])
            elif k == "import_media":
                bpath = "/" + "/".join(x for x in str(p.get("bin_path", "")).split("/") if x)
                bucket = self._bins.setdefault(bpath, [])
                for item in p.get("items") or []:
                    entry = dict(item)
                    if not any(existing.get("path") == entry.get("path") for existing in bucket):
                        bucket.append(entry)
            elif k == "create_timeline":
                name = str(p["name"])
                if name in self._timelines:
                    raise ClobberError(f"FakeTransport: refusing to overwrite existing timeline {name!r}")
                cut = editor.SceneCut.model_validate(p["scene_cut"])
                self._timelines[name] = _snapshot_from_cut(name, cut)
            elif k == "add_timeline_marker":
                tl_name = str(p["timeline_name"])
                snap = self._timelines.get(tl_name)
                if snap is None:
                    raise KeyError(f"FakeTransport: unknown timeline {tl_name!r}")
                marker = _marker_row(
                    frame=float(p.get("frame", 0)),
                    color=str(p.get("color", "Orange")),
                    name=str(p.get("name", "")),
                    note=str(p.get("note", "")),
                    duration=float(p.get("duration", 1)),
                    custom_data=str(p.get("custom_data", "")),
                )
                snap.markers.append(marker)
                snap.fingerprint = compute_fingerprint(snap.items, snap.markers)
            elif k == "create_master_timeline":
                name = str(p["name"])
                if name in self._timelines:
                    raise ClobberError(f"FakeTransport: refusing to overwrite existing master timeline {name!r}")
                mc = editor.MasterCut.model_validate(p["master_cut"])
                base = float(mc.start_frame)
                items: list[ItemRow] = []
                for entry in mc.entries:
                    items.append(
                        ItemRow(
                            track_type="video",
                            track_index=1,
                            name=entry.timeline_name,
                            start=base + float(entry.record_in),
                            duration=float(entry.duration),
                            source_in=0.0,
                            enabled=True,
                            flags=1,
                            media_path=entry.scene_cut or entry.timeline_name,
                            media_ref=entry.timeline_name,
                            item_id=f"master:{entry.scene_id}",
                        )
                    )
                fp = compute_fingerprint(items, [])
                self._timelines[name] = TimelineSnapshot(
                    name=name,
                    timeline_id=f"tl:{name}",
                    sequence_ids=[f"seq:{name}"],
                    start_frame=base,
                    fps=float(mc.fps),
                    items=items,
                    markers=[],
                    fingerprint=fp,
                    read_at=fsio.now_iso(),
                )

        receipt_path = f"/{RECEIPTS_BIN}/{run_id}/ok"
        self._bins.setdefault(receipt_path, [])
        self._receipts.append(receipt_path)
        return {"run_id": run_id, "ok": True, "receipt": receipt_path}

    # -- Human-action simulation helpers (for tests & round-trip verification) ---------------
    def add_marker(
        self,
        timeline_name: str,
        *,
        frame: float = 0.0,
        color: str = "Red",
        name: str = "",
        note: str = "",
        custom_data: str = "",
    ) -> MarkerRow:
        snap = self._timelines.get(timeline_name)
        if snap is None:
            raise KeyError(f"unknown timeline {timeline_name!r}")
        m = _marker_row(frame=frame, color=color, name=name, note=note, custom_data=custom_data)
        snap.markers.append(m)
        snap.fingerprint = compute_fingerprint(snap.items, snap.markers)
        snap.read_at = fsio.now_iso()
        return m

    def swap_take(
        self,
        timeline_name: str,
        clip_id: str,
        new_take_id: str,
        *,
        new_media_path: str | None = None,
    ) -> None:
        snap = self._timelines.get(timeline_name)
        if snap is None:
            raise KeyError(f"unknown timeline {timeline_name!r}")
        matched = False
        for it in snap.items:
            if it.track_type == "video" and it.track_index == 1:
                if clip_id in (it.item_id or "") or clip_id in (it.name or "") or clip_id in (it.media_ref or ""):
                    it.name = f"{new_take_id}.mp4"
                    it.media_ref = new_take_id
                    if new_media_path:
                        it.media_path = new_media_path
                    elif it.media_path:
                        it.media_path = re.sub(r"SC\d+_SH\d+_C\d+_T\d+", new_take_id, it.media_path)
                    matched = True
        if not matched:
            raise KeyError(f"no V1 item for {clip_id} on {timeline_name}")
        snap.fingerprint = compute_fingerprint(snap.items, snap.markers)
        snap.read_at = fsio.now_iso()

    def trim_clip(self, timeline_name: str, clip_id: str, delta_frames: int = -12) -> None:
        snap = self._timelines.get(timeline_name)
        if snap is None:
            raise KeyError(f"unknown timeline {timeline_name!r}")
        for it in snap.items:
            if it.track_type == "video" and it.track_index == 1:
                if clip_id in (it.item_id or "") or clip_id in (it.name or "") or clip_id in (it.media_ref or ""):
                    it.duration = max(1.0, float(it.duration) + float(delta_frames))
        snap.fingerprint = compute_fingerprint(snap.items, snap.markers)
        snap.read_at = fsio.now_iso()

    def simulate_human_edit(
        self,
        timeline_name: str,
        *,
        swap_clip_id: str | None = None,
        to_take_id: str | None = None,
        trim_clip_id: str | None = None,
        delta_frames: int = -12,
        add_marker: dict[str, Any] | None = None,
    ) -> TimelineSnapshot:
        """Simulate a human editing a handed-off timeline in DaVinci Resolve."""
        if swap_clip_id and to_take_id:
            self.swap_take(timeline_name, swap_clip_id, to_take_id)
        if trim_clip_id:
            self.trim_clip(timeline_name, trim_clip_id, delta_frames=delta_frames)
        if add_marker:
            self.add_marker(
                timeline_name,
                frame=float(add_marker.get("frame", 0)),
                color=str(add_marker.get("color", "Red")),
                name=str(add_marker.get("name", "")),
                note=str(add_marker.get("note", "")),
                custom_data=str(add_marker.get("custom_data", "")),
            )
        snap = self.timeline(timeline_name)
        assert snap is not None
        return snap


# ---------------------------------------------------------------------------------------------
# Additive Lua generator + LuaMenuTransport (real DaVinci Resolve 21 Free Edition)
# ---------------------------------------------------------------------------------------------
def _lua_str(val: Any) -> str:
    s = str(val or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "")
    return f'"{s}"'


def render_lua(ops: Sequence[BridgeOp], *, run_id: str) -> str:
    """Compile ``BridgeOp`` list into an additive-only Resolve Lua script."""
    lines = [
        "-- AGY_Bridge.lua — additive-only DaVinci Resolve 21 bridge (generated by agy_studio.resolve_bridge)",
        "local resolve = Resolve()",
        "if not resolve then return end",
        "local pm = resolve:GetProjectManager()",
        "local proj = pm and pm:GetCurrentProject()",
        "if not proj then return end",
        "local mp = proj:GetMediaPool()",
        "local root = mp:GetRootFolder()",
        "",
        "local function ensure_bin(path)",
        "  local cur = root",
        '  for part in string.gmatch(path, "[^/]+") do',
        "    local found = nil",
        "    for _, sub in ipairs(cur:GetSubFolderList() or {}) do",
        "      if sub:GetName() == part then found = sub; break end",
        "    end",
        "    if not found then found = mp:AddSubFolder(cur, part) end",
        "    cur = found or cur",
        "  end",
        "  return cur",
        "end",
        "",
    ]
    for op in ops:
        k, p = op.kind, op.params
        if k == "ensure_bin":
            lines.append(f"ensure_bin({_lua_str(p.get('path', ''))})")
        elif k == "import_media":
            lines.append(f"do local b = ensure_bin({_lua_str(p.get('bin_path', ''))}); mp:SetCurrentFolder(b)")
            for item in p.get("items") or []:
                abs_path = item.get("abs_path") or item.get("path") or ""
                lines.append(f"  local imported = mp:ImportMedia({{{_lua_str(abs_path)}}}) or {{}}")
                meta = item.get("metadata") or {}
                kws = item.get("keywords") or []
                if kws:
                    meta = {**meta, "Keywords": ", ".join(str(x) for x in kws)}
                for mk, mv in meta.items():
                    lines.append(f"  for _, ci in ipairs(imported) do ci:SetMetadata({_lua_str(mk)}, {_lua_str(mv)}) end")
            lines.append("end")
        elif k == "create_timeline":
            tl_name = p["name"]
            bin_path = p.get("bin_path", "/")
            otio_abs = p.get("otio_abs_path")
            lines.append(f"do local b = ensure_bin({_lua_str(bin_path)}); mp:SetCurrentFolder(b)")
            if otio_abs:
                lines.append(f"  local tl = mp:ImportTimelineFromFile({_lua_str(otio_abs)}, {{timelineName={_lua_str(tl_name)}}})")
                lines.append(f"  if not tl then tl = mp:CreateEmptyTimeline({_lua_str(tl_name)}) end")
            else:
                lines.append(f"  local tl = mp:CreateEmptyTimeline({_lua_str(tl_name)})")
            for m in p.get("markers") or []:
                lines.append(
                    f"  if tl then tl:AddMarker({int(m.get('frame', 0))}, {_lua_str(m.get('color', 'Yellow'))}, "
                    f"{_lua_str(m.get('name', ''))}, {_lua_str(m.get('note', ''))}, "
                    f"{int(m.get('duration', 1))}, {_lua_str(m.get('custom_data', ''))}) end"
                )
            lines.append("end")
        elif k == "add_timeline_marker":
            lines.append(
                f"do local count = proj:GetTimelineCount() or 0; "
                f"for idx = 1, count do local t = proj:GetTimelineByIndex(idx); "
                f"if t and t:GetName() == {_lua_str(p.get('timeline_name'))} then "
                f"t:AddMarker({int(p.get('frame', 0))}, {_lua_str(p.get('color', 'Orange'))}, "
                f"{_lua_str(p.get('name', ''))}, {_lua_str(p.get('note', ''))}, "
                f"{int(p.get('duration', 1))}, {_lua_str(p.get('custom_data', ''))}); break end end end"
            )
        elif k == "create_master_timeline":
            tl_name = p["name"]
            otio_abs = p.get("otio_abs_path")
            lines.append('do local b = ensure_bin("/MASTER"); mp:SetCurrentFolder(b)')
            if otio_abs:
                lines.append(f"  local tl = mp:ImportTimelineFromFile({_lua_str(otio_abs)}, {{timelineName={_lua_str(tl_name)}}})")
                lines.append(f"  if not tl then tl = mp:CreateEmptyTimeline({_lua_str(tl_name)}) end")
            else:
                lines.append(f"  local tl = mp:CreateEmptyTimeline({_lua_str(tl_name)})")
            lines.append("end")

    lines.append(f"ensure_bin({_lua_str(f'/{RECEIPTS_BIN}/{run_id}/ok')})")
    source = "\n".join(lines) + "\n"
    assert_lua_additive(source)
    return source


class LuaMenuTransport:
    """Real DaVinci Resolve Free 21 transport via ``Fusion/Scripts/Utility/AGY_Bridge.lua`` + UI menu click."""

    def __init__(self, *, scripts_dir: Path | None = None) -> None:
        self.scripts_dir = Path(scripts_dir) if scripts_dir else UTILITY_SCRIPTS_DIR

    def execute(self, ops: Sequence[BridgeOp], *, run_id: str) -> dict[str, Any]:
        lua_source = render_lua(ops, run_id=run_id)
        assert_lua_additive(lua_source)
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        script_path = self.scripts_dir / BRIDGE_SCRIPT_NAME
        fsio.atomic_write_text(script_path, lua_source)

        import resolve_sync_trigger  # noqa: PLC0415 - legacy module, real path only

        triggered = resolve_sync_trigger.trigger_menu_script("AGY_Bridge")
        return {
            "run_id": run_id,
            "ok": bool(triggered),
            "script_path": str(script_path),
            "receipt": f"/{RECEIPTS_BIN}/{run_id}/ok",
        }


# ---------------------------------------------------------------------------------------------
# ResolveBridge
# ---------------------------------------------------------------------------------------------
class ResolveBridge:
    """Non-destructive bridge between AGY's project folder and DaVinci Resolve (§10)."""

    def __init__(self, project: Project, transport: Any, db: Any = None) -> None:
        self.project = project
        self.transport = transport
        self.db = db if db is not None else getattr(transport, "db", None)

    def _scene_folder(self, scene_id: str) -> str:
        try:
            slug = self.project.load_scene(scene_id).slug
        except Exception:  # noqa: BLE001
            slug = ""
        clean = re.sub(r"[^A-Za-z0-9_]+", "_", slug).strip("_").upper()
        return f"{scene_id}_{clean}" if clean else scene_id

    def _existing_timelines(self) -> set[str]:
        names: set[str] = set()
        if self.db is not None and hasattr(self.db, "timelines"):
            try:
                names.update(self.db.timelines())
            except Exception:  # noqa: BLE001
                pass
        for sc in self.project.list_scenes():
            h = self.project.load_handoff(sc.id)
            if h and h.timeline_name:
                names.add(h.timeline_name)
            snap_data = editor.read_handoff_snapshot(self.project, sc.id)
            if snap_data and isinstance(snap_data.get("timelines"), dict):
                names.update(snap_data["timelines"].keys())
        return names

    def next_timeline_name(self, base: str, *, suffix: str = "") -> str:
        existing = self._existing_timelines()
        v = 1
        while True:
            cand = f"{base}_v{v}{suffix}"
            if cand not in existing:
                return cand
            v += 1

    def preflight(self, scene_id: str, timeline_name: str) -> None:
        """§10.4.5 Pre-flight check before any Resolve timeline creation or modification."""
        fsio.validate_id(scene_id, "scene id")
        handoff = self.project.load_handoff(scene_id)
        if handoff is not None and handoff.timeline_name == timeline_name:
            if handoff.ownership == Ownership.HUMAN or handoff.legacy:
                raise ClobberError(
                    f"Timeline {timeline_name!r} is owned by the human editor — AGY never modifies or overwrites it."
                )
            if self.db is not None and handoff.fingerprint:
                live_fp = self.db.fingerprint(timeline_name, exclude_marker_names=handoff.agy_marker_names)
                if live_fp is not None and live_fp != handoff.fingerprint:
                    raise ClobberError(
                        f"Timeline {timeline_name!r} fingerprint changed since handoff ({live_fp} != {handoff.fingerprint}) "
                        "— refusing to touch a modified timeline."
                    )
        if self.db is not None and hasattr(self.db, "timelines"):
            try:
                if timeline_name in self.db.timelines():
                    raise ClobberError(
                        f"Timeline {timeline_name!r} already exists in DaVinci Resolve — refusing to overwrite."
                    )
            except ClobberError:
                raise
            except Exception:  # noqa: BLE001
                pass

    def _bible_bin_ops(self) -> list[BridgeOp]:
        ops = [
            BridgeOp(kind="ensure_bin", params={"path": "/00_BIBLE/CHARACTERS"}),
            BridgeOp(kind="ensure_bin", params={"path": "/00_BIBLE/ENVIRONMENTS"}),
            BridgeOp(kind="ensure_bin", params={"path": "/00_BIBLE/STYLE"}),
        ]
        char_items: list[dict[str, Any]] = []
        for ent in self.project.list_entities():
            for st in ent.states:
                for uri in st.reference_pack.image_uris():
                    p = self.project.abs(uri)
                    if p.is_file():
                        char_items.append(
                            {
                                "path": uri,
                                "abs_path": str(p),
                                "metadata": {"Scene": "BIBLE", "Comments": f"{ent.id} ({st.id})"},
                                "keywords": [ent.id, st.id, "reference_pack"],
                            }
                        )
        if char_items:
            ops.append(BridgeOp(kind="import_media", params={"bin_path": "/00_BIBLE/CHARACTERS", "items": char_items}))

        env_items: list[dict[str, Any]] = []
        for env in self.project.list_environments():
            for uri in [env.geometry_still, *env.derived_angles.model_dump().values()]:
                if isinstance(uri, str) and uri and self.project.abs(uri).is_file():
                    env_items.append(
                        {
                            "path": uri,
                            "abs_path": str(self.project.abs(uri)),
                            "metadata": {"Scene": "BIBLE", "Comments": env.id},
                            "keywords": [env.id, "environment"],
                        }
                    )
        if env_items:
            ops.append(BridgeOp(kind="import_media", params={"bin_path": "/00_BIBLE/ENVIRONMENTS", "items": env_items}))
        return ops

    def deliver_scene(self, scene_id: str, cut: editor.SceneCut | None = None) -> ResolveHandoff:
        """Assemble/deliver ``scene_id`` into a brand-new ``SCxx_v<n>`` timeline and transfer ownership to ``human``."""
        fsio.validate_id(scene_id, "scene id")
        if cut is None:
            cut = editor.assemble_scene(self.project, scene_id)
        base = cut.timeline_base or scene_id
        tl_name = self.next_timeline_name(base)
        self.preflight(scene_id, tl_name)

        sc_folder = self._scene_folder(scene_id)
        tl_bin = f"/SCENES/{sc_folder}/TIMELINES"
        ops: list[BridgeOp] = [
            *self._bible_bin_ops(),
            BridgeOp(kind="ensure_bin", params={"path": f"/SCENES/{sc_folder}/BOARDS"}),
            BridgeOp(kind="ensure_bin", params={"path": f"/SCENES/{sc_folder}/AUDIO"}),
            BridgeOp(kind="ensure_bin", params={"path": tl_bin}),
            BridgeOp(kind="ensure_bin", params={"path": f"/REGEN/{scene_id}"}),
        ]

        shots = self.project.list_shots(scene_id)
        board_items = [
            {
                "path": s.board_frame,
                "abs_path": str(self.project.abs(s.board_frame)),
                "metadata": {"Scene": scene_id, "Shot": s.id, "Comments": s.intention},
                "keywords": [scene_id, s.id, "board"],
            }
            for s in shots
            if s.board_frame and self.project.abs(s.board_frame).is_file()
        ]
        if board_items:
            ops.append(
                BridgeOp(kind="import_media", params={"bin_path": f"/SCENES/{sc_folder}/BOARDS", "items": board_items})
            )

        pairs = editor.active_clips(self.project, scene_id)
        fanout = editor.anchor_fanout(c for _, c in pairs)
        entity_ids = [e.id for e in self.project.list_entities()]
        by_shot: dict[str, list[dict[str, Any]]] = {}
        for shot, clip in pairs:
            for take in self.project.list_takes(clip.id):
                staged = editor.stage_take_media(self.project, take)
                if staged is None:
                    continue
                meta, kws = editor.take_metadata(
                    self.project, take, clip=clip, shot=shot, fanout=fanout, entity_ids=entity_ids
                )
                by_shot.setdefault(shot.id, []).append(
                    {
                        "path": self.project.rel(staged),
                        "abs_path": str(staged),
                        "take_id": take.id,
                        "clip_id": clip.id,
                        "metadata": meta,
                        "keywords": kws,
                    }
                )
        for shot_id, items in sorted(by_shot.items()):
            bpath = f"/SCENES/{sc_folder}/TAKES/{shot_id}"
            ops.append(BridgeOp(kind="ensure_bin", params={"path": bpath}))
            ops.append(BridgeOp(kind="import_media", params={"bin_path": bpath, "items": items}))

        audio_items = [
            {
                "path": c.media,
                "abs_path": str(self.project.abs(c.media)),
                "metadata": dict(c.metadata),
                "keywords": list(c.keywords),
            }
            for c in cut.clips
            if c.role in ("sfx", "music") and self.project.abs(c.media).is_file()
        ]
        if audio_items:
            ops.append(
                BridgeOp(kind="import_media", params={"bin_path": f"/SCENES/{sc_folder}/AUDIO", "items": audio_items})
            )

        otio_abs = str(self.project.abs(cut.otio_path)) if cut.otio_path else None
        ops.append(
            BridgeOp(
                kind="create_timeline",
                params={
                    "name": tl_name,
                    "bin_path": tl_bin,
                    "otio_abs_path": otio_abs,
                    "scene_cut": cut.model_dump(mode="json"),
                    "markers": [
                        {
                            "frame": m.frame,
                            "color": m.color.value.capitalize(),
                            "name": m.name,
                            "note": m.note,
                            "duration": m.duration,
                            "custom_data": m.custom_data,
                        }
                        for m in cut.markers
                    ],
                },
            )
        )

        run_id = f"deliver_{scene_id}_{tl_name}_{fsio.stable_hash([scene_id, tl_name, fsio.now_iso()], 8)}"
        self.transport.execute(ops, run_id=run_id)

        snap = None
        if self.db is not None and hasattr(self.db, "timeline"):
            try:
                snap = self.db.timeline(tl_name)
            except Exception:  # noqa: BLE001
                snap = None
        if snap is None:
            snap = _snapshot_from_cut(tl_name, cut)

        fp = snap.fingerprint_excluding(())
        editor.record_timeline_snapshot(self.project, scene_id, tl_name, snap, agent=AGENT)
        opt_in = bool(self.project.config.settings.open_decisions.orange_marker_default)
        handoff = ResolveHandoff(
            scene_id=scene_id,
            timeline_name=tl_name,
            version=cut.version,
            otio_path=cut.otio_path,
            bin_path=tl_bin,
            fingerprint=fp,
            ownership=Ownership.HUMAN,
            orange_marker_opt_in=opt_in,
            agy_marker_names=[],
            ts=fsio.now_iso(),
            generated_by=fsio.generated_by(AGENT, notes=f"delivered {tl_name}"),
        )
        self.project.save_handoff(handoff)
        fsio.append_jsonl(
            self.project.path("logs", "resolve_bridge.jsonl"),
            {"ts": handoff.ts, "op": "deliver_scene", "scene_id": scene_id, "timeline": tl_name, "fingerprint": fp},
        )
        return handoff

    @staticmethod
    def handoff_message(handoff: ResolveHandoff) -> str:
        """§10.4.3 Handoff message in chat."""
        return f"{handoff.timeline_name} is on the timeline; it's yours — I'll only read it from here."

    def deliver_regen(
        self,
        scene_id: str,
        takes: Sequence[Take],
        *,
        add_orange_marker: bool | None = None,
    ) -> list[str]:
        """Deliver post-handoff regenerated takes to ``/REGEN/<SC_ID>/`` without modifying the human's cut (§10.4.6).

        When ``add_orange_marker`` is enabled, an Orange ``AGY_REGEN_READY`` marker is added and its name is recorded
        in ``handoff.agy_marker_names`` so the human timeline's fingerprint is completely unchanged (acceptance 8).
        """
        fsio.validate_id(scene_id, "scene id")
        handoff = self.project.load_handoff(scene_id)
        regen_bin = f"/REGEN/{scene_id}"
        sc_folder = self._scene_folder(scene_id)
        ops: list[BridgeOp] = [BridgeOp(kind="ensure_bin", params={"path": regen_bin})]

        pairs = editor.active_clips(self.project, scene_id)
        fanout = editor.anchor_fanout(c for _, c in pairs)
        entity_ids = [e.id for e in self.project.list_entities()]

        snap = editor.snapshot_entry(self.project, scene_id, handoff.timeline_name) if handoff else None
        base_frame = snap.base_frame() if snap else 86400.0

        placed: list[str] = []
        new_marker_names: list[str] = []
        want_marker = (
            add_orange_marker
            if add_orange_marker is not None
            else bool(
                handoff
                and handoff.orange_marker_opt_in
                and self.project.config.settings.open_decisions.orange_marker_default
            )
        )

        items_for_regen: list[dict[str, Any]] = []
        for take in takes:
            staged = editor.stage_take_media(self.project, take)
            if staged is None:
                continue
            clip = self.project.load_clip(take.clip_id)
            shot_id = parse_artifact_id(clip.id)["shot"]
            shot = self.project.load_shot(shot_id) if shot_id else None
            meta, kws = editor.take_metadata(
                self.project, take, clip=clip, shot=shot, fanout=fanout, entity_ids=entity_ids
            )
            entry = {
                "path": self.project.rel(staged),
                "abs_path": str(staged),
                "take_id": take.id,
                "clip_id": clip.id,
                "metadata": meta,
                "keywords": [*kws, "regen"],
            }
            items_for_regen.append(entry)
            if shot_id:
                shot_bin = f"/SCENES/{sc_folder}/TAKES/{shot_id}"
                ops.append(BridgeOp(kind="ensure_bin", params={"path": shot_bin}))
                ops.append(BridgeOp(kind="import_media", params={"bin_path": shot_bin, "items": [entry]}))
            placed.append(f"{regen_bin}/{take.id}")

            if want_marker and handoff and handoff.timeline_name and not handoff.legacy:
                frame = 0
                if snap is not None:
                    for it in snap.track_items("video", 1):
                        if clip.id in (it.item_id or "") or clip.id in (it.name or "") or clip.id in (it.media_ref or ""):
                            frame = max(0, int(round(float(it.start) - base_frame)))
                            break
                m_name = f"{AGY_MARKER_PREFIX} {clip.id} {take.id}"
                new_marker_names.append(m_name)
                ops.append(
                    BridgeOp(
                        kind="add_timeline_marker",
                        params={
                            "timeline_name": handoff.timeline_name,
                            "frame": frame,
                            "color": MarkerColor.ORANGE.value.capitalize(),
                            "name": m_name,
                            "note": f"New take {take.id} ready in {regen_bin}",
                            "duration": 1,
                            "custom_data": f"agy_regen:{take.id}",
                        },
                    )
                )

        if items_for_regen:
            ops.insert(1, BridgeOp(kind="import_media", params={"bin_path": regen_bin, "items": items_for_regen}))

        if handoff and new_marker_names:
            merged = list(dict.fromkeys([*handoff.agy_marker_names, *new_marker_names]))
            handoff = handoff.model_copy(update={"agy_marker_names": merged})
            self.project.save_handoff(handoff)

        run_id = f"regen_{scene_id}_{fsio.stable_hash([scene_id, [t.id for t in takes], fsio.now_iso()], 8)}"
        self.transport.execute(ops, run_id=run_id)
        fsio.append_jsonl(
            self.project.path("logs", "resolve_bridge.jsonl"),
            {"ts": fsio.now_iso(), "op": "deliver_regen", "scene_id": scene_id, "placed": placed},
        )
        return placed

    def propose_regen_timeline(self, scene_id: str, cut: editor.SceneCut | None = None) -> str:
        """Create a separate ``SCxx_v<n+1>_regen`` timeline when the human explicitly asks for one (§10.4.6)."""
        fsio.validate_id(scene_id, "scene id")
        if cut is None:
            cut = editor.assemble_scene(self.project, scene_id)
        tl_name = self.next_timeline_name(cut.timeline_base or scene_id, suffix="_regen")
        self.preflight(scene_id, tl_name)
        sc_folder = self._scene_folder(scene_id)
        tl_bin = f"/SCENES/{sc_folder}/TIMELINES"
        otio_abs = str(self.project.abs(cut.otio_path)) if cut.otio_path else None
        ops = [
            BridgeOp(kind="ensure_bin", params={"path": tl_bin}),
            BridgeOp(
                kind="create_timeline",
                params={
                    "name": tl_name,
                    "bin_path": tl_bin,
                    "otio_abs_path": otio_abs,
                    "scene_cut": cut.model_dump(mode="json"),
                    "markers": [],
                },
            ),
        ]
        run_id = f"propose_regen_{scene_id}_{tl_name}"
        self.transport.execute(ops, run_id=run_id)
        snap = self.db.timeline(tl_name) if (self.db and hasattr(self.db, "timeline")) else _snapshot_from_cut(tl_name, cut)
        editor.record_timeline_snapshot(
            self.project, scene_id, tl_name, snap, agent=AGENT, extra={"kind": "proposed_regen"}
        )
        return tl_name

    def deliver_review(self, scene_id: str, gate: str, media_paths: Iterable[str | Path]) -> str:
        """Deliver review contact sheets / comparison videos into ``/REVIEW/<SC_ID>/<GATE>/``."""
        fsio.validate_id(scene_id, "scene id")
        bin_path = f"/REVIEW/{scene_id}/{gate.upper()}"
        items: list[dict[str, Any]] = []
        for raw in media_paths:
            p = Path(raw) if Path(raw).is_absolute() else self.project.abs(str(raw))
            if p.is_file():
                rel = self.project.rel(p) if str(p.resolve()).startswith(str(self.project.root.resolve())) else p.name
                items.append(
                    {
                        "path": rel,
                        "abs_path": str(p.resolve()),
                        "metadata": {"Scene": scene_id, "Comments": f"{gate} review asset"},
                        "keywords": [scene_id, gate.upper(), "review"],
                    }
                )
        ops = [BridgeOp(kind="ensure_bin", params={"path": bin_path})]
        if items:
            ops.append(BridgeOp(kind="import_media", params={"bin_path": bin_path, "items": items}))
        run_id = f"review_{scene_id}_{gate}_{fsio.stable_hash([scene_id, gate, len(items)], 6)}"
        self.transport.execute(ops, run_id=run_id)
        return bin_path

    def create_master(self, master_cut: Path | str | editor.MasterCut) -> str:
        """Create a brand-new ``MASTER_v<n>`` timeline in ``/MASTER/`` nesting human-approved scene timelines (§10.4.8)."""
        mc = (
            master_cut
            if isinstance(master_cut, editor.MasterCut)
            else editor.load_master_cut(self.project, master_cut)
        )
        existing = self._existing_timelines()
        name = mc.name
        v = mc.version
        while name in existing:
            v += 1
            name = f"MASTER_v{v}"
        otio_abs = str(self.project.abs(mc.otio_path)) if mc.otio_path else None
        ops = [
            BridgeOp(kind="ensure_bin", params={"path": "/MASTER"}),
            BridgeOp(
                kind="create_master_timeline",
                params={"name": name, "otio_abs_path": otio_abs, "master_cut": mc.model_dump(mode="json")},
            ),
        ]
        run_id = f"master_{name}_{fsio.stable_hash([name, fsio.now_iso()], 8)}"
        self.transport.execute(ops, run_id=run_id)
        fsio.append_jsonl(
            self.project.path("logs", "resolve_bridge.jsonl"),
            {"ts": fsio.now_iso(), "op": "create_master", "timeline": name, "scenes": [e.scene_id for e in mc.entries]},
        )
        return name
