"""DaVinci Resolve delivery is a deliverable: the menu-click route can fail (Accessibility, menu not enumerated,
no osascript) and the direct scripting API must take over; if both fail the editor gets exact manual steps.

Before this suite, ``LuaMenuTransport`` returned ``ok=True`` for ANY AppleScript status string — including
``RESOLVE_NOT_RUNNING`` and ``NOT_FOUND:...`` — so a scene could be marked HANDED_OFF with an empty media pool.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from agy_studio.resolve_bridge import (
    BridgeOp,
    FailoverTransport,
    LuaMenuTransport,
    ResolveDeliveryError,
    ScriptApiTransport,
)


# ---------------------------------------------------------------------------------------------
# A tiny in-memory stand-in for Resolve's scripting objects (only the calls the transport makes)
# ---------------------------------------------------------------------------------------------
class _Clip:
    def __init__(self, path: str) -> None:
        self.path = path
        self.meta: dict[str, str] = {}

    def SetMetadata(self, k: str, v: str) -> bool:  # noqa: N802 - Resolve API casing
        self.meta[k] = v
        return True


class _Folder:
    def __init__(self, name: str) -> None:
        self.name = name
        self.subs: list[_Folder] = []
        self.clips: list[_Clip] = []

    def GetName(self) -> str:  # noqa: N802
        return self.name

    def GetSubFolderList(self) -> list[_Folder]:  # noqa: N802
        return list(self.subs)


class _Timeline:
    def __init__(self, name: str) -> None:
        self.name = name
        self.markers: list[tuple[Any, ...]] = []

    def GetName(self) -> str:  # noqa: N802
        return self.name

    def AddMarker(self, *args: Any) -> bool:  # noqa: N802
        self.markers.append(args)
        return True


class _MediaPool:
    def __init__(self, proj: _Project) -> None:
        self.root = _Folder("Master")
        self.current = self.root
        self.proj = proj
        self.import_calls: list[str] = []

    def GetRootFolder(self) -> _Folder:  # noqa: N802
        return self.root

    def AddSubFolder(self, parent: _Folder, name: str) -> _Folder:  # noqa: N802
        f = _Folder(name)
        parent.subs.append(f)
        return f

    def SetCurrentFolder(self, f: _Folder) -> bool:  # noqa: N802
        self.current = f
        return True

    def ImportMedia(self, paths: list[str]) -> list[_Clip]:  # noqa: N802
        out = [_Clip(p) for p in paths]
        self.current.clips.extend(out)
        self.import_calls.extend(paths)
        return out

    def ImportTimelineFromFile(self, path: str, opts: dict[str, Any]) -> _Timeline | None:  # noqa: N802
        if not Path(path).is_file():
            return None
        tl = _Timeline(opts["timelineName"])
        self.proj.timelines.append(tl)
        return tl

    def CreateEmptyTimeline(self, name: str) -> _Timeline:  # noqa: N802
        tl = _Timeline(name)
        self.proj.timelines.append(tl)
        return tl


class _Project:
    def __init__(self) -> None:
        self.timelines: list[_Timeline] = []
        self.mp = _MediaPool(self)

    def GetMediaPool(self) -> _MediaPool:  # noqa: N802
        return self.mp

    def GetTimelineCount(self) -> int:  # noqa: N802
        return len(self.timelines)

    def GetTimelineByIndex(self, idx: int) -> _Timeline:  # noqa: N802
        return self.timelines[idx - 1]


class _PM:
    def __init__(self, proj: _Project | None) -> None:
        self.proj = proj

    def GetCurrentProject(self) -> _Project | None:  # noqa: N802
        return self.proj


class _Resolve:
    def __init__(self, proj: _Project | None = None) -> None:
        self.pm = _PM(proj)

    def GetProjectManager(self) -> _PM:  # noqa: N802
        return self.pm


def _bins(folder: _Folder, prefix: str = "") -> list[str]:
    out = []
    for sub in folder.subs:
        p = f"{prefix}/{sub.name}"
        out.append(p)
        out.extend(_bins(sub, p))
    return out


def _ops(tmp_path: Path) -> list[BridgeOp]:
    clip = tmp_path / "SC01_SH01_C01_T01.mp4"
    clip.write_bytes(b"\x00" * 16)
    otio = tmp_path / "SC01_v1.otio"
    otio.write_text("{}")
    return [
        BridgeOp(kind="ensure_bin", params={"path": "/SCENES/SC01/TAKES/SC01_SH01"}),
        BridgeOp(kind="import_media", params={"bin_path": "/SCENES/SC01/TAKES/SC01_SH01", "items": [
            {"abs_path": str(clip), "metadata": {"Scene": "SC01"}, "keywords": ["SC01", "take"]}]}),
        BridgeOp(kind="create_timeline", params={"name": "SC01_v1", "bin_path": "/SCENES/SC01/TIMELINES",
                                                 "otio_abs_path": str(otio),
                                                 "markers": [{"frame": 0, "color": "Blue", "name": "AGY_G6",
                                                              "note": "", "duration": 1, "custom_data": ""}]}),
        BridgeOp(kind="add_timeline_marker", params={"timeline_name": "SC01_v1", "frame": 24, "color": "Orange",
                                                     "name": "AGY_REGEN_READY", "note": "", "duration": 1,
                                                     "custom_data": ""}),
    ]


# ---------------------------------------------------------------------------------------------
# ScriptApiTransport
# ---------------------------------------------------------------------------------------------
def test_script_api_transport_lands_bins_media_timeline_and_markers(tmp_path: Path) -> None:
    proj = _Project()
    tr = ScriptApiTransport(_Resolve(proj))
    res = tr.execute(_ops(tmp_path), run_id="run1")
    assert res["ok"] and res["transport"] == "script_api" and res["imported"] == 1
    bins = _bins(proj.mp.root)
    assert "/SCENES/SC01/TAKES/SC01_SH01" in bins
    assert "/SCENES/SC01/TIMELINES" in bins
    assert any(b.endswith("/run1/ok") for b in bins)  # receipt bin, same contract as the Lua route
    clip = next(c for f in _walk(proj.mp.root) for c in f.clips)
    assert clip.meta["Scene"] == "SC01" and clip.meta["Keywords"] == "SC01, take"
    tl = proj.timelines[0]
    assert tl.name == "SC01_v1"
    assert len(tl.markers) == 2  # G6 marker from create_timeline + the add_timeline_marker op


def _walk(folder: _Folder) -> list[_Folder]:
    out = [folder]
    for s in folder.subs:
        out.extend(_walk(s))
    return out


def test_script_api_transport_no_open_project_is_actionable() -> None:
    tr = ScriptApiTransport(_Resolve(None))
    with pytest.raises(ResolveDeliveryError, match="No project is open"):
        tr.execute([], run_id="r")


def test_script_api_transport_missing_module_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "DaVinciResolveScript", raising=False)
    monkeypatch.setenv("RESOLVE_SCRIPT_API_MODULES", "/definitely/not/here")
    import agy_studio.resolve_bridge as rb

    monkeypatch.setattr(rb, "_RESOLVE_MODULE_DIRS", ())
    with pytest.raises(ResolveDeliveryError, match="RESOLVE_SCRIPT_API_MODULES"):
        ScriptApiTransport().execute([], run_id="r")


def test_script_api_transport_resolve_not_running_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mod = types.ModuleType("DaVinciResolveScript")
    fake_mod.scriptapp = lambda name: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "DaVinciResolveScript", fake_mod)
    with pytest.raises(ResolveDeliveryError, match="External scripting using"):
        ScriptApiTransport().execute([], run_id="r")


# ---------------------------------------------------------------------------------------------
# LuaMenuTransport: status strings are no longer "truthy == delivered"
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("status", ["RESOLVE_NOT_RUNNING", "NOT_FOUND:Workspace>Scripts>Utility", ""])
def test_lua_menu_transport_treats_non_click_status_as_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                              status: str) -> None:
    fake = types.ModuleType("resolve_sync_trigger")
    fake.trigger_menu_script = lambda stem: status  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "resolve_sync_trigger", fake)
    tr = LuaMenuTransport(scripts_dir=tmp_path / "Utility")
    with pytest.raises(ResolveDeliveryError, match="not triggered"):
        tr.execute(_ops(tmp_path), run_id="r")
    assert (tmp_path / "Utility" / "AGY_Bridge.lua").is_file()  # the manual one-click route is still installed


def test_lua_menu_transport_clicked_is_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("resolve_sync_trigger")
    fake.trigger_menu_script = lambda stem: "CLICKED:Workspace>Scripts>Utility>AGY_Bridge"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "resolve_sync_trigger", fake)
    res = LuaMenuTransport(scripts_dir=tmp_path / "Utility").execute(_ops(tmp_path), run_id="r")
    assert res["ok"] and res["transport"] == "lua_menu"


# ---------------------------------------------------------------------------------------------
# FailoverTransport
# ---------------------------------------------------------------------------------------------
class _Boom:
    name = "boom"

    def execute(self, ops: Any, *, run_id: str) -> dict[str, Any]:
        raise PermissionError("macOS blocked UI automation (Accessibility)")


def test_failover_uses_script_api_when_menu_click_fails(tmp_path: Path) -> None:
    proj = _Project()
    tr = FailoverTransport(_Boom(), ScriptApiTransport(_Resolve(proj)))
    res = tr.execute(_ops(tmp_path), run_id="r")
    assert res["transport"] == "script_api" and tr.last_used == "script_api"
    assert res["failover_errors"] and "Accessibility" in res["failover_errors"][0]
    assert proj.mp.import_calls  # the clip really landed


def test_failover_all_routes_down_gives_manual_steps(tmp_path: Path) -> None:
    tr = FailoverTransport(_Boom(), ScriptApiTransport(_Resolve(None)))
    with pytest.raises(ResolveDeliveryError) as info:
        tr.execute(_ops(tmp_path), run_id="r")
    msg = str(info.value)
    assert "boom: macOS blocked UI automation" in msg
    assert "script_api: No project is open" in msg
    assert "Workspace -> Scripts -> Utility -> AGY_Bridge" in msg
    assert "OTIO" in msg
