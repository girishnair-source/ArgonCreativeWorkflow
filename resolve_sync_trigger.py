#!/usr/bin/env python3
"""Closed-loop DaVinci Resolve (Free Edition) sync trigger for "LIKE. FOLLOW. KILL OR DIE.".

DaVinci Resolve Free exposes no external scripting socket (``scriptapp("Resolve")`` returns None from
both Python and Resolve's own ``fuscript``) - that path is Studio-only. The only inbound route is a script
executed *inside* Resolve from Workspace -> Scripts. This tool:

  1. installs the Lua in-app builder (no Python runtime dependency) into Resolve's Scripts menu folders
     (Resolve enumerates script *names* at start-up but re-reads the file on every click),
  2. clicks Workspace -> Scripts -> Sync_LikeFollowKillOrDie_S01 through macOS UI automation
     (System Events; needs a one-time Accessibility grant for the app hosting this terminal),
  3. verifies the result against Resolve's own project database (read-only copy, SQLite): the V1/A1 spine
     of each timeline plus the run receipt. Resolve Free's in-app script sandbox (verified on 21.1.1) has no
     file writes at all - ``io``, ``os.execute``, ``bmd.writefile`` are nil - so the Lua script stores the
     receipt *inside the project* as a ``_sync_receipt`` bin whose sub-bins encode status/timestamp/version/
     per-timeline counts, and this tool reads them back from the database.

Usage:
    .venv/bin/python resolve_sync_trigger.py            # install + trigger + verify
    .venv/bin/python resolve_sync_trigger.py --install-only
    .venv/bin/python resolve_sync_trigger.py --verify-only
"""
from __future__ import annotations

import argparse
import plistlib
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
RESOLVE_APP = Path("/Applications/DaVinci Resolve/DaVinci Resolve.app")
# Expected V1 spine per timeline: item count and total frames — derived from the Casting Bible so a
# re-cut episode (e.g. EP02 3x120f -> 9x192f) is verified against the plan, not a stale constant.
def _expected_from_bible() -> dict[str, dict[str, int]]:
    try:
        sys.path.insert(0, str(WORKSPACE_ROOT))
        from assemble_microdrama import episode_table  # noqa: WPS433 (local import keeps CLI light)

        return {r["name"]: {"items": int(r["shots"]), "frames": int(r["frames"])} for r in episode_table()}
    except Exception:  # pragma: no cover - fallback if the workspace modules cannot be imported
        return {"EP01_Daring": {"items": 9, "frames": 1728}, "EP02_TheParcel": {"items": 9, "frames": 1728}}


EXPECTED = _expected_from_bible()

# AppleScript: open Workspace -> Scripts, find the script (top level or one submenu deep) and click it.
_APPLESCRIPT = r'''
on run argv
    set targetName to item 1 of argv
    set bundleId to item 2 of argv
    set AppleScript's text item delimiters to " | "
    tell application "System Events"
        set procs to (every application process whose bundle identifier is bundleId)
        if (count of procs) is 0 then return "RESOLVE_NOT_RUNNING"
        set p to item 1 of procs
        set frontmost of p to true
        delay 1.0
        tell p
            set wsItem to menu bar item "Workspace" of menu bar 1
            click wsItem
            delay 0.7
            set scriptsItem to menu item "Scripts" of menu 1 of wsItem
            click scriptsItem
            delay 0.9
            set scriptsMenu to menu 1 of scriptsItem
            set seen to {}
            repeat with mi in (menu items of scriptsMenu)
                set miName to name of mi
                if miName is not missing value then set end of seen to miName
                if miName is targetName then
                    click mi
                    return "CLICKED:Workspace>Scripts>" & targetName
                end if
            end repeat
            repeat with mi in (menu items of scriptsMenu)
                set miName to name of mi
                if miName is not missing value and (count of menus of mi) > 0 then
                    click mi
                    delay 0.7
                    set subMenu to menu 1 of mi
                    repeat with smi in (menu items of subMenu)
                        set smiName to name of smi
                        if smiName is not missing value then set end of seen to (miName & "/" & smiName)
                        if smiName is targetName then
                            click smi
                            return "CLICKED:Workspace>Scripts>" & miName & ">" & targetName
                        end if
                    end repeat
                end if
            end repeat
            key code 53
            delay 0.2
            key code 53
            delay 0.2
            key code 53
            return "NOT_FOUND:" & (seen as text)
        end tell
    end tell
end run
'''

# Resolve 21 project library schema: timeline -> sequence -> track -> items (track Type 0 = video, 1 = audio).
_DB_QUERY = """
SELECT tl.Name AS timeline, CAST(tr.Type AS INTEGER) AS track_type, COUNT(*) AS items,
       MIN(CAST(i.Start AS INTEGER)) AS first_start,
       MAX(CAST(i.Start AS INTEGER) + CAST(i.Duration AS INTEGER)) AS last_end,
       SUM(CAST(i.Duration AS INTEGER)) AS sum_dur
FROM Sm2TiItem i
JOIN Sm2TiTrack tr ON tr.Sm2TiTrack_id = i.Sm2TiTrack_id
JOIN Sm2Sequence s ON s.Sm2Sequence_id = tr.Sequence
JOIN Sm2Timeline tl ON tl.Sm2Timeline_id = s.Sm2Timeline_id
GROUP BY tl.Name, track_type ORDER BY tl.Name, track_type
"""
_DB_ITEMS_QUERY = """
SELECT tl.Name AS timeline, i.Name AS item, CAST(i.Start AS INTEGER) AS Start, CAST(i.Duration AS INTEGER) AS Duration
FROM Sm2TiItem i
JOIN Sm2TiTrack tr ON tr.Sm2TiTrack_id = i.Sm2TiTrack_id
JOIN Sm2Sequence s ON s.Sm2Sequence_id = tr.Sequence
JOIN Sm2Timeline tl ON tl.Sm2Timeline_id = s.Sm2Timeline_id
WHERE CAST(tr.Type AS INTEGER) = 0 ORDER BY tl.Name, Start
"""
# Receipt written by the Lua script as sub-bins of the `_sync_receipt` bin (MpFolder = parent folder id).
_DB_RECEIPT_QUERY = """
SELECT c.Name FROM Sm2MpFolder c
JOIN Sm2MpFolder p ON c.MpFolder = p.Sm2MpFolder_id
WHERE p.Name = ? ORDER BY c.Name
"""


def _paths():
    sys.path.insert(0, str(WORKSPACE_ROOT))
    import assemble_microdrama as am  # noqa: WPS433 (runtime import keeps CLI start-up light)

    return am


def install_scripts() -> Path:
    """(Re)generate the Python console loader + Lua menu script and install the Lua one into Resolve."""
    am = _paths()
    py_loader = am._write_resolve_console_loader_script(am.ASSET_DIR / "load_into_resolve_console.py")
    installed = sorted(am.RESOLVE_USER_SCRIPTS_ROOT.glob(f"*/{am.RESOLVE_MENU_SCRIPT_STEM}.lua"))
    print(f"Installed Lua menu script into {len(installed)} Resolve Scripts folder(s):")
    for p in installed:
        print(f"  - {p}")
    return py_loader.with_suffix(".lua")


def resolve_bundle_id() -> str:
    info = RESOLVE_APP / "Contents" / "Info.plist"
    if info.exists():
        with info.open("rb") as fh:
            return str(plistlib.load(fh).get("CFBundleIdentifier", "com.blackmagic-design.DaVinciResolve"))
    return "com.blackmagic-design.DaVinciResolve"


def resolve_is_running() -> bool:
    res = subprocess.run(["pgrep", "-f", "DaVinci Resolve.app/Contents/MacOS/Resolve"], capture_output=True, text=True)
    return bool(res.stdout.strip())


def trigger_menu_script(script_stem: str) -> str:
    """Click the Resolve menu entry via System Events. Returns the AppleScript status string."""
    osascript = shutil.which("osascript")
    if not osascript:
        raise RuntimeError("osascript not found - macOS UI automation unavailable")
    res = subprocess.run(
        [osascript, "-", script_stem, resolve_bundle_id()],
        input=_APPLESCRIPT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if res.returncode != 0:
        err = (res.stderr or res.stdout).strip()
        if "assistive access" in err or "-25211" in err or "-1719" in err or "not allowed" in err:
            raise PermissionError(
                "macOS blocked UI automation (Accessibility). Grant access once:\n"
                "  System Settings -> Privacy & Security -> Accessibility -> enable the app that hosts this "
                "terminal (your IDE / Terminal), and if prompted also allow it to control 'System Events'.\n"
                f"  Raw error: {err}"
            )
        raise RuntimeError(f"osascript failed: {err}")
    return res.stdout.strip()


def clear_receipts() -> None:
    am = _paths()
    for p in am.RESOLVE_SYNC_RECEIPT_CANDIDATES:
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass


_LUA_TOKEN = re.compile(
    r'\s*(?:(?P<lb>\{)|(?P<rb>\})|(?P<eq>=)|(?P<comma>,)|(?P<lsq>\[)|(?P<rsq>\])'
    r'|(?P<str>"(?:\\.|[^"\\])*")|(?P<num>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)|(?P<ident>[A-Za-z_]\w*))',
    re.S,
)
_LUA_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "'": "'", "a": "\a", "b": "\b", "f": "\f", "v": "\v"}


def _lua_unescape(body: str) -> str:
    out, i = [], 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            if nxt.isdigit():  # \ddd decimal escape
                j = i + 1
                while j < len(body) and j < i + 4 and body[j].isdigit():
                    j += 1
                out.append(chr(int(body[i + 1:j])))
                i = j
                continue
            out.append(_LUA_ESCAPES.get(nxt, nxt))
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def parse_lua_table(text: str):
    """Parse the Lua table literal produced by Fusion's bmd.writefile into Python dict/list values."""
    tokens = []
    pos = 0
    while pos < len(text):
        m = _LUA_TOKEN.match(text, pos)
        if not m or m.end() == pos:
            if text[pos:].strip() == "":
                break
            raise ValueError(f"unexpected Lua receipt syntax near: {text[pos:pos + 30]!r}")
        pos = m.end()
        tokens.append((m.lastgroup, m.group(m.lastgroup)))
    idx = 0

    def value():
        nonlocal idx
        kind, tok = tokens[idx]
        if kind == "lb":
            return table()
        idx += 1
        if kind == "str":
            return _lua_unescape(tok[1:-1])
        if kind == "num":
            return float(tok) if any(c in tok for c in ".eE") else int(tok)
        if kind == "ident":
            return {"true": True, "false": False, "nil": None}.get(tok, tok)
        raise ValueError(f"unexpected token {tok!r} in Lua receipt")

    def table():
        nonlocal idx
        idx += 1  # consume '{'
        keyed: dict = {}
        seq: list = []
        while tokens[idx][0] != "rb":
            kind, tok = tokens[idx]
            if kind == "lsq":  # ["key"] = value  /  [1] = value
                idx += 1
                key = value()
                idx += 2  # ']' '='
                keyed[key] = value()
            elif kind == "ident" and idx + 1 < len(tokens) and tokens[idx + 1][0] == "eq":
                idx += 2
                keyed[tok] = value()
            else:
                seq.append(value())
            if tokens[idx][0] == "comma":
                idx += 1
        idx += 1  # consume '}'
        if keyed and seq:
            keyed.update({i + 1: v for i, v in enumerate(seq)})
            return keyed
        return keyed if keyed else seq

    return value()


def _db_snapshot():
    """Copy Resolve's live project library (Resolve keeps it open) and return a read-only connection + temp dir."""
    am = _paths()
    db = am.RESOLVE_PROJECT_DB
    if not db.exists():
        return None, None
    tmp = tempfile.TemporaryDirectory()
    snap = Path(tmp.name) / "Project.db"
    shutil.copy2(db, snap)
    for sidecar in ("-wal", "-shm"):
        side = db.with_name(db.name + sidecar)
        if side.exists():
            shutil.copy2(side, snap.with_name(snap.name + sidecar))
    return sqlite3.connect(f"file:{snap}?mode=ro", uri=True), tmp


_RECEIPT_TL = re.compile(r"^(?P<name>.+?)__items_(?P<items>\d+)__frames_(?P<frames>\d+)__markers_(?P<markers>\d+)$")


def parse_receipt_bins(names: list[str]) -> dict | None:
    """Decode the `_sync_receipt` sub-bin names written by the Lua script (see receipt_fields there)."""
    if not names:
        return None
    rec: dict = {"status": None, "finished_at": None, "finished_epoch": None, "resolve_version": None,
                 "timelines": [], "error": "", "drt_backups": None, "log": []}
    for raw in names:
        body = raw[3:] if re.match(r"^\d\d_", raw) else raw
        if body.startswith("status_"):
            rec["status"] = body[len("status_"):]
        elif body.startswith("at_"):
            rec["finished_at"] = body[len("at_"):]
            try:
                rec["finished_epoch"] = time.mktime(time.strptime(rec["finished_at"], "%Y%m%d_%H%M%S"))
            except ValueError:
                rec["finished_epoch"] = None
        elif body.startswith("resolve_"):
            rec["resolve_version"] = body[len("resolve_"):].replace("_", ".")
        elif body.startswith("drt_backups_"):
            rec["drt_backups"] = body[len("drt_backups_"):].replace("_of_", "/")
        elif body.startswith("error_"):
            rec["error"] = body[len("error_"):]
        else:
            m = _RECEIPT_TL.match(body)
            if m:
                rec["timelines"].append({"name": m.group("name"), "item_count": int(m.group("items")),
                                         "duration_frames": int(m.group("frames")), "markers": int(m.group("markers"))})
            else:
                rec["log"].append(raw)
    return rec if rec["status"] else None


def read_db_receipt() -> dict | None:
    am = _paths()
    conn, tmp = _db_snapshot()
    if conn is None:
        return None
    try:
        names = [row[0] for row in conn.execute(_DB_RECEIPT_QUERY, (am.RESOLVE_SYNC_RECEIPT_BIN,))]
    except sqlite3.DatabaseError:
        names = []
    finally:
        conn.close()
        tmp.cleanup()
    rec = parse_receipt_bins(names)
    if rec:
        rec["_receipt_path"] = f"project DB bin '{am.RESOLVE_SYNC_RECEIPT_BIN}' ({len(names)} fields)"
    return rec


def find_receipt(since: float | None = None) -> dict | None:
    """Return the run receipt: a file receipt if the environment could write one, else the in-project bin
    receipt. With ``since`` (epoch seconds) only a receipt produced after that moment counts."""
    am = _paths()
    for p in am.RESOLVE_SYNC_RECEIPT_CANDIDATES:
        if p.exists() and (since is None or p.stat().st_mtime >= since):
            try:
                data = parse_lua_table(p.read_text(encoding="utf-8"))
            except (ValueError, IndexError):
                continue  # still being written
            if isinstance(data, dict) and data.get("status"):
                data["_receipt_path"] = str(p)
                return data
    rec = read_db_receipt()
    if rec and (since is None or (rec.get("finished_epoch") or 0) >= since):
        return rec
    return None


def read_project_db() -> dict[str, dict]:
    """Read Resolve's project library (copied first - Resolve keeps the live file open) and return per-timeline
    V1/A1 statistics plus the V1 item list. Empty dict if the DB is absent."""
    conn, tmp = _db_snapshot()
    if conn is None:
        return {}
    try:
        out: dict[str, dict] = {}
        for name, track_type, items, first, last, sum_dur in conn.execute(_DB_QUERY):
            entry = out.setdefault(name, {"v1_items": 0, "a1_items": 0, "v1_frames": 0, "v1_gap_frames": 0, "items": []})
            if track_type == 0:
                entry["v1_items"] = items
                entry["v1_frames"] = int(last - first)
                entry["v1_gap_frames"] = int((last - first) - sum_dur)
            elif track_type == 1:
                entry["a1_items"] = items
        for name, item, start, dur in conn.execute(_DB_ITEMS_QUERY):
            out.setdefault(name, {"v1_items": 0, "a1_items": 0, "v1_frames": 0, "v1_gap_frames": 0, "items": []})
            out[name]["items"].append({"name": item, "start": int(start), "duration": int(dur)})
        return out
    finally:
        conn.close()
        tmp.cleanup()


def find_latest_timeline_state(db_state: dict[str, dict], base_name: str) -> tuple[str, dict] | tuple[None, None]:
    """Find the latest versioned timeline for an episode (e.g. EP01_Daring_v02 or EP01_Daring)."""
    candidates = []
    for name, data in db_state.items():
        if name == base_name:
            candidates.append((1, name, data))
        elif name.startswith(base_name + "_v"):
            suffix = name[len(base_name) + 2 :]
            if suffix.isdigit():
                candidates.append((int(suffix), name, data))
    if not candidates:
        return None, None
    candidates.sort(key=lambda c: c[0])
    return candidates[-1][1], candidates[-1][2]


def verify(db_state: dict[str, dict], receipt: dict | None, db_mtime: float | None) -> bool:
    ok = True
    print("\n== Resolve project database (ground truth) ==")
    if not db_state:
        print("  project DB not found - cannot verify")
        ok = False
    checked_timelines = set()
    for base_name, exp in EXPECTED.items():
        tl_name, st = find_latest_timeline_state(db_state, base_name)
        if not st or tl_name is None:
            print(f"  [MISSING ] {base_name}")
            ok = False
            continue
        checked_timelines.add(tl_name)
        good = st["v1_items"] == exp["items"] and st["v1_frames"] == exp["frames"] and st["v1_gap_frames"] == 0
        ok = ok and good
        ver_info = f" (version: {tl_name})" if tl_name != base_name else ""
        print(
            f"  [{'OK      ' if good else 'MISMATCH'}] {base_name}{ver_info}: V1 {st['v1_items']} items / {st['v1_frames']} frames "
            f"(gaps {st['v1_gap_frames']}f), A1 {st['a1_items']} items  (expected {exp['items']} items / {exp['frames']} f)"
        )
        for it in st["items"]:
            print(f"             {it['name']}: {it['start'] - 86400}-{it['start'] - 86400 + it['duration']} ({it['duration']}f)")
    extra = sorted(set(db_state) - checked_timelines)
    if extra:
        print(f"  preserved / other timelines in project: {', '.join(extra)}")
    if db_mtime:
        print(f"  project DB last saved: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(db_mtime))}")

    print("\n== In-app receipt ==")
    if receipt is None:
        print("  no receipt found (the in-app script did not finish, or Resolve has not saved yet); "
              "DB verification above is authoritative")
    else:
        print(f"  {receipt['_receipt_path']}: status={receipt.get('status')} resolve={receipt.get('resolve_version')} "
              f"at {receipt.get('finished_at')}")
        if receipt.get("error"):
            print(f"  error: {receipt['error']}")
        for tl in receipt.get("timelines", []):
            print(f"  {tl.get('name')}: {tl.get('item_count')} items, {tl.get('duration_frames')} frames, {tl.get('markers')} markers")
        if receipt.get("drt_backups") is not None:
            print(f"  .drt backups: {receipt['drt_backups']}")
        for line in receipt.get("log", []):
            print(f"  log: {line}")
        ok = ok and receipt.get("status") == "success"
    return ok



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--install-only", action="store_true", help="install the menu script, do not click it")
    ap.add_argument("--verify-only", action="store_true", help="only verify the current Resolve project state")
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for Resolve to finish")
    args = ap.parse_args()
    am = _paths()

    if args.verify_only:
        db_path = am.RESOLVE_PROJECT_DB
        return 0 if verify(read_project_db(), find_receipt(), db_path.stat().st_mtime if db_path.exists() else None) else 1

    install_scripts()
    if args.install_only:
        print("\nIn DaVinci Resolve: Workspace -> Scripts -> " + am.RESOLVE_MENU_SCRIPT_STEM)
        print("(Resolve lists script names at start-up: if the entry is missing, restart Resolve once.)")
        return 0

    if not resolve_is_running():
        print("DaVinci Resolve is not running - open it, then re-run this command.")
        return 2

    clear_receipts()
    db_path = am.RESOLVE_PROJECT_DB
    db_mtime_before = db_path.stat().st_mtime if db_path.exists() else 0.0
    started = time.time() - 2.0  # small clock slack: the Lua timestamp has 1 s resolution

    status = trigger_menu_script(am.RESOLVE_MENU_SCRIPT_STEM)
    print(f"UI automation: {status}")
    if not status.startswith("CLICKED"):
        print("Could not find the menu entry. Resolve enumerates Workspace -> Scripts at start-up: "
              "restart DaVinci Resolve once after the first install, then re-run.")
        return 3

    # The receipt is the Lua script's last step (after its final SaveProject), so a receipt newer than the
    # click means the whole run has landed in the project database.
    deadline = time.time() + args.timeout
    receipt = None
    while time.time() < deadline:
        receipt = find_receipt(since=started)
        if receipt:
            break
        time.sleep(1.5)
    else:
        db_mtime = db_path.stat().st_mtime if db_path.exists() else 0.0
        if db_mtime > db_mtime_before:
            print(f"\nNo fresh receipt within {args.timeout:.0f}s, but the project was saved after the click - "
                  "verifying the database state only.")
        else:
            print(f"\nNo receipt and no project save within {args.timeout:.0f}s - the script may not have run inside Resolve "
                  "(check ~/Library/Application Support/Blackmagic Design/DaVinci Resolve/logs/ResolveDebug.txt).")

    db_mtime = db_path.stat().st_mtime if db_path.exists() else None
    return 0 if verify(read_project_db(), receipt, db_mtime) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (PermissionError, RuntimeError, TimeoutError) as exc:
        print(f"\n{exc}")
        sys.exit(1)
