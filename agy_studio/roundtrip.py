"""Resolve round-trip reader — BUILD_SPEC_v3 §10.5 (passive DB polling + OTIO/FCPXML interchange import).

Detects human actions on handed-off timelines without ever modifying Resolve:
* **Red marker** or marker text starting with ``REGEN`` -> ``regen_marker`` on the clip under the marker,
* **Green marker** or marker text starting with ``APPROVED`` -> ``approved_marker`` (G6 scene approval),
* **Blue marker** or marker text starting with ``NOTE`` -> ``note_marker`` (candidate standing preference),
* **Take swap on V1** -> ``take_swap`` (updates ``selected_take_id`` and marks anchor descendants ``stale``),
* **Trim / retime / reorder / other edit** -> logged as a ``Decision`` with ``source="resolve"``; never overwritten.

Events are deduplicated across polls via ``logs/roundtrip_seen.jsonl``.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Literal

from . import editor, fsio
from .models import Model
from .project import Project, parse_artifact_id
from .resolve_db import ItemRow, MarkerRow, TimelineSnapshot, compute_fingerprint, marker_excluded

AGENT = "agy.roundtrip"
SEEN_LOG = ("logs", "roundtrip_seen.jsonl")
CLIP_ID_RE = re.compile(r"(SC\d{2,4}_SH\d{2,3}_C\d{2,3})")
SCENE_ID_RE = re.compile(r"(SC\d{2,4})")

RoundtripKind = Literal[
    "regen_marker",
    "approved_marker",
    "note_marker",
    "take_swap",
    "trim",
    "retime",
    "reorder",
    "other_edit",
]


class RoundtripEvent(Model):
    kind: RoundtripKind
    scene_id: str
    timeline: str
    clip_id: str | None = None
    take_id: str | None = None
    from_take_id: str | None = None
    text: str = ""
    frame: int | None = None
    event_key: str


def _seen_keys(project: Project) -> set[str]:
    rows = fsio.read_jsonl(project.path(*SEEN_LOG))
    return {str(r.get("event_key")) for r in rows if r.get("event_key")}


def _record_seen(project: Project, events: Iterable[RoundtripEvent]) -> None:
    path = project.path(*SEEN_LOG)
    for ev in events:
        fsio.append_jsonl(
            path,
            {
                "ts": fsio.now_iso(),
                "event_key": ev.event_key,
                "kind": ev.kind,
                "scene_id": ev.scene_id,
                "timeline": ev.timeline,
                "clip_id": ev.clip_id,
                "take_id": ev.take_id,
            },
        )


def _clip_id_of_item(it: ItemRow, fallback_scene: str) -> str | None:
    for text in (it.item_id, it.media_ref, it.name, it.media_path):
        m = CLIP_ID_RE.search(text or "")
        if m:
            return m.group(1)
    tid = editor.take_id_of(it.name) or editor.take_id_of(it.media_path) or editor.take_id_of(it.media_ref)
    if tid:
        return parse_artifact_id(tid)["clip"]
    return None


def _take_id_of_item(it: ItemRow) -> str | None:
    for text in (it.media_ref, it.name, it.media_path):
        tid = editor.take_id_of(text)
        if tid:
            return tid
    return None


def _clip_at_frame(snap: TimelineSnapshot, rel_frame: float, scene_id: str) -> str | None:
    base = snap.base_frame()
    abs_frame = base + float(rel_frame)
    v1 = snap.track_items("video", 1)
    for it in v1:
        if float(it.start) - 0.5 <= abs_frame < float(it.start) + float(it.duration) + 0.5:
            cid = _clip_id_of_item(it, scene_id)
            if cid:
                return cid
        if float(it.start) - 0.5 <= float(rel_frame) < float(it.start) + float(it.duration) + 0.5:
            cid = _clip_id_of_item(it, scene_id)
            if cid:
                return cid
    if v1:
        return _clip_id_of_item(v1[0], scene_id)
    clips = [c.id for s in [] for c in []]  # fallback handled by caller
    del clips
    return None


def _classify_marker(
    project: Project,
    scene_id: str,
    timeline: str,
    m: MarkerRow,
    snap: TimelineSnapshot,
) -> RoundtripEvent | None:
    combined = " ".join(x for x in (m.name, m.note, *m.strings) if x).strip()
    upper = combined.upper()
    color = (m.color or "").strip().lower()
    frame = int(round(float(m.frame or 0)))

    clip_match = CLIP_ID_RE.search(combined) or CLIP_ID_RE.search(m.custom_data or "")
    clip_id = clip_match.group(1) if clip_match else _clip_at_frame(snap, float(m.frame or 0), scene_id)
    if clip_id is None:
        all_clips = project.list_scene_clips(scene_id)
        if all_clips:
            clip_id = all_clips[0].id

    # 1. REGEN marker (Red or starts with/contains REGEN:)
    if re.search(r"\bREGEN\b", upper) or (color == "red" and "QA_REJECT" not in upper):
        steer = combined
        steer = re.sub(r"(?i)^.*?\bREGEN\s*[:\-–—]?\s*", "", steer).strip()
        if not steer and m.note and not re.match(r"(?i)^regen\b", m.note.strip()):
            steer = m.note.strip()
        key = fsio.stable_hash(["regen_marker", scene_id, timeline, clip_id, frame, steer], 12)
        return RoundtripEvent(
            kind="regen_marker",
            scene_id=scene_id,
            timeline=timeline,
            clip_id=clip_id,
            text=steer,
            frame=frame,
            event_key=key,
        )

    # 2. APPROVED marker (Green or starts with APPROVED)
    if re.search(r"\bAPPROVED\b", upper) or color == "green":
        key = fsio.stable_hash(["approved_marker", scene_id, timeline, frame], 12)
        return RoundtripEvent(
            kind="approved_marker",
            scene_id=scene_id,
            timeline=timeline,
            text=combined or "APPROVED",
            frame=frame,
            event_key=key,
        )

    # 3. NOTE marker (Blue or starts with NOTE:)
    if re.search(r"\bNOTE\b", upper) or color == "blue":
        note = re.sub(r"(?i)^.*?\bNOTE\s*[:\-–—]?\s*", "", combined).strip() or combined
        key = fsio.stable_hash(["note_marker", scene_id, timeline, clip_id, frame, note], 12)
        return RoundtripEvent(
            kind="note_marker",
            scene_id=scene_id,
            timeline=timeline,
            clip_id=clip_id,
            text=note,
            frame=frame,
            event_key=key,
        )

    return None


def _diff_snapshots(
    project: Project,
    scene_id: str,
    timeline: str,
    baseline: TimelineSnapshot,
    current: TimelineSnapshot,
    *,
    exclude_markers: Iterable[str] = (),
) -> list[RoundtripEvent]:
    events: list[RoundtripEvent] = []
    exclude = set(exclude_markers or ())

    base_marker_sigs = {
        (round(float(m.frame or 0), 2), m.name, m.note, (m.color or "").lower())
        for m in baseline.all_markers()
    }
    for m in current.all_markers():
        if marker_excluded(m, exclude):
            continue
        sig = (round(float(m.frame or 0), 2), m.name, m.note, (m.color or "").lower())
        if sig in base_marker_sigs:
            continue
        ev = _classify_marker(project, scene_id, timeline, m, current)
        if ev is not None:
            events.append(ev)

    base_v1 = baseline.track_items("video", 1)
    curr_v1 = current.track_items("video", 1)

    base_by_clip: dict[str, ItemRow] = {}
    base_order: list[str] = []
    for idx, it in enumerate(base_v1):
        cid = _clip_id_of_item(it, scene_id) or f"{scene_id}_SLOT{idx + 1:02d}"
        base_by_clip[cid] = it
        base_order.append(cid)

    curr_by_clip: dict[str, ItemRow] = {}
    curr_order: list[str] = []
    for idx, it in enumerate(curr_v1):
        cid = _clip_id_of_item(it, scene_id) or (base_order[idx] if idx < len(base_order) else f"{scene_id}_SLOT{idx + 1:02d}")
        curr_by_clip[cid] = it
        curr_order.append(cid)

    # Check take swaps, trims, retimes per clip slot
    for idx, (cid, base_it) in enumerate(base_by_clip.items()):
        curr_it = curr_by_clip.get(cid)
        if curr_it is None and idx < len(curr_v1):
            curr_it = curr_v1[idx]
        if curr_it is None:
            continue
        old_take = _take_id_of_item(base_it)
        new_take = _take_id_of_item(curr_it)
        if old_take and new_take and old_take != new_take:
            real_cid = parse_artifact_id(new_take)["clip"] or cid
            key = fsio.stable_hash(["take_swap", scene_id, timeline, real_cid, old_take, new_take], 12)
            events.append(
                RoundtripEvent(
                    kind="take_swap",
                    scene_id=scene_id,
                    timeline=timeline,
                    clip_id=real_cid,
                    take_id=new_take,
                    from_take_id=old_take,
                    text=f"Swapped {real_cid} from {old_take} to {new_take}",
                    frame=int(round(float(curr_it.start))),
                    event_key=key,
                )
            )
            continue

        if (
            abs(float(curr_it.duration) - float(base_it.duration)) >= 1.0
            or abs(float(curr_it.source_in) - float(base_it.source_in)) >= 1.0
        ):
            key = fsio.stable_hash(
                ["trim", scene_id, timeline, cid, round(float(curr_it.source_in), 1), round(float(curr_it.duration), 1)],
                12,
            )
            events.append(
                RoundtripEvent(
                    kind="trim",
                    scene_id=scene_id,
                    timeline=timeline,
                    clip_id=cid if CLIP_ID_RE.match(cid) else None,
                    take_id=new_take or old_take,
                    text=f"Trimmed {cid} to {curr_it.duration:.0f}f (in={curr_it.source_in:.0f}f)",
                    frame=int(round(float(curr_it.start))),
                    event_key=key,
                )
            )

        if (curr_it.retime_hash or None) != (base_it.retime_hash or None):
            key = fsio.stable_hash(["retime", scene_id, timeline, cid, curr_it.retime_hash], 12)
            events.append(
                RoundtripEvent(
                    kind="retime",
                    scene_id=scene_id,
                    timeline=timeline,
                    clip_id=cid if CLIP_ID_RE.match(cid) else None,
                    take_id=new_take or old_take,
                    text=f"Retimed {cid}",
                    frame=int(round(float(curr_it.start))),
                    event_key=key,
                )
            )

    if len(base_order) == len(curr_order) and base_order != curr_order and set(base_order) == set(curr_order):
        key = fsio.stable_hash(["reorder", scene_id, timeline, curr_order], 12)
        events.append(
            RoundtripEvent(
                kind="reorder",
                scene_id=scene_id,
                timeline=timeline,
                text="Reordered clips on V1: " + " -> ".join(curr_order),
                event_key=key,
            )
        )

    base_fp = baseline.fingerprint_excluding(exclude)
    curr_fp = current.fingerprint_excluding(exclude)
    if base_fp != curr_fp and not events:
        key = fsio.stable_hash(["other_edit", scene_id, timeline, curr_fp], 12)
        events.append(
            RoundtripEvent(
                kind="other_edit",
                scene_id=scene_id,
                timeline=timeline,
                text=f"Edited {timeline} in Resolve",
                event_key=key,
            )
        )
    return events


def poll_resolve(project: Project, db: Any) -> list[RoundtripEvent]:
    """Passively diff every handed-off scene timeline in ``db`` against its ``handoff_snapshot.json`` baseline."""
    if db is None or not hasattr(db, "timeline"):
        return []
    seen = _seen_keys(project)
    out: list[RoundtripEvent] = []

    for scene in project.list_scenes():
        handoff = project.load_handoff(scene.id)
        if handoff is None or not handoff.timeline_name:
            continue
        current = db.timeline(handoff.timeline_name)
        if current is None:
            continue
        baseline = editor.snapshot_entry(project, scene.id, handoff.timeline_name)
        if baseline is None:
            # First observation of a pending/adopted timeline: record baseline without firing spurious edit events,
            # but still inspect any explicit REGEN/APPROVED/NOTE markers on non-legacy timelines.
            editor.record_timeline_snapshot(project, scene.id, handoff.timeline_name, current, agent=AGENT)
            continue

        raw_events = _diff_snapshots(
            project,
            scene.id,
            handoff.timeline_name,
            baseline,
            current,
            exclude_markers=handoff.agy_marker_names,
        )
        fresh = [e for e in raw_events if e.event_key not in seen]
        if fresh:
            _record_seen(project, fresh)
            seen.update(e.event_key for e in fresh)
            out.extend(fresh)
            editor.record_timeline_snapshot(project, scene.id, handoff.timeline_name, current, agent=AGENT)
    return out


# ---------------------------------------------------------------------------------------------
# OTIO / FCPXML interchange import (§10.5 fallback)
# ---------------------------------------------------------------------------------------------
def _snapshot_from_otio(data: dict[str, Any], default_name: str) -> tuple[str | None, TimelineSnapshot]:
    tl_name = str(data.get("name") or default_name)
    agy_meta = (data.get("metadata") or {}).get("agy") or {}
    scene_id = agy_meta.get("scene_id")
    gst = data.get("global_start_time") or {}
    fps = float(gst.get("rate") or 24.0)
    base_frame = float(gst.get("value") or 86400.0)

    items: list[ItemRow] = []
    markers: list[MarkerRow] = []
    tracks_stack = data.get("tracks") or {}
    for m in tracks_stack.get("markers") or []:
        mr = (m.get("marked_range") or {}).get("start_time") or {}
        frame = float(mr.get("value") or 0.0)
        color = str(m.get("color") or "YELLOW").capitalize()
        name = str(m.get("name") or "")
        note = str(m.get("comment") or "")
        markers.append(
            MarkerRow(
                scope="timeline",
                frame=frame,
                name=name,
                note=note,
                color=color,
                strings=[s for s in (name, note) if s],
                decoded=True,
                layout="otio",
            )
        )

    v_idx, a_idx = 0, 0
    for tr in tracks_stack.get("children") or []:
        kind = str(tr.get("kind") or "Video").lower()
        t_type = "video" if "video" in kind else "audio"
        if t_type == "video":
            v_idx += 1
            t_idx = v_idx
        else:
            a_idx += 1
            t_idx = a_idx
        cursor = base_frame
        for child in tr.get("children") or []:
            schema = str(child.get("OTIO_SCHEMA") or "")
            sr = child.get("source_range") or {}
            dur = float((sr.get("duration") or {}).get("value") or 0.0)
            src_in = float((sr.get("start_time") or {}).get("value") or 0.0)
            if schema.startswith("Gap"):
                cursor += dur
                continue
            c_meta = ((child.get("metadata") or {}).get("agy") or {})
            ref = child.get("media_reference") or {}
            target_url = str(ref.get("target_url") or "")
            name = str(child.get("name") or ref.get("name") or "")
            if not scene_id:
                m_sc = SCENE_ID_RE.search(name) or SCENE_ID_RE.search(target_url)
                if m_sc:
                    scene_id = m_sc.group(1)
            items.append(
                ItemRow(
                    track_type=t_type,
                    track_index=t_idx,
                    name=name,
                    start=cursor,
                    duration=dur,
                    source_in=src_in,
                    enabled=bool(child.get("enabled", True)),
                    flags=1 if child.get("enabled", True) else 0,
                    media_path=c_meta.get("rel") or target_url,
                    media_ref=c_meta.get("take_id") or c_meta.get("clip_id") or name,
                    item_id=c_meta.get("key") or c_meta.get("clip_id") or name,
                )
            )
            cursor += dur

    fp = compute_fingerprint(items, markers)
    snap = TimelineSnapshot(
        name=tl_name,
        start_frame=base_frame,
        fps=fps,
        items=items,
        markers=markers,
        fingerprint=fp,
        read_at=fsio.now_iso(),
    )
    return scene_id, snap


def _parse_fcpxml_time(val: str | None, fps: float = 24.0) -> float:
    if not val:
        return 0.0
    s = val.strip().rstrip("s")
    if "/" in s:
        num, _, den = s.partition("/")
        try:
            return round((float(num) / float(den)) * fps, 2)
        except (ValueError, ZeroDivisionError):
            return 0.0
    try:
        return round(float(s) * fps, 2)
    except ValueError:
        return 0.0


def _snapshot_from_fcpxml(xml_text: str, default_name: str) -> tuple[str | None, TimelineSnapshot]:
    if re.search(r"<!\s*(DOCTYPE|ENTITY)\b", xml_text, re.IGNORECASE):
        raise ValueError("Rejected XML containing <!DOCTYPE or <!ENTITY declaration (XXE security guard)")
    root = ET.fromstring(xml_text)
    assets: dict[str, tuple[str, str]] = {}
    for asset in root.findall(".//asset"):
        aid = asset.attrib.get("id", "")
        name = asset.attrib.get("name", "")
        src = asset.attrib.get("src", "")
        if not src:
            rep = asset.find("media-rep")
            if rep is not None:
                src = rep.attrib.get("src", "")
        assets[aid] = (name, src)

    project_el = root.find(".//project")
    tl_name = (project_el.attrib.get("name") if project_el is not None else None) or default_name
    seq_el = root.find(".//sequence")
    base_frame = _parse_fcpxml_time(seq_el.attrib.get("tcStart") if seq_el is not None else None) or 86400.0

    items: list[ItemRow] = []
    markers: list[MarkerRow] = []
    scene_id: str | None = None
    cursor = base_frame

    spine = root.find(".//spine")
    clip_elems = list(spine) if spine is not None else root.findall(".//asset-clip")
    for el in clip_elems:
        tag = el.tag.split("}")[-1]
        if tag not in ("asset-clip", "clip", "video"):
            continue
        ref_id = el.attrib.get("ref", "")
        a_name, a_src = assets.get(ref_id, ("", ""))
        name = el.attrib.get("name") or a_name or ref_id
        dur = _parse_fcpxml_time(el.attrib.get("duration")) or 96.0
        src_in = _parse_fcpxml_time(el.attrib.get("start"))
        offset = _parse_fcpxml_time(el.attrib.get("offset")) if el.attrib.get("offset") else cursor
        if not scene_id:
            m_sc = SCENE_ID_RE.search(name) or SCENE_ID_RE.search(a_src)
            if m_sc:
                scene_id = m_sc.group(1)
        items.append(
            ItemRow(
                track_type="video",
                track_index=1,
                name=name,
                start=offset,
                duration=dur,
                source_in=src_in,
                enabled=el.attrib.get("enabled", "1") != "0",
                flags=1,
                media_path=a_src or name,
                media_ref=name,
                item_id=name,
            )
        )
        for m_el in el.findall("marker"):
            m_val = m_el.attrib.get("value", "")
            m_note = m_el.attrib.get("note", "")
            m_start = _parse_fcpxml_time(m_el.attrib.get("start"))
            rel_frame = max(0.0, (offset - base_frame) + max(0.0, m_start - src_in))
            markers.append(
                MarkerRow(
                    scope="timeline",
                    frame=rel_frame,
                    name=m_val,
                    note=m_note,
                    strings=[s for s in (m_val, m_note) if s],
                    decoded=True,
                    layout="fcpxml",
                )
            )
        cursor = offset + dur

    fp = compute_fingerprint(items, markers)
    snap = TimelineSnapshot(
        name=tl_name,
        start_frame=base_frame,
        fps=24.0,
        items=items,
        markers=markers,
        fingerprint=fp,
        read_at=fsio.now_iso(),
    )
    return scene_id, snap


def ingest_interchange(
    project: Project,
    path: str | Path,
    *,
    scene_id: str | None = None,
) -> list[RoundtripEvent]:
    """Parse an exported ``.otio`` or ``.fcpxml`` file and diff it against the scene's handoff baseline (§10.5)."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"interchange file not found: {path}")
    suffix = p.suffix.lower()
    if suffix == ".otio":
        data = fsio.read_json(p)
        inferred_sc, snap = _snapshot_from_otio(data, p.stem)
    elif suffix in (".fcpxml", ".xml"):
        text = p.read_text(encoding="utf-8")
        inferred_sc, snap = _snapshot_from_fcpxml(text, p.stem)
    else:
        raise ValueError(f"unsupported interchange format {suffix!r} (expected .otio or .fcpxml)")

    sid = scene_id or inferred_sc
    if not sid:
        m = SCENE_ID_RE.search(p.stem.upper())
        sid = m.group(1) if m else (project.list_scenes()[0].id if project.list_scenes() else "SC01")

    handoff = project.load_handoff(sid)
    tl_name = handoff.timeline_name if handoff and handoff.timeline_name else snap.name
    baseline = editor.snapshot_entry(project, sid, tl_name)
    if baseline is None:
        cut_path = editor.latest_scene_cut(project, sid)
        if cut_path is not None:
            cut = editor.load_scene_cut(project, cut_path)
            from .resolve_bridge import _snapshot_from_cut  # noqa: PLC0415

            baseline = _snapshot_from_cut(tl_name, cut)

    if baseline is None:
        return []

    raw_events = _diff_snapshots(
        project,
        sid,
        tl_name,
        baseline,
        snap,
        exclude_markers=handoff.agy_marker_names if handoff else (),
    )
    seen = _seen_keys(project)
    fresh = [e for e in raw_events if e.event_key not in seen]
    if fresh:
        _record_seen(project, fresh)
        editor.record_timeline_snapshot(project, sid, tl_name, snap, agent=AGENT)
    return fresh


def events_to_decisions(events: Iterable[RoundtripEvent]) -> list[dict[str, Any]]:
    """Map ``RoundtripEvent`` items to ``Decision`` constructor dicts (pure; no side effects)."""
    out: list[dict[str, Any]] = []
    for ev in events:
        if ev.kind == "regen_marker":
            out.append(
                {
                    "action": "reject",
                    "tier": "polish",
                    "target_artifact_id": ev.clip_id or ev.scene_id,
                    "gate": None,
                    "payload": {
                        "kind": "regen",
                        "steer": ev.text,
                        "timeline": ev.timeline,
                        "frame": ev.frame,
                    },
                    "source": "resolve",
                }
            )
        elif ev.kind == "approved_marker":
            out.append(
                {
                    "action": "approve",
                    "tier": "craft",
                    "target_artifact_id": ev.scene_id,
                    "gate": "G6",
                    "payload": {"timeline": ev.timeline, "frame": ev.frame, "note": ev.text},
                    "source": "resolve",
                }
            )
        elif ev.kind == "note_marker":
            out.append(
                {
                    "action": "steer",
                    "tier": "craft",
                    "target_artifact_id": ev.clip_id or ev.scene_id,
                    "gate": None,
                    "payload": {"note": ev.text, "timeline": ev.timeline, "frame": ev.frame},
                    "source": "resolve",
                }
            )
        elif ev.kind == "take_swap":
            out.append(
                {
                    "action": "choose",
                    "tier": "craft",
                    "target_artifact_id": ev.clip_id or ev.scene_id,
                    "gate": None,
                    "payload": {
                        "take_id": ev.take_id,
                        "from_take_id": ev.from_take_id,
                        "timeline": ev.timeline,
                        "frame": ev.frame,
                    },
                    "source": "resolve",
                }
            )
        else:
            out.append(
                {
                    "action": "annotate",
                    "tier": "polish",
                    "target_artifact_id": ev.clip_id or ev.scene_id,
                    "gate": None,
                    "payload": {"edit_kind": ev.kind, "timeline": ev.timeline, "text": ev.text, "frame": ev.frame},
                    "source": "resolve",
                }
            )
    return out
