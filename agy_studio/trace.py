"""``agy_studio trace`` — where did the time go?

A production that "ran for an hour and made nothing" is almost always one of: an LLM/agent call that stalled, a gate or
question waiting for a human that nobody relayed, or a provider job polling. This report is built purely from the
logs the engine already writes (``logs/agent_invocations.jsonl``, ``logs/state_transitions.jsonl``,
``logs/gates.jsonl``, per-take ``dispatch.jsonl``, ``questions.jsonl``) so it works on any production folder.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .fsio import read_jsonl
from .project import Project


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _ago(dt: datetime | None, now: datetime) -> str:
    if dt is None:
        return "never"
    secs = max(0.0, (now - dt).total_seconds())
    if secs < 90:
        return f"{secs:.0f}s ago"
    if secs < 5400:
        return f"{secs / 60:.0f} min ago"
    return f"{secs / 3600:.1f} h ago"


def _safe_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return read_jsonl(path) if path.is_file() else []
    except Exception:  # noqa: BLE001 - a half-written line must not break the diagnostic
        return []


def collect_trace(project: Project, *, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    logs = project.path("logs")
    events: list[tuple[datetime, str]] = []

    # --- agent / LLM calls -------------------------------------------------------------------
    calls = _safe_jsonl(logs / "agent_invocations.jsonl")
    per_role: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0, "total_s": 0.0, "max_s": 0.0, "errors": 0})
    for c in calls:
        role = str(c.get("agent") or "?")
        d = float(c.get("duration_s") or 0.0)
        r = per_role[role]
        r["n"] += 1
        r["total_s"] += d
        r["max_s"] = max(r["max_s"], d)
        if c.get("error"):
            r["errors"] += 1
        ts = _parse_ts(c.get("ts"))
        if ts:
            events.append((ts, f"agent {role} ({d:.0f}s{', ERROR' if c.get('error') else ''})"))

    # --- state transitions / gates / questions ------------------------------------------------
    for t in _safe_jsonl(logs / "state_transitions.jsonl"):
        ts = _parse_ts(t.get("ts"))
        if ts:
            events.append((ts, f"{t.get('scene_id') or 'GLOBAL'}: {t.get('from_stage')} -> {t.get('to_stage')} ({t.get('reason', '')})"))
    for g in _safe_jsonl(logs / "gates.jsonl"):
        ts = _parse_ts(g.get("ts"))
        if ts:
            events.append((ts, f"gate {g.get('gate')} {g.get('scene_id') or ''} {g.get('event') or g.get('status') or ''}".strip()))
    open_questions = []
    for q in _safe_jsonl(project.path("questions.jsonl")):
        if q.get("status") == "open":
            open_questions.append(q)
        ts = _parse_ts(q.get("ts") or q.get("asked_at"))
        if ts:
            events.append((ts, f"question {q.get('id')} [{q.get('status')}]: {str(q.get('text', ''))[:90]}"))

    # --- provider dispatch (video takes) -----------------------------------------------------
    dispatch_counts: dict[str, int] = defaultdict(int)
    inflight: list[str] = []
    for dlog in project.path("scenes").rglob("dispatch.jsonl") if project.path("scenes").is_dir() else []:
        rows = _safe_jsonl(dlog)
        last_by_take: dict[str, dict[str, Any]] = {}
        for r in rows:
            dispatch_counts[str(r.get("event"))] += 1
            last_by_take[str(r.get("take_id"))] = r
            ts = _parse_ts(r.get("ts"))
            if ts:
                events.append((ts, f"take {r.get('take_id')}: {r.get('event')} {r.get('provider_model') or ''}".rstrip()))
        for tid, r in last_by_take.items():
            if r.get("event") in ("submitted", "resume", "poll_timeout"):
                inflight.append(f"{tid} ({r.get('event')} {_ago(_parse_ts(r.get('ts')), now)})")

    events.sort(key=lambda e: e[0])
    last = events[-1][0] if events else None
    gaps = []
    for (t0, e0), (t1, e1) in zip(events, events[1:]):
        gap = (t1 - t0).total_seconds()
        if gap >= 180:
            gaps.append((gap, t0, e0, e1))
    gaps.sort(key=lambda g: -g[0])

    scenes: dict[str, str] = {}
    try:
        from .state import StateStore  # noqa: PLC0415

        st = StateStore(project).load()
    except Exception:  # noqa: BLE001
        st = None
    return {
        "now": now,
        "events": events,
        "per_role": dict(per_role),
        "dispatch_counts": dict(dispatch_counts),
        "inflight": inflight,
        "open_questions": open_questions,
        "last_activity": last,
        "gaps": gaps[:5],
        "scenes": scenes if st is None else {sid: sc.stage.value for sid, sc in st.scenes.items()},
    }


def render_trace(project: Project, *, limit: int = 25, now: datetime | None = None) -> str:
    t = collect_trace(project, now=now)
    now = t["now"]
    out = [f"{project.project_id} — where the time went",
           "  scenes               : " + (", ".join(f"{k}={v}" for k, v in t["scenes"].items()) or "none started (no ingest/G1 yet)"),
           f"  last engine activity : {_ago(t['last_activity'], now)}"
           + (f" ({t['last_activity'].isoformat(timespec='seconds')})" if t["last_activity"] else "")]
    if t["open_questions"]:
        out.append(f"  WAITING ON YOU       : {len(t['open_questions'])} open question(s) — the engine does nothing until answered:")
        for q in t["open_questions"]:
            out.append(f"    {q.get('id')} [{q.get('scene_id') or 'GLOBAL'}]: {str(q.get('text', ''))[:160]}")
    if t["inflight"]:
        out.append(f"  provider jobs in flight: {', '.join(t['inflight'])}")
    if t["per_role"]:
        out.append("  agent / LLM calls (role: calls, total, slowest, errors):")
        for role, r in sorted(t["per_role"].items(), key=lambda kv: -kv[1]["total_s"]):
            out.append(f"    {role:<24} {int(r['n']):>3}  {r['total_s']:>7.0f}s  max {r['max_s']:>5.0f}s  err {int(r['errors'])}")
        total = sum(r["total_s"] for r in t["per_role"].values())
        out.append(f"    {'TOTAL in LLM calls':<24} {sum(int(r['n']) for r in t['per_role'].values()):>3}  {total:>7.0f}s")
    else:
        out.append("  agent / LLM calls    : none recorded — nothing has reached the engine's agents yet")
    if t["dispatch_counts"]:
        out.append("  video dispatch events: " + ", ".join(f"{k}={v}" for k, v in sorted(t["dispatch_counts"].items())))
    else:
        out.append("  video dispatch events: none — no take has been submitted to Omni/Veo yet")
    if t["gaps"]:
        out.append("  biggest silent gaps (nothing logged between two events):")
        for gap, t0, e0, e1 in t["gaps"]:
            out.append(f"    {gap / 60:>5.1f} min  after  '{e0[:70]}'  ->  '{e1[:70]}'")
    if t["events"]:
        out.append(f"  last {min(limit, len(t['events']))} events:")
        for ts, e in t["events"][-limit:]:
            out.append(f"    {ts.astimezone(timezone.utc).strftime('%H:%M:%S')}  {e[:120]}")
    out.append("  tips: a big silent gap right after an 'agent …' line = a stalled LLM/HTTP call (now capped by request "
               "timeouts); an open question = the engine is waiting for an answer; 'submitted' with no 'completed' = "
               "provider still rendering (poll) or a dead job (`resume`).")
    return "\n".join(out)
