"""AGY Studio live dashboard — one data model, rendered as ``DASHBOARD.md`` (and served by ``dashboard_web``).

Everything here is READ from files the engine already writes; the dashboard never computes anything the pipeline
does not know. Sections (see ``collect_dashboard``):

* status      — scene stages, gates waiting on the Director, open questions, blockers, control mode
* time        — time spent per stage / per agent role, last activity, silent gaps (shares ``trace.collect_trace``)
* tokens/cost — Omni/Veo token counts from ``dispatch.jsonl``, LLM tokens from ``agent_invocations.jsonl``,
                budget caps / spend from ``state.json`` (prices flagged as estimates when the registry says so)
* eta         — remaining stages per scene × observed (or default) stage durations
* risks       — derived warnings: stale token, budget near cap, provider fallback in use, RAI re-prompts, stuck jobs,
                Resolve delivery failures, static / low-motion takes, long-open questions
* bibles      — Global Context Bible, scene bibles, working-set hashes, locked packs / environments / voice specs
* outputs     — takes in flight, completed takes, selected takes, images produced
* qa_failures — rejected / hard-rejected takes WITH the exact prompt, negative prompt, reference images and sheet
* decisions   — the latest decisions incl. autopilot self-approvals
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .fsio import read_jsonl, read_yaml
from .project import Project
from .trace import _ago, _parse_ts, collect_trace

DASHBOARD_NAME = "DASHBOARD.md"
TOKEN_CACHE_MIN = 45  # GcloudToken caches for 45 min; a re-auth may be needed after that on long runs

# Default dwell per stage (seconds) when this production has no observation yet.
DEFAULT_STAGE_S: dict[str, float] = {
    "pending": 5, "working_set": 10, "design": 120, "awaiting_G2": 0, "direction": 150, "awaiting_G3": 0,
    "decomposition": 20, "audio_plan": 30, "awaiting_G4": 0, "lint": 5, "generating": 0, "qa": 60,
    "awaiting_G5": 0, "assembly": 40, "handed_off": 0,
}
STAGE_ORDER = ["pending", "working_set", "design", "awaiting_G2", "direction", "awaiting_G3", "decomposition",
               "audio_plan", "awaiting_G4", "lint", "generating", "qa", "awaiting_G5", "assembly", "handed_off"]
PER_TAKE_RENDER_S = 240.0  # provider render default until measured


def _safe_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return read_jsonl(path) if path.is_file() else []
    except Exception:  # noqa: BLE001
        return []


def _safe_yaml(path: Path) -> dict[str, Any]:
    try:
        data = read_yaml(path) if path.is_file() else None
    except Exception:  # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


def _safe_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------------------------
# collectors
# ---------------------------------------------------------------------------------------------
def _stage_durations(transitions: list[dict[str, Any]]) -> dict[str, float]:
    """Median observed dwell per stage (seconds) from state_transitions.jsonl."""
    enter: dict[tuple[str | None, str], datetime] = {}
    samples: dict[str, list[float]] = {}
    for t in transitions:
        ts = _parse_ts(t.get("ts"))
        if ts is None:
            continue
        sid = t.get("scene_id")
        frm, to = str(t.get("from_stage")), str(t.get("to_stage"))
        key = (sid, frm)
        if key in enter:
            samples.setdefault(frm, []).append((ts - enter.pop(key)).total_seconds())
        enter[(sid, to)] = ts
    return {k: float(statistics.median(v)) for k, v in samples.items() if v}


def _takes(project: Project) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        clips = list(project.iter_all_clips())
    except Exception:  # noqa: BLE001
        return out
    for clip in clips:
        try:
            takes = project.list_takes(clip.id)
        except Exception:  # noqa: BLE001
            continue
        for t in takes:
            tdir = project.take_dir(t.id)
            payload = _safe_json(tdir / "payload.json")
            bundle = payload.get("bundle") or {}
            qa = t.qa
            sheets = sorted(str(project.rel(p)) for p in tdir.glob("*contact_3x3*.png")) if tdir.is_dir() else []
            out.append({
                "take_id": t.id, "clip_id": clip.id, "scene_id": clip.scene_id, "model": t.model,
                "provider_model": t.provider_model, "dispatch_status": getattr(t.dispatch_status, "value", str(t.dispatch_status)),
                "status": getattr(t.status, "value", str(t.status)), "operation_name": t.operation_name,
                "video_uri": t.video_uri, "selected": clip.selected_take_id == t.id,
                "prompt": bundle.get("prompt"), "negative_prompt": bundle.get("negative_prompt"),
                "reference_images": list(bundle.get("reference_images") or []), "first_frame": bundle.get("first_frame"),
                "last_frame": bundle.get("last_frame"), "contact_sheets": sheets,
                "qa": None if qa is None else {
                    "composite": qa.composite, "recommendation": getattr(qa.recommendation, "value", str(qa.recommendation)),
                    "hard_reject_reasons": list(qa.hard_reject_reasons), "artifact_flags": list(qa.artifact_flags),
                    "transcript_match": qa.transcript_match, "prompt_adherence": qa.prompt_adherence,
                    "notes": list(qa.notes)[:5]},
                "error": t.error,
            })
    return out


def _dispatch_usage(project: Project) -> dict[str, int]:
    totals = {"usage_input_tokens": 0, "usage_output_tokens": 0, "usage_thought_tokens": 0, "usage_total_tokens": 0}
    scenes = project.path("scenes")
    if not scenes.is_dir():
        return totals
    for dlog in scenes.rglob("dispatch.jsonl"):
        for r in _safe_jsonl(dlog):
            if r.get("event") == "completed":
                for k in totals:
                    if isinstance(r.get(k), int):
                        totals[k] += int(r[k])
    return totals


def _llm_usage(calls: list[dict[str, Any]]) -> dict[str, Any]:
    tot = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "thought_tokens": 0, "total_tokens": 0, "with_usage": 0}
    by_model: dict[str, int] = {}
    for c in calls:
        tot["calls"] += 1
        u = (c.get("outputs") or {}).get("usage") or {}
        if u:
            tot["with_usage"] += 1
            for k in ("input_tokens", "output_tokens", "thought_tokens", "total_tokens"):
                tot[k] += int(u.get(k) or 0)
            by_model[str(u.get("model") or c.get("model") or "?")] = by_model.get(str(u.get("model") or c.get("model") or "?"), 0) + int(u.get("total_tokens") or 0)
    tot["by_model"] = by_model
    return tot


def collect_dashboard(project: Project, *, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    cfg = project.config
    st_settings = cfg.settings
    trace = collect_trace(project, now=now)
    logs = project.path("logs")
    transitions = _safe_jsonl(logs / "state_transitions.jsonl")
    calls = _safe_jsonl(logs / "agent_invocations.jsonl")
    gates_log = _safe_jsonl(logs / "gates.jsonl")
    decisions = _safe_jsonl(project.path("decisions.jsonl"))
    safety = _safe_jsonl(logs / "safety_retries.jsonl")
    bridge = _safe_jsonl(logs / "resolve_bridge.jsonl")
    budget_log = _safe_jsonl(logs / "budget.jsonl")

    # --- state --------------------------------------------------------------------------------
    scenes: dict[str, dict[str, Any]] = {}
    budget_total = 0.0
    global_stage = None
    try:
        from .state import StateStore  # noqa: PLC0415

        st = StateStore(project).load()
        global_stage = getattr(st.global_stage, "value", str(st.global_stage))
        budget_total = float(getattr(st, "budget_spent_total", 0.0) or 0.0)
        for sid, sc in st.scenes.items():
            waiting = [g for g, s in sc.gates.items() if s == "awaiting"]
            scenes[sid] = {"stage": sc.stage.value, "gates_waiting": waiting, "blocked_by": list(sc.blocked_by),
                           "budget_cap": sc.budget_cap, "budget_spent": sc.budget_spent}
    except Exception:  # noqa: BLE001
        pass

    takes = _takes(project)
    by_scene_takes: dict[str, list[dict[str, Any]]] = {}
    for t in takes:
        by_scene_takes.setdefault(str(t["scene_id"]), []).append(t)

    # --- time / ETA ---------------------------------------------------------------------------
    observed = _stage_durations(transitions)
    completed_render: list[float] = []
    scenes_dir = project.path("scenes")
    if scenes_dir.is_dir():
        for dlog in scenes_dir.rglob("dispatch.jsonl"):
            sub: dict[str, datetime] = {}
            for r in _safe_jsonl(dlog):
                ts = _parse_ts(r.get("ts"))
                if ts is None:
                    continue
                if r.get("event") == "submitted":
                    sub[str(r.get("take_id"))] = ts
                elif r.get("event") == "completed" and str(r.get("take_id")) in sub:
                    completed_render.append((ts - sub.pop(str(r.get("take_id")))).total_seconds())
    render_s = float(statistics.median(completed_render)) if completed_render else PER_TAKE_RENDER_S
    autopilot = str(st_settings.control_mode) == "autopilot"
    eta: dict[str, Any] = {"per_scene": {}, "total_s": 0.0, "render_s_per_take": render_s,
                           "basis": "observed" if completed_render or observed else "defaults"}
    for sid, sc in scenes.items():
        stage = sc["stage"]
        if stage not in STAGE_ORDER:
            continue
        remaining = STAGE_ORDER[STAGE_ORDER.index(stage):]
        secs = 0.0
        human_waits = []
        try:
            clips = [c for c in project.iter_all_clips() if c.scene_id == sid]
        except Exception:  # noqa: BLE001
            clips = []
        n_clips = max(1, len(clips)) if clips else 3
        for s_name in remaining:
            if s_name.startswith("awaiting_"):
                if not autopilot:
                    human_waits.append(s_name)
                continue
            if s_name == "generating":
                pending_clips = [c for c in clips if not c.selected_take_id] or clips or [None] * n_clips
                secs += len(pending_clips) * int(st_settings.takes_per_clip) * render_s
            else:
                secs += observed.get(s_name, DEFAULT_STAGE_S.get(s_name, 30.0))
        eta["per_scene"][sid] = {"seconds": round(secs), "human_gates_ahead": human_waits}
        eta["total_s"] += secs

    # --- tokens / cost ------------------------------------------------------------------------
    video_usage = _dispatch_usage(project)
    llm_usage = _llm_usage(calls)
    price_is_estimate = True
    try:
        from .registry import default_registry  # noqa: PLC0415

        price_is_estimate = any(m.cost_is_estimate for m in default_registry().video_models())
    except Exception:  # noqa: BLE001
        pass

    # --- risks --------------------------------------------------------------------------------
    risks: list[dict[str, str]] = []
    first_call = min((_parse_ts(c.get("ts")) for c in calls if _parse_ts(c.get("ts"))), default=None)
    if first_call and (now - first_call).total_seconds() > TOKEN_CACHE_MIN * 60:
        risks.append({"level": "info", "text": f"Session older than {TOKEN_CACHE_MIN} min — the gcloud token will be "
                                               f"refreshed on the next call; if sign-in expired you will see an AuthError (fix: gcloud auth login)."})
    for sid, sc in scenes.items():
        cap, spent = sc.get("budget_cap"), float(sc.get("budget_spent") or 0.0)
        if cap and spent >= 0.8 * float(cap):
            risks.append({"level": "warn" if spent < cap else "error",
                          "text": f"{sid}: budget {spent:.2f} of cap {cap:.2f} ({spent / cap:.0%}) — `raise-cap {sid} <new cap>` will be needed to continue" if spent < cap else f"{sid}: budget cap {cap:.2f} reached — generation is blocked until `raise-cap {sid} <new cap>`"})
        if sc["blocked_by"]:
            risks.append({"level": "warn", "text": f"{sid}: blocked by {', '.join(sc['blocked_by'])}"})
    for q in trace["open_questions"]:
        age = (now - (_parse_ts(q.get("ts") or q.get("asked_at")) or now)).total_seconds()
        lvl = "error" if age > 300 else "warn"
        risks.append({"level": lvl, "text": f"question {q.get('id')} open {age / 60:.0f} min — the engine waits: "
                                            f"{str(q.get('text', ''))[:120]}"})
    try:
        from .registry import default_registry  # noqa: PLC0415

        reg = default_registry()
        for t in takes:
            if t["dispatch_status"] == "completed" and t["model"] in ("omni", "veo"):
                spec = reg.get(t["model"])
                primary = cfg.model_pins.get(spec.id) or spec.provider_model
                if t["provider_model"] and t["provider_model"] != primary:
                    risks.append({"level": "info", "text": f"{t['take_id']}: rendered by fallback provider {t['provider_model']} (primary {primary})"})
    except Exception:  # noqa: BLE001
        pass
    if safety:
        risks.append({"level": "warn", "text": f"{len(safety)} RAI safety re-prompt(s) so far — prompts were rewritten; check "
                                               f"logs/safety_retries.jsonl for what changed"})
    for s_ in trace["inflight"]:
        risks.append({"level": "warn", "text": f"provider job in flight: {s_} (median render {render_s:.0f}s) — `resume` re-polls if it looks stuck"})
    for t in takes:
        qa = t.get("qa") or {}
        flags = set(qa.get("artifact_flags") or [])
        if "static_video" in flags:
            risks.append({"level": "error", "text": f"{t['take_id']}: STATIC video (a still, not a generated take) — rejected"})
        elif "low_motion" in flags:
            risks.append({"level": "info", "text": f"{t['take_id']}: low motion — check it is not a slow zoom on a still"})
    last_bridge = bridge[-1] if bridge else None
    if last_bridge and last_bridge.get("error"):
        risks.append({"level": "error", "text": f"Resolve delivery failed last time: {str(last_bridge.get('error'))[:140]}"})
    if not scenes:
        risks.append({"level": "info", "text": "No scene has started — run `ingest <script>` then `g1` and approve G1."})

    # --- bibles -------------------------------------------------------------------------------
    gb = _safe_yaml(project.path("bible", "global_context_bible.yaml"))
    bibles: dict[str, Any] = {
        "global_context_bible": "bible/global_context_bible.yaml" if gb else None,
        "characters_locked": sorted(k for k, v in (gb.get("characters") or {}).items()
                                    if str((v or {}).get("approval_status", "")).lower() == "locked") if gb else [],
        "style_lock": bool(project.load_style_lock()) if hasattr(project, "load_style_lock") else None,
        "scenes": {},
        "model_policy": {"video_primary": st_settings.video_model_primary or "omni (registry default)",
                         "video_secondary": st_settings.video_model_secondary or "other engine (auto)",
                         "pins": dict(cfg.model_pins), "provider_fallbacks": dict(st_settings.provider_fallbacks)},
    }
    voices_dir = project.path("library", "voices")
    bibles["voice_specs_locked"] = sorted(p.name for p in voices_dir.iterdir()) if voices_dir.is_dir() else []
    for sid in scenes:
        sd = project.scene_dir(sid)
        ws = _safe_yaml(sd / "working_set.yaml")
        env = ws.get("environment_id") or ws.get("environment")
        cast = ws.get("entity_states") or ws.get("characters") or []
        bibles["scenes"][sid] = {"scene_bible": str(project.rel(sd / "SCENE_BIBLE.md")) if (sd / "SCENE_BIBLE.md").is_file() else None,
                                 "working_set_hash": ws.get("working_set_hash"),
                                 "environment": env.get("id") if isinstance(env, dict) else env,
                                 "characters": [c.get("id", str(c)) if isinstance(c, dict) else str(c) for c in cast][:12]}

    # --- outputs ------------------------------------------------------------------------------
    images = 0
    for folder in ("library/packs", "library/environments", "library/characters"):
        d = project.path(*folder.split("/"))
        if d.is_dir():
            images += sum(1 for p in d.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    boards = sum(1 for p in scenes_dir.rglob("boards/*.png")) if scenes_dir.is_dir() else 0
    outputs = {
        "images_in_library": images, "storyboard_frames": boards,
        "takes_total": len(takes),
        "takes_in_flight": [t for t in takes if t["dispatch_status"] in ("submitted", "running")],
        "takes_completed": sum(1 for t in takes if t["dispatch_status"] == "completed"),
        "takes_selected": [t["take_id"] for t in takes if t["selected"]],
        "final_mp4s": sorted(str(project.rel(p)) for p in project.path("post").rglob("*.mp4")) if project.path("post").is_dir() else [],
        "handoffs": {sid: (project.load_handoff(sid).timeline_name if project.load_handoff(sid) else None) for sid in scenes},
    }

    # --- QA failures --------------------------------------------------------------------------
    qa_failures = []
    for t in takes:
        qa = t.get("qa") or {}
        rejected = t["status"] == "rejected" or bool(qa.get("hard_reject_reasons")) or t["dispatch_status"] in ("failed", "filtered")
        if rejected:
            qa_failures.append(t)

    recent_decisions = [{"id": d.get("id"), "ts": d.get("ts"), "action": d.get("action"), "gate": d.get("gate"),
                         "target": d.get("target_artifact_id"), "user": d.get("user"), "source": d.get("source"),
                         "rationale": str(d.get("rationale") or "")[:100]} for d in decisions[-12:]]

    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "project": {"id": project.project_id, "title": cfg.title, "control_mode": st_settings.control_mode,
                    "content_format": st_settings.content_format, "aspect_ratio": st_settings.aspect_ratio,
                    "language": cfg.language, "takes_per_clip": st_settings.takes_per_clip, "global_stage": global_stage},
        "status": {"scenes": scenes, "open_questions": trace["open_questions"],
                   "last_activity": trace["last_activity"].isoformat(timespec="seconds") if trace["last_activity"] else None,
                   "last_activity_ago": _ago(trace["last_activity"], now)},
        "time": {"per_role": trace["per_role"], "llm_total_s": sum(r["total_s"] for r in trace["per_role"].values()),
                 "stage_medians_s": observed, "gaps": [(g, t0.isoformat(timespec="seconds"), e0, e1) for g, t0, e0, e1 in trace["gaps"]],
                 "events": [(ts.isoformat(timespec="seconds"), e) for ts, e in trace["events"][-40:]]},
        "tokens": {"video": video_usage, "llm": llm_usage},
        "cost": {"spent_total": budget_total, "per_scene": {sid: {"cap": sc["budget_cap"], "spent": sc["budget_spent"]} for sid, sc in scenes.items()},
                 "prices_are_estimates": price_is_estimate, "ledger_events": len(budget_log)},
        "eta": eta,
        "risks": risks,
        "bibles": bibles,
        "outputs": outputs,
        "qa_failures": qa_failures,
        "decisions": recent_decisions,
        "gates_log_tail": gates_log[-8:],
        "takes": takes,
    }


# ---------------------------------------------------------------------------------------------
# markdown renderer
# ---------------------------------------------------------------------------------------------
def _fmt_s(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def render_markdown(d: dict[str, Any]) -> str:
    p = d["project"]
    L: list[str] = [f"# AGY Studio — {p['id']} · {p['title']}",
                    f"_updated {d['generated_at']} · mode **{p['control_mode']}** · {p['content_format']} {p['aspect_ratio']} · "
                    f"lang {p['language']} · {p['takes_per_clip']} take(s)/clip · last engine activity **{d['status']['last_activity_ago']}**_", ""]
    # banner for auth / blocking
    errs = [r for r in d["risks"] if r["level"] == "error"]
    if errs:
        L += ["> [!CAUTION]", *[f"> {r['text']}" for r in errs], ""]
    # status
    L += ["## Status", "", "| Scene | Stage | Waiting on you | Blocked by | ETA (engine time left) |", "|---|---|---|---|---|"]
    for sid, sc in d["status"]["scenes"].items():
        eta = d["eta"]["per_scene"].get(sid, {})
        gates = ", ".join(sc["gates_waiting"]) or "—"
        hw = eta.get("human_gates_ahead") or []
        eta_txt = _fmt_s(eta.get("seconds", 0)) + (f" + {len(hw)} human gate(s)" if hw else "")
        L.append(f"| {sid} | `{sc['stage']}` | {gates} | {', '.join(sc['blocked_by']) or '—'} | {eta_txt} |")
    if not d["status"]["scenes"]:
        L.append("| — | no scene started | run `ingest` → `g1` | | |")
    if d["status"]["open_questions"]:
        L += ["", "**Open questions (the engine is waiting):**"]
        for q in d["status"]["open_questions"]:
            L.append(f"- `{q.get('id')}` [{q.get('scene_id') or 'GLOBAL'}] {str(q.get('text', ''))[:200]}")
    # risks
    L += ["", "## Things that may go wrong / need you", ""]
    if d["risks"]:
        icon = {"error": "🔴", "warn": "🟠", "info": "🔵"}
        L += [f"- {icon.get(r['level'], '•')} {r['text']}" for r in d["risks"]]
    else:
        L.append("- nothing flagged")
    # time & tokens & cost
    lu, vu = d["tokens"]["llm"], d["tokens"]["video"]
    L += ["", "## Time, tokens, cost", "",
          f"- LLM calls: **{lu['calls']}** · {_fmt_s(d['time']['llm_total_s'])} total · tokens in/out/thought: "
          f"{lu['input_tokens']:,}/{lu['output_tokens']:,}/{lu['thought_tokens']:,} (usage known for {lu['with_usage']} calls)",
          f"- Video (Omni/Veo) tokens in/out/total: {vu['usage_input_tokens']:,}/{vu['usage_output_tokens']:,}/{vu['usage_total_tokens']:,}",
          f"- Spend: **≈ {d['cost']['spent_total']:.2f}** total" + (" _(registry prices are estimates until confirmed against billing)_" if d["cost"]["prices_are_estimates"] else ""),
          f"- ETA basis: {d['eta']['basis']}; median provider render {_fmt_s(d['eta']['render_s_per_take'])} per take; engine time left ≈ **{_fmt_s(d['eta']['total_s'])}**"]
    if d["time"]["per_role"]:
        L += ["", "| Agent role | calls | total | slowest | errors |", "|---|---|---|---|---|"]
        for role, r in sorted(d["time"]["per_role"].items(), key=lambda kv: -kv[1]["total_s"]):
            L.append(f"| {role} | {int(r['n'])} | {_fmt_s(r['total_s'])} | {_fmt_s(r['max_s'])} | {int(r['errors'])} |")
    if d["time"]["gaps"]:
        L += ["", "Silent gaps (nothing logged): " + "; ".join(f"{g / 60:.0f} min after '{e0[:50]}'" for g, _, e0, _ in d["time"]["gaps"][:3])]
    # bibles
    b = d["bibles"]
    L += ["", "## Bibles & policy in use", "",
          f"- Global Context Bible: {'`' + b['global_context_bible'] + '`' if b['global_context_bible'] else '— (not built yet)'}; "
          f"locked characters: {', '.join(b['characters_locked']) or 'none'}; voice specs: {', '.join(b['voice_specs_locked']) or 'none'}",
          f"- Video engines: primary **{b['model_policy']['video_primary']}**, secondary **{b['model_policy']['video_secondary']}**"
          + (f"; pins {b['model_policy']['pins']}" if b['model_policy']['pins'] else "")
          + (f"; provider chains {b['model_policy']['provider_fallbacks']}" if b['model_policy']['provider_fallbacks'] else "")]
    for sid, sb in b["scenes"].items():
        L.append(f"- {sid}: scene bible {'`' + sb['scene_bible'] + '`' if sb['scene_bible'] else '—'} · working set `{sb['working_set_hash'] or '—'}` · env {sb['environment'] or '—'} · cast {', '.join(sb['characters']) or '—'}")
    # outputs
    o = d["outputs"]
    L += ["", "## Outputs", "",
          f"- Images in library: {o['images_in_library']} · storyboard frames: {o['storyboard_frames']}",
          f"- Takes: {o['takes_completed']} completed / {o['takes_total']} total · selected: {', '.join(o['takes_selected']) or 'none yet'}"]
    for t in o["takes_in_flight"]:
        L.append(f"  - ⏳ {t['take_id']} on {t['provider_model']} ({t['dispatch_status']})")
    for sid, tl in o["handoffs"].items():
        if tl:
            L.append(f"- {sid}: timeline **{tl}** delivered to Resolve")
    for m in o["final_mp4s"][:5]:
        L.append(f"- final: `{m}`")
    # QA failures
    L += ["", "## Takes that failed QA (with prompt & references)", ""]
    if not d["qa_failures"]:
        L.append("- none")
    for t in d["qa_failures"][-10:]:
        qa = t.get("qa") or {}
        why = ", ".join(qa.get("hard_reject_reasons") or []) or (t.get("error") or {}).get("message") if isinstance(t.get("error"), dict) else ", ".join(qa.get("hard_reject_reasons") or []) or str(t.get("error") or t["dispatch_status"])
        L.append(f"<details><summary><b>{t['take_id']}</b> ({t['provider_model'] or t['model']}) — {why}</summary>")
        L.append("")
        if qa:
            L.append(f"- composite {qa.get('composite')} · recommendation {qa.get('recommendation')} · transcript {qa.get('transcript_match')} · adherence {qa.get('prompt_adherence')} · flags {', '.join(qa.get('artifact_flags') or []) or '—'}")
        if t.get("prompt"):
            L += ["- prompt:", "", "```", str(t["prompt"])[:1500], "```"]
        if t.get("negative_prompt"):
            L.append(f"- negative: `{str(t['negative_prompt'])[:300]}`")
        refs = [r for r in [t.get("first_frame"), t.get("last_frame"), *t.get("reference_images", [])] if r]
        if refs:
            L.append("- references: " + ", ".join(f"[{Path(r).name}]({r})" for r in refs))
        for sh in t.get("contact_sheets", []):
            L.append(f"- contact sheet: ![{Path(sh).name}]({sh})")
        L += ["", "</details>", ""]
    # decisions
    L += ["## Latest decisions", ""]
    for dd in d["decisions"][-8:]:
        L.append(f"- `{dd['id']}` {dd['action']} {dd.get('gate') or ''} {dd.get('target') or ''} by {dd.get('user')}"
                 + (f" ({dd['source']})" if dd.get("source") else "") + (f" — {dd['rationale']}" if dd.get("rationale") else ""))
    if not d["decisions"]:
        L.append("- none yet")
    L += ["", "---", "_Refresh: `python -m agy_studio --project " + p["id"] + " dashboard --write` · live page: `dashboard serve --open`_"]
    return "\n".join(L) + "\n"


def write_dashboard(project: Project) -> Path:
    """Render ``DASHBOARD.md`` into the production folder (called by the engine after every step)."""
    path = project.root / DASHBOARD_NAME
    path.write_text(render_markdown(collect_dashboard(project)), encoding="utf-8")
    return path
