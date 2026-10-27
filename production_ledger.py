#!/usr/bin/env python3
"""production_ledger.py — Scene-by-scene production log, token & cost ledger for AGY film projects.

What it tracks (per scene + project totals)
  * Time      : wall-clock window, Director review / idle time, tooling pauses and net production time.
  * Generation: images (Nano Banana Pro / generate_image), Gemini Omni 1.1 Flash 1080p scene clips,
                Omni 1.1 Flash 360p SFX videos, Veo 3.1 clips, Lyria-002 scores and planner LLM calls.
  * Reshoots  : safety-filter retries, duplicate takes and re-takes of already delivered clips.
  * Tokens    : real API usage when the provider returns it (Omni 1.1 does), documented estimates otherwise.
  * Cost      : computed from the unit prices in ``06_production_logs/model_unit_prices.json``.

Everything is written to one append-only log file — ``projects/<project>/06_production_logs/production_ledger.jsonl``
— and summarised in ``PRODUCTION_REPORT.md`` (+ ``production_summary.csv``) next to it.

Usage (from the workspace root):
  .venv/bin/python production_ledger.py sync                 # import new generations (transcript, dispatch logs, stems)
  .venv/bin/python production_ledger.py scene start SC05 --title "Doosra Naam: Kanta"
  .venv/bin/python production_ledger.py scene pause SC05 --note "ledger setup"   (also: resume / end)
  .venv/bin/python production_ledger.py report               # regenerate the report and print the totals
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT = os.environ.get("AGY_LEDGER_PROJECT", "Do_Dulhane_Gem")
IST = timezone(timedelta(hours=5, minutes=30))

# Unit keys understood by the cost engine, per model.
MODELS: dict[str, dict[str, str]] = {
    "nano-banana-pro": {"label": "Nano Banana Pro (generate_image) — keyframes, sheets & plates", "kind": "image"},
    "omni-1.1-flash-1080p": {"label": "Gemini Omni 1.1 Flash — 1080p scene clips", "kind": "video"},
    "omni-1.1-flash-360p-sfx": {"label": "Gemini Omni 1.1 Flash — 360p SFX videos", "kind": "video"},
    "veo-3.1": {"label": "Vertex AI Veo 3.1 — Act-1 pilot clips", "kind": "video"},
    "lyria-002": {"label": "Vertex AI Lyria-002 — background scores", "kind": "music"},
    "gemini-2.5-pro-planner": {"label": "Gemini 2.5 Pro — planner LLM calls", "kind": "llm"},
}

# Token accounting used when a provider does not report usage (backfill).
# Omni figures were MEASURED from the live API on 10-Oct-2026 (SC04_B07 1080p 10s clip, SC04_B17 360p 10s SFX).
TOKEN_ESTIMATES = {
    "omni_1080p_output_per_s": 8688.0,   # 86,880 output video tokens / 10 s
    "omni_1080p_input_per_call": 3992,   # 3 reference images (3,300) + prompt text (692)
    "omni_1080p_thought_per_call": 516,
    "omni_360p_output_per_s": 1931.0,    # 19,310 output video tokens / 10 s
    "omni_360p_input_per_call": 79,
    "omni_360p_thought_per_call": 457,
    "image_output_tokens": 1120,         # Google's published accounting for a 1K/2K Nano Banana Pro image
    "image_input_tokens_per_ref": 560,   # per reference image passed to the image model
    "chars_per_token": 4,
    "lyria_clip_seconds": 32.8,          # one lyria-002 call returns ~32.8 s of 48 kHz audio
}

DEFAULT_PRICES: dict[str, Any] = {
    "currency": "USD",
    "confirmed_by_director": False,
    "bill_filtered_attempts": False,
    "models": {
        "nano-banana-pro": {"per_image": 0.134},
        "omni-1.1-flash-1080p": {"per_second": 0.20},
        "omni-1.1-flash-360p-sfx": {"per_second": 0.05},
        "veo-3.1": {"per_second": 0.40},
        "lyria-002": {"per_call": 0.06},
        "gemini-2.5-pro-planner": {"per_call": 0.02},
    },
    "notes": "Placeholder values from agy_studio/registry.yaml until the Director confirms the real unit prices.",
}

IMG_FILE_RX = re.compile(r"^(?P<name>.+)_(?P<ms>\d{13})\.(?:jpg|jpeg|png|webp)$", re.I)
SFX_SUBMIT_RX = re.compile(r"\[Omni 1\.1 360p SFX\] Submitted (?P<stem>\S+?\.wav) \((?P<dur>\d+)s\) -> (?P<op>\S+)")


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def parse_ts(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=IST)
    s = str(value).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=IST)


def iso(dt: datetime) -> str:
    return dt.astimezone(IST).isoformat(timespec="seconds")


def now() -> datetime:
    return datetime.now(tz=IST)


def eid(*parts: Any) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]


def fmt_dur(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, _ = divmod(rem, 60)
    if d:
        return f"{d}d {h}h {m:02d}m"
    return f"{h}h {m:02d}m"


def fmt_int(n: float) -> str:
    return f"{int(round(n)):,}"


def money(v: float, cur: str = "USD") -> str:
    sym = "$" if cur.upper() == "USD" else (cur + " ")
    return f"{sym}{v:,.2f}"


class Ledger:
    def __init__(self, project: str = DEFAULT_PROJECT):
        self.project = project
        self.proj_dir = ROOT / "projects" / project
        self.prod_dir = ROOT / "productions" / project
        self.log_dir = self.proj_dir / "06_production_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.log_dir / "production_ledger.jsonl"
        self.prices_path = self.log_dir / "model_unit_prices.json"
        self.config_path = self.log_dir / "ledger_config.json"
        self.report_md = self.log_dir / "PRODUCTION_REPORT.md"
        self.report_csv = self.log_dir / "production_summary.csv"
        self._events: list[dict[str, Any]] | None = None
        self._ids: set[str] | None = None
        self._index: dict[str, int] = {}

    # -- storage ------------------------------------------------------------------------------
    def events(self) -> list[dict[str, Any]]:
        """All events; when an id appears more than once the LAST line wins (append-only upserts)."""
        if self._events is None:
            self._events, self._index = [], {}
            if self.ledger_path.exists():
                for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = ev.get("id") or f"_anon_{len(self._events)}"
                    if key in self._index:
                        self._events[self._index[key]] = ev
                    else:
                        self._index[key] = len(self._events)
                        self._events.append(ev)
            self._ids = set(self._index)
        return self._events

    def append(self, event: dict[str, Any], *, upsert: bool = False) -> bool:
        """Append one event. With ``upsert=True`` an existing id is superseded when its content changed."""
        self.events()
        assert self._ids is not None and self._events is not None
        event = dict(event)
        event.setdefault("ts", iso(now()))
        event.setdefault("project", self.project)
        event.setdefault("id", eid(event.get("event"), event.get("ts"), event.get("scene"), event.get("name")))
        key = event["id"]
        if key in self._ids:
            if not upsert or self._events[self._index[key]] == event:
                return False
        with self.ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        if key in self._index:
            self._events[self._index[key]] = event
        else:
            self._index[key] = len(self._events)
            self._events.append(event)
            self._ids.add(key)
        return True

    def config(self) -> dict[str, Any]:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        return {}

    def prices(self) -> dict[str, Any]:
        if not self.prices_path.exists():
            self.prices_path.write_text(json.dumps(DEFAULT_PRICES, indent=2), encoding="utf-8")
        data = json.loads(self.prices_path.read_text(encoding="utf-8"))
        merged = json.loads(json.dumps(DEFAULT_PRICES))
        merged.update({k: v for k, v in data.items() if k != "models"})
        merged["models"].update(data.get("models", {}))
        return merged

    # -- scenes -------------------------------------------------------------------------------
    def scene_event(self, action: str, scene: str, *, title: str | None = None, note: str | None = None,
                    at: str | None = None, source: str = "live") -> bool:
        ts = parse_ts(at) if at else now()
        assert ts is not None
        ev: dict[str, Any] = {"event": f"scene_{action}", "scene": scene.upper(), "ts": iso(ts), "source": source}
        if title:
            ev["title"] = title
        if note:
            ev["note"] = note
        ev["id"] = eid("scene", action, scene.upper(), iso(ts))
        return self.append(ev)

    def windows(self) -> list[dict[str, Any]]:
        """Ordered scene windows built from scene_start / scene_end / pause / resume events."""
        starts: dict[str, dict[str, Any]] = {}
        for e in sorted(self.events(), key=lambda x: x.get("ts", "")):
            ev, sc = e.get("event"), e.get("scene")
            if ev == "scene_start" and sc not in starts:
                starts[sc] = {"scene": sc, "title": e.get("title", ""), "start": parse_ts(e["ts"]), "end": None,
                              "pauses": [], "_open_pause": None}
            elif sc in starts:
                w = starts[sc]
                if ev == "scene_end":
                    w["end"] = parse_ts(e["ts"])
                elif ev == "scene_pause":
                    w["_open_pause"] = (parse_ts(e["ts"]), e.get("note", ""))
                elif ev == "scene_resume" and w["_open_pause"]:
                    p_start, note = w["_open_pause"]
                    w["pauses"].append((p_start, parse_ts(e["ts"]), note))
                    w["_open_pause"] = None
                if e.get("title"):
                    w["title"] = e["title"]
        ordered = sorted(starts.values(), key=lambda w: w["start"])
        for i, w in enumerate(ordered):
            if w["end"] is None:
                w["end"] = ordered[i + 1]["start"] if i + 1 < len(ordered) else None
            if w["_open_pause"]:
                p_start, note = w["_open_pause"]
                w["pauses"].append((p_start, w["end"] or now(), note))
            w.pop("_open_pause", None)
        return ordered

    def scene_of(self, ts: datetime | None) -> str:
        if ts is None:
            return "UNASSIGNED"
        for w in self.windows():
            end = w["end"] or now() + timedelta(days=3650)
            if w["start"] <= ts < end:
                return w["scene"]
        return "UNASSIGNED"

    # -- live logging helpers (called from hooks) ---------------------------------------------
    def log_generation(self, event: str, **fields: Any) -> bool:
        ts = parse_ts(fields.pop("ts", None)) or now()
        fields.setdefault("scene", self.scene_of(ts))
        fields["event"] = event
        fields["ts"] = iso(ts)
        fields.setdefault("source", "live")
        fields.setdefault("id", eid(event, fields.get("model"), fields.get("operation") or fields.get("name"), iso(ts)))
        return self.append(fields)

    # -- sync / backfill ----------------------------------------------------------------------
    def sync(self, verbose: bool = True) -> Counter:
        added: Counter = Counter()
        cfg = self.config()
        start_floor = parse_ts(cfg.get("project_start")) or datetime(2000, 1, 1, tzinfo=IST)
        for tpath in cfg.get("transcripts", []):
            if Path(tpath).exists():
                a = self._sync_transcript(Path(tpath), [Path(p) for p in cfg.get("artifact_dirs", [])], start_floor)
                added.update(a)
        added.update(self._sync_dispatch(start_floor))
        added.update(self._sync_audio([Path(p) for p in cfg.get("task_log_dirs", [])], start_floor))
        added.update(self._sync_planner_calls(start_floor))
        if verbose:
            print("[ledger] sync added:", dict(added) or "nothing new")
        return added

    def _sync_transcript(self, tpath: Path, artifact_dirs: list[Path], floor: datetime) -> Counter:
        added: Counter = Counter()
        rows: list[dict[str, Any]] = []
        for line in tpath.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        # 1) Director review / idle gaps: time between my last step and the Director's next message.
        prev_ts: datetime | None = None
        for r in rows:
            ts = parse_ts(r.get("created_at"))
            if ts is None:
                continue
            if r.get("type") == "USER_INPUT" and r.get("source") == "USER_EXPLICIT" and prev_ts is not None and ts >= floor:
                gap = (ts - prev_ts).total_seconds()
                if gap >= 30:
                    c = r.get("content") or ""
                    kind = "director_review" if "approved this document" in c or "Comments on artifact" in c else "director_idle"
                    if self.append({"event": "idle", "kind": kind, "start": iso(prev_ts), "end": iso(ts), "ts": iso(ts),
                                    "seconds": round(gap, 1), "source": "transcript",
                                    "id": eid("idle", tpath.name, r.get("step_index"))}):
                        added["idle"] += 1
            prev_ts = ts
        # 2) Image generations (generate_image tool calls) matched to the files they produced.
        files: dict[str, list[tuple[int, Path]]] = defaultdict(list)
        for d in artifact_dirs:
            if d.exists():
                for p in d.iterdir():
                    m = IMG_FILE_RX.match(p.name)
                    if m:
                        files[m.group("name")].append((int(m.group("ms")), p))
        for v in files.values():
            v.sort()
        used: set[Path] = set()
        for r in rows:
            ts = parse_ts(r.get("created_at"))
            if ts is None or ts < floor:
                continue
            for idx, tc in enumerate(r.get("tool_calls") or []):
                name, args = _tc_name_args(tc)
                if name != "generate_image":
                    continue
                img = str(args.get("ImageName") or "image")
                refs = args.get("ImagePaths") or []
                n_refs = len(refs) if isinstance(refs, list) else 0
                p_chars = len(str(args.get("Prompt") or ""))
                t_ms = int(ts.timestamp() * 1000)
                match = None
                for ms, p in files.get(img, []):
                    if p in used:
                        continue
                    if t_ms - 5_000 <= ms <= t_ms + 15 * 60_000:
                        match = (ms, p)
                        break
                status = "completed" if match else "failed"
                if match:
                    used.add(match[1])
                in_tok = math.ceil(p_chars / TOKEN_ESTIMATES["chars_per_token"]) + n_refs * TOKEN_ESTIMATES["image_input_tokens_per_ref"]
                out_tok = TOKEN_ESTIMATES["image_output_tokens"] if match else 0
                ev = {"event": "image_generation", "model": "nano-banana-pro", "name": img, "ts": iso(ts),
                      "scene": self.scene_of(ts), "status": status, "ref_images": n_refs, "prompt_chars": p_chars,
                      "input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": in_tok + out_tok,
                      "tokens_source": "estimated", "file": str(match[1]) if match else None,
                      "finished_ts": iso(datetime.fromtimestamp(match[0] / 1000, tz=IST)) if match else None,
                      "source": "transcript", "id": eid("img", tpath.name, r.get("step_index"), idx, img)}
                if self.append(ev, upsert=True):
                    added["image_generation"] += 1
        return added

    def _sync_dispatch(self, floor: datetime) -> Counter:
        added: Counter = Counter()
        attempts: list[dict[str, Any]] = []
        for f in sorted(glob.glob(str(self.prod_dir / "scenes" / "**" / "dispatch.jsonl"), recursive=True)):
            take_dir = Path(f).parent
            take_id = take_dir.name
            open_attempts: list[dict[str, Any]] = []
            for line in Path(f).read_text(encoding="utf-8").splitlines():
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ev, ts = e.get("event"), parse_ts(e.get("ts"))
                if ev == "submitted":
                    open_attempts.append({"take_id": take_id, "take_dir": take_dir, "model": e.get("model"),
                                          "provider_model": e.get("provider_model"), "operation": e.get("operation_name"),
                                          "attempt": e.get("attempt"), "safety_level": e.get("safety_level"),
                                          "submitted": ts, "finished": None, "status": "in_flight", "usage": None})
                elif open_attempts and ev in ("completed", "result_filtered", "result_failed"):
                    # overlapping runs can interleave in one take folder: match by operation id, else oldest open
                    op = e.get("operation_name")
                    cur = next((a for a in open_attempts if op and a["operation"] == op), open_attempts[0])
                    open_attempts.remove(cur)
                    cur["finished"] = ts
                    cur["status"] = {"completed": "completed", "result_filtered": "filtered"}.get(ev, "failed")
                    if isinstance(e.get("usage"), dict):
                        cur["usage"] = e["usage"]
                    elif e.get("usage_output_tokens") is not None or e.get("usage_input_tokens") is not None:
                        cur["usage"] = {"total_input_tokens": int(e.get("usage_input_tokens") or 0),
                                        "total_output_tokens": int(e.get("usage_output_tokens") or 0),
                                        "total_thought_tokens": int(e.get("usage_thought_tokens") or 0)}
                    attempts.append(cur)
            attempts.extend(open_attempts)
        # durations of delivered takes
        dur_cache_path = self.log_dir / ".take_durations.json"
        dur_cache = json.loads(dur_cache_path.read_text()) if dur_cache_path.exists() else {}
        for a in attempts:
            mp4 = a["take_dir"] / "take.mp4"
            key = str(mp4)
            if a["status"] == "completed" and mp4.exists() and key not in dur_cache:
                dur_cache[key] = _ffprobe_duration(mp4)
        dur_cache_path.write_text(json.dumps(dur_cache))
        clip_dur: dict[str, float] = {}
        for a in attempts:
            clip = a["take_id"].rsplit("_", 1)[0]
            d = dur_cache.get(str(a["take_dir"] / "take.mp4"))
            if a["status"] == "completed" and d:
                clip_dur.setdefault(clip, d)
        # reshoot classification per clip (chronological)
        by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for a in attempts:
            by_clip[a["take_id"].rsplit("_", 1)[0]].append(a)
        for clip, lst in by_clip.items():
            lst.sort(key=lambda a: a["submitted"] or floor)
            for i, a in enumerate(lst):
                if i == 0:
                    a["reshoot"], a["reshoot_reason"] = False, None
                    continue
                prev = lst[i - 1]
                a["reshoot"] = True
                if prev["status"] in ("filtered", "failed"):
                    a["reshoot_reason"] = "safety_retry"
                elif prev["finished"] and a["submitted"] and prev["finished"] > a["submitted"]:
                    a["reshoot_reason"] = "duplicate_parallel_run"
                else:
                    a["reshoot_reason"] = "retake"
        for a in attempts:
            if a["submitted"] is None or a["submitted"] < floor:
                continue
            clip = a["take_id"].rsplit("_", 1)[0]
            model = "veo-3.1" if str(a["model"]).startswith("veo") else "omni-1.1-flash-1080p"
            dur = dur_cache.get(str(a["take_dir"] / "take.mp4")) if a["status"] == "completed" else None
            dur_src = "measured"
            if not dur:
                dur, dur_src = clip_dur.get(clip, 8.0 if model == "veo-3.1" else 10.0), "planned"
            usage = a.get("usage")
            if usage:
                in_tok = int(usage.get("total_input_tokens") or 0)
                out_tok = int(usage.get("total_output_tokens") or 0)
                th_tok = int(usage.get("total_thought_tokens") or 0)
                tok_src = "api"
            elif model.startswith("omni"):
                in_tok = TOKEN_ESTIMATES["omni_1080p_input_per_call"]
                out_tok = int(round(TOKEN_ESTIMATES["omni_1080p_output_per_s"] * round(dur))) if a["status"] == "completed" else 0
                th_tok = TOKEN_ESTIMATES["omni_1080p_thought_per_call"]
                tok_src = "estimated"
            else:
                in_tok = out_tok = th_tok = 0
                tok_src = "not_reported"
            ev = {"event": "video_generation", "model": model, "provider_model": a["provider_model"],
                  "purpose": "scene_clip", "name": clip, "take_id": a["take_id"], "operation": a["operation"],
                  "attempt": a["attempt"], "status": a["status"], "resolution": "1080p", "duration_s": round(dur, 2),
                  "duration_source": dur_src, "ts": iso(a["submitted"]), "scene": self.scene_of(a["submitted"]),
                  "finished_ts": iso(a["finished"]) if a["finished"] else None,
                  "gen_seconds": round((a["finished"] - a["submitted"]).total_seconds(), 1) if a["finished"] else None,
                  "reshoot": a.get("reshoot", False), "reshoot_reason": a.get("reshoot_reason"),
                  "input_tokens": in_tok, "output_tokens": out_tok, "thought_tokens": th_tok,
                  "total_tokens": in_tok + out_tok + th_tok, "tokens_source": tok_src, "source": "dispatch_log",
                  "id": eid("vid", a["take_id"], a["operation"])}
            if self.append(ev, upsert=True):
                added["video_generation"] += 1
        return added

    def _sync_audio(self, task_dirs: list[Path], floor: datetime) -> Counter:
        added: Counter = Counter()
        ops: dict[str, str] = {}
        for d in task_dirs:
            for log in sorted(d.glob("*.log")) if d.exists() else []:
                try:
                    txt = log.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for m in SFX_SUBMIT_RX.finditer(txt):
                    ops[m.group("stem")] = m.group("op")
        live_names = {(e.get("event"), e.get("name")) for e in self.events() if e.get("source") == "live"}
        for marker in sorted(glob.glob(str(self.proj_dir / "04_video_clips" / "**" / "audio_stems" / "*.ok"), recursive=True)):
            mp = Path(marker)
            ts = datetime.fromtimestamp(mp.stat().st_mtime, tz=IST)
            if ts < floor:
                continue
            stem = mp.name.replace(".omni_sfx.ok", ".wav").replace(".lyria.ok", ".wav")
            if mp.name.endswith(".lyria.ok"):
                if ("music_generation", stem) in live_names:
                    continue
                wav = mp.with_name(stem)
                secs = _wav_seconds(wav)
                ev = {"event": "music_generation", "model": "lyria-002", "name": stem, "status": "completed",
                      "api_calls": 1, "audio_seconds_generated": TOKEN_ESTIMATES["lyria_clip_seconds"],
                      "stem_seconds": secs, "ts": iso(ts), "scene": self.scene_of(ts), "source": "stem_marker",
                      "id": eid("lyria", stem, int(mp.stat().st_mtime))}
                if self.append(ev):
                    added["music_generation"] += 1
            elif mp.name.endswith(".omni_sfx.ok"):
                if ("video_generation", stem) in live_names:
                    continue
                dur = 10.0
                ev = {"event": "video_generation", "model": "omni-1.1-flash-360p-sfx",
                      "provider_model": "gemini-omni-1.1-flash-preview", "purpose": "sfx_360p", "name": stem,
                      "operation": ops.get(stem), "status": "completed", "resolution": "360p", "duration_s": dur,
                      "duration_source": "planned", "ts": iso(ts), "scene": self.scene_of(ts), "reshoot": False,
                      "input_tokens": TOKEN_ESTIMATES["omni_360p_input_per_call"],
                      "output_tokens": int(TOKEN_ESTIMATES["omni_360p_output_per_s"] * dur),
                      "thought_tokens": TOKEN_ESTIMATES["omni_360p_thought_per_call"], "tokens_source": "estimated",
                      "source": "stem_marker", "id": eid("sfx", stem, int(mp.stat().st_mtime))}
                ev["total_tokens"] = ev["input_tokens"] + ev["output_tokens"] + ev["thought_tokens"]
                if self.append(ev):
                    added["video_generation"] += 1
        return added

    def _sync_planner_calls(self, floor: datetime) -> Counter:
        added: Counter = Counter()
        for f in sorted(glob.glob(str(self.prod_dir / "logs" / "agent_calls" / "*.json"))):
            try:
                data = json.loads(Path(f).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if data.get("mode") != "llm":
                continue
            ts = parse_ts(data.get("ts"))
            if ts is None or ts < floor:
                continue
            ev = {"event": "llm_call", "model": "gemini-2.5-pro-planner", "name": f"{data.get('agent')}:{data.get('scene_id')}",
                  "status": "completed", "ts": iso(ts), "scene": self.scene_of(ts), "source": "agent_calls",
                  "tokens_source": "not_reported", "id": eid("llm", Path(f).name)}
            if self.append(ev):
                added["llm_call"] += 1
        return added

    # -- reporting ----------------------------------------------------------------------------
    def compute(self) -> dict[str, Any]:
        prices = self.prices()
        bill_filtered = bool(prices.get("bill_filtered_attempts"))
        wins = self.windows()
        evs = self.events()
        idles = [(parse_ts(e["start"]), parse_ts(e["end"]), e.get("kind")) for e in evs if e.get("event") == "idle"]
        scenes: dict[str, dict[str, Any]] = {}
        order = [w["scene"] for w in wins]
        for w in wins:
            start, end = w["start"], (w["end"] or now())
            wall = (end - start).total_seconds()
            idle = sum(_overlap(start, end, a, b) for a, b, _ in idles if a and b)
            review = sum(_overlap(start, end, a, b) for a, b, k in idles if a and b and k == "director_review")
            pause = sum(_overlap(start, end, a, b) for a, b, _ in w["pauses"] if a and b)
            pause_idle = sum(_overlap(max(start, pa), min(end, pb), a, b)
                             for pa, pb, _ in w["pauses"] if pa and pb for a, b, _k in idles if a and b)
            scenes[w["scene"]] = {
                "scene": w["scene"], "title": w.get("title", ""), "start": start, "end": w["end"],
                "in_progress": w["end"] is None, "wall_s": wall, "idle_s": idle, "review_s": review,
                "pause_s": max(0.0, pause - pause_idle), "pause_notes": [n for _, _, n in w["pauses"]],
                "images_attempted": 0, "images_ok": 0,
                "gen": defaultdict(lambda: {"attempts": 0, "ok": 0, "filtered": 0, "failed": 0, "seconds": 0.0,
                                            "billable_seconds": 0.0, "calls": 0, "tokens": 0, "tokens_api": 0,
                                            "in_tok": 0, "out_tok": 0}),
                "reshoots": Counter(), "tokens": 0, "tokens_api": 0, "cost": 0.0, "cost_by_model": Counter(),
            }
        for e in evs:
            kind = e.get("event")
            if kind not in ("image_generation", "video_generation", "music_generation", "llm_call"):
                continue
            sc = e.get("scene") or self.scene_of(parse_ts(e.get("ts")))
            if sc not in scenes:
                continue
            s = scenes[sc]
            model = e.get("model")
            g = s["gen"][model]
            g["attempts"] += 1
            st = e.get("status")
            if st == "completed":
                g["ok"] += 1
            elif st == "filtered":
                g["filtered"] += 1
            elif st in ("failed", "in_flight"):
                g["failed"] += 1
            tok = int(e.get("total_tokens") or 0)
            g["tokens"] += tok
            g["in_tok"] += int(e.get("input_tokens") or 0)
            g["out_tok"] += int(e.get("output_tokens") or 0)
            s["tokens"] += tok
            if e.get("tokens_source") == "api":
                g["tokens_api"] += tok
                s["tokens_api"] += tok
            if kind == "image_generation":
                s["images_attempted"] += 1
                s["images_ok"] += 1 if st == "completed" else 0
            if kind == "video_generation":
                secs = float(e.get("duration_s") or 0)
                g["seconds"] += secs if st == "completed" else 0
                if st == "completed" or (bill_filtered and st == "filtered"):
                    g["billable_seconds"] += secs
                if e.get("reshoot"):
                    s["reshoots"][e.get("reshoot_reason") or "retake"] += 1
            if kind in ("music_generation", "llm_call"):
                g["calls"] += int(e.get("api_calls") or 1) if st == "completed" else 0
            s["cost_by_model"][model] += self._cost(e, prices, bill_filtered)
        for s in scenes.values():
            s["cost"] = sum(s["cost_by_model"].values())
            s["production_s"] = max(0.0, s["wall_s"] - s["idle_s"] - s["pause_s"])
        return {"scenes": scenes, "order": order, "prices": prices}

    @staticmethod
    def _cost(e: dict[str, Any], prices: dict[str, Any], bill_filtered: bool) -> float:
        model = e.get("model")
        p = prices["models"].get(model, {})
        st = e.get("status")
        billable = st == "completed" or (bill_filtered and st == "filtered")
        if not billable:
            return 0.0
        kind = e.get("event")
        cost = 0.0
        if kind == "image_generation":
            cost += float(p.get("per_image", 0)) + float(p.get("per_ref_image", 0)) * int(e.get("ref_images") or 0)
        elif kind == "video_generation":
            if "per_1m_output_tokens" in p or "per_1m_input_tokens" in p:
                cost += int(e.get("input_tokens") or 0) / 1e6 * float(p.get("per_1m_input_tokens", 0))
                cost += (int(e.get("output_tokens") or 0) + int(e.get("thought_tokens") or 0)) / 1e6 * float(p.get("per_1m_output_tokens", 0))
            else:
                cost += float(e.get("duration_s") or 0) * float(p.get("per_second", 0))
                cost += float(p.get("per_call", 0))
        elif kind in ("music_generation", "llm_call"):
            cost += int(e.get("api_calls") or 1) * float(p.get("per_call", 0))
        return cost

    def report(self) -> str:
        data = self.compute()
        scenes, order, prices = data["scenes"], data["order"], data["prices"]
        cur = prices.get("currency", "USD")
        tot = Counter()
        tot_gen: dict[str, Counter] = defaultdict(Counter)
        reshoot_tot = Counter()
        for sc in order:
            s = scenes[sc]
            tot["wall_s"] += s["wall_s"]
            tot["idle_s"] += s["idle_s"]
            tot["review_s"] += s["review_s"]
            tot["pause_s"] += s["pause_s"]
            tot["production_s"] += s["production_s"]
            tot["tokens"] += s["tokens"]
            tot["tokens_api"] += s["tokens_api"]
            tot["images_attempted"] += s["images_attempted"]
            tot["images_ok"] += s["images_ok"]
            tot["cost"] += s["cost"]
            reshoot_tot.update(s["reshoots"])
            for m, g in s["gen"].items():
                for k, v in g.items():
                    tot_gen[m][k] += v
                tot_gen[m]["cost"] += s["cost_by_model"][m]
        vid_models = [m for m in MODELS if MODELS[m]["kind"] == "video"]
        videos_attempted = sum(tot_gen[m]["attempts"] for m in vid_models)
        videos_ok = sum(tot_gen[m]["ok"] for m in vid_models)
        videos_filtered = sum(tot_gen[m]["filtered"] for m in vid_models)
        reshoots = sum(reshoot_tot.values())
        conf = "✅ confirmed by the Director" if prices.get("confirmed_by_director") else "⚠️ PLACEHOLDER prices — not yet confirmed"

        L: list[str] = []
        L.append(f"# Production Ledger — `{self.project}`")
        L.append(f"_Generated {now().strftime('%d %b %Y, %H:%M IST')} · log file: `06_production_logs/production_ledger.jsonl` · unit prices: {conf}_\n")
        L.append("## 🎬 Project Totals")
        L.append("| Metric | Total |")
        L.append("| :--- | ---: |")
        L.append(f"| **Total time (wall-clock, all scenes)** | **{fmt_dur(tot['wall_s'])}** |")
        L.append(f"| ↳ Net production time (agent + rendering) | {fmt_dur(tot['production_s'])} |")
        L.append(f"| ↳ Waiting on the Director (reviews, approvals, time away) | {fmt_dur(tot['idle_s'])} |")
        if tot["pause_s"]:
            L.append(f"| ↳ Tooling pauses (e.g. this ledger setup) | {fmt_dur(tot['pause_s'])} |")
        L.append(f"| **Total tokens consumed (generation APIs)** | **{fmt_int(tot['tokens'])}** |")
        L.append(f"| ↳ reported live by the API / estimated for back-filled history | {fmt_int(tot['tokens_api'])} / {fmt_int(tot['tokens'] - tot['tokens_api'])} |")
        L.append(f"| **Total images generated** | **{fmt_int(tot['images_ok'])}** (of {fmt_int(tot['images_attempted'])} requests) |")
        L.append(f"| **Total videos generated** | **{fmt_int(videos_ok)}** (of {fmt_int(videos_attempted)} requests; {fmt_int(videos_filtered)} safety-blocked) |")
        L.append(f"| **Total video reshoots** | **{fmt_int(reshoots)}** ({_reshoot_text(reshoot_tot)}) |")
        L.append(f"| Lyria-002 scores | {fmt_int(tot_gen['lyria-002']['calls'])} |")
        L.append(f"| **Total cost** | **{money(tot['cost'], cur)}** |\n")

        L.append("## 🎞️ Scene-by-Scene")
        L.append("| Scene | Window (IST) | Wall-clock | Production | Waiting on Director | Images | Omni 1080p clips | Omni 360p SFX | Veo 3.1 | Lyria | Reshoots | Tokens | Cost |")
        L.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for sc in order:
            s = scenes[sc]
            g = s["gen"]
            win = f"{s['start'].astimezone(IST).strftime('%d %b %H:%M')} → " + (
                "*in progress*" if s["in_progress"] else s["end"].astimezone(IST).strftime("%d %b %H:%M"))
            title = f"**{sc}** {s['title']}".strip()
            L.append(
                f"| {title} | {win} | {fmt_dur(s['wall_s'])} | {fmt_dur(s['production_s'])} | {fmt_dur(s['idle_s'])} | "
                f"{s['images_ok']}/{s['images_attempted']} | {_va(g, 'omni-1.1-flash-1080p')} | {_va(g, 'omni-1.1-flash-360p-sfx')} | "
                f"{_va(g, 'veo-3.1')} | {g['lyria-002']['calls'] if 'lyria-002' in g else 0} | {sum(s['reshoots'].values())} | "
                f"{fmt_int(s['tokens'])} | {money(s['cost'], cur)} |")
        L.append(f"| **TOTAL** | | **{fmt_dur(tot['wall_s'])}** | **{fmt_dur(tot['production_s'])}** | **{fmt_dur(tot['idle_s'])}** | "
                 f"**{tot['images_ok']}/{tot['images_attempted']}** | **{_vt(tot_gen, 'omni-1.1-flash-1080p')}** | **{_vt(tot_gen, 'omni-1.1-flash-360p-sfx')}** | "
                 f"**{_vt(tot_gen, 'veo-3.1')}** | **{tot_gen['lyria-002']['calls']}** | **{reshoots}** | **{fmt_int(tot['tokens'])}** | **{money(tot['cost'], cur)}** |")
        L.append("\n_Clip cells read `delivered/requests`. Production = wall-clock − waiting on Director − tooling pauses._\n")

        L.append("## 🧮 Cost by Model (unit price × usage)")
        L.append("| Model | Requests | Delivered | Safety-blocked | Seconds / calls billed | Tokens | Unit price | Cost |")
        L.append("| :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: |")
        for m, meta in MODELS.items():
            g = tot_gen.get(m)
            if not g or not g["attempts"]:
                continue
            if meta["kind"] == "video":
                usage = f"{g['billable_seconds']:.0f} s"
            elif meta["kind"] == "image":
                usage = f"{g['ok']} images"
            else:
                usage = f"{g['calls']} calls"
            L.append(f"| {meta['label']} | {g['attempts']} | {g['ok']} | {g['filtered']} | {usage} | {fmt_int(g['tokens'])} | "
                     f"{_price_text(prices['models'].get(m, {}), cur)} | {money(g['cost'], cur)} |")
        L.append(f"| **Total** | | | | | **{fmt_int(tot['tokens'])}** | | **{money(tot['cost'], cur)}** |\n")

        L.append("## 🔁 Video Reshoots by Scene")
        L.append("| Scene | Safety-filter retries | Duplicate takes (parallel runs) | Re-takes of delivered clips | Total |")
        L.append("| :--- | ---: | ---: | ---: | ---: |")
        for sc in order:
            r = scenes[sc]["reshoots"]
            if sum(r.values()):
                L.append(f"| {sc} | {r['safety_retry']} | {r['duplicate_parallel_run']} | {r['retake']} | {sum(r.values())} |")
        L.append(f"| **Total** | {reshoot_tot['safety_retry']} | {reshoot_tot['duplicate_parallel_run']} | {reshoot_tot['retake']} | **{reshoots}** |\n")

        L.append("## 📝 How the numbers are computed")
        L.append("- **Time** comes from the conversation transcript and scene start/end markers. *Waiting on Director* = time between my hand-back and your next message (reviews, approvals, time away); background rendering counts as production time.")
        L.append("- **Images** = every `generate_image` request, matched to the file it produced (unmatched = failed/blocked, not charged).")
        L.append("- **Videos** = every submission recorded in the take `dispatch.jsonl` logs (incl. automatic safety re-prompts), plus the 360p SFX videos behind each `A3` stem.")
        L.append("- **Tokens**: Gemini Omni 1.1 returns real token usage — logged live from now on. Back-filled history uses Omni figures measured from the API (1080p ≈ 8,688 output tokens/s, 360p ≈ 1,931/s) and Google's published 1,120 output tokens per Nano Banana Pro image. Veo 3.1 and Lyria-002 do not report tokens (billed per second / per clip).")
        L.append("- **Not included**: the Antigravity agent's own reasoning tokens (not exposed to this workspace).")
        if prices.get("notes"):
            L.append(f"- **Pricing notes**: {prices['notes']}")
        L.append(f"- Safety-blocked attempts are {'**charged**' if prices.get('bill_filtered_attempts') else '**counted but not charged**'}.")
        md = "\n".join(L) + "\n"
        self.report_md.write_text(md, encoding="utf-8")

        with self.report_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["scene", "title", "start_ist", "end_ist", "wall_clock_h", "production_h", "director_idle_h",
                        "images_ok", "images_requested", "omni_1080p_ok", "omni_1080p_requests", "omni_360p_sfx_ok",
                        "veo_ok", "lyria_calls", "reshoots", "tokens", f"cost_{cur.lower()}"])
            for sc in order:
                s = scenes[sc]
                g = s["gen"]
                w.writerow([sc, s["title"], iso(s["start"]), iso(s["end"]) if s["end"] else "", round(s["wall_s"] / 3600, 2),
                            round(s["production_s"] / 3600, 2), round(s["idle_s"] / 3600, 2), s["images_ok"],
                            s["images_attempted"], g["omni-1.1-flash-1080p"]["ok"], g["omni-1.1-flash-1080p"]["attempts"],
                            g["omni-1.1-flash-360p-sfx"]["ok"], g["veo-3.1"]["ok"], g["lyria-002"]["calls"],
                            sum(s["reshoots"].values()), s["tokens"], round(s["cost"], 2)])
        summary = (f"[ledger] TOTAL time {fmt_dur(tot['wall_s'])} (production {fmt_dur(tot['production_s'])}, "
                   f"Director idle {fmt_dur(tot['idle_s'])}) | tokens {fmt_int(tot['tokens'])} | images {tot['images_ok']} | "
                   f"videos {videos_ok}/{videos_attempted} | reshoots {reshoots} | cost {money(tot['cost'], cur)}")
        print(summary)
        print(f"[ledger] report -> {self.report_md}")
        return md


# ----------------------------------------------------------------------------------------------
# small utilities
# ----------------------------------------------------------------------------------------------
def _tc_name_args(tc: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    name = tc.get("name") or (tc.get("function") or {}).get("name")
    args = tc.get("args") or tc.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    out: dict[str, Any] = {}
    for k, v in (args or {}).items():
        if isinstance(v, str):
            try:
                out[k] = json.loads(v)
            except json.JSONDecodeError:
                out[k] = v
        else:
            out[k] = v
    return name, out


def _ffprobe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                              "default=noprint_wrappers=1:nokey=1", str(path)], capture_output=True, text=True, timeout=30)
        return round(float(out.stdout.strip()), 2)
    except Exception:  # noqa: BLE001
        return None


def _wav_seconds(path: Path) -> float | None:
    try:
        import wave
        with wave.open(str(path), "rb") as w:
            return round(w.getnframes() / float(w.getframerate()), 2)
    except Exception:  # noqa: BLE001
        return None


def _overlap(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> float:
    lo, hi = max(a0, b0), min(a1, b1)
    return max(0.0, (hi - lo).total_seconds())


def _va(g: dict[str, Any], model: str) -> str:
    if model not in g or not g[model]["attempts"]:
        return "—"
    return f"{g[model]['ok']}/{g[model]['attempts']}"


def _vt(tot_gen: dict[str, Counter], model: str) -> str:
    g = tot_gen.get(model)
    if not g or not g["attempts"]:
        return "—"
    return f"{g['ok']}/{g['attempts']}"


def _price_text(p: dict[str, Any], cur: str) -> str:
    sym = "$" if cur.upper() == "USD" else cur + " "
    parts = []
    if "per_image" in p:
        parts.append(f"{sym}{p['per_image']}/image")
    if "per_second" in p:
        parts.append(f"{sym}{p['per_second']}/s")
    if "per_1m_input_tokens" in p:
        parts.append(f"{sym}{p['per_1m_input_tokens']}/1M in-tok")
    if "per_1m_output_tokens" in p:
        parts.append(f"{sym}{p['per_1m_output_tokens']}/1M out-tok")
    if "per_call" in p:
        parts.append(f"{sym}{p['per_call']}/call")
    return " + ".join(parts) or "—"


def _reshoot_text(c: Counter) -> str:
    return (f"{c['safety_retry']} safety retries · {c['duplicate_parallel_run']} duplicate takes · "
            f"{c['retake']} re-takes")


# ----------------------------------------------------------------------------------------------
# hook API used by pro_audio_director.py and the shoot scripts (never raises)
# ----------------------------------------------------------------------------------------------
def project_from_path(path: str | os.PathLike[str]) -> str | None:
    parts = Path(path).resolve().parts
    if "projects" in parts:
        i = parts.index("projects")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def safe_log(event: str, *, project: str | None = None, **fields: Any) -> None:
    try:
        proj = project or DEFAULT_PROJECT
        if not (ROOT / "projects" / proj).exists():
            return
        Ledger(proj).log_generation(event, **fields)
    except Exception as exc:  # noqa: BLE001 - logging must never break a shoot
        print(f"[ledger] warning: could not log {event}: {exc}")


def usage_dict(interaction: Any) -> dict[str, Any] | None:
    """Plain-dict copy of an Omni/Gemini interaction ``usage`` block (None when absent)."""
    def g(obj: Any, key: str) -> Any:
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    u = g(interaction, "usage")
    if not u:
        return None
    out: dict[str, Any] = {}
    for k in ("total_input_tokens", "total_output_tokens", "total_thought_tokens", "total_tokens", "total_cached_tokens"):
        v = g(u, k)
        if v is not None:
            out[k] = int(v)
    for k in ("input_tokens_by_modality", "output_tokens_by_modality"):
        v = g(u, k)
        if v:
            out[k] = {str(g(m, "modality")): int(g(m, "tokens") or 0) for m in v}
    return out or None


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sync")
    sub.add_parser("report")
    sc = sub.add_parser("scene")
    sc.add_argument("action", choices=["start", "end", "pause", "resume"])
    sc.add_argument("scene")
    sc.add_argument("--title")
    sc.add_argument("--note")
    sc.add_argument("--at", help="ISO timestamp (default: now)")
    pr = sub.add_parser("set-price")
    pr.add_argument("model", choices=list(MODELS))
    pr.add_argument("unit", choices=["per_image", "per_ref_image", "per_second", "per_call", "per_1m_input_tokens", "per_1m_output_tokens"])
    pr.add_argument("value", type=float)
    args = ap.parse_args(list(argv) if argv is not None else None)

    led = Ledger(args.project)
    if args.cmd == "sync":
        led.sync()
    elif args.cmd == "report":
        led.report()
    elif args.cmd == "scene":
        ok = led.scene_event(args.action, args.scene, title=args.title, note=args.note, at=args.at)
        print(f"[ledger] scene {args.action} {args.scene.upper()}: {'logged' if ok else 'already logged'}")
    elif args.cmd == "set-price":
        data = led.prices()
        data["models"].setdefault(args.model, {})[args.unit] = args.value
        led.prices_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[ledger] {args.model}.{args.unit} = {args.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
