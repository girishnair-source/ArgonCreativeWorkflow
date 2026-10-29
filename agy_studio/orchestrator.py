"""AGY orchestration engine — scene loop, gates, invalidation hooks, resume (BUILD_SPEC_v3 §9, §11, §12).

Antigravity (AGY, in chat) is the orchestrator named by the spec. This module is the deterministic engine AGY
drives through the CLI (``.venv/bin/python -m agy_studio ...``):

* ``Studio.advance(scene)`` runs every automatic step of the §11 scene loop until a human gate, a blocker or
  missing creative input is reached. It never waits: gates are persisted as ``awaiting:<gate>`` in
  ``state.json`` and AGY relays the rendered gate text to the human (one message per gate, one line per stage).
* ``Studio.respond(gate, scene, text)`` parses the human's reply (§9.3 format), records one Decision per item
  (steer/reject also become scoped Preferences) and applies the effects (selection, regen, anchors, locks).
* ``Studio.tick()`` advances every scheduled scene, polls Resolve round-trip events and works the regen queue,
  so the human can edit scene N in Resolve while scene N+1 is produced (acceptance 11).

Everything is resumable: all progress lives in the project folder; killing the process loses nothing and never
duplicates a dispatch (payload-hash dedupe in ``state.json`` + persisted attempt counters; acceptance 9).
Workstream modules are imported lazily so a fault in one area never takes down ``status``.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from . import continuity
from .fsio import (
    append_jsonl,
    generated_by,
    next_version_path,
    now_iso,
    read_record,
    save_versioned,
    stable_hash,
    validate_id,
)
from .models import (
    ApprovalStatus,
    ClipStatus,
    DecisionAction,
    DecisionTier,
    DispatchStatus,
    GateId,
    GenClip,
    GlobalStage,
    Ownership,
    Recommendation,
    Scene,
    SceneStage,
    SceneStatus,
    Take,
    TakeStatus,
    Tier,
)
from .project import PACK_ROLES, Project, find_project, parse_artifact_id
from .registry import Registry, default_registry
from .state import StateStore

S = SceneStage
AgentMode = Literal["llm", "manual"]

IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
MAX_IMPORT_BYTES = 40 * 1024 * 1024
MAX_AUTO_REGENS = 2  # automatic re-rolls after QA hard rejects / dispatch failures before asking the human
MAX_LINT_BOUNCES = 2
MAX_ACTIVE_SCENES = 3  # §12 parallelism bound (on top of per-scene budget caps and pack/voice contention)
REQUIRED_PACK_ROLES: dict[Tier, tuple[str, ...]] = {
    Tier.PRINCIPAL: ("front", "three_quarter", "profile", "rear", "face_closeup"),
    Tier.SUPPORTING: ("front", "three_quarter", "face_closeup"),
    Tier.INCIDENTAL: ("single_ref",),
}
ENV_ROLES = ("geometry", "reverse", "angle_45L", "angle_45R", "wide_establishing")
HARD_SFX_WORDS = ("gunshot", "gun shot", "explosion", "impact", "crash", "slam", "punch", "shatter", "bang", "thud")
ACTIVE_STAGES = frozenset({S.WORKING_SET, S.DESIGN, S.AWAITING_G2, S.DIRECTION, S.AWAITING_G3, S.DECOMPOSITION,
                           S.AUDIO_PLAN, S.AWAITING_G4, S.LINT, S.GENERATING, S.QA, S.AWAITING_G5, S.ASSEMBLY})
# §15 open decisions: chat key -> (ProjectSettings.open_decisions field, accepted values)
OPEN_DECISION_FIELDS: dict[str, tuple[str, dict[str, Any]]] = {
    "OD1": ("identity_conditioning", {"native": "native_refs", "native_refs": "native_refs", "refs": "native_refs",
                                      "lora": "lora"}),
    "OD2": ("incidental_conditioning", {"still": "single_still_if_line_or_gt2s",
                                        "single_still_if_line_or_gt2s": "single_still_if_line_or_gt2s",
                                        "text": "text_only", "text_only": "text_only"}),
    "OD3": ("stem_separation", {"on": True, "yes": True, "true": True, "off": False, "no": False, "false": False}),
    "OD4": ("voice_reference_audio", {"on": True, "yes": True, "true": True, "off": False, "no": False,
                                      "false": False}),
    "OD5": ("default_model_tossup", {"omni": "omni", "veo": "veo", "pilot": None, "none": None}),
    "OD6": ("roundtrip_transport", {"polling": "db_polling", "db_polling": "db_polling", "otio": "otio_export",
                                    "otio_export": "otio_export", "fcpxml": "otio_export"}),
    "OD7": ("orange_marker_default", {"on": True, "yes": True, "true": True, "off": False, "no": False,
                                      "false": False}),
}


class StudioError(RuntimeError):
    """Plain-language problem AGY relays to the human (never a stack trace)."""


@dataclass
class StepReport:
    """Chat-ready result of an engine step: one line per completed stage, the gate text, blockers, needs."""

    scene_id: str | None = None
    lines: list[str] = field(default_factory=list)
    gate_text: str | None = None
    blocked: list[str] = field(default_factory=list)
    needs: list[str] = field(default_factory=list)
    stage: str | None = None

    def text(self) -> str:
        out = list(self.lines)
        if self.gate_text:
            out.append(self.gate_text)
        out.extend(f"NEEDS: {n}" for n in dict.fromkeys(self.needs))
        out.extend(f"BLOCKED: {b}" for b in dict.fromkeys(self.blocked))
        return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------
def _pack_roles(pack: Any) -> dict[str, str | None]:
    t = pack.turnaround
    return {"front": t.front, "three_quarter": t.three_quarter, "profile": t.profile, "rear": t.rear,
            "face_closeup": pack.face_closeup, "expression_sheet": pack.expression_sheet,
            "single_ref": pack.single_ref}


def _env_roles(env: Any) -> dict[str, str | None]:
    d = env.derived_angles
    roles: dict[str, str | None] = {"geometry": env.geometry_still, "reverse": d.reverse, "angle_45L": d.angle_45L,
                                    "angle_45R": d.angle_45R, "wide_establishing": d.wide_establishing}
    for key, uri in d.inserts.items():
        roles[f"insert_{key}"] = uri
    return roles


def _in_library(uri: str | None) -> bool:
    return bool(uri) and str(uri).startswith("library/")


def norm_take_id(clip_id: str, value: str | None) -> str | None:
    """``T2`` / ``2`` / ``T02`` / full id -> ``<clip>_T02``."""
    if not value:
        return None
    v = value.strip().strip("\"'")
    if v.startswith(clip_id + "_"):
        return v
    m = re.fullmatch(r"[Tt]?(\d{1,3})", v)
    if m:
        return f"{clip_id}_T{int(m.group(1)):02d}"
    return v


def _strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


class Studio:
    """The engine behind every AGY chat command for one project."""

    def __init__(self, project: Project, *, registry: Registry | None = None, fake: bool | None = None,
                 agent_mode: AgentMode | None = None, transport: Any = None, db: Any = None,
                 user: str = "director", review_to_resolve: bool | None = None) -> None:
        from .budget import BudgetManager
        from .decisions import DecisionLog, PreferenceStore, QuestionLog
        from .gates import GateManager

        self.project = project
        self.registry = registry or default_registry()
        self.fake = fake
        mode = agent_mode or os.environ.get("AGY_AGENT_MODE", "llm")
        if mode not in ("llm", "manual"):
            raise StudioError(f"agent mode must be llm or manual, not {mode!r}")
        self.agent_mode: AgentMode = mode  # type: ignore[assignment]
        self.user = user
        self.state = StateStore(project)
        self.decisions = DecisionLog(project)
        self.prefs = PreferenceStore(project)
        self.questions = QuestionLog(project)
        self.gates = GateManager(project, self.state, self.decisions)
        self.budget = BudgetManager(project, self.registry, self.state)
        self._transport = transport
        self._db = db
        self._bridge: Any = None
        self.review_to_resolve = review_to_resolve

    @classmethod
    def open(cls, workspace: Path, project_id: str | None = None, **kw: Any) -> "Studio":
        return cls(find_project(Path(workspace), project_id), **kw)

    # ------------------------------------------------------------------------------------------
    # environment
    # ------------------------------------------------------------------------------------------
    @property
    def settings(self) -> Any:
        return self.project.config.settings

    def is_fake(self) -> bool:
        """True only for the unit-test double (``fake=True`` under pytest); raises outside pytest, never silent."""
        from .adapters.base import resolve_fake

        return resolve_fake(self.fake)

    def is_autopilot(self) -> bool:
        """``control_mode == autopilot``: G1 is approved once by the human, G2-G5 are answered by AGY itself."""
        return str(getattr(self.settings, "control_mode", "director") or "director").lower() == "autopilot"

    def _autopilot_approve(self, gate: GateId, sid: str, rep: StepReport, *, text: str = "approve all") -> bool:
        """Autopilot: the gate packet + review artifacts are still written (audit trail), then AGY answers it.

        Returns True so :meth:`advance` keeps stepping; the nested ``respond`` already moved the scene forward and
        its report text is folded into ``rep``.
        """
        rep.lines.append(f"{sid}: {gate.value} auto-approved (autopilot mode; review kept in reviews/)")
        rep.gate_text = None
        out = self.respond(gate.value, sid, text, user="autopilot", source="autopilot")
        rep.lines.extend(ln for ln in out.splitlines() if ln.strip())
        return True

    def _require_vertex_auth(self, *, force: bool = False) -> Any:
        """Fail-fast credential pre-flight for every real generation wave (skipped in fake / test mode).

        Runs ``gcloud auth print-access-token`` + project lookup once per process (``force=True`` re-checks) and
        raises :class:`~agy_studio.adapters.auth.AuthError` with gcloud's real error text and the fix. Nothing is
        planned, hashed or submitted before this passes.
        """
        from .adapters.auth import require_vertex_auth

        if self.is_fake():
            return None
        cached = getattr(self, "_auth_status", None)
        if cached is not None and not force:
            return cached
        status = require_vertex_auth()
        self._auth_status = status
        return status

    @property
    def db(self) -> Any:
        """Read path into Resolve (passive DB snapshot reads; never clicks menus to poll)."""
        if self._db is None:
            if self.is_fake():
                tr = self._get_transport()
                self._db = getattr(tr, "db", None) or (tr if hasattr(tr, "timeline") else None)
            else:
                from .resolve_db import ResolveDB, default_db_path

                s = self.settings
                path = Path(s.resolve_db_path) if s.resolve_db_path else default_db_path(
                    s.resolve_project_name or self.project.project_id)
                self._db = ResolveDB(path)
        return self._db

    def _get_transport(self) -> Any:
        if self._transport is None:
            from . import resolve_bridge as rb

            # Real mode: menu-click Lua bridge first, then the direct scripting API. Delivery into the editor is a
            # deliverable — a transport failure is surfaced with the manual one-click fallback, never swallowed.
            self._transport = (rb.FakeTransport() if self.is_fake()
                               else rb.FailoverTransport(rb.LuaMenuTransport(), rb.ScriptApiTransport()))
        return self._transport

    @property
    def bridge(self) -> Any:
        if self._bridge is None:
            from .resolve_bridge import ResolveBridge

            self._bridge = ResolveBridge(self.project, self._get_transport(), db=self.db)
        return self._bridge

    # ------------------------------------------------------------------------------------------
    # lookups
    # ------------------------------------------------------------------------------------------
    def scene(self, sid: str) -> Scene:
        return self.project.load_scene(sid)

    def schedule(self) -> list[str]:
        st = self.state.load()
        return list(st.schedule) or [s.id for s in self.project.list_scenes()]

    def scene_model(self, sid: str) -> str:
        sc = self.scene(sid)
        if sc.model_override:
            return sc.model_override
        if sc.breakdown and sc.breakdown.recommended_model:
            return sc.breakdown.recommended_model
        return self.settings.open_decisions.default_model_tossup or "omni"

    def display_name(self, entity_id: str) -> str:
        try:
            ent = self.project.load_entity(entity_id)
            return ent.display_name or entity_id
        except (FileNotFoundError, ValueError):
            return entity_id

    def active_clips(self, sid: str) -> list[GenClip]:
        """Clips of the current decomposition, in shot/clip order (stale-by-redecomposition clips excluded)."""
        out: list[GenClip] = []
        for shot in self.project.list_shots(sid):
            ids = list(shot.gen_clips) or [c.id for c in self.project.list_clips(shot.id)
                                           if c.status != ClipStatus.STALE]
            for cid in ids:
                try:
                    out.append(self.project.load_clip(cid))
                except FileNotFoundError:
                    continue
        return out

    @staticmethod
    def _needs_take(clip: GenClip) -> bool:
        return (not clip.selected_take_id) or clip.status in (ClipStatus.STALE, ClipStatus.NEEDS_RELINT)

    def _live_takes(self, cid: str) -> list[Take]:
        return [t for t in self.project.list_takes(cid) if t.status != TakeStatus.REJECTED
                and t.dispatch_status not in (DispatchStatus.FAILED, DispatchStatus.FILTERED)]

    def _candidates(self, cid: str) -> list[Take]:
        return [t for t in self.project.list_takes(cid) if t.status != TakeStatus.REJECTED
                and t.dispatch_status == DispatchStatus.COMPLETED]

    def frontier(self, sid: str) -> list[str]:
        """Clips that need a take and whose anchor clip (if any) already has a current selected take."""
        from .decomposer import plan_anchor_waves

        clips = self.active_clips(sid)
        by_id = {c.id: c for c in clips}
        order = [cid for wave in plan_anchor_waves(clips) for cid in wave]
        order += [c.id for c in clips if c.id not in order]
        ready: list[str] = []
        for cid in order:
            clip = by_id.get(cid)
            if clip is None or not self._needs_take(clip):
                continue
            anchor = clip.anchor_from_clip_id
            if anchor and anchor in by_id and self._needs_take(by_id[anchor]):
                continue
            ready.append(cid)
        return ready

    # ------------------------------------------------------------------------------------------
    # cursor-backed counters (persisted => identical seeds after a kill => payload dedupe)
    # ------------------------------------------------------------------------------------------
    def _cursor(self, key: str, default: Any = None) -> Any:
        return self.state.load().resume_cursor.get(key, default)

    def _attempt(self, cid: str) -> int:
        return int(self._cursor(f"attempt:{cid}", 0))

    def _bump_attempt(self, cid: str) -> int:
        n = self._attempt(cid) + 1
        self.state.set_cursor(f"attempt:{cid}", n)
        return n

    def _bump_failure(self, cid: str) -> int:
        n = int(self._cursor(f"failures:{cid}", 0)) + 1
        self.state.set_cursor(f"failures:{cid}", n)
        return n

    def _seed_base(self, cid: str, attempt: int) -> int | None:
        if attempt <= 0:
            return None  # shot generator default seeds (same as before a kill -> dedupe)
        return (int(stable_hash(cid, 8), 16) + attempt * 7919) % 2_000_000_000

    def _n_takes(self, cid: str) -> int:
        return int(self._cursor(f"ntakes:{cid}", 0)) or int(self.settings.takes_per_clip)

    # ------------------------------------------------------------------------------------------
    # questions / blockers
    # ------------------------------------------------------------------------------------------
    def ask(self, sid: str | None, kind: str, text: str, *, options: Iterable[str] = (),
            blocks: Iterable[str] = ()) -> Any:
        """Blocking question (§9.5), de-duplicated per scene + kind while it is open."""
        key = f"q:{sid or 'GLOBAL'}:{kind}"
        qid = self._cursor(key)
        if qid:
            q = next((q for q in self.questions.all() if q.id == qid), None)
            if q is not None and q.status == "open":
                return q
        stage = self.state.scene(sid).stage.value if sid and sid in self.state.load().scenes else None
        q = self.questions.ask(text, scene_id=sid, stage=stage, blocks=list(blocks), options=list(options))
        self.state.set_cursor(key, q.id)
        self.state.set_cursor(f"qkind:{q.id}", {"kind": kind, "scene_id": sid})
        return q

    def _block(self, sid: str, rep: StepReport, *blockers: str) -> bool:
        current = list(self.state.scene(sid).blocked_by)
        self.state.set_blocked(sid, current + list(blockers))
        rep.blocked.extend(blockers)
        return False

    def _auth_block(self, sid: str, rep: StepReport, exc: BaseException) -> bool:
        """Fail fast on a Google Cloud credential problem: show the full error NOW and block the scene on it.

        The blocker is ``question:<id> auth``; :meth:`_blocker_resolved` re-checks gcloud on every tick and clears
        it automatically once ``gcloud auth login`` has been run — nothing else is required from the human.
        """
        from .adapters.auth import describe_for_chat

        text = describe_for_chat(exc)
        q = self.ask(sid, "auth", text)
        rep.lines.append(text)
        return self._block(sid, rep, f"question:{q.id} auth")

    def _clear_blockers(self, sid: str, rep: StepReport) -> bool:
        blockers = list(self.state.scene(sid).blocked_by)
        if not blockers:
            return True
        remaining = [b for b in blockers if not self._blocker_resolved(sid, b)]
        if remaining != blockers:
            self.state.set_blocked(sid, remaining)
        if remaining:
            rep.blocked.extend(remaining)
            return False
        return True

    def _blocker_resolved(self, sid: str, blocker: str) -> bool:
        kind, _, rest = blocker.partition(":")
        if kind == "question":
            qid = rest.split(" ", 1)[0]
            q = next((q for q in self.questions.all() if q.id == qid), None)
            if q is not None and q.status == "open" and rest.endswith(" auth"):
                # Self-healing: once the human has run `gcloud auth login`, the next tick re-checks the credential
                # and clears the blocker by itself — no code change, no manual `answer`, no offline stand-in.
                from .adapters.auth import AuthError

                try:
                    status = self._require_vertex_auth(force=True)
                except AuthError:
                    return False
                self.questions.answer(qid, f"gcloud sign-in verified ({status.summary()})")
                return True
            return q is None or q.status == "answered"
        if kind == "pack":
            asset, _, _owner = rest.partition("@")
            return self._state_locked(asset) or self.state.owner(asset) in (None, sid)
        if kind == "voice":
            char, _, _owner = rest.partition("@")
            vp = self.project.load_voice_profile(char)
            return bool(vp and vp.anchor_take) or self.state.owner(self._voice_key(char)) in (None, sid)
        if kind == "budget":
            try:
                return bool(self.budget.check(sid, float(rest or 0)).ok)
            except ValueError:
                return False
        return False  # resolve:/dispatch: blockers are cleared explicitly (`unblock SC`) after AGY fixes the cause

    def unblock(self, sid: str) -> str:
        validate_id(sid, "scene id")
        before = list(self.state.scene(sid).blocked_by)
        self.state.set_blocked(sid, [])
        return f"{sid}: cleared blockers {before}" if before else f"{sid}: nothing was blocking"

    # ------------------------------------------------------------------------------------------
    # gates
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _gate_key(gate: GateId | str, scene_id: str | None) -> str:
        return f"gate:{GateId(gate).value}:{scene_id or 'GLOBAL'}"

    def _open_gate(self, packet: Any, media: Iterable[str] = ()) -> str:
        text = self.gates.open(packet)
        self.state.set_cursor(self._gate_key(packet.gate, packet.scene_id),
                              {"item_ids": [it.item_id for it in packet.items], "opened_at": now_iso(),
                               "title": packet.title})
        deliver = self.review_to_resolve if self.review_to_resolve is not None else self.is_fake()
        media = [m for m in media if m]
        if deliver and packet.scene_id and media and GateId(packet.gate) in (GateId.G3, GateId.G5):
            try:
                where = self.bridge.deliver_review(packet.scene_id, GateId(packet.gate).value, media)
                text += f"\nSame media in Resolve: {where}"
            except Exception as exc:  # noqa: BLE001 — review copy in Resolve is a convenience, never fatal
                text += f"\n(Resolve review bin not updated: {exc})"
        return text

    def _blocks(self, sid: str, what: str) -> list[str]:
        sched = self.schedule()
        out = [f"{sid} {what}"]
        if sid in sched and sched.index(sid) + 1 < len(sched):
            out.append(f"{sched[sched.index(sid) + 1]} entry")
        return out

    def _risk(self, sid: str) -> str | None:
        sc = self.scene(sid)
        return sc.breakdown.risk_tier.value if sc.breakdown else None

    @staticmethod
    def _decision_for(decisions: list[Any], item_id: str) -> str | None:
        for d in decisions:
            if d.target_artifact_id == item_id:
                return d.id
        return decisions[-1].id if decisions else None

    @staticmethod
    def _review_media(review_dir: Path | str | None, needle: str | None = None) -> list[str]:
        if not review_dir:
            return []
        manifest = Path(review_dir) / "manifest.yaml"
        if not manifest.exists():
            return []
        try:
            data = read_record(manifest)
        except Exception:  # noqa: BLE001
            return []
        paths = [s for s in _strings(data) if s.startswith("/") and Path(s).suffix.lower() in
                 (".png", ".jpg", ".mp4", ".mov", ".wav")]
        if needle:
            hits = [p for p in paths if needle in p]
            sheets = [p for p in paths if "contact" in Path(p).name.lower()]
            return list(dict.fromkeys(hits + sheets))
        return list(dict.fromkeys(paths))

    def respond(self, gate: str, scene_id: str | None, text: str, *, user: str | None = None,
                source: str = "chat") -> str:
        """Apply the human's reply to an open gate (§9.3 format); returns AGY's one-line confirmation + effects."""
        from .gates import parse_gate_response

        gate_id = GateId(gate.upper())
        if gate_id == GateId.G6:
            if not scene_id:
                raise StudioError("G6 needs a scene, e.g. `approve SC01`.")
            return self.approve_scene(scene_id, source=source, user=user)
        if gate_id in (GateId.G1, GateId.G7):
            scene_id = None
        status = self.state.gate_status(scene_id, gate_id)
        if status != "awaiting":
            where = f" · {scene_id}" if scene_id else ""
            raise StudioError(f"{gate_id.value}{where} is not waiting for you (status: {status}).")
        cur = self._cursor(self._gate_key(gate_id, scene_id)) or {}
        response = parse_gate_response(gate_id, text, item_ids=tuple(cur.get("item_ids", ())))
        decisions = self.gates.close(gate_id, scene_id, response, user=user or self.user, source=source)
        handler = {GateId.G1: self._apply_g1, GateId.G2: self._apply_g2, GateId.G3: self._apply_g3,
                   GateId.G4: self._apply_g4, GateId.G5: self._apply_g5, GateId.G7: self._apply_g7}[gate_id]
        lines = [f"Got it — {response.interpretation or self._interpret(response)} "
                 f"({len(decisions)} decision(s) recorded)."]
        lines += handler(scene_id, response, decisions)
        return "\n".join(x for x in lines if x)

    @staticmethod
    def _interpret(response: Any) -> str:
        if response.approve_all:
            head = "approve all"
        elif response.reject_all:
            head = "reject all"
        else:
            head = ""
        parts = []
        for ch in response.choices:
            bit = f"{ch.item_id}:{ch.action}"
            if ch.value:
                bit += f" {ch.value}"
            if ch.steer:
                bit += f' "{ch.steer}"'
            parts.append(bit)
        parts += [f"anchor {c}={t}" for c, t in response.anchors.items()]
        body = ", ".join(x for x in [head, *parts] if x) or "no item choices"
        return body + (f"; note: {response.free_text}" if response.free_text else "")

    # ------------------------------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------------------------------
    def status(self, sid: str | None = None) -> str:
        st = self.state.load()
        lines = [f"{self.project.project_id} · {self.project.config.title} · {st.global_stage.value}"
                 + (" · PAUSED" if st.paused else "")]
        if st.gates:
            lines.append("global gates: " + ", ".join(f"{k}={v}" for k, v in sorted(st.gates.items())))
        for s_id in ([sid] if sid else self.schedule()):
            sc = st.scenes.get(s_id)
            if sc is None:
                lines.append(f"{s_id}: not scheduled yet")
                continue
            parts = [f"{s_id}: {sc.stage.value}"]
            if sc.awaiting:
                parts.append(sc.awaiting)
            clips = self.active_clips(s_id)
            if clips:
                parts.append(f"clips {sum(1 for c in clips if not self._needs_take(c))}/{len(clips)} selected")
            cap = f" of {sc.budget_cap:.2f}" if sc.budget_cap is not None else ""
            parts.append(f"spend {sc.budget_spent:.2f}{cap}")
            handoff = self.project.load_handoff(s_id)
            if handoff:
                parts.append(f"timeline {handoff.timeline_name} ({handoff.ownership.value})")
            if sc.paused:
                parts.append("paused")
            if sc.blocked_by:
                parts.append("blocked: " + "; ".join(sc.blocked_by))
            lines.append(" · ".join(parts))
        if st.regen_queue:
            lines.append("regen queue: " + ", ".join(str(r.get("clip_id")) for r in st.regen_queue))
        open_q = self.questions.open()
        if open_q:
            lines.append(f"open questions: {len(open_q)} (`questions`)")
        pending = self.gates.pending()
        if pending:
            lines.append("awaiting you: " + ", ".join(
                f"{p.get('gate', '?')} {p.get('scene_id') or ''}".strip() for p in pending))
        return "\n".join(lines)

    # ------------------------------------------------------------------------------------------
    # scene loop driver
    # ------------------------------------------------------------------------------------------
    def _handlers(self) -> dict[SceneStage, Any]:
        return {S.PENDING: self._h_pending, S.WORKING_SET: self._h_working_set, S.DESIGN: self._h_design,
                S.DIRECTION: self._h_direction, S.DECOMPOSITION: self._h_decomposition,
                S.AUDIO_PLAN: self._h_audio_plan, S.LINT: self._h_lint, S.GENERATING: self._h_generating,
                S.QA: self._h_qa, S.ASSEMBLY: self._h_assembly}

    def advance(self, sid: str, *, max_steps: int = 40) -> StepReport:
        """Run automatic steps for one scene until a gate, a blocker or missing creative input."""
        validate_id(sid, "scene id")
        rep = StepReport(scene_id=sid)
        self.state.ensure_scene(sid)
        handlers = self._handlers()
        from .adapters.auth import AuthError

        for _ in range(max_steps):
            if self.state.is_paused(sid):
                rep.blocked.append(f"{sid} is paused (`resume {sid}`)")
                break
            if not self._clear_blockers(sid, rep):
                break
            handler = handlers.get(self.state.scene(sid).stage)
            if handler is None:
                break
            try:
                keep_going = handler(sid, rep)
            except AuthError as exc:  # any stage (G2 images, G3 boards, G5 video...): fail fast, same blocker
                self._auth_block(sid, rep, exc)
                break
            if not keep_going:
                break
        rep.stage = self.state.scene(sid).stage.value
        self._refresh_dashboard()
        return rep

    def _refresh_dashboard(self) -> None:
        """Rewrite ``DASHBOARD.md`` (status, time, tokens, ETA, risks, outputs, QA failures). Never raises."""
        try:
            from .dashboard import write_dashboard  # noqa: PLC0415

            write_dashboard(self.project)
        except Exception:  # noqa: BLE001 - the dashboard is a view; it must never break the pipeline
            pass

    def tick(self, *, poll: bool = True) -> str:
        """One orchestration pass over the whole project (round-trip poll, regen queue, every scene)."""
        lines: list[str] = []
        st = self.state.load()
        if poll and st.global_stage not in (GlobalStage.PREPRODUCTION, GlobalStage.AWAITING_G1):
            try:
                lines += self.poll()
            except Exception as exc:  # noqa: BLE001 — Resolve may simply be closed
                lines.append(f"(Resolve round-trip skipped: {exc})")
        lines += self.process_regens()
        if st.global_stage in (GlobalStage.PREPRODUCTION, GlobalStage.AWAITING_G1):
            lines.append("Waiting for G1 (Bible, StyleLock, voice profiles, schedule) before any scene starts.")
            return "\n".join(lines)
        active = sum(1 for sc in st.scenes.values() if sc.stage in ACTIVE_STAGES)
        for sid in self.schedule():
            before = self.state.load().scenes.get(sid)
            if (before is None or before.stage == S.PENDING) and active >= MAX_ACTIVE_SCENES:
                continue
            rep = self.advance(sid)
            if (before is None or before.stage == S.PENDING) and rep.stage != S.PENDING.value:
                active += 1
            text = rep.text()
            if text:
                lines.append(text)
        self._refresh_dashboard()
        return "\n".join(lines) or "Nothing to do right now."

    # ---- PENDING / WORKING_SET ---------------------------------------------------------------
    def _h_pending(self, sid: str, rep: StepReport) -> bool:
        if self.state.gate_status(None, GateId.G1) != "passed":
            rep.blocked.append("waiting for G1 (Bible, StyleLock, voice profiles, schedule)")
            return False
        self.state.transition(sid, S.WORKING_SET, reason="G1 passed; scene scheduled")
        return True

    def _scene_state_ids(self, sid: str) -> list[str]:
        return continuity.all_scene_states(self.scene(sid))

    def _state_locked(self, state_id: str) -> bool:
        pair = self.project.entity_state_index().get(state_id)
        if pair is None:
            return False
        st = pair[1]
        if st.approval_status != ApprovalStatus.LOCKED:  # type: ignore[attr-defined]
            return False
        pack = st.reference_pack  # type: ignore[attr-defined]
        if pack.text_only:
            return True
        uris = pack.image_uris()
        return bool(uris) and all(_in_library(u) for u in uris)

    @staticmethod
    def _env_locked(env: Any) -> bool:
        return env.approval_status == ApprovalStatus.LOCKED and _in_library(env.geometry_still)

    def _h_working_set(self, sid: str, rep: StepReport) -> bool:
        from .bible_hierarchy import build_scene_bible
        from .working_set import build_working_set

        blockers = []
        for state_id in self._scene_state_ids(sid):
            if not self._state_locked(state_id):
                owner = self.state.claim(state_id, sid)
                if owner != sid:
                    blockers.append(f"pack:{state_id}@{owner}")
        if blockers:  # §12: first-appearance scene owns generation; others wait
            return self._block(sid, rep, *blockers)
        ws = build_working_set(self.project, sid)
        try:
            build_scene_bible(self.project, sid)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.budget.ensure_cap(sid)
        except Exception as exc:  # noqa: BLE001
            rep.lines.append(f"{sid}: budget cap not set yet ({exc})")
        self.state.transition(sid, S.DESIGN, reason=f"working set {ws.working_set_hash}")
        rep.lines.append(f"{sid}: working set built (hash {ws.working_set_hash})")
        return True

    # ---- DESIGN / G2 -------------------------------------------------------------------------
    def design_delta(self, sid: str) -> list[str]:
        """Entity states + environment of this scene that are not locked in library/ yet (delta only)."""
        out = [s for s in self._scene_state_ids(sid) if not self._state_locked(s)]
        scene = self.scene(sid)
        if scene.environment_id:
            env = self.project.load_environment(scene.environment_id)
            if not self._env_locked(env):
                out.append(env.id)
        return out

    def _design_manifest_path(self, sid: str) -> Path:
        return self.project.path("scenes", sid, "design", "candidates.yaml")

    def _design_manifest(self, sid: str) -> dict[str, dict[str, str]]:
        p = self._design_manifest_path(sid)
        if not p.exists():
            return {}
        data = read_record(p) or {}
        return {k: dict(v) for k, v in (data.get("candidates") or {}).items()}

    def _save_design_manifest(self, sid: str, cands: dict[str, dict[str, str]]) -> None:
        save_versioned(self._design_manifest_path(sid),
                       {"candidates": cands, "generated_by": generated_by("agy.design").model_dump(mode="json")})

    def _required_roles(self, item: str) -> tuple[str, ...]:
        pair = self.project.entity_state_index().get(item)
        if pair is not None:
            ent, st = pair
            if st.reference_pack.text_only:  # type: ignore[attr-defined]
                return ()
            if ent.type.value == "prop":
                return ("front", "three_quarter") if ent.tier == Tier.PRINCIPAL else ("single_ref",)
            return REQUIRED_PACK_ROLES.get(ent.tier, ("single_ref",))
        return ("geometry",)

    def design_candidates(self, sid: str) -> dict[str, dict[str, str]]:
        """item id -> {role: project-relative path} staged for G2 (manifest + proposed records outside library/)."""
        manifest = self._design_manifest(sid)
        idx = self.project.entity_state_index()
        out: dict[str, dict[str, str]] = {}
        for item in self.design_delta(sid):
            files = {r: u for r, u in manifest.get(item, {}).items() if self.project.abs(u).is_file()}
            if item in idx:
                roles = _pack_roles(idx[item][1].reference_pack)  # type: ignore[attr-defined]
            else:
                roles = _env_roles(self.project.load_environment(item))
            for role, uri in roles.items():
                if uri and not _in_library(uri) and self.project.abs(uri).is_file():
                    files.setdefault(role, uri)
            if files:
                out[item] = files
        return out

    def design_missing(self, sid: str) -> dict[str, list[str]]:
        cands = self.design_candidates(sid)
        missing: dict[str, list[str]] = {}
        for item in self.design_delta(sid):
            need = [r for r in self._required_roles(item) if r not in cands.get(item, {})]
            if need:
                missing[item] = need
        return missing

    def _h_design(self, sid: str, rep: StepReport) -> bool:
        from .gates import GateItem, GatePacket
        from .review import build_g2_review

        delta = self.design_delta(sid)
        if not delta:
            self.gates.auto_pass(GateId.G2, sid, reason="nothing new: every pack and the geometry are locked")
            self.state.transition(sid, S.DIRECTION, reason="G2 auto-pass (nothing new)")
            rep.lines.append(f"{sid}: G2 auto-passed (nothing new to design)")
            return True
        missing = self.design_missing(sid)
        if missing:  # the Director's own sheets (bootstrap_intake.py -> library/imports/<Name>/) come first, in every mode
            imported = self.auto_import_reference_sheets(sid)
            if imported:
                rep.lines.extend(imported)
                missing = self.design_missing(sid)
        if missing and self.is_autopilot():  # nobody to ask: generate the missing reference images right away
            rep.lines.append(self.design_generate(sid))
            missing = self.design_missing(sid)
        if missing:  # workspace token-saving gate: ask for existing reference sheets before generating
            listing = "; ".join(f"{self.display_name(k.split('_STATE_')[0])} [{k}] needs {', '.join(v)}"
                                for k, v in missing.items())
            q = self.ask(sid, "design",
                         f"{sid} needs reference images before anything is generated: {listing}. Do you already "
                         f"have character reference sheets or location images for these? If yes, drop them in "
                         f"the project folder and tell me the file names so I reuse them and save tokens; "
                         f"otherwise say 'generate' and I will create new ones with Nano Banana Pro.",
                         options=("generate", "I have them"))
            rep.needs.append(f"{sid}: reference images — `design {sid} import <ITEM> role=<file> ...` or "
                             f"`design {sid} generate` after the editor answers {q.id}")
            return self._block(sid, rep, f"question:{q.id} reference sheets")
        cands = self.design_candidates(sid)
        review_dir = build_g2_review(self.project, sid, {k: [str(self.project.abs(u)) for u in v.values()]
                                                         for k, v in cands.items()})
        items = []
        for item, files in cands.items():
            what = "environment geometry + angles" if item not in self.project.entity_state_index() else \
                f"reference pack ({', '.join(sorted(files))})"
            items.append(GateItem(item_id=item, summary=f"{self.display_name(item.split('_STATE_')[0])}: {what}",
                                  options=["ok", 'regen "<what to change>"'],
                                  media=[str(self.project.abs(u)) for u in files.values()]))
        packet = GatePacket(gate=GateId.G2, scene_id=sid, title="New packs & geometry", risk=self._risk(sid),
                            model=self._image_model_id(), blocks=self._blocks(sid, "direction"),
                            review_dir=str(review_dir), resolve_bin=f"/{self.project.project_id}/00_BIBLE",
                            items=items,
                            expected_format='CHAR_X_STATE_Y:ok, ENV_Z:regen "<what to change>" — or `approve all`',
                            defaults="Vision tier — never auto-proceeds; silence means AGY waits.")
        rep.gate_text = self._open_gate(packet)
        self.state.transition(sid, S.AWAITING_G2, reason=f"{len(items)} new design item(s) staged")
        if self.is_autopilot():
            return self._autopilot_approve(GateId.G2, sid, rep)
        return False

    def _image_model_id(self) -> str:
        try:
            return self.registry.first_of_kind("image").id
        except Exception:  # noqa: BLE001
            return "nano_banana_pro"

    def _image_adapter(self) -> Any:
        return self.registry.adapter(self._image_model_id(), project=self.project, fake=self.fake)

    def auto_import_reference_sheets(self, sid: str) -> list[str]:
        """Stage the Director's own reference images from ``library/imports/<Name>/`` for the pending design items.

        ``<Name>`` is matched (case/space-insensitive) against entity ids, display names and environment ids. The first
        image becomes ``front`` (``geometry`` for an environment), the second ``three_quarter``, further files
        ``insert_<n>``. Missing views are then generated FROM these sheets (``design_generate`` passes staged images as
        references) — never from scratch. Returns one line per staged item; nothing when there are no imports.
        """
        imports_dir = self.project.path("library", "imports")
        if not imports_dir.is_dir():
            return []
        folders = {self._norm_name(d.name): d for d in imports_dir.iterdir() if d.is_dir()}
        if not folders:
            return []
        idx = self.project.entity_state_index()
        manifest = self._design_manifest(sid)
        lines: list[str] = []
        for item in self.design_delta(sid):
            if manifest.get(item):
                continue  # already staged (earlier scene or manual import)
            keys: list[str] = [self._norm_name(item)]
            if item in idx:
                ent = idx[item][0]
                keys += [self._norm_name(ent.id), self._norm_name(ent.id.removeprefix("CHAR_").removeprefix("PROP_")),
                         self._norm_name(getattr(ent, "display_name", None) or ""), self._norm_name(item.split("_STATE_")[0])]
            else:
                keys += [self._norm_name(item.removeprefix("ENV_"))]
            folder = next((folders[k] for k in keys if k and k in folders), None)
            if folder is None:
                continue
            files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            if not files:
                continue
            roles = list(self._required_roles(item)) or (["geometry"] if item not in idx else ["single_ref"])
            mapping: dict[str, str | Path] = {}
            for i, f in enumerate(files):
                role = roles[i] if i < len(roles) else f"insert_{i - len(roles) + 1}"
                mapping[role] = f
            try:
                self.design_import(sid, item, mapping)
            except StudioError as exc:
                lines.append(f"{item}: could not use {folder.name}/ ({exc})")
                continue
            lines.append(f"{item}: using the Director's {len(mapping)} reference image(s) from library/imports/{folder.name}/ "
                         f"({', '.join(mapping)})")
        return lines

    @staticmethod
    def _norm_name(text: str) -> str:
        return "".join(ch for ch in str(text).lower() if ch.isalnum())

    def design_import(self, sid: str, item: str, files: dict[str, str | Path]) -> str:
        """Copy the editor's existing reference images INTO the project (never referenced in place)."""
        validate_id(sid, "scene id")
        validate_id(item, "design item id")
        if item not in self.design_delta(sid):
            raise StudioError(f"{item} is not a pending design item of {sid} (pending: {self.design_delta(sid)})")
        allowed = set(PACK_ROLES) | set(ENV_ROLES) if item in self.project.entity_state_index() else set(ENV_ROLES)
        manifest = self._design_manifest(sid)
        staged = dict(manifest.get(item, {}))
        dest_dir = self.project.path("scenes", sid, "design", "imports", item)
        for role, src in files.items():
            if role not in allowed and not role.startswith("insert_"):
                raise StudioError(f"unknown role {role!r} for {item}; use one of {sorted(allowed)}")
            src_path = Path(src).expanduser()
            if not src_path.is_absolute():
                src_path = (self.project.root / src_path)
            src_path = src_path.resolve()
            if not src_path.is_file() or src_path.is_symlink():
                raise StudioError(f"{src} is not a regular file")
            if src_path.suffix.lower() not in IMAGE_EXTS:
                raise StudioError(f"{src_path.name}: only {sorted(IMAGE_EXTS)} images can be imported")
            if src_path.stat().st_size > MAX_IMPORT_BYTES:
                raise StudioError(f"{src_path.name} is larger than {MAX_IMPORT_BYTES // (1024 * 1024)} MB")
            dest = next_version_path(dest_dir, role, src_path.suffix.lower())
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest)
            staged[role] = self.project.rel(dest)
        manifest[item] = staged
        self._save_design_manifest(sid, manifest)
        self._answer_open(sid, "design", f"editor supplied reference images for {item}")
        return f"{sid}: imported {len(files)} reference image(s) for {item} (staged for G2; nothing generated)."

    def design_generate(self, sid: str, items: Iterable[str] | None = None, *, steer: str = "") -> str:
        """Generate missing reference images with the image model (only after the editor said 'generate')."""
        missing = self.design_missing(sid)
        targets = [i for i in (items or missing.keys()) if i in self.design_delta(sid)]
        if not targets:
            return f"{sid}: nothing to generate."
        img = self._image_adapter()
        style = self.project.load_style_lock()
        idx = self.project.entity_state_index()
        manifest = self._design_manifest(sid)
        made = 0
        for item in targets:
            staged = dict(manifest.get(item, {}))
            roles = missing.get(item) or list(self._required_roles(item))
            out_dir = self.project.path("scenes", sid, "design", "generated", item)
            out_dir.mkdir(parents=True, exist_ok=True)
            if item in idx:
                ent, st = idx[item]
                refs = self._derived_refs(st, staged)
                base = (f"{ent.canonical_description}. {st.delta_description}".strip(". ")  # type: ignore[attr-defined]
                        + ". Neutral grey studio backdrop, even soft studio lighting, full body unless noted.")
            else:
                env = self.project.load_environment(item)
                refs = []
                base = f"{env.canonical_description}. Empty location plate, no people."
            if style:
                base += f" Medium: {style.medium}. Lens language: {style.lens_language}."
            if steer:
                base += f" Direction note: {steer}."
            for role in roles:
                prompt = f"{base} View: {role.replace('_', ' ')}."
                dest = next_version_path(out_dir, role, ".png")
                result = img.generate(prompt, dest, reference_images=[str(self.project.abs(r)) for r in refs],
                                      aspect_ratio="3:4" if item in idx else self.settings.aspect_ratio)
                path = Path(result) if result else dest
                staged[role] = self.project.rel(path)
                made += 1
            manifest[item] = staged
        self._save_design_manifest(sid, manifest)
        self._answer_open(sid, "design", "generate")
        return f"{sid}: generated {made} reference image(s) for {', '.join(targets)} (staged for G2)."

    def _derived_refs(self, st: Any, staged: dict[str, str] | None = None) -> list[str]:
        """Already-imported images for this state + locked pack images of the state this one derives from."""
        own = list((staged or {}).values()) + list(st.reference_pack.image_uris())  # type: ignore[attr-defined]
        out: list[str] = [u for u in dict.fromkeys(own) if self.project.abs(u).is_file()]
        if getattr(st, "derived_from", None):
            pair = self.project.entity_state_index().get(st.derived_from)
            if pair is not None and self._state_locked(st.derived_from):
                for u in pair[1].reference_pack.image_uris()[:3]:  # type: ignore[attr-defined]
                    if u not in out and self.project.abs(u).is_file():
                        out.append(u)
        return out[:5]

    def _answer_open(self, sid: str | None, kind: str, answer: str, decision_id: str | None = None) -> None:
        qid = self._cursor(f"q:{sid or 'GLOBAL'}:{kind}")
        if not qid:
            return
        q = next((q for q in self.questions.all() if q.id == qid), None)
        if q is not None and q.status == "open":
            self.questions.answer(qid, answer, decision_id=decision_id)

    def _promote_design(self, sid: str, item: str, files: dict[str, str], decision_id: str | None) -> str:
        idx = self.project.entity_state_index()
        prov = generated_by("agy.g2", notes=f"approved by decision {decision_id}")
        if item in idx:
            ent, st = idx[item]
            abs_files = {r: self.project.abs(u) for r, u in files.items() if r in PACK_ROLES}
            pack = self.project.promote_pack(item, abs_files, approved_by=self.user, provenance=prov)
            for s in ent.states:
                if s.id == item:
                    s.reference_pack = pack
                    s.approval_status = ApprovalStatus.LOCKED
            self.project.save_entity(ent, confirm_decision_id=decision_id if ent.approval_status ==
                                     ApprovalStatus.LOCKED else None)
            self.state.release(item)
            return f"{item}: pack locked in library/ ({len(abs_files)} image(s))"
        env = self.project.load_environment(item)
        named = {f"{r}{Path(u).suffix.lower()}": self.project.abs(u) for r, u in files.items()}
        rels = self.project.promote_library_files(("environments", item), named, provenance=prov)
        by_role = {Path(name).stem: rel for name, rel in rels.items()}
        updates = dict(env.derived_angles.model_dump())
        for role in ("reverse", "angle_45L", "angle_45R", "wide_establishing"):
            if role in by_role:
                updates[role] = by_role[role]
        for role, rel in by_role.items():
            if role.startswith("insert_"):
                updates.setdefault("inserts", {})[role[len("insert_"):]] = rel
        was_locked = env.approval_status == ApprovalStatus.LOCKED
        env.derived_angles = type(env.derived_angles).model_validate(updates)
        if "geometry" in by_role:
            env.geometry_still = by_role["geometry"]
            cands = [c for c in env.geometry_candidates if c != by_role["geometry"]]
            env.geometry_candidates = ([by_role["geometry"]] + cands)[:4]
        env.approval_status = ApprovalStatus.LOCKED
        self.project.save_environment(env, confirm_decision_id=decision_id if was_locked else None)
        set_note = self._ensure_set3d(item)
        return f"{item}: geometry + angles locked in library/{set_note}"

    def _ensure_set3d(self, env_id: str) -> str:
        """3D world building is OPT-IN in every mode — it is never started by a gate.

        A grey-box set costs an LLM design pass, a Blender export and (if rendered) a Cycles render per camera, and it
        only pays off when a human wants to walk the room and frame cameras. Locking an environment at G2 therefore
        just reports how to build it on demand (``set3d build <ENV>`` then ``set3d open-blender <ENV>``); when a set
        already exists its Blender export is refreshed so the .blend stays in sync with the locked geometry.
        """
        from .set3d import store
        from .set3d.blender_export import export_blender_scene

        if store.has_set(self.project, env_id):
            try:
                export_blender_scene(self.project, env_id)
            except Exception:  # noqa: BLE001
                pass
            return (f" (3D set at bible/sets/{env_id}/set.json — `set3d open-blender {env_id}` to view it in Blender, "
                    f"`set3d serve {env_id}` for the browser FPV)")
        return (f" (no 3D set — optional: `set3d build {env_id}` then `set3d open-blender {env_id}` if you want to "
                f"walk the room and frame cameras)")

    def _apply_g2(self, sid: str | None, response: Any, decisions: list[Any]) -> list[str]:
        from .bible_hierarchy import build_global_context_bible, build_scene_bible

        assert sid
        cands = self.design_candidates(sid)
        explicit = {c.item_id: c for c in response.choices}
        approve, regen = [], {}
        for item in cands:
            ch = explicit.get(item)
            if ch is not None and ch.action in ("regen", "reject", "alternatives"):
                regen[item] = ch.steer or ch.value or response.free_text
            elif ch is not None or response.approve_all:
                approve.append(item)
        if response.reject_all:
            regen.update({i: response.free_text for i in cands if i not in explicit})
            approve = [i for i in approve if i in explicit]
        lines = [self._promote_design(sid, item, cands[item], self._decision_for(decisions, item)) for item in approve]
        if regen or self.design_delta(sid):
            self.state.transition(sid, S.DESIGN, reason="G2: some design items need another pass")
            for item, steer in regen.items():
                lines.append(f"{item}: redesign requested — {steer or 'no note'}")
                self.state.set_cursor(f"design_steer:{sid}:{item}", steer or "")
            if regen:
                manifest = self._design_manifest(sid)
                for item in regen:
                    manifest.pop(item, None)
                self._save_design_manifest(sid, manifest)
            lines.append(self.advance(sid).text())
            return lines
        try:
            build_scene_bible(self.project, sid)
            build_global_context_bible(self.project)
        except Exception:  # noqa: BLE001
            pass
        self.state.transition(sid, S.DIRECTION, reason="G2 approved")
        lines.append(self.advance(sid).text())
        return lines

    # ---- DIRECTION / G3 ----------------------------------------------------------------------
    def _h_direction(self, sid: str, rep: StepReport) -> bool:
        from .gates import GateItem, GatePacket
        from .review import build_g3_review

        shots = self.project.list_shots(sid)
        if not shots and self.agent_mode == "llm":
            try:
                rep.lines.append(self.direct(sid, mode="llm"))
            except Exception as exc:  # noqa: BLE001
                rep.lines.append(f"{sid}: Director agent unavailable ({exc})")
            shots = self.project.list_shots(sid)
        if not shots:
            rep.needs.append(f"{sid}: direction — `direct {sid} --manual-file <shots.yaml>` (agents in manual mode)")
            return False
        missing = [s.id for s in shots if not s.board_frame or not self.project.abs(s.board_frame).is_file()]
        if missing:
            self.make_boards(sid, missing)
            shots = self.project.list_shots(sid)
        animatic = self.make_animatic(sid)
        review_dir = build_g3_review(self.project, sid)
        alts = self._cursor(f"alternatives:{sid}", {}) or {}
        items = []
        for shot in shots:
            lines_txt = ""
            if shot.lines:
                scene = self.scene(sid)
                lines_txt = " · lines: " + " / ".join(
                    f"{self.display_name(ln.character_id)}: «{ln.text}»" for lid in shot.lines
                    if (ln := scene.line(lid)) is not None)
            items.append(GateItem(item_id=shot.id,
                                  summary=f"{shot.coverage_type} · {shot.duration_s:g}s · {shot.intention}{lines_txt}",
                                  options=["ok", 'regen "<note>"', 'alternatives 2 "<differ in>"', *alts.get(shot.id, [])],
                                  media=[str(self.project.abs(shot.board_frame))] if shot.board_frame else []))
        notes = [f"animatic: {animatic}"] if animatic else []
        strategies = self._cursor(f"strategies:{sid}", []) or []
        if len(strategies) > 1:
            notes.append("other coverage strategies: " + "; ".join(strategies[1:]) +
                         " — reply `strategy <label>` to switch")
        packet = GatePacket(gate=GateId.G3, scene_id=sid, title="Boards", risk=self._risk(sid),
                            model=self._image_model_id(), blocks=self._blocks(sid, "decomposition"),
                            review_dir=str(review_dir), resolve_bin=f"/REVIEW/{sid}/G3", items=items,
                            expected_format='SH01:ok, SH02:regen "tighter, lower angle", SH03:alternatives 2 '
                                            '"differ in lens" — or `approve all`',
                            defaults="Craft tier — never auto-proceeds; silence means AGY waits.", notes=notes)
        media = self._review_media(review_dir) or [str(self.project.abs(s.board_frame)) for s in shots
                                                   if s.board_frame]
        rep.gate_text = self._open_gate(packet, media)
        self.state.transition(sid, S.AWAITING_G3, reason=f"{len(shots)} shot(s) boarded")
        if self.is_autopilot():
            return self._autopilot_approve(GateId.G3, sid, rep)
        return False

    def _relevant_prefs(self, sid: str, agent: str | None = None) -> list[Any]:
        chars = [continuity.entity_id_of_state(s, [e.id for e in self.project.list_entities()]) or s
                 for s in self._scene_state_ids(sid)]
        try:
            return list(self.prefs.relevant(scene_id=sid, character_ids=tuple(chars), agent=agent, user=self.user))
        except TypeError:
            return list(self.prefs.relevant(scene_id=sid))

    def _run_agent(self, role: str, sid: str | None, task: str, *, mode: AgentMode | None = None,
                   manual_output: Any = None, feedback: Iterable[str] = (), n_variants: int = 3,
                   global_context: Any = None) -> Any:
        from .agents.runner import run_agent
        from .working_set import build_working_set

        ws = build_working_set(self.project, sid) if sid else None
        return run_agent(self.project, role, task=task, scene_id=sid, working_set=ws,
                         global_context=global_context, feedback=tuple(feedback),
                         preferences=tuple(self._relevant_prefs(sid, role)) if sid else (),
                         mode=mode or self.agent_mode, manual_output=manual_output, registry=self.registry,
                         n_variants=n_variants, fake=self.fake)

    def _parse_shots(self, sid: str, raw_shots: list[Any], *, start: int = 1) -> list[Any]:
        from .models import CreativeShot

        shots = []
        for k, raw in enumerate(raw_shots, start=start):
            data = dict(raw) if isinstance(raw, dict) else raw.model_dump(mode="json")
            data.setdefault("id", f"{sid}_SH{k:02d}")
            data["scene_id"] = sid
            data.setdefault("order", k)
            for key in ("gen_clips", "board_frame"):
                data.pop(key, None)
            shot = CreativeShot.model_validate(data)
            if parse_artifact_id(shot.id)["scene"] != sid or not parse_artifact_id(shot.id)["shot"]:
                raise StudioError(f"shot id {shot.id} must look like {sid}_SH01")
            shots.append(shot)
        return shots

    def direct(self, sid: str, *, mode: AgentMode | None = None, manual_output: Any = None,
               only: Iterable[str] | None = None, steer: str = "") -> str:
        """Director -> Cinematographer proposals (2–3 coverage strategies); recommended one becomes the shot list."""
        validate_id(sid, "scene id")
        task = ("Propose 2-3 coverage strategies for this scene. Each strategy is a list of complete CreativeShots "
                "(id, intention, coverage_type, duration_s, environment_angle, camera_side, beat_ids, actions "
                "[ONE atomic physical action each, entity IDs only], lines [line ids in order], line_action_index, "
                "spec.visual with all 9 fields [lighting = physical sources only, palette = no light sources], "
                "spec.audio_spec.ambience/diegetic_sfx). Respect the 180-degree axis, the active entity states and "
                "the dialogue rules: exact Hindi in Devanagari is spoken only by the intended speaker; every other "
                "visible mouth stays closed. Stage multi-shot coverage (master, OTS, reverse reaction) as separate "
                "shots. Return proposals as {label, summary, shots:[...]}.")
        feedback = [steer] if steer else []
        rounds = 0
        while True:
            rounds += 1
            result = self._run_agent("director", sid, task, mode=mode, manual_output=manual_output, feedback=feedback)
            strategies = []
            for i, raw_prop in enumerate(result.proposals or []):
                prop = raw_prop.get("content") if isinstance(raw_prop, dict) and isinstance(raw_prop.get("content"), dict) else raw_prop
                raw = prop.get("shots") or prop.get("creative_shots") if isinstance(prop, dict) else None
                if raw is None and isinstance(prop, dict) and "intention" in prop:
                    raw = [prop]
                if not raw:
                    continue
                label = str(prop.get("label") or (raw_prop.get("id") if isinstance(raw_prop, dict) else None) or chr(65 + i))
                strategies.append((label, str(prop.get("summary", "")), self._parse_shots(sid, raw)))
            if not strategies:
                raise StudioError(f"The Director returned no usable shots for {sid}.")
            issues = self._continuity_issues(sid, strategies[0][2])
            if not issues or (mode or self.agent_mode) != "llm" or rounds >= int(self.settings.max_agent_rounds):
                break
            self.state.bump_agent_round(sid, "director_continuity")
            feedback = feedback + [f"Continuity Supervisor: {i}" for i in issues]
        label, summary, shots = strategies[0]
        keep = set(only or [])
        existing = {s.id: s for s in self.project.list_shots(sid)}
        for shot in shots:
            if keep and shot.id not in keep:
                continue
            old = existing.get(shot.id)
            if old is not None and old.gen_clips:
                shot.version = old.version + 1
            self.project.save_shot(shot)
        scene = self.scene(sid)
        if not keep:
            scene.creative_shots = [s.id for s in shots]
        else:
            scene.creative_shots = list(dict.fromkeys([*scene.creative_shots, *[s.id for s in shots if s.id in keep]]))
        self.project.save_scene(scene)
        self.state.set_cursor(f"strategies:{sid}", [f"{lbl}: {summ}" for lbl, summ, _ in strategies])
        self._stage_strategies(sid, strategies)
        msg = f"{sid}: {len(shots)} shot(s) from strategy {label}" + (f" ({summary})" if summary else "")
        if issues:
            msg += f"; continuity notes for you: {'; '.join(issues)}"
        return msg

    def _stage_strategies(self, sid: str, strategies: list[tuple[str, str, list[Any]]]) -> None:
        save_versioned(self.project.path("scenes", sid, "direction", "strategies.yaml"), {
            "strategies": [{"label": lbl, "summary": summ, "shots": [s.model_dump(mode="json") for s in shots]}
                           for lbl, summ, shots in strategies],
            "generated_by": generated_by("agy.direction").model_dump(mode="json")})

    def switch_strategy(self, sid: str, label: str) -> str:
        p = self.project.path("scenes", sid, "direction", "strategies.yaml")
        if not p.exists():
            raise StudioError(f"{sid} has no staged coverage strategies.")
        data = read_record(p) or {}
        for strat in data.get("strategies", []):
            if str(strat.get("label")).upper() == label.upper():
                return self.direct(sid, mode="manual", manual_output={"proposals": [strat]})
        raise StudioError(f"{sid}: no strategy labelled {label}.")

    def _continuity_issues(self, sid: str, shots: list[Any]) -> list[str]:
        """Deterministic Continuity Supervisor checks on proposed shots (entities, axis side, lines)."""
        scene = self.scene(sid)
        known = {e.id for e in self.project.list_entities()} | set(self.project.entity_state_index())
        known |= {e.id for e in self.project.list_environments()}
        issues = []
        sides = {s.camera_side for s in shots if s.camera_side}
        if len(sides) > 1:
            issues.append(f"camera crosses the axis ({', '.join(sorted(sides))}); keep one side of "
                          f"'{self._axis(scene)}' unless a crossing shot is intended")
        covered = [lid for s in shots for lid in s.lines]
        for ln in scene.dialogue:
            if covered.count(ln.line_id) != 1:
                issues.append(f"line {ln.line_id} must be covered exactly once (found {covered.count(ln.line_id)})")
        for s in shots:
            for tok in continuity.entity_tokens(" ".join([*s.actions, s.spec.visual.subject_and_action])):
                if tok not in known and not tok.startswith(sid):
                    issues.append(f"{s.id}: unknown entity token {tok}")
        return issues

    def _axis(self, scene: Scene) -> str:
        if scene.environment_id:
            try:
                return self.project.load_environment(scene.environment_id).axis_of_action or "axis of action"
            except FileNotFoundError:
                pass
        return "axis of action"

    def _shot_refs(self, sid: str, shot: Any) -> list[str]:
        scene = self.scene(sid)
        idx = self.project.entity_state_index()
        tokens = continuity.entity_tokens(" ".join([shot.spec.visual.subject_and_action,
                                                    shot.spec.visual.composition, *shot.actions]))
        refs: list[str] = []
        # 1. If the shot uses a 3D camera (cam:<NAME>), pass the 3D camera's clean blockout render FIRST
        #    so the image model locks onto the exact 3D camera angle, horizon, and perspective.
        if scene.environment_id and getattr(shot, "environment_angle", None):
            ang = str(shot.environment_angle)
            if ang.startswith("cam:"):
                cam_name = ang.split(":", 1)[1]
                from .set3d.store import latest_render_dir

                rdir = latest_render_dir(self.project, scene.environment_id, cam_name)
                if rdir is not None and (rdir / "blockout.png").exists():
                    refs.append(self.project.rel(rdir / "blockout.png"))

        # 2. Pass the isolated cast identity crop sheet if present, plus active character/prop reference packs
        cast_crop = self.project.path("starter_kit", "cast_exact_from_master.png")
        if cast_crop.exists():
            refs.append(self.project.rel(cast_crop))
        for state_id in scene.entity_states_active:
            ent_id = continuity.entity_id_of_state(state_id, [e for e in tokens]) or ""
            if (ent_id or state_id in idx) and state_id in idx:
                refs += idx[state_id][1].reference_pack.image_uris()[:2]  # type: ignore[attr-defined]

        # 3. Pass the environment's locked geometry still / styled camera angle for room material continuity
        if scene.environment_id:
            env = self.project.load_environment(scene.environment_id)
            angle = env.angle_uri(shot.environment_angle)
            if angle:
                refs.append(angle)
            if env.geometry_still:
                refs.append(env.geometry_still)
        try:
            limit = int(self.registry.get(self._image_model_id()).supports.max_reference_images) or 14
        except Exception:  # noqa: BLE001
            limit = 14
        return [str(self.project.abs(r)) for r in dict.fromkeys(refs) if self.project.abs(r).exists()][:limit]

    def _named(self, text: str) -> str:
        for tok in continuity.entity_tokens(text):
            name = self.display_name(tok)
            if name != tok:
                text = re.sub(rf"\b{re.escape(tok)}\b", name, text)
        return text

    def make_boards(self, sid: str, shot_ids: Iterable[str] | None = None) -> list[str]:
        img = self._image_adapter()
        wanted = set(shot_ids or [])
        made = []
        scene = self.scene(sid)
        for shot in self.project.list_shots(sid):
            if wanted and shot.id not in wanted:
                continue
            v = shot.spec.visual
            spatial_note = ""
            if scene.environment_id and getattr(shot, "environment_angle", None) and str(shot.environment_angle).startswith("cam:"):
                cam_name = str(shot.environment_angle).split(":", 1)[1]
                try:
                    from .set3d.blender_export import compute_view_space_blocking_summary
                    from .set3d.store import load_set

                    layout = load_set(self.project, scene.environment_id)
                    cam_obj = layout.camera(cam_name)
                    if cam_obj is not None:
                        spatial_note = " 3D CAMERA SPATIAL BLOCKING: " + compute_view_space_blocking_summary(layout, cam_obj)
                except Exception:  # noqa: BLE001
                    spatial_note = ""
            prompt = self._named(
                f"Photorealistic live-action storyboard keyframe with the real actors and props in the exact same room. "
                f"{v.subject_and_action}.{spatial_note} Camera: {v.camera}. Lens: {v.lens_and_aspect}. "
                f"Lighting: {v.lighting}. Palette: {v.palette}. Composition: {v.composition}. Atmosphere: "
                f"{v.atmosphere}. Mood: {v.mood}. Match the exact room architecture, wall color, wooden dining table/bed, "
                f"and character facial identities & wardrobes from the reference images; mouths of non-speaking characters "
                f"100% CLOSED. {v.attribution}.")
            dest = next_version_path(self.project.path("scenes", sid, "boards"), shot.id, ".png")
            dest.parent.mkdir(parents=True, exist_ok=True)
            result = img.generate(prompt, dest, reference_images=self._shot_refs(sid, shot),
                                  aspect_ratio=self.settings.aspect_ratio)
            shot.board_frame = self.project.rel(Path(result) if result else dest)
            self.project.save_shot(shot)
            made.append(shot.id)
        return made

    def make_animatic(self, sid: str) -> str | None:
        """Best-effort board slideshow (``boards/animatic_v<n>.mp4``) for the G3 review."""
        from .procs import run

        shots = [s for s in self.project.list_shots(sid) if s.board_frame and self.project.abs(s.board_frame).is_file()]
        if not shots:
            return None
        w, h = (540, 960) if self.settings.height >= self.settings.width else (960, 540)
        dest = next_version_path(self.project.path("scenes", sid, "boards"), "animatic", ".mp4")
        argv = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n"]
        chains = []
        for i, shot in enumerate(shots):
            argv += ["-loop", "1", "-t", f"{max(1.0, shot.duration_s or 3.0):.2f}", "-i",
                     str(self.project.abs(shot.board_frame))]
            chains.append(f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
                          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24[v{i}]")
        concat = "".join(f"[v{i}]" for i in range(len(shots))) + f"concat=n={len(shots)}:v=1:a=0[out]"
        argv += ["-filter_complex", ";".join(chains + [concat]), "-map", "[out]", "-pix_fmt", "yuv420p"]
        for codec in ("libx264", "mpeg4"):
            try:
                run([*argv, "-c:v", codec, str(dest)], timeout=300)
                return str(dest)
            except Exception:  # noqa: BLE001
                if dest.exists():
                    return str(dest)
        return None

    def _apply_g3(self, sid: str | None, response: Any, decisions: list[Any]) -> list[str]:
        assert sid
        shots = self.project.list_shots(sid)
        explicit = {c.item_id: c for c in response.choices}
        redo: dict[str, str] = {}
        alts: list[Any] = []
        chosen: list[Any] = []
        for shot in shots:
            ch = explicit.get(shot.id)
            if ch is not None and ch.action in ("regen", "reject"):
                redo[shot.id] = ch.steer or ch.value or response.free_text
            elif ch is not None and ch.action == "alternatives":
                alts.append(ch)
            elif ch is not None and ch.action == "choose" and ch.value:
                chosen.append(ch)
            elif response.reject_all and ch is None:
                redo[shot.id] = response.free_text
        lines: list[str] = []
        for ch in chosen:
            lines.append(self.choose_alternative(ch.item_id, str(ch.value)))
        if redo or alts or chosen:
            self.state.transition(sid, S.DIRECTION, reason="G3: revisions requested")
            for shot in shots:
                if shot.id in redo:
                    shot.board_frame = None
                    self.project.save_shot(shot)
            for ch in alts:
                lines.append(self.alternatives(ch.item_id, int(ch.value or 2) if str(ch.value or "").isdigit() else 2,
                                               ch.steer or "", record=False))
            if redo:
                if self.agent_mode == "llm":
                    try:
                        lines.append(self.direct(sid, mode="llm", only=list(redo),
                                                 steer="; ".join(f"{k}: {v}" for k, v in redo.items() if v)))
                    except Exception as exc:  # noqa: BLE001
                        lines.append(f"Director agent unavailable ({exc}); AGY will revise {', '.join(redo)}.")
                else:
                    lines.append("NEEDS: revised shots for " + ", ".join(f"{k} ({v})" for k, v in redo.items()) +
                                 f" — `direct {sid} --manual-file <file> --only {','.join(redo)}`")
            lines.append(self.advance(sid).text())
            return lines
        for shot in shots:
            if shot.approval_status != ApprovalStatus.LOCKED:
                shot.approval_status = ApprovalStatus.LOCKED
                self.project.save_shot(shot)
        self.state.transition(sid, S.DECOMPOSITION, reason="G3 approved")
        lines.append(self.advance(sid).text())
        return lines

    def alternatives(self, shot_or_clip: str, n: int, differ_in: str = "", *, record: bool = True,
                     manual_output: Any = None) -> str:
        """`alternatives <shot_id> N "<differ in>"` (shots) or N extra takes for a clip."""
        ids = parse_artifact_id(shot_or_clip)
        n = max(1, min(int(n), 4))
        if record:
            self.decisions.record(tier=DecisionTier.CRAFT, target_artifact_id=shot_or_clip,
                                  action=DecisionAction.REQUEST_ALTERNATIVES,
                                  payload={"n": n, "differ_in": differ_in}, rationale=differ_in, user=self.user)
        if ids["clip"]:
            self.state.set_cursor(f"ntakes:{ids['clip']}", n)
            return self.regen(ids["clip"], differ_in, record=False, keep_current=True)
        if not ids["shot"]:
            raise StudioError(f"{shot_or_clip} is not a shot or clip id")
        sid = ids["scene"]
        shot = self.project.load_shot(ids["shot"])
        task = (f"Propose {n} alternative versions of shot {shot.id} that differ in: {differ_in or 'coverage'}. "
                f"Keep its actions, lines and continuity; return proposals as {{label, summary, shots:[one shot]}}.")
        result = self._run_agent("director", sid, task, manual_output=manual_output, n_variants=n,
                                 feedback=[differ_in] if differ_in else ())
        staged = []
        alt_dir = self.project.path("scenes", sid, "direction", "alternatives")
        for i, raw_prop in enumerate((result.proposals or [])[:n], start=1):
            prop = raw_prop.get("content") if isinstance(raw_prop, dict) and isinstance(raw_prop.get("content"), dict) else raw_prop
            raw = prop.get("shots") if isinstance(prop, dict) else None
            raw = raw or [prop]
            alt = self._parse_shots(sid, [{**raw[0], "id": shot.id}])[0]
            label = f"A{i}"
            save_versioned(alt_dir / f"{shot.id}_{label}.yaml", alt)
            staged.append(f"choose {label}")
        alts = dict(self._cursor(f"alternatives:{sid}", {}) or {})
        alts[shot.id] = staged
        self.state.set_cursor(f"alternatives:{sid}", alts)
        return f"{shot.id}: {len(staged)} alternative(s) staged — they appear as options at G3."

    def choose_alternative(self, shot_id: str, label: str) -> str:
        ids = parse_artifact_id(shot_id)
        p = self.project.path("scenes", ids["scene"], "direction", "alternatives", f"{shot_id}_{label.upper()}.yaml")
        if not p.exists():
            raise StudioError(f"no alternative {label} for {shot_id}")
        from .models import CreativeShot

        alt = CreativeShot.model_validate(read_record(p))
        alt.board_frame = None
        self.project.save_shot(alt)
        return f"{shot_id}: switched to alternative {label.upper()} (new board at the next G3)."

    # ---- DECOMPOSITION / AUDIO PLAN / G4 -----------------------------------------------------
    def _h_decomposition(self, sid: str, rep: StepReport) -> bool:
        from .decomposer import decompose_scene

        out = decompose_scene(self.project, self.registry, sid, model_id=self.scene_model(sid))
        n = sum(len(v) for v in out.values())
        flagged = [c.id for clips in out.values() for c in clips if c.flags]
        self.state.transition(sid, S.AUDIO_PLAN, reason=f"{n} clip(s)")
        rep.lines.append(f"{sid}: decomposed into {n} clip(s)" +
                         (f"; flagged {', '.join(flagged)}" if flagged else ""))
        return True

    def _cue_plan_path(self, sid: str) -> Path:
        return self.project.path("scenes", sid, "audio", "cue_plan.yaml")

    def cue_plan(self, sid: str) -> list[dict[str, Any]]:
        p = self._cue_plan_path(sid)
        return list((read_record(p) or {}).get("cues", [])) if p.exists() else []

    def plan_audio(self, sid: str, *, mode: AgentMode | None = None, manual_output: Any = None,
                   steers: dict[str, str] | None = None) -> list[str]:
        """Voice Director deliveries, Sound Designer ambience/SFX flags, Composer cue plan (idempotent)."""
        scene = self.scene(sid)
        env = self.project.load_environment(scene.environment_id) if scene.environment_id else None
        clips = self.active_clips(sid)
        lines: list[str] = []
        deliveries: dict[str, str] = {}
        cues: list[dict[str, Any]] | None = None
        mode = mode or self.agent_mode
        if manual_output is not None or mode == "llm":
            try:
                res = self._run_agent("voice_director", sid,
                                      "Write one delivery note per dialogue line (emotion, pace, volume, breath) "
                                      "that keeps each character's locked voice profile; off-screen lines are "
                                      "heard while every on-screen mouth stays closed. Return {deliveries: "
                                      "{line_id: note}, cues: [{cue_id, start_s, duration_s, prompt}]}.",
                                      mode=mode, manual_output=manual_output, n_variants=1)
                raw_first = (res.proposals or [{}])[0] if res.proposals else {}
                first = raw_first.get("content") if isinstance(raw_first, dict) and isinstance(raw_first.get("content"), dict) else raw_first
                deliveries = {str(k): str(v) for k, v in (first.get("deliveries") or {}).items()}
                cues = first.get("cues")
            except Exception as exc:  # noqa: BLE001
                lines.append(f"{sid}: Voice Director unavailable ({exc}); using script parentheticals")
        for clip in clips:
            changed = False
            for ln in clip.spec.audio_spec.all_lines():
                wanted = deliveries.get(ln.line_id or "")
                if wanted and wanted != ln.delivery:
                    ln.delivery = wanted
                    changed = True
                elif not ln.delivery:
                    src = scene.line(ln.line_id) if ln.line_id else None
                    perf = scene.performance_states.get(ln.character_id, "") or clip.performance_state
                    ln.delivery = "; ".join(x for x in ((src.parenthetical if src else ""), perf) if x) or "natural"
                    changed = True
                if steers and clip.id in steers and steers[clip.id] and steers[clip.id] not in ln.delivery:
                    ln.delivery = f"{ln.delivery}; {steers[clip.id]}"
                    changed = True
            if not clip.spec.audio_spec.ambience and env is not None and env.room_tone:
                clip.spec.audio_spec.ambience = env.room_tone
                changed = True
            for sfx in clip.spec.audio_spec.diegetic_sfx:
                if any(w in sfx.lower() for w in HARD_SFX_WORDS):
                    flag = f"sfx_supplement:{sfx}"
                    if flag not in clip.flags:
                        clip.flags = [*clip.flags, flag]
                        changed = True
            if changed:
                self.project.save_clip(clip)
        if cues or not self.cue_plan(sid):
            total = round(sum(c.duration_s for c in clips), 2)
            moods = "; ".join(dict.fromkeys(b.summary for b in scene.beats)) or scene.slug
            cues = cues or [{"cue_id": f"{sid}_CUE01", "start_s": 0.0, "duration_s": total,
                             "prompt": f"Instrumental underscore for {scene.slug.replace('_', ' ')}: {moods}. "
                                       f"No vocals; sits under native dialogue."}]
            save_versioned(self._cue_plan_path(sid), {"cues": cues,
                                                       "generated_by": generated_by("agy.composer").model_dump(mode="json")})
        return lines

    def _h_audio_plan(self, sid: str, rep: StepReport) -> bool:
        from .gates import GateItem, GatePacket
        from .voice_audition import prepare_scene_voices

        rep.lines += self.plan_audio(sid)
        voice_report: dict[str, Any] = {}
        try:
            voice_report = prepare_scene_voices(self.project, sid, fake=bool(self.is_fake()))
        except Exception:  # noqa: BLE001
            pass
        control_mode = str(getattr(self.settings, "control_mode", "director") or "director").lower()
        # G4 is a craft gate: GateManager.auto_pass() refuses it by design. In hybrid mode (all voices already locked
        # in the Global Bible) and in autopilot the packet is still opened and then answered by AGY — a recorded
        # decision, not a silent skip.
        self_answer = self.is_autopilot() or (control_mode == "hybrid" and bool(voice_report.get("auto_pass_eligible")))
        items = []
        for clip in self.active_clips(sid):
            a = clip.spec.audio_spec
            said = [f"{self.display_name(ln.character_id)}: «{ln.text}» ({ln.transliteration or '—'}) — "
                    f"delivery: {ln.delivery}; voice: {ln.voice}" for ln in a.dialogue]
            said += [f"OFF-SCREEN {self.display_name(ln.character_id)}: «{ln.text}» ({ln.transliteration or '—'}) "
                     f"— delivery: {ln.delivery}; voice: {ln.voice}; all on-screen mouths closed" for ln in a.off_screen]
            summary = " | ".join(said) or "no dialogue (all mouths closed)"
            summary += f" | ambience: {a.ambience or '—'} | sfx: {', '.join(a.diegetic_sfx) or '—'} | music: none"
            items.append(GateItem(item_id=clip.id, summary=summary, flags=list(clip.flags),
                                  options=["ok", 'regen "<delivery note>"']))
        notes = [f"cue {c.get('cue_id')}: {c.get('duration_s')}s from {c.get('start_s', 0)}s — {c.get('prompt')}"
                 for c in self.cue_plan(sid)]
        for aud in voice_report.get("auditioned", []):
            notes.append(f"voice spec {aud['character_id']} (goes into the video prompt; no audio generated): "
                         f"{aud['description']}")
        for reuse in voice_report.get("reused_from_global", []):
            notes.append(f"voice locked in Global Bible {reuse['character_id']}: {reuse.get('anchor_wav') or reuse['voice_profile_id']}")
        packet = GatePacket(gate=GateId.G4, scene_id=sid, title="Audio spec per clip + cue plan",
                            risk=self._risk(sid), model=self.scene_model(sid), blocks=self._blocks(sid, "generation"),
                            items=items, notes=notes,
                            expected_format='C01:ok, C02:regen "angrier, faster, breathless" — or `approve all`',
                            defaults="Craft tier — never auto-proceeds; silence means AGY waits.")
        rep.gate_text = self._open_gate(packet)
        self.state.transition(sid, S.AWAITING_G4, reason="audio_spec + cue plan ready")
        if self_answer:
            return self._autopilot_approve(GateId.G4, sid, rep)
        return False

    def _apply_g4(self, sid: str | None, response: Any, decisions: list[Any]) -> list[str]:
        from .voice_audition import lock_approved_voices_to_global_bible

        assert sid
        steers = {c.item_id: (c.steer or c.value or response.free_text) for c in response.choices
                  if c.action in ("regen", "reject")}
        if response.reject_all:
            steers.update({c.id: response.free_text for c in self.active_clips(sid) if c.id not in steers})
        lines: list[str] = []
        if steers:
            self.state.transition(sid, S.AUDIO_PLAN, reason="G4: delivery changes requested")
            lines += self.plan_audio(sid, mode="manual" if self.agent_mode == "manual" else None, steers=steers)
            lines.append(self.advance(sid).text())
            return lines
        try:
            lock_approved_voices_to_global_bible(self.project, sid)
        except Exception:  # noqa: BLE001
            pass
        self.state.transition(sid, S.LINT, reason="G4 approved")
        lines.append(self.advance(sid).text())
        return lines

    # ---- LINT / GENERATING / QA / G5 ---------------------------------------------------------
    def _fill_anchor(self, cid: str) -> None:
        clip = self.project.load_clip(cid)
        if not clip.anchor_from_clip_id:
            return
        anchor = self.project.load_clip(clip.anchor_from_clip_id)
        if not anchor.selected_take_id:
            return
        take = self.project.load_take(anchor.selected_take_id)
        if take.last_frame_uri and (clip.conditioning.first_frame_anchor != take.last_frame_uri or
                                    clip.conditioning.first_frame_anchor_take_id != take.id):
            clip.conditioning = clip.conditioning.model_copy(update={
                "first_frame_anchor": take.last_frame_uri, "first_frame_anchor_take_id": take.id})
            self.project.save_clip(clip)

    def _voice_key(self, char: str) -> str:
        vp = self.project.load_voice_profile(char)
        return vp.id if vp else f"VOICE_{char}"

    def _speakers_needing_anchor(self, clips: Iterable[GenClip]) -> list[str]:
        out = []
        for clip in clips:
            for char in continuity.speakers_in_clip(clip):
                vp = self.project.load_voice_profile(char)
                if vp is not None and not vp.anchor_take and char not in out:
                    out.append(char)
        return out

    def _h_lint(self, sid: str, rep: StepReport) -> bool:
        from .linter import lint_shot

        frontier = self.frontier(sid)
        need_gen = [cid for cid in frontier if not self._live_takes(cid)]
        if not need_gen:
            self.state.transition(sid, S.GENERATING, reason="nothing new to dispatch")
            return True
        for cid in need_gen:
            self._fill_anchor(cid)
        blockers = []
        for char in self._speakers_needing_anchor(self.project.load_clip(c) for c in need_gen):
            owner = self.state.claim(self._voice_key(char), sid)
            if owner != sid:
                blockers.append(f"voice:{char}@{owner}")
        if blockers:  # unanchored voice: the first-appearance scene picks the anchor; others wait
            return self._block(sid, rep, *blockers)
        reports: dict[str, Any] = {}
        for shot_id in sorted({parse_artifact_id(c)["shot"] for c in need_gen}):
            for clip, report in lint_shot(self.project, self.registry, shot_id, apply_fixes=True, save=True):
                reports[clip.id] = report
        failing = {c: reports[c] for c in need_gen if c in reports and reports[c].blocking}
        if failing:
            if any(r.bounce_to_decomposer for r in failing.values()) and \
                    self.state.bump_agent_round(sid, "lint_bounce") <= MAX_LINT_BOUNCES:
                self.state.transition(sid, S.DECOMPOSITION, reason="lint bounce: " + ", ".join(failing))
                rep.lines.append(f"{sid}: lint bounced {', '.join(failing)} back to the Shot Decomposer")
                return True
            detail = "\n".join(f"  {c}: " + "; ".join(f"{f.rule} {f.message}" for f in r.findings
                                                     if f.severity == "error") for c, r in failing.items())
            q = self.ask(sid, "lint", f"{sid}: pre-flight lint stopped {len(failing)} clip(s) — nothing was sent "
                                      f"to the video models.\n{detail}")
            return self._block(sid, rep, f"question:{q.id} lint")
        planned = round(sum(self.budget.estimate_clip(self.project.load_clip(c), takes=self._n_takes(c))
                            for c in need_gen), 4)
        check = self.budget.check(sid, planned)
        if not check.ok:  # §12: overrun -> ask, never silently continue
            self.ask(sid, "budget", f"{check.message} Reply `raise-cap {sid} <new cap>` to continue.")
            return self._block(sid, rep, f"budget:{planned}")
        self.state.set_cursor(f"wave:{sid}", need_gen)
        self.state.transition(sid, S.GENERATING, reason=f"{len(need_gen)} clip(s) passed lint")
        rep.lines.append(f"{sid}: lint passed for {len(need_gen)} clip(s) (planned ≈ {planned:.2f})")
        return True

    def _h_generating(self, sid: str, rep: StepReport) -> bool:
        from .adapters.auth import AuthError
        from .budget import BudgetExceeded
        from .shot_generator import LintGateError, ModelFallbackNeeded, generate_clip, resume_inflight

        # Fail fast (BEFORE any take folder / payload / provider job exists): one cheap gcloud check. A broken
        # sign-in becomes a blocking question carrying gcloud's own error + the fix. No seed retry, no provider
        # fallback and never an offline stand-in — those only waste minutes and hide the real problem.
        try:
            self._require_vertex_auth()
        except AuthError as exc:
            return self._auth_block(sid, rep, exc)

        try:
            resumed = resume_inflight(self.project, self.registry, self.state, fake=self.fake)
            if resumed:
                rep.lines.append(f"{sid}: resumed {len(resumed)} in-flight take(s) (no resubmission)")
        except AuthError as exc:
            return self._auth_block(sid, rep, exc)
        except Exception as exc:  # noqa: BLE001
            rep.lines.append(f"{sid}: could not resume in-flight takes ({exc})")
        wave = [c for c in self.frontier(sid) if not self._live_takes(c)]
        made = 0
        for cid in wave:
            try:
                takes = generate_clip(self.project, self.registry, self.state, cid, n_takes=self._n_takes(cid),
                                      budget=self.budget, fake=self.fake,
                                      seed_base=self._seed_base(cid, self._attempt(cid)))
            except LintGateError as exc:
                self.state.transition(sid, S.LINT, reason=f"{cid}: {exc}")
                return True
            except AuthError as exc:
                return self._auth_block(sid, rep, exc)
            except ModelFallbackNeeded as exc:
                clip = self.project.load_clip(cid)
                clip.model_target = clip.model_target.model_copy(update={"preferred": exc.model_id, "fallback": None})
                self.project.save_clip(clip)
                self.state.transition(sid, S.LINT, reason=f"{cid}: re-lint for fallback model {exc.model_id}")
                rep.lines.append(f"{cid}: preferred model failed; re-linting for {exc.model_id}")
                return True
            except BudgetExceeded as exc:
                self.ask(sid, "budget", f"{exc} Reply `raise-cap {sid} <new cap>` to continue.")
                return self._block(sid, rep, "budget:0")
            except Exception as exc:  # noqa: BLE001 — provider error: retry with new seeds, then ask
                if self._bump_failure(cid) > MAX_AUTO_REGENS:
                    q = self.ask(sid, f"dispatch_{cid}", f"{cid} failed to generate repeatedly: {exc}")
                    return self._block(sid, rep, f"question:{q.id} dispatch {cid}")
                self._bump_attempt(cid)
                rep.lines.append(f"{cid}: generation failed ({exc}); will retry with new seeds")
                continue
            ok = [t for t in takes if t.dispatch_status == DispatchStatus.COMPLETED]
            if not ok and not self._live_takes(cid):
                if self._bump_failure(cid) > MAX_AUTO_REGENS:
                    errs = "; ".join(str(t.error) for t in takes if t.error)[:400]
                    q = self.ask(sid, f"dispatch_{cid}", f"{cid}: every take failed or was filtered ({errs}).")
                    return self._block(sid, rep, f"question:{q.id} dispatch {cid}")
                self._bump_attempt(cid)
            made += len(ok)
        self.state.transition(sid, S.QA, reason=f"{made} take(s) generated")
        if wave:
            rep.lines.append(f"{sid}: generated {made} take(s) for {len(wave)} clip(s)")
        return True

    def _prev_take(self, clip: GenClip) -> Take | None:
        clips = self.active_clips(clip.scene_id)
        ids = [c.id for c in clips]
        if clip.id not in ids or ids.index(clip.id) == 0:
            return None
        prev = clips[ids.index(clip.id) - 1]
        return self.project.load_take(prev.selected_take_id) if prev.selected_take_id else None

    def _reject_takes(self, takes: Iterable[Take], reason: str) -> None:
        from .vault import index_rejected_take

        for t in takes:
            if t.status != TakeStatus.REJECTED:
                t.status = TakeStatus.REJECTED
                self.project.save_take(t)
                append_jsonl(self.project.path("logs", "take_rejections.jsonl"),
                             {"ts": now_iso(), "take_id": t.id, "reason": reason})
                try:
                    index_rejected_take(self.project, t, reason=reason)
                except Exception:  # noqa: BLE001
                    pass

    def _recommended_take(self, cid: str) -> Take | None:
        ranked = sorted(self._candidates(cid), key=lambda t: (t.rank if t.rank is not None else 999, t.id))
        for t in ranked:
            if t.qa is None or (t.qa.recommendation != Recommendation.REJECT and not t.qa.hard_reject_reasons):
                return t
        return None

    def _h_qa(self, sid: str, rep: StepReport) -> bool:
        from .qa_critic import rank_takes, route_selection, score_take

        frontier = self.frontier(sid)
        if not frontier:
            self.state.transition(sid, S.ASSEMBLY, reason="every clip has a selected take")
            return True
        scene = self.scene(sid)
        human, regen, auto = [], [], []
        for cid in frontier:
            cands = self._candidates(cid)
            if not cands:
                regen.append(cid) if not self._live_takes(cid) else None
                continue
            clip = self.project.load_clip(cid)
            prev = self._prev_take(clip)
            for t in cands:
                if t.qa is None:
                    score_take(self.project, self.registry, clip, t, prev_take=prev, fake=self.fake)
            ranked = rank_takes(self.project, clip, self._candidates(cid))
            acceptable = [t for t in ranked if t.qa is not None and t.qa.recommendation != Recommendation.REJECT
                          and not t.qa.hard_reject_reasons]
            if not acceptable:
                if self._attempt(cid) < MAX_AUTO_REGENS:
                    reasons = sorted({r for t in ranked if t.qa for r in t.qa.hard_reject_reasons}) or ["QA reject"]
                    self._reject_takes(ranked, "auto re-roll: " + ", ".join(reasons))
                    self._bump_attempt(cid)
                    regen.append(cid)
                    rep.lines.append(f"{cid}: all takes rejected by QA ({', '.join(reasons)}); re-rolling")
                    continue
                human.append(cid)
                continue
            if route_selection(self.project, scene, clip, ranked) == "auto_select" and \
                    not self._speakers_needing_anchor([clip]):
                self._select_take(cid, acceptable[0].id, auto=True)
                auto.append(cid)
            else:
                human.append(cid)
        if auto:
            append_jsonl(self.project.path("logs", "auto_selections.jsonl"),
                         {"ts": now_iso(), "scene_id": sid, "clips": auto, "reason": "low risk, composite ≥ threshold"})
            rep.lines.append(f"{sid}: auto-selected {len(auto)} low-risk clip(s): {', '.join(auto)}")
        if regen:
            self.state.transition(sid, S.GENERATING, reason="re-roll after QA: " + ", ".join(regen))
            return True
        if human and self.is_autopilot():
            # No human picks in autopilot: open the G5 packet (review folder + audit record) and answer it with the
            # QA-recommended take per clip — or, once the automatic re-rolls are exhausted, the best-ranked take —
            # plus first voice anchors taken from those picks. Every choice is a recorded decision (source=autopilot).
            parts: list[str] = []
            anchors: dict[str, str] = {}
            for cid in human:
                best = self._recommended_take(cid)
                if best is None:
                    ranked = rank_takes(self.project, self.project.load_clip(cid), self._candidates(cid))
                    if not ranked:
                        continue
                    best = ranked[0]
                    parts.append(f"{cid}:choose {best.id}")
                    rep.lines.append(f"{cid}: autopilot picked best-ranked {best.id} despite QA flags "
                                     f"(auto re-rolls exhausted) — check it at G6")
                else:
                    parts.append(f"{cid}:ok")
                for char in self._speakers_needing_anchor([self.project.load_clip(cid)]):
                    anchors.setdefault(char, best.id)
            packet = self._g5_packet(sid, human, auto_selected=auto)
            rep.gate_text = self._open_gate(packet, self._review_media(packet.review_dir))
            self.state.transition(sid, S.AWAITING_G5, reason=f"autopilot: {len(human)} clip(s) picked by QA rank")
            reply = ", ".join([*parts, *(f"anchor {c}={t}" for c, t in anchors.items())]) or "approve all"
            return self._autopilot_approve(GateId.G5, sid, rep, text=reply)
        if human:
            packet = self._g5_packet(sid, human, auto_selected=auto)
            rep.gate_text = self._open_gate(packet, self._review_media(packet.review_dir))
            self.state.transition(sid, S.AWAITING_G5, reason=f"{len(human)} clip(s) need your pick")
            return False
        if auto and not human:
            self.gates.auto_pass(GateId.G5, sid, reason=f"low-risk auto-select: {', '.join(auto)}", clip_ids=auto)
        self.state.transition(sid, S.LINT if self.frontier(sid) else S.ASSEMBLY, reason="wave selected")
        return True

    def _clip_summary(self, clip: GenClip) -> str:
        bits = [clip.spec.visual.subject_and_action]
        for ln in clip.spec.audio_spec.dialogue:
            bits.append(f"{self.display_name(ln.character_id)}: «{ln.text}»")
        for ln in clip.spec.audio_spec.off_screen:
            bits.append(f"off-screen {self.display_name(ln.character_id)}: «{ln.text}»")
        return f"{clip.duration_s:g}s · " + " · ".join(b for b in bits if b)

    def _g5_packet(self, sid: str, clip_ids: list[str], *, auto_selected: Iterable[str] = ()) -> Any:
        from .gates import GateItem, GatePacket
        from .invalidation import AnchorGraph
        from .review import build_g5_review

        review_dir = build_g5_review(self.project, sid, clip_ids)
        graph = AnchorGraph.build(self.project)
        thr = self.settings.qa
        items = []
        clips = [self.project.load_clip(c) for c in clip_ids]
        for clip in clips:
            ranked = sorted(self._candidates(clip.id), key=lambda t: (t.rank if t.rank is not None else 999, t.id))
            best = self._recommended_take(clip.id)
            qa = best.qa if best is not None else (ranked[0].qa if ranked else None)
            sims = [v for v in (qa.speaker_similarity.values() if qa else []) if v is not None]
            flags = sorted(set((qa.artifact_flags + qa.unwanted_audio_flags + qa.hard_reject_reasons) if qa else []))
            items.append(GateItem(
                item_id=clip.id,
                recommendation=best.id if best is not None else None,
                qa_composite=qa.composite if qa else None,
                transcript_ok=(qa.transcript_match >= thr.transcript_match_min)
                if qa is not None and qa.transcript_match is not None else None,
                speaker_sim=min(sims) if sims else None,
                flags=flags + [f for f in clip.flags if f not in flags],
                anchor_fanout=len(graph.anchor_descendants(clip.id)),
                options=[t.id for t in ranked],
                summary=self._clip_summary(clip),
                media=self._review_media(review_dir, clip.id)))
        anchor_prompts = []
        for char in self._speakers_needing_anchor(clips):
            cands = [t.id for c in clips if char in continuity.speakers_in_clip(c) for t in self._candidates(c.id)]
            anchor_prompts.append(f"Pick {self.display_name(char)}'s voice anchor — all future takes will be scored "
                                  f"against it. Reply `anchor {char}=<take id>` (candidates: {', '.join(cands[:6])}).")
        notes = [f"auto-selected (low risk): {', '.join(auto_selected)}"] if auto_selected else []
        return GatePacket(gate=GateId.G5, scene_id=sid, title="Take selection", risk=self._risk(sid),
                          model=self.scene_model(sid), blocks=self._blocks(sid, "assembly"),
                          review_dir=str(review_dir), resolve_bin=f"/REVIEW/{sid}/G5", items=items,
                          expected_format='C01:T2, C02:ok, C03:regen "slower delivery, less head movement" — plus '
                                          'any free-text steer; `approve all` takes every recommendation',
                          defaults="Craft tier — never auto-proceeds; silence means AGY waits.",
                          voice_anchor_prompts=anchor_prompts, notes=notes)

    def _descendant_fanout(self, cid: str) -> Any:
        from .invalidation import AnchorGraph

        fan = AnchorGraph.build(self.project).fanout(cid)
        clips = [c for c in fan.clips if c != cid]
        takes = [t for t in fan.takes if not t.startswith(cid + "_")]
        return fan.model_copy(update={"clips": clips, "takes": takes})

    def _select_take(self, cid: str, take_id: str, *, decision_id: str | None = None, auto: bool = False) -> list[str]:
        from .invalidation import apply_invalidation

        clip = self.project.load_clip(cid)
        take = self.project.load_take(take_id)
        if take.clip_id != cid:
            raise StudioError(f"{take_id} belongs to {take.clip_id}, not {cid}")
        if take.dispatch_status != DispatchStatus.COMPLETED or not take.video_uri:
            raise StudioError(f"{take_id} has no finished video yet")
        previous = clip.selected_take_id if clip.status != ClipStatus.STALE else None
        for t in self.project.list_takes(cid):
            if t.id != take_id and t.status == TakeStatus.SELECTED:
                t.status = TakeStatus.CANDIDATE
                self.project.save_take(t)
        take.status = TakeStatus.SELECTED
        self.project.save_take(take)
        clip.selected_take_id = take_id
        clip.status = ClipStatus.SELECTED
        self.project.save_clip(clip)
        out = [f"{cid}: {'auto-' if auto else ''}selected {take_id}"]
        if previous and previous != take_id:
            fan = self._descendant_fanout(cid)
            if fan.clips:
                apply_invalidation(self.project, self.state, fan, reason=f"{cid}: take {previous} -> {take_id}",
                                   decision_id=decision_id)
                out.append(f"Stale (anchored on {cid}'s last frame): {', '.join(fan.clips)}")
        return out

    def set_voice_anchor(self, char: str, take_id: str, *, decision_id: str | None = None) -> str:
        """First anchor: set immediately (§4.3). Re-anchor: Vision-tier confirm after the fan-out is shown."""
        from .invalidation import AnchorGraph, describe_fanout

        validate_id(char, "character id")
        vp = self.project.load_voice_profile(char)
        if vp is None:
            raise StudioError(f"{char} has no voice profile")
        take = self.project.load_take(take_id)
        clip = self.project.load_clip(take.clip_id)
        if char not in continuity.speakers_in_clip(clip):
            raise StudioError(f"{self.display_name(char)} does not speak in {take.clip_id}")
        if vp.anchor_take == take_id:
            return f"{char}: {take_id} is already the voice anchor"
        if vp.anchor_take:
            fan = AnchorGraph.build(self.project).fanout_voice(char)
            self.state.request_confirm(vp.id, action="reanchor_voice",
                                       fanout={"fanout": fan.model_dump(mode="json"), "character_id": char,
                                               "take_id": take_id})
            return (f"Re-anchoring {self.display_name(char)}'s voice is a Vision-tier change.\n{describe_fanout(fan)}\n"
                    f"Reply `confirm {vp.id}` to re-anchor (takes are re-scored, never regenerated).")
        self._write_anchor(vp, take, confirm_decision_id=None)
        return f"{self.display_name(char)}'s voice anchor is {take_id}; future takes are scored against it."

    def _write_anchor(self, vp: Any, take: Take, *, confirm_decision_id: str | None) -> None:
        from .qa_critic import rescore_voice

        wav = self.project.abs(take.dialogue_wav_uri) if take.dialogue_wav_uri else None
        if wav is None or not wav.is_file():
            from .media import extract_wav

            wav = extract_wav(self.project.abs(take.video_uri or ""), self.project.take_dir(take.id) / "dialogue.wav")
            take.dialogue_wav_uri = self.project.rel(wav)
            self.project.save_take(take)
        rels = self.project.promote_library_files(("voices", vp.character_id), {"anchor.wav": wav},
                                                  provenance=generated_by("agy.voice_anchor", notes=take.id))
        vp.anchor_take = take.id
        vp.anchor_wav = rels["anchor.wav"]
        self.project.save_voice_profile(vp, confirm_decision_id=confirm_decision_id)
        self.state.release(vp.id)
        rescore_voice(self.project, self.registry, vp.character_id)

    def _apply_g5(self, sid: str | None, response: Any, decisions: list[Any]) -> list[str]:
        assert sid
        cur_items = list((self._cursor(self._gate_key(GateId.G5, sid)) or {}).get("item_ids", []))
        explicit = {c.item_id for c in response.choices}
        lines: list[str] = []
        regen: list[tuple[str, str, int | None]] = []
        for ch in response.choices:
            cid = ch.item_id
            did = self._decision_for(decisions, cid)
            if ch.action == "choose":
                lines += self._select_take(cid, norm_take_id(cid, ch.value) or "", decision_id=did)
            elif ch.action == "ok":
                best = self._recommended_take(cid)
                if best is None:
                    regen.append((cid, ch.steer or "", None))
                    lines.append(f"{cid}: no acceptable take to approve — regenerating")
                else:
                    lines += self._select_take(cid, best.id, decision_id=did)
            elif ch.action in ("regen", "reject"):
                regen.append((cid, ch.steer or ch.value or "", None))
            elif ch.action == "alternatives":
                n = int(ch.value) if str(ch.value or "").isdigit() else 2
                regen.append((cid, ch.steer or "", n))
        if response.approve_all:
            for cid in cur_items:
                if cid not in explicit and (best := self._recommended_take(cid)) is not None:
                    lines += self._select_take(cid, best.id, decision_id=self._decision_for(decisions, cid))
        if response.reject_all:
            regen += [(cid, response.free_text, None) for cid in cur_items if cid not in explicit]
        for char, tid in response.anchors.items():
            lines.append(self.set_voice_anchor(char, norm_take_id(self.project.load_take(tid).clip_id, tid) or tid,
                                               decision_id=self._decision_for(decisions, char)))
        selected_clips = [self.project.load_clip(c) for c in cur_items if not self._needs_take(self.project.load_clip(c))]
        for char in self._speakers_needing_anchor(selected_clips):
            first = next((c.selected_take_id for c in selected_clips if char in continuity.speakers_in_clip(c)), None)
            q = self.ask(sid, f"anchor_{char}", f"Pick {self.display_name(char)}'s voice anchor — all future takes "
                                                f"will be scored against it (suggested: {first}). Reply "
                                                f"`anchor {char}=<take id>`.")
            lines.append(f"Open question {q.id}: voice anchor for {self.display_name(char)}")
        for cid, steer, n in regen:
            lines.append(self._prepare_regen(cid, steer, self._decision_for(decisions, cid), n_takes=n))
        if regen:
            self.state.transition(sid, S.LINT, reason=f"G5: regenerate {', '.join(c for c, _, _ in regen)}")
        elif self.frontier(sid):
            self.state.transition(sid, S.LINT, reason="G5 closed; next anchor wave / remaining picks")
        else:
            self.state.transition(sid, S.ASSEMBLY, reason="G5: every clip selected")
        lines.append(self.advance(sid).text())
        return lines

    def apply_steer_to_clip(self, clip: GenClip, steer: str) -> GenClip:
        """Fold a human steer into the clip spec (delivery notes + performance) so the new payload differs."""
        steer = (steer or "").strip()
        if not steer:
            return clip
        for ln in clip.spec.audio_spec.all_lines():
            if steer not in ln.delivery:
                ln.delivery = f"{ln.delivery}; {steer}" if ln.delivery else steer
        if steer not in clip.performance_state:
            clip.performance_state = f"{clip.performance_state}; direction: {steer}" if clip.performance_state \
                else f"direction: {steer}"
        return clip

    def _prepare_regen(self, cid: str, steer: str, decision_id: str | None, *, n_takes: int | None = None,
                       keep_current: bool = False) -> str:
        clip = self.project.load_clip(cid)
        if steer:
            self.apply_steer_to_clip(clip, steer)
        if clip.status == ClipStatus.SELECTED and not keep_current:
            clip.status = ClipStatus.NEEDS_RELINT
        self.project.save_clip(clip)
        if not keep_current:
            self._reject_takes([t for t in self._candidates(cid) if t.status != TakeStatus.SELECTED],
                               f"regen requested ({decision_id}): {steer}")
        if n_takes:
            self.state.set_cursor(f"ntakes:{cid}", int(n_takes))
        key = f"regen_attempt:{decision_id}" if decision_id else None
        if key and self._cursor(key) is not None:
            attempt = int(self._cursor(key))  # resumed after a kill: same attempt => same seeds => dedupe
            self.state.set_cursor(f"attempt:{cid}", attempt)
        else:
            attempt = self._bump_attempt(cid)
            if key:
                self.state.set_cursor(key, attempt)
        return f"{cid}: regenerating (attempt {attempt}){' — ' + steer if steer else ''}"

    # ---- regen requests (chat `regen`, Resolve REGEN markers) --------------------------------
    def regen(self, clip_id: str, steer: str = "", *, source: str = "chat", user: str | None = None,
              record: bool = True, keep_current: bool = False) -> str:
        ids = parse_artifact_id(clip_id)
        if not ids["clip"]:
            raise StudioError(f"{clip_id} is not a clip id (e.g. SC01_SH02_C01)")
        self.project.load_clip(clip_id)
        decision_id = None
        if record:
            d = self.decisions.record(tier=DecisionTier.POLISH, target_artifact_id=clip_id,
                                      action=DecisionAction.REJECT, payload={"kind": "regen", "steer": steer},
                                      rationale=steer, source=source, user=user or self.user)
            decision_id = d.id
        self.state.enqueue_regen(clip_id, steer=steer, decision_id=decision_id, source=source)
        if keep_current:
            self.state.set_cursor(f"keep_current:{decision_id or clip_id}", True)
        return "\n".join(self.process_regens(only=[clip_id]))

    def _dequeue_regen(self, item: dict[str, Any]) -> None:
        with self.state.mutate() as st:
            st.regen_queue = [r for r in st.regen_queue if not (r.get("clip_id") == item.get("clip_id") and
                                                              r.get("decision_id") == item.get("decision_id"))]

    def process_regens(self, only: Iterable[str] | None = None) -> list[str]:
        """Work the regen queue. Items leave the queue only after they are handled (kill-safe)."""
        wanted = set(only or [])
        lines: list[str] = []
        for item in list(self.state.load().regen_queue):
            cid = str(item.get("clip_id"))
            if wanted and cid not in wanted:
                continue
            sid = parse_artifact_id(cid)["scene"]
            stage = self.state.ensure_scene(sid).stage
            steer = str(item.get("steer") or "")
            did = item.get("decision_id")
            keep = bool(self._cursor(f"keep_current:{did or cid}", False))
            if stage in (S.HANDED_OFF, S.DONE, S.ASSEMBLY):
                lines += self._regen_after_handoff(sid, cid, steer, did, keep_current=keep)
            elif stage in (S.AWAITING_G5, S.QA, S.GENERATING, S.LINT):
                lines.append(self._prepare_regen(cid, steer, did, keep_current=keep))
                if stage == S.AWAITING_G5:
                    self.state.transition(sid, S.LINT, reason=f"regen {cid}")
                lines.append(self.advance(sid).text())
            else:  # not generated yet: the steer simply shapes the first generation
                clip = self.apply_steer_to_clip(self.project.load_clip(cid), steer)
                self.project.save_clip(clip)
                lines.append(f"{cid}: not generated yet — your note will shape its first takes")
            self._dequeue_regen(item)
        return lines

    def _regen_after_handoff(self, sid: str, cid: str, steer: str, decision_id: str | None, *,
                             keep_current: bool = False) -> list[str]:
        """Post-handoff regen: new takes go to /REGEN/ beside the human's cut; their timeline is never touched."""
        from .linter import lint_shot
        from .qa_critic import rank_takes, score_take
        from .shot_generator import generate_clip

        lines = [self._prepare_regen(cid, steer, decision_id, keep_current=True)]
        before = {t.id for t in self.project.list_takes(cid)}
        shot_id = parse_artifact_id(cid)["shot"]
        reports = {c.id: r for c, r in lint_shot(self.project, self.registry, shot_id, apply_fixes=True, save=True)}
        rep = reports.get(cid)
        if rep is None or rep.blocking:
            detail = "; ".join(f"{f.rule} {f.message}" for f in (rep.findings if rep else []) if f.severity == "error")
            q = self.ask(sid, f"lint_{cid}", f"{cid}: regen blocked by pre-flight lint — {detail or 'no report'}")
            return lines + [f"{cid}: lint blocked the regen (question {q.id}); nothing was dispatched"]
        takes = generate_clip(self.project, self.registry, self.state, cid, n_takes=self._n_takes(cid),
                              budget=self.budget, fake=self.fake, seed_base=self._seed_base(cid, self._attempt(cid)))
        new = [t for t in takes if t.id not in before and t.dispatch_status == DispatchStatus.COMPLETED]
        clip = self.project.load_clip(cid)
        for t in new:
            score_take(self.project, self.registry, clip, t, prev_take=self._prev_take(clip), fake=self.fake)
        if new:
            rank_takes(self.project, clip, self._candidates(cid))
        clip = self.project.load_clip(cid)
        if clip.status == ClipStatus.NEEDS_RELINT and clip.selected_take_id:
            clip.status = ClipStatus.SELECTED  # the human's pick stays in force until they swap in Resolve
            self.project.save_clip(clip)
        if not new:
            return lines + [f"{cid}: no new take finished — will retry on the next pass"]
        handoff = self.project.load_handoff(sid)
        if handoff is None:
            return lines + [f"{cid}: {len(new)} new take(s) ready (scene not in Resolve yet)"]
        where = self.bridge.deliver_regen(sid, [self.project.load_take(t.id) for t in new])
        lines.append(f"{cid}: {len(new)} new take(s) in {', '.join(where) if isinstance(where, list) else where} — "
                     f"your timeline {handoff.timeline_name} was not touched; swap them in when you like.")
        return lines

    # ---- ASSEMBLY / HANDOFF / G6 -------------------------------------------------------------
    def ensure_cues(self, sid: str) -> list[str]:
        cues = self.cue_plan(sid)
        made = []
        if not cues:
            return made
        music = None
        for cue in cues:
            cue_id = validate_id(str(cue.get("cue_id")), "cue id")
            dest = self.project.path("scenes", sid, "audio", "cues", f"{cue_id}.wav")
            if dest.exists():
                continue
            music = music or self.registry.adapter("existing_music_gen", project=self.project, fake=self.fake)
            dest.parent.mkdir(parents=True, exist_ok=True)
            music.generate_cue(str(cue.get("prompt", "")), float(cue.get("duration_s") or 30.0), dest)
            made.append(cue_id)
        return made

    def ensure_sfx_supplement(self, sid: str) -> list[str]:
        made = []
        sfx = None
        for clip in self.active_clips(sid):
            for k, flag in enumerate([f for f in clip.flags if f.startswith("sfx_supplement:")], start=1):
                dest = self.project.path("scenes", sid, "audio", "sfx_supplement", f"{clip.id}_{k:02d}.wav")
                if dest.exists():
                    continue
                sfx = sfx or self.registry.adapter("existing_sfx_gen", project=self.project, fake=self.fake)
                dest.parent.mkdir(parents=True, exist_ok=True)
                sfx.generate(flag.split(":", 1)[1], 2.0, dest)
                made.append(dest.name)
        return made

    def _h_assembly(self, sid: str, rep: StepReport) -> bool:
        from .editor import assemble_scene

        try:
            cues = self.ensure_cues(sid)
            if cues:
                rep.lines.append(f"{sid}: music cue(s) generated: {', '.join(cues)}")
            sfx = self.ensure_sfx_supplement(sid)
            if sfx:
                rep.lines.append(f"{sid}: {len(sfx)} hard-effect stem(s) for the SFX supplement")
        except Exception as exc:  # noqa: BLE001 — music/SFX are layered extras; never block the picture cut
            rep.lines.append(f"{sid}: music/SFX supplement skipped ({exc})")
        cut = assemble_scene(self.project, sid)
        try:
            handoff = self.bridge.deliver_scene(sid, cut)
        except Exception as exc:  # noqa: BLE001
            # The .mp4 / OTIO deliverables exist on disk; what failed is pushing them into the editor. Say exactly
            # what to do (the FailoverTransport message includes the one-click manual route) and block G6 on it —
            # `unblock <SID>` + `advance <SID>` retries delivery once Resolve is ready; nothing is regenerated.
            where = f"post/ and scenes/{sid}/assembly/ (OTIO: {cut.otio_path})" if cut.otio_path else "post/"
            rep.lines.append(f"{sid}: scene cut assembled and saved under {where}.\n"
                             f"DaVinci Resolve delivery did not complete:\n{exc}\n"
                             f"When Resolve is ready, reply `unblock {sid}` and `advance {sid}` to push it again.")
            return self._block(sid, rep, f"resolve: {str(exc).splitlines()[0][:160]}")
        scene = self.scene(sid)
        scene.status = SceneStatus.HANDED_OFF
        scene.resolve = scene.resolve.model_copy(update={
            "timeline_name": handoff.timeline_name, "bin_path": handoff.bin_path,
            "handoff_fingerprint": handoff.fingerprint, "ownership": Ownership.HUMAN})
        self.project.save_scene(scene)
        self.state.transition(sid, S.HANDED_OFF, reason=f"delivered {handoff.timeline_name}")
        rep.lines.append(self.bridge.handoff_message(handoff))
        return False

    def approve_scene(self, sid: str, *, source: str = "chat", user: str | None = None,
                      decision: Any = None) -> str:
        """G6: the human approved their edited timeline (Resolve APPROVED marker or chat `approve SC`)."""
        from .vault import freeze_and_backup_scene

        validate_id(sid, "scene id")
        stage = self.state.ensure_scene(sid).stage
        if stage not in (S.HANDED_OFF, S.DONE):
            raise StudioError(f"{sid} is at {stage.value}; it can be approved after it is handed off in Resolve.")
        d = decision or self.decisions.record(tier=DecisionTier.CRAFT, gate=GateId.G6.value, target_artifact_id=sid,
                                              action=DecisionAction.APPROVE, payload={}, source=source,
                                              user=user or self.user)
        self.state.set_gate(sid, GateId.G6, "passed")
        exit_state = self._exit_state(sid)
        save_versioned(self.project.path("scenes", sid, "exit_state.yaml"), exit_state)
        scene = self.scene(sid)
        scene.status = SceneStatus.DONE
        self.project.save_scene(scene)
        self.state.transition(sid, S.DONE, reason="G6 approved", decision_id=d.id)
        freeze_info = {}
        try:
            freeze_info = freeze_and_backup_scene(self.project, sid, decision_id=d.id)
        except Exception:  # noqa: BLE001
            pass
        lines = [f"{sid} approved ({source}); exit state recorded. Scene done."]
        if freeze_info.get("frozen_manifest"):
            lines.append(f"Frozen timeline: {freeze_info['frozen_manifest']} · backup: {freeze_info.get('backup_otio')} "
                         f"· rejected takes vaulted: {len(freeze_info.get('archived_rejected_takes', []))}")
        lines += self._flag_next_scene(sid, exit_state)
        return "\n".join(lines)

    def _exit_state(self, sid: str) -> dict[str, Any]:
        scene = self.scene(sid)
        clips = self.active_clips(sid)
        shots = [s.id for s in self.project.list_shots(sid)]
        states: list[str] = list(scene.entity_states_active)
        last_frame = None
        if clips:
            last = clips[-1]
            states = continuity.active_states_at(scene, shots, last.shot_id, last.sequence_index)
            if last.selected_take_id:
                last_frame = self.project.load_take(last.selected_take_id).last_frame_uri
        handoff = self.project.load_handoff(sid)
        return {"scene_id": sid, "approved_at": now_iso(), "declared": scene.exit_state,
                "entity_states_at_end": states, "last_frame": last_frame,
                "timeline": handoff.timeline_name if handoff else None,
                "generated_by": generated_by("agy.g6").model_dump(mode="json")}

    def _flag_next_scene(self, sid: str, exit_state: dict[str, Any]) -> list[str]:
        sched = self.schedule()
        if sid not in sched or sched.index(sid) + 1 >= len(sched):
            return []
        nxt = sched[sched.index(sid) + 1]
        ws_path = self.project.path("scenes", nxt, "working_set.yaml")
        if not ws_path.exists():
            return []
        used = (read_record(ws_path) or {}).get("prev_scene_exit_state") or {}
        fresh = {k: v for k, v in exit_state.items() if k in ("declared", "entity_states_at_end")}
        if not used or all(used.get(k) == v for k, v in fresh.items()) or used == exit_state.get("declared"):
            return []
        scene = self.scene(nxt)
        scene.status = SceneStatus.NEEDS_REVIEW
        self.project.save_scene(scene)
        q = self.ask(nxt, "entry_review", f"{sid}'s approved ending differs from what {nxt} was planned against. "
                                          f"Review {nxt}'s opening (`show working_set {nxt}`)?")
        return [f"{nxt} needs review: its entry state changed (question {q.id})"]

    # ---- round-trip (Resolve -> AGY) ---------------------------------------------------------
    def poll(self) -> list[str]:
        from .roundtrip import events_to_decisions, poll_resolve

        events = poll_resolve(self.project, self.db)
        if not events:
            return []
        specs = events_to_decisions(events)
        lines = []
        for ev, spec in zip(events, specs):
            spec = dict(spec)
            d = self.decisions.record(tier=DecisionTier(spec.get("tier", "craft")),
                                      target_artifact_id=spec.get("target_artifact_id") or ev.scene_id,
                                      action=DecisionAction(spec.get("action", "annotate")),
                                      payload=dict(spec.get("payload") or {}), rationale=str(ev.text or ""),
                                      gate=spec.get("gate"), source="resolve", user=self.user)
            lines += self._apply_event(ev, d)
        return lines

    def _apply_event(self, ev: Any, decision: Any) -> list[str]:
        kind = ev.kind
        if kind == "regen_marker" and ev.clip_id:
            self.state.enqueue_regen(ev.clip_id, steer=ev.text or "", decision_id=decision.id, source="resolve")
            return [f"Resolve REGEN marker on {ev.clip_id}: \"{ev.text or ''}\" — queued"]
        if kind == "take_swap" and ev.clip_id and ev.take_id:
            try:
                out = self._select_take(ev.clip_id, ev.take_id, decision_id=decision.id)
            except Exception as exc:  # noqa: BLE001
                return [f"Resolve take swap on {ev.clip_id} noted but not applied ({exc})"]
            return [f"Resolve: you swapped {ev.clip_id} to {ev.take_id}", *out[1:]]
        if kind == "approved_marker" and ev.scene_id:
            try:
                return [self.approve_scene(ev.scene_id, source="resolve", decision=decision)]
            except StudioError as exc:
                return [f"APPROVED marker on {ev.timeline}: {exc}"]
        if kind == "note_marker":
            q = self.ask(ev.scene_id, f"note_{decision.id}",
                         f"Resolve NOTE on {ev.timeline}: \"{ev.text}\" — keep it as a standing preference for "
                         f"{ev.scene_id}? (yes/no)", options=("yes", "no"))
            self.state.set_cursor(f"note:{q.id}", {"scene_id": ev.scene_id, "text": ev.text, "decision_id": decision.id})
            return [f"Resolve NOTE on {ev.timeline} — asked whether to keep it as a preference ({q.id})"]
        return [f"Resolve: {kind.replace('_', ' ')} on {ev.timeline or ev.scene_id} logged (no regeneration)"]

    def answer(self, qid: str, text: str) -> str:
        q = next((q for q in self.questions.all() if q.id == qid), None)
        if q is None:
            raise StudioError(f"no question {qid}")
        d = self.decisions.record(tier=DecisionTier.CRAFT, target_artifact_id=q.scene_id or self.project.project_id,
                                  action=DecisionAction.ANSWER_QUESTION, payload={"question_id": qid, "answer": text},
                                  rationale=text, user=self.user)
        self.questions.answer(qid, text, decision_id=d.id)
        lines = [f"{qid} answered."]
        note = self._cursor(f"note:{qid}")
        if note and text.strip().lower() in ("yes", "y", "keep", "ok"):
            p = self.prefs.add(scope=f"scene:{note['scene_id']}", text=note["text"],
                               source_decision_ids=[note["decision_id"], d.id])
            lines.append(f"Kept as preference {p.id} for {note['scene_id']}.")
        kind = (self._cursor(f"qkind:{qid}") or {}).get("kind", "")
        if kind.startswith("anchor_") and "=" in text:
            char, _, tid = text.replace("anchor", "").strip().partition("=")
            lines.append(self.set_voice_anchor(char.strip(), tid.strip(), decision_id=d.id))
        if q.scene_id:
            lines.append(self.advance(q.scene_id).text())
        return "\n".join(x for x in lines if x)

    # ---- Vision-tier confirm, locks, overrides -----------------------------------------------
    def propose_pack_change(self, state_id: str, files: dict[str, str | Path], *, why: str = "") -> str:
        """Editing a LOCKED pack: stage the new images, print the fan-out, wait for `confirm <state_id>`."""
        from .invalidation import AnchorGraph, describe_fanout

        if state_id not in self.project.entity_state_index():
            raise StudioError(f"{state_id} is not an entity state")
        staged: dict[str, str] = {}
        dest_dir = self.project.path("library_staging", state_id)
        for role, src in files.items():
            if role not in PACK_ROLES:
                raise StudioError(f"unknown pack role {role}")
            src_path = Path(src).expanduser().resolve()
            if not src_path.is_file() or src_path.suffix.lower() not in IMAGE_EXTS or \
                    src_path.stat().st_size > MAX_IMPORT_BYTES:
                raise StudioError(f"{src} must be an image file under {MAX_IMPORT_BYTES // (1024 * 1024)} MB")
            dest = next_version_path(dest_dir, role, src_path.suffix.lower())
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest)
            staged[role] = self.project.rel(dest)
        fan = AnchorGraph.build(self.project).fanout(state_id)
        self.state.request_confirm(state_id, action="edit_pack",
                                   fanout={"fanout": fan.model_dump(mode="json"), "files": staged, "why": why})
        return (f"{state_id} is locked. Changing it affects:\n{describe_fanout(fan)}\n"
                f"Reply `confirm {state_id}` to apply (affected clips become stale; nothing regenerates by itself).")

    def confirm(self, artifact_id: str, *, user: str | None = None) -> str:
        from .invalidation import FanOut, apply_invalidation
        from .qa_critic import rescore_voice

        pending = self.state.consume_confirm(artifact_id)  # raises ConfirmRequiredError if no fan-out was shown
        d = self.decisions.record(tier=DecisionTier.VISION, target_artifact_id=artifact_id,
                                  action=DecisionAction.APPROVE, payload={"kind": "confirm", **pending},
                                  rationale=f"confirm {artifact_id}", user=user or self.user)
        info = pending.get("fanout") or {}
        fan = FanOut.model_validate(info["fanout"]) if info.get("fanout") else None
        action = pending.get("action")
        if action == "reanchor_voice":
            char = info["character_id"]
            vp = self.project.load_voice_profile(char)
            self._write_anchor(vp, self.project.load_take(info["take_id"]), confirm_decision_id=d.id)
            touched = apply_invalidation(self.project, self.state, fan, reason="voice re-anchor",
                                         decision_id=d.id) if fan else []
            rescored = rescore_voice(self.project, self.registry, char)
            return (f"{self.display_name(char)} re-anchored to {info['take_id']}; {len(touched)} clip(s) flagged "
                    f"for voice review, {len(rescored)} take(s) re-scored (none regenerated).")
        if action == "edit_pack":
            pair = self.project.entity_state_index()[artifact_id]
            ent = pair[0]
            abs_files = {r: self.project.abs(u) for r, u in info["files"].items()}
            pack = self.project.promote_pack(artifact_id, abs_files, approved_by=user or self.user,
                                             provenance=generated_by("agy.confirm", notes=d.id))
            for s in ent.states:
                if s.id == artifact_id:
                    s.reference_pack = pack
                    s.approval_status = ApprovalStatus.LOCKED
                    s.version += 1
            self.project.save_entity(ent, confirm_decision_id=d.id)
            touched = apply_invalidation(self.project, self.state, fan, reason=f"pack change {artifact_id}",
                                         decision_id=d.id) if fan else []
            return f"{artifact_id}: new pack version locked; {len(touched)} artifact(s) marked stale."
        return f"{artifact_id}: confirmed ({action})."

    def lock(self, artifact_id: str, *, user: str | None = None) -> str:
        """`lock <id>` — Vision for Bible records, Craft for shots/takes."""
        try:
            ids = parse_artifact_id(artifact_id)
        except ValueError:
            ids = {}
        tier = DecisionTier.CRAFT if ids else DecisionTier.VISION
        d = self.decisions.record(tier=tier, target_artifact_id=artifact_id, action=DecisionAction.LOCK, payload={},
                                  user=user or self.user)
        if ids.get("take"):
            return "\n".join(self._select_take(ids["clip"], artifact_id, decision_id=d.id))
        if ids.get("shot") and not ids.get("clip"):
            shot = self.project.load_shot(artifact_id)
            shot.approval_status = ApprovalStatus.LOCKED
            self.project.save_shot(shot)
            return f"{artifact_id}: shot locked"
        if artifact_id == "STYLE_LOCK":
            style = self.project.load_style_lock()
            if style is None:
                raise StudioError("there is no StyleLock yet")
            style.approval_status = ApprovalStatus.LOCKED
            self.project.save_style_lock(style)
            return "StyleLock locked"
        if artifact_id.startswith("VOICE_"):
            char = next((e.id for e in self.project.list_entities() if e.voice_profile_id == artifact_id), None)
            vp = self.project.load_voice_profile(char) if char else None
            if vp is None:
                raise StudioError(f"no voice profile {artifact_id}")
            vp.approval_status = ApprovalStatus.LOCKED
            self.project.save_voice_profile(vp)
            return f"{artifact_id}: voice profile locked in library/"
        if (self.project.path("bible", "entities", f"{artifact_id}.yaml")).exists():
            ent = self.project.load_entity(artifact_id)
            ent.approval_status = ApprovalStatus.LOCKED
            self.project.save_entity(ent)
            return f"{artifact_id}: Bible entry locked (packs lock at G2)"
        if (self.project.path("bible", "environments", f"{artifact_id}.yaml")).exists():
            return f"{artifact_id}: environments lock at G2 together with their geometry (`status`)"
        raise StudioError(f"don't know how to lock {artifact_id}")

    def set_model(self, sid: str, model_id: str, *, user: str | None = None) -> str:
        """`model SC12 omni|veo` — override; clips without a selected take are re-linted for the new model."""
        validate_id(sid, "scene id")
        spec = self.registry.get(model_id)
        if spec.kind.value != "video_av":
            raise StudioError(f"{model_id} is not a video model")
        d = self.decisions.record(tier=DecisionTier.CRAFT, target_artifact_id=sid, action=DecisionAction.OVERRIDE,
                                  payload={"model": model_id}, rationale=f"model {sid} {model_id}",
                                  user=user or self.user)
        scene = self.scene(sid)
        scene.model_override = model_id
        self.project.save_scene(scene)
        other = next((m.id for m in self.registry.video_models() if m.id != model_id), None)
        n = 0
        for clip in self.active_clips(sid):
            if self._needs_take(clip):
                clip.model_target = clip.model_target.model_copy(update={"preferred": model_id, "fallback": other})
                clip.status = ClipStatus.NEEDS_RELINT if clip.status != ClipStatus.STALE else clip.status
                self.project.save_clip(clip)
                n += 1
        return f"{sid}: model set to {model_id} ({d.id}); {n} clip(s) will be re-linted before generation."

    def pin_model(self, model_id: str, provider_model: str, *, user: str | None = None) -> str:
        self.registry.get(model_id)
        cfg = self.project.config
        old = cfg.model_pins.get(model_id)
        cfg.model_pins = {**cfg.model_pins, model_id: provider_model}
        self.project.save_config(cfg)
        self.decisions.record(tier=DecisionTier.VISION, target_artifact_id=self.project.project_id,
                              action=DecisionAction.OVERRIDE, payload={"model_pin": {model_id: provider_model}},
                              user=user or self.user)
        affected = [c.id for c in self.project.iter_all_clips() for t in self.project.list_takes(c.id)
                    if t.model == model_id and t.provider_model and t.provider_model != provider_model]
        return (f"{model_id} pinned to {provider_model} (was {old}). WARNING: {len(set(affected))} clip(s) have takes "
                f"from another version — they stay as they are; new takes use the pin.")

    def set_open_decision(self, key: str, value: str, *, user: str | None = None) -> str:
        raw_key = key.strip()
        key = raw_key.upper()
        if key not in OPEN_DECISION_FIELDS:
            by_field = {fname.upper(): k for k, (fname, _) in OPEN_DECISION_FIELDS.items()}
            key = by_field.get(key, key)
        if key not in OPEN_DECISION_FIELDS:
            raise StudioError(f"unknown open decision {raw_key}; use OD1..OD7")
        field_name, accepted = OPEN_DECISION_FIELDS[key]
        norm = value.strip().lower()
        if norm not in accepted:
            raise StudioError(f"{key} accepts {sorted(accepted)}")
        cfg = self.project.config
        od = cfg.settings.open_decisions.model_copy(update={field_name: accepted[norm], "confirmed_by_human": True})
        cfg.settings = cfg.settings.model_copy(update={"open_decisions": od})
        self.project.save_config(cfg)
        self.decisions.record(tier=DecisionTier.VISION, target_artifact_id=key, action=DecisionAction.CHOOSE,
                              payload={field_name: accepted[norm]}, user=user or self.user)
        return f"{key} ({field_name}) = {accepted[norm]}"

    # ---- pre-production / G1 -----------------------------------------------------------------
    def ingest(self, script_path: Path, *, episode: int | None = None) -> str:
        from .preproduction import build_schedule, compute_breakdown, ingest_script

        res = ingest_script(self.project, Path(script_path), episode=episode)
        for scene in self.project.list_scenes():
            if scene.breakdown is None:
                scene.breakdown = compute_breakdown(self.project, self.registry, scene)
                self.project.save_scene(scene)
        schedule, pilot = build_schedule(self.project, self.project.list_scenes())
        self.state.set_schedule(schedule, pilot)
        lines = [f"Script ingested: {len(res.scenes)} scene(s); pilot {pilot}; schedule {' → '.join(schedule)}"]
        if res.new_characters:
            lines.append("new characters: " + ", ".join(res.new_characters))
        for amb in res.ambiguities:
            q = self.ask(None, f"ambiguity_{stable_hash(str(amb), 6)}", f"Script question: {amb}")
            lines.append(f"question {q.id}: {amb}")
        return "\n".join(lines)

    def open_g1(self) -> str:
        from .bible_hierarchy import build_global_context_bible
        from .gates import GateItem, GatePacket
        from .preproduction import build_schedule, g1_packet_items

        st = self.state.load()
        if not st.schedule:
            schedule, pilot = build_schedule(self.project, self.project.list_scenes())
            self.state.set_schedule(schedule, pilot)
        try:
            build_global_context_bible(self.project)
        except Exception:  # noqa: BLE001
            pass
        items = []
        for raw in g1_packet_items(self.project):
            d = dict(raw)
            item_id = f"OD{d['number']}" if d.get("kind") == "open_decision" and d.get("number") else str(d.get("item_id") or d.get("id"))
            items.append(GateItem(item_id=item_id, summary=str(d.get("summary", "")),
                                  options=[str(o) for o in d.get("options", [])],
                                  media=[str(m) for m in d.get("media", [])], flags=[str(f) for f in d.get("flags", [])]))
        packet = GatePacket(gate=GateId.G1, scene_id=None,
                            title="Bible, state timeline, StyleLock, voice profiles, schedule", items=items,
                            blocks=["every scene"],
                            expected_format='`approve all`, or per item: CHAR_RISHI:regen "older, heavier build", '
                                            'OD3:choose on',
                            defaults="Vision tier — never auto-proceeds; silence means AGY waits.")
        text = self._open_gate(packet)
        self.state.set_global_stage(GlobalStage.AWAITING_G1, reason="G1 opened")
        return text

    def _apply_g1(self, scene_id: str | None, response: Any, decisions: list[Any]) -> list[str]:
        from .bible_hierarchy import build_global_context_bible

        lines: list[str] = []
        redo = []
        od_names = set(OPEN_DECISION_FIELDS) | {fname.upper() for fname, _ in OPEN_DECISION_FIELDS.values()}
        for ch in response.choices:
            if ch.item_id.upper() in od_names and ch.action == "choose" and ch.value:
                lines.append(self.set_open_decision(ch.item_id, str(ch.value)))
            elif ch.action in ("regen", "reject"):
                redo.append(f"{ch.item_id} ({ch.steer or ch.value or 'revise'})")
        if response.reject_all or redo:
            self.state.set_gate(None, GateId.G1, "rejected")
            lines.append("AGY will revise " + (", ".join(redo) if redo else "the Bible") +
                         " and re-open G1 (`g1`). Your notes are saved as preferences.")
            return lines
        if not (response.approve_all or all(c.action in ("ok", "choose") for c in response.choices)):
            return lines + ["G1 stays open — reply `approve all` when the Bible is right."]
        lines += self._lock_bible(self._decision_for(decisions, self.project.project_id))
        try:
            build_global_context_bible(self.project)
        except Exception:  # noqa: BLE001
            pass
        self.state.set_gate(None, GateId.G1, "passed")
        cfg = self.project.config
        od = cfg.settings.open_decisions.model_copy(update={"confirmed_by_human": True})
        cfg.settings = cfg.settings.model_copy(update={"open_decisions": od})
        self.project.save_config(cfg)
        self.state.set_global_stage(GlobalStage.SCENE_LOOP, reason="G1 approved")
        lines.append("G1 approved — Bible, StyleLock and voice profiles are locked. Starting the scene loop.")
        lines.append(self.tick(poll=False))
        return lines

    def _lock_bible(self, decision_id: str | None) -> list[str]:
        out = []
        style = self.project.load_style_lock()
        if style is not None and style.approval_status != ApprovalStatus.LOCKED:
            style.approval_status = ApprovalStatus.LOCKED
            self.project.save_style_lock(style)
            out.append("StyleLock locked")
        for ent in self.project.list_entities():
            if ent.approval_status != ApprovalStatus.LOCKED:
                ent.approval_status = ApprovalStatus.LOCKED
                self.project.save_entity(ent)
            if ent.type.value == "character":
                vp = self.project.load_voice_profile(ent.id)
                if vp is not None and vp.approval_status != ApprovalStatus.LOCKED:
                    vp.approval_status = ApprovalStatus.LOCKED
                    self.project.save_voice_profile(vp)
                    out.append(f"{vp.id} locked")
        return out

    # ---- global post / G7 ---------------------------------------------------------------------
    def master(self) -> str:
        from .editor import assemble_master
        from .gates import GateItem, GatePacket

        approved = [s.id for s in self.project.list_scenes() if s.status in (SceneStatus.DONE,
                                                                             SceneStatus.HUMAN_APPROVED)]
        if not approved:
            raise StudioError("No scene is approved yet — mark APPROVED in Resolve or say `approve SC01`.")
        master_path = assemble_master(self.project, approved)
        name = self.bridge.create_master(master_path)
        notes = self._post_notes(approved)
        self.state.set_global_stage(GlobalStage.POST, reason="master assembled")
        packet = GatePacket(gate=GateId.G7, scene_id=None, title="Picture lock", items=[
            GateItem(item_id="MASTER", summary=f"{name} nests {len(approved)} approved scene timeline(s): "
                                               f"{', '.join(approved)}", media=[str(master_path)])],
            notes=notes, blocks=["delivery"], expected_format="`approve all` to lock picture, or notes",
            defaults="Vision tier — never auto-proceeds.")
        text = self._open_gate(packet)
        self.state.set_global_stage(GlobalStage.AWAITING_G7, reason="G7 opened")
        return text

    def _post_notes(self, scene_ids: list[str]) -> list[str]:
        style = self.project.load_style_lock()
        color = ["# Color notes (Colorist proposal — notes only, never applied to your timelines)", ""]
        mix = ["# Mix notes (Mixer proposal)", "",
               "- A1–A2: native dialogue + ambience from each take (keep dialogue around -12 dBFS peaks)",
               "- A3–A4: SFX supplement stems (hard effects only)",
               "- A7–A8: music cues ducked ~8–10 dB under dialogue; no music inside takes", ""]
        cue_rows = ["scene,cue_id,start_s,duration_s,file"]
        for sid in scene_ids:
            scene = self.scene(sid)
            color.append(f"- {sid} ({scene.slug}): lighting {scene.lighting_setup_id or '—'}; grade ref "
                         f"{style.lut_or_grade_ref if style else '—'}")
            for cue in self.cue_plan(sid):
                f = self.project.path("scenes", sid, "audio", "cues", f"{cue.get('cue_id')}.wav")
                cue_rows.append(f"{sid},{cue.get('cue_id')},{cue.get('start_s', 0)},{cue.get('duration_s')},"
                                f"{self.project.rel(f) if f.exists() else ''}")
        outs = []
        for name, body in (("color_notes", "\n".join(color) + "\n"), ("mix_notes", "\n".join(mix) + "\n")):
            p = next_version_path(self.project.path("post"), name, ".md")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")
            outs.append(self.project.rel(p))
        p = next_version_path(self.project.path("post"), "cue_sheet", ".csv")
        p.write_text("\n".join(cue_rows) + "\n", encoding="utf-8")
        outs.append(self.project.rel(p))
        return [f"notes: {', '.join(outs)}"]

    def _apply_g7(self, scene_id: str | None, response: Any, decisions: list[Any]) -> list[str]:
        if not response.approve_all and not any(c.action in ("ok", "choose") for c in response.choices):
            self.state.set_global_stage(GlobalStage.POST, reason="G7 notes")
            return ["Picture not locked — AGY will address your notes and offer a new MASTER version."]
        s = self.settings
        proposal = {"timeline": "latest MASTER_v<n>", "format": "mp4", "codec": "H.264",
                    "resolution": f"{s.width}x{s.height}", "fps": s.fps, "audio": "AAC 48 kHz stereo",
                    "note": "Proposed only — you approve and trigger the render in Resolve.",
                    "generated_by": generated_by("agy.delivery").model_dump(mode="json")}
        p = next_version_path(self.project.path("post"), "render_proposal", ".yaml")
        save_versioned(p, proposal)
        self.state.set_global_stage(GlobalStage.DELIVERY, reason="G7 picture lock")
        return [f"Picture locked. Render settings proposed in {self.project.rel(p)} — you trigger the render."]

    # ---- §9.4 inspection & control commands -----------------------------------------------------
    def show_working_set(self, sid: str) -> str:
        """`show working_set SC12` — everything the scene's agents see (§4.5)."""
        from .working_set import show_working_set

        validate_id(sid, "scene id")
        return show_working_set(self.project, sid)

    def pin(self, sid: str, path: str | Path, why: str, *, user: str | None = None) -> str:
        """`pin SC12 <path> "<why>"` — copy an extra reference into the scene's working set."""
        from .working_set import PIN_EXTENSIONS, pin as pin_ref

        validate_id(sid, "scene id")
        src = Path(path).expanduser()
        if not src.is_file():
            raise StudioError(f"{path} is not a file I can read")
        if src.suffix.lower() not in PIN_EXTENSIONS:
            raise StudioError(f"{src.suffix or 'that file type'} can't be pinned "
                              f"(allowed: {', '.join(sorted(PIN_EXTENSIONS))})")
        d = self.decisions.record(tier=DecisionTier.CRAFT, target_artifact_id=sid, action=DecisionAction.ANNOTATE,
                                  payload={"pin": str(src), "why": why}, rationale=why, user=user or self.user)
        ref = pin_ref(self.project, sid, src, why, decision_id=d.id)
        return f"{sid}: pinned {ref.path} — {ref.why} ({d.id}). The scene's agents see it from the next step."

    def bible(self, artifact_id: str = "GLOBAL") -> str:
        """`bible [GLOBAL|SC01|HERO_01|STYLE_LOCK]` — human-readable Global Bible, Scene Bible, or one record."""
        from .bible_hierarchy import GLOBAL_BIBLE_MD_REL, build_global_context_bible, build_scene_bible
        from .fsio import dump_yaml

        key = (artifact_id or "GLOBAL").strip().upper()
        if key in ("", "GLOBAL", "ALL", "GLOBAL_BIBLE"):
            build_global_context_bible(self.project, save=True)
            md_path = self.project.abs(GLOBAL_BIBLE_MD_REL)
            return md_path.read_text(encoding="utf-8")
        if self.project.path("bible", "scenes", f"{key}.yaml").is_file():
            build_scene_bible(self.project, key, save=True)
            sc_md = self.project.scene_dir(key) / "SCENE_BIBLE.md"
            return sc_md.read_text(encoding="utf-8")
        if key in ("STYLE", "STYLE_LOCK", "STYLELOCK"):
            style = self.project.load_style_lock()
            if style is None:
                return "There is no StyleLock yet (it is proposed during pre-production and locked at G1)."
            return f"StyleLock · {style.approval_status.value} · v{style.version}\n" + dump_yaml(
                style.model_dump(mode="json", exclude={"generated_by"}, exclude_none=True))
        validate_id(key, "bible id")
        if self.project.path("bible", "entities", f"{key}.yaml").exists():
            ent = self.project.load_entity(key)
            lines = [f"{ent.display_name or ent.id} ({ent.id}) · {ent.type.value} · {ent.tier.value} · "
                     f"{ent.approval_status.value} · v{ent.version}", ent.canonical_description]
            if ent.voice_profile_id:
                lines.append(f"voice: {ent.voice_profile_id} (`voice {ent.id}`)")
            for st in ent.states:
                pack = st.reference_pack
                if pack.text_only:
                    pack_txt = "text only (incidental)"
                else:
                    uris = pack.image_uris()
                    in_lib = sum(1 for u in uris if _in_library(u))
                    pack_txt = f"{len(uris)} reference image(s), {in_lib} locked in library/"
                lines.append(f"  state {st.id} · {st.approval_status.value} · {pack_txt}"
                             + (f" · scenes {', '.join(st.scenes_active)}" if st.scenes_active else "")
                             + (f" — {st.delta_description}" if st.delta_description else ""))
            return "\n".join(lines)
        if self.project.path("bible", "environments", f"{key}.yaml").exists():
            env = self.project.load_environment(key)
            return f"Location {env.id} · {env.approval_status.value}\n" + dump_yaml(
                env.model_dump(mode="json", exclude={"generated_by"}, exclude_none=True))
        raise StudioError(f"{artifact_id} is not in the Bible (try `bible`, `bible SC01`, or a character/prop/location id)")

    def voice(self, char: str) -> str:
        """`voice HERO_01` — voice profile, anchor and the latest speaker-similarity readings."""
        char = validate_id(char.strip().upper(), "character id")
        vp = self.project.load_voice_profile(char)
        if vp is None:
            return f"{self.display_name(char)} has no voice profile yet (proposed in pre-production, locked at G1)."
        lines = [f"{vp.id} for {self.display_name(char)} · {vp.approval_status.value} · v{vp.version}",
                 f"voice: {vp.description}"]
        if vp.language:
            lines.append(f"language: {vp.language}")
        for state_id, delta in vp.state_variants.items():
            lines.append(f"  in {state_id}: {delta}")
        if vp.anchor_take:
            lines.append(f"anchor: {vp.anchor_take} ({self.project.abs(vp.anchor_wav) if vp.anchor_wav else 'no wav'})"
                         " — every new take is scored against it")
        else:
            lines.append("anchor: not set yet — you'll be asked inside the first G5 where this character speaks")
        scores = []
        for clip in self.project.iter_all_clips():
            if char not in continuity.speakers_in_clip(clip) or not clip.selected_take_id:
                continue
            try:
                take = self.project.load_take(clip.selected_take_id)
            except FileNotFoundError:
                continue
            sim = getattr(getattr(take, "qa", None), "speaker_similarity", None)
            if isinstance(sim, dict):
                sim = sim.get(char)
            if sim is not None:
                scores.append(f"{clip.id} {float(sim):.2f}")
        if scores:
            lines.append("speaker similarity of selected takes: " + ", ".join(scores[-8:]))
        return "\n".join(lines)

    def questions_text(self, sid: str | None = None) -> str:
        """`questions` — open blocking questions with the scene/stage they block (§9.5)."""
        qs = self.questions.open(sid)
        if not qs:
            return "No open questions."
        out = []
        for q in qs:
            where = q.scene_id or "project"
            if q.stage:
                where += f" · {q.stage}"
            line = f"{q.id} [{where}] {q.text}"
            if q.options:
                line += f" (options: {', '.join(q.options)})"
            out.append(line)
        return "\n".join(out) + "\nReply with `answer <id> \"...\"`."

    def pause(self, sid: str | None = None, *, user: str | None = None) -> str:
        if sid:
            validate_id(sid, "scene id")
            self.state.ensure_scene(sid)
        self.state.set_paused(True, sid)
        self.decisions.record(tier=DecisionTier.POLISH, target_artifact_id=sid or self.project.project_id,
                              action=DecisionAction.OVERRIDE, payload={"paused": True}, rationale="pause",
                              user=user or self.user)
        return f"{sid or 'Everything'} paused — nothing new is dispatched until `resume{' ' + sid if sid else ''}`."

    def unpause(self, sid: str | None = None, *, user: str | None = None) -> str:
        if sid:
            validate_id(sid, "scene id")
        self.state.set_paused(False, sid)
        self.decisions.record(tier=DecisionTier.POLISH, target_artifact_id=sid or self.project.project_id,
                              action=DecisionAction.OVERRIDE, payload={"paused": False}, rationale="resume",
                              user=user or self.user)
        return f"{sid or 'Everything'} resumed."

    def prefs_text(self, scope: str | None = None) -> str:
        """`prefs` — standing preferences learned from your steers/rejects (§4.8)."""
        ps = self.prefs.list(scope)
        if not ps:
            return "No standing preferences yet — every steer or reject you give becomes one."
        return "\n".join(f"{p.id} [{p.scope}] {p.text} (weight {p.weight:g}; from {', '.join(p.source_decision_ids)})"
                         for p in ps)

    def budget_text(self, sid: str | None = None) -> str:
        if sid:
            validate_id(sid, "scene id")
        return self.budget.report(sid)

    def raise_budget(self, sid: str, cap: float, *, user: str | None = None) -> str:
        """Human answer to a budget overrun question: raise the scene cap (never done silently, §12)."""
        validate_id(sid, "scene id")
        d = self.decisions.record(tier=DecisionTier.CRAFT, target_artifact_id=sid, action=DecisionAction.OVERRIDE,
                                  payload={"budget_cap": float(cap)}, rationale=f"raise budget cap to {cap}",
                                  user=user or self.user)
        new_cap = self.budget.raise_cap(sid, float(cap), decision_id=d.id)
        return f"{sid}: budget cap is now {new_cap:.2f} ({d.id}). " + self.advance(sid).text()

    def handoff_text(self, sid: str) -> str:
        """`handoff SC12` — where the scene's timeline is and who owns it (§10.4)."""
        validate_id(sid, "scene id")
        h = self.project.load_handoff(sid)
        if h is None:
            stage = self.state.scene(sid).stage.value if sid in self.state.load().scenes else "not started"
            return f"{sid} has not been handed off yet (stage: {stage})."
        owner = "yours — AGY only reads it" if h.ownership == Ownership.HUMAN else "still AGY's"
        lines = [f"{h.timeline_name} is on the timeline; it's {owner}. Handed off {h.ts}."]
        if h.bin_path:
            lines.append(f"Media bin: {h.bin_path}")
        if h.legacy:
            lines.append("(Adopted from the pre-v3 episode pipeline; treated as human-owned.)")
        regen_ready = [t for t in self.state.load().regen_queue if str(t.get("clip_id", "")).startswith(sid + "_")]
        if regen_ready:
            lines.append(f"{len(regen_ready)} regeneration(s) queued for this scene.")
        return "\n".join(lines)

    def decisions_text(self, n: int = 15, sid: str | None = None) -> str:
        rows = self.decisions.for_scene(sid) if sid else self.decisions.all()
        if not rows:
            return "No decisions recorded yet."
        return "\n".join(f"{d.id} {d.ts} {d.tier.value}/{d.action.value} {d.target_artifact_id}"
                         + (f" [{d.gate}]" if d.gate else "") + (f" — {d.rationale}" if d.rationale else "")
                         for d in rows[-max(1, n):])

    # ---- v4 extensions (reanchor, control mode, searchable rejected vault) -------------------
    def reanchor_shot(self, shot_id: str, screenshot_path: str | Path, *, steer: str = "",
                      user: str | None = None) -> str:
        """Gate 3 3D/Blender screenshot-to-keyframe re-anchoring."""
        from .set3d.blender_export import reanchor_shot_from_screenshot

        res = reanchor_shot_from_screenshot(
            self.project, shot_id, screenshot_path, image_adapter=self._image_adapter(), steer=steer
        )
        self.decisions.record(
            tier=DecisionTier.CRAFT,
            gate=GateId.G3.value,
            target_artifact_id=shot_id,
            action=DecisionAction.OVERRIDE,
            payload={"screenshot": res["screenshot"], "board_frame": res["board_frame"], "steer": steer},
            rationale=f"reanchor {shot_id} from 3D screenshot",
            user=user or self.user,
        )
        return (f"{shot_id}: re-anchored from 3D screenshot ({res['screenshot']}) + "
                f"{len(res['reference_images'])} Bible/3D reference(s) → {res['board_frame']}")

    def set_control_mode(self, mode: str, *, content_format: str | None = None, user: str | None = None) -> str:
        """Set project control_mode (director | hybrid | autopilot) and optional content_format."""
        norm = mode.strip().lower()
        if norm not in ("director", "hybrid", "autopilot"):
            raise StudioError("control_mode must be director, hybrid, or autopilot")
        cfg = self.project.config
        updates: dict[str, Any] = {"control_mode": norm}
        if content_format:
            updates["content_format"] = content_format.strip().lower()
        cfg.settings = cfg.settings.model_copy(update=updates)
        self.project.save_config(cfg)
        self.decisions.record(
            tier=DecisionTier.VISION,
            target_artifact_id=self.project.project_id,
            action=DecisionAction.OVERRIDE,
            payload=updates,
            rationale=f"mode {norm}",
            user=user or self.user,
        )
        return f"{self.project.project_id}: control_mode set to {norm}" + (
            f" (format: {updates['content_format']})" if "content_format" in updates else ""
        )

    def set_model_policy(self, *, primary: str | None = None, secondary: str | None = None,
                         fallbacks: dict[str, list[str]] | None = None, clear_fallbacks: bool = False,
                         user: str | None = None) -> str:
        """Per-production model policy (project.yaml ``settings``): primary / secondary video engine and provider
        fallback chains. Registry.yaml stays the global capability source; this only narrows or reorders it."""
        from .decomposer import _NO_FALLBACK, video_model_ids

        video = video_model_ids(self.registry)
        updates: dict[str, Any] = {}
        if primary is not None:
            prim = primary.strip()
            if prim.lower() in ("", "default", "registry"):
                updates["video_model_primary"] = None
            elif prim not in video:
                raise StudioError(f"primary video model must be one of {video}, not {prim!r}")
            else:
                updates["video_model_primary"] = prim
        if secondary is not None:
            sec = secondary.strip()
            if sec.lower() in ("default", "registry", "auto"):
                updates["video_model_secondary"] = None
            elif sec.lower() in _NO_FALLBACK:
                updates["video_model_secondary"] = "none"
            elif sec not in video:
                raise StudioError(f"secondary video model must be one of {video} or 'none', not {sec!r}")
            else:
                updates["video_model_secondary"] = sec
        cfg = self.project.config
        if clear_fallbacks:
            updates["provider_fallbacks"] = {}
        if fallbacks:
            merged = dict(cfg.settings.provider_fallbacks) if not clear_fallbacks else {}
            for mid, chain in fallbacks.items():
                self.registry.get(mid)  # must be a registry id
                merged[mid] = [str(x).strip() for x in chain if str(x).strip()]
            updates["provider_fallbacks"] = merged
        if not updates:
            return self.describe_model_policy()
        cfg.settings = cfg.settings.model_copy(update=updates)
        self.project.save_config(cfg)
        self.decisions.record(tier=DecisionTier.VISION, target_artifact_id=self.project.project_id,
                              action=DecisionAction.OVERRIDE, payload={"model_policy": updates},
                              rationale="model policy", user=user or self.user)
        return self.describe_model_policy()

    def describe_model_policy(self) -> str:
        """Effective model policy for this production: engines, provider chains (registry vs. override), pins."""
        st = self.settings
        video = [m for m in self.registry.video_models()]
        prim = st.video_model_primary or "omni (registry default)"
        if st.video_model_secondary is None:
            sec = "other enabled video model (auto)"
        elif str(st.video_model_secondary).lower() in ("", "none", "null", "off"):
            sec = "NONE — fail fast on the primary, no engine fallback"
        else:
            sec = st.video_model_secondary
        lines = [f"{self.project.project_id} model policy",
                 f"  video primary   : {prim}",
                 f"  video secondary : {sec}",
                 "  provider chains :"]
        for spec in self.registry.all():
            if not spec.enabled:
                continue
            pin = self.project.config.model_pins.get(spec.id)
            head = pin or spec.provider_model
            override = st.provider_fallbacks.get(spec.id)
            chain = override if override is not None else list(spec.provider_fallbacks)
            src = "project.yaml" if override is not None else "registry.yaml"
            tail = " -> ".join(chain) if chain else "(no provider fallback)"
            lines.append(f"    {spec.id:<20} {head}{' [pinned]' if pin else ''} -> {tail}   [{src}]")
        lines.append(f"  video engines available: {', '.join(m.id for m in video)}")
        lines.append("  change with: `models --primary veo --secondary omni|none --fallback omni=a,b --fallback veo=`")
        return "\n".join(lines)

    def search_vault(self, query: str = "", *, character_id: str | None = None,
                     scene_id: str | None = None) -> str:
        """Search the Rejected Video Vault (`library/rejected_takes_vault.jsonl`)."""
        from .vault import search_rejected_takes

        hits = search_rejected_takes(self.project, query, character_id=character_id, scene_id=scene_id)
        if not hits:
            return "No rejected takes matched your query."
        lines = [f"Found {len(hits)} rejected take(s) in library/rejected_takes_vault.jsonl:"]
        for item in hits[:20]:
            h = item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            lines.append(
                f"  - {h.get('take_id')} ({h.get('scene_id')} · {h.get('model')}): "
                f"video={h.get('video_uri')} | reason={h.get('rejection_reason')} | "
                f"prompt={str(h.get('prompt', ''))[:90]}"
            )
        return "\n".join(lines)

    # ---- resume -------------------------------------------------------------------------------
    def resume(self) -> str:
        """Called after a crash/kill: finish in-flight takes (never resubmitting) and continue every scene."""
        from .shot_generator import resume_inflight

        lines = []
        takes = resume_inflight(self.project, self.registry, self.state, fake=self.fake)
        if takes:
            lines.append(f"Resumed {len(takes)} in-flight take(s) without resubmitting.")
        lines.append(self.tick())
        return "\n".join(lines)
