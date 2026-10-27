"""Orchestration state — ``state.json`` (BUILD_SPEC_v3 §12).

* per-scene state machine with an explicit transition table (forced resets require a logged reason),
* atomic persistence after every mutation (``StateStore.mutate``), re-entrant within a thread,
* dispatch de-duplication by ``payload_hash`` (acceptance criterion 9),
* first-appearance ownership of unlocked packs / unanchored voices (§12 contention),
* Vision-tier confirm handshake (fan-out must be shown before ``confirm <artifact_id>`` is accepted),
* regen queue, pause/resume, per-scene budget spend, resume cursor.

Every transition is also appended to ``logs/state_transitions.jsonl`` (append-only audit trail).
"""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path
from typing import Any, Iterator

from .fsio import append_jsonl, atomic_write_json, locked, now_iso, read_json
from .models import GateId, GlobalStage, ProjectState, SceneStage, SceneState, StageTransition
from .project import Project

S = SceneStage
ALLOWED_TRANSITIONS: dict[SceneStage, frozenset[SceneStage]] = {
    S.PENDING: frozenset({S.WORKING_SET}),
    S.WORKING_SET: frozenset({S.DESIGN}),
    S.DESIGN: frozenset({S.AWAITING_G2, S.DIRECTION}),  # DIRECTION when G2 auto-passes (nothing new)
    S.AWAITING_G2: frozenset({S.DIRECTION, S.DESIGN}),
    S.DIRECTION: frozenset({S.AWAITING_G3}),
    S.AWAITING_G3: frozenset({S.DECOMPOSITION, S.DIRECTION}),
    S.DECOMPOSITION: frozenset({S.AUDIO_PLAN}),
    S.AUDIO_PLAN: frozenset({S.AWAITING_G4}),
    S.AWAITING_G4: frozenset({S.LINT, S.AUDIO_PLAN, S.DECOMPOSITION}),
    S.LINT: frozenset({S.GENERATING, S.DECOMPOSITION}),  # bounce to Shot Decomposer
    S.GENERATING: frozenset({S.QA, S.LINT}),
    S.QA: frozenset({S.AWAITING_G5, S.ASSEMBLY, S.GENERATING, S.LINT}),  # next anchor wave -> LINT
    S.AWAITING_G5: frozenset({S.ASSEMBLY, S.GENERATING, S.LINT}),
    S.ASSEMBLY: frozenset({S.HANDED_OFF}),
    S.HANDED_OFF: frozenset({S.DONE, S.GENERATING}),  # post-handoff regen is delivered beside the human's cut
    S.DONE: frozenset({S.GENERATING}),
}

GATE_STAGE: dict[GateId, SceneStage] = {
    GateId.G2: S.AWAITING_G2,
    GateId.G3: S.AWAITING_G3,
    GateId.G4: S.AWAITING_G4,
    GateId.G5: S.AWAITING_G5,
    GateId.G6: S.HANDED_OFF,
}
GATE_STATUSES = frozenset({"pending", "awaiting", "passed", "auto_passed", "rejected"})
GLOBAL_GATES = frozenset({GateId.G1, GateId.G7})


class IllegalTransitionError(RuntimeError):
    pass


class DuplicateDispatchError(RuntimeError):
    pass


class ConfirmRequiredError(PermissionError):
    pass


_local = threading.local()


class StateStore:
    """Read/modify/write access to ``state.json`` under an inter-process lock."""

    def __init__(self, project: Project) -> None:
        self.project = project
        self.path: Path = project.state_path
        self.transitions_log: Path = project.path("logs", "state_transitions.jsonl")

    # ------------------------------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------------------------------
    def load(self) -> ProjectState:
        held = getattr(_local, "held", {}).get(str(self.path))
        if held is not None:
            return held
        if self.path.exists():
            return ProjectState.model_validate(read_json(self.path))
        return ProjectState(project_id=self.project.project_id)

    @contextlib.contextmanager
    def mutate(self) -> Iterator[ProjectState]:
        """Atomic read-modify-write. Nested calls in the same thread share one state object and one write."""
        key = str(self.path)
        held: dict[str, ProjectState] = getattr(_local, "held", None) or {}
        _local.held = held
        if key in held:
            yield held[key]
            return
        with locked(self.path):
            state = (ProjectState.model_validate(read_json(self.path)) if self.path.exists()
                     else ProjectState(project_id=self.project.project_id))
            held[key] = state
            try:
                yield state
                state.write_seq += 1
                state.updated_at = now_iso()
                atomic_write_json(self.path, state)
            finally:
                held.pop(key, None)

    # ------------------------------------------------------------------------------------------
    # Scenes
    # ------------------------------------------------------------------------------------------
    def scene(self, scene_id: str) -> SceneState:
        st = self.load()
        if scene_id not in st.scenes:
            raise KeyError(f"scene {scene_id} not in state.json (call ensure_scene)")
        return st.scenes[scene_id]

    def ensure_scene(self, scene_id: str) -> SceneState:
        with self.mutate() as st:
            if scene_id not in st.scenes:
                st.scenes[scene_id] = SceneState(scene_id=scene_id, updated_at=now_iso())
            return st.scenes[scene_id]

    def transition(self, scene_id: str, to_stage: SceneStage | str, *, reason: str = "",
                   decision_id: str | None = None, force: bool = False) -> SceneState:
        to_stage = SceneStage(to_stage)
        with self.mutate() as st:
            sc = st.scenes.get(scene_id)
            if sc is None:
                sc = st.scenes[scene_id] = SceneState(scene_id=scene_id)
            frm = sc.stage
            if frm == to_stage:
                return sc
            if not force and to_stage not in ALLOWED_TRANSITIONS.get(frm, frozenset()):
                raise IllegalTransitionError(f"{scene_id}: {frm.value} -> {to_stage.value} is not allowed")
            if force and not reason:
                raise IllegalTransitionError("forced transitions require a reason")
            tr = StageTransition(ts=now_iso(), from_stage=frm.value, to_stage=to_stage.value,
                                 reason=("[forced] " if force else "") + reason, decision_id=decision_id)
            sc.history.append(tr)
            sc.stage = to_stage
            sc.awaiting = f"awaiting:{to_stage.value.split('_', 1)[1]}" if to_stage.value.startswith("awaiting_") \
                else ("awaiting:G6" if to_stage == S.HANDED_OFF else None)
            sc.updated_at = tr.ts
            append_jsonl(self.transitions_log, {"scene_id": scene_id, **tr.model_dump(mode="json")})
            return sc

    def set_gate(self, scene_id: str | None, gate: GateId | str, status: str) -> None:
        gate = GateId(gate)
        if status not in GATE_STATUSES:
            raise ValueError(f"gate status {status!r} not in {sorted(GATE_STATUSES)}")
        with self.mutate() as st:
            if scene_id is None:
                if gate not in GLOBAL_GATES:
                    raise ValueError(f"{gate.value} is a scene gate; pass a scene id")
                st.gates[gate.value] = status
            else:
                if gate in GLOBAL_GATES:
                    raise ValueError(f"{gate.value} is a global gate; pass scene_id=None")
                sc = st.scenes.setdefault(scene_id, SceneState(scene_id=scene_id))
                sc.gates[gate.value] = status
                sc.updated_at = now_iso()

    def gate_status(self, scene_id: str | None, gate: GateId | str) -> str:
        gate = GateId(gate)
        st = self.load()
        if scene_id is None:
            return st.gates.get(gate.value, "pending")
        sc = st.scenes.get(scene_id)
        return sc.gates.get(gate.value, "pending") if sc else "pending"

    def set_global_stage(self, stage: GlobalStage | str, reason: str = "") -> None:
        stage = GlobalStage(stage)
        with self.mutate() as st:
            frm = st.global_stage
            st.global_stage = stage
            append_jsonl(self.transitions_log, {"scene_id": None, "ts": now_iso(), "from_stage": frm.value,
                                                "to_stage": stage.value, "reason": reason})

    def set_schedule(self, schedule: list[str], pilot: str | None) -> None:
        with self.mutate() as st:
            st.schedule = list(schedule)
            st.pilot_scene_id = pilot
            for sid in schedule:
                st.scenes.setdefault(sid, SceneState(scene_id=sid, updated_at=now_iso()))

    # ------------------------------------------------------------------------------------------
    # Dispatch de-duplication (acceptance 9)
    # ------------------------------------------------------------------------------------------
    def dispatched_take(self, payload_hash: str) -> str | None:
        return self.load().dispatched_payloads.get(payload_hash)

    def record_dispatch(self, payload_hash: str, take_id: str) -> None:
        with self.mutate() as st:
            existing = st.dispatched_payloads.get(payload_hash)
            if existing and existing != take_id:
                raise DuplicateDispatchError(f"payload {payload_hash} already dispatched as {existing}")
            st.dispatched_payloads[payload_hash] = take_id

    # ------------------------------------------------------------------------------------------
    # Pack / voice contention (§12: first-appearance scene owns generation; others wait)
    # ------------------------------------------------------------------------------------------
    def claim(self, asset_id: str, scene_id: str) -> str:
        with self.mutate() as st:
            return st.pack_locks.setdefault(asset_id, scene_id)

    def release(self, asset_id: str) -> None:
        with self.mutate() as st:
            st.pack_locks.pop(asset_id, None)

    def owner(self, asset_id: str) -> str | None:
        return self.load().pack_locks.get(asset_id)

    # ------------------------------------------------------------------------------------------
    # Vision-tier confirm handshake (§9.3: fan-out printed, then `confirm <artifact_id>`)
    # ------------------------------------------------------------------------------------------
    def request_confirm(self, artifact_id: str, *, action: str, fanout: dict[str, Any]) -> None:
        with self.mutate() as st:
            st.pending_confirms[artifact_id] = {"action": action, "fanout": fanout, "requested_at": now_iso()}

    def pending_confirm(self, artifact_id: str) -> dict[str, Any] | None:
        return self.load().pending_confirms.get(artifact_id)

    def consume_confirm(self, artifact_id: str) -> dict[str, Any]:
        """Pop the pending confirm; raises ConfirmRequiredError if the fan-out was never shown."""
        with self.mutate() as st:
            pending = st.pending_confirms.pop(artifact_id, None)
            if pending is None:
                raise ConfirmRequiredError(
                    f"no pending confirm for {artifact_id}: AGY must show the fan-out before `confirm {artifact_id}`")
            return pending

    # ------------------------------------------------------------------------------------------
    # Regen queue / pause / budget / cursor
    # ------------------------------------------------------------------------------------------
    def enqueue_regen(self, clip_id: str, *, steer: str = "", decision_id: str | None = None,
                      source: str = "chat") -> None:
        with self.mutate() as st:
            if any(r.get("clip_id") == clip_id and r.get("decision_id") == decision_id for r in st.regen_queue):
                return
            st.regen_queue.append({"clip_id": clip_id, "steer": steer, "decision_id": decision_id,
                                   "source": source, "ts": now_iso()})

    def take_regens(self, scene_id: str | None = None) -> list[dict[str, Any]]:
        """Remove and return queued regens (optionally only those of one scene)."""
        with self.mutate() as st:
            keep, out = [], []
            for r in st.regen_queue:
                (out if scene_id is None or str(r.get("clip_id", "")).startswith(scene_id + "_") else keep).append(r)
            st.regen_queue = keep
            return out

    def set_paused(self, paused: bool, scene_id: str | None = None) -> None:
        with self.mutate() as st:
            if scene_id is None:
                st.paused = paused
            else:
                st.scenes.setdefault(scene_id, SceneState(scene_id=scene_id)).paused = paused

    def is_paused(self, scene_id: str | None = None) -> bool:
        st = self.load()
        if st.paused:
            return True
        return bool(scene_id and st.scenes.get(scene_id) and st.scenes[scene_id].paused)

    def add_spend(self, scene_id: str, amount: float) -> float:
        with self.mutate() as st:
            sc = st.scenes.setdefault(scene_id, SceneState(scene_id=scene_id))
            sc.budget_spent = round(sc.budget_spent + amount, 4)
            st.budget_spent_total = round(st.budget_spent_total + amount, 4)
            return sc.budget_spent

    def set_budget_cap(self, scene_id: str, cap: float | None) -> None:
        with self.mutate() as st:
            st.scenes.setdefault(scene_id, SceneState(scene_id=scene_id)).budget_cap = cap

    def bump_agent_round(self, scene_id: str, loop: str) -> int:
        with self.mutate() as st:
            sc = st.scenes.setdefault(scene_id, SceneState(scene_id=scene_id))
            sc.agent_rounds[loop] = sc.agent_rounds.get(loop, 0) + 1
            return sc.agent_rounds[loop]

    def set_blocked(self, scene_id: str, blockers: list[str]) -> None:
        with self.mutate() as st:
            st.scenes.setdefault(scene_id, SceneState(scene_id=scene_id)).blocked_by = list(dict.fromkeys(blockers))

    def set_cursor(self, key: str, value: Any) -> None:
        with self.mutate() as st:
            st.resume_cursor[key] = value
