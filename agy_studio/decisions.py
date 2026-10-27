"""Decisions, preferences and blocking questions — BUILD_SPEC_v3 §4.8, §9.2, §9.5 (Workstream A).

* ``decisions.jsonl`` and ``questions.jsonl`` are append-only (``fsio.append_jsonl``). Ids (``D000001``,
  ``Q000001``) are allocated under a dedicated ``fsio.locked`` allocation lock so concurrent writers never reuse
  an id. (A separate ``*.alloc`` lock is used because ``append_jsonl`` takes the file's own lock and flock locks
  are per open-file-description — nesting the same lock would deadlock.)
* Every human action is a :class:`Decision`. A ``steer`` or ``reject`` that carries text automatically becomes a
  scoped :class:`Preference` in ``preferences.yaml`` (acceptance criterion 7); preferences are injected into
  future working sets by scope (project / scene / character / agent / user).
* A question is answered by appending an updated copy of it — the latest copy per id wins; nothing is rewritten.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

from .fsio import (
    append_jsonl,
    canonical_json,
    generated_by,
    locked,
    now_iso,
    read_jsonl,
    read_yaml,
    save_versioned,
    validate_id,
)
from .models import (
    Decision,
    DecisionAction,
    DecisionSource,
    DecisionTier,
    GateId,
    Preference,
    Question,
    SceneState,
)
from .project import Project, parse_artifact_id

PREFERENCE_ACTIONS = frozenset({DecisionAction.STEER, DecisionAction.REJECT})
_SCENE_REF_KEYS = ("scene_id", "take_id", "anchor_take", "clip_id", "shot_id", "value")


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------
def _max_seq(rows: Iterable[dict[str, Any]], prefix: str) -> int:
    pat = re.compile(rf"^{prefix}(\d+)$")
    best = 0
    for row in rows:
        m = pat.match(str(row.get("id", "")))
        if m:
            best = max(best, int(m.group(1)))
    return best


def _alloc_lock_target(path: Path) -> Path:
    return path.with_name(path.name + ".alloc")


def _scene_of(artifact_id: Any) -> str | None:
    if not isinstance(artifact_id, str):
        return None
    try:
        return parse_artifact_id(artifact_id)["scene"]
    except ValueError:
        return None


def decision_text(decision: Decision) -> str:
    """The free text a decision carries (``payload.text`` > ``payload.steer`` > rationale)."""
    payload = decision.payload or {}
    for key in ("text", "steer"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return (decision.rationale or "").strip()


def infer_scope(target_artifact_id: str, payload: dict[str, Any] | None = None) -> str:
    """Preference scope for a decision: explicit ``payload['scope']`` or inferred from the target.

    scene/shot/clip/take ids -> ``scene:<SC>``; ``CHAR_*`` (incl. ``CHAR_X_STATE_Y`` and ``VOICE_CHAR_X``) ->
    ``character:<CHAR_X>``; anything else -> ``project``.
    """
    explicit = (payload or {}).get("scope")
    if explicit:
        return str(explicit).strip()
    target = (target_artifact_id or "").strip()
    scene = _scene_of(target)
    if scene:
        return f"scene:{scene}"
    if target.startswith("VOICE_CHAR_"):
        target = target[len("VOICE_"):]
    if target.startswith("CHAR_"):
        return f"character:{target.split('_STATE_', 1)[0]}"
    return "project"


def preference_is_relevant(pref: Preference, *, scene_id: str | None = None, character_ids: Iterable[str] = (),
                           agent: str | None = None, user: str | None = None) -> bool:
    """Scene scoping rule: project-wide, this scene, characters in this scene, the asking agent / user."""
    if pref.scope == "project":
        return True
    kind, _, value = pref.scope.partition(":")
    if kind == "scene":
        return scene_id is not None and value == scene_id
    if kind == "character":
        return value in set(character_ids)
    if kind == "agent":
        return agent is not None and value == agent
    if kind == "user":
        return user is not None and value == user
    return False


def decision_touches_scene(decision: Decision, scene_id: str) -> bool:
    """True if the decision targets (or its payload references) an artifact of ``scene_id``."""
    if _scene_of(decision.target_artifact_id) == scene_id:
        return True
    payload = decision.payload or {}
    for key in _SCENE_REF_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and (value == scene_id or _scene_of(value) == scene_id):
            return True
    return False


# ---------------------------------------------------------------------------------------------
# Preferences (preferences.yaml)
# ---------------------------------------------------------------------------------------------
class PreferenceStore:
    """Scoped standing notes from the director. Ids ``P000001``…; file versioned via ``save_versioned``."""

    def __init__(self, project: Project) -> None:
        self.project = project
        self.path = project.preferences_path

    def _read(self) -> dict[str, Any]:
        data = read_yaml(self.path) if self.path.exists() else None
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(f"{self.path} is not a mapping")
        if not isinstance(data.get("preferences"), list):
            data["preferences"] = []
        return data

    def _all(self) -> list[Preference]:
        out: list[Preference] = []
        for raw in self._read()["preferences"]:
            try:
                out.append(Preference.model_validate(raw))
            except ValueError:
                continue  # a hand-mangled row never breaks the store
        return out

    def _add(self, *, scope: str, text: str, weight: float, source_decision_ids: Iterable[str],
             dedupe_decision_id: str | None = None) -> Preference:
        text = (text or "").strip()
        if not text:
            raise ValueError("a preference needs some text")
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"preference weight must be a finite number >= 0, got {weight}")
        ids = [validate_id(str(d), "decision id") for d in source_decision_ids]
        with locked(self.path):
            data = self._read()
            rows = data["preferences"]
            if dedupe_decision_id:
                for raw in rows:
                    if dedupe_decision_id in (raw.get("source_decision_ids") or []):
                        return Preference.model_validate(raw)
            pref = Preference(id=f"P{_max_seq(rows, 'P') + 1:06d}", scope=scope, text=text, weight=weight,
                              source_decision_ids=ids, created_at=now_iso())
            rows.append(pref.model_dump(mode="json"))
            data["generated_by"] = generated_by("agy.decisions").model_dump(mode="json")
            save_versioned(self.path, data)
        return pref

    def add(self, *, scope: str, text: str, weight: float = 1.0,
            source_decision_ids: Iterable[str] = ()) -> Preference:
        return self._add(scope=scope, text=text, weight=weight, source_decision_ids=source_decision_ids)

    def from_decision(self, decision: Decision) -> Preference | None:
        """Preference for a decision's text (idempotent: one preference per decision id)."""
        text = decision_text(decision)
        if not text:
            return None
        weight = decision.payload.get("weight", 1.0)
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            weight = 1.0
        return self._add(scope=infer_scope(decision.target_artifact_id, decision.payload), text=text,
                         weight=weight, source_decision_ids=[decision.id], dedupe_decision_id=decision.id)

    def list(self, scope: str | None = None, active_only: bool = True) -> list[Preference]:
        return [p for p in self._all() if (scope is None or p.scope == scope) and (p.active or not active_only)]

    def for_decision(self, decision_id: str) -> list[Preference]:
        return [p for p in self._all() if decision_id in p.source_decision_ids]

    def relevant(self, scene_id: str | None = None, character_ids: Iterable[str] = (), agent: str | None = None,
                 user: str | None = None) -> list[Preference]:
        chars = tuple(character_ids)
        return [p for p in self._all() if p.active and preference_is_relevant(
            p, scene_id=scene_id, character_ids=chars, agent=agent, user=user)]

    def deactivate(self, pref_id: str, *, decision_id: str) -> Preference:
        """Retire a preference (never deleted; the deactivating decision is recorded in the file)."""
        validate_id(pref_id, "preference id")
        validate_id(decision_id, "decision id")
        with locked(self.path):
            data = self._read()
            for i, raw in enumerate(data["preferences"]):
                if raw.get("id") != pref_id:
                    continue
                pref = Preference.model_validate(raw)
                if not pref.active:
                    return pref
                pref.active = False
                data["preferences"][i] = pref.model_dump(mode="json")
                if not isinstance(data.get("deactivations"), list):
                    data["deactivations"] = []
                data["deactivations"].append({"preference_id": pref_id, "decision_id": decision_id, "ts": now_iso()})
                data["generated_by"] = generated_by("agy.decisions").model_dump(mode="json")
                save_versioned(self.path, data)
                return pref
        raise KeyError(f"no preference {pref_id}")


# ---------------------------------------------------------------------------------------------
# Decisions (decisions.jsonl)
# ---------------------------------------------------------------------------------------------
class DecisionLog:
    """Append-only decision log. ``record`` persists before returning (§1.4: decisions persist before AGY acts)."""

    def __init__(self, project: Project, preferences: PreferenceStore | None = None) -> None:
        self.project = project
        self.path = project.decisions_path
        self.preferences = preferences or PreferenceStore(project)
        self.last_preference: Preference | None = None  # preference auto-created by the last record()

    def record(self, *, tier: DecisionTier | str, target_artifact_id: str, action: DecisionAction | str,
               payload: dict[str, Any] | None = None, rationale: str = "", gate: GateId | str | None = None,
               source: DecisionSource | str = "chat", user: str = "director") -> Decision:
        tier = DecisionTier(tier)
        action = DecisionAction(action)
        source = DecisionSource(source)
        gate_value = GateId(gate).value if gate else None
        target = str(target_artifact_id or "").strip()
        if not target or len(target) > 200 or any(ch in target for ch in "\r\n\t"):
            raise ValueError(f"invalid decision target {target_artifact_id!r}")
        try:  # normalise to plain JSON now so the returned record equals the persisted one
            payload = json.loads(canonical_json(dict(payload or {})))
        except TypeError as exc:
            raise TypeError(f"decision payload must be JSON-serialisable: {exc}") from exc
        rationale = str(rationale or "").strip()
        user = str(user or "").strip() or "director"
        draft = Decision(id="D000000", ts=now_iso(), user=user, tier=tier, gate=gate_value, target_artifact_id=target,
                         source=source, action=action, payload=payload, rationale=rationale)
        wants_preference = action in PREFERENCE_ACTIONS and bool(decision_text(draft))
        if wants_preference:  # validate the scope BEFORE anything is written
            Preference(id="P000000", scope=infer_scope(target, payload), text=decision_text(draft), created_at=draft.ts)
        with locked(_alloc_lock_target(self.path)):
            decision = draft.model_copy(update={"id": f"D{_max_seq(read_jsonl(self.path), 'D') + 1:06d}",
                                                "ts": now_iso()})
            append_jsonl(self.path, decision)
        self.last_preference = self.preferences.from_decision(decision) if wants_preference else None
        return decision

    def all(self) -> list[Decision]:
        out: list[Decision] = []
        for row in read_jsonl(self.path):
            try:
                out.append(Decision.model_validate(row))
            except ValueError:
                continue
        return out

    def get(self, decision_id: str) -> Decision | None:
        return next((d for d in self.all() if d.id == decision_id), None)

    def for_target(self, artifact_id: str, prefix: bool = True) -> list[Decision]:
        return [d for d in self.all() if d.target_artifact_id == artifact_id
                or (prefix and d.target_artifact_id.startswith(artifact_id + "_"))]

    def for_scene(self, scene_id: str) -> list[Decision]:
        validate_id(scene_id, "scene id")
        return [d for d in self.all() if decision_touches_scene(d, scene_id)]

    def backfill_preferences(self) -> list[Preference]:
        """Resume safety: create preferences for steer/reject decisions whose preference write was lost."""
        linked = {d for p in self.preferences.list(active_only=False) for d in p.source_decision_ids}
        created: list[Preference] = []
        for d in self.all():
            if d.action in PREFERENCE_ACTIONS and d.id not in linked and decision_text(d):
                pref = self.preferences.from_decision(d)
                if pref:
                    created.append(pref)
        return created


# ---------------------------------------------------------------------------------------------
# Blocking questions (questions.jsonl)
# ---------------------------------------------------------------------------------------------
class QuestionLog:
    """Questions surfaced with the scene/stage they block; AGY keeps working elsewhere (§9.5).

    With ``state`` given, an open question's id is added to that scene's ``blocked_by`` and removed on answer.
    """

    def __init__(self, project: Project, state: Any = None) -> None:
        self.project = project
        self.path = project.questions_path
        self.state = state

    def _rows(self) -> list[Question]:
        out: list[Question] = []
        for row in read_jsonl(self.path):
            try:
                out.append(Question.model_validate(row))
            except ValueError:
                continue
        return out

    def _set_blocker(self, scene_id: str, qid: str, add: bool) -> None:
        if self.state is None:
            return
        with self.state.mutate() as st:
            sc = st.scenes.setdefault(scene_id, SceneState(scene_id=scene_id))
            blockers = [b for b in sc.blocked_by if b != qid]
            sc.blocked_by = [*blockers, qid] if add else blockers

    def ask(self, text: str, *, scene_id: str | None = None, stage: str | None = None,
            blocks: Iterable[str] = (), options: Iterable[str] = ()) -> Question:
        text = (text or "").strip()
        if not text:
            raise ValueError("a question needs text")
        if scene_id is not None:
            validate_id(scene_id, "scene id")
        with locked(_alloc_lock_target(self.path)):
            q = Question(id=f"Q{_max_seq(read_jsonl(self.path), 'Q') + 1:06d}", ts=now_iso(), text=text,
                         scene_id=scene_id, stage=stage, blocks=[str(b) for b in blocks],
                         options=[str(o) for o in options])
            append_jsonl(self.path, q)
        if scene_id:
            self._set_blocker(scene_id, q.id, add=True)
        return q

    def answer(self, question_id: str, answer: str, *, decision_id: str | None = None) -> Question:
        current = self.get(question_id)
        if current is None:
            raise KeyError(f"no question {question_id}")
        answer = (answer or "").strip()
        if not answer:
            raise ValueError("an answer needs text")
        if decision_id is not None:
            validate_id(decision_id, "decision id")
        updated = Question.model_validate({**current.model_dump(mode="json"), "status": "answered",
                                           "answer": answer, "answered_at": now_iso(), "decision_id": decision_id})
        append_jsonl(self.path, updated)
        if current.scene_id:
            self._set_blocker(current.scene_id, current.id, add=False)
        return updated

    def get(self, question_id: str) -> Question | None:
        latest = None
        for q in self._rows():
            if q.id == question_id:
                latest = q
        return latest

    def all(self) -> list[Question]:
        """Latest copy of every question, ordered by first appearance."""
        latest: dict[str, Question] = {}
        for q in self._rows():
            latest[q.id] = q  # dict keeps first-insertion order; value is replaced by later copies
        return list(latest.values())

    def open(self, scene_id: str | None = None) -> list[Question]:
        return [q for q in self.all() if q.status == "open" and (scene_id is None or q.scene_id == scene_id)]
