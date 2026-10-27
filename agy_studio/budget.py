"""Per-scene budget caps — BUILD_SPEC_v3 §12 (Workstream A).

* A scene's cap defaults to ``estimate × settings.budget_cap_multiplier`` (estimate from the scene's clips via the
  registry's per-second / per-call prices, else from the scene breakdown).
* Every spend is persisted in ``state.json`` (survives kills and reloads) and journaled in ``logs/budget.jsonl``.
* An overrun never continues silently: :meth:`BudgetManager.require` raises :class:`BudgetExceeded` carrying a
  plain-language question for the director; only a recorded decision can raise a cap.
"""

from __future__ import annotations

import math
from typing import Any

from .decisions import DecisionLog
from .fsio import append_jsonl, now_iso, validate_id
from .invalidation import load_scene_clips
from .models import GenClip, Model, ModelKind
from .project import Project
from .registry import Registry, UnknownModelError, default_registry
from .state import StateStore

_QA_KINDS = (ModelKind.TRANSCRIBE, ModelKind.VISION_JUDGE, ModelKind.SPEAKER_SIMILARITY)


class BudgetCheck(Model):
    ok: bool
    scene_id: str
    cap: float | None = None
    spent: float = 0.0
    planned: float = 0.0
    remaining: float | None = None
    message: str = ""


class BudgetExceeded(RuntimeError):
    """Raised instead of spending past a cap. ``.check.message`` is the question to put to the director."""

    def __init__(self, check: BudgetCheck) -> None:
        super().__init__(check.message)
        self.check = check


def _money(x: float | None) -> str:
    return "—" if x is None else f"${x:,.2f}"


def _amount(value: Any, what: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number, got {value!r}") from None
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{what} must be a finite amount >= 0, got {value!r}")
    return round(v, 4)


class BudgetManager:
    def __init__(self, project: Project, registry: Registry | None = None, state: StateStore | None = None) -> None:
        self.project = project
        self.registry = registry if registry is not None else default_registry()
        self.state = state if state is not None else StateStore(project)
        self.ledger_path = project.path("logs", "budget.jsonl")

    # -- estimates ---------------------------------------------------------------------------------
    def _qa_cost_per_take(self) -> float:
        total = 0.0
        for kind in _QA_KINDS:  # one transcription, one vision-judge pass, one speaker-similarity pass per take
            try:
                total += self.registry.first_of_kind(kind).estimate_cost(0.0, calls=1)
            except UnknownModelError:
                continue
        return total

    def _clip_cost(self, model_id: str, duration_s: float, takes: int) -> float:
        if takes < 0:
            raise ValueError("takes must be >= 0")
        spec = self.registry.get(model_id)
        return spec.estimate_cost(spec.snap_duration(float(duration_s)), calls=takes) + self._qa_cost_per_take() * takes

    def _model_for(self, clip: GenClip) -> str:
        try:
            override = self.project.load_scene(clip.scene_id).model_override
        except (OSError, ValueError):
            override = None
        return override or clip.model_target.preferred

    def estimate_clip(self, clip: GenClip, *, model_id: str | None = None, takes: int | None = None) -> float:
        n = int(takes if takes is not None else self.project.config.settings.takes_per_clip)
        return round(self._clip_cost(model_id or self._model_for(clip), clip.duration_s, n), 4)

    def estimate_scene(self, scene_id: str) -> float:
        validate_id(scene_id, "scene id")
        scene = self.project.load_scene(scene_id)
        clips = load_scene_clips(self.project, scene_id)
        if clips:
            return round(sum(self.estimate_clip(c) for c in clips), 4)
        bd = scene.breakdown
        if bd is None:
            return 0.0
        if bd.estimated_cost > 0:
            return round(float(bd.estimated_cost), 4)
        if bd.estimated_clips > 0 and bd.estimated_duration_s > 0:
            per_clip = bd.estimated_duration_s / bd.estimated_clips
            model = scene.model_override or bd.recommended_model
            takes = int(self.project.config.settings.takes_per_clip)
            return round(self._clip_cost(model, per_clip, takes) * bd.estimated_clips, 4)
        return 0.0

    # -- caps / checks -----------------------------------------------------------------------------
    def _scene_numbers(self, scene_id: str) -> tuple[float | None, float]:
        sc = self.state.load().scenes.get(scene_id)
        return (sc.budget_cap, round(sc.budget_spent, 4)) if sc else (None, 0.0)

    def ensure_cap(self, scene_id: str) -> float | None:
        """The scene's cap; derives and stores ``estimate × multiplier`` when none is set (None: no estimate yet)."""
        validate_id(scene_id, "scene id")
        cap, _ = self._scene_numbers(scene_id)
        if cap is not None:
            return cap
        estimate = self.estimate_scene(scene_id)
        if estimate <= 0:
            return None
        multiplier = float(self.project.config.settings.budget_cap_multiplier)
        cap = round(estimate * multiplier, 4)
        with self.state.mutate() as st:  # re-check under the state lock: another process may have set it
            sc = st.scenes.get(scene_id)
            if sc is not None and sc.budget_cap is not None:
                return sc.budget_cap
            self.state.set_budget_cap(scene_id, cap)
        self._journal("cap_set", scene_id, cap=cap, estimate=estimate, multiplier=multiplier)
        return cap

    def check(self, scene_id: str, planned: float) -> BudgetCheck:
        planned = _amount(planned, "planned spend")
        cap = self.ensure_cap(scene_id)
        _, spent = self._scene_numbers(scene_id)
        if cap is None:
            ok = planned == 0
            msg = (f"{scene_id}: nothing to spend." if ok else
                   f"{scene_id} has no budget cap yet (there is no estimate to base it on) — tell me a cap, "
                   f"e.g. `budget {scene_id} 5.00`, before I spend {_money(planned)}.")
            return BudgetCheck(ok=ok, scene_id=scene_id, cap=None, spent=spent, planned=planned, message=msg)
        remaining = round(cap - spent, 4)
        ok = spent + planned <= cap + 1e-9
        if ok:
            msg = (f"{scene_id}: {_money(spent)} of {_money(cap)} spent; this step needs {_money(planned)} — fine "
                   f"({_money(remaining - planned)} left after it).")
        else:
            suggest = math.ceil((spent + planned) * 1.1 * 100) / 100
            msg = (f"{scene_id} would go over budget: {_money(spent)} spent + {_money(planned)} planned = "
                   f"{_money(spent + planned)}, cap {_money(cap)}. Raise the cap (e.g. `budget {scene_id} "
                   f"{suggest:.2f}`) or tell me what to cut — I won't continue until you decide.")
        return BudgetCheck(ok=ok, scene_id=scene_id, cap=cap, spent=spent, planned=planned, remaining=remaining,
                           message=msg)

    def require(self, scene_id: str, planned: float) -> BudgetCheck:
        """Like :meth:`check` but raises :class:`BudgetExceeded` (ask, never continue) when over the cap."""
        chk = self.check(scene_id, planned)
        if not chk.ok:
            self._journal("blocked", scene_id, planned=chk.planned, spent=chk.spent, cap=chk.cap)
            raise BudgetExceeded(chk)
        return chk

    # -- spend / caps ------------------------------------------------------------------------------
    def charge(self, scene_id: str, amount: float, *, ref: str, note: str = "") -> float:
        """Record actual spend (persisted in state.json); returns the scene's new total."""
        validate_id(scene_id, "scene id")
        amount = _amount(amount, "charge")
        ref = " ".join(str(ref or "").split())
        if not ref or len(ref) > 200:
            raise ValueError("a charge needs a short ref (e.g. the take id)")
        spent = self.state.add_spend(scene_id, amount)
        self._journal("charge", scene_id, amount=amount, ref=ref, note=" ".join(str(note or "").split()), spent=spent)
        return spent

    def raise_cap(self, scene_id: str, new_cap: float, *, decision_id: str) -> float:
        """Change a cap — only with the id of a decision that is on record."""
        validate_id(scene_id, "scene id")
        new_cap = _amount(new_cap, "budget cap")
        validate_id(decision_id, "decision id")
        if DecisionLog(self.project).get(decision_id) is None:
            raise PermissionError(f"decision {decision_id} is not on record — only the director's decision can "
                                  "change a budget cap")
        old_cap, _ = self._scene_numbers(scene_id)
        self.state.set_budget_cap(scene_id, new_cap)
        self._journal("cap_raised", scene_id, old_cap=old_cap, new_cap=new_cap, decision_id=decision_id)
        return new_cap

    # -- reporting ---------------------------------------------------------------------------------
    @staticmethod
    def _scene_line(scene_id: str, cap: float | None, spent: float) -> str:
        if cap is None:
            return f"{scene_id}: {_money(spent)} spent (no cap set yet)."
        left = cap - spent
        if left < -1e-9:
            return f"{scene_id}: {_money(spent)} spent — {_money(-left)} OVER the {_money(cap)} cap."
        pct = spent / cap * 100 if cap else 0.0
        return f"{scene_id}: {_money(spent)} of {_money(cap)} spent ({pct:.0f}%), {_money(left)} left."

    def report(self, scene_id: str | None = None) -> str:
        st = self.state.load()
        if scene_id is not None:
            validate_id(scene_id, "scene id")
            cap, spent = self._scene_numbers(scene_id)
            lines = [self._scene_line(scene_id, cap, spent)]
            try:
                estimate = self.estimate_scene(scene_id)
            except (OSError, ValueError, UnknownModelError):
                estimate = 0.0
            if estimate:
                lines.append(f"The planned work for {scene_id} is estimated at {_money(estimate)} "
                             f"({self.project.config.settings.takes_per_clip} takes per clip).")
        else:
            rows = [self._scene_line(sid, sc.budget_cap, sc.budget_spent) for sid, sc in sorted(st.scenes.items())
                    if sc.budget_cap is not None or sc.budget_spent]
            lines = ["Budget:", *[f"  {r}" for r in rows]] if rows else ["Budget: nothing spent yet."]
            lines.append(f"Total spent: {_money(st.budget_spent_total)}.")
        if any(m.cost_is_estimate for m in self.registry.video_models()):
            lines.append("Model prices are estimates until they are checked against billing.")
        return "\n".join(lines)

    def _journal(self, event: str, scene_id: str, **fields: Any) -> None:
        append_jsonl(self.ledger_path, {"ts": now_iso(), "event": event, "scene_id": scene_id, **fields})
