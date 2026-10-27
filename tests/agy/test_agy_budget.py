"""Workstream A — per-scene budget caps (BUILD_SPEC_v3 §12). Prices come from a private copy of registry.yaml."""

from __future__ import annotations

import pytest

from agy_studio.budget import BudgetExceeded, BudgetManager
from agy_studio.decisions import DecisionLog
from agy_studio.fsio import read_jsonl
from agy_studio.models import GenClip, SceneBreakdown
from agy_studio.project import parse_artifact_id
from agy_studio.state import StateStore

QA_PER_TAKE = 0.002 + 0.003 + 0.0  # gemini_asr + gemini_vision_judge + speaker_sim_local, one call each per take


def _clip(cid: str, duration: float, model: str = "omni") -> GenClip:
    ids = parse_artifact_id(cid)
    return GenClip(id=cid, shot_id=ids["shot"], scene_id=ids["scene"], sequence_index=int(cid[-2:]),
                   duration_s=duration, model_target={"preferred": model})


def test_clip_estimates_use_registry_prices_and_snapped_durations(demo_project, registry) -> None:
    bm = BudgetManager(demo_project, registry=registry)
    clip = _clip("SC01_SH01_C01", 6.0)
    assert bm.estimate_clip(clip) == pytest.approx(2 * 0.20 * 6 + 2 * QA_PER_TAKE)  # takes_per_clip = 2
    assert bm.estimate_clip(clip, model_id="veo", takes=1) == pytest.approx(0.40 * 6 + QA_PER_TAKE)
    odd = _clip("SC01_SH01_C02", 5.0)
    assert bm.estimate_clip(odd, model_id="veo", takes=1) == pytest.approx(0.40 * 6 + QA_PER_TAKE)  # 5 s -> 6 s
    scene = demo_project.load_scene("SC01")
    scene.model_override = "veo"  # chat: `model SC01 veo`
    demo_project.save_scene(scene)
    assert bm.estimate_clip(clip, takes=1) == pytest.approx(0.40 * 6 + QA_PER_TAKE)


def test_cap_defaults_to_estimate_times_multiplier(demo_project, registry) -> None:
    for c in (_clip("SC01_SH01_C01", 6.0), _clip("SC01_SH02_C01", 8.0)):
        demo_project.save_clip(c)
    bm = BudgetManager(demo_project, registry=registry)
    estimate = bm.estimate_scene("SC01")
    assert estimate == pytest.approx(2 * 0.20 * 14 + 4 * QA_PER_TAKE)
    cap = bm.ensure_cap("SC01")
    assert cap == pytest.approx(estimate * 1.25)  # settings.budget_cap_multiplier
    assert StateStore(demo_project).load().scenes["SC01"].budget_cap == pytest.approx(cap)
    assert bm.ensure_cap("SC01") == cap  # stored once, never silently re-derived
    assert [r["event"] for r in read_jsonl(bm.ledger_path)] == ["cap_set"]


def test_breakdown_estimate_before_any_clip_exists(demo_project, registry) -> None:
    bm = BudgetManager(demo_project, registry=registry)
    scene = demo_project.load_scene("SC02")
    scene.breakdown = SceneBreakdown(estimated_cost=4.0)
    demo_project.save_scene(scene)
    assert bm.estimate_scene("SC02") == 4.0
    scene.breakdown = SceneBreakdown(estimated_clips=2, estimated_duration_s=12.0, recommended_model="veo")
    demo_project.save_scene(scene)
    assert bm.estimate_scene("SC02") == pytest.approx(2 * (2 * 0.40 * 6 + 2 * QA_PER_TAKE))


def test_charge_persists_across_state_reloads(demo_project, registry) -> None:
    bm = BudgetManager(demo_project, registry=registry)
    assert bm.charge("SC01", 1.25, ref="SC01_SH01_C01_T01") == 1.25
    assert bm.charge("SC01", 0.5, ref="SC01_SH01_C01_T02", note="retry after a filter") == 1.75
    reloaded = StateStore(demo_project).load()  # a brand-new store reads state.json from disk
    assert reloaded.scenes["SC01"].budget_spent == 1.75 and reloaded.budget_spent_total == 1.75
    again = BudgetManager(demo_project, registry=registry, state=StateStore(demo_project))
    assert again.check("SC01", 0).spent == 1.75
    charges = [r for r in read_jsonl(demo_project.path("logs", "budget.jsonl")) if r["event"] == "charge"]
    assert [(r["ref"], r["amount"], r["spent"]) for r in charges] == [
        ("SC01_SH01_C01_T01", 1.25, 1.25), ("SC01_SH01_C01_T02", 0.5, 1.75)]


def test_an_overrun_asks_instead_of_continuing(demo_project, registry) -> None:
    bm = BudgetManager(demo_project, registry=registry)
    bm.state.set_budget_cap("SC01", 3.0)
    bm.charge("SC01", 2.5, ref="SC01_SH01_C01_T01")
    ok = bm.require("SC01", 0.5)
    assert ok.ok and ok.remaining == pytest.approx(0.5) and "fine" in ok.message
    with pytest.raises(BudgetExceeded) as caught:
        bm.require("SC01", 1.0)
    check = caught.value.check
    assert (check.ok, check.cap, check.spent, check.planned) == (False, 3.0, 2.5, 1.0)
    assert "over budget" in check.message and "`budget SC01" in check.message and "won't continue" in check.message
    assert str(caught.value) == check.message
    assert read_jsonl(bm.ledger_path)[-1]["event"] == "blocked"
    assert StateStore(demo_project).load().scenes["SC01"].budget_spent == 2.5  # nothing was spent


def test_only_a_recorded_decision_can_raise_a_cap(demo_project, registry) -> None:
    bm = BudgetManager(demo_project, registry=registry)
    bm.state.set_budget_cap("SC01", 3.0)
    with pytest.raises(PermissionError):
        bm.raise_cap("SC01", 10.0, decision_id="D000042")
    assert StateStore(demo_project).load().scenes["SC01"].budget_cap == 3.0
    d = DecisionLog(demo_project).record(tier="vision", target_artifact_id="SC01", action="override",
                                         payload={"budget_cap": 10.0})
    assert bm.raise_cap("SC01", 10.0, decision_id=d.id) == 10.0
    assert StateStore(demo_project).load().scenes["SC01"].budget_cap == 10.0
    assert bm.require("SC01", 6.0).ok
    assert read_jsonl(bm.ledger_path)[-1]["decision_id"] == d.id
    with pytest.raises(ValueError):
        bm.raise_cap("SC01", -1, decision_id=d.id)
    with pytest.raises(ValueError):
        bm.raise_cap("SC01", 20, decision_id="../D000001")


@pytest.mark.parametrize("amount", [-1, float("nan"), float("inf"), "lots"])
def test_bad_amounts_are_refused(demo_project, registry, amount) -> None:
    bm = BudgetManager(demo_project, registry=registry)
    with pytest.raises(ValueError):
        bm.charge("SC01", amount, ref="SC01_SH01_C01_T01")
    with pytest.raises(ValueError):
        bm.check("SC01", amount)


def test_charge_needs_a_ref_and_a_safe_scene_id(demo_project, registry) -> None:
    bm = BudgetManager(demo_project, registry=registry)
    with pytest.raises(ValueError):
        bm.charge("SC01", 1.0, ref="  ")
    with pytest.raises(ValueError):
        bm.charge("../SC01", 1.0, ref="x")
    assert StateStore(demo_project).load().budget_spent_total == 0.0


def test_scene_without_any_estimate_asks_for_a_cap(demo_project, registry) -> None:
    bm = BudgetManager(demo_project, registry=registry)
    assert bm.check("SC02", 0).ok
    chk = bm.check("SC02", 1.0)
    assert not chk.ok and chk.cap is None and "`budget SC02" in chk.message
    with pytest.raises(BudgetExceeded):
        bm.require("SC02", 1.0)


def test_report_is_plain_language(demo_project, registry) -> None:
    bm = BudgetManager(demo_project, registry=registry)
    assert bm.report().startswith("Budget: nothing spent yet.")
    bm.state.set_budget_cap("SC01", 4.0)
    bm.charge("SC01", 1.0, ref="SC01_SH01_C01_T01")
    text = bm.report()
    assert "SC01: $1.00 of $4.00 spent (25%), $3.00 left." in text and "Total spent: $1.00." in text
    assert "estimates until they are checked against billing" in text
    assert bm.report("SC01").startswith("SC01: $1.00 of $4.00 spent")
    bm.charge("SC01", 4.0, ref="SC01_SH01_C01_T02")
    assert "OVER the $4.00 cap" in bm.report("SC01")
