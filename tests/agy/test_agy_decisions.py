"""Workstream A — decisions, preferences and blocking questions (BUILD_SPEC_v3 §4.8, §9.2, §9.5). No network."""

from __future__ import annotations

import threading

import pytest

from agy_studio.decisions import (
    DecisionLog,
    PreferenceStore,
    QuestionLog,
    decision_text,
    decision_touches_scene,
    infer_scope,
    preference_is_relevant,
)
from agy_studio.fsio import append_jsonl, read_jsonl, read_yaml
from agy_studio.models import Decision, DecisionAction, DecisionTier, Preference
from agy_studio.state import StateStore


# ---------------------------------------------------------------------------------------------
# DecisionLog
# ---------------------------------------------------------------------------------------------
def test_record_appends_sequential_ids_and_persists_before_returning(demo_project) -> None:
    log = DecisionLog(demo_project)
    a = log.record(tier="craft", target_artifact_id="SC01_SH02_C01", action="choose",
                   payload={"take_id": "SC01_SH02_C01_T02"}, gate="G5")
    b = log.record(tier=DecisionTier.VISION, target_artifact_id="CHAR_RISHI_STATE_INJURED", action=DecisionAction.LOCK)
    assert (a.id, b.id) == ("D000001", "D000002")
    rows = read_jsonl(demo_project.decisions_path)
    assert [r["id"] for r in rows] == ["D000001", "D000002"]
    assert Decision.model_validate(rows[0]) == a  # the returned record IS the persisted one
    assert a.gate == "G5" and a.source.value == "chat" and a.user == "director"
    fresh = DecisionLog(demo_project)
    assert fresh.get("D000002") == b and fresh.get("D999999") is None
    assert [d.id for d in fresh.for_target("SC01_SH02")] == ["D000001"]
    assert fresh.for_target("SC01_SH02", prefix=False) == []
    assert [d.id for d in fresh.for_scene("SC01")] == ["D000001"]
    assert fresh.for_scene("SC02") == []


def test_steer_and_reject_with_text_become_scoped_preferences(demo_project) -> None:
    log = DecisionLog(demo_project)
    steer = log.record(tier="craft", target_artifact_id="SC01_SH02_C03", action="steer", gate="G5",
                       payload={"command": "regen", "steer": "slower delivery, less head movement"})
    pref = log.last_preference
    assert pref is not None and pref.scope == "scene:SC01"
    assert pref.text == "slower delivery, less head movement" and pref.source_decision_ids == [steer.id]
    log.record(tier="vision", target_artifact_id="CHAR_RISHI_STATE_INJURED", action="reject",
               payload={"text": "the scrape belongs on the LEFT forearm"})
    assert log.last_preference.scope == "character:CHAR_RISHI"
    log.record(tier="craft", target_artifact_id="SC01", action="steer", payload={"text": "warmer", "scope": "project"})
    assert log.last_preference.scope == "project"
    log.record(tier="craft", target_artifact_id="SC01_SH01_C01", action="approve", payload={"text": "lovely"})
    assert log.last_preference is None  # only steer / reject become standing notes
    log.record(tier="craft", target_artifact_id="SC01_SH01_C01", action="steer")
    assert log.last_preference is None  # nothing to remember without text
    store = PreferenceStore(demo_project)
    assert [p.id for p in store.list()] == ["P000001", "P000002", "P000003"]
    assert store.from_decision(steer).id == "P000001"  # idempotent: one preference per decision
    assert len(store.list()) == 3
    assert [p.id for p in store.for_decision(steer.id)] == ["P000001"]


def test_invalid_scope_is_refused_before_anything_is_written(demo_project) -> None:
    log = DecisionLog(demo_project)
    with pytest.raises(ValueError):
        log.record(tier="craft", target_artifact_id="SC01", action="steer",
                   payload={"text": "x", "scope": "scene:../SC02"})
    assert read_jsonl(demo_project.decisions_path) == []
    assert PreferenceStore(demo_project).list() == []


@pytest.mark.parametrize("bad", [
    {"target_artifact_id": ""},
    {"target_artifact_id": "SC01\nINJECTED"},
    {"tier": "cosmic"},
    {"action": "delete_everything"},
    {"gate": "G9"},
    {"source": "email"},
])
def test_record_validates_its_inputs(demo_project, bad) -> None:
    base = {"tier": "craft", "target_artifact_id": "SC01", "action": "approve"}
    with pytest.raises(ValueError):
        DecisionLog(demo_project).record(**{**base, **bad})
    assert read_jsonl(demo_project.decisions_path) == []


def test_payload_must_be_json_serialisable(demo_project) -> None:
    with pytest.raises(TypeError):
        DecisionLog(demo_project).record(tier="craft", target_artifact_id="SC01", action="approve",
                                         payload={"x": object()})
    assert read_jsonl(demo_project.decisions_path) == []


def test_concurrent_writers_never_reuse_an_id(demo_project) -> None:
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            log = DecisionLog(demo_project)
            for i in range(5):
                log.record(tier="polish", target_artifact_id=f"SC01_SH0{1 + n % 2}", action="annotate",
                           payload={"worker": n, "i": i})
        except BaseException as exc:  # surfaced by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    ids = [r["id"] for r in read_jsonl(demo_project.decisions_path)]
    assert sorted(ids) == [f"D{i:06d}" for i in range(1, 31)]


def test_backfill_recreates_a_preference_lost_to_a_kill(demo_project) -> None:
    d = Decision(id="D000001", ts="2026-01-01T00:00:00+00:00", tier="craft", target_artifact_id="SC02",
                 action="steer", payload={"text": "more echo in the stairwell"})
    append_jsonl(demo_project.decisions_path, d)  # the process died before the preference was written
    log = DecisionLog(demo_project)
    created = log.backfill_preferences()
    assert [(p.scope, p.text) for p in created] == [("scene:SC02", "more echo in the stairwell")]
    assert log.backfill_preferences() == []
    assert log.record(tier="craft", target_artifact_id="SC02", action="approve").id == "D000002"


# ---------------------------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------------------------
def test_preference_scoping_and_deactivation(demo_project) -> None:
    store = PreferenceStore(demo_project)
    p_proj = store.add(scope="project", text="no lens flares")
    p_sc1 = store.add(scope="scene:SC01", text="keep the camera low", weight=2)
    p_sc2 = store.add(scope="scene:SC02", text="harsher bulb light")
    p_char = store.add(scope="character:CHAR_GUARD", text="the guard is only ever heard, never seen")
    p_agent = store.add(scope="agent:director", text="prefer fewer cuts")
    assert {p.id for p in store.relevant(scene_id="SC01", character_ids=["CHAR_GUARD"])} == {
        p_proj.id, p_sc1.id, p_char.id}
    assert {p.id for p in store.relevant(scene_id="SC02")} == {p_proj.id, p_sc2.id}
    assert p_agent.id in {p.id for p in store.relevant(agent="director")}
    assert [p.id for p in store.list(scope="scene:SC01")] == [p_sc1.id] and p_sc1.weight == 2.0
    d = DecisionLog(demo_project).record(tier="craft", target_artifact_id="SC01", action="override")
    retired = store.deactivate(p_sc1.id, decision_id=d.id)
    assert retired.active is False
    assert p_sc1.id not in {p.id for p in store.list()}
    assert p_sc1.id in {p.id for p in store.list(active_only=False)}  # retired, never deleted
    data = read_yaml(demo_project.preferences_path)
    assert data["deactivations"][0]["preference_id"] == p_sc1.id
    assert data["deactivations"][0]["decision_id"] == d.id
    assert any((demo_project.root / "_history").glob("preferences_v*.yaml"))  # earlier versions archived
    with pytest.raises(ValueError):
        store.add(scope="project", text="   ")
    with pytest.raises(ValueError):
        store.add(scope="project", text="x", weight=float("nan"))
    with pytest.raises(ValueError):
        store.add(scope="everyone", text="x")
    with pytest.raises(KeyError):
        store.deactivate("P999999", decision_id=d.id)


def test_scope_and_relevance_helpers() -> None:
    assert infer_scope("SC03_SH01_C02_T01") == "scene:SC03"
    assert infer_scope("SC03") == "scene:SC03"
    assert infer_scope("CHAR_RISHI_STATE_INJURED") == "character:CHAR_RISHI"
    assert infer_scope("VOICE_CHAR_RISHI") == "character:CHAR_RISHI"
    assert infer_scope("STYLE_LOCK") == "project"
    assert infer_scope("SC03", {"scope": "agent:editor"}) == "agent:editor"
    d = Decision(id="D000001", ts="t", tier="craft", target_artifact_id="CHAR_RISHI", action="choose",
                 payload={"take_id": "SC04_SH01_C01_T02"}, rationale="because")
    assert decision_touches_scene(d, "SC04") and not decision_touches_scene(d, "SC01")
    assert decision_text(d) == "because"
    assert decision_text(d.model_copy(update={"payload": {"steer": " slower "}})) == "slower"
    pref = Preference(id="P000001", scope="user:girish", text="x", created_at="t")
    assert preference_is_relevant(pref, user="girish") and not preference_is_relevant(pref, scene_id="SC01")


# ---------------------------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------------------------
def test_questions_block_their_scene_until_answered(demo_project) -> None:
    state = StateStore(demo_project)
    qlog = QuestionLog(demo_project, state=state)
    q = qlog.ask("Should the guard ever be seen in SC01?", scene_id="SC01", stage="decomposed",
                 blocks=["SC01 G3"], options=["off-screen only", "silhouette in the doorway"])
    q2 = qlog.ask("Night or dusk for SC02?", scene_id="SC02")
    assert (q.id, q2.id) == ("Q000001", "Q000002")
    assert state.load().scenes["SC01"].blocked_by == [q.id]
    assert [x.id for x in qlog.open()] == [q.id, q2.id]
    assert [x.id for x in qlog.open("SC01")] == [q.id]
    d = DecisionLog(demo_project).record(tier="craft", target_artifact_id="SC01", action="answer_question",
                                         payload={"question_id": q.id, "text": "off-screen only"})
    answered = qlog.answer(q.id, "off-screen only", decision_id=d.id)
    assert answered.status == "answered" and answered.decision_id == d.id and answered.answered_at
    assert qlog.get(q.id).answer == "off-screen only"
    assert [x.id for x in qlog.open()] == [q2.id]
    assert state.load().scenes["SC01"].blocked_by == []
    assert state.load().scenes["SC02"].blocked_by == [q2.id]  # the other scene stays blocked
    assert len(read_jsonl(demo_project.questions_path)) == 3  # appended, never rewritten
    assert [x.id for x in qlog.all()] == [q.id, q2.id]
    with pytest.raises(KeyError):
        qlog.answer("Q999999", "x")
    with pytest.raises(ValueError):
        qlog.answer(q2.id, "   ")
    with pytest.raises(ValueError):
        qlog.ask("   ")
    with pytest.raises(ValueError):
        qlog.ask("x", scene_id="../SC01")
