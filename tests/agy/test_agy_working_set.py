"""Workstream A — scene working sets, the only context an agent sees for a scene (BUILD_SPEC_v3 §4.5, acceptance 5)."""

from __future__ import annotations

import json

import pytest

from agy_studio.decisions import DecisionLog, PreferenceStore
from agy_studio.fsio import PathSecurityError
from agy_studio.working_set import (
    build_working_set,
    load_working_set,
    pin,
    resolve_refs,
    show_working_set,
    working_set_hash,
    working_set_path,
)

SC01_LINE = "भाई, लाइव चालू है... दो हज़ार लोग देख रहे हैं!"
SC02_LINE = "भाग! सीढ़ियों से नीचे!"


def _dump(ws) -> str:
    return json.dumps(ws.model_dump(mode="json"), ensure_ascii=False)


def test_working_set_points_into_the_library_for_exactly_this_scenes_assets(demo_project) -> None:
    ws = build_working_set(demo_project, "SC01")
    assert [p.id for p in ws.entity_states] == [
        "CHAR_RISHI_STATE_DEFAULT", "CHAR_MOHNISH_STATE_DEFAULT", "PROP_PHONE_STATE_DEFAULT", "CHAR_RISHI_STATE_INJURED"]
    assert all(p.ref.startswith("library/packs/") and p.ref.endswith("/pack.yaml") for p in ws.entity_states)
    assert all(demo_project.abs(p.ref).is_file() for p in ws.entity_states)  # pointers, never copies
    assert [p.id for p in ws.voice_profiles] == ["VOICE_CHAR_MOHNISH", "VOICE_CHAR_RISHI", "VOICE_CHAR_GUARD"]
    assert all(p.ref.startswith("library/voices/") and demo_project.abs(p.ref).is_file() for p in ws.voice_profiles)
    assert ws.environment.id == "ENV_TERRACE" and ws.style_lock.approval_status == "locked"
    assert ws.story_so_far == [] and ws.prev_scene_exit_state == {}
    assert ws.next_scene_entry_needs == {"scene_id": "SC02", "entry_state": {"CHAR_RISHI": "CHAR_RISHI_STATE_INJURED"},
                                         "entity_states_active": ["CHAR_RISHI_STATE_INJURED",
                                                                  "CHAR_MOHNISH_STATE_DEFAULT"]}
    assert "CHAR_GUARD_STATE_DEFAULT" not in [p.id for p in ws.entity_states]  # heard only, never seen
    assert ws.working_set_hash == working_set_hash(ws)
    assert load_working_set(demo_project, "SC01") == ws


def test_no_other_scenes_dialogue_ever_enters(demo_project) -> None:
    sc01 = _dump(build_working_set(demo_project, "SC01"))
    assert SC01_LINE in sc01
    for leak in (SC02_LINE, "SC02_L001", "Bhaag", "flee down the stairwell"):
        assert leak not in sc01
    sc02_ws = build_working_set(demo_project, "SC02")
    sc02 = _dump(sc02_ws)
    assert SC02_LINE in sc02
    for leak in (SC01_LINE, "SC01_L00", "Bhai, live", "नीचे उतरो"):
        assert leak not in sc02
    assert sc02_ws.story_so_far == ["SC01 (terrace_dare): Mohnish starts the live stream; Rishi climbs the parapet; "
                                    "Rishi plays to the camera until the guard's shout startles him"]
    assert sc02_ws.prev_scene_exit_state["CHAR_RISHI"] == "CHAR_RISHI_STATE_INJURED"
    assert sc02_ws.next_scene_entry_needs == {}


def test_hash_is_stable_and_changes_when_a_scene_preference_is_added(demo_project) -> None:
    first = build_working_set(demo_project, "SC01")
    path = working_set_path(demo_project, "SC01")
    saved = path.read_bytes()
    second = build_working_set(demo_project, "SC01")
    assert first.working_set_hash == second.working_set_hash and len(first.working_set_hash) == 16
    assert path.read_bytes() == saved and not (path.parent / "_history").exists()  # unchanged -> not rewritten
    PreferenceStore(demo_project).add(scope="scene:SC02", text="harsher bulb light")
    assert build_working_set(demo_project, "SC01").working_set_hash == first.working_set_hash  # another scene's note
    PreferenceStore(demo_project).add(scope="scene:SC01", text="keep the camera low and handheld")
    third = build_working_set(demo_project, "SC01")
    assert third.working_set_hash != first.working_set_hash
    assert [p.text for p in third.scene_preferences] == ["keep the camera low and handheld"]
    assert load_working_set(demo_project, "SC01").working_set_hash == third.working_set_hash
    assert third.generated_by.working_set_hash == third.working_set_hash
    assert any((path.parent / "_history").iterdir())  # the previous working set is archived, never lost


def test_character_and_project_notes_follow_the_cast(demo_project) -> None:
    store = PreferenceStore(demo_project)
    store.add(scope="project", text="no lens flares")
    store.add(scope="character:CHAR_GUARD", text="the guard is only ever heard, never seen")
    store.add(scope="agent:director", text="prefer fewer cuts")
    sc01 = build_working_set(demo_project, "SC01")
    sc02 = build_working_set(demo_project, "SC02")
    assert {p.text for p in sc01.scene_preferences} == {
        "no lens flares", "the guard is only ever heard, never seen", "prefer fewer cuts"}
    assert {p.text for p in sc02.scene_preferences} == {"no lens flares", "prefer fewer cuts"}


def test_feedback_history_holds_only_this_scenes_decisions(demo_project) -> None:
    log = DecisionLog(demo_project)
    mine = log.record(tier="craft", target_artifact_id="SC01_SH02_C01", action="steer", payload={"steer": "slower"})
    log.record(tier="craft", target_artifact_id="SC02", action="steer", payload={"text": "more echo"})
    ws = build_working_set(demo_project, "SC01")
    assert [f["decision_id"] for f in ws.feedback_history] == [mine.id]
    assert ws.feedback_history[0]["text"] == "slower" and ws.feedback_history[0]["target"] == "SC01_SH02_C01"
    assert "more echo" not in _dump(ws)


def test_resolve_refs_loads_records_and_flags_later_changes(demo_project) -> None:
    ws = build_working_set(demo_project, "SC01")
    refs = resolve_refs(demo_project, ws)
    assert refs["warnings"] == []
    assert set(refs["entity_states"]) == {p.id for p in ws.entity_states}
    assert all(img.is_file() for e in refs["entity_states"].values() for img in e["images"])
    assert refs["environment"].id == "ENV_TERRACE"
    assert set(refs["voice_profiles"]) == {p.id for p in ws.voice_profiles}
    vp = demo_project.load_voice_profile("CHAR_RISHI")
    demo_project.save_voice_profile(vp.model_copy(update={"anchor_take": "SC01_SH02_C01_T01"}))  # new library version
    assert any("VOICE_CHAR_RISHI changed" in w for w in resolve_refs(demo_project, ws)["warnings"])


def test_unlocked_project_points_at_proposed_bible_records(unlocked_project) -> None:
    ws = build_working_set(unlocked_project, "SC01")
    assert all(p.ref.startswith("bible/entities/") and p.approval_status == "proposed" for p in ws.entity_states)
    assert all(p.ref == "bible/casting.yaml" and p.approval_status == "proposed" for p in ws.voice_profiles)
    assert resolve_refs(unlocked_project, ws)["warnings"] == []


def test_pin_copies_the_file_into_the_project(demo_project, tmp_path) -> None:
    src = tmp_path / "mood ref (final).png"
    src.write_bytes(b"\x89PNG fake")
    before = build_working_set(demo_project, "SC01").working_set_hash
    ref = pin(demo_project, "SC01", src, "the exact sodium-lamp colour we want")
    assert ref.path == "scenes/SC01/pinned/mood_ref_final.png"
    copied = demo_project.abs(ref.path)
    assert copied.read_bytes() == b"\x89PNG fake"
    src.write_bytes(b"changed later")  # the project's copy no longer depends on the original
    assert copied.read_bytes() == b"\x89PNG fake"
    second = pin(demo_project, "SC01", src, "a second look")
    assert second.path == "scenes/SC01/pinned/mood_ref_final_2.png"
    assert copied.read_bytes() == b"\x89PNG fake"  # never overwritten
    ws = build_working_set(demo_project, "SC01")
    assert [p.path for p in ws.pinned_extra_refs] == [ref.path, second.path]
    assert ws.working_set_hash != before
    assert resolve_refs(demo_project, ws)["pinned"][0] == copied


def test_pin_refuses_unsafe_inputs(demo_project, tmp_path) -> None:
    script = tmp_path / "run.sh"
    script.write_text("echo hi")
    with pytest.raises(ValueError):
        pin(demo_project, "SC01", script, "not a reference")
    folder = tmp_path / "folder.png"
    folder.mkdir()
    with pytest.raises(ValueError):
        pin(demo_project, "SC01", folder, "a directory is not a file")
    with pytest.raises(FileNotFoundError):
        pin(demo_project, "SC01", tmp_path / "missing.png", "missing")
    ok = tmp_path / "ok.png"
    ok.write_bytes(b"x")
    with pytest.raises(ValueError):
        pin(demo_project, "SC01", ok, "   ")
    with pytest.raises(PathSecurityError):
        pin(demo_project, "../SC01", ok, "escape attempt")
    pinned = demo_project.path("scenes", "SC01", "pinned")
    assert not pinned.exists() or not [p for p in pinned.iterdir() if p.suffix in (".sh", ".png")]
    assert build_working_set(demo_project, "SC01").pinned_extra_refs == []


def test_show_working_set_is_plain_language(demo_project) -> None:
    PreferenceStore(demo_project).add(scope="scene:SC01", text="keep the camera low")
    text = show_working_set(demo_project, "SC01")
    assert text.startswith("Working set — SC01 terrace_dare (hash ")
    assert SC01_LINE in text and SC02_LINE not in text
    assert "SC01_L003 CHAR_GUARD (off-screen): कौन है वहाँ? नीचे उतरो! — Kaun hai wahan? Neeche utro!" in text
    assert "[scene:SC01] keep the camera low" in text
    assert "Previous scene ends with: — (first scene)" in text
    assert "Next scene needs: SC02 starts with CHAR_RISHI: CHAR_RISHI_STATE_INJURED" in text
