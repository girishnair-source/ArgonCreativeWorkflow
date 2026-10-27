"""Workstream A — human gates G1–G7: chat rendering, reply parsing, persistence, auto-pass rules (BUILD_SPEC_v3 §9)."""

from __future__ import annotations

import pytest

from agy_studio.decisions import DecisionLog, PreferenceStore
from agy_studio.fsio import PathSecurityError, read_jsonl, read_yaml
from agy_studio.gates import (
    AUTO_PASS_GATES,
    GATE_TIERS,
    NEVER_AUTO,
    GateItem,
    GateManager,
    GatePacket,
    GateResponse,
    parse_gate_response,
    render_gate,
)
from agy_studio.models import (
    AudioLine,
    AudioSpec,
    ClipStatus,
    Conditioning,
    DecisionAction,
    DecisionTier,
    GateId,
    GenClip,
    SceneBreakdown,
    ShotSpec,
)
from agy_studio.project import parse_artifact_id

CLIPS = [f"SC01_SH02_C{i:02d}" for i in range(1, 5)]


def _g5_packet(**kw) -> GatePacket:
    items = [GateItem(item_id=c, recommendation="T2", qa_composite=0.8, transcript_ok=True, speaker_sim=0.9)
             for c in CLIPS]
    return GatePacket(gate="G5", scene_id="SC01", items=items, **kw)


# ---------------------------------------------------------------------------------------------
# Rendering (§9.3)
# ---------------------------------------------------------------------------------------------
def test_gate_tiers_match_the_spec() -> None:
    assert {g.value: t.value for g, t in GATE_TIERS.items()} == {
        "G1": "vision", "G2": "vision", "G3": "craft", "G4": "craft", "G5": "craft", "G6": "craft", "G7": "vision"}
    assert AUTO_PASS_GATES == {GateId.G2, GateId.G5}


def test_header_matches_the_chat_contract() -> None:
    packet = GatePacket(gate="G5", scene_id="SC12", risk="high", model="veo", blocks=["SC12 assembly", "SC13 entry"],
                        items=[GateItem(item_id=f"SC12_SH04_C{i:02d}") for i in range(1, 7)])
    assert render_gate(packet).splitlines()[0] == (
        "[GATE G5 · SC12 · Take selection · 6 clips · risk: high · model: veo · blocks: SC12 assembly, SC13 entry]")


def test_render_shows_paths_metrics_reply_format_defaults_and_voice_anchor_prompt(demo_project) -> None:
    review = demo_project.path("scenes", "SC01", "review", "G5_manual")
    review.mkdir(parents=True)
    (review / "SC01_SH02_C01_T01.mp4").write_bytes(b"x")
    (review / "notes.txt").write_text("not media")
    item = GateItem(item_id="SC01_SH02_C01", recommendation="T2", qa_composite=0.82, transcript_ok=True,
                    speaker_sim=0.91, anchor_fanout=3, media=["scenes/SC01/review/G5_manual/SC01_SH02_C01_T02.mp4"])
    item2 = GateItem(item_id="SC01_SH02_C02", recommendation="T1", qa_composite=0.64, transcript_ok=False,
                     speaker_sim={"CHAR_RISHI": None}, flags=["lip-sync drift at 3.2s"])
    packet = GatePacket(gate="G5", scene_id="SC01", review_dir="scenes/SC01/review/G5_manual",
                        resolve_bin="AGY/SC01/review", items=[item, item2], voice_anchor_prompts=["CHAR_RISHI"])
    text = render_gate(packet, project_root=demo_project.root)
    root = str(demo_project.root)
    assert f"Review folder: {root}/scenes/SC01/review/G5_manual/" in text
    assert "Resolve bin: AGY/SC01/review" in text
    assert f"{root}/scenes/SC01/review/G5_manual/SC01_SH02_C01_T01.mp4" in text  # loose review media, absolute
    assert f"{root}/scenes/SC01/review/G5_manual/SC01_SH02_C01_T02.mp4" in text  # item media, absolute
    assert "notes.txt" not in text
    assert ("• SC01_SH02_C01 — recommend T2 · QA 0.82 · transcript ✓ · speaker-sim 0.91 · flags: none · "
            "anchor fan-out: 3 clips") in text
    assert ("• SC01_SH02_C02 — recommend T1 · QA 0.64 · transcript ✗ · speaker-sim CHAR_RISHI — · "
            "flags: lip-sync drift at 3.2s") in text
    assert "How to read this:" in text  # metrics are explained in plain language
    assert "• Pick CHAR_RISHI's voice anchor — all future takes will be scored against it." in text
    assert "Reply with: anchor CHAR_RISHI=<take id>" in text
    assert "How to reply:" in text and "C01:T2, C02:T1" in text
    assert "Defaults:" in text and NEVER_AUTO in text


def test_render_without_items_still_explains_reply_and_defaults() -> None:
    text = render_gate(GatePacket(gate="G7", notes=["picture lock covers SC01–SC02"]))
    assert text.splitlines()[0] == "[GATE G7 · Picture lock]"
    assert "approve all" in text and "reject all" in text and NEVER_AUTO in text
    assert "• picture lock covers SC01–SC02" in text


def test_packet_and_item_ids_are_validated() -> None:
    with pytest.raises(ValueError):
        GatePacket(gate="G5", scene_id="../SC01")
    with pytest.raises(ValueError):
        GateItem(item_id="SC01 SH02")
    with pytest.raises(ValueError):
        GatePacket(gate="G8")


# ---------------------------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------------------------
def test_parse_the_canonical_g5_reply_with_short_ids_and_a_free_text_note() -> None:
    resp = parse_gate_response("G5", 'C01:T2, C02:T1, C03:regen "slower delivery, less head movement", C04:ok. '
                                     "Overall: keep the handheld energy", item_ids=CLIPS)
    by_item = {c.item_id: c for c in resp.choices}
    assert list(by_item) == CLIPS
    assert (by_item["SC01_SH02_C01"].action, by_item["SC01_SH02_C01"].value) == ("choose", "SC01_SH02_C01_T02")
    assert by_item["SC01_SH02_C02"].value == "SC01_SH02_C02_T01"
    assert by_item["SC01_SH02_C03"].action == "regen"
    assert by_item["SC01_SH02_C03"].steer == "slower delivery, less head movement"
    assert by_item["SC01_SH02_C04"].action == "ok"
    assert "keep the handheld energy" in resp.free_text
    assert not resp.approve_all and not resp.reject_all and resp.unresolved == []
    assert resp.interpretation.startswith("Understood:") and "\n" not in resp.interpretation


def test_parse_reply_variants() -> None:
    curly = parse_gate_response("G5", "C03:regen \u201cslower delivery, less head movement\u201d", item_ids=CLIPS)
    assert [(c.item_id, c.steer) for c in curly.choices] == [("SC01_SH02_C03", "slower delivery, less head movement")]
    assert parse_gate_response("G5", "approve all").approve_all
    rej = parse_gate_response("G3", 'reject all "the boards ignore the axis of action"')
    assert rej.reject_all and rej.free_text == "the boards ignore the axis of action"
    alt = parse_gate_response("G3", 'SH02:alternatives 3 "differ in lens"', item_ids=["SC01_SH01", "SC01_SH02"])
    assert alt.choices[0].model_dump() == {"item_id": "SC01_SH02", "action": "alternatives", "value": "3",
                                           "steer": "differ in lens"}
    anchor = parse_gate_response("G5", "anchor CHAR_RISHI=SC01_SH01_C01_T02", item_ids=CLIPS)
    assert anchor.anchors == {"CHAR_RISHI": "SC01_SH01_C01_T02"} and anchor.choices == []
    cmd = parse_gate_response("G5", 'regen C3 "more breath between words"; pick T1 for c1', item_ids=CLIPS)
    assert [(c.item_id, c.action, c.value, c.steer) for c in cmd.choices] == [
        ("SC01_SH02_C03", "regen", None, "more breath between words"),
        ("SC01_SH02_C01", "choose", "SC01_SH02_C01_T01", None)]
    last = parse_gate_response("G5", "C01:T2, C01:T1", item_ids=CLIPS)
    assert [c.value for c in last.choices] == ["SC01_SH02_C01_T01"]  # a later instruction wins
    unknown = parse_gate_response("G5", "C09:T1", item_ids=CLIPS)
    assert unknown.unresolved == ["C09"] and unknown.choices == [] and "C09" in unknown.interpretation
    polite = parse_gate_response("G5", "C02:ok, thanks!", item_ids=CLIPS)
    assert polite.free_text == "" and [c.action for c in polite.choices] == ["ok"]
    exc = parse_gate_response("G5", 'approve all except C03:regen "softer"', item_ids=CLIPS)
    assert exc.approve_all and [(c.item_id, c.action, c.steer) for c in exc.choices] == [
        ("SC01_SH02_C03", "regen", "softer")]
    nothing = parse_gate_response("G5", "thanks!", item_ids=CLIPS)
    assert nothing.choices == [] and nothing.free_text == "" and "nothing is recorded" in nothing.interpretation


def test_option_choices_for_open_decisions_and_alternatives(demo_project) -> None:
    g1 = parse_gate_response("G1", "OD3:choose on, OD1: choose native", item_ids=["OD1", "OD3", "CHAR_RISHI"])
    assert [(c.item_id, c.action, c.value) for c in g1.choices] == [("OD3", "choose", "on"),
                                                                    ("OD1", "choose", "native")]
    assert g1.free_text == "" and "OD3: choose on" in g1.interpretation
    g3 = parse_gate_response("G3", "SH01:choose A1", item_ids=["SC01_SH01", "SC01_SH02"])
    assert [(c.item_id, c.action, c.value) for c in g3.choices] == [("SC01_SH01", "choose", "A1")]
    g5 = parse_gate_response("G5", "C02: pick 2, C03: choose T1", item_ids=CLIPS)
    assert [(c.item_id, c.value) for c in g5.choices] == [("SC01_SH02_C02", "SC01_SH02_C02_T02"),
                                                          ("SC01_SH02_C03", "SC01_SH02_C03_T01")]
    # a longer sentence after the verb is a note, never a choice
    note = parse_gate_response("G3", "SH02: pick a wider lens", item_ids=["SC01_SH01", "SC01_SH02"])
    assert note.choices == [] and note.free_text == "SH02: pick a wider lens"
    assert parse_gate_response("G5", "C03: use 2 seconds less", item_ids=CLIPS).choices == []
    gm = GateManager(demo_project)
    gm.open(GatePacket(gate="G1", items=[GateItem(item_id="OD3"), GateItem(item_id="OD1")]))
    _, decisions = gm.respond("G1", None, "OD3:choose on")
    assert [(d.target_artifact_id, d.action, d.payload) for d in decisions] == [
        ("OD3", DecisionAction.CHOOSE, {"command": "choose", "value": "on"})]
    assert gm.state.gate_status(None, GateId.G1) == "awaiting"  # OD1 is still unanswered


def test_quoted_text_never_triggers_commands() -> None:
    resp = parse_gate_response("G5", 'C03:regen "approve all of it, C01:T1, reject all the noise"', item_ids=CLIPS)
    assert not resp.approve_all and not resp.reject_all
    assert [(c.item_id, c.action, c.steer) for c in resp.choices] == [
        ("SC01_SH02_C03", "regen", "approve all of it, C01:T1, reject all the noise")]


# ---------------------------------------------------------------------------------------------
# GateManager
# ---------------------------------------------------------------------------------------------
def test_open_persists_the_packet_and_sets_awaiting(demo_project) -> None:
    gm = GateManager(demo_project)
    text = gm.open(_g5_packet())
    assert text.startswith("[GATE G5 · SC01 · Take selection · 4 clips]")
    assert gm.state.gate_status("SC01", "G5") == "awaiting"
    records = list(demo_project.path("scenes", "SC01", "review").glob("G5_*/gate.yaml"))
    assert len(records) == 1
    rec = read_yaml(records[0])
    assert (rec["status"], rec["tier"], rec["rendered"]) == ("awaiting", "craft", text)
    assert [i["item_id"] for i in rec["packet"]["items"]] == CLIPS
    assert [(p["gate"], p["scene_id"], p["items"]) for p in gm.pending()] == [("G5", "SC01", 4)]
    assert gm.latest_packet("G5", "SC01").items[0].item_id == CLIPS[0]


def test_reply_records_one_decision_per_choice_and_passes_the_gate(demo_project) -> None:
    gm = GateManager(demo_project)
    gm.open(_g5_packet())
    _, decisions = gm.respond("G5", "SC01", "C01:T2, C02:T1, C03:ok, C04:ok")
    assert [d.action for d in decisions] == [DecisionAction.CHOOSE, DecisionAction.CHOOSE,
                                             DecisionAction.APPROVE, DecisionAction.APPROVE]
    assert all(d.gate == "G5" and d.tier == DecisionTier.CRAFT for d in decisions)
    assert decisions[0].payload == {"command": "choose", "value": "SC01_SH02_C01_T02", "take_id": "SC01_SH02_C01_T02"}
    assert decisions[2].payload == {"command": "ok", "recommendation": "T2"}
    assert gm.state.gate_status("SC01", "G5") == "passed"
    assert [d.id for d in DecisionLog(demo_project).all()] == [d.id for d in decisions]
    assert PreferenceStore(demo_project).list() == []  # nothing steered -> no standing notes
    assert [r["event"] for r in read_jsonl(gm.log_path)] == ["open", "close"]
    assert gm.pending() == []


def test_regen_and_free_text_send_the_gate_back_and_become_notes(demo_project) -> None:
    gm = GateManager(demo_project)
    gm.open(_g5_packet())
    _, decisions = gm.respond("G5", "SC01", 'C01:T2, C03:regen "slower delivery". Overall: keep the handheld energy')
    assert [d.action for d in decisions] == [DecisionAction.CHOOSE, DecisionAction.STEER, DecisionAction.STEER]
    assert decisions[1].payload == {"command": "regen", "steer": "slower delivery"}
    assert decisions[2].target_artifact_id == "SC01" and "keep the handheld energy" in decisions[2].payload["text"]
    assert gm.state.gate_status("SC01", "G5") == "rejected"
    queue = gm.state.load().regen_queue
    assert [(r["clip_id"], r["steer"], r["decision_id"]) for r in queue] == [
        ("SC01_SH02_C03", "slower delivery", decisions[1].id)]
    prefs = PreferenceStore(demo_project).list()
    assert [p.scope for p in prefs] == ["scene:SC01", "scene:SC01"]
    assert prefs[0].text == "slower delivery" and "handheld" in prefs[1].text


def test_partial_reply_leaves_the_gate_waiting(demo_project) -> None:
    gm = GateManager(demo_project)
    gm.open(_g5_packet())
    gm.respond("G5", "SC01", "C01:T2")
    assert gm.state.gate_status("SC01", "G5") == "awaiting"
    _, decisions = gm.respond("G5", "SC01", "approve all")
    assert decisions[0].payload["recommendations"] == {c: "T2" for c in CLIPS}
    assert gm.state.gate_status("SC01", "G5") == "passed"


def test_approve_all_except_excludes_the_listed_items(demo_project) -> None:
    gm = GateManager(demo_project)
    gm.open(_g5_packet())
    _, decisions = gm.respond("G5", "SC01", 'approve all except C03:regen "softer"')
    approve = decisions[0]
    assert approve.action == DecisionAction.APPROVE and approve.payload["except"] == ["SC01_SH02_C03"]
    assert "SC01_SH02_C03" not in approve.payload["recommendations"]
    assert gm.state.gate_status("SC01", "G5") == "rejected"


def test_voice_anchor_reply_is_a_vision_decision_on_the_voice_profile(demo_project) -> None:
    gm = GateManager(demo_project)
    gm.open(_g5_packet(voice_anchor_prompts=["CHAR_RISHI"]))
    _, decisions = gm.respond("G5", "SC01", "approve all, anchor CHAR_RISHI=C01_T2")
    anchor = [d for d in decisions if d.payload.get("command") == "anchor"]
    assert len(anchor) == 1 and anchor[0].tier == DecisionTier.VISION
    assert anchor[0].target_artifact_id == "VOICE_CHAR_RISHI"
    assert anchor[0].payload["anchor_take"] == "SC01_SH02_C01_T02"


def test_global_gates_live_in_logs_and_scope_is_enforced(demo_project) -> None:
    gm = GateManager(demo_project)
    text = gm.open(GatePacket(gate="G1", notes=["3 characters, 2 locations, 2 scenes"]))
    assert text.startswith("[GATE G1 · Bible, style lock & voices]")
    assert list(demo_project.path("logs", "gates").glob("G1_*/gate.yaml"))
    assert gm.state.gate_status(None, "G1") == "awaiting"
    with pytest.raises(ValueError):
        gm.open(GatePacket(gate="G1", scene_id="SC01"))
    with pytest.raises(ValueError):
        gm.open(GatePacket(gate="G5"))
    _, decisions = gm.respond("G1", None, "approve all")
    assert decisions[0].tier == DecisionTier.VISION and decisions[0].target_artifact_id == "PROJECT"
    assert gm.state.gate_status(None, "G1") == "passed"


def test_close_refuses_a_reply_meant_for_another_gate(demo_project) -> None:
    with pytest.raises(ValueError):
        GateManager(demo_project).close("G4", "SC01", GateResponse(gate="G5", approve_all=True))


def test_open_refuses_media_outside_the_project(demo_project, tmp_path) -> None:
    outside = tmp_path / "elsewhere.mp4"
    outside.write_bytes(b"x")
    packet = GatePacket(gate="G5", scene_id="SC01", items=[GateItem(item_id=CLIPS[0], media=[str(outside)])])
    gm = GateManager(demo_project)
    with pytest.raises(PathSecurityError):
        gm.open(packet)
    assert gm.state.gate_status("SC01", "G5") == "pending"


# ---------------------------------------------------------------------------------------------
# Auto-pass: only G2 (nothing new) and G5 (low-risk auto-select)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("gate", ["G1", "G3", "G4", "G6", "G7"])
def test_auto_pass_is_refused_for_vision_and_craft_gates(demo_project, gate) -> None:
    gm = GateManager(demo_project)
    with pytest.raises(ValueError, match="never auto-proceeds"):
        gm.auto_pass(gate, "SC01", reason="nothing new")
    assert DecisionLog(demo_project).all() == []
    st = gm.state.load()
    assert st.gates == {} and "SC01" not in st.scenes


def test_g2_auto_passes_when_every_pack_and_geometry_is_locked(demo_project) -> None:
    gm = GateManager(demo_project)
    assert gm.auto_pass_blockers("G2", "SC01") == []
    d = gm.auto_pass("G2", "SC01", reason="every pack and the terrace geometry are already in the library")
    assert (d.payload, d.user, d.tier, d.gate) == ({"auto": True}, "agy", DecisionTier.VISION, "G2")
    assert gm.state.gate_status("SC01", "G2") == "auto_passed"
    assert read_jsonl(gm.log_path)[-1]["event"] == "auto_pass"
    with pytest.raises(ValueError):
        gm.auto_pass("G2", "SC01", reason="   ")


def test_g2_refuses_to_auto_pass_new_packs(unlocked_project) -> None:
    gm = GateManager(unlocked_project)
    assert any("is new" in b for b in gm.auto_pass_blockers("G2", "SC01"))
    with pytest.raises(ValueError, match="can't auto-pass"):
        gm.auto_pass("G2", "SC01", reason="looks fine")
    assert gm.state.gate_status("SC01", "G2") == "pending"


def test_g5_auto_select_needs_low_risk_and_every_voice_anchor(demo_project) -> None:
    gm = GateManager(demo_project)
    blockers = gm.auto_pass_blockers("G5", "SC01")
    assert any("no scene breakdown" in b for b in blockers)
    assert any("CHAR_RISHI speaks for the first time" in b for b in blockers)
    with pytest.raises(ValueError):
        gm.auto_pass("G5", "SC01", reason="low risk")
    scene = demo_project.load_scene("SC01")
    scene.breakdown = SceneBreakdown(risk_tier="low")
    demo_project.save_scene(scene)
    for char in ("CHAR_RISHI", "CHAR_MOHNISH", "CHAR_GUARD"):
        vp = demo_project.load_voice_profile(char)
        demo_project.save_voice_profile(vp.model_copy(update={"anchor_take": "SC01_SH01_C01_T01"}))
    assert gm.auto_pass_blockers("G5", "SC01") == []
    gm.auto_pass("G5", "SC01", reason="low-risk scene; every recommended take passed QA")
    assert gm.state.gate_status("SC01", "G5") == "auto_passed"


def _speaking_clip(cid: str, char: str, line_id: str, text: str, **kw) -> GenClip:
    ids = parse_artifact_id(cid)
    audio = AudioSpec(dialogue=[AudioLine(character_id=char, line_id=line_id, text=text)])
    return GenClip(id=cid, shot_id=ids["shot"], scene_id=ids["scene"], sequence_index=1, duration_s=6, lines=[line_id],
                   spec=ShotSpec(audio_spec=audio),
                   conditioning=Conditioning(reference_entity_states=["CHAR_MOHNISH_STATE_DEFAULT"]), **kw)


def test_g5_anchor_guard_covers_only_the_auto_selected_clips(demo_project) -> None:
    """Wave 1 auto-selects Mohnish's clip; Rishi's first line (a later wave) must not block that auto-pass."""
    scene = demo_project.load_scene("SC01")
    scene.breakdown = SceneBreakdown(risk_tier="low")
    demo_project.save_scene(scene)
    demo_project.save_clip(_speaking_clip("SC01_SH01_C01", "CHAR_MOHNISH", "SC01_L001",
                                          "भाई, लाइव चालू है... दो हज़ार लोग देख रहे हैं!",
                                          status=ClipStatus.SELECTED, selected_take_id="SC01_SH01_C01_T01"))
    demo_project.save_clip(_speaking_clip("SC01_SH02_C01", "CHAR_RISHI", "SC01_L002", "देख, अब असली खेल शुरू होगा।"))
    vp = demo_project.load_voice_profile("CHAR_MOHNISH")
    demo_project.save_voice_profile(vp.model_copy(update={"anchor_take": "SC01_SH01_C01_T01"}))
    gm = GateManager(demo_project)
    assert gm.auto_pass_blockers("G5", "SC01") == []  # default scope: the clips already selected
    assert any("CHAR_RISHI speaks for the first time" in b
               for b in gm.auto_pass_blockers("G5", "SC01", clip_ids=["SC01_SH02_C01"]))
    d = gm.auto_pass("G5", "SC01", reason="low-risk auto-select", clip_ids=["SC01_SH01_C01"])
    assert d.payload == {"auto": True, "clips": ["SC01_SH01_C01"]}
    assert gm.state.gate_status("SC01", "G5") == "auto_passed"
