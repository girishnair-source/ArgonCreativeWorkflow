"""End-to-end acceptance tests for AGY Studio (BUILD_SPEC_v3 §14 AC 1–11 + Workstreams B/C/D/E)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agy_studio.adapters.fake import SimulatedKill, default_backend, reset_default_backend
from agy_studio.agents.runner import AgentScopeError, run_agent
from agy_studio.decomposer import decompose_scene, decompose_shot
from agy_studio.editor import assemble_scene
from agy_studio.fsio import atomic_write_yaml, read_jsonl, read_yaml
from agy_studio.invalidation import AnchorGraph, ConfirmRequiredError, apply_invalidation
from agy_studio.linter import LintContext, lint_clip, lint_shot
from agy_studio.migrate_legacy import migrate_lfkod
from agy_studio.models import (
    ApprovalStatus,
    AudioLine,
    ClipStatus,
    DecisionAction,
    DecisionTier,
    DispatchStatus,
    EntityState,
    GlobalStage,
    ModelTarget,
    SceneStage,
)
from agy_studio.orchestrator import Studio
from agy_studio.project import Project
from agy_studio.prompt_engineer import build_prompt_bundle
from agy_studio.registry import Registry
from agy_studio.resolve_bridge import ClobberError, FakeTransport, ResolveBridge, assert_lua_additive
from agy_studio.resolve_db import decode_marker_blob, encode_marker_blob
from agy_studio.roundtrip import events_to_decisions, ingest_interchange, poll_resolve
from agy_studio.shot_generator import LintGateError, check_lint_gate, generate_clip, resume_inflight
from agy_studio.state import StateStore
from agy_studio.testing import build_demo_project
from agy_studio.working_set import build_working_set, show_working_set


def seed_two_scene_fixture(root: Path) -> tuple[Project, Registry]:
    reset_default_backend()
    proj = build_demo_project(root)
    return proj, Registry()


# ---------------------------------------------------------------------------
# AC 1: Pre-flight lint gate — all 18 rules L1–L18 + hard block before dispatch
# ---------------------------------------------------------------------------

def test_ac1_all_18_lint_rules_triggered_and_gate_blocks(tmp_path: Path) -> None:
    project, registry = seed_two_scene_fixture(tmp_path / "proj_ac1")
    decompose_scene(project, registry, "SC01")
    base_clip = project.load_clip("SC01_SH01_C01")
    state = StateStore(project)

    # 1. Unlinted clip -> blocked by check_lint_gate and generate_clip
    unlinted = base_clip.model_copy(deep=True)
    unlinted.lint_report = None
    project.save_clip(unlinted)
    with pytest.raises(LintGateError):
        check_lint_gate(project.load_clip("SC01_SH01_C01"))
    with pytest.raises(LintGateError):
        generate_clip(project, registry, state, "SC01_SH01_C01", fake=True)

    ctx = LintContext.build(project, registry, "SC01", "SC01_SH01")
    triggered: set[str] = set()

    # L1: banned words (hyperrealistic, photorealistic, stunning, 4K, 8K, cinematic, ...)
    c = base_clip.model_copy(deep=True)
    c.spec.visual.subject_and_action += " hyperrealistic photorealistic 8K masterpiece"
    _, rep = lint_clip(ctx, c, apply_fixes=False)
    assert "L1" in {f.rule for f in rep.findings}
    triggered.add("L1")

    # L2: > 1 physical verb in subject_and_action
    c = base_clip.model_copy(deep=True)
    c.spec.visual.subject_and_action = "CHAR_RISHI runs and jumps and grabs and throws the phone"
    _, rep = lint_clip(ctx, c, apply_fixes=False)
    assert "L2" in {f.rule for f in rep.findings}
    triggered.add("L2")

    # L3: unregistered uppercase token
    c = base_clip.model_copy(deep=True)
    c.spec.visual.subject_and_action = "CHAR_RISHI faces CHAR_UNKNOWN_VILLAIN on the roof"
    _, rep = lint_clip(ctx, c, apply_fixes=False)
    assert "L3" in {f.rule for f in rep.findings}
    triggered.add("L3")

    # L4: inactive entity state referenced
    e_rishi = project.load_entity("CHAR_RISHI")
    e_rishi.states.append(
        EntityState(
            id="CHAR_RISHI_STATE_ALT",
            delta_description="Alt",
            scenes_active=["SC02"],
            reference_pack=e_rishi.states[0].reference_pack,
        )
    )
    project.save_entity(e_rishi, confirm_decision_id="D000001")
    ctx_l4 = LintContext.build(project, registry, "SC01", "SC01_SH01")
    c = base_clip.model_copy(deep=True)
    c.spec.visual.subject_and_action = "CHAR_RISHI in CHAR_RISHI_STATE_ALT holds PROP_PHONE"
    _, rep = lint_clip(ctx_l4, c, apply_fixes=False)
    assert "L4" in {f.rule for f in rep.findings}
    triggered.add("L4")

    # L5: reference pack not locked / missing roles
    e_unlocked = project.load_entity("CHAR_RISHI")
    e_unlocked.states[0].approval_status = ApprovalStatus.PROPOSED
    project.save_entity(e_unlocked, confirm_decision_id="D000001")
    ctx_l5 = LintContext.build(project, registry, "SC01", "SC01_SH01")
    _, rep = lint_clip(ctx_l5, base_clip.model_copy(deep=True), apply_fixes=False)
    assert "L5" in {f.rule for f in rep.findings}
    triggered.add("L5")
    e_unlocked.states[0].approval_status = ApprovalStatus.LOCKED
    project.save_entity(e_unlocked, confirm_decision_id="D000001")

    # L6: conditioning missing spatial_anchor or entity pack
    c = base_clip.model_copy(deep=True)
    c.conditioning.spatial_anchor = None
    _, rep = lint_clip(ctx, c, apply_fixes=False)
    assert "L6" in {f.rule for f in rep.findings}
    triggered.add("L6")

    # L7: lighting has colour words / palette has light sources
    c = base_clip.model_copy(deep=True)
    c.spec.visual.lighting = "teal and orange warm light"
    c.spec.visual.palette = "sodium streetlamp key light"
    _, rep = lint_clip(ctx, c, apply_fixes=False)
    assert "L7" in {f.rule for f in rep.findings}
    triggered.add("L7")

    # L8: duration outside model's registry limits
    c = base_clip.model_copy(deep=True)
    c.duration_s = 25.0
    _, rep = lint_clip(ctx, c, apply_fixes=False)
    assert "L8" in {f.rule for f in rep.findings}
    triggered.add("L8")

    # L9: first-frame anchor missing when anchor_from_clip_id is set
    c = base_clip.model_copy(deep=True)
    c.anchor_from_clip_id = "SC01_SH01_C99"
    c.conditioning.first_frame_anchor = None
    _, rep = lint_clip(ctx, c, apply_fixes=False)
    assert "L9" in {f.rule for f in rep.findings}
    triggered.add("L9")

    # L10: empty visual field
    c = base_clip.model_copy(deep=True)
    c.spec.visual.mood = ""
    _, rep = lint_clip(ctx, c, apply_fixes=False)
    assert "L10" in {f.rule for f in rep.findings}
    triggered.add("L10")

    # L11: missing required negative prompts
    c = base_clip.model_copy(deep=True)
    c.spec.negative = []
    _, rep = lint_clip(ctx, c, apply_fixes=False)
    assert "L11" in {f.rule for f in rep.findings}
    triggered.add("L11")

    # L12: unsupported model / capability
    c = base_clip.model_copy(deep=True)
    c.model_target = ModelTarget(preferred="nonexistent_video_model", fallback=None)
    _, rep = lint_clip(ctx, c, apply_fixes=False)
    assert "L12" in {f.rule for f in rep.findings}
    triggered.add("L12")

    # L13: camera side crosses axis_of_action
    c = base_clip.model_copy(deep=True)
    c.camera_side = "north_of_axis"
    c.spec.visual.camera = "camera on the north side looking south"
    _, rep = lint_clip(ctx, c, apply_fixes=False)
    assert "L13" in {f.rule for f in rep.findings}
    triggered.add("L13")

    # L14: non-verbatim dialogue text
    c = base_clip.model_copy(deep=True)
    c.spec.audio_spec.dialogue[0].text = "यह संवाद स्क्रिप्ट में नहीं है"
    _, rep = lint_clip(ctx, c, apply_fixes=False)
    assert "L14" in {f.rule for f in rep.findings}
    triggered.add("L14")

    # L15: voice description does not match locked VoiceProfile
    c = base_clip.model_copy(deep=True)
    c.spec.audio_spec.dialogue[0].voice = "wrong voice description"
    _, rep = lint_clip(ctx, c, apply_fixes=False)
    assert "L15" in {f.rule for f in rep.findings}
    triggered.add("L15")

    # L16: speakers per clip > max_speakers_per_clip
    c = base_clip.model_copy(deep=True)
    c.spec.audio_spec.dialogue = [
        c.spec.audio_spec.dialogue[0],
        AudioLine(
            line_id="SC01_L002",
            character_id="CHAR_RISHI",
            text="कूदूँ क्या?",
            transliteration="Koodoon kya?",
            delivery="defiant",
            voice="young Hindi tenor",
        ),
    ]
    _, rep = lint_clip(ctx, c, apply_fixes=False)
    assert "L16" in {f.rule for f in rep.findings}
    triggered.add("L16")

    # L17: music != "none" in audio_spec
    c = base_clip.model_copy(deep=True)
    c.spec.audio_spec.music = "dramatic background score"
    _, rep = lint_clip(ctx, c, apply_fixes=False)
    assert "L17" in {f.rule for f in rep.findings}
    triggered.add("L17")

    # L18: speech word count exceeds clip duration budget
    c = base_clip.model_copy(deep=True)
    c.duration_s = 4.0
    c.spec.audio_spec.dialogue[0].text = (
        "एक दो तीन चार पांच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस बीस"
    )
    _, rep = lint_clip(ctx, c, apply_fixes=False)
    assert "L18" in {f.rule for f in rep.findings}
    triggered.add("L18")

    assert triggered == {f"L{i}" for i in range(1, 19)}

    # 3. Clean clip passes lint with auto-fixes applied, then spec mutation invalidates lint_report
    results = lint_shot(project, registry, "SC01_SH01", apply_fixes=True, save=True)
    assert all(r.passed and not r.blocking for _, r in results)
    check_lint_gate(project.load_clip("SC01_SH01_C01"))

    mutated = project.load_clip("SC01_SH01_C01")
    mutated.spec.visual.mood = "mutated mood after lint"
    project.save_clip(mutated)
    with pytest.raises(LintGateError):
        check_lint_gate(project.load_clip("SC01_SH01_C01"))


# ---------------------------------------------------------------------------
# AC 2 & AC 10: Decomposition determinism & hot-reloadable registry
# ---------------------------------------------------------------------------

def test_ac2_and_ac10_decomposer_determinism_and_hot_reload_registry(tmp_path: Path) -> None:
    project1, registry1 = seed_two_scene_fixture(tmp_path / "p1")
    project2, registry2 = seed_two_scene_fixture(tmp_path / "p2")

    out1 = decompose_scene(project1, registry1, "SC01")
    out2 = decompose_scene(project2, registry2, "SC01")

    clips1 = [c.model_dump(exclude={"generated_by", "created_at"}) for v in out1.values() for c in v]
    clips2 = [c.model_dump(exclude={"generated_by", "created_at"}) for v in out2.values() for c in v]
    assert clips1 == clips2

    # Multi-speaker shot automatically splits into 1-speaker clips with anchor_from_clip_id
    sh02_clips = out1["SC01_SH02"]
    assert len(sh02_clips) >= 2
    for c in sh02_clips:
        speakers = {ln.character_id for ln in c.spec.audio_spec.all_lines()}
        assert len(speakers) <= 1
    assert sh02_clips[1].anchor_from_clip_id == sh02_clips[0].id

    # AC 10: Hot-reload registry YAML (lower max_duration_s from 10 -> 4) without code edits
    reg_path = tmp_path / "custom_registry.yaml"
    raw_reg = read_yaml(registry1.path)
    for m in raw_reg["models"]:
        if m["id"] == "omni":
            m["max_duration_s"] = 4
            m["allowed_durations_s"] = [4]
    atomic_write_yaml(reg_path, raw_reg)
    hot_registry = Registry(reg_path)

    shot = project1.load_shot("SC01_SH01")
    shot.duration_s = 8.0
    shot.actions = ["CHAR_RISHI raises PROP_PHONE", "CHAR_RISHI steps back from the edge"]
    project1.save_shot(shot)
    hot_clips = decompose_shot(project1, hot_registry, "SC01_SH01", model_id="omni")
    assert len(hot_clips) >= 2
    assert all(c.duration_s <= 4.0 for c in hot_clips)


# ---------------------------------------------------------------------------
# AC 3 & AC 4: Audio-visual unity & Voice consistency + re-anchor re-score only
# ---------------------------------------------------------------------------

def test_ac3_and_ac4_audio_visual_unity_and_voice_reanchor(tmp_path: Path) -> None:
    project, registry = seed_two_scene_fixture(tmp_path / "proj_ac34")
    # Verify no TTS or lip-sync in registry
    for m in registry.all():
        assert m.kind not in ("tts", "lipsync", "lip_sync")

    orch = Studio(project, registry=registry, fake=True)
    orch.open_g1()
    orch.respond("G1", None, "approve all")  # G1 approved -> tick advances SC01 to AWAITING_G3
    orch.respond("G3", "SC01", "approve all")  # -> AWAITING_G4
    orch.respond("G4", "SC01", "approve all")  # -> lint -> gen -> QA -> AWAITING_G5

    takes_c01 = project.list_takes("SC01_SH01_C01")
    assert len(takes_c01) >= 2
    for t in takes_c01:
        assert t.video_uri and project.abs(t.video_uri).is_file()
        assert t.dialogue_wav_uri and project.abs(t.dialogue_wav_uri).is_file()
        assert t.qa is not None

    # Set initial voice anchor on CHAR_MOHNISH (first appearance in SC01_SH01_C01)
    vp_before = project.load_voice_profile("CHAR_MOHNISH")
    assert vp_before is not None and vp_before.anchor_take is None

    orch.respond(
        "G5",
        "SC01",
        "SC01_SH01_C01:T01, SC01_SH02_C01:T01, anchor CHAR_MOHNISH=SC01_SH01_C01_T01, anchor CHAR_RISHI=SC01_SH02_C01_T01",
    )
    vp_after = project.load_voice_profile("CHAR_MOHNISH")
    assert vp_after is not None
    assert vp_after.anchor_take == "SC01_SH01_C01_T01"
    assert vp_after.anchor_wav and project.abs(vp_after.anchor_wav).is_file()

    # Record take counts before re-anchor
    take_ids_before = [t.id for t in project.list_takes("SC01_SH01_C01")]
    prompt_msg = orch.set_voice_anchor("CHAR_MOHNISH", "SC01_SH01_C01_T02")
    assert "confirm VOICE_CHAR_MOHNISH" in prompt_msg
    msg = orch.confirm("VOICE_CHAR_MOHNISH")
    assert "re-scored (none regenerated)" in msg
    take_ids_after = [t.id for t in project.list_takes("SC01_SH01_C01")]
    # AC 4: Re-anchoring re-scores existing takes and NEVER regenerates new takes
    assert take_ids_after == take_ids_before
    vp_reanchored = project.load_voice_profile("CHAR_MOHNISH")
    assert vp_reanchored is not None and vp_reanchored.anchor_take == "SC01_SH01_C01_T02"


# ---------------------------------------------------------------------------
# AC 5, AC 6, AC 7: Working-set isolation, Dependency invalidation, Preference learning
# ---------------------------------------------------------------------------

def test_ac5_ac6_ac7_working_set_invalidation_and_preferences(tmp_path: Path) -> None:
    project, registry = seed_two_scene_fixture(tmp_path / "proj_ac567")
    orch = Studio(project, registry=registry, fake=True)

    # AC 5: Working-set isolation (SC01 working set contains only SC01 dialogue/beats, not SC02's dialogue/beats)
    ws = build_working_set(project, "SC01", save=True)
    assert ws.scene.id == "SC01"
    ws_text = show_working_set(project, "SC01")
    assert "SC02_L001" not in ws_text
    assert "SC02_B01" not in ws_text
    with pytest.raises(AgentScopeError):
        run_agent(project, "director", task="direct", working_set=None, registry=registry, fake=True)
    with pytest.raises(AgentScopeError):
        run_agent(
            project,
            "director",
            task="direct",
            working_set=ws,
            global_context={"raw_script": "forbidden"},
            registry=registry,
            fake=True,
        )

    # Advance SC01 through G5 so clips & takes exist
    orch.open_g1()
    orch.respond("G1", None, "approve all")
    orch.respond("G3", "SC01", "approve all")
    orch.respond("G4", "SC01", "approve all")
    orch.respond(
        "G5",
        "SC01",
        "SC01_SH01_C01:T01, SC01_SH02_C01:T01, anchor CHAR_MOHNISH=SC01_SH01_C01_T01, anchor CHAR_RISHI=SC01_SH02_C01_T01",
    )
    if orch.state.scene("SC01").stage == SceneStage.AWAITING_G5:
        orch.respond("G5", "SC01", "approve all, anchor CHAR_GUARD=SC01_SH02_C02_T01")

    # AC 7: Preference learning — steer decision creates scoped Preference and updates clip spec / working_set_hash
    hash_before = build_working_set(project, "SC01", save=True).working_set_hash
    orch.decisions.record(
        tier=DecisionTier.CRAFT,
        target_artifact_id="SC01_SH01_C01",
        action=DecisionAction.STEER,
        payload={"steer": "handheld urgency with tighter framing", "scope": "scene:SC01"},
        rationale="handheld urgency with tighter framing",
    )
    prefs = orch.prefs.list()
    assert any("handheld urgency" in p.text for p in prefs)
    hash_after = build_working_set(project, "SC01", save=True).working_set_hash
    assert hash_after != hash_before
    clip_steered = orch.apply_steer_to_clip(project.load_clip("SC01_SH01_C01"), "handheld urgency with tighter framing")
    compiled = build_prompt_bundle(project, registry, clip_steered)
    assert "handheld urgency" in compiled.prompt

    # AC 6: Dependency invalidation
    graph = AnchorGraph.build(project)
    # 1) Take swap on SC01_SH02_C01 marks only its anchor descendant SC01_SH02_C02 stale
    fo_anchor = graph.fanout("SC01_SH02_C01")
    assert "SC01_SH02_C02" in fo_anchor.clips
    assert "SC01_SH01_C01" not in fo_anchor.clips
    changed = apply_invalidation(project, orch.state, fo_anchor, reason="take swap on SC01_SH02_C01")
    assert "SC01_SH02_C02" in changed
    assert project.load_clip("SC01_SH02_C02").status == ClipStatus.STALE
    assert project.load_clip("SC01_SH01_C01").status != ClipStatus.STALE

    # 2) Pack change requires confirmation and marks only clips referencing that state stale
    fo_pack = graph.fanout("CHAR_RISHI_STATE_DEFAULT")
    assert "SC01_SH01_C01" in fo_pack.clips
    assert fo_pack.requires_confirm
    with pytest.raises(ConfirmRequiredError):
        apply_invalidation(project, orch.state, fo_pack, reason="pack updated", decision_id=None)
    dec = orch.decisions.record(
        tier=DecisionTier.VISION,
        target_artifact_id="CHAR_RISHI_STATE_DEFAULT",
        action=DecisionAction.APPROVE,
        rationale="confirm pack update",
    )
    changed_pack = apply_invalidation(project, orch.state, fo_pack, reason="pack updated", decision_id=dec.id)
    assert "SC01_SH01_C01" in changed_pack


# ---------------------------------------------------------------------------
# AC 8: Non-destructive Resolve handoff & round-trip safety
# ---------------------------------------------------------------------------

def test_ac8_non_destructive_resolve_handoff_and_roundtrip(tmp_path: Path) -> None:
    project, registry = seed_two_scene_fixture(tmp_path / "proj_ac8")
    transport = FakeTransport()
    bridge = ResolveBridge(project, transport=transport)

    # Marker protobuf/zstd round-trip
    markers_in = [{"frame": 24, "color": "Blue", "name": "AGY_REGEN", "note": "tighter lens", "duration": 1}]
    blob = encode_marker_blob(markers_in)
    markers_out, decoded_ok = decode_marker_blob(blob)
    assert decoded_ok
    assert any(m.color == "Blue" and "tighter lens" in m.note for m in markers_out)

    # Lua additive guard rejects destructive API calls
    with pytest.raises(ClobberError):
        assert_lua_additive("timeline:DeleteClips(items)")

    orch = Studio(project, registry=registry, fake=True, transport=transport)
    orch.open_g1()
    orch.respond("G1", None, "approve all")
    orch.respond("G3", "SC01", "approve all")
    orch.respond("G4", "SC01", "approve all")
    orch.respond(
        "G5",
        "SC01",
        "approve all, anchor CHAR_MOHNISH=SC01_SH01_C01_T01, anchor CHAR_RISHI=SC01_SH02_C01_T01",
    )
    if orch.state.scene("SC01").stage == SceneStage.AWAITING_G5:
        orch.respond("G5", "SC01", "approve all, anchor CHAR_GUARD=SC01_SH02_C02_T01")

    # Now handed off to Resolve (G6)
    assert orch.state.scene("SC01").stage == SceneStage.HANDED_OFF
    assert "SC01_v1" in transport.timelines()

    # Overwriting existing timeline SC01_v1 is forbidden
    with pytest.raises(ClobberError):
        bridge.preflight("SC01", "SC01_v1")

    # Simulate human editing SC01_v1 in Resolve
    transport.trim_clip("SC01_v1", "SC01_SH01_C01", delta_frames=-12)
    items_before = transport.timeline("SC01_v1").items

    # Post-handoff regen goes to /REGEN/ bin + optional Orange marker, NEVER modifies timeline items/trims
    regen_take = project.load_take("SC01_SH01_C01_T02")
    bridge.deliver_regen("SC01", [regen_take], add_orange_marker=True)
    snap_after = transport.timeline("SC01_v1")
    assert snap_after is not None
    assert snap_after.items == items_before
    assert any("/REGEN" in b for b in transport.bins())

    # Poll Resolve detects human trim + Red regen marker
    transport.add_marker("SC01_v1", frame=10, color="Red", note="warmer backlight", name="REGEN")
    events = poll_resolve(project, transport)
    kinds = {e.kind for e in events}
    assert "trim" in kinds
    assert "regen_marker" in kinds
    decs = events_to_decisions(events)
    assert len(decs) >= 2

    # XXE security guard on XML interchange import
    bad_xml = tmp_path / "evil.fcpxml"
    bad_xml.write_text('<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><fcpxml>&xxe;</fcpxml>')
    with pytest.raises(ValueError, match="DOCTYPE|ENTITY"):
        ingest_interchange(project, bad_xml, scene_id="SC01")


# ---------------------------------------------------------------------------
# AC 9: Resumability & crash-safe in-flight deduplication
# ---------------------------------------------------------------------------

def test_ac9_crash_recovery_never_resubmits_inflight_take(tmp_path: Path) -> None:
    project, registry = seed_two_scene_fixture(tmp_path / "proj_ac9")
    decompose_scene(project, registry, "SC01")
    lint_shot(project, registry, "SC01_SH01", apply_fixes=True, save=True)
    state = StateStore(project)

    # Simulate process kill right after take 1 is submitted to provider
    backend = default_backend()
    backend.kill_after_submit = True
    with pytest.raises(SimulatedKill):
        generate_clip(project, registry, state, "SC01_SH01_C01", n_takes=2, fake=True)
    backend.kill_after_submit = False

    t1 = project.load_take("SC01_SH01_C01_T01")
    assert t1.dispatch_status in (DispatchStatus.SUBMITTED, DispatchStatus.RUNNING)
    assert t1.operation_name is not None
    op_name = t1.operation_name
    assert backend.submits_for(t1.payload_hash) == 1

    # Resume in-flight takes -> polls existing operation_name, completes T01 without resubmitting
    resumed = resume_inflight(project, registry, state, fake=True)
    assert any(t.id == "SC01_SH01_C01_T01" for t in resumed)
    t1_done = project.load_take("SC01_SH01_C01_T01")
    assert t1_done.dispatch_status == DispatchStatus.COMPLETED
    assert t1_done.operation_name == op_name
    assert backend.submits_for(t1.payload_hash) == 1

    # Calling generate_clip again finishes remaining takes (T02) and never resubmits T01
    all_takes = generate_clip(project, registry, state, "SC01_SH01_C01", n_takes=2, fake=True)
    assert len(all_takes) == 2
    assert all(t.dispatch_status == DispatchStatus.COMPLETED for t in all_takes)
    assert backend.submits_for(t1.payload_hash) == 1


# ---------------------------------------------------------------------------
# AC 11: Full 2-scene end-to-end production loop + G7 Master assembly
# ---------------------------------------------------------------------------

def test_ac11_two_scene_end_to_end_and_master_assembly(tmp_path: Path) -> None:
    project, registry = seed_two_scene_fixture(tmp_path / "proj_ac11")
    transport = FakeTransport()
    orch = Studio(project, registry=registry, fake=True, transport=transport)

    # 1. Open and approve G1 (with an Open Decision choice)
    g1_text = orch.open_g1()
    assert "[GATE G1" in g1_text
    orch.respond("G1", None, "OD3:choose on, approve all")
    assert project.config.settings.open_decisions.confirmed_by_human

    # 2. Drive SC01 through G3 -> G4 -> G5 -> G6 (HANDED_OFF; G2 auto-passes since packs are locked)
    if orch.state.scene("SC01").stage != SceneStage.AWAITING_G3:
        orch.advance("SC01")
    assert orch.state.scene("SC01").stage == SceneStage.AWAITING_G3
    orch.respond("G3", "SC01", "approve all")
    assert orch.state.scene("SC01").stage == SceneStage.AWAITING_G4
    orch.respond("G4", "SC01", "approve all")
    orch.respond(
        "G5",
        "SC01",
        "approve all, anchor CHAR_MOHNISH=SC01_SH01_C01_T01, anchor CHAR_RISHI=SC01_SH02_C01_T01",
    )
    if orch.state.scene("SC01").stage == SceneStage.AWAITING_G5:
        orch.respond("G5", "SC01", "approve all, anchor CHAR_GUARD=SC01_SH02_C02_T01")
    assert orch.state.scene("SC01").stage == SceneStage.HANDED_OFF

    # 3. While SC01 is at G6 (being edited in Resolve), advance SC02 in parallel through G6!
    orch.advance("SC02")
    if orch.state.scene("SC02").stage == SceneStage.AWAITING_G2:
        orch.respond("G2", "SC02", "approve all")
    assert orch.state.scene("SC02").stage == SceneStage.AWAITING_G3
    orch.respond("G3", "SC02", "approve all")
    orch.respond("G4", "SC02", "approve all")
    while orch.state.scene("SC02").stage == SceneStage.AWAITING_G5:
        orch.respond("G5", "SC02", "approve all")
    assert orch.state.scene("SC02").stage == SceneStage.HANDED_OFF

    # 4. Approve both scenes at G6 (via Green APPROVED marker in Resolve for SC01, and chat for SC02)
    transport.add_marker("SC01_v1", frame=0, color="Green", note="approved", name="APPROVED")
    orch.poll()
    assert orch.state.scene("SC01").stage == SceneStage.DONE

    orch.approve_scene("SC02")
    assert orch.state.scene("SC02").stage == SceneStage.DONE

    # 5. Open G7 Master Assembly and approve final film
    g7_text = orch.master()
    assert "[GATE G7" in g7_text
    assert "MASTER_v1" in transport.timelines()
    orch.respond("G7", None, "approve all")
    assert orch.state.load().global_stage == GlobalStage.DELIVERY

    # Verify agent invocations were logged with working_set_hash
    invocations = read_jsonl(project.path("logs", "agent_invocations.jsonl"))
    assert len(invocations) >= 2
    assert any(row.get("working_set_hash") for row in invocations)


# ---------------------------------------------------------------------------
# Phase 0 Legacy Migration smoke test
# ---------------------------------------------------------------------------

def test_phase0_migrate_legacy_non_destructive(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    proj = migrate_lfkod(repo_root, productions_dir=tmp_path, project_id="LFKOD_S01", import_media=False)
    assert proj.project_id == "LFKOD_S01"
    assert len(proj.list_entities()) >= 4
    assert len(proj.list_environments()) >= 2
    assert len(proj.list_scenes()) >= 2
    total_shots = sum(len(proj.list_shots(s.id)) for s in proj.list_scenes())
    total_beats = sum(len(s.beats) for s in proj.list_scenes())
    assert total_shots == 18 or total_beats == 8
