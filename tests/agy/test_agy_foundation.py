"""Foundation tests: registry, project folder, locking/versioning, state machine, integrity hashes, helpers."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import yaml

from agy_studio.agent_log import list_agent_calls, load_agent_call, log_agent_call
from agy_studio.continuity import active_states_at, all_scene_states, count_words, entity_tokens
from agy_studio.fsio import (
    ImmutableArtifactError,
    PathSecurityError,
    clip_lint_hash,
    compute_payload_hash,
    read_jsonl,
    safe_join,
    validate_id,
    write_new,
)
from agy_studio.models import GenClip, PromptBundle, SceneStage, ShotSpec, VisualSpec
from agy_studio.procs import DisallowedCommandError, run
from agy_studio.project import LockedAssetError, parse_artifact_id
from agy_studio.registry import Registry
from agy_studio.state import ConfirmRequiredError, DuplicateDispatchError, IllegalTransitionError, StateStore


# ---------------------------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------------------------
def test_registry_loads_video_models_and_constraints(registry: Registry) -> None:
    assert {m.id for m in registry.video_models()} == {"omni", "veo"}
    assert registry.max_duration_s("omni") == 10
    assert registry.max_duration_s("veo") == 8
    assert registry.supports("omni", "refs_with_first_frame") is True
    assert registry.supports("veo", "refs_with_first_frame") is False
    assert registry.get("veo").snap_duration(5.1) == 6.0
    for kind in ("music", "sfx", "image", "transcribe", "speaker_similarity"):
        assert registry.first_of_kind(kind)


def test_registry_hot_reload_changes_constraints(registry: Registry) -> None:
    data = yaml.safe_load(registry.path.read_text())
    for m in data["models"]:
        if m["id"] == "veo":
            m["max_duration_s"] = 6
            m["supports"]["max_speakers_per_clip"] = 1
    registry.path.write_text(yaml.safe_dump(data, sort_keys=False))
    future = time.time() + 5
    os.utime(registry.path, (future, future))
    assert registry.max_duration_s("veo") == 6
    assert registry.max_speakers_per_clip("veo") == 1


def test_registry_rejects_adapter_outside_namespace(tmp_path: Path) -> None:
    bad = tmp_path / "registry.yaml"
    bad.write_text(yaml.safe_dump({"models": [{"id": "x", "kind": "llm", "provider_model": "p", "adapter": "os.path"}]}))
    with pytest.raises(ValueError, match="agy_studio.adapters"):
        Registry(bad)


# ---------------------------------------------------------------------------------------------
# Project folder, locking, append-only library
# ---------------------------------------------------------------------------------------------
def test_demo_project_is_complete(demo_project) -> None:
    p = demo_project
    assert [e.id for e in p.list_entities()] == ["CHAR_GUARD", "CHAR_MOHNISH", "CHAR_RISHI", "PROP_PHONE"]
    assert [s.id for s in p.list_scenes()] == ["SC01", "SC02"]
    assert [s.id for s in p.list_shots("SC01")] == ["SC01_SH01", "SC01_SH02"]
    assert p.load_style_lock().approval_status == "locked"
    vp = p.load_voice_profile("CHAR_RISHI")
    assert vp is not None and vp.approval_status == "locked"
    assert p.casting()["voice_profiles"]["CHAR_RISHI"]["library_version"] == 1
    rishi = p.load_entity("CHAR_RISHI")
    pack = rishi.state("CHAR_RISHI_STATE_DEFAULT").reference_pack
    assert pack.face_closeup and p.abs(pack.face_closeup).is_file()
    assert pack.face_closeup.startswith("library/packs/CHAR_RISHI_STATE_DEFAULT/v1/")
    env = p.load_environment("ENV_TERRACE")
    assert env.angle_uri("reverse") and p.abs(env.angle_uri("reverse")).is_file()


def test_locked_entity_edit_requires_confirm_and_versions(demo_project) -> None:
    p = demo_project
    rishi = p.load_entity("CHAR_RISHI")
    rishi.canonical_description += ", small scar on the chin"
    with pytest.raises(LockedAssetError):
        p.save_entity(rishi)
    p.save_entity(rishi, confirm_decision_id="D000001")
    again = p.load_entity("CHAR_RISHI")
    assert again.version == 2 and again.derived_from == "CHAR_RISHI@v1"
    assert list((p.path("bible", "entities", "_history")).glob("CHAR_RISHI_v*.yaml"))


def test_voice_anchor_set_once_then_reanchor_needs_confirm(demo_project) -> None:
    p = demo_project
    vp = p.load_voice_profile("CHAR_RISHI")
    vp.anchor_take = "SC01_SH01_C01_T01"
    vp.anchor_wav = "library/voices/CHAR_RISHI/anchor.wav"
    p.save_voice_profile(vp)  # first anchor: allowed without confirm
    assert p.load_voice_profile("CHAR_RISHI").anchor_take == "SC01_SH01_C01_T01"
    vp2 = p.load_voice_profile("CHAR_RISHI")
    vp2.anchor_take = "SC01_SH02_C01_T02"
    with pytest.raises(LockedAssetError):
        p.save_voice_profile(vp2)
    p.save_voice_profile(vp2, confirm_decision_id="D000002")
    final = p.load_voice_profile("CHAR_RISHI")
    assert final.anchor_take == "SC01_SH02_C01_T02" and final.version == 3
    assert p.voice_profile_path("CHAR_RISHI", 1).is_file()  # old versions are never removed


def test_library_promotion_is_append_only(demo_project, tmp_path: Path) -> None:
    from agy_studio.fsio import generated_by
    from agy_studio.media import write_png

    p = demo_project
    v1 = p.path("library", "packs", "CHAR_MOHNISH_STATE_DEFAULT", "v1", "front.png")
    before = v1.read_bytes()
    src = write_png(tmp_path / "front.png", 32, 32, (1, 2, 3))
    pack = p.promote_pack("CHAR_MOHNISH_STATE_DEFAULT", {"front": src}, approved_by="director",
                          provenance=generated_by("test"))
    assert pack.turnaround.front.startswith("library/packs/CHAR_MOHNISH_STATE_DEFAULT/v2/")
    assert v1.read_bytes() == before


def test_path_safety(tmp_path: Path) -> None:
    with pytest.raises(PathSecurityError):
        safe_join(tmp_path, "..", "escape.txt")
    with pytest.raises(PathSecurityError):
        validate_id("../SC01")
    with pytest.raises(ValueError):
        parse_artifact_id("SC01_SH01/../../etc")
    assert parse_artifact_id("SC12_SH04_C02_T01") == {
        "scene": "SC12", "shot": "SC12_SH04", "clip": "SC12_SH04_C02", "take": "SC12_SH04_C02_T01"}
    f = tmp_path / "a.txt"
    write_new(f, "x")
    with pytest.raises(ImmutableArtifactError):
        write_new(f, "y")


def test_procs_only_runs_allow_listed_binaries() -> None:
    with pytest.raises(DisallowedCommandError):
        run(["rm", "-rf", "/tmp/nothing"])
    with pytest.raises(DisallowedCommandError):
        run(["bash", "-c", "echo hi"])


# ---------------------------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------------------------
def test_state_transitions_persist_and_resume(demo_project) -> None:
    store = StateStore(demo_project)
    store.ensure_scene("SC01")
    store.transition("SC01", SceneStage.WORKING_SET, reason="start")
    with pytest.raises(IllegalTransitionError):
        store.transition("SC01", SceneStage.GENERATING)
    with pytest.raises(IllegalTransitionError):
        store.transition("SC01", SceneStage.LINT, force=True)
    store.transition("SC01", SceneStage.DESIGN)
    store.transition("SC01", SceneStage.AWAITING_G2)
    restarted = StateStore(demo_project)  # simulates AGY being killed and restarted
    sc = restarted.scene("SC01")
    assert sc.stage == SceneStage.AWAITING_G2 and sc.awaiting == "awaiting:G2"
    assert len(read_jsonl(demo_project.path("logs", "state_transitions.jsonl"))) == 3


def test_dispatch_dedupe_survives_restart(demo_project) -> None:
    StateStore(demo_project).record_dispatch("abc123", "SC01_SH01_C01_T01")
    store = StateStore(demo_project)
    assert store.dispatched_take("abc123") == "SC01_SH01_C01_T01"
    store.record_dispatch("abc123", "SC01_SH01_C01_T01")  # idempotent
    with pytest.raises(DuplicateDispatchError):
        store.record_dispatch("abc123", "SC01_SH01_C01_T02")


def test_confirm_handshake_requires_fanout_first(demo_project) -> None:
    store = StateStore(demo_project)
    with pytest.raises(ConfirmRequiredError):
        store.consume_confirm("CHAR_RISHI_STATE_DEFAULT")
    store.request_confirm("CHAR_RISHI_STATE_DEFAULT", action="edit_pack", fanout={"clips": ["SC01_SH01_C01"]})
    assert store.consume_confirm("CHAR_RISHI_STATE_DEFAULT")["fanout"]["clips"] == ["SC01_SH01_C01"]
    with pytest.raises(ConfirmRequiredError):
        store.consume_confirm("CHAR_RISHI_STATE_DEFAULT")


def test_nested_mutate_is_one_atomic_write(demo_project) -> None:
    store = StateStore(demo_project)
    store.ensure_scene("SC01")
    seq = store.load().write_seq
    with store.mutate() as outer:
        with store.mutate() as inner:
            assert inner is outer
            inner.paused = True
    st = store.load()
    assert st.write_seq == seq + 1 and st.paused is True


# ---------------------------------------------------------------------------------------------
# Continuity helpers
# ---------------------------------------------------------------------------------------------
def test_active_states_follow_mid_scene_transition(demo_project) -> None:
    scene = demo_project.load_scene("SC01")
    order = scene.creative_shots
    assert "CHAR_RISHI_STATE_DEFAULT" in active_states_at(scene, order, "SC01_SH01", 1)
    assert "CHAR_RISHI_STATE_DEFAULT" in active_states_at(scene, order, "SC01_SH02", 1)
    at_c2 = active_states_at(scene, order, "SC01_SH02", 2)
    assert "CHAR_RISHI_STATE_INJURED" in at_c2 and "CHAR_RISHI_STATE_DEFAULT" not in at_c2
    assert "CHAR_RISHI_STATE_INJURED" in all_scene_states(scene)


def test_devanagari_word_count_and_entity_tokens() -> None:
    assert count_words("भाई, लाइव चालू है... दो हज़ार लोग देख रहे हैं!") == 10
    assert count_words("देख, अब असली खेल शुरू होगा।") == 6
    assert entity_tokens("CHAR_RISHI climbs; CHAR_RISHI waves at PROP_PHONE") == ["CHAR_RISHI", "PROP_PHONE"]


# ---------------------------------------------------------------------------------------------
# Integrity hashes and agent log
# ---------------------------------------------------------------------------------------------
def test_lint_hash_tracks_spec_but_not_bookkeeping() -> None:
    clip = GenClip(id="SC01_SH01_C01", shot_id="SC01_SH01", scene_id="SC01", sequence_index=1, duration_s=6,
                   spec=ShotSpec(visual=VisualSpec(subject_and_action="CHAR_RISHI climbs onto the parapet ledge")))
    h = clip_lint_hash(clip)
    clip.flags.append("merged_short_clip")
    assert clip_lint_hash(clip) == h
    clip.spec.visual.mood = "dread"
    assert clip_lint_hash(clip) != h


def test_payload_hash_is_content_addressed(demo_project) -> None:
    p = demo_project
    ref = p.path("scenes", "SC01", "design", "probe.bin")
    ref.write_bytes(b"one")
    bundle = PromptBundle(model_id="omni", prompt="x", reference_images=[p.rel(ref)], duration_s=6)
    h1 = compute_payload_hash(p.root, model_id="omni", provider_model="m", bundle=bundle, seed=1)
    assert h1 == compute_payload_hash(p.root, model_id="omni", provider_model="m", bundle=bundle, seed=1)
    ref.write_bytes(b"two")
    assert compute_payload_hash(p.root, model_id="omni", provider_model="m", bundle=bundle, seed=1) != h1


def test_agent_calls_are_immutable_and_reproducible(demo_project) -> None:
    a = log_agent_call(demo_project, agent="Director", scene_id="SC01", working_set_hash="ws1", task="t",
                       inputs={"x": 1}, outputs={"proposals": []}, mode="manual")
    b = log_agent_call(demo_project, agent="Director", scene_id="SC01", working_set_hash="ws1", task="t",
                       inputs={"x": 2}, outputs={"proposals": []}, mode="manual")
    assert a != b and load_agent_call(a)["working_set_hash"] == "ws1"
    assert len(list_agent_calls(demo_project, scene_id="SC01", agent="Director")) == 2


# ---------------------------------------------------------------------------------------------
# Media helpers (ffmpeg)
# ---------------------------------------------------------------------------------------------
def test_media_helpers_roundtrip(ffmpeg_available, tmp_path: Path) -> None:
    from agy_studio import media

    clip = media.make_test_clip(tmp_path / "t.mp4", duration_s=1.0)
    assert 0.8 <= media.duration_s(clip) <= 1.3 and media.has_audio(clip)
    first = media.extract_frame(clip, tmp_path / "first.png", at="first")
    last = media.extract_frame(clip, tmp_path / "last.png", at="last")
    assert first.is_file() and last.is_file()
    wav = media.extract_wav(clip, tmp_path / "d.wav")
    data, sr = media.read_wav_mono(wav)
    assert sr == 16000 and data.size > 8000
    png = media.write_png(tmp_path / "c.png", 16, 16, (250, 10, 10))
    rgb = media.read_image_rgb(png, (8, 8))
    assert rgb.shape == (8, 8, 3) and rgb[..., 0].mean() > 200
    with pytest.raises(FileExistsError):
        media.extract_frame(clip, first, at="first")
