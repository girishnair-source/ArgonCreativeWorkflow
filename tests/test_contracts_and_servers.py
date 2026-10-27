"""Automated Verification Harness for the Spec-Driven DaVinci Resolve 20 NLE MCP Server.

Tests:
1. Strict Pydantic V2 API contracts (`extra="forbid"`, DaVinci Resolve 20 Free Edition limits).
2. Security boundaries (path traversal `../` rejection, SSRF private/loopback URL rejection).
3. Remote `https://` media asset download & content-hashed caching into `./generated_assets/`.
4. All 7 NLE editing domains in `resolve_mcp_server.py` including Free Edition fallbacks.
5. The Section 5 Checklist Verification Scenario (`SpaceSequence` timeline assembly & frame inspection).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

import resolve_mcp_server as srv
from schemas import (
    ASSET_DIR,
    CreateSubclipRequest,
    IngestMediaItem,
    InspectFramesRequest,
    ProjectConfigRequest,
    RenderTimelineRequest,
    WORKSPACE_ROOT,
)


@pytest.fixture(autouse=True)
def _enable_mock_resolve(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Enable deterministic in-memory DaVinci Resolve 20 Free Edition engine for tests."""
    monkeypatch.setenv("RESOLVE_MOCK_MODE", "1")
    srv.reset_mock_resolve_engine()
    yield


@pytest.fixture()
def sample_media_files() -> Iterator[dict[str, Path]]:
    """Create temporary sample media assets inside WORKSPACE_ROOT/generated_assets."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    video_a = ASSET_DIR / "test_astronaut_earth.mp4"
    video_b = ASSET_DIR / "test_reverse_shot.mp4"
    audio_dlg = ASSET_DIR / "test_dialogue_a1.wav"
    audio_music = ASSET_DIR / "test_music_a3.wav"
    lut_file = ASSET_DIR / "test_cinema.cube"

    video_a.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
    video_b.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
    audio_dlg.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    audio_music.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    lut_file.write_text("LUT_3D_SIZE 2\n0.0 0.0 0.0\n1.0 1.0 1.0\n", encoding="utf-8")

    yield {
        "video_a": video_a,
        "video_b": video_b,
        "audio_dlg": audio_dlg,
        "audio_music": audio_music,
        "lut": lut_file,
    }

    for p in (video_a, video_b, audio_dlg, audio_music, lut_file):
        p.unlink(missing_ok=True)


# ============================================================================
# 1. Pydantic V2 Contract & Security Boundary Tests
# ============================================================================


def test_pydantic_strict_extra_forbid() -> None:
    """Verify undeclared fields are strictly rejected by `StrictContract`."""
    with pytest.raises(ValidationError):
        ProjectConfigRequest.model_validate(
            {
                "project_name": "ValidName",
                "width": 1920,
                "height": 1080,
                "unexpected_field": "rejected",
            }
        )


def test_resolve_20_free_edition_limits_enforced() -> None:
    """Verify resolutions above 3840 UHD or frame rates above 60fps are rejected for Resolve 20 Free."""
    with pytest.raises(ValidationError):
        ProjectConfigRequest(project_name="8K_Blocked", width=7680, height=4320, frame_rate="24")

    with pytest.raises(ValidationError):
        ProjectConfigRequest(project_name="HighFPS_Blocked", width=1920, height=1080, frame_rate="120")


def test_security_path_traversal_rejected() -> None:
    """Verify `../` path traversal outside workspace is rejected across contracts."""
    with pytest.raises(ValidationError, match="escapes allowed directory boundary"):
        IngestMediaItem(source="../../etc/passwd")

    with pytest.raises(ValidationError, match="escapes allowed directory boundary"):
        InspectFramesRequest(output_dir="../../../tmp/evil_frames")

    with pytest.raises(ValidationError, match="escapes allowed directory boundary"):
        RenderTimelineRequest(target_dir="../../outside_renders")


def test_security_ssrf_url_validation() -> None:
    """Verify non-HTTPS and private/loopback/metadata IPs are rejected for remote media ingest."""
    with pytest.raises(ValidationError, match="https://"):
        IngestMediaItem(source="http://storage.googleapis.com/clip.mp4")

    with pytest.raises(ValidationError, match="not permitted"):
        IngestMediaItem(source="https://localhost/clip.mp4")

    with pytest.raises(ValidationError, match="not permitted"):
        IngestMediaItem(source="https://169.254.169.254/latest/meta-data/")

    with pytest.raises(ValidationError, match="restricted"):
        IngestMediaItem(source="https://192.168.1.10/internal.mp4")


def test_subclip_frame_bounds_validation() -> None:
    """Verify `mark_out <= mark_in` is rejected."""
    with pytest.raises(ValidationError, match="mark_out must be strictly greater"):
        CreateSubclipRequest(clip_name="shot.mp4", mark_in=50, mark_out=20)


# ============================================================================
# 2. Remote HTTPS URL Auto-Download & Ingest Tests
# ============================================================================


async def test_ingest_remote_https_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify remote `https://` URLs are downloaded to `./generated_assets/` and imported."""

    class _FakeStreamResponse:
        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self, chunk_size: int = 65536) -> AsyncIterator[bytes]:
            yield b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"

        async def __aenter__(self) -> _FakeStreamResponse:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

    class _FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def stream(self, method: str, url: str) -> _FakeStreamResponse:
            return _FakeStreamResponse()

        async def __aenter__(self) -> _FakeAsyncClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    res = await srv.ingest_media(
        items=[
            {
                "source": "https://storage.googleapis.com/renders/astronaut_shot01.mp4",
                "target_bin": "Video/SpaceScene",
                "custom_name": "astronaut_shot01.mp4",
                "generation_id": "gen_op_998877",
            }
        ]
    )
    assert res["status"] == "success"
    assert len(res["imported_clips"]) == 1
    clip_meta = res["imported_clips"][0]
    assert clip_meta["bin_path"] == "Video/SpaceScene"
    assert clip_meta["generation_id"] == "gen_op_998877"
    downloaded_path = Path(clip_meta["local_path"])
    assert downloaded_path.exists()
    downloaded_path.unlink(missing_ok=True)


# ============================================================================
# 3. Section 5 Checklist Verification Scenario (`SpaceSequence`)
# ============================================================================


async def test_section_5_space_sequence_verification(
    sample_media_files: dict[str, Path],
) -> None:
    """Verify the Section 5 prompt: append 5s astronaut clip to 'SpaceSequence' & report frames."""
    rel_video = str(sample_media_files["video_a"].relative_to(WORKSPACE_ROOT))

    append_res = await srv.append_clip_to_timeline(
        file_path=rel_video,
        track_index=1,
        timeline_name="SpaceSequence",
        start_frame=0,
        end_frame=120,  # 5 seconds @ 24 fps = 120 frames
    )
    assert append_res["status"] == "success"
    assert append_res["timeline_name"] == "SpaceSequence"
    assert append_res["total_timeline_frames"] == 120

    structure = srv.get_timeline_structure()
    assert structure["status"] == "success"
    assert structure["timeline_name"] == "SpaceSequence"
    assert structure["duration_frames"] == 120
    video_tracks = [t for t in structure["tracks"] if t["track_type"] == "video"]
    assert len(video_tracks[0]["clips"]) == 1
    assert video_tracks[0]["clips"][0]["duration_frames"] == 120


# ============================================================================
# 4. Complex Multi-Track NLE Editing, Free-Edition Fallbacks & Gap Detection
# ============================================================================


async def test_complex_multitrack_editing_and_free_edition_fallbacks(
    sample_media_files: dict[str, Path],
) -> None:
    """Verify bins, subclips, gap detection, B-roll layering, Free-Edition fallbacks, grading, audio ducking & render."""
    rel_video_a = str(sample_media_files["video_a"].relative_to(WORKSPACE_ROOT))
    rel_video_b = str(sample_media_files["video_b"].relative_to(WORKSPACE_ROOT))
    rel_audio_dlg = str(sample_media_files["audio_dlg"].relative_to(WORKSPACE_ROOT))
    rel_audio_music = str(sample_media_files["audio_music"].relative_to(WORKSPACE_ROOT))
    rel_lut = str(sample_media_files["lut"].relative_to(WORKSPACE_ROOT))

    # 1. Configure project & create timeline
    proj_res = srv.configure_project("MicroDrama_EP01", width=1920, height=1080, frame_rate="24")
    assert proj_res["status"] == "success"

    tl_res = srv.create_or_manage_timeline(action="create", timeline_name="DirectorCut_v1")
    assert tl_res["status"] == "success"

    # 2. Ingest media & create subclip + metadata
    ingest_res = await srv.ingest_media(
        items=[
            {"source": rel_video_a, "target_bin": "Video/Scene1", "generation_id": "gen_101"},
            {"source": rel_video_b, "target_bin": "Video/Scene1", "generation_id": "gen_102"},
        ]
    )
    assert ingest_res["status"] == "success"

    sub_res = srv.create_subclip(
        clip_name=sample_media_files["video_a"].name,
        mark_in=10,
        mark_out=70,
        subclip_name="astronaut_closeup_subclip",
        target_bin="Video/Subclips",
    )
    assert sub_res["status"] == "success"
    assert sub_res["duration_frames"] == 60

    meta_res = srv.set_clip_metadata(
        clip_name="astronaut_closeup_subclip",
        scene="1",
        shot="1A",
        clip_color="Cyan",
        generation_id="gen_101_sub",
    )
    assert meta_res["status"] == "success"

    # 3. Place clips on V1 with a deliberate 24-frame gap between Shot 1 and Shot 2, plus B-Roll on V2
    place_res = srv.place_clips_on_timeline(
        placements=[
            {
                "clip_name": sample_media_files["video_a"].name,
                "track_index": 1,
                "start_frame": 0,
                "end_frame": 120,
                "record_frame": 86400,
            },
            {
                "clip_name": sample_media_files["video_b"].name,
                "track_index": 1,
                "start_frame": 0,
                "end_frame": 120,
                "record_frame": 86544,  # Leaves a 24-frame gap [86520..86544]
            },
            {
                "clip_name": "astronaut_closeup_subclip",
                "track_index": 2,
                "media_type": "video_only",
                "record_frame": 86500,  # B-roll cutaway bridging the cut on V2
            },
        ]
    )
    assert place_res["status"] == "success"
    assert place_res["total_timeline_frames"] == 264

    # 4. Verify automatic gap detection in get_timeline_structure
    struct_res = srv.get_timeline_structure()
    assert struct_res["status"] == "success"
    v1_gaps = [
        g
        for g in struct_res["detected_gaps"]
        if g["track_type"] == "video" and g["track_index"] == 1
    ]
    assert len(v1_gaps) == 1
    assert v1_gaps[0]["gap_start_frame"] == 86520
    assert v1_gaps[0]["duration_frames"] == 24

    # 5. Test Transform + Resolve 20 Free Edition SmartReframe fallback (1.7778x zoom)
    xf_res = srv.transform_timeline_item(
        track_type="video",
        track_index=1,
        item_index=0,
        pan=5.0,
        opacity=95.0,
        stabilize=True,
        smart_reframe=True,
    )
    assert xf_res["status"] == "success"
    assert xf_res["applied_properties"]["ZoomX"] == 1.7778
    assert len(xf_res["fallback_notes"]) == 1

    # 6. Test Titles & Resolve 20 Free Edition Subtitles (.srt fallback)
    title_res = srv.insert_title_or_generator(
        element_type="title",
        template_name="Text+",
        text_content="ORBITAL STATION ALPHA",
        track_index=2,
    )
    assert title_res["status"] == "success"

    sub_auto_res = srv.manage_subtitles(
        mode="auto_transcribe",
        subtitle_lines=[
            {"start_frame": 0, "end_frame": 48, "text": "Houston, visual contact confirmed."},
            {"start_frame": 50, "end_frame": 110, "text": "Telemetry nominal across all sectors."},
        ],
        fps=24.0,
    )
    assert sub_auto_res["status"] == "success"
    assert sub_auto_res["captions_count"] == 2
    assert sub_auto_res["fallback_note"] is not None
    if sub_auto_res["srt_path"]:
        Path(sub_auto_res["srt_path"]).unlink(missing_ok=True)

    # 7. Test Color Grading (LUT + CDL + Shot-to-Shot CopyGrade)
    grade_res = srv.apply_color_grade(
        track_index=1,
        item_index=0,
        node_index=1,
        lut_path=rel_lut,
        cdl_slope="1.05 1.00 0.95",
        cdl_saturation="1.10",
    )
    assert grade_res["status"] == "success"
    assert len(grade_res["applied_operations"]) == 2

    copy_grade_res = srv.apply_color_grade(
        track_index=1,
        item_index=1,
        copy_grade_from_item_index=0,
    )
    assert copy_grade_res["status"] == "success"

    # 8. Test Fairlight Multi-Track Audio Sync, Free-Edition VoiceIsolation fallback & Music Ducking
    dlg_place = await srv.place_synced_audio(
        audio_clip_name=rel_audio_dlg,
        audio_role="dialogue",
        sync_to_video_track_index=1,
        sync_to_video_item_index=0,
        frame_offset=0,
    )
    assert dlg_place["status"] == "success"
    assert dlg_place["audio_track_index"] == 1

    music_place = await srv.place_synced_audio(
        audio_clip_name=rel_audio_music,
        audio_role="music_bed",
        record_frame=86400,
    )
    assert music_place["status"] == "success"
    assert music_place["audio_track_index"] == 3

    dlg_cfg = srv.configure_audio_clip(
        audio_track_index=1,
        item_index=0,
        voice_isolation_enabled=True,
    )
    assert dlg_cfg["status"] == "success"
    assert dlg_cfg["applied_settings"]["Volume"] == 3.0
    assert len(dlg_cfg["fallback_notes"]) == 1

    duck_cfg = srv.configure_audio_clip(
        audio_track_index=3,
        item_index=0,
        duck_against_dialogue_track=1,
        duck_reduction_db=-14.0,
    )
    assert duck_cfg["status"] == "success"
    assert duck_cfg["applied_settings"]["Volume"] == -14.0

    # 9. Test Critic Playhead Export, Markers & Render Queue
    frame_res = srv.export_frame_at_playhead(
        output_dir="./debug_frames",
        timecodes=["01:00:00:00", "01:00:05:00"],
        filename_prefix="test_critic",
    )
    assert frame_res["status"] == "success"
    assert len(frame_res["exported_frames"]) == 2
    for ef in frame_res["exported_frames"]:
        Path(ef["frame_path"]).unlink(missing_ok=True)

    marker_res = srv.manage_markers(
        action="add",
        target="timeline",
        frame_id=86520,
        color="Red",
        name="Continuity Gap",
        note="24-frame gap detected between V1 clips",
        custom_data="generation_id=gen_102",
    )
    assert marker_res["status"] == "success"
    assert "86520" in marker_res["markers"]

    render_res = srv.render_timeline(
        target_dir="./rendered_exports",
        custom_filename="test_master_uhd",
        format_extension="mp4",
        video_codec="H264",
        width=1920,
        height=1080,
    )
    assert render_res["status"] == "success"
    assert render_res["completion_percentage"] == 100
    if render_res["output_file_path"]:
        Path(render_res["output_file_path"]).unlink(missing_ok=True)


# ============================================================================
# 5. Casting Director Continuity & "LIKE. FOLLOW. KILL OR DIE." Assembly Test
# ============================================================================


async def test_casting_director_and_microdrama_episode_assembly() -> None:
    """Verify deterministic Casting Director prompt compilation, ImagePaths binding, and EP01/EP02 assembly."""
    cb = pytest.importorskip("projects.like_share_kill_or_die.casting_bible")
    from assemble_microdrama import build_and_assemble_microdrama

    CAST_BIBLE = cb.CAST_BIBLE
    compile_shot_prompt = cb.compile_shot_prompt

    fake_anchors = {
        "ANCHOR_CAST_EP1": "/tmp/ref_cast_ep1.jpg",
        "ANCHOR_LOC_EP1": "/tmp/ref_loc_ep1.jpg",
    }
    compiled = compile_shot_prompt(
        "EP1_SHOT_05",
        anchor_paths_by_id=fake_anchors,
        previous_shot_image_path="/tmp/ep1_shot04.jpg",
    )
    assert compiled.shot_id == "EP1_SHOT_05"
    assert compiled.aspect_ratio.value == "9:16"
    # Verify verbatim character physical prompt is injected for consistency
    assert CAST_BIBLE["CHAR_RISHI"].immutable_physical_prompt in compiled.full_visual_prompt
    assert CAST_BIBLE["CHAR_MOHNISH"].immutable_physical_prompt in compiled.full_visual_prompt
    assert CAST_BIBLE["CHAR_GUARD"].immutable_physical_prompt in compiled.full_visual_prompt
    # Verify all 3 reference image paths (Cast + Location + Prev Shot) are bound
    assert compiled.reference_image_paths == [
        "/tmp/ref_cast_ep1.jpg",
        "/tmp/ref_loc_ep1.jpg",
        "/tmp/ep1_shot04.jpg",
    ]

    summary = await build_and_assemble_microdrama()
    assert Path(summary["manifest_path"]).exists()
    assert "EP01_Daring" in summary["episodes"]
    assert "EP02_TheParcel" in summary["episodes"]

    # Pro NLE Cut (8f in / 8f out handles trimmed per shot dynamically based on on-disk assets):
    from assemble_microdrama import episode_table

    expected_frames = {r["name"]: r["frames"] for r in episode_table()}
    for ep_name, exp_frames in expected_frames.items():
        ep_info = summary["episodes"][ep_name]
        assert ep_info["timeline_structure"]["status"] == "success"
        assert ep_info["timeline_structure"]["duration_frames"] == exp_frames
        assert ep_info["timeline_structure"]["resolution"] == "1080x1920"
        v1_gaps = [
            g
            for g in ep_info["timeline_structure"]["detected_gaps"]
            if g["track_type"] == "video" and g["track_index"] == 1
        ]
        assert len(v1_gaps) == 0
        assert ep_info["subtitles"]["status"] == "success"
        assert ep_info["render"]["status"] == "success"


# ============================================================================
# 6. Multi-Format Continuity, Token-Saving Casting Gate & Dialogue Agent Tests
# ============================================================================


def test_casting_agent_reference_sheet_gate_and_user_import() -> None:
    """Verify Casting Agent reuses existing sheets and halts to ask the user before generating missing sheets."""
    from casting_agent import (
        ReferenceSheetConfirmationRequiredError,
        enforce_reference_sheet_gate,
        import_existing_reference_sheet,
        preflight_reference_sheet_check,
    )
    from new_project_template import build_starter_project

    starter = build_starter_project()
    # Clean any previously imported test sheets for CHAR_LEO / CHAR_MAYA
    report = preflight_reference_sheet_check(
        character_ids=["CHAR_LEO"],
        location_ids=["LOC_WORKSHOP"],
        custom_characters=starter.characters,
        custom_locations=starter.locations,
    )
    assert not report.ready_to_proceed_without_generation
    assert len(report.missing_sheets) == 2

    # Gate must block unapproved generation so tokens are never wasted silently
    with pytest.raises(ReferenceSheetConfirmationRequiredError, match="blocked automatic token spend"):
        enforce_reference_sheet_gate(
            character_ids=["CHAR_LEO"],
            location_ids=["LOC_WORKSHOP"],
            custom_characters=starter.characters,
            custom_locations=starter.locations,
        )

    # Simulate user providing an existing character sheet image (0 tokens spent)
    sample_sheet = ASSET_DIR / "temp_user_leo_sheet.jpg"
    sample_sheet.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01")
    try:
        reg_entry = import_existing_reference_sheet(
            "CHAR_LEO",
            sample_sheet,
            entity_type="character",
            display_name="Leo",
        )
        assert reg_entry.source_origin == "user_supplied"
        assert Path(reg_entry.file_path).exists()

        # Now CHAR_LEO is reused from disk and only LOC_WORKSHOP needs approval
        resolved = enforce_reference_sheet_gate(
            character_ids=["CHAR_LEO"],
            location_ids=["LOC_WORKSHOP"],
            approved_to_generate_ids={"LOC_WORKSHOP"},
            custom_characters=starter.characters,
            custom_locations=starter.locations,
        )
        assert "CHAR_LEO" in resolved
    finally:
        sample_sheet.unlink(missing_ok=True)
        Path(ASSET_DIR / "anchors" / "char_leo.jpg").unlink(missing_ok=True)


def test_dialogue_agent_pacing_and_single_speaker_isolation() -> None:
    """Verify word-count pacing budget, voice consistency lock, and single-speaker mouth-closed isolation."""
    from dialogue_agent import (
        CadencePreset,
        CharacterVoiceProfile,
        DeliveryMode,
        SceneDialogueLine,
        compile_speaker_isolation_prompt,
        validate_dialogue_pacing,
    )

    protag_vp = CharacterVoiceProfile(
        character_id="CHAR_RISHI",
        speaker_name="Rishi",
        vocal_gender_age_tag="21-year-old urban Indian male",
        pitch_band="warm mid-tenor (135-165 Hz)",
        timbre_and_texture="youthful chest resonance, slightly breathy natural edge, crisp consonant articulation",
        accent_and_dialect="natural Mumbai Hinglish conversational cadence",
        default_cadence=CadencePreset.NATURAL_CONVERSATIONAL,
        target_wpm=135,
        vocal_quirks="quick intake of breath before defensive or high-stakes lines",
    )
    device_vp = CharacterVoiceProfile(
        character_id="CHAR_MASTER_VOICE",
        speaker_name="The Master (Device / Sphere Voice)",
        vocal_gender_age_tag="synthetic pitch-shifted ominous male voice",
        pitch_band="deep sub-baritone (85-110 Hz) with metallic device resonance",
        timbre_and_texture="cold, unhurried, clinical menace with subtle speaker-grille filtering",
        accent_and_dialect="crisp neutral Indian English mixed with cold Hindi commands",
        default_cadence=CadencePreset.SLOW_GRAVITAS,
        target_wpm=112,
        vocal_quirks="surgical precision with zero emotional warmth",
    )

    # 1. Human-paced 11-word line in an 8s shot -> passes comfortably
    good_line = SceneDialogueLine(
        shot_id="EP02_SHOT_03",
        intended_speaker_id="CHAR_RISHI",
        intended_speaker_visual_anchor="Rishi in olive hoodie on screen-right",
        visible_silent_character_anchors={
            "CHAR_FATHER": "Father in white kurta on screen-left",
            "CHAR_RIA": "Ria in yellow kurti in foreground-right",
        },
        spoken_text="Papa, woh raat ko sirf doston ke saath padhai thi.",
        shot_duration_sec=8.0,
    )
    pacing_ok = validate_dialogue_pacing(good_line, protag_vp)
    assert pacing_ok.is_human_paced is True
    assert pacing_ok.pacing_status == "OPTIMAL_HUMAN_PACE"

    iso_prompts = compile_speaker_isolation_prompt(good_line, protag_vp)
    pos = iso_prompts["positive_dialogue_prompt"]
    neg = iso_prompts["negative_dialogue_prompt"]
    # Verify intended speaker + voice profile lock
    assert "ONLY Rishi in olive hoodie on screen-right (CHAR_RISHI) speaks" in pos
    assert "warm mid-tenor (135-165 Hz)" in pos
    # Verify silent listeners are explicitly locked with mouths CLOSED
    assert "Father in white kurta on screen-left (CHAR_FATHER) remains 100% SILENT with mouth CLOSED" in pos
    assert "Ria in yellow kurti in foreground-right (CHAR_RIA) remains 100% SILENT with mouth CLOSED" in pos
    assert "silent listener moving mouth" in neg

    # 2. Device / Off-screen voice -> ALL visible on-camera characters must keep mouths closed
    device_line = SceneDialogueLine(
        shot_id="EP02_SHOT_04",
        intended_speaker_id="CHAR_MASTER_VOICE",
        intended_speaker_visual_anchor="Rishi holding smartphone at ear",
        visible_silent_character_anchors={"CHAR_RIA": "Ria watching from table"},
        spoken_text="First task complete. Your parcel arrives at the door now.",
        delivery_mode=DeliveryMode.DEVICE_SPEAKER,
        shot_duration_sec=8.0,
    )
    device_iso = compile_speaker_isolation_prompt(device_line, device_vp)
    assert "ON-CAMERA MOUTH-CLOSED LOCK" in device_iso["positive_dialogue_prompt"]
    assert "zero on-camera lip-sync" in device_iso["positive_dialogue_prompt"]

    # 3. Over-stuffed 28-word line in a 5s shot -> flagged as over budget
    over_line = SceneDialogueLine(
        shot_id="TEST_OVER_BUDGET",
        intended_speaker_id="CHAR_RISHI",
        intended_speaker_visual_anchor="Rishi on rooftop",
        spoken_text=(
            "Listen to me very carefully right now because we have absolutely no time left "
            "and if the security guard catches us up here tonight we are all finished forever!"
        ),
        shot_duration_sec=5.0,
    )
    pacing_bad = validate_dialogue_pacing(over_line, protag_vp)
    assert pacing_bad.is_human_paced is False
    assert pacing_bad.pacing_status == "OVER_BUDGET_RISK_OF_CLIPPING"


def test_scalable_multi_format_cartoon_and_20min_episode_compiler() -> None:
    """Verify cartoon/20-min episode compilation, 14-ref Nano Banana Pro + 10-ref Omni 1.1, and scene-boundary reset."""
    from new_project_template import build_starter_project
    from production_bible import VisualMedium, compile_episode_shots

    project = build_starter_project()
    assert project.style_lock.medium == VisualMedium.CARTOON_3D_STYLIZED
    assert project.style_lock.primary_image_model == "gemini-3-pro-image"
    assert project.style_lock.max_image_references == 14
    assert project.style_lock.primary_video_model == "gemini-omni-1.1-flash"
    assert project.style_lock.max_video_references == 10

    rich_ref_map = {
        "CHAR_LEO": "/tmp/leo_front.jpg",
        "CHAR_LEO_SIDE": "/tmp/leo_side.jpg",
        "CHAR_LEO_BACK": "/tmp/leo_back.jpg",
        "CHAR_LEO_EXPR": "/tmp/leo_expressions.jpg",
        "CHAR_LEO_WARDROBE": "/tmp/leo_apron.jpg",
        "CHAR_MAYA": "/tmp/maya_front.jpg",
        "CHAR_MAYA_SIDE": "/tmp/maya_side.jpg",
        "CHAR_MAYA_BACK": "/tmp/maya_back.jpg",
        "CHAR_MAYA_EXPR": "/tmp/maya_expressions.jpg",
        "CHAR_MAYA_WARDROBE": "/tmp/maya_vest.jpg",
        "LOC_WORKSHOP": "/tmp/workshop_wide.jpg",
        "LOC_WORKSHOP_REVERSE": "/tmp/workshop_reverse.jpg",
        "LOC_WORKSHOP_BENCH": "/tmp/workshop_bench.jpg",
        "LOC_WORKSHOP_LIGHTING": "/tmp/workshop_lighting.jpg",
        "LOC_ROOFTOP_DOCK": "/tmp/dock_plate.jpg",
        "STYLE_BIBLE": "/tmp/style_bible_3d.jpg",
    }

    compiled = compile_episode_shots(
        project,
        episode_id="EP01",
        reference_paths_by_id=rich_ref_map,
        shot_keyframes_by_id={
            "EP01_SC01_SH001": "/tmp/sc01_sh001_kf.jpg",
        },
    )
    assert len(compiled) == 2
    sh1, sh2 = compiled[0], compiled[1]
    assert sh1.image_model == "gemini-3-pro-image"
    assert sh1.video_model == "gemini-omni-1.1-flash"
    # Verify Nano Banana Pro receives up to 14 reference images and Omni 1.1 receives up to 10
    assert len(sh1.reference_image_paths) == 14
    assert len(sh1.omni_video_reference_paths) == 10
    assert len(sh1.fallback_3_reference_paths) == 3
    assert "CARTOON_3D_STYLIZED" in sh1.positive_prompt
    assert "live-action real photography" in sh1.negative_prompt
    # Verify Scene 2 (new location LOC_ROOFTOP_DOCK) did NOT inherit Scene 1's keyframe
    assert "/tmp/sc01_sh001_kf.jpg" not in sh2.reference_image_paths
    assert sh2.resolve_bin_path == "Video/EP01/SC02_ROOFTOP"



