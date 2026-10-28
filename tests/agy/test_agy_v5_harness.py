"""Tests for AGY Studio v5 Harness Upgrades:
1. Gemini TTS 3.8 (`gemini_tts_3_8` / `gemini-3.8-tts`) native acoustic delivery (no ElevenLabs bracket stripping).
2. Two-Tier `360p` Draft -> `4K` Finalize video generation on `gemini-omni-1.5-flash` (`omni_1_5_flash`) reusing the
   exact same prompt, seed, and reference images.
3. Automatic `3x3` video frame contact sheet PNG builder & 6 Visual Continuity Critic failure modes + remediations.
4. Virtual Cinema Optical Glass (`6 Camera Bodies x 11 Lenses x 3 Apertures`) & `33 Physical Camera Rigs`.
5. Two-Tier `SceneSlate` (`kind: "slate"`, up to 5 background audio tracks with `none|light|medium|heavy` ducking
   and `-14 LUFS`) -> `DaVinci Resolve` Timeline promotion.
6. Git-style Timeline Version Control (`branch`, `commit`, `undo`, `redo`, `pull_request`, `merge`, `reset`, `revert`),
   Optimistic Concurrency (`expected_state_token`), Keyframe Pinning (`0.0..20.0s`), and Windowed Retakes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agy_studio.generic_bible import (
    CameraRig33,
    CineAperture,
    CineCameraBody,
    CineLens,
    CineOpticalSpec,
    compile_generic_bible_prompt,
)
from agy_studio.harness_v5 import (
    ConcurrencyConflictError,
    DraftFinalizeController,
    DuckingAmount,
    GeminiTTS38Line,
    KeyframePin,
    KeyframePinSpec,
    SceneSlate,
    SlateAudioTrack,
    SlateClipItem,
    SlateController,
    TimelineGitRepo,
    VisualContinuityCritic,
    REMEDIATION_RECIPES,
    VisualContinuityFailureMode,
    WindowedRetakeSpec,
    build_3x3_contact_sheet,
)
from agy_studio.registry import default_registry


def test_registry_has_omni_1_5_flash_360p_to_4k_and_gemini_tts_3_8() -> None:
    reg = default_registry()
    omni_15 = reg.get("omni_1_5_flash")
    assert omni_15.provider_model == "gemini-omni-1.5-flash"
    assert omni_15.draft_resolution == "360p"
    assert omni_15.finalize_resolution == "4k"
    assert omni_15.reuse_prompt_on_finalize is True
    assert omni_15.supports.two_tier_draft_finalize is True
    assert omni_15.supports.reuse_prompt_on_finalize is True
    assert "360p" in omni_15.resolutions and "4k" in omni_15.resolutions

    tts = reg.get("gemini_tts_3_8")
    assert tts.native_voice_model == "gemini-3.8-tts"
    assert tts.kind.value == "video_av"


def test_gemini_tts_3_8_preserves_native_acting_tags() -> None:
    line = GeminiTTS38Line(
        character_id="CHAR_RISHI",
        voice_id="VOICE_RISHI_BARITONE",
        language="hi-IN",
        native_script_text="[tense whisper] यह पार्सल किसने भेजा?",
        phonetic_guide="Yeh parcel kisne bheja?",
        acoustic_delivery="hushed urgent baritone",
        off_screen=False,
    )
    formatted = line.format_tts_prompt()
    assert "gemini-3.8-tts" in formatted
    assert "[tense whisper] यह पार्सल किसने भेजा?" in formatted
    assert "ON-CAMERA SINGLE SPEAKER" in formatted


def test_virtual_cinema_optical_glass_and_33_rigs() -> None:
    assert len(CineCameraBody) == 6
    assert len(CineLens) == 11
    assert len(CineAperture) == 3
    assert len(CameraRig33) == 33

    optics = CineOpticalSpec(
        camera_body=CineCameraBody.ARRI_ALEXA_35,
        lens=CineLens.COOKE_S4_PRIME,
        focal_length_mm=35,
        aperture=CineAperture.SHALLOW_F1_4,
        rig=CameraRig33.STEADICAM_ORBIT_180,
    )
    prompt = compile_generic_bible_prompt(
        subject_and_character="CHAR_RISHI seated at dining table",
        scene_and_theme="INT. RISHI'S HOUSE - MORNING",
        action_and_physics="Rishi looks down at his smartphone",
        camera_and_framing="OTS from behind Ria's left shoulder",
        lighting_and_color="Warm morning daylight through sheer curtains",
        optics=optics,
        single_speaker_id="CHAR_FATHER",
    )
    assert "ARRI Alexa 35" in prompt
    assert "35mm Cooke S4/i Prime" in prompt
    assert "f/1.4" in prompt
    assert "180-degree Steadicam arc orbit" in prompt


def test_two_tier_360p_draft_to_4k_finalize_reuses_exact_same_prompt(tmp_path: Path, ffmpeg_available: bool) -> None:
    """The controller records REAL videos (it never generates stubs) and the critic measures them."""
    from agy_studio import media

    ctl = DraftFinalizeController(tmp_path / "SC01")
    exact_prompt = "Wide master: Rishi at the dining table, warm tungsten, 35mm, no text overlays."
    refs = ["library/packs/CHAR_RISHI_STATE_DEFAULT/v1/front.png"]
    draft_mp4 = media.make_test_clip(tmp_path / "SC01_C001_draft_360p.mp4", duration_s=3.0, size=(640, 360))
    draft = ctl.register_draft_360p("SC01_C001", exact_prompt, draft_mp4, seed=777, reference_image_paths=refs)
    assert draft.stage == "draft_360p" and draft.critic_report is not None
    assert draft.critic_report.status == "accepted" and draft.critic_report.motion["static"] is False
    assert draft.finalize_resolution == "4k"
    assert Path(draft.draft_contact_sheet_path).exists()

    final_mp4 = media.make_test_clip(tmp_path / "SC01_C001_final_4k.mp4", duration_s=3.0, size=(640, 360))
    finalized = ctl.finalize_4k(draft, final_mp4)
    assert finalized.stage == "finalized_4k"
    assert finalized.finalized_prompt == exact_prompt
    assert finalized.prompt_sha256 == draft.prompt_sha256
    assert Path(finalized.final_4k_video_path).exists()
    assert Path(finalized.final_4k_contact_sheet_path).exists()

    # a still padded into an mp4 is rejected as the draft — it never reaches 4K
    still_mp4 = media.make_test_clip(tmp_path / "SC01_C002_draft_360p.mp4", duration_s=3.0, color="gray", static=True)
    rejected = ctl.register_draft_360p("SC01_C002", exact_prompt, still_mp4, seed=1)
    assert rejected.stage == "critic_rejected"
    assert rejected.critic_report is not None and rejected.critic_report.failure_mode is not None
    assert rejected.critic_report.failure_mode.value == "static_frames"
    assert "never substitute stills" in (rejected.critic_report.remediation or "")
    with pytest.raises(ValueError, match="has not been accepted"):
        ctl.finalize_4k(rejected, final_mp4)

    # a non-video file is an error, not a grey placeholder sheet
    (tmp_path / "not_a_video.mp4").write_text("{}")
    with pytest.raises(Exception):
        ctl.register_draft_360p("SC01_C003", exact_prompt, tmp_path / "not_a_video.mp4")


def test_contact_sheet_samples_across_the_whole_clip_not_the_first_second(tmp_path: Path, ffmpeg_available: bool) -> None:
    """Regression: the old filter took frames 0,3,..,24 (= first second). Frames must span the clip."""
    import subprocess

    from agy_studio import media

    # 9 s clip whose colour changes at t=6 s: the old sampler never saw the change.
    clip = tmp_path / "two_tone.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=red:size=160x90:rate=24:duration=6",
                    "-f", "lavfi", "-i", "color=c=blue:size=160x90:rate=24:duration=3",
                    "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-pix_fmt", "yuv420p",
                    str(clip)], check=True)
    frames = media.sample_frames_rgb(clip, n=9, size=(16, 9))
    reds = [bool(f[..., 0].mean() > f[..., 2].mean()) for f in frames]
    assert reds[0] is True and reds[-1] is False and 5 <= sum(reds) <= 7, reds
    sheet = build_3x3_contact_sheet(clip, tmp_path / "sheet.png")
    assert sheet.is_file() and sheet.stat().st_size > 0
    assert media.analyze_motion(clip)["static"] is False


def test_visual_continuity_critic_rejects_and_provides_remediation(tmp_path: Path, ffmpeg_available: bool) -> None:
    from agy_studio import media

    clip = media.make_test_clip(tmp_path / "take.mp4", duration_s=2.0, size=(320, 180))
    contact = build_3x3_contact_sheet(clip)
    critic = VisualContinuityCritic()
    for mode in VisualContinuityFailureMode:
        if mode is VisualContinuityFailureMode.STATIC_FRAMES:
            continue  # measured, cannot be injected
        report = critic.evaluate("TAKE_01", contact, video_path=clip, checks={mode.value: False})
        assert report.status == "rejected"
        assert report.failure_mode == mode
        assert report.remediation == REMEDIATION_RECIPES[mode]
    # a human/LLM "pass" can never override the measured static check
    still = media.make_test_clip(tmp_path / "still.mp4", duration_s=2.0, color="gray", static=True)
    sheet = build_3x3_contact_sheet(still)
    report = critic.evaluate("TAKE_02", sheet, video_path=still, checks={"static_frames": True})
    assert report.status == "rejected" and report.failure_mode == VisualContinuityFailureMode.STATIC_FRAMES
    # sheet-only evaluation (no video at hand) still measures the 9 tiles
    report = critic.evaluate("TAKE_03", sheet)
    assert report.status == "rejected" and report.motion.get("source") == "sheet"


def test_scene_slate_ducking_max_5_tracks_state_token_and_promotion(tmp_path: Path) -> None:
    ctl = SlateController(tmp_path / "SC01")
    slate = SceneSlate(
        scene_id="SC01",
        clips=[
            SlateClipItem(clip_id="SC01_C001", video_path="takes/c1_4k.mp4", trim_in_s=0.0, trim_out_s=4.5, resolution="4k"),
            SlateClipItem(clip_id="SC01_C002", video_path="takes/c2_4k.mp4", trim_in_s=0.5, trim_out_s=6.0, resolution="4k"),
        ],
        audio_tracks=[
            SlateAudioTrack(
                track_id="BGM_1",
                role="music",
                audio_path="audio/bgm.wav",
                gain_db=-6.0,
                ducking=DuckingAmount.HEAVY,
            ),
            SlateAudioTrack(
                track_id="VO_1",
                role="voiceover",
                audio_path="audio/vo.wav",
                gain_db=0.0,
                ducking=DuckingAmount.NONE,
            ),
        ],
    )
    assert slate.total_duration_s() == 10.0
    assert slate.audio_tracks[0].effective_ducked_gain_db == -21.0

    saved = ctl.save_slate(slate)
    assert saved.state_token.startswith("st_")

    # Optimistic concurrency conflict check
    with pytest.raises(ConcurrencyConflictError):
        ctl.save_slate(slate, expected_state_token="st_wrong_token")

    promoted = ctl.promote_to_timeline(require_4k_clips=True)
    assert promoted.kind == "timeline"
    assert promoted.promoted_to_timeline is True


def test_timeline_git_repo_branch_commit_undo_redo_pr_merge_revert(tmp_path: Path) -> None:
    repo = TimelineGitRepo(tmp_path)
    c1 = repo.commit({"clips": ["C001"]}, "Initial rough cut")
    c2 = repo.commit({"clips": ["C001", "C002"]}, "Add OTS reaction shot")

    undone = repo.undo()
    assert undone.commit_id == c1.commit_id
    redone = repo.redo()
    assert redone.commit_id == c2.commit_id

    repo.create_branch("director-experiment")
    c3 = repo.commit({"clips": ["C001", "C002", "C003_4K"]}, "Add 4K climax close-up")
    assert c3.branch == "director-experiment"

    pr = repo.open_pull_request("director-experiment", "main", "Promote 4K climax cut")
    merged = repo.merge_pull_request(pr.pr_id)
    assert merged.branch == "main"
    assert merged.snapshot["clips"] == ["C001", "C002", "C003_4K"]

    reverted = repo.revert(merged.commit_id)
    assert reverted.snapshot["clips"] == ["C001", "C002"]


def test_keyframe_pinning_and_windowed_retakes() -> None:
    pins = KeyframePinSpec(
        clip_id="SC01_C001",
        duration_s=8.0,
        pins=[
            KeyframePin(image_path="kf_start.png", timestamp_s=0.0),
            KeyframePin(image_path="kf_mid.png", timestamp_s=4.25),
            KeyframePin(image_path="kf_end.png", timestamp_s=8.0),
        ],
    )
    assert len(pins.pins) == 3

    with pytest.raises(ValueError):
        KeyframePinSpec(
            clip_id="SC01_C001",
            duration_s=5.0,
            pins=[KeyframePin(image_path="kf_late.png", timestamp_s=6.5)],
        )

    retake = WindowedRetakeSpec(
        clip_id="SC01_C001",
        mode="audio_video",
        start_s=2.4,
        end_s=5.8,
        retake_prompt="Ria glances up from her phone toward Father with a subtle smirk.",
    )
    assert retake.preserve_outside_window is True
