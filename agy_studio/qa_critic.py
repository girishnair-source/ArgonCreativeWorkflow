"""QA Critic — BUILD_SPEC_v3 §6 Production (QA Critic) & §7.2 Post-generation QA.

Scores every completed take across visual and audio axes:
* ``identity_similarity`` (per referenced character/prop against locked pack images),
* ``spatial_continuity`` (first frame vs ``first_frame_anchor`` or previous clip's last frame),
* ``prompt_adherence`` & ``artifact_flags`` (burned-in subtitles, morphing, extra limbs),
* ``transcript_match`` (ASR vs verbatim expected dialogue in Devanagari/Hindi),
* ``speaker_similarity`` (per speaker against ``VoiceProfile.anchor_wav``; ``None`` until the voice anchor is set),
* ``music_present`` & ``unwanted_audio_flags`` (spontaneous music, narration, extra/unscripted voices).

Hard rejects: spontaneous music, transcript match below threshold, extra/unscripted voice, speaker similarity below
the reject floor, burned-in subtitles, or missing dialogue audio when dialogue was scripted.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any, Literal

from . import continuity, fsio, media
from .adapters.base import short_error, extract_dialogue_wav, staged_write
from .models import (
    DispatchStatus,
    GenClip,
    QAReport,
    Recommendation,
    RiskTier,
    Scene,
    Take,
    TakeStatus,
)
from .project import Project
from .registry import Registry
from .shot_generator import FIRST_FRAME_NAME, LAST_FRAME_NAME

AGENT = "agy.qa_critic"
QA_YAML_NAME = "qa.yaml"
_WORD_RE = re.compile(r"[\w\u0900-\u097F]+", re.UNICODE)


def _norm_words(text: str | None) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def compute_transcript_match(expected: str, actual: str) -> float:
    """Similarity in ``[0.0, 1.0]`` between scripted dialogue and ASR transcript (Devanagari-aware)."""
    exp_words = _norm_words(expected)
    act_words = _norm_words(actual)
    if not exp_words and not act_words:
        return 1.0
    if not exp_words or not act_words:
        return 0.0
    exp_joined = " ".join(exp_words)
    act_joined = " ".join(act_words)
    if exp_joined == act_joined:
        return 1.0
    seq_ratio = difflib.SequenceMatcher(None, exp_joined, act_joined).ratio()
    overlap = len(set(exp_words) & set(act_words)) / float(max(len(set(exp_words)), len(set(act_words))))
    return round(max(0.0, min(1.0, 0.65 * seq_ratio + 0.35 * overlap)), 4)


def _ensure_take_artifacts(project: Project, registry: Registry, clip: GenClip, take: Take, *, fake: bool | None) -> Take:
    if not take.video_uri:
        return take
    video = project.abs(take.video_uri)
    if not video.is_file():
        return take
    tdir = project.take_dir(take.id)
    changed = False
    first_png = tdir / FIRST_FRAME_NAME
    if not first_png.is_file():
        staged_write(first_png, lambda s: media.extract_frame(video, s, at="first", overwrite=True))
    if take.first_frame_uri != project.rel(first_png):
        take.first_frame_uri = project.rel(first_png)
        changed = True
    last_png = tdir / LAST_FRAME_NAME
    if not last_png.is_file():
        staged_write(last_png, lambda s: media.extract_frame(video, s, at="last", overwrite=True))
    if take.last_frame_uri != project.rel(last_png):
        take.last_frame_uri = project.rel(last_png)
        changed = True
    wav = project.abs(take.dialogue_wav_uri) if take.dialogue_wav_uri else None
    if wav is None or not wav.is_file():
        wav = extract_dialogue_wav(project, take)
        if wav is not None and wav.is_file():
            take.dialogue_wav_uri = project.rel(wav)
            changed = True
    if take.transcript is None:
        if wav is not None and wav.is_file():
            asr = registry.adapter("gemini_asr", project=project, fake=fake, registry=registry)
            expected = " ".join(continuity.clip_line_texts(clip))
            take.transcript = asr.transcribe(wav, language_hint="hi", expected_text=expected)
        else:
            take.transcript = ""
        changed = True
    if changed:
        project.save_take(take)
    return take


def _pack_images_by_entity(project: Project, clip: GenClip) -> dict[str, list[Path]]:
    idx = project.entity_state_index()
    ent_ids = [e.id for e in project.list_entities()]
    out: dict[str, list[Path]] = {}
    for state_id in clip.conditioning.reference_entity_states:
        pair = idx.get(state_id)
        if pair is None:
            continue
        ent, st = pair
        paths = [project.abs(u) for u in st.reference_pack.image_uris() if project.abs(u).is_file()]
        if paths:
            out[ent.id] = paths
    if not out:
        scene = project.load_scene(clip.scene_id)
        for state_id in scene.entity_states_active:
            ent_id = continuity.entity_id_of_state(state_id, ent_ids)
            pair = idx.get(state_id)
            if ent_id and pair is not None:
                paths = [project.abs(u) for u in pair[1].reference_pack.image_uris() if project.abs(u).is_file()]
                if paths:
                    out[ent_id] = paths
    return out


def score_take(
    project: Project,
    registry: Registry,
    clip: GenClip,
    take: Take,
    *,
    prev_take: Take | None = None,
    fake: bool | None = None,
) -> QAReport:
    """Compute and persist ``QAReport`` (`takes/<TAKE_ID>/qa.yaml` + `take.qa`)."""
    take = _ensure_take_artifacts(project, registry, clip, take, fake=fake)
    thr = project.config.settings.qa

    vision = registry.adapter("gemini_vision_judge", project=project, fake=fake, registry=registry)
    asr = registry.adapter("gemini_asr", project=project, fake=fake, registry=registry)
    spk_sim = registry.adapter("speaker_sim_local", project=project, fake=fake, registry=registry)

    frames = [
        project.abs(u)
        for u in (take.first_frame_uri, take.last_frame_uri)
        if u and project.abs(u).is_file()
    ]
    pack_map = _pack_images_by_entity(project, clip)
    identity_similarity: dict[str, float] = {}
    for ent_id, pack_imgs in pack_map.items():
        identity_similarity[ent_id] = float(vision.identity_similarity(frames, pack_imgs))

    anchor_uri = clip.conditioning.first_frame_anchor or (prev_take.last_frame_uri if prev_take else None)
    spatial_continuity: float | None = None
    if anchor_uri and take.first_frame_uri and project.abs(anchor_uri).is_file() and project.abs(take.first_frame_uri).is_file():
        spatial_continuity = float(vision.spatial_match(project.abs(take.first_frame_uri), project.abs(anchor_uri)))

    prompt_adherence = float(vision.prompt_adherence(frames, clip.spec.visual.subject_and_action)) if frames else 0.0
    artifact_flags = list(vision.artifact_flags(frames)) if frames else ["missing_frames"]

    # Measured, model-free: is this a moving picture at all? A storyboard still padded into an mp4 (slideshow /
    # ffmpeg zoompan) has (near-)identical sampled frames and is never an acceptable take.
    motion_stats: dict[str, Any] | None = None
    if take.video_uri and project.abs(take.video_uri).is_file():
        try:
            motion_stats = media.analyze_motion(project.abs(take.video_uri), n=9)
        except Exception as exc:  # noqa: BLE001 - an undecodable "video" is itself a hard reject below
            motion_stats = {"error": short_error(exc), "static": True, "max_diff": 0.0, "mean_diff": 0.0}
        if motion_stats.get("static"):
            artifact_flags.append("static_video")
        elif motion_stats.get("low_motion"):
            artifact_flags.append("low_motion")
    # Provenance: a real take was submitted to a provider (operation_name) — an mp4 placed into the take folder by
    # a script (storyboard still + zoompan, slideshow) never was, and is never acceptable.
    unprovenanced = bool(take.video_uri) and not take.operation_name

    expected_text = " ".join(continuity.clip_line_texts(clip)).strip()
    wav_path = project.abs(take.dialogue_wav_uri) if take.dialogue_wav_uri else None
    has_wav = wav_path is not None and wav_path.is_file()
    actual_text = (take.transcript or "").strip()
    transcript_match = compute_transcript_match(expected_text, actual_text)

    audio_info = (
        asr.classify_audio(wav_path)
        if has_wav
        else {"music_present": False, "narration_present": False, "extra_voice": False, "n_speakers": 0, "notes": "no wav"}
    )
    music_present = bool(audio_info.get("music_present", False))
    unwanted_audio_flags: list[str] = []
    if music_present:
        unwanted_audio_flags.append("music_present")
    if audio_info.get("narration_present"):
        unwanted_audio_flags.append("narration_present")
    if audio_info.get("extra_voice"):
        unwanted_audio_flags.append("extra_voice")
    if not expected_text and (actual_text or int(audio_info.get("n_speakers", 0) or 0) > 0):
        unwanted_audio_flags.append("unscripted_speech")

    speakers = continuity.speakers_in_clip(clip)
    speaker_similarity: dict[str, float | None] = {}
    for char_id in speakers:
        vp = project.load_voice_profile(char_id)
        if vp is None or not vp.anchor_wav or not has_wav or not project.abs(vp.anchor_wav).is_file():
            speaker_similarity[char_id] = None
        else:
            speaker_similarity[char_id] = float(spk_sim.similarity(wav_path, project.abs(vp.anchor_wav)))

    hard_reject_reasons: list[str] = []
    if unprovenanced:
        hard_reject_reasons.append("unprovenanced_video (no provider operation behind this file — not a generated take)")
    if motion_stats is not None and motion_stats.get("static"):
        hard_reject_reasons.append(
            f"static_video (sampled frames do not change: max_diff={float(motion_stats.get('max_diff', 0.0)):.4f}"
            f"{'; ' + str(motion_stats['error']) if motion_stats.get('error') else ''}) — a still/slideshow, not a video take")
    if music_present:
        hard_reject_reasons.append("spontaneous_music")
    if expected_text and not has_wav:
        hard_reject_reasons.append("missing_dialogue_audio")
    if expected_text and transcript_match < float(thr.transcript_match_min):
        hard_reject_reasons.append(f"transcript_mismatch ({transcript_match:.2f} < {thr.transcript_match_min:.2f})")
    if any(f in unwanted_audio_flags for f in ("extra_voice", "narration_present", "unscripted_speech")):
        hard_reject_reasons.append("extra_or_unscripted_voice")
    for char_id, sim in speaker_similarity.items():
        if sim is not None and sim < float(thr.speaker_similarity_reject):
            hard_reject_reasons.append(f"speaker_voice_mismatch:{char_id} ({sim:.2f} < {thr.speaker_similarity_reject:.2f})")
        elif sim is not None and sim < float(thr.speaker_similarity_warn):
            flag = f"voice_drift_warning:{char_id}"
            if flag not in unwanted_audio_flags:
                unwanted_audio_flags.append(flag)

    for ent_id, id_sim in identity_similarity.items():
        if id_sim < float(thr.identity_similarity_min):
            flag = f"identity_drift:{ent_id}"
            if flag not in artifact_flags:
                artifact_flags.append(flag)

    if any("subtitle" in f.lower() for f in artifact_flags):
        hard_reject_reasons.append("burned_in_subtitles")

    components: list[float] = [prompt_adherence, transcript_match]
    if identity_similarity:
        components.append(sum(identity_similarity.values()) / len(identity_similarity))
    if spatial_continuity is not None:
        components.append(spatial_continuity)
    known_sims = [s for s in speaker_similarity.values() if s is not None]
    if known_sims:
        components.append(sum(known_sims) / len(known_sims))

    raw_composite = sum(components) / len(components) if components else 0.0
    if hard_reject_reasons:
        composite = round(min(0.35, raw_composite * 0.4), 4)
        recommendation = Recommendation.REJECT
    elif artifact_flags or unwanted_audio_flags or raw_composite < float(thr.auto_select_composite_min):
        composite = round(raw_composite, 4)
        recommendation = Recommendation.REVIEW
    else:
        composite = round(raw_composite, 4)
        recommendation = Recommendation.ACCEPT

    report = QAReport(
        take_id=take.id,
        identity_similarity=identity_similarity,
        spatial_continuity=spatial_continuity,
        prompt_adherence=prompt_adherence,
        artifact_flags=artifact_flags,
        transcript=actual_text,
        transcript_match=transcript_match,
        speaker_similarity=speaker_similarity,
        music_present=music_present,
        unwanted_audio_flags=unwanted_audio_flags,
        hard_reject_reasons=hard_reject_reasons,
        composite=composite,
        recommendation=recommendation,
        generated_by=fsio.generated_by(AGENT),
    )
    fsio.save_versioned(project.take_dir(take.id) / QA_YAML_NAME, report)
    take.qa = report
    project.save_take(take)
    return report


def rank_takes(project: Project, clip: GenClip, takes: list[Take]) -> list[Take]:
    """Order ``takes`` best-first (non-rejected before rejected, higher ``qa.composite`` first) and save ``rank``."""

    def _key(t: Take) -> tuple[int, float, str]:
        qa = t.qa
        rejected = (
            t.status == TakeStatus.REJECTED
            or t.dispatch_status != DispatchStatus.COMPLETED
            or qa is None
            or qa.recommendation == Recommendation.REJECT
            or bool(qa.hard_reject_reasons)
        )
        comp = float(qa.composite) if qa is not None else -1.0
        return (1 if rejected else 0, -comp, t.id)

    ordered = sorted(takes, key=_key)
    out: list[Take] = []
    for idx, take in enumerate(ordered, start=1):
        if take.rank != idx:
            take.rank = idx
            project.save_take(take)
        out.append(take)
    return out


def route_selection(
    project: Project,
    scene: Scene,
    clip: GenClip,
    ranked: list[Take],
) -> Literal["auto_select", "human"]:
    """Decide whether the top take can be auto-selected (§7.2, §9.1) or must go to the human at G5."""
    settings = project.config.settings
    if not settings.auto_select_low_risk:
        return "human"
    if scene.breakdown is None or scene.breakdown.risk_tier != RiskTier.LOW:
        return "human"
    if clip.flags:
        return "human"
    for char_id in continuity.speakers_in_clip(clip):
        vp = project.load_voice_profile(char_id)
        if vp is None or not vp.anchor_take or not vp.anchor_wav:
            return "human"
    if not ranked:
        return "human"
    best = ranked[0]
    qa = best.qa
    if qa is None or qa.hard_reject_reasons or qa.recommendation != Recommendation.ACCEPT:
        return "human"
    if qa.composite < float(settings.qa.auto_select_composite_min):
        return "human"
    if qa.artifact_flags or qa.unwanted_audio_flags:
        return "human"
    return "auto_select"


def rescore_voice(
    project: Project,
    registry: Registry,
    character_id: str,
    *,
    fake: bool | None = None,
) -> list[str]:
    """Re-score ``speaker_similarity`` for every existing take where ``character_id`` speaks (never regenerates)."""
    fsio.validate_id(character_id, "character id")
    vp = project.load_voice_profile(character_id)
    anchor_wav = project.abs(vp.anchor_wav) if (vp is not None and vp.anchor_wav) else None
    has_anchor = anchor_wav is not None and anchor_wav.is_file()
    spk_sim = registry.adapter("speaker_sim_local", project=project, fake=fake, registry=registry)
    thr = project.config.settings.qa

    rescored: list[str] = []
    for clip in project.iter_all_clips():
        if character_id not in continuity.speakers_in_clip(clip):
            continue
        takes = project.list_takes(clip.id)
        for take in takes:
            if take.dispatch_status != DispatchStatus.COMPLETED or not take.video_uri:
                continue
            if take.qa is None:
                score_take(project, registry, clip, take, fake=fake)
                rescored.append(take.id)
                continue
            wav = project.abs(take.dialogue_wav_uri) if take.dialogue_wav_uri else None
            if wav is None or not wav.is_file():
                wav = extract_dialogue_wav(project, take)
                if wav is not None and wav.is_file():
                    take.dialogue_wav_uri = project.rel(wav)
            sim: float | None = None
            if has_anchor and wav is not None and wav.is_file():
                sim = float(spk_sim.similarity(wav, anchor_wav))

            qa = take.qa.model_copy(deep=True)
            qa.speaker_similarity[character_id] = sim
            qa.hard_reject_reasons = [
                r for r in qa.hard_reject_reasons if not r.startswith(f"speaker_voice_mismatch:{character_id}")
            ]
            qa.unwanted_audio_flags = [
                f for f in qa.unwanted_audio_flags if f != f"voice_drift_warning:{character_id}"
            ]
            if sim is not None and sim < float(thr.speaker_similarity_reject):
                qa.hard_reject_reasons.append(
                    f"speaker_voice_mismatch:{character_id} ({sim:.2f} < {thr.speaker_similarity_reject:.2f})"
                )
            elif sim is not None and sim < float(thr.speaker_similarity_warn):
                qa.unwanted_audio_flags.append(f"voice_drift_warning:{character_id}")

            comps = [qa.prompt_adherence, qa.transcript_match]
            if qa.identity_similarity:
                comps.append(sum(qa.identity_similarity.values()) / len(qa.identity_similarity))
            if qa.spatial_continuity is not None:
                comps.append(qa.spatial_continuity)
            known_sims = [s for s in qa.speaker_similarity.values() if s is not None]
            if known_sims:
                comps.append(sum(known_sims) / len(known_sims))
            raw_comp = sum(comps) / len(comps) if comps else 0.0
            if qa.hard_reject_reasons:
                qa.composite = round(min(0.35, raw_comp * 0.4), 4)
                qa.recommendation = Recommendation.REJECT
            elif qa.artifact_flags or qa.unwanted_audio_flags or raw_comp < float(thr.auto_select_composite_min):
                qa.composite = round(raw_comp, 4)
                qa.recommendation = Recommendation.REVIEW
            else:
                qa.composite = round(raw_comp, 4)
                qa.recommendation = Recommendation.ACCEPT
            qa.generated_by = fsio.generated_by(AGENT, notes=f"rescore_voice {character_id}")

            fsio.save_versioned(project.take_dir(take.id) / QA_YAML_NAME, qa)
            take.qa = qa
            project.save_take(take)
            rescored.append(take.id)
        if takes:
            rank_takes(project, clip, project.list_takes(clip.id))
    return rescored
