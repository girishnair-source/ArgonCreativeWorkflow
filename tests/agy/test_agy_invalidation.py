"""Workstream A — anchor graph & invalidation (BUILD_SPEC_v3 §4.7, acceptance 4 and 6). No network."""

from __future__ import annotations

import pytest

from agy_studio.decisions import DecisionLog
from agy_studio.fsio import read_jsonl
from agy_studio.invalidation import AnchorGraph, FanOut, apply_invalidation, describe_fanout, request_confirm
from agy_studio.models import AudioLine, AudioSpec, ClipStatus, Conditioning, GenClip, SceneStatus, ShotSpec, Take
from agy_studio.project import parse_artifact_id
from agy_studio.state import ConfirmRequiredError, StateStore
from agy_studio.working_set import build_working_set

RISHI, RISHI_HURT = "CHAR_RISHI_STATE_DEFAULT", "CHAR_RISHI_STATE_INJURED"
MOHNISH, PHONE = "CHAR_MOHNISH_STATE_DEFAULT", "PROP_PHONE_STATE_DEFAULT"
LINES = {
    "CHAR_MOHNISH": ("SC01_L001", "भाई, लाइव चालू है... दो हज़ार लोग देख रहे हैं!"),
    "CHAR_RISHI": ("SC01_L002", "देख, अब असली खेल शुरू होगा।"),
    "CHAR_GUARD": ("SC01_L003", "कौन है वहाँ? नीचे उतरो!"),
}


def _line(char: str) -> AudioLine:
    line_id, text = LINES[char]
    return AudioLine(character_id=char, line_id=line_id, text=text)


def _clip(cid: str, *, refs: list[str], angle: str, speaker: str | None = None, off_screen: str | None = None,
          anchor_from: str | None = None) -> GenClip:
    ids = parse_artifact_id(cid)
    audio = AudioSpec(dialogue=[_line(speaker)] if speaker else [], off_screen=[_line(off_screen)] if off_screen else [])
    return GenClip(id=cid, shot_id=ids["shot"], scene_id=ids["scene"], sequence_index=int(cid[-2:]), duration_s=6,
                   lines=[_line(c).line_id for c in (speaker, off_screen) if c], spec=ShotSpec(audio_spec=audio),
                   environment_angle=angle, anchor_from_clip_id=anchor_from,
                   conditioning=Conditioning(reference_entity_states=refs,
                                             first_frame_anchor_take_id=f"{anchor_from}_T01" if anchor_from else None))


@pytest.fixture
def staged(demo_project):
    """SC01 with four generated clips: SH01_C01 (Mohnish speaks) and an anchored chain SH02_C01 -> C02 -> C03."""
    clips = [
        _clip("SC01_SH01_C01", refs=[RISHI, MOHNISH, PHONE], angle="wide_establishing", speaker="CHAR_MOHNISH"),
        _clip("SC01_SH02_C01", refs=[RISHI, MOHNISH], angle="reverse", speaker="CHAR_RISHI"),
        _clip("SC01_SH02_C02", refs=[RISHI_HURT, MOHNISH], angle="reverse", off_screen="CHAR_GUARD",
              anchor_from="SC01_SH02_C01"),
        _clip("SC01_SH02_C03", refs=[RISHI_HURT], angle="reverse", anchor_from="SC01_SH02_C02"),
    ]
    for c in clips:
        demo_project.save_clip(c)
        demo_project.save_take(Take(id=f"{c.id}_T01", clip_id=c.id, model="omni", payload_hash=f"hash_{c.id}"))
    return demo_project


def test_anchor_descendants_are_transitive_and_take_ids_map_to_their_clip(staged) -> None:
    g = AnchorGraph.build(staged)
    assert g.anchor_descendants("SC01_SH02_C01") == ["SC01_SH02_C02", "SC01_SH02_C03"]
    assert g.anchor_descendants("SC01_SH02_C02") == ["SC01_SH02_C03"]
    assert g.anchor_descendants("SC01_SH01_C01") == []
    fo = g.fanout("SC01_SH02_C01_T01")  # a different take selected for C01
    assert (fo.kind, fo.action, fo.requires_confirm) == ("anchor", "stale", False)
    assert (fo.clips, fo.shots, fo.scenes) == (["SC01_SH02_C02", "SC01_SH02_C03"], ["SC01_SH02"], ["SC01"])
    assert fo.takes == ["SC01_SH02_C02_T01", "SC01_SH02_C03_T01"]
    assert "2 later clips" in fo.summary and fo.summary == describe_fanout(fo)


def test_unrelated_clip_bytes_are_untouched_by_an_invalidation(staged) -> None:
    """Acceptance 6: only the fan-out is rewritten; SH01_C01's clip.yaml stays byte-identical."""
    unrelated = staged.clip_dir("SC01_SH01_C01") / "clip.yaml"
    before = unrelated.read_bytes()
    root_record = (staged.clip_dir("SC01_SH02_C01") / "clip.yaml").read_bytes()
    changed = apply_invalidation(staged, StateStore(staged), AnchorGraph(staged).fanout("SC01_SH02_C01"),
                                 reason="SC01_SH02_C01 regenerated")
    assert changed == ["SC01_SH02_C02", "SC01_SH02_C03"]
    assert unrelated.read_bytes() == before
    assert not (unrelated.parent / "_history").exists()
    assert (staged.clip_dir("SC01_SH02_C01") / "clip.yaml").read_bytes() == root_record
    assert staged.load_clip("SC01_SH02_C02").status == ClipStatus.STALE
    assert staged.load_clip("SC01_SH02_C03").status == ClipStatus.STALE
    assert any((staged.clip_dir("SC01_SH02_C02") / "_history").iterdir())  # previous record archived, never lost
    log = read_jsonl(staged.path("logs", "invalidation.jsonl"))
    assert log[-1]["reason"] == "SC01_SH02_C01 regenerated" and log[-1]["changed"] == changed


def test_pack_change_needs_a_vision_confirm(staged) -> None:
    state, log = StateStore(staged), DecisionLog(staged)
    fo = AnchorGraph(staged).fanout(RISHI_HURT)
    assert (fo.kind, fo.action, fo.requires_confirm) == ("pack", "stale", True)
    assert fo.clips == ["SC01_SH02_C02", "SC01_SH02_C03"]
    shown = request_confirm(state, fo)
    assert f"confirm {RISHI_HURT}" in shown and state.pending_confirm(RISHI_HURT) is not None
    with pytest.raises(ConfirmRequiredError):
        apply_invalidation(staged, state, fo, reason="new scrape design")
    craft = log.record(tier="craft", target_artifact_id=RISHI_HURT, action="approve")
    with pytest.raises(ConfirmRequiredError):
        apply_invalidation(staged, state, fo, reason="new scrape design", decision_id=craft.id)
    other = log.record(tier="vision", target_artifact_id=MOHNISH, action="lock")
    with pytest.raises(ConfirmRequiredError):
        apply_invalidation(staged, state, fo, reason="new scrape design", decision_id=other.id)
    refused = log.record(tier="vision", target_artifact_id=RISHI_HURT, action="reject", payload={"text": "no"})
    with pytest.raises(ConfirmRequiredError):
        apply_invalidation(staged, state, fo, reason="new scrape design", decision_id=refused.id)
    with pytest.raises(ConfirmRequiredError):
        apply_invalidation(staged, state, fo, reason="new scrape design", decision_id="D999999")
    assert staged.load_clip("SC01_SH02_C02").status == ClipStatus.PENDING  # nothing moved without a confirm
    confirm = log.record(tier="vision", target_artifact_id=RISHI_HURT, action="lock", payload={"confirm": RISHI_HURT})
    changed = apply_invalidation(staged, state, fo, reason="new scrape design", decision_id=confirm.id)
    assert changed == ["SC01_SH02_C02", "SC01_SH02_C03"]
    assert state.pending_confirm(RISHI_HURT) is None  # the confirm handshake is consumed
    assert staged.load_clip("SC01_SH01_C01").status == ClipStatus.PENDING
    assert staged.load_clip("SC01_SH02_C01").status == ClipStatus.PENDING


def test_entity_change_covers_every_state_of_that_entity(staged) -> None:
    fo = AnchorGraph(staged).fanout("CHAR_RISHI")
    assert fo.kind == "pack" and fo.requires_confirm
    assert fo.clips == ["SC01_SH01_C01", "SC01_SH02_C01", "SC01_SH02_C02", "SC01_SH02_C03"]
    assert AnchorGraph(staged).fanout(PHONE).clips == ["SC01_SH01_C01"]


def test_voice_reanchor_marks_only_that_characters_speaking_clips(staged) -> None:
    state = StateStore(staged)
    g = AnchorGraph(staged)
    fo = g.fanout_voice("CHAR_RISHI")
    assert (fo.root, fo.kind, fo.action, fo.requires_confirm) == (
        "VOICE_CHAR_RISHI", "voice", "needs_voice_review", True)
    assert fo.clips == ["SC01_SH02_C01"] and fo.takes == ["SC01_SH02_C01_T01"]
    assert "Nothing is regenerated" in fo.summary
    assert g.fanout("VOICE_CHAR_RISHI").clips == fo.clips
    with pytest.raises(ConfirmRequiredError):
        apply_invalidation(staged, state, fo, reason="Rishi re-anchored")
    request_confirm(state, fo)
    confirm = DecisionLog(staged).record(tier="vision", target_artifact_id="VOICE_CHAR_RISHI", action="choose",
                                         payload={"command": "anchor", "anchor_take": "SC01_SH02_C01_T01"})
    others = ("SC01_SH01_C01", "SC01_SH02_C02", "SC01_SH02_C03")
    before = {c: (staged.clip_dir(c) / "clip.yaml").read_bytes() for c in others}
    changed = apply_invalidation(staged, state, fo, reason="Rishi re-anchored", decision_id=confirm.id)
    assert changed == ["SC01_SH02_C01"]
    assert staged.load_clip("SC01_SH02_C01").status == ClipStatus.NEEDS_VOICE_REVIEW
    assert {c: (staged.clip_dir(c) / "clip.yaml").read_bytes() for c in others} == before
    assert state.pending_confirm("VOICE_CHAR_RISHI") is None
    assert g.fanout_voice("CHAR_GUARD").clips == ["SC01_SH02_C02"]  # off-screen lines count as speaking
    assert g.fanout_voice("CHAR_MOHNISH").clips == ["SC01_SH01_C01"]


def test_statuses_are_never_downgraded(staged) -> None:
    state = StateStore(staged)
    g = AnchorGraph(staged)
    apply_invalidation(staged, state, g.fanout("SC01_SH02_C01"), reason="C01 regenerated")  # C02, C03 -> stale
    confirm = DecisionLog(staged).record(tier="vision", target_artifact_id="VOICE_CHAR_GUARD", action="lock")
    assert apply_invalidation(staged, state, g.fanout_voice("CHAR_GUARD"), reason="guard re-anchored",
                              decision_id=confirm.id) == []
    assert staged.load_clip("SC01_SH02_C02").status == ClipStatus.STALE
    relint = g.fanout("SC01_SH02")
    assert (relint.kind, relint.action, relint.requires_confirm) == ("shot_spec", "needs_relint", False)
    assert relint.clips == ["SC01_SH02_C01", "SC01_SH02_C02", "SC01_SH02_C03"]
    assert apply_invalidation(staged, state, relint, reason="shot spec edited") == ["SC01_SH02_C01"]
    assert staged.load_clip("SC01_SH02_C01").status == ClipStatus.NEEDS_RELINT
    assert staged.load_clip("SC01_SH02_C03").status == ClipStatus.STALE


def test_environment_angle_fanout(staged) -> None:
    g = AnchorGraph(staged)
    rev = g.fanout("ENV_TERRACE:reverse")
    assert rev.kind == "environment_angle" and rev.requires_confirm and rev.action == "stale"
    assert rev.clips == ["SC01_SH02_C01", "SC01_SH02_C02", "SC01_SH02_C03"]
    assert "confirm ENV_TERRACE`" in rev.summary
    assert g.fanout("ENV_TERRACE").clips == ["SC01_SH01_C01", "SC01_SH02_C01", "SC01_SH02_C02", "SC01_SH02_C03"]
    assert g.fanout("ENV_STAIRWELL").clips == []
    confirm = DecisionLog(staged).record(tier="vision", target_artifact_id="ENV_TERRACE", action="lock")
    assert apply_invalidation(staged, StateStore(staged), rev, reason="reverse angle redrawn",
                              decision_id=confirm.id) == rev.clips
    assert staged.load_clip("SC01_SH01_C01").status == ClipStatus.PENDING


def test_exit_state_change_flags_the_next_scene_for_review(staged) -> None:
    build_working_set(staged, "SC02")
    g = AnchorGraph(staged)
    fo = g.fanout("SC01")
    assert (fo.kind, fo.scenes, fo.action, fo.clips, fo.requires_confirm) == (
        "exit_state", ["SC02"], "needs_review", [], False)
    assert "SC02 must be reviewed again" in fo.summary
    assert apply_invalidation(staged, StateStore(staged), fo, reason="SC01 now ends on the stairs") == ["SC02"]
    assert staged.load_scene("SC02").status == SceneStatus.NEEDS_REVIEW
    assert staged.load_scene("SC01").status == SceneStatus.PENDING
    assert g.fanout("SC02").scenes == []  # last scene: nothing after it


def test_invalidation_refuses_bad_input(staged) -> None:
    state = StateStore(staged)
    g = AnchorGraph(staged)
    with pytest.raises(ValueError):
        apply_invalidation(staged, state, g.fanout("SC01_SH02_C01"), reason="   ")
    tampered = FanOut(root="SC01_SH02_C01", kind="anchor", clips=["../../outside"], action="stale")
    with pytest.raises(ValueError):
        apply_invalidation(staged, state, tampered, reason="tampered fan-out")
    with pytest.raises(ValueError):
        g.fanout("NOT_A_THING")
    with pytest.raises(ValueError):
        apply_invalidation(staged, state, g.fanout("SC01_SH02_C01"), reason="x", decision_id="../D1")
    assert staged.load_clip("SC01_SH02_C02").status == ClipStatus.PENDING
