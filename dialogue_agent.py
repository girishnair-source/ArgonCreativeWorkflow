"""Dialogue Agent & Acoustic Continuity Engine for Multi-Format AI Video Production.

Ensures across cartoons, feature films, 20-minute episodes, and vertical microdramas that:
1. Human Pacing & Word-Count Budgets: Every line of dialogue fits naturally inside the shot's
   speaking window (accounting for pre-roll breath, action beats, and post-roll cut cushion)
   so speech never sounds rushed, robotic, or clipped at the edit point.
2. Cross-Generation Acoustic Voice Consistency: Every character has an immutable
   `CharacterVoiceProfile` (pitch band in Hz, vocal timbre, cadence WPM, dialect/accent)
   injected identically into every shot where that character speaks.
3. Single-Speaker Isolation & Lip-Lock Protocol: In multi-character or off-screen shots,
   strictly locks ONLY the intended speaker's mouth and voice while explicitly locking all
   other visible characters' mouths CLOSED in both positive and negative prompts.
"""

from __future__ import annotations

import argparse
import json
import re
from enum import StrEnum
from typing import Annotated, Any

from pydantic import Field, model_validator

from schemas import StrictContract


class DeliveryMode(StrEnum):
    """How the spoken line is delivered relative to the camera frame."""

    ON_CAMERA = "on_camera"
    OFF_SCREEN = "off_screen"
    DEVICE_SPEAKER = "device_speaker"
    VOICEOVER_INTERNAL = "voiceover_internal"


class CadencePreset(StrEnum):
    """Standard human dramatic speaking rate presets."""

    SLOW_GRAVITAS = "slow_gravitas"          # ~1.7 words/sec (102 WPM)
    NATURAL_CONVERSATIONAL = "natural"       # ~2.2 words/sec (132 WPM)
    URGENT_THRILLER = "urgent_thriller"      # ~2.6 words/sec (156 WPM)
    COMEDIC_SNAPPY = "comedic_snappy"        # ~2.8 words/sec (168 WPM)


CADENCE_WPS_MAP: dict[CadencePreset, float] = {
    CadencePreset.SLOW_GRAVITAS: 1.7,
    CadencePreset.NATURAL_CONVERSATIONAL: 2.2,
    CadencePreset.URGENT_THRILLER: 2.6,
    CadencePreset.COMEDIC_SNAPPY: 2.8,
}


class CharacterVoiceProfile(StrictContract):
    """Immutable acoustic voice print for a character across all episodes and scenes."""

    character_id: Annotated[str, Field(pattern=r"^CHAR_[A-Z0-9_]+$")]
    speaker_name: str
    vocal_gender_age_tag: Annotated[
        str,
        Field(
            min_length=5,
            description="E.g., '21-year-old Indian male', '52-year-old authoritative father', 'stylized energetic 10yo cartoon boy'.",
        ),
    ]
    pitch_band: Annotated[
        str,
        Field(
            min_length=5,
            description="E.g., 'warm mid-tenor (130-165 Hz)', 'deep baritone (95-120 Hz)', 'bright mezzo-soprano (195-235 Hz)'.",
        ),
    ]
    timbre_and_texture: Annotated[
        str,
        Field(
            min_length=10,
            description="Acoustic texture, resonance, breathiness, and consonant clarity.",
        ),
    ]
    accent_and_dialect: Annotated[
        str,
        Field(
            min_length=5,
            description="Regional accent, language mix (e.g., 'natural urban Mumbai Hinglish', 'neutral General American English').",
        ),
    ]
    default_cadence: CadencePreset = CadencePreset.NATURAL_CONVERSATIONAL
    target_wpm: Annotated[int, Field(ge=80, le=200)] = 132
    vocal_quirks: str = Field(
        default="Natural human micro-pauses and breath intake before speaking.",
        description="Consistent speech mannerisms that anchor character identity.",
    )

    def compile_acoustic_lock_token(self) -> str:
        """Return the verbatim voice identity token injected into every shot where this character speaks."""
        return (
            f"VOICE IDENTITY LOCK [{self.speaker_name} ({self.character_id})]: "
            f"{self.vocal_gender_age_tag}, {self.pitch_band}, {self.timbre_and_texture}, "
            f"speaking in {self.accent_and_dialect} at ~{self.target_wpm} WPM ({self.vocal_quirks})"
        )


class DialoguePacingReport(StrictContract):
    """Validation result checking if a dialogue line fits humanly inside a shot's duration."""

    shot_id: str
    speaker_id: str | None
    spoken_text: str
    word_count: int
    estimated_syllables: int
    shot_duration_sec: float
    effective_speaking_window_sec: float
    estimated_speech_duration_sec: float
    max_recommended_words: int
    hard_ceiling_words: int
    is_human_paced: bool
    pacing_status: str
    recommendation: str


class SceneDialogueLine(StrictContract):
    """Single-speaker validated dialogue specification for a shot or beat."""

    shot_id: str
    intended_speaker_id: Annotated[
        str,
        Field(pattern=r"^CHAR_[A-Z0-9_]+$", description="The ONLY character permitted to speak in this beat."),
    ]
    intended_speaker_visual_anchor: Annotated[
        str,
        Field(
            min_length=10,
            description="Screen position and wardrobe cue of the speaker, e.g., 'Rishi in olive hoodie on screen-right'.",
        ),
    ]
    visible_silent_character_anchors: dict[str, str] = Field(
        default_factory=dict,
        description="Map of silent character_id -> visual position/wardrobe anchor who MUST keep their mouths closed.",
    )
    spoken_text: Annotated[str, Field(min_length=1)]
    spoken_hindi_devanagari: str | None = Field(
        default=None,
        description="Authentic Devanagari Hindi text to ensure native Indian accent and accurate speech synthesis.",
    )
    subtitle_text: str | None = None
    delivery_mode: DeliveryMode = DeliveryMode.ON_CAMERA
    emotional_direction: str = Field(
        default="natural conversational realism",
        description="Acting performance note (e.g., 'hushed trembling whisper', 'confident teasing grin').",
    )
    shot_duration_sec: Annotated[float, Field(ge=2.0, le=30.0)] = 8.0
    pre_roll_silence_sec: Annotated[float, Field(ge=0.2, le=5.0)] = 0.6
    post_roll_silence_sec: Annotated[float, Field(ge=0.2, le=5.0)] = 0.6
    action_pause_beats_sec: Annotated[float, Field(ge=0.0, le=10.0)] = 0.3
    cadence_override: CadencePreset | None = None

    @model_validator(mode="after")
    def _ensure_speaker_not_in_silent_list(self) -> SceneDialogueLine:
        if (
            self.delivery_mode == DeliveryMode.ON_CAMERA
            and self.intended_speaker_id in self.visible_silent_character_anchors
        ):
            raise ValueError(
                f"intended_speaker_id '{self.intended_speaker_id}' cannot also be in "
                "visible_silent_character_anchors when delivery_mode is ON_CAMERA."
            )
        return self


def count_words_and_syllables(text: str) -> tuple[int, int]:
    """Count spoken words and approximate syllables (supporting English + Romanized Hinglish)."""
    cleaned = re.sub(r"[^\w\s'-]", " ", text.strip())
    words = [w for w in cleaned.split() if w]
    word_count = len(words)
    syllables = 0
    for w in words:
        vowel_groups = re.findall(r"[aeiouyAEIOUY]+", w)
        syllables += max(1, len(vowel_groups))
    return word_count, syllables


def validate_dialogue_pacing(
    line: SceneDialogueLine,
    voice_profile: CharacterVoiceProfile | None = None,
) -> DialoguePacingReport:
    """Validate that `line.spoken_text` fits comfortably within the shot's speaking window."""
    words, syllables = count_words_and_syllables(line.spoken_text)

    if line.cadence_override is not None:
        wps = CADENCE_WPS_MAP[line.cadence_override]
    elif voice_profile is not None:
        wps = voice_profile.target_wpm / 60.0
    else:
        wps = CADENCE_WPS_MAP[CadencePreset.NATURAL_CONVERSATIONAL]

    # Effective speaking window leaves room for actor reaction at start, physical pause beats, and clean tail cut
    effective_window = max(
        0.5,
        line.shot_duration_sec
        - line.pre_roll_silence_sec
        - line.post_roll_silence_sec
        - line.action_pause_beats_sec,
    )

    # Syllables also constrain speed (~3.3 syllables/sec for natural speech)
    word_duration = words / wps if wps > 0 else 0.0
    syllable_duration = syllables / 3.3
    estimated_speech_sec = round(max(word_duration, syllable_duration), 2)

    max_recommended_words = max(2, int(effective_window * wps))
    hard_ceiling_words = max(3, int(effective_window * 2.75))

    if words <= max_recommended_words and estimated_speech_sec <= effective_window:
        is_ok = True
        status = "OPTIMAL_HUMAN_PACE"
        rec = (
            f"Fits comfortably ({words} words / ~{estimated_speech_sec:.1f}s in {effective_window:.1f}s window). "
            f"Leaves {line.pre_roll_silence_sec:.1f}s pre-roll breath and {line.post_roll_silence_sec:.1f}s tail cushion."
        )
    elif words <= hard_ceiling_words:
        is_ok = True
        status = "BRISK_BUT_ACCEPTABLE"
        rec = (
            f"Slightly brisk ({words} words vs {max_recommended_words} optimal for {line.shot_duration_sec:.0f}s shot). "
            "Consider trimming 2-3 filler words for stronger dramatic pause."
        )
    else:
        is_ok = False
        status = "OVER_BUDGET_RISK_OF_CLIPPING"
        excess = words - max_recommended_words
        rec = (
            f"TOO LONG for {line.shot_duration_sec:.0f}s shot ({words} words requires ~{estimated_speech_sec:.1f}s, "
            f"but speaking window is {effective_window:.1f}s). Trim at least {excess} words or split across 2 shots."
        )

    return DialoguePacingReport(
        shot_id=line.shot_id,
        speaker_id=line.intended_speaker_id,
        spoken_text=line.spoken_text,
        word_count=words,
        estimated_syllables=syllables,
        shot_duration_sec=line.shot_duration_sec,
        effective_speaking_window_sec=round(effective_window, 2),
        estimated_speech_duration_sec=estimated_speech_sec,
        max_recommended_words=max_recommended_words,
        hard_ceiling_words=hard_ceiling_words,
        is_human_paced=is_ok,
        pacing_status=status,
        recommendation=rec,
    )


def compile_speaker_isolation_prompt(
    line: SceneDialogueLine,
    voice_profile: CharacterVoiceProfile,
) -> dict[str, str]:
    """Compile positive and negative prompt clauses guaranteeing ONLY the intended speaker speaks.

    Prevents:
    - Multi-mouth ventriloquism (listeners moving their mouths along with the speaker)
    - Voice swapping (wrong character's gender/timbre assigned to the speaker)
    - On-camera lip movement during off-screen / phone / narrator audio
    """
    voice_lock = voice_profile.compile_acoustic_lock_token()
    start_t = line.pre_roll_silence_sec
    end_t = max(start_t + 1.0, line.shot_duration_sec - line.post_roll_silence_sec)

    silent_clauses: list[str] = []
    for silent_cid, visual_desc in line.visible_silent_character_anchors.items():
        silent_clauses.append(f"{visual_desc} ({silent_cid}) remains 100% SILENT with mouth CLOSED")

    line_spec = (
        f'Exact spoken line in authentic colloquial Hindi (Devanagari): "{line.spoken_hindi_devanagari}" '
        f'(phonetic: "{line.spoken_text}"). Native Indian Hindi pronunciation with zero English accent.'
        if line.spoken_hindi_devanagari
        else f'Exact spoken line: "{line.spoken_text}".'
    )

    if line.delivery_mode == DeliveryMode.ON_CAMERA:
        speaker_clause = (
            f"SINGLE-SPEAKER LIP & VOICE LOCK: ONLY {line.intended_speaker_visual_anchor} "
            f"({line.intended_speaker_id}) speaks on camera between {start_t:.1f}s and {end_t:.1f}s "
            f"with {line.emotional_direction}. {voice_lock}. {line_spec}"
        )
        if silent_clauses:
            listener_clause = (
                "SILENT LISTENER MOUTH LOCK: "
                + "; ".join(silent_clauses)
                + " — zero lip movement on listeners, reacting only with eyes and subtle posture."
            )
        else:
            listener_clause = "No other character speaks or vocalizes."
    else:
        # Off-screen, phone speaker, or internal voiceover: ALL visible on-camera mouths must stay closed!
        all_on_screen = [line.intended_speaker_visual_anchor, *line.visible_silent_character_anchors.values()]
        speaker_clause = (
            f"OFF-CAMERA / DEVICE AUDIO LOCK ({line.delivery_mode.value}): "
            f"The voice of {voice_profile.speaker_name} emanates strictly via {line.delivery_mode.value} "
            f"between {start_t:.1f}s and {end_t:.1f}s with {line.emotional_direction}. {voice_lock}. {line_spec}"
        )
        listener_clause = (
            "ON-CAMERA MOUTH-CLOSED LOCK: Every visible character on screen ("
            + ", ".join(f for f in all_on_screen if f)
            + ") keeps their mouth COMPLETELY CLOSED in silent reaction — zero on-camera lip-sync."
        )

    negative_clause = (
        "wrong character speaking, multiple characters moving lips simultaneously, "
        "silent listener moving mouth, ventriloquism, voice gender swap, inconsistent vocal pitch, "
        "rushed chipmunk speech, cut-off final word, overlapping gibberish voices"
    )

    return {
        "voice_identity_lock": voice_lock,
        "positive_dialogue_prompt": f"{speaker_clause} {listener_clause}",
        "negative_dialogue_prompt": negative_clause,
    }


def _load_default_voice_profiles() -> dict[str, CharacterVoiceProfile]:
    """Load voice profiles from active project if present; otherwise return empty dict."""
    try:
        from projects.like_share_kill_or_die.casting_bible import get_voice_profiles

        return get_voice_profiles()  # type: ignore[return-value]
    except ImportError:
        return {}


DEFAULT_VOICE_PROFILES: dict[str, CharacterVoiceProfile] = _load_default_voice_profiles()


def audit_episode_dialogues(lines: list[SceneDialogueLine]) -> dict[str, Any]:
    """Audit a list of scene dialogue lines for word-count pacing and speaker isolation."""
    reports: list[dict[str, Any]] = []
    all_passed = True
    for line in lines:
        vp = DEFAULT_VOICE_PROFILES.get(
            line.intended_speaker_id,
            CharacterVoiceProfile(
                character_id=line.intended_speaker_id,
                speaker_name=line.intended_speaker_id,
                vocal_gender_age_tag="adult human speaker",
                pitch_band="natural mid-range",
                timbre_and_texture="clear conversational resonance",
                accent_and_dialect="natural conversational delivery",
            ),
        )
        pacing = validate_dialogue_pacing(line, vp)
        compiled = compile_speaker_isolation_prompt(line, vp)
        if not pacing.is_human_paced:
            all_passed = False
        reports.append(
            {
                "pacing": pacing.model_dump(),
                "compiled_prompts": compiled,
            }
        )
    return {
        "total_lines": len(lines),
        "all_human_paced": all_passed,
        "line_reports": reports,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Dialogue Agent CLI: Validate word-count pacing & compile single-speaker voice locks."
    )
    parser.add_argument("--text", type=str, help="Spoken dialogue line to test.")
    parser.add_argument("--speaker", type=str, default="CHAR_PROTAGONIST", help="Character ID (e.g. CHAR_PROTAGONIST).")
    parser.add_argument(
        "--speaker-anchor",
        type=str,
        default="Protagonist on screen-right",
        help="Visual anchor for the intended speaker.",
    )
    parser.add_argument("--duration", type=float, default=8.0, help="Shot duration in seconds (default: 8.0).")
    parser.add_argument(
        "--mode",
        type=str,
        default="on_camera",
        choices=[m.value for m in DeliveryMode],
        help="Delivery mode (on_camera, off_screen, device_speaker, voiceover_internal).",
    )
    args = parser.parse_args()

    if args.text:
        line = SceneDialogueLine(
            shot_id="TEST_SHOT_01",
            intended_speaker_id=args.speaker,
            intended_speaker_visual_anchor=args.speaker_anchor,
            spoken_text=args.text,
            shot_duration_sec=args.duration,
            delivery_mode=DeliveryMode(args.mode),
        )
        res = audit_episode_dialogues([line])
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
