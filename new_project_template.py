"""Starter Template for Editors: Spin Up Any Cartoon, 20-Min Episode, Movie, or Microdrama.

Copy this file (e.g. `cp new_project_template.py my_new_show.py`), customize your:
1. `GlobalStyleLock` (Live-Action Cinema, 2D Cel Cartoon, 3D Stylized Animation, Anime, Stop-Motion)
2. `characters` & `voice_profiles` (Immutable physical & vocal identity across all episodes)
3. `locations` (Architectural & lighting locks per setting)
4. `episodes` -> `scenes` -> `shots` (Scales from 9-shot vertical microdramas to 200-shot 20-min episodes)

Run pre-flight checks before spending any tokens:
    .venv/bin/python new_project_template.py
"""

from __future__ import annotations

import json

from dialogue_agent import CadencePreset, CharacterVoiceProfile, DeliveryMode, SceneDialogueLine
from production_bible import (
    CharacterCastSpec,
    EpisodeSpec,
    GlobalStyleLock,
    LocationScoutSpec,
    ProductionScale,
    ProjectBibleSpec,
    ScalableShotSpec,
    SceneSpec,
    VisualMedium,
    compile_episode_shots,
    preflight_project_continuity,
)
from schemas import AspectRatio


def build_starter_project() -> ProjectBibleSpec:
    """Example multi-scene project showing how to configure a Cartoon, 20-Min Episode, or Movie."""
    return ProjectBibleSpec(
        project_id="STARTER_SHOW_S01",
        title="Starter Show — Multi-Scene Template",
        style_lock=GlobalStyleLock(
            # Switch medium to:
            # - VisualMedium.LIVE_ACTION_CINEMA
            # - VisualMedium.CARTOON_2D_CEL
            # - VisualMedium.CARTOON_3D_STYLIZED
            # - VisualMedium.ANIME_2D
            # - VisualMedium.STOP_MOTION_CLAY
            medium=VisualMedium.CARTOON_3D_STYLIZED,
            scale=ProductionScale.SHORT_EPISODE_20MIN,
            aspect_ratio=AspectRatio.LANDSCAPE,
            width=1920,
            height=1080,
            fps=24.0,
            color_palette_and_lut_lock=(
                "Warm amber key lights against deep teal shadows, tactile fabric and surface shaders."
            ),
            lens_or_rendering_signature=(
                "Stylized 3D feature animation render with 35mm virtual cinema camera depth of field."
            ),
        ),
        characters={
            "CHAR_LEO": CharacterCastSpec(
                character_id="CHAR_LEO",
                name="Leo",
                age=12,
                role_description="Curious young inventor protagonist",
                immutable_physical_prompt=(
                    "12-year-old boy with tousled chestnut-brown curls, bright hazel eyes, "
                    "freckled nose bridge, and expressive round eyebrows."
                ),
                wardrobe_by_state={
                    "workshop": "Mustard-yellow canvas work apron over a navy-blue rolled-sleeve shirt, brass goggles on forehead.",
                    "expedition": "Weatherproof olive parka with glowing compass badge on chest.",
                },
                anchor_sheet_id="ANCHOR_CAST_LEO",
            ),
            "CHAR_MAYA": CharacterCastSpec(
                character_id="CHAR_MAYA",
                name="Maya",
                age=13,
                role_description="Sharp-witted navigator partner",
                immutable_physical_prompt=(
                    "13-year-old girl with sleek black bob haircut, warm bronze skin, "
                    "sharp dark-brown eyes, and a confident grin."
                ),
                wardrobe_by_state={
                    "workshop": "Burgundy bomber jacket over charcoal overalls with utility belt.",
                    "expedition": "Crimson windbreaker with silver reflective trim.",
                },
                anchor_sheet_id="ANCHOR_CAST_MAYA",
            ),
        },
        voice_profiles={
            "CHAR_LEO": CharacterVoiceProfile(
                character_id="CHAR_LEO",
                speaker_name="Leo",
                vocal_gender_age_tag="12-year-old energetic boy",
                pitch_band="bright youthful alto-tenor (175-210 Hz)",
                timbre_and_texture="warm, enthusiastic, clear consonant articulation",
                accent_and_dialect="neutral English conversational animation delivery",
                default_cadence=CadencePreset.NATURAL_CONVERSATIONAL,
                target_wpm=135,
            ),
            "CHAR_MAYA": CharacterVoiceProfile(
                character_id="CHAR_MAYA",
                speaker_name="Maya",
                vocal_gender_age_tag="13-year-old confident girl",
                pitch_band="crisp mezzo-soprano (200-240 Hz)",
                timbre_and_texture="grounded, witty, decisive cadence",
                accent_and_dialect="neutral English conversational animation delivery",
                default_cadence=CadencePreset.NATURAL_CONVERSATIONAL,
                target_wpm=140,
            ),
        },
        locations={
            "LOC_WORKSHOP": LocationScoutSpec(
                location_id="LOC_WORKSHOP",
                name="Attic Clockwork Workshop",
                setting_type="INT",
                time_of_day="Late Afternoon Golden Hour",
                architectural_prompt=(
                    "Timber-beamed attic workshop filled with brass gears, drafting tables, "
                    "and a large circular stained-glass window on the back wall."
                ),
                lighting_and_lens_prompt=(
                    "Warm 3500K sunbeams cutting through dust motes from the circular window, "
                    "cool 5600K fill from a desk lamp."
                ),
                anchor_plate_id="ANCHOR_LOC_WORKSHOP",
            ),
            "LOC_ROOFTOP_DOCK": LocationScoutSpec(
                location_id="LOC_ROOFTOP_DOCK",
                name="Skyline Airship Dock",
                setting_type="EXT",
                time_of_day="Blue Hour Dusk",
                architectural_prompt=(
                    "Weathered wooden pier extending from a clocktower roof over a canyon of glowing lanterns."
                ),
                lighting_and_lens_prompt=(
                    "Deep indigo twilight sky (6500K) contrasted with warm 2800K hanging dock lanterns."
                ),
                anchor_plate_id="ANCHOR_LOC_ROOFTOP_DOCK",
            ),
        },
        episodes=[
            EpisodeSpec(
                episode_id="EP01",
                episode_number=1,
                title="The Compass Core",
                target_runtime_seconds=1200,  # 20-minute episode structure
                scenes=[
                    SceneSpec(
                        scene_id="SC01_WORKSHOP",
                        scene_number=1,
                        title="Act I — The Ignition Test",
                        location_id="LOC_WORKSHOP",
                        shots=[
                            ScalableShotSpec(
                                shot_id="EP01_SC01_SH001",
                                shot_number=1,
                                duration_seconds=8,
                                character_ids=["CHAR_LEO", "CHAR_MAYA"],
                                wardrobe_state_by_character={"CHAR_LEO": "workshop", "CHAR_MAYA": "workshop"},
                                camera_framing_and_motion="Medium Two-Shot at drafting table, 35mm lens, slow dolly-in.",
                                action_beat_prompt=(
                                    "Leo taps the brass compass core on the table; it hums with a soft cyan pulse "
                                    "while Maya watches closely from screen-left."
                                ),
                                dialogue_line=SceneDialogueLine(
                                    shot_id="EP01_SC01_SH001",
                                    intended_speaker_id="CHAR_LEO",
                                    intended_speaker_visual_anchor="Leo in mustard-yellow apron on screen-right",
                                    visible_silent_character_anchors={
                                        "CHAR_MAYA": "Maya in burgundy bomber jacket on screen-left"
                                    },
                                    spoken_text="Hold on, the needle isn't pointing north—it's locking onto the dock!",
                                    delivery_mode=DeliveryMode.ON_CAMERA,
                                    emotional_direction="wide-eyed excitement and disbelief",
                                    shot_duration_sec=8.0,
                                ),
                            ),
                        ],
                    ),
                    SceneSpec(
                        scene_id="SC02_ROOFTOP",
                        scene_number=2,
                        title="Act I — Departure on the Pier",
                        location_id="LOC_ROOFTOP_DOCK",
                        shots=[
                            ScalableShotSpec(
                                shot_id="EP01_SC02_SH001",
                                shot_number=2,
                                duration_seconds=8,
                                character_ids=["CHAR_MAYA", "CHAR_LEO"],
                                wardrobe_state_by_character={"CHAR_MAYA": "expedition", "CHAR_LEO": "expedition"},
                                camera_framing_and_motion="Wide tracking shot along the lantern-lit pier, 28mm lens.",
                                action_beat_prompt=(
                                    "Maya steps onto the wooden pier in the evening wind, turning back toward Leo."
                                ),
                                dialogue_line=SceneDialogueLine(
                                    shot_id="EP01_SC02_SH001",
                                    intended_speaker_id="CHAR_MAYA",
                                    intended_speaker_visual_anchor="Maya in crimson windbreaker in foreground-left",
                                    visible_silent_character_anchors={
                                        "CHAR_LEO": "Leo in olive parka in background-right"
                                    },
                                    spoken_text="If that signal fades before sunrise, we lose the trail forever.",
                                    delivery_mode=DeliveryMode.ON_CAMERA,
                                    emotional_direction="urgent, determined focus",
                                    shot_duration_sec=8.0,
                                ),
                            ),
                        ],
                    ),
                ],
            )
        ],
    )


if __name__ == "__main__":
    project = build_starter_project()
    preflight = preflight_project_continuity(project, episode_id="EP01")
    compiled = compile_episode_shots(project, episode_id="EP01")
    print("=== PREFLIGHT REPORT (CASTING & DIALOGUE GATES) ===")
    print(json.dumps(preflight, indent=2))
    print("\n=== COMPILED SHOT PAYLOADS ===")
    print(json.dumps([c.model_dump() for c in compiled], indent=2))
