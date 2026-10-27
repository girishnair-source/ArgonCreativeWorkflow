"""Deterministic demo project builder (no network) shared by all tests and offline demos.

``build_demo_project`` creates a complete, consistent mini production under ``<productions_dir>/<project_id>``:

* locked StyleLock with visual + audio global negatives,
* characters CHAR_RISHI (principal, DEFAULT + INJURED states), CHAR_MOHNISH (supporting), CHAR_GUARD
  (incidental, off-screen voice only in SC01) and prop PROP_PHONE, each with a reference pack in ``library/``,
* locked voice profiles VOICE_CHAR_RISHI / VOICE_CHAR_MOHNISH / VOICE_CHAR_GUARD (Hindi, Devanagari),
* environments ENV_TERRACE and ENV_STAIRWELL with geometry + derived angles + lighting + room tone,
* SC01 "terrace_dare" with Devanagari dialogue (one off-screen line), a mid-scene transition
  (CHAR_RISHI DEFAULT -> INJURED at SC01_SH02 clip 2) and two CreativeShots; SC02 "stairwell_escape".

With ``lock=False`` every bible record stays ``proposed`` and packs point into ``scenes/SC01/design/``.
"""

from __future__ import annotations

from pathlib import Path

from .fsio import atomic_write_yaml, generated_by
from .media import write_png
from .models import (
    ApprovalStatus,
    AudioSpec,
    Beat,
    CreativeShot,
    DerivedAngles,
    Entity,
    EntityState,
    EntityType,
    Environment,
    LightingSetup,
    MidSceneTransition,
    ReferencePack,
    Scene,
    SceneDialogueLine,
    ShotSpec,
    StyleLock,
    Tier,
    Turnaround,
    VisualSpec,
    VoiceProfile,
)
from .project import Project

DEMO_PROJECT_ID = "DEMO_S01"

_FULL = {"front": (200, 60, 60), "three_quarter": (190, 70, 70), "profile": (180, 80, 80),
         "rear": (170, 90, 90), "face_closeup": (220, 120, 100)}
_SUPPORTING = {"front": (60, 60, 200), "three_quarter": (70, 70, 190), "face_closeup": (100, 120, 220)}
_SINGLE = {"single_ref": (90, 160, 90)}


def _pack(project: Project, state_id: str, roles: dict[str, tuple[int, int, int]], *, lock: bool,
          shade: int = 0) -> ReferencePack:
    design = project.path("scenes", "SC01", "design")
    files = {}
    for role, rgb in roles.items():
        color = tuple(max(0, min(255, c + shade)) for c in rgb)
        files[role] = write_png(design / f"{state_id}_{role}.png", 64, 96, color)  # type: ignore[arg-type]
    if lock:
        return project.promote_pack(state_id, files, approved_by="director",
                                    provenance=generated_by("agy.testing.demo", model="synthetic"))
    rel = {role: project.rel(p) for role, p in files.items()}
    return ReferencePack(
        turnaround=Turnaround(front=rel.get("front"), three_quarter=rel.get("three_quarter"),
                              profile=rel.get("profile"), rear=rel.get("rear")),
        face_closeup=rel.get("face_closeup"), single_ref=rel.get("single_ref"),
        generated_by=generated_by("agy.testing.demo", model="synthetic"),
    )


def _environment(project: Project, env_id: str, description: str, axis: str, room_tone: str,
                 lighting: LightingSetup, *, lock: bool, base: tuple[int, int, int]) -> Environment:
    design = project.path("scenes", "SC01", "design")
    names = ("geometry", "reverse", "angle_45L", "angle_45R", "wide_establishing")
    files = {f"{n}.png": write_png(design / f"{env_id}_{n}.png", 96, 64,
                                   (base[0], min(255, base[1] + 10 * i), base[2])) for i, n in enumerate(names)}
    status = ApprovalStatus.LOCKED if lock else ApprovalStatus.PROPOSED
    if lock:
        rels = project.promote_library_files(("environments", env_id), files,
                                             provenance=generated_by("agy.testing.demo", model="synthetic"))
    else:
        rels = {name: project.rel(p) for name, p in files.items()}
    return Environment(
        id=env_id,
        canonical_description=description,
        geometry_candidates=[rels["geometry.png"]],
        geometry_still=rels["geometry.png"],
        axis_of_action=axis,
        derived_angles=DerivedAngles(reverse=rels["reverse.png"], angle_45L=rels["angle_45L.png"],
                                     angle_45R=rels["angle_45R.png"], wide_establishing=rels["wide_establishing.png"]),
        lighting_setups=[lighting],
        room_tone=room_tone,
        approval_status=status,
        generated_by=generated_by("agy.testing.demo"),
    )


def build_demo_project(productions_dir: Path, project_id: str = DEMO_PROJECT_ID, *, lock: bool = True,
                       with_shots: bool = True) -> Project:
    project = Project.create(Path(productions_dir), project_id, "Demo — Terrace Dare", language="hi")
    status = ApprovalStatus.LOCKED if lock else ApprovalStatus.PROPOSED
    prov = generated_by("agy.testing.demo")

    project.save_style_lock(StyleLock(
        medium="live-action digital video, natural skin texture, gritty street realism",
        aspect_ratio="9:16",
        lens_language="35mm and 50mm primes, shallow depth of field, handheld micro-movement",
        lut_or_grade_ref="teal-and-sodium night grade",
        reference_films=["Gangs of Wasseypur", "Ugly (2013)"],
        donts=["glossy CGI skin", "text overlays", "lens flares on every shot"],
        global_negative=["background music", "score", "narration", "subtitles", "on-screen text", "echo",
                         "robotic voice", "extra fingers", "warped limbs"],
        approval_status=status, generated_by=prov,
    ))

    # ---- entities --------------------------------------------------------------------------
    rishi_default = EntityState(id="CHAR_RISHI_STATE_DEFAULT", delta_description="baseline look",
                                reference_pack=_pack(project, "CHAR_RISHI_STATE_DEFAULT", _FULL, lock=lock),
                                scenes_active=["SC01"], approval_status=status)
    rishi_injured = EntityState(id="CHAR_RISHI_STATE_INJURED", derived_from="CHAR_RISHI_STATE_DEFAULT",
                                delta_description="fresh bleeding scrape on the left forearm, torn left flannel sleeve",
                                state_specific_negative=["clean unhurt forearms", "intact sleeves"],
                                reference_pack=_pack(project, "CHAR_RISHI_STATE_INJURED", _FULL, lock=lock, shade=-30),
                                scenes_active=["SC01", "SC02"], approval_status=status)
    project.save_entity(Entity(
        id="CHAR_RISHI", type=EntityType.CHARACTER, tier=Tier.PRINCIPAL, display_name="Rishi",
        canonical_description="Indian man, 23, lean build, sharp jaw with light stubble, messy black hair, "
                              "open royal-blue plaid flannel over a charcoal t-shirt, black wristband on right wrist",
        voice_profile_id="VOICE_CHAR_RISHI", states=[rishi_default, rishi_injured],
        approval_status=status, generated_by=prov,
    ))
    project.save_entity(Entity(
        id="CHAR_MOHNISH", type=EntityType.CHARACTER, tier=Tier.SUPPORTING, display_name="Mohnish",
        canonical_description="Indian man, 22, round face, thick-rimmed glasses, curly hair, mustard hoodie",
        voice_profile_id="VOICE_CHAR_MOHNISH",
        states=[EntityState(id="CHAR_MOHNISH_STATE_DEFAULT", delta_description="baseline look",
                            reference_pack=_pack(project, "CHAR_MOHNISH_STATE_DEFAULT", _SUPPORTING, lock=lock),
                            scenes_active=["SC01", "SC02"], approval_status=status)],
        approval_status=status, generated_by=prov,
    ))
    project.save_entity(Entity(
        id="CHAR_GUARD", type=EntityType.CHARACTER, tier=Tier.INCIDENTAL, display_name="Security Guard",
        canonical_description="Indian man, 55, grey moustache, khaki security uniform, peaked cap",
        voice_profile_id="VOICE_CHAR_GUARD",
        states=[EntityState(id="CHAR_GUARD_STATE_DEFAULT", delta_description="baseline look",
                            reference_pack=_pack(project, "CHAR_GUARD_STATE_DEFAULT", _SINGLE, lock=lock),
                            scenes_active=["SC01"], approval_status=status)],
        approval_status=status, generated_by=prov,
    ))
    project.save_entity(Entity(
        id="PROP_PHONE", type=EntityType.PROP, tier=Tier.INCIDENTAL, display_name="Mohnish's phone",
        canonical_description="black smartphone in a cracked clear case, live-stream app open",
        states=[EntityState(id="PROP_PHONE_STATE_DEFAULT", delta_description="screen on",
                            reference_pack=_pack(project, "PROP_PHONE_STATE_DEFAULT", _SINGLE, lock=lock, shade=40),
                            scenes_active=["SC01"], approval_status=status)],
        approval_status=status, generated_by=prov,
    ))

    # ---- voices ----------------------------------------------------------------------------
    hindi = "Hindi (colloquial Mumbai), Devanagari script, natural native Indian articulation, zero English accent"
    for vp in (
        VoiceProfile(id="VOICE_CHAR_RISHI", character_id="CHAR_RISHI", language=hindi,
                     description="male, early 20s, warm light baritone, quick confident cadence, playful bravado",
                     state_variants={"CHAR_RISHI_STATE_INJURED": "strained, shallow breaths, words clipped by pain"},
                     approval_status=status, generated_by=prov),
        VoiceProfile(id="VOICE_CHAR_MOHNISH", character_id="CHAR_MOHNISH", language=hindi,
                     description="male, early 20s, higher nasal tenor, fast nervous cadence, anxious giggle",
                     approval_status=status, generated_by=prov),
        VoiceProfile(id="VOICE_CHAR_GUARD", character_id="CHAR_GUARD", language=hindi,
                     description="male, mid 50s, gruff rasping bass, slow authoritative cadence, shouting",
                     approval_status=status, generated_by=prov),
    ):
        project.save_voice_profile(vp)

    # ---- environments ----------------------------------------------------------------------
    project.save_environment(_environment(
        project, "ENV_TERRACE",
        "Flat concrete rooftop terrace of a six-storey Mumbai chawl at night: waist-high parapet wall on the "
        "east edge, rusted water tanks, tangled cable bundles, laundry lines",
        "CHAR_RISHI on the parapet (frame left) faces CHAR_MOHNISH (frame right); camera stays south of the line",
        "night wind gusting across the rooftop, distant traffic hum, faint dog barks far below",
        LightingSetup(id="LS_NIGHT_SODIUM", description="sodium-vapour street lamp below the parapet as hard key "
                      "from frame left; phone screen as soft fill on faces; distant windows as practical backlight"),
        lock=lock, base=(70, 80, 90)))
    project.save_environment(_environment(
        project, "ENV_STAIRWELL",
        "Narrow concrete stairwell of the same chawl: chipped green railings, peeling posters, a single caged bulb",
        "descending flight runs top-left to bottom-right; camera stays on the railing side",
        "hollow concrete echo of footsteps, buzzing bulb, muffled television from a flat",
        LightingSetup(id="LS_STAIR_BULB", description="single caged tungsten bulb on the landing ceiling as key; "
                      "spill from an open flat door as edge light"),
        lock=lock, base=(60, 90, 70)))

    # ---- scenes ----------------------------------------------------------------------------
    sc01 = Scene(
        id="SC01", slug="terrace_dare", order=1, episode=1,
        script_text="EXT. CHAWL TERRACE - NIGHT\nMohnish streams live as Rishi mounts the parapet...",
        beats=[
            Beat(beat_id="SC01_B1", summary="Mohnish starts the live stream; Rishi climbs the parapet",
                 performance_states={"CHAR_RISHI": "cocky", "CHAR_MOHNISH": "nervous excitement"}),
            Beat(beat_id="SC01_B2", summary="Rishi plays to the camera until the guard's shout startles him",
                 performance_states={"CHAR_RISHI": "showboating then startled", "CHAR_MOHNISH": "panicked"}),
        ],
        performance_states={"CHAR_RISHI": "cocky", "CHAR_MOHNISH": "nervous excitement"},
        entity_states_active=["CHAR_RISHI_STATE_DEFAULT", "CHAR_MOHNISH_STATE_DEFAULT", "PROP_PHONE_STATE_DEFAULT"],
        environment_id="ENV_TERRACE", lighting_setup_id="LS_NIGHT_SODIUM", time_of_day="night",
        mid_scene_transitions=[MidSceneTransition(
            entity_id="CHAR_RISHI", from_state="CHAR_RISHI_STATE_DEFAULT", to_state="CHAR_RISHI_STATE_INJURED",
            at_shot_id="SC01_SH02", at_clip_index=2, note="scrapes his forearm on the parapet as he stumbles")],
        dialogue=[
            SceneDialogueLine(line_id="SC01_L001", character_id="CHAR_MOHNISH", parenthetical="whispering, excited",
                              text="भाई, लाइव चालू है... दो हज़ार लोग देख रहे हैं!",
                              transliteration="Bhai, live chaalu hai... do hazaar log dekh rahe hain!", language="hi"),
            SceneDialogueLine(line_id="SC01_L002", character_id="CHAR_RISHI", parenthetical="grinning at the lens",
                              text="देख, अब असली खेल शुरू होगा।",
                              transliteration="Dekh, ab asli khel shuru hoga.", language="hi"),
            SceneDialogueLine(line_id="SC01_L003", character_id="CHAR_GUARD", parenthetical="shouting from the stairwell",
                              text="कौन है वहाँ? नीचे उतरो!", transliteration="Kaun hai wahan? Neeche utro!",
                              language="hi", off_screen=True),
        ],
        exit_state={"CHAR_RISHI": "CHAR_RISHI_STATE_INJURED", "CHAR_MOHNISH": "CHAR_MOHNISH_STATE_DEFAULT",
                    "position": "both running toward the stairwell door"},
        generated_by=prov,
    )
    sc02 = Scene(
        id="SC02", slug="stairwell_escape", order=2, episode=1,
        script_text="INT. CHAWL STAIRWELL - NIGHT\nRishi and Mohnish race down the stairs...",
        beats=[Beat(beat_id="SC02_B1", summary="The boys flee down the stairwell",
                    performance_states={"CHAR_RISHI": "pained urgency", "CHAR_MOHNISH": "terrified"})],
        entity_states_active=["CHAR_RISHI_STATE_INJURED", "CHAR_MOHNISH_STATE_DEFAULT"],
        environment_id="ENV_STAIRWELL", lighting_setup_id="LS_STAIR_BULB", time_of_day="night",
        dialogue=[SceneDialogueLine(line_id="SC02_L001", character_id="CHAR_RISHI", parenthetical="breathless",
                                    text="भाग! सीढ़ियों से नीचे!", transliteration="Bhaag! Seedhiyon se neeche!",
                                    language="hi")],
        entry_state={"CHAR_RISHI": "CHAR_RISHI_STATE_INJURED"},
        generated_by=prov,
    )

    if with_shots:
        ambience = "night wind gusting across the rooftop, distant traffic hum, faint dog barks far below"
        sh1 = CreativeShot(
            id="SC01_SH01", scene_id="SC01", order=1,
            intention="Establish the dare: Rishi mounts the parapet while Mohnish streams",
            coverage_type="wide", duration_s=6.0, environment_angle="wide_establishing", camera_side="south",
            beat_ids=["SC01_B1"], actions=["CHAR_RISHI climbs onto the parapet ledge"],
            lines=["SC01_L001"], line_action_index={"SC01_L001": 0},
            spec=ShotSpec(
                visual=VisualSpec(
                    subject_and_action="CHAR_RISHI climbs onto the parapet ledge",
                    camera="static wide shot from the south side of the terrace with a slow 5% push-in",
                    lens_and_aspect="24mm lens, 9:16 vertical frame",
                    lighting="sodium-vapour street lamp below the parapet as hard key from frame left; "
                             "phone screen as soft fill",
                    palette="desaturated concrete greys, deep teal shadows, warm skin tones",
                    composition="CHAR_RISHI on the parapet in the upper-left third; CHAR_MOHNISH holding "
                                "PROP_PHONE in the lower-right foreground",
                    atmosphere="light haze, gusting wind moving the laundry lines",
                    mood="reckless thrill with rising dread",
                    attribution="gritty Mumbai street-thriller realism",
                ),
                audio_spec=AudioSpec(ambience=ambience, diegetic_sfx=["shoe scuff on concrete", "wind gusts"]),
            ),
        )
        sh2 = CreativeShot(
            id="SC01_SH02", scene_id="SC01", order=2,
            intention="Rishi's bravado breaks when the guard's shout startles him",
            coverage_type="over-the-shoulder reverse", duration_s=8.0, environment_angle="reverse",
            camera_side="south", beat_ids=["SC01_B2"],
            actions=["CHAR_RISHI raises both arms toward the phone camera", "CHAR_RISHI stumbles backward off the ledge"],
            lines=["SC01_L002", "SC01_L003"], line_action_index={"SC01_L002": 0, "SC01_L003": 1},
            spec=ShotSpec(
                visual=VisualSpec(
                    subject_and_action="CHAR_RISHI raises both arms toward the phone camera",
                    camera="over-the-shoulder from behind CHAR_MOHNISH, handheld",
                    lens_and_aspect="50mm lens, 9:16 vertical frame",
                    lighting="sodium-vapour street lamp from below as hard key; phone screen as fill",
                    palette="desaturated greys, teal shadows, warm skin tones",
                    composition="CHAR_RISHI centred on the parapet, CHAR_MOHNISH's shoulder in the left foreground",
                    atmosphere="wind-blown haze",
                    mood="bravado tipping into panic",
                    attribution="gritty Mumbai street-thriller realism",
                ),
                audio_spec=AudioSpec(ambience=ambience, diegetic_sfx=["wind gusts", "sharp scrape on concrete"]),
            ),
        )
        project.save_shot(sh1)
        project.save_shot(sh2)
        sc01.creative_shots = [sh1.id, sh2.id]

    project.save_scene(sc01)
    project.save_scene(sc02)
    atomic_write_yaml(project.path("bible", "state_timeline.yaml"), {
        "entries": [
            {"entity_id": "CHAR_RISHI", "state_id": "CHAR_RISHI_STATE_DEFAULT", "from": "SC01", "until": "SC01_SH02_C02"},
            {"entity_id": "CHAR_RISHI", "state_id": "CHAR_RISHI_STATE_INJURED", "from": "SC01_SH02_C02", "until": None},
            {"entity_id": "CHAR_MOHNISH", "state_id": "CHAR_MOHNISH_STATE_DEFAULT", "from": "SC01", "until": None},
        ],
        "generated_by": prov.model_dump(mode="json"),
    })
    return project
