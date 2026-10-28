"""End-to-end tests for the AGY Studio v4 Architecture Rework:
- Tier-0 Generic Bible (7-layer / 4-layer prompt compiler, multi-shot commercial escalation, locked folder layout)
- Two-Layer Bible Hierarchy (GlobalContextBible + SceneBible) & Strict 3-Tier Reference Resolver (Scene -> Global -> AskUser)
- Gate 3 3D Set & Blender Export (.obj, .mtl, .gltf, open_in_blender.py) + 3D Screenshot-to-Keyframe Re-Anchoring
- Gate 4 Voice Discovery, Audition WAV synthesis & Global Voice Bible Lock
- Gate 5 Interactive HTML Video Review (index.html with videos, compiled prompts, and reference images)
- Gate 6 Scene Timeline Freeze/Backup + Searchable Rejected Video Vault (rejected_takes_vault.jsonl + .html)
"""

from __future__ import annotations

from pathlib import Path

from agy_studio import cli, media
from agy_studio.bible_hierarchy import (
    build_global_context_bible,
    build_scene_bible,
    load_global_context_bible,
    promote_scene_assets_to_global_library,
    resolve_reference,
)
from agy_studio.generic_bible import (
    COMMERCIAL_ESCALATION_TEMPLATE,
    ContentFormat,
    ControlMode,
    compile_generic_bible_prompt,
    export_shareable_bundle,
    validate_workspace_layout,
)
from agy_studio.models import (
    ApprovalStatus,
    AudioLine,
    AudioSpec,
    Conditioning,
    CreativeShot,
    DispatchStatus,
    Entity,
    EntityState,
    EntityType,
    Environment,
    GenClip,
    ProjectConfig,
    ProjectSettings,
    ReferencePack,
    Scene,
    SceneDialogueLine,
    ShotSpec,
    Take,
    TakeStatus,
    Tier,
    Turnaround,
    VisualSpec,
)
from agy_studio.project import Project
from agy_studio.review import build_g5_review
from agy_studio.set3d import designer
from agy_studio.set3d.blender_export import export_blender_scene, reanchor_shot_from_screenshot
from agy_studio.vault import freeze_and_backup_scene, index_rejected_take, search_rejected_takes
from agy_studio.voice_audition import lock_approved_voices_to_global_bible, prepare_scene_voices


def _seed_v4_project(tmp_path: Path) -> Project:
    ws = tmp_path / "ws"
    prods = Project.productions_dir(ws)
    cfg = ProjectConfig(
        project_id="TEST_V4",
        title="V4 Blueprint Test",
        language="hi",
        settings=ProjectSettings(fps=24, width=1080, height=1920, aspect_ratio="9:16"),
    )
    proj = Project.create(prods, "TEST_V4", "V4 Blueprint Test", language="hi", config=cfg)

    # Create a base character sheet in library/
    hero_sheet = proj.abs("library/packs/CHAR_HERO_STATE_DEFAULT/v1/front.png")
    media.make_test_png(hero_sheet, label="HERO_BASE")

    ent = Entity(
        id="CHAR_HERO",
        type=EntityType.CHARACTER,
        tier=Tier.PRINCIPAL,
        display_name="Rishi",
        canonical_description="24-year-old Indian man, athletic build, sharp jawline",
        approval_status=ApprovalStatus.LOCKED,
        states=[
            EntityState(
                id="CHAR_HERO_STATE_DEFAULT",
                delta_description="Casual grey hoodie",
                reference_pack=ReferencePack(turnaround=Turnaround(front=proj.rel(hero_sheet))),
                scenes_active=["SC01"],
                approval_status=ApprovalStatus.LOCKED,
            ),
            EntityState(
                id="CHAR_HERO_STATE_INJURED",
                derived_from="CHAR_HERO_STATE_DEFAULT",
                delta_description="Bruised cheek and torn sleeve",
                reference_pack=ReferencePack(),
                scenes_active=["SC02"],
                approval_status=ApprovalStatus.PROPOSED,
            ),
        ],
    )
    proj.save_entity(ent)

    env_img = proj.abs("library/environments/ENV_ROOM/v1/geometry.png")
    media.make_test_png(env_img, label="ENV_ROOM")
    env = Environment(
        id="ENV_ROOM",
        canonical_description="Compact Mumbai bedroom with desk and neon window",
        geometry_still=proj.rel(env_img),
        approval_status=ApprovalStatus.LOCKED,
    )
    proj.save_environment(env)

    sc1 = Scene(
        id="SC01",
        slug="INT. ROOM - NIGHT",
        order=1,
        environment_id="ENV_ROOM",
        entity_states_active=["CHAR_HERO_STATE_DEFAULT"],
        dialogue=[
            SceneDialogueLine(
                line_id="SC01_L001",
                character_id="CHAR_HERO",
                text="भाई, लाइव चालू है!",
                transliteration="Bhai, live chaalu hai!",
                language="Hindi",
            )
        ],
    )
    proj.save_scene(sc1)

    shot1 = CreativeShot(
        id="SC01_SH01",
        scene_id="SC01",
        order=1,
        intention="Hero reacts to the parcel on his desk",
        coverage_type="medium_close_up",
        duration_s=5.0,
        environment_angle="cam:WIDE_MASTER",
        lines=["SC01_L001"],
        spec=ShotSpec(
            visual=VisualSpec(
                subject_and_action="CHAR_HERO stares at the parcel on the desk",
                camera="steady push-in",
                lens_and_aspect="35mm anamorphic, 9:16",
                lighting="desk lamp key light and window rim",
                palette="deep slate and amber",
                composition="medium close-up",
                atmosphere="humid night air",
                mood="tense",
                attribution="directed by AGY",
            ),
            audio_spec=AudioSpec(
                dialogue=[
                    AudioLine(
                        character_id="CHAR_HERO",
                        text="भाई, लाइव चालू है!",
                        line_id="SC01_L001",
                        delivery="urgent whisper",
                        voice="baritone",
                        transliteration="Bhai, live chaalu hai!",
                    )
                ],
                ambience="quiet ceiling fan hum",
            ),
        ),
    )
    proj.save_shot(shot1)

    clip1 = GenClip(
        id="SC01_SH01_C01",
        shot_id="SC01_SH01",
        scene_id="SC01",
        sequence_index=1,
        duration_s=5.0,
        lines=["SC01_L001"],
        spec=shot1.spec,
        conditioning=Conditioning(
            reference_uris=[proj.rel(hero_sheet)],
            reference_entity_states=["CHAR_HERO_STATE_DEFAULT"],
            spatial_anchor=proj.rel(env_img),
        ),
    )
    proj.save_clip(clip1)
    return proj


def test_generic_bible_prompt_and_commercial_escalation() -> None:
    assert len(COMMERCIAL_ESCALATION_TEMPLATE) == 4
    assert COMMERCIAL_ESCALATION_TEMPLATE[0].start_s == 0.0
    assert COMMERCIAL_ESCALATION_TEMPLATE[-1].end_s == 15.0

    prompt_7 = compile_generic_bible_prompt(
        scene_and_theme="sleek titanium smartwatch on wet volcanic rock",
        subject_and_character="matte black chronograph with glowing amber hands",
        action_and_physics="water droplets bead and roll across the sapphire crystal",
        camera_and_framing="100mm macro lens, shallow depth of field",
        lighting_and_color="crisp rim light with warm tungsten reflections",
        audio_and_dialogue="subtle mechanical tick and raindrop impact",
        style_and_negatives="photorealistic commercial, no text overlays",
        beats=COMMERCIAL_ESCALATION_TEMPLATE,
        compact_4_layer=False,
    )
    assert "(0-3s)" in prompt_7
    assert "(3-7s)" in prompt_7
    assert "(7-12s)" in prompt_7
    assert "(12-15s)" in prompt_7
    assert "Hard cut" in prompt_7

    prompt_4 = compile_generic_bible_prompt(
        scene_and_theme="sleek titanium smartwatch",
        subject_and_character="chronograph",
        action_and_physics="water droplets roll across glass",
        camera_and_framing="100mm macro",
        compact_4_layer=True,
    )
    assert "100mm macro" in prompt_4
    assert ControlMode.DIRECTOR.value == "director"
    assert ContentFormat.COMMERCIAL_AD.value == "commercial_ad"


def test_locked_folder_layout_and_bundle_export(tmp_path: Path) -> None:
    ws_root = Path(__file__).resolve().parents[2]
    report = validate_workspace_layout(ws_root)
    assert report["valid"] is True, f"Missing shareable paths: {report['missing_required_paths']}"

    dest = tmp_path / "exported_bundle"
    res = export_shareable_bundle(ws_root, dest)
    assert res == dest.resolve()
    assert (dest / "FOLDER_STRUCTURE.md").is_file()
    assert (dest / "agy_studio" / "generic_bible.py").is_file()
    assert (dest / "productions" / ".gitkeep").is_file()


def test_two_layer_bible_hierarchy_and_three_tier_lookup(tmp_path: Path) -> None:
    proj = _seed_v4_project(tmp_path)
    gb = build_global_context_bible(proj)
    assert "CHAR_HERO" in gb.characters
    assert "ENV_ROOM" in gb.locations
    assert proj.abs("bible/GLOBAL_BIBLE.md").is_file()

    # Build SC01 SceneBible (CHAR_HERO_STATE_DEFAULT is locked in library -> reused)
    sb1 = build_scene_bible(proj, "SC01")
    assert "CHAR_HERO_STATE_DEFAULT" in sb1.reused_asset_ids

    # Tier 1 lookup: SceneBible hit
    r1 = resolve_reference(proj, "SC01", "CHAR_HERO")
    assert r1.source_tier == "scene_bible"
    assert r1.sheet_uri is not None

    # Create SC02 where CHAR_HERO_STATE_INJURED has no sheet yet -> falls back to Tier 2 GlobalContextBible base sheet!
    sc2 = Scene(
        id="SC02",
        slug="INT. ROOM - LATER",
        order=2,
        environment_id="ENV_ROOM",
        entity_states_active=["CHAR_HERO_STATE_INJURED"],
    )
    proj.save_scene(sc2)
    sb2 = build_scene_bible(proj, "SC02")
    # Clear the fallback sheet on SceneBible override to test pure Tier-2 fallback
    sb2.character_overrides["CHAR_HERO"].sheet_uri = None
    from agy_studio.bible_hierarchy import scene_bible_path
    from agy_studio.fsio import save_versioned

    save_versioned(scene_bible_path(proj, "SC02"), sb2)

    r2 = resolve_reference(proj, "SC02", "CHAR_HERO_STATE_INJURED")
    assert r2.source_tier == "global_bible"
    assert r2.sheet_uri == gb.characters["CHAR_HERO"].base_sheet_uri

    # Tier 3 lookup: completely unknown character -> opens a blocking question to ask the user
    r3 = resolve_reference(proj, "SC02", "CHAR_UNKNOWN_VILLAIN", ask_on_missing=True)
    assert r3.source_tier == "ask_user"
    assert r3.question_id is not None

    # Promote a newly generated Scene 2 sheet to Global Library without mutating Global Bible base sheet
    injured_sheet = proj.abs("library/packs/CHAR_HERO_STATE_INJURED/v1/front.png")
    media.make_test_png(injured_sheet, label="HERO_INJURED")
    sb2.character_overrides["CHAR_HERO"].sheet_uri = proj.rel(injured_sheet)
    save_versioned(scene_bible_path(proj, "SC02"), sb2)

    promoted = promote_scene_assets_to_global_library(proj, "SC02")
    assert "CHAR_HERO_STATE_INJURED" in promoted
    gb_after = load_global_context_bible(proj)
    assert gb_after.characters["CHAR_HERO"].base_sheet_uri == gb.characters["CHAR_HERO"].base_sheet_uri
    assert gb_after.reusable_states_library["CHAR_HERO_STATE_INJURED"] == proj.rel(injured_sheet)


def test_gate3_blender_export_and_screenshot_reanchor(tmp_path: Path) -> None:
    proj = _seed_v4_project(tmp_path)
    designer.build_set(proj, "ENV_ROOM", mode="llm", fake=True)
    b_res = export_blender_scene(proj, "ENV_ROOM")
    assert proj.abs(b_res["obj"]).is_file()
    assert proj.abs(b_res["mtl"]).is_file()
    assert proj.abs(b_res["gltf"]).is_file()
    assert proj.abs(b_res["blender_script"]).is_file()

    # User takes a screenshot in Blender / 3D viewer and re-anchors SC01_SH01
    user_screen = tmp_path / "blender_viewport.png"
    media.make_test_png(user_screen, label="BLENDER_SCREENSHOT")

    re_res = reanchor_shot_from_screenshot(
        proj, "SC01_SH01", user_screen, image_adapter=None, steer="low Dutch angle from desk corner"
    )
    assert proj.abs(re_res["board_frame"]).is_file()
    assert "SC01_SH01_C01" in re_res["stale_clips"]
    updated_shot = proj.load_shot("SC01_SH01")
    assert updated_shot.board_frame == re_res["board_frame"]


def test_gate4_voice_spec_lock_without_audio(tmp_path: Path) -> None:
    proj = _seed_v4_project(tmp_path)
    # First time CHAR_HERO speaks: a physical voice SPEC (text) is staged — no audition audio is synthesised,
    # because Omni/Veo generate the speech themselves from the per-line `voice` text.
    rep1 = prepare_scene_voices(proj, "SC01", fake=True)
    assert len(rep1["auditioned"]) == 1
    assert rep1["auto_pass_eligible"] is False
    spec = rep1["auditioned"][0]["description"]
    assert "Hz" in spec and "pace" in spec and "same voice in every shot" in spec
    assert "audition_wav" not in rep1["auditioned"][0]
    assert not list((proj.scene_dir("SC01") / "audio").rglob("*.wav")) if (proj.scene_dir("SC01") / "audio").exists() else True
    vp = proj.load_voice_profile("CHAR_HERO")
    assert vp is not None and vp.reference_audio is None and vp.anchor_wav is None  # anchor comes from a real take

    # Approve at Gate 4 -> locks the spec in the Global Bible (still no audio, no fake anchor)
    locked = lock_approved_voices_to_global_bible(proj, "SC01")
    assert "CHAR_HERO" in locked
    vp = proj.load_voice_profile("CHAR_HERO")
    assert vp is not None and vp.approval_status.value == "locked" and vp.anchor_wav is None
    assert not list(proj.path("library", "voices").rglob("*.wav")) if proj.path("library", "voices").exists() else True

    # Next scene with the same speaker reuses the locked spec automatically
    rep2 = prepare_scene_voices(proj, "SC01", fake=True)
    assert len(rep2["reused_from_global"]) == 1
    assert rep2["reused_from_global"][0]["description"] == spec
    assert len(rep2["auditioned"]) == 0
    assert rep2["auto_pass_eligible"] is True


def test_gate5_html_review_and_gate6_freeze_and_rejected_vault(tmp_path: Path) -> None:
    proj = _seed_v4_project(tmp_path)
    # Create 2 takes for SC01_SH01_C01: T01 (selected) and T02 (rejected)
    vid1 = proj.take_dir("SC01_SH01_C01_T01") / "video.mp4"
    vid2 = proj.take_dir("SC01_SH01_C01_T02") / "video.mp4"
    media.make_test_clip(vid1, duration_s=2.0)
    media.make_test_clip(vid2, duration_s=2.0)

    t1 = Take(
        id="SC01_SH01_C01_T01",
        clip_id="SC01_SH01_C01",
        model="omni",
        payload_hash="hash_t1",
        dispatch_status=DispatchStatus.COMPLETED,
        status=TakeStatus.SELECTED,
        video_uri=proj.rel(vid1),
    )
    t2 = Take(
        id="SC01_SH01_C01_T02",
        clip_id="SC01_SH01_C01",
        model="omni",
        payload_hash="hash_t2",
        dispatch_status=DispatchStatus.COMPLETED,
        status=TakeStatus.REJECTED,
        video_uri=proj.rel(vid2),
    )
    proj.save_take(t1)
    proj.save_take(t2)

    clip = proj.load_clip("SC01_SH01_C01")
    clip.takes = [t1.id, t2.id]
    clip.selected_take_id = t1.id
    proj.save_clip(clip)

    # Gate 5 review builds index.html showing video takes, prompts, and reference images
    rev_dir = build_g5_review(proj, "SC01", ["SC01_SH01_C01"])
    html_file = rev_dir / "index.html"
    assert html_file.is_file()
    html_text = html_file.read_text(encoding="utf-8")
    assert "SC01_SH01_C01" in html_text
    assert "CHAR_HERO stares at the parcel" in html_text

    # Index rejected take & freeze scene at Gate 6
    index_rejected_take(proj, t2, rejection_source="user_g5", reason="lip sync slightly late")
    frozen = freeze_and_backup_scene(proj, "SC01", decision_id="DEC_G6_TEST")
    assert proj.abs(frozen["frozen_manifest"]).is_file()
    assert proj.abs(frozen["backup_otio"]).is_file()
    assert proj.abs(frozen["exported_scene_mp4"]).is_file()
    assert proj.abs("library/rejected_takes_vault.html").is_file()

    # Search the Rejected Video Vault by prompt / reason / character
    hits = search_rejected_takes(proj, "parcel", character_id="CHAR_HERO", scene_id="SC01")
    assert any(h.take_id == "SC01_SH01_C01_T02" for h in hits)

    # Verify CLI commands for mode and search-takes
    out_mode = cli.run(["--workspace", str(proj.root.parents[1]), "--project", "TEST_V4", "mode", "hybrid", "--format", "commercial_ad"], fake=True)
    assert "hybrid" in out_mode
    out_search = cli.run(["--workspace", str(proj.root.parents[1]), "--project", "TEST_V4", "search-takes", "parcel"], fake=True)
    assert "SC01_SH01_C01_T02" in out_search
