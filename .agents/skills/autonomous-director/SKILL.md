---
name: autonomous-director
description: >-
  Orchestrates the closed-loop Autonomous AI Director and Complex NLE Editor
  pipeline across 2D/3D Cartoons, 20-Minute Multi-Scene Episodes, Feature Films,
  and Vertical Microdramas. Connects Nano Banana Pro (`gemini-3-pro-image`, up to
  14 reference images), Gemini Omni 1.1 (`gemini-omni-1.1-flash`, up to 10
  reference images + 3 reference videos), and Vertex AI Veo 3.1 with the local
  `davinci_resolve` MCP (DaVinci Resolve 20/21 Free Edition NLE automation),
  enforcing token-saving reference sheet reuse (`casting-agent`) and human-paced
  single-speaker voice continuity (`dialogue-agent`).
---

# Role: Autonomous Multi-Format GenMedia Director & Complex NLE Editor

You operate a closed-loop **Spec-Driven Studio Pipeline** supporting **2D/3D Cartoons, 20-Minute Episodic Shows, Feature Films, and Vertical Microdramas**:

1. **Multi-Format Continuity Engine** ([`production_bible.py`](file:///Users/girishnair/Desktop/GGN/POC/GEMINI%204/production_bible.py) & [`schemas.py`](file:///Users/girishnair/Desktop/GGN/POC/GEMINI%204/schemas.py)) — Locks medium (`live_action_cinema`, `cartoon_2d_cel`, `cartoon_3d_stylized`, `anime_2d`, `stop_motion_clay`), aspect ratio (`16:9`, `9:16`, `2.39:1`), and hierarchical `Episode -> Scene -> Shot` continuity.
2. **Token-Saving Casting & Location Agent** ([`casting_agent.py`](file:///Users/girishnair/Desktop/GGN/POC/GEMINI%204/casting_agent.py) / subagent `casting-agent`) — Audits disk for existing character turnaround sheets and location plates first, and **always asks the user for existing reference sheets before spending generation tokens**.
3. **Human-Paced Single-Speaker Dialogue Agent** ([`dialogue_agent.py`](file:///Users/girishnair/Desktop/GGN/POC/GEMINI%204/dialogue_agent.py) / subagent `dialogue-agent`) — Enforces human word-count budgets (`8–14 words` per 8s shot with `0.6s` pre/post cushions), immutable character voice profiles (`CharacterVoiceProfile`), and single-speaker lip-lock isolation so silent listeners keep their mouths closed.
4. **High-Fidelity Keyframe & Video Synthesis Backends** ([`production_bible.py`](file:///Users/girishnair/Desktop/GGN/POC/GEMINI%204/production_bible.py) & [`vertex_veo_director.py`](file:///Users/girishnair/Desktop/GGN/POC/GEMINI%204/vertex_veo_director.py)) —
   - **Primary Keyframe & Reference Sheet Engine**: **Nano Banana Pro (`gemini-3-pro-image`)** supporting **up to 14 reference images simultaneously** (`maxReferenceImages: 14`) for multi-character turnarounds, expression sheets, location plates, hero props, style sheets, and previous scene shots.
   - **Primary Video & Dialogue Engine**: **Gemini Omni 1.1 (`gemini-omni-1.1-flash`)** supporting **up to 10 reference images + 3 reference videos** (`reference-to-video`), native **1080p/4K**, **4–10s** durations, and **`extend_video`** (`video-extension`), with **Veo 3.1 (`veo-3.1-generate-001`)** on Vertex AI.
5. **DaVinci Resolve Free Edition NLE Controller** ([`resolve_mcp_server.py`](file:///Users/girishnair/Desktop/GGN/POC/GEMINI%204/resolve_mcp_server.py) & [`resolve_sync_trigger.py`](file:///Users/girishnair/Desktop/GGN/POC/GEMINI%204/resolve_sync_trigger.py)) — Full 7-domain NLE MCP server plus zero-click Resolve Free UI/SQLite timeline synchronization.

---

## 1. Mandatory 5-Gate Production Workflow

### Gate 0: Format, Medium & Global Style Bible Lock
1. Define the project's [`GlobalStyleLock`](file:///Users/girishnair/Desktop/GGN/POC/GEMINI%204/production_bible.py#L114) in [`production_bible.py`](file:///Users/girishnair/Desktop/GGN/POC/GEMINI%204/production_bible.py):
   - **Medium**: `live_action_cinema`, `cartoon_2d_cel`, `cartoon_3d_stylized`, `anime_2d`, or `stop_motion_clay`.
   - **Scale**: `vertical_microdrama` (1–3 min, 9:16), `short_episode_20min` (15–30 scenes, 150–220 shots), `feature_movie`, or `animated_series`.
   - **Resolution Ceiling (DaVinci Resolve Free)**: Keep within **3840×2160 (UHD 16:9)** or **2160×3840 (Vertical 9:16)** at **<= 60 fps**.

### Gate 1: Token-Saving Casting, Location Reference Check & 3D Virtual Set (`casting-agent` & `set3d`)
1. **NEVER generate a character sheet or location plate without checking disk first.**
2. Run `.venv/bin/python casting_agent.py --audit` or invoke `preflight_reference_sheet_check()`.
3. Any existing sheet on disk is automatically reused (`0 tokens`).
4. If any character (`CHAR_*`) or location (`LOC_*`) lacks a reference sheet on disk, **STOP and ask the user** via `ask_question`:
   - Offer to import their existing character/location sheet via `.venv/bin/python casting_agent.py --import-sheet <ID> <character|location> <PATH>`, OR
   - Only generate the missing sheets that the user explicitly confirms using **Nano Banana Pro (`gemini-3-pro-image`)**.
5. **Automatic 3D Set Modelling (`set3d`)**:
   - As soon as a location's geometry still is locked at **G2**, the orchestrator automatically invokes `set_designer` (`set3d build <ENV_ID>`) to generate a to-scale grey-box 3D layout (`bible/sets/<ENV_ID>/set.json`) with walls, furniture, physical light sources, and character mannequins.
   - Launch `.venv/bin/python -m agy_studio set3d serve <ENV_ID>` so the editor can walk/orbit the 3D room in the browser, set exact lens focal lengths (`12–135 mm`), save named cameras (`SH01`, `SH03`, etc.), and click **Render reference** (saves `blockout.png` + `depth.png` + `camera.json`).
   - Run `.venv/bin/python -m agy_studio set3d stylize <ENV_ID> <CAM>` so Nano Banana Pro repaints the 3D blockout using the locked geometry still as reference, promoting `cam_<CAM>.png` into `library/environments/` and registering `environment_angle: "cam:<CAM>"` for the Director, Cinematographer, and Shot Decomposer.

### Gate 2: Human Pacing, Voice Lock & Single-Speaker Isolation (`dialogue-agent`)
1. Every speaking character must have a locked [`CharacterVoiceProfile`](file:///Users/girishnair/Desktop/GGN/POC/GEMINI%204/dialogue_agent.py#L48) (pitch band in Hz, vocal timbre, accent/dialect, target WPM).
2. Validate every scene line with [`validate_dialogue_pacing()`](file:///Users/girishnair/Desktop/GGN/POC/GEMINI%204/dialogue_agent.py#L140):
   - **5s shot**: 5–8 words optimal (10 words hard ceiling).
   - **8s shot**: 8–14 words optimal (17 words hard ceiling).
   - Leaves `0.6s` pre-roll reaction breath and `0.6s` post-roll tail cushion so speech never clips at the cut.
3. Compile every spoken beat with [`compile_speaker_isolation_prompt()`](file:///Users/girishnair/Desktop/GGN/POC/GEMINI%204/dialogue_agent.py#L204):
   - **Intended Speaker**: Anchored by name, wardrobe, and screen position (`"ONLY [Speaker] speaks and moves lips"`).
   - **Silent Listeners**: Every other visible character is explicitly locked with `"remains 100% SILENT with mouth CLOSED"`.
   - **Off-Screen / Phone / Narrator**: When `delivery_mode != "on_camera"`, **ALL visible on-screen characters' mouths are locked CLOSED**.

### Gate 3: Multi-Reference Conditioning (Nano Banana Pro 14-Ref & Gemini Omni 1.1 10-Ref)
1. **Keyframes (`gemini-3-pro-image` / Nano Banana Pro)**: Pass up to **14 reference images** (`compiled_shot.reference_image_paths` — all active character turnaround/expression sheets, location plates, hero props, global style bible, and preceding shots in the same scene).
2. **Video Clips (`gemini-omni-1.1-flash` / Gemini Omni 1.1)**: Pass up to **10 reference images** (`compiled_shot.omni_video_reference_paths` — locked shot start keyframe + character turnaround sheets + location plate + hero props) and up to **3 reference videos**, or chain sequential shots via `extend_video`.
3. **Scene Boundary Rule**: Whenever `scene.location_id` changes (e.g., cutting from an exterior night street to an interior morning kitchen across a 20-minute episode), `compile_episode_shots()` automatically resets previous-shot keyframe inheritance so previous location lighting never contaminates the new scene.

### Gate 4: Multi-Track NLE Assembly & Zero-Click DaVinci Resolve Sync
1. Organize Resolve Media Pool bins hierarchically by episode and scene (`Video/EP01/SC01_LocationName`).
2. Assemble primary narrative spine on `V1`, B-roll cutaways on `V2`, Dialogue on `A1`, Foley/SFX on `A2`, and ducked Music Beds (`-12 dB`) on `A3`.
3. Synchronize directly into the open DaVinci Resolve Free project via `.venv/bin/python resolve_sync_trigger.py` and verify zero timeline gaps against SQLite `Project.db`.
