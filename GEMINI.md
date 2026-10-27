# Zero-Touch Creative Editor Mode (Mandatory Workspace Rules)

The user operating this workspace is a **Creative Video Editor / Director**, NOT a software engineer.
Follow these rules strictly on every turn:

1. **NEVER Ask the Editor to Run Python, Virtual Environments, Git, or Terminal Commands**:
   - You (Antigravity) must execute all terminal commands, Git clones/pulls, Python scripts, environment setups, and DaVinci Resolve sync triggers yourself using `run_command`.
   - If the editor starts with a GitHub source URL (e.g., *"Start a new production in AGY Studio using this source : https://github.com/girishnair-source/ArgonCreativeWorkflow.git"*) and `agy_studio/` is not yet present in the current working directory, automatically clone or sync that repository into the workspace first via `run_command`.
   - If `.venv`, `ffmpeg`, or `gcloud` are missing or unauthenticated, automatically run `bash setup_studio_env.sh` via `run_command`. If Google Cloud needs authentication, run `gcloud auth login` and `gcloud auth application-default login` via `run_command` so the Google sign-in browser window pops open automatically for the editor, or tell them the single command `gcloud auth login` only if needed.

2. **Interactive Creative Intake — Mandatory Gate at Project Start (Use `ask_question` in Plain Filmmaking Language)**:
   - Whenever the editor starts a new project (whether via *"Start a new production in AGY Studio using this source : https://github.com/girishnair-source/ArgonCreativeWorkflow.git"* or *"Start a new production in AGY Studio and guide me step by step from script to Final Edited Video"*), first silently verify the environment (`live_demo_preflight.py` / `setup_studio_env.sh`), and **MUST FIRST** ask via `ask_question` to lock the **Operating Mode**, **Format**, **Language**, **Production Scope**, **3D Blocking Engine**, and **Reference Sheet Reuse**, and prompt the editor for their **Project Title, Project ID (optional — auto-derived from title if omitted), and Script / Story Brief**:
     1. **Operating Mode (The 3 Studio Modes — `ControlMode`)**:
        - **Mode 1 — `director` (Ultimate Director Control)**: Human review and explicit approval at every single gate (`G1` Global Bible $\rightarrow$ `G2` Character/Location Sheets & 3D Set $\rightarrow$ `G3` Storyboard & 3D Camera Angles $\rightarrow$ `G4` Voice Auditions $\rightarrow$ `G5` Video Takes $\rightarrow$ `G6` DaVinci Resolve Scene Cut).
        - **Mode 2 — `hybrid` (Smart Hybrid Control)**: Stops at core creative gates (`G1` Global Bible, `G3` Storyboard/3D Framing, `G5` Video Takes, `G6` Resolve Cut), while auto-passing `G2` and `G4` whenever approved Global Bible character sheets, sets, or locked voice prints are reused.
        - **Mode 3 — `autopilot` (Seamless End-to-End Autopilot)**: Approve `G1` (Global Bible) once, and AGY Studio runs `G2` through `G5` autonomously with Critic QA self-healing (`threshold >= 0.85`), pausing only at `G6` when the assembled DaVinci Resolve timeline is ready.
     2. **Content Format & Aspect Ratio (`ContentFormat`)**:
        - Ask whether the project is a **Vertical Microdrama (`microdrama_9_16`, `9:16`)**, **20-Minute Episode (`episode_16_9`, `16:9`)**, **Feature Film (`feature_film`, `16:9` / `2.39:1`)**, **Multi-Shot Commercial / Marketing Spot (`marketing_spot`, time-coded beat escalation)**, or **2D/3D Cartoon**.
     3. **Primary Dialogue Language, Production Scope & Script Intake**:
        - Ask for the **Primary Language** (e.g., `Hindi / Hinglish`, `English`, `Tamil / Telugu / Regional`) via `ask_question`.
        - Ask for the **Production Scope** via `ask_question`:
          - `(Recommended for Live Demo) Pilot Scene 1 Only (SC01 — 3 to 4 shots, fast 8-min end-to-end to Final Edited Video & DaVinci Resolve)`
          - `Full Script (All Scenes in the Screenplay)`
        - If the user has not yet shared their **Project Title**, **Project ID** (auto-generate a clean ID like `AUDIENCE_EP01` if not provided), or **Script / Story Brief**, ask them in plain English to paste their script or story beats right after submitting the intake modal.
     4. **3D Camera Blocking Engine & Token-Saving Reference Sheet Gate (`casting_agent.py` & `set3d`)**:
        - Ask whether the director prefers **Zero-Blender 60 FPS Browser 3D FPV (`set3d` Procedural Three.js + GLSL)**, **Native Blender 5.2 (`.blend`)**, or **Both**.
        - Before generating ANY character or location images, ALWAYS ask:
          *"Do you already have Character Reference Sheets or Location Reference Images for these characters/settings? (If yes, drop them into this folder or share their filenames so we reuse them and save tokens; otherwise I will generate new ones for you.)"*
        - Once locations and characters are locked at **G2**, automatically build the 1:1 3D set (`bible/sets/<ENV_ID>/set.json` + standalone 60 FPS `<ENV_ID>_FPV_Director.html` + `.blend`). When re-anchoring off-axis/OTS shots from 3D screenshots, always use `compute_view_space_blocking_summary` + isolated character crops so wide-master backgrounds never bleed into OTS angles.
     5. **Human-Paced Single-Speaker Dialogue Gate (`dialogue_agent.py`) & Exact Native Script Mandate**: Automatically write or validate all scene dialogue so that:
        - Every line fits human speaking pace (`8–14 words` per 8s shot, `5–8 words` per 5s shot, with `0.6s` pre/post-roll breath cushions).
        - Every character's acoustic voice print (`CharacterVoiceProfile`) stays identical across every shot and episode.
        - **Native Script Dialogues (Any Language)**: Always write and request prompts using the exact dialogue in its **authentic native script** alongside phonetic transliteration when helpful.
        - **Voice-Over vs. On-Camera Lip-Sync Gate**:
          - **Voice-Over / Off-Screen / Narrator / Phone Audio (`off_screen=True`)**: **ALL visible characters' mouths must remain 100% SHUT (`CLOSED`)**.
          - **On-Camera Dialogue (`off_screen=False`)**: **Only the intended speaking character lip-syncs properly** to the line, while all other visible characters in the frame keep their mouths `100% CLOSED`.

3. **Primary Media Generation Tier — Nano Banana Pro (14 References) & Gemini Omni 1.1 / Vertex Veo 3.1**:
   - **Keyframes & Reference Sheets**: Default to **Nano Banana Pro (`gemini-3-pro-image`)** on Vertex AI, passing **up to 14 reference images** simultaneously (`compiled_shot.reference_image_paths` — character turnarounds, expression sheets, location plates, hero props, style bible, and preceding scene shots), or built-in `generate_image` (up to 3 refs).
   - **Video & Dialogue Clips**: Default to **Gemini Omni 1.1 (`gemini-omni-1.1-flash`)** (up to 10 reference images + 3 reference videos, native **1080p/4K**, **4–10s** durations, and **`extend_video`**) or **Vertex AI Veo 3.1 (`veo-3.1-generate-001`)** via [`vertex_veo_director.py`](file:///Users/girishnair/Desktop/GGN/AGY/MDC/vertex_veo_director.py).
   - **Cinematic Multi-Shot Staging**: Stage and prompt scenes as dynamic cinematic multi-shot sequences (e.g. `[Beat 1: 0s-4s Master / OTS Angle] -> [Cinematic Cut at 4s to Beat 2: Reverse Reaction Close-Up]`) to maximize dramatic pacing and visual momentum.

4. **Zero-Click Final Edited Video (`.mp4`) & DaVinci Resolve Dual Delivery**:
   - Write all story/character/scene specs into Python modules yourself behind the scenes.
   - Generate the keyframes, voice stems, and video clips yourself.
   - At **Gate G6**, ALWAYS deliver **BOTH**:
     1. **A Standalone Final Edited Master `.mp4` Video** (`productions/<PROJECT_ID>/post/<PROJECT_ID>_FINAL_EDITED_VIDEO.mp4`, loudnormed to `-14 LUFS` with dialogue, room tone, foley, and ducked score mixed together) so the Director and audience can play the finished movie immediately.
     2. **An Editable 5-Track DaVinci Resolve Timeline** (`SC01_v1` / `MASTER_v1` with `V1` picture, `A1` dialogue, `A2` room tone, `A3` foley, `A4` ducked score, subtitles, and QA markers) pushed automatically into DaVinci Resolve.

5. **Live Audience Demo Deterministic Anti-Hallucination Contract (`G0` → `G6`)**:
   - **Never Auto-Approve Gates Behind the Scenes**: In `director` or `hybrid` mode, NEVER run `gate G1 "approve all"`, `gate G2 "approve all"`, or `gate G3 "approve all"` until you have FIRST presented a visual review artifact (`g1_bible_review.md`, `g2_visual_review.md`, `g3_storyboard_review.md`, `g5_video_takes_review.md`) containing embedded PNG/JPG images and clickable `[filename](file:///...)` links, AND received the Director's explicit approval.
   - **Single-Source-of-Truth Reference Chain (Zero Character/Room Drift)**:
     - At **Gate G2**, always embed and verify every character's canonical `front.png` + `face_closeup.png` and every environment's `cam_WIDE_MASTER.png` in `g2_visual_review.md` so no stale/mismatched packs exist.
     - At **Gate G3**, every storyboard keyframe generation MUST pass the 3 locked reference anchors: `(1) 3D Camera Blockout PNG (cam:<NAME>_3d_clean.png)` + `(2) Approved G2 Master Room Plate (cam_WIDE_MASTER.png)` + `(3) Approved G2 Character Sheet(s)` — NEVER generate a shot from text alone.
   - **Visible 3D World Launch at Gate G2**:
     - Whenever a 3D set is built at **G2**, ALWAYS build the `.blend` file (`open_in_blender.py --save-and-render`), render all `cam:<NAME>` 9:16/16:9 camera blockouts, AND automatically open Blender 5.2 (`/Applications/Blender.app <ENV_ID>.blend`) or the 60 FPS `<ENV_ID>_FPV_Director.html` viewer on the Director's screen so the audience sees the 3D set live.
   - **Sequential Subagent Guard & PATH Guard**:
     - Always prepend `/opt/homebrew/bin:/usr/local/bin` to `PATH` when invoking `ffmpeg`, `gcloud`, or `blender`.
     - When invoking `image-generator` subagents, run **1 subagent at a time** (or up to 3 images inside a single subagent) to prevent concurrent `429 RESOURCE_EXHAUSTED` rate-limit errors during a live demo.
   - **Two-Tier `360p` Draft → `3×3` Contact Sheet Critic → `4K` Finalize (`Gemini Omni 1.5 Flash` + `Gemini TTS 3.8`)**:
     - At **Gate G4**, use **Gemini TTS 3.8 (`gemini-3.8-tts`)** with native script dialogue + `[acoustic delivery tags]` intact (never ElevenLabs).
     - At **Gate G5**, generate `360p` draft video takes first, run `build_3x3_contact_sheet()` + `VisualContinuityCritic`, and promote the winning take to `4K` reusing the **exact same prompt, seed, and reference bundle** (`reuse_prompt_on_finalize=True`), then assemble both the **Final Edited `.mp4` Video** and the **5-track `-14 LUFS` DaVinci Resolve Timeline** at **Gate G6** with `StateTokenGuard` and `TimelineGitRepo` commit history.

