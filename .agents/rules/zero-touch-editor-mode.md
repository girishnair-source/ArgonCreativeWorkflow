# Zero-Touch Creative Editor Mode (Mandatory Workspace Rules)

The user operating this workspace is a **Creative Video Editor / Director**, NOT a software engineer.
Follow these rules strictly on every turn:

1. **NEVER Ask the Editor to Run Python, Virtual Environments, or Terminal Commands**:
   - You (Antigravity) must execute all terminal commands, Python scripts, environment setups, and DaVinci Resolve sync triggers yourself using `run_command`.
   - If `.venv`, `ffmpeg`, or `gcloud` are missing or unauthenticated, automatically run `bash setup_studio_env.sh` via `run_command`. If Google Cloud needs authentication, run `gcloud auth login` and `gcloud auth application-default login` via `run_command` so the Google sign-in browser window pops open automatically for the editor, or tell them the single command `gcloud auth login` only if needed.

2. **Interactive Creative Intake (Use `ask_question` in Plain Filmmaking Language)**:
   - When the editor asks to create a new show, episode, cartoon, or movie:
     1. **Format & Style**: Ask via `ask_question` whether the project is a **2D/3D Cartoon**, **20-Minute Episode**, **Feature Movie**, or **Vertical Microdrama (9:16)**, and confirm aspect ratio (`16:9` vs `9:16`).
     2. **Token-Saving Reference Sheet Gate (`casting_agent.py`)**: Before generating ANY character or location images, ALWAYS ask the editor first:
        *"Do you already have Character Reference Sheets or Location Reference Images for these characters/settings? (If yes, drop them into this folder or share their filenames so we reuse them and save tokens; otherwise I will generate new ones for you.)"*
     3. **Human-Paced Single-Speaker Dialogue Gate (`dialogue_agent.py`) & Exact Hindi Mandate**: Automatically write or validate all scene dialogue so that:
        - Every line fits human speaking pace (`8–14 words` per 8s shot, `5–8 words` per 5s shot, with `0.6s` pre/post-roll breath cushions).
        - Every character's acoustic voice print (`CharacterVoiceProfile`) stays identical across every shot and episode.
        - **Native Script Dialogues (Any Language)**: Dialogues can be in **any language** (Hindi, Tamil, Telugu, Malayalam, English, Spanish, Japanese, etc.). Always write and request prompts using the exact dialogue in its **authentic native script** alongside phonetic transliteration when helpful (e.g. Devanagari `भाई, लाइव चालू है... दो हज़ार लोग देख रहे हैं!` for Hindi) so Gemini Omni / Veo renders pronunciation clearly with authentic native articulation.
        - **Voice-Over vs. On-Camera Lip-Sync Gate**:
          - **Voice-Over / Off-Screen / Narrator / Phone Audio (`off_screen=True`)**: **ALL visible characters' mouths must remain 100% SHUT (`CLOSED`)**.
          - **On-Camera Dialogue (`off_screen=False`)**: **Only the intended speaking character lip-syncs properly** to the line, while all other visible characters in the frame keep their mouths `100% CLOSED`.

3. **Primary Media Generation Tier — Nano Banana Pro (14 References) & Gemini Omni 1.1 / Vertex Veo 3.1**:
   - **Keyframes & Reference Sheets**: Default to **Nano Banana Pro (`gemini-3-pro-image`)** on Vertex AI, passing **up to 14 reference images** simultaneously (`compiled_shot.reference_image_paths` — character turnarounds, expression sheets, location plates, hero props, style bible, and preceding scene shots), or built-in `generate_image` (up to 3 refs).
   - **Video & Dialogue Clips**: Default to **Gemini Omni 1.1 (`gemini-omni-1.1-flash`)** (up to 10 reference images + 3 reference videos, native **1080p/4K**, **4–10s** durations, and **`extend_video`**) or **Vertex AI Veo 3.1 (`veo-3.1-generate-001`)** via [`vertex_veo_director.py`](../../vertex_veo_director.py).
   - **Cinematic Multi-Shot Staging**: Stage and prompt scenes as dynamic cinematic multi-shot sequences (e.g. `[Beat 1: 0s-4s Master / OTS Angle] -> [Cinematic Cut at 4s to Beat 2: Reverse Reaction Close-Up]`) to maximize dramatic pacing and visual momentum.

4. **Zero-Click DaVinci Resolve Delivery**:
   - Write all story/character/scene specs into Python modules yourself behind the scenes.
   - Generate the keyframes and video clips yourself.
   - Run `.venv/bin/python assemble_microdrama.py` and `.venv/bin/python resolve_sync_trigger.py` yourself so the finished edit, audio tracks, subtitles, and markers appear automatically inside the editor's open **DaVinci Resolve** timeline without the editor touching a single script.

6. **Fail-Fast Credential Contract — NO Offline Fallbacks, EVER (`agy_studio/adapters/auth.py`)**:
   - Every real generation call (Gemini Omni / Veo video at `G5`, Gemini TTS at `G4`, Nano Banana Pro images, Lyria music, Gemini text/ASR/vision judges) is authenticated with the local `gcloud auth print-access-token`. The built-in `image-generator` subagent uses Antigravity's own credentials, so **"G3 images worked" proves NOTHING about G4/G5** — always run `.venv/bin/python live_demo_preflight.py` (which actually mints a token) before `G4`.
   - When any step raises `AuthError` / prints `STOPPED — Google Cloud sign-in problem`, **STOP IMMEDIATELY** and paste the complete message to the Director verbatim (it already contains gcloud's real error and the exact fix, e.g. `gcloud config set context_aware/use_client_certificate true` + `gcloud auth login` + `gcloud auth application-default login`). Run those commands via `run_command` (the browser sign-in opens for the editor), then re-run the **same** gate command — the `auth` blocker clears itself on the next tick.
   - **FORBIDDEN while a credential problem is open** (these waste minutes, produce unusable output and hide the real block): setting `AGY_FAKE_GENERATION=1`, macOS `say` / `espeak` / any offline TTS, `ffmpeg zoompan` / Ken-Burns stills or slideshows passed off as video, writing ad-hoc `prepare_g4_*.py` / `prepare_g5_*.py` / "workaround" scripts, retrying with new seeds, or switching provider models. The only fix for a credential error is a credential fix.
   - Never hard-code or guess a GCP project: `VERTEX_PROJECT_ID` / `GOOGLE_CLOUD_PROJECT` or `gcloud config get-value project` must resolve, otherwise the harness stops with the `gcloud config set project <ID>` instruction. To demo on a non-corporate account set `AGY_GCLOUD_ACCOUNT=<account>` (or `gcloud config set account <account>`).

7. **Autopilot Mode Contract (`control_mode=autopilot`) — Script → Editor with ONE Human Click**:
   - The Director approves **G1 only**. From there `tick` / `advance` runs `G2 → G3 → G4 → G5 → G6` with no questions and no waiting gates: missing reference images are generated immediately (no "do you have reference sheets?" prompt), G2/G3/G4 packets are opened for the audit trail and answered by AGY (`source=autopilot`), G5 picks the QA-recommended take per clip and sets first voice anchors itself, and the scene cut is pushed into DaVinci Resolve at `G6`.
   - **The 3D set (`set3d` + Blender export) is skipped in autopilot** — there is no human to frame cameras in it. `set3d build <ENV_ID>` remains available on demand.
   - Do NOT call `ask_question` between G1 and G6 in autopilot, and do NOT stop to present `g2_visual_review.md` / `g3_storyboard_review.md` for approval — present the finished `G6` timeline and the review folders (`scenes/<SID>/review/`) afterwards instead.
   - The fail-fast credential contract (rule 6) still applies: the only thing that may stop an autopilot run is an `AuthError`, which must be shown verbatim and fixed — never worked around.
