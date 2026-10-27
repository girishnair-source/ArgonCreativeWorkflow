# AGY Studio — Autonomous AI Director & NLE Pipeline Guide (`AGENTS.md`)

## 1. Zero-Touch Creative Editor Mode
The operator of this workspace is a **Creative Video Editor / Director**. Antigravity executes all Python commands, virtual-environment setup, 3D set servers, media generation passes, and DaVinci Resolve timeline syncs directly via `run_command`.

## 2. Environment & Venv Commands (Blaze-Free)
All pipeline commands run inside the workspace `.venv`:

```bash
# Bootstrap or verify local macOS studio environment (.venv, ffmpeg, gcloud)
bash setup_studio_env.sh

# Inspect overall project and scene status
.venv/bin/python -m agy_studio status

# View recent decisions or open questions
.venv/bin/python -m agy_studio decisions 20
.venv/bin/python -m agy_studio questions

# Advance a scene or run a full orchestration tick
.venv/bin/python -m agy_studio advance SC01
.venv/bin/python -m agy_studio tick --no-poll
```

## 3. Gate Workflow (`G0` → `G7`)
- **Mandatory Step 0 (Project / Conversation Start — 3-Mode Intake via `ask_question`)**:
  - At the start of any new project or production conversation, ALWAYS call `ask_question` first to lock:
    1. **Operating Mode (`ControlMode`)**:
       - `director` (*Ultimate Director Control* — human approval at every gate `G1`–`G6`)
       - `hybrid` (*Smart Hybrid Control* — stops at `G1`, `G3`, `G5`, `G6`; auto-passes `G2`/`G4` when reusing Global Bible assets/voices)
       - `autopilot` (*Seamless Autopilot* — approve `G1` once, runs `G2`–`G5` autonomously with QA self-healing to `G6` DaVinci Resolve cut)
    2. **Content Format (`ContentFormat`)**: `microdrama_9_16` (`9:16`), `episode_16_9` (`16:9`), `feature_film` (`16:9`/`2.39:1`), or `marketing_spot` (multi-shot commercial escalation).
    3. **3D Blocking Engine**: Zero-Blender 60 FPS Browser 3D FPV (`<ENV_ID>_FPV_Director.html` / `set3d serve`), Native Blender 5.2 (`.blend`), or Both.
  - Persist the selection immediately via `.venv/bin/python -m agy_studio --project <PROJECT_ID> set-mode <director|hybrid|autopilot> --format <FORMAT>`.
- **Gate G0 — Script Intake**: `.venv/bin/python -m agy_studio ingest <script.fountain>`
- **Gate G1 — Production Bible, StyleLock, Voice Profiles & Schedule**: `.venv/bin/python -m agy_studio g1` → `.venv/bin/python -m agy_studio gate G1 "approve all"`
- **Gate G2 — Visual Design (Character/Prop Packs, Environment Geometry & `set3d` Virtual Sets)**:
  - Check or generate design candidates: `.venv/bin/python -m agy_studio design SC01 status` / `generate`
  - Build & serve 3D sets:
    - `.venv/bin/python -m agy_studio set3d build ENV_RISHIS_HOUSE`
    - `.venv/bin/python -m agy_studio set3d serve ENV_RISHIS_HOUSE --host 127.0.0.1 --port 8765`
    - `.venv/bin/python -m agy_studio set3d stylize ENV_RISHIS_HOUSE WIDE_MASTER`
  - Approve Gate G2: `.venv/bin/python -m agy_studio gate G2 SC01 "approve all"`
- **Gate G3 — Direction, Cinematography & Shot Decomposition**:
  - `.venv/bin/python -m agy_studio direct SC01`
  - Every shot references a locked `cam:<NAME>` spatial anchor from `set3d` and enforces single-speaker human-paced dialogue budgets (`2.2 wps`, `0.6s` pre/post-roll cushions).
- **Gate G4 — Keyframe & Audio Pre-Flight Review**:
  - `.venv/bin/python -m agy_studio audio SC01`
- **Gate G5 — Video Clip Generation & QA Critic Selection**:
  - Generates multi-reference takes via Gemini Omni 1.1 / Vertex AI Veo 3.1 and runs automated QA (`transcript_match`, `speaker_similarity`, `identity_similarity`).
- **Gate G6 — DaVinci Resolve Scene Timeline Review & Round-Trip Sync**:
  - `.venv/bin/python -m agy_studio poll` / `.venv/bin/python -m agy_studio approve SC01`
- **Gate G7 — Master Episode Assembly**:
  - `.venv/bin/python -m agy_studio master`

## 4. Media & Video Handling Rules
- **Do NOT load raw video/audio binaries into chat**: Never run `view_file` on large `.mp4`, `.mov`, `.mkv`, or `.wav` files; inspect frame contact sheets or metadata instead.
- **Active Production ("Like. Follow. Die or Kill.")**: See [`productions/LIKE_FOLLOW_DIE_OR_KILL/HANDOFF.md`](../productions/LIKE_FOLLOW_DIE_OR_KILL/HANDOFF.md) and [`productions/LIKE_FOLLOW_DIE_OR_KILL/starter_kit/`](../productions/LIKE_FOLLOW_DIE_OR_KILL/starter_kit/) (as well as [`productions/LFKOD_S01/HANDOFF.md`](../productions/LFKOD_S01/HANDOFF.md)) for the complete Starter Kit, screenplay PDF/Fountain script, master character/location references, Blender 5.2 `.blend` set, Zero-Blender 60 FPS Procedural Three.js/GLSL 3D FPV HTML viewers, and locked `cam:` angles.
