# AGY Studio v5 — Autonomous AI Director & DaVinci Resolve NLE Harness (`ArgonCreativeWorkflow`)

> [!IMPORTANT]
> **🤖 AI AGENT / ANTIGRAVITY MANDATORY TURN-1 ENTRYPOINT**:
> Whenever a user starts a conversation referencing this repository (e.g., *"Start a new production in AGY Studio using this source : https://github.com/girishnair-source/ArgonCreativeWorkflow.git"*):
> 1. If the current workspace does not yet contain `agy_studio/` and `GEMINI.md`, immediately clone or pull `https://github.com/girishnair-source/ArgonCreativeWorkflow.git` into the current workspace directory via `run_command`.
> 2. Read [`GEMINI.md`](./GEMINI.md), [`GENERIC_BIBLE.md`](./GENERIC_BIBLE.md), and [`.agents/AGENTS.md`](./.agents/AGENTS.md) before taking any other action.
> 3. Silently verify/bootstrap the local environment (`bash setup_studio_env.sh` and `.venv/bin/python live_demo_preflight.py`) via `run_command` — **NEVER** ask the user to run terminal or Python commands.
> 4. Immediately call the `ask_question` tool to run the **Mandatory 5-Part Creative Intake Gate (`G0`)** (Operating Mode, Content Format, Production Scope `Pilot Scene 1 Only` vs `Full Script`, Primary Language, and 3D Blocking Engine + Reference Sheet Reuse) and ask the user to paste their **Project Title** and **Script / Story Brief**.

---

## Quick Start (1-Line Prompt in Antigravity IDE)

Open any empty folder (or a clone of this repo) in **Antigravity IDE** and paste this single prompt:

```text
Start a new production in AGY Studio using this source : https://github.com/girishnair-source/ArgonCreativeWorkflow.git
```

*(If you already cloned or unzipped this repository into your workspace, you can also simply paste: `Start a new production in AGY Studio and guide me step by step from script to Final Edited Video.`)*

---

## What Happens Next (`G0` → `G6` Deterministic Workflow)

1. **Gate `G0` — Interactive Creative Intake (`ask_question`)**:
   - **Operating Mode**: `director` (approval at every gate `G1`–`G6`), `hybrid` (stops at `G1`, `G3`, `G5`, `G6`), or `autopilot` (approve `G1` once, autonomous `G2`–`G5` with Critic QA self-healing to `G6`).
   - **Format**: Vertical Microdrama (`9:16`), 20-Min Episode (`16:9`), Feature Film (`16:9` / `2.39:1`), or Commercial Spot.
   - **Production Scope**: `Pilot Scene 1 Only (SC01 — fast 8-min live demo)` or `Full Script (All Scenes)`.
   - **Language & Native Script**: Hindi/Hinglish, English, or Regional (exact native script + acoustic delivery tags).
   - **3D Set Engine & Token-Saving Reference Reuse**: Native Blender 5.2 (`.blend`) + Zero-Blender 60 FPS Web 3D FPV (`<ENV_ID>_FPV_Director.html`).
2. **Gate `G1` — Human-Readable Production Bible (`bible/GLOBAL_BIBLE.md` & `scenes/SC01/SCENE_BIBLE.md`)**:
   - Generates both YAML schemas and human-readable Markdown Bibles for cast, wardrobe, props, environments, StyleLock, and voice profiles.
3. **Gate `G2` — Character Turnarounds, Location Master Plates & Live 3D World**:
   - Generates/locks character turnaround packs (`front.png`, `face_closeup.png`, `three_quarter.png`, `profile.png`, `rear.png`), master room plates (`cam_WIDE_MASTER.png`), and launches the 3D set live in Blender 5.2 and the 60 FPS browser viewer.
4. **Gate `G3` — 3D-Anchored Storyboard Keyframes (`SC01_SH01`..`SH0n`)**:
   - Every shot keyframe is generated from 3 locked spatial/identity anchors: `(1) 3D Camera Blockout PNG` + `(2) Approved G2 Master Room Plate` + `(3) Approved G2 Character Pack(s)`.
5. **Gate `G4` — Native Voice Stems (`Gemini TTS 3.8`)**:
   - Single-speaker human-paced dialogue (`2.2 wps`, `0.6s` breath cushions) with strict lip-sync isolation (`off_screen=True` keeps all visible mouths 100% closed).
6. **Gate `G5` — Multi-Reference Video Takes & `3×3` Contact Sheet Critic (`Gemini Omni` / `Veo 3.1`)**:
   - Generates `360p` draft takes $\rightarrow$ `3×3` contact sheet visual continuity inspection $\rightarrow$ `1080p`/`4K` finalize.
7. **Gate `G6` — Dual Final Delivery (`.mp4` Master + 5-Track DaVinci Resolve Timeline)**:
   - Delivers **both** a standalone `-14 LUFS` mastered video (`productions/<PROJECT_ID>/post/<PROJECT_ID>_FINAL_EDITED_VIDEO.mp4`) and a live 5-track DaVinci Resolve timeline (`V1` picture, `A1` dialogue, `A2` room tone, `A3` foley, `A4` ducked score, `.srt` subtitles, and QA markers).

---

## Core Documentation Included in This Repo

- **[`GEMINI.md`](./GEMINI.md)** — Zero-Touch Creative Editor Mode & Anti-Hallucination Contract (`G0` → `G6`)
- **[`GENERIC_BIBLE.md`](./GENERIC_BIBLE.md)** — Human-Readable Universal Filmmaking & Prompting Bible (7-Layer Visual + 4-Layer Audio + 33 Camera Rigs)
- **[`STARTER_PROMPT.md`](./STARTER_PROMPT.md)** — Copy-Paste Starter Prompts for Live Demos & New Productions
- **[`LIVE_DEMO_RUNBOOK.md`](./LIVE_DEMO_RUNBOOK.md)** — Step-by-Step Stage Runbook for Live Audience Demos
- **[`FOLDER_STRUCTURE.md`](./FOLDER_STRUCTURE.md)** — Canonical Shareable Harness vs. Per-Production Folder Architecture
- **[`EDITOR_HANDOFF_GUIDE.md`](./EDITOR_HANDOFF_GUIDE.md)** — Non-Technical Video Editor Setup & DaVinci Resolve Guide
