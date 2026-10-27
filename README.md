# AGY Studio v5 — Autonomous AI Director NLE Harness 
(ArgonCreativeWorkflow)

> [!IMPORTANT]
> **🤖 AI AGENT / ANTIGRAVITY MANDATORY TURN-1 ENTRYPOINT**:
> Whenever a user starts a conversation referencing this repository (e.g., *"Start a new production in AGY Studio using this source : https://github.com/girishnair-source/ArgonCreativeWorkflow.git"*):
> 1. If the current workspace does not yet contain `agy_studio/` and `GEMINI.md`, immediately clone or pull `https://github.com/girishnair-source/ArgonCreativeWorkflow.git` into the current workspace directory via `run_command`.
> 2. Read [`GEMINI.md`](./GEMINI.md), [`GENERIC_BIBLE.md`](./GENERIC_BIBLE.md), and [`.agents/AGENTS.md`](./.agents/AGENTS.md) before taking any other action.
> 3. Silently verify/bootstrap the local environment (`bash setup_studio_env.sh` and `.venv/bin/python live_demo_preflight.py`) via `run_command` — **NEVER** ask the user to run terminal or Python commands.
> 4. Immediately call the `ask_question` tool to run the **Mandatory 5-Part Creative Intake Gate (`G0`)** (Operating Mode, Content Format, Production Scope `Pilot Scene 1 Only` vs `Full Script`, Primary Language, and 3D Blocking Engine + Reference Sheet Reuse) and ask the user to paste their **Project Title** and **Script / Story Brief**.

---

## 1. Quick Start (1-Line Prompt in Antigravity IDE — Zero Coding)

Open any empty folder (or a clone of this repo) in **Antigravity IDE** and paste this single plain-English prompt:

```text
Start a new production in AGY Studio using this source : https://github.com/girishnair-source/ArgonCreativeWorkflow.git
```

*(If you already cloned or unzipped this repository into your workspace, you can also paste: `Start a new production in AGY Studio and guide me step by step from script to Final Edited Video.`)*

---

## 2. What the Starter Kit Brings In

The **AGY Studio v5 Starter Kit** is a production-hardened filmmaking harness that turns raw screenplays or story briefs into broadcast-mastered multi-track timelines and finished `.mp4` movies without writing a single line of code.

### A. Triple-Tier NLE & Open-Interchange Failover (Works With *Any* Video Editor)
The Starter Kit never dead-ends if a specific editing app or plugin is missing. At **Gate `G6`**, it executes an automatic 3-tier NLE failover chain alongside a standalone `-14 LUFS` mastered `.mp4` delivery:

1. **Tier 1 — Live NLE MCP Connection (Primary)**:
   - Automatically detects and connects to your active NLE MCP server (e.g., `davinci_resolve` MCP) to build a live, editable 5-track timeline (`V1` picture, `A1` dialogue stems, `A2` room tone, `A3` spot foley, `A4` sidechain-ducked score, `.srt` subtitles, and color-coded QA markers).
2. **Tier 2 — Native NLE Scripting Fallback (`resolve_sync_trigger.py`)**:
   - If the MCP server is offline, the Starter Kit automatically falls back to direct local NLE scripting API execution inside the editor (`DaVinciResolveScript` / Fusion bridge) to assemble the timeline natively.
3. **Tier 3 — Universal Open-Source Interchange Formats (`OTIO` / `OIIO` / `FCPXML` / `EDL`) + Master `.mp4`**:
   - If no NLE is open—or if the editor works in **Adobe Premiere Pro**, **Final Cut Pro**, **Avid Media Composer**, **Blender VSE**, or **Nuke**—the Starter Kit automatically generates open-standard interchange packages (**OpenTimelineIO `.otio`**, **OpenImageIO (`OIIO`)** color/frame-accurate image sequences, **FCPXML**, and **CMX 3600 EDL**) plus a standalone broadcast-leveled **`<PROJECT_ID>_FINAL_EDITED_VIDEO.mp4`** (`-14 LUFS`).

### B. Multi-Layer Cloud, 3D & Model Failovers (Zero Live-Demo Crashes)
- **3D Set Engine Failover**: Uses native **Blender 5.2 (`.blend`)** when installed for headless Eevee/Cycles camera blockouts, and automatically pairs or falls back to a **Zero-Blender 60 FPS Procedural Three.js / WebGL / GLSL Browser 3D Viewer (`<ENV_ID>_FPV_Director.html`)** so 3D spatial camera blocking works on *any* laptop with zero 3D software installed.
- **Model & Auth Failover Chain**: Automatically cascades across model tiers (`Gemini Omni` $\rightarrow$ `Veo 3.1 Standard` $\rightarrow$ `Veo 3.1 Fast`), resolves multi-account Google Cloud OAuth/ADC credentials automatically, self-heals RAI safety filter triggers via prompt softening, and persists provider operation IDs in `dispatch.jsonl` so interrupted renders resume without re-billing.

### C. Deep Filmmaking & Post-Production Domain Knowledge Built In
- **Universal Filmmaking & Prompting Bible ([`GENERIC_BIBLE.md`](./GENERIC_BIBLE.md))**:
  - Encodes a deterministic **7-Layer Visual Architecture** (`Subject`, `Context`, `Action`, `Style`, `Camera`, `Composition`, `Ambiance`) + **4-Layer Audio Spec** (`Dialogue`, `Ambient`, `SFX`, `Music`) + **6 Cinema Camera Bodies** (ARRI Alexa 35, RED V-Raptor, Sony Venice 2, IMAX 70mm, Panavision DXL2, Arriflex 16SR3) $\times$ **6 Optical Lenses** (Cooke S4/i, ARRI Master Anamorphic, Panavision C-Series, Helios 44-2, Angenieux Optimo, Laowa 24mm Probe) $\times$ **33 Camera Rigs**.
- **3-Tier Hierarchical Memory (`GLOBAL_BIBLE.md` $\rightarrow$ `SCENE_BIBLE.md` $\rightarrow$ Shot Working Set)**:
  - Eliminates LLM "context rot" across 20-minute episodes and feature films by compiling only the active scene's locked character turnaround packs, wardrobe states, hero props, and 3D `cam:<NAME>` coordinates into each shot's prompt bundle.
- **Human-Paced Single-Speaker Dialogue & Lip-Sync Physics ([`dialogue_agent.py`](./dialogue_agent.py))**:
  - Enforces natural human speaking cadence (`2.2 words/sec`, `8–14 words` per `8s` shot with `0.6s` breath cushions), preserves authentic native scripts (Devanagari Hindi, Hinglish, English, Tamil, Telugu) with `[acoustic delivery tags]`, and enforces **Single-Speaker Lip Lock** (`off_screen=True` forces all visible mouths `100% CLOSED`; on-camera lines isolate lip movement strictly to the active speaker).
- **Token-Saving Casting Audit ([`casting_agent.py`](./casting_agent.py)) & Two-Tier `360p` Draft $\rightarrow$ `3×3` Critic $\rightarrow$ `4K` Finalize**:
  - Audits existing character/location sheets before spending generation tokens, renders fast `360p` draft takes first, inspects a `3×3` frame contact sheet across 6 continuity failure modes (`Identity`, `Lighting`, `Prop`, `Wardrobe`, `Eyeline`, `Scale`), and promotes the winning take to `1080p`/`4K` reusing the exact seed and prompt (`reuse_prompt_on_finalize=True`).
- **Git-For-Video Timeline Version Control (`TimelineGitRepo` & `StateTokenGuard`)**:
  - Tracks every NLE cut with cryptographic `state_token` optimistic locking and Git-style `branch`, `commit`, `PR`, and `undo` history so AI edits never overwrite a human editor's manual trims.

---

## 3. What the Antigravity (AGY) Harness Enables

**Antigravity (AGY)** acts as the **Zero-Touch Creative Operating System** that bridges non-technical filmmakers with complex cloud AI models, 3D engines, and NLE timelines:

1. **100% Zero-Code Experience for Directors & Video Editors**:
   - Creative directors never touch Python, virtual environments, FFmpeg flags, Git commands, or JSON schemas. You direct in plain filmmaking language (*"Give me a tighter 50mm Over-The-Shoulder angle from behind Father"*, *"Make the room lighting warmer"*), and AGY executes all underlying engineering invisibly.
2. **Plug-and-Play Flexibility With Any Editor, 3D Tool, or MCP**:
   - Because AGY natively orchestrates **MCP servers**, **specialized subagents** (`casting-agent`, `dialogue-agent`, `image-generator`), **background task queues**, and **local CLI/API bridges**, the harness is completely modular. Swap DaVinci Resolve for Premiere/Final Cut (`OTIO`/`OIIO`/`FCPXML`), swap Blender 5.2 for Three.js WebGL or Unreal/USD, or plug in any new MCP server without rewriting your production workflow.
3. **Deterministic Human-in-the-Loop Gate Governance (`G0` $\rightarrow$ `G6`)**:
   - AGY enforces structured interactive intake (`ask_question`) and rich visual Markdown review artifacts (embedded character turnarounds, interactive 3D viewers, `3×3` video contact sheets) at every gate, giving you three operating modes:
     - **`director` Mode**: Explicit human sign-off at every gate (`G1` Bible $\rightarrow$ `G2` Cast/3D Set $\rightarrow$ `G3` Storyboard $\rightarrow$ `G4` Voice $\rightarrow$ `G5` Video Takes $\rightarrow$ `G6` NLE Cut).
     - **`hybrid` Mode**: Stops at core creative gates (`G1`, `G3`, `G5`, `G6`) while auto-passing `G2`/`G4` when reusing approved Global Bible assets.
     - **`autopilot` Mode**: Approve `G1` once and let AGY run `G2`–`G5` autonomously with Critic QA self-healing (`threshold >= 0.85`) straight to your `G6` timeline.

---

## 4. Why Argon (`Gemini Omni`) Is the Best Model to Work With AGY

The **Argon Creative Workflow** is purpose-built around **Argon (`Gemini Omni` / `reference_to_video`)** because its multimodal architecture solves the hardest continuity bottlenecks in AI filmmaking:

1. **Deep Multi-Reference Spatial & Identity Grounding (Up to 10 Reference Images + 3 Reference Videos)**:
   - Standard image-to-video models only accept a single starting frame—so as soon as the camera cuts to a reverse Over-The-Shoulder (OTS) angle or a character turns around, faces, clothing, and wall geometry morph.
   - **Argon** ingests the **3D camera blockout render**, **multi-angle character turnaround sheets (`front`, `3/4`, `profile`, `rear`)**, **master room plate**, **hero prop pack**, and **preceding video clip** simultaneously in a single `reference_to_video` call, preserving `>96%` facial, wardrobe, and architectural continuity across every cut.
2. **Native Time-Coded Multi-Shot Direction & Optical Physics**:
   - Argon natively executes time-coded multi-beat shot progressions within a single continuous generation (e.g., `(0-2.5s) Low-angle ledge slip -> (2.5-5.5s) Whip-tilt rescue pull & fall -> (5.5-8.0s) Overhead relieved laughter`) while honoring specific cinema lens focal lengths, depth-of-field bokeh, and camera rig motion.
3. **Joint Audio-Visual Performance & Native-Script Lip-Sync Isolation**:
   - Argon generates synchronized performance audio and facial micro-expressions together—supporting authentic native scripts (Hindi/Devanagari, Hinglish, English, Regional) and obeying AGY's **Single-Speaker Lip-Sync Rules** so only the active speaker's mouth moves while other visible characters remain naturally silent.
4. **Seamless Two-Tier Draft-to-Finalize Promotion (`360p` $\rightarrow$ `1080p`/`4K`) & Video Extension**:
   - Argon's fast `360p` draft generation pairs with AGY's `3×3` `VisualContinuityCritic` to validate blocking and physics in seconds before deterministically promoting the winning seed and reference bundle to `1080p`/`4K` master quality.

---

## 5. Core Documentation Included in This Repo

- **[`GEMINI.md`](./GEMINI.md)** — Zero-Touch Creative Editor Mode & Anti-Hallucination Contract (`G0` → `G6`)
- **[`GENERIC_BIBLE.md`](./GENERIC_BIBLE.md)** — Human-Readable Universal Filmmaking & Prompting Bible (7-Layer Visual + 4-Layer Audio + 33 Camera Rigs)
- **[`STARTER_PROMPT.md`](./STARTER_PROMPT.md)** — Copy-Paste Starter Prompts for Live Demos & New Productions
- **[`LIVE_DEMO_RUNBOOK.md`](./LIVE_DEMO_RUNBOOK.md)** — Step-by-Step Stage Runbook for Live Audience Demos
- **[`FOLDER_STRUCTURE.md`](./FOLDER_STRUCTURE.md)** — Canonical Shareable Harness vs. Per-Production Folder Architecture
- **[`EDITOR_HANDOFF_GUIDE.md`](./EDITOR_HANDOFF_GUIDE.md)** — Non-Technical Video Editor Setup & DaVinci Resolve Guide
