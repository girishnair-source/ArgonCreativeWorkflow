# AGY Studio v5 — One-Line Starter Prompt (`STARTER_PROMPT.md`)

All studio rules, environment checks, 3D world builders, and DaVinci Resolve / Final Edited `.mp4` delivery steps are built directly into the Starter Kit repository (`https://github.com/girishnair-source/ArgonCreativeWorkflow.git` — `GEMINI.md`, `GENERIC_BIBLE.md`, and `.agents/`).

## Option A — Start Directly From GitHub in Any Empty Folder (Recommended for Live Demo)
Open any empty folder in **Antigravity IDE** and paste this single plain-English prompt:

```text
Start a new production in AGY Studio using this source : https://github.com/girishnair-source/ArgonCreativeWorkflow.git
```

Antigravity will automatically clone `https://github.com/girishnair-source/ArgonCreativeWorkflow.git` into the workspace, verify the studio environment (`setup_studio_env.sh` + `live_demo_preflight.py`), and launch the interactive Creative Intake modal (`ask_question`).

---

## Option B — Start Inside an Already-Cloned or Unzipped Starter Kit Folder
If you have already cloned the repo or unzipped `AGY_STUDIO_STARTER_KIT.zip` and opened that folder in **Antigravity IDE**, paste:

```text
Start a new production in AGY Studio and guide me step by step from script to Final Edited Video.
```

---

## What Antigravity Will Ask You Next (`ask_question`)
Immediately after you run the starter prompt, Antigravity will pop up an interactive selection modal asking for:
1. **Operating Mode** (`Director`, `Hybrid`, or `Autopilot`)
2. **Content Format** (`Vertical Microdrama 9:16`, `20-Min Episode 16:9`, `Feature Film`, or `Commercial Spot`)
3. **Production Scope** (`Pilot Scene 1 Only (SC01 — Live Demo Mode)` or `Full Script (All Scenes)`)
4. **Primary Language** (`Hindi / Hinglish`, `English`, or regional language)
5. **3D Set Engine & Existing Reference Images** (`Blender 5.2 + 60 FPS Web 3D Viewer`, and whether you have character/location sheets to reuse)
6. **Your Project Title & Script / Story Brief** (which you paste or attach in the next reply; `PROJECT_ID` is auto-generated from your title if you don't specify one)
