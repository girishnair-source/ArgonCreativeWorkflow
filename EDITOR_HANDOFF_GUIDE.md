# AI Director & DaVinci Resolve Studio — Quick-Start Guide for Editors

You do **not** need to write any code or run any Python scripts. **Antigravity** handles all setup, file organization, AI image/video generation, and **DaVinci Resolve** timeline assembly automatically.

---

## Before You Start (What You Need on Your Mac)
1. **Antigravity** app installed.
2. **DaVinci Resolve** (Free or Studio Edition) installed.

---

## Step 1: Open the Project Folder & Pick the Best Model
1. Double-click **`Autonomous_AI_Director_Studio_Kit.zip`** to unzip it (for example, on your Desktop).
2. Open **Antigravity**, click **File > Open Folder**, and select the unzipped **`Autonomous_AI_Director_Studio_Kit`** folder.
3. In the **Antigravity Model Selector dropdown** (bottom of the chat box), select **`Gemini 3.1 Pro`** (or **`Claude Opus 4.6 (Thinking)`**).
4. Open **DaVinci Resolve**, create a new blank project (or open your target project), and leave DaVinci Resolve open in the background.

---

## Step 2: One-Time Setup (Copy & Paste This into Antigravity)
If this is your first time using the kit on this Mac, copy and paste the box below into the **Antigravity chat** and press **Enter**:

```text
Please set up my Mac for this studio kit from scratch. Check and automatically install everything required (including Homebrew packages, FFmpeg, Google Cloud CLI, and the local Python environment) by running setup_studio_env.sh yourself. If Google Cloud login is needed, trigger the browser login for me so I only have to click my Google account in the browser, and set my GCP project. Do not ask me to run terminal commands myself.
```

* **What happens next?**
  * Antigravity will silently install all required tools in the background (approve any permission prompts in Antigravity if asked).
  * A Google Sign-In tab will open in your web browser—just click your Google account and click **Allow**.

---

## Step 3: Create Your New Show, Cartoon, 20-Min Episode, or Movie
Whenever you want to create a new episode or movie from scratch, copy and paste the prompt below into **Antigravity** (and attach or paste your script/story idea):

```text
I want to create a brand-new video project from scratch and have you automatically assemble it onto my open DaVinci Resolve timeline.

Here is my story / script:
[PASTE YOUR SCRIPT, SCENE LIST, OR STORY IDEA HERE — OR ATTACH A PDF/DOC]

Please follow the Studio Workflow automatically:
1. Ask me what format and visual style I want (e.g., 2D Cartoon, 3D Stylized Animation, 20-Minute Multi-Scene Episode, Feature Movie, or 9:16 Vertical Microdrama) and the target DaVinci Resolve project name.
2. CASTING & LOCATION GATE (Save Tokens): Before generating any character or location reference sheets, list the characters and locations needed and ask me if I already have existing Character Reference Sheets or Location Reference Images to drop in. Only generate sheets for the ones I don't already have.
3. DIALOGUE & VOICE CONTINUITY GATE: Use the Dialogue Agent to write/check all scene dialogue using exact Hindi dialogue in authentic Devanagari script with zero English accent, matching human speaking pace for each shot duration, ensuring acoustic continuity across episodes, and strictly locking on-camera lips so ONLY the intended speaker speaks.
4. MULTI-REFERENCE MEDIA GENERATION & CINEMATIC MULTI-SHOT STAGING: Stage scenes as cinematic multi-shot sequences (motivated camera cuts, master to reaction close-ups) using Nano Banana Pro (gemini-3-pro-image with up to 14 reference images) for keyframes/sheets and Gemini Omni 1.1 (gemini-omni-1.1-flash with up to 10 reference images + 3 reference videos at 1080p/4K) for 8–10s video clips, stitch the master export with .srt captions, and automatically sync the edited multi-track sequence directly into my open DaVinci Resolve timeline without me running scripts.
```

---

## How to Give Existing Character or Location Sheets to Antigravity
When Antigravity's **Casting Agent** asks if you already have Character Reference Sheets or Location Concept Art:
* Simply **drag and drop your `.jpg` or `.png` images** into the Antigravity chat (or drop them into the `generated_assets/anchors/` folder) and tell Antigravity:
  * *"Use this image for [Character Name / Location Name]."*
* Antigravity will lock those sheets into the project bible and **spend zero tokens** regenerating them!

---

## Non-Destructive Editing & Timeline Versioning Guarantee
* **Your manual edits in DaVinci Resolve are 100% safe**:
  * You can freely trim clips, adjust audio levels, add transitions, and color grade directly in DaVinci Resolve.
  * The automated sync script **never deletes or overwrites** your existing timelines or media bins.
* **Automatic Timeline Versioning**:
  * If you re-run an Antigravity command to generate or sync new shots, the script checks your active project.
  * If `EP01_Daring` already exists, it leaves your cut untouched and creates **`EP01_Daring_v02`** (then `_v03`, etc.).
  * You can switch between cuts using DaVinci's timeline dropdown or view both simultaneously using **Stacked Timelines**.

---

## Troubleshooting (Plain English)
* **If Google Cloud asks you to re-authenticate**:
  Just tell Antigravity in chat: *"Please refresh my Google Cloud login"* (or if you ever need to type one command in Terminal, type `gcloud auth login` and press Enter).
* **If DaVinci Resolve doesn't show the timeline yet**:
  Make sure DaVinci Resolve is open with a project window visible (not stuck on the Project Manager welcome popup), then tell Antigravity: *"Sync the timeline to DaVinci Resolve now."*

