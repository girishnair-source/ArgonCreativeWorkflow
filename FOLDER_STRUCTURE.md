# AGY Studio — Locked Folder Structure & Shareable Solution Guide

This document defines the **locked folder layout** for AGY Studio so any filmmaker, microdrama creator, or marketing team knows:
1. **What to share** when packaging AGY Studio for another user or team (**Shareable Core Bundle**).
2. **Where production assets live** for each show, film, or campaign (**Per-Production Folder `productions/<PROJECT_ID>/`**).

---

## 1. What to Share With Others (Shareable Solution Bundle)

To share AGY Studio with a teammate or customer so they can open it in **Google Antigravity** and run the full pipeline immediately, include **only the Shareable Core** (or export it automatically with `.venv/bin/python -m agy_studio export-bundle <DEST_DIR>`):

```text
agy_studio_root/
├── GEMINI.md                        # Mandatory Zero-Touch Creative Editor Mode rules
├── FOLDER_STRUCTURE.md              # This locked folder layout & sharing specification
├── README.md                        # Quickstart & architecture overview
├── EDITOR_HANDOFF_GUIDE.md          # Editor guide for DaVinci Resolve + Antigravity
├── setup_studio_env.sh              # One-step macOS/Linux bootstrapper (.venv, ffmpeg, gcloud)
├── pyproject.toml                   # Python package & tool configuration
├── requirements.txt                 # Locked Python dependencies
├── mcp_config.json                  # DaVinci Resolve MCP server registration
├── resolve_mcp_server.py            # 20-tool DaVinci Resolve NLE MCP server
├── resolve_sync_trigger.py          # Automatic timeline sync trigger
├── vertex_veo_director.py           # Vertex AI Veo / Gemini Omni multi-ref director
├── casting_agent.py                 # Token-saving character & location reference gate
├── dialogue_agent.py                # Human-paced single-speaker dialogue & voice validator
├── pro_audio_director.py            # 4-layer audio & stem planner
├── production_bible.py              # Base bible utilities
├── production_ledger.py             # Cost & asset ledger utilities
├── schemas.py                       # Shared Pydantic schemas
├── new_project_template.py          # Starter template for new shows/commercials
│
├── .agents/                         # Antigravity Customizations Root (MUST BE SHARED)
│   ├── AGENTS.md                    # Pipeline command reference & Gate G1-G7 rules
│   ├── agents/                      # Specialist subagents
│   │   ├── casting-agent.md         # Token-saving reference sheet auditor
│   │   └── dialogue-agent.md        # Single-speaker speech pacing & native script agent
│   ├── rules/                       # Always-on workspace rules
│   │   ├── zero-touch-editor-mode.md
│   │   └── generic_bible.md         # Tier-0 Generic Filmmaking & Multi-Shot Commercial Bible
│   ├── skills/                      # On-demand skills
│   │   ├── autonomous-director/SKILL.md
│   │   └── prompt-videos/SKILL.md   # Replicate multi-shot commercial & 7-layer video prompting
│   └── plugins/
│       └── autonomous-director/     # Packaged plugin bundle
│
├── agy_studio/                      # Deterministic v4 Orchestration Engine (MUST BE SHARED)
│   ├── __init__.py & __main__.py
│   ├── cli.py                       # Unified CLI (.venv/bin/python -m agy_studio ...)
│   ├── generic_bible.py             # Tier-0 Generic Bible, Multi-Shot Compiler & Bundle Exporter
│   ├── bible_hierarchy.py           # GlobalContextBible, SceneBible & 3-Tier Reference Resolver
│   ├── voice_audition.py            # Gate 4 Voice Discovery, Audition & Global Bible Lock
│   ├── vault.py                     # Searchable Rejected Video Vault & Gate 6 Freeze/Backup
│   ├── orchestrator.py              # Gate G1-G7 state machine & autopilot/director modes
│   ├── models.py                    # Strict Pydantic v2 data models
│   ├── preproduction.py             # Script ingest, shot breakdown & G1 packet builder
│   ├── working_set.py               # Per-scene context & 3-tier reference lookup
│   ├── decomposer.py                # Scene-to-Shot & Shot-to-Clip decomposition
│   ├── shot_generator.py            # Multi-reference video & keyframe dispatch
│   ├── qa_critic.py                 # Automated hallucination, lip-sync & transcript critic
│   ├── review.py                    # Contact sheets + Gate 5 Interactive HTML Review UI
│   ├── editor.py                    # OpenTimelineIO & 8-track NLE cut assembler
│   ├── resolve_bridge.py            # DaVinci Resolve timeline sync
│   ├── roundtrip.py                 # Passive Resolve marker/swap/trim polling
│   ├── set3d/                       # 3D Virtual Set (Three.js viewer + Blender .obj/.gltf export)
│   ├── adapters/                    # Gemini Omni 1.1 Flash (racoon-cookies), Veo 3.1, Nano Banana Pro
│   └── agents/prompts/              # 24 film-department role prompts
│
├── tests/                           # Automated verification suite
└── productions/                     # Per-project workspace directory (share empty or with sample)
```

> [!IMPORTANT]
> **What NOT to include when sharing the studio**:
> Exclude `.venv/`, `__pycache__/`, `.pytest_cache/`, `*.lock`, and any private/unreleased `productions/<PROJECT_ID>/` folders (unless intentionally sharing a demo production like `LFKOD_S01`).
> Run `.venv/bin/python -m agy_studio export-bundle /path/to/export_folder` to build a clean shareable zip/folder automatically.

---

## 2. Locked Per-Production Folder Structure (`productions/<PROJECT_ID>/`)

Every project created under `productions/<PROJECT_ID>/` follows this exact locked hierarchy:

```text
productions/<PROJECT_ID>/
├── project.yaml                         # Project settings, aspect ratio, language & control_mode
├── state.json                           # Resumable state machine cursor across Gates G1-G7
├── decisions.jsonl                      # Immutable audit log of every human & agent decision
├── questions.jsonl                      # Blocking questions log (3-tier lookup fallback)
├── preferences.yaml                     # Scoped preferences learned from steers/rejects
├── HANDOFF.md                           # Human-readable production status & camera map
│
├── script/                              # Gate 0: Immutable versioned screenplays / briefs
│   └── script_v1.fountain
│
├── bible/                               # Gate 1: GLOBAL_KNOWLEDGE / CONTEXT_BIBLE (Read-Only Base)
│   ├── global_context_bible.yaml        # Single compact index of all scenes, shots, base sheets & voices
│   ├── GLOBAL_BIBLE.md                  # Human-readable Global Context Bible summary
│   ├── style_lock.yaml                  # Global visual & audio negative prompts / lens rules
│   ├── state_timeline.yaml              # Continuity state transitions across scenes
│   ├── casting.yaml                     # Global character -> locked voice profile registry
│   ├── entities/                        # Base character & prop records (CHAR_*.yaml, PROP_*.yaml)
│   ├── environments/                    # Base location records (ENV_*.yaml)
│   └── sets/                            # 3D Virtual Sets & Blender Exports per location
│       └── <ENV_ID>/
│           ├── set.json                 # Metric 3D room layout, lights, cameras & mannequins
│           ├── renders/                 # Browser/3D camera blockout + depth renders
│           ├── styled/                  # Nano Banana Pro style-passed camera plates
│           └── blender/                 # Exported .obj, .mtl, .gltf & open_in_blender.py
│
├── library/                             # Content-Addressed Global Asset Library (Reused Across Scenes)
│   ├── characters/                      # Locked character packs
│   ├── props/                           # Locked prop packs
│   ├── environments/                    # Locked location plates & cam:<NAME> angles
│   ├── packs/                           # Versioned turnaround packs (v1, v2...)
│   ├── voices/                          # Locked character voice profiles & audition WAV anchors
│   ├── rejected_takes_vault.jsonl       # Gate 5/6 Searchable Rejected Video Metadata Index
│   └── rejected_takes_vault.html        # Interactive browser UI to search & filter rejected videos
│
├── scenes/                              # Scene-Wise Production Loop (Gates G2 -> G6 per scene)
│   └── SC01/
│       ├── scene.yaml                   # Scene beats, dialogue & entry/exit continuity
│       ├── scene_bible.yaml             # Gate 2: SCENE BIBLE (scene-specific costume/lighting deltas)
│       ├── working_set.yaml             # Resolved 3-tier pointers (SceneBible -> GlobalBible)
│       ├── design/                      # Gate 2 candidate delta renders before library lock
│       ├── shots/                       # Gate 3 ShotSpecs (SH01.yaml, SH02.yaml...)
│       ├── boards/                      # Gate 3 Storyboard keyframes & animatics
│       ├── clips/                       # Decomposed GenClips with 7-layer visual + 4-layer audio specs
│       ├── audio/                       # Gate 4 voice auditions, SFX supplements & music cues
│       ├── takes/                       # Gate 5 generated video takes & QA Critic JSON reports
│       ├── review/                      # Gate 2/3/5 review packets & Interactive HTML Dashboards
│       │   └── G5_<TIMESTAMP>/
│       │       ├── index.html           # Interactive HTML video review with prompts & refs used
│       │       └── manifest.yaml
│       ├── assembly/                    # Gate 6 OpenTimelineIO cuts, media links & frozen snapshots
│       ├── backups/                     # Gate 6 Frozen .otio / .drt timeline backups
│       └── exports/                     # Gate 6 Versioned rendered scene cuts (SC01_v1.mp4)
│
├── post/                                # Gate 7 Master Episode / Film Assembly (master_v1.otio)
└── reviews/                             # Exported visual review markdowns & reference carousels
```
