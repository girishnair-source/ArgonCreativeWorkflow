"""AGY chat-command toolkit (BUILD_SPEC_v3 §9.4) — ``.venv/bin/python -m agy_studio <command> ...``.

Antigravity runs these commands on the editor's behalf (Zero-Touch Creative Editor Mode); the editor only
talks to AGY in chat and works in DaVinci Resolve. Every command prints chat-ready plain language. Problems are
printed as one ``PROBLEM:`` line (exit code 2) instead of a stack trace; set ``AGY_DEBUG=1`` to see tracebacks.

Examples::

    python -m agy_studio status
    python -m agy_studio tick
    python -m agy_studio gate G5 SC01 'C01:T2, C02:regen "slower delivery", C03:ok'
    python -m agy_studio regen SC01_SH02_C01 "less head movement"
    python -m agy_studio approve SC01
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

GATES = ("G1", "G2", "G3", "G4", "G5", "G6", "G7")
SCENE_GATES = ("G2", "G3", "G4", "G5", "G6")
MANUAL_MAX_BYTES = 2 * 1024 * 1024


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def find_workspace(start: Path | None = None) -> Path:
    """Workspace root: $AGY_WORKSPACE, else the nearest parent holding ``productions/`` or ``agy_studio/``."""
    env = os.environ.get("AGY_WORKSPACE")
    if env:
        return Path(env).expanduser().resolve()
    here = (start or Path.cwd()).resolve()
    for p in (here, *here.parents):
        if (p / "productions").is_dir() or (p / "agy_studio").is_dir():
            return p
    return Path(__file__).resolve().parents[1]


def _load_manual(path: str | None) -> Any:
    """Agent output authored by AGY itself (manual agent mode): a YAML or JSON file."""
    if not path:
        return None
    import yaml

    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    if p.stat().st_size > MANUAL_MAX_BYTES:
        raise ValueError(f"{path} is larger than {MANUAL_MAX_BYTES // 1024} KB")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _pairs(items: list[str], what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in items:
        key, sep, value = raw.partition("=")
        if not sep or not key.strip() or not value.strip():
            raise ValueError(f"{what} must look like role=/path/to/file (got {raw!r})")
        out[key.strip()] = value.strip()
    return out


def _join(words: list[str] | None) -> str:
    return " ".join(words or []).strip()


# ---------------------------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agy_studio", description="AGY film production toolkit (BUILD_SPEC_v3)")
    p.add_argument("--workspace", help="workspace root (default: auto-detect)")
    p.add_argument("--project", help="project id under productions/ (default: most recent)")
    p.add_argument("--fake", action="store_true", help="use fake adapters (no paid generation, no Resolve)")
    p.add_argument("--manual", action="store_true", help="agents run in manual mode (AGY authors their output)")
    p.add_argument("--user", default="director", help="who is deciding (recorded on every Decision)")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="command")

    def add(name: str, help_text: str, aliases: tuple[str, ...] = ()) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help_text, aliases=list(aliases))

    c = add("init", "create a new production folder (locks 1 of the 3 studio modes: director | hybrid | autopilot)")
    c.add_argument("project_id")
    c.add_argument("title", nargs="+")
    c.add_argument("--mode", dest="control_mode", choices=("director", "hybrid", "autopilot"), default="director",
                   help="Operating mode: director (human gate at G1-G6), hybrid (auto G2/G4 reuse), autopilot (G1 straight to G6)")
    c.add_argument("--format", dest="content_format",
                   choices=("microdrama_9_16", "episode_16_9", "feature_film", "marketing_spot", "commercial_ad"),
                   default=None)
    c.add_argument("--language", default="hi")
    c.add_argument("--aspect", choices=("9:16", "16:9"), default="9:16")
    c.add_argument("--fps", type=int, default=24)

    c = add("status", "project / scene status")
    c.add_argument("scene", nargs="?")

    c = add("tick", "one orchestration pass (round-trip poll, regen queue, every scene)")
    c.add_argument("--repeat", type=int, default=1)
    c.add_argument("--interval", type=float, default=30.0)
    c.add_argument("--no-poll", action="store_true")

    c = add("advance", "run one scene's automatic steps until a gate or blocker")
    c.add_argument("scene")

    c = add("resume", "resume a paused scene, or (no scene) resume everything after a pause or crash")
    c.add_argument("scene", nargs="?")

    c = add("pause", "pause a scene or everything")
    c.add_argument("scene", nargs="?")

    c = add("gate", "answer an open gate, e.g. gate G5 SC01 'C01:T2, C02:ok'", aliases=("respond",))
    c.add_argument("gate", type=str.upper, choices=GATES)
    c.add_argument("rest", nargs="+", help="[SCENE] reply text")

    c = add("approve", "G6: approve a scene (same as the APPROVED marker in Resolve)")
    c.add_argument("scene")

    c = add("regen", "regenerate one clip with an optional steer")
    c.add_argument("clip")
    c.add_argument("steer", nargs="*")

    c = add("alternatives", "N alternative specs for a shot or clip")
    c.add_argument("target")
    c.add_argument("n", type=int)
    c.add_argument("differ_in", nargs="*")
    c.add_argument("--manual-file")

    c = add("choose-alt", "pick one of the alternatives offered for a shot")
    c.add_argument("shot")
    c.add_argument("label")

    c = add("strategy", "switch a scene to another proposed coverage strategy")
    c.add_argument("scene")
    c.add_argument("label")

    c = add("lock", "lock a Bible record, shot, take or the STYLE_LOCK")
    c.add_argument("artifact")

    c = add("confirm", "confirm a Vision-tier change after its fan-out was shown")
    c.add_argument("artifact")

    c = add("model", "set a scene's video model (omni or veo)")
    c.add_argument("scene")
    c.add_argument("model_id")

    c = add("pin-model", "pin a registry model to a provider model/version")
    c.add_argument("model_id")
    c.add_argument("provider_model")

    c = add("decide", "set an open decision, e.g. decide OD3 off")
    c.add_argument("key")
    c.add_argument("value")

    c = add("ws", "show the scene's working set", aliases=("working-set",))
    c.add_argument("scene")

    c = add("pin", "pin an extra reference file into a scene's working set")
    c.add_argument("scene")
    c.add_argument("path")
    c.add_argument("why", nargs="+")

    c = add("bible", "show the full human-readable Global Bible (default), a Scene Bible (SC01), or one record")
    c.add_argument("artifact", nargs="?", default="GLOBAL")

    c = add("voice", "show a character's voice profile and anchor")
    c.add_argument("character")

    c = add("questions", "open blocking questions")
    c.add_argument("scene", nargs="?")

    c = add("answer", "answer a blocking question")
    c.add_argument("qid")
    c.add_argument("text", nargs="+")

    c = add("prefs", "standing preferences learned from steers/rejects")
    c.add_argument("scope", nargs="?")

    c = add("budget", "spend vs caps")
    c.add_argument("scene", nargs="?")

    c = add("raise-cap", "raise a scene's budget cap (answer to an overrun question)")
    c.add_argument("scene")
    c.add_argument("amount", type=float)

    c = add("handoff", "where a scene's timeline is and who owns it")
    c.add_argument("scene")

    add("master", "assemble MASTER from human-approved scene timelines and open G7")

    c = add("ingest", "ingest a screenplay into the project")
    c.add_argument("script")
    c.add_argument("--episode", type=int)

    add("g1", "open G1 (Bible, state timeline, StyleLock, voice profiles, schedule)")

    c = add("design", "character/location design: status | generate [ITEM..] | import ITEM role=path..")
    c.add_argument("scene")
    c.add_argument("action", choices=("status", "generate", "import"))
    c.add_argument("args", nargs="*")
    c.add_argument("--steer", default="")

    c = add("direct", "run Director/Cinematographer/Storyboard for a scene")
    c.add_argument("scene")
    c.add_argument("--manual-file")
    c.add_argument("--only", help="comma-separated shot ids")
    c.add_argument("--steer", default="")

    c = add("audio", "plan deliveries, ambience and music cues for a scene")
    c.add_argument("scene")
    c.add_argument("--manual-file")

    c = add("anchor", "set or re-anchor a character's voice to an approved take")
    c.add_argument("character")
    c.add_argument("take")

    c = add("pack-change", "propose new images for a LOCKED pack (shows fan-out; needs confirm)")
    c.add_argument("state_id")
    c.add_argument("files", nargs="+", help="role=/path/to/image")
    c.add_argument("--why", default="")

    add("poll", "read Resolve round-trip events now (markers, take swaps, approvals)")

    c = add("unblock", "clear a scene's blockers after the cause was fixed")
    c.add_argument("scene")

    c = add("decisions", "recent decisions")
    c.add_argument("n", nargs="?", type=int, default=15)
    c.add_argument("--scene")

    add("registry", "model capability registry (durations, speakers, features)")

    c = add("reanchor", "Gate 3: re-anchor a shot keyframe from a 3D/Blender viewport screenshot + Bible refs")
    c.add_argument("shot")
    c.add_argument("screenshot")
    c.add_argument("--steer", default="")

    c = add("search-takes", "search the Rejected Video Vault (library/rejected_takes_vault.jsonl)")
    c.add_argument("query", nargs="*")
    c.add_argument("--character")
    c.add_argument("--scene")

    c = add("export-bundle", "export the locked Shareable Solution Bundle to a destination folder")
    c.add_argument("dest")
    c.add_argument("--sample-project", help="optional sample production id to include (e.g. LFKOD_S01)")

    c = add("mode", "set project control mode (director | hybrid | autopilot) and optional content format",
            aliases=("set-mode",))
    c.add_argument("control_mode", choices=("director", "hybrid", "autopilot"))
    c.add_argument("--format", dest="content_format")

    c = add("set3d", "3D virtual set: build the grey-box, open the camera viewer, style saved cameras")
    s3 = c.add_subparsers(dest="action", required=True)
    b = s3.add_parser("build", help="author the grey-box set for an environment (Set Designer agent)")
    b.add_argument("env")
    b.add_argument("--steer", default="", help="editor direction for the designer")
    b.add_argument("--manual-file", help="layout AGY authored itself (YAML/JSON) instead of calling the LLM")
    b.add_argument("--fresh", action="store_true", help="drop previously saved cameras")
    v = s3.add_parser("serve", help="open the browser viewer (orbit/fly, focal length, save + render cameras)")
    v.add_argument("env", nargs="?")
    v.add_argument("--port", type=int, default=8765)
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--open", action="store_true", help="open the default browser")
    s3.add_parser("status", help="which environments have a set, their cameras and renders")
    ca = s3.add_parser("cameras", help="list the saved cameras of a set")
    ca.add_argument("env")
    st = s3.add_parser("stylize", help="repaint a rendered camera in the locked style -> angle cam:<NAME>")
    st.add_argument("env")
    st.add_argument("camera")
    st.add_argument("--steer", default="")
    st.add_argument("--with-characters", action="store_true", help="keep anonymous silhouettes where mannequins stand")
    st.add_argument("--no-promote", action="store_true", help="only write the candidate, do not register the angle")
    sh = s3.add_parser("show", help="print the set layout JSON")
    sh.add_argument("env")
    eb = s3.add_parser("export-blender", help="export Wavefront .obj/.mtl, .gltf, and open_in_blender.py for a 3D set")
    eb.add_argument("env")

    c = add("draft-finalize", "run Two-Tier 360p Draft -> 3x3 Contact Sheet + Critic QA -> 4K Finalize (Omni 1.5 Flash)")
    c.add_argument("scene")
    c.add_argument("clip")
    c.add_argument("--prompt", required=True, help="video generation prompt (reused identically for 360p and 4K)")
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--draft-only", action="store_true", help="only run 360p draft + 3x3 contact sheet critic")

    c = add("contact-sheet", "extract a 3x3 frame contact sheet PNG from a video take")
    c.add_argument("video_path")
    c.add_argument("--output", help="optional output PNG path")

    c = add("slate", "manage Two-Tier SceneSlate (rough cut + sidechain ducking) and promote to timeline")
    sl = c.add_subparsers(dest="action", required=True)
    sl_p = sl.add_parser("promote", help="promote scene slate.json to timeline_promoted.json")
    sl_p.add_argument("scene")
    sl_p.add_argument("--require-4k", action="store_true")

    c = add("timeline-git", "Git-style version control for scene/project timelines (branch, commit, undo, redo, pr, merge)")
    tg = c.add_subparsers(dest="action", required=True)
    tg.add_parser("head", help="show current timeline branch HEAD commit")
    tg_b = tg.add_parser("branch", help="create and checkout a timeline branch")
    tg_b.add_argument("name")
    tg.add_parser("undo", help="undo the last timeline commit")
    tg.add_parser("redo", help="redo the previously undone timeline commit")
    return p


# ---------------------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------------------
def _registry_text() -> str:
    from .registry import default_registry

    lines = []
    for m in default_registry().all():
        bits = [f"{m.id} ({m.kind.value}{'' if m.enabled else ', disabled'})", f"→ {m.provider_model}"]
        if m.max_duration_s:
            bits.append(f"≤{m.max_duration_s:g}s")
        if m.supports.max_speakers_per_clip:
            bits.append(f"speakers≤{m.supports.max_speakers_per_clip}")
        if m.supports.max_reference_images:
            bits.append(f"refs≤{m.supports.max_reference_images}")
        feats = [f for f in ("native_audio", "reference_to_video", "first_frame", "extend_video",
                             "voice_reference_audio") if m.supports_feature(f)]
        if feats:
            bits.append("+" + ",".join(feats))
        lines.append(" ".join(bits))
    return "\n".join(lines)


def _init(ws: Path, a: argparse.Namespace) -> str:
    from .bible_hierarchy import build_global_context_bible
    from .generic_bible import ContentFormat, ControlMode
    from .models import ProjectConfig, ProjectSettings
    from .project import Project

    w, h = (1080, 1920) if a.aspect == "9:16" else (1920, 1080)
    title = _join(a.title)
    cfg = ProjectConfig(project_id=a.project_id, title=title, language=a.language,
                        settings=ProjectSettings(fps=a.fps, width=w, height=h, aspect_ratio=a.aspect))
    proj = Project.create(Project.productions_dir(ws), a.project_id, title, language=a.language, config=cfg)
    mode_val = ControlMode(getattr(a, "control_mode", "director") or "director")
    default_fmt = "microdrama_9_16" if a.aspect == "9:16" else "episode_16_9"
    fmt_val = ContentFormat(getattr(a, "content_format", None) or default_fmt)
    build_global_context_bible(proj, control_mode=mode_val, content_format=fmt_val, save=True)
    return (f"Created {proj.project_id} “{title}” ({a.aspect}, {a.fps} fps, "
            f"mode={mode_val.value}, format={fmt_val.value}) at {proj.root}")


def run(argv: list[str] | None = None) -> str:
    a = build_parser().parse_args(argv)
    ws = Path(a.workspace).expanduser().resolve() if a.workspace else find_workspace()
    if a.cmd == "init":
        return _init(ws, a)
    if a.cmd == "registry":
        return _registry_text()
    if a.cmd == "export-bundle":
        from .generic_bible import SHAREABLE_SOLUTION_PATHS, export_shareable_bundle

        out_dest = export_shareable_bundle(ws, Path(a.dest).expanduser(), include_sample_project=a.sample_project)
        return (f"Exported shareable bundle to {out_dest} ({len(SHAREABLE_SOLUTION_PATHS)} core paths"
                + (f" + sample project {a.sample_project}" if a.sample_project else "") + ")")

    from .orchestrator import Studio

    studio = Studio.open(ws, a.project, fake=True if a.fake else None,
                         agent_mode="manual" if a.manual else None, user=a.user)
    s = studio
    cmd = a.cmd
    table: dict[str, Callable[[], Any]] = {
        "status": lambda: s.status(a.scene),
        "advance": lambda: s.advance(a.scene).text() or f"{a.scene}: nothing to do right now",
        "pause": lambda: s.pause(a.scene, user=a.user),
        "approve": lambda: s.approve_scene(a.scene, user=a.user),
        "regen": lambda: s.regen(a.clip, _join(a.steer), user=a.user),
        "choose-alt": lambda: s.choose_alternative(a.shot, a.label),
        "strategy": lambda: s.switch_strategy(a.scene, a.label),
        "lock": lambda: s.lock(a.artifact, user=a.user),
        "confirm": lambda: s.confirm(a.artifact, user=a.user),
        "model": lambda: s.set_model(a.scene, a.model_id, user=a.user),
        "pin-model": lambda: s.pin_model(a.model_id, a.provider_model, user=a.user),
        "decide": lambda: s.set_open_decision(a.key, a.value, user=a.user),
        "ws": lambda: s.show_working_set(a.scene),
        "working-set": lambda: s.show_working_set(a.scene),
        "pin": lambda: s.pin(a.scene, a.path, _join(a.why), user=a.user),
        "bible": lambda: s.bible(a.artifact),
        "voice": lambda: s.voice(a.character),
        "questions": lambda: s.questions_text(a.scene),
        "answer": lambda: s.answer(a.qid, _join(a.text)),
        "prefs": lambda: s.prefs_text(a.scope),
        "budget": lambda: s.budget_text(a.scene),
        "raise-cap": lambda: s.raise_budget(a.scene, a.amount, user=a.user),
        "handoff": lambda: s.handoff_text(a.scene),
        "master": s.master,
        "ingest": lambda: s.ingest(Path(a.script).expanduser(), episode=a.episode),
        "g1": s.open_g1,
        "anchor": lambda: s.set_voice_anchor(a.character.upper(), a.take),
        "pack-change": lambda: s.propose_pack_change(a.state_id, _pairs(a.files, "pack files"), why=a.why),
        "poll": lambda: "\n".join(s.poll()) or "No new Resolve events.",
        "unblock": lambda: s.unblock(a.scene),
        "decisions": lambda: s.decisions_text(a.n, a.scene),
        "reanchor": lambda: s.reanchor_shot(a.shot.upper(), a.screenshot, steer=a.steer, user=a.user),
        "search-takes": lambda: s.search_vault(_join(a.query), character_id=a.character, scene_id=a.scene),
        "mode": lambda: s.set_control_mode(a.control_mode, content_format=a.content_format, user=a.user),
        "set-mode": lambda: s.set_control_mode(a.control_mode, content_format=a.content_format, user=a.user),
    }
    if cmd in table:
        return str(table[cmd]())
    if cmd == "tick":
        outs = []
        for i in range(max(1, a.repeat)):
            if i:
                time.sleep(max(0.0, a.interval))
            outs.append(s.tick(poll=not a.no_poll))
        return "\n".join(outs)
    if cmd == "resume":
        if a.scene:
            return s.unpause(a.scene, user=a.user) + "\n" + s.advance(a.scene).text()
        lines = []
        if s.state.is_paused(None):
            lines.append(s.unpause(None, user=a.user))
        lines.append(s.resume())
        return "\n".join(x for x in lines if x)
    if cmd in ("gate", "respond"):
        rest = list(a.rest)
        scene = None
        if a.gate in SCENE_GATES:
            if len(rest) < (1 if a.gate == "G6" else 2):
                raise ValueError(f"{a.gate} needs a scene and your reply, e.g. gate {a.gate} SC01 'approve all'")
            scene = rest.pop(0).upper()
        return s.respond(a.gate, scene, _join(rest) or "approve all", user=a.user)
    if cmd == "alternatives":
        return s.alternatives(a.target, a.n, _join(a.differ_in), manual_output=_load_manual(a.manual_file))
    if cmd == "design":
        if a.action == "status":
            missing = s.design_missing(a.scene)
            cands = s.design_candidates(a.scene)
            lines = [f"{a.scene} design — still needed: " + (", ".join(
                f"{k} ({', '.join(v)})" for k, v in missing.items()) or "nothing")]
            for item, roles in cands.items():
                lines.append(f"  candidate {item}: " + ", ".join(f"{r}={p}" for r, p in roles.items()))
            return "\n".join(lines)
        if a.action == "generate":
            return s.design_generate(a.scene, a.args or None, steer=a.steer)
        if not a.args:
            raise ValueError("design SC import needs ITEM role=/path ...")
        return s.design_import(a.scene, a.args[0], _pairs(a.args[1:], "design files"))
    if cmd == "direct":
        only = [x.strip() for x in a.only.split(",") if x.strip()] if a.only else None
        manual = _load_manual(a.manual_file)
        return s.direct(a.scene, mode="manual" if manual is not None else None, manual_output=manual,
                        only=only, steer=a.steer)
    if cmd == "audio":
        manual = _load_manual(a.manual_file)
        return "\n".join(s.plan_audio(a.scene, mode="manual" if manual is not None else None,
                                      manual_output=manual))
    if cmd == "set3d":
        return _set3d(s, a)
    if cmd == "contact-sheet":
        from .harness_v5 import build_3x3_contact_sheet

        out = build_3x3_contact_sheet(Path(a.video_path), Path(a.output) if a.output else None)
        return f"3x3 Contact Sheet generated: {out}"
    if cmd == "draft-finalize":
        from .harness_v5 import DraftFinalizeController

        ctl = DraftFinalizeController(s.project.scene_dir(a.scene.upper()))
        rec = ctl.generate_draft_360p(a.clip.upper(), a.prompt, seed=a.seed)
        if not a.draft_only and rec.critic_report and rec.critic_report.status == "accepted":
            rec = ctl.finalize_4k(rec)
        return (
            f"{rec.clip_id} ({rec.provider_model}): stage={rec.stage} "
            f"draft={rec.draft_video_path} contact_3x3={rec.draft_contact_sheet_path} "
            f"final_4k={rec.final_4k_video_path}"
        )
    if cmd == "slate":
        from .harness_v5 import SlateController

        ctl = SlateController(s.project.scene_dir(a.scene.upper()))
        promoted = ctl.promote_to_timeline(require_4k_clips=a.require_4k)
        return f"Promoted {promoted.scene_id} slate -> timeline ({len(promoted.clips)} clips, token={promoted.state_token})"
    if cmd == "timeline-git":
        from .harness_v5 import TimelineGitRepo

        repo = TimelineGitRepo(s.project.root)
        if a.action == "head":
            h = repo.current_head()
            return f"HEAD: {h.commit_id} ({h.branch}) — {h.message}" if h else "No commits yet on current branch."
        if a.action == "branch":
            bname = repo.create_branch(a.name)
            return f"Switched to timeline branch {bname}"
        if a.action == "undo":
            c = repo.undo()
            return f"Undid to commit {c.commit_id}: {c.message}"
        if a.action == "redo":
            c = repo.redo()
            return f"Redid to commit {c.commit_id}: {c.message}"
    raise ValueError(f"unknown command {cmd}")


def _set3d(s: Any, a: argparse.Namespace) -> str:
    from .set3d import designer, store
    from .set3d.blender_export import export_blender_scene

    proj = s.project
    if a.action == "build":
        manual = _load_manual(a.manual_file)
        layout = designer.build_set(proj, a.env.upper(), mode="manual" if manual is not None else "llm",
                                    manual_output=manual, steer=a.steer, fake=s.fake, keep_cameras=not a.fresh)
        try:
            export_blender_scene(proj, layout.environment_id)
        except Exception:  # noqa: BLE001
            pass
        return (f"3D set for {layout.environment_id} ready (v{layout.version}): {len(layout.primitives)} blocks, "
                f"{len(layout.lights)} lights, {len(layout.mannequins)} mannequins.\n"
                f"Open it with: set3d serve {layout.environment_id}\n{layout.notes}".rstrip())
    if a.action == "serve":
        from .set3d.server import serve

        env = a.env.upper() if a.env else None
        if env and not store.has_set(proj, env):
            raise FileNotFoundError(f"{env} has no 3D set yet — run `set3d build {env}` first")
        if not store.list_sets(proj):
            raise FileNotFoundError("no 3D sets yet — run `set3d build <ENV_ID>` first")
        serve(proj, host=a.host, port=a.port, env_id=env, open_browser=a.open)
        return ""
    if a.action == "status":
        envs = store.list_sets(proj)
        if not envs:
            return "No 3D sets yet. Build one with `set3d build <ENV_ID>` once the environment exists in the Bible."
        return "\n".join(store.load_set(proj, e).summary() for e in envs)
    if a.action == "cameras":
        lay = store.load_set(proj, a.env.upper())
        if not lay.cameras:
            return f"{lay.environment_id}: no saved cameras yet — open `set3d serve {lay.environment_id}` and save some."
        return lay.summary()
    if a.action == "show":
        return store.load_set(proj, a.env.upper()).model_dump_json(indent=2)
    if a.action == "export-blender":
        res = export_blender_scene(proj, a.env.upper())
        return (f"Blender & 3D assets exported for {res['environment_id']}:\n"
                f"  - OBJ: {res['obj']}\n  - MTL: {res['mtl']}\n  - glTF: {res['gltf']}\n"
                f"  - Blender script: {res['blender_script']} (`blender --python {proj.abs(res['blender_script'])}`)")
    if a.action == "stylize":
        from .set3d.stylepass import stylize_camera

        res = stylize_camera(proj, a.env.upper(), a.camera, image_adapter=s._image_adapter(), steer=a.steer,
                             with_characters=a.with_characters, promote=not a.no_promote, user=s.user)
        lines = [f"Styled reference for {a.env.upper()} camera {a.camera}: {proj.abs(res['styled'])}"]
        if res["library"]:
            lines.append(f"Registered as angle cam:{a.camera} → {res['library']} (use environment_angle: cam:{a.camera})")
        else:
            lines.append("Candidate only (not registered). Re-run without --no-promote to register it.")
        return "\n".join(lines)
    raise ValueError(f"unknown set3d action {a.action}")


def main(argv: list[str] | None = None) -> int:
    try:
        out = run(argv)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("Stopped. Nothing is lost — `resume` continues from the last finished step.")
        return 130
    except Exception as exc:  # noqa: BLE001 — chat-facing: one plain line, no stack trace
        from .adapters.auth import AuthError, describe_for_chat

        if os.environ.get("AGY_DEBUG"):
            traceback.print_exc()
        if isinstance(exc, AuthError):
            # Fail fast + tell the human exactly what to run. Never fall back to offline / fake generation.
            print(describe_for_chat(exc))
            return 3
        msg = str(exc).strip() or exc.__class__.__name__
        print(f"PROBLEM: {msg} [{exc.__class__.__name__}]")
        return 2
    if out:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
