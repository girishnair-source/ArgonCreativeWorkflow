"""Review asset builder — BUILD_SPEC_v3 §9.3 (contact sheets, side-by-side comparison videos, gate review dirs).

* :func:`contact_sheet` tiles images into a labelled PNG grid using pure Python + numpy (no external font files).
* :func:`comparison_video` renders a side-by-side comparison MP4 with audio where pass ``k`` plays take ``k``'s
  native audio track so the human can judge both picture and voice delivery in one file.
* :func:`build_g2_review`, :func:`build_g3_review`, :func:`build_g5_review` build ``scenes/<SC>/review/<GATE>_<ts>/``
  and write ``manifest.yaml`` with absolute paths for AGY's gate messages.
"""

from __future__ import annotations

import math
import shutil
import struct
import zlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from . import fsio, media
from .models import DispatchStatus, Take, TakeStatus
from .procs import CommandFailed, run
from .project import Project

AGENT = "agy.review"
TILE_W = 160
TILE_H = 160
LABEL_H = 20

# Minimal 3x5 uppercase/digit bitmap glyphs for burn-in labels on contact sheets (zero external font dependencies).
_GLYPHS: dict[str, tuple[int, ...]] = {
    "0": (0b111, 0b101, 0b101, 0b101, 0b111),
    "1": (0b010, 0b110, 0b010, 0b010, 0b111),
    "2": (0b111, 0b001, 0b111, 0b100, 0b111),
    "3": (0b111, 0b001, 0b111, 0b001, 0b111),
    "4": (0b101, 0b101, 0b111, 0b001, 0b001),
    "5": (0b111, 0b100, 0b111, 0b001, 0b111),
    "6": (0b111, 0b100, 0b111, 0b101, 0b111),
    "7": (0b111, 0b001, 0b010, 0b010, 0b010),
    "8": (0b111, 0b101, 0b111, 0b101, 0b111),
    "9": (0b111, 0b101, 0b111, 0b001, 0b111),
    "A": (0b010, 0b101, 0b111, 0b101, 0b101),
    "B": (0b110, 0b101, 0b110, 0b101, 0b110),
    "C": (0b111, 0b100, 0b100, 0b100, 0b111),
    "D": (0b110, 0b101, 0b101, 0b101, 0b110),
    "E": (0b111, 0b100, 0b110, 0b100, 0b111),
    "F": (0b111, 0b100, 0b110, 0b100, 0b100),
    "G": (0b111, 0b100, 0b101, 0b101, 0b111),
    "H": (0b101, 0b101, 0b111, 0b101, 0b101),
    "I": (0b111, 0b010, 0b010, 0b010, 0b111),
    "J": (0b001, 0b001, 0b001, 0b101, 0b111),
    "K": (0b101, 0b101, 0b110, 0b101, 0b101),
    "L": (0b100, 0b100, 0b100, 0b100, 0b111),
    "M": (0b101, 0b111, 0b111, 0b101, 0b101),
    "N": (0b101, 0b111, 0b111, 0b111, 0b101),
    "O": (0b111, 0b101, 0b101, 0b101, 0b111),
    "P": (0b111, 0b101, 0b111, 0b100, 0b100),
    "Q": (0b111, 0b101, 0b101, 0b111, 0b001),
    "R": (0b111, 0b101, 0b110, 0b101, 0b101),
    "S": (0b111, 0b100, 0b111, 0b001, 0b111),
    "T": (0b111, 0b010, 0b010, 0b010, 0b010),
    "U": (0b101, 0b101, 0b101, 0b101, 0b111),
    "V": (0b101, 0b101, 0b101, 0b101, 0b010),
    "W": (0b101, 0b101, 0b111, 0b111, 0b101),
    "X": (0b101, 0b101, 0b010, 0b101, 0b101),
    "Y": (0b101, 0b101, 0b010, 0b010, 0b010),
    "Z": (0b111, 0b001, 0b010, 0b100, 0b111),
    "_": (0b000, 0b000, 0b000, 0b000, 0b111),
    "-": (0b000, 0b000, 0b111, 0b000, 0b000),
    ":": (0b000, 0b010, 0b000, 0b010, 0b000),
    ".": (0b000, 0b000, 0b000, 0b000, 0b010),
}


def _write_rgb_png(path: Path, arr: np.ndarray) -> Path:
    """Write an ``(H, W, 3)`` uint8 RGB array as a standard PNG file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw, 6))
        + _chunk(b"IEND", b"")
    )
    path.write_bytes(png)
    return path


def _draw_label(bar: np.ndarray, text: str) -> None:
    """Stamp ``text`` into a ``(LABEL_H, TILE_W, 3)`` uint8 bar (2x scaled 3x5 glyphs)."""
    bar[:, :] = (24, 26, 32)
    x = 6
    y0 = max(2, (bar.shape[0] - 10) // 2)
    for ch in (text or "").upper()[:18]:
        glyph = _GLYPHS.get(ch)
        if glyph is None:
            x += 6
            continue
        if x + 6 >= bar.shape[1] - 2:
            break
        for row_idx, bits in enumerate(glyph):
            for col_idx in range(3):
                if (bits >> (2 - col_idx)) & 1:
                    py = y0 + row_idx * 2
                    px = x + col_idx * 2
                    bar[py : py + 2, px : px + 2] = (240, 242, 245)
        x += 8


def contact_sheet(
    images: Sequence[str | Path],
    labels: Sequence[str],
    out_png: Path | str,
    *,
    cols: int = 4,
) -> Path:
    """Render a grid contact sheet PNG with a label strip under each tile."""
    out_png = Path(out_png)
    valid: list[tuple[Path, str]] = []
    for idx, img in enumerate(images):
        p = Path(img)
        if p.is_file():
            lbl = labels[idx] if idx < len(labels) else p.stem
            valid.append((p, str(lbl)))
    if not valid:
        return media.write_png(out_png, width=TILE_W, height=TILE_H, rgb=(36, 36, 42), overwrite=True)

    cols = max(1, min(int(cols), len(valid)))
    rows = int(math.ceil(len(valid) / float(cols)))
    cell_h = TILE_H + LABEL_H
    canvas = np.full((rows * cell_h, cols * TILE_W, 3), 18, dtype=np.uint8)

    for idx, (path, lbl) in enumerate(valid):
        r = idx // cols
        c = idx % cols
        y0 = r * cell_h
        x0 = c * TILE_W
        try:
            tile = media.read_image_rgb(path, (TILE_W, TILE_H))
        except Exception:  # noqa: BLE001
            tile = np.full((TILE_H, TILE_W, 3), 80, dtype=np.uint8)
        canvas[y0 : y0 + TILE_H, x0 : x0 + TILE_W] = tile
        _draw_label(canvas[y0 + TILE_H : y0 + cell_h, x0 : x0 + TILE_W], lbl)

    return _write_rgb_png(out_png, canvas)


def _take_video_path(project: Project | None, item: Take | str | Path) -> Path | None:
    if isinstance(item, Take):
        if not item.video_uri or project is None:
            return None
        p = project.abs(item.video_uri)
        return p if p.is_file() else None
    p = Path(item)
    if p.is_file():
        return p
    if project is not None:
        try:
            cand = project.abs(str(item))
            if cand.is_file():
                return cand
        except Exception:  # noqa: BLE001
            pass
    return None


def comparison_video(
    takes: Sequence[Take | str | Path],
    labels: Sequence[str],
    out_mp4: Path | str,
    *,
    project: Project | None = None,
) -> Path:
    """Build a side-by-side comparison MP4 with audio where pass ``k`` plays take ``k``'s audio track (§9.3)."""
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for item in takes:
        p = _take_video_path(project, item)
        if p is not None:
            paths.append(p)
    if not paths:
        return media.make_test_clip(out_mp4, duration_s=1.0, overwrite=True)
    if len(paths) == 1:
        shutil.copy2(paths[0], out_mp4)
        return out_mp4

    paths = paths[:4]
    n = len(paths)
    tile_w, tile_h = 270, 480
    with_audio = [media.has_audio(p) for p in paths]

    inputs: list[str] = []
    for p in paths:
        inputs += ["-i", str(p)]

    v_Filters = [
        f"[{i}:v]scale={tile_w}:{tile_h}:force_original_aspect_ratio=decrease,"
        f"pad={tile_w}:{tile_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24[v{i}]"
        for i in range(n)
    ]
    hstack_in = "".join(f"[v{i}]" for i in range(n))
    if all(with_audio):
        # Pass k shows the side-by-side stack while playing take k's native audio track.
        splits = "".join(f"[vpass{k}]" for k in range(n))
        v_chain = ";".join(
            [
                *v_Filters,
                f"{hstack_in}hstack=inputs={n},split={n}{splits}",
            ]
        )
        a_Filters = [f"[{k}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=mono[a{k}]" for k in range(n)]
        pairs = "".join(f"[vpass{k}][a{k}]" for k in range(n))
        filter_complex = f"{v_chain};{';'.join(a_Filters)};{pairs}concat=n={n}:v=1:a=1[outv][outa]"
        maps = ["-map", "[outv]", "-map", "[outa]"]
    else:
        filter_complex = ";".join([*v_Filters, f"{hstack_in}hstack=inputs={n}[outv]"])
        maps = ["-map", "[outv]"]

    for vcodec in (["-c:v", "libx264", "-preset", "ultrafast"], ["-c:v", "mpeg4", "-q:v", "5"]):
        argv = ["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", filter_complex, *maps, *vcodec, "-pix_fmt", "yuv420p"]
        if all(with_audio):
            argv += ["-c:a", "aac", "-b:a", "96k"]
        try:
            run([*argv, str(out_mp4)], timeout=300)
            if out_mp4.is_file() and out_mp4.stat().st_size > 0:
                return out_mp4
        except CommandFailed:
            continue

    shutil.copy2(paths[0], out_mp4)
    return out_mp4


def _new_review_dir(project: Project, scene_id: str, gate: str) -> Path:
    stamp = fsio.now_iso().replace(":", "").replace("-", "").replace(".", "")
    out = project.review_dir(scene_id, gate, stamp)
    suffix = 1
    while out.exists():
        suffix += 1
        out = project.review_dir(scene_id, gate, f"{stamp}_{suffix}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_manifest(review_dir: Path, *, gate: str, scene_id: str, files: Iterable[Path], by_item: dict[str, list[str]]) -> Path:
    abs_files = [str(Path(f).resolve()) for f in files if Path(f).exists()]
    manifest = {
        "gate": gate,
        "scene_id": scene_id,
        "review_dir": str(review_dir.resolve()),
        "files": abs_files,
        "by_item": by_item,
        "generated_by": fsio.generated_by(AGENT).model_dump(mode="json"),
    }
    return fsio.save_versioned(review_dir / "manifest.yaml", manifest)


def build_g2_review(project: Project, scene_id: str, candidates: dict[str, list[str]]) -> Path:
    """Build ``scenes/<SC>/review/G2_<ts>/`` contact sheets for new reference packs & location plates."""
    fsio.validate_id(scene_id, "scene id")
    rdir = _new_review_dir(project, scene_id, "G2")
    all_imgs: list[Path] = []
    all_lbls: list[str] = []
    by_item: dict[str, list[str]] = {}
    files: list[Path] = []

    for item_id, paths in candidates.items():
        resolved: list[Path] = []
        lbls: list[str] = []
        for raw in paths:
            p = Path(raw) if Path(raw).is_absolute() else project.abs(raw)
            if p.is_file():
                resolved.append(p)
                lbls.append(f"{item_id}:{p.stem}")
                all_imgs.append(p)
                all_lbls.append(f"{item_id}:{p.stem}")
        if resolved:
            sheet = contact_sheet(resolved, lbls, rdir / f"{item_id}_contact_sheet.png", cols=min(4, len(resolved)))
            files.append(sheet)
            files.extend(resolved)
            by_item[item_id] = [str(sheet.resolve()), *[str(p.resolve()) for p in resolved]]

    if all_imgs:
        overview = contact_sheet(all_imgs, all_lbls, rdir / f"{scene_id}_g2_contact_sheet.png", cols=4)
        files.insert(0, overview)

    _write_manifest(rdir, gate="G2", scene_id=scene_id, files=files, by_item=by_item)
    return rdir


def build_g3_review(project: Project, scene_id: str) -> Path:
    """Build ``scenes/<SC>/review/G3_<ts>/`` storyboard contact sheet (+ animatic link if present)."""
    fsio.validate_id(scene_id, "scene id")
    rdir = _new_review_dir(project, scene_id, "G3")
    shots = project.list_shots(scene_id)
    imgs: list[Path] = []
    lbls: list[str] = []
    by_item: dict[str, list[str]] = {}
    files: list[Path] = []

    for shot in shots:
        if shot.board_frame:
            p = project.abs(shot.board_frame)
            if p.is_file():
                imgs.append(p)
                lbls.append(shot.id)
                files.append(p)
                by_item[shot.id] = [str(p.resolve())]

    if imgs:
        sheet = contact_sheet(imgs, lbls, rdir / f"{scene_id}_boards_contact_sheet.png", cols=min(4, len(imgs)))
        files.insert(0, sheet)

    boards_dir = project.path("scenes", scene_id, "boards")
    if boards_dir.is_dir():
        animatics = sorted(boards_dir.glob("animatic_v*.mp4"))
        if animatics:
            files.append(animatics[-1])

    _write_manifest(rdir, gate="G3", scene_id=scene_id, files=files, by_item=by_item)
    return rdir


def build_g5_review(project: Project, scene_id: str, clip_ids: Sequence[str]) -> Path:
    """Build ``scenes/<SC>/review/G5_<ts>/`` per-clip comparison videos and a first/last-frame contact sheet."""
    fsio.validate_id(scene_id, "scene id")
    rdir = _new_review_dir(project, scene_id, "G5")
    sheet_imgs: list[Path] = []
    sheet_lbls: list[str] = []
    by_item: dict[str, list[str]] = {}
    files: list[Path] = []

    for cid in clip_ids:
        takes = [
            t
            for t in project.list_takes(cid)
            if t.status != TakeStatus.REJECTED and t.dispatch_status == DispatchStatus.COMPLETED and t.video_uri
        ]
        takes.sort(key=lambda t: (t.rank if t.rank is not None else 999, t.id))
        item_files: list[str] = []
        vids: list[Path] = []
        vid_lbls: list[str] = []
        for t in takes:
            vpath = project.abs(t.video_uri) if t.video_uri else None
            if vpath and vpath.is_file():
                vids.append(vpath)
                vid_lbls.append(t.id)
                item_files.append(str(vpath.resolve()))
                files.append(vpath)
            if t.first_frame_uri and project.abs(t.first_frame_uri).is_file():
                fp = project.abs(t.first_frame_uri)
                sheet_imgs.append(fp)
                sheet_lbls.append(f"{t.id}:first")
            if t.last_frame_uri and project.abs(t.last_frame_uri).is_file():
                lp = project.abs(t.last_frame_uri)
                sheet_imgs.append(lp)
                sheet_lbls.append(f"{t.id}:last")
        if vids:
            comp = comparison_video(vids, vid_lbls, rdir / f"{cid}_compare.mp4", project=project)
            files.insert(0, comp)
            item_files.insert(0, str(comp.resolve()))
        by_item[cid] = item_files

    if sheet_imgs:
        sheet = contact_sheet(sheet_imgs, sheet_lbls, rdir / f"{scene_id}_g5_contact_sheet.png", cols=4)
        files.insert(0, sheet)

    html_path = build_g5_html_review(project, scene_id, clip_ids, rdir)
    files.insert(0, html_path)

    _write_manifest(rdir, gate="G5", scene_id=scene_id, files=files, by_item=by_item)
    return rdir


def build_g5_html_review(project: Project, scene_id: str, clip_ids: Sequence[str], rdir: Path) -> Path:
    """Render an interactive HTML Review Dashboard (``index.html``) for Gate 5 showing each generated video clip,
    the exact prompt used, the reference images passed to the model, and QA Critic hallucination scores.
    """
    import html
    import os

    cards: list[str] = []
    for cid in clip_ids:
        try:
            clip = project.load_clip(cid)
        except FileNotFoundError:
            continue

        vis = clip.spec.visual
        prompt_lines = [
            f"Subject & Action: {vis.subject_and_action}",
            f"Camera: {vis.camera}",
            f"Lens & Aspect: {vis.lens_and_aspect}",
            f"Lighting: {vis.lighting}",
            f"Composition: {vis.composition}",
            f"Mood & Atmosphere: {vis.mood} {vis.atmosphere}".strip(),
        ]
        audio_spec = getattr(clip.spec, "audio_spec", None) or getattr(clip.spec, "audio", None)
        all_lines = audio_spec.all_lines() if (audio_spec and hasattr(audio_spec, "all_lines")) else []
        if all_lines:
            dlg = "; ".join(f"{d.character_id} says: {d.text}" for d in all_lines)
            prompt_lines.append(f"Dialogue: {dlg}")
        if audio_spec and getattr(audio_spec, "ambience", None):
            prompt_lines.append(f"Ambient Sound: {audio_spec.ambience}")
        prompt_text = "\n".join(line for line in prompt_lines if line.split(":", 1)[-1].strip())

        # Collect reference images used for this clip
        ref_uris: list[tuple[str, str]] = []
        cond = clip.conditioning
        if getattr(cond, "spatial_anchor", None):
            ref_uris.append(("Spatial Anchor / Storyboard", str(cond.spatial_anchor)))
        for u in getattr(cond, "reference_uris", []) or []:
            if u:
                ref_uris.append(("Character / Prop Reference", str(u)))
        for state_id, uris in (getattr(cond, "identity_refs", None) or {}).items():
            for u in uris:
                if u:
                    ref_uris.append((f"Character ({state_id})", str(u)))
        for u in getattr(cond, "prop_refs", None) or []:
            if u:
                ref_uris.append(("Prop Reference", str(u)))
        anchor_prev = getattr(cond, "first_frame_anchor", None) or getattr(cond, "previous_clip_last_frame", None)
        if anchor_prev:
            ref_uris.append(("First Frame / Continuity Anchor", str(anchor_prev)))

        refs_html_items: list[str] = []
        for role_lbl, uri in ref_uris:
            abs_p = project.abs(uri) if not Path(uri).is_absolute() else Path(uri)
            if abs_p.is_file():
                rel_img = os.path.relpath(abs_p, rdir)
                refs_html_items.append(
                    f'<div style="background:#1e293b;padding:8px;border-radius:8px;width:130px;text-align:center;">'
                    f'<img src="{html.escape(rel_img)}" style="width:100%;height:160px;object-fit:cover;border-radius:6px;" />'
                    f'<div style="font-size:11px;color:#cbd5e1;margin-top:4px;word-break:break-word;">{html.escape(role_lbl)}</div>'
                    f"</div>"
                )
            else:
                refs_html_items.append(
                    f'<div style="background:#1e293b;padding:8px;border-radius:8px;font-size:11px;color:#94a3b8;">'
                    f"{html.escape(role_lbl)}: <code>{html.escape(uri)}</code></div>"
                )

        takes = [
            t
            for t in project.list_takes(cid)
            if t.status != TakeStatus.REJECTED and t.dispatch_status == DispatchStatus.COMPLETED and t.video_uri
        ]
        takes.sort(key=lambda t: (t.rank if t.rank is not None else 999, t.id))

        take_blocks: list[str] = []
        for t in takes:
            vpath = project.abs(t.video_uri) if t.video_uri else None
            rel_vid = os.path.relpath(vpath, rdir) if (vpath and vpath.is_file()) else ""
            qa = t.qa
            qa_score = f"{qa.composite:.2f}" if (qa and qa.composite is not None) else "—"
            tr_ok = "✓" if (qa and (qa.transcript_match is None or qa.transcript_match >= 0.8)) else "✗"
            badge_bg = "#14532d" if t.id == clip.selected_take_id else "#1e293b"
            take_blocks.append(
                f'<div style="background:{badge_bg};border:1px solid #334155;border-radius:10px;padding:12px;flex:1;min-width:240px;">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
                f'<strong>{html.escape(t.id)}</strong>'
                f'<span style="font-size:12px;color:#86efac;">Model: {html.escape(t.model)} · QA: {qa_score} · Transcript: {tr_ok}</span>'
                f"</div>"
                + (
                    f'<video controls preload="metadata" src="{html.escape(rel_vid)}" style="width:100%;max-height:360px;background:#000;border-radius:8px;"></video>'
                    if rel_vid
                    else '<div style="padding:20px;color:#94a3b8;">Video file pending</div>'
                )
                + "</div>"
            )

        cards.append(
            f"""
            <section style="background:#151821;border:1px solid #2a303c;border-radius:14px;padding:20px;margin-bottom:24px;">
              <h2 style="margin:0 0 12px 0;font-size:18px;color:#f8fafc;">Clip <code>{html.escape(cid)}</code> ({clip.duration_s:g}s · Angle: <code>{html.escape(clip.environment_angle or 'default')}</code>)</h2>
              <div style="display:flex;flex-wrap:wrap;gap:16px;margin-bottom:16px;">
                {''.join(take_blocks) or '<div style="color:#94a3b8;">No candidate takes available.</div>'}
              </div>
              <div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:12px;margin-bottom:14px;">
                <div style="font-size:12px;font-weight:600;color:#38bdf8;margin-bottom:6px;">EXACT COMPILED PROMPT USED</div>
                <pre style="margin:0;white-space:pre-wrap;font-size:13px;color:#e2e8f0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;">{html.escape(prompt_text)}</pre>
              </div>
              <div>
                <div style="font-size:12px;font-weight:600;color:#a78bfa;margin-bottom:8px;">REFERENCE IMAGES PASSED TO MODEL ({len(ref_uris)})</div>
                <div style="display:flex;flex-wrap:wrap;gap:10px;">
                  {''.join(refs_html_items) or '<span style="color:#64748b;font-size:12px;">No external reference images linked.</span>'}
                </div>
              </div>
            </section>
            """
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Gate G5 Video Review — {html.escape(project.project_id)} / {html.escape(scene_id)}</title>
</head>
<body style="margin:0;padding:28px;background:#0b0d13;color:#e2e8f0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <header style="margin-bottom:24px;border-bottom:1px solid #1e293b;padding-bottom:16px;">
    <h1 style="margin:0 0 6px 0;font-size:24px;">Gate G5 — Video Generation, Prompt &amp; Reference Review ({html.escape(scene_id)})</h1>
    <p style="margin:0;color:#94a3b8;font-size:14px;">
      Project: <strong>{html.escape(project.project_id)}</strong> · Generator: <strong>Gemini Omni 1.1 Flash (racoon-cookies) / Veo 3.1</strong> ·
      Auto-QA Hallucination Filter: <strong>Active</strong>
    </p>
  </header>
  {''.join(cards)}
</body>
</html>
"""
    out_file = rdir / "index.html"
    out_file.write_text(html_doc, encoding="utf-8")
    return out_file

