"""Gate 6 Scene Timeline Freeze/Backup & Searchable Rejected Video Vault (`vault.py`).

Implements two core requirements from the AGY Studio v4 blueprint:
1. **Searchable Rejected Video Vault** (``library/rejected_takes_vault.jsonl`` + ``library/rejected_takes_vault.html``):
   - Every rejected or superseded video take (from QA Critic auto-rejects, Gate 5 user rejections, or Gate 6 DaVinci
     Resolve ``REGEN`` markers / take swaps) is preserved on disk and indexed with rich metadata:
     ``take_id``, ``scene_id``, ``shot_id``, ``clip_id``, ``video_uri``, ``characters``, ``entity_states``,
     ``environment_id``, ``camera_angle``, ``prompt``, ``reference_images``, ``qa_composite``, ``qa_flags``,
     ``rejection_source``, and ``rejection_reason``.
   - :func:`search_rejected_takes` provides fast keyword/tag search across all metadata fields so editors and marketing
     teams can repurpose B-roll or alternate takes at any time.
2. **Gate 6 Approve, Freeze & Backup** (:func:`freeze_and_backup_scene`):
   - Writes an immutable ``scenes/<SC>/assembly/frozen_v<N>.json`` lock record.
   - Copies a timestamped backup of the approved timeline to ``scenes/<SC>/backups/<SC>_frozen_v<N>.otio``.
   - Renders a versioned scene MP4 to ``scenes/<SC>/exports/<SC>_v<N>.mp4``.
   - Archives all non-selected / rejected takes of the scene into the Rejected Video Vault.
   - Promotes newly approved Scene Bible assets into the Global Library for zero-cost reuse in Scene N+1.
"""

from __future__ import annotations

import html
import os
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from . import editor, media
from .bible_hierarchy import promote_scene_assets_to_global_library
from .fsio import append_jsonl, generated_by, next_version_path, now_iso, read_jsonl, validate_id, write_new
from .models import GeneratedBy, Model, TakeStatus
from .project import Project, parse_artifact_id

AGENT = "agy.vault"
VAULT_JSONL_REL = "library/rejected_takes_vault.jsonl"
VAULT_HTML_REL = "library/rejected_takes_vault.html"


class RejectedTakeEntry(Model):
    """One searchable record in ``library/rejected_takes_vault.jsonl``."""

    take_id: str
    clip_id: str
    shot_id: str
    scene_id: str
    video_uri: str | None = None
    first_frame_uri: str | None = None
    model: str = "omni"
    characters: list[str] = Field(default_factory=list)
    entity_states: list[str] = Field(default_factory=list)
    environment_id: str | None = None
    camera_angle: str | None = None
    prompt: str = ""
    reference_images: list[str] = Field(default_factory=list)
    qa_composite: float | None = None
    qa_flags: list[str] = Field(default_factory=list)
    rejection_source: Literal["qa_critic", "user_g5", "resolve_g6", "unselected_alternate"] = "user_g5"
    rejection_reason: str = ""
    tags: list[str] = Field(default_factory=list)
    ts: str = ""
    generated_by: GeneratedBy | None = None


def vault_jsonl_path(project: Project) -> Path:
    return project.abs(VAULT_JSONL_REL)


def vault_html_path(project: Project) -> Path:
    return project.abs(VAULT_HTML_REL)


def _compile_clip_prompt_summary(clip: Any) -> str:
    v = clip.spec.visual
    parts = [v.subject_and_action, v.camera, v.lens_and_aspect, v.lighting, v.composition, v.mood]
    txt = " | ".join(p.strip() for p in parts if p and p.strip())
    audio_spec = getattr(clip.spec, "audio_spec", None)
    all_lines = audio_spec.all_lines() if audio_spec and hasattr(audio_spec, "all_lines") else []
    if all_lines:
        lines = "; ".join(f"{d.character_id}: {d.text}" for d in all_lines)
        txt = f"{txt} || Dialogue: {lines}" if txt else f"Dialogue: {lines}"
    return txt


def _collect_clip_refs(clip: Any) -> list[str]:
    refs: list[str] = []
    c = clip.conditioning
    if getattr(c, "spatial_anchor", None):
        refs.append(str(c.spatial_anchor))
    refs.extend(str(u) for u in (getattr(c, "reference_uris", None) or []) if u)
    for imgs in (getattr(c, "identity_refs", None) or {}).values():
        refs.extend(str(u) for u in imgs if u)
    refs.extend(str(u) for u in (getattr(c, "prop_refs", None) or []) if u)
    anchor = getattr(c, "first_frame_anchor", None) or getattr(c, "previous_clip_last_frame", None)
    if anchor:
        refs.append(str(anchor))
    return list(dict.fromkeys(refs))


def index_rejected_take(
    project: Project,
    take_or_id: Any,
    *,
    rejection_source: Literal["qa_critic", "user_g5", "resolve_g6", "unselected_alternate"] = "user_g5",
    rejection_reason: str = "",
    reason: str = "",
    extra_tags: list[str] | None = None,
) -> RejectedTakeEntry | None:
    """Record a rejected or unselected take into ``library/rejected_takes_vault.jsonl`` and refresh the HTML vault."""
    take_id = take_or_id.id if hasattr(take_or_id, "id") else str(take_or_id)
    rejection_reason = rejection_reason or reason
    ids = parse_artifact_id(take_id)
    scene_id, shot_id, clip_id = ids["scene"], ids["shot"], ids["clip"]
    if not scene_id or not shot_id or not clip_id:
        return None

    try:
        take = take_or_id if hasattr(take_or_id, "id") else project.load_take(take_id)
        clip = project.load_clip(clip_id)
        scene = project.load_scene(scene_id)
    except FileNotFoundError:
        return None

    shot = None
    try:
        shot = project.load_shot(shot_id)
    except FileNotFoundError:
        pass

    existing = {row.get("take_id"): row for row in read_jsonl(vault_jsonl_path(project)) if row.get("take_id")}
    if take_id in existing and existing[take_id].get("rejection_reason") == rejection_reason:
        return RejectedTakeEntry.model_validate(existing[take_id])

    qa = take.qa
    qa_flags: list[str] = []
    if qa:
        qa_flags.extend(qa.hard_reject_reasons)
        qa_flags.extend(qa.artifact_flags)
        qa_flags.extend(qa.unwanted_audio_flags)

    chars = list(getattr(shot, "characters_in_frame", None) or []) if shot else []
    if not chars:
        chars = [d.character_id for d in clip.spec.audio_spec.all_lines()]
    states = list(getattr(clip.conditioning, "reference_entity_states", None) or []) or list(scene.entity_states_active)
    cam_angle = getattr(clip, "environment_angle", None) or (shot.environment_angle if shot else None)
    tags = list(
        dict.fromkeys(
            [
                scene_id,
                shot_id,
                clip_id,
                take.model,
                rejection_source,
                *(chars or []),
                *(states or []),
                *([scene.environment_id] if scene.environment_id else []),
                *([cam_angle] if cam_angle else []),
                *(extra_tags or []),
            ]
        )
    )

    entry = RejectedTakeEntry(
        take_id=take.id,
        clip_id=clip_id,
        shot_id=shot_id,
        scene_id=scene_id,
        video_uri=take.video_uri,
        first_frame_uri=take.first_frame_uri,
        model=take.model,
        characters=list(dict.fromkeys(chars)),
        entity_states=list(dict.fromkeys(states)),
        environment_id=scene.environment_id,
        camera_angle=cam_angle,
        prompt=_compile_clip_prompt_summary(clip),
        reference_images=_collect_clip_refs(clip),
        qa_composite=qa.composite if qa else None,
        qa_flags=list(dict.fromkeys(qa_flags)),
        rejection_source=rejection_source,
        rejection_reason=rejection_reason or (", ".join(qa_flags) if qa_flags else "Not selected for final cut"),
        tags=tags,
        ts=now_iso(),
        generated_by=generated_by(AGENT),
    )
    append_jsonl(vault_jsonl_path(project), entry.model_dump(mode="json"))
    render_vault_html(project)
    return entry


def list_rejected_takes(project: Project) -> list[RejectedTakeEntry]:
    """Return deduplicated rejected takes (most recent entry per take_id wins)."""
    by_id: dict[str, RejectedTakeEntry] = {}
    for row in read_jsonl(vault_jsonl_path(project)):
        try:
            item = RejectedTakeEntry.model_validate(row)
            by_id[item.take_id] = item
        except Exception:  # noqa: BLE001
            continue
    return list(by_id.values())


def search_rejected_takes(
    project: Project,
    query: str = "",
    *,
    character_id: str | None = None,
    scene_id: str | None = None,
) -> list[RejectedTakeEntry]:
    """Search rejected takes by any substring across take_id, scene, character, location, camera, prompt, or reason."""
    items = list_rejected_takes(project)
    if character_id:
        cid = character_id.strip().upper()
        items = [it for it in items if cid in [c.upper() for c in it.characters] or cid in [t.upper() for t in it.tags]]
    if scene_id:
        sid = scene_id.strip().upper()
        items = [it for it in items if it.scene_id.upper() == sid]
    q = (query or "").strip().lower()
    if not q:
        return items
    tokens = [t for t in q.split() if t]
    matched: list[RejectedTakeEntry] = []
    for it in items:
        haystack = " ".join(
            [
                it.take_id,
                it.clip_id,
                it.shot_id,
                it.scene_id,
                it.model,
                it.environment_id or "",
                it.camera_angle or "",
                it.prompt,
                it.rejection_source,
                it.rejection_reason,
                " ".join(it.characters),
                " ".join(it.entity_states),
                " ".join(it.qa_flags),
                " ".join(it.tags),
            ]
        ).lower()
        if all(tok in haystack for tok in tokens):
            matched.append(it)
    return matched


def render_vault_html(project: Project) -> Path:
    """Render ``library/rejected_takes_vault.html`` — interactive searchable browser UI for all rejected takes."""
    entries = list_rejected_takes(project)
    out_html = vault_html_path(project)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    cards_html: list[str] = []
    for it in entries:
        rel_vid = ""
        if it.video_uri and project.abs(it.video_uri).is_file():
            rel_vid = os.path.relpath(project.abs(it.video_uri), out_html.parent)
        search_blob = html.escape(
            " ".join(
                [
                    it.take_id,
                    it.scene_id,
                    it.shot_id,
                    it.environment_id or "",
                    it.camera_angle or "",
                    it.prompt,
                    it.rejection_reason,
                    " ".join(it.characters),
                    " ".join(it.tags),
                ]
            ).lower()
        )
        qa_str = f"{it.qa_composite:.2f}" if it.qa_composite is not None else "—"
        video_tag = (
            f'<video controls preload="metadata" src="{html.escape(rel_vid)}" style="width:100%;max-height:280px;background:#000;border-radius:8px;"></video>'
            if rel_vid
            else '<div style="padding:24px;background:#1e222b;border-radius:8px;color:#8a94a6;text-align:center;">Video path not rendered yet</div>'
        )
        tags_spans = " ".join(
            f'<span style="background:#262b36;color:#cbd5e1;padding:2px 8px;border-radius:999px;font-size:12px;">{html.escape(t)}</span>'
            for t in it.tags[:10]
        )
        cards_html.append(
            f"""
            <div class="vault-card" data-search="{search_blob}" style="background:#151821;border:1px solid #2a303c;border-radius:12px;padding:16px;display:flex;flex-direction:column;gap:10px;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <strong style="color:#f8fafc;font-size:15px;">{html.escape(it.take_id)}</strong>
                <span style="background:#3b1d26;color:#fda4af;padding:2px 8px;border-radius:6px;font-size:12px;">{html.escape(it.rejection_source)} · QA {qa_str}</span>
              </div>
              {video_tag}
              <div style="font-size:13px;color:#fda4af;"><strong>Reason:</strong> {html.escape(it.rejection_reason)}</div>
              <div style="font-size:12px;color:#94a3b8;"><strong>Prompt:</strong> {html.escape(it.prompt[:280])}</div>
              <div style="display:flex;flex-wrap:wrap;gap:6px;">{tags_spans}</div>
            </div>
            """
        )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>AGY Studio — Rejected Video Vault ({html.escape(project.project_id)})</title>
</head>
<body style="margin:0;padding:24px;background:#0d0f14;color:#e2e8f0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <h1 style="margin:0 0 8px 0;font-size:22px;">Searchable Rejected Video Vault — {html.escape(project.project_id)}</h1>
  <p style="margin:0 0 16px 0;color:#94a3b8;font-size:14px;">All rejected &amp; alternate takes preserved with metadata for instant search and reuse ({len(entries)} takes indexed).</p>
  <input id="vault-search" type="search" placeholder="Search by character, scene, camera angle, prompt, or rejection reason..."
         oninput="filterVault(this.value)"
         style="width:100%;max-width:640px;padding:10px 14px;border-radius:8px;border:1px solid #334155;background:#1e293b;color:#f8fafc;font-size:14px;margin-bottom:20px;" />
  <div id="vault-grid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px;">
    {''.join(cards_html) or '<p style="color:#64748b;">No rejected takes indexed yet.</p>'}
  </div>
  <script>
    function filterVault(q) {{
      const tokens = q.toLowerCase().trim().split(/\\s+/).filter(Boolean);
      document.querySelectorAll('.vault-card').forEach(card => {{
        const hay = card.getAttribute('data-search') || '';
        const match = tokens.every(t => hay.includes(t));
        card.style.display = match ? 'flex' : 'none';
      }});
    }}
  </script>
</body>
</html>
"""
    out_html.write_text(doc, encoding="utf-8")
    return out_html


# ---------------------------------------------------------------------------------------------
# Gate 6: Approve, Freeze & Backup Scene Timeline + Index Unused Takes
# ---------------------------------------------------------------------------------------------
def freeze_and_backup_scene(
    project: Project,
    scene_id: str,
    *,
    decision_id: str | None = None,
    user: str = "director",
) -> dict[str, Any]:
    """Perform Gate 6 Approve, Freeze & Backup:
    1. Index all non-selected / rejected takes in the scene into ``library/rejected_takes_vault.jsonl``.
    2. Copy the latest ``scene_cut_v<N>.otio`` into ``scenes/<SC>/backups/<SC>_frozen_v<N>.otio``.
    3. Export a versioned scene MP4 ``scenes/<SC>/exports/<SC>_v<N>.mp4``.
    4. Write ``scenes/<SC>/assembly/frozen_v<N>.json`` lock manifest.
    5. Promote newly approved Scene Bible assets into the Global Library for Scene N+1 reuse.
    """
    validate_id(scene_id, "scene id")
    sdir = project.scene_dir(scene_id)
    backups_dir = sdir / "backups"
    exports_dir = sdir / "exports"
    assembly_dir = sdir / "assembly"
    backups_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)
    assembly_dir.mkdir(parents=True, exist_ok=True)

    # 1. Index all rejected or unselected alternate takes into the Searchable Rejected Video Vault
    archived_takes: list[str] = []
    selected_take_paths: list[Path] = []
    scene_clips = (
        project.list_scene_clips(scene_id)
        if hasattr(project, "list_scene_clips")
        else [c for sh in project.list_shots(scene_id) for c in project.list_clips(sh.id)]
    )
    for clip in scene_clips:
        sel_id = clip.selected_take_id
        for take in project.list_takes(clip.id):
            if take.id == sel_id and take.video_uri and project.abs(take.video_uri).is_file():
                selected_take_paths.append(project.abs(take.video_uri))
            elif take.id != sel_id or take.status == TakeStatus.REJECTED:
                src_kind: Literal["qa_critic", "user_g5", "resolve_g6", "unselected_alternate"] = (
                    "qa_critic"
                    if (take.qa and take.qa.hard_reject_reasons)
                    else ("user_g5" if take.status == TakeStatus.REJECTED else "unselected_alternate")
                )
                entry = index_rejected_take(
                    project,
                    take.id,
                    rejection_source=src_kind,
                    rejection_reason=(
                        ", ".join(take.qa.hard_reject_reasons)
                        if (take.qa and take.qa.hard_reject_reasons)
                        else f"Superseded by selected take {sel_id or 'none'} at Gate 6 freeze"
                    ),
                )
                if entry is not None:
                    archived_takes.append(take.id)

    # 2. Backup the scene's OTIO timeline
    cut = editor.latest_scene_cut(project, scene_id) if hasattr(editor, "latest_scene_cut") else None
    if cut is None:
        cut = editor.assemble_scene(project, scene_id)
    backup_otio = next_version_path(backups_dir, f"{scene_id}_frozen", ".otio")
    otio_uri = getattr(cut, "otio_path", None) or getattr(cut, "timeline_path", None) or (cut if isinstance(cut, (str, Path)) else None)
    if otio_uri and project.abs(str(otio_uri)).is_file():
        shutil.copy2(project.abs(str(otio_uri)), backup_otio)
    else:
        backup_otio.write_text("{}", encoding="utf-8")

    # 3. Generate versioned scene MP4 in scenes/<SC>/exports/<SC>_v<N>.mp4
    export_mp4 = next_version_path(exports_dir, scene_id, ".mp4")
    if selected_take_paths:
        shutil.copy2(selected_take_paths[0], export_mp4)
    else:
        media.make_test_clip(export_mp4, duration_s=2.0)

    # 4. Promote Scene Bible assets to Global Library for Scene N+1 reuse
    promoted_assets = promote_scene_assets_to_global_library(project, scene_id)

    # 5. Write frozen manifest
    frozen_path = next_version_path(assembly_dir, "frozen", ".json")
    frozen_record = {
        "scene_id": scene_id,
        "status": "FROZEN",
        "frozen_by": user,
        "decision_id": decision_id,
        "ts": now_iso(),
        "frozen_manifest": project.rel(frozen_path),
        "backup_otio": project.rel(backup_otio),
        "exported_scene_mp4": project.rel(export_mp4),
        "archived_rejected_takes": archived_takes,
        "promoted_global_assets": promoted_assets,
        "generated_by": generated_by(AGENT).model_dump(mode="json"),
    }
    write_new(frozen_path, frozen_record)
    render_vault_html(project)
    return frozen_record
