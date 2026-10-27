"""Token-Saving Casting & Location Reference Sheet Agent (`casting_agent.py`).

Prevents wasting image/video generation tokens on characters or locations that already exist by:
1. Scanning `./generated_assets/` and `reference_registry.json` for existing Character Reference
   Sheets (`CHAR_*`), Location Reference Plates (`LOC_*`), and Master Anchor Sheets (`ANCHOR_*`).
2. Enforcing a mandatory Pre-Flight Reference Gate (`preflight_reference_sheet_check`):
   - Automatically reuses any existing character/location sheet found on disk (0 tokens spent).
   - For any missing character or location, halts generation and prompts the user/editor to either:
     (a) supply an existing reference image path/file (`import_existing_reference_sheet`), OR
     (b) explicitly confirm generating only the missing sheet (`approved_to_generate_ids`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from schemas import (
    ASSET_DIR,
    AnchorSheetSpec,
    CharacterCastSpec,
    LocationScoutSpec,
    StrictContract,
    WORKSPACE_ROOT,
    resolve_safe_workspace_path,
)

ANCHOR_DIR: Path = ASSET_DIR / "anchors"
REGISTRY_PATH: Path = ASSET_DIR / "reference_sheet_registry.json"


def _load_default_bibles() -> tuple[dict[str, CharacterCastSpec], dict[str, LocationScoutSpec], dict[str, AnchorSheetSpec]]:
    """Optionally load active project bibles if present; otherwise return empty generic dicts."""
    try:
        from projects.like_share_kill_or_die.casting_bible import (
            CAST_BIBLE,
            LOCATION_BIBLE,
            MASTER_ANCHOR_SHEETS,
        )

        return dict(CAST_BIBLE), dict(LOCATION_BIBLE), dict(MASTER_ANCHOR_SHEETS)
    except ImportError:
        return {}, {}, {}


# Standard candidate filenames for built-in or previously generated sheets
DEFAULT_KNOWN_ANCHOR_FILES: dict[str, list[str]] = {
    "ANCHOR_CAST_EP1": ["anchor_cast_ep1.jpg", "anchors/anchor_cast_ep1.jpg"],
    "ANCHOR_LOC_EP1": ["anchor_loc_ep1.jpg", "anchors/anchor_loc_ep1.jpg"],
    "ANCHOR_CAST_EP2": ["anchor_cast_ep2.jpg", "anchors/anchor_cast_ep2.jpg"],
    "ANCHOR_LOC_EP2": ["anchor_loc_ep2.jpg", "anchors/anchor_loc_ep2.jpg"],
}


class ReferenceSheetConfirmationRequiredError(RuntimeError):
    """Raised when a character or location sheet is missing and user confirmation has not been granted."""

    def __init__(self, message: str, report: CastingPreflightReport) -> None:
        super().__init__(message)
        self.report = report


class RegisteredReferenceSheet(StrictContract):
    """Metadata for an on-disk Character or Location Reference Sheet."""

    entity_id: str = Field(description="E.g. CHAR_RISHI, LOC_TERRACE, or ANCHOR_CAST_EP1.")
    entity_type: Annotated[str, Field(pattern=r"^(character|location|anchor)$")]
    display_name: str
    file_path: str
    sha256: str
    source_origin: Annotated[str, Field(pattern=r"^(existing_on_disk|user_supplied|newly_generated)$")]


class MissingReferencePrompt(StrictContract):
    """Prompt item asking the user for an existing sheet before spending tokens to generate one."""

    entity_id: str
    entity_type: Annotated[str, Field(pattern=r"^(character|location|anchor)$")]
    display_name: str
    anchor_id: str
    question_for_user: str
    suggested_generation_prompt: str


class CastingPreflightReport(StrictContract):
    """Result of auditing character and location reference sheets before generation."""

    ready_to_proceed_without_generation: bool
    tokens_saved_count: int
    reused_sheets: list[RegisteredReferenceSheet]
    missing_sheets: list[MissingReferencePrompt]
    user_action_summary: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reference_registry() -> dict[str, dict[str, Any]]:
    """Load the persistent reference sheet registry from `generated_assets/reference_sheet_registry.json`."""
    if not REGISTRY_PATH.exists():
        return {}
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_reference_registry(registry: dict[str, dict[str, Any]]) -> Path:
    """Persist the reference sheet registry to disk."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    safe_path = resolve_safe_workspace_path(REGISTRY_PATH, ASSET_DIR)
    safe_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    return safe_path


def _discover_sheet_on_disk(entity_id: str, anchor_id: str, image_name: str = "") -> Path | None:
    """Search workspace asset folders and registry for an existing reference sheet image."""
    registry = load_reference_registry()
    for key in (entity_id, anchor_id):
        if key in registry:
            recorded_path = Path(registry[key].get("file_path", ""))
            if recorded_path.exists() and recorded_path.is_file():
                return recorded_path

    candidates: list[Path] = []
    # 1. Check known mapping
    for rel in DEFAULT_KNOWN_ANCHOR_FILES.get(anchor_id, []):
        candidates.append(ASSET_DIR / rel)

    # 2. Check standard naming patterns across generated_assets/ and subdirectories
    stems = [
        entity_id.lower(),
        anchor_id.lower(),
    ]
    if image_name:
        stems.append(image_name.lower())

    search_dirs = [
        ASSET_DIR,
        ANCHOR_DIR,
        ASSET_DIR / "cast",
        ASSET_DIR / "locations",
    ]
    for sdir in search_dirs:
        if not sdir.exists():
            continue
        for stem in stems:
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                candidates.append(sdir / f"{stem}{ext}")

    for cand in candidates:
        if cand.exists() and cand.is_file() and cand.stat().st_size > 0:
            return cand.resolve()

    return None


def import_existing_reference_sheet(
    entity_id: str,
    source_path: str | Path,
    entity_type: str = "character",
    display_name: str | None = None,
) -> RegisteredReferenceSheet:
    """Import an editor-supplied character or location reference sheet so 0 generation tokens are spent."""
    src = resolve_safe_workspace_path(source_path, WORKSPACE_ROOT)
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"Supplied reference sheet does not exist: {src}")

    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    ext = src.suffix.lower() or ".jpg"
    dest = resolve_safe_workspace_path(ANCHOR_DIR / f"{entity_id.lower()}{ext}", ASSET_DIR)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)

    digest = _sha256_file(dest)
    entry = RegisteredReferenceSheet(
        entity_id=entity_id,
        entity_type=entity_type,
        display_name=display_name or entity_id,
        file_path=str(dest),
        sha256=digest,
        source_origin="user_supplied",
    )

    reg = load_reference_registry()
    reg[entity_id] = entry.model_dump()
    default_cast, default_loc, _ = _load_default_bibles()
    if entity_id in default_cast:
        reg[default_cast[entity_id].anchor_sheet_id] = entry.model_dump()
    if entity_id in default_loc:
        reg[default_loc[entity_id].anchor_plate_id] = entry.model_dump()
    save_reference_registry(reg)
    return entry


def preflight_reference_sheet_check(
    character_ids: list[str] | None = None,
    location_ids: list[str] | None = None,
    custom_characters: dict[str, CharacterCastSpec] | None = None,
    custom_locations: dict[str, LocationScoutSpec] | None = None,
) -> CastingPreflightReport:
    """Check whether required Character & Location reference sheets already exist before generating."""
    default_cast, default_loc, default_anchors = _load_default_bibles()
    cast_map = {**default_cast, **(custom_characters or {})}
    loc_map = {**default_loc, **(custom_locations or {})}
    anchor_map = dict(default_anchors)

    target_chars = character_ids if character_ids is not None else list(cast_map.keys())
    target_locs = location_ids if location_ids is not None else list(loc_map.keys())

    reused: list[RegisteredReferenceSheet] = []
    missing: list[MissingReferencePrompt] = []
    reg = load_reference_registry()
    reg_updated = False

    for cid in target_chars:
        if cid not in cast_map:
            continue
        cspec = cast_map[cid]
        anchor_obj = anchor_map.get(cspec.anchor_sheet_id)
        img_name = anchor_obj.image_name if anchor_obj else cspec.anchor_sheet_id.lower()
        found = _discover_sheet_on_disk(cid, cspec.anchor_sheet_id, img_name)
        if found is not None:
            sheet = RegisteredReferenceSheet(
                entity_id=cid,
                entity_type="character",
                display_name=cspec.name,
                file_path=str(found),
                sha256=_sha256_file(found),
                source_origin=reg.get(cid, {}).get("source_origin", "existing_on_disk"),
            )
            reused.append(sheet)
            if cid not in reg:
                reg[cid] = sheet.model_dump()
                reg_updated = True
        else:
            gen_prompt = anchor_obj.prompt if anchor_obj else cspec.immutable_physical_prompt
            missing.append(
                MissingReferencePrompt(
                    entity_id=cid,
                    entity_type="character",
                    display_name=cspec.name,
                    anchor_id=cspec.anchor_sheet_id,
                    question_for_user=(
                        f"Character Reference Sheet for '{cspec.name}' ({cid}) is not on disk yet. "
                        "Do you already have an existing character sheet image to link (saves generation tokens), "
                        "or should I generate a new reference sheet?"
                    ),
                    suggested_generation_prompt=gen_prompt,
                )
            )

    for lid in target_locs:
        if lid not in loc_map:
            continue
        lspec = loc_map[lid]
        anchor_obj = anchor_map.get(lspec.anchor_plate_id)
        img_name = anchor_obj.image_name if anchor_obj else lspec.anchor_plate_id.lower()
        found = _discover_sheet_on_disk(lid, lspec.anchor_plate_id, img_name)
        if found is not None:
            sheet = RegisteredReferenceSheet(
                entity_id=lid,
                entity_type="location",
                display_name=lspec.name,
                file_path=str(found),
                sha256=_sha256_file(found),
                source_origin=reg.get(lid, {}).get("source_origin", "existing_on_disk"),
            )
            reused.append(sheet)
            if lid not in reg:
                reg[lid] = sheet.model_dump()
                reg_updated = True
        else:
            gen_prompt = anchor_obj.prompt if anchor_obj else lspec.architectural_prompt
            missing.append(
                MissingReferencePrompt(
                    entity_id=lid,
                    entity_type="location",
                    display_name=lspec.name,
                    anchor_id=lspec.anchor_plate_id,
                    question_for_user=(
                        f"Location Reference Plate for '{lspec.name}' ({lid}) is not on disk yet. "
                        "Do you already have an existing location plate image to link (saves generation tokens), "
                        "or should I generate a new location sheet?"
                    ),
                    suggested_generation_prompt=gen_prompt,
                )
            )

    if reg_updated:
        save_reference_registry(reg)

    ready = len(missing) == 0
    if ready:
        summary = (
            f"All {len(reused)} required character and location reference sheets already exist on disk. "
            "0 generation tokens needed for reference sheets."
        )
    else:
        missing_names = ", ".join(f"{m.display_name} ({m.entity_id})" for m in missing)
        summary = (
            f"STOP & ASK USER BEFORE GENERATING: {len(reused)} sheets reused from disk, "
            f"but {len(missing)} sheets are missing [{missing_names}]. "
            "Ask the user if they have existing reference sheets to import before generating new ones."
        )

    return CastingPreflightReport(
        ready_to_proceed_without_generation=ready,
        tokens_saved_count=len(reused),
        reused_sheets=reused,
        missing_sheets=missing,
        user_action_summary=summary,
    )


def enforce_reference_sheet_gate(
    character_ids: list[str] | None = None,
    location_ids: list[str] | None = None,
    approved_to_generate_ids: set[str] | None = None,
    custom_characters: dict[str, CharacterCastSpec] | None = None,
    custom_locations: dict[str, LocationScoutSpec] | None = None,
) -> dict[str, str]:
    """Return a map of `entity_id / anchor_id -> local_file_path` or raise if unapproved sheets are missing."""
    report = preflight_reference_sheet_check(
        character_ids=character_ids,
        location_ids=location_ids,
        custom_characters=custom_characters,
        custom_locations=custom_locations,
    )
    approved = approved_to_generate_ids or set()
    unapproved_missing = [
        m for m in report.missing_sheets if m.entity_id not in approved and m.anchor_id not in approved
    ]
    if unapproved_missing:
        questions = "\n".join(f"  - {m.question_for_user}" for m in unapproved_missing)
        raise ReferenceSheetConfirmationRequiredError(
            f"Casting Agent blocked automatic token spend on {len(unapproved_missing)} missing reference sheet(s).\n"
            f"Please ask the user first:\n{questions}",
            report=report,
        )

    resolved_paths: dict[str, str] = {}
    default_cast, default_loc, _ = _load_default_bibles()
    cast_map = {**default_cast, **(custom_characters or {})}
    loc_map = {**default_loc, **(custom_locations or {})}
    for sheet in report.reused_sheets:
        resolved_paths[sheet.entity_id] = sheet.file_path
        if sheet.entity_id in cast_map:
            resolved_paths[cast_map[sheet.entity_id].anchor_sheet_id] = sheet.file_path
        if sheet.entity_id in loc_map:
            resolved_paths[loc_map[sheet.entity_id].anchor_plate_id] = sheet.file_path
    return resolved_paths


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Casting Agent CLI: Audit existing character/location reference sheets or register user sheets."
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Audit all character and location reference sheets on disk.",
    )
    parser.add_argument(
        "--import-sheet",
        nargs=3,
        metavar=("ENTITY_ID", "ENTITY_TYPE", "FILE_PATH"),
        help="Register an existing reference sheet (e.g., --import-sheet CHAR_HERO character ./my_hero.jpg).",
    )
    args = parser.parse_args()

    if args.import_sheet:
        eid, etype, fpath = args.import_sheet
        registered = import_existing_reference_sheet(eid, fpath, entity_type=etype)
        print(json.dumps(registered.model_dump(), indent=2))
    else:
        report = preflight_reference_sheet_check()
        print(json.dumps(report.model_dump(), indent=2))


if __name__ == "__main__":
    _cli()
