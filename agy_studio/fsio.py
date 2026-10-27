"""Atomic, append-only, path-safe file IO for the project folder (BUILD_SPEC_v3 §1.6, §3).

Rules enforced here:
* every write is atomic (temp file in the same directory + fsync + ``os.replace``),
* immutable artifacts use ``write_new`` (refuses to overwrite — new versions are new files),
* versioned records archive the previous content to ``_history/`` before being replaced,
* append-only logs (decisions.jsonl, questions.jsonl, agent calls) use ``append_jsonl`` with an exclusive lock,
* IDs are validated against ``models.ID_PATTERN`` and every path is resolved and checked to stay inside the
  project root (no traversal via IDs, pinned paths or YAML content),
* YAML is only ever read with ``yaml.safe_load`` (no arbitrary object deserialisation).
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml
from pydantic import BaseModel

from . import AGY_VERSION
from .models import ID_PATTERN, GeneratedBy

_ID_RE = re.compile(ID_PATTERN)


class PathSecurityError(ValueError):
    """Raised when a path or ID would escape the project folder."""


class ImmutableArtifactError(FileExistsError):
    """Raised when code tries to overwrite an immutable artifact."""


# ---------------------------------------------------------------------------------------------
# Time / hashing / provenance
# ---------------------------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def compact_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(_jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(obj: Any, length: int = 16) -> str:
    """Deterministic content hash (canonical JSON, sorted keys). Used for working-set and payload hashes."""
    return sha256_bytes(canonical_json(obj).encode("utf-8"))[:length]


def generated_by(
    agent: str,
    *,
    model: str | None = None,
    model_version: str | None = None,
    seed: int | None = None,
    working_set_hash: str | None = None,
    notes: str | None = None,
) -> GeneratedBy:
    return GeneratedBy(
        agent=agent,
        model=model,
        model_version=model_version,
        seed=seed,
        working_set_hash=working_set_hash,
        ts=now_iso(),
        tool_version=AGY_VERSION,
        notes=notes,
    )


# ---------------------------------------------------------------------------------------------
# IDs and path safety
# ---------------------------------------------------------------------------------------------
def validate_id(value: str, kind: str = "id") -> str:
    pat = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,95}$") if "project" in kind.lower() else _ID_RE
    if not isinstance(value, str) or not pat.match(value):
        raise PathSecurityError(f"invalid {kind} {value!r}: must match {ID_PATTERN}")
    return value


def safe_join(root: Path, *parts: str | Path) -> Path:
    """Join ``parts`` under ``root`` and guarantee the result stays inside ``root`` (after symlink resolution)."""
    root_resolved = Path(root).resolve()
    candidate = root_resolved.joinpath(*[str(p) for p in parts]).resolve()
    if candidate != root_resolved and not str(candidate).startswith(str(root_resolved) + os.sep):
        raise PathSecurityError(f"path {candidate} escapes project root {root_resolved}")
    return candidate


def to_rel(root: Path, path: Path | str) -> str:
    """Project-relative POSIX path (raises PathSecurityError if the path is outside the root)."""
    root_resolved = Path(root).resolve()
    p = Path(path)
    p = (p if p.is_absolute() else root_resolved / p).resolve()
    if p != root_resolved and not str(p).startswith(str(root_resolved) + os.sep):
        raise PathSecurityError(f"path {p} is outside project root {root_resolved}")
    return p.relative_to(root_resolved).as_posix()


def from_rel(root: Path, rel: str) -> Path:
    if Path(rel).is_absolute():
        raise PathSecurityError(f"expected a project-relative path, got absolute {rel!r}")
    return safe_join(root, rel)


# ---------------------------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------------------------
def atomic_write_bytes(path: Path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        with contextlib.suppress(OSError):
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    return path


def atomic_write_text(path: Path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def dump_yaml(data: Any) -> str:
    return yaml.safe_dump(_jsonable(data), sort_keys=False, allow_unicode=True, width=120)


def atomic_write_json(path: Path, data: Any) -> Path:
    return atomic_write_text(path, json.dumps(_jsonable(data), indent=2, ensure_ascii=False) + "\n")


def atomic_write_yaml(path: Path, data: Any) -> Path:
    return atomic_write_text(path, dump_yaml(data))


def write_new(path: Path, data: bytes | str | dict | list | BaseModel) -> Path:
    """Write an immutable artifact. Refuses to overwrite an existing file (new versions are new files)."""
    path = Path(path)
    if path.exists():
        raise ImmutableArtifactError(f"refusing to overwrite immutable artifact {path}")
    if isinstance(data, bytes):
        return atomic_write_bytes(path, data)
    if isinstance(data, str):
        return atomic_write_text(path, data)
    if path.suffix in (".yaml", ".yml"):
        return atomic_write_yaml(path, data)
    return atomic_write_json(path, data)


def next_version_path(directory: Path, stem: str, suffix: str) -> Path:
    """``<dir>/<stem>_v<n><suffix>`` with n = 1 + highest existing version."""
    directory = Path(directory)
    pat = re.compile(rf"^{re.escape(stem)}_v(\d+){re.escape(suffix)}$")
    highest = 0
    if directory.exists():
        for p in directory.iterdir():
            m = pat.match(p.name)
            if m:
                highest = max(highest, int(m.group(1)))
    return directory / f"{stem}_v{highest + 1}{suffix}"


def save_versioned(path: Path, data: Any) -> Path:
    """Atomically replace ``path`` after archiving its previous content into ``<dir>/_history/``.

    Used for mutable records (bible YAML, shot/clip YAML) so that no prior version is ever lost.
    """
    path = Path(path)
    if path.exists():
        hist_dir = path.parent / "_history"
        hist_dir.mkdir(parents=True, exist_ok=True)
        archived = next_version_path(hist_dir, path.stem, path.suffix)
        atomic_write_bytes(archived, path.read_bytes())
    if path.suffix in (".yaml", ".yml"):
        return atomic_write_yaml(path, data)
    return atomic_write_json(path, data)


# ---------------------------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------------------------
def read_yaml(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_record(path: Path) -> Any:
    return read_yaml(path) if Path(path).suffix in (".yaml", ".yml") else read_json(path)


# ---------------------------------------------------------------------------------------------
# Append-only JSONL with an exclusive lock
# ---------------------------------------------------------------------------------------------
@contextlib.contextmanager
def locked(path: Path) -> Iterator[None]:
    """Exclusive advisory lock on ``<path>.lock`` (serialises concurrent writers across processes)."""
    lock_path = Path(path).with_name(Path(path).name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def append_jsonl(path: Path, record: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_jsonable(record), ensure_ascii=False, sort_keys=False) + "\n"
    with locked(path):
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            continue  # a torn final line from a killed writer is ignored, never fatal
    return out


# ---------------------------------------------------------------------------------------------
# Integrity hashes shared by linter / shot generator (acceptance criteria 1 and 9)
# ---------------------------------------------------------------------------------------------
LINT_HASH_FIELDS = frozenset({
    "spec", "conditioning", "duration_s", "lines", "model_target", "performance_state",
    "environment_angle", "camera_side", "anchor_from_clip_id",
})


def clip_lint_hash(clip: BaseModel) -> str:
    """Hash of everything the pre-flight linter inspects on a GenClip.

    The linter stores it in ``LintReport.spec_hash`` (computed AFTER auto-fixes). The Shot Generator recomputes
    it right before dispatch and refuses to send anything whose hash differs — so no payload can ever be
    dispatched without a passing lint of exactly that spec (acceptance criterion 1).
    """
    return stable_hash(clip.model_dump(mode="json", include=set(LINT_HASH_FIELDS)), length=24)


def compute_payload_hash(root: Path, *, model_id: str, provider_model: str, bundle: BaseModel,
                         seed: int | None) -> str:
    """Content-addressed dispatch hash: prompt + settings + sha256 of every referenced media file.

    Identical work always hashes identically, so a resumed run finds it in ``state.dispatched_payloads`` and
    never dispatches it twice (acceptance criterion 9).
    """
    data = bundle.model_dump(mode="json")
    rels = [*data.get("reference_images", []), data.get("first_frame"), data.get("last_frame"),
            *data.get("voice_reference_audio", [])]
    files: dict[str, str] = {}
    for rel in rels:
        if not rel:
            continue
        p = from_rel(root, rel)
        files[rel] = sha256_file(p) if p.is_file() else "missing"
    return stable_hash({"model_id": model_id, "provider_model": provider_model, "bundle": data,
                        "seed": seed, "files": files}, length=24)
