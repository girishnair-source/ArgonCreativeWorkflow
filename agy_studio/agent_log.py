"""Agent-call log (BUILD_SPEC_v3 §12 / acceptance criterion 5).

Every agent invocation is written as an immutable JSON record ``logs/agent_calls/<ts>_<agent>_<scene>.json``
holding the full inputs (working-set hash + task + feedback + the exact prompt), the outputs and provenance.
Together with ``scenes/<SC>/working_set.yaml`` this makes any agent input reproducible.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .fsio import ImmutableArtifactError, append_jsonl, generated_by, now_iso, read_json, write_new
from .project import Project

_SAFE = re.compile(r"[^a-z0-9_]+")


def _slug(text: str) -> str:
    return _SAFE.sub("_", (text or "agent").lower()).strip("_")[:48] or "agent"


def log_agent_call(
    project: Project,
    *,
    agent: str,
    scene_id: str | None,
    working_set_hash: str | None,
    task: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    model: str | None = None,
    mode: str = "llm",
    prompt: str | None = None,
    duration_s: float | None = None,
    error: str | None = None,
) -> Path:
    """Write one immutable agent-call record and return its path."""
    scene_part = scene_id if scene_id and re.fullmatch(r"[A-Z][A-Z0-9_]{0,31}", scene_id) else "GLOBAL"
    record = {
        "ts": now_iso(),
        "agent": agent,
        "scene_id": scene_id,
        "working_set_hash": working_set_hash,
        "mode": mode,  # llm | manual (AGY authored the proposal itself) | deterministic
        "model": model,
        "task": task,
        "inputs": inputs,
        "prompt": prompt,
        "outputs": outputs,
        "duration_s": duration_s,
        "error": error,
        "generated_by": generated_by(agent, model=model, working_set_hash=working_set_hash).model_dump(mode="json"),
    }
    for _ in range(5):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = project.agent_calls_dir / f"{ts}_{_slug(agent)}_{scene_part}.json"
        try:
            written = write_new(path, record)
            append_jsonl(
                project.path("logs", "agent_invocations.jsonl"),
                {
                    "ts": record["ts"],
                    "agent": agent,
                    "scene_id": scene_id,
                    "working_set_hash": working_set_hash,
                    "mode": mode,
                    "model": model,
                    "task": task,
                    "log_path": project.rel(written),
                    "error": error,
                },
            )
            return written
        except ImmutableArtifactError:
            continue
    raise ImmutableArtifactError("could not allocate a unique agent-call log name")


def list_agent_calls(project: Project, *, scene_id: str | None = None, agent: str | None = None) -> list[Path]:
    out = []
    for p in sorted(project.agent_calls_dir.glob("*.json")):
        if scene_id and not p.stem.endswith("_" + scene_id):
            continue
        if agent and f"_{_slug(agent)}_" not in p.name:
            continue
        out.append(p)
    return out


def load_agent_call(path: Path) -> dict[str, Any]:
    return read_json(path)
