"""Agent runner — BUILD_SPEC_v3 §1 (agents propose, humans decide), §4.5 (working-set isolation), §12 (logging).

``run_agent`` is the ONLY way an agent is invoked:

* the role prompt comes from ``agents/prompts/<role>.md``,
* scene roles get ONLY the scene working set (plus the records its pointers reference) — never global context or the
  raw script; global roles get ``global_context``; only the Script Analyst may read raw script text,
* scoped preferences and human feedback are injected,
* the output envelope ``{proposals: [2–4 variants each with id, content, rationale, provenance], rationale,
  questions_for_human}`` is validated with Pydantic,
* ``mode="llm"`` calls ``registry.adapter("agent_llm", project=..., fake=...).generate_json(system, user, schema=...)``
  (imported lazily through the registry); ``mode="manual"`` means AGY authored ``manual_output`` itself — it is
  validated and logged exactly the same way,
* EVERY call (including failed validations) is logged with ``agent_log.log_agent_call`` incl. the working-set hash
  and the exact prompt (acceptance criterion 5),
* agents never write the Bible or Resolve: the runner's only side effect is the immutable agent-call log.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import Field, ValidationError, field_validator

from ..agent_log import log_agent_call
from ..fsio import read_record, stable_hash
from ..models import Model, Preference, SceneWorkingSet
from ..project import Project
from .roles import RoleSpec, effective_max_rounds, get_role

MIN_VARIANTS = 2
MAX_VARIANTS = 4
RAW_SCRIPT_KEYS = frozenset({"raw_script", "script", "script_text", "screenplay", "full_script", "script_raw"})
_PROPOSAL_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
_RECORD_SUFFIXES = (".yaml", ".yml", ".json")


class AgentScopeError(ValueError):
    """A role was invoked with the wrong kind of context (scene isolation / global context rules)."""


class AgentOutputError(ValueError):
    """The agent output failed validation (the failed call is still logged)."""


class AgentRoundLimitError(RuntimeError):
    """A negotiation loop exceeded its round cap — escalate to the human as a blocking question."""


# ---------------------------------------------------------------------------------------------
# Output envelope
# ---------------------------------------------------------------------------------------------
class AgentProposal(Model):
    """One proposal variant: a concrete option the human can choose."""

    id: str
    content: Any
    rationale: str
    provenance: list[str] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def _v_id(cls, v: str) -> str:
        v = str(v).strip()
        if not _PROPOSAL_ID_RE.match(v):
            raise ValueError(f"proposal id {v!r} must match {_PROPOSAL_ID_RE.pattern}")
        return v

    @field_validator("content")
    @classmethod
    def _v_content(cls, v: Any) -> Any:
        if v is None or (isinstance(v, (str, list, dict)) and not v):
            raise ValueError("proposal content must not be empty")
        return v

    @field_validator("rationale")
    @classmethod
    def _v_rationale(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("every proposal needs a rationale")
        return str(v).strip()

    @field_validator("provenance", mode="before")
    @classmethod
    def _v_provenance(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [v] if v.strip() else []
        if isinstance(v, dict):
            return [f"{k}: {val}" for k, val in v.items()]
        if isinstance(v, (list, tuple)):
            return [p if isinstance(p, str) else json.dumps(p, ensure_ascii=False, sort_keys=True) for p in v]
        return v


class AgentOutput(Model):
    proposals: list[AgentProposal] = Field(min_length=MIN_VARIANTS, max_length=MAX_VARIANTS)
    rationale: str
    questions_for_human: list[str] = Field(default_factory=list)

    @field_validator("rationale")
    @classmethod
    def _v_rationale(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("the output needs an overall rationale")
        return str(v).strip()

    @field_validator("proposals")
    @classmethod
    def _v_unique(cls, v: list[AgentProposal]) -> list[AgentProposal]:
        ids = [p.id for p in v]
        if len(ids) != len(set(ids)):
            raise ValueError(f"proposal ids must be unique, got {ids}")
        return v

    @field_validator("questions_for_human", mode="before")
    @classmethod
    def _v_questions(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        return [q if isinstance(q, str) else json.dumps(q, ensure_ascii=False) for q in v]


class AgentResult(Model):
    role: str
    proposals: list[dict[str, Any]]
    rationale: str
    questions_for_human: list[str] = Field(default_factory=list)
    log_path: str  # absolute path of the immutable agent-call record
    mode: str
    scene_id: str | None = None
    working_set_hash: str | None = None
    model: str | None = None
    log_rel: str | None = None  # project-relative POSIX path of the same record


def output_json_schema() -> dict[str, Any]:
    return AgentOutput.model_json_schema()


# ---------------------------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------------------------
def _normalize_preferences(preferences: Iterable[Any], *, role: str, scene_id: str | None) -> list[dict[str, Any]]:
    """Active preferences that apply to this call (drops other scenes' / other agents' scoped preferences)."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pref in preferences or ():
        if isinstance(pref, Preference):
            item = {"id": pref.id, "scope": pref.scope, "text": pref.text, "weight": pref.weight,
                    "active": pref.active}
        elif isinstance(pref, dict):
            item = {"id": pref.get("id"), "scope": pref.get("scope", "project"), "text": str(pref.get("text", "")),
                    "weight": pref.get("weight", 1.0), "active": pref.get("active", True)}
        else:
            item = {"id": None, "scope": "project", "text": str(pref), "weight": 1.0, "active": True}
        if not item["active"] or not item["text"].strip():
            continue
        scope = str(item["scope"] or "project")
        if scope.startswith("scene:") and scope.split(":", 1)[1] != (scene_id or ""):
            continue
        if scope.startswith("agent:") and scope.split(":", 1)[1] != role:
            continue
        key = item["id"] or item["text"]
        if key in seen:
            continue
        seen.add(key)
        item.pop("active", None)
        out.append(item)
    return out


def _normalize_feedback(feedback: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    for item in feedback or ():
        if isinstance(item, Model):
            out.append(item.model_dump(mode="json", exclude_none=True))
        elif isinstance(item, (dict, list, str, int, float)):
            out.append(item)
        else:
            out.append(str(item))
    return out


def _entity_view(record: dict[str, Any], state_id: str) -> dict[str, Any]:
    states = [s for s in record.get("states", []) if s.get("id") == state_id]
    keep = ("id", "type", "tier", "display_name", "canonical_description", "voice_profile_id", "approval_status",
            "version")
    view = {k: record.get(k) for k in keep if k in record}
    view["state"] = states[0] if states else None
    return view


def _voice_view(record: dict[str, Any], voice_id: str) -> dict[str, Any] | None:
    for char_id, entry in (record.get("voice_profiles") or {}).items():
        profile = (entry or {}).get("profile") or {}
        if profile.get("id") == voice_id or char_id == voice_id or (entry or {}).get("voice_profile_id") == voice_id:
            return profile or dict(entry)
    return None


def _resolve_pointer(project: Project | None, pointer: Any) -> dict[str, Any]:
    """Read the record a working-set RefPointer points at (only refs listed in the working set are ever read)."""
    ptr = pointer.model_dump(mode="json") if isinstance(pointer, Model) else dict(pointer)
    out: dict[str, Any] = {"id": ptr.get("id"), "ref": ptr.get("ref"), "version": ptr.get("version"),
                           "approval_status": ptr.get("approval_status")}
    if project is None or not ptr.get("ref"):
        return out
    try:
        path = project.abs(str(ptr["ref"]))
        if path.suffix.lower() not in _RECORD_SUFFIXES or not path.is_file():
            out["record"] = None
            out["note"] = "pointer is not a readable record (media or missing)"
            return out
        record = read_record(path) or {}
        if isinstance(record, dict) and isinstance(record.get("states"), list) and ptr.get("id") != record.get("id"):
            out["record"] = _entity_view(record, str(ptr.get("id")))
        elif isinstance(record, dict) and "voice_profiles" in record:
            out["record"] = _voice_view(record, str(ptr.get("id")))
        else:
            if isinstance(record, dict):
                record = {k: v for k, v in record.items() if k != "generated_by"}
                if record.get("set3d_uri"):
                    try:
                        set_path = project.abs(str(record["set3d_uri"]))
                        if set_path.is_file():
                            from ..set3d.schema import SetLayout

                            lay = SetLayout.model_validate_json(set_path.read_text(encoding="utf-8"))
                            record["set3d_summary"] = {
                                "bounds_m": list(lay.bounds),
                                "axis_of_action": [list(p) for p in lay.axis_of_action] if lay.axis_of_action else None,
                                "landmarks": [f"{p.id} ({p.label or p.kind} at {list(p.position)})"
                                              for p in lay.primitives[:25]],
                                "mannequins": [f"{m.character_id} ({m.pose}, {m.height_m}m at {list(m.position)})"
                                               for m in lay.mannequins],
                                "saved_cameras": {f"cam:{c.name}": f"{c.describe()} ({c.note})" if c.note
                                                  else c.describe() for c in lay.cameras},
                            }
                    except Exception:  # noqa: BLE001
                        pass
            out["record"] = record
    except Exception as exc:  # noqa: BLE001 — a broken pointer must not crash the agent call; it is reported
        out["record"] = None
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def working_set_hash_of(ws: SceneWorkingSet) -> str:
    """The working set's own hash (workstream A sets it); computed deterministically if absent."""
    if ws.working_set_hash:
        return ws.working_set_hash
    return stable_hash(ws.model_dump(mode="json", exclude={"working_set_hash", "generated_by"}), length=16)


def scene_context(project: Project | None, ws: SceneWorkingSet) -> dict[str, Any]:
    """Everything a scene agent may see: the working set + the records its pointers reference. Nothing else."""
    data = ws.model_dump(mode="json", exclude_none=True, exclude={"generated_by"})
    data["resolved_refs"] = {
        "style_lock": _resolve_pointer(project, ws.style_lock),
        "entity_states": [_resolve_pointer(project, p) for p in ws.entity_states],
        "voice_profiles": [_resolve_pointer(project, p) for p in ws.voice_profiles],
        "environment": _resolve_pointer(project, ws.environment) if ws.environment else None,
    }
    return data


def _check_global_context(spec: RoleSpec, global_context: Any) -> dict[str, Any]:
    if not isinstance(global_context, dict):
        raise AgentScopeError(f"{spec.name}: global_context must be a dict of Bible-level data")
    if not spec.reads_raw_script:
        leaked = sorted(k for k in global_context if str(k).lower() in RAW_SCRIPT_KEYS)
        if leaked:
            raise AgentScopeError(
                f"{spec.name} reads the Bible, never the raw script (§1); remove {leaked} from global_context")
    return json.loads(json.dumps(global_context, ensure_ascii=False, default=str))


def _envelope_instructions(n_variants: int, spec: RoleSpec) -> str:
    return (
        "\n\n## Output envelope (MANDATORY — reply with ONE JSON object, no prose outside it)\n"
        "{\n"
        '  "proposals": [ {"id": "V1", "content": <your ' + spec.output_kind + ' proposal>, '
        '"rationale": "why this option", "provenance": ["working-set / bible / preference / feedback refs used"]} ],\n'
        '  "rationale": "overall reasoning and the trade-off between the variants",\n'
        '  "questions_for_human": ["blocking questions only, in plain filmmaking language"]\n'
        "}\n"
        f"Return exactly {n_variants} meaningfully different variants (allowed range {MIN_VARIANTS}–{MAX_VARIANTS}); "
        "ids unique (V1, V2, ...). Every variant needs a rationale and a non-empty provenance list.\n"
        "You PROPOSE only: never claim to have written the Bible, the library or DaVinci Resolve — the human decides "
        "at the gate and AGY applies the decision.\n"
    )


def build_prompt(spec: RoleSpec, *, task: str, context_label: str, context: dict[str, Any],
                 preferences: list[dict[str, Any]], feedback: list[Any], n_variants: int,
                 round_index: int | None = None, round_cap: int | None = None) -> tuple[str, str]:
    """(system, user) prompt pair. Deterministic for identical inputs (sorted JSON)."""
    system = spec.load_prompt().rstrip() + _envelope_instructions(n_variants, spec)
    parts = [
        f"# ROLE\n{spec.title or spec.name} ({spec.name}; scope={spec.scope}; output_kind={spec.output_kind})",
        f"# TASK\n{task.strip()}",
    ]
    if round_index is not None:
        parts.append(f"# NEGOTIATION ROUND\n{round_index} of at most {round_cap}")
    parts.append(f"# {context_label}\n" + json.dumps(context, ensure_ascii=False, indent=2, sort_keys=True))
    parts.append("# PREFERENCES (scoped; honour them unless they conflict with a hard rule — then ask)\n"
                 + (json.dumps(preferences, ensure_ascii=False, indent=2, sort_keys=True) if preferences else "[]"))
    parts.append("# FEEDBACK TO ADDRESS\n"
                 + (json.dumps(feedback, ensure_ascii=False, indent=2, sort_keys=True) if feedback else "[]"))
    parts.append(f"# REPLY\nOne JSON object with {n_variants} proposal variants, following the output envelope.")
    return system, "\n\n".join(parts)


def _parse_manual(manual_output: Any) -> Any:
    if isinstance(manual_output, AgentOutput):
        return manual_output.model_dump(mode="json")
    if isinstance(manual_output, (bytes, str)):
        return json.loads(manual_output)
    return manual_output


def _prefill_ids(raw: Any) -> Any:
    """Assign V1..Vn to proposals that arrived without an id (ids are presentation handles, not content)."""
    if isinstance(raw, dict) and isinstance(raw.get("proposals"), list):
        raw = dict(raw)
        fixed = []
        for i, prop in enumerate(raw["proposals"], start=1):
            if isinstance(prop, dict) and not prop.get("id"):
                prop = {**prop, "id": f"V{i}"}
            fixed.append(prop)
        raw["proposals"] = fixed
    return raw


def _validate(raw: Any) -> AgentOutput:
    return AgentOutput.model_validate(_prefill_ids(raw))


# ---------------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------------
def run_agent(
    project: Project,
    role: str,
    *,
    task: str,
    scene_id: str | None = None,
    working_set: SceneWorkingSet | dict[str, Any] | None = None,
    global_context: dict[str, Any] | None = None,
    feedback: Iterable[Any] = (),
    preferences: Iterable[Any] = (),
    mode: Literal["llm", "manual"] = "llm",
    manual_output: Any = None,
    registry: Any = None,
    n_variants: int = 3,
    fake: bool | None = None,
    round_index: int | None = None,
    temperature: float = 0.4,
) -> AgentResult:
    """Invoke one §6 role and return its validated proposals (see module docstring for the rules)."""
    spec = get_role(role)
    if mode not in ("llm", "manual"):
        raise ValueError(f"mode must be 'llm' or 'manual', got {mode!r}")
    if not task or not str(task).strip():
        raise ValueError("run_agent needs a task description")
    n_variants = max(MIN_VARIANTS, min(MAX_VARIANTS, int(n_variants)))

    # -- scope enforcement (§4.5) ---------------------------------------------------------------
    ws: SceneWorkingSet | None = None
    if working_set is not None:
        ws = working_set if isinstance(working_set, SceneWorkingSet) else SceneWorkingSet.model_validate(working_set)
    scene_call = spec.scope == "scene" and (ws is not None or not spec.cross_cutting or scene_id is not None)
    if spec.scope == "global" and ws is not None:
        raise AgentScopeError(f"{spec.name} is a global role: pass global_context, not a scene working set")
    if scene_call:
        if ws is None:
            raise AgentScopeError(
                f"{spec.name} is a scene role and REQUIRES the scene working set (it never sees global context)")
        if global_context is not None:
            raise AgentScopeError(
                f"{spec.name} is a scene role: it receives ONLY the working set, never global context or the script")
        if scene_id is not None and scene_id != ws.scene_id:
            raise AgentScopeError(f"scene_id {scene_id} does not match the working set's scene {ws.scene_id}")
        scene_id = ws.scene_id
        ws_hash: str | None = working_set_hash_of(ws)
        context = scene_context(project, ws)
        context_label = f"SCENE WORKING SET {scene_id} (the ONLY context you may use)"
        context_hash = ws_hash
    else:
        if global_context is None:
            raise AgentScopeError(f"{spec.name} runs at the global level here and needs global_context")
        if scene_id is not None:
            raise AgentScopeError(f"{spec.name}: a global call has no scene_id (pass a working set for scene work)")
        context = _check_global_context(spec, global_context)
        context_label = "GLOBAL CONTEXT (Bible-level data)"
        ws_hash = None
        context_hash = stable_hash(context, length=16)

    cap = effective_max_rounds(spec, project.config.settings)
    if round_index is not None and spec.loop and int(round_index) > cap:
        raise AgentRoundLimitError(
            f"{spec.name}: round {round_index} exceeds the {spec.loop} cap of {cap} — ask the human instead")

    prefs = _normalize_preferences(preferences, role=spec.name, scene_id=scene_id)
    fb = _normalize_feedback(feedback)
    system, user = build_prompt(spec, task=str(task), context_label=context_label, context=context,
                                preferences=prefs, feedback=fb, n_variants=n_variants,
                                round_index=round_index if spec.loop else None, round_cap=cap if spec.loop else None)
    exact_prompt = f"=== SYSTEM ===\n{system}\n\n=== USER ===\n{user}"
    inputs: dict[str, Any] = {
        "role": spec.name, "scope": "scene" if scene_call else "global", "task": str(task),
        "scene_id": scene_id, "working_set_hash": ws_hash, "context_hash": context_hash,
        "n_variants": n_variants, "preferences": prefs, "feedback": fb,
        "round_index": round_index, "round_cap": cap if spec.loop else None,
        "prompt_file": f"agents/prompts/{spec.prompt_file}",
    }

    started = time.monotonic()
    model_name: str | None = None
    attempts: list[dict[str, Any]] = []
    output: AgentOutput | None = None
    error: str | None = None
    try:
        if mode == "manual":
            if manual_output is None:
                raise AgentOutputError("manual mode needs manual_output (the proposals AGY authored itself)")
            raw = _parse_manual(manual_output)
            attempts.append({"raw": raw})
            output = _validate(raw)
        else:
            from ..registry import default_registry  # lazy: adapters are workstream C's

            reg = registry or default_registry()
            spec_llm = reg.get("agent_llm")
            model_name = spec_llm.provider_model
            adapter = reg.adapter("agent_llm", project=project, fake=fake)
            user_prompt = user
            for attempt in range(2):
                raw = adapter.generate_json(system, user_prompt, schema=output_json_schema(),
                                            temperature=temperature)
                attempts.append({"raw": raw})
                try:
                    output = _validate(raw)
                    break
                except ValidationError as exc:
                    attempts[-1]["validation_error"] = str(exc)
                    if attempt == 1:
                        raise
                    user_prompt = (user + "\n\n# YOUR PREVIOUS REPLY FAILED VALIDATION — fix it and reply again\n"
                                   + str(exc))
                    exact_prompt += f"\n\n=== RETRY USER ===\n{user_prompt}"
    except (ValidationError, AgentOutputError, json.JSONDecodeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # adapter / registry failure: still logged, then re-raised
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        outputs: dict[str, Any] = {"validated": output.model_dump(mode="json") if output else None,
                                   "attempts": attempts}
        log_path = log_agent_call(project, agent=spec.name, scene_id=scene_id, working_set_hash=ws_hash,
                                  task=str(task), inputs=inputs, outputs=outputs, model=model_name, mode=mode,
                                  prompt=exact_prompt, duration_s=round(time.monotonic() - started, 3),
                                  error=error)
    if output is None:
        raise AgentOutputError(f"{spec.name} output failed validation ({error}); logged at {log_path}")
    return AgentResult(
        role=spec.name,
        proposals=[p.model_dump(mode="json") for p in output.proposals],
        rationale=output.rationale,
        questions_for_human=list(output.questions_for_human),
        log_path=str(Path(log_path)),
        mode=mode,
        scene_id=scene_id,
        working_set_hash=ws_hash,
        model=model_name,
        log_rel=project.rel(log_path),
    )
