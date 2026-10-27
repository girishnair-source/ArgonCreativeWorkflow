"""Agent role registry — BUILD_SPEC_v3 §6 (departments and roles) and §12 (negotiation caps).

Every role is data: which scope it works in, which prompt file defines it, what kind of proposal it returns and how
many negotiation rounds it may take. The runner (``agents.runner.run_agent``) enforces the scope rules:

* ``scope="scene"`` roles REQUIRE a ``SceneWorkingSet`` and never receive global context or the raw script (§4.5),
* ``scope="global"`` roles receive ``global_context`` (Bible-level data) and never a scene working set,
* ``cross_cutting`` scene roles (Continuity Supervisor, Composer, Sound Designer, Editor, Mixer) may ALSO be called
  once at the global level (state timeline, score themes, sound palette, master assembly, stems layout) — but a call
  is always either scene-scoped (working set only) or global (global context only), never both,
* only the Script Analyst may read raw script text (everyone else reads the Bible, §1).

Director <-> Cinematographer <-> Continuity Supervisor negotiate for at most ``settings.max_agent_rounds`` rounds
(§12, default 3); unresolved disagreements become blocking questions for the human.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

PROMPTS_DIR = Path(__file__).with_name("prompts")
DIRECTION_LOOP = "direction_loop"
DEFAULT_LOOP_ROUNDS = 3


class RoleSpec(BaseModel):
    """One §6 role. ``prompt_file`` is relative to ``agents/prompts/``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    scope: Literal["global", "scene"]
    prompt_file: str
    output_kind: str
    max_rounds: int = 1
    department: str = ""
    title: str = ""
    gate: str | None = None  # human gate where this role's proposals are decided (G1..G7)
    cross_cutting: bool = False
    reads_raw_script: bool = False
    loop: str | None = None  # negotiation loop membership (cap = settings.max_agent_rounds)
    implementation: str | None = None  # deterministic module that performs the role (agent reviews/explains)

    @property
    def prompt_path(self) -> Path:
        return PROMPTS_DIR / self.prompt_file

    def load_prompt(self) -> str:
        path = self.prompt_path
        if not path.is_file():
            raise FileNotFoundError(f"role prompt missing for {self.name}: {path}")
        return path.read_text(encoding="utf-8")


def _r(name: str, scope: Literal["global", "scene"], output_kind: str, *, department: str, title: str,
       gate: str | None = None, cross_cutting: bool = False, reads_raw_script: bool = False,
       loop: str | None = None, implementation: str | None = None, max_rounds: int = 1) -> RoleSpec:
    return RoleSpec(name=name, scope=scope, prompt_file=f"{name}.md", output_kind=output_kind,
                    max_rounds=DEFAULT_LOOP_ROUNDS if loop else max_rounds, department=department, title=title,
                    gate=gate, cross_cutting=cross_cutting, reads_raw_script=reads_raw_script, loop=loop,
                    implementation=implementation)


_ROLE_LIST: tuple[RoleSpec, ...] = (
    # Cross-cutting
    _r("continuity_supervisor", "scene", "continuity_report", department="cross-cutting",
       title="Continuity Supervisor", gate="G3", cross_cutting=True, loop=DIRECTION_LOOP,
       implementation="agy_studio.continuity"),
    # Story department (global)
    _r("script_analyst", "global", "scene_list", department="story", title="Script Analyst", gate="G1",
       reads_raw_script=True, implementation="agy_studio.preproduction.ingest_script"),
    _r("story_editor", "global", "story_structure", department="story", title="Story Editor", gate="G1"),
    _r("breakdown_analyst", "global", "breakdown_schedule", department="story", title="Breakdown Analyst",
       gate="G1", implementation="agy_studio.preproduction.compute_breakdown"),
    # Art department (delta-only; promoted to the library at G2)
    _r("production_designer", "global", "style_lock", department="art", title="Production Designer", gate="G1"),
    _r("character_designer", "scene", "reference_pack_plan", department="art", title="Character Designer",
       gate="G2"),
    _r("location_designer", "scene", "environment_plan", department="art", title="Location Designer", gate="G2"),
    _r("set_designer", "global", "set_layout", department="art", title="Set Designer (3D grey-box)", gate="G2",
       implementation="agy_studio.set3d.designer"),
    # Direction department (per scene)
    _r("director", "scene", "coverage_plan", department="direction", title="Director", gate="G3",
       loop=DIRECTION_LOOP),
    _r("cinematographer", "scene", "visual_spec", department="direction", title="Cinematographer", gate="G3",
       loop=DIRECTION_LOOP),
    _r("storyboard_artist", "scene", "board_plan", department="direction", title="Storyboard Artist", gate="G3"),
    _r("shot_decomposer", "scene", "clip_plan", department="direction", title="Shot Decomposer", gate="G4",
       implementation="agy_studio.decomposer"),
    # Audio department
    _r("casting_director", "global", "voice_profiles", department="audio", title="Casting Director", gate="G1"),
    _r("voice_director", "scene", "delivery_plan", department="audio", title="Voice Director", gate="G4"),
    _r("composer", "scene", "cue_plan", department="audio", title="Composer", gate="G4", cross_cutting=True),
    _r("sound_designer", "scene", "sound_plan", department="audio", title="Sound Designer", gate="G4",
       cross_cutting=True),
    # Production department
    _r("prompt_engineer", "scene", "prompt_review", department="production", title="Prompt Engineer",
       implementation="agy_studio.prompt_engineer"),
    _r("preflight_linter", "scene", "lint_review", department="production", title="Pre-flight Linter",
       implementation="agy_studio.linter"),
    _r("shot_generator", "scene", "dispatch_plan", department="production", title="Shot Generator",
       implementation="agy_studio.shot_generator"),
    _r("qa_critic", "scene", "qa_review", department="production", title="QA Critic", gate="G5",
       implementation="agy_studio.qa_critic"),
    _r("vfx_fixer", "scene", "fix_plan", department="production", title="VFX Fixer (optional)", gate="G5"),
    # Post department
    _r("editor", "scene", "edit_plan", department="post", title="Editor", gate="G6", cross_cutting=True,
       implementation="agy_studio.editor"),
    _r("colorist", "scene", "color_notes", department="post", title="Colorist", gate="G6"),
    _r("mixer", "scene", "mix_plan", department="post", title="Mixer", gate="G6", cross_cutting=True),
)

ROLES: dict[str, RoleSpec] = {spec.name: spec for spec in _ROLE_LIST}

DIRECTION_LOOP_ROLES: tuple[str, ...] = tuple(r.name for r in _ROLE_LIST if r.loop == DIRECTION_LOOP)


def normalize_role_name(role: str) -> str:
    return (role or "").strip().lower().replace("-", "_").replace(" ", "_")


def get_role(role: str | RoleSpec) -> RoleSpec:
    if isinstance(role, RoleSpec):
        return role
    key = normalize_role_name(role)
    if key not in ROLES:
        raise KeyError(f"unknown agent role {role!r}; known roles: {sorted(ROLES)}")
    return ROLES[key]


def effective_max_rounds(role: str | RoleSpec, settings: object | None = None) -> int:
    """Negotiation cap for ``role``: loop roles use ``settings.max_agent_rounds`` (§12), others their own cap."""
    spec = get_role(role)
    if spec.loop:
        value = getattr(settings, "max_agent_rounds", None) if settings is not None else None
        return int(value) if value else spec.max_rounds
    return spec.max_rounds


def roles_by_department() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for spec in _ROLE_LIST:
        out.setdefault(spec.department, []).append(spec.name)
    return out
