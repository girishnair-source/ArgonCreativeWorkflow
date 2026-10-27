"""AGY agents layer (BUILD_SPEC_v3 §6): role registry + the single runner every agent call goes through.

Agents PROPOSE (2–4 variants with rationale + provenance); humans decide at the gates; AGY applies decisions.
Agents never write the Bible, the library or DaVinci Resolve.
"""

from .roles import DIRECTION_LOOP_ROLES, PROMPTS_DIR, ROLES, RoleSpec, effective_max_rounds, get_role
from .runner import (
    AgentOutput,
    AgentOutputError,
    AgentProposal,
    AgentResult,
    AgentRoundLimitError,
    AgentScopeError,
    run_agent,
)

__all__ = [
    "AgentOutput",
    "AgentOutputError",
    "AgentProposal",
    "AgentResult",
    "AgentRoundLimitError",
    "AgentScopeError",
    "DIRECTION_LOOP_ROLES",
    "PROMPTS_DIR",
    "ROLES",
    "RoleSpec",
    "effective_max_rounds",
    "get_role",
    "run_agent",
]
