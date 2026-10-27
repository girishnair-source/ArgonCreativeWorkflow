"""AGY Film Studio — multi-agent film production system orchestrated by Antigravity.

Implements docs/BUILD_SPEC_v3.md. The project folder (productions/<PROJECT_ID>/) is the system of record;
this package provides the deterministic machinery (data model, registry, state machine, linter, adapters,
QA, Resolve Bridge) that Antigravity invokes from chat.
"""

AGY_VERSION = "3.0.0"
SPEC_VERSION = "v3"

__all__ = ["AGY_VERSION", "SPEC_VERSION"]
