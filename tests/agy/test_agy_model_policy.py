"""Per-production model policy: primary / secondary video engine and provider fallback chains in project.yaml.

registry.yaml remains the global capability source of truth; a production can narrow or reorder it without editing
the repository (and without code changes), and the CLI `models` command shows the effective policy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agy_studio.adapters.fake import FakeVideoAdapter
from agy_studio.cli import run as cli_run
from agy_studio.decomposer import resolve_model_target
from agy_studio.models import GenerationPayload, PromptBundle
from agy_studio.orchestrator import Studio, StudioError
from agy_studio.registry import Registry
from agy_studio.testing import build_demo_project


def _scene(project):
    return project.load_scene("SC01")


def test_registry_defaults_when_no_policy(tmp_path: Path) -> None:
    project = build_demo_project(tmp_path / "p")
    reg = Registry()
    t = resolve_model_target(reg, _scene(project), settings=project.config.settings)
    assert t.preferred == "omni" and t.fallback == "veo"


def test_project_primary_and_secondary_engine(tmp_path: Path) -> None:
    project = build_demo_project(tmp_path / "p")
    reg = Registry()
    orch = Studio(project, registry=reg, fake=True)
    out = orch.set_model_policy(primary="veo", secondary="omni")
    assert "video primary   : veo" in out and "video secondary : omni" in out
    project = orch.project
    t = resolve_model_target(reg, _scene(project), settings=project.config.settings)
    assert (t.preferred, t.fallback) == ("veo", "omni")
    # persisted in project.yaml
    assert project.config.settings.video_model_primary == "veo"
    # an explicit per-scene / per-call model still wins over the project primary
    t2 = resolve_model_target(reg, _scene(project), "omni", settings=project.config.settings)
    assert (t2.preferred, t2.fallback) == ("omni", None)  # secondary == preferred -> no engine fallback


def test_secondary_none_means_fail_fast_on_primary(tmp_path: Path) -> None:
    project = build_demo_project(tmp_path / "p")
    reg = Registry()
    orch = Studio(project, registry=reg, fake=True)
    out = orch.set_model_policy(primary="omni", secondary="none")
    assert "NONE" in out
    t = resolve_model_target(reg, _scene(project), settings=project.config.settings)
    assert t.preferred == "omni" and t.fallback is None


def test_invalid_engine_ids_are_rejected(tmp_path: Path) -> None:
    project = build_demo_project(tmp_path / "p")
    orch = Studio(project, registry=Registry(), fake=True)
    with pytest.raises(StudioError, match="primary video model must be one of"):
        orch.set_model_policy(primary="sora")
    with pytest.raises(StudioError, match="secondary video model must be one of"):
        orch.set_model_policy(secondary="kling")
    with pytest.raises(Exception, match="not in registry"):
        orch.set_model_policy(fallbacks={"not_a_model": ["x"]})


def test_provider_fallback_chain_override_is_used_by_adapters(tmp_path: Path) -> None:
    project = build_demo_project(tmp_path / "p")
    reg = Registry()
    orch = Studio(project, registry=reg, fake=True)
    orch.set_model_policy(fallbacks={"omni": ["gemini-omni-1.1-flash-preview"], "veo": []})
    spec = reg.get("omni")
    adapter = FakeVideoAdapter(spec, project=project, registry=reg)
    assert adapter.model_chain() == [spec.provider_model, "gemini-omni-1.1-flash-preview"]
    bundle = PromptBundle(model_id="omni", prompt="x", duration_s=6.0, aspect_ratio="9:16", resolution="1080p")
    payload = GenerationPayload(model_id="omni", provider_model=spec.provider_model, bundle=bundle, seed=1,
                                payload_hash="ab" * 32)
    assert adapter.provider_chain(payload) == [spec.provider_model, "gemini-omni-1.1-flash-preview"]
    veo = FakeVideoAdapter(reg.get("veo"), project=project, registry=reg)
    assert veo.model_chain() == [reg.get("veo").provider_model]  # empty override = primary provider only
    # registry chain is untouched for other models and after clearing
    orch.set_model_policy(clear_fallbacks=True)
    adapter = FakeVideoAdapter(spec, project=orch.project, registry=reg)
    assert adapter.model_chain() == [spec.provider_model, *spec.provider_fallbacks]


def test_cli_models_command_round_trip(tmp_path: Path) -> None:
    project = build_demo_project(tmp_path / "productions")
    ws = str(project.root.parents[1])
    out = cli_run(["--workspace", ws, "--project", project.project_id, "models", "--primary", "veo",
                   "--secondary", "none", "--fallback", "omni=gemini-omni-1.1-flash", "--fallback", "veo="],
                  fake=True)
    assert "video primary   : veo" in out and "NONE" in out
    assert "omni " in out and "gemini-omni-1.1-flash   [project.yaml]" in out
    assert "(no provider fallback)   [project.yaml]" in out
    shown = cli_run(["--workspace", ws, "--project", project.project_id, "models"], fake=True)
    assert shown == out
