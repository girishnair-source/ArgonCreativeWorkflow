"""Fail-fast credential handling (no offline fallback, ever).

Covers the root cause behind the "Gate G5 video failed while Gate G3 images worked" incident: ``gcloud auth
print-access-token`` blocked by Context-Aware Access used to surface as an opaque ``CalledProcessError ... exit
status 1`` that was logged as a per-model failure, re-rolled with new seeds, handed to the next provider model and
finally patched over with offline stand-ins. Now it is one :class:`AuthError` with gcloud's real text + the fix.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest

from agy_studio.adapters import auth
from agy_studio.adapters.auth import AuthError
from agy_studio.adapters.base import LegacyVideoAdapter, PollResult, VideoAdapter
from agy_studio.models import DispatchStatus, GenerationPayload, PromptBundle, Take, TakeStatus
from agy_studio.orchestrator import Studio
from agy_studio.registry import Registry
from agy_studio.state import SceneStage
from agy_studio.testing import build_demo_project

CAA_STDERR = (
    "ERROR: (gcloud.auth.print-access-token) There was a problem refreshing your current auth tokens: "
    "Access was blocked by Context Aware Access.\n"
    "You can try to fix this by updating the context_aware/use_client_certificate flag to true:\n"
    "  gcloud config set context_aware/use_client_certificate true\n"
    "Please run:\n  $ gcloud auth login\n"
)


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def _fake_gcloud(tmp_path: Path, *, stdout: str = "", stderr: str = "", rc: int = 0) -> Path:
    """A stand-in ``gcloud`` executable with scripted output (the real binary is never called in tests)."""
    script = tmp_path / "bin" / "gcloud"
    script.parent.mkdir(parents=True, exist_ok=True)
    body = "#!/bin/sh\n"
    if stdout:
        body += f"printf '%s' '{stdout}'\n"
    if stderr:
        body += f"printf '%s' '{stderr}' 1>&2\n"
    body += f"exit {rc}\n"
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


@pytest.fixture
def no_env_project(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in auth.PROJECT_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("AGY_GCLOUD_ACCOUNT", raising=False)
    monkeypatch.delenv("CLOUDSDK_CORE_ACCOUNT", raising=False)


# ---------------------------------------------------------------------------------------------
# auth.py: the error the human sees
# ---------------------------------------------------------------------------------------------
def test_context_aware_access_block_becomes_one_actionable_auth_error(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    gcloud = _fake_gcloud(tmp_path, stderr=CAA_STDERR, rc=1)
    monkeypatch.setenv("AGY_GCLOUD", str(gcloud))
    with pytest.raises(AuthError) as info:
        auth.gcloud_access_token()
    msg = str(info.value)
    # gcloud's own words are relayed, not swallowed into "exit status 1"
    assert "Access was blocked by Context Aware Access" in msg
    assert "exit status" not in msg
    # the dedicated CAA remedy + the login commands, with the resolved gcloud path
    assert f"{gcloud} config set context_aware/use_client_certificate true" in msg
    assert f"{gcloud} auth login" in msg
    assert f"{gcloud} auth application-default login" in msg
    assert "print-access-token" in msg  # how to verify
    assert "NO video / voice / image" in msg  # the consequence is explicit


def test_expired_login_error_without_caa_has_no_caa_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gcloud = _fake_gcloud(tmp_path, stderr="ERROR: (gcloud.auth.print-access-token) You do not currently have an "
                                          "active account selected. Please run: $ gcloud auth login", rc=1)
    monkeypatch.setenv("AGY_GCLOUD", str(gcloud))
    with pytest.raises(AuthError) as info:
        auth.gcloud_access_token()
    assert "context_aware" not in str(info.value)
    assert f"{gcloud} auth login" in str(info.value)


def test_valid_token_is_returned_verbatim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gcloud = _fake_gcloud(tmp_path, stdout="FAKE-ACCESS-TOKEN-FOR-TESTS\n")
    monkeypatch.setenv("AGY_GCLOUD", str(gcloud))
    assert auth.gcloud_access_token() == "FAKE-ACCESS-TOKEN-FOR-TESTS"
    assert auth.gcloud_access_token("someone@example.com") == "FAKE-ACCESS-TOKEN-FOR-TESTS"


def test_missing_gcloud_binary_is_an_install_instruction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGY_GCLOUD", str(tmp_path / "nope" / "gcloud"))
    monkeypatch.setenv("PATH", str(tmp_path / "emptybin"))
    monkeypatch.setattr(auth, "GCLOUD_CANDIDATE_DIRS", ())
    with pytest.raises(AuthError) as info:
        auth.find_gcloud()
    assert "brew install --cask google-cloud-sdk" in str(info.value)
    assert auth.find_gcloud(required=False) is None


def test_homebrew_sdk_dir_is_added_to_path_when_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sdk_bin = tmp_path / "share" / "google-cloud-sdk" / "bin"
    sdk_bin.mkdir(parents=True)
    monkeypatch.setattr(auth, "GCLOUD_CANDIDATE_DIRS", (str(sdk_bin), str(tmp_path / "does-not-exist")))
    monkeypatch.setenv("PATH", "/usr/bin")
    auth.ensure_tool_dirs_on_path()
    parts = os.environ["PATH"].split(os.pathsep)
    assert parts[0] == str(sdk_bin)
    assert str(tmp_path / "does-not-exist") not in parts
    auth.ensure_tool_dirs_on_path()  # idempotent
    assert os.environ["PATH"].split(os.pathsep).count(str(sdk_bin)) == 1


def test_no_project_is_an_error_not_a_hidden_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      no_env_project: None) -> None:
    gcloud = _fake_gcloud(tmp_path, stdout="(unset)\n")
    monkeypatch.setenv("AGY_GCLOUD", str(gcloud))
    with pytest.raises(AuthError) as info:
        auth.vertex_project()
    assert "gcloud config set project" in str(info.value)
    assert "ggn-non-prod" not in str(info.value)
    monkeypatch.setenv("VERTEX_PROJECT_ID", "customer-demo-123")
    assert auth.vertex_project() == "customer-demo-123"


def test_require_vertex_auth_runs_the_real_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   no_env_project: None) -> None:
    gcloud = _fake_gcloud(tmp_path, stdout="ok-output\n")
    monkeypatch.setenv("AGY_GCLOUD", str(gcloud))
    monkeypatch.setenv("VERTEX_PROJECT_ID", "customer-demo-123")
    status = auth.require_vertex_auth()
    assert status.project == "customer-demo-123"
    assert status.gcloud == str(gcloud)
    assert "token=OK" in status.summary()


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (AuthError("x"), True),
        (subprocess.CalledProcessError(1, ["gcloud", "auth", "print-access-token"]), True),
        (RuntimeError("HTTP 401 UNAUTHENTICATED: Request had invalid authentication credentials"), True),
        (RuntimeError("403 PERMISSION_DENIED: caller lacks aiplatform.endpoints.predict"), True),
        (RuntimeError("HTTP 429 RESOURCE_EXHAUSTED"), False),
        (RuntimeError("The prompt was blocked by Responsible AI safety filters"), False),
        (TimeoutError("poll timed out"), False),
    ],
)
def test_is_auth_failure_classification(exc: BaseException, expected: bool) -> None:
    assert auth.is_auth_failure(exc) is expected


def test_as_auth_error_403_mentions_project_and_api(monkeypatch: pytest.MonkeyPatch) -> None:
    err = auth.as_auth_error(RuntimeError("403 PERMISSION_DENIED: Vertex AI API has not been used"),
                             context="submitting SC01_SH01_C01_T01")
    assert "PERMISSION_DENIED" in str(err)
    assert "aiplatform.googleapis.com" in str(err)
    assert "submitting SC01_SH01_C01_T01" in str(err)


# ---------------------------------------------------------------------------------------------
# dispatch(): an auth failure never becomes a per-model failure / RAI re-roll / next provider
# ---------------------------------------------------------------------------------------------
class _CountingAdapter(VideoAdapter):
    """Minimal VideoAdapter whose submit raises what the test scripts; counts every attempt."""

    def __init__(self, spec: Any, *, project: Any, exc_factory: Any) -> None:
        super().__init__(spec, project=project, fake=False)
        self.submits = 0
        self.polls = 0
        self._exc_factory = exc_factory

    def submit(self, payload: GenerationPayload, *, provider_model: str | None = None, attempt: int = 0) -> str:
        self.submits += 1
        raise self._exc_factory()

    def poll(self, operation_name: str) -> PollResult:
        self.polls += 1
        return PollResult(status=DispatchStatus.RUNNING)

    def download(self, result: PollResult, dest: Path) -> Path:
        raise AssertionError("never reached")


def _payload_and_take(project: Any, registry: Registry) -> tuple[GenerationPayload, Take, Any]:
    spec = registry.get("omni")
    bundle = PromptBundle(model_id="omni", prompt="A quiet kitchen at dawn.", duration_s=6.0, aspect_ratio="9:16",
                          resolution="1080p")
    payload = GenerationPayload(model_id="omni", provider_model=spec.provider_model, bundle=bundle, seed=1,
                                payload_hash="deadbeef" * 8)
    take = Take(id="SC01_SH01_C01_T01", clip_id="SC01_SH01_C01", model="omni", provider_model=spec.provider_model,
                seed=1, payload_hash=payload.payload_hash, dispatch_status=DispatchStatus.PLANNED,
                status=TakeStatus.CANDIDATE)
    return payload, take, spec


def test_dispatch_raises_auth_error_on_first_submit_without_trying_fallback_models(tmp_path: Path) -> None:
    project = build_demo_project(tmp_path / "productions")
    registry = Registry()
    payload, take, spec = _payload_and_take(project, registry)
    assert len(spec.provider_fallbacks) >= 1, "test needs a provider fallback chain to prove it is NOT used"

    adapter = _CountingAdapter(spec, project=project,
                               exc_factory=lambda: AuthError("gcloud said: Access was blocked by Context Aware Access"))
    saved: list[Take] = []
    with pytest.raises(AuthError, match="Context Aware Access"):
        adapter.dispatch(payload, take=take, on_update=saved.append, poll_interval_s=0, timeout_s=5)
    assert adapter.submits == 1  # not max_rai_retries x len(provider_chain)
    assert adapter.polls == 0
    assert saved == []  # nothing persisted as submitted/failed
    log = project.take_dir(take.id) / "dispatch.jsonl"
    if log.exists():
        text = log.read_text()
        assert "submit_failed" not in text and "gave_up" not in text


def test_dispatch_wraps_raw_gcloud_calledprocesserror_as_auth_error(tmp_path: Path) -> None:
    """Even an un-wrapped CalledProcessError from a stray gcloud call stops the wave immediately."""
    project = build_demo_project(tmp_path / "productions")
    registry = Registry()
    payload, take, spec = _payload_and_take(project, registry)
    adapter = _CountingAdapter(
        spec, project=project,
        exc_factory=lambda: subprocess.CalledProcessError(1, ["gcloud", "auth", "print-access-token"]))
    with pytest.raises(AuthError):
        adapter.dispatch(payload, take=take, on_update=lambda t: None, poll_interval_s=0, timeout_s=5)
    assert adapter.submits == 1


def test_dispatch_still_walks_provider_chain_for_non_auth_failures(tmp_path: Path) -> None:
    """Regression guard: genuine provider errors keep the existing fallback behaviour."""
    project = build_demo_project(tmp_path / "productions")
    registry = Registry()
    payload, take, spec = _payload_and_take(project, registry)
    adapter = _CountingAdapter(spec, project=project, exc_factory=lambda: RuntimeError("HTTP 503 UNAVAILABLE"))
    out = adapter.dispatch(payload, take=take, on_update=lambda t: None, poll_interval_s=0, timeout_s=5)
    assert out.dispatch_status == DispatchStatus.FAILED
    assert adapter.submits == len(adapter.provider_chain(payload))


# ---------------------------------------------------------------------------------------------
# LegacyVideoAdapter.call_with_auth: 401 -> one refresh; broken refresh or 403 -> AuthError
# ---------------------------------------------------------------------------------------------
class _Legacy(LegacyVideoAdapter):
    def submit(self, payload: GenerationPayload, *, provider_model: str | None = None, attempt: int = 0) -> str:
        raise AssertionError

    def poll(self, operation_name: str) -> PollResult:
        raise AssertionError


class _Token:
    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def get(self, force: bool = False) -> str:
        self.calls += 1
        if self.fail:
            raise subprocess.CalledProcessError(1, ["gcloud", "auth", "print-access-token"])
        return "tok"


class _Http(RuntimeError):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status


def test_call_with_auth_refreshes_once_on_401_then_succeeds(tmp_path: Path) -> None:
    project = build_demo_project(tmp_path / "productions")
    token = _Token()
    adapter = _Legacy(Registry().get("omni"), project=project, token=token)
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _Http(401)
        return "ok"

    assert adapter.call_with_auth(fn) == "ok"
    assert token.calls == 1 and calls["n"] == 2


def test_call_with_auth_broken_refresh_is_auth_error(tmp_path: Path) -> None:
    project = build_demo_project(tmp_path / "productions")
    adapter = _Legacy(Registry().get("omni"), project=project, token=_Token(fail=True))
    with pytest.raises(AuthError):
        adapter.call_with_auth(lambda: (_ for _ in ()).throw(_Http(401)))


def test_call_with_auth_403_is_auth_error_without_refresh(tmp_path: Path) -> None:
    project = build_demo_project(tmp_path / "productions")
    token = _Token()
    adapter = _Legacy(Registry().get("omni"), project=project, token=token)
    with pytest.raises(AuthError, match="PERMISSION_DENIED"):
        adapter.call_with_auth(lambda: (_ for _ in ()).throw(_Http(403)))
    assert token.calls == 0


def test_call_with_auth_non_auth_errors_pass_through(tmp_path: Path) -> None:
    project = build_demo_project(tmp_path / "productions")
    adapter = _Legacy(Registry().get("omni"), project=project, token=_Token())
    with pytest.raises(_Http):
        adapter.call_with_auth(lambda: (_ for _ in ()).throw(_Http(503)))


# ---------------------------------------------------------------------------------------------
# Orchestrator: pre-flight before the wave, blocking question, self-healing once gcloud is fixed
# ---------------------------------------------------------------------------------------------
def test_generating_wave_stops_before_any_take_and_self_heals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.agy.test_agy_acceptance import seed_two_scene_fixture

    project, registry = seed_two_scene_fixture(tmp_path / "proj_auth")
    orch = Studio(project, registry=registry, fake=True)
    orch.open_g1()
    orch.respond("G1", None, "OD3:choose on, approve all")
    if orch.state.scene("SC01").stage != SceneStage.AWAITING_G3:
        orch.advance("SC01")
    orch.respond("G3", "SC01", "approve all")
    assert orch.state.scene("SC01").stage == SceneStage.AWAITING_G4

    # Simulate a real (non-fake) studio whose gcloud sign-in is blocked by Context-Aware Access.
    broken = {"value": True}

    def fake_require(*, force: bool = False) -> Any:
        if broken["value"]:
            raise AuthError("Google Cloud sign-in failed — gcloud said: Access was blocked by Context Aware Access")
        return auth.AuthStatus(gcloud="/opt/homebrew/share/google-cloud-sdk/bin/gcloud", account="demo@example.com",
                               project="customer-demo-123")

    monkeypatch.setattr(orch, "_require_vertex_auth", fake_require)
    takes_before = sum(len(project.list_takes(c.id)) for c in project.iter_all_clips())

    text = orch.respond("G4", "SC01", "approve all")
    sc = orch.state.scene("SC01")
    assert sc.stage == SceneStage.GENERATING  # we stopped AT the wave, not somewhere later
    assert any(b.endswith(" auth") for b in sc.blocked_by), sc.blocked_by
    assert "Context Aware Access" in text
    assert "no offline stand-in" in text
    takes_after = sum(len(project.list_takes(c.id)) for c in project.iter_all_clips())
    assert takes_after == takes_before  # fail fast: nothing was planned, hashed or submitted
    open_q = [q for q in orch.questions.open() if q.scene_id == "SC01"]
    assert len(open_q) == 1

    # Retrying while still broken: same question, no duplicate, still nothing generated.
    orch.advance("SC01")
    assert len([q for q in orch.questions.open() if q.scene_id == "SC01"]) == 1
    assert sum(len(project.list_takes(c.id)) for c in project.iter_all_clips()) == takes_before

    # The human runs `gcloud auth login`; the next tick clears the blocker by itself and generation proceeds.
    broken["value"] = False
    orch.advance("SC01")
    assert not orch.questions.open()
    assert orch.state.scene("SC01").stage in (SceneStage.AWAITING_G5, SceneStage.QA, SceneStage.GENERATING)
    assert sum(len(project.list_takes(c.id)) for c in project.iter_all_clips()) > takes_before


def test_cli_renders_auth_error_as_stop_message(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from agy_studio import cli

    monkeypatch.setattr(cli, "run", lambda argv=None: (_ for _ in ()).throw(AuthError("gcloud said: blocked")))
    rc = cli.main(["status"])
    out = capsys.readouterr().out
    assert rc == 3
    assert out.startswith("STOPPED — Google Cloud sign-in problem")
    assert "gcloud said: blocked" in out


# ---------------------------------------------------------------------------------------------
# There is no offline / fake production mode: the test double is unreachable outside pytest
# ---------------------------------------------------------------------------------------------
def test_fake_backend_is_unreachable_outside_pytest(monkeypatch: pytest.MonkeyPatch) -> None:
    from agy_studio.adapters import base as adapters_base

    monkeypatch.setattr(adapters_base, "_under_pytest", lambda: False)
    with pytest.raises(adapters_base.FakeModeUnavailable, match="not available"):
        adapters_base.resolve_fake(True)
    assert adapters_base.resolve_fake(False) is False
    assert adapters_base.resolve_fake(None) is False  # no environment switch exists any more
    monkeypatch.setenv("AGY_FAKE_GENERATION", "1")
    assert adapters_base.fake_mode() is False


def test_cli_has_no_fake_flag() -> None:
    from agy_studio.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--fake", "status"])


# ---------------------------------------------------------------------------------------------
# Every Gemini request carries a hard timeout (a stalled connection must fail, never hang a gate)
# ---------------------------------------------------------------------------------------------
def test_request_timeouts_are_set_and_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    from agy_studio.adapters import base as adapters_base

    assert adapters_base.request_timeout_s("text") == 120.0
    assert adapters_base.request_timeout_s("image") == 300.0
    assert adapters_base.request_timeout_s("video") == 180.0
    monkeypatch.setenv("AGY_TIMEOUT_TEXT_S", "45")
    assert adapters_base.request_timeout_s("text") == 45.0
    opts = adapters_base.genai_http_options("text")
    assert opts.timeout == 45_000  # google-genai takes milliseconds
    monkeypatch.setenv("AGY_TIMEOUT_TEXT_S", "not-a-number")
    assert adapters_base.request_timeout_s("text") == 120.0


def test_trace_report_on_a_fresh_project(tmp_path: Path) -> None:
    from agy_studio.testing import build_demo_project
    from agy_studio.trace import render_trace

    project = build_demo_project(tmp_path / "p")
    text = render_trace(project)
    assert "where the time went" in text
    assert "no take has been submitted" in text
