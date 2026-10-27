"""Google Cloud credential discovery for every real (non-fake) adapter — **fail fast, never fall back**.

Every Vertex AI call in AGY Studio (Gemini Omni / Veo video, Nano Banana Pro images, Gemini text / ASR / vision
judges, Lyria music) is authenticated with the local ``gcloud`` user credential. When that credential is missing,
expired or blocked (e.g. by corporate Context-Aware Access) there is exactly ONE correct behaviour:

* stop immediately — before a single take folder, payload or provider job is created,
* show the human the *real* ``gcloud`` error text and the exact commands that fix it,
* never substitute offline output (``AGY_FAKE_GENERATION``, macOS ``say``, ``ffmpeg zoompan`` stills, ad-hoc
  ``prepare_*.py`` scripts). Offline stand-ins waste minutes, produce unusable media and hide the credential block.

This module is the single source of truth for that rule:

* :func:`find_gcloud` — locates the ``gcloud`` binary (``PATH`` plus the usual macOS / Linux / Windows install dirs,
  including Homebrew's ``/opt/homebrew/share/google-cloud-sdk/bin``) and raises :class:`AuthError` when absent.
* :func:`gcloud_access_token` — runs ``gcloud auth print-access-token`` with stderr captured; a non-zero exit becomes
  an :class:`AuthError` carrying gcloud's own message plus remediation (Context-Aware Access gets its dedicated hint).
* :func:`require_vertex_auth` — one cheap pre-flight (binary + token + project) that the orchestrator runs before it
  enters a generation wave and that ``live_demo_preflight.py`` runs on stage.
* :func:`is_auth_failure` / :func:`as_auth_error` — classify provider exceptions (HTTP 401 / 403, UNAUTHENTICATED,
  PERMISSION_DENIED, gcloud failures) so adapters re-raise them instead of retrying, re-rolling seeds or trying the
  next provider model (every one of those would fail identically).

Only the standard library is used here so the module can be imported from the legacy generators as well.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ENV_KEYS = ("VERTEX_PROJECT_ID", "GOOGLE_CLOUD_PROJECT")
GCLOUD_TIMEOUT_S = 60.0

# Directories where gcloud / ffmpeg commonly live but which GUI-launched processes (Antigravity, Resolve, IDEs) do
# not have on PATH. Only existing directories are prepended, in this order.
GCLOUD_CANDIDATE_DIRS: tuple[str, ...] = (
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/opt/homebrew/share/google-cloud-sdk/bin",
    "/usr/local/share/google-cloud-sdk/bin",
    "/usr/local/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/bin",
    "/opt/homebrew/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/bin",
    "~/google-cloud-sdk/bin",
    "~/Downloads/google-cloud-sdk/bin",
    "/usr/lib/google-cloud-sdk/bin",
    "/snap/bin",
    "/snap/google-cloud-cli/current/bin",
    "/opt/google-cloud-sdk/bin",
    "~/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin",
    "C:/Program Files (x86)/Google/Cloud SDK/google-cloud-sdk/bin",
    "C:/Program Files/Google/Cloud SDK/google-cloud-sdk/bin",
)

_CAA_MARKERS = ("context aware access", "context-aware access", "context_aware")
_LOGIN_MARKERS = ("gcloud auth login", "reauthentication", "reauth", "invalid_grant", "token has been expired",
                  "expired or revoked", "no credentialed accounts", "does not have any valid credentials",
                  "you do not currently have an active account")
_AUTH_STATUS = frozenset({401, 403})
_AUTH_WORDS = ("UNAUTHENTICATED", "PERMISSION_DENIED", "ACCESS_TOKEN_SCOPE_INSUFFICIENT", "invalid authentication",
               "Request had invalid authentication credentials", "credentials were not found", "Reauthentication")


class AuthError(RuntimeError):
    """Google Cloud credentials are missing, expired, blocked or not permitted for the Vertex project.

    The message is complete and human-facing: it contains the real tool error and the commands that fix it. Callers
    must let it propagate (or convert it to a blocking question) — never retry, re-seed, fall back to another provider
    model or switch to offline generation.
    """

    def __init__(self, message: str, *, cause: str = "", remedy: str = "") -> None:
        super().__init__(message)
        self.cause = cause
        self.remedy = remedy


@dataclass
class AuthStatus:
    """Result of :func:`require_vertex_auth` (also rendered by ``live_demo_preflight.py``)."""

    gcloud: str
    account: str | None
    project: str
    token_ok: bool = True
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        acct = self.account or "(unknown account)"
        return f"gcloud={self.gcloud} account={acct} project={self.project} token={'OK' if self.token_ok else 'FAIL'}"


# ---------------------------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------------------------
def ensure_tool_dirs_on_path() -> None:
    """Prepend the well-known SDK / Homebrew directories that exist on this machine to ``PATH`` (idempotent)."""
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    added: list[str] = []
    for raw in GCLOUD_CANDIDATE_DIRS:
        d = str(Path(os.path.expanduser(raw)))
        if d in parts or d in added or not Path(d).is_dir():
            continue
        added.append(d)
    if added:
        os.environ["PATH"] = os.pathsep.join([*added, *parts])


def find_gcloud(*, required: bool = True) -> str | None:
    """Absolute path of the ``gcloud`` executable, or raise :class:`AuthError` (``required=False`` -> None)."""
    ensure_tool_dirs_on_path()
    explicit = os.environ.get("AGY_GCLOUD") or os.environ.get("CLOUDSDK_GCLOUD")
    candidates = [explicit] if explicit else []
    candidates += [shutil.which("gcloud"), shutil.which("gcloud.cmd") if sys.platform.startswith("win") else None]
    for cand in candidates:
        if cand and Path(cand).exists():
            return str(Path(cand))
    if not required:
        return None
    raise AuthError(
        "Google Cloud CLI (gcloud) was not found on this machine, so no Vertex AI model can be called.\n"
        "Install it, then sign in:\n"
        "  macOS : brew install --cask google-cloud-sdk\n"
        "  other : https://cloud.google.com/sdk/docs/install\n"
        "  then  : gcloud auth login && gcloud auth application-default login\n"
        "If it is installed in a custom location, export AGY_GCLOUD=/full/path/to/gcloud.",
        cause="gcloud not found", remedy="install gcloud",
    )


def _gcloud_env() -> dict[str, str]:
    env = dict(os.environ)
    tag = "datacloud.antigravity"
    prev = env.get("CLOUDSDK_METRICS_ENVIRONMENT", "")
    env["CLOUDSDK_METRICS_ENVIRONMENT"] = f"{prev} {tag}".strip() if prev else tag
    env.setdefault("CLOUDSDK_CORE_DISABLE_PROMPTS", "1")  # never hang on an interactive prompt
    return env


def run_gcloud(args: list[str], *, timeout: float = GCLOUD_TIMEOUT_S) -> subprocess.CompletedProcess[str]:
    """Run ``gcloud <args>`` non-interactively with stdout AND stderr captured (never swallowed)."""
    gcloud = find_gcloud()
    try:
        return subprocess.run([gcloud, *args], text=True, capture_output=True, env=_gcloud_env(),
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise AuthError(
            f"`gcloud {' '.join(args)}` did not answer within {int(timeout)}s. gcloud is probably waiting for an "
            f"interactive sign-in or a network proxy.\nRun in a terminal:\n  {gcloud} auth login\n"
            f"  {gcloud} auth application-default login",
            cause="gcloud timeout", remedy="gcloud auth login",
        ) from exc


# ---------------------------------------------------------------------------------------------
# Token / account / project
# ---------------------------------------------------------------------------------------------
def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def _gcloud_error_lines(stderr: str) -> str:
    """gcloud's ``ERROR:`` block (plus its follow-up hints) without the unrelated WARNING preamble."""
    lines = [ln.rstrip() for ln in (stderr or "").splitlines()]
    starts = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("ERROR")]
    if not starts:
        return _clean(stderr)
    return _clean(" ".join(lines[starts[0]:]))


def format_gcloud_failure(stderr: str, *, command: str, gcloud: str) -> AuthError:
    """Turn a failed gcloud invocation into one complete, human-facing :class:`AuthError`."""
    detail = _gcloud_error_lines(stderr) or "(gcloud printed no error text)"
    low = (stderr or "").lower()
    lines = [f"Google Cloud sign-in failed — `{command}` returned an error, so NO video / voice / image can be "
             f"generated right now.", f"gcloud said: {detail}", "", "Fix it once in a terminal, then re-run the gate:"]
    remedy: list[str] = []
    if any(m in low for m in _CAA_MARKERS):
        lines.append("  # Your account is blocked by corporate Context-Aware Access — enable the client certificate:")
        remedy.append(f"{gcloud} config set context_aware/use_client_certificate true")
    remedy.append(f"{gcloud} auth login")
    remedy.append(f"{gcloud} auth application-default login")
    lines += [f"  {cmd}" for cmd in remedy]
    lines.append(f"Verify with:  {gcloud} auth print-access-token")
    if "quota" in low or "project" in low and "not found" in low:
        lines.append("Also check the active project:  gcloud config set project <YOUR_VERTEX_PROJECT_ID>")
    return AuthError("\n".join(lines), cause=detail, remedy="; ".join(remedy))


def gcloud_access_token(account: str | None = None, *, timeout: float = GCLOUD_TIMEOUT_S) -> str:
    """A fresh ``gcloud auth print-access-token`` (optionally for a specific account). Raises :class:`AuthError`."""
    args = ["auth", "print-access-token"]
    if account:
        args.append(account)
    gcloud = find_gcloud()
    proc = run_gcloud(args, timeout=timeout)
    token = (proc.stdout or "").strip().splitlines()
    if proc.returncode == 0 and token and token[-1]:
        return token[-1].strip()
    raise format_gcloud_failure(proc.stderr or proc.stdout, command=f"gcloud {' '.join(args)}", gcloud=gcloud)


def gcloud_accounts() -> tuple[str | None, list[str]]:
    """``(active_account, all_accounts)`` from ``gcloud auth list`` (empty when gcloud cannot answer)."""
    proc = run_gcloud(["auth", "list", "--format=value(account,status)"], timeout=30)
    if proc.returncode != 0:
        return None, []
    active: str | None = None
    accounts: list[str] = []
    for line in (proc.stdout or "").splitlines():
        parts = line.split()
        if not parts:
            continue
        accounts.append(parts[0])
        if len(parts) > 1 and parts[1].upper().startswith("ACTIVE"):
            active = parts[0]
    return active, accounts


def preferred_account() -> str | None:
    """``AGY_GCLOUD_ACCOUNT`` / ``CLOUDSDK_CORE_ACCOUNT`` when set, else the active gcloud account (None = default)."""
    explicit = os.environ.get("AGY_GCLOUD_ACCOUNT") or os.environ.get("CLOUDSDK_CORE_ACCOUNT")
    if explicit:
        return explicit
    return None


def gcloud_project() -> str | None:
    """``gcloud config get-value project`` or None when unset / unavailable."""
    proc = run_gcloud(["config", "get-value", "project", "--quiet"], timeout=30)
    if proc.returncode != 0:
        return None
    proj = (proc.stdout or "").strip()
    return proj if proj and proj != "(unset)" else None


def vertex_project() -> str:
    """Vertex project: ``VERTEX_PROJECT_ID`` / ``GOOGLE_CLOUD_PROJECT`` -> active gcloud project. Never a hidden default.

    A silently substituted project would send a customer's generation to the wrong billing account and surface as a
    confusing 403 minutes later, so a missing project is an :class:`AuthError` with the one command that sets it.
    """
    for key in PROJECT_ENV_KEYS:
        if os.environ.get(key, "").strip():
            return os.environ[key].strip()
    proj = gcloud_project()
    if proj:
        return proj
    raise AuthError(
        "No Google Cloud project is configured for Vertex AI, so generation cannot start.\n"
        "Set it once (either works):\n"
        "  gcloud config set project <YOUR_VERTEX_PROJECT_ID>\n"
        "  export VERTEX_PROJECT_ID=<YOUR_VERTEX_PROJECT_ID>",
        cause="no project", remedy="gcloud config set project",
    )


def require_vertex_auth(*, account: str | None = None) -> AuthStatus:
    """Fast pre-flight: gcloud present, access token mints, project known. Raises :class:`AuthError` otherwise."""
    gcloud = find_gcloud()
    account = account or preferred_account()
    gcloud_access_token(account)
    active, _ = gcloud_accounts()
    project = vertex_project()
    status = AuthStatus(gcloud=gcloud, account=account or active, project=project)
    return status


# ---------------------------------------------------------------------------------------------
# Exception classification (used by every adapter's retry loop)
# ---------------------------------------------------------------------------------------------
def http_status_of(exc: BaseException) -> int | None:
    for attr in ("status", "code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value
    text = str(exc)
    if re.search(r"\b401\b", text) or "UNAUTHENTICATED" in text:
        return 401
    if re.search(r"\b403\b", text) or "PERMISSION_DENIED" in text:
        return 403
    return None


def is_auth_failure(exc: BaseException | str) -> bool:
    """True for credential / permission failures that no retry, seed change or provider fallback can repair."""
    if isinstance(exc, AuthError):
        return True
    if isinstance(exc, BaseException):
        if isinstance(exc, subprocess.CalledProcessError) and exc.cmd and "gcloud" in str(exc.cmd[0]):
            return True
        if http_status_of(exc) in _AUTH_STATUS:
            return True
        text = str(exc)
    else:
        text = exc
    return any(word.lower() in text.lower() for word in _AUTH_WORDS)


def as_auth_error(exc: BaseException, *, context: str = "") -> AuthError:
    """Wrap a provider / gcloud exception as a complete :class:`AuthError` (idempotent for AuthError)."""
    if isinstance(exc, AuthError):
        return exc
    gcloud = find_gcloud(required=False) or "gcloud"
    detail = _clean(str(exc)) or type(exc).__name__
    status = http_status_of(exc)
    where = f" while {context}" if context else ""
    if status == 403:
        head = (f"Vertex AI refused the request{where} (HTTP 403 PERMISSION_DENIED). The signed-in Google account "
                f"has no access to this model / project, or the Vertex AI API is not enabled.")
        tips = ["Check the active account and project:  gcloud auth list && gcloud config get-value project",
                "Enable the API once:  gcloud services enable aiplatform.googleapis.com",
                "Or switch account:   gcloud config set account <ACCOUNT> && gcloud auth login"]
    else:
        head = (f"Vertex AI rejected the credentials{where} (HTTP {status or 401} UNAUTHENTICATED). The local gcloud "
                f"token is missing, expired or blocked.")
        tips = [f"{gcloud} auth login", f"{gcloud} auth application-default login",
                f"Verify with:  {gcloud} auth print-access-token"]
    msg = "\n".join([head, f"Provider said: {detail}", "", "Fix it once in a terminal, then re-run the gate:",
                     *(f"  {t}" for t in tips)])
    return AuthError(msg, cause=detail, remedy="; ".join(tips))


def raise_if_auth_failure(exc: BaseException, *, context: str = "") -> None:
    """Re-raise ``exc`` as :class:`AuthError` when it is a credential / permission failure; otherwise return."""
    if is_auth_failure(exc):
        raise as_auth_error(exc, context=context) from exc


def describe_for_chat(exc: BaseException) -> str:
    """Chat-ready rendering of an :class:`AuthError` (or any exception) — one block, no stack trace."""
    if isinstance(exc, AuthError):
        return f"STOPPED — Google Cloud sign-in problem (nothing was generated, no offline stand-in was made):\n{exc}"
    return str(exc)


__all__: list[str] = [
    "AuthError",
    "AuthStatus",
    "GCLOUD_CANDIDATE_DIRS",
    "as_auth_error",
    "describe_for_chat",
    "ensure_tool_dirs_on_path",
    "find_gcloud",
    "format_gcloud_failure",
    "gcloud_access_token",
    "gcloud_accounts",
    "gcloud_project",
    "http_status_of",
    "is_auth_failure",
    "preferred_account",
    "raise_if_auth_failure",
    "require_vertex_auth",
    "run_gcloud",
    "vertex_project",
]
