"""Allow-listed subprocess execution (security rule: no shell, argv lists only, known binaries only).

Every external process the studio runs goes through :func:`run`. Arguments are passed as a list (never through a
shell), the binary must be on :data:`ALLOWED_BINARIES`, and NUL bytes are rejected. ``gcloud`` calls carry the
Antigravity attribution tag in ``CLOUDSDK_METRICS_ENVIRONMENT`` (matching the existing pipeline).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

ALLOWED_BINARIES = frozenset({"ffmpeg", "ffprobe", "osascript", "gcloud", "sqlite3"})
_GCLOUD_TAG = "datacloud.antigravity"


class DisallowedCommandError(PermissionError):
    """Raised when code tries to execute a binary that is not allow-listed."""


class CommandFailed(RuntimeError):
    def __init__(self, argv: list[str], returncode: int, stdout: str | bytes, stderr: str | bytes) -> None:
        self.argv = argv
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        tail = stderr[-800:] if isinstance(stderr, str) else stderr[-800:].decode("utf-8", "replace")
        super().__init__(f"{Path(argv[0]).name} exited {returncode}: {tail.strip()}")


_EXTRA_BIN_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")


def which(binary: str) -> str | None:
    found = shutil.which(binary)
    if found:
        return found
    for d in _EXTRA_BIN_DIRS:
        cand = Path(d) / binary
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def gcloud_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base if base is not None else os.environ)
    prev = env.get("CLOUDSDK_METRICS_ENVIRONMENT", "")
    if _GCLOUD_TAG not in prev.split():
        env["CLOUDSDK_METRICS_ENVIRONMENT"] = f"{prev} {_GCLOUD_TAG}".strip()
    return env


def run(
    argv: Sequence[str | Path],
    *,
    timeout: float = 600,
    check: bool = True,
    input_text: str | None = None,
    binary_output: bool = False,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    """Run an allow-listed binary with an argv list. Raises :class:`CommandFailed` on non-zero exit if ``check``."""
    if not argv:
        raise ValueError("empty argv")
    args = [str(a) for a in argv]
    name = Path(args[0]).name
    if name not in ALLOWED_BINARIES:
        raise DisallowedCommandError(f"{name!r} is not an allow-listed binary ({sorted(ALLOWED_BINARIES)})")
    if any("\x00" in a for a in args):
        raise ValueError("NUL byte in command arguments")
    exe = args[0] if os.path.isabs(args[0]) else which(args[0])
    if not exe:
        raise FileNotFoundError(f"{name} not found on PATH")
    if name == "gcloud":
        env = gcloud_env(env)
    proc = subprocess.run(  # noqa: S603 - argv list, allow-listed binary, shell=False
        [exe, *args[1:]],
        capture_output=True,
        text=not binary_output,
        input=input_text if not binary_output else None,
        timeout=timeout,
        env=env,
        cwd=str(cwd) if cwd else None,
        check=False,
        shell=False,
    )
    if check and proc.returncode != 0:
        raise CommandFailed(args, proc.returncode, proc.stdout, proc.stderr)
    return proc
