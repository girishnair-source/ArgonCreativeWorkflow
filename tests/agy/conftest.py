"""Shared fixtures for the AGY Film Studio test-suite (tests/agy). No network access is ever required."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from agy_studio.registry import DEFAULT_REGISTRY_PATH, Registry
from agy_studio.testing import build_demo_project


@pytest.fixture
def productions_dir(tmp_path: Path) -> Path:
    d = tmp_path / "productions"
    d.mkdir()
    return d


@pytest.fixture
def demo_project(productions_dir: Path):
    """Fully locked demo production (packs, voices, environments locked; SC01 has two CreativeShots)."""
    return build_demo_project(productions_dir)


@pytest.fixture
def unlocked_project(productions_dir: Path):
    """Same production with every bible record still `proposed` (pre-G1 / pre-G2)."""
    return build_demo_project(productions_dir, lock=False)


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    """A private copy of registry.yaml so tests can edit it (hot reload) without touching the real file."""
    path = tmp_path / "registry.yaml"
    shutil.copy2(DEFAULT_REGISTRY_PATH, path)
    return Registry(path)


@pytest.fixture
def ffmpeg_available() -> bool:
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg/ffprobe not installed")
    return True
