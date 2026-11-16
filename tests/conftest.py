"""Shared fixtures: everything is generated on the fly so CI needs no downloads and no GPU."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def make_video(
    path: Path,
    *,
    seconds: float = 2.0,
    size: str = "320x240",
    fps: int = 24,
    audio: bool = True,
) -> Path:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    cmd += ["-f", "lavfi", "-i", f"testsrc2=s={size}:r={fps}:d={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True)
    return path


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """2 s, 320x240, 24 fps (48 frames) with a sine audio track."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    return make_video(tmp_path_factory.mktemp("media") / "sample.mp4")


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A random-weight x2 Compact (SRVGGNet) model that spandrel recognises. Runs on CPU."""
    import torch
    from spandrel.architectures.Compact import Compact

    torch.manual_seed(0)
    net = Compact(num_feat=8, num_conv=2, upscale=2)
    path = tmp_path_factory.mktemp("models") / "tiny_x2.pth"
    torch.save(net.state_dict(), path)
    return path


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Never touch the real ~/.cache/vupscale during tests."""
    cache = tmp_path / "cache"
    monkeypatch.setenv("VUPSCALE_CACHE_DIR", str(cache))
    return cache


@pytest.fixture(autouse=True)
def wide_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep Rich from wrapping long temp paths so output assertions stay simple."""
    monkeypatch.setenv("COLUMNS", "300")
