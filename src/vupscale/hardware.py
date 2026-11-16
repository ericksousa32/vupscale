"""Environment and hardware detection.

Stage 1 ships the probes used by ``vupscale doctor``. Auto-tuning (``RuntimeConfig``,
empirical memory calibration, ``AdaptiveBatchController``) arrives in stage 2.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

import psutil

Environment = Literal["colab", "runpod", "local"]

NVENC_ENCODERS = ("h264_nvenc", "hevc_nvenc", "av1_nvenc")
CPU_ENCODERS = ("libx264", "libx265", "libsvtav1", "libaom-av1")


def detect_environment() -> Environment:
    """Colab and RunPod are identified by what only they provide, not by folder names."""
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return "colab"
    if any(key.startswith("RUNPOD_") for key in os.environ):
        return "runpod"
    try:
        if importlib.util.find_spec("google.colab") is not None:
            return "colab"
    except (ImportError, ValueError):
        pass
    return "local"


def ffmpeg_version(binary: str = "ffmpeg") -> str | None:
    """First token of ``ffmpeg -version`` (e.g. ``6.1.1``), or ``None`` if not installed."""
    if shutil.which(binary) is None:
        return None
    try:
        out = subprocess.run(
            [binary, "-version"], capture_output=True, text=True, timeout=10, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    parts = out.split()
    return parts[2] if len(parts) >= 3 and parts[1] == "version" else None


@lru_cache(maxsize=32)
def encoder_works(encoder: str) -> bool:
    """Run a REAL one-frame encode.

    Listing ``ffmpeg -encoders`` is not enough: the NVENC encoders are compiled into most
    builds but fail at runtime on GPUs without the hardware block (A100, H100) or without
    the codec (AV1 on a T4).
    """
    if shutil.which("ffmpeg") is None:
        return False
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "lavfi", "-i", "color=c=black:s=256x256:r=25",
        "-frames:v", "1", "-pix_fmt", "yuv420p", "-c:v", encoder, "-f", "null", "-",
    ]  # fmt: skip
    try:
        return subprocess.run(cmd, capture_output=True, timeout=30, check=False).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@dataclass
class GpuInfo:
    name: str
    compute_cap: tuple[int, int]
    vram_total_gb: float
    vram_free_gb: float

    @property
    def generation(self) -> str:
        major, minor = self.compute_cap
        names = {
            (6, 1): "Pascal",
            (7, 0): "Volta",
            (7, 5): "Turing",
            (8, 0): "Ampere",
            (8, 6): "Ampere",
            (8, 7): "Ampere",
            (8, 9): "Ada Lovelace",
            (9, 0): "Hopper",
            (10, 0): "Blackwell",
            (12, 0): "Blackwell",
        }
        return names.get((major, minor), f"sm_{major}{minor}")


def gpu_info(index: int = 0) -> GpuInfo | None:
    """Details of a CUDA device, or ``None`` when CUDA is unavailable."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info(index)
    return GpuInfo(
        name=torch.cuda.get_device_name(index),
        compute_cap=torch.cuda.get_device_capability(index),
        vram_total_gb=total / 2**30,
        vram_free_gb=free / 2**30,
    )


@dataclass
class SystemReport:
    environment: Environment
    python: str
    torch: str | None
    cuda_build: str | None
    gpu: GpuInfo | None
    cpu_count: int
    ram_total_gb: float
    ram_free_gb: float
    ffmpeg: str | None
    ffprobe: bool
    encoders: dict[str, bool] = field(default_factory=dict)

    @property
    def problems(self) -> list[str]:
        issues: list[str] = []
        if self.ffmpeg is None:
            issues.append("ffmpeg not found (run scripts/setup_cloud.sh or apt-get install ffmpeg)")
        if not self.ffprobe:
            issues.append("ffprobe not found (it ships with ffmpeg)")
        if self.torch is None:
            issues.append("PyTorch is not installed")
        if self.ffmpeg and not any(self.encoders.get(e) for e in ("libx264", "h264_nvenc")):
            issues.append("no working H.264 encoder in this ffmpeg build")
        return issues

    @property
    def warnings(self) -> list[str]:
        notes: list[str] = []
        if self.torch is not None and self.gpu is None:
            notes.append("no CUDA GPU detected: inference will run on CPU (very slow)")
        if self.gpu is not None and not any(self.encoders.get(e) for e in NVENC_ENCODERS):
            notes.append("GPU has no working NVENC: encoding will use the CPU")
        return notes


def collect_report() -> SystemReport:
    import platform

    try:
        import torch

        torch_version: str | None = torch.__version__
        cuda_build: str | None = torch.version.cuda
    except ImportError:
        torch_version = cuda_build = None

    vm = psutil.virtual_memory()
    has_ffmpeg = shutil.which("ffmpeg") is not None
    encoders = {e: encoder_works(e) for e in (*NVENC_ENCODERS, *CPU_ENCODERS)} if has_ffmpeg else {}
    return SystemReport(
        environment=detect_environment(),
        python=platform.python_version(),
        torch=torch_version,
        cuda_build=cuda_build,
        gpu=gpu_info(),
        cpu_count=os.cpu_count() or 1,
        ram_total_gb=vm.total / 2**30,
        ram_free_gb=vm.available / 2**30,
        ffmpeg=ffmpeg_version(),
        ffprobe=shutil.which("ffprobe") is not None,
        encoders=encoders,
    )
