"""Environment and hardware detection: what this machine is, with no opinions.

Decisions based on these facts live in ``tuning.py``. NVIDIA GPUs (CUDA) and AMD GPUs
(ROCm, which PyTorch exposes through the same ``torch.cuda`` interface) are both handled.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import psutil

Environment = Literal["colab", "runpod", "local"]

Vendor = Literal["nvidia", "amd"]

NVENC_ENCODERS = ("h264_nvenc", "hevc_nvenc", "av1_nvenc")
VAAPI_ENCODERS = ("h264_vaapi", "hevc_vaapi", "av1_vaapi")
CPU_ENCODERS = ("libx264", "libx265", "libsvtav1", "libaom-av1")

# PCI vendor ids, in order of preference for VAAPI encoding.
_VAAPI_VENDORS = ("0x1002", "0x8086")  # AMD, Intel
# RDNA 2 chips that are not in the ROCm build targets but run as gfx1030 with an override.
_RDNA2_OVERRIDE = ("gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036")


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


def vaapi_device(sysfs: Path = Path("/sys/class/drm")) -> str | None:
    """The VAAPI render node to encode on, or ``None`` when there is none.

    ``VUPSCALE_VAAPI_DEVICE`` overrides the choice. Otherwise an AMD render node is
    preferred over an Intel one (a desktop may expose both), then any other.
    """
    override = os.environ.get("VUPSCALE_VAAPI_DEVICE")
    if override:
        return override
    nodes: list[tuple[int, str]] = []
    try:
        entries = sorted(sysfs.glob("renderD*"))
    except OSError:
        return None
    for entry in entries:
        try:
            vendor = (entry / "device" / "vendor").read_text().strip().lower()
        except OSError:
            vendor = ""
        rank = _VAAPI_VENDORS.index(vendor) if vendor in _VAAPI_VENDORS else len(_VAAPI_VENDORS)
        nodes.append((rank, f"/dev/dri/{entry.name}"))
    return min(nodes)[1] if nodes else None


def encoder_input_args(encoder: str) -> list[str]:
    """Global FFmpeg arguments an encoder needs before the inputs."""
    if encoder.endswith("_vaapi"):
        device = vaapi_device()
        return ["-vaapi_device", device] if device else []
    return []


def encoder_upload_filter(encoder: str, pix_fmt: str = "yuv420p") -> str:
    """Last filters before the encoder: a CPU pixel format, or an upload to the GPU."""
    if encoder.endswith("_vaapi"):
        surface = "p010" if "10" in pix_fmt else "nv12"
        return f"format={surface},hwupload"
    return f"format={pix_fmt}"


@lru_cache(maxsize=64)
def encoder_works(encoder: str, width: int = 256, height: int = 256) -> bool:
    """Run a REAL one-frame encode at ``width`` x ``height``.

    Listing ``ffmpeg -encoders`` is not enough: NVENC and VAAPI encoders are compiled
    into most builds but fail at runtime on GPUs without the hardware block (A100, H100),
    without the codec (AV1 on a T4) or above the size the hardware accepts.
    """
    if shutil.which("ffmpeg") is None:
        return False
    device_args = encoder_input_args(encoder)
    if encoder.endswith("_vaapi") and not device_args:
        return False
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", *device_args,
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r=25",
        "-frames:v", "1", "-vf", encoder_upload_filter(encoder), "-c:v", encoder,
        "-f", "null", "-",
    ]  # fmt: skip
    try:
        return subprocess.run(cmd, capture_output=True, timeout=60, check=False).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _amd_generation(arch: str) -> str:
    prefixes = (
        ("gfx120", "RDNA 4"),
        ("gfx115", "RDNA 3.5"),
        ("gfx110", "RDNA 3"),
        ("gfx103", "RDNA 2"),
        ("gfx101", "RDNA 1"),
        ("gfx95", "CDNA 4"),
        ("gfx94", "CDNA 3"),
        ("gfx90a", "CDNA 2"),
        ("gfx908", "CDNA"),
    )
    return next((name for prefix, name in prefixes if arch.startswith(prefix)), arch or "AMD")


@dataclass
class GpuInfo:
    name: str
    compute_cap: tuple[int, int]
    vram_total_gb: float
    vram_free_gb: float
    vendor: Vendor = "nvidia"
    arch: str = ""  # AMD target, e.g. "gfx1200"; empty on NVIDIA

    @property
    def is_amd(self) -> bool:
        return self.vendor == "amd"

    @property
    def label(self) -> str:
        """Short architecture id: ``sm_89`` on NVIDIA, ``gfx1200`` on AMD."""
        if self.is_amd:
            return self.arch or "gfx?"
        major, minor = self.compute_cap
        return f"sm_{major}{minor}"

    @property
    def generation(self) -> str:
        if self.is_amd:
            return _amd_generation(self.arch)
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
        return names.get(self.compute_cap, self.label)

    @property
    def native_bf16(self) -> bool:
        """bfloat16 runs on matrix hardware rather than being emulated.

        On AMD, PyTorch reports the gfx target as the "capability" (gfx1030 -> (10, 3)),
        so NVIDIA's ``major >= 8`` rule would wrongly include RDNA 2.
        """
        if self.is_amd:
            return self.arch.startswith(("gfx11", "gfx12", "gfx90a", "gfx94", "gfx95"))
        return self.compute_cap[0] >= 8


def gpu_info(index: int = 0) -> GpuInfo | None:
    """Details of a CUDA or ROCm device, or ``None`` when there is no usable GPU."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info(index)
    amd = getattr(torch.version, "hip", None) is not None
    arch = ""
    if amd:
        props = torch.cuda.get_device_properties(index)
        arch = str(getattr(props, "gcnArchName", "")).split(":")[0]
    return GpuInfo(
        name=torch.cuda.get_device_name(index),
        compute_cap=tuple(torch.cuda.get_device_capability(index)),  # type: ignore[arg-type]
        vram_total_gb=total / 2**30,
        vram_free_gb=free / 2**30,
        vendor="amd" if amd else "nvidia",
        arch=arch,
    )


def gpu_compute_check() -> str | None:
    """Run a small half-precision convolution on the GPU. ``None`` means it worked.

    Catches the GPU that PyTorch can see but cannot run kernels on, typically an AMD
    chip that is not among the targets of the installed ROCm build.
    """
    try:
        import torch

        x = torch.rand(1, 3, 64, 64, device="cuda", dtype=torch.float16)
        w = torch.rand(8, 3, 3, 3, device="cuda", dtype=torch.float16)
        torch.nn.functional.conv2d(x, w, padding=1).sum().item()
    except Exception as exc:  # anything at all means "not usable"
        return f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}".strip()
    return None


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
    rocm_build: str | None = None
    gpu_error: str | None = None  # result of gpu_compute_check()
    vaapi: str | None = None  # render node used for VAAPI
    amd_device_present: bool = False  # /dev/kfd exists
    amd_device_usable: bool = False  # ... and this user may open it
    hsa_override: str | None = None

    @property
    def problems(self) -> list[str]:
        issues: list[str] = []
        if self.ffmpeg is None:
            issues.append("ffmpeg not found (run scripts/setup_cloud.sh or apt-get install ffmpeg)")
        if not self.ffprobe:
            issues.append("ffprobe not found (it ships with ffmpeg)")
        if self.torch is None:
            issues.append("PyTorch is not installed")
        h264 = ("libx264", "h264_nvenc", "h264_vaapi")
        if self.ffmpeg and not any(self.encoders.get(e) for e in h264):
            issues.append("no working H.264 encoder in this ffmpeg build")
        if self.gpu is not None and self.gpu_error:
            hint = ""
            if self.gpu.is_amd and self.gpu.arch in _RDNA2_OVERRIDE and not self.hsa_override:
                hint = " (try HSA_OVERRIDE_GFX_VERSION=10.3.0)"
            issues.append(
                f"the GPU is visible but a test convolution failed{hint}: {self.gpu_error}"
            )
        return issues

    @property
    def warnings(self) -> list[str]:
        notes: list[str] = []
        if self.torch is not None and self.gpu is None:
            if self.amd_device_present and self.rocm_build is None:
                notes.append(
                    "an AMD GPU is present but this PyTorch build has no ROCm support: "
                    "install the ROCm build of PyTorch (see docs/getting-started.md)"
                )
            elif self.amd_device_present and not self.amd_device_usable:
                notes.append(
                    "no permission to use the AMD GPU: add your user to the 'render' and "
                    "'video' groups, then log in again"
                )
            else:
                notes.append("no GPU detected: inference will run on CPU (very slow)")
        if self.gpu is not None:
            hardware = NVENC_ENCODERS if not self.gpu.is_amd else VAAPI_ENCODERS
            if not any(self.encoders.get(e) for e in hardware):
                if self.gpu.is_amd:
                    notes.append(
                        "no working VAAPI encoder: encoding will use the CPU "
                        "(check the Mesa VA-API driver; see docs/troubleshooting.md)"
                    )
                else:
                    notes.append("GPU has no working NVENC: encoding will use the CPU")
            if self.gpu.is_amd and self.gpu.arch in _RDNA2_OVERRIDE and not self.hsa_override:
                notes.append(
                    f"{self.gpu.arch} is not a ROCm build target; if inference fails, "
                    "set HSA_OVERRIDE_GFX_VERSION=10.3.0"
                )
        return notes


def collect_report() -> SystemReport:
    import platform

    try:
        import torch

        torch_version: str | None = torch.__version__
        cuda_build: str | None = torch.version.cuda
        rocm_build: str | None = getattr(torch.version, "hip", None)
    except ImportError:
        torch_version = cuda_build = rocm_build = None

    vm = psutil.virtual_memory()
    has_ffmpeg = shutil.which("ffmpeg") is not None
    names = (*NVENC_ENCODERS, *VAAPI_ENCODERS, *CPU_ENCODERS)
    encoders = {e: encoder_works(e) for e in names} if has_ffmpeg else {}
    gpu = gpu_info()
    kfd = Path("/dev/kfd")
    return SystemReport(
        environment=detect_environment(),
        python=platform.python_version(),
        torch=torch_version,
        cuda_build=cuda_build,
        gpu=gpu,
        cpu_count=os.cpu_count() or 1,
        ram_total_gb=vm.total / 2**30,
        ram_free_gb=vm.available / 2**30,
        ffmpeg=ffmpeg_version(),
        ffprobe=shutil.which("ffprobe") is not None,
        encoders=encoders,
        rocm_build=rocm_build,
        gpu_error=gpu_compute_check() if gpu is not None else None,
        vaapi=vaapi_device(),
        amd_device_present=kfd.exists(),
        amd_device_usable=kfd.exists() and os.access(kfd, os.R_OK | os.W_OK),
        hsa_override=os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
    )
