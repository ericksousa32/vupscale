"""Auto-tuning: turn the detected hardware plus the model traits into a ``RuntimeConfig``.

``hardware.py`` answers "what is this machine?"; this module answers "how should we run
on it?". Memory sizing is *measured* (see :func:`calibrate`) instead of estimated with a
formula, because activation memory differs by orders of magnitude between architectures.
"""

from __future__ import annotations

import gc
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import (
    Codec,
    CompileMode,
    Device,
    OptionsError,
    Precision,
    UpscaleOptions,
    compute_output_size,
)
from .hardware import Environment, GpuInfo, detect_environment, encoder_works, gpu_info
from .models import cache_dir

if TYPE_CHECKING:
    import numpy as np
    import torch

TILE_LADDER = (768, 512, 384, 256)
MAX_BATCH = 16
MAX_QUEUE_DEPTH = 8
VRAM_SAFETY = 0.75  # use at most 75% of the free VRAM (25% margin)
RAM_FRACTION = 0.15  # raw frames waiting in queues may use 15% of the free host RAM
H264_HW_MAX = 4096  # widest frame H.264 hardware encoders accept (NVENC and AMD VCN)
HEVC_HW_MAX = 8192
HW_BACKENDS = ("nvenc", "vaapi")  # tried in this order by --codec auto
COMPILE_MIN_FRAMES = 1500

# (units, tile_size) -> peak bytes. ``units`` are whole frames when ``tile_size`` is None,
# otherwise tiles of ``tile_size`` px. Must raise an out-of-memory error when it does not fit.
Probe = Callable[[int, "int | None"], int]
# (frames, tile_size, tile_batch) -> upscaled frames
ProcessFn = Callable[["np.ndarray", "int | None", int], "np.ndarray"]


class TuningError(RuntimeError):
    """No configuration fits in memory."""


# --------------------------------------------------------------------------- memory helpers


def is_oom(exc: BaseException) -> bool:
    """True for CUDA/CPU out-of-memory errors, including the ones cuDNN raises as RuntimeError."""
    if isinstance(exc, MemoryError):
        return True
    try:
        import torch

        oom_type = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
        if isinstance(exc, oom_type):
            return True
    except ImportError:
        pass
    text = str(exc)
    return isinstance(exc, RuntimeError) and (
        "out of memory" in text.lower() or "CUDNN_STATUS_ALLOC_FAILED" in text
    )


def free_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def measure_peak(fn: Callable[[], Any], device: torch.device) -> int:
    """Run ``fn`` and return the peak CUDA memory allocated meanwhile (0 on CPU)."""
    import torch

    if device.type != "cuda":
        fn()
        return 0
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device))


def frame_ram_bytes(in_size: tuple[int, int], out_size: tuple[int, int]) -> int:
    """Host RAM held by one frame in flight: decoded RGB input plus upscaled RGB output."""
    return 3 * (in_size[0] * in_size[1] + out_size[0] * out_size[1])


def max_batch_for_ram(frame_bytes: int, ram_free_bytes: float) -> int:
    return max(1, int(ram_free_bytes * RAM_FRACTION // (2 * frame_bytes)))


def queue_depth_for_ram(batch_size: int, frame_bytes: int, ram_free_bytes: float) -> int:
    """Batches each queue (decode->GPU and GPU->encode) may hold within the RAM budget."""
    depth = int(ram_free_bytes * RAM_FRACTION // (2 * batch_size * frame_bytes))
    return max(1, min(MAX_QUEUE_DEPTH, depth))


def tiles_per_frame(height: int, width: int, tile: int, overlap: int) -> int:
    step = max(1, tile - overlap)

    def count(length: int) -> int:
        return 1 if length <= tile else math.ceil((length - overlap) / step)

    return count(height) * count(width)


# --------------------------------------------------------------------------- decisions


@dataclass(frozen=True)
class ModelTraits:
    """The parts of a spandrel descriptor that drive tuning."""

    scale: int
    supports_half: bool = True
    supports_bfloat16: bool = True
    tiling_recommended: bool = True

    @classmethod
    def from_descriptor(cls, descriptor: Any) -> ModelTraits:
        return cls(
            scale=int(descriptor.scale),
            supports_half=bool(descriptor.supports_half),
            supports_bfloat16=bool(descriptor.supports_bfloat16),
            tiling_recommended=getattr(descriptor.tiling, "name", "SUPPORTED") == "SUPPORTED",
        )


def select_dtype(
    precision: Precision,
    device_type: str,
    compute_cap: tuple[int, int],
    traits: ModelTraits,
    *,
    native_bf16: bool | None = None,
) -> tuple[torch.dtype, list[str]]:
    """fp16 where the model supports it, then bf16 (where the GPU runs it natively), then fp32.

    ``native_bf16`` comes from ``GpuInfo.native_bf16``; without it the NVIDIA rule
    (compute capability 8.0+) is applied to ``compute_cap``.

    fp16 has 10 mantissa bits against 7 for bf16 (less banding) at the same speed, so bf16
    is only the fallback for models that overflow in fp16.
    """
    import torch

    notes: list[str] = []
    if device_type != "cuda":
        if precision not in (Precision.auto, Precision.fp32):
            notes.append(f"precision {precision.value} ignored on CPU, using fp32")
        return torch.float32, notes

    if native_bf16 is None:
        native_bf16 = compute_cap[0] >= 8
    if precision is Precision.auto:
        if traits.supports_half:
            return torch.float16, notes
        if traits.supports_bfloat16 and native_bf16:
            notes.append("model does not support fp16, using bf16")
            return torch.bfloat16, notes
        notes.append("model does not support half precision on this GPU, using fp32 (slower)")
        return torch.float32, notes
    if precision is Precision.fp16:
        if not traits.supports_half:
            notes.append("model declares no fp16 support: expect black frames or artifacts")
        return torch.float16, notes
    if precision is Precision.bf16:
        if not traits.supports_bfloat16:
            notes.append("model declares no bf16 support: expect artifacts")
        if not native_bf16:
            notes.append("this GPU has no native bf16 acceleration: it will be slow")
        return torch.bfloat16, notes
    return torch.float32, notes


@dataclass(frozen=True)
class EncoderChoice:
    encoder: str  # ffmpeg encoder name
    preset: str
    notes: tuple[str, ...] = ()

    @property
    def is_hardware(self) -> bool:
        return self.encoder.endswith(tuple(f"_{backend}" for backend in HW_BACKENDS))


def select_encoder(
    codec: Codec,
    width: int,
    height: int,
    cpu_count: int,
    works: Callable[[str], bool] = encoder_works,
) -> EncoderChoice:
    """Pick the ffmpeg encoder for the OUTPUT resolution.

    ``auto`` uses the first hardware encoder whose real test encode succeeds: NVENC, then
    VAAPI (AMD and Intel on Linux), then the CPU. It switches to HEVC above 4096 px, the
    limit of H.264 encoders on both NVIDIA and AMD hardware.
    """
    notes: list[str] = []
    longest = max(width, height)
    few_cores = cpu_count <= 4

    def cpu(family: str) -> EncoderChoice:
        if family == "h264":
            return EncoderChoice("libx264", "faster" if few_cores else "medium", tuple(notes))
        if family == "hevc":
            return EncoderChoice("libx265", "faster" if few_cores else "medium", tuple(notes))
        if works("libsvtav1"):
            return EncoderChoice("libsvtav1", "8" if few_cores else "6", tuple(notes))
        return EncoderChoice("libaom-av1", "6", tuple(notes))

    if codec in (Codec.h264, Codec.hevc, Codec.av1):
        return cpu(codec.value)

    explicit = codec is not Codec.auto
    if explicit:
        family, backend = codec.value.split("_", 1)
        backends: tuple[str, ...] = (backend,)
    else:
        family, backends = "h264", HW_BACKENDS
    if family == "h264" and longest > H264_HW_MAX:
        if explicit:
            notes.append(f"{codec.value} is limited to {H264_HW_MAX} px, switching to HEVC")
        family = "hevc"
    if longest > HEVC_HW_MAX:
        notes.append(f"output above {HEVC_HW_MAX} px exceeds hardware encoder limits, using CPU")
        return cpu(family)
    for backend in backends:
        encoder = f"{family}_{backend}"
        if works(encoder):
            return EncoderChoice(encoder, "p5" if backend == "nvenc" else "", tuple(notes))
    if explicit:
        notes.append(f"{family}_{backends[0]} does not work on this machine, encoding on CPU")
    return cpu(family)


def should_compile(
    mode: CompileMode,
    device_type: str,
    compute_cap: tuple[int, int],
    total_frames: int,
    vendor: str = "nvidia",
    arch: str = "",
) -> bool:
    """``auto`` compiles only where it was measured to pay off, and only for jobs long
    enough to amortise the compile time.

    Measured: +31% on an RTX 4090 (Ampere+ rule), +32% on an RX 9060 XT (RDNA 4).
    RDNA 2 and RDNA 3 stay opt-in (``--compile on``) until measured.
    """
    if mode is CompileMode.on:
        return True
    if mode is CompileMode.off:
        return False
    if device_type != "cuda" or total_frames < COMPILE_MIN_FRAMES:
        return False
    if vendor == "amd":
        return arch.startswith("gfx12")
    return compute_cap[0] >= 8


# --------------------------------------------------------------------------- calibration


@dataclass(frozen=True)
class Calibration:
    tile_size: int | None
    units: int  # frames per model call (no tiling) or tiles per model call
    peak_bytes: int = 0
    from_cache: bool = False


def tile_candidates(tile_option: int, frame_hw: tuple[int, int]) -> list[int | None]:
    """Tile sizes to try, best first. A tile as large as the frame means no tiling."""
    longest = max(frame_hw)
    if tile_option == -1:
        return [None]
    if tile_option > 0:
        return [tile_option if tile_option < longest else None]
    return [None, *(t for t in TILE_LADDER if t < longest)]


def calibration_cache_file() -> Path:
    return cache_dir().parent / "calibration.json"


def _load_cache(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_cache(path: Path, data: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
        os.replace(tmp, path)
    except OSError:
        pass  # the cache is an optimisation, never a reason to fail


def calibrate(
    probe: Probe,
    *,
    frame_hw: tuple[int, int],
    budget_bytes: float,
    tile_option: int = 0,
    fixed_frames: int = 0,
    max_frames: int = MAX_BATCH,
    max_tiles: int = MAX_BATCH,
    tile_overlap: int | None = None,
    cache_key: str | None = None,
    cache_file: Path | None = None,
) -> Calibration:
    """Find the largest workload per model call that really fits, by running it.

    Tries the whole frame first, then progressively smaller tiles. For the first size
    that fits, two measurements give the cost per extra unit, which is extrapolated and
    then verified. Results are cached per (GPU, model, resolution, dtype).
    """

    def attempt(units: int, tile: int | None) -> int | None:
        failed = False
        peak = 0
        try:
            peak = probe(units, tile)
        except Exception as exc:
            if not is_oom(exc):
                raise
            failed = True
        if failed:  # cleaned up outside the except block so the traceback frees its tensors
            free_memory()
            return None
        return peak if peak <= budget_bytes else None

    def steady(units: int, tile: int | None) -> int | None:
        """Two runs back to back. The first run of a shape is the easy one: cuDNN empties
        the allocator cache to autotune. Only the second shows the steady state a render
        lives in (measured on an RTX 4090: a batch passed once, then failed every time)."""
        first = attempt(units, tile)
        if first is None:
            return None
        second = attempt(units, tile)
        return None if second is None else max(first, second)

    path = cache_file or calibration_cache_file()
    cache = _load_cache(path) if cache_key else {}
    hit = cache.get(cache_key) if cache_key else None
    if isinstance(hit, dict) and isinstance(hit.get("units"), int):
        tile = hit.get("tile")
        peak = steady(hit["units"], tile)  # confirm it still fits today
        if peak is not None:
            return Calibration(tile, hit["units"], peak, from_cache=True)

    result: Calibration | None = None
    for tile in tile_candidates(tile_option, frame_hw):
        if tile is None and fixed_frames > 0:
            peak = attempt(fixed_frames, None)
            if peak is None:
                continue
            result = Calibration(None, fixed_frames, peak)
            break
        peak1 = attempt(1, tile)
        if peak1 is None:
            continue
        limit = max_frames
        if tile is not None:
            # Tiles are probed on a single frame, so one frame's worth is the ceiling.
            limit = max_tiles
            if tile_overlap is not None:
                limit = min(limit, tiles_per_frame(*frame_hw, tile, tile_overlap))
        result = Calibration(tile, 1, peak1)
        if limit >= 2 and (peak2 := attempt(2, tile)) is not None:
            result = Calibration(tile, 2, peak2)
            per_unit = max(peak2 - peak1, 1)
            units = min(limit, max(2, int((budget_bytes - peak1) // per_unit) + 1))
            while units > 2:
                peak = attempt(units, tile)
                if peak is not None:
                    result = Calibration(tile, units, peak)
                    break
                units = max(2, units // 2)
        break

    # The winner must also hold in steady state, or it steps down.
    while result is not None and result.units > 1:
        peak = steady(result.units, result.tile_size)
        if peak is not None:
            result = Calibration(result.tile_size, result.units, peak)
            break
        result = Calibration(result.tile_size, max(1, result.units // 2), result.peak_bytes)

    if result is None:
        raise TuningError(
            "the model does not fit in memory for this video, even with the smallest tiles. "
            "Try a lighter model, --pre-height to shrink the input, or --precision fp16."
        )
    if cache_key:
        cache[cache_key] = {"tile": result.tile_size, "units": result.units}
        _save_cache(path, cache)
    return result


# --------------------------------------------------------------------------- runtime config


@dataclass
class RuntimeConfig:
    environment: Environment
    device: torch.device
    dtype: torch.dtype
    batch_size: int  # frames read and processed together
    tile_size: int | None
    tile_overlap: int
    tile_batch: int  # tiles per model call when tiling
    compile_model: bool
    video_codec: str  # ffmpeg encoder name
    encoder_preset: str
    queue_depth: int
    out_width: int
    out_height: int
    gpu_name: str
    compute_cap: tuple[int, int]
    vram_total_gb: float
    vram_free_gb: float
    ram_free_gb: float
    cpu_count: int
    calibrated: bool = False
    notes: list[str] = field(default_factory=list)
    gpu_vendor: str = ""  # "nvidia", "amd", or "" on CPU
    gpu_label: str = ""  # "sm_89", "gfx1200", or "" on CPU

    @property
    def encoder_label(self) -> str:
        """``hevc_nvenc (p5)``, or just ``hevc_vaapi`` for encoders without a preset."""
        return (
            f"{self.video_codec} ({self.encoder_preset})"
            if self.encoder_preset
            else self.video_codec
        )

    @property
    def precision_name(self) -> str:
        return {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}.get(
            str(self.dtype).removeprefix("torch."), str(self.dtype)
        )


_UNSET: Any = object()


def build_runtime_config(
    options: UpscaleOptions,
    traits: ModelTraits,
    *,
    width: int,
    height: int,
    total_frames: int = 0,
    make_probe: Callable[[torch.device, torch.dtype], Probe] | None = None,
    model_id: str = "",
    gpu: GpuInfo | None = _UNSET,
    ram_free_bytes: float | None = None,
    cpu_count: int | None = None,
    works: Callable[[str], bool] | None = None,
    use_cache: bool = True,
) -> RuntimeConfig:
    """Resolve every ``auto`` in ``options`` for a video of ``width`` x ``height``.

    ``width``/``height`` are the size fed to the model (after any pre-scale).
    ``make_probe`` builds the measuring function once device and dtype are known; without
    it (e.g. ``--dry-run``) conservative rules of thumb are used instead of calibration.
    """
    import psutil
    import torch

    if gpu is _UNSET:
        gpu = gpu_info()
    if ram_free_bytes is None:
        ram_free_bytes = float(psutil.virtual_memory().available)
    cpu_count = cpu_count or os.cpu_count() or 1
    notes: list[str] = []

    if options.device is Device.cuda and gpu is None:
        raise OptionsError("--device cuda requested but no CUDA GPU is available")
    on_gpu = gpu is not None and options.device is not Device.cpu
    device = torch.device("cuda" if on_gpu else "cpu")
    compute_cap = gpu.compute_cap if on_gpu and gpu else (0, 0)

    dtype, dtype_notes = select_dtype(
        options.precision,
        device.type,
        compute_cap,
        traits,
        native_bf16=gpu.native_bf16 if on_gpu and gpu else None,
    )
    notes += dtype_notes

    out_w, out_h = compute_output_size(
        width, height, traits.scale, options.out_scale, options.target_height
    )
    if works is None:

        def works(encoder: str) -> bool:
            # Hardware encoders are tested at the real output size: limits differ by chip.
            return encoder_works(encoder, out_w, out_h)

    encoder = select_encoder(options.codec, out_w, out_h, cpu_count, works)
    notes += encoder.notes

    frame_bytes = frame_ram_bytes((width, height), (out_w, out_h))
    ram_batch_cap = max_batch_for_ram(frame_bytes, ram_free_bytes)
    frame_hw = (height, width)
    calibrated = False

    if on_gpu and gpu and make_probe is not None:
        key = None
        if use_cache:
            key = "|".join(
                str(part)
                for part in (
                    "v2", gpu.name, model_id, f"{width}x{height}", dtype, torch.__version__,
                    f"tile={options.tile_size}", f"batch={options.batch_size}",
                )
            )  # fmt: skip
        found = calibrate(
            make_probe(device, dtype),
            frame_hw=frame_hw,
            budget_bytes=gpu.vram_free_gb * 2**30 * VRAM_SAFETY,
            tile_option=options.tile_size,
            fixed_frames=options.batch_size,
            max_frames=min(MAX_BATCH, ram_batch_cap),
            tile_overlap=options.tile_overlap,
            cache_key=key,
        )
        tile_size, units, calibrated = found.tile_size, found.units, True
    else:
        candidates = tile_candidates(options.tile_size, frame_hw)
        forced = options.tile_size != 0
        if forced:
            tile_size = candidates[0]
        elif on_gpu and gpu:
            roomy = gpu.vram_total_gb >= 20 or width * height < 1920 * 1080
            tile_size = None if roomy else next((t for t in candidates if t and t <= 512), None)
        else:
            small = width * height <= 768 * 768
            tile_size = None if small else next((t for t in candidates if t and t <= 512), None)
        big_gpu = bool(on_gpu and gpu and gpu.vram_total_gb >= 20)
        units = (4 if big_gpu else 1) if tile_size is None else (4 if on_gpu else 1)
        if on_gpu:
            notes.append("memory not calibrated yet: batch and tile are conservative estimates")

    if tile_size is None:
        batch_size = options.batch_size or min(units, ram_batch_cap)
        tile_batch = 0
    else:
        per_frame = tiles_per_frame(height, width, tile_size, options.tile_overlap)
        tile_batch = min(units, per_frame)
        batch_size = options.batch_size or 1  # with tiling the tiles are the batch
        if not traits.tiling_recommended and options.tile_size == 0:
            notes.append("this model discourages tiling, but the full frame does not fit in VRAM")
    if batch_size > ram_batch_cap:
        notes.append(
            f"batch size {batch_size} exceeds the host RAM budget ({ram_batch_cap} frames)"
        )

    return RuntimeConfig(
        environment=detect_environment(),
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        tile_size=tile_size,
        tile_overlap=options.tile_overlap,
        tile_batch=tile_batch,
        compile_model=should_compile(
            options.compile,
            device.type,
            compute_cap,
            total_frames,
            vendor=gpu.vendor if on_gpu and gpu else "",
            arch=gpu.arch if on_gpu and gpu else "",
        ),
        video_codec=encoder.encoder,
        encoder_preset=encoder.preset,
        queue_depth=queue_depth_for_ram(batch_size, frame_bytes, ram_free_bytes),
        out_width=out_w,
        out_height=out_h,
        gpu_name=gpu.name if on_gpu and gpu else "CPU",
        compute_cap=compute_cap,
        vram_total_gb=gpu.vram_total_gb if on_gpu and gpu else 0.0,
        vram_free_gb=gpu.vram_free_gb if on_gpu and gpu else 0.0,
        ram_free_gb=ram_free_bytes / 2**30,
        cpu_count=cpu_count,
        calibrated=calibrated,
        notes=notes,
        gpu_vendor=gpu.vendor if on_gpu and gpu else "",
        gpu_label=gpu.label if on_gpu and gpu else "",
    )


# --------------------------------------------------------------------------- OOM recovery


class AdaptiveBatchController:
    """Runs batches and, on out-of-memory, degrades the settings and retries.

    No frame is lost and order is preserved. Degradation order: tiles per call -> frames
    per batch -> enable tiling / smaller tiles. Changes stick for the rest of the render.
    """

    def __init__(
        self,
        batch_size: int,
        tile_size: int | None,
        tile_batch: int,
        frame_hw: tuple[int, int],
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        self.batch_size = max(1, batch_size)
        self.tile_size = tile_size
        self.tile_batch = max(1, tile_batch) if tile_size is not None else 0
        self.frame_hw = frame_hw
        self.events: list[str] = []
        self._on_event = on_event

    @classmethod
    def from_config(cls, cfg: RuntimeConfig, frame_hw: tuple[int, int], **kw: Any):
        return cls(cfg.batch_size, cfg.tile_size, cfg.tile_batch, frame_hw, **kw)

    def _record(self, message: str) -> None:
        self.events.append(message)
        if self._on_event is not None:
            self._on_event(message)

    def _degrade(self) -> None:
        if self.tile_size is not None and self.tile_batch > 1:
            self.tile_batch //= 2
            self._record(f"out of memory: tiles per call reduced to {self.tile_batch}")
            return
        if self.batch_size > 1:
            self.batch_size //= 2
            self._record(f"out of memory: batch size reduced to {self.batch_size}")
            return
        ceiling = self.tile_size if self.tile_size is not None else max(self.frame_hw)
        smaller = next((t for t in TILE_LADDER if t < ceiling), None)
        if smaller is None:
            raise TuningError(
                "out of memory even with batch size 1 and the smallest tiles. "
                "Try a lighter model or shrink the input with --pre-height."
            )
        self.tile_size, self.tile_batch = smaller, 1
        self._record(f"out of memory: tile size set to {smaller}")

    def run(self, process: ProcessFn, frames: np.ndarray) -> np.ndarray:
        """Process ``frames`` (N, H, W, 3), splitting and retrying as memory allows."""
        import numpy as np

        outputs: list[np.ndarray] = []
        done, total = 0, len(frames)
        while done < total:
            chunk = frames[done : done + self.batch_size]
            out = None
            try:
                out = process(chunk, self.tile_size, self.tile_batch)
            except Exception as exc:
                if not is_oom(exc):
                    raise
            if out is None:  # handled here so the failed call's tensors are released first
                free_memory()
                self._degrade()
                continue
            if done == 0 and len(chunk) == total:
                return out  # the common case: one call, result passed through untouched
            # Split batch: the engine may reuse its output buffer on the next call.
            outputs.append(out.copy())
            done += len(chunk)
        return np.concatenate(outputs)
