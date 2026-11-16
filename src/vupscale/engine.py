"""Model loading and GPU inference: uint8 in, uint8 out, optional seamless tiling."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from .config import UpscaleOptions, compute_output_size
from .models import ModelError
from .tuning import (
    ModelTraits,
    Probe,
    RuntimeConfig,
    build_runtime_config,
    is_oom,
    measure_peak,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def load_model(path: str | Path) -> Any:
    """Load a super-resolution checkpoint with spandrel and check it is usable for video."""
    from spandrel import ImageModelDescriptor, ModelLoader

    path = Path(path)
    try:
        descriptor = ModelLoader(device="cpu").load_from_file(path)
    except Exception as exc:
        raise ModelError(f"could not load {path.name} as a model: {exc}") from exc
    if not isinstance(descriptor, ImageModelDescriptor):
        raise ModelError(
            f"{path.name} is not an image-to-image model ({type(descriptor).__name__})"
        )
    if descriptor.purpose not in ("SR", "Restoration"):
        raise ModelError(f"{path.name} is a {descriptor.purpose} model, not a general upscaler")
    if descriptor.input_channels != 3 or descriptor.output_channels != 3:
        raise ModelError(
            f"{path.name} uses {descriptor.input_channels}->{descriptor.output_channels} "
            "channels; only RGB models are supported"
        )
    return descriptor


def model_id(path: Path) -> str:
    """Cheap identity for caches: name plus size."""
    return f"{path.name}:{path.stat().st_size}"


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Start offsets covering ``length``. The last tile is aligned to the end, so every
    tile has the same size (fixed shapes keep cuDNN autotuning and torch.compile happy)."""
    if length <= tile:
        return [0]
    step = max(1, tile - overlap)
    starts = list(range(0, length - tile, step))
    starts.append(length - tile)
    return starts


def blend_ramp(length: int, overlap: int, fade_start: bool, fade_end: bool) -> torch.Tensor:
    """1D blending weights: 1 in the middle, a raised-cosine fade on edges shared with a
    neighbour. The fade is close to zero right at the tile border, where the model has the
    least context, which is what hides the seams."""
    weights = torch.ones(length, dtype=torch.float32)
    n = min(overlap, length // 2)
    if n > 0:
        ramp = torch.sin(0.5 * math.pi * (torch.arange(n, dtype=torch.float32) + 0.5) / n) ** 2
        if fade_start:
            weights[:n] = ramp
        if fade_end:
            weights[-n:] = ramp.flip(0)
    return weights


class UpscaleEngine:
    """Runs a spandrel model on batches of RGB frames.

    ``process_batch`` takes and returns ``uint8`` arrays of shape ``(B, H, W, 3)``. Frames
    cross the PCIe bus as uint8 in both directions (4x less traffic than float32, and the
    output is ``scale**2`` times larger than the input); all conversions happen on the GPU.
    """

    def __init__(
        self,
        descriptor: Any,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        *,
        tile_overlap: int = 32,
        out_size: tuple[int, int] | None = None,  # (width, height); None = native scale
    ) -> None:
        self.descriptor = descriptor
        self.device = torch.device(device)
        self.dtype = dtype
        self.scale = int(descriptor.scale)
        self.tile_overlap = tile_overlap
        self.out_size = out_size
        self.notes: list[str] = []

        descriptor.eval()
        descriptor.to(self.device, dtype)
        self._model = descriptor.model.to(memory_format=torch.channels_last)
        self._call_fn: Callable[[Any, torch.Tensor], torch.Tensor] = getattr(
            descriptor, "_call_fn", lambda model, x: model(x)
        )
        self._compiled: Any = None
        self._fixed_units = 0  # pad model calls to this many items once compiled
        self._pinned_in: torch.Tensor | None = None
        self._pinned_out: torch.Tensor | None = None
        self._ramps: dict[tuple[int, int, bool, bool], torch.Tensor] = {}
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    # ------------------------------------------------------------------ setup

    def compile(self, fixed_units: int) -> None:
        """Compile the model. Shapes are kept fixed (``fixed_units`` items per call) so it
        never recompiles mid-render; any compile failure falls back to eager mode."""
        try:
            self._compiled = torch.compile(
                self._model, mode="max-autotune-no-cudagraphs", dynamic=False
            )
            self._fixed_units = max(1, fixed_units)
        except Exception as exc:
            self._drop_compiled(exc)

    def _drop_compiled(self, exc: BaseException) -> None:
        self._compiled = None
        self._fixed_units = 0
        self.notes.append(f"torch.compile unavailable, running eager ({type(exc).__name__})")

    # ------------------------------------------------------------------ model calls

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """One model call on ``(N, 3, h, w)`` in [0, 1]; returns ``(N, 3, h*s, w*s)``."""
        n, _, h, w = x.shape
        pad_w, pad_h = self.descriptor.size_requirements.get_padding(w, h)
        if pad_w or pad_h:
            reflect = (0, min(pad_w, w - 1), 0, min(pad_h, h - 1))
            x = F.pad(x, reflect, mode="reflect")
            rest = (0, pad_w - reflect[1], 0, pad_h - reflect[3])
            if rest[1] or rest[3]:
                x = F.pad(x, rest, mode="replicate")
        x = x.contiguous(memory_format=torch.channels_last)

        if self._compiled is not None:
            if n < self._fixed_units:  # keep the compiled shape: repeat the last item
                x = torch.cat([x, x[-1:].expand(self._fixed_units - n, -1, -1, -1)])
            try:
                y = self._call_fn(self._compiled, x)
            except Exception as exc:
                if is_oom(exc):
                    raise
                self._drop_compiled(exc)
                y = self._call_fn(self._model, x[:n])
        else:
            y = self._call_fn(self._model, x)
        return y[:n, :, : h * self.scale, : w * self.scale]

    def _ramp(self, length: int, fade_start: bool, fade_end: bool) -> torch.Tensor:
        key = (length, self.tile_overlap, fade_start, fade_end)
        if key not in self._ramps:
            self._ramps[key] = blend_ramp(
                length * self.scale, self.tile_overlap * self.scale, fade_start, fade_end
            ).to(self.device)
        return self._ramps[key]

    def _forward_tiled(self, x: torch.Tensor, tile: int, tile_batch: int) -> torch.Tensor:
        """Upscale in overlapping tiles, several tiles per model call, blended seamlessly."""
        b, c, h, w = x.shape
        s = self.scale
        th, tw = min(tile, h), min(tile, w)
        ys, xs = tile_starts(h, th, self.tile_overlap), tile_starts(w, tw, self.tile_overlap)
        if len(ys) == 1 and len(xs) == 1:
            return self._forward(x)

        out = torch.zeros((b, c, h * s, w * s), dtype=torch.float32, device=x.device)
        weight = torch.zeros((1, 1, h * s, w * s), dtype=torch.float32, device=x.device)
        for y0 in ys:
            wy = self._ramp(th, y0 > 0, y0 + th < h)
            for x0 in xs:
                wx = self._ramp(tw, x0 > 0, x0 + tw < w)
                weight[..., y0 * s : (y0 + th) * s, x0 * s : (x0 + tw) * s] += wy[:, None] * wx

        jobs = [(i, y0, x0) for i in range(b) for y0 in ys for x0 in xs]
        tile_batch = max(1, tile_batch)
        for start in range(0, len(jobs), tile_batch):
            group = jobs[start : start + tile_batch]
            tiles = torch.stack([x[i, :, y0 : y0 + th, x0 : x0 + tw] for i, y0, x0 in group])
            result = self._forward(tiles).float()
            for (i, y0, x0), piece in zip(group, result, strict=True):
                mask = self._ramp(th, y0 > 0, y0 + th < h)[:, None] * self._ramp(
                    tw, x0 > 0, x0 + tw < w
                )
                out[i, :, y0 * s : (y0 + th) * s, x0 * s : (x0 + tw) * s] += piece * mask
            del tiles, result
        return out.div_(weight)

    # ------------------------------------------------------------------ host <-> device

    def _upload(self, frames: np.ndarray) -> torch.Tensor:
        if self.device.type != "cuda":
            return torch.from_numpy(frames)
        if (
            self._pinned_in is None
            or self._pinned_in.shape[1:] != frames.shape[1:]
            or (self._pinned_in.shape[0] < frames.shape[0])
        ):
            self._pinned_in = torch.empty(frames.shape, dtype=torch.uint8).pin_memory()
        staged = self._pinned_in[: frames.shape[0]]
        staged.copy_(torch.from_numpy(frames))
        return staged.to(self.device, non_blocking=True)

    def _download(self, y: torch.Tensor) -> np.ndarray:
        if self.device.type != "cuda":
            return y.numpy()
        if (
            self._pinned_out is None
            or self._pinned_out.shape[1:] != y.shape[1:]
            or (self._pinned_out.shape[0] < y.shape[0])
        ):
            self._pinned_out = torch.empty(y.shape, dtype=torch.uint8).pin_memory()
        staged = self._pinned_out[: y.shape[0]]
        staged.copy_(y, non_blocking=True)
        torch.cuda.synchronize(self.device)
        # Copy out of the reused pinned buffer: the caller may queue this for the encoder.
        return staged.numpy().copy()

    # ------------------------------------------------------------------ public API

    @torch.inference_mode()
    def process_batch(
        self, frames: np.ndarray, tile_size: int | None = None, tile_batch: int = 1
    ) -> np.ndarray:
        """Upscale ``(B, H, W, 3)`` uint8 RGB frames; returns ``(B, H', W', 3)`` uint8."""
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ValueError(
                f"expected uint8 frames (B, H, W, 3), got {frames.dtype} {frames.shape}"
            )
        x = self._upload(np.ascontiguousarray(frames))
        x = x.permute(0, 3, 1, 2).to(self.dtype).div_(255.0)

        tiled = tile_size is not None
        y = self._forward_tiled(x, tile_size, tile_batch) if tiled else self._forward(x)
        del x
        # Clamp before any resize so out-of-range model output cannot bleed into neighbours.
        y = y.float().clamp_(0.0, 1.0)

        if self.out_size is not None:
            out_w, out_h = self.out_size
            if (y.shape[-1], y.shape[-2]) != (out_w, out_h):
                y = F.interpolate(
                    y, size=(out_h, out_w), mode="bicubic", align_corners=False, antialias=True
                )
        y = y.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
        return self._download(y.permute(0, 2, 3, 1).contiguous())

    def make_probe(self, frame_hw: tuple[int, int]) -> Probe:
        """Measuring function for calibration: runs the real code path on dummy frames.

        ``units`` are whole frames without tiling, or tiles per model call on one frame.
        """
        height, width = frame_hw

        def probe(units: int, tile: int | None) -> int:
            count = units if tile is None else 1
            dummy = np.full((count, height, width, 3), 127, dtype=np.uint8)
            return measure_peak(lambda: self.process_batch(dummy, tile, units), self.device)

        return probe


def create_engine(
    options: UpscaleOptions,
    descriptor: Any,
    *,
    width: int,
    height: int,
    total_frames: int = 0,
    model_path: Path | None = None,
    **tuning_overrides: Any,
) -> tuple[UpscaleEngine, RuntimeConfig]:
    """Build the engine and its calibrated ``RuntimeConfig`` for a ``width`` x ``height`` input."""
    traits = ModelTraits.from_descriptor(descriptor)
    out_size = compute_output_size(
        width, height, traits.scale, options.out_scale, options.target_height
    )
    native = out_size == (width * traits.scale, height * traits.scale)
    built: list[UpscaleEngine] = []

    def new_engine(device: torch.device, dtype: torch.dtype) -> UpscaleEngine:
        engine = UpscaleEngine(
            descriptor,
            device,
            dtype,
            tile_overlap=options.tile_overlap,
            out_size=None if native else out_size,
        )
        built.append(engine)
        return engine

    cfg = build_runtime_config(
        options,
        traits,
        width=width,
        height=height,
        total_frames=total_frames,
        make_probe=lambda device, dtype: new_engine(device, dtype).make_probe((height, width)),
        model_id=model_id(model_path) if model_path else "",
        **tuning_overrides,
    )
    engine = built[0] if built else new_engine(cfg.device, cfg.dtype)
    if cfg.compile_model:
        engine.compile(cfg.batch_size if cfg.tile_size is None else cfg.tile_batch)
    cfg.notes += engine.notes
    return engine, cfg
