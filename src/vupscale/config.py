"""Single source of truth for every option shared by the CLI and the Python API."""

from __future__ import annotations

import glob
import math
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any

VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".ts", ".flv", ".wmv", ".mpg", ".mpeg"}
)


class Device(str, Enum):
    auto = "auto"
    cuda = "cuda"
    cpu = "cpu"


class Precision(str, Enum):
    auto = "auto"
    fp16 = "fp16"
    bf16 = "bf16"
    fp32 = "fp32"


class CompileMode(str, Enum):
    auto = "auto"
    on = "on"
    off = "off"


class Codec(str, Enum):
    auto = "auto"
    h264 = "h264"
    hevc = "hevc"
    av1 = "av1"
    h264_nvenc = "h264_nvenc"
    hevc_nvenc = "hevc_nvenc"
    av1_nvenc = "av1_nvenc"


class OptionsError(ValueError):
    """Raised when a combination of options is invalid."""


@dataclass
class UpscaleOptions:
    """All knobs of a run. Defaults here are the defaults of the CLI and of ``upscale_video``."""

    # --- input / output -------------------------------------------------
    input: Path
    model: str
    output: Path | None = None  # None -> next to the input, named after scale/height

    # --- hardware (auto unless overridden) ------------------------------
    device: Device = Device.auto
    precision: Precision = Precision.auto
    batch_size: int = 0  # 0 = auto
    tile_size: int = 0  # 0 = auto, -1 = disabled
    tile_overlap: int = 32
    compile: CompileMode = CompileMode.auto

    # --- output size ----------------------------------------------------
    out_scale: float = 0.0  # 0 = native scale of the model
    target_height: int = 0  # 0 = off

    # --- frame selection (VideoHelperSuite semantics) -------------------
    fps: float = 0.0  # 0 = keep original
    max_frames: int = 0  # 0 = all
    skip_frames: int = 0
    every_nth: int = 1
    pre_width: int = 0
    pre_height: int = 0

    # --- encoding -------------------------------------------------------
    codec: Codec = Codec.auto
    quality: int = 20  # crf on CPU encoders, cq on NVENC
    pix_fmt: str = "yuv420p"
    no_audio: bool = False

    # --- resilience -----------------------------------------------------
    checkpoint_dir: Path | None = None
    resume: bool = False
    segment_seconds: float = 180.0

    # --- misc -----------------------------------------------------------
    dry_run: bool = False

    def __post_init__(self) -> None:
        self.input = Path(self.input).expanduser()
        if self.output is not None:
            self.output = Path(self.output).expanduser()
        if self.checkpoint_dir is not None:
            self.checkpoint_dir = Path(self.checkpoint_dir).expanduser()
        self.device = Device(self.device)
        self.precision = Precision(self.precision)
        self.compile = CompileMode(self.compile)
        self.codec = Codec(self.codec)
        self.validate()

    def validate(self) -> None:
        def check(ok: bool, message: str) -> None:
            if not ok:
                raise OptionsError(message)

        check(bool(str(self.model).strip()), "model must not be empty")
        check(self.batch_size >= 0, "batch_size must be >= 0 (0 = auto)")
        check(
            self.tile_size in (0, -1) or self.tile_size >= 64,
            "tile_size must be 0 (auto), -1 (off) or >= 64",
        )
        check(self.tile_overlap >= 0, "tile_overlap must be >= 0")
        if self.tile_size > 0:
            check(
                self.tile_overlap * 2 < self.tile_size,
                "tile_overlap must be smaller than half of tile_size",
            )
        check(self.out_scale >= 0, "out_scale must be >= 0 (0 = native)")
        check(self.target_height >= 0, "target_height must be >= 0 (0 = off)")
        check(
            not (self.out_scale > 0 and self.target_height > 0),
            "use either out_scale or target_height, not both",
        )
        check(self.fps >= 0, "fps must be >= 0 (0 = original)")
        check(self.max_frames >= 0, "max_frames must be >= 0 (0 = all)")
        check(self.skip_frames >= 0, "skip_frames must be >= 0")
        check(self.every_nth >= 1, "every_nth must be >= 1")
        check(self.pre_width >= 0 and self.pre_height >= 0, "pre_width/pre_height must be >= 0")
        check(0 <= self.quality <= 51, "quality must be between 0 and 51")
        check(self.segment_seconds > 0, "segment_seconds must be > 0")
        check(
            not (self.resume and self.checkpoint_dir is None),
            "resume requires checkpoint_dir",
        )

    @property
    def audio_disabled_reason(self) -> str | None:
        """Why audio cannot be kept in sync, or ``None`` when it can be muxed."""
        if self.no_audio:
            return "disabled by --no-audio"
        if self.every_nth > 1:
            return "every_nth > 1 changes playback speed"
        # --fps resamples (drops/duplicates frames) without changing duration,
        # and --skip-frames is compensated with an audio offset, so both stay in sync.
        return None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Enum):
                value = value.value
            elif isinstance(value, Path):
                value = str(value)
            out[f.name] = value
        return out


def output_is_directory(output: Path) -> bool:
    """An existing folder, or a path without extension, means "put the file in here"."""
    return output.is_dir() or output.suffix == ""


def output_name(input_path: Path, scale: float | str, target_height: int = 0) -> str:
    """``clip.mp4`` -> ``clip_x4.mp4`` (or ``clip_2160p.mp4`` with a target height)."""
    if target_height > 0:
        suffix = f"_{target_height}p"
    elif isinstance(scale, str):
        suffix = f"_x{scale}"
    else:
        suffix = f"_x{scale:g}"
    return f"{input_path.stem}{suffix}.mp4"


def resolve_output(options: UpscaleOptions, scale: float | str) -> Path:
    """Final output path. ``scale`` may be a placeholder string before the model is loaded."""
    name = output_name(options.input, scale, options.target_height)
    if options.output is None:
        return options.input.with_name(name)
    if output_is_directory(options.output):
        return options.output / name
    return options.output


def expand_inputs(spec: str | Path) -> list[Path]:
    """Resolve a file, a directory or a glob pattern into a sorted list of video files."""
    path = Path(spec).expanduser()
    if path.is_file():
        return [path]
    if path.is_dir():
        found = sorted(
            p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not found:
            raise OptionsError(f"no video files found in directory: {path}")
        return found
    text = str(path)
    if glob.has_magic(text):
        found = sorted(
            Path(p)
            for p in glob.glob(text, recursive=True)
            if Path(p).is_file() and Path(p).suffix.lower() in VIDEO_EXTENSIONS
        )
        if not found:
            raise OptionsError(f"no video files match pattern: {spec}")
        return found
    raise OptionsError(f"input not found: {spec}")


def _even(value: float) -> int:
    """Nearest even integer >= 2 (yuv420p needs even dimensions)."""
    return max(2, math.floor(value / 2 + 0.5) * 2)


def compute_input_size(
    width: int, height: int, pre_width: int = 0, pre_height: int = 0
) -> tuple[int, int]:
    """Size fed to the model after the optional pre-scale. A zero axis keeps the aspect ratio."""
    if pre_width and pre_height:
        return _even(pre_width), _even(pre_height)
    if pre_width:
        return _even(pre_width), _even(height * pre_width / width)
    if pre_height:
        return _even(width * pre_height / height), _even(pre_height)
    return width, height


def compute_output_size(
    width: int, height: int, scale: int, out_scale: float = 0.0, target_height: int = 0
) -> tuple[int, int]:
    """Final (width, height) of the video given the size fed to the model."""
    if target_height > 0:
        out_h = _even(target_height)
        return _even(width * out_h / height), out_h
    factor = out_scale if out_scale > 0 else scale
    return _even(width * factor), _even(height * factor)
