"""Job planning and orchestration.

Stage 1 ships job planning (inputs -> one ``UpscaleOptions`` per video) and the dry-run
preview. The streaming render loop arrives in stage 5 of docs/PLAN.md.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from rich.table import Table

from .config import (
    OptionsError,
    UpscaleOptions,
    expand_inputs,
    output_is_directory,
    resolve_output,
)
from .console import get_console
from .models import describe_spec


def make_jobs(input_spec: str | Path, model: str, **options: Any) -> list[UpscaleOptions]:
    """Expand a file, directory or glob into one validated job per video.

    With several inputs, ``output`` (if given) is treated as a directory and each video
    gets its own checkpoint sub-directory, so resumes never mix segments.
    """
    inputs = expand_inputs(input_spec)
    checkpoint_dir = options.pop("checkpoint_dir", None)
    base = UpscaleOptions(input=inputs[0], model=model, checkpoint_dir=checkpoint_dir, **options)
    if len(inputs) == 1:
        return [base]

    if base.output is not None and not output_is_directory(base.output):
        raise OptionsError(
            f"{len(inputs)} inputs matched: --output must be a directory, not a file "
            f"({base.output})"
        )
    return [
        replace(
            base,
            input=path,
            checkpoint_dir=None if base.checkpoint_dir is None else base.checkpoint_dir / path.stem,
        )
        for path in inputs
    ]


def describe_output(options: UpscaleOptions) -> str:
    """Where the result will go, as far as it can be known before loading the model."""
    return str(resolve_output(options, "<scale>"))


def render_plan(options: UpscaleOptions) -> Table:
    def auto(value: Any, is_auto: bool) -> str:
        return "auto" if is_auto else str(value)

    table = Table(title=f"vupscale plan: {options.input.name}", show_header=False, min_width=60)
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()

    table.add_row("input", str(options.input))
    table.add_row("output", describe_output(options))
    table.add_row("model", describe_spec(options.model))
    table.add_section()
    table.add_row("device", options.device.value)
    table.add_row("precision", options.precision.value)
    table.add_row("batch size", auto(options.batch_size, options.batch_size == 0))
    tile = "off" if options.tile_size == -1 else auto(options.tile_size, options.tile_size == 0)
    table.add_row("tile size", f"{tile} (overlap {options.tile_overlap})")
    table.add_row("torch.compile", options.compile.value)
    table.add_section()
    if options.target_height > 0:
        size = f"resize to {options.target_height}p on GPU"
    elif options.out_scale > 0:
        size = f"x{options.out_scale:g} of the input (resized on GPU)"
    else:
        size = "native scale of the model"
    table.add_row("output size", size)

    frames = []
    if options.fps > 0:
        frames.append(f"fps={options.fps:g}")
    if options.skip_frames:
        frames.append(f"skip {options.skip_frames}")
    if options.every_nth > 1:
        frames.append(f"every {options.every_nth}th")
    if options.max_frames:
        frames.append(f"max {options.max_frames}")
    if options.pre_width or options.pre_height:
        frames.append(f"pre-scale {options.pre_width or 'auto'}x{options.pre_height or 'auto'}")
    table.add_row("frames", ", ".join(frames) or "all, original fps")
    table.add_section()
    table.add_row("codec", f"{options.codec.value} (quality {options.quality}, {options.pix_fmt})")
    reason = options.audio_disabled_reason
    table.add_row("audio", f"off ({reason})" if reason else "copied from the input")
    if options.checkpoint_dir is not None:
        mode = "resume" if options.resume else "fresh"
        table.add_row(
            "checkpoints",
            f"{options.checkpoint_dir} ({mode}, ~{options.segment_seconds:g}s segments)",
        )
    return table


def run(options: UpscaleOptions) -> Path:
    """Upscale one video. Returns the output path."""
    if options.dry_run:
        get_console().print(render_plan(options))
        return Path(describe_output(options))
    raise NotImplementedError(
        "The render pipeline is not implemented yet (stages 2-5 of docs/PLAN.md). "
        "Use --dry-run, `vupscale doctor` and `vupscale models` for now."
    )
