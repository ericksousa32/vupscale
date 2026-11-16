"""Command line interface. ``vupscale video.mp4 -m realesrgan-x4`` is all that is required."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import typer
from rich.table import Table
from typer.core import TyperGroup

from . import __version__
from .config import Codec, CompileMode, Device, OptionsError, Precision
from .console import get_console
from .models import MODEL_ALIASES, ModelError, cache_dir, cached_path


class DefaultCommandGroup(TyperGroup):
    """Route ``vupscale <video> ...`` to the ``upscale`` command.

    Deliberately avoids importing ``click``: recent Typer releases vendor it instead of
    depending on it, so it may not be installed.
    """

    default_command = "upscale"
    _group_flags = frozenset(
        {"--help", "-h", "--version", "-V", "--install-completion", "--show-completion"}
    )

    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and args[0] not in self._group_flags:
            args = [self.default_command, *args]
        return super().parse_args(ctx, args)


app = typer.Typer(
    cls=DefaultCommandGroup,
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Upscale videos with PyTorch super-resolution models.\n\n"
        "Minimal usage:  vupscale video.mp4 -m realesrgan-x4\n\n"
        "Everything else is auto-tuned for the detected GPU and can be overridden."
    ),
)

HW = "Hardware (auto-tuned)"
SIZE = "Output size"
FRAMES = "Frame selection"
ENC = "Encoding"
CKPT = "Checkpointing"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"vupscale {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True, help="Show version."
    ),
) -> None:
    pass


@app.command()
def upscale(
    input_arg: Optional[str] = typer.Argument(  # noqa: UP045 - typer needs Optional on 3.10
        None, metavar="INPUT", help="Video file, directory or glob pattern.", show_default=False
    ),
    input_opt: Optional[str] = typer.Option(  # noqa: UP045
        None, "--input", "-i", help="Same as the INPUT argument.", show_default=False
    ),
    # Required, but validated by hand: `Option(...)` is not enforced by every
    # Typer/click combination found on Colab and RunPod images.
    model: Optional[str] = typer.Option(  # noqa: UP045
        None,
        "--model",
        "-m",
        help="[required] Alias (see `vupscale models`), local .pth/.safetensors or URL.",
        show_default=False,
    ),
    output: Optional[Path] = typer.Option(  # noqa: UP045
        None,
        "--output",
        "-o",
        help="Output file (or directory). Default: <input>_x<scale>.mp4 next to the input.",
        show_default=False,
    ),
    # hardware
    device: Device = typer.Option(Device.auto, help="Inference device.", rich_help_panel=HW),
    precision: Precision = typer.Option(
        Precision.auto, help="auto picks what the model supports.", rich_help_panel=HW
    ),
    batch_size: int = typer.Option(
        0, min=0, help="Frames per batch. 0 = auto.", rich_help_panel=HW
    ),
    tile_size: int = typer.Option(
        0, min=-1, help="Tile size in px. 0 = auto, -1 = off.", rich_help_panel=HW
    ),
    tile_overlap: int = typer.Option(32, min=0, help="Tile overlap in px.", rich_help_panel=HW),
    compile_: CompileMode = typer.Option(
        CompileMode.auto, "--compile", help="torch.compile the model.", rich_help_panel=HW
    ),
    # output size
    out_scale: float = typer.Option(
        0.0, min=0.0, help="Final scale vs. input. 0 = native model scale.", rich_help_panel=SIZE
    ),
    target_height: int = typer.Option(
        0, min=0, help="Final height in px (e.g. 2160). 0 = off.", rich_help_panel=SIZE
    ),
    # frames
    fps: float = typer.Option(
        0.0, min=0.0, help="Force frame rate. 0 = original.", rich_help_panel=FRAMES
    ),
    max_frames: int = typer.Option(0, min=0, help="Frame cap. 0 = all.", rich_help_panel=FRAMES),
    skip_frames: int = typer.Option(0, min=0, help="Skip first N frames.", rich_help_panel=FRAMES),
    every_nth: int = typer.Option(1, min=1, help="Keep every Nth frame.", rich_help_panel=FRAMES),
    pre_width: int = typer.Option(
        0, min=0, help="Resize width before upscaling. 0 = keep.", rich_help_panel=FRAMES
    ),
    pre_height: int = typer.Option(
        0, min=0, help="Resize height before upscaling. 0 = keep.", rich_help_panel=FRAMES
    ),
    # encoding
    codec: Codec = typer.Option(
        Codec.auto, help="auto = NVENC when it really works, else CPU.", rich_help_panel=ENC
    ),
    quality: int = typer.Option(
        20,
        "--quality",
        "--crf",
        min=0,
        max=51,
        help="Lower is better (crf / NVENC cq).",
        rich_help_panel=ENC,
    ),
    pix_fmt: str = typer.Option("yuv420p", help="Output pixel format.", rich_help_panel=ENC),
    no_audio: bool = typer.Option(
        False, "--no-audio", help="Drop the audio track.", rich_help_panel=ENC
    ),
    # checkpointing
    checkpoint_dir: Optional[Path] = typer.Option(  # noqa: UP045
        None,
        help="Write resumable segments here (e.g. Google Drive).",
        rich_help_panel=CKPT,
        show_default=False,
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Continue from valid segments in --checkpoint-dir.",
        rich_help_panel=CKPT,
    ),
    segment_seconds: float = typer.Option(
        180.0, min=1.0, help="Processing time per checkpoint segment.", rich_help_panel=CKPT
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the resolved plan and exit without processing."
    ),
) -> None:
    """Upscale a video, a directory of videos or a glob pattern."""
    from . import pipeline

    console = get_console()
    if input_arg and input_opt:
        raise typer.BadParameter("give the input once: either positional or with --input")
    input_spec = input_arg or input_opt
    if not input_spec:
        raise typer.BadParameter("missing INPUT (video file, directory or glob)")
    if not model:
        raise typer.BadParameter("missing --model / -m (run `vupscale models` for aliases)")

    try:
        jobs = pipeline.make_jobs(
            input_spec,
            model,
            output=output,
            device=device,
            precision=precision,
            batch_size=batch_size,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            compile=compile_,
            out_scale=out_scale,
            target_height=target_height,
            fps=fps,
            max_frames=max_frames,
            skip_frames=skip_frames,
            every_nth=every_nth,
            pre_width=pre_width,
            pre_height=pre_height,
            codec=codec,
            quality=quality,
            pix_fmt=pix_fmt,
            no_audio=no_audio,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            segment_seconds=segment_seconds,
            dry_run=dry_run,
        )
        for job in jobs:
            result = pipeline.run(job)
            if not dry_run:
                console.print(f"[green]done:[/green] {result}")
    except (OptionsError, ModelError) as exc:
        console.print(f"[bold red]error:[/bold red] {exc}")
        raise typer.Exit(2) from exc
    except NotImplementedError as exc:
        console.print(f"[bold yellow]not available yet:[/bold yellow] {exc}")
        raise typer.Exit(1) from exc


@app.command()
def doctor() -> None:
    """Show the detected environment, GPU, FFmpeg and which encoders really work."""
    from .hardware import CPU_ENCODERS, NVENC_ENCODERS, collect_report

    console = get_console()
    with console.status("probing hardware and encoders..."):
        report = collect_report()

    def mark(ok: bool) -> str:
        return "[green]yes[/green]" if ok else "[dim]no[/dim]"

    table = Table(title=f"vupscale {__version__} doctor", show_header=False, min_width=60)
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    table.add_row("environment", report.environment)
    table.add_row("python", report.python)
    torch_text = report.torch or "[red]not installed[/red]"
    if report.torch and report.cuda_build:
        torch_text += f" (CUDA {report.cuda_build})"
    table.add_row("torch", torch_text)
    if report.gpu:
        gpu = report.gpu
        major, minor = gpu.compute_cap
        table.add_row("gpu", f"{gpu.name} ({gpu.generation}, sm_{major}{minor})")
        table.add_row("vram", f"{gpu.vram_free_gb:.1f} GB free of {gpu.vram_total_gb:.1f} GB")
    else:
        table.add_row("gpu", "[yellow]none (CPU only)[/yellow]")
    table.add_row("cpu", f"{report.cpu_count} cores")
    table.add_row("ram", f"{report.ram_free_gb:.1f} GB free of {report.ram_total_gb:.1f} GB")
    table.add_section()
    table.add_row("ffmpeg", report.ffmpeg or "[red]not found[/red]")
    table.add_row("ffprobe", mark(report.ffprobe))
    if report.encoders:
        table.add_row(
            "nvenc (tested)",
            "  ".join(
                f"{e.removesuffix('_nvenc')}: {mark(report.encoders[e])}" for e in NVENC_ENCODERS
            ),
        )
        table.add_row(
            "cpu encoders",
            "  ".join(f"{e}: {mark(report.encoders[e])}" for e in CPU_ENCODERS),
        )
        from .config import Codec
        from .tuning import select_encoder

        hd = select_encoder(Codec.auto, 3840, 2160, report.cpu_count).encoder
        big = select_encoder(Codec.auto, 7680, 4320, report.cpu_count).encoder
        table.add_row("codec auto", f"{hd} (up to 4096 px), {big} (larger)")
    console.print(table)

    for note in report.warnings:
        console.print(f"[yellow]warning:[/yellow] {note}")
    for issue in report.problems:
        console.print(f"[bold red]problem:[/bold red] {issue}")
    if report.problems:
        raise typer.Exit(1)
    console.print("[green]ready.[/green]")


@app.command()
def bench(
    model: str = typer.Option(..., "--model", "-m", help="Alias, local file or URL."),
    size: str = typer.Option("1280x720", help="Input resolution to simulate, WIDTHxHEIGHT."),
    seconds: float = typer.Option(5.0, min=0.0, help="How long to measure."),
    device: Device = typer.Option(Device.auto, help="Inference device."),
    precision: Precision = typer.Option(Precision.auto, help="Override the precision."),
    batch_size: int = typer.Option(0, min=0, help="Frames per batch. 0 = auto."),
    tile_size: int = typer.Option(0, min=-1, help="Tile size. 0 = auto, -1 = off."),
    target_height: int = typer.Option(0, min=0, help="Final height in px. 0 = native."),
    compile_: CompileMode = typer.Option(CompileMode.off, "--compile", help="torch.compile."),
) -> None:
    """Calibrate and measure inference speed for a model on synthetic frames (no video needed)."""
    import time

    import numpy as np

    from .config import UpscaleOptions
    from .engine import create_engine, load_model
    from .models import resolve_model
    from .tuning import AdaptiveBatchController, TuningError

    console = get_console()
    try:
        width, height = (int(v) for v in size.lower().split("x"))
    except ValueError as exc:
        raise typer.BadParameter("size must look like 1920x1080") from exc

    try:
        path = resolve_model(model)
        descriptor = load_model(path)
        options = UpscaleOptions(
            input=Path("bench"), model=model, device=device, precision=precision,
            batch_size=batch_size, tile_size=tile_size, target_height=target_height,
            compile=compile_,
        )  # fmt: skip
        with console.status("calibrating memory..."):
            engine, cfg = create_engine(
                options, descriptor, width=width, height=height, model_path=path
            )
        controller = AdaptiveBatchController.from_config(cfg, (height, width))
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 256, (cfg.batch_size, height, width, 3), dtype=np.uint8)
        with console.status("measuring..."):
            controller.run(engine.process_batch, frames)  # warm-up (cuDNN autotune, compile)
            engine.timings = {}
            done, start = 0, time.perf_counter()
            while done == 0 or time.perf_counter() - start < seconds:
                controller.run(engine.process_batch, frames)
                done += len(frames)
            elapsed = time.perf_counter() - start
    except (OptionsError, ModelError, TuningError) as exc:
        console.print(f"[bold red]error:[/bold red] {exc}")
        raise typer.Exit(2) from exc

    table = Table(title=f"vupscale bench: {path.name}", show_header=False, min_width=60)
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    table.add_row("model", f"{descriptor.architecture.name} x{descriptor.scale}")
    table.add_row("device", f"{cfg.gpu_name} ({cfg.precision_name})")
    table.add_row("resolution", f"{width}x{height} -> {cfg.out_width}x{cfg.out_height}")
    table.add_row("batch size", f"{controller.batch_size}")
    tile = "off (full frames)"
    if controller.tile_size is not None:
        tile = f"{controller.tile_size} px, {controller.tile_batch} tiles per call"
    table.add_row("tiling", tile)
    table.add_row("calibrated", "yes" if cfg.calibrated else "no (CPU or estimate)")
    table.add_row("codec (auto)", f"{cfg.video_codec} ({cfg.encoder_preset})")
    if cfg.device.type == "cuda":
        import torch

        peak = torch.cuda.max_memory_allocated(cfg.device) / 2**30
        table.add_row("vram peak", f"{peak:.1f} GB of {cfg.vram_total_gb:.1f} GB")
    table.add_section()
    table.add_row("speed", f"[bold green]{done / elapsed:.2f} fps[/bold green] (inference only)")
    labels = {
        "upload": "upload (RAM -> GPU)",
        "model": "model",
        "convert": "convert to uint8 (GPU)",
        "download": "download (GPU -> RAM)",
        "host copy": "host copy (RAM)",
    }
    timings = engine.timings or {}
    measured = sum(timings.values()) or 1.0
    table.add_section()
    for stage, label in labels.items():
        if stage in timings:
            spent = timings[stage]
            table.add_row(
                label, f"{1000 * spent / done:8.1f} ms/frame  {100 * spent / measured:5.1f}%"
            )
    console.print(table)
    for note in (*cfg.notes, *engine.notes, *controller.events):
        console.print(f"[yellow]note:[/yellow] {note}")


@app.command()
def models() -> None:
    """List the built-in model aliases and what is already cached."""
    console = get_console()
    table = Table(title="vupscale models", min_width=60)
    table.add_column("alias", style="bold cyan", no_wrap=True)
    table.add_column("scale", justify="right")
    table.add_column("cached", justify="center")
    table.add_column("description")
    for info in MODEL_ALIASES.values():
        cached = cached_path(info.url).is_file()
        table.add_row(
            info.alias,
            f"x{info.scale}",
            "[green]yes[/green]" if cached else "[dim]no[/dim]",
            info.description,
        )
    console.print(table)
    console.print(f"cache: {cache_dir()}")
    console.print("Any local .pth/.safetensors file or HTTP(S) URL also works with --model.")


if __name__ == "__main__":
    app()
