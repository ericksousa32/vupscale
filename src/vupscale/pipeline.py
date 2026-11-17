"""The render pipeline: decode -> upscale -> encode, three stages running concurrently.

FFmpeg decodes on one thread and encodes on another; the main thread only feeds the GPU.
Bounded queues between the stages keep host RAM in check and reveal which stage is the
bottleneck. With a checkpoint directory the output is produced in resumable segments.
"""

from __future__ import annotations

import contextlib
import hashlib
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from . import __version__
from .checkpoint import CheckpointStore, atomic_copy, fingerprint
from .config import (
    OptionsError,
    UpscaleOptions,
    expand_inputs,
    output_is_directory,
    resolve_output,
)
from .console import get_console, progress_refresh_rate
from .engine import UpscaleEngine, create_engine, load_model
from .models import resolve_model
from .tuning import AdaptiveBatchController, ModelTraits, RuntimeConfig, build_runtime_config
from .video_io import (
    AudioSource,
    DecodePlan,
    VideoError,
    VideoMetadata,
    VideoStreamReader,
    VideoStreamWriter,
    concat_segments,
    part_path,
    plan_decode,
    probe_video,
)

# Mounts where writing a stream directly is slow or unreliable (FUSE): encode locally first.
_SLOW_MOUNTS = ("/content/drive",)
_models: dict[Path, Any] = {}


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


def _descriptor(path: Path) -> Any:
    """Load each model once per process (batch jobs reuse it)."""
    if path not in _models:
        _models[path] = load_model(path)
    return _models[path]


def _is_slow_mount(path: Path) -> bool:
    text = str(path.absolute())
    return any(text == mount or text.startswith(mount + "/") for mount in _SLOW_MOUNTS)


def _work_dir(key: Path) -> Path:
    """Fresh local scratch directory for this job (leftovers of a killed run are dropped)."""
    tag = hashlib.sha1(str(key.absolute()).encode()).hexdigest()[:12]
    path = Path(tempfile.gettempdir()) / "vupscale_work" / tag
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- reporting


def _summary(
    options: UpscaleOptions,
    meta: VideoMetadata,
    plan: DecodePlan,
    cfg: RuntimeConfig,
    descriptor: Any,
    output: Path,
    audio: AudioSource | None,
    start_frame: int = 0,
) -> Table:
    table = Table(
        title=f"vupscale {__version__}: {meta.path.name}", show_header=False, min_width=60
    )
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    table.add_row(
        "input",
        f"{meta.width}x{meta.height}, {meta.fps_float:.3f} fps, {meta.total_frames} frames, "
        f"{meta.duration:.1f} s",
    )
    table.add_row("output", str(output))
    table.add_row("model", f"{descriptor.architecture.name} x{descriptor.scale}")
    table.add_section()
    device = cfg.gpu_name
    if cfg.device.type == "cuda":
        device += f" (sm_{cfg.compute_cap[0]}{cfg.compute_cap[1]}, {cfg.environment})"
    table.add_row("device", device)
    table.add_row("precision", cfg.precision_name)
    table.add_row("batch size", str(cfg.batch_size))
    tiling = "off (full frames)"
    if cfg.tile_size is not None:
        tiling = f"up to {cfg.tile_size} px, {cfg.tile_batch} tiles per call"
    table.add_row("tiling", tiling)
    table.add_row("torch.compile", "on" if cfg.compile_model else "off")
    table.add_section()
    table.add_row("resolution", f"{plan.width}x{plan.height} -> {cfg.out_width}x{cfg.out_height}")
    frames = f"{plan.total_frames} at {float(plan.fps):.3f} fps"
    if start_frame:
        frames += f"  (resuming at frame {start_frame})"
    table.add_row("frames", frames)
    table.add_row(
        "codec",
        f"{cfg.video_codec} ({cfg.encoder_preset}, quality {options.quality}, {options.pix_fmt})",
    )
    if audio is not None:
        table.add_row("audio", f"{meta.audio_codec} from the input")
    else:
        reason = options.audio_disabled_reason or "the input has no audio track"
        table.add_row("audio", f"off ({reason})")
    if options.checkpoint_dir is not None:
        table.add_row(
            "checkpoints", f"{options.checkpoint_dir} (~{options.segment_seconds:g} s segments)"
        )
    return table


class _VramColumn(TextColumn):
    def __init__(self, cfg: RuntimeConfig) -> None:
        super().__init__("")
        self._cfg = cfg

    def render(self, task: Any) -> Text:
        if self._cfg.device.type != "cuda":
            return Text("")
        import torch

        used = torch.cuda.memory_reserved(self._cfg.device) / 2**30
        return Text(f"{used:.1f}/{self._cfg.vram_total_gb:.0f} GB", style="magenta")


@dataclass
class _Monitor:
    """Tracks speed and which stage limits throughput, from the queue fill levels."""

    depth: int
    started: float = field(default_factory=time.perf_counter)
    frames: int = 0
    waits: dict[str, int] = field(default_factory=lambda: {"gpu": 0, "decode": 0, "encode": 0})

    def sample(self, decoded_waiting: int, encode_waiting: int, frames: int) -> None:
        self.frames += frames
        if encode_waiting >= self.depth:
            self.waits["encode"] += 1  # the encoder cannot keep up
        elif decoded_waiting == 0:
            self.waits["decode"] += 1  # the GPU is waiting for frames
        else:
            self.waits["gpu"] += 1

    @property
    def fps(self) -> float:
        elapsed = time.perf_counter() - self.started
        return self.frames / elapsed if elapsed > 0 else 0.0

    @property
    def bottleneck(self) -> str:
        return max(self.waits, key=lambda stage: self.waits[stage]) if self.frames else "-"


# --------------------------------------------------------------------------- the render


def _job_identity(
    options: UpscaleOptions, plan: DecodePlan, cfg: RuntimeConfig, model_path: Path
) -> dict[str, Any]:
    """Everything that shapes the pixels and the container of the output."""
    return {
        "input": fingerprint(options.input),
        "model": fingerprint(model_path),
        "size": [cfg.out_width, cfg.out_height],
        "model_input": [plan.width, plan.height],
        "fps": str(plan.fps),
        "frames": plan.total_frames,
        "selection": [options.fps, options.skip_frames, options.every_nth, options.max_frames],
        "encoder": cfg.video_codec,
        "quality": options.quality,
        "pix_fmt": options.pix_fmt,
    }


def _render_frames(
    reader: VideoStreamReader,
    engine: UpscaleEngine,
    controller: AdaptiveBatchController,
    cfg: RuntimeConfig,
    plan: DecodePlan,
    start_frame: int,
    new_writer: Callable[[int], VideoStreamWriter],
    segment_done: Callable[[VideoStreamWriter, int, int], None],
    segment_seconds: float | None,
) -> tuple[int, _Monitor]:
    """Pump frames through the three stages. Returns (frames rendered, monitor).

    ``new_writer(index)`` builds the writer for the file starting at frame ``index``;
    ``segment_done(writer, start, end)`` is called once that file is closed cleanly.
    Without ``segment_seconds`` everything goes into a single file.
    """
    monitor = _Monitor(depth=cfg.queue_depth)
    position = start_frame
    segment_start = start_frame
    writer: VideoStreamWriter | None = None
    opened = time.monotonic()
    interrupted = False

    progress = Progress(
        TextColumn("[bold]upscaling"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[green]{task.fields[fps]:.2f} fps"),
        _VramColumn(cfg),
        TimeRemainingColumn(),
        TextColumn("[dim]limit: {task.fields[limit]}"),
        console=get_console(),
        refresh_per_second=progress_refresh_rate(),
    )
    try:
        with progress:
            task = progress.add_task(
                "", total=plan.total_frames, completed=start_frame, fps=0.0, limit="-"
            )
            try:
                for batch in reader:
                    if writer is None:
                        writer = new_writer(position).__enter__()
                        segment_start, opened = position, time.monotonic()
                    decoded_waiting = reader.queued
                    writer.write(controller.run(engine.process_batch, batch))
                    position += len(batch)
                    monitor.sample(decoded_waiting, writer.queued, len(batch))
                    progress.update(
                        task, completed=position, fps=monitor.fps, limit=monitor.bottleneck
                    )
                    due = (
                        segment_seconds is not None and time.monotonic() - opened >= segment_seconds
                    )
                    if due and position < plan.total_frames:
                        writer.close()
                        segment_done(writer, segment_start, position - 1)
                        writer = None
            except KeyboardInterrupt:
                interrupted = True  # keep what is rendered: finish the current file cleanly
        if writer is not None:
            writer.close()
            segment_done(writer, segment_start, position - 1)
            writer = None
    finally:
        if writer is not None:
            writer.abort()
    if interrupted:
        raise KeyboardInterrupt
    return position - start_frame, monitor


def run(options: UpscaleOptions) -> Path:
    """Upscale one video. Returns the output path."""
    console = get_console()
    meta = probe_video(options.input)
    model_path = resolve_model(options.model)
    descriptor = _descriptor(model_path)
    plan = plan_decode(meta, options)
    if plan.total_frames <= 0:
        raise VideoError(f"{meta.path.name}: the frame options select no frames")
    output = resolve_output(options, options.out_scale or descriptor.scale)
    if output.resolve() == meta.path.resolve():
        raise OptionsError("the output would overwrite the input; choose another --output")
    suffix = output.suffix or ".mp4"
    audio = None
    if meta.has_audio and not options.no_audio:
        audio = AudioSource(meta.path, meta.audio_codec, plan.audio_offset, plan.audio_duration)

    if options.dry_run:
        cfg = build_runtime_config(
            options,
            ModelTraits.from_descriptor(descriptor),
            width=plan.width,
            height=plan.height,
            total_frames=plan.total_frames,
        )
        console.print(_summary(options, meta, plan, cfg, descriptor, output, audio))
        for note in cfg.notes:
            console.print(f"[yellow]note:[/yellow] {note}")
        console.print("[dim]dry run: nothing was rendered[/dim]")
        return output

    with console.status("loading the model and calibrating memory..."):
        engine, cfg = create_engine(
            options,
            descriptor,
            width=plan.width,
            height=plan.height,
            total_frames=plan.total_frames,
            model_path=model_path,
        )
    if meta.is_hdr:
        cfg.notes.append("HDR input: colours are converted as SDR and will look washed out")
    if meta.is_vfr:
        cfg.notes.append("variable frame rate input: the output uses the average frame rate")

    store: CheckpointStore | None = None
    start_frame = 0
    if options.checkpoint_dir is not None:
        store = CheckpointStore(options.checkpoint_dir, suffix)
        try:
            job = _job_identity(options, plan, cfg, model_path)
            if options.resume and (finished := store.completed_output(job)) is not None:
                console.print(f"[green]already rendered:[/green] {finished}")
                store.close()
                return finished
            start_frame = store.open(job, resume=options.resume)
        except BaseException:
            store.close()
            raise

    console.print(_summary(options, meta, plan, cfg, descriptor, output, audio, start_frame))
    for note in cfg.notes:
        console.print(f"[yellow]note:[/yellow] {note}")

    work = _work_dir(options.checkpoint_dir or output)
    engine.set_output_ring(cfg.queue_depth + 2)  # results reach the encoder without a copy
    controller = AdaptiveBatchController.from_config(
        cfg,
        (plan.height, plan.width),
        on_event=lambda message: console.print(f"[yellow]{message}[/yellow]"),
    )

    def writer_for(path: Path, **extra: Any) -> VideoStreamWriter:
        return VideoStreamWriter(
            path,
            cfg.out_width,
            cfg.out_height,
            plan.fps,
            encoder=cfg.video_codec,
            preset=cfg.encoder_preset,
            quality=options.quality,
            pix_fmt=options.pix_fmt,
            queue_depth=cfg.queue_depth,
            **extra,
        )

    def reader_from(frame: int) -> VideoStreamReader:
        return VideoStreamReader(
            meta, plan_decode(meta, options, frame), cfg.batch_size, queue_depth=cfg.queue_depth
        )

    started = time.perf_counter()
    rendered, monitor = 0, None
    try:
        if store is not None:
            checkpoints = store
            try:
                if start_frame < plan.total_frames:
                    with reader_from(start_frame) as reader:
                        rendered, monitor = _render_frames(
                            reader, engine, controller, cfg, plan, start_frame,
                            lambda index: writer_for(
                                work / f"seg_{index:06d}{suffix}", faststart=False
                            ),
                            lambda writer, start, end: checkpoints.publish(
                                writer.output, start, end
                            ),
                            options.segment_seconds,
                        )  # fmt: skip
            finally:
                store.wait()  # whatever was finished is safely stored, even after Ctrl+C
            with console.status("joining segments..."):
                concat_segments(store.segments(), output, audio)
            store.finish(output)
        else:
            direct = not _is_slow_mount(output)
            target = part_path(output) if direct else work / f"render{suffix}"
            try:
                with reader_from(0) as reader:
                    rendered, monitor = _render_frames(
                        reader, engine, controller, cfg, plan, 0,
                        lambda index: writer_for(target, audio=audio),
                        lambda writer, start, end: None,
                        None,
                    )  # fmt: skip
                if rendered == 0:
                    raise VideoError(f"{meta.path.name}: no frames could be decoded")
                if direct:
                    target.replace(output)
                else:
                    with console.status(f"copying to {output.parent}..."):
                        atomic_copy(target, output)
            finally:
                with contextlib.suppress(OSError):
                    target.unlink()
    except KeyboardInterrupt:
        if store is not None:
            console.print(
                "[yellow]interrupted:[/yellow] finished segments are saved; "
                "run again with --resume to continue"
            )
        else:
            console.print("[yellow]interrupted:[/yellow] partial output discarded")
        raise
    finally:
        if store is not None:
            store.close()
        shutil.rmtree(work, ignore_errors=True)

    elapsed = time.perf_counter() - started
    line = f"[green]done:[/green] {output}  ({elapsed:.1f} s"
    if monitor is not None and rendered:
        line += f", {rendered / elapsed:.2f} fps, limited by {monitor.bottleneck}"
    console.print(line + ")")
    if controller.events:
        console.print(f"[yellow]{len(controller.events)} out-of-memory adjustment(s) made[/yellow]")
    return output
