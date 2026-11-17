"""Zero-disk video I/O: FFmpeg decodes to and encodes from raw RGB pipes.

Decode and encode each run in their own thread behind a bounded queue, so the GPU never
waits for FFmpeg (and FFmpeg never waits for the GPU) beyond the queue depth.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import queue
import shutil
import subprocess
import tempfile
import threading
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from .config import UpscaleOptions, compute_input_size
from .hardware import ffmpeg_version

_QUIET = ["-hide_banner", "-loglevel", "error", "-nostdin"]
_SWS_FLAGS = "accurate_rnd+full_chroma_int"
_PIPE_BYTES = 1 << 20
# Audio codecs that can be stream-copied into each container; anything else is re-encoded.
_COPYABLE_AUDIO = {
    ".mp4": {"aac", "mp3", "ac3", "eac3", "alac"},
    ".m4v": {"aac", "mp3", "ac3", "eac3", "alac"},
    ".mov": {"aac", "mp3", "ac3", "eac3", "alac", "pcm_s16le", "pcm_s24le"},
    ".webm": {"opus", "vorbis"},
}


class VideoError(RuntimeError):
    """FFmpeg/ffprobe failed, or a video could not be read or written."""


def _require(binary: str) -> str:
    if shutil.which(binary) is None:
        raise VideoError(f"{binary} not found: install FFmpeg (bash scripts/setup_cloud.sh)")
    return binary


def _grow_pipe(stream: Any) -> None:
    """Ask Linux for a 1 MB pipe (default 64 KB) to cut syscalls on large frames."""
    try:
        import fcntl

        fcntl.fcntl(stream.fileno(), 1031, _PIPE_BYTES)  # F_SETPIPE_SZ
    except (ImportError, OSError, ValueError, AttributeError):
        pass


def _passthrough_args() -> list[str]:
    """Stop FFmpeg from duplicating/dropping frames to fake a constant frame rate."""
    version = ffmpeg_version() or ""
    digits = "".join(ch if ch.isdigit() else " " for ch in version.lstrip("nN-")).split()
    major = int(digits[0]) if digits else 99
    return ["-vsync", "passthrough"] if major < 5 else ["-fps_mode", "passthrough"]


# --------------------------------------------------------------------------- probing


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    width: int  # display size, after any rotation flag
    height: int
    fps: Fraction
    is_vfr: bool
    total_frames: int
    duration: float
    codec: str
    pix_fmt: str
    color_space: str  # as reported by ffprobe, "" when untagged
    color_transfer: str
    has_audio: bool
    audio_codec: str

    @property
    def fps_float(self) -> float:
        return float(self.fps)

    @property
    def is_hdr(self) -> bool:
        return self.color_transfer in ("smpte2084", "arib-std-b67")


def _fraction(text: str | None) -> Fraction | None:
    try:
        value = Fraction(text or "")
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def _ffprobe_json(path: Path, *args: str) -> dict[str, Any]:
    cmd = [_require("ffprobe"), "-v", "error", "-of", "json", *args, str(path)]
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise VideoError(f"cannot read {path.name}: {done.stderr.strip() or 'ffprobe failed'}")
    try:
        return json.loads(done.stdout or "{}")
    except ValueError as exc:
        raise VideoError(f"cannot read {path.name}: unexpected ffprobe output") from exc


def count_frames(path: Path) -> int:
    """Exact number of video packets (frames). Demuxes the file without decoding it."""
    data = _ffprobe_json(
        path, "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets"
    )
    try:
        return int(data["streams"][0]["nb_read_packets"])
    except (KeyError, IndexError, ValueError, TypeError):
        return 0


def probe_video(path: str | Path) -> VideoMetadata:
    path = Path(path)
    if not path.is_file():
        raise VideoError(f"video not found: {path}")
    data = _ffprobe_json(path, "-show_format", "-show_streams")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None or not video.get("width"):
        raise VideoError(f"{path.name} has no video stream")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    real = _fraction(video.get("r_frame_rate"))
    average = _fraction(video.get("avg_frame_rate"))
    fps = average or real
    if fps is None:
        raise VideoError(f"{path.name}: could not determine the frame rate")
    is_vfr = bool(real and average and abs(real - average) / real > 0.01)

    duration = 0.0
    for source in (video.get("duration"), data.get("format", {}).get("duration")):
        with contextlib.suppress(TypeError, ValueError):
            duration = float(source)
            break
    try:
        total = int(video.get("nb_frames"))
    except (TypeError, ValueError):
        total = count_frames(path) or round(duration * fps)
    if duration <= 0 and total:
        duration = float(total / fps)

    width, height = int(video["width"]), int(video["height"])
    rotation = 0.0
    for side in video.get("side_data_list", []):
        with contextlib.suppress(TypeError, ValueError):
            rotation = float(side.get("rotation", rotation))
    with contextlib.suppress(TypeError, ValueError):
        rotation = float(video.get("tags", {}).get("rotate", rotation))
    if round(abs(rotation)) % 180 == 90:  # FFmpeg auto-rotates while decoding
        width, height = height, width

    return VideoMetadata(
        path=path,
        width=width,
        height=height,
        fps=fps,
        is_vfr=is_vfr,
        total_frames=total,
        duration=duration,
        codec=video.get("codec_name", ""),
        pix_fmt=video.get("pix_fmt", ""),
        color_space=video.get("color_space", "") or "",
        color_transfer=video.get("color_transfer", "") or "",
        has_audio=audio is not None,
        audio_codec=(audio or {}).get("codec_name", ""),
    )


def hwaccel_works(path: Path) -> bool:
    """True when NVDEC can decode this file (frees the CPU, which is scarce on Colab)."""
    cmd = [
        "ffmpeg", *_QUIET, "-hwaccel", "cuda", "-i", str(path),
        "-frames:v", "1", "-f", "null", "-",
    ]  # fmt: skip
    try:
        return subprocess.run(cmd, capture_output=True, timeout=30, check=False).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# --------------------------------------------------------------------------- decode plan


@dataclass(frozen=True)
class DecodePlan:
    """Everything needed to decode exactly the frames the options select."""

    width: int  # size of the frames delivered (model input)
    height: int
    fps: Fraction  # frame rate of the delivered frames
    frames: int  # frames this run will deliver
    total_frames: int  # frames of the whole job (before ``start_frame``)
    seek_seconds: float | None  # fast input seek, or None
    filters: str
    audio_offset: float  # where the kept audio starts in the source, seconds
    audio_duration: float  # length of the whole job, seconds


def _input_matrix(meta: VideoMetadata) -> str:
    # Only names every FFmpeg release accepts for scale's in_color_matrix.
    known = {
        "bt709": "bt709",
        "smpte170m": "bt601",
        "bt470bg": "bt601",
        "bt2020nc": "bt2020",
        "bt2020c": "bt2020",
    }
    if meta.color_space in known:
        return known[meta.color_space]
    # Untagged: follow the usual convention, HD is BT.709 and SD is BT.601.
    return "bt709" if meta.height >= 720 or meta.width >= 1280 else "bt601"


def plan_decode(meta: VideoMetadata, options: UpscaleOptions, start_frame: int = 0) -> DecodePlan:
    """Translate the VideoHelperSuite-style options into an FFmpeg decode plan.

    Order of operations, as in VHS: force the frame rate, skip the first frames, keep every
    Nth frame, cap the count. Keeping every Nth frame divides the frame rate, so duration
    (and audio sync) is preserved. ``start_frame`` resumes a job at that output frame.
    """
    forced = options.fps > 0
    base_fps = Fraction(options.fps).limit_denominator(1001) if forced else meta.fps
    nth = options.every_nth

    available = round(meta.duration * base_fps) if forced else meta.total_frames
    after_skip = max(0, available - options.skip_frames)
    total = math.ceil(after_skip / nth)
    if options.max_frames > 0:
        total = min(total, options.max_frames)
    remaining = max(0, total - start_frame)
    skip = options.skip_frames + start_frame * nth  # in frames of the base timeline

    width, height = compute_input_size(
        meta.width, meta.height, options.pre_width, options.pre_height
    )
    filters: list[str] = []
    seek: float | None = None
    if forced:
        filters.append(f"fps={base_fps.numerator}/{base_fps.denominator}")
    if skip > 0:
        if forced or meta.is_vfr:
            # Exact but slow: decodes and discards everything before the start.
            filters.append(f"trim=start_frame={skip},setpts=PTS-STARTPTS")
        else:
            # Constant frame rate: seek straight there. Half a frame early so rounding
            # can never land on the wrong side of the target frame.
            seek = float((skip - Fraction(1, 2)) / base_fps)
    if nth > 1:
        filters.append(f"select=not(mod(n\\,{nth}))")
    size = f"{width}:{height}:" if (width, height) != (meta.width, meta.height) else ""
    filters.append(f"scale={size}flags=lanczos+{_SWS_FLAGS}:in_color_matrix={_input_matrix(meta)}")
    filters.append("format=rgb24")

    out_fps = base_fps / nth
    return DecodePlan(
        width=width,
        height=height,
        fps=out_fps,
        frames=remaining,
        total_frames=total,
        seek_seconds=seek,
        filters=",".join(filters),
        audio_offset=float(options.skip_frames / base_fps),
        audio_duration=float(total / out_fps) if total else 0.0,
    )


# --------------------------------------------------------------------------- stderr helper


class _Stderr:
    """Drains a process' stderr on a daemon thread so the pipe can never fill and block."""

    def __init__(self, stream: Any) -> None:
        self.lines: deque[str] = deque(maxlen=200)
        self._thread = threading.Thread(target=self._drain, args=(stream,), daemon=True)
        self._thread.start()

    def _drain(self, stream: Any) -> None:
        with contextlib.suppress(ValueError, OSError):
            for raw in iter(stream.readline, b""):
                self.lines.append(raw.decode("utf-8", "replace").rstrip())

    def tail(self, count: int = 8) -> str:
        self._thread.join(timeout=2)
        return "\n".join(list(self.lines)[-count:]) or "(no error output)"


def _kill(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.kill()
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        with contextlib.suppress(OSError, ValueError):
            if stream is not None:
                stream.close()
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        proc.wait(timeout=10)


# --------------------------------------------------------------------------- reader


class VideoStreamReader:
    """Decodes a video into batches of ``(N, H, W, 3)`` uint8 RGB frames.

    Batches are views of a small ring of preallocated buffers: a batch stays valid until
    ``queue_depth + 1`` later batches have been taken, so use it (e.g. upload it to the
    GPU) before asking for many more.
    """

    def __init__(
        self,
        meta: VideoMetadata,
        plan: DecodePlan,
        batch_size: int = 1,
        *,
        queue_depth: int = 2,
        hwaccel: bool = False,
    ) -> None:
        self.meta = meta
        self.plan = plan
        self.batch_size = max(1, batch_size)
        self.frames_read = 0
        self._frame_bytes = plan.width * plan.height * 3
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, queue_depth))
        shape = (self.batch_size, plan.height, plan.width, 3)
        self._buffers = [np.empty(shape, dtype=np.uint8) for _ in range(max(1, queue_depth) + 2)]
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._stderr: _Stderr | None = None
        self._thread: threading.Thread | None = None
        self._hwaccel = hwaccel

    def command(self) -> list[str]:
        cmd = [_require("ffmpeg"), *_QUIET]
        if self._hwaccel:
            cmd += ["-hwaccel", "cuda"]
        if self.plan.seek_seconds is not None:
            cmd += ["-ss", f"{self.plan.seek_seconds:.6f}"]
        cmd += ["-i", str(self.meta.path), "-map", "0:v:0", "-an", "-sn", "-dn"]
        cmd += ["-vf", self.plan.filters, *_passthrough_args()]
        cmd += ["-frames:v", str(self.plan.frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        return cmd

    def __enter__(self) -> VideoStreamReader:
        self._proc = subprocess.Popen(
            self.command(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )
        _grow_pipe(self._proc.stdout)
        self._stderr = _Stderr(self._proc.stderr)
        self._thread = threading.Thread(target=self._produce, name="vupscale-decode", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _put(self, item: Any) -> None:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _produce(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        stdout = self._proc.stdout
        try:
            slot = 0
            while not self._stop.is_set() and self.frames_read < self.plan.frames:
                buffer = self._buffers[slot]
                slot = (slot + 1) % len(self._buffers)
                view = memoryview(buffer).cast("B")
                got = 0
                while got < len(view):
                    count = stdout.readinto(view[got:])
                    if not count:
                        break
                    got += count
                frames, partial = divmod(got, self._frame_bytes)
                if partial:
                    raise VideoError("decoder stopped in the middle of a frame")
                if frames:
                    self.frames_read += frames
                    self._put(buffer[:frames])
                if got < len(view):
                    break
            code = self._proc.wait()
            if code != 0 and not self._stop.is_set():
                assert self._stderr is not None
                raise VideoError(f"FFmpeg failed while decoding:\n{self._stderr.tail()}")
            self._put(None)
        except BaseException as exc:  # delivered to the consumer thread
            self._put(exc)

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            item = self._queue.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    @property
    def queued(self) -> int:
        return self._queue.qsize()

    def close(self) -> None:
        self._stop.set()
        _kill(self._proc)
        if self._thread is not None:
            self._thread.join(timeout=10)


# --------------------------------------------------------------------------- writer


@dataclass(frozen=True)
class AudioSource:
    """Audio to mux into the output: taken from ``path`` starting at ``offset`` seconds."""

    path: Path
    codec: str = ""
    offset: float = 0.0
    # Length to keep, normally the video's. Never ``-shortest``: that would let a short
    # audio track cut frames off the video. 0 = the rest of the source.
    duration: float = 0.0


def audio_codec_args(source_codec: str, container: str) -> list[str]:
    """Stream-copy the audio when the container allows it, otherwise re-encode."""
    container = container.lower()
    if container in (".mkv", ".mka"):
        return ["-c:a", "copy"]
    if source_codec in _COPYABLE_AUDIO.get(container, set()):
        return ["-c:a", "copy"]
    if container == ".webm":
        return ["-c:a", "libopus", "-b:a", "160k"]
    return ["-c:a", "aac", "-b:a", "192k"]


def encoder_args(encoder: str, preset: str, quality: int, container: str = ".mp4") -> list[str]:
    """Rate control and speed settings for each supported encoder."""
    args = ["-c:v", encoder]
    if encoder.endswith("_nvenc"):
        args += ["-preset", preset or "p5", "-tune", "hq", "-rc", "vbr", "-cq", str(quality)]
        args += ["-b:v", "0"]
    elif encoder == "libaom-av1":
        args += ["-crf", str(quality), "-b:v", "0", "-cpu-used", preset or "6", "-row-mt", "1"]
    else:
        args += ["-crf", str(quality), "-preset", preset or "medium"]
        if encoder == "libx265":
            args += ["-x265-params", "log-level=error"]
    if encoder in ("libx265", "hevc_nvenc") and container.lower() in (".mp4", ".mov", ".m4v"):
        args += ["-tag:v", "hvc1"]  # required for playback on Apple devices
    return args


def _audio_inputs(audio: AudioSource | None) -> list[str]:
    if audio is None:
        return []
    args: list[str] = []
    if audio.offset > 0:
        args += ["-ss", f"{audio.offset:.6f}"]
    if audio.duration > 0:
        args += ["-t", f"{audio.duration:.6f}"]
    return [*args, "-i", str(audio.path)]


def _fps_text(fps: Fraction | float) -> str:
    value = Fraction(fps).limit_denominator(100000)
    return f"{value.numerator}/{value.denominator}"


class VideoStreamWriter:
    """Encodes ``(N, H, W, 3)`` uint8 RGB batches into a video file, on its own thread.

    ``write`` returns as soon as the batch is queued. The array must stay untouched until
    it has been encoded: with ``queue_depth`` batches waiting plus one being written, a
    producer that reuses buffers needs a ring of at least ``queue_depth + 2``.
    """

    def __init__(
        self,
        output: Path,
        width: int,
        height: int,
        fps: Fraction | float,
        *,
        encoder: str = "libx264",
        preset: str = "medium",
        quality: int = 20,
        pix_fmt: str = "yuv420p",
        audio: AudioSource | None = None,
        faststart: bool = True,
        queue_depth: int = 2,
    ) -> None:
        self.output = Path(output)
        self.width, self.height = width, height
        self.fps = fps
        self.encoder, self.preset, self.quality, self.pix_fmt = encoder, preset, quality, pix_fmt
        self.audio = audio
        self.faststart = faststart
        self.frames_written = 0
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, queue_depth))
        self._error: BaseException | None = None
        self._proc: subprocess.Popen | None = None
        self._stderr: _Stderr | None = None
        self._thread: threading.Thread | None = None
        self._closed = False

    def command(self) -> list[str]:
        suffix = self.output.suffix
        cmd = [_require("ffmpeg"), *_QUIET, "-y"]
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.width}x{self.height}"]
        cmd += ["-r", _fps_text(self.fps), "-i", "pipe:0", *_audio_inputs(self.audio)]
        cmd += ["-map", "0:v:0"]
        if self.audio is not None:
            cmd += ["-map", "1:a:0?", *audio_codec_args(self.audio.codec, suffix)]
        chain = f"scale=flags={_SWS_FLAGS}:out_color_matrix=bt709:out_range=tv"
        if self.width % 2 or self.height % 2:
            chain = f"pad=ceil(iw/2)*2:ceil(ih/2)*2,{chain}"
        cmd += ["-vf", f"{chain},format={self.pix_fmt}"]
        cmd += encoder_args(self.encoder, self.preset, self.quality, suffix)
        cmd += ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
        cmd += ["-color_range", "tv"]
        if self.faststart and suffix.lower() in (".mp4", ".mov", ".m4v"):
            cmd += ["-movflags", "+faststart"]
        return [*cmd, str(self.output)]

    def __enter__(self) -> VideoStreamWriter:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(self.command(), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        _grow_pipe(self._proc.stdin)
        self._stderr = _Stderr(self._proc.stderr)
        self._thread = threading.Thread(target=self._consume, name="vupscale-encode", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, *rest: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    def _consume(self) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        stdin = self._proc.stdin
        while True:
            item = self._queue.get()
            if item is None:
                return
            if self._error is not None:
                continue  # keep draining so the producer never blocks on a dead encoder
            try:
                stdin.write(memoryview(np.ascontiguousarray(item)).cast("B"))
                self.frames_written += len(item)
            except (OSError, ValueError) as exc:
                self._error = exc

    def _failure(self) -> VideoError:
        assert self._stderr is not None
        return VideoError(
            f"FFmpeg failed while encoding {self.output.name}:\n{self._stderr.tail()}"
        )

    def write(self, frames: np.ndarray) -> None:
        """Queue a batch for encoding; blocks while the queue is full."""
        if frames.shape[1:] != (self.height, self.width, 3) or frames.dtype != np.uint8:
            raise ValueError(
                f"expected uint8 frames (N, {self.height}, {self.width}, 3), got {frames.shape}"
            )
        while True:
            if self._error is not None:
                raise self._failure()
            try:
                self._queue.put(frames, timeout=0.1)
                return
            except queue.Full:
                continue

    @property
    def queued(self) -> int:
        return self._queue.qsize()

    def close(self) -> None:
        """Flush everything, wait for FFmpeg and raise if the encode failed."""
        if self._closed:
            return
        self._closed = True
        assert self._proc is not None and self._thread is not None
        self._queue.put(None)
        self._thread.join()
        with contextlib.suppress(OSError, ValueError):
            self._proc.stdin.close()  # type: ignore[union-attr]
        code = self._proc.wait()
        if self._error is not None or code != 0:
            raise self._failure()

    def abort(self) -> None:
        """Stop immediately, without finishing the file."""
        if self._closed:
            return
        self._closed = True
        self._error = self._error or VideoError("aborted")
        _kill(self._proc)
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)
        if self._thread is not None:
            self._thread.join(timeout=10)


# --------------------------------------------------------------------------- segments


def part_path(output: Path) -> Path:
    """Temporary sibling that keeps the extension, so FFmpeg still knows the format."""
    return output.with_name(f"{output.stem}.part{output.suffix}")


def concat_segments(segments: list[Path], output: Path, audio: AudioSource | None = None) -> Path:
    """Join checkpoint segments without re-encoding and mux the audio; atomic on success."""
    if not segments:
        raise VideoError("no segments to concatenate")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix
    temp = part_path(output)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as listing:
        for segment in segments:
            escaped = str(Path(segment).resolve()).replace("'", "'\\''")
            listing.write(f"file '{escaped}'\n")
    try:
        cmd = [_require("ffmpeg"), *_QUIET, "-y", "-f", "concat", "-safe", "0", "-i", listing.name]
        cmd += [*_audio_inputs(audio), "-map", "0:v:0", "-c:v", "copy"]
        if audio is not None:
            cmd += ["-map", "1:a:0?", *audio_codec_args(audio.codec, suffix)]
        mp4_like = suffix.lower() in (".mp4", ".mov", ".m4v")
        if mp4_like and probe_video(segments[0]).codec == "hevc":
            cmd += ["-tag:v", "hvc1"]
        if mp4_like:
            cmd += ["-movflags", "+faststart"]
        done = subprocess.run([*cmd, str(temp)], capture_output=True, text=True, check=False)
        if done.returncode != 0:
            temp.unlink(missing_ok=True)
            raise VideoError(f"FFmpeg failed while joining segments:\n{done.stderr.strip()[-800:]}")
        os.replace(temp, output)
    finally:
        Path(listing.name).unlink(missing_ok=True)
    return output
