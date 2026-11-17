"""Video I/O tests use real FFmpeg on tiny synthetic clips."""

from __future__ import annotations

import subprocess
import time
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from conftest import make_video, requires_ffmpeg
from vupscale.config import UpscaleOptions
from vupscale.video_io import (
    AudioSource,
    VideoError,
    VideoStreamReader,
    VideoStreamWriter,
    _input_matrix,
    audio_codec_args,
    concat_segments,
    count_frames,
    encoder_args,
    part_path,
    plan_decode,
    probe_video,
)

pytestmark = requires_ffmpeg
W, H, FPS, COUNT = 64, 48, 24, 40


def frame_value(index: int) -> int:
    return 10 + index * 5


def write_indexed(path: Path, count: int = COUNT, **kw) -> Path:
    """A clip whose frame ``i`` is a flat grey of value ``10 + 5 * i``."""
    frames = np.empty((count, H, W, 3), dtype=np.uint8)
    for i in range(count):
        frames[i] = frame_value(i)
    with VideoStreamWriter(path, W, H, FPS, quality=8, preset="ultrafast", **kw) as writer:
        writer.write(frames[: count // 2])
        writer.write(frames[count // 2 :])
    return path


@pytest.fixture(scope="module")
def indexed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_indexed(tmp_path_factory.mktemp("indexed") / "indexed.mp4")


def read_all(path: Path, batch_size: int = 7, start_frame: int = 0, **options):
    meta = probe_video(path)
    plan = plan_decode(meta, UpscaleOptions(input=path, model="m", **options), start_frame)
    with VideoStreamReader(meta, plan, batch_size) as reader:
        batches = [batch.copy() for batch in reader]
    frames = np.concatenate(batches) if batches else np.empty((0, plan.height, plan.width, 3))
    return plan, frames


def indices(frames: np.ndarray) -> list[int]:
    return [round((float(f.mean()) - 10) / 5) for f in frames]


# ------------------------------------------------------------------ probe


def test_probe(sample_video: Path):
    meta = probe_video(sample_video)
    assert (meta.width, meta.height) == (320, 240)
    assert meta.fps == 24 and not meta.is_vfr and meta.total_frames == 48
    assert meta.duration == pytest.approx(2.0, abs=0.1)
    assert meta.has_audio and meta.audio_codec == "aac" and meta.codec == "h264"
    assert count_frames(sample_video) == 48


def test_probe_errors(tmp_path: Path):
    with pytest.raises(VideoError, match="not found"):
        probe_video(tmp_path / "nope.mp4")
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video")
    with pytest.raises(VideoError, match="cannot read"):
        probe_video(junk)


def test_probe_swaps_size_for_rotated_video(sample_video: Path, tmp_path: Path):
    rotated = tmp_path / "rotated.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-display_rotation", "90", "-i", str(sample_video),
         "-c", "copy", str(rotated)],
        check=True,
    )  # fmt: skip
    meta = probe_video(rotated)
    assert (meta.width, meta.height) == (240, 320)
    plan, frames = read_all(rotated, max_frames=2)
    assert frames.shape == (2, 320, 240, 3)  # what FFmpeg really delivers


# ------------------------------------------------------------------ reader


def test_reads_every_frame_in_order(indexed: Path):
    plan, frames = read_all(indexed)
    assert plan.frames == COUNT and plan.fps == FPS and plan.seek_seconds is None
    assert frames.shape == (COUNT, H, W, 3) and frames.dtype == np.uint8
    assert indices(frames) == list(range(COUNT))


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({"skip_frames": 1}, list(range(1, COUNT))),
        ({"skip_frames": 13}, list(range(13, COUNT))),
        ({"max_frames": 9}, list(range(9))),
        ({"every_nth": 3}, list(range(0, COUNT, 3))),
        ({"skip_frames": 7, "every_nth": 3, "max_frames": 5}, [7, 10, 13, 16, 19]),
        ({"skip_frames": COUNT - 1}, [COUNT - 1]),
        ({"skip_frames": COUNT + 5}, []),
    ],
)
def test_frame_selection_is_exact(indexed: Path, options: dict, expected: list[int]):
    plan, frames = read_all(indexed, **options)
    assert plan.frames == len(expected)
    assert indices(frames) == expected


def test_resume_point_continues_exactly_where_it_stopped(indexed: Path):
    options = {"skip_frames": 4, "every_nth": 2}
    full_plan, full = read_all(indexed, **options)
    plan, rest = read_all(indexed, start_frame=10, **options)
    assert plan.total_frames == full_plan.total_frames == 18 and plan.frames == 8
    assert indices(rest) == indices(full)[10:] == list(range(24, 40, 2))
    assert plan.audio_offset == full_plan.audio_offset  # audio belongs to the whole job


def test_every_nth_divides_the_frame_rate_to_keep_duration(indexed: Path):
    plan, _ = read_all(indexed, every_nth=2, skip_frames=12)
    assert plan.fps == 12 and plan.frames == 14
    assert plan.audio_offset == pytest.approx(0.5)
    assert plan.audio_duration == pytest.approx(14 / 12)


def test_forced_fps_resamples_without_changing_duration(indexed: Path):
    plan, frames = read_all(indexed, fps=12)
    assert plan.fps == 12 and plan.frames == 20
    assert len(frames) == 20 and indices(frames)[:3] == [0, 2, 4]
    plan, frames = read_all(indexed, fps=12, skip_frames=5)  # uses the exact trim path
    assert len(frames) == plan.frames == 15 and indices(frames)[0] == 10


def test_pre_scale(indexed: Path):
    plan, frames = read_all(indexed, pre_height=24, max_frames=3)
    assert (plan.width, plan.height) == (32, 24) and frames.shape == (3, 24, 32, 3)
    assert indices(frames) == [0, 1, 2]


def test_stopping_early_does_not_hang(sample_video: Path):
    meta = probe_video(sample_video)
    plan = plan_decode(meta, UpscaleOptions(input=sample_video, model="m"))
    start = time.monotonic()
    with VideoStreamReader(meta, plan, batch_size=2, queue_depth=1) as reader:
        for batch in reader:
            assert batch.shape == (2, 240, 320, 3)
            break
    assert time.monotonic() - start < 10


def test_decode_failure_is_reported(sample_video: Path, tmp_path: Path):
    broken = tmp_path / "broken.mp4"
    data = sample_video.read_bytes()
    meta = probe_video(sample_video)
    broken.write_bytes(data[: len(data) // 3])  # truncated after probing the good file
    plan = plan_decode(meta, UpscaleOptions(input=broken, model="m"))
    bad_meta = replace(meta, path=broken)
    with pytest.raises(VideoError), VideoStreamReader(bad_meta, plan, 4) as reader:
        assert sum(len(b) for b in reader) < 48
        raise VideoError("fewer frames than planned")  # either path is a reported failure


# ------------------------------------------------------------------ writer


def test_writer_output_is_what_was_asked(tmp_path: Path, sample_video: Path):
    out = write_indexed(
        tmp_path / "out.mp4",
        audio=AudioSource(sample_video, "aac", offset=0.5, duration=COUNT / FPS),
    )
    meta = probe_video(out)
    assert (meta.width, meta.height, meta.total_frames) == (W, H, COUNT)
    assert meta.fps == FPS and meta.pix_fmt == "yuv420p" and meta.color_space == "bt709"
    assert meta.has_audio and meta.audio_codec == "aac"
    assert meta.duration == pytest.approx(COUNT / FPS, abs=0.15)


def test_fractional_frame_rates_survive(tmp_path: Path):
    out = tmp_path / "ntsc.mp4"
    with VideoStreamWriter(out, W, H, Fraction(30000, 1001), preset="ultrafast") as writer:
        writer.write(np.zeros((6, H, W, 3), dtype=np.uint8))
    assert probe_video(out).fps == Fraction(30000, 1001)


@pytest.mark.parametrize("colour", [(200, 30, 30), (30, 200, 30), (30, 30, 200), (128, 128, 128)])
def test_colours_survive_a_round_trip(tmp_path: Path, colour: tuple[int, int, int]):
    """A BT.601/BT.709 mix-up shifts saturated colours by 10-20 levels."""
    out = tmp_path / "colour.mp4"
    frames = np.empty((4, 720, 1280, 3), dtype=np.uint8)  # HD: the BT.709 case
    frames[:] = colour
    with VideoStreamWriter(out, 1280, 720, 24, quality=4, preset="ultrafast") as writer:
        writer.write(frames)
    _, decoded = read_all(out)
    assert np.abs(decoded.reshape(-1, 3).mean(axis=0) - colour).max() <= 3


def test_untagged_sd_video_is_read_as_bt601(tmp_path: Path):
    clip = tmp_path / "sd.mp4"
    source = "color=c=0xC81E1E:s=64x48:r=24:d=0.5"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", source,
         "-vf", "scale=out_color_matrix=smpte170m,format=yuv420p", "-c:v", "libx264", "-crf", "4",
         str(clip)],
        check=True,
    )  # fmt: skip
    # Older FFmpeg leaves the file untagged, newer ones tag it; both must decode as BT.601.
    meta = probe_video(clip)
    assert meta.color_space in ("", "bt470bg", "smpte170m")
    untagged = replace(meta, color_space="")
    assert _input_matrix(untagged) == "bt601"
    assert _input_matrix(replace(untagged, color_space="bt470bg")) == "bt601"
    assert _input_matrix(replace(untagged, width=1920, height=1080)) == "bt709"
    assert _input_matrix(replace(untagged, color_space="bt2020nc")) == "bt2020"
    _, decoded = read_all(clip)
    assert np.abs(decoded.reshape(-1, 3).mean(axis=0) - (200, 30, 30)).max() <= 4


def test_writer_reports_ffmpeg_errors(tmp_path: Path):
    frames = np.zeros((2, H, W, 3), dtype=np.uint8)
    bad = VideoStreamWriter(tmp_path / "x.mp4", W, H, FPS, encoder="no_such_encoder")
    with pytest.raises(VideoError, match="FFmpeg failed while encoding"), bad as writer:
        for _ in range(50):  # must not block forever on a dead encoder
            writer.write(frames)
    with VideoStreamWriter(tmp_path / "y.mp4", W, H, FPS, preset="ultrafast") as writer:
        writer.write(frames)
        pytest.raises(ValueError, writer.write, np.zeros((1, H + 2, W, 3), dtype=np.uint8))


def test_abort_on_exception_leaves_no_hanging_process(tmp_path: Path):
    target = VideoStreamWriter(tmp_path / "z.mp4", W, H, FPS, preset="ultrafast")
    with pytest.raises(KeyboardInterrupt), target as writer:
        writer.write(np.zeros((2, H, W, 3), dtype=np.uint8))
        raise KeyboardInterrupt
    assert writer._proc.poll() is not None


def test_encoder_arguments():
    nvenc = encoder_args("hevc_nvenc", "p5", 20)
    assert nvenc[:2] == ["-c:v", "hevc_nvenc"] and "-cq" in nvenc and "hvc1" in nvenc
    assert "-crf" not in nvenc
    x264 = encoder_args("libx264", "faster", 18)
    assert x264 == ["-c:v", "libx264", "-crf", "18", "-preset", "faster"]
    assert "hvc1" in encoder_args("libx265", "medium", 20, ".mov")
    assert "hvc1" not in encoder_args("libx265", "medium", 20, ".mkv")
    assert "-cpu-used" in encoder_args("libaom-av1", "6", 30)


def test_audio_is_copied_only_where_the_container_allows():
    assert audio_codec_args("aac", ".mp4") == ["-c:a", "copy"]
    assert audio_codec_args("opus", ".mp4")[:2] == ["-c:a", "aac"]
    assert audio_codec_args("flac", ".mkv") == ["-c:a", "copy"]
    assert audio_codec_args("aac", ".webm")[:2] == ["-c:a", "libopus"]


@pytest.mark.parametrize("encoder", ["libx264", "libx265"])
def test_real_encoders(tmp_path: Path, encoder: str):
    out = tmp_path / f"{encoder}.mp4"
    with VideoStreamWriter(out, W, H, FPS, encoder=encoder, preset="ultrafast") as writer:
        writer.write(np.full((5, H, W, 3), 90, dtype=np.uint8))
    assert probe_video(out).total_frames == 5


# ------------------------------------------------------------------ segments


def test_concat_segments_with_audio(tmp_path: Path, sample_video: Path):
    parts = []
    for n in range(3):
        part = tmp_path / f"seg_{n}.mp4"
        frames = np.empty((8, H, W, 3), dtype=np.uint8)
        for i in range(8):
            frames[i] = frame_value(n * 8 + i)
        with VideoStreamWriter(
            part, W, H, FPS, quality=8, preset="ultrafast", faststart=False
        ) as w:
            w.write(frames)
        parts.append(part)
    out = concat_segments(parts, tmp_path / "joined.mp4", AudioSource(sample_video, "aac", 0, 1.0))
    meta = probe_video(out)
    assert meta.total_frames == 24 and meta.has_audio
    assert meta.duration == pytest.approx(1.0, abs=0.15)
    assert not part_path(out).exists()
    _, frames = read_all(out)
    assert indices(frames) == list(range(24))  # no duplicated or dropped frame at the joins


def test_concat_failure_leaves_no_partial_file(tmp_path: Path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"junk")
    with pytest.raises(VideoError):
        concat_segments([bad], tmp_path / "out.mp4")
    assert not (tmp_path / "out.mp4").exists() and not part_path(tmp_path / "out.mp4").exists()
    with pytest.raises(VideoError, match="no segments"):
        concat_segments([], tmp_path / "out.mp4")


def test_a_short_audio_track_never_cuts_the_video(tmp_path: Path):
    short = make_video(tmp_path / "short.mp4", seconds=0.5, size="64x48")
    out = write_indexed(tmp_path / "long.mp4", audio=AudioSource(short, "aac"))
    assert probe_video(out).total_frames == COUNT
