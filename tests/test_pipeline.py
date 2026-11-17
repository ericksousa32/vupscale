"""End-to-end renders on CPU with the tiny model and a synthetic clip."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_ffmpeg
from vupscale import pipeline, upscale_video
from vupscale.checkpoint import MANIFEST, CheckpointError
from vupscale.config import UpscaleOptions
from vupscale.tuning import AdaptiveBatchController
from vupscale.video_io import VideoError, VideoStreamReader, plan_decode, probe_video

pytestmark = requires_ffmpeg


@pytest.fixture
def clip(tmp_path: Path, sample_video: Path) -> Path:
    """48 frames, 320x240, 24 fps, with audio."""
    path = tmp_path / "clip.mp4"
    shutil.copyfile(sample_video, path)
    return path


def decode(path: Path) -> np.ndarray:
    meta = probe_video(path)
    plan = plan_decode(meta, UpscaleOptions(input=path, model="m"))
    with VideoStreamReader(meta, plan, 8) as reader:
        return np.concatenate([batch.copy() for batch in reader])


def fail_after(monkeypatch: pytest.MonkeyPatch, batches: int, error: type[BaseException]) -> None:
    """Make the render die after ``batches`` batches, like a crash or a Ctrl+C."""
    real = AdaptiveBatchController.run
    seen = {"count": 0}

    def flaky(self, process, frames):
        seen["count"] += 1
        if seen["count"] > batches:
            raise error("simulated")
        return real(self, process, frames)

    monkeypatch.setattr(AdaptiveBatchController, "run", flaky)


def segments(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.glob("seg_*.mp4"))


# ------------------------------------------------------------------ plain renders


def test_render(clip: Path, tiny_model: Path):
    out = upscale_video(clip, model=str(tiny_model))
    assert out == clip.with_name("clip_x2.mp4")
    meta = probe_video(out)
    assert (meta.width, meta.height) == (640, 480)
    assert meta.total_frames == 48 and meta.fps == 24
    assert meta.has_audio and meta.duration == pytest.approx(2.0, abs=0.15)
    assert not list(clip.parent.glob("*.part*"))


def test_render_is_a_real_upscale_of_the_input(clip: Path, tiny_model: Path):
    """Same frames in the same order: compare against the source reduced to grey levels."""
    out = upscale_video(clip, model=str(tiny_model), no_audio=True)
    source, result = decode(clip), decode(out)
    assert len(source) == len(result) == 48
    # A random-weight model changes the picture, but frame-to-frame motion must line up:
    # the frame differences of the output track those of the input.
    moved_in = np.abs(np.diff(source.astype(np.int16), axis=0)).mean(axis=(1, 2, 3))
    moved_out = np.abs(np.diff(result.astype(np.int16), axis=0)).mean(axis=(1, 2, 3))
    assert np.corrcoef(moved_in, moved_out)[0, 1] > 0.5
    assert not probe_video(out).has_audio


def test_options_shape_the_output(clip: Path, tiny_model: Path, tmp_path: Path):
    out = upscale_video(
        clip,
        model=str(tiny_model),
        output=tmp_path / "result.mkv",
        target_height=360,
        skip_frames=6,
        every_nth=2,
        max_frames=10,
        codec="hevc",
    )
    meta = probe_video(out)
    assert out.name == "result.mkv" and meta.codec == "hevc"
    assert (meta.width, meta.height, meta.total_frames) == (480, 360, 10)
    assert meta.fps == 12  # every 2nd frame: half the rate, same duration, audio in sync
    assert meta.has_audio


def test_output_directory_and_batch_input(clip: Path, tiny_model: Path, tmp_path: Path):
    shutil.copyfile(clip, clip.with_name("other.mp4"))
    outs = upscale_video(clip.parent, model=str(tiny_model), output=tmp_path / "out", max_frames=3)
    assert [o.name for o in outs] == ["clip_x2.mp4", "other_x2.mp4"]
    assert all(probe_video(o).total_frames == 3 for o in outs)


def test_refuses_bad_jobs(clip: Path, tiny_model: Path):
    with pytest.raises(VideoError, match="select no frames"):
        upscale_video(clip, model=str(tiny_model), skip_frames=500)
    with pytest.raises(Exception, match="overwrite the input"):
        upscale_video(clip, model=str(tiny_model), output=clip)


def test_failure_leaves_no_partial_output(clip: Path, tiny_model: Path, monkeypatch):
    fail_after(monkeypatch, 5, RuntimeError)
    with pytest.raises(RuntimeError):
        upscale_video(clip, model=str(tiny_model))
    assert [p.name for p in clip.parent.iterdir()] == ["clip.mp4"]


def test_dry_run_renders_nothing(clip: Path, tiny_model: Path):
    out = upscale_video(clip, model=str(tiny_model), dry_run=True)
    assert out == clip.with_name("clip_x2.mp4") and not out.exists()


def test_oom_during_a_render_is_survived(clip: Path, tiny_model: Path, monkeypatch):
    import torch

    from vupscale.engine import UpscaleEngine

    real = UpscaleEngine._forward
    calls = {"n": 0}

    def tight(self, x):
        calls["n"] += 1
        if calls["n"] == 3 and x.shape[-1] == 320:  # one full-frame call fails
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return real(self, x)

    monkeypatch.setattr(UpscaleEngine, "_forward", tight)
    out = upscale_video(clip, model=str(tiny_model), max_frames=12, no_audio=True)
    assert probe_video(out).total_frames == 12


# ------------------------------------------------------------------ checkpoints


def test_segmented_render_matches_a_plain_one(clip: Path, tiny_model: Path, tmp_path: Path):
    plain = upscale_video(clip, model=str(tiny_model), output=tmp_path / "plain.mp4")
    ckpt = tmp_path / "ckpt"
    out = upscale_video(
        clip, model=str(tiny_model), output=tmp_path / "seg.mp4",
        checkpoint_dir=ckpt, segment_seconds=0.001,
    )  # fmt: skip
    meta = probe_video(out)
    assert meta.total_frames == 48 and meta.has_audio
    assert meta.duration == pytest.approx(2.0, abs=0.15)
    # Many tiny segments joined: same frames, no duplicate or missing one at the joins.
    a, b = decode(plain).astype(np.int16), decode(out).astype(np.int16)
    per_frame = np.abs(a - b).mean(axis=(1, 2, 3))
    assert per_frame.max() < 6, per_frame.max()
    # Finished: segments are gone, the manifest records the result.
    assert segments(ckpt) == []
    manifest = json.loads((ckpt / MANIFEST).read_text())
    assert manifest["completed"] and manifest["output"] == str(out)


def test_resume_after_a_crash(clip: Path, tiny_model: Path, tmp_path: Path, monkeypatch):
    ckpt, out = tmp_path / "ckpt", tmp_path / "out.mp4"
    job = dict(model=str(tiny_model), output=out, checkpoint_dir=ckpt, segment_seconds=0.001)
    plain = upscale_video(clip, model=str(tiny_model), output=tmp_path / "plain.mp4")

    with monkeypatch.context() as patch:
        fail_after(patch, 20, RuntimeError)
        with pytest.raises(RuntimeError):
            upscale_video(clip, **job)
    saved = segments(ckpt)
    assert saved and not out.exists()
    done = max(int(name[11:17]) for name in saved) + 1
    assert 0 < done <= 20  # everything before the crash, minus the open segment

    with pytest.raises(CheckpointError, match="--resume"):
        upscale_video(clip, **job)  # refusing to silently discard the progress

    calls = {"n": 0}
    real = AdaptiveBatchController.run

    def counting(self, process, frames):
        calls["n"] += len(frames)
        return real(self, process, frames)

    monkeypatch.setattr(AdaptiveBatchController, "run", counting)
    result = upscale_video(clip, resume=True, **job)
    assert calls["n"] == 48 - done  # only the missing frames were rendered
    assert probe_video(result).total_frames == 48
    a, b = decode(plain).astype(np.int16), decode(result).astype(np.int16)
    assert np.abs(a - b).mean(axis=(1, 2, 3)).max() < 6


def test_ctrl_c_saves_the_open_segment(clip: Path, tiny_model: Path, tmp_path: Path, monkeypatch):
    ckpt = tmp_path / "ckpt"
    job = dict(model=str(tiny_model), output=tmp_path / "out.mp4", checkpoint_dir=ckpt)
    with monkeypatch.context() as patch:
        fail_after(patch, 10, KeyboardInterrupt)
        with pytest.raises(KeyboardInterrupt):
            upscale_video(clip, **job)  # default 180 s segments: all frames in one open file
    assert segments(ckpt) == ["seg_000000-000009.mp4"]
    assert probe_video(upscale_video(clip, resume=True, **job)).total_frames == 48


def test_resume_discards_damaged_segments(
    clip: Path, tiny_model: Path, tmp_path: Path, monkeypatch
):
    ckpt = tmp_path / "ckpt"
    job = dict(model=str(tiny_model), output=tmp_path / "out.mp4", checkpoint_dir=ckpt,
               segment_seconds=0.001)  # fmt: skip
    with monkeypatch.context() as patch:
        fail_after(patch, 30, RuntimeError)
        with pytest.raises(RuntimeError):
            upscale_video(clip, **job)
    saved = segments(ckpt)
    assert len(saved) >= 3
    victim = ckpt / saved[1]
    victim.write_bytes(victim.read_bytes()[:200])  # e.g. a copy cut short by a disconnect
    (ckpt / "seg_000090-000099.mp4.part").write_bytes(b"junk")

    result = upscale_video(clip, resume=True, **job)
    assert probe_video(result).total_frames == 48
    assert not list(ckpt.glob("*.part"))


def test_resume_refuses_a_different_job(clip: Path, tiny_model: Path, tmp_path: Path, monkeypatch):
    ckpt = tmp_path / "ckpt"
    job = dict(model=str(tiny_model), output=tmp_path / "out.mp4", checkpoint_dir=ckpt,
               segment_seconds=0.001)  # fmt: skip
    with monkeypatch.context() as patch:
        fail_after(patch, 10, RuntimeError)
        with pytest.raises(RuntimeError):
            upscale_video(clip, **job)
    with pytest.raises(CheckpointError, match="quality"):
        upscale_video(clip, resume=True, quality=30, **job)
    with pytest.raises(CheckpointError, match="size"):
        upscale_video(clip, resume=True, target_height=360, **job)
    assert probe_video(upscale_video(clip, resume=True, **job)).total_frames == 48


def test_rerunning_a_finished_job_is_instant(
    clip: Path, tiny_model: Path, tmp_path: Path, monkeypatch
):
    job = dict(model=str(tiny_model), output=tmp_path / "out.mp4",
               checkpoint_dir=tmp_path / "ckpt", resume=True, max_frames=8)  # fmt: skip
    first = upscale_video(clip, **job)
    fail_after(monkeypatch, 0, AssertionError)  # any rendering now would blow up
    assert upscale_video(clip, **job) == first
    first.unlink()
    monkeypatch.undo()
    assert probe_video(upscale_video(clip, **job)).total_frames == 8  # output gone: redo


def test_resume_with_nothing_saved_starts_from_scratch(
    clip: Path, tiny_model: Path, tmp_path: Path
):
    out = upscale_video(
        clip, model=str(tiny_model), output=tmp_path / "out.mp4",
        checkpoint_dir=tmp_path / "fresh", resume=True, max_frames=5,
    )  # fmt: skip
    assert probe_video(out).total_frames == 5


def test_slow_mounts_are_written_through_local_disk(
    clip: Path, tiny_model: Path, tmp_path: Path, monkeypatch
):
    drive = tmp_path / "drive"
    monkeypatch.setattr(pipeline, "_SLOW_MOUNTS", (str(drive),))
    out = upscale_video(
        clip, model=str(tiny_model), output=drive / "videos" / "o.mp4", max_frames=4
    )
    assert probe_video(out).total_frames == 4
    assert [p.name for p in out.parent.iterdir()] == ["o.mp4"]


def test_a_finished_checkpoint_dir_can_host_a_new_job(clip: Path, tiny_model: Path, tmp_path: Path):
    common = dict(model=str(tiny_model), checkpoint_dir=tmp_path / "ckpt", resume=True)
    upscale_video(clip, output=tmp_path / "a.mp4", max_frames=4, **common)
    second = upscale_video(clip, output=tmp_path / "b.mp4", max_frames=6, **common)
    assert probe_video(second).total_frames == 6
