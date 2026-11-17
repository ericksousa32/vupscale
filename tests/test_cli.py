from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

import vupscale
from vupscale import pipeline
from vupscale.cli import app
from vupscale.config import OptionsError, resolve_output
from vupscale.video_io import probe_video

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(result) -> str:
    """Output without ANSI styling (Rich forces colours on GitHub Actions)."""
    return _ANSI.sub("", result.output)


@pytest.fixture
def video(tmp_path: Path, sample_video: Path) -> Path:
    path = tmp_path / "clip.mp4"
    shutil.copyfile(sample_video, path)
    return path


@pytest.fixture
def weights(tiny_model: Path) -> Path:
    return tiny_model


def test_help_and_version():
    assert runner.invoke(app, ["--help"]).exit_code == 0
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and vupscale.__version__ in plain(result)
    result = runner.invoke(app, ["upscale", "--help"])
    assert result.exit_code == 0 and "--tile-size" in plain(result)


def test_models_lists_the_catalog():
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "realesrgan-x4" in plain(result) and "realesr-animevideov3" in plain(result)


def test_doctor_runs():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code in (0, 1)  # 1 only when ffmpeg/torch are really missing
    assert "environment" in plain(result) and "ffmpeg" in plain(result)


@pytest.mark.parametrize("style", ["positional", "explicit", "flag"])
def test_dry_run_in_every_invocation_style(style: str, video: Path, weights: Path):
    args = {
        "positional": [str(video), "-m", str(weights)],  # default command
        "explicit": ["upscale", str(video), "-m", str(weights)],
        "flag": ["-i", str(video), "-m", str(weights)],
    }[style]
    result = runner.invoke(app, [*args, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "clip_x2.mp4" in plain(result) and "nothing was rendered" in plain(result)


def test_dry_run_reflects_overrides(video: Path, weights: Path):
    result = runner.invoke(
        app,
        [str(video), "-m", str(weights), "--dry-run", "--target-height", "2160",
         "--every-nth", "2", "--tile-size", "128", "--codec", "hevc", "--crf", "18"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    out = plain(result)
    assert "clip_2160p.mp4" in out and "up to 128 px" in out and "libx265" in out
    assert "quality 18" in out and "320x240 -> 2880x2160" in out
    assert "24 at 12.000 fps" in out and "aac from the input" in out  # audio stays in sync


def test_errors_are_friendly(video: Path, weights: Path, tmp_path: Path):
    missing = runner.invoke(app, [str(tmp_path / "nope.mp4"), "-m", str(weights)])
    assert missing.exit_code == 2 and "input not found" in plain(missing)
    conflict = runner.invoke(
        app, [str(video), "-m", str(weights), "--out-scale", "2", "--target-height", "1080"]
    )
    assert conflict.exit_code == 2 and "either" in plain(conflict)
    assert runner.invoke(app, [str(video)]).exit_code == 2  # --model is required
    assert runner.invoke(app, ["-m", str(weights)]).exit_code == 2  # input is required


def test_real_run_end_to_end(video: Path, weights: Path):
    result = runner.invoke(app, [str(video), "-m", str(weights), "--max-frames", "6"])
    assert result.exit_code == 0, plain(result)
    assert "done:" in plain(result)
    assert probe_video(video.with_name("clip_x2.mp4")).total_frames == 6


def test_directory_input_makes_one_job_per_video(tmp_path: Path, weights: Path):
    for name in ("a.mp4", "b.mkv", "readme.txt"):
        (tmp_path / name).touch()
    ckpt = tmp_path / "ckpt"
    jobs = pipeline.make_jobs(tmp_path, str(weights), checkpoint_dir=ckpt, output=tmp_path / "out")
    assert [j.input.name for j in jobs] == ["a.mp4", "b.mkv"]
    assert [j.checkpoint_dir for j in jobs] == [ckpt / "a", ckpt / "b"]
    assert resolve_output(jobs[1], 2) == tmp_path / "out" / "b_x2.mp4"
    with pytest.raises(OptionsError, match="directory"):
        pipeline.make_jobs(tmp_path, str(weights), output=tmp_path / "single.mp4")


def test_python_api_matches_the_cli(video: Path, weights: Path):
    result = vupscale.upscale_video(video, model=str(weights), dry_run=True, out_scale=1.5)
    assert result == video.with_name("clip_x1.5.mp4")
    assert not result.exists()
    with pytest.raises(OptionsError):
        vupscale.upscale_video(video, model=str(weights), every_nth=0)
    with pytest.raises(TypeError):
        vupscale.upscale_video(video, model=str(weights), not_an_option=1)


def test_bench_runs_the_real_engine_on_cpu(tiny_model: Path):
    result = runner.invoke(
        app,
        ["bench", "-m", str(tiny_model), "--size", "96x64", "--seconds", "0", "--device", "cpu"],
    )
    out = plain(result)
    assert result.exit_code == 0, out
    assert "96x64 -> 192x128" in out and "fps" in out and "fp32" in out
    assert "ms/frame" in out and "model" in out and "upload" in out
    assert runner.invoke(app, ["bench", "-m", str(tiny_model), "--size", "big"]).exit_code == 2
