from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import vupscale
from vupscale import pipeline
from vupscale.cli import app
from vupscale.config import OptionsError

runner = CliRunner()


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.touch()  # planning never opens the file
    return path


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    path = tmp_path / "model.pth"
    path.write_bytes(b"x")
    return path


def test_help_and_version():
    assert runner.invoke(app, ["--help"]).exit_code == 0
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and vupscale.__version__ in result.output
    result = runner.invoke(app, ["upscale", "--help"])
    assert result.exit_code == 0 and "--tile-size" in result.output


def test_models_lists_the_catalog():
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "realesrgan-x4" in result.output and "realesr-animevideov3" in result.output


def test_doctor_runs():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code in (0, 1)  # 1 only when ffmpeg/torch are really missing
    assert "environment" in result.output and "ffmpeg" in result.output


@pytest.mark.parametrize("style", ["positional", "explicit", "flag"])
def test_dry_run_in_every_invocation_style(style: str, video: Path, weights: Path):
    args = {
        "positional": [str(video), "-m", str(weights)],  # default command
        "explicit": ["upscale", str(video), "-m", str(weights)],
        "flag": ["-i", str(video), "-m", str(weights)],
    }[style]
    result = runner.invoke(app, [*args, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "clip_x<scale>.mp4" in result.output


def test_dry_run_reflects_overrides(video: Path, weights: Path):
    result = runner.invoke(
        app,
        [str(video), "-m", str(weights), "--dry-run", "--target-height", "2160",
         "--every-nth", "2", "--tile-size", "512", "--codec", "hevc", "--crf", "18"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    out = result.output
    assert "clip_2160p.mp4" in out and "512" in out and "hevc" in out and "quality 18" in out
    assert "every_nth" in out  # audio switched off, with the reason


def test_errors_are_friendly(video: Path, weights: Path, tmp_path: Path):
    missing = runner.invoke(app, [str(tmp_path / "nope.mp4"), "-m", str(weights)])
    assert missing.exit_code == 2 and "input not found" in missing.output
    conflict = runner.invoke(
        app, [str(video), "-m", str(weights), "--out-scale", "2", "--target-height", "1080"]
    )
    assert conflict.exit_code == 2 and "either" in conflict.output
    assert runner.invoke(app, [str(video)]).exit_code == 2  # --model is required
    assert runner.invoke(app, ["-m", str(weights)]).exit_code == 2  # input is required


def test_real_run_reports_not_implemented_yet(video: Path, weights: Path):
    result = runner.invoke(app, [str(video), "-m", str(weights)])
    assert result.exit_code == 1 and "not available yet" in result.output


def test_directory_input_makes_one_job_per_video(tmp_path: Path, weights: Path):
    for name in ("a.mp4", "b.mkv", "readme.txt"):
        (tmp_path / name).touch()
    ckpt = tmp_path / "ckpt"
    jobs = pipeline.make_jobs(tmp_path, str(weights), checkpoint_dir=ckpt, output=tmp_path / "out")
    assert [j.input.name for j in jobs] == ["a.mp4", "b.mkv"]
    assert [j.checkpoint_dir for j in jobs] == [ckpt / "a", ckpt / "b"]
    assert pipeline.describe_output(jobs[1]) == str(tmp_path / "out" / "b_x<scale>.mp4")
    with pytest.raises(OptionsError, match="directory"):
        pipeline.make_jobs(tmp_path, str(weights), output=tmp_path / "single.mp4")


def test_python_api_matches_the_cli(video: Path, weights: Path):
    result = vupscale.upscale_video(video, model=str(weights), dry_run=True, out_scale=2)
    assert result == video.with_name("clip_x<scale>.mp4")
    with pytest.raises(OptionsError):
        vupscale.upscale_video(video, model=str(weights), every_nth=0)
    with pytest.raises(TypeError):
        vupscale.upscale_video(video, model=str(weights), not_an_option=1)
