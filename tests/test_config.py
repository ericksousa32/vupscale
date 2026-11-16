from __future__ import annotations

from pathlib import Path

import pytest

from vupscale.config import (
    Codec,
    OptionsError,
    Precision,
    UpscaleOptions,
    expand_inputs,
    output_name,
    resolve_output,
)


def opts(**kw) -> UpscaleOptions:
    return UpscaleOptions(input=Path("clip.mp4"), model="m.pth", **kw)


def test_defaults_match_the_plan():
    o = opts()
    assert (o.batch_size, o.tile_size, o.tile_overlap) == (0, 0, 32)
    assert (o.fps, o.max_frames, o.skip_frames, o.every_nth) == (0, 0, 0, 1)
    assert (o.quality, o.pix_fmt, o.no_audio) == (20, "yuv420p", False)
    assert o.codec is Codec.auto and o.precision is Precision.auto
    assert o.segment_seconds == 180


def test_strings_are_coerced_for_the_python_api():
    o = opts(precision="fp16", codec="hevc", device="cpu", compile="off", output="out.mp4")
    assert o.precision is Precision.fp16 and o.codec is Codec.hevc
    assert o.output == Path("out.mp4")


@pytest.mark.parametrize(
    "bad",
    [
        {"batch_size": -1},
        {"tile_size": 32},
        {"tile_size": 128, "tile_overlap": 64},
        {"every_nth": 0},
        {"quality": 52},
        {"out_scale": 2, "target_height": 1080},
        {"resume": True},
        {"precision": "fp8"},
        {"segment_seconds": 0},
    ],
)
def test_invalid_options_are_rejected(bad):
    with pytest.raises((OptionsError, ValueError)):
        opts(**bad)


def test_audio_is_dropped_only_when_sync_is_impossible():
    assert opts().audio_disabled_reason is None
    assert opts(skip_frames=10, fps=12).audio_disabled_reason is None
    assert "every_nth" in opts(every_nth=2).audio_disabled_reason
    assert "no-audio" in opts(no_audio=True).audio_disabled_reason


def test_output_naming():
    assert output_name(Path("a/clip.mov"), 4) == "clip_x4.mp4"
    assert output_name(Path("clip.mp4"), 1.5) == "clip_x1.5.mp4"
    assert output_name(Path("clip.mp4"), 4, target_height=2160) == "clip_2160p.mp4"


def test_resolve_output(tmp_path: Path):
    src = tmp_path / "clip.mp4"
    assert resolve_output(UpscaleOptions(src, "m"), 2) == tmp_path / "clip_x2.mp4"
    out_dir = tmp_path / "out"
    assert resolve_output(UpscaleOptions(src, "m", output=out_dir), 2) == out_dir / "clip_x2.mp4"
    explicit = tmp_path / "final.mkv"
    assert resolve_output(UpscaleOptions(src, "m", output=explicit), 2) == explicit


def test_expand_inputs(tmp_path: Path):
    for name in ("b.mp4", "a.mkv", "notes.txt"):
        (tmp_path / name).touch()
    assert expand_inputs(tmp_path / "b.mp4") == [tmp_path / "b.mp4"]
    assert [p.name for p in expand_inputs(tmp_path)] == ["a.mkv", "b.mp4"]
    assert [p.name for p in expand_inputs(str(tmp_path / "*.mp4"))] == ["b.mp4"]
    with pytest.raises(OptionsError):
        expand_inputs(tmp_path / "missing.mp4")
    with pytest.raises(OptionsError):
        expand_inputs(str(tmp_path / "*.webm"))
