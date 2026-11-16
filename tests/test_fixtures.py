"""The fixtures feed every later stage, so make sure they are what they claim to be."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from conftest import requires_ffmpeg
from vupscale import hardware


@requires_ffmpeg
def test_sample_video_has_48_frames_and_audio(sample_video: Path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-show_streams", "-of", "json",
         str(sample_video)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    streams = {s["codec_type"]: s for s in json.loads(out)["streams"]}
    assert int(streams["video"]["nb_read_packets"]) == 48
    assert (streams["video"]["width"], streams["video"]["height"]) == (320, 240)
    assert "audio" in streams


def test_tiny_model_is_a_valid_sr_model(tiny_model: Path):
    import torch
    from spandrel import ImageModelDescriptor, ModelLoader

    descriptor = ModelLoader().load_from_file(tiny_model)
    assert isinstance(descriptor, ImageModelDescriptor)
    assert descriptor.scale == 2 and descriptor.purpose == "SR"
    out = descriptor(torch.rand(1, 3, 16, 24))
    assert out.shape == (1, 3, 32, 48)


def test_environment_detection(monkeypatch):
    for key in ("COLAB_RELEASE_TAG", "COLAB_GPU"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RUNPOD_POD_ID", "abc")
    assert hardware.detect_environment() == "runpod"
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-colab")
    assert hardware.detect_environment() == "colab"


@requires_ffmpeg
def test_encoder_probe_really_encodes():
    hardware.encoder_works.cache_clear()
    assert hardware.encoder_works("libx264") is True
    assert hardware.encoder_works("definitely_not_an_encoder") is False
