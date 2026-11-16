"""Engine tests run the real tiny model on CPU."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import torch

from vupscale.config import UpscaleOptions
from vupscale.engine import UpscaleEngine, blend_ramp, create_engine, load_model, tile_starts
from vupscale.models import ModelError
from vupscale.tuning import AdaptiveBatchController


@pytest.fixture(scope="module")
def descriptor(tiny_model: Path):
    return load_model(tiny_model)


@pytest.fixture(scope="module")
def frames() -> np.ndarray:
    """Smooth gradients plus noise: 3 frames of 96x128."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:96, 0:128]
    base = np.stack([xx * 2, yy * 2.5, (xx + yy)], axis=-1) % 256
    noise = rng.integers(-20, 20, size=(3, 96, 128, 3))
    return np.clip(base[None] + noise, 0, 255).astype(np.uint8)


def reference(descriptor, frames: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255
    y = descriptor(x)
    return (y.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).numpy()


def test_load_model_rejects_junk(tmp_path: Path):
    junk = tmp_path / "junk.pth"
    junk.write_bytes(b"not a checkpoint")
    with pytest.raises(ModelError, match="could not load"):
        load_model(junk)


def test_full_frame_matches_calling_the_model_directly(descriptor, frames):
    engine = UpscaleEngine(descriptor)
    out = engine.process_batch(frames)
    assert out.shape == (3, 192, 256, 3) and out.dtype == np.uint8
    assert np.array_equal(out, reference(descriptor, frames))


def test_input_is_validated(descriptor, frames):
    engine = UpscaleEngine(descriptor)
    with pytest.raises(ValueError):
        engine.process_batch(frames.astype(np.float32))
    with pytest.raises(ValueError):
        engine.process_batch(frames[0])


def test_tile_layout():
    assert tile_starts(100, 128, 16) == [0]
    assert tile_starts(128, 64, 16) == [0, 48, 64]
    starts = tile_starts(1080, 512, 32)
    assert starts[0] == 0 and starts[-1] + 512 == 1080
    assert all(b - a <= 512 - 32 for a, b in pairwise(starts))  # always overlapping
    ramp = blend_ramp(64, 16, True, False)
    assert 0 < ramp[0] < 0.01 and ramp[15] > 0.99 and ramp[-1] == 1


@pytest.mark.parametrize("tile_batch", [1, 4, 64])
def test_tiling_is_seamless(descriptor, frames, tile_batch: int):
    engine = UpscaleEngine(descriptor, tile_overlap=16)
    full = engine.process_batch(frames).astype(np.int16)
    tiled = engine.process_batch(frames, tile_size=64, tile_batch=tile_batch).astype(np.int16)
    assert tiled.shape == full.shape
    diff = np.abs(tiled - full)
    assert diff.max() <= 2, f"max diff {diff.max()}"
    assert diff.mean() < 0.05


def test_tile_larger_than_the_frame_is_a_plain_call(descriptor, frames):
    engine = UpscaleEngine(descriptor)
    assert np.array_equal(engine.process_batch(frames, tile_size=512), engine.process_batch(frames))


def test_output_resize(descriptor, frames):
    engine = UpscaleEngine(descriptor, out_size=(192, 144))  # x1.5 from an x2 model
    out = engine.process_batch(frames)
    assert out.shape == (3, 144, 192, 3)
    # Still the same picture: compare against the native result reduced the same way.
    native = torch.from_numpy(UpscaleEngine(descriptor).process_batch(frames))
    small = torch.nn.functional.interpolate(
        native.permute(0, 3, 1, 2).float(), size=(144, 192), mode="bicubic", antialias=True
    )
    expect = small.clamp(0, 255).round().permute(0, 2, 3, 1).numpy()
    assert np.abs(out.astype(np.int16) - expect).max() <= 2


def test_size_requirements_are_padded_and_cropped(descriptor, frames):
    from spandrel import SizeRequirements

    original = descriptor.size_requirements
    descriptor.size_requirements = SizeRequirements(minimum=16, multiple_of=16)
    try:
        odd = frames[:, :90, :101]  # not multiples of 16
        out = UpscaleEngine(descriptor).process_batch(odd)
        assert out.shape == (3, 180, 202, 3)
        ref = reference(descriptor, odd)
        assert np.array_equal(out, ref)
    finally:
        descriptor.size_requirements = original


def test_compile_failure_falls_back_to_eager(descriptor, frames, monkeypatch):
    engine = UpscaleEngine(descriptor)
    expected = engine.process_batch(frames)

    def boom(model, x):
        raise RuntimeError("inductor exploded")

    engine._compiled, engine._fixed_units = object(), 4
    real_call = engine._call_fn
    engine._call_fn = lambda model, x: (
        boom(model, x) if model is engine._compiled else real_call(model, x)
    )
    assert np.array_equal(engine.process_batch(frames), expected)
    assert engine._compiled is None and "eager" in engine.notes[0]


def test_fixed_shape_padding_when_compiled(descriptor, frames):
    engine = UpscaleEngine(descriptor)
    expected = engine.process_batch(frames)
    seen: list[int] = []
    real_call = engine._call_fn

    def spy(model, x):
        seen.append(x.shape[0])
        return real_call(engine._model, x)

    engine._compiled, engine._fixed_units, engine._call_fn = object(), 8, spy
    assert np.array_equal(engine.process_batch(frames), expected)
    assert seen == [8]  # 3 frames padded to the compiled batch of 8, extras discarded


def test_probe_runs_the_real_path(descriptor):
    probe = UpscaleEngine(descriptor).make_probe((48, 64))
    assert probe(2, None) == 0  # CPU reports no VRAM
    assert probe(4, 32) == 0


def test_create_engine_on_cpu(descriptor, tiny_model: Path, frames):
    options = UpscaleOptions(input=Path("clip.mp4"), model=str(tiny_model), target_height=144)
    engine, cfg = create_engine(
        options, descriptor, width=128, height=96, total_frames=48, model_path=tiny_model, gpu=None
    )
    assert cfg.device.type == "cpu" and cfg.precision_name == "fp32"
    assert (cfg.out_width, cfg.out_height) == (192, 144) and cfg.tile_size is None
    controller = AdaptiveBatchController.from_config(cfg, (96, 128))
    out = controller.run(engine.process_batch, frames)
    assert out.shape == (3, 144, 192, 3)


def test_engine_recovers_from_oom_through_the_controller(descriptor, frames, monkeypatch):
    engine = UpscaleEngine(descriptor, tile_overlap=16)
    expected = engine.process_batch(frames)
    real_forward = engine._forward

    def limited(x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] * x.shape[2] * x.shape[3] > 96 * 128:  # one full frame at most
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return real_forward(x)

    monkeypatch.setattr(engine, "_forward", limited)
    controller = AdaptiveBatchController(
        batch_size=3, tile_size=None, tile_batch=0, frame_hw=(96, 128)
    )
    out = controller.run(engine.process_batch, frames)
    assert controller.batch_size == 1 and len(controller.events) == 1
    assert np.array_equal(out, expected)
