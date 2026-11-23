"""Auto-tuning is tested on CPU with a simulated GPU: a probe that 'uses' memory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from vupscale import tuning
from vupscale.config import (
    Codec,
    CompileMode,
    OptionsError,
    Precision,
    UpscaleOptions,
    compute_input_size,
    compute_output_size,
)
from vupscale.hardware import GpuInfo
from vupscale.tuning import (
    AdaptiveBatchController,
    ModelTraits,
    TuningError,
    build_runtime_config,
    calibrate,
    select_dtype,
    select_encoder,
    should_compile,
)

GB = 2**30
T4 = GpuInfo("Tesla T4", (7, 5), 14.6, 14.5)
A100 = GpuInfo("NVIDIA A100-SXM4-40GB", (8, 0), 39.5, 39.0)
RTX4090 = GpuInfo("NVIDIA GeForce RTX 4090", (8, 9), 24.0, 23.5)


class FakeGpu:
    """Memory model: fixed weights plus a cost per megapixel of input per unit."""

    def __init__(self, capacity_gb: float, gb_per_megapixel: float, base_gb: float = 0.2):
        self.capacity = capacity_gb * GB
        self.per_mp = gb_per_megapixel * GB
        self.base = base_gb * GB
        self.frame_hw = (1080, 1920)
        self.calls: list[tuple[int, int | None]] = []

    def probe(self, units: int, tile: int | None) -> int:
        self.calls.append((units, tile))
        h, w = self.frame_hw if tile is None else (tile, tile)
        need = self.base + units * self.per_mp * (h * w / 1e6)
        if need > self.capacity:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return int(need)


def opts(**kw) -> UpscaleOptions:
    return UpscaleOptions(input=Path("clip.mp4"), model="m.pth", **kw)


def all_work(_: str) -> bool:
    return True


def none_work(encoder: str) -> bool:
    return not encoder.endswith(("_nvenc", "_vaapi"))


# ------------------------------------------------------------------ sizes


def test_sizes_are_even_and_keep_aspect():
    assert compute_input_size(1920, 1080) == (1920, 1080)
    assert compute_input_size(1920, 1080, pre_height=540) == (960, 540)
    assert compute_input_size(1920, 1080, pre_width=853) == (854, 480)
    assert compute_output_size(1920, 1080, 4) == (7680, 4320)
    assert compute_output_size(1920, 1080, 4, out_scale=2) == (3840, 2160)
    assert compute_output_size(1920, 1080, 4, target_height=1440) == (2560, 1440)
    assert compute_output_size(853, 481, 1)[0] % 2 == 0


# ------------------------------------------------------------------ precision


def test_precision_prefers_fp16_everywhere_the_model_allows():
    ok = ModelTraits(scale=4)
    assert select_dtype(Precision.auto, "cuda", (7, 5), ok)[0] is torch.float16
    assert select_dtype(Precision.auto, "cuda", (8, 9), ok)[0] is torch.float16
    assert select_dtype(Precision.auto, "cpu", (0, 0), ok)[0] is torch.float32


def test_precision_respects_models_without_fp16():
    no_half = ModelTraits(scale=4, supports_half=False)
    assert select_dtype(Precision.auto, "cuda", (8, 0), no_half)[0] is torch.bfloat16
    # A T4 has no native bf16, so fp32 is the safe choice there.
    dtype, notes = select_dtype(Precision.auto, "cuda", (7, 5), no_half)
    assert dtype is torch.float32 and notes
    dtype, notes = select_dtype(Precision.fp16, "cuda", (7, 5), no_half)
    assert dtype is torch.float16 and "black frames" in notes[0]


# ------------------------------------------------------------------ encoder


def test_encoder_auto_uses_nvenc_and_switches_to_hevc_above_4096():
    assert select_encoder(Codec.auto, 3840, 2160, 2, all_work).encoder == "h264_nvenc"
    assert select_encoder(Codec.auto, 7680, 4320, 2, all_work).encoder == "hevc_nvenc"
    assert select_encoder(Codec.auto, 2160, 4320, 2, all_work).encoder == "hevc_nvenc"


def test_encoder_falls_back_to_cpu_without_nvenc():
    hd = select_encoder(Codec.auto, 1920, 1080, 2, none_work)  # A100/H100 case
    assert (hd.encoder, hd.preset) == ("libx264", "faster")
    assert select_encoder(Codec.auto, 7680, 4320, 32, none_work).encoder == "libx265"
    forced = select_encoder(Codec.hevc_nvenc, 1920, 1080, 32, none_work)
    assert forced.encoder == "libx265" and forced.preset == "medium" and forced.notes


def test_encoder_explicit_choices():
    assert select_encoder(Codec.h264, 1920, 1080, 8, all_work).encoder == "libx264"
    assert select_encoder(Codec.av1, 1920, 1080, 8, all_work).encoder == "libsvtav1"
    assert select_encoder(Codec.av1_nvenc, 1920, 1080, 8, all_work).encoder == "av1_nvenc"
    too_wide = select_encoder(Codec.h264_nvenc, 7680, 4320, 8, all_work)
    assert too_wide.encoder == "hevc_nvenc" and "4096" in too_wide.notes[0]
    assert select_encoder(Codec.auto, 15360, 8640, 8, all_work).encoder == "libx265"


def test_compile_only_when_it_pays_off():
    assert should_compile(CompileMode.auto, "cuda", (8, 9), 5000)
    assert not should_compile(CompileMode.auto, "cuda", (8, 9), 300)  # too short to amortise
    assert not should_compile(CompileMode.auto, "cuda", (7, 5), 5000)  # T4
    assert not should_compile(CompileMode.auto, "cpu", (0, 0), 5000)
    assert should_compile(CompileMode.on, "cuda", (7, 5), 1)
    assert not should_compile(CompileMode.off, "cuda", (8, 9), 5000)


# ------------------------------------------------------------------ calibration


def test_calibration_keeps_full_frames_when_they_fit():
    gpu = FakeGpu(capacity_gb=24, gb_per_megapixel=1.0)  # 1080p frame ~2.1 GB
    found = calibrate(gpu.probe, frame_hw=gpu.frame_hw, budget_bytes=24 * GB * 0.75)
    assert found.tile_size is None
    assert found.units == 8  # 0.2 + 8 * 2.07 = 16.8 GB <= 18 GB budget, 9 would not fit
    assert found.peak_bytes <= 24 * GB * 0.75


def test_calibration_falls_back_to_tiles_on_a_small_gpu():
    gpu = FakeGpu(capacity_gb=14.5, gb_per_megapixel=8.0)  # 1080p frame ~16.8 GB: OOM
    found = calibrate(gpu.probe, frame_hw=gpu.frame_hw, budget_bytes=14.5 * GB * 0.75)
    assert found.tile_size == 768  # largest tile that fits (4.9 GB each)
    assert found.units == 2
    assert gpu.calls[0] == (1, None)  # the full frame was tried first


def test_calibration_respects_overrides_and_limits():
    gpu = FakeGpu(capacity_gb=24, gb_per_megapixel=1.0)
    budget = 24 * GB * 0.75
    assert (
        calibrate(gpu.probe, frame_hw=gpu.frame_hw, budget_bytes=budget, tile_option=512).tile_size
        == 512
    )
    assert (
        calibrate(gpu.probe, frame_hw=gpu.frame_hw, budget_bytes=budget, tile_option=-1).tile_size
        is None
    )
    assert calibrate(gpu.probe, frame_hw=gpu.frame_hw, budget_bytes=budget, max_frames=3).units == 3
    assert (
        calibrate(gpu.probe, frame_hw=gpu.frame_hw, budget_bytes=budget, fixed_frames=2).units == 2
    )
    # A tile at least as big as the frame is just the full frame.
    assert tuning.tile_candidates(512, (240, 320)) == [None]


def test_calibration_fails_clearly_when_nothing_fits():
    gpu = FakeGpu(capacity_gb=1, gb_per_megapixel=100.0)
    with pytest.raises(TuningError, match="lighter model"):
        calibrate(gpu.probe, frame_hw=gpu.frame_hw, budget_bytes=0.75 * GB)


def test_calibration_does_not_swallow_other_errors():
    def broken(units: int, tile: int | None) -> int:
        raise ValueError("shape mismatch")

    with pytest.raises(ValueError):
        calibrate(broken, frame_hw=(1080, 1920), budget_bytes=GB)


def test_calibration_cache_is_reused_and_revalidated(tmp_path: Path):
    cache = tmp_path / "calibration.json"
    kwargs = dict(frame_hw=(1080, 1920), budget_bytes=18 * GB, cache_key="k", cache_file=cache)
    gpu = FakeGpu(capacity_gb=24, gb_per_megapixel=1.0)
    first = calibrate(gpu.probe, **kwargs)
    assert not first.from_cache and cache.is_file()

    gpu.calls.clear()
    second = calibrate(gpu.probe, **kwargs)
    assert second.from_cache and (second.tile_size, second.units) == (None, 8)
    assert gpu.calls == [(8, None), (8, None)]  # only the steady-state confirmation

    # Less free VRAM today: the cached answer no longer fits and is recomputed.
    tighter = FakeGpu(capacity_gb=8, gb_per_megapixel=1.0)
    third = calibrate(tighter.probe, **{**kwargs, "budget_bytes": 6 * GB})
    assert not third.from_cache and third.units == 2

    cache.write_text("not json")
    assert calibrate(gpu.probe, **kwargs).units == 8  # a corrupt cache is ignored


# ------------------------------------------------------------------ runtime config


def config(gpu, fake: FakeGpu | None = None, works=all_work, ram_gb=11.0, cpus=2, **kw):
    size = kw.pop("size", (1920, 1080))
    traits = kw.pop("traits", ModelTraits(scale=4))
    frames = kw.pop("total_frames", 3000)
    if fake is not None:
        fake.frame_hw = (size[1], size[0])
    return build_runtime_config(
        opts(**kw),
        traits,
        width=size[0],
        height=size[1],
        total_frames=frames,
        make_probe=(lambda device, dtype: fake.probe) if fake else None,
        gpu=gpu,
        ram_free_bytes=ram_gb * GB,
        cpu_count=cpus,
        works=works,
        use_cache=False,
    )


def test_config_for_colab_t4_1080p_x4():
    cfg = config(T4, FakeGpu(14.5, 8.0))
    assert cfg.device.type == "cuda" and cfg.dtype is torch.float16
    assert cfg.tile_size == 768 and cfg.tile_batch == 2 and cfg.batch_size == 1
    assert (cfg.out_width, cfg.out_height) == (7680, 4320)
    assert cfg.video_codec == "hevc_nvenc"  # 8K output is beyond H.264 NVENC
    assert not cfg.compile_model and cfg.calibrated
    # 8K RGB frames are ~100 MB each: the queues must stay tiny on 11 GB of RAM.
    assert 1 <= cfg.queue_depth <= 8
    in_flight = 2 * cfg.queue_depth * cfg.batch_size * 3 * (1920 * 1080 + 7680 * 4320)
    assert in_flight <= 0.15 * 11 * GB


def test_config_for_t4_with_a_4k_target_uses_h264_nvenc():
    cfg = config(T4, FakeGpu(14.5, 8.0), target_height=2160)
    assert (cfg.out_width, cfg.out_height) == (3840, 2160)
    assert cfg.video_codec == "h264_nvenc" and cfg.encoder_preset == "p5"


def test_config_for_rtx4090_batches_full_frames_and_compiles():
    cfg = config(RTX4090, FakeGpu(23.5, 1.0), size=(1280, 720), ram_gb=60, cpus=16)
    assert cfg.tile_size is None and cfg.tile_batch == 0
    assert cfg.batch_size == 16 and cfg.compile_model
    assert cfg.queue_depth >= 2


def test_config_for_a100_encodes_on_cpu():
    cfg = config(A100, FakeGpu(39, 1.0), works=none_work, ram_gb=80, cpus=8, target_height=2160)
    assert cfg.video_codec == "libx264" and cfg.encoder_preset == "medium"
    assert cfg.tile_size is None and cfg.batch_size > 1


def test_host_ram_caps_the_batch_size():
    # Plenty of VRAM, but 8K frames on a 12 GB host: RAM is the limit, not the GPU.
    cfg = config(A100, FakeGpu(39, 0.1), ram_gb=4.0)
    assert cfg.batch_size <= tuning.max_batch_for_ram(3 * (1920 * 1080 + 7680 * 4320), 4.0 * GB)
    assert cfg.batch_size < 16


def test_manual_overrides_win():
    cfg = config(T4, FakeGpu(14.5, 0.5), batch_size=3, tile_size=-1, precision="fp32",
                 codec="hevc", compile="on")  # fmt: skip
    assert cfg.batch_size == 3 and cfg.tile_size is None
    assert cfg.dtype is torch.float32 and cfg.video_codec == "libx265" and cfg.compile_model


def test_config_without_probe_uses_conservative_rules():
    cfg = config(T4)  # e.g. --dry-run
    assert cfg.tile_size == 512 and cfg.batch_size == 1 and not cfg.calibrated
    assert any("not calibrated" in n for n in cfg.notes)
    assert config(RTX4090, size=(1280, 720), ram_gb=60).tile_size is None


def test_config_on_cpu():
    cfg = config(None, works=none_work)
    assert cfg.device.type == "cpu" and cfg.dtype is torch.float32
    assert cfg.tile_size == 512 and cfg.batch_size == 1 and not cfg.compile_model
    assert cfg.gpu_name == "CPU"
    assert config(None, works=none_work, size=(320, 240)).tile_size is None
    assert config(T4, device="cpu").device.type == "cpu"
    with pytest.raises(OptionsError, match="no CUDA GPU"):
        config(None, device="cuda")


def test_discouraged_tiling_is_reported():
    traits = ModelTraits(scale=4, tiling_recommended=False)
    cfg = config(T4, FakeGpu(14.5, 8.0), traits=traits)
    assert cfg.tile_size is not None and any("discourages tiling" in n for n in cfg.notes)


def test_traits_from_a_real_spandrel_descriptor(tiny_model: Path):
    from spandrel import ModelLoader

    traits = ModelTraits.from_descriptor(ModelLoader().load_from_file(tiny_model))
    assert traits == ModelTraits(scale=2, supports_half=True, supports_bfloat16=True)


# ------------------------------------------------------------------ OOM recovery


class FakeEngine:
    """Upscales by repeating pixels; runs out of memory above a pixel budget per call."""

    def __init__(self, capacity_px: int, scale: int = 2):
        self.capacity_px = capacity_px
        self.scale = scale
        self.calls: list[tuple[int, int | None, int]] = []

    def __call__(self, frames: np.ndarray, tile: int | None, tile_batch: int) -> np.ndarray:
        self.calls.append((len(frames), tile, tile_batch))
        n, h, w, _ = frames.shape
        load = n * h * w if tile is None else tile_batch * tile * tile
        if load > self.capacity_px:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return frames.repeat(self.scale, axis=1).repeat(self.scale, axis=2)


def numbered_frames(n: int, h: int = 1080, w: int = 1920) -> np.ndarray:
    frames = np.zeros((n, h, w, 3), dtype=np.uint8)
    frames[:, 0, 0, 0] = np.arange(n)
    return frames


def test_controller_passes_through_when_memory_is_fine():
    engine = FakeEngine(capacity_px=10**12)
    ctl = AdaptiveBatchController(4, None, 0, (1080, 1920))
    out = ctl.run(engine, numbered_frames(4))
    assert out.shape == (4, 2160, 3840, 3) and not ctl.events
    assert engine.calls == [(4, None, 0)]


def test_controller_halves_the_batch_and_keeps_every_frame_in_order():
    engine = FakeEngine(capacity_px=2 * 1080 * 1920)  # two full frames fit, not eight
    ctl = AdaptiveBatchController(8, None, 0, (1080, 1920))
    out = ctl.run(engine, numbered_frames(8))
    assert out.shape[0] == 8 and list(out[:, 0, 0, 0]) == list(range(8))
    assert ctl.batch_size == 2 and len(ctl.events) == 2
    # The lower setting sticks: the next batch does not hit OOM again.
    engine.calls.clear()
    ctl.run(engine, numbered_frames(8))
    assert all(n == 2 for n, _, _ in engine.calls) and len(ctl.events) == 2


def test_controller_enables_then_shrinks_tiles():
    engine = FakeEngine(capacity_px=600 * 600)  # not even one frame; a 512 tile fits
    seen: list[str] = []
    ctl = AdaptiveBatchController(2, None, 0, (1080, 1920), on_event=seen.append)
    out = ctl.run(engine, numbered_frames(3))
    assert list(out[:, 0, 0, 0]) == [0, 1, 2]
    assert (ctl.batch_size, ctl.tile_size, ctl.tile_batch) == (1, 512, 1)
    assert seen == ctl.events and "tile size set to 768" in seen[1]


def test_controller_reduces_tiles_per_call_first():
    engine = FakeEngine(capacity_px=2 * 512 * 512)
    ctl = AdaptiveBatchController(1, 512, 8, (1080, 1920))
    ctl.run(engine, numbered_frames(2))
    assert (ctl.tile_size, ctl.tile_batch) == (512, 2)


def test_controller_gives_up_with_a_clear_error():
    ctl = AdaptiveBatchController(1, 256, 1, (1080, 1920))
    with pytest.raises(TuningError, match="smallest tiles"):
        ctl.run(FakeEngine(capacity_px=10), numbered_frames(1))


def test_controller_lets_real_bugs_through():
    def broken(frames, tile, tile_batch):
        raise ValueError("bad shape")

    with pytest.raises(ValueError):
        AdaptiveBatchController(2, None, 0, (8, 8)).run(broken, numbered_frames(2, 8, 8))


def test_oom_detection():
    assert tuning.is_oom(torch.cuda.OutOfMemoryError("x"))
    assert tuning.is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert tuning.is_oom(RuntimeError("cuDNN error: CUDNN_STATUS_ALLOC_FAILED"))
    assert tuning.is_oom(MemoryError())
    assert not tuning.is_oom(RuntimeError("size mismatch"))


def test_calibration_requires_the_winner_to_fit_twice_in_a_row():
    """Seen on an RTX 4090: a batch fits on its first run and never again."""
    calls: list[int] = []

    def first_run_only(units: int, tile: int | None) -> int:
        calls.append(units)
        if units >= 4 and calls.count(units) > 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return units * GB

    kwargs = dict(frame_hw=(1080, 1920), budget_bytes=4 * GB, max_frames=4)
    found = calibrate(first_run_only, **kwargs)
    assert found.units == 2 and calls == [1, 2, 4, 4, 2, 2]


def test_stale_cache_entry_that_only_fits_once_is_recalibrated(tmp_path: Path):
    cache = tmp_path / "calibration.json"
    cache.write_text('{"k": {"tile": null, "units": 4}}')
    calls: list[int] = []

    def first_run_only(units: int, tile: int | None) -> int:
        calls.append(units)
        if units >= 4 and calls.count(units) > 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return units * GB

    found = calibrate(
        first_run_only, frame_hw=(1080, 1920), budget_bytes=4 * GB, max_frames=4,
        cache_key="k", cache_file=cache,
    )  # fmt: skip
    assert found.units == 2 and not found.from_cache
    assert '"units": 2' in cache.read_text()


def test_controller_copies_results_of_split_batches():
    """The engine may reuse its output buffer, so split batches must not alias it."""
    shared = np.zeros((1, 4, 4, 3), dtype=np.uint8)

    def reusing_engine(frames, tile, tile_batch):
        if len(frames) > 1:
            raise MemoryError
        shared[:] = frames[0, 0, 0, 0]
        return shared

    frames = numbered_frames(3, 4, 4)
    out = AdaptiveBatchController(2, None, 0, (4, 4)).run(reusing_engine, frames)
    assert list(out[:, 0, 0, 0]) == [0, 1, 2]
    # A batch that fits in one call is passed through without a copy.
    assert AdaptiveBatchController(1, None, 0, (4, 4)).run(reusing_engine, frames[:1]) is shared
