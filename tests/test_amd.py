"""AMD (ROCm + VAAPI) support, tested without AMD hardware.

ROCm exposes the GPU through ``torch.cuda``, so the engine is shared with NVIDIA. What
differs, and is tested here, is every decision that reads the GPU's identity, plus the
VAAPI encoding path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vupscale import hardware
from vupscale.config import Codec, CompileMode, Precision, UpscaleOptions
from vupscale.engine import UpscaleEngine, backend_defaults
from vupscale.hardware import GpuInfo, SystemReport, encoder_upload_filter, vaapi_device
from vupscale.tuning import (
    ModelTraits,
    build_runtime_config,
    select_dtype,
    select_encoder,
    should_compile,
)
from vupscale.video_io import VideoStreamWriter, encoder_args

GB = 2**30
RX9060XT = GpuInfo("AMD Radeon RX 9060 XT", (12, 0), 15.9, 15.5, "amd", "gfx1200")
RX7900XTX = GpuInfo("AMD Radeon RX 7900 XTX", (11, 0), 24.0, 23.6, "amd", "gfx1100")
RX6800 = GpuInfo("AMD Radeon RX 6800", (10, 3), 16.0, 15.7, "amd", "gfx1030")
RX6600 = GpuInfo("AMD Radeon RX 6600", (10, 3), 8.0, 7.8, "amd", "gfx1032")
RTX5090 = GpuInfo("NVIDIA GeForce RTX 5090", (12, 0), 31.4, 30.9)


def vaapi_only(encoder: str) -> bool:
    return not encoder.endswith("_nvenc")


# ------------------------------------------------------------------ identity


def test_amd_generations_and_labels():
    assert (RX9060XT.generation, RX9060XT.label) == ("RDNA 4", "gfx1200")
    assert RX7900XTX.generation == "RDNA 3" and RX6800.generation == "RDNA 2"
    assert GpuInfo("MI300X", (9, 4), 192, 190, "amd", "gfx942").generation == "CDNA 3"
    # Same "capability" (12, 0) as an RTX 5090, but a different GPU entirely.
    assert (RTX5090.generation, RTX5090.label) == ("Blackwell", "sm_120")
    assert not RTX5090.is_amd and RX9060XT.is_amd


def test_native_bf16_is_not_read_from_the_capability_on_amd():
    assert RX9060XT.native_bf16 and RX7900XTX.native_bf16
    assert not RX6800.native_bf16  # (10, 3) would pass NVIDIA's major >= 8 rule
    assert RTX5090.native_bf16
    assert not GpuInfo("Tesla T4", (7, 5), 15, 15).native_bf16


def test_precision_on_amd():
    ok = ModelTraits(scale=4)
    no_half = ModelTraits(scale=4, supports_half=False)
    assert select_dtype(Precision.auto, "cuda", (10, 3), ok, native_bf16=False)[0] is torch.float16
    assert (
        select_dtype(Precision.auto, "cuda", (12, 0), no_half, native_bf16=True)[0]
        is torch.bfloat16
    )
    dtype, notes = select_dtype(Precision.auto, "cuda", (10, 3), no_half, native_bf16=False)
    assert dtype is torch.float32 and notes  # RDNA 2 has no bf16 hardware


def test_compile_stays_opt_in_on_amd():
    assert not should_compile(CompileMode.auto, "cuda", (12, 0), 50_000, vendor="amd")
    assert should_compile(CompileMode.auto, "cuda", (12, 0), 50_000, vendor="nvidia")
    assert should_compile(CompileMode.on, "cuda", (12, 0), 10, vendor="amd")


# ------------------------------------------------------------------ encoders


def test_auto_prefers_nvenc_then_vaapi_then_cpu():
    assert select_encoder(Codec.auto, 2560, 1440, 8, lambda e: True).encoder == "h264_nvenc"
    choice = select_encoder(Codec.auto, 2560, 1440, 8, vaapi_only)
    assert (choice.encoder, choice.preset, choice.is_hardware) == ("h264_vaapi", "", True)
    assert select_encoder(Codec.auto, 7680, 4320, 8, vaapi_only).encoder == "hevc_vaapi"
    cpu = select_encoder(Codec.auto, 2560, 1440, 8, lambda e: e.startswith("lib"))
    assert cpu.encoder == "libx264" and not cpu.is_hardware


def test_explicit_vaapi_codecs():
    assert select_encoder(Codec.av1_vaapi, 2560, 1440, 8, vaapi_only).encoder == "av1_vaapi"
    wide = select_encoder(Codec.h264_vaapi, 7680, 4320, 8, vaapi_only)
    assert wide.encoder == "hevc_vaapi" and "h264_vaapi is limited to 4096" in wide.notes[0]
    missing = select_encoder(Codec.hevc_vaapi, 2560, 1440, 8, lambda e: e.startswith("lib"))
    assert missing.encoder == "libx265" and "hevc_vaapi does not work" in missing.notes[0]


def test_vaapi_rate_control():
    assert encoder_args("hevc_vaapi", "", 24) == [
        "-c:v", "hevc_vaapi", "-rc_mode", "CQP", "-global_quality", "24", "-tag:v", "hvc1",
    ]  # fmt: skip
    av1 = encoder_args("av1_vaapi", "", 51)
    assert av1[av1.index("-global_quality") + 1] == "255"  # AV1 quantisers run 0-255
    assert "-preset" not in encoder_args("h264_vaapi", "", 20)


def test_vaapi_frames_are_uploaded_to_the_gpu():
    assert encoder_upload_filter("h264_vaapi") == "format=nv12,hwupload"
    assert encoder_upload_filter("hevc_vaapi", "yuv420p10le") == "format=p010,hwupload"
    assert encoder_upload_filter("libx264", "yuv420p") == "format=yuv420p"


def test_writer_command_for_vaapi(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VUPSCALE_VAAPI_DEVICE", "/dev/dri/renderD129")
    cmd = VideoStreamWriter(tmp_path / "o.mp4", 2560, 1440, 30, encoder="hevc_vaapi").command()
    assert cmd[cmd.index("-vaapi_device") + 1] == "/dev/dri/renderD129"
    assert cmd.index("-vaapi_device") < cmd.index("-i")  # a global option, before inputs
    chain = cmd[cmd.index("-vf") + 1]
    assert chain.endswith("out_color_matrix=bt709:out_range=tv,format=nv12,hwupload")
    assert "-colorspace" in cmd and "-pix_fmt" in cmd  # input format only
    cpu = VideoStreamWriter(tmp_path / "o.mp4", 64, 48, 30, encoder="libx264").command()
    assert "-vaapi_device" not in cpu


def test_vaapi_without_a_render_node_never_works(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("VUPSCALE_VAAPI_DEVICE", raising=False)
    monkeypatch.setattr(hardware, "vaapi_device", lambda: None)
    hardware.encoder_works.cache_clear()
    assert not hardware.encoder_works("h264_vaapi")
    hardware.encoder_works.cache_clear()


# ------------------------------------------------------------------ render node


def fake_drm(root: Path, nodes: dict[str, str]) -> Path:
    for name, vendor in nodes.items():
        device = root / name / "device"
        device.mkdir(parents=True)
        (device / "vendor").write_text(vendor + "\n")
    return root


def test_render_node_prefers_amd(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("VUPSCALE_VAAPI_DEVICE", raising=False)
    sysfs = fake_drm(tmp_path, {"renderD128": "0x8086", "renderD129": "0x1002"})
    assert vaapi_device(sysfs) == "/dev/dri/renderD129"  # Intel iGPU + Radeon
    other = fake_drm(tmp_path / "b", {"renderD128": "0x10de", "renderD130": "0x8086"})
    assert vaapi_device(other) == "/dev/dri/renderD130"
    assert vaapi_device(tmp_path / "empty") is None
    monkeypatch.setenv("VUPSCALE_VAAPI_DEVICE", "/dev/dri/renderD200")
    assert vaapi_device(sysfs) == "/dev/dri/renderD200"


# ------------------------------------------------------------------ runtime config


class FakeGpu:
    def __init__(self, capacity_gb: float):
        self.capacity = capacity_gb * GB

    def probe(self, units: int, tile: int | None) -> int:
        need = 0.2 * GB + units * 1.0 * GB
        if need > self.capacity:
            raise torch.OutOfMemoryError("HIP out of memory. Tried to allocate 2.00 GiB")
        return int(need)


def test_runtime_config_for_rx_9060_xt():
    cfg = build_runtime_config(
        UpscaleOptions(input=Path("clip.mp4"), model="m.pth", target_height=1440),
        ModelTraits(scale=2),
        width=1280,
        height=720,
        total_frames=71_259,
        make_probe=lambda device, dtype: FakeGpu(15.5).probe,
        gpu=RX9060XT,
        ram_free_bytes=24 * GB,
        cpu_count=12,
        works=vaapi_only,
        use_cache=False,
    )
    assert cfg.device.type == "cuda" and cfg.dtype is torch.float16  # ROCm uses "cuda"
    assert cfg.calibrated and cfg.batch_size > 1  # "HIP out of memory" is recognised
    assert cfg.video_codec == "h264_vaapi" and not cfg.compile_model
    assert (cfg.gpu_vendor, cfg.gpu_label) == ("amd", "gfx1200")


# ------------------------------------------------------------------ engine


def test_backend_defaults(monkeypatch):
    monkeypatch.delenv("VUPSCALE_CHANNELS_LAST", raising=False)
    monkeypatch.delenv("VUPSCALE_CUDNN_BENCHMARK", raising=False)
    assert backend_defaults(torch.device("cpu")) == (True, False)
    monkeypatch.setattr(torch.version, "hip", "7.2.1", raising=False)
    assert backend_defaults(torch.device("cuda")) == (False, False)  # ROCm
    monkeypatch.setenv("VUPSCALE_CHANNELS_LAST", "1")
    monkeypatch.setenv("VUPSCALE_CUDNN_BENCHMARK", "yes")
    assert backend_defaults(torch.device("cuda")) == (True, True)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.delenv("VUPSCALE_CHANNELS_LAST")
    monkeypatch.delenv("VUPSCALE_CUDNN_BENCHMARK")
    assert backend_defaults(torch.device("cuda")) == (True, True)  # NVIDIA


def test_both_layouts_give_the_same_pixels(tiny_model: Path, monkeypatch):
    import numpy as np

    from vupscale.engine import load_model

    frames = np.random.default_rng(1).integers(0, 256, (2, 40, 56, 3), dtype=np.uint8)
    outputs = []
    for flag in ("1", "0"):
        monkeypatch.setenv("VUPSCALE_CHANNELS_LAST", flag)
        engine = UpscaleEngine(load_model(tiny_model))
        assert engine.channels_last is (flag == "1")
        outputs.append(engine.process_batch(frames, tile_size=32, tile_batch=3))
    assert np.abs(outputs[0].astype(int) - outputs[1].astype(int)).max() <= 1


# ------------------------------------------------------------------ doctor


def report(**overrides) -> SystemReport:
    base = dict(
        environment="local", python="3.12", torch="2.9.1+rocm7.2.1", cuda_build=None,
        gpu=RX9060XT, cpu_count=12, ram_total_gb=32, ram_free_gb=24, ffmpeg="6.1.1",
        ffprobe=True, encoders={"libx264": True, "h264_vaapi": True}, rocm_build="7.2.1",
        vaapi="/dev/dri/renderD128", amd_device_present=True, amd_device_usable=True,
    )  # fmt: skip
    base.update(overrides)
    return SystemReport(**base)


def test_doctor_is_happy_with_a_working_amd_setup():
    assert report().problems == [] and report().warnings == []


def test_doctor_explains_common_amd_setups():
    cuda_torch = report(gpu=None, rocm_build=None, cuda_build="13.2")
    assert "no ROCm support" in cuda_torch.warnings[0]
    no_groups = report(gpu=None, amd_device_usable=False)
    assert "'render' and 'video' groups" in no_groups.warnings[0]
    no_vaapi = report(encoders={"libx264": True})
    assert "no working VAAPI encoder" in no_vaapi.warnings[0]


def test_doctor_points_rdna2_cards_at_the_override():
    broken = report(gpu=RX6600, gpu_error="RuntimeError: HIP error: invalid device function")
    assert "HSA_OVERRIDE_GFX_VERSION=10.3.0" in broken.problems[0]
    assert any("gfx1032" in note for note in broken.warnings)
    assert report(gpu=RX6600, hsa_override="10.3.0").warnings == []
    assert report(gpu=RX6800).warnings == []  # gfx1030 is a build target


def test_doctor_on_nvidia_is_unchanged():
    nvidia = report(
        gpu=GpuInfo("Tesla T4", (7, 5), 14.6, 14.5), rocm_build=None, cuda_build="13.0",
        encoders={"libx264": True, "h264_nvenc": True}, vaapi=None, amd_device_present=False,
    )  # fmt: skip
    assert nvidia.problems == [] and nvidia.warnings == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_compute_check_on_a_real_gpu():
    assert hardware.gpu_compute_check() is None
