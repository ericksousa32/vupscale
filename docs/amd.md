# AMD GPUs

vupscale runs on AMD Radeon GPUs under Linux, through the ROCm build of PyTorch for
inference and VAAPI for hardware encoding.

> **Status:** tested on an RX 9060 XT (RDNA 4) under Linux: inference, calibration,
> `torch.compile`, VAAPI encoding and complete renders. RDNA 2 and RDNA 3 are covered by
> tests with simulated GPUs but have not been run on real cards.

## What is supported

| Generation | Examples | Inference | Hardware encoding |
|---|---|---|---|
| RDNA 4 | RX 9060 XT, RX 9070 XT | ROCm, officially supported by AMD | H.264, HEVC, AV1 |
| RDNA 3 | RX 7900 XTX, RX 7800 XT, RX 7600 | ROCm, officially supported by AMD | H.264, HEVC, AV1 |
| RDNA 2 | RX 6800 XT, RX 6700 XT, RX 6600 | ROCm, community-supported (see below) | H.264, HEVC |

RDNA 2 cards are not on AMD's PyTorch support list. The RX 6800 and RX 6900 (`gfx1030`)
usually work with the standard ROCm build of PyTorch. Other RDNA 2 cards (`gfx1031` to
`gfx1036`) need `HSA_OVERRIDE_GFX_VERSION=10.3.0` to run as `gfx1030`; `vupscale doctor`
says so when it applies.

Windows is not supported: hardware encoding there goes through AMD's AMF, which
vupscale does not implement. WSL2 can run inference, but has no VAAPI encoder, so
encoding falls back to the CPU.

## Install

### 1. Driver and permissions

The `amdgpu` kernel driver that ships with current Linux distributions is enough for
both ROCm and VAAPI. For ROCm itself, follow AMD's
[ROCm on Radeon](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/) guide for
your distribution.

Your user needs access to the GPU device files:

```bash
sudo usermod -aG render,video $USER     # then log out and back in
```

### 2. PyTorch for ROCm

Install the ROCm build of PyTorch before vupscale. A regular `pip install torch`
installs the CUDA build, which cannot see an AMD GPU.

Use the command from AMD's
[Install PyTorch for ROCm](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html)
page (AMD's own wheels, recommended for Radeon) or from the
[PyTorch selector](https://pytorch.org/get-started/locally/) with "ROCm" as the
platform. Then check:

```bash
python -c "import torch; print(torch.version.hip, torch.cuda.get_device_name(0))"
```

It should print a ROCm version and the name of your card. ROCm GPUs appear through
`torch.cuda` in PyTorch; that is expected.

### 3. FFmpeg with VAAPI

| Distribution | Packages |
|---|---|
| Ubuntu, Debian | `sudo apt install ffmpeg mesa-va-drivers vainfo` |
| Arch | `sudo pacman -S ffmpeg libva-mesa-driver libva-utils` |
| Fedora | `ffmpeg` and `mesa-va-drivers-freeworld` from RPM Fusion, plus `libva-utils` |

Fedora's own Mesa package has H.264 and HEVC VAAPI support removed for patent reasons;
the RPM Fusion "freeworld" driver restores it. Check with `vainfo`: the encode profiles
appear as `VAEntrypointEncSlice` lines.

### 4. vupscale

```bash
pip install "git+https://github.com/ericksousa32/vupscale"
vupscale doctor
```

A working setup looks like this:

```
│ torch          │ 2.9.1+rocm7.2.1 (ROCm 7.2.1)                 │
│ gpu            │ AMD Radeon RX 9060 XT (RDNA 4, gfx1200)      │
│ gpu compute    │ ok                                            │
│ vaapi (tested) │ h264: yes  hevc: yes  av1: yes  /dev/dri/renderD128 │
│ codec auto     │ h264_vaapi (up to 4096 px), hevc_vaapi (larger) │
```

`gpu compute` runs a small convolution on the card. It catches the case where PyTorch
can see the GPU but has no kernels for it.

## What is different on AMD

The command line and the Python API are the same. These defaults differ:

| Setting | NVIDIA | AMD | Why |
|---|---|---|---|
| Precision | fp16 | fp16 | Same rule. bf16 is the fallback only on RDNA 3 and newer, which have bf16 hardware |
| `--compile auto` | On for long renders on Ampere and newer | On for long renders on RDNA 4; off on RDNA 2 and 3 | Measured +32% on RDNA 4; RDNA 2 and 3 not measured, `--compile on` enables it |
| Memory layout | `channels_last` | Default (NCHW) | Measured no gain on RDNA 4 |
| Kernel autotuning | cuDNN benchmark on | MIOpen exhaustive search off | Measured no gain on RDNA 4, and exhaustive search runs for every new shape |
| `--codec auto` | NVENC | VAAPI | Hardware encoder of each vendor on Linux |
| `--quality` | NVENC constant quality | VAAPI constant QP | See [Video options](video.md#quality) |

Memory calibration, tiling, out-of-memory recovery, checkpoints and everything else
work the same way.

MIOpen compiles its kernels on first use and caches them (in `~/.cache/miopen` by
default), so the first render or benchmark on a new card, or after a ROCm update, is
slower than the following ones.

## Measured on an RX 9060 XT

Ubuntu 24.04 container on a Fedora host, PyTorch 2.15 for ROCm, FFmpeg 6.1,
`realesr-animevideov3`, November 2026.

| Test | Speed | Model time |
|---|---|---|
| `bench`, 1280x720 to 5120x2880, defaults | 17.6 fps | 45.7 ms/frame |
| same, `VUPSCALE_CHANNELS_LAST=1` | 17.3 fps | 46.8 ms/frame |
| same, `VUPSCALE_CUDNN_BENCHMARK=1` | 17.4 fps | 46.3 ms/frame |
| same, `--compile on` | 23.3 fps | 31.4 ms/frame |
| Complete render, 770x576 to 1926x1440, `h264_vaapi`, with audio | 21.4 fps | |

The complete render was limited by the GPU, with `torch.compile` off (300 frames is too
short for `auto` to compile). The same render on a Tesla T4 ran at 15.0 fps.

## Tuning on your card

Two environment variables switch the AMD-specific defaults, so they can be measured
with `vupscale bench`:

| Variable | Default on AMD | Effect when set to `1` |
|---|---|---|
| `VUPSCALE_CHANNELS_LAST` | `0` | Use the NHWC memory layout |
| `VUPSCALE_CUDNN_BENCHMARK` | `0` | Let MIOpen search exhaustively for the fastest kernels |

```bash
vupscale bench -m realesr-animevideov3 --size 1280x720
VUPSCALE_CHANNELS_LAST=1 vupscale bench -m realesr-animevideov3 --size 1280x720
VUPSCALE_CUDNN_BENCHMARK=1 vupscale bench -m realesr-animevideov3 --size 1280x720
vupscale bench -m realesr-animevideov3 --size 1280x720 --compile on
```

Run each a second time before comparing: the first run includes MIOpen's kernel
compilation. They work on NVIDIA too, where both default to `1`.

## Other settings

| Variable | Meaning |
|---|---|
| `VUPSCALE_VAAPI_DEVICE` | Render node to encode on, e.g. `/dev/dri/renderD129`. By default an AMD node is preferred over an Intel one |
| `HSA_OVERRIDE_GFX_VERSION` | ROCm setting. `10.3.0` lets RDNA 2 cards other than the RX 6800/6900 use the `gfx1030` kernels |

## Limitations

- Linux only.
- Hardware decoding is not used (as on NVIDIA).
- Measured on one RDNA 4 card only. `vupscale bench` results from RDNA 2 and RDNA 3
  cards would tell whether `torch.compile` should be on by default there too.
- With `torch.compile`, PyTorch on recent ROCm may log a notice about "origami GEMM
  selection"; vupscale hides it, and it does not affect the result.
