# AMD GPUs

vupscale runs on AMD Radeon GPUs under Linux, through the ROCm build of PyTorch for
inference and VAAPI for hardware encoding.

> **Status:** implemented and covered by tests with simulated AMD GPUs, but not yet
> measured on AMD hardware. The defaults below are conservative; the
> [tuning section](#tuning-on-your-card) explains how to check them on your card.

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
| `--compile auto` | On for long renders on Ampere and newer | Off | Not measured on AMD yet; `--compile on` enables it |
| Memory layout | `channels_last` | Default (NCHW) | MIOpen, AMD's convolution library, is most tuned for NCHW |
| Kernel autotuning | cuDNN benchmark on | MIOpen exhaustive search off | Exhaustive search runs for every new shape, and calibration produces many |
| `--codec auto` | NVENC | VAAPI | Hardware encoder of each vendor on Linux |
| `--quality` | NVENC constant quality | VAAPI constant QP | See [Video options](video.md#quality) |

Memory calibration, tiling, out-of-memory recovery, checkpoints and everything else
work the same way.

MIOpen compiles its kernels on first use and caches them (in `~/.cache/miopen` by
default), so the first render or benchmark on a new card, or after a ROCm update, is
slower than the following ones.

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
- No measured speeds yet. Please share `vupscale bench` results from AMD cards.
