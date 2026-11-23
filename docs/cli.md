# CLI reference

```
vupscale [INPUT] -m MODEL [OPTIONS]     render (same as `vupscale upscale ...`)
vupscale doctor                         check the machine
vupscale models                         list model aliases
vupscale bench -m MODEL [OPTIONS]       measure inference speed
vupscale --version
```

When the first argument is not a command name, `upscale` is assumed, so
`vupscale video.mp4 -m realesrgan-x4` and `vupscale upscale video.mp4 -m realesrgan-x4`
are the same.

## upscale

### Input, model, output

| Option | Default | Meaning |
|---|---|---|
| `INPUT` or `-i`, `--input` | required | A video file, a directory, or a glob pattern such as `"clips/*.mp4"` |
| `-m`, `--model` | required | A built-in alias, a local `.pth` / `.safetensors` file, or an HTTP(S) URL. See [Models](models.md) |
| `-o`, `--output` | next to the input | Output file, or a directory. A path without an extension is treated as a directory |
| `--dry-run` | off | Print the resolved configuration and exit without rendering |

Default output names:

| Situation | Name |
|---|---|
| Native model scale | `<input>_x4.mp4` |
| `--out-scale 1.5` | `<input>_x1.5.mp4` |
| `--target-height 2160` | `<input>_2160p.mp4` |

With a directory or glob as input, each video is rendered in turn with the model loaded
once. `--output` must then be a directory, and each video gets its own sub-directory
under `--checkpoint-dir`.

The container follows the output extension (`.mp4`, `.mkv`, `.mov`, `.webm`).

### Hardware (auto-tuned)

All of these default to automatic. See [Auto-tuning](auto-tuning.md) for how the
automatic values are chosen.

| Option | Default | Meaning |
|---|---|---|
| `--device` | `auto` | `auto`, `cuda` or `cpu` |
| `--precision` | `auto` | `auto`, `fp16`, `bf16` or `fp32` |
| `--batch-size` | `0` | Frames per batch. `0` = automatic |
| `--tile-size` | `0` | Largest tile in pixels. `0` = automatic, `-1` = no tiling |
| `--tile-overlap` | `32` | Overlap between neighbouring tiles, in input pixels |
| `--compile` | `auto` | `auto`, `on` or `off` for `torch.compile` |

`--tile-size` must be at least 64, and `--tile-overlap` must be less than half of it.

### Output size

| Option | Default | Meaning |
|---|---|---|
| `--out-scale` | `0` | Final size relative to the input, e.g. `2` or `1.5`. `0` = the model's own scale |
| `--target-height` | `0` | Final height in pixels, e.g. `2160`. Width follows the aspect ratio. `0` = off |

Use one or the other, not both. The model always runs at its native scale; the result
is then resized on the GPU. See [Video options](video.md#output-size).

### Frame selection

These follow the ComfyUI VideoHelperSuite options and are applied in this order.

| Option | Default | Meaning |
|---|---|---|
| `--fps` | `0` | Resample to this frame rate first. `0` = keep the original |
| `--skip-frames` | `0` | Drop this many frames from the start |
| `--every-nth` | `1` | Keep one frame in every N. The output frame rate is divided by N |
| `--max-frames` | `0` | Stop after this many output frames. `0` = all |
| `--pre-width`, `--pre-height` | `0` | Resize before upscaling. A zero axis keeps the aspect ratio |

### Encoding

| Option | Default | Meaning |
|---|---|---|
| `--codec` | `auto` | `auto`; `h264`, `hevc`, `av1` (CPU encoders); `h264_nvenc`, `hevc_nvenc`, `av1_nvenc` (NVIDIA); `h264_vaapi`, `hevc_vaapi`, `av1_vaapi` (AMD and Intel on Linux) |
| `--quality`, `--crf` | `20` | 0 to 51, lower is better and larger. CRF on CPU encoders, constant quality (`-cq`) on NVENC, constant QP on VAAPI |
| `--pix-fmt` | `yuv420p` | Output pixel format |
| `--no-audio` | off | Do not copy the audio track |

### Checkpointing

| Option | Default | Meaning |
|---|---|---|
| `--checkpoint-dir` | none | Render in resumable segments stored here |
| `--resume` | off | Continue from the valid segments in `--checkpoint-dir` |
| `--segment-seconds` | `180` | Processing time per segment |

See [Checkpoints and resume](checkpoints.md).

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `2` | Invalid options, unreadable video, model problem, checkpoint mismatch, or nothing fits in memory |
| `130` | Interrupted with Ctrl+C |

## doctor

```bash
vupscale doctor
```

Prints environment (`colab`, `runpod` or `local`), Python and PyTorch versions (with the
CUDA or ROCm version), GPU name, generation and architecture (`sm_89`, `gfx1200`), free
VRAM and RAM, the result of a small test convolution on the GPU, CPU cores, FFmpeg
version, and the result of a real test encode for each NVENC, VAAPI and CPU encoder. The
VAAPI row, with the render node used, appears on machines with an AMD GPU or a render
node. The `codec auto` row shows what
`--codec auto` resolves to for outputs up to 4096 px wide and for larger ones.

Exits with code 1 when FFmpeg, ffprobe, PyTorch or a working H.264 encoder is missing, or
when the test convolution fails. Warnings explain common AMD setup problems: a CUDA
build of PyTorch on an AMD machine, missing `render`/`video` group membership, and RDNA 2
cards that need `HSA_OVERRIDE_GFX_VERSION`.

## models

```bash
vupscale models
```

Lists the built-in aliases with their scale and whether each is already downloaded, and
prints the cache directory.

## bench

```bash
vupscale bench -m realesrgan-x4 --size 1920x1080
```

Calibrates memory and measures inference speed on synthetic frames. No video and no
FFmpeg are needed. See [Performance](performance.md#vupscale-bench).

| Option | Default | Meaning |
|---|---|---|
| `-m`, `--model` | required | Alias, local file or URL |
| `--size` | `1280x720` | Input resolution to simulate, `WIDTHxHEIGHT` |
| `--seconds` | `5` | How long to measure |
| `--device` | `auto` | `auto`, `cuda` or `cpu` |
| `--precision` | `auto` | Override the precision |
| `--batch-size` | `0` | Frames per batch. `0` = automatic |
| `--tile-size` | `0` | `0` = automatic, `-1` = no tiling |
| `--target-height` | `0` | Final height. `0` = native scale |
| `--compile` | `off` | `auto`, `on` or `off`. Off by default so that a quick benchmark does not pay the compile time |

## Environment variables

| Variable | Meaning |
|---|---|
| `VUPSCALE_CACHE_DIR` | Where models and calibration results are cached. Default `~/.cache/vupscale` |
| `VUPSCALE_VAAPI_DEVICE` | Render node for VAAPI encoding, e.g. `/dev/dri/renderD129` |
| `VUPSCALE_CHANNELS_LAST` | `1` or `0`: memory layout for the model. See [AMD GPUs](amd.md#tuning-on-your-card) |
| `VUPSCALE_CUDNN_BENCHMARK` | `1` or `0`: cuDNN / MIOpen kernel autotuning. See [AMD GPUs](amd.md#tuning-on-your-card) |
