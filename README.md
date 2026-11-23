# vupscale

CLI and Python API tool for upscaling videos with PyTorch (.pth) models, zero-disk streaming via FFmpeg, and adaptive auto-tuning for Google Colab, RunPod, and local GPUs.

> **Status: beta.** Feature complete and covered by tests. Developed on `dev`; `main`
> receives tested pull requests.

## Install

```bash
pip install "git+https://github.com/ericksousa32/vupscale"
```

PyTorch is not pinned, so on Google Colab and RunPod the pre-installed build is reused
and nothing heavy is downloaded. FFmpeg must be on the `PATH`
(`bash scripts/setup_cloud.sh` installs it when missing).

GPUs: NVIDIA through CUDA, and AMD Radeon RDNA 2 or newer on Linux through ROCm. For AMD,
install the ROCm build of PyTorch first; see [docs/amd.md](docs/amd.md).

## Usage

Only the input and the model are required. Everything else is auto-tuned.

```bash
vupscale video.mp4 -m realesrgan-x4          # -> video_x4.mp4
vupscale video.mp4 -m realesrgan-x4 --target-height 2160
vupscale "clips/*.mp4" -m realesr-animevideov3 -o upscaled/
vupscale video.mp4 -m realesrgan-x4 --dry-run   # show the plan, process nothing
```

```python
from vupscale import upscale_video

upscale_video(
    "/content/drive/MyDrive/videos/clip.mp4",
    model="realesrgan-x4",
    checkpoint_dir="/content/drive/MyDrive/vupscale_ckpt",
    resume=True,
)
```

`--model` accepts a built-in alias, a local `.pth` / `.safetensors` file or an HTTP(S)
URL (downloaded once and cached in `~/.cache/vupscale/models`).

### Resumable renders

With `--checkpoint-dir` the video is rendered in segments (about 3 minutes of work each,
see `--segment-seconds`). Segments are encoded on local disk and copied to the checkpoint
directory when complete, so it can live on Google Drive. After a disconnect, run the same
command with `--resume`: valid segments are kept, damaged ones are redone, and only the
missing frames are rendered. Ctrl+C saves the segment in progress.

The progress line shows speed, VRAM, ETA and `limit:` - which stage is holding the
render back (`gpu`, `decode` or `encode`).

| Command | What it does |
|---|---|
| `vupscale doctor` | Environment, GPU, FFmpeg, and which encoders really work (tested with a real encode) |
| `vupscale bench -m MODEL --size 1920x1080` | Calibrate memory and measure inference speed on synthetic frames |
| `vupscale models` | Built-in model aliases and what is cached |
| `vupscale upscale --help` | Every option, grouped by topic |

## Documentation

Full documentation lives in [docs/](docs/README.md):
[getting started](docs/getting-started.md) ·
[CLI reference](docs/cli.md) ·
[Python API](docs/python-api.md) ·
[models](docs/models.md) ·
[video options](docs/video.md) ·
[checkpoints](docs/checkpoints.md) ·
[AMD GPUs](docs/amd.md) ·
[auto-tuning](docs/auto-tuning.md) ·
[performance](docs/performance.md) ·
[architecture](docs/architecture.md) ·
[development](docs/development.md) ·
[troubleshooting](docs/troubleshooting.md)

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```

Tests generate a synthetic video and a tiny random model on the fly, so they run on CPU
without downloads. Work happens on `dev`; `main` receives tested pull requests.
