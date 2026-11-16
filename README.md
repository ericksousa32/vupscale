# vupscale

CLI and Python API tool for upscaling videos with PyTorch (.pth) models, zero-disk streaming via FFmpeg, and adaptive auto-tuning for Google Colab, RunPod, and local GPUs.

> **Status: under construction.** Stage 1 of 5 is in place (package, CLI, model catalog,
> hardware probes). The render pipeline lands in the next stages; see
> [docs/PLAN.md](docs/PLAN.md). Until then `--dry-run`, `doctor` and `models` work.

## Install

```bash
pip install "git+https://github.com/ericksousa32/vupscale"
```

PyTorch is not pinned, so on Google Colab and RunPod the pre-installed build is reused
and nothing heavy is downloaded. FFmpeg must be on the `PATH`
(`bash scripts/setup_cloud.sh` installs it when missing).

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

| Command | What it does |
|---|---|
| `vupscale doctor` | Environment, GPU, FFmpeg, and which encoders really work (tested with a real encode) |
| `vupscale models` | Built-in model aliases and what is cached |
| `vupscale upscale --help` | Every option, grouped by topic |

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```

Tests generate a synthetic video and a tiny random model on the fly, so they run on CPU
without downloads. Work happens on `dev`; `main` receives tested pull requests.
