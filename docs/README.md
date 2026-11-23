# vupscale documentation

vupscale upscales videos with PyTorch super-resolution models. Frames stream from FFmpeg
to the GPU and back to FFmpeg through pipes, so no image files are written to disk, and
the batch size, tiling, precision and codec are tuned automatically for the machine it
runs on. NVIDIA GPUs (CUDA) and AMD GPUs (ROCm, on Linux) are supported.

```bash
vupscale video.mp4 -m realesrgan-x4
```

## Using vupscale

| Page | What it covers |
|---|---|
| [Getting started](getting-started.md) | Install on Colab, RunPod or a local machine, and render a first video |
| [CLI reference](cli.md) | Every command and option |
| [Python API](python-api.md) | `upscale_video()` for notebooks and scripts |
| [Models](models.md) | Built-in aliases, your own checkpoints, choosing a model |
| [Video options](video.md) | Frame selection, output size, codecs, quality, audio, colour |
| [Checkpoints and resume](checkpoints.md) | Surviving disconnects on long renders |
| [AMD GPUs](amd.md) | Running on Radeon RDNA 2 to RDNA 4 under Linux: setup, defaults, tuning |
| [Troubleshooting](troubleshooting.md) | Common errors and known limitations |

## Understanding vupscale

| Page | What it covers |
|---|---|
| [Auto-tuning](auto-tuning.md) | How precision, batch size, tiling and codec are chosen |
| [Performance](performance.md) | Measured speeds, `vupscale bench`, finding the bottleneck |
| [Architecture](architecture.md) | Modules, the three-stage pipeline, buffer lifetimes |
| [Development](development.md) | Tests, CI, branch workflow, adding a model alias |
