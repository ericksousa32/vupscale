# Getting started

## Requirements

- Python 3.10 or newer
- PyTorch (any recent build; a CUDA build for GPU rendering)
- FFmpeg and ffprobe on the `PATH`
- A GPU is strongly recommended: NVIDIA (CUDA), or AMD Radeon RDNA 2 or newer on Linux
  (ROCm; see [AMD GPUs](amd.md)). CPU rendering works but is very slow.

## Install

```bash
pip install "git+https://github.com/ericksousa32/vupscale"
```

PyTorch is deliberately not pinned. On Google Colab and RunPod the PyTorch that is
already installed satisfies the requirement, so nothing large is downloaded.

To try the development branch, add `@dev` to the URL.

### Google Colab

```python
!pip install -q "git+https://github.com/ericksousa32/vupscale"
!vupscale doctor
```

Colab already ships FFmpeg. Select a GPU runtime first (Runtime > Change runtime type).

### RunPod

Use the official PyTorch template. Many RunPod images do not include FFmpeg:

```bash
apt-get update && apt-get install -y ffmpeg tmux
pip install "git+https://github.com/ericksousa32/vupscale"
vupscale doctor
```

On Ubuntu 24.04 images pip may refuse with `externally-managed-environment`; add
`--break-system-packages` to the `pip install` command.

For long renders, run inside `tmux` so that closing the browser tab does not kill the
job, and attach a volume at `/workspace` so files survive the pod.

### Local machine with an AMD GPU

Install the ROCm build of PyTorch and FFmpeg with VAAPI first; [AMD GPUs](amd.md)
walks through it. Then install vupscale as above and run `vupscale doctor`.

### From a clone

```bash
git clone https://github.com/ericksousa32/vupscale
cd vupscale
bash scripts/setup_cloud.sh      # installs FFmpeg if missing, then `pip install -e .`
```

## Check the machine

```bash
vupscale doctor
```

`doctor` prints the detected environment, GPU, memory, FFmpeg version and which encoders
work. Encoders are tested with a real one-frame encode, because FFmpeg lists hardware
encoders even on GPUs that cannot use them. It exits with an error only when something
required is missing (FFmpeg, ffprobe, PyTorch or an H.264 encoder).

## First render

```bash
vupscale video.mp4 -m realesr-animevideov3
```

This writes `video_x4.mp4` next to the input, with the original audio. Only the input
and the model are required.

Before a long render it is worth trying a short piece:

```bash
vupscale video.mp4 -m realesr-animevideov3 --max-frames 300
```

To see what would happen without rendering anything:

```bash
vupscale video.mp4 -m realesr-animevideov3 --dry-run
```

## Reading the output

A render prints a summary table and then a progress line:

```
upscaling ━━━━━━━━━━━━━━━━━━ 300/300 15.47 fps 2.8/15 GB 0:00:00 limit: gpu
done: /content/video_1440p.mp4  (20.0 s, 15.00 fps, limited by gpu)
```

- `15.47 fps` is the end-to-end speed, including decoding and encoding.
- `2.8/15 GB` is the VRAM reserved by PyTorch out of the GPU total.
- `limit:` names the stage holding the render back: `gpu` (the model), `decode`
  (FFmpeg cannot supply frames fast enough) or `encode` (the encoder cannot keep up).
  See [Performance](performance.md).

## Next steps

- Pick a model: [Models](models.md)
- Choose the output resolution, codec and quality: [Video options](video.md)
- Make long renders resumable: [Checkpoints and resume](checkpoints.md)
