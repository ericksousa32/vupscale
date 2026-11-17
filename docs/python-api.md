# Python API

The API mirrors the command line: same options, same defaults, same behaviour.

```python
from vupscale import upscale_video

output = upscale_video("clip.mp4", model="realesrgan-x4")
```

## upscale_video

```python
upscale_video(input, model, output=None, **options) -> Path | list[Path]
```

| Argument | Meaning |
|---|---|
| `input` | A video file, a directory, or a glob pattern |
| `model` | A built-in alias, a local checkpoint, or a URL |
| `output` | Output file or directory. `None` = next to the input |
| `**options` | Any option from the [CLI reference](cli.md), with dashes replaced by underscores |

It returns the output path, or a list of paths when `input` matches several videos.

Option names map directly: `--target-height 2160` becomes `target_height=2160`,
`--no-audio` becomes `no_audio=True`, `--checkpoint-dir` becomes `checkpoint_dir=...`.
Choices can be passed as strings (`precision="fp16"`, `codec="hevc_nvenc"`).

Progress is printed with Rich and renders correctly inside Jupyter and Colab cells.

### Google Colab with Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")

!pip install -q "git+https://github.com/ericksousa32/vupscale"

from vupscale import upscale_video

upscale_video(
    "/content/drive/MyDrive/videos/clip.mp4",
    model="realesr-animevideov3",
    target_height=1440,
    checkpoint_dir="/content/drive/MyDrive/vupscale_ckpt",
    resume=True,
)
```

With `resume=True`, running the same cell again after a disconnect continues the
render, and running it after the job has finished returns immediately. See
[Checkpoints and resume](checkpoints.md).

### Preview without rendering

```python
upscale_video("clip.mp4", model="realesrgan-x4", dry_run=True)
```

Prints the resolved configuration and returns the path the output would have.

## Errors

| Exception | Module | Raised when |
|---|---|---|
| `OptionsError` | `vupscale` | An option value or combination is invalid, or the input does not exist |
| `ModelError` | `vupscale` | A model cannot be found, downloaded or loaded, or is not an RGB upscaler |
| `VideoError` | `vupscale.video_io` | A video cannot be read, FFmpeg fails, or no frames are selected |
| `TuningError` | `vupscale.tuning` | The model does not fit in memory even with the smallest tiles |
| `CheckpointError` | `vupscale.checkpoint` | The checkpoint directory belongs to a different or unfinished job |

`OptionsError` is a `ValueError`; the others are `RuntimeError`. `KeyboardInterrupt`
propagates after the current segment has been saved (with checkpoints) or the partial
output removed (without).

## Other exports

| Name | Meaning |
|---|---|
| `UpscaleOptions` | Dataclass holding every option; the single source of defaults for the CLI and the API |
| `resolve_model(spec)` | Turn an alias, path or URL into a local file, downloading if needed |
| `MODEL_ALIASES` | The built-in catalog, `alias -> ModelInfo(url, scale, description)` |
| `__version__` | Package version |

## Lower-level building blocks

The modules below are importable and tested, but they are internal: their interfaces
may change between versions. [Architecture](architecture.md) describes how they fit
together.

```python
from vupscale.engine import load_model, create_engine
from vupscale.video_io import probe_video, plan_decode, VideoStreamReader, VideoStreamWriter
from vupscale.tuning import build_runtime_config, AdaptiveBatchController
```
