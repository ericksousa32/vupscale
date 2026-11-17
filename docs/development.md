# Development

## Setup

```bash
git clone https://github.com/ericksousa32/vupscale
cd vupscale
pip install -e ".[dev]"
```

FFmpeg must be installed for the video tests. No GPU is needed.

## Checks

```bash
ruff check .
ruff format --check .
pytest
```

CI runs the same three commands on Python 3.10 and 3.12 with CPU-only PyTorch, on every
push to `main` or `dev` and on every pull request.

## Branches

Work happens on `dev`. `main` receives pull requests from `dev` once the change has been
tested, including on a real GPU where it matters.

## Tests

The suite needs no downloads and no GPU. Everything it uses is generated on the fly:

| Fixture | What it is |
|---|---|
| `sample_video` | 2 seconds of FFmpeg's `testsrc2`, 320x240, 24 fps (48 frames), with a sine audio track |
| `tiny_model` | A random-weight 2x Compact model that spandrel recognises |
| `isolated_cache` | Points `VUPSCALE_CACHE_DIR` at a temporary directory for every test |

| File | Covers |
|---|---|
| `test_config.py` | Option defaults, validation, output naming, input expansion |
| `test_models.py` | Catalog, local paths, downloads against a local HTTP server, cache, truncated downloads |
| `test_cli.py` | Commands, invocation styles, error messages, a real render through the CLI |
| `test_tuning.py` | Precision, encoder choice, calibration, runtime configuration, out-of-memory recovery |
| `test_engine.py` | Inference against a direct model call, tiling, resize, padding, compile fallback |
| `test_video_io.py` | Probing, exact frame selection, colours, writer, audio, segment joining |
| `test_pipeline.py` | End-to-end renders, checkpoints, resume after crash and Ctrl+C, damaged segments |
| `test_fixtures.py` | That the fixtures are what they claim to be |

### Testing GPU logic without a GPU

Tuning code takes its hardware facts as arguments, so tests pass in fake ones:

- `GpuInfo("Tesla T4", (7, 5), 14.6, 14.5)` stands in for a detected GPU.
- `FakeGpu` in `test_tuning.py` is a probe with a simple memory model that raises
  `torch.cuda.OutOfMemoryError` above its capacity. Calibration runs against it.
- Encoder availability is injected as a function (`works=`).

Out-of-memory recovery is tested end to end by making the engine's forward pass raise on
chosen calls.

### What the tests cannot cover

No test runs on CUDA. Pinned memory, fp16, real VRAM measurement and NVENC are only
checked by hand. After changing `engine.py` or `tuning.py`, verify on a GPU:

```bash
vupscale doctor
vupscale bench -m realesr-animevideov3 --size 1920x1080
vupscale bench -m realesrgan-x4 --size 1920x1080        # exercises the memory limits
vupscale some_video.mp4 -m realesr-animevideov3 --max-frames 300
```

Problems found this way so far, each now covered by a test of the logic involved:

- A host copy that dominated 8K renders on a fast GPU.
- A calibrated batch that fit on its first run and never again.
- FFmpeg releases that reject a colour matrix name others accept.
- Typer releases that no longer install `click`.

### Checking exact frames

`test_video_io.py` writes a clip whose frame `i` is a flat grey of value `10 + 5 * i`,
then reads it back and recovers the indices. This is how frame selection, seeking and
segment joins are proved to neither drop nor duplicate a frame.

## Adding a model alias

Add a `ModelInfo` entry to `_CATALOG` in `models.py`:

```python
(
    ModelInfo(
        "my-alias",  # lower case
        "https://example.com/model.pth",  # a stable release asset ending in .pth
        4,  # scale
        "One line on what it is for.",
    ),
)
```

`test_catalog_is_consistent` checks the entry's shape. Confirm the URL is a stable
release asset and that the model loads: `vupscale bench -m my-alias --size 640x360`.

## Adding an option

1. Add the field and its default to `UpscaleOptions` in `config.py`, with validation in
   `validate()`.
2. Add the matching `typer.Option` in `cli.py` and pass it through to `make_jobs`.
3. If it changes the output, add it to `_job_identity` in `pipeline.py` so that a
   resumed render cannot mix settings.
4. Document it in `docs/cli.md`.

`upscale_video()` picks up new options automatically through `UpscaleOptions`.

## Conventions

- Line length 100, formatted with `ruff format`.
- Modules import `torch` lazily where they can, so `vupscale doctor` can report a
  missing PyTorch instead of crashing.
- Anything written to a user-visible location goes through a temporary name and a
  rename.
- Messages printed to the user are in English.
