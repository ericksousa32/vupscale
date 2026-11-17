# Models

vupscale loads checkpoints with [spandrel](https://github.com/chaiNNer-org/spandrel),
which recognises the architecture from the weights. `--model` accepts three things:

| Form | Example |
|---|---|
| Built-in alias | `-m realesrgan-x4` |
| Local file | `-m /workspace/models/4x_foolhardy_Remacri.pth` |
| HTTP(S) URL | `-m https://example.com/model.safetensors` |

## Built-in aliases

| Alias | Scale | Architecture | Notes |
|---|---|---|---|
| `realesrgan-x4` | 4x | ESRGAN (RRDB) | Real-world footage, best quality of the set, heavy |
| `realesrgan-x2` | 2x | ESRGAN (RRDB) | Real-world footage at 2x |
| `realesrgan-x4-anime` | 4x | ESRGAN, 6 blocks | Illustrations and anime, medium weight |
| `realesr-animevideov3` | 4x | Compact | Anime video. Very fast |
| `realesr-general-x4v3` | 4x | Compact | General purpose. Very fast, less detail than `realesrgan-x4` |

All five are the official Real-ESRGAN releases. `vupscale models` shows which are
already downloaded.

## Downloads and cache

Aliases and URLs are downloaded once to `~/.cache/vupscale/models` (or under
`VUPSCALE_CACHE_DIR`). Downloads go to a `.part` file and are renamed only when the size
matches what the server announced, so an interrupted download never leaves a broken
model behind.

## Using other models

Any image super-resolution or restoration model that spandrel supports should load:
ESRGAN, Real-ESRGAN, Compact, SPAN, SwinIR, HAT, DAT, OmniSR and others. Community
models from sites such as OpenModelDB work by path or URL.

Requirements checked at load time:

- It must be an image-to-image model with purpose `SR` or `Restoration`. Face-restoration
  and inpainting models are rejected.
- It must take and return 3 channels (RGB).

Only the built-in aliases and the Compact test model are exercised by the test suite.
Other architectures rely on spandrel's own handling and have not been run by this
project; try a short piece with `--max-frames` first.

### Restoration models (1x)

Models with scale 1 (denoisers, deblockers) work the same way: the output has the input
resolution unless `--out-scale` or `--target-height` says otherwise.

## Choosing a model

Speed differs by an order of magnitude or more between the Compact models and the full
ESRGAN ones. Measured inference speed for 1080p input:

| Model | Tesla T4 | RTX 4090 |
|---|---|---|
| `realesr-animevideov3` | 3.2 fps | 19 fps (21 with `torch.compile`) |
| `realesrgan-x4` | 0.09 fps | 0.68 fps |

Smaller inputs are proportionally faster: cost follows the number of input pixels.
[Performance](performance.md) has the full measurements and how to take your own.

Practical guidance:

- On a T4, the Compact models are the realistic choice for anything longer than a clip.
- A 4x model used for a 1.5x or 2x enlargement does most of its work for nothing; a 2x
  model is the better fit when one exists for the content.
- Models are trained for a kind of content. An anime model tends to over-smooth live
  action. Compare candidates on a representative 10 to 20 second piece before
  committing hours to a render.

```bash
vupscale video.mp4 -m realesrgan-x2 --skip-frames 30000 --max-frames 600 -o sample_a.mp4
vupscale video.mp4 -m realesr-general-x4v3 --target-height 2160 \
  --skip-frames 30000 --max-frames 600 -o sample_b.mp4
```

## What vupscale reads from the model

| Property | Used for |
|---|---|
| Scale | Output size, default file name |
| `supports_half`, `supports_bfloat16` | Choosing the precision. See [Auto-tuning](auto-tuning.md#precision) |
| Tiling recommendation | A note is shown if tiling is needed for a model that discourages it |
| Size requirements | Inputs and tiles are padded to the multiple the model needs, then cropped |
