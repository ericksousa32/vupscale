# Auto-tuning

Every hardware option defaults to `auto`. This page explains what `auto` does, so that
the choices in the summary table make sense and you know when to override them.

Decisions are based on two things: the machine (`vupscale doctor` shows what is
detected) and what the model declares about itself through spandrel.

## Precision

| Situation | Precision |
|---|---|
| CPU | fp32 |
| GPU, model supports half precision | fp16 |
| GPU with bf16 hardware (NVIDIA Ampere and newer, AMD RDNA 3 and newer), model supports bf16 but not fp16 | bf16 |
| Otherwise | fp32 |

fp16 is preferred over bf16 everywhere, including on recent GPUs: it keeps 10 bits of
mantissa against 7, which means less banding in smooth gradients, at the same speed.
bf16 is the fallback for models that overflow in fp16.

Some transformer models declare that they do not support fp16. Forcing it with
`--precision fp16` typically gives black frames; vupscale prints a warning when an
override contradicts the model.

## Batch size and tiling: calibration

Memory use differs by orders of magnitude between architectures, so vupscale does not
estimate it. Before rendering it **measures** it, by running the real code path on
dummy frames:

1. Try one whole frame. If it does not fit, try tiles of at most 768, 512, 384 and
   256 px in turn, until one fits.
2. Run two units (frames, or tiles) to learn the cost of each extra unit, extrapolate
   the largest count that fits, and run that to verify it.
3. Confirm the winner with two runs back to back. The first run of a new shape is the
   easy one, because cuDNN empties the allocator cache while it autotunes; only the
   second shows the steady state of a real render.

"Fits" means the peak stays within 75% of the VRAM that was free at the start. The
batch is capped at 16 frames.

Results are cached in `~/.cache/vupscale/calibration.json`, keyed by GPU, model,
resolution, precision and PyTorch version. A cached result is confirmed with a quick
run before it is trusted, since the free VRAM may differ from last time.

While calibrating, recent PyTorch versions print `CUDACachingAllocator ... OOM` warnings
for each attempt that did not fit. They are expected.

### Tiling

When the whole frame does not fit, it is processed in overlapping tiles:

- The calibrated tile size is an upper bound. Tiles are sized to divide the frame
  evenly, which avoids recomputing large overlaps: a 1080p frame with a 768 px limit is
  covered by six 662x556 tiles.
- Several tiles go through the model in one call (`tiles per call` in the summary).
- Overlaps are blended with a raised-cosine ramp that gives almost no weight to the very
  edge of each tile, where the model has the least context. In tests, tiled output
  matches full-frame output to within 2 levels out of 255.
- With tiling active the batch is one frame; the parallelism comes from the tiles.

Tiled and full-frame rendering were measured at the same speed for `realesrgan-x4` on an
RTX 4090, so tiling is chosen by what fits, not by speed.

### Host RAM

Raw frames are large: an 8K RGB frame is about 100 MB. The frames waiting in the decode
and encode queues are limited to 15% of the free host RAM. This caps both the batch size
and the queue depth, and is what keeps a 12 GB Colab instance alive on 8K output.

### Overriding

| Option | Effect |
|---|---|
| `--batch-size N` | Use exactly N frames per batch |
| `--tile-size N` | Tile at most N px. `-1` disables tiling |
| `--tile-overlap N` | Overlap in input pixels (default 32) |

## Out-of-memory recovery

If the GPU still runs out of memory mid-render (another process took VRAM, or a scene
behaves differently), the render does not abort. Settings are reduced and the failed
batch is retried, in this order:

1. Halve the tiles per call.
2. Halve the batch size.
3. Enable tiling, or move to the next smaller tile size.

No frame is lost or reordered, and the reduced settings stay for the rest of the render.
Each adjustment is printed. The render stops with an error only if a single 256 px tile
does not fit.

This also applies with `--tile-size -1`: tiling is enabled rather than aborting.

## Codec

See [Video options](video.md#codec). In short: NVENC, then VAAPI, when a real test encode
works at the output resolution; H.264 up to 4096 px and HEVC above.

## torch.compile

With `--compile auto`, the model is compiled when all of these hold:

- the GPU is an NVIDIA one from the Ampere generation or newer (compute capability
  8.0+), or an AMD RDNA 4 one; on RDNA 2 and RDNA 3 it stays off until measured, and
  `--compile on` enables it, and
- the job has at least 1500 frames, so the compile time is paid back.

Compilation uses `max-autotune-no-cudagraphs` with fixed shapes; the last, smaller batch
is padded so that nothing recompiles mid-render. If compilation fails for any reason,
the render continues uncompiled and says so.

Measured with `realesr-animevideov3`, compiling made the model 31% faster on an RTX 4090
and 32% faster on an RX 9060 XT.
It has not been measured on the larger ESRGAN models. The first batch takes noticeably
longer while the model compiles.

## AMD GPUs

On ROCm the same calibration and recovery run unchanged. The memory layout and kernel
autotuning defaults differ; see [AMD GPUs](amd.md#what-is-different-on-amd).

## Without a GPU

On CPU the precision is fp32, the batch is one frame, and frames larger than about
768x768 are tiled at 512 px. Nothing is calibrated.

## Seeing the decisions

```bash
vupscale video.mp4 -m realesrgan-x4 --dry-run     # estimate, without calibrating
vupscale bench -m realesrgan-x4 --size 1920x1080  # calibrate and measure
```

`--dry-run` skips calibration, so its batch and tile values are conservative estimates;
a note says so. `bench` and a real render calibrate.
