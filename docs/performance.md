# Performance

## Measured speeds

These were measured in November 2026 with the development version. Treat them as
reference points, not guarantees: drivers, PyTorch versions and the video itself all
move the numbers.

### Inference only (`vupscale bench`, 1920x1080 input)

| Model | Output | Tesla T4 (Colab) | RTX 4090 (RunPod) |
|---|---|---|---|
| `realesr-animevideov3` | 7680x4320 | 3.2 fps | 19.1 fps |
| `realesr-animevideov3`, `torch.compile` | 7680x4320 | not measured | 21.1 fps |
| `realesr-animevideov3` | 3840x2160 | 3.4 fps | 19.0 fps |
| `realesrgan-x4` | 7680x4320 | 0.09 fps (tiled) | 0.68 fps |

The 3840x2160 figure for the RTX 4090 predates an optimisation that removed a host
copy; expect it to be somewhat higher now.

The RTX 4090 was about 7 times faster than the T4 on both models.

### Complete renders (decode, upscale, encode)

| GPU | Model | Input | Output | Speed |
|---|---|---|---|---|
| Tesla T4 | `realesr-animevideov3` | 770x576 | 1926x1440, `h264_nvenc` | 15.0 fps |
| Tesla T4 | `realesrgan-x2` | 770x576 | 1540x1152, `h264_nvenc` | about 1.9 fps |

Complete renders have not yet been measured on an RTX 4090.

## What drives the speed

- **Input pixels.** The model's cost follows the size of what goes in. A 720p input is
  about 2.25 times faster than 1080p with the same model.
- **The model.** In the measurements above, the Compact model was roughly 30 times faster
  than the full ESRGAN one.
- **Output size barely matters for the model**, since it always runs at its native
  scale. It matters for transfer and encoding: very large outputs (8K) make those
  stages significant on a fast GPU.
- **Batch size has little effect.** On a T4, batches of 1, 4 and 8 gave 2.9, 2.9 and
  3.2 fps. The largest batch that fits is used.

To estimate a render: frames divided by fps. A 40 minute, 30 fps video is 72,000 frames;
at 3 fps that is under 7 hours.

## vupscale bench

`bench` measures the inference path alone, on synthetic frames. It needs no video and no
FFmpeg, so it is the quickest way to compare models, GPUs or settings.

```bash
vupscale bench -m realesr-animevideov3 --size 1920x1080
```

```
│ batch size              │ 12                             │
│ tiling                  │ off (full frames)              │
│ vram peak               │ 7.0 GB of 23.5 GB              │
├─────────────────────────┼────────────────────────────────┤
│ speed                   │ 19.06 fps (inference only)     │
├─────────────────────────┼────────────────────────────────┤
│ upload (RAM -> GPU)     │      0.9 ms/frame    1.8%      │
│ model                   │     39.1 ms/frame   74.5%      │
│ convert to uint8 (GPU)  │      4.9 ms/frame    9.3%      │
│ download (GPU -> RAM)   │      7.6 ms/frame   14.4%      │
```

| Row | Meaning |
|---|---|
| upload | Copying input frames to the GPU and converting them to the model's precision |
| model | The network itself, including tiling |
| convert to uint8 | Resizing (if any), clamping and rounding the result on the GPU |
| download | Copying the result back to pinned host memory |

Useful comparisons:

```bash
vupscale bench -m MODEL --size 1280x720 --target-height 1440   # your real resolutions
vupscale bench -m MODEL --size 1920x1080 --compile on          # is compiling worth it?
vupscale bench -m MODEL --size 1920x1080 --tile-size 512       # force tiling
```

`bench` leaves `torch.compile` off unless asked, so a quick run does not pay the compile
time. A real render of 1500 frames or more on an Ampere-or-newer GPU turns it on by
default.

## Finding the bottleneck in a render

The progress line ends with `limit:`, and the final line with `limited by`:

| Value | Meaning | What helps |
|---|---|---|
| `gpu` | The model is the slowest stage. This is the normal case | A lighter model, a smaller input (`--pre-height`), `--every-nth`, a faster GPU |
| `decode` | The GPU is waiting for FFmpeg to supply frames | More CPU cores; a less demanding source codec |
| `encode` | The encoder cannot keep up and its queue is full | NVENC instead of a CPU encoder; a smaller output; a faster preset |

It is derived from the queues between the stages: a full encode queue points at the
encoder, an empty decode queue points at the decoder.

On GPUs without NVENC (A100, H100), CPU encoding of 4K or 8K output is the likely limit.

## Why frames travel as uint8

Frames cross between RAM and the GPU as 8-bit integers in both directions, and are
converted on the GPU. This moves a quarter of the data that float32 would. It matters
most on the way back: a 4x model's output has 16 times the pixels of its input.

Results are written into a small ring of pinned buffers and handed to the encoder
thread directly. An earlier version copied each result once more in RAM; on an RTX 4090
at 8K that copy alone took 578 ms per frame against 40 ms for the model, and removing it
took the speed from 1.6 to 19 fps.
