# Architecture

vupscale started as a translation of a ComfyUI VideoHelperSuite workflow (load video,
meta batch manager, upscale with model, video combine) into a standalone streaming
pipeline.

## Modules

| Module | Responsibility |
|---|---|
| `config.py` | `UpscaleOptions`: every option, its default and its validation. Output naming and size arithmetic |
| `cli.py` | Typer commands. Translates arguments into `UpscaleOptions` and errors into exit codes |
| `__init__.py` | `upscale_video()`, the Python entry point |
| `models.py` | Alias catalog, model resolution, cached atomic downloads |
| `hardware.py` | What the machine is: environment, GPU, RAM, FFmpeg, real encoder tests |
| `tuning.py` | How to run on it: precision, codec, calibration, `RuntimeConfig`, out-of-memory recovery |
| `engine.py` | Model loading and inference: uint8 transfer, tiling, resize, `torch.compile` |
| `video_io.py` | ffprobe metadata, decode planning, threaded FFmpeg reader and writer, segment joining |
| `checkpoint.py` | Segment store, manifest, validation on resume |
| `pipeline.py` | Orchestration: wires everything together, progress display |
| `console.py` | Shared Rich console that also works in Jupyter and Colab |

`hardware.py` answers "what is this machine?" and has no opinions. `tuning.py` takes
those facts plus the model's traits and makes the decisions. Keeping them apart lets the
decisions be tested on a CPU-only machine with simulated GPUs.

## The pipeline

```
┌────────────────┐   queue    ┌──────────────────────────────┐   queue    ┌────────────────┐
│ decode thread  │ ─────────> │ main thread                  │ ─────────> │ encode thread  │
│ ffmpeg -> RGB  │  (depth N) │ upload, model, convert, down │  (depth N) │ RGB -> ffmpeg  │
└────────────────┘            └──────────────────────────────┘            └────────────────┘
```

Three stages run concurrently. FFmpeg decodes in one subprocess and encodes in another;
a Python thread per subprocess moves bytes through the pipes. The main thread only
drives the GPU. Without this, the GPU would sit idle while FFmpeg encodes.

The queues are bounded. Their depth comes from the host RAM budget (see
[Auto-tuning](auto-tuning.md#host-ram)), and their fill levels are how the progress line
knows which stage is the bottleneck.

### One batch, step by step

1. The decode thread reads raw `rgb24` bytes from FFmpeg's stdout straight into a
   preallocated NumPy buffer (`readinto`, no intermediate copy) and queues the batch.
2. The main thread copies it into a pinned buffer and uploads it as uint8.
3. On the GPU: convert to the model's precision, run the model (whole frames or tiles),
   optionally resize, clamp, round, convert back to uint8.
4. The result is downloaded into a pinned buffer and a view of it is queued for the
   encoder.
5. The encode thread writes it to FFmpeg's stdin.

`AdaptiveBatchController` wraps step 3: on an out-of-memory error it reduces the
settings and retries the same frames.

## Buffer lifetimes

To avoid copying large frames, buffers are reused, which means a batch is only valid for
a limited time. The rules:

| Buffer ring | Size | A batch is valid until |
|---|---|---|
| Reader output | queue depth + 2 | queue depth + 1 later batches have been taken |
| Engine output | queue depth + 2 | that many further `process_batch` calls |

The engine ring size follows from the writer: at most `depth` batches wait in its queue
and one is being written, so with the one being produced, `depth + 2` distinct buffers
are needed. `VideoStreamWriter.write` blocks while the queue is full, which is what
guarantees the ring never wraps onto a batch still in use.

When the controller has to split a batch after an out-of-memory error, it copies each
partial result, because the engine reuses its buffer on the next call.

Outside the pipeline, `UpscaleEngine.process_batch` returns independent copies; the ring
is enabled explicitly with `set_output_ring`.

## Decode planning

`plan_decode` turns the frame options into an FFmpeg command and computes exactly how
many frames will come out, so progress, segment names and resume points are exact.

- Filter order: `fps`, then skip, then `select` every Nth, then scale and colour
  conversion to `rgb24`.
- Skipping uses a fast input seek (`-ss`, half a frame early to be immune to rounding)
  on constant-frame-rate sources, and an exact `trim` filter otherwise.
- `-fps_mode passthrough` stops FFmpeg from duplicating frames to fake a constant rate.
- A resume point is expressed as "start at output frame K" and translated back into
  source frames.

## Checkpointed renders

```
render segment -> local disk -> verify frame count -> copy as .part -> rename
                                                  (background thread)
all segments -> concat demuxer, stream copy + audio -> output.part -> rename
```

Segments carry no audio; it is muxed once at the end. See
[Checkpoints and resume](checkpoints.md).

## Atomic writes

Nothing visible to the user is ever half-written. Models, calibration cache, manifest,
segments and final outputs are all written under a temporary name and renamed when
complete. A failed or interrupted render leaves either a finished file or no file.

## Error handling

- FFmpeg's stderr is drained continuously into a ring of recent lines by a daemon
  thread, so a full pipe can never deadlock the process, and failures can be reported
  with FFmpeg's own message.
- A dead encoder does not block the producer: the writer thread keeps draining its
  queue and the next `write` raises.
- Exceptions in the decode thread are passed through the queue and re-raised in the
  main thread.
- On any exit path, subprocesses are killed and threads joined.

## Design decisions

Choices that are not obvious from the code, and why they were made.

| Decision | Reason |
|---|---|
| `hardware.py` (facts) is separate from `tuning.py` (decisions) | The decisions can be tested on a CPU-only machine with simulated GPUs |
| Memory is measured, not estimated | Activation memory differs by orders of magnitude between architectures; no formula holds |
| Calibration confirms with two runs back to back | On an RTX 4090 a batch was seen to fit on its first run and never again |
| fp16 is preferred over bf16, even on recent GPUs | More mantissa bits, so less banding, at the same speed |
| Tiles are sized to divide the frame evenly | Fixed 768 px tiles on 1080p recomputed 71% more pixels than the frame has |
| Raised-cosine blending between tiles | Gives almost no weight to tile edges, where the model has the least context |
| With tiling, the batch is one frame | The tiles per call carry the parallelism; it keeps memory predictable |
| Results reach the encoder without a host copy | At 8K on a fast GPU that copy cost far more than the model |
| `--every-nth` divides the frame rate | Duration and audio sync are preserved, as in VideoHelperSuite |
| Audio is trimmed explicitly, never with `-shortest` | `-shortest` lets a short audio track truncate the video |
| Segments are cut by processing time | A fixed frame count means seconds of work with one model and half an hour with another |
| NVENC is detected with a real encode | FFmpeg lists NVENC encoders on GPUs that cannot use them |
| No NVDEC decoding | Not needed so far; worth revisiting if `limit: decode` shows up in practice |
