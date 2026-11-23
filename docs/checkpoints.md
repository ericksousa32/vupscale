# Checkpoints and resume

Long renders on Colab or a rented GPU can be cut off at any moment. With a checkpoint
directory, vupscale renders the video in segments and can continue from the last
finished one.

```bash
vupscale video.mp4 -m realesrgan-x2 --checkpoint-dir /workspace/ckpt --resume
```

Run the same command again after an interruption and it continues. Run it again after
it has finished and it returns immediately.

## How it works

1. The output is cut into segments by **processing time**, not by frame count:
   `--segment-seconds` (default 180) of work per segment. A fixed frame count would mean
   seconds of work with a light model and half an hour with a heavy one.
2. Each segment is encoded to fast local disk (`/tmp/vupscale_work`).
3. When a segment is complete, its frame count is verified and it is copied to the
   checkpoint directory as `name.part`, then renamed. The copy runs on a background
   thread, so a slow destination does not stall the GPU.
4. At the end the segments are joined without re-encoding, the audio is added, and the
   segments are deleted.

Because of steps 2 and 3, the checkpoint directory can live on Google Drive or another
slow mount: it only ever receives complete files.

Segments are named after the output frames they hold, inclusive:
`seg_000000-000743.mp4`.

## What survives an interruption

| Event | What is kept |
|---|---|
| Ctrl+C, or stopping a Colab cell | Everything, including the segment in progress, which is closed cleanly |
| Crash, disconnect, pod terminated | Every segment already copied to the checkpoint directory. At most `--segment-seconds` of work is lost |

Lower `--segment-seconds` to lose less on a hard stop, at the cost of more files and a
little more overhead.

## Resuming

With `--resume`, vupscale reads the checkpoint directory and:

- Keeps the unbroken run of segments starting at frame 0 whose frame counts check out.
- Deletes segments that are damaged, incomplete, or that come after a gap.
- Seeks to the first missing frame and renders only what is left.

The summary table shows `resuming at frame N`.

### The job must be the same

A `manifest.json` in the checkpoint directory records what the job is. Resuming is
refused, with a message naming what changed, if any of these differ:

- the input video (file name, size, and a hash of its first 8 MB)
- the model
- the output resolution, the frame rate, and the frame-selection options
- the encoder, `--quality` and `--pix-fmt`

Otherwise the result would be a video stitched from mismatched pieces.

These can change freely between runs: batch size, tiling, precision, `torch.compile`,
`--no-audio`, and the output path.

The encoder must match exactly, including hardware (NVENC, VAAPI) versus CPU. If you may
resume on a machine without the same hardware encoder (an A100, or a different GPU
vendor), start the job with a CPU codec such as
`--codec hevc` so that both machines can produce it.

## Without `--resume`

If the checkpoint directory holds an unfinished render, a run without `--resume` stops
with an error instead of discarding the progress. Pass `--resume` to continue, or point
at an empty directory to start over.

A directory whose last job finished can be reused for a new job.

## Finished jobs

When a job completes, the manifest is marked as completed and records the output path.
Running the same job again with `--resume` prints `already rendered` and returns the
existing file, which is convenient when re-executing a notebook cell. If the output file
has since been deleted, the render starts again from the beginning.

## Several videos

With a directory or glob as input, each video uses its own sub-directory of
`--checkpoint-dir`, named after the video, so their segments never mix.

## Disk space

- The checkpoint directory holds all finished segments until the end.
- While the segments are being joined, the segments and the final file exist at the
  same time. Allow at least twice the final size on the checkpoint and output volumes.
- The local work directory holds one segment at a time.

## Outputs on Google Drive without checkpoints

Even without `--checkpoint-dir`, an output under `/content/drive` is rendered to local
disk first and copied over at the end, because streaming an encode directly onto the
Drive mount is slow and unreliable.

## Python

```python
upscale_video(
    "/content/drive/MyDrive/videos/clip.mp4",
    model="realesrgan-x2",
    checkpoint_dir="/content/drive/MyDrive/vupscale_ckpt",
    resume=True,
    segment_seconds=120,
)
```
