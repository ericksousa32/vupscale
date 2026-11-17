# Troubleshooting

Start with `vupscale doctor`: it shows what was detected and flags anything missing.

## Installation

**`ffmpeg not found`**
Install FFmpeg: `apt-get update && apt-get install -y ffmpeg`, or run
`bash scripts/setup_cloud.sh` from a clone. Many RunPod images ship without it.

**`error: externally-managed-environment` from pip**
Ubuntu 24.04 images protect the system Python. Add `--break-system-packages` to the
`pip install` command, or use a virtual environment.

**pip downloads a multi-gigabyte PyTorch**
PyTorch was not already installed in the environment pip is using. On Colab and the
RunPod PyTorch templates it is; elsewhere, install the PyTorch build for your CUDA
version first.

## GPU and encoders

**`doctor` shows `gpu: none (CPU only)`**
PyTorch cannot see a CUDA device. On Colab, select a GPU runtime. Elsewhere, check that
the installed PyTorch is a CUDA build (`python -c "import torch; print(torch.version.cuda)"`).

**`nvenc (tested)` shows `no` on a machine with an NVIDIA GPU**
Either FFmpeg was missing when the test ran, or the GPU has no NVENC hardware. A100 and
H100 cards have none, and a T4 has no AV1. vupscale then encodes on the CPU.

**`h264_nvenc is limited to 4096 px, switching to HEVC`**
Expected for outputs wider than 4096 px. Use `--target-height` for a smaller output if
you need H.264.

## Memory

**`CUDACachingAllocator ... memory allocation failed with OOM` during "calibrating memory"**
Expected. Calibration deliberately tries batches that are too large to find the limit,
and recent PyTorch versions log each failed attempt.

**`out of memory: batch size reduced to ...` during a render**
The render adapted and continued; no frames were lost. If it happens on every run,
another process is probably using VRAM.

**`the model does not fit in memory for this video, even with the smallest tiles`**
A single 256 px tile does not fit. Use a lighter model, shrink the input with
`--pre-height`, or free VRAM held by other processes.

**The Colab session crashes with RAM exhausted**
Very large outputs hold a lot of raw frames. Lower the output size with
`--target-height`, or set `--batch-size 1`.

## Output

**Black or corrupted frames**
The model probably does not work in half precision. Try `--precision fp32`.

**Visible grid or seams**
Increase `--tile-overlap` (for example to 64), or try `--tile-size -1` if the frame fits
in VRAM.

**Colours look washed out**
The input is probably HDR, which is not tone-mapped. A note is printed at the start of
the render when this is detected.

**The picture looks stretched**
The source has non-square pixels (anamorphic DVD or broadcast material). The display
aspect ratio is not carried over. Resize to square pixels first with `--pre-width` and
`--pre-height`.

**The file does not open in a browser or editor**
It may be HEVC. Use `--codec h264_nvenc` or `--codec h264`, and keep the output at most
4096 px wide.

**The file is larger than expected**
Raise `--quality` (24 is noticeably smaller than the default 20) or use HEVC. See
[Video options](video.md#quality).

## Speed

**The render is slow**
Look at `limit:` in the progress line. `gpu` means the model itself is the limit; see
[Performance](performance.md#finding-the-bottleneck-in-a-render) for what helps in each
case. Compare models with `vupscale bench`.

**The first batch takes a long time**
With `torch.compile` active (long renders on recent GPUs), the first batch includes
compilation. Use `--compile off` to skip it for short tests.

## Checkpoints

**`holds an unfinished render. Pass --resume to continue it`**
The checkpoint directory has segments from an earlier run. Add `--resume`, or use an
empty directory to start over.

**`cannot resume: the job changed since the checkpoint was made (...)`**
The message lists what differs. Restore those settings, or use an empty directory.
Renaming the input file also counts as a change.

**Resuming on a different machine is refused because of the encoder**
`--codec auto` picked NVENC on one machine and a CPU encoder on the other. Start jobs
that may move between machines with an explicit CPU codec, such as `--codec hevc`.

**The job finished, but running it again re-renders everything**
That happens when the output file was moved or deleted, or when `--resume` was not
passed.

## Interrupting

**Closing the terminal killed the render**
Run long jobs inside `tmux` (or `screen`), and use `--checkpoint-dir` with `--resume`.

## Known limitations

- One GPU is used; multi-GPU rendering is not supported.
- Only RGB models (3 channels in and out).
- Output is 8-bit SDR, BT.709.
- HDR input is not tone-mapped.
- Variable-frame-rate input is written at its average frame rate.
- Non-square pixel aspect ratios are not preserved.
- Only the first audio track is copied; subtitles and chapters are not.
- Hardware (NVDEC) decoding is not used.
- Models are per-frame: there is no temporal model support, so some flicker between
  frames is inherent to the model used.

## Reporting a problem

Include the output of `vupscale doctor`, the full command, and the summary table and
error message from the run.
