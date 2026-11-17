# Video options

## Frame selection

The frame options follow ComfyUI VideoHelperSuite and are applied in a fixed order:

1. `--fps` resamples the video to a new frame rate (frames are dropped or repeated; the
   duration does not change).
2. `--skip-frames` drops frames from the start.
3. `--every-nth` keeps one frame in every N.
4. `--max-frames` caps the number of output frames.

`--every-nth` divides the output frame rate, so the video keeps its duration: a 30 fps
input with `--every-nth 2` becomes a 15 fps output of the same length. This halves the
work at the cost of smoothness.

Audio stays in sync with every combination: it is started at the same point as the
video and trimmed to the same length.

### Examples

```bash
# 10 seconds starting at the 1000th frame of a 30 fps video
vupscale in.mp4 -m realesrgan-x4 --skip-frames 1000 --max-frames 300

# Half the frame rate, half the render time
vupscale in.mp4 -m realesrgan-x4 --every-nth 2

# Convert 60 fps footage to 30 fps before upscaling
vupscale in.mp4 -m realesrgan-x4 --fps 30
```

### Seeking

On constant-frame-rate videos, `--skip-frames` (and resuming from a checkpoint) seeks
straight to the target frame. With `--fps` or a variable-frame-rate source it falls back
to decoding and discarding everything before the target, which is exact but slower.

### Pre-scaling

`--pre-width` and `--pre-height` resize the video before it reaches the model, with a
Lanczos filter. Giving only one keeps the aspect ratio. This is the main lever when a
model is too slow or too large for the GPU: model cost follows the number of input
pixels.

```bash
# 4K source, 4x model: shrink to 1080p first, upscale back to 4K
vupscale in_4k.mp4 -m realesrgan-x4 --pre-height 1080 --target-height 2160
```

## Output size

By default the output is the input multiplied by the model's scale. Two options change
that:

| Option | Result |
|---|---|
| `--out-scale 2` | Twice the input size, whatever the model's scale |
| `--target-height 1440` | 1440 pixels tall, width from the aspect ratio |

The model still runs at its native scale. Its result is resized on the GPU (bicubic,
antialiased) before it is copied back to RAM, so a smaller output also means less data
to transfer and encode.

Output dimensions are always rounded to even numbers, as `yuv420p` requires.

## Codec

`--codec auto` (the default) picks an encoder for the output resolution:

| Output | Encoder chosen |
|---|---|
| Longest side up to 4096 px | `h264_nvenc`, or `libx264` without NVENC |
| Longest side up to 8192 px | `hevc_nvenc`, or `libx265` without NVENC |
| Larger than 8192 px | `libx265` |

H.264 NVENC cannot encode frames wider than 4096 px, which is why 4x from 1080p (8K)
switches to HEVC.

NVENC is used only if a real test encode succeeds on the machine. This matters because:

- Data-centre GPUs such as the A100 and H100 have no NVENC hardware.
- A Tesla T4 has H.264 and HEVC NVENC but no AV1.
- AV1 NVENC needs an RTX 40-series, L4, L40S or newer.

Explicit choices:

| `--codec` | Encoder |
|---|---|
| `h264`, `hevc` | `libx264`, `libx265` (CPU) |
| `av1` | `libsvtav1`, or `libaom-av1` if SVT-AV1 is missing (CPU) |
| `h264_nvenc`, `hevc_nvenc`, `av1_nvenc` | The NVENC encoder; falls back to the CPU encoder of the same family, with a note, if it does not work here |

HEVC files are roughly a third smaller than H.264 at similar quality, but some browsers
and older editors cannot open them. HEVC in `.mp4` / `.mov` is tagged `hvc1` so that
Apple devices play it.

### Quality

`--quality` (alias `--crf`) runs from 0 to 51; lower is better and larger. The default
is 20. It maps to CRF for CPU encoders and to constant-quality (`-cq`, VBR) for NVENC.
The two scales are similar but not identical, so the same number gives somewhat
different sizes on NVENC and on x264.

As a rule of thumb, adding 6 roughly halves the file size.

File size depends heavily on the content, because the encoders target a quality level,
not a bitrate. To predict it, render a representative piece and scale up:

```bash
vupscale in.mp4 -m realesrgan-x2 --skip-frames 30000 --max-frames 600 -o sample.mp4
ls -lh sample.mp4      # final size ~ sample size x (total frames / 600)
```

### Encoder speed settings

| Encoder | Setting |
|---|---|
| NVENC | preset `p5`, `-tune hq` |
| `libx264`, `libx265` | preset `faster` on machines with 4 cores or fewer, otherwise `medium` |
| `libsvtav1` | preset 8 on 4 cores or fewer, otherwise 6 |

## Container

The output extension selects the container: `.mp4` (default), `.mkv`, `.mov` or `.webm`.
MP4 and MOV files are written with `faststart` so they begin playing before they are
fully downloaded.

## Audio

The first audio track of the input is copied into the output.

- It is stream-copied (no quality loss) when the container supports the codec: AAC,
  MP3, AC-3, E-AC-3 and ALAC in MP4/MOV, anything in MKV, Opus and Vorbis in WebM.
- Otherwise it is re-encoded: AAC at 192 kbit/s, or Opus at 160 kbit/s for WebM.
- A short audio track never shortens the video.
- `--no-audio` drops it.

Only the first audio track is kept. Subtitles, chapters and additional tracks are not
copied.

## Colour

- The input's colour matrix is taken from its metadata. Untagged video follows the
  usual convention: BT.709 for HD (720p and up), BT.601 for SD.
- The output is always BT.709, limited range, and tagged as such.
- Rotation flags (phone videos) are honoured: the output is upright.

## Limitations

- **HDR** (PQ or HLG) input is converted as if it were SDR and looks washed out. A note
  is printed when this happens.
- **Variable frame rate** input is written at its average frame rate.
- **Non-square pixels** (anamorphic DVD or broadcast material) are treated as square;
  the display aspect ratio flag is not carried over.
- Output is 8-bit.
