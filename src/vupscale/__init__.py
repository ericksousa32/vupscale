"""vupscale: video upscaling with PyTorch models and zero-disk FFmpeg streaming."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import OptionsError, UpscaleOptions
from .models import MODEL_ALIASES, ModelError, resolve_model

__version__ = "0.1.0.dev0"

__all__ = [
    "MODEL_ALIASES",
    "ModelError",
    "OptionsError",
    "UpscaleOptions",
    "__version__",
    "resolve_model",
    "upscale_video",
]


def upscale_video(
    input: str | Path,  # noqa: A002 - mirrors the CLI argument name
    model: str,
    output: str | Path | None = None,
    **options: Any,
) -> Path | list[Path]:
    """Upscale a video (or every video in a directory / matching a glob).

    Same options and defaults as the ``vupscale`` command; see :class:`UpscaleOptions`.
    Only ``input`` and ``model`` are required::

        upscale_video("clip.mp4", model="realesrgan-x4")

    Returns the output path, or a list of paths when ``input`` matches several videos.
    """
    from . import pipeline

    jobs = pipeline.make_jobs(input, model, output=output, **options)
    results = [pipeline.run(job) for job in jobs]
    return results[0] if len(results) == 1 else results
