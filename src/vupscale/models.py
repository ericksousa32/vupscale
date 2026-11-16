"""Model catalog (aliases), resolution of ``--model`` and cached downloads."""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from .console import get_console

MODEL_EXTENSIONS = (".pth", ".safetensors", ".pt", ".ckpt")
_RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"


class ModelError(RuntimeError):
    """Raised when a model spec cannot be resolved or downloaded."""


@dataclass(frozen=True)
class ModelInfo:
    alias: str
    url: str
    scale: int
    description: str

    @property
    def filename(self) -> str:
        return _filename_from_url(self.url)


_CATALOG = (
    ModelInfo(
        "realesrgan-x4",
        f"{_RELEASES}/v0.1.0/RealESRGAN_x4plus.pth",
        4,
        "Real-ESRGAN x4plus. General real-world footage, best quality, heavy.",
    ),
    ModelInfo(
        "realesrgan-x2",
        f"{_RELEASES}/v0.2.1/RealESRGAN_x2plus.pth",
        2,
        "Real-ESRGAN x2plus. General real-world footage, 2x.",
    ),
    ModelInfo(
        "realesrgan-x4-anime",
        f"{_RELEASES}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        4,
        "Real-ESRGAN x4plus anime (6 blocks). Illustrations and anime, medium weight.",
    ),
    ModelInfo(
        "realesr-animevideov3",
        f"{_RELEASES}/v0.2.5.0/realesr-animevideov3.pth",
        4,
        "Compact model for anime video. Very fast, good default for a T4.",
    ),
    ModelInfo(
        "realesr-general-x4v3",
        f"{_RELEASES}/v0.2.5.0/realesr-general-x4v3.pth",
        4,
        "Compact general-purpose model. Very fast, lighter detail than x4plus.",
    ),
)

MODEL_ALIASES: dict[str, ModelInfo] = {m.alias: m for m in _CATALOG}


def cache_dir() -> Path:
    """Where downloaded models live. Override with ``VUPSCALE_CACHE_DIR``."""
    root = os.environ.get("VUPSCALE_CACHE_DIR")
    base = Path(root).expanduser() if root else Path.home() / ".cache" / "vupscale"
    return base / "models"


def is_url(spec: str) -> bool:
    return urlparse(spec).scheme in ("http", "https")


def _filename_from_url(url: str) -> str:
    name = Path(unquote(urlparse(url).path)).name
    if not name.lower().endswith(MODEL_EXTENSIONS):
        # Opaque URLs (query-string downloads): keep them apart in the cache.
        digest = hashlib.sha1(url.encode()).hexdigest()[:12]
        name = f"{name or 'model'}-{digest}.pth"
    return name


def cached_path(url: str) -> Path:
    return cache_dir() / _filename_from_url(url)


def download(url: str, dest: Path, *, quiet: bool = False) -> Path:
    """Download ``url`` to ``dest`` atomically (``.part`` + rename) with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "vupscale"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            expected = int(response.headers.get("Content-Length") or 0)
            columns = (
                TextColumn("[bold]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
            )
            written = 0
            with (
                Progress(*columns, console=get_console(), transient=True, disable=quiet) as bar,
                open(part, "wb") as fh,
            ):
                task = bar.add_task(dest.name, total=expected or None)
                while chunk := response.read(1 << 20):
                    fh.write(chunk)
                    written += len(chunk)
                    bar.update(task, advance=len(chunk))
    except (urllib.error.URLError, OSError) as exc:
        part.unlink(missing_ok=True)
        raise ModelError(f"failed to download model from {url}: {exc}") from exc

    if written == 0 or (expected and written != expected):
        part.unlink(missing_ok=True)
        raise ModelError(
            f"incomplete download from {url}: got {written} bytes, expected {expected or '> 0'}"
        )
    os.replace(part, dest)
    return dest


def resolve_model(spec: str | Path, *, quiet: bool = False) -> Path:
    """Turn an alias, a local path or an HTTP(S) URL into a local model file."""
    text = str(spec).strip()
    if not text:
        raise ModelError("empty model spec")

    info = MODEL_ALIASES.get(text.lower())
    url = info.url if info else (text if is_url(text) else None)
    if url is not None:
        dest = cached_path(url)
        if dest.is_file() and dest.stat().st_size > 0:
            return dest
        return download(url, dest, quiet=quiet)

    path = Path(text).expanduser()
    if path.is_file():
        return path

    hint = ""
    if not path.suffix:
        known = ", ".join(sorted(MODEL_ALIASES))
        hint = f" (not a known alias either; available: {known})"
    raise ModelError(f"model file not found: {text}{hint}")


def describe_spec(spec: str | Path) -> str:
    """Human description of where a model spec points, without touching the network."""
    text = str(spec).strip()
    info = MODEL_ALIASES.get(text.lower())
    if info:
        state = "cached" if cached_path(info.url).is_file() else "will download"
        return f"{info.alias} -> {info.filename} (x{info.scale}, {state})"
    if is_url(text):
        state = "cached" if cached_path(text).is_file() else "will download"
        return f"{_filename_from_url(text)} ({state})"
    path = Path(text).expanduser()
    return f"{path} ({'found' if path.is_file() else 'NOT FOUND'})"
