"""Resumable renders: finished segments live in a checkpoint directory with a manifest.

Segments are encoded on fast local disk and only then copied to the checkpoint directory
(``name.part`` + rename), so a slow or flaky mount such as Google Drive never sees a
half-written file. On resume every segment is validated by counting its frames.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .video_io import VideoError, count_frames

MANIFEST = "manifest.json"
_HEAD_BYTES = 8 << 20
_SEGMENT = re.compile(r"^seg_(\d{6,})-(\d{6,})$")


class CheckpointError(RuntimeError):
    """The checkpoint directory cannot be used for this job."""


def fingerprint(path: Path) -> dict[str, Any]:
    """Cheap identity of a file: its size and the SHA-1 of its first 8 MB."""
    digest = hashlib.sha1()
    with open(path, "rb") as fh:
        digest.update(fh.read(_HEAD_BYTES))
    return {"name": path.name, "size": path.stat().st_size, "sha1_head": digest.hexdigest()}


def atomic_copy(source: Path, destination: Path) -> None:
    """Copy so that ``destination`` either does not exist or is complete."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".part")
    try:
        shutil.copyfile(source, temp)
        os.replace(temp, destination)
    except BaseException:
        with contextlib.suppress(OSError):
            temp.unlink()
        raise


class CheckpointStore:
    def __init__(self, directory: Path, suffix: str = ".mp4") -> None:
        self.directory = Path(directory)
        self.suffix = suffix
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vupscale-publish")
        self._pending: list[Future[Path]] = []

    # ------------------------------------------------------------------ manifest

    @property
    def manifest_path(self) -> Path:
        return self.directory / MANIFEST

    def read_manifest(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.manifest_path.read_text())
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _write_manifest(self, data: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        temp = self.manifest_path.with_name(MANIFEST + ".part")
        temp.write_text(json.dumps(data, indent=1, sort_keys=True))
        os.replace(temp, self.manifest_path)

    # ------------------------------------------------------------------ segments

    def segment_path(self, start: int, end: int) -> Path:
        """Segment holding output frames ``start``..``end`` inclusive."""
        return self.directory / f"seg_{start:06d}-{end:06d}{self.suffix}"

    def _listed(self) -> list[tuple[int, int, Path]]:
        found = []
        if self.directory.is_dir():
            for path in self.directory.iterdir():
                match = _SEGMENT.match(path.stem)
                if match and path.suffix == self.suffix and path.is_file():
                    found.append((int(match[1]), int(match[2]), path))
        return sorted(found)

    def segments(self) -> list[Path]:
        return [path for _, _, path in self._listed()]

    def _wipe(self) -> None:
        for _, _, path in self._listed():
            path.unlink(missing_ok=True)
        if self.directory.is_dir():
            for stray in self.directory.glob("*.part"):
                stray.unlink(missing_ok=True)

    # ------------------------------------------------------------------ lifecycle

    def open(self, job: dict[str, Any], *, resume: bool) -> int:
        """Prepare the directory for ``job`` and return the first frame still to render.

        ``job`` describes everything that shapes the output. Resuming with a different
        job is refused: the result would be a video stitched from mismatched pieces.
        """
        previous = self.read_manifest()
        leftovers = self._listed()
        if not resume:
            if leftovers or (previous and not previous.get("completed")):
                raise CheckpointError(
                    f"{self.directory} holds an unfinished render. Pass --resume to continue "
                    "it, or use an empty checkpoint directory to start over."
                )
            self._write_manifest({"job": job, "completed": False})
            return 0

        if previous is None:
            if leftovers:
                raise CheckpointError(
                    f"{self.directory} has segments but no readable {MANIFEST}; "
                    "use an empty checkpoint directory"
                )
            self._write_manifest({"job": job, "completed": False})
            return 0

        old = previous.get("job", {})
        changed = sorted(k for k in set(old) | set(job) if old.get(k) != job.get(k))
        if changed:
            details = ", ".join(f"{k}: {old.get(k)!r} -> {job.get(k)!r}" for k in changed)
            raise CheckpointError(
                f"cannot resume: the job changed since the checkpoint was made ({details}). "
                "Restore the original settings or use an empty checkpoint directory."
            )
        if previous.get("completed"):
            self._write_manifest({"job": job, "completed": False})

        # Keep the contiguous run of valid segments from frame 0; drop everything else.
        next_frame = 0
        for start, end, path in leftovers:
            valid = start == next_frame and end >= start
            if valid:
                try:
                    valid = count_frames(path) == end - start + 1
                except VideoError:
                    valid = False
            if valid:
                next_frame = end + 1
            else:
                path.unlink(missing_ok=True)
        for stray in self.directory.glob("*.part"):
            stray.unlink(missing_ok=True)
        return next_frame

    def completed_output(self, job: dict[str, Any]) -> Path | None:
        """The finished file of an identical, already completed job, if it still exists."""
        previous = self.read_manifest()
        if not previous or not previous.get("completed") or previous.get("job") != job:
            return None
        output = Path(previous.get("output", ""))
        return output if output.is_file() else None

    def publish(self, local: Path, start: int, end: int) -> None:
        """Verify a finished local segment and move it into the checkpoint directory.

        Runs on a background thread so a slow destination does not stall the GPU.
        """
        expected = end - start + 1
        target = self.segment_path(start, end)

        def work() -> Path:
            actual = count_frames(local)
            if actual != expected:
                raise CheckpointError(
                    f"segment {target.name} has {actual} frames, expected {expected}"
                )
            atomic_copy(local, target)
            local.unlink(missing_ok=True)
            return target

        self._pending.append(self._pool.submit(work))

    def wait(self) -> None:
        """Block until every published segment is safely stored; re-raise failures."""
        pending, self._pending = self._pending, []
        for future in pending:
            future.result()

    def finish(self, output: Path) -> None:
        """Mark the job done and free the space taken by the segments."""
        manifest = self.read_manifest() or {}
        manifest.update(completed=True, output=str(output))
        self._write_manifest(manifest)
        self._wipe()

    def close(self) -> None:
        self._pool.shutdown(wait=True)
