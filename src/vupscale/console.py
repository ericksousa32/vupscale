"""Shared Rich console that renders cleanly both in terminals and in Jupyter/Colab cells."""

from __future__ import annotations

from functools import lru_cache

from rich.console import Console


def in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]
    except Exception:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ in {"ZMQInteractiveShell", "Shell"}


@lru_cache(maxsize=1)
def get_console() -> Console:
    if in_notebook():
        return Console(force_jupyter=True, width=110)
    return Console()


# Notebooks re-render the whole output on every refresh; keep it low there.
def progress_refresh_rate() -> float:
    return 2.0 if in_notebook() else 8.0
