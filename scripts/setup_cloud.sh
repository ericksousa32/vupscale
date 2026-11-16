#!/usr/bin/env bash
# Idempotent setup for Google Colab, RunPod and plain Linux boxes.
#
# Usually you do not need this script at all:
#     pip install "git+https://github.com/ericksousa32/vupscale"
# reuses the PyTorch that Colab/RunPod already ship. Use this script when ffmpeg is
# missing or when you cloned the repository and want an editable install.
set -euo pipefail

log() { printf '\033[1;36m[vupscale]\033[0m %s\n' "$*"; }

SUDO=""
if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
fi

# 1. ffmpeg / ffprobe -------------------------------------------------------
if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
    log "ffmpeg already installed: $(ffmpeg -version | head -n1 | cut -d' ' -f3)"
elif command -v apt-get >/dev/null 2>&1; then
    log "installing ffmpeg via apt-get..."
    $SUDO apt-get update -qq
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq ffmpeg >/dev/null
else
    log "ffmpeg not found and apt-get is unavailable: install ffmpeg manually." >&2
    exit 1
fi

# 2. the package --------------------------------------------------------------
PYTHON="${PYTHON:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if "$PYTHON" -c "import torch" >/dev/null 2>&1; then
    log "reusing PyTorch $("$PYTHON" -c 'import torch; print(torch.__version__)')"
else
    log "PyTorch not found: pip will install the default build."
fi

if [ -f "$REPO_ROOT/pyproject.toml" ]; then
    log "installing vupscale (editable) from $REPO_ROOT"
    "$PYTHON" -m pip install -q -e "$REPO_ROOT"
else
    log "installing vupscale from GitHub"
    "$PYTHON" -m pip install -q "git+https://github.com/ericksousa32/vupscale"
fi

# 3. sanity check ----------------------------------------------------------------
"$PYTHON" -m vupscale.cli doctor || true
