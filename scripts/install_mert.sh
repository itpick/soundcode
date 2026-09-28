#!/bin/zsh
# MERT-v1-95M (m-a-p, CC BY-NC 4.0, allowed for private research) onto the
# external drive, plus transformers/nnAudio in the main .venv (no pip in it:
# uv pip install --python .venv/bin/python, per docs/superpowers/plans/2026-09-26-compare-and-render.md).
set -e
ROOT=${0:A:h:h}
DEST=${SOUNDCODE_MERT:-"/Volumes/ExFAT 2/infinity-engine/models/mert"}
[ -d "/Volumes/ExFAT 2" ] || { echo "external drive not mounted"; exit 2; }
mkdir -p "$DEST"

uv pip install --quiet --python "$ROOT/.venv/bin/python" "transformers==4.46.3" nnAudio
"$ROOT/.venv/bin/hf" download m-a-p/MERT-v1-95M --local-dir "$DEST"
echo "MERT-v1-95M ready at $DEST"
