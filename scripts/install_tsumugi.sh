#!/bin/zsh
# Install tsumugi (MIT) into external/tsumugi at the tested commit, with its own venv.
set -e
ROOT=${0:A:h:h}
DEST=${SOUNDCODE_TSUMUGI:-$ROOT/external/tsumugi}
COMMIT=020edc1
free_gb=$(df -g /System/Volumes/Data | awk 'NR==2 {print $4}')
if [ "$free_gb" -lt 3 ]; then echo "only ${free_gb} GB free; need ~1 GB (stop at 3 GB)"; exit 1; fi
[ -d "$DEST/.git" ] || git clone https://github.com/anime-song/tsumugi.git "$DEST"
git -C "$DEST" fetch --quiet origin && git -C "$DEST" checkout --quiet "$COMMIT"
(cd "$DEST" && uv sync --locked --python 3.11)
echo "tsumugi $COMMIT ready at $DEST"
