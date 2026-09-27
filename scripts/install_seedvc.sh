#!/bin/zsh
# Seed-VC (GPL-3.0) singing voice conversion, own venv; checkpoints on the external drive.
set -e
ROOT=${0:A:h:h}
DEST=${SOUNDCODE_SEEDVC:-$ROOT/external/seed-vc}
CK="/Volumes/ExFAT 2/infinity-engine/models/seed-vc-checkpoints"
SPIKE="/Volumes/ExFAT 2/infinity-engine/scratch/svs/seed-vc/checkpoints"
[ -d "/Volumes/ExFAT 2" ] || { echo "external drive not mounted"; exit 2; }
[ -d "$DEST/.git" ] || git clone https://github.com/Plachtaa/seed-vc.git "$DEST"
git -C "$DEST" checkout --quiet 51383ef
git -C "$DEST" apply --check "$ROOT/scripts/seedvc.patch" 2>/dev/null && git -C "$DEST" apply "$ROOT/scripts/seedvc.patch"
if [ ! -e "$DEST/checkpoints" ]; then
  mkdir -p "$CK"; [ -d "$SPIKE" ] && cp -R "$SPIKE/." "$CK/"
  ln -s "$CK" "$DEST/checkpoints"
fi
cd "$DEST" && uv venv --quiet --python 3.11 .venv
uv pip install --quiet --python .venv/bin/python torch==2.13.0 torchaudio numpy==1.26.4 scipy==1.13.1 \
  librosa==0.10.2 munch einops "huggingface-hub>=0.28.1" transformers==4.46.3 soundfile pyyaml \
  descript-audio-codec==1.0.0 "setuptools<80"
echo "seed-vc 51383ef ready at $DEST"
