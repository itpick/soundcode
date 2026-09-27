#!/bin/zsh
# DiffSinger bank (Azure Cobalt v0.4.28, CC BY-SA 4.0) + NSF-HiFiGAN vocoder
# (CC BY-NC-SA) onto the external drive. Reuses the 2026-09-26 spike download.
set -e
DEST=${SOUNDCODE_DIFFSINGER:-"/Volumes/ExFAT 2/infinity-engine/models/diffsinger"}
SPIKE="/Volumes/ExFAT 2/infinity-engine/scratch/svs/banks"
[ -d "${DEST:h}" ] || { echo "external drive not mounted: ${DEST:h}"; exit 2; }
mkdir -p "$DEST"
[ -d "$DEST/AC0.4.28" ] || cp -R "$SPIKE/ac/AC0.4.28" "$DEST/"
[ -f "$DEST/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx" ] || \
  cp "$SPIKE/voc/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx" "$DEST/"
diff -rq -x '._*' "$SPIKE/ac/AC0.4.28" "$DEST/AC0.4.28" && echo "diffsinger bank ready at $DEST"
