#!/bin/zsh
# Install Seed-VC (GPL-3.0) on a remote CUDA box over ssh (default: framepick),
# under ~/infinity-engine/seed-vc, pinned + patched, with its checkpoints.
set -e
ROOT=${0:A:h:h}
HOST=${SOUNDCODE_SEEDVC_HOST:-framepick}
ssh -o BatchMode=yes "$HOST" 'mkdir -p ~/infinity-engine' 2>/dev/null
scp -q "$ROOT/scripts/seedvc.patch" "$HOST":infinity-engine/seedvc.patch 2>/dev/null
ssh -o BatchMode=yes "$HOST" 'bash -s' 2>/dev/null <<'REMOTE'
set -e
cd ~/infinity-engine
[ -d seed-vc/.git ] || git clone -q https://github.com/Plachtaa/seed-vc.git
cd seed-vc && git checkout -q 51383ef
git apply --check ../seedvc.patch 2>/dev/null && git apply ../seedvc.patch || true
[ -x .venv/bin/python ] || uv venv -q --python 3.11 .venv
uv pip install -q --python .venv/bin/python torch==2.13.0 torchaudio numpy==1.26.4 scipy==1.13.1 \
  librosa==0.10.2 munch einops "huggingface-hub>=0.28.1" transformers==4.46.3 soundfile pyyaml \
  descript-audio-codec==1.0.0 "setuptools<80"
HF_HUB_CACHE=$PWD/checkpoints/hf_cache .venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
for repo, fn in [("Plachta/Seed-VC","config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"),
                 ("Plachta/Seed-VC","DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"),
                 ("lj1995/VoiceConversionWebUI","rmvpe.pt"),("funasr/campplus","campplus_cn_common.bin")]:
    hf_hub_download(repo_id=repo, filename=fn, cache_dir="./checkpoints")
for fn in ["config.json","bigvgan_generator.pt"]:
    hf_hub_download(repo_id="nvidia/bigvgan_v2_44khz_128band_512x", filename=fn)
hf_hub_download(repo_id="openai/whisper-base", filename="model.safetensors")
PY
LD_LIBRARY_PATH=/run/opengl-driver/lib .venv/bin/python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "seed-vc ready on $(hostname) at $PWD"
REMOTE
