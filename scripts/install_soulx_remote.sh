#!/bin/zsh
# Install SoulX-Singer (Apache-2.0) on a remote CUDA box (default: framepick) under
# ~/infinity-engine/soulx-singer, with its own uv venv (python 3.10) and weights.
set -e
HOST=${SOUNDCODE_SOULX_HOST:-${SOUNDCODE_SEEDVC_HOST:-framepick}}
ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" 'bash -s' <<'REMOTE'
set -e
mkdir -p ~/infinity-engine && cd ~/infinity-engine
[ -d soulx-singer/.git ] || git clone -q https://github.com/Soul-AILab/SoulX-Singer.git soulx-singer
cd soulx-singer
[ -x .venv/bin/python ] || uv venv -q --python 3.10 .venv
uv pip install -q --python .venv/bin/python -r requirements.txt
uv pip install -q --python .venv/bin/python "huggingface_hub[cli]"
# the pinned torch 2.2 has no Blackwell (sm_120, RTX 50xx) kernels: use a cu128 build
uv pip install -q --python .venv/bin/python "torch==2.9.1" "torchaudio==2.9.1" --index-url https://download.pytorch.org/whl/cu128
.venv/bin/hf download Soul-AILab/SoulX-Singer --local-dir pretrained_models/SoulX-Singer >/dev/null
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt .venv/bin/python -m nltk.downloader -q -d .venv/nltk_data cmudict averaged_perceptron_tagger_eng >/dev/null 2>&1 || true
.venv/bin/hf download Soul-AILab/SoulX-Singer-Preprocess --local-dir pretrained_models/SoulX-Singer-Preprocess >/dev/null
LD_LIBRARY_PATH=/run/opengl-driver/lib .venv/bin/python -c "import torch; print('cuda', torch.cuda.is_available())"
echo "soulx-singer $(git rev-parse --short HEAD) ready on $(hostname) at $PWD; $(du -sh pretrained_models | cut -f1) weights"
REMOTE
