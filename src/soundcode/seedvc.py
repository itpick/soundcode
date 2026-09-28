"""Seed-VC (Plachtaa/seed-vc @51383ef, GPL-3.0): singing voice conversion.

Runs in its own checkout and venv as a subprocess — locally (external/seed-vc)
or, with SOUNDCODE_SEEDVC_HOST set (e.g. `framepick`), on a remote CUDA box
over ssh (~/infinity-engine/seed-vc there; scripts/install_seedvc_remote.sh).
Reference = the original singer (separated lead stem); f0 is preserved.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from .sing_score import SingError

_FLAGS = ("--length-adjust 1.0 --inference-cfg-rate 0.7 --f0-condition True "
          "--auto-f0-adjust False --semi-tone-shift 0 --fp16 False")
REMOTE_DIR = "infinity-engine/seed-vc"


def home() -> Path:
    root = Path(os.environ.get("SOUNDCODE_SEEDVC",
                               Path(__file__).resolve().parents[2] / "external" / "seed-vc"))
    if not (root / ".venv" / "bin" / "python").exists():
        raise SingError(f"Seed-VC not installed at {root}; run scripts/install_seedvc.sh")
    return root


def _last_line(proc: subprocess.CompletedProcess) -> str:
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    tail = [ln for ln in tail if ln.strip() and not ln.startswith("=") and "AUTHORIZED" not in ln]
    return tail[-1] if tail else f"exit {proc.returncode}"


def _remote(host: str, src: Path, ref: Path, out: Path, steps: int) -> Path:
    job = f"infinity-engine/jobs/{uuid.uuid4().hex[:10]}"
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host]

    def run(cmd: list[str], what: str, timeout: int = 600) -> None:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise SingError(f"Seed-VC on {host}: {what} timed out after {timeout}s") from exc
        if proc.returncode != 0:
            raise SingError(f"Seed-VC on {host}: {what} failed ({_last_line(proc)})")

    scp = ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    part = out.with_suffix(".part")
    # NixOS keeps the NVIDIA driver libs outside the default path; harmless elsewhere
    remote = (f"cd {REMOTE_DIR} && LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH "
              f"HF_HUB_CACHE=$PWD/checkpoints/hf_cache "
              f".venv/bin/python inference.py --source ~/{job}/{shlex.quote(src.name)} "
              f"--target ~/{job}/{shlex.quote(ref.name)} --output ~/{job}/out "
              f"--diffusion-steps {steps} {_FLAGS} && ls ~/{job}/out/vc_*.wav")
    try:
        run([*ssh, f"mkdir -p {job}"], "mkdir")
        run([*scp, str(src), str(ref), f"{host}:{job}/"], "upload")
        run([*ssh, remote], "inference", timeout=1800)
        run([*scp, f"{host}:{job}/out/vc_*.wav", str(part)], "download")
        part.rename(out)                         # atomic: a cut download never reaches the cache
    finally:
        part.unlink(missing_ok=True)
        try:
            subprocess.run([*ssh, f"rm -rf {job}"], capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            pass
    return out


def convert(src: Path, ref: Path, out: Path, steps: int = 30) -> Path:
    src, ref, out = Path(src).resolve(), Path(ref).resolve(), Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    host = os.environ.get("SOUNDCODE_SEEDVC_HOST")
    if host:
        return _remote(host, src, ref, out, steps)
    root = home()
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [str(root / ".venv" / "bin" / "python"), "inference.py", "--source", str(src),
               "--target", str(ref), "--output", tmp, "--diffusion-steps", str(steps),
               *_FLAGS.split()]
        env = {**os.environ, "HF_HUB_CACHE": str(root / "checkpoints" / "hf_cache")}
        try:
            proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, env=env,
                                  timeout=1800)
        except subprocess.TimeoutExpired as exc:
            raise SingError("Seed-VC timed out after 1800s") from exc
        wavs = sorted(Path(tmp).glob("vc_*.wav"))
        if proc.returncode != 0 or not wavs:
            raise SingError(f"Seed-VC failed: {_last_line(proc)}")
        shutil.move(str(wavs[0]), out)
    return out
