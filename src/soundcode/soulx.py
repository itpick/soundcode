"""SoulX-Singer (Soul-AILab, Apache-2.0): zero-shot, melody-conditioned singing.

One model sings lyrics + melody (f0 from :contour.vox) in the voice of a short
reference clip of the original singer — no separate voice-conversion step.
Runs on a remote CUDA box over ssh (~/infinity-engine/soulx-singer there;
scripts/install_soulx_remote.sh), like seedvc.py.

Metadata format, pinned from SoulX-Singer@81aeb3a (preprocess/tools/midi_parser.py):
  one dict per segment: index, language, time [ms0, ms1] (song time), duration
  (s per note), text (word per note; <SP> rests), phoneme (en_ARPAbet-with-stress
  per note; filled remotely by SoulX's own g2p_transform), note_pitch (MIDI; 0 for
  rests), note_type (1 rest, 2 a word's first note, 3 continuation / melisma),
  f0 (Hz at 24 kHz / hop 480 = 50 frames per second).
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import uuid
from pathlib import Path

import numpy as np

from . import sing_score as ss
from .sing_score import SingError

F0_RATE = 50.0
SR = 24000
MAX_SEG_S = 15.0
REST_MIN_S = 0.05
REMOTE_DIR = "infinity-engine/soulx-singer"


def _phoneme(word: str) -> str:
    """SoulX's format (en_ + stressed ARPAbet). The remote run refills these with
    SoulX's own g2p_transform; this local copy keeps previews and tests honest."""
    if word == "<SP>":
        return "<SP>"
    phs, hit = ss.g2p(word)                       # also loads the dictionary
    prons = ss._cmu.get(word) if hit else None
    if prons:
        return "en_" + "-".join(prons[0])
    return "en_" + "-".join(p[3:].upper() for p in phs)


def _notes_with_words(doc) -> list[tuple[float, float, int, str, int]]:
    """(start, end, midi, text, type) with every word sung.

    Driven by the words, not the notes: a note spanning several words is split
    at the word boundaries (same pitch), a word covering several notes sings
    them as a melisma (type 3), and a word no note overlaps gets one at the
    nearest note's pitch. Notes outside every word are sung on 'ah'.
    (Note-driven assignment silently dropped 11 of River's 61 words.)"""
    notes = ss.vocal_notes(doc, ss.vocal_stream(doc))
    ws = [w for w in ss.words(doc) if w[1] > w[0]]
    out: list[tuple[float, float, int, str, int]] = []

    def nearest_pitch(t: float) -> int:
        if not notes:
            return 60
        return min(notes, key=lambda n: 0 if n[0] <= t < n[1] else min(abs(n[0] - t), abs(n[1] - t)))[2]

    for k, (w0, w1, word) in enumerate(ws):
        w1 = min(w1, ws[k + 1][0]) if k + 1 < len(ws) else w1
        if w1 - w0 < 0.03:
            w1 = w0 + 0.03
        pieces = [(max(a, w0), min(b, w1), p) for a, b, p in notes if b > w0 and a < w1]
        pieces = [x for x in pieces if x[1] - x[0] >= 0.02]
        if not pieces:
            pieces = [(w0, w1, nearest_pitch((w0 + w1) / 2))]
        pieces[0] = (w0 if pieces[0][0] - w0 < 0.15 else pieces[0][0], pieces[0][1], pieces[0][2])
        for m, (a, b, p) in enumerate(pieces):
            out.append((a, b, p, word, 2 if m == 0 else 3))
    covered = [(a, b) for a, b, *_ in out]
    for a, b, p in notes:                               # notes no word touches: sung on 'ah'
        if b - a >= 0.1 and not any(ca < b and cb > a for ca, cb in covered):
            out.append((a, b, p, "ah", 2))
    out.sort(key=lambda x: x[0])
    clean: list[tuple[float, float, int, str, int]] = []
    for x in out:                                       # keep strictly monotonic, no overlaps
        if clean and x[0] < clean[-1][1]:
            x = (clean[-1][1], x[1], x[2], x[3], x[4])
        if x[1] - x[0] >= 0.01:
            clean.append(x)
    return clean


def _f0(doc, t0: float, t1: float, notes) -> list[float]:
    from .contour import STEP_S, read_contour

    t = t0 + np.arange(round((t1 - t0) * F0_RATE)) / F0_RATE
    hz = np.zeros(len(t))
    for a, b, p, _, _ in notes:
        hz[(t >= a) & (t < b)] = 440.0 * 2 ** ((p - 69) / 12)
    for start, vals in read_contour(doc):
        seg_t = start + np.arange(len(vals)) * STEP_S
        inside = (t >= seg_t[0]) & (t < seg_t[-1] + STEP_S)
        hz[inside] = 440.0 * 2 ** ((np.interp(t[inside], seg_t, vals) - 6900) / 1200)
    return [round(float(x), 1) for x in hz]


def _segment(doc, t0: float, t1: float, notes, index: str) -> dict:
    items: list[tuple[str, float, int, int]] = []           # text, dur, pitch, type
    cur = t0
    for a, b, p, text, typ in notes:
        a, b = max(a, t0), min(b, t1)
        if b <= a:
            continue
        if a - cur >= REST_MIN_S:
            items.append(("<SP>", a - cur, 0, 1))
        elif items:                                           # tiny gap: extend the previous note
            items[-1] = (items[-1][0], items[-1][1] + (a - cur), items[-1][2], items[-1][3])
        else:
            a = cur
        items.append((text, b - a, p, typ))
        cur = b
    if t1 - cur > 1e-6:
        items.append(("<SP>", t1 - cur, 0, 1))
    merged: list[tuple[str, float, int, int]] = []       # no zero-length tokens after rounding
    for it in items:
        if merged and it[1] < 0.01:
            merged[-1] = (merged[-1][0], merged[-1][1] + it[1], merged[-1][2], merged[-1][3])
        else:
            merged.append(it)
    items = merged
    durs = [round(d, 2) for _, d, _, _ in items]
    durs[-1] = round(durs[-1] + (t1 - t0) - sum(durs), 2)
    return {"index": index, "language": "English", "time": [int(round(t0 * 1000)), int(round(t1 * 1000))],
            "duration": " ".join(f"{d:.2f}" for d in durs),
            "text": " ".join(x[0] for x in items),
            "phoneme": " ".join(_phoneme(x[0]) for x in items),
            "note_pitch": " ".join(str(x[2]) for x in items),
            "note_type": " ".join(str(x[3]) for x in items),
            "f0": " ".join(str(v) for v in _f0(doc, t0, t1, notes))}


def metadata(doc, t0: float, t1: float) -> list[dict]:
    """SoulX segments covering [t0, t1], <= 15 s each, cut between notes: at a
    rest (>= 0.3 s) once a segment is half full, else at the widest gap before
    it would overflow. A note is never split."""
    notes = [n for n in _notes_with_words(doc) if n[1] > t0 and n[0] < t1]
    cuts, start, best = [], t0, None                    # best = (gap, mid) in the current segment
    for (a0, b0, *_), (a1, b1, *_) in zip(notes, notes[1:]):
        mid, gap = (b0 + a1) / 2, a1 - b0
        if gap >= 0.3 and mid - start >= MAX_SEG_S * 0.5:
            cuts.append((start, mid)); start, best = mid, None
            continue
        if mid - start > 1.0 and (best is None or gap >= best[0]):
            best = (gap, mid)
        if b1 - start > MAX_SEG_S:
            cut = best[1] if best else start + MAX_SEG_S
            cuts.append((start, cut)); start, best = cut, None
    cuts.append((start, t1))
    fixed = []
    for a, b in cuts:                                   # a tail longer than 15 s with no notes
        while b - a > MAX_SEG_S:
            fixed.append((a, a + MAX_SEG_S))
            a += MAX_SEG_S
        fixed.append((a, b))
    return [_segment(doc, a, b, [n for n in notes if n[1] > a and n[0] < b],
                     f"vocal_{int(a * 1000)}_{int(b * 1000)}") for a, b in fixed]


def prompt_window(doc, want_s: float = 8.0) -> tuple[float, float]:
    notes = _notes_with_words(doc)
    if not notes:
        raise SingError("no vocal notes for a SoulX prompt")
    t0 = max(0.0, notes[0][0] - 0.2)
    end = t0 + want_s
    for (_, b0, *_), (a1, *_) in zip(notes, notes[1:]):      # end at a rest after ~want_s
        if b0 >= t0 + want_s * 0.6 and a1 - b0 >= 0.2:
            end = (b0 + a1) / 2
            break
    return t0, min(end, t0 + max(12.0, want_s * 1.25))


CONTROL = "melody"          # "melody" (f0 curve) or "score" (MIDI notes)
PROMPT_S = 12.0
SEED = 0                    # SoulX is a diffusion model: fixed so re-renders and A/B tests repeat
ALIGN = 2                   # bump when the .sc → SoulX score mapping changes (invalidates cached vocals)


def settings() -> dict:
    """Everything besides the .sc and the reference that changes a SoulX render (for the cache key)."""
    return {"control": CONTROL, "prompt_s": PROMPT_S, "seed": SEED, "align": ALIGN}


def render(doc, ref_wav: Path, prompt: tuple[float, float] | None = None,
           control: str | None = None, prompt_s: float | None = None,
           n_steps: int | None = None, cfg: float | None = None, seed: int | None = SEED) -> np.ndarray:
    """Sing the whole vocal remotely; returns mono float32 at 44.1 kHz.

    `n_steps`/`cfg` override SoulX's diffusion steps and guidance scale (config defaults 32 / 3).
    `seed` fixes torch/numpy/random so a render is repeatable; None leaves it random."""
    import librosa
    import soundfile as sf

    host = os.environ.get("SOUNDCODE_SOULX_HOST") or os.environ.get("SOUNDCODE_SEEDVC_HOST") or "framepick"
    p0, p1 = prompt or prompt_window(doc, prompt_s or PROMPT_S)
    dur = doc.duration or max(n[1] for n in _notes_with_words(doc)) + 1.0
    target = metadata(doc, 0.0, dur)
    pmeta = metadata(doc, p0, p1)[:1]
    pmeta[0]["time"] = [0, int(round((p1 - p0) * 1000))]
    y, _ = librosa.load(str(ref_wav), sr=SR, mono=True, offset=p0, duration=p1 - p0)
    job = f"infinity-engine/jobs/{uuid.uuid4().hex[:10]}"
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host]
    scp = ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]

    def run(cmd, what, timeout=600):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise SingError(f"SoulX on {host}: {what} timed out after {timeout}s") from exc
        if proc.returncode != 0:
            tail = [ln for ln in (proc.stderr or proc.stdout).strip().splitlines()
                    if ln.strip() and "AUTHORIZED" not in ln and not ln.startswith("=")]
            raise SingError(f"SoulX on {host}: {what} failed ({tail[-1] if tail else proc.returncode})")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "target.json").write_text(json.dumps(target))
        (tmp / "prompt.json").write_text(json.dumps(pmeta))
        sf.write(str(tmp / "prompt.wav"), y, SR)
        fill = ("import json,sys; from preprocess.tools.g2p import g2p_transform as g\n"
                "for p in sys.argv[1:]:\n"
                "    d=json.load(open(p))\n"
                "    for s in d: s['phoneme']=' '.join(g(s['text'].split(),'English'))\n"
                "    json.dump(d,open(p,'w'))\n")
        (tmp / "fill.py").write_text(fill)
        run_py = ("import runpy,sys,random; import numpy, torch\n"
                  "s=int(sys.argv.pop(1))\n"
                  "if s>=0: random.seed(s); numpy.random.seed(s); torch.manual_seed(s)\n"
                  "runpy.run_module('cli.inference', run_name='__main__')\n")
        (tmp / "run.py").write_text(run_py)
        conf = "soulxsinger/config/soulxsinger.yaml"
        edit = ""
        if n_steps is not None or cfg is not None:
            subs = []
            if n_steps is not None:
                subs.append(f"-e 's/^\\( *n_steps:\\).*/\\1 {int(n_steps)}/'")
            if cfg is not None:
                subs.append(f"-e 's/^\\( *cfg:\\).*/\\1 {float(cfg)}/'")
            edit = f"sed {' '.join(subs)} {conf} > ~/{job}/config.yaml && "
            conf = f"~/{job}/config.yaml"
        remote = (f"cd {REMOTE_DIR} && export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH "
                  f"PYTHONPATH=$PWD && {edit}.venv/bin/python ~/{job}/fill.py ~/{job}/target.json ~/{job}/prompt.json && "
                  f".venv/bin/python ~/{job}/run.py {-1 if seed is None else int(seed)} --device cuda "
                  f"--model_path pretrained_models/SoulX-Singer/model.pt "
                  f"--config {conf} "
                  f"--prompt_wav_path ~/{job}/prompt.wav --prompt_metadata_path ~/{job}/prompt.json "
                  f"--target_metadata_path ~/{job}/target.json "
                  f"--phoneset_path soulxsinger/utils/phoneme/phone_set.json "
                  f"--save_dir ~/{job}/out --auto_shift --pitch_shift 0 --control {control or CONTROL}")
        out = tmp / "generated.wav"
        try:
            run([*ssh, f"mkdir -p {job}"], "mkdir")
            run([*scp, *(str(tmp / f) for f in ("target.json", "prompt.json", "prompt.wav", "fill.py", "run.py")),
                 f"{host}:{job}/"], "upload")
            run([*ssh, remote], "inference", timeout=1800)
            run([*scp, f"{host}:{job}/out/generated.wav", str(out)], "download")
        finally:
            try:
                subprocess.run([*ssh, f"rm -rf {shlex.quote(job)}"], capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                pass
        g, _ = librosa.load(str(out), sr=44100, mono=True)
    return g.astype(np.float32)
