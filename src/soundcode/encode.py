"""Encoder: audio -> .sc

Every stage is independent and fails soft. If a stage cannot produce evidence
its stream is omitted entirely and a `meta warn` is recorded on a neighbour —
per spec §4.3, an encoder must not emit a stream it has no evidence for, and
an unmarked value is a claim. Guessing quietly is the one thing we never do.

Confidence sources, per spec §4.4: librosa beat strength (grid), chroma
template margin (harmony), pyin voiced-probability (notes/contour), onset
strength percentile (percussion). Stages with no native confidence get a
stream-level estimate from a proxy check.
"""

from __future__ import annotations

import math
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

_PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_DRUM_VOICES = ("kick", "snare", "hat")

# basic-pitch reports pitch bend in 1/3-semitone bins, and an in-tune note
# reads +1 bin (measured on pure tones), so that is the zero point.
BASIC_PITCH_BEND_ZERO = 1.0


def bend_to_cents(bend) -> float:
    if bend is None or not np.size(bend):
        return 0.0
    return (float(np.mean(bend)) - BASIC_PITCH_BEND_ZERO) * 100.0 / 3.0


# Loudness gate: a 2 s block of a stem is transcribed only if it is above an
# absolute floor AND within GATE_REL_DB of the full mix. Separation leaves
# faint bleed in every stem; basic-pitch's floor is relative to the stem's own
# peak, so a near-silent stem otherwise yields confident invented notes.
GATE_ABS_DB = -50.0
GATE_REL_DB = 35.0
GATE_BLOCK_S = 2.0
AMP_FLOOR = 0.40
ONSET_THRESHOLD = 0.6
FRAME_THRESHOLD = 0.4


def _block_db(y: np.ndarray, sr: int) -> np.ndarray:
    n = max(1, int(np.ceil(len(y) / (GATE_BLOCK_S * sr))))
    out = np.full(n, -np.inf)
    for i in range(n):
        seg = y[int(i * GATE_BLOCK_S * sr): int((i + 1) * GATE_BLOCK_S * sr)]
        r = float(np.sqrt(np.mean(np.square(seg, dtype=np.float64)))) if seg.size else 0.0
        out[i] = 20 * math.log10(r) if r > 0 else -np.inf
    return out


GATE_PEAK_WIN_S = 0.1


def _block_peak_db(y: np.ndarray, sr: int) -> np.ndarray:
    """Loudest 100 ms RMS inside each block: a sparse soft note is real music
    even when its 2 s mean falls under the floor."""
    n = max(1, int(np.ceil(len(y) / (GATE_BLOCK_S * sr))))
    w = max(1, int(GATE_PEAK_WIN_S * sr))
    out = np.full(n, -np.inf)
    for i in range(n):
        seg = y[int(i * GATE_BLOCK_S * sr): int((i + 1) * GATE_BLOCK_S * sr)]
        if not seg.size:
            continue
        k = len(seg) // w
        frames = seg[: k * w].reshape(k, w) if k else seg[None, :]
        r = float(np.sqrt(np.max(np.mean(np.square(frames, dtype=np.float64), axis=1))))
        out[i] = 20 * math.log10(r) if r > 0 else -np.inf
    return out


def active_blocks(stem: np.ndarray, mix: np.ndarray, sr: int) -> np.ndarray:
    """Absolute floor on the block's loudest 100 ms; relative rule on block means."""
    s, m = _block_db(stem, sr), _block_db(mix, sr)
    peak = _block_peak_db(stem, sr)
    m = np.pad(m, (0, max(0, len(s) - len(m))), constant_values=-np.inf)[: len(s)]
    return (peak >= GATE_ABS_DB) & (s >= m - GATE_REL_DB)


def active_level_db(stem: np.ndarray, mask: np.ndarray, sr: int) -> float | None:
    parts = [stem[int(i * GATE_BLOCK_S * sr): int((i + 1) * GATE_BLOCK_S * sr)]
             for i, on in enumerate(mask) if on]
    if not parts:
        return None
    y = np.concatenate(parts)
    r = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    return 20 * math.log10(r) if r > 0 else None


def merge_same_pitch(events: list[tuple]) -> list[tuple]:
    out: list[list] = []
    last: dict[int, int] = {}
    for e in sorted(events, key=lambda e: e[0]):
        start, end, midi, amp, bend = e
        j = last.get(midi)
        if j is not None and start <= out[j][1] + 0.03:
            out[j][1] = max(out[j][1], end)
            out[j][3] = max(out[j][3], amp)
            continue
        last[midi] = len(out)
        out.append([start, end, midi, amp, bend])
    return [tuple(e) for e in out]


def needs_fallback(st: "Stage") -> bool:
    """pyin stands in only when basic-pitch could not run, never for a gated stem."""
    return not st.ok and not st.gated


def vocal_stem_name(path: Path) -> str:
    """`vocals` when encoder_stems fell back to lead+backing, else the lead stem."""
    return "vocals" if Path(path).stem == "vocals" else "lead_vocals"


def _stage_lines(st: "Stage") -> list[str]:
    fields = "".join(f" {k}={v}" for k, v in st.header_fields.items())
    out = [f":{st.name}{fields}", f"meta    src={st.src}  conf={st.conf:.2f}"]
    if st.stem:
        level = f"  level={st.level_db:.1f}dB" if st.level_db is not None else ""
        out.append(f"meta    stem={st.stem}{level}")
    out += [f'meta    warn="{w}"' for w in st.warns]
    return out + st.lines + [""]

# Populated by separate_stems() when separation fails. Stem loss silently
# costs us :notes.* and :text.* — the streams that matter most — so the reason
# is recorded in the file rather than left to be inferred from absences.
STEM_FAILURE: list[str] = []
# Separation succeeded but with caveats the .sc file must carry (a lossy sum
# check, an unsplit vocal). Written as `# NOTE:` lines in the header.
STEM_NOTES: list[str] = []

# A lead stem this far below the backing stem means the karaoke model filed
# the whole vocal as backing; melody and lyrics then use lead + backing.
_LEAD_SILENT_DB = 20.0


@dataclass
class Stage:
    """One analysis stage's output: .sc lines plus how much we trust them."""

    name: str
    lines: list[str] = field(default_factory=list)
    conf: float = 0.0
    src: str = ""
    warns: list[str] = field(default_factory=list)
    ok: bool = False
    stem: str = ""                # source stem, for the renderer and compare
    gated: bool = False           # silent under the loudness gate: no fallback
    header_fields: dict[str, str] = field(default_factory=dict)   # e.g. inst=
    level_db: float | None = None  # source stem RMS over its active blocks


def _log(msg: str) -> None:
    print(f"  {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# audio + stems
# --------------------------------------------------------------------------

def load_audio(path: str, sr: int = 44100) -> tuple[np.ndarray, int]:
    import librosa

    y, _ = librosa.load(path, sr=sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y, sr


def encoder_stems(result, notes: list[str]) -> dict[str, Path]:
    """Separation stems under the names the stages below expect. The lead
    vocal drives melody and lyrics; backing vocals are left for Milestone 2."""
    stems = {k: v for k, v in result.stems.items()
             if k in ("drums", "bass", "guitar", "piano", "other")}
    lead, backing = result.levels["lead_vocals"], result.levels["backing_vocals"]
    if lead < backing - _LEAD_SILENT_DB:
        stems["vocals"] = result.mixes["vocals"]
        notes.append(f"lead vocal stem near-silent ({lead:.0f} dB vs backing "
                     f"{backing:.0f} dB); vocals were not split into lead/backing")
    else:
        stems["vocals"] = result.stems["lead_vocals"]
    return stems


def separate_stems(path: str, workdir: Path) -> dict[str, Path]:
    """Seven-stem separation (see separate.py); fails soft into STEM_FAILURE."""
    from . import separate as sep

    _log("separating stems (RoFormer vocals, karaoke lead/backing, demucs 6-stem)")
    try:
        result = sep.separate(path, workdir / "stems")
    except Exception as exc:  # fail soft on anything: encode from the mix
        reason = str(exc) if isinstance(exc, sep.SeparationError) \
            else f"{type(exc).__name__}: {exc}"
        _log(f"separation FAILED: {reason}")
        STEM_FAILURE.append(reason)
        return {}
    if not result.report.ok:
        r = result.report
        STEM_NOTES.append(f"stem sum check failed (level diff {r.level_diff_db:+.1f} dB, "
                          f"residual {r.residual_db:.1f} dB); stem-derived streams "
                          "are less reliable")
        _log(STEM_NOTES[-1])
    return encoder_stems(result, STEM_NOTES)


def _module_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def stage_grid(y: np.ndarray, sr: int, duration: float) -> tuple[Stage, dict]:
    """Tempo curve, meter and bar anchors."""
    import librosa

    st = Stage("grid", src="librosa:beat_track")
    mono = y.mean(0)
    try:
        onset_env = librosa.onset.onset_strength(y=mono, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr,
                                               units="time", trim=False)
        tempo = float(np.atleast_1d(tempo)[0])
        if not beats.size or tempo <= 0:
            st.warns.append("no stable pulse detected")
            return st, {}
    except Exception as exc:                             # noqa: BLE001
        st.warns.append(f"beat tracking failed: {exc}")
        return st, {}

    # confidence: how peaked the onset envelope is at detected beats
    frames = librosa.time_to_frames(beats, sr=sr)
    frames = frames[(frames >= 0) & (frames < onset_env.size)]
    strength = float(np.mean(onset_env[frames])) if frames.size else 0.0
    st.conf = float(np.clip(strength / (np.percentile(onset_env, 95) + 1e-9), 0, 1))

    downbeat = float(beats[0])
    beats_per_bar = 4
    bar_dur = beats_per_bar * 60.0 / tempo

    st.lines = [f"meter   @{downbeat:.3f}   4/4"]
    # anchor at least every 16 bars so drift cannot accumulate (spec §4.3.1)
    n_bars = max(int((duration - downbeat) / bar_dur), 1)
    for bar in range(1, n_bars + 2, 16):
        st.lines.append(f"anchor  bar {bar}    @{downbeat + (bar - 1) * bar_dur:.3f}")

    # One tempo: the one every note below is positioned with (downbeat +
    # (bar-1) * bar_dur). Per-window tempo estimates used to be written here as
    # a curve, but notes were never placed on it, so the renderer drifted by
    # seconds against the transcription. A real tempo curve comes back with a
    # beat/downbeat tracker whose beats the notes are also placed on.
    st.lines.append(f"tempo   @0.000   {tempo:.2f}")

    st.ok = True
    return st, {"tempo": tempo, "beats": beats, "downbeat": downbeat,
                "bar_dur": bar_dur, "n_bars": n_bars}


def stage_struct(y: np.ndarray, sr: int, grid: dict, duration: float) -> Stage:
    """Sections via spectral novelty. Plan B from the spec — allin1 is fragile."""
    import librosa

    st = Stage("struct", src="librosa:novelty")
    mono = y.mean(0)
    try:
        chroma = librosa.feature.chroma_cqt(y=mono, sr=sr)
        bounds = librosa.segment.agglomerative(chroma, k=max(2, min(6, int(duration // 12))))
        times = librosa.frames_to_time(bounds, sr=sr)
    except Exception as exc:                             # noqa: BLE001
        st.warns.append(f"segmentation failed: {exc}")
        return st

    edges = sorted({0.0, *[float(t) for t in times], duration})
    rms = librosa.feature.rms(y=mono)[0]
    peak = float(np.max(rms)) or 1.0

    bar_dur = grid.get("bar_dur", 2.0)
    downbeat = grid.get("downbeat", 0.0)

    def to_bar(t: float) -> int:
        return max(int(round((t - downbeat) / bar_dur)) + 1, 1)

    labels = ["intro", "verse", "chorus", "verse", "chorus", "outro", "coda"]
    for i, (a, b) in enumerate(zip(edges, edges[1:])):
        if b - a < 3.0:
            continue
        seg = rms[int(a / duration * rms.size):max(int(b / duration * rms.size), 1)]
        energy = float(np.mean(seg) / peak) if seg.size else 0.0
        label = labels[min(i, len(labels) - 1)]
        st.lines.append(
            f"{label:<8}{to_bar(a)}-{to_bar(b) - 1:<6} "
            f"{'vocal' if energy > 0.45 else 'inst':<6} energy={energy:.2f}"
        )
    st.conf = 0.55          # segmentation without a trained model is weak; say so
    st.ok = bool(st.lines)
    if st.ok:
        st.warns.append("sections from spectral novelty only; labels are positional")
    return st


def stage_harmony(y: np.ndarray, sr: int, grid: dict) -> Stage:
    """Bar-level chords by chroma template matching."""
    import librosa

    st = Stage("harmony", src="librosa:chroma-template")
    if not grid:
        st.warns.append("no grid — chords need bar positions")
        return st
    mono = y.mean(0)
    try:
        chroma = librosa.feature.chroma_cqt(y=mono, sr=sr)
    except Exception as exc:                             # noqa: BLE001
        st.warns.append(f"chroma failed: {exc}")
        return st

    maj = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
    minr = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)
    templates, names = [], []
    for pc in range(12):
        templates.append(np.roll(maj, pc)); names.append(f"{_PC_NAMES[pc]}")
        templates.append(np.roll(minr, pc)); names.append(f"{_PC_NAMES[pc]}m")
    T = np.array(templates)
    T /= np.linalg.norm(T, axis=1, keepdims=True)

    bar_dur, downbeat = grid["bar_dur"], grid["downbeat"]
    chords, confs = [], []
    for bar in range(grid["n_bars"]):
        t0 = downbeat + bar * bar_dur
        f0 = librosa.time_to_frames(t0, sr=sr)
        f1 = librosa.time_to_frames(t0 + bar_dur, sr=sr)
        seg = chroma[:, max(f0, 0):max(f1, f0 + 1)]
        if not seg.size:
            chords.append(None); confs.append(0.0); continue
        v = seg.mean(1)
        n = np.linalg.norm(v)
        if n < 1e-6:
            chords.append(None); confs.append(0.0); continue
        scores = T @ (v / n)
        best = int(np.argmax(scores))
        runner = float(np.sort(scores)[-2])
        margin = float(scores[best]) - runner          # separation == confidence
        chords.append(names[best])
        confs.append(float(np.clip(0.5 + margin * 3.0, 0.0, 0.99)))

    # emit in rows of four bars, marking any chord we are not sure about
    for start in range(0, len(chords), 4):
        row = chords[start:start + 4]
        rc = confs[start:start + 4]
        cells = []
        for ch, cf in zip(row, rc):
            if ch is None:
                cells.append("N")
            elif cf < 0.80:
                cells.append(f"{ch} ?{cf:.2f}")
            else:
                cells.append(ch)
        st.lines.append(f"bars {start + 1}-{start + len(row)}   " + " | ".join(cells))

    st.conf = float(np.mean([c for c in confs if c])) if any(confs) else 0.0
    st.ok = bool(st.lines)
    return st


def stage_percussion(stem: Path | None, y: np.ndarray, sr: int, grid: dict) -> Stage:
    """Onsets on the drum stem, classified into kick/snare/hat by band energy."""
    import librosa

    st = Stage("perc.drums", src="librosa:onset+bandsplit")
    if not grid:
        st.warns.append("no grid — percussion events need bar positions")
        return st
    mask = None
    if stem is not None:
        d, _ = librosa.load(str(stem), sr=sr, mono=True)
        st.stem = "drums"
        mask = active_blocks(d, y.mean(0), sr)
        st.level_db = active_level_db(d, mask, sr)
        if not mask.any():
            st.warns.append("stem silent (below the loudness gate throughout)")
            return st
    else:
        d = y.mean(0)
        st.warns.append("no drum stem; onsets taken from the full mix")

    onsets = librosa.onset.onset_detect(y=d, sr=sr, units="time", backtrack=True)
    if mask is not None:
        onsets = [t for t in onsets if mask[min(int(t / GATE_BLOCK_S), len(mask) - 1)]]
    if not len(onsets):
        st.warns.append("no onsets found")
        return st

    S = np.abs(librosa.stft(d, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low, mid, high = freqs < 150, (freqs >= 150) & (freqs < 1200), freqs >= 6000

    bar_dur, downbeat = grid["bar_dur"], grid["downbeat"]
    events: list[str] = []
    for t in onsets:
        f = librosa.time_to_frames(t, sr=sr)
        if f < 0 or f >= S.shape[1]:
            continue
        band = S[:, f]
        e = np.array([band[low].sum(), band[mid].sum(), band[high].sum()])
        if e.sum() <= 0:
            continue
        voice = _DRUM_VOICES[int(np.argmax(e))]
        vel = int(np.clip(20 + 107 * (e.sum() / (S.sum(0).max() + 1e-9)) ** 0.4, 1, 127))

        pos = (t - downbeat) / bar_dur
        bar = int(pos) + 1
        beat = (pos - int(pos)) * 4 + 1
        if bar < 1:
            continue
        grid_t = downbeat + (bar - 1) * bar_dur + (round((beat - 1) * 4) / 4) * (bar_dur / 4)
        dev = (t - grid_t) * 1000.0
        snapped = round((beat - 1) * 4) / 4 + 1
        if abs(dev) > 120:                     # too far off-grid to attribute
            continue
        events.append(f"{bar}:{snapped:.2f} {voice} {vel} dev{dev:+.0f}ms")

    for i in range(0, len(events), 4):
        st.lines.append(" | ".join(events[i:i + 4]))
    st.conf = 0.62 if stem is not None else 0.40
    st.ok = bool(st.lines)
    return st


def stage_notes(stem: Path | None, name: str, sr: int, grid: dict,
                fmin: str, fmax: str, inst: str) -> tuple[Stage, list]:
    """Monophonic note events via pyin. Voiced-probability is the confidence."""
    import librosa

    st = Stage(f"notes.{name}", src="librosa:pyin")
    if stem is None or not grid:
        st.warns.append("stem or grid unavailable")
        return st, []
    y, _ = librosa.load(str(stem), sr=sr, mono=True)
    if float(np.abs(y).max()) < 1e-3:
        st.warns.append("stem is silent")
        return st, []
    try:
        f0, voiced, vprob = librosa.pyin(
            y, sr=sr, fmin=librosa.note_to_hz(fmin), fmax=librosa.note_to_hz(fmax),
            frame_length=2048)
    except Exception as exc:                             # noqa: BLE001
        st.warns.append(f"pyin failed: {exc}")
        return st, []

    times = librosa.times_like(f0, sr=sr)
    rms = librosa.feature.rms(y=y, frame_length=2048)[0]
    rms = np.interp(np.linspace(0, 1, f0.size), np.linspace(0, 1, rms.size), rms)
    peak_rms = float(rms.max()) or 1.0

    # group contiguous voiced frames of similar pitch into notes
    notes, cur = [], None
    for i, (t, hz, v) in enumerate(zip(times, f0, voiced)):
        if not v or not np.isfinite(hz):
            if cur:
                notes.append(cur); cur = None
            continue
        cents = 1200 * math.log2(hz / 440.0) + 6900
        if cur and abs(cents - cur["cents"][-1]) < 60:
            cur["cents"].append(cents); cur["end"] = t
            cur["conf"].append(float(vprob[i])); cur["rms"].append(float(rms[i]))
        else:
            if cur:
                notes.append(cur)
            cur = {"start": float(t), "end": float(t), "cents": [cents],
                   "conf": [float(vprob[i])], "rms": [float(rms[i])]}
    if cur:
        notes.append(cur)

    bar_dur, downbeat = grid["bar_dur"], grid["downbeat"]
    beat_dur = bar_dur / 4
    kept = []
    for n in notes:
        dur = n["end"] - n["start"]
        if dur < 0.08:
            continue
        cents = float(np.median(n["cents"]))
        conf = float(np.mean(n["conf"]))
        vel = int(np.clip(20 + 107 * (float(np.mean(n["rms"])) / peak_rms) ** 0.5, 1, 127))
        pos = (n["start"] - downbeat) / bar_dur
        bar = int(pos) + 1
        if bar < 1:
            continue
        beat = round((pos - int(pos)) * 4 * 2) / 2 + 1
        beats = max(round(dur / beat_dur * 2) / 2, 0.5)
        from .pitch import cents_to_name
        atom = cents_to_name(round(cents))
        mark = "" if conf >= 0.80 else f" ?{conf:.2f}"
        kept.append(f"{bar}:{beat:.1f}  {atom:<8} {beats:.1f}b {vel}{mark}")
        st.lines.append(kept[-1])

    st.conf = float(np.mean([float(np.mean(n["conf"])) for n in notes])) if notes else 0.0
    st.ok = bool(st.lines)
    return st, notes


def stage_notes_poly(stem: Path | None, name: str, grid: dict,
                     min_amp: float = AMP_FLOOR, mix: np.ndarray | None = None,
                     sr: int = 44100, stem_name: str = "") -> Stage:
    """Polyphonic note transcription via basic-pitch.

    pyin is monophonic and returns noise on real polyphonic material — it was
    scoring 0.02-0.12 voiced-probability on actual songs. basic-pitch is a
    trained polyphonic model and is what the spec called for; this replaces
    pyin wherever it is importable.
    """
    st = Stage(f"notes.{name}", src="basic-pitch:onnx")
    if stem is None or not grid:
        st.warns.append("stem or grid unavailable")
        return st
    try:
        from basic_pitch.inference import predict
    except Exception as exc:                             # noqa: BLE001
        st.warns.append(f"basic-pitch unavailable ({type(exc).__name__})")
        return st

    try:
        _, _, events = predict(str(stem), onset_threshold=ONSET_THRESHOLD,
                               frame_threshold=FRAME_THRESHOLD)
    except Exception as exc:                             # noqa: BLE001
        st.warns.append(f"basic-pitch failed: {exc}")
        return st
    if not events:
        st.warns.append("no notes detected")
        return st

    st.stem = stem_name
    if mix is not None:
        import librosa
        y_stem, _ = librosa.load(str(stem), sr=sr, mono=True)
        mask = active_blocks(y_stem, mix, sr)
        st.level_db = active_level_db(y_stem, mask, sr)
        if not mask.any():
            st.warns = ["stem silent (below the loudness gate throughout)"]
            st.gated = True
            return st
        events = [e for e in events
                  if mask[min(int(e[0] / GATE_BLOCK_S), len(mask) - 1)]]
    events = merge_same_pitch(events)
    if not events:
        st.warns.append("no notes above the loudness gate")
        st.gated = mix is not None
        return st

    from .pitch import cents_to_name

    bar_dur, downbeat = grid["bar_dur"], grid["downbeat"]
    beat_dur = bar_dur / 4
    amps = [e[3] for e in events]
    peak = max(amps) or 1.0

    kept = 0
    for start, end, midi, amp, bend in sorted(events, key=lambda e: e[0]):
        if amp < min_amp * peak:
            continue
        dur = end - start
        if dur < 0.05:
            continue
        pos = (start - downbeat) / bar_dur
        bar = int(pos) + 1
        if bar < 1:
            continue
        beat = round((pos - int(pos)) * 4 * 2) / 2 + 1
        beats = max(round(dur / beat_dur * 2) / 2, 0.5)
        cents = midi * 100 + bend_to_cents(bend)
        vel = int(np.clip(20 + 107 * (amp / peak) ** 0.5, 1, 127))
        conf = float(np.clip(amp / peak, 0, 1))
        mark = "" if conf >= 0.80 else f" ?{conf:.2f}"
        st.lines.append(f"{bar}:{beat:.1f}  {cents_to_name(round(cents)):<8} "
                        f"{beats:.1f}b {vel}{mark}")
        kept += 1

    st.conf = float(np.mean([min(a / peak, 1.0) for a in amps]))
    st.ok = kept > 0
    if st.ok:
        st.warns.append(f"{kept} of {len(events)} detected notes kept "
                        f"(amplitude floor {min_amp:.2f} of peak)")
    return st


def stage_tsumugi(stems: dict[str, Path], mix_path: Path, mix: np.ndarray, sr: int,
                  grid: dict, work: Path) -> tuple[list[Stage], set[str]]:
    """tsumugi per stem + a mix-level vote -> :instruments, note streams, drums.

    Stems that fail (TsumugiError) are left out of `handled`; encode() runs
    the basic-pitch path for those, so one bad stem never loses the rest.
    """
    import librosa
    import soundfile as sf

    from . import gm
    from . import inventory as inv
    from . import tsumugi as ts
    from . import tsumugi_sc as tsc

    if not grid:
        return [], set()
    work.mkdir(parents=True, exist_ok=True)
    handled: set[str] = set()
    inventory = Stage("instruments", src=f"tsumugi@{ts.PINNED_COMMIT}+mix-vote")
    stages: list[Stage] = []
    taken: set[str] = set()
    confs: list[float] = []

    if not mix_path.exists():
        sf.write(str(mix_path), mix, sr)
    try:
        mix_tracks = tsc.read_tracks(ts.transcribe(mix_path, work / "mix.mid", "default"))
    except ts.TsumugiError as exc:
        STEM_NOTES.append(f"tsumugi mix-level vote unavailable ({exc})")
        mix_tracks = []

    for stem, path in stems.items():
        y_stem, _ = librosa.load(str(path), sr=sr, mono=True)
        mask = active_blocks(y_stem, mix, sr)
        level = active_level_db(y_stem, mask, sr)
        if not mask.any():
            inventory.lines.append(f"# {stem}: silent (below the loudness gate)")
            handled.add(stem)
            continue
        try:
            midi = ts.transcribe(path, work / f"{stem}.mid", ts.STEM_MODEL[stem])
            try:
                midi = ts.add_velocity(midi, path, work / f"{stem}.vel.mid")
            except ts.TsumugiError as exc:
                STEM_NOTES.append(f"{stem}: velocity model failed ({exc}); velocities are flat")
            tracks = tsc.read_tracks(midi)
        except ts.TsumugiError as exc:
            STEM_NOTES.append(f"tsumugi failed on {stem} ({exc}); basic-pitch used")
            continue
        handled.add(stem)

        if stem == "drums":
            tracks, why = tsc.drop_bleed(tsc.merge_same_inst(tracks))
            st = Stage("perc.drums", src=f"tsumugi:drums_v1_5@{ts.PINNED_COMMIT}", conf=0.69,
                       stem="drums", level_db=level)
            st.header_fields = {"inst": "drums.kit"}
            st.warns += why
            for t in tracks:
                st.lines += tsc.note_lines(t, grid)
            st.ok = bool(st.lines)
            stages.append(st)
            inventory.lines.append(f"drums    {'drums.kit' if st.ok else 'none'}")
            continue

        prior = inv.STEM_FAMILY.get(stem, "other")
        if not tracks:
            inventory.lines.append(f"{stem:<8} none (no notes)")
            continue
        try:
            top = ts.refine(path, midi, ts.REFINE_STEM.get(stem, "other"),
                            work / f"{stem}.refine.json")
        except ts.TsumugiError as exc:
            top = []
            STEM_NOTES.append(f"{stem}: refinement failed ({exc})")
        decision = inv.decide(stem, prior, top,
                              inv.mix_votes(tracks, mix_tracks, prefer=prior))
        if decision.reassigned:
            again = inv.FAMILY_REFINE_STEM.get(decision.family, "other")
            try:
                top = ts.refine(path, midi, again, work / f"{stem}.refine.{again}.json")
                decision.conf *= top[0][1] if top else 1.0
            except ts.TsumugiError:
                top = []
        # the stem's refined class relabels its tracks when it fits the decided family
        klass = top[0][0] if top and gm.TSUMUGI[top[0][0]]["family"] == decision.family else None
        if klass is not None:
            for t in tracks:
                t.klass, t.inst = klass, gm.TSUMUGI[klass]["inst"]
        # bleed is judged once, after relabelling: a stem's own instrument
        # split over several program tracks is one part, not bleed of others
        tracks, why = tsc.drop_bleed(tsc.merge_same_inst(tracks))
        if not tracks:
            stages.append(Stage(f"notes.{stem}", warns=why))       # "omitted — …"
            inventory.lines.append(f"{stem:<8} none (bleed)")
            continue
        for t in tracks:
            st = Stage(tsc.stream_name(t.inst, stem, taken),
                       src=f"tsumugi:{ts.STEM_MODEL[stem]}@{ts.PINNED_COMMIT}",
                       conf=decision.conf, stem=stem, level_db=level)
            st.header_fields = {"inst": t.inst}
            if decision.warn:
                st.warns.append(decision.warn)
            st.warns += why
            st.lines = tsc.note_lines(t, grid)
            st.ok = bool(st.lines)
            stages.append(st)
        label = tracks[0].inst if tracks else "none"
        mark = "" if decision.conf >= 0.80 else f" ?{decision.conf:.2f}"
        runner = f"   | {top[1][0]} {top[1][1]:.2f}" if len(top) > 1 else ""
        inventory.lines.append(f"{stem:<8} {label}{mark}{runner}")
        if decision.warn:
            inventory.warns.append(decision.warn)
        confs.append(decision.conf)

    inventory.conf = float(np.mean(confs)) if confs else 0.0
    inventory.ok = bool(inventory.lines)
    return ([inventory] + stages if inventory.ok else stages), handled


def stage_mix(y: np.ndarray, sr: int) -> Stage:
    """Measured production scalars — evidence for the bridge's prose."""
    import librosa

    st = Stage("mix", src="librosa:spectral")
    mono = y.mean(0)
    rms = librosa.feature.rms(y=mono)[0]
    lufs = 20 * math.log10(float(np.sqrt(np.mean(mono ** 2))) + 1e-9)
    dyn = 20 * math.log10((float(np.percentile(rms, 95)) + 1e-9) /
                          (float(np.percentile(rms, 10)) + 1e-9))

    S = np.abs(librosa.stft(mono, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    bands = {"low": freqs < 250, "mid": (freqs >= 250) & (freqs < 4000), "high": freqs >= 4000}

    st.lines.append(f"lufs_int      {lufs:.1f}")
    st.lines.append(f"lufs_range    {dyn:.1f}")
    if y.shape[0] == 2:
        widths = []
        L, R = np.abs(librosa.stft(y[0], n_fft=2048)), np.abs(librosa.stft(y[1], n_fft=2048))
        for nm, sel in bands.items():
            num = float(np.mean(np.abs(L[sel] - R[sel])))
            den = float(np.mean(np.abs(L[sel] + R[sel]))) + 1e-9
            widths.append(f"{nm} {np.clip(num / den, 0, 1):.2f}")
        st.lines.append("stereo_width  " + ", ".join(widths))
    cent = float(np.mean(librosa.feature.spectral_centroid(y=mono, sr=sr)))
    roll = float(np.mean(librosa.feature.spectral_rolloff(y=mono, sr=sr, roll_percent=0.95)))
    st.lines.append(f"centroid      {cent:.0f}Hz")
    st.lines.append(f"rolloff_95    {roll:.0f}Hz")
    st.conf = 0.95              # these are measurements, not inferences
    st.ok = True
    return st


def stage_contour(stem: Path | None, stem_name: str, mix: np.ndarray, sr: int) -> Stage:
    """The lead vocal's f0 curve (:contour.vox), gated like every stem."""
    import librosa

    from . import contour as ct

    st = Stage("contour.vox", src="torchcrepe:full", stem=stem_name)
    st.header_fields = {"rate": str(ct.RATE_HZ)}
    if stem is None:
        st.warns.append("no vocal stem")
        return st
    y, _ = librosa.load(str(stem), sr=sr, mono=True)
    mask = active_blocks(y, mix, sr)
    if not mask.any():
        st.warns.append("vocal stem silent (below the loudness gate)")
        return st
    try:
        times, cents, voiced, conf = ct.extract(stem)
    except Exception as exc:                             # noqa: BLE001
        st.warns.append(f"f0 extraction failed: {exc}")
        return st
    # blank frames in gated blocks, so bleed never becomes a contour
    blk = np.minimum((times / GATE_BLOCK_S).astype(int), len(mask) - 1)
    voiced = voiced & mask[blk]
    st.lines = ct.contour_lines(ct.phrases(times, cents, voiced))
    st.conf, st.ok = conf, bool(st.lines)
    return st


def lyric_cells(words: list[tuple[float, float, str, float]], grid: dict) -> list[str]:
    """Words at performed timing (3-decimal beats), with durations."""
    from . import tsumugi_sc as tsc

    beat_s = grid["bar_dur"] / 4
    cells, last = [], None
    for start, end, word, prob in sorted(words, key=lambda w: w[0]):
        if not word:
            continue
        if last is not None:
            start = max(start, last + 0.02)            # collisions: nudge 20 ms apart
        last = start
        pos = tsc.position(start, grid)
        dur = max(end - start, 0.05)
        d = f"{dur:.3f}s" if pos.startswith("@") else f"{dur / beat_s:.3f}b"
        mark = "" if prob >= 0.80 else f" ?{prob:.2f}"
        cells.append(f'{pos} "{word}" {d}{mark}')
    return cells


def stage_lyrics(stem: Path | None, sr: int, grid: dict) -> Stage:
    """Word-level lyrics from the vocal stem, if a Whisper backend is present."""
    st = Stage("text.vox", src="whisper")
    if stem is None or not grid:
        st.warns.append("no vocal stem")
        return st
    if not _module_available("whisper") and not _module_available("faster_whisper"):
        st.warns.append("no whisper backend installed")
        return st
    try:
        if _module_available("faster_whisper"):
            from faster_whisper import WhisperModel
            model = WhisperModel("base", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(str(stem), word_timestamps=True)
            words = [(w.start, w.end, w.word.strip(), w.probability)
                     for s in segments for w in (s.words or [])]
        else:
            import whisper
            model = whisper.load_model("base")
            res = model.transcribe(str(stem), word_timestamps=True)
            words = [(w["start"], w["end"], w["word"].strip(), w.get("probability", 0.8))
                     for s in res["segments"] for w in s.get("words", [])]
    except Exception as exc:                             # noqa: BLE001
        st.warns.append(f"transcription failed: {exc}")
        return st

    cells = lyric_cells(words, grid)
    for i in range(0, len(cells), 5):
        st.lines.append(" | ".join(cells[i:i + 5]))
    st.conf = float(np.mean([w[3] for w in words])) if words else 0.0
    st.ok = bool(st.lines)
    return st


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def encode(path: str, out_path: str | None = None,
           workdir: str | None = None, title: str | None = None) -> str:
    """Analyse `path` and write a .sc file. Returns the .sc text."""
    src = Path(path)
    wd = Path(workdir or tempfile.mkdtemp(prefix="sc-")) / (src.stem + ".scw")
    wd.mkdir(parents=True, exist_ok=True)
    STEM_FAILURE.clear()
    STEM_NOTES.clear()

    _log(f"loading {src.name}")
    y, sr = load_audio(str(src))
    duration = y.shape[1] / sr

    stems = separate_stems(str(src), wd)
    if stems:
        _log(f"stems: {', '.join(sorted(stems))}")

    _log("grid")
    grid_st, grid = stage_grid(y, sr, duration)
    _log("structure"); struct_st = stage_struct(y, sr, grid, duration)
    _log("harmony");   harm_st = stage_harmony(y, sr, grid)
    mono_mix = y.mean(0)
    # tsumugi transcribes every stem it can; basic-pitch covers the rest
    ts_stems = {(vocal_stem_name(v) if k == "vocals" else k): v for k, v in stems.items()}
    _log("tsumugi (notes, drums, instrument inventory)")
    ts_stages, handled = (stage_tsumugi(ts_stems, wd / "mix.wav", mono_mix, sr, grid,
                                        wd / "tsumugi") if stems else ([], set()))
    vox_name = vocal_stem_name(stems["vocals"]) if "vocals" in stems else ""
    _log("vocal f0 contour")
    contour_st = stage_contour(stems.get("vocals"), vox_name or "lead_vocals", mono_mix, sr)
    skipped = Stage("skipped")                       # ok=False, no warns: not written

    _log("fallback transcription for stems tsumugi did not handle")
    perc_st = skipped if "drums" in handled else stage_percussion(stems.get("drums"), y, sr, grid)
    if "bass" in handled:
        bass_st = skipped
    else:
        bass_st = stage_notes_poly(stems.get("bass"), "bass", grid, mix=mono_mix, sr=sr,
                                   stem_name="bass")
        if needs_fallback(bass_st):
            bass_st, _ = stage_notes(stems.get("bass"), "bass", sr, grid, "E1", "E4",
                                     "bass.electric")
    if vox_name in handled:
        vox_st = skipped
    else:
        vox_st = stage_notes_poly(stems.get("vocals"), "vox", grid, mix=mono_mix, sr=sr,
                                  stem_name=vox_name)
        if needs_fallback(vox_st):
            vox_st, _ = stage_notes(stems.get("vocals"), "vox", sr, grid, "C2", "C6",
                                    "voice.lead")
    other_st, guitar_st, piano_st = (
        skipped if n in handled else
        stage_notes_poly(stems.get(n), n, grid, mix=mono_mix, sr=sr, stem_name=n)
        for n in ("other", "guitar", "piano"))
    _log("lyrics");    text_st = stage_lyrics(stems.get("vocals"), sr, grid)
    _log("mix");       mix_st = stage_mix(y, sr)

    bpm = grid.get("tempo", 0.0)
    key = _estimate_key(y, sr)

    lines: list[str] = [
        "%sc        0.3",
        "%profile   metric-tonal",
        "%residual  none",
        "",
        f'@title     "{title or src.stem}"',
        f'@source    {src.name}',
        "@offset    0.000",
        f"@duration  {duration:.3f}",
        f"@sr        {sr}",
        "",
        "# @style and @mix prose are written by the decode bridge from the",
        "# measured evidence below. Left empty by the encoder on purpose:",
        "# asserting a style we did not measure would be a claim (spec 4.4).",
        '@style     ""',
        "",
    ]

    if STEM_FAILURE:
        lines += [
            "# WARNING: stem separation failed, so :notes.*, :contour.* and",
            "# :text.* could not be produced. Everything below was derived from",
            "# the full mix and is correspondingly less reliable.",
            f"#   reason: {STEM_FAILURE[0]}",
            "",
        ]

    for note in STEM_NOTES:
        lines.append(f"# NOTE: {note}")
    if STEM_NOTES:
        lines.append("")

    if key:
        lines += [":tuning", "ref          A4 = 440.0Hz", "temperament  12tet", ""]

    for st in (grid_st, struct_st, harm_st, *ts_stages, contour_st, perc_st, bass_st, vox_st,
               guitar_st, piano_st, other_st, text_st, mix_st):
        if not st.ok:
            if st.warns:
                lines.append(f"# :{st.name} omitted — {'; '.join(st.warns)}")
            continue
        lines += _stage_lines(st)

    text = "\n".join(lines) + "\n"
    dest = Path(out_path) if out_path else src.with_suffix(".sc")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    _log(f"wrote {dest}  ({len(text)} bytes, tempo {bpm:.1f}, key {key or '?'})")
    return text


def _estimate_key(y: np.ndarray, sr: int) -> str:
    """Krumhansl-style key estimate. Reported, never used to override notes."""
    import librosa

    try:
        chroma = librosa.feature.chroma_cqt(y=y.mean(0), sr=sr).mean(1)
    except Exception:                                    # noqa: BLE001
        return ""
    maj = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minr = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    best, score = "", -1.0
    for pc in range(12):
        for name, prof in (("", maj), ("m", minr)):
            r = float(np.corrcoef(chroma, np.roll(prof, pc))[0, 1])
            if r > score:
                score, best = r, f"{_PC_NAMES[pc]}{name}"
    return best
