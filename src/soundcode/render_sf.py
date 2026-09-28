"""Sampled renderer: .sc -> audio through a General MIDI SoundFont.

Each stream is rendered on its own (so `compare` can score it against its
source stem), then mixed. Pitch: chords round to the nearest semitone; a
monophonic line keeps its cents offset as pitch bend, so tuning survives.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import gm
from .expand import Note, expand
from .model import Document

BEND_RANGE_SEMITONES = 2
_ORDER = {"off": 0, "bend": 1, "on": 2}


@dataclass(frozen=True, order=True)
class MidiEvent:
    time: float
    order: int
    kind: str
    key: int = 0
    value: int = 0


def _event(time: float, kind: str, key: int = 0, value: int = 0) -> MidiEvent:
    return MidiEvent(time, _ORDER[kind], kind, key, value)


def stream_events(notes: list[Note], target: gm.Target) -> list[MidiEvent]:
    mono = target.family in gm.MONO_FAMILIES
    out: list[MidiEvent] = []
    for n in notes:
        if target.drums:
            key = gm.drum_note(n.voice or "")
        elif n.cents is None:
            continue
        else:
            key = int(round(n.cents / 100.0))
            if mono:
                offset = (n.cents - key * 100) / 100.0          # semitones
                value = 8192 + round(offset / BEND_RANGE_SEMITONES * 8192)
                out.append(_event(n.start, "bend", value=max(0, min(16383, value))))
        key = max(0, min(127, key))
        out.append(_event(n.start, "on", key, max(1, min(127, n.vel))))
        out.append(_event(n.start + max(n.dur, 0.02), "off", key))
    return out


SF2_URL = "https://github.com/mrbumpy409/GeneralUser-GS/raw/main/GeneralUser-GS.sf2"
SF2_SHA256 = "9575028c7a1f589f5770fccc8cff2734566af40cd26ed836944e9a5152688cfe"
DEFAULT_SF2 = Path("models") / "soundfonts" / "GeneralUser-GS.sf2"
_TAIL_S = 1.5          # release tails after the last note
_CHUNK = 512           # samples per generate() call between events


class SoundFontError(RuntimeError):
    """No usable SoundFont; the message says how to get one."""


def _download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as r, dest.open("wb") as fh:
        fh.write(r.read())


def soundfont_path() -> Path:
    override = os.environ.get("SOUNDCODE_SOUNDFONT")
    if override:
        p = Path(override)
        if not p.is_file():
            raise SoundFontError(f"SOUNDCODE_SOUNDFONT={p} does not exist")
        return p
    p = DEFAULT_SF2
    if p.is_file():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    part = p.with_suffix(".part")
    try:
        _download(SF2_URL, part)
    except Exception as exc:  # offline, HTTP error
        part.unlink(missing_ok=True)
        raise SoundFontError(f"could not download the SoundFont ({exc}); set "
                             "SOUNDCODE_SOUNDFONT to any .sf2, or use --engine mock") from exc
    digest = hashlib.sha256(part.read_bytes()).hexdigest()
    if digest != SF2_SHA256:
        part.unlink(missing_ok=True)
        raise SoundFontError(f"SoundFont download failed its checksum ({digest[:12]}…); "
                             "set SOUNDCODE_SOUNDFONT to any .sf2, or use --engine mock")
    part.rename(p)
    return p


def _synth_stream(events: list[MidiEvent], target: gm.Target, sfpath: Path,
                  sr: int, n: int) -> np.ndarray:
    import tinysoundfont

    synth = tinysoundfont.Synth(samplerate=sr)
    sfid = synth.sfload(str(sfpath))
    ch = 9 if target.drums else 0
    synth.program_select(ch, sfid, target.bank, target.preset, target.drums)
    synth.pitchbend_range(ch, BEND_RANGE_SEMITONES)
    out = np.zeros((n, 2), dtype=np.float32)
    pos = 0
    for ev in sorted(events):
        stop = min(int(ev.time * sr), n)
        while pos < stop:
            k = min(_CHUNK, stop - pos)
            out[pos:pos + k] = np.frombuffer(synth.generate(k), dtype=np.float32).reshape(-1, 2)
            pos += k
        if ev.kind == "on":
            synth.noteon(ch, ev.key, ev.value)
        elif ev.kind == "off":
            synth.noteoff(ch, ev.key)
        else:
            synth.pitchbend(ch, ev.value)
    while pos < n:
        k = min(_CHUNK, n - pos)
        out[pos:pos + k] = np.frombuffer(synth.generate(k), dtype=np.float32).reshape(-1, 2)
        pos += k
    return out


# Must equal encode.GATE_BLOCK_S: `meta level` is the stem's RMS over the 2 s
# blocks where it plays, gaps inside those blocks included, so the render is
# measured the same way. (Measuring only sounding samples biased every sparse
# part low by 10*log10(sounding fraction).)
LEVEL_BLOCK_S = 2.0


def _rms_db(y: np.ndarray, sr: int) -> float:
    b = max(1, int(LEVEL_BLOCK_S * sr))
    blocks = [y[i:i + b] for i in range(0, len(y), b)]
    active = [x for x in blocks if x.size and np.abs(x).max() > 1e-4]
    if not active:
        return float("-inf")
    a = np.concatenate(active).astype(np.float64)
    return 20 * float(np.log10(np.sqrt(np.mean(a ** 2))))


def render_streams(doc: Document, sr: int | None = None,
                   sf2: Path | None = None, no_fx: bool = False) -> dict[str, np.ndarray]:
    sr = sr or doc.sample_rate
    sfpath = sf2 or soundfont_path()
    notes = expand(doc)
    total = doc.duration or (max((x.start + x.dur for x in notes), default=1.0))
    n = int((total + _TAIL_S) * sr)
    by_stream: dict[str, list[Note]] = {}
    for x in notes:
        by_stream.setdefault(x.stream, []).append(x)

    out: dict[str, np.ndarray] = {}
    for name, stream_notes in by_stream.items():
        target = gm.target_for(name, stream_notes[0].inst)
        s = doc.stream(name)
        y = _drums_from_kit(doc, s, stream_notes, target, sfpath, sr, n)
        if y is None:
            y = _synth_stream(stream_events(stream_notes, target), target, sfpath, sr, n)
        if not no_fx:
            from .fx import apply as apply_fx, parse_fx
            f = parse_fx(s)
            if f is not None:
                y = apply_fx(y, sr, f)
        level = s.meta.get("level") if s is not None else None
        if level is not None:
            have = _rms_db(y, sr)
            if np.isfinite(have):
                y *= 10 ** ((float(level.rstrip("dB")) - have) / 20)
        out[name] = y
    return out


def _drums_from_kit(doc, s, stream_notes, target, sfpath, sr: int, n: int):
    """A perc stream with `meta kit` plays the song's own one-shots; voices the
    kit lacks fall back to the General MIDI kit. None = no kit, render as usual."""
    if s is None or s.kind != "perc" or not s.meta.get("kit"):
        return None
    kit_dir = Path(s.meta["kit"])
    if not kit_dir.is_absolute() and getattr(doc, "path", None):
        kit_dir = Path(doc.path).parent / kit_dir
    if not kit_dir.is_dir():
        s.warns.append(f"drum kit {kit_dir} not found; General MIDI kit used")
        return None
    from . import kit as kitmod

    hits = [(x.start, x.voice or "", x.vel) for x in stream_notes]
    y, missing = kitmod.play(hits, kitmod.load(kit_dir, sr), n, sr)
    missing = set(missing)
    rest = [x for x in stream_notes if (x.start, x.voice or "", x.vel) in missing]
    if rest:
        y = y + _synth_stream(stream_events(rest, target), target, sfpath, sr, n)
    return y


def mix(doc: Document, streams: dict[str, np.ndarray], sr: int,
        with_vocals: bool = False) -> np.ndarray:
    """Mix rendered streams. Vocal streams are left out unless asked for: until
    Milestone 2 gives them a real singing voice, a "voice oohs" line is a
    distraction when judging the instruments."""
    from .render import _GAIN, _PAN, section_gains

    n = max((y.shape[0] for y in streams.values()), default=sr)
    buf = np.zeros((n, 2), dtype=np.float32)
    for name, y in streams.items():
        s = doc.stream(name)
        target = gm.target_for(name, s.fields.get("inst", "unknown") if s else "unknown")
        if target.family == "voice" and not with_vocals:
            continue
        has_level = s is not None and "level" in s.meta
        gain = 1.0 if has_level else _GAIN.get(target.family, _GAIN["unknown"])
        from .fx import parse_fx
        pan = 0.0 if parse_fx(s) is not None else _PAN.get(target.family, 0.0)
        buf[:, 0] += y[:, 0] * gain * (1.0 - max(pan, 0.0))
        buf[:, 1] += y[:, 1] * gain * (1.0 + min(pan, 0.0))
    for start_s, end_s, g in section_gains(doc):
        a, b = int(start_s * sr), min(int(end_s * sr), n)
        if b > a:
            buf[a:b] *= g
    peak = float(np.abs(buf).max())
    if peak > 0:
        buf *= 0.89 / peak
    return buf


def render(doc: Document, sr: int | None = None, sf2: Path | None = None,
           with_vocals: bool = False, voice_ref: Path | None = None,
           no_fx: bool = False) -> np.ndarray:
    sr = sr or doc.sample_rate
    streams = render_streams(doc, sr, sf2, no_fx=no_fx)
    if with_vocals:
        import sys

        from . import sing
        from .sing_score import NoVocalError, vocal_stream
        try:
            name = vocal_stream(doc)
            wav, warns = sing.sing(doc, voice_ref)
        except NoVocalError as exc:
            print(f"with-vocals: {exc}; rendering instruments only", file=sys.stderr)
            return mix(doc, streams, sr, with_vocals)
        for w in warns:
            print(f"with-vocals: {w}", file=sys.stderr)
        streams[name] = _load_stream(wav, sr, max((y.shape[0] for y in streams.values()), default=0),
                                     doc.stream(name))
    return mix(doc, streams, sr, with_vocals)


VOICE_RANGE_DB = 30.0


def match_level(y: np.ndarray, sr: int, level_db: float) -> np.ndarray:
    """Scale a synthesized part so its RMS over the 2 s blocks where it really
    sounds (within VOICE_RANGE_DB of its loudest block) equals `level_db`.
    A vocoder's faint noise floor in every block must not count as singing."""
    b = max(1, int(LEVEL_BLOCK_S * sr))
    blocks = [y[i:i + b] for i in range(0, len(y), b)]
    db = np.array([10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-20) for x in blocks])
    keep = [x for x, d in zip(blocks, db) if d >= db.max() - VOICE_RANGE_DB]
    have = 20 * np.log10(np.sqrt(np.mean(np.concatenate(keep).astype(np.float64) ** 2)) + 1e-20)
    return (y * 10 ** ((level_db - have) / 20)).astype(np.float32)


def _load_stream(path: Path, sr: int, n: int, stream) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    n = max(n, len(y))
    out = np.zeros((n, 2), np.float32)
    out[: len(y), 0] = out[: len(y), 1] = y
    level = stream.meta.get("level") if stream is not None else None
    if level is not None and np.abs(out).max() > 0:
        mono = match_level(out[:, 0], sr, float(level.rstrip("dB")))
        out[:, 0] = out[:, 1] = mono
    return out


def render_to_file(doc: Document, path: str, sr: int | None = None,
                   with_vocals: bool = False, voice_ref: Path | None = None,
                   no_fx: bool = False) -> tuple[int, float]:
    import soundfile as sf

    sr = sr or doc.sample_rate
    audio = render(doc, sr, with_vocals=with_vocals, voice_ref=voice_ref, no_fx=no_fx)
    sf.write(path, audio, sr, subtype="PCM_16")
    return len(expand(doc)), audio.shape[0] / sr
