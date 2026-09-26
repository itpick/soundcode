"""Compare a render against the original, stem by stem.

Every metric returns None when it has nothing to measure (a silent side),
never NaN: a report must say "no evidence", not print a number that lies.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SR = 22050
_HOP = 512
_SILENT_DB = -70.0


def json_safe(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def level_db(y: np.ndarray) -> float | None:
    r = float(np.sqrt(np.mean(np.square(y, dtype=np.float64)))) if y.size else 0.0
    return 20 * math.log10(r) if r > 0 else None


def _silent(y: np.ndarray) -> bool:
    lv = level_db(y)
    return lv is None or lv < _SILENT_DB


def note_f1(ref_iv, ref_hz, est_iv, est_hz, octave_agnostic: bool = False) -> float | None:
    import mir_eval

    if len(ref_iv) == 0 and len(est_iv) == 0:
        return None
    if len(ref_iv) == 0 or len(est_iv) == 0:
        return 0.0
    ref_hz, est_hz = np.asarray(ref_hz, float), np.asarray(est_hz, float)
    if octave_agnostic:
        fold = lambda h: 440.0 * 2 ** (((12 * np.log2(h / 440.0)) % 12) / 12)  # noqa: E731
        ref_hz, est_hz = fold(ref_hz), fold(est_hz)
    return float(mir_eval.transcription.precision_recall_f1_overlap(
        np.asarray(ref_iv, float), ref_hz, np.asarray(est_iv, float), est_hz,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)[2])


def chroma_blocks(y_ref, y_est, blocks) -> list[float | None]:
    import librosa

    n = min(len(y_ref), len(y_est))
    y_ref, y_est = y_ref[:n], y_est[:n]
    cr = librosa.feature.chroma_cqt(y=y_ref, sr=SR, hop_length=_HOP)
    ce = librosa.feature.chroma_cqt(y=y_est, sr=SR, hop_length=_HOP)
    out: list[float | None] = []
    for a, b in blocks:
        i, j = int(a * SR), int(b * SR)
        if _silent(y_ref[i:j]) or _silent(y_est[i:j]):
            out.append(None)
            continue
        fa, fb = int(a * SR / _HOP), max(int(b * SR / _HOP), int(a * SR / _HOP) + 1)
        u, v = cr[:, fa:fb].mean(1), ce[:, fa:fb].mean(1)
        out.append(float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)))
    return out


def onset_f1(y_ref, y_est) -> float | None:
    import librosa
    import mir_eval

    if _silent(y_ref) and _silent(y_est):
        return None
    on_r = librosa.onset.onset_detect(y=y_ref, sr=SR, units="time")
    on_e = librosa.onset.onset_detect(y=y_est, sr=SR, units="time")
    if len(on_r) == 0 or len(on_e) == 0:
        return 0.0
    return float(mir_eval.onset.f_measure(on_r, on_e, window=0.07)[0])


def energy_corr(y_ref, y_est) -> float | None:
    import librosa

    n = min(len(y_ref), len(y_est))
    if _silent(y_ref[:n]) or _silent(y_est[:n]):
        return None
    a = librosa.feature.rms(y=y_ref[:n], hop_length=_HOP)[0]
    b = librosa.feature.rms(y=y_est[:n], hop_length=_HOP)[0]
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def blocks_from_grid(doc, duration: float) -> list[tuple[float, float]]:
    from .expand import build_grid

    if doc is not None and doc.stream("grid") is not None:
        grid = build_grid(doc)
        edges, bar = [], 1
        while True:
            t = grid.time_of(bar, 1.0)
            if t >= duration or bar > 10000:
                break
            edges.append(t)
            bar += 1
        edges.append(duration)
        blocks = [(a, b) for a, b in zip(edges, edges[1:]) if b > a]
        if blocks:
            return blocks
    edges = list(np.arange(0.0, duration, 2.0)) + [duration]
    return [(float(a), float(b)) for a, b in zip(edges, edges[1:]) if b > a]


_NAME_STEM = {"vox": "lead_vocals", "bvox": "backing_vocals", "drums": "drums",
              "bass": "bass", "guitar": "guitar", "piano": "piano", "other": "other"}


def stem_for_stream(doc, name: str) -> str:
    from .separate import STEMS

    s = doc.stream(name)
    if s is not None and s.meta.get("stem") in STEMS:
        return s.meta["stem"]
    track = name.split(".", 1)[1] if "." in name else name
    return _NAME_STEM.get(track, "other")


def transcribe(path) -> tuple[np.ndarray, np.ndarray]:
    from basic_pitch.inference import predict

    _, _, events = predict(str(path))
    if not events:
        return np.zeros((0, 2)), np.zeros(0)
    events = sorted(events, key=lambda e: e[0])
    iv = np.array([[e[0], max(e[1], e[0] + 0.01)] for e in events])
    hz = 440.0 * 2 ** ((np.array([e[2] for e in events]) - 69) / 12)
    return iv, hz


def _mono(y: np.ndarray, sr: int) -> np.ndarray:
    """(n, 2) render output -> mono at SR. (separate.read_stereo is (2, n): use .mean(0).)"""
    import librosa

    m = y.mean(1) if y.ndim == 2 else y
    return librosa.resample(m.astype(np.float32), orig_sr=sr, target_sr=SR) if sr != SR else m


def _write_wav(path: Path, y_mono: np.ndarray) -> None:
    import soundfile as sf
    sf.write(str(path), y_mono, SR, subtype="FLOAT")


def run(original, sc_or_wav, out_dir, engine: str = "sf2",
        render_streams=None, stems_dir=None) -> dict:
    from .parser import parse_file
    from .separate import STEMS, default_out_dir, read_stereo, separate

    original, sc_or_wav, out = Path(original), Path(sc_or_wav), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    notes_side: dict[str, str] = {}

    # original stems
    sd = Path(stems_dir) if stems_dir else default_out_dir(original)
    if not (sd / "manifest.json").exists() and stems_dir is None:
        separate(original, sd)
    orig = {s: read_stereo(sd / f"{s}.wav", sr=SR).mean(0) for s in STEMS}
    n = max(len(y) for y in orig.values())

    # rendered side, per stem
    rend = {s: np.zeros(n, np.float32) for s in STEMS}
    doc = None
    if sc_or_wav.suffix == ".sc":
        doc = parse_file(str(sc_or_wav))
        if render_streams is None:
            if engine == "mock":
                raise ValueError("compare needs per-stream renders; use --engine sf2")
            from .render_sf import render_streams
        streams = render_streams(doc, sr=doc.sample_rate)
        for name, y in streams.items():
            m = _mono(y, doc.sample_rate)[:n]
            rend[stem_for_stream(doc, name)][:len(m)] += m
    else:
        rs = default_out_dir(sc_or_wav)
        separate(sc_or_wav, rs)
        notes_side["render"] = "stems from re-separation (less reliable pairing)"
        for s in STEMS:
            m = read_stereo(rs / f"{s}.wav", sr=SR).mean(0)[:n]
            rend[s][:len(m)] = m

    duration = n / SR
    blocks = blocks_from_grid(doc, duration)
    report: dict = {"original": str(original), "render": str(sc_or_wav), "engine": engine,
                    "notes": notes_side, "blocks": blocks, "stems": {}}
    for s in STEMS:
        yo, yr = orig[s][:n], rend[s][:n]
        po, pr = out / f"stem-{s}.orig.wav", out / f"stem-{s}.render.wav"
        _write_wav(po, yo)
        _write_wav(pr, yr)
        lo, lr = level_db(yo), level_db(yr)
        row = {"level_orig_db": lo, "level_render_db": lr,
               "level_diff_db": (lr - lo) if lo is not None and lr is not None else None,
               "onset_f1": onset_f1(yo, yr), "energy_corr": energy_corr(yo, yr),
               "chroma_blocks": chroma_blocks(yo, yr, blocks)}
        vals = [v for v in row["chroma_blocks"] if v is not None]
        row["chroma"] = float(np.mean(vals)) if vals else None
        if s == "drums":
            row["notes_f1"] = row["notes_f1_octave"] = None
        else:
            (io, ho), (ir, hr) = transcribe(po), transcribe(pr)
            row["notes_f1"] = note_f1(io, ho, ir, hr)
            row["notes_f1_octave"] = note_f1(io, ho, ir, hr, octave_agnostic=True)
        row["sound"] = None                      # CLAP: enabled in Plan 2
        report["stems"][s] = row

    _plots(out, orig, rend, report)
    (out / "report.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False))
    _html(out, report)
    report["report"] = str(out / "report.html")
    return report


def _plots(out: Path, orig: dict, rend: dict, report: dict) -> None:
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for s in orig:
        fig, ax = plt.subplots(1, 4, figsize=(22, 3.2))
        for j, (y, tag) in enumerate([(orig[s], "original"), (rend[s], "render")]):
            S = librosa.amplitude_to_db(np.abs(librosa.stft(y + 1e-9, hop_length=_HOP)), ref=np.max)
            librosa.display.specshow(S, sr=SR, hop_length=_HOP, x_axis="time", y_axis="log",
                                     ax=ax[j], vmin=-80)
            ax[j].set_title(f"{s} — {tag} spectrogram")
            C = librosa.feature.chroma_cqt(y=y + 1e-9, sr=SR, hop_length=_HOP)
            librosa.display.specshow(C, sr=SR, hop_length=_HOP, x_axis="time", y_axis="chroma",
                                     ax=ax[2 + j])
            ax[2 + j].set_title(f"{s} — {tag} chroma")
        fig.tight_layout()
        fig.savefig(out / f"{s}.png", dpi=60)
        plt.close(fig)

    names = list(report["stems"])
    grid = np.array([[np.nan if v is None else v for v in report["stems"][s]["chroma_blocks"]]
                     for s in names], dtype=float)
    fig, ax = plt.subplots(figsize=(max(6, grid.shape[1] * 0.3), 3))
    im = ax.imshow(grid, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("bar (or 2 s block)")
    ax.set_title("chroma similarity: where the render diverges")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "bars.png", dpi=70)
    plt.close(fig)


def _fmt(v, spec=".2f"):
    return "—" if v is None else format(v, spec)


def _html(out: Path, report: dict) -> None:
    rows = []
    for s, r in report["stems"].items():
        rows.append(
            f"<tr><td>{s}</td><td>{_fmt(r['level_orig_db'], '.1f')}</td>"
            f"<td>{_fmt(r['level_render_db'], '.1f')}</td><td>{_fmt(r['level_diff_db'], '+.1f')}</td>"
            f"<td>{_fmt(r['notes_f1'])}</td><td>{_fmt(r['notes_f1_octave'])}</td>"
            f"<td>{_fmt(r['chroma'])}</td><td>{_fmt(r['onset_f1'])}</td><td>{_fmt(r['energy_corr'])}</td>"
            f"<td><audio controls preload=none src='stem-{s}.orig.wav'></audio></td>"
            f"<td><audio controls preload=none src='stem-{s}.render.wav'></audio></td></tr>")
    imgs = "".join(f"<h3>{s}</h3><img src='{s}.png' style='max-width:100%'>" for s in report["stems"])
    notes = "".join(f"<p><b>note:</b> {v}</p>" for v in report["notes"].values())
    (out / "report.html").write_text(f"""<!doctype html><meta charset=utf-8>
<title>compare — {Path(report['original']).name}</title>
<style>body{{font:14px system-ui;margin:16px}}td,th{{padding:4px 8px;border-bottom:1px solid #ddd}}
table{{border-collapse:collapse;overflow-x:auto;display:block}}</style>
<h1>{Path(report['original']).name} vs {Path(report['render']).name}</h1>{notes}
<table><tr><th>stem</th><th>orig dB</th><th>render dB</th><th>Δ dB</th><th>note F1</th>
<th>F1 any-oct</th><th>chroma</th><th>onset F1</th><th>energy r</th><th>original</th><th>render</th></tr>
{''.join(rows)}</table><h2>Where it diverges</h2><img src='bars.png' style='max-width:100%'>{imgs}
""", encoding="utf-8")
