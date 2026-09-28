"""Combining metrics into 0-100 scores: per slice, per part, per song.

Spec 2026-09-28-benchmark-scorer, Task 4, "From metrics to 0-100". This
module only combines already-computed metric dicts (`score_slice`'s input
is the flat `{metric_name: value|None}` that Task 5's `score_song` derives
per slice, e.g. `level_diff_db_abs = |level_diff_db|`, `lag_ms_abs =
|lag_ms|`) — it never touches audio.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import anchors

# part string -> anchors.json key (the metric set and floor/ceiling/weight/
# axis to score that part against). `residual` is deliberately absent: it
# is never scored (design doc).
PART_TYPE: dict[str, str] = {
    "piano": "pitched",
    "guitar": "pitched",
    "other": "pitched",
    "bass": "bass",
    "drums": "drums",
    "lead_vocals": "vocal",
    "backing_vocals": "vocal",
    "mix": "mix",
}

# Part score = 0.4 What + 0.4 Sound + 0.2 Dyn, renormalised over the axes
# that aren't None (design doc, "Combining").
AXIS_WEIGHTS: dict[str, float] = {"what": 0.4, "sound": 0.4, "dyn": 0.2}

AXES = ("what", "sound", "dyn")


def score_slice(metrics: dict, part_type: str) -> dict:
    """Score one slice's flat metrics dict against `part_type`'s anchors.

    Returns `{axes: {what, sound, dyn}, score, metrics: {name: {raw, score}}}`.
    A metric name with no anchor entry for `part_type` is ignored (not an
    error) — it is simply left out of `metrics` and never affects an axis.
    An axis with no non-None metric score is `None` (excluded, not 0); the
    overall score is the `AXIS_WEIGHTS`-weighted mean of the axes that
    aren't None, renormalised over just those axes, or `None` if every axis
    is None.
    """
    part_anchors = anchors.ANCHORS.get(part_type, {})
    out_metrics: dict[str, dict] = {}
    axis_pairs: dict[str, list[tuple[float, float]]] = {axis: [] for axis in AXES}

    for name, raw in metrics.items():
        entry = part_anchors.get(name)
        if entry is None:
            continue
        score = anchors.to_score(raw, entry)
        out_metrics[name] = {"raw": raw, "score": score}
        if score is not None:
            axis_pairs[entry["axis"]].append((score, float(entry.get("weight", 1.0))))

    axes: dict[str, float | None] = {}
    for axis in AXES:
        pairs = axis_pairs[axis]
        total_w = sum(w for _, w in pairs)
        axes[axis] = (sum(s * w for s, w in pairs) / total_w) if total_w > 0 else None

    num = 0.0
    den = 0.0
    for axis, w in AXIS_WEIGHTS.items():
        if axes[axis] is not None:
            num += axes[axis] * w
            den += w
    score = (num / den) if den > 0 else None

    return {"axes": axes, "score": score, "metrics": out_metrics}


def score_part(
    song: dict,
    sections: list[tuple[str, dict]],
    windows: list[tuple[str, dict]],
    part_type: str,
    *,
    silent: bool = False,
    missing: bool = False,
) -> dict:
    """Combine one part's song/section/window metric dicts into its scored form.

    `song`, and each `(label, metrics)` pair in `sections`/`windows`, are
    flat metric dicts as `score_slice` takes. Returns `{song, sections:[…],
    windows:[…], worst: {label, score} | None, missing, silent}`.

    `silent` (the original is below the activity gate for the whole song):
    not scored, not a failure — `song` is `None`, `sections`/`windows` are
    empty, `worst` is `None`.
    `missing` (an active original part has no rebuilt counterpart): scores
    0 rather than being left unscored, so it drags the song average down.
    """
    if silent:
        return {"song": None, "sections": [], "windows": [], "worst": None,
                "missing": False, "silent": True}

    if missing:
        zero_slice = {"axes": {axis: None for axis in AXES}, "score": 0.0, "metrics": {}}
        return {"song": zero_slice, "sections": [], "windows": [], "worst": None,
                "missing": True, "silent": False}

    song_scored = score_slice(song, part_type)
    sections_scored = [{"label": label, **score_slice(m, part_type)} for label, m in sections]
    windows_scored = [{"label": label, **score_slice(m, part_type)} for label, m in windows]

    worst: dict | None = None
    for entry in sections_scored:
        if entry["score"] is None:
            continue
        if worst is None or entry["score"] < worst["score"]:
            worst = {"label": entry["label"], "score": entry["score"]}

    return {"song": song_scored, "sections": sections_scored, "windows": windows_scored,
            "worst": worst, "missing": False, "silent": False}


def _mix_score(mix) -> float | None:
    """The mix's score, from a `score_part`-shaped dict (its `song.score`,
    or `None` if the mix itself came back `silent`) — the mix is scored the
    same way as any other part, via `score_part(..., "mix")`, so this takes
    the same shape `parts`' values do, not a bare number.

    `mix=None` is accepted as "no mix score to blend in" (see `song_score`'s
    fallback). Anything else that isn't a dict with a `song` key is a
    caller bug, not a silent/missing mix, and raises rather than being
    quietly treated as absent.
    """
    if mix is None:
        return None
    if not isinstance(mix, dict) or "song" not in mix:
        raise ValueError(
            "song_score's `mix` must be a score_part-shaped dict (with a "
            f"'song' key) or None, not {mix!r}")
    song = mix["song"]
    return song["score"] if song is not None else None


def song_score(parts: dict, energy: dict, mix) -> float | None:
    """The energy-share-weighted mean of the non-silent parts, blended 50/50
    with the mix score.

    - A missing active part (`result["missing"]`) counts as 0 but still
      keeps its energy weight, dragging the average down.
    - A part that is neither missing nor silent but whose own song score is
      `None` (e.g. every one of its metrics came back `None`) is treated
      the same way: it counts as 0 while keeping its weight. Only
      `silent` opts a part out of the average entirely.
    - A silent part is excluded, including its weight, so the remaining
      parts' shares are implicitly renormalised over what's left.
    - If there is no weight left to average over on the parts side (no
      non-silent parts, or every non-silent part's energy share is 0), the
      song score is the mix score alone.
    - `mix` must be a `score_part`-shaped dict (see `_mix_score`) or
      `None`; `None` means there is no mix score to blend in.
    - `None` overall only when there is nothing to average on either side
      (no usable parts weight AND no mix score).
    """
    total = 0.0
    total_w = 0.0
    for name, result in parts.items():
        if result.get("silent"):
            continue
        w = float(energy.get(name, 0.0))
        song = result.get("song")
        score = 0.0 if result.get("missing") else (song.get("score") if song else None)
        if score is None:
            score = 0.0
        total += w * score
        total_w += w
    parts_score = (total / total_w) if total_w > 0 else None

    mix_score = _mix_score(mix)

    if parts_score is None and mix_score is None:
        return None
    if parts_score is None:
        return mix_score
    if mix_score is None:
        return parts_score
    return 0.5 * parts_score + 0.5 * mix_score


# --------------------------------------------------------------------------
# score_song: from audio files to the nested result dict (Task 5)
# --------------------------------------------------------------------------

VOCAL_PARTS = ("lead_vocals", "backing_vocals")
BASS_F0_FMIN = 30.0          # design doc: bass f0 via torchcrepe with fmin 30 Hz

_WHISPER = None


def _words(wav) -> list[tuple[str, float, float]]:
    """faster-whisper `small` word stamps for `wav`: `[(word, start, end)]`.

    The one place the scorer runs Whisper: `score_song` calls it once per
    file and shares the result between `sung_wer` and `word_mae_s` (tests
    monkeypatch it). The model is loaded once per process.
    """
    global _WHISPER
    if _WHISPER is None:
        from faster_whisper import WhisperModel

        from ..lyrics import asr_download_root
        _WHISPER = WhisperModel("small", device="cpu", compute_type="int8",
                                download_root=asr_download_root())
    segs, _ = _WHISPER.transcribe(str(wav), language="en", word_timestamps=True)
    return [(w.word, float(w.start), float(w.end)) for s in segs for w in (s.words or [])]


def _load_mono(path: Path):
    import librosa

    from .metrics import SR
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y.astype(np.float32)


def _load_stereo(path: Path, sr: int):
    """(2, n) float32 at `sr`; a mono file is duplicated."""
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y.astype(np.float32)


def _fit(y, n: int):
    """Unequal lengths: the original sets the length. A shorter side is
    zero-padded (a rebuild that stops early is silent there, and scored as
    such), a longer one is cut at the original's end."""
    if y.shape[-1] >= n:
        return y[..., :n]
    pad = [(0, 0)] * (y.ndim - 1) + [(0, n - y.shape[-1])]
    return np.pad(y, pad)


def _abs(v):
    return None if v is None else abs(float(v))


def _mean(vals):
    vals = [float(v) for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _word_pairs(ref_words, est_words) -> list[tuple[float, float]]:
    """(ref start, est start) for each word the two transcriptions agree on,
    aligned with difflib on normalised words."""
    import difflib

    from ..compare import heard_words

    def tokens(words):
        out = []
        for w, start, _end in words:
            out += [(n, float(start)) for n in heard_words([w])]
        return out

    r, e = tokens(ref_words), tokens(est_words)
    sm = difflib.SequenceMatcher(None, [w for w, _ in r], [w for w, _ in e], autojunk=False)
    pairs = []
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            pairs.append((r[blk.a + k][1], e[blk.b + k][1]))
    return pairs


def _word_mae(pairs, a: float, b: float) -> float | None:
    """Median |start difference| over matched words whose original start is in [a, b)."""
    d = [abs(te - tr) for tr, te in pairs if a <= tr < b]
    return float(np.median(d)) if d else None


def _decay_ratio_err(ref_y, est_y, sr, vo_ref, vo_est, a, b) -> float | None:
    """Mean over voices hit on both sides in [a, b) of |decay_est / decay_ref - 1|."""
    from . import drums

    errs = []
    for v in drums.VOICES:
        r = drums._in_range(np.asarray(vo_ref.get(v, [])), a, b)
        e = drums._in_range(np.asarray(vo_est.get(v, [])), a, b)
        if r.size == 0 or e.size == 0:
            continue
        dr, de = drums.decay_s(ref_y, sr, r), drums.decay_s(est_y, sr, e)
        if dr and de is not None:
            errs.append(abs(de / dr - 1.0))
    return _mean(errs)


def _width(seg) -> float | None:
    """Stereo width: side RMS / (mid RMS + side RMS), on a (2, n) segment."""
    if seg.shape[-1] == 0:
        return None
    left, right = seg[0].astype(np.float64), seg[1].astype(np.float64)
    mid = np.sqrt(np.mean(((left + right) / 2) ** 2))
    side = np.sqrt(np.mean(((left - right) / 2) ** 2))
    return float(side / (mid + side)) if mid + side > 1e-12 else None


def _lufs(seg, sr: int) -> float | None:
    """Integrated loudness (pyloudnorm, BS.1770) of a (2, n) segment; None if
    too short to gate (< 400 ms) or silent."""
    import pyloudnorm

    if seg.shape[-1] < int(0.4 * sr) + 1:
        return None
    with np.errstate(divide="ignore"):
        v = pyloudnorm.Meter(sr).integrated_loudness(seg.T.astype(np.float64))
    return float(v) if np.isfinite(v) else None


def _seg(y, sr, a, b):
    return y[..., max(int(a * sr), 0):max(int(b * sr), 0)]


def _slice_entries(scored: list[dict], slcs, raws, states) -> list[dict]:
    """Annotate `score_part`'s slice entries with span, state flags and raw
    metrics. A `missing` slice (original active, rebuild below the gate
    there) scores 0: the section-level form of "missing active part -> 0"."""
    out = []
    for e, s, raw, st in zip(scored, slcs, raws, states):
        e = {**e, "a": s.a, "b": s.b, "silent": st == "silent", "missing": st == "missing",
             "raw": raw}
        if st == "missing":
            e["axes"] = {axis: None for axis in AXES}
            e["score"] = 0.0
            e["metrics"] = {}
        out.append(e)
    return out


def _missing_share(secs, sec_states) -> float:
    """Share of the original's active section time where the rebuild is missing."""
    active = sum(s.b - s.a for s, st in zip(secs, sec_states) if st != "silent")
    missing = sum(s.b - s.a for s, st in zip(secs, sec_states) if st == "missing")
    return missing / active if active > 0 else 0.0


def _assemble(part_type, song_s, secs, wins, song_m, sec_ms, win_ms, states) -> dict:
    """`score_part` on the flat metric dicts, then annotate every slice entry
    with its time span, its state flags and every raw metric (anchored or not).

    Missing sections pull the song down in proportion to their time: the
    song's axes and score are the whole-song metrics' ones scaled by the
    share of the original's active time the rebuild covers
    (`score_present` keeps the unscaled score)."""
    res = score_part(song_m, [(s.label, m) for s, m in zip(secs, sec_ms)],
                     [(s.label, m) for s, m in zip(wins, win_ms)], part_type)
    share = _missing_share(secs, states["sections"])
    song = {**res["song"], "a": song_s.a, "b": song_s.b, "silent": False, "missing": False,
            "raw": song_m, "score_present": res["song"]["score"], "missing_share": share}
    if share > 0:
        keep = 1.0 - share
        song["axes"] = {k: (None if v is None else v * keep) for k, v in song["axes"].items()}
        song["score"] = None if song["score"] is None else song["score"] * keep
    res["song"] = song
    res["sections"] = _slice_entries(res["sections"], secs, sec_ms, states["sections"])
    res["windows"] = _slice_entries(res["windows"], wins, win_ms, states["windows"])
    # re-pick the worst section from the annotated entries (same rule as
    # score_part: first lowest wins) so it carries its own span even when two
    # sections share a label (e.g. two "verse"s)
    scored = [e for e in res["sections"] if e["score"] is not None]
    if scored:
        w = min(scored, key=lambda e: e["score"])
        res["worst"] = {"label": w["label"], "score": w["score"], "a": w["a"], "b": w["b"]}
    return res


def _lags(y_ref, y_est, sr, secs, wins, states):
    """Per-window drift, per-section lag, and the song lag.

    The song lag is the mean |lag| over the windows where both sides are
    active: the whole-song cross-correlation of a part that drifts
    progressively has no one lag. A window where the original is active but
    the rebuild isn't is flagged `missing` in its drift entry (lag unknown,
    None, never "in sync"), and listed in `missing_windows` rather than
    silently dropped."""
    from . import align

    dr = align.drift(y_ref, y_est, sr, list(wins) + list(secs))
    win_dr, sec_dr = dr[:len(wins)], dr[len(wins):]
    for d, st in zip(win_dr, states["windows"]):
        d["missing"] = st == "missing"
        if st == "missing":
            d["lag_ms"], d["drift"] = None, False
    song_lag = _mean(abs(d["lag_ms"]) for d, st in zip(win_dr, states["windows"])
                     if st == "ok" and d["lag_ms"] is not None)
    missing_windows = [d["label"] for d in win_dr if d["missing"]]
    return win_dr, [d["lag_ms"] for d in sec_dr], song_lag, missing_windows


def _states(y_ref, y_est, sr, secs, wins) -> dict:
    """Per section / window: "silent" where the original is below the gate
    (not scored), "missing" where the original is active but the rebuild
    isn't (scored 0), else "ok". Decided on the full-length original."""
    from . import slices

    def state(s):
        if not slices.active(y_ref, sr, s.a, s.b):
            return "silent"
        return "ok" if slices.active(y_est, sr, s.a, s.b) else "missing"
    return {"sections": [state(s) for s in secs], "windows": [state(s) for s in wins]}


def _score_part_audio(key: str, ref_path: Path, est_path: Path, y_ref, y_est, doc, song_s,
                      secs, wins, drum_cache: Path) -> tuple[dict, list[dict]]:
    """One active part with a rebuilt counterpart: every metric on every
    slice. Heavy features (librosa, basic-pitch, torchcrepe, MERT, tsumugi,
    Whisper) run once per file; the slices only index into them."""
    from .. import compare
    from . import drums, embed, metrics

    part_type = PART_TYPE[key]
    sr = metrics.SR
    y_est = _fit(y_est, y_ref.shape[-1])
    pitched, f0 = key in metrics.NOTE_F1_PARTS, key in metrics.F0_PARTS
    fmin = BASS_F0_FMIN if key == "bass" else 50.0
    f_ref = metrics.Features.of(y_ref, sr, pitched, f0, f0_fmin=fmin)
    f_est = metrics.Features.of(y_est, sr, pitched, f0, f0_fmin=fmin)
    fr_ref, fr_est = embed.frames(ref_path), embed.frames(est_path)

    states = _states(y_ref, y_est, sr, secs, wins)
    win_dr, sec_lags, song_lag, missing_windows = _lags(y_ref, y_est, sr, secs, wins, states)

    vo_ref = vo_est = None
    if key == "drums":
        vo_ref, vo_est = drums.voice_onsets(ref_path, drum_cache), drums.voice_onsets(est_path, drum_cache)
    pairs = None
    if key in VOCAL_PARTS:
        w_ref, w_est = _words(ref_path), _words(est_path)
        pairs = _word_pairs(w_ref, w_est)

    def one(s, lag):
        m = metrics.slice_metrics(f_ref, f_est, s, key)
        m["level_diff_db_abs"] = _abs(m["level_diff_db"])
        m["lag_ms"] = lag
        m["lag_ms_abs"] = _abs(lag)
        m["mert"] = embed.cosine(fr_ref, fr_est, s.a, s.b)
        yr, ye = metrics._slice_y(f_ref, s.a, s.b), metrics._slice_y(f_est, s.a, s.b)
        m["spectral_db"] = compare.spectral_db(yr, ye, sr) if yr.size and ye.size else None
        if vo_ref is not None:
            f1s = drums.voice_f1(vo_ref, vo_est, s.a, s.b)
            m["drum_voice_f1"] = _mean(f1s.values())
            m["drum_voice_f1_by_voice"] = f1s
            m["decay_ratio_err"] = _decay_ratio_err(f_ref.y, f_est.y, sr, vo_ref, vo_est, s.a, s.b)
        if pairs is not None:
            m["word_mae_s"] = _word_mae(pairs, s.a, s.b)
        return m

    sec_ms = [one(s, lag) if st == "ok" else {}
              for s, lag, st in zip(secs, sec_lags, states["sections"])]
    win_ms = [one(s, d["lag_ms"]) if st == "ok" else {}
              for s, d, st in zip(wins, win_dr, states["windows"])]
    song_m = one(song_s, song_lag)
    song_m["missing_windows"] = missing_windows
    if key in VOCAL_PARTS:
        song_m["voice_sim"] = compare.voice_similarity(f_ref.y, f_est.y, sr)
        ws_est = compare.sung_wer(doc, est_path, words=w_est)
        ws_ref = compare.sung_wer(doc, ref_path, words=w_ref)
        song_m["sung_wer"], song_m["sung_wer_original"] = ws_est, ws_ref
        song_m["sung_wer_excess"] = (max(0.0, ws_est - ws_ref)
                                     if ws_est is not None and ws_ref is not None else None)

    res = _assemble(part_type, song_s, secs, wins, song_m, sec_ms, win_ms, states)
    return res, win_dr


def _score_mix(original: Path, rebuild_mix: Path, song_s, secs, wins) -> tuple[dict, list[dict]]:
    """The rebuilt mix against the original mix: chroma, onset F1, lag, MERT,
    LUFS-I and stereo width differences, spectral distance."""
    import soundfile as sf

    from .. import compare
    from . import embed, metrics, slices

    sr = metrics.SR
    y_ref = _load_mono(original)
    y_est = _fit(_load_mono(rebuild_mix), y_ref.shape[-1])
    if not slices.active(y_ref, sr, song_s.a, song_s.b):
        return score_part({}, [], [], "mix", silent=True), []
    if not slices.active(y_est, sr, song_s.a, song_s.b):
        return score_part({}, [], [], "mix", missing=True), []

    st_sr = sf.info(str(original)).samplerate
    st_ref = _load_stereo(original, st_sr)
    st_est = _fit(_load_stereo(rebuild_mix, st_sr), st_ref.shape[-1])
    f_ref = metrics.Features.of(y_ref, sr, False, False)
    f_est = metrics.Features.of(y_est, sr, False, False)
    fr_ref, fr_est = embed.frames(original), embed.frames(rebuild_mix)
    states = _states(y_ref, y_est, sr, secs, wins)
    win_dr, sec_lags, song_lag, missing_windows = _lags(y_ref, y_est, sr, secs, wins, states)

    def one(s, lag):
        m = metrics.slice_metrics(f_ref, f_est, s, "mix")
        m["lag_ms"], m["lag_ms_abs"] = lag, _abs(lag)
        m["mert"] = embed.cosine(fr_ref, fr_est, s.a, s.b)
        yr, ye = metrics._slice_y(f_ref, s.a, s.b), metrics._slice_y(f_est, s.a, s.b)
        m["spectral_db"] = compare.spectral_db(yr, ye, sr) if yr.size and ye.size else None
        sr_, se_ = _seg(st_ref, st_sr, s.a, s.b), _seg(st_est, st_sr, s.a, s.b)
        lr, le = _lufs(sr_, st_sr), _lufs(se_, st_sr)
        m["lufs_diff"] = (le - lr) if lr is not None and le is not None else None
        m["lufs_diff_abs"] = _abs(m["lufs_diff"])
        wr, we = _width(sr_), _width(se_)
        m["width_diff"] = abs(wr - we) if wr is not None and we is not None else None
        return m

    sec_ms = [one(s, lag) if st == "ok" else {}
              for s, lag, st in zip(secs, sec_lags, states["sections"])]
    win_ms = [one(s, d["lag_ms"]) if st == "ok" else {}
              for s, d, st in zip(wins, win_dr, states["windows"])]
    song_m = one(song_s, song_lag)
    song_m["missing_windows"] = missing_windows
    return _assemble("mix", song_s, secs, wins, song_m, sec_ms, win_ms, states), win_dr


def _json_default(o):
    if isinstance(o, np.generic):
        v = o.item()
        return None if isinstance(v, float) and not np.isfinite(v) else v
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def score_song(original: Path, stems_dir: Path, sc: Path, parts_dir: Path,
               rebuild_mix: Path, out_dir: Path, *, cache_dir: Path | None = None) -> dict:
    """Score a rebuild against the original, per part, per section and per
    20 s window, and write `scores.json` to `out_dir`.

    - Parts come from `render_sf.PART_KEYS` (`residual` is never scored):
      the original stem is `stems_dir/<key>.wav`, the rebuild
      `parts_dir/<key>.wav`.
    - A part whose original is below the gate for the whole song (or has no
      stem) is `silent`; an active original whose rebuilt part is absent or
      silent is `missing` (score 0, flagged).
    - Unequal lengths: the original sets the length; a shorter rebuild is
      zero-padded (and a longer one cut), so a rebuild that stops early is
      scored as missing there, never excused as silent.
    - A section or window where the original is active but the rebuild is
      below the gate is `missing` and scores 0; the song score is scaled by
      the share of the original's active section time the rebuild covers.
    - `cache_dir` holds the per-file tsumugi drum-voice cache (default: the
      MERT cache directory, `out/bench/cache`).

    Returns `{song, duration, parts, mix, score, worst, drift, energy, files}`.
    """
    import json

    import soundfile as sf

    from .. import compare
    from ..parser import parse_file
    from ..render_sf import PART_KEYS, PART_LABELS
    from . import embed, metrics, slices

    original, stems_dir, parts_dir = Path(original), Path(stems_dir), Path(parts_dir)
    rebuild_mix, out_dir = Path(rebuild_mix), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    drum_cache = Path(cache_dir) if cache_dir is not None else embed.CACHE_DIR

    doc = parse_file(str(sc))
    duration = float(sf.info(str(original)).duration)
    song_s = slices.song(duration)
    secs, wins = slices.sections(doc, duration), slices.windows(duration)

    parts: dict[str, dict] = {}
    drift: dict[str, list] = {}
    energy: dict[str, float] = {}
    for key in PART_KEYS:
        if key not in PART_TYPE:          # residual
            continue
        ref_path, est_path = stems_dir / f"{key}.wav", parts_dir / f"{key}.wav"
        has_ref, has_est = ref_path.is_file(), est_path.is_file()
        if not (has_ref or has_est):
            continue
        y_ref = _load_mono(ref_path) if has_ref else None
        energy[key] = float(np.sum(np.square(y_ref, dtype=np.float64))) if has_ref else 0.0
        ref_active = has_ref and slices.active(y_ref, metrics.SR, 0.0, duration)
        y_est = _load_mono(est_path) if has_est and ref_active else None
        est_active = y_est is not None and slices.active(y_est, metrics.SR, 0.0, duration)
        if not ref_active:
            res, dr = score_part({}, [], [], PART_TYPE[key], silent=True), []
        elif not est_active:
            res, dr = score_part({}, [], [], PART_TYPE[key], missing=True), []
        else:
            res, dr = _score_part_audio(key, ref_path, est_path, y_ref, y_est, doc, song_s,
                                        secs, wins, drum_cache)
        # an original with no stem / below the gate, but a rebuilt part: noted, not scored
        res["extra"] = bool(not ref_active and has_est)
        res.update({"type": PART_TYPE[key], "label": PART_LABELS[key],
                    "files": {"original": str(ref_path) if has_ref else None,
                              "rebuild": str(est_path) if has_est else None}})
        parts[key] = res
        drift[key] = dr

    total = sum(energy.values())
    shares = {k: (v / total if total > 0 else 0.0) for k, v in energy.items()}
    for k in parts:
        parts[k]["energy"] = shares.get(k, 0.0)

    has_mix = rebuild_mix.is_file()
    if has_mix:
        mix, drift["mix"] = _score_mix(original, rebuild_mix, song_s, secs, wins)
    else:
        mix, drift["mix"] = score_part({}, [], [], "mix", missing=True), []
    mix.update({"type": "mix", "label": "Mix",
                "files": {"original": str(original), "rebuild": str(rebuild_mix) if has_mix else None}})

    worst = None
    for key, res in parts.items():
        if res["silent"]:
            continue
        if res["missing"]:
            cand = {"part": key, "label": "not rebuilt", "score": 0.0, "a": 0.0, "b": duration}
        elif res["worst"] is not None:
            cand = {"part": key, **res["worst"]}
        else:
            continue
        if worst is None or cand["score"] < worst["score"]:
            worst = cand

    result = {"song": original.stem, "duration": duration, "parts": parts, "mix": mix,
              "score": song_score(parts, shares, mix), "worst": worst, "drift": drift,
              "energy": shares,
              "sections": [{"label": s.label, "a": s.a, "b": s.b} for s in secs],
              "files": {"original": str(original), "sc": str(sc), "stems": str(stems_dir),
                        "parts": str(parts_dir), "rebuild_mix": str(rebuild_mix)}}
    (out_dir / "scores.json").write_text(json.dumps(compare.json_safe(result), indent=2,
                                                    default=_json_default, allow_nan=False))
    return result
