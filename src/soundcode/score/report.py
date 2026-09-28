"""The terminal table and the HTML report for one scored song.

Spec 2026-09-28-benchmark-scorer, Task 5, "Outputs". Both take the dict
`scorer.score_song` returns. The HTML is self-contained (inline CSS, no
external resources); every piece of text in it goes through `html.escape`.
"""

from __future__ import annotations

import html as _html
from pathlib import Path

import numpy as np

EXCERPT_S = 15.0
N_WORST = 3
DRIFT_MS = 30.0
PALETTE = ("#2f6fdf", "#d9480f", "#2b8a3e", "#ae3ec9", "#e8a100", "#0c8599", "#c2255c", "#5c7cfa")


def _mmss(t: float | None) -> str:
    if t is None:
        return "?"
    t = max(float(t), 0.0)
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _num(v) -> str:
    return "—" if v is None else f"{v:.0f}"


def _part_rows(result: dict) -> list[tuple[str, dict]]:
    """Parts sorted by score ascending (missing = 0 first), silent parts last."""
    def key(item):
        res = item[1]
        if res["silent"]:
            return (1, 0.0)
        s = res["song"]["score"] if res["song"] is not None else None
        return (0, 0.0 if s is None else s)
    return sorted(result["parts"].items(), key=key)


def _row(name: str, res: dict) -> str:
    if res["silent"]:
        cells, worst = ["—"] * 4, "— (silent in the original)"
    elif res["missing"]:
        cells, worst = ["—", "—", "—", "0"], "not rebuilt"
    else:
        ax = res["song"]["axes"]
        cells = [_num(ax["what"]), _num(ax["sound"]), _num(ax["dyn"]), _num(res["song"]["score"])]
        w = res["worst"]
        worst = f"{w['label']} @ {_mmss(w.get('a'))} ({w['score']:.0f})" if w else "—"
    return f"{name:<15}| {cells[0]:>5} | {cells[1]:>5} | {cells[2]:>5} | {cells[3]:>5} | {worst}"


def table(result: dict) -> str:
    """Fixed-width `part | what | sound | dyn | score | worst (label @ m:ss)`
    rows, sorted by score ascending; `—` for silent parts, `not rebuilt` for
    missing ones; then the mix and the song."""
    head = f"{'part':<15}| {'what':>5} | {'sound':>5} | {'dyn':>5} | {'score':>5} | worst (label @ m:ss)"
    lines = [f"{result['song']}  ({_mmss(result['duration'])})", head, "-" * len(head)]
    lines += [_row(name, res) for name, res in _part_rows(result)]
    lines.append("-" * len(head))
    lines.append(_row("mix", result["mix"]))
    w = result.get("worst")
    worst = f"{w['part']} {w['label']} @ {_mmss(w.get('a'))} ({w['score']:.0f})" if w else "—"
    lines.append(f"{'song':<15}| {'':>5} | {'':>5} | {'':>5} | {_num(result['score']):>5} | {worst}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

def _e(s) -> str:
    return _html.escape(str(s), quote=True)


def _color(score: float) -> str:
    """Red (0) -> amber -> green (100)."""
    h = 120.0 * min(max(score, 0.0), 100.0) / 100.0
    return f"hsl({h:.0f}, 62%, 40%)"


def _heatmap(result: dict) -> str:
    labels = [s["label"] for s in result.get("sections", [])]
    head = "".join(f"<th>{_e(lb)}<br><small>{_e(_mmss(s['a']))}</small></th>"
                   for lb, s in zip(labels, result.get("sections", [])))
    rows = []
    items = [(k, r) for k, r in result["parts"].items()] + [("mix", result["mix"])]
    for key, res in items:
        name = f"{_e(res.get('label', key))}<br><small>{_e(key)}</small>"
        if res["silent"]:
            cells = f"<td class=na colspan={max(len(labels), 1)}>— silent in the original</td>"
            song = "<td class=na>—</td>"
        elif res["missing"]:
            cells = f"<td class=miss colspan={max(len(labels), 1)}>not rebuilt</td>"
            song = f"<td style='background:{_color(0)}'>0</td>"
        else:
            # by position, not label: two sections may share a label ("verse")
            out = []
            for i, lb in enumerate(labels):
                s = res["sections"][i] if i < len(res["sections"]) else None
                if s is None or s["score"] is None:
                    out.append("<td class=na>·</td>")
                else:
                    out.append(f"<td style='background:{_color(s['score'])}' "
                               f"title='{_e(key)} · {_e(lb)}'>{s['score']:.0f}</td>")
            cells = "".join(out)
            sc = res["song"]["score"]
            song = ("<td class=na>—</td>" if sc is None
                    else f"<td style='background:{_color(sc)}'><b>{sc:.0f}</b></td>")
        rows.append(f"<tr><th class=rowh>{name}</th>{cells}{song}</tr>")
    return (f"<table class=heat><tr><th></th>{head}<th>song</th></tr>"
            + "".join(rows) + "</table>")


def _svg_lines(series: dict[str, list[tuple[float, float | None]]], duration: float,
               unit: str, hlines: tuple[float, ...] = (), sym: bool = False,
               floor: float = 1.0) -> str:
    """A small line chart: one polyline per series (broken at None), with
    optional dashed reference lines. Pure inline SVG, no scripts."""
    vals = [v for pts in series.values() for _, v in pts if v is not None]
    if not vals:
        return "<p class=muted>no data</p>"
    w, h, pl, pr, pt, pb = 720, 200, 48, 12, 10, 24
    top = max(max(abs(v) for v in vals), *(abs(x) for x in hlines), floor) * 1.1
    lo, hi = (-top, top) if sym or min(vals) < 0 else (0.0, top)
    dur = max(duration, 1e-6)
    X = lambda t: pl + (w - pl - pr) * min(max(t / dur, 0.0), 1.0)  # noqa: E731
    Y = lambda v: pt + (h - pt - pb) * (1.0 - (v - lo) / (hi - lo))  # noqa: E731
    parts = [f"<svg viewBox='0 0 {w} {h}' class=chart role=img>"]
    for v in (lo, 0.0, hi) if lo < 0 else (lo, hi / 2, hi):
        parts.append(f"<line x1={pl} x2={w - pr} y1={Y(v):.1f} y2={Y(v):.1f} class=grid />"
                     f"<text x={pl - 4} y={Y(v) + 4:.1f} class=tick text-anchor=end>{v:.0f}</text>")
    for v in hlines:
        parts.append(f"<line x1={pl} x2={w - pr} y1={Y(v):.1f} y2={Y(v):.1f} class=thr />")
    n_ticks = 6
    for i in range(n_ticks + 1):
        t = dur * i / n_ticks
        parts.append(f"<text x={X(t):.1f} y={h - 6} class=tick text-anchor=middle>{_e(_mmss(t))}</text>")
    parts.append(f"<text x=4 y=12 class=tick>{_e(unit)}</text>")
    legend = []
    for i, (name, pts) in enumerate(series.items()):
        c = PALETTE[i % len(PALETTE)]
        run: list[str] = []
        runs = []
        for t, v in pts:
            if v is None:
                if run:
                    runs.append(run)
                run = []
            else:
                run.append(f"{X(t):.1f},{Y(v):.1f}")
        if run:
            runs.append(run)
        for r in runs:
            if len(r) == 1:
                x, y = r[0].split(",")
                parts.append(f"<circle cx={x} cy={y} r=3 fill='{c}' />")
            else:
                parts.append(f"<polyline fill=none stroke='{c}' stroke-width=2 points='{' '.join(r)}' />")
        legend.append(f"<span class=key><i style='background:{c}'></i>{_e(name)}</span>")
    parts.append("</svg>")
    return "".join(parts) + f"<div class=legend>{''.join(legend)}</div>"


def _window_series(result: dict, metric: str, parts=None) -> dict:
    out = {}
    for key, res in list(result["parts"].items()) + [("mix", result["mix"])]:
        if parts is not None and key not in parts:
            continue
        if res["silent"] or res["missing"] or not res["windows"]:
            continue
        pts = [((w["a"] + w["b"]) / 2, (w.get("raw") or {}).get(metric)) for w in res["windows"]]
        if any(v is not None for _, v in pts):
            out[key] = pts
    return out


def _lag_series(result: dict) -> dict:
    out = {}
    for key, wins in result.get("drift", {}).items():
        pts = [((w["a"] + w["b"]) / 2, w["lag_ms"]) for w in wins]
        if any(v is not None for _, v in pts):
            out[key] = pts
    return out


def worst_slices(result: dict, n: int = N_WORST) -> list[dict]:
    """The `n` lowest-scoring (part, section) slices over the scored parts."""
    cands = []
    for key, res in result["parts"].items():
        if res["silent"] or res["missing"]:
            continue
        for s in res["sections"]:
            if s["score"] is not None:
                cands.append({"part": key, "label": s["label"], "score": s["score"],
                              "a": s["a"], "b": s["b"], "files": res.get("files", {})})
    cands.sort(key=lambda c: c["score"])
    return cands[:n]


def _cut(src: Path, dst: Path, start: float, length: float) -> None:
    import soundfile as sf

    info = sf.info(str(src))
    i0 = int(round(start * info.samplerate))
    y, sr = sf.read(str(src), start=min(i0, info.frames),
                    frames=int(round(length * info.samplerate)), always_2d=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), y if np.size(y) else np.zeros(1, np.float32), sr)


def excerpts(result: dict, out_dir: Path, n: int = N_WORST) -> list[dict]:
    """Cut a 15 s listen pair per worst slice into `out_dir/excerpts/`:
    centred in the slice, clamped to the original stem's length, the same
    span cut from the original stem and from the rebuilt part."""
    import soundfile as sf

    out = []
    for i, w in enumerate(worst_slices(result, n), 1):
        orig, reb = w["files"].get("original"), w["files"].get("rebuild")
        if not orig or not reb:
            continue
        dur = sf.info(orig).duration
        length = min(EXCERPT_S, dur)
        start = min(max((w["a"] + w["b"]) / 2 - length / 2, 0.0), max(dur - length, 0.0))
        o = Path(out_dir) / "excerpts" / f"{i}-original.wav"
        r = Path(out_dir) / "excerpts" / f"{i}-rebuild.wav"
        _cut(Path(orig), o, start, length)
        _cut(Path(reb), r, start, length)
        out.append({**w, "n": i, "start": start, "length": length,
                    "original_wav": f"excerpts/{i}-original.wav",
                    "rebuild_wav": f"excerpts/{i}-rebuild.wav"})
    return out


_CSS = """
:root{--bg:#fbfbfa;--fg:#1c1c1a;--muted:#6b6b66;--line:#e2e1dc;--card:#fff;--na:#ecebe6;
--miss:#8a1c1c;--thr:#c92a2a}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#161615;--fg:#ecebe6;
--muted:#9a9a93;--line:#34332f;--card:#1f1f1d;--na:#2a2a27;--miss:#e06666;--thr:#ff6b6b}}
:root[data-theme="dark"]{--bg:#161615;--fg:#ecebe6;--muted:#9a9a93;--line:#34332f;--card:#1f1f1d;
--na:#2a2a27;--miss:#e06666;--thr:#ff6b6b}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,sans-serif;
margin:0 auto;padding:16px;max-width:1100px}
h1{font-size:22px;margin:4px 0}h2{font-size:17px;margin:28px 0 8px}
.muted,small{color:var(--muted)}
.big{font-size:40px;font-weight:700;line-height:1}
.scroll{overflow-x:auto}
table{border-collapse:collapse}
td,th{padding:5px 8px;border-bottom:1px solid var(--line);text-align:center}
.heat td{color:#fff;min-width:52px;font-variant-numeric:tabular-nums}
.heat td.na{background:var(--na);color:var(--muted)}
.heat td.miss{background:var(--na);color:var(--miss);font-weight:600}
th.rowh{text-align:left;white-space:nowrap}
.chart{width:100%;height:auto;background:var(--card);border:1px solid var(--line);border-radius:6px}
.chart .grid{stroke:var(--line)}.chart .thr{stroke:var(--thr);stroke-dasharray:5 4}
.chart .tick{fill:var(--muted);font-size:11px}
.legend{margin:4px 0 0}.key{margin-right:14px;white-space:nowrap}
.key i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}
audio{width:220px;max-width:100%}
.worst td{text-align:left}
"""


def html(result: dict, out_dir: Path) -> Path:
    """Write `out_dir/report.html` (and the worst-slice excerpts); return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ex = excerpts(result, out_dir)
    dur = result["duration"]

    vocal = [k for k, r in result["parts"].items() if r.get("type") == "vocal"]
    lag = _svg_lines(_lag_series(result), dur, "lag ms", hlines=(DRIFT_MS, -DRIFT_MS), sym=True,
                     floor=DRIFT_MS * 2)
    f0 = (_svg_lines(_window_series(result, "f0_cents", vocal), dur, "cents")
          if vocal else "<p class=muted>no vocal parts</p>")
    level = _svg_lines(_window_series(result, "level_diff_db"), dur, "dB (rebuild − original)",
                       sym=True, floor=3.0)

    rows = []
    for w in ex:
        rows.append(
            f"<tr><td>{w['n']}</td><td>{_e(w['part'])}</td><td>{_e(w['label'])}</td>"
            f"<td>{_e(_mmss(w['a']))}–{_e(_mmss(w['b']))}</td>"
            f"<td style='color:#fff;background:{_color(w['score'])}'>{w['score']:.0f}</td>"
            f"<td><audio controls preload=none src='{_e(w['original_wav'])}'></audio></td>"
            f"<td><audio controls preload=none src='{_e(w['rebuild_wav'])}'></audio></td></tr>")
    worst_tbl = ("<table class=worst><tr><th>#</th><th>part</th><th>section</th><th>time</th>"
                 "<th>score</th><th>original</th><th>rebuild</th></tr>" + "".join(rows) + "</table>"
                 if rows else "<p class=muted>no scored slices</p>")

    w = result.get("worst")
    worst_line = (f"worst: {_e(w['part'])} · {_e(w['label'])} @ {_e(_mmss(w.get('a')))} "
                  f"({w['score']:.0f})" if w else "")
    drifting = [k for k, wins in result.get("drift", {}).items() if any(x["drift"] for x in wins)]
    drift_line = (f"drift (|lag| &gt; {DRIFT_MS:.0f} ms) in: {_e(', '.join(drifting))}"
                  if drifting else "no drift over 30 ms")
    score = result["score"]
    page = f"""<!doctype html>
<html lang=en><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(result['song'])} score</title><style>{_CSS}</style></head><body>
<h1>{_e(result['song'])}</h1>
<p class=muted>{_e(_mmss(dur))} · scored per part, section and 20 s window</p>
<div class=big>{_e(_num(score))}</div>
<p>{worst_line}<br>{drift_line}</p>
<h2>Parts × sections</h2><div class=scroll>{_heatmap(result)}</div>
<h2>Lag per window</h2><p class=muted>dashed: the ±{DRIFT_MS:.0f} ms drift threshold</p>{lag}
<h2>Vocal pitch error per window</h2>{f0}
<h2>Level difference per window</h2>{level}
<h2>Worst slices</h2><div class=scroll>{worst_tbl}</div>
<h2>Table</h2><pre class=scroll>{_e(table(result))}</pre>
</body></html>
"""
    path = out_dir / "report.html"
    path.write_text(page, encoding="utf-8")
    return path
