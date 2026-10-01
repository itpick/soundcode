// .sc text helpers, ported from the native encoder (src/soundcode/encode.py,
// tsumugi_sc.py, pitch.py, contour.py) so the browser output follows the same
// conventions: one tempo, one downbeat, anchors every 16 bars, 3-decimal
// bar:beat positions, durations in beats after the downbeat and in seconds
// before it.

const SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
export const PC_NAMES = SHARP;
export const BEATS_PER_BAR = 4;

export function centsToName(cents) {
  const nearest = Math.round(cents / 100);
  const offset = Math.round(cents - nearest * 100);
  const semis = ((nearest % 12) + 12) % 12;
  const octave = Math.floor(nearest / 12);
  let name = `${SHARP[semis]}${octave - 1}`;
  if (offset) name += `${offset > 0 ? '+' : ''}${offset}c`;
  return name;
}

export function f3(x) { return (Math.round(x * 1000) / 1000).toFixed(3); }

// Python's round-half-even on .xxx5 is not reproduced; positions differ by at
// most 1 ms from what Python would print, well under every tolerance.
export function position(t, grid) {
  if (t < grid.downbeat - 1e-6) return `@${f3(t)}`;
  let beats = (t - grid.downbeat) / (grid.barDur / BEATS_PER_BAR);
  beats = Math.round(beats * 1000) / 1000;
  const bar = Math.floor(beats / BEATS_PER_BAR);
  const beat = beats - bar * BEATS_PER_BAR;
  return `${bar + 1}:${(beat + 1).toFixed(3)}`;
}

export function duration(pos, seconds, grid, floor = 0.01) {
  const d = Math.max(seconds, floor);
  return pos.startsWith('@') ? `${d.toFixed(3)}s` : `${(d / (grid.barDur / BEATS_PER_BAR)).toFixed(3)}b`;
}

export function clamp(x, a, b) { return Math.min(b, Math.max(a, x)); }

export function stageLines(st) {
  const fields = Object.entries(st.fields || {}).map(([k, v]) => ` ${k}=${v}`).join('');
  const out = [`:${st.name}${fields}`, `meta    src=${st.src}  conf=${(st.conf ?? 0).toFixed(2)}`];
  if (st.stem) {
    const level = st.levelDb != null ? `  level=${st.levelDb.toFixed(1)}dB` : '';
    out.push(`meta    stem=${st.stem}${level}`);
  }
  for (const [k, v] of Object.entries(st.meta || {})) out.push(`meta    ${k}=${v}`);
  for (const w of st.warns || []) out.push(`meta    warn="${String(w).replace(/"/g, "'")}"`);
  return out.concat(st.lines, ['']);
}

// contour.py phrases(): resample a voiced f0 track onto a 50 Hz grid, bridging
// gaps shorter than 60 ms, and cut it into runs.
export const CONTOUR_RATE = 50;
export function contourPhrases(times, cents, voiced) {
  const STEP = 1 / CONTOUR_RATE, GAP = 0.06;
  if (!voiced.some(Boolean)) return [];
  const t0 = times[0], t1 = times[times.length - 1];
  const grid = [];
  for (let t = t0; t <= t1 + 1e-9; t += STEP) grid.push(t);
  const vt = [], vc = [];
  for (let i = 0; i < times.length; i++) if (voiced[i]) { vt.push(times[i]); vc.push(cents[i]); }
  const interp = (x, xs, ys) => {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
    let lo = 0, hi = xs.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
    return ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / (xs[hi] - xs[lo]);
  };
  const vf = voiced.map(v => (v ? 1 : 0));
  const v = grid.map(t => interp(t, times, vf) > 0.5);
  const c = grid.map(t => interp(t, vt, vc));
  const out = [];
  let i = 0; const n = grid.length;
  while (i < n) {
    if (!v[i]) { i++; continue; }
    let j = i;
    while (j < n) {
      if (v[j]) { j++; continue; }
      let k = j;
      while (k < n && !v[k]) k++;
      if ((k - j) * STEP < GAP && k < n) j = k; else break;
    }
    out.push([grid[i], c.slice(i, j).map(x => Math.round(x))]);
    i = j;
  }
  return out;
}

export function contourLines(ph, perLine = 50) {
  const lines = [];
  for (const [start, vals] of ph) {
    for (let k = 0; k < vals.length; k += perLine) {
      lines.push(`f0  @${f3(start + k / CONTOUR_RATE)}  ` + vals.slice(k, k + perLine).join(' '));
    }
  }
  return lines;
}

export function chunkJoin(cells, per) {
  const lines = [];
  for (let i = 0; i < cells.length; i += per) lines.push(cells.slice(i, i + per).join(' | '));
  return lines;
}
