// Signal analysis on the full mix: essentia.js (WASM) for beats, key, chroma
// and predominant melody; a small JS STFT for :mix scalars and drum onsets.

import { stft, binFreq } from './stft.js';
import { PC_NAMES, BEATS_PER_BAR, position, duration, clamp, centsToName, chunkJoin } from './sc.js';

const ESSENTIA = 'https://cdn.jsdelivr.net/npm/essentia.js@0.1.3/dist';
let _essentia = null;

export async function loadEssentia() {
  if (_essentia) return _essentia;
  const { EssentiaWASM } = await import(`${ESSENTIA}/essentia-wasm.es.js`);
  const { default: Essentia } = await import(`${ESSENTIA}/essentia.js-core.es.js`);
  _essentia = new Essentia(EssentiaWASM);
  return _essentia;
}

const free = (...vs) => { for (const v of vs) { try { v && v.delete && v.delete(); } catch { /* ignore */ } } };

// ---------------------------------------------------------------- grid ----
export function stageGrid(E, mono, sr, dur, lowFlux) {
  const st = { name: 'grid', src: 'essentia.js:RhythmExtractor2013', conf: 0, warns: [], lines: [] };
  const v = E.arrayToVector(mono);
  const r = E.RhythmExtractor2013(v, 208, 'multifeature', 40);
  const ticks = Array.from(E.vectorToArray(r.ticks));
  const bpm = r.bpm;
  free(v, r.ticks, r.estimates, r.bpmIntervals);
  if (!ticks.length || !(bpm > 0)) { st.warns.push('no stable pulse detected'); return { st, grid: null }; }
  // multifeature confidence is 0..5.32; >3.5 is "excellent" per essentia docs
  st.conf = clamp(r.confidence / 3.5, 0, 1);

  // Downbeat: of the first four beats, the one whose every-fourth beats carry
  // the most low-band onset energy (kick drums usually land on 1). The native
  // encoder takes the first beat; this is a cheap improvement, not a model.
  let phase = 0;
  if (lowFlux) {
    let best = -1;
    for (let p = 0; p < Math.min(4, ticks.length); p++) {
      let s = 0, n = 0;
      for (let i = p; i < ticks.length; i += 4) { s += lowFlux.at(ticks[i]); n++; }
      if (n && s / n > best) { best = s / n; phase = p; }
    }
  }
  const barDur = BEATS_PER_BAR * 60 / bpm;
  // step back whole bars to the start, so the opening is in bar:beat too
  let downbeat = ticks[phase];
  while (downbeat - barDur >= 0) downbeat -= barDur;
  const nBars = Math.max(Math.floor((dur - downbeat) / barDur), 1);
  st.lines.push(`meter   @${downbeat.toFixed(3)}   4/4`);
  for (let bar = 1; bar <= nBars + 1; bar += 16) {
    st.lines.push(`anchor  bar ${bar}    @${(downbeat + (bar - 1) * barDur).toFixed(3)}`);
  }
  st.lines.push(`tempo   @0.000   ${bpm.toFixed(2)}`);
  if (phase) st.warns.push(`downbeat = beat ${phase + 1} of the first bar by low-band onset energy`);
  st.ok = true;
  return { st, grid: { tempo: bpm, downbeat, barDur, nBars, ticks } };
}

// ----------------------------------------------------------------- key ----
export function stageKey(E, mono) {
  const v = E.arrayToVector(mono);
  const k = E.KeyExtractor(v);
  free(v);
  return { key: k.key, scale: k.scale, strength: k.strength };
}

// ------------------------------------------------------- chroma (HPCP) ----
// Returns frames of 12-bin chroma with index 0 = C.
export function chromaFrames(E, mono, sr, frameSize = 4096, hop = 2048) {
  const frames = [];
  const times = [];
  const win = new Float32Array(frameSize);
  for (let start = 0; start + frameSize <= mono.length; start += hop) {
    win.set(mono.subarray(start, start + frameSize));
    const fv = E.arrayToVector(win);
    const w = E.Windowing(fv, true, frameSize, 'blackmanharris62');
    const s = E.Spectrum(w.frame, frameSize);
    const p = E.SpectralPeaks(s.spectrum, 0.00001, 5000, 60, 40, 'magnitude', sr);
    const h = E.HPCP(p.frequencies, p.magnitudes, true, 500, 0, 5000, false, 40, false, 'unitMax', 440, sr, 12, 'squaredCosine', 1);
    const a = E.vectorToArray(h.hpcp);
    const c = new Float32Array(12);
    for (let i = 0; i < 12; i++) c[(9 + i) % 12] = a[i];   // HPCP bin 0 is A (ref 440)
    frames.push(c);
    times.push((start + frameSize / 2) / sr);
    free(fv, w.frame, s.spectrum, p.frequencies, p.magnitudes, h.hpcp);
  }
  return { frames, times };
}

export function stageHarmony(chroma, grid) {
  const st = { name: 'harmony', src: 'essentia.js:HPCP+chord-template', conf: 0, warns: [], lines: [] };
  if (!grid) { st.warns.push('no grid'); return st; }
  const maj = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], min = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0];
  const T = [], names = [];
  for (let pc = 0; pc < 12; pc++) {
    for (const [tpl, suf] of [[maj, ''], [min, 'm']]) {
      const r = tpl.map((_, i) => tpl[(i - pc + 12) % 12]);
      const n = Math.hypot(...r);
      T.push(r.map(x => x / n)); names.push(PC_NAMES[pc] + suf);
    }
  }
  const chords = [], confs = [];
  for (let bar = 0; bar < grid.nBars; bar++) {
    const t0 = grid.downbeat + bar * grid.barDur, t1 = t0 + grid.barDur;
    const v = new Float64Array(12); let n = 0;
    chroma.times.forEach((t, i) => { if (t >= t0 && t < t1) { for (let k = 0; k < 12; k++) v[k] += chroma.frames[i][k]; n++; } });
    const norm = Math.hypot(...v);
    if (!n || norm < 1e-6) { chords.push(null); confs.push(0); continue; }
    const scores = T.map(t => t.reduce((s, x, k) => s + x * v[k] / norm, 0));
    const order = scores.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]);
    const margin = order[0][0] - order[1][0];
    chords.push(names[order[0][1]]);
    confs.push(clamp(0.5 + margin * 3, 0, 0.99));
  }
  for (let s = 0; s < chords.length; s += 4) {
    const cells = chords.slice(s, s + 4).map((ch, i) => {
      const cf = confs[s + i];
      return ch == null ? 'N' : (cf < 0.8 ? `${ch} ?${cf.toFixed(2)}` : ch);
    });
    st.lines.push(`bars ${s + 1}-${s + cells.length}   ` + cells.join(' | '));
  }
  const nz = confs.filter(Boolean);
  st.conf = nz.length ? nz.reduce((a, b) => a + b, 0) / nz.length : 0;
  st.ok = st.lines.length > 0;
  return st;
}

// ------------------------------------------------- spectral features ----
// One STFT of the mono mix (and L/R for width), shared by :mix and drums.
export function spectral(ch) {
  const S = stft(ch.mono, ch.sr, 2048, 512);
  const SL = ch.stereo ? stft(ch.L, ch.sr, 2048, 512) : null;
  const SR = ch.stereo ? stft(ch.R, ch.sr, 2048, 512) : null;
  return { S, SL, SR };
}

function percentile(arr, p) {
  const a = Float64Array.from(arr).sort();
  const idx = (a.length - 1) * p / 100;
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  return a[lo] + (a[hi] - a[lo]) * (idx - lo);
}

// Native :mix semantics (encode.py stage_mix): "lufs_int" is the RMS level of
// the mono mix in dBFS and "lufs_range" the p95/p10 frame-RMS spread -- kept
// identical so the two encoders' numbers are comparable. True EBU R128 is
// reported alongside as a comment.
export function stageMix(E, ch, spec) {
  const st = { name: 'mix', src: 'js:spectral', conf: 0.95, warns: [], lines: [] };
  const { mono, sr } = ch;
  let ss = 0; for (const x of mono) ss += x * x;
  const lufs = 20 * Math.log10(Math.sqrt(ss / mono.length) + 1e-9);
  const rms = [];
  const fl = 2048, hop = 512;
  for (let f = 0; f * hop < mono.length; f++) {
    const c = f * hop - fl / 2; let s = 0;
    for (let i = 0; i < fl; i++) { const k = c + i; const x = k >= 0 && k < mono.length ? mono[k] : 0; s += x * x; }
    rms.push(Math.sqrt(s / fl));
  }
  const dyn = 20 * Math.log10((percentile(rms, 95) + 1e-9) / (percentile(rms, 10) + 1e-9));
  st.lines.push(`lufs_int      ${lufs.toFixed(1)}`);
  st.lines.push(`lufs_range    ${dyn.toFixed(1)}`);
  const { S, SL, SR } = spec;
  const nb = S.mags[0].length;
  const freqs = Array.from({ length: nb }, (_, i) => binFreq(i, S.nfft, sr));
  if (SL) {
    const bands = { low: f => f < 250, mid: f => f >= 250 && f < 4000, high: f => f >= 4000 };
    const parts = [];
    for (const [nm, sel] of Object.entries(bands)) {
      let num = 0, den = 0;
      for (let t = 0; t < SL.mags.length; t++) {
        const l = SL.mags[t], r = SR.mags[t];
        for (let i = 0; i < nb; i++) if (sel(freqs[i])) { num += Math.abs(l[i] - r[i]); den += Math.abs(l[i] + r[i]); }
      }
      parts.push(`${nm} ${clamp(num / (den + 1e-9), 0, 1).toFixed(2)}`);
    }
    st.lines.push('stereo_width  ' + parts.join(', '));
  }
  let cSum = 0, rSum = 0;
  for (const m of S.mags) {
    let tot = 0, wsum = 0;
    for (let i = 0; i < nb; i++) { tot += m[i]; wsum += m[i] * freqs[i]; }
    cSum += tot > 0 ? wsum / tot : 0;
    // librosa rolloff: smallest bin whose cumulative energy (magnitude) >= 95%
    let acc = 0, ri = nb - 1;
    for (let i = 0; i < nb; i++) { acc += m[i]; if (acc >= 0.95 * tot) { ri = i; break; } }
    rSum += freqs[ri];
  }
  st.lines.push(`centroid      ${(cSum / S.mags.length).toFixed(0)}Hz`);
  st.lines.push(`rolloff_95    ${(rSum / S.mags.length).toFixed(0)}Hz`);
  try {
    if (ch.stereo) {
      const lv = E.arrayToVector(ch.L), rv = E.arrayToVector(ch.R);
      const lo = E.LoudnessEBUR128(lv, rv, 0.1, sr, false);
      st.comment = `# EBU R128 (essentia.js): integrated ${lo.integratedLoudness.toFixed(1)} LUFS, range ${lo.loudnessRange.toFixed(1)} LU`;
      free(lv, rv, lo.momentaryLoudness, lo.shortTermLoudness);
    }
  } catch (e) { st.warns.push(`EBU R128 failed: ${e}`); }
  st.ok = true;
  return { st, rms, rmsHop: hop / sr };
}

// --------------------------------------------------------------- drums ----
// Band-wise spectral flux on the full mix (no drum stem in the browser).
// band (Hz), peak-picking threshold (in SDs of the band's flux) and min gap
// Tuned on Discipline against the native (stem-based) drum events:
// onset F1 kick 0.53, snare 0.50, hat 0.80 (was 0.52/0.23/0.80 with a
// 180-320 Hz snare band). One clip -- treat as a starting point.
const DRUM_PARAMS = {
  kick: { band: [40, 100], delta: 1.2, minGap: 0.2 },
  snare: { band: [1500, 5000], delta: 3.5, minGap: 0.3 },
  hat: { band: [7000, 16000], delta: 0.8, minGap: 0.08 },
};
const DRUM_BANDS = Object.fromEntries(Object.entries(DRUM_PARAMS).map(([k, v]) => [k, v.band]));

function bandFlux(S, lo, hi) {
  const nb = S.mags[0].length;
  const i0 = Math.max(1, Math.round(lo * S.nfft / S.sr)), i1 = Math.min(nb - 1, Math.round(hi * S.nfft / S.sr));
  const flux = new Float32Array(S.mags.length);
  let prev = null;
  for (let t = 0; t < S.mags.length; t++) {
    const m = S.mags[t];
    const cur = new Float32Array(i1 - i0 + 1);
    for (let i = i0; i <= i1; i++) cur[i - i0] = Math.log1p(100 * m[i]);
    if (prev) { let s = 0; for (let i = 0; i < cur.length; i++) s += Math.max(0, cur[i] - prev[i]); flux[t] = s / cur.length; }
    prev = cur;
  }
  return flux;
}

export function lowBandFlux(spec) {
  const S = spec.S;
  const flux = bandFlux(S, ...DRUM_BANDS.kick);
  const fps = S.sr / S.hop;
  return { flux, at: t => { const f = Math.round(t * fps); let m = 0; for (let k = f - 2; k <= f + 2; k++) if (k >= 0 && k < flux.length) m = Math.max(m, flux[k]); return m; } };
}

function pickPeaks(flux, fps, { w = 3, avg = 10, delta = 1.0, minGap = 0.06 } = {}) {
  let mean = 0; for (const x of flux) mean += x; mean /= flux.length;
  let sd = 0; for (const x of flux) sd += (x - mean) ** 2; sd = Math.sqrt(sd / flux.length);
  const peaks = []; let last = -1e9;
  for (let t = 0; t < flux.length; t++) {
    let isMax = true;
    for (let k = Math.max(0, t - w); k <= Math.min(flux.length - 1, t + w); k++) if (flux[k] > flux[t]) { isMax = false; break; }
    if (!isMax) continue;
    let local = 0, n = 0;
    for (let k = Math.max(0, t - avg); k <= Math.min(flux.length - 1, t + avg); k++) { local += flux[k]; n++; }
    if (flux[t] < local / n + delta * sd) continue;
    if ((t - last) / fps < minGap) continue;
    peaks.push(t); last = t;
  }
  return peaks;
}

// Median-filter HPSS (Fitzgerald 2010) on the magnitude spectrogram up to
// 8 kHz: the percussive share of energy per 2 s block. Gates the drum stream
// the way the native encoder's loudness gate silences an empty drum stem.
function median(arr) { const a = Float32Array.from(arr).sort(); return a[a.length >> 1]; }
export function percussiveShare(S, blockS = 2, kt = 17, kf = 17) {
  const nb = Math.min(S.mags[0].length, Math.round(8000 * S.nfft / S.sr));
  const T = S.mags.length, ht = kt >> 1, hf = kf >> 1;
  const fps = S.sr / S.hop, nBlocks = Math.ceil(T / fps / blockS);
  const P = new Float64Array(nBlocks), H = new Float64Array(nBlocks);
  const buf = [];
  for (let t = 0; t < T; t++) {
    const m = S.mags[t], blk = Math.min(nBlocks - 1, Math.floor(t / fps / blockS));
    for (let f = 0; f < nb; f++) {
      buf.length = 0;
      for (let k = Math.max(0, t - ht); k <= Math.min(T - 1, t + ht); k++) buf.push(S.mags[k][f]);
      const h = median(buf);
      buf.length = 0;
      for (let k = Math.max(0, f - hf); k <= Math.min(nb - 1, f + hf); k++) buf.push(m[k]);
      const p = median(buf);
      // soft (Wiener) masks, power 2
      const e = m[f] * m[f], d = h * h + p * p + 1e-12;
      H[blk] += e * h * h / d; P[blk] += e * p * p / d;
    }
  }
  return Array.from(P, (p, i) => p / (p + H[i] + 1e-12));
}

// Calibrated on the four spike clips only: Discipline (rock kit) blocks read
// 0.27-0.35; the three drumless clips 0.07-0.25.
export const PERC_GATE = 0.26;

export function stageDrums(spec, grid, perc, override = {}) {
  const st = { name: 'perc.drums', fields: { inst: 'drums.kit' }, src: 'js:bandflux+hpss-gate', conf: 0.35, warns: [], lines: [] };
  if (!grid) { st.warns.push('no grid'); return st; }
  const S = spec.S, fps = S.sr / S.hop;
  const open = perc ? perc.map(x => x >= PERC_GATE) : null;
  st.debug = perc ? perc.map(x => x.toFixed(2)).join(' ') : '';
  if (open && !open.some(Boolean)) {
    st.gated = true;
    st.why = `percussive share below ${PERC_GATE} in every 2 s block (max ${Math.max(...perc).toFixed(2)})`;
    return st;
  }
  st.warns.push('no drum stem in the browser; onsets from band-wise spectral flux on the full mix');
  if (open) st.warns.push(`kept in ${open.filter(Boolean).length} of ${open.length} 2-s blocks (HPSS percussive share >= ${PERC_GATE})`);
  const events = [];
  for (const [voice, p0] of Object.entries(DRUM_PARAMS)) {
    const p = { ...p0, ...(override[voice] || {}) };
    const flux = bandFlux(S, ...p.band);
    const peaks = pickPeaks(flux, fps, { delta: p.delta, minGap: p.minGap });
    let peak = 0; for (const p of peaks) peak = Math.max(peak, flux[p]);
    for (const p of peaks) {
      const t = p / fps;
      if (t < grid.downbeat - 1e-6) continue;
      if (open && !open[Math.min(open.length - 1, Math.floor(t / 2))]) continue;
      const vel = Math.round(clamp(30 + 97 * Math.sqrt(flux[p] / (peak || 1)), 1, 127));
      events.push([t, voice, vel]);
    }
  }
  events.sort((a, b) => a[0] - b[0]);
  st.lines = events.map(([t, v, vel]) => `${position(t, grid)} ${v} ${vel}`);
  st.ok = st.lines.length > 0;
  return st;
}

// -------------------------------------------------------------- melody ----
// essentia's PredominantPitchMelodia (Salamon & Gomez 2012) on the mix: the
// classic non-ML melody extractor. Stands in for CREPE on a separated vocal.
export function melody(E, mono, sr) {
  const v = E.arrayToVector(mono);
  const eq = E.EqualLoudness(v, sr);
  const hop = 128;
  const m = E.PredominantPitchMelodia(eq.signal, 10, 3, 2048, false, 0.8, hop, 1, 40, 20000, 100, 80, 20, 0.9, 0.9, 27.5625, 55, sr, 100, false, 0.2);
  const pitch = E.vectorToArray(m.pitch).slice();
  const conf = E.vectorToArray(m.pitchConfidence).slice();
  free(v, eq.signal, m.pitch, m.pitchConfidence);
  const times = Array.from(pitch, (_, i) => i * hop / sr);
  return { pitch, conf, times, hop };
}

export function segmentMelody(E, pitchHz, mono, sr, hop) {
  const pv = E.arrayToVector(pitchHz), sv = E.arrayToVector(mono);
  const r = E.PitchContourSegmentation(pv, sv, hop, 0.1, 60, -2, sr, 440);
  const out = { onset: Array.from(E.vectorToArray(r.onset)), duration: Array.from(E.vectorToArray(r.duration)), midi: Array.from(E.vectorToArray(r.MIDIpitch)) };
  free(pv, sv, r.onset, r.duration, r.MIDIpitch);
  return out;
}

// ------------------------------------------------------------- struct ----
// Bar-level chroma + loudness novelty; k sections like the native encoder.
export function stageStruct(chroma, mixRms, grid, dur, words) {
  const st = { name: 'struct', src: 'js:bar-novelty', conf: 0.5, warns: [], lines: [] };
  if (!grid) { st.lines.push(`all     @0.000-@${dur.toFixed(3)} ${words.length ? 'vocal' : 'inst'}  energy=1.00`); st.ok = true; return st; }
  const nB = grid.nBars;
  const feat = [];
  for (let b = 0; b < nB; b++) {
    const t0 = grid.downbeat + b * grid.barDur, t1 = t0 + grid.barDur;
    const v = new Float64Array(13); let n = 0;
    chroma.times.forEach((t, i) => { if (t >= t0 && t < t1) { for (let k = 0; k < 12; k++) v[k] += chroma.frames[i][k]; n++; } });
    const nn = Math.hypot(...v.slice(0, 12)) || 1;
    for (let k = 0; k < 12; k++) v[k] /= nn;
    const r0 = Math.floor(t0 / mixRms.hop), r1 = Math.min(mixRms.rms.length, Math.ceil(t1 / mixRms.hop));
    let e = 0; for (let i = r0; i < r1; i++) e += mixRms.rms[i]; v[12] = e / Math.max(1, r1 - r0);
    feat.push(v);
  }
  const peakE = Math.max(...feat.map(f => f[12])) || 1;
  const k = Math.max(2, Math.min(6, Math.floor(dur / 12)));
  // novelty at bar boundary b: distance between the 2-bar windows either side
  const nov = [];
  for (let b = 2; b <= nB - 2; b++) {
    const mean = (a, z) => { const m = new Float64Array(13); for (let i = a; i < z; i++) for (let j = 0; j < 13; j++) m[j] += feat[i][j] / (z - a); return m; };
    const A = mean(b - 2, b), B = mean(b, b + 2);
    let d = 0; for (let j = 0; j < 12; j++) d += (A[j] - B[j]) ** 2;
    d += 4 * ((A[12] - B[12]) / peakE) ** 2;
    nov.push([d, b]);
  }
  nov.sort((a, b) => b[0] - a[0]);
  const bounds = [];
  for (const [, b] of nov) {
    if (bounds.length >= k - 1) break;
    if (bounds.every(x => Math.abs(x - b) >= 4)) bounds.push(b);
  }
  const edges = [0, ...bounds.sort((a, b) => a - b), nB];
  const labels = ['intro', 'verse', 'chorus', 'verse', 'chorus', 'outro', 'coda'];
  for (let i = 0; i < edges.length - 1; i++) {
    const a = edges[i], z = edges[i + 1];
    let e = 0; for (let b = a; b < z; b++) e += feat[b][12]; e /= (z - a) * peakE;
    const t0 = grid.downbeat + a * grid.barDur, t1 = grid.downbeat + z * grid.barDur;
    const vocal = words.some(w => w.start < t1 && w.end > t0);
    st.lines.push(`${labels[Math.min(i, labels.length - 1)].padEnd(8)}${a + 1}-${String(z).padEnd(6)} ${(vocal ? 'vocal' : 'inst').padEnd(6)} energy=${e.toFixed(2)}`);
  }
  st.warns.push('sections from bar-level chroma/loudness novelty; labels are positional; vocal flag = ASR heard words');
  st.ok = true;
  return st;
}

export { position, duration, centsToName, chunkJoin };
