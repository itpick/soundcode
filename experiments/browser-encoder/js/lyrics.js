// Lyrics: Whisper (transformers.js) word timestamps on the mix, optionally
// reconciled with LRCLIB's published lyrics (port of src/soundcode/lyrics.py).

import { position, duration, chunkJoin } from './sc.js';

const TRANSFORMERS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6';
let _asr = null, _asrInfo = null;

// `model` is a Hugging Face repo id (fetched from huggingface.co), or
// `local:<dir>` for a copy served from this page's own origin under models/
// (tools/export_whisper.py makes one; used when the HF CDN is unreachable).
export async function loadWhisper(model, prefer, onFile) {
  if (_asr) return _asrInfo;
  const T = await import(TRANSFORMERS);
  const local = model.startsWith('local:');
  if (local) {
    T.env.allowLocalModels = true; T.env.allowRemoteModels = false;
    T.env.localModelPath = new URL('models/', location.href).href;
    model = model.slice(6);
  } else {
    T.env.allowLocalModels = false;
  }
  let device = 'wasm';
  if (prefer !== 'wasm' && navigator.gpu) {
    try { if (await navigator.gpu.requestAdapter()) device = 'webgpu'; } catch { /* no adapter */ }
  }
  // Hub repos ship an int8 decoder (54 MB vs 209 MB) that suits WASM; the
  // local export has fp32 only (dynamic quantisation skips the merged
  // decoder's subgraphs).
  const dtype = device === 'webgpu' || local
    ? { encoder_model: 'fp32', decoder_model_merged: 'fp32' }
    : { encoder_model: 'fp32', decoder_model_merged: 'int8' };
  const files = {};
  const progress_callback = p => {
    if (p.file && (p.status === 'progress' || p.status === 'done')) {
      files[p.file] = { loaded: p.loaded ?? files[p.file]?.loaded ?? 0, total: p.total ?? files[p.file]?.total ?? 0 };
      onFile && onFile(p);
    }
  };
  _asr = await T.pipeline('automatic-speech-recognition', model, { device, dtype, progress_callback });
  const wasm = T.env.backends?.onnx?.wasm;
  _asrInfo = { device, dtype, model: local ? `local:${model}` : model, files, threads: wasm?.numThreads ?? null, version: T.env.version };
  return _asrInfo;
}

export const HALLUCINATIONS = new Set(['thank you', 'thanks for watching', 'thank you for watching', 'you', 'bye',
  'subtitles by the amaraorg community']);

export async function transcribe(audio16k) {
  const out = await _asr(audio16k, { return_timestamps: 'word', chunk_length_s: 30, stride_length_s: 5, language: 'english', task: 'transcribe' });
  // drop non-lyric annotations: [Music], (dramatic music), ♪ -- they can span words
  const words = []; let depth = 0;
  for (const c of out.chunks || []) {
    const word = c.text.trim();
    if (!word || c.timestamp[0] == null) continue;
    const opens = (word.match(/[[(]/g) || []).length, closes = (word.match(/[\])]/g) || []).length;
    const skip = depth > 0 || opens > 0 || /^♪+$/.test(word);
    depth = Math.max(0, depth + opens - closes);
    if (skip) continue;
    words.push({ start: c.timestamp[0], end: c.timestamp[1] ?? c.timestamp[0] + 0.3, word: word.replace(/♪/g, ''), prob: null });
  }
  return { text: out.text, words };
}

export function normalise(w) { return w.toLowerCase().replace(/[^a-z0-9']/g, '').replace(/^'+|'+$/g, ''); }

// ------------------------------------------------------------- LRCLIB ----
export async function lrclib(title, artist) {
  const qs = new URLSearchParams({ track_name: title, ...(artist ? { artist_name: artist } : {}) });
  const r = await fetch(`https://lrclib.net/api/search?${qs}`);
  if (!r.ok) throw new Error(`LRCLIB HTTP ${r.status}`);
  const found = (await r.json()).filter(x => x.syncedLyrics || x.plainLyrics);
  if (!found.length) return null;
  const rec = found[0];
  const lines = [];
  if (rec.syncedLyrics) {
    for (const raw of rec.syncedLyrics.split('\n')) {
      const m = raw.trim().match(/^\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)$/);
      if (m && m[3].trim()) lines.push({ t: +m[1] * 60 + +m[2], text: m[3].trim() });
    }
  } else {
    for (const l of (rec.plainLyrics || '').split('\n')) if (l.trim()) lines.push({ t: null, text: l.trim() });
  }
  return { lines, id: rec.id, synced: !!rec.syncedLyrics };
}

function refWords(lines, start, end) {
  const timed = lines.some(l => l.t != null);
  const out = [];
  for (const l of lines) {
    if (timed && (l.t == null || !(start - 1 <= l.t && l.t <= end))) continue;
    for (const x of l.text.replace(/-/g, ' ').split(/\s+/)) { const w = normalise(x); if (w) out.push({ t: l.t, w }); }
  }
  return out;
}

// difflib-like opcodes from an LCS alignment.
function opcodes(a, b) {
  const n = a.length, m = b.length;
  const L = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) L[i][j] = a[i] === b[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const blocks = []; let i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { const s = [i, j]; while (i < n && j < m && a[i] === b[j]) { i++; j++; } blocks.push([s[0], s[1], i - s[0]]); }
    else if (L[i + 1][j] >= L[i][j + 1]) i++; else j++;
  }
  blocks.push([n, m, 0]);
  const ops = []; let ai = 0, bj = 0;
  for (const [bi, bjj, size] of blocks) {
    const tag = ai < bi && bj < bjj ? 'replace' : ai < bi ? 'delete' : bj < bjj ? 'insert' : null;
    if (tag) ops.push([tag, ai, bi, bj, bjj]);
    if (size) ops.push(['equal', bi, bi + size, bjj, bjj + size]);
    ai = bi + size; bj = bjj + size;
  }
  const M = blocks.reduce((s, x) => s + x[2], 0);
  return { ops, ratio: n + m ? 2 * M / (n + m) : 0 };
}

const CONFIDENT = 0.9;
// The native encoder accepts a published-lyrics match at ratio >= 0.5 with
// large-v3-turbo on a separated vocal. whisper-base on a centre-channel
// approximation mishears more, so the browser accepts 0.35. Discipline's true
// window matches at 0.38 but other windows of that repetitive song score
// 0.32-0.35, so the offset (not the words) is a weak call there.
export const MATCH_RATIO = 0.35;
export function reconcile(asr, ref) {
  const a = asr.map(w => normalise(w.word)), b = ref.map(r => r.w);
  const { ops, ratio } = opcodes(a, b);
  if (ratio < MATCH_RATIO || !ref.length) return { words: asr, ratio };
  const out = [];
  for (const [tag, i1, i2, j1, j2] of ops) {
    if (tag === 'equal') for (let i = i1; i < i2; i++) out.push({ ...asr[i], word: b[j1 + i - i1], prob: 0.9 });
    else if (tag === 'replace') {
      const n = Math.max(i2 - i1, j2 - j1);
      for (let k = 0; k < n; k++) {
        if (k < i2 - i1 && k < j2 - j1) {
          const w = asr[i1 + k];
          if ((w.prob ?? 0) >= CONFIDENT) out.push({ ...w, word: normalise(w.word), alt: b[j1 + k] });
          else out.push({ ...w, word: b[j1 + k], prob: 0.7, alt: normalise(w.word) || null });
        } else if (k < j2 - j1) out.push({ start: null, end: null, word: b[j1 + k], prob: 0.5 });
        else out.push(asr[i1 + k]);
      }
    } else if (tag === 'delete') for (let i = i1; i < i2; i++) out.push(asr[i]);
    else for (let j = j1; j < j2; j++) out.push({ start: null, end: null, word: b[j], prob: 0.5 });
  }
  while (out.length && out[out.length - 1].start == null) out.pop();
  let k = 0;
  while (k < out.length) {
    if (out[k].start != null) { k++; continue; }
    let j = k; while (j < out.length && out[j].start == null) j++;
    const nxt = j < out.length ? out[j].start : null;
    let prev = k > 0 ? out[k - 1].end : null; const n = j - k;
    if (prev == null) { const end = nxt != null ? nxt : 0.4 * n; prev = Math.max(0, end - 0.4 * n); }
    const end = nxt != null ? nxt : prev + 0.4 * n;
    const step = Math.max((end - prev) / n, 0.01);
    for (let m = 0; m < n; m++) { const a0 = prev + m * step; out[k + m] = { ...out[k + m], start: a0, end: a0 + Math.min(step, 0.6), prob: 0.5 }; }
    k = j;
  }
  return { words: out, ratio };
}

export function bestOffset(asr, lines, dur) {
  const starts = [...new Set([0, ...lines.filter(l => l.t != null).map(l => l.t)])].sort((x, y) => x - y);
  let best = 0, bestR = 0;
  for (const off of starts) {
    const ref = refWords(lines, off, off + dur);
    if (!ref.length) continue;
    const { ratio } = reconcile(asr, ref);
    if (ratio > bestR + 1e-9) { best = off; bestR = ratio; }
  }
  return bestR >= MATCH_RATIO ? best : 0;
}

export function referenceFor(asr, lines, dur) {
  const off = bestOffset(asr, lines, dur);
  return { ref: refWords(lines, off, off + dur), offset: off };
}

// ------------------------------------------------------------- stage ----
export function lyricStage(words, grid, src) {
  const st = { name: 'text.vox', fields: { lang: 'en', align: 'word' }, src, conf: 0.7, warns: [], lines: [] };
  const cells = []; let last = null;
  for (const w of [...words].sort((a, b) => a.start - b.start)) {
    if (!w.word) continue;
    let start = w.start;
    if (last != null) start = Math.max(start, last + 0.02);
    last = start;
    const pos = position(start, grid);
    const d = duration(pos, Math.max(w.end - start, 0.05), grid, 0.05);
    const mark = w.prob != null && w.prob < 0.8 ? ` ?${w.prob.toFixed(2)}` : '';
    const alt = w.alt && w.alt !== w.word ? ` alt="${w.alt}"` : '';
    cells.push(`${pos} "${w.word.replace(/"/g, "'")}" ${d}${mark}${alt}`);
  }
  st.lines = chunkJoin(cells, 5);
  st.ok = st.lines.length > 0;
  return st;
}
