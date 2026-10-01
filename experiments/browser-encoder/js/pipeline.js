// The whole browser encoder: audio file bytes in, .sc text out.

import { decode, channels, resampleMono } from './audio.js';
import { centreVocal } from './stft.js';
import * as A from './analysis.js';
import * as N from './notes.js';
import * as L from './lyrics.js';
import { stageLines, contourPhrases, contourLines, CONTOUR_RATE, position, duration, centsToName, clamp } from './sc.js';

export const DEFAULTS = { whisperModel: 'onnx-community/whisper-base_timestamped', backend: 'auto', lrclib: true, vocalInput: 'centre' };

function timer(log) {
  const stages = [];
  return {
    stages,
    async run(name, fn) {
      const t0 = performance.now();
      log(`▶ ${name}`);
      try {
        const r = await fn();
        const ms = performance.now() - t0;
        stages.push({ name, ms, ok: true });
        log(`  ✓ ${name} ${(ms / 1000).toFixed(2)} s`);
        return r;
      } catch (e) {
        const ms = performance.now() - t0;
        stages.push({ name, ms, ok: false, error: String(e && e.stack || e) });
        log(`  ✗ ${name} failed after ${(ms / 1000).toFixed(2)} s: ${e}`);
        return undefined;
      }
    },
  };
}

// Bytes fetched from the network, grouped by package / model repo.
export function downloads() {
  const groups = {};
  for (const e of performance.getEntriesByType('resource')) {
    const u = new URL(e.name);
    if (u.hostname === location.hostname) continue;
    let key = u.hostname;
    const m = u.pathname.match(/^\/npm\/((?:@[^/]+\/)?[^/@]+@[^/]+)/);
    if (m) key = m[1];
    const hf = u.pathname.match(/^\/([^/]+\/[^/]+)\/resolve\//);
    if (hf) key = `hf:${hf[1]}`;
    const g = groups[key] || (groups[key] = { files: 0, bytes: 0, opaque: 0 });
    g.files++;
    const b = e.encodedBodySize || e.transferSize || 0;
    g.bytes += b; if (!b) g.opaque++;
  }
  return groups;
}

export async function encode(bytes, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  const log = o.log || (m => console.log(m));
  const T = timer(log);
  const backends = {};
  const notes = [];
  const fileName = o.fileName || 'input.mp3';
  const title = o.title || fileName.replace(/\.[^.]+$/, '');

  // 1. decode + resample
  const buf = await T.run('decode (Web Audio)', () => decode(bytes));
  if (!buf) throw new Error('could not decode audio');
  const ch = channels(buf);
  const dur = buf.duration;
  const a22 = await T.run('resample 22.05 kHz (basic-pitch)', () => resampleMono(ch.mono, ch.sr, 22050));
  // Whisper and the melody tracker hear the centre channel (no model can
  // separate in the page cheaply; this is the free approximation).
  const useCentre = o.vocalInput === 'centre' && ch.stereo;
  const voc = useCentre ? await T.run('centre-channel extract (JS)', () => centreVocal(ch.L, ch.R, ch.sr)) : ch.mono;
  const a16 = await T.run('resample 16 kHz (whisper)', () => resampleMono(voc, ch.sr, 16000));

  // 2. essentia: grid, key, chroma, mix, drums
  const E = await T.run('load essentia.js (WASM)', () => A.loadEssentia());
  const spec = await T.run('STFT (JS)', () => A.spectral(ch));
  const low = spec ? A.lowBandFlux(spec) : null;
  const g = await T.run('tempo + beats (RhythmExtractor2013)', () => A.stageGrid(E, ch.mono, ch.sr, dur, low));
  const grid = g?.grid || null;
  const tgrid = grid || { downbeat: Infinity, barDur: 2 };      // no pulse: every time in seconds
  const key = await T.run('key (KeyExtractor)', () => A.stageKey(E, ch.mono));
  const chroma = await T.run('chroma (HPCP)', () => A.chromaFrames(E, ch.mono, ch.sr));
  const harmony = chroma ? A.stageHarmony(chroma, grid) : null;
  if (harmony && key) harmony.lines.unshift(`tonal_center ${key.key} ${key.scale} ?${clamp(key.strength, 0, 0.99).toFixed(2)}`);
  const mix = await T.run('mix scalars', () => A.stageMix(E, ch, spec));
  const perc = await T.run('percussive share (HPSS)', () => A.percussiveShare(spec.S));
  const drums = await T.run('drums (band flux)', () => A.stageDrums(spec, grid, perc, o.drumParams));

  // 3. melody on the mix
  const mel = await T.run('melody (PredominantPitchMelodia)', () => A.melody(E, voc, ch.sr));

  // 4. lyrics
  let words = [], lyricSrc = `whisper-js:${o.whisperModel.split('/').pop()}`, lyricWarns = [], offset = 0;
  const asrInfo = await T.run('load whisper (transformers.js)', () => L.loadWhisper(o.whisperModel, o.backend, o.onModelFile));
  if (asrInfo) backends.whisper = asrInfo.device;
  const heard = asrInfo ? await T.run('transcribe (whisper)', () => L.transcribe(a16)) : null;
  if (heard) {
    words = heard.words;
    const norm = heard.text.toLowerCase().replace(/[^a-z0-9 ]/g, '').replace(/\s+/g, ' ').trim();
    if (L.HALLUCINATIONS.has(norm) || !words.length) {
      lyricWarns.push(`ASR heard only "${heard.text.trim()}" (a common hallucination on non-speech); no lyrics`);
      words = [];
    }
  }
  if (words.length && o.lrclib && o.title) {
    const ref = await T.run('LRCLIB lookup', () => L.lrclib(o.title, o.artist));
    if (ref === undefined) lyricWarns.push('LRCLIB lookup failed (network/CORS); ASR only');
    else if (!ref) lyricWarns.push('LRCLIB: no match; ASR only');
    else {
      const { ref: rw, offset: off } = L.referenceFor(words, ref.lines, dur);
      const r = L.reconcile(words, rw);
      if (r.ratio >= L.MATCH_RATIO) { words = r.words; lyricSrc = `lrclib+${lyricSrc.split(':')[1]}`; offset = off; }
      else lyricWarns.push(`published lyrics did not match this recording (ratio ${r.ratio.toFixed(2)}); ASR only`);
    }
  }

  // 5. lead vocal: melody gated to where words were heard
  let lead = [], leadStage = null, contourStage = null;
  if (mel && words.length) {
    await T.run('lead vocal notes + contour', () => {
      const regions = words.map(w => [w.start - 0.15, w.end + 0.15]);
      const inVocal = t => regions.some(([a, b]) => t >= a && t <= b);
      const gated = new Float32Array(mel.pitch.length);
      const voiced = [], cents = [];
      let cs = 0, cn = 0;
      for (let i = 0; i < mel.pitch.length; i++) {
        const hz = mel.pitch[i];
        const v = hz > 0 && inVocal(mel.times[i]);
        if (v) { gated[i] = hz; cs += mel.conf[i]; cn++; }
        voiced.push(v);
        cents.push(v ? 1200 * Math.log2(hz / 440) + 6900 : 0);
      }
      const seg = A.segmentMelody(E, gated, voc, ch.sr, mel.hop);
      const rmsAt = t => { const i = Math.min(mix.rms.length - 1, Math.round(t / mix.rmsHop)); return mix.rms[Math.max(0, i)]; };
      const peak = Math.max(...mix.rms);
      lead = seg.onset.map((s, i) => ({ start: s, end: s + seg.duration[i], midi: Math.round(seg.midi[i]) }));
      leadStage = { name: 'notes.lead', fields: { inst: 'voice.lead' }, src: 'essentia.js:PitchContourSegmentation', stem: 'lead_vocals',
        conf: cn ? cs / cn : 0, warns: [`melody from the ${useCentre ? 'centre channel' : 'full mix'} (no vocal stem), kept only where ASR heard words`], lines: [] };
      for (const n of lead) {
        const pos = position(n.start, tgrid);
        const vel = Math.round(clamp(20 + 107 * Math.sqrt(rmsAt(n.start) / peak), 1, 127));
        leadStage.lines.push(`${pos}  ${centsToName(n.midi * 100)}  ${duration(pos, n.end - n.start, tgrid)} ${vel}`);
      }
      leadStage.ok = leadStage.lines.length > 0;
      contourStage = { name: 'contour.vox', fields: { rate: String(CONTOUR_RATE) }, src: 'essentia.js:PredominantPitchMelodia', stem: 'lead_vocals',
        conf: cn ? cs / cn : 0, warns: [], lines: contourLines(contourPhrases(mel.times, cents, voiced)) };
      contourStage.ok = contourStage.lines.length > 0;
    });
  }

  // 6. notes
  const bpInfo = o.skipNotes ? null : await T.run('load basic-pitch (TF.js)', () => N.loadBasicPitch(o.backend));
  if (bpInfo) backends.basicPitch = bpInfo.backend;
  const raw = bpInfo ? await T.run('notes (basic-pitch)', () => N.transcribe(a22)) : null;
  const noteSt = raw ? N.noteStages(raw, tgrid, lead) : [];

  // 7. struct + assemble
  const rmsForStruct = mix ? { rms: mix.rms, hop: mix.rmsHop } : { rms: [1], hop: dur };
  const struct = chroma ? A.stageStruct(chroma, rmsForStruct, grid, dur, words) : null;
  const text = words.length ? L.lyricStage(words, tgrid, lyricSrc) : null;
  if (text) text.warns.push('transformers.js returns no per-word probabilities; conf is a stream-level estimate');

  const out = [];
  out.push('%sc        0.3', `%profile   ${grid ? 'metric-tonal' : 'free-tonal'}`, '%residual  none', '');
  out.push(`@title     "${title.replace(/"/g, "'")}"`);
  out.push(`@source    ${fileName}`);
  if (o.artist) out.push(`@artist    "${o.artist.replace(/"/g, "'")}"`);
  out.push(`@offset    ${offset.toFixed(3)}`);
  out.push(`@duration  ${dur.toFixed(3)}`);
  out.push(`@sr        ${ch.sr}`);
  out.push('');
  out.push('# Encoded in the browser (experiments/browser-encoder): full mix only,');
  out.push(`# no source separation. backends: ${Object.entries(backends).map(([k, v]) => `${k}=${v}`).join(', ') || 'n/a'}`);
  out.push('@style     ""', '');
  out.push(':tuning', 'ref          A4 = 440.0Hz', 'temperament  12tet', '');
  const emit = st => { if (st && st.ok) out.push(...stageLines(st)); };
  if (g && g.st) emit(g.st);
  emit(struct);
  emit(harmony);
  if (drums && drums.gated) out.push(`# :perc.drums omitted — ${drums.why}`, '');
  emit(drums);
  for (const st of noteSt) emit(st);
  emit(leadStage);
  emit(contourStage);
  if (text) { text.warns.push(...lyricWarns); emit(text); }
  else if (lyricWarns.length) out.push(`# :text.vox omitted — ${lyricWarns.join('; ')}`, '');
  if (mix) { emit(mix.st); if (mix.st.comment) out.push(mix.st.comment); }
  const sc = out.join('\n').replace(/\n+$/, '') + '\n';

  return {
    sc, stages: T.stages, backends, downloads: downloads(), duration: dur,
    summary: {
      bpm: grid?.tempo, key: key && `${key.key} ${key.scale}`, words: words.length, lyricSrc,
      notes: raw?.length ?? null, lead: lead.length, drums: drums?.lines.length ?? 0, perc: drums?.debug, offset,
      whisper: asrInfo ? { model: asrInfo.model, device: asrInfo.device, dtype: asrInfo.dtype, threads: asrInfo.threads, files: asrInfo.files } : null,
    },
  };
}
