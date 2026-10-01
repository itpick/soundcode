// Polyphonic notes from the full mix with Spotify's basic-pitch (TF.js).

import { position, duration, centsToName, clamp } from './sc.js';

const TFJS = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.19.0/+esm';           // the version basic-pitch's bundle imports
const BASIC_PITCH = 'https://cdn.jsdelivr.net/npm/@spotify/basic-pitch@1.0.1';

// Native thresholds (encode.py) so the two encoders are comparable.
const ONSET_THRESHOLD = 0.6, FRAME_THRESHOLD = 0.4, AMP_FLOOR = 0.40;
const BEND_ZERO = 1.0;          // basic-pitch reads an in-tune note as +1 bin
const BASS_SPLIT = 48;          // MIDI below C3 -> :notes.bass

let _bp = null, _tf = null;

export async function loadBasicPitch(prefer = 'auto') {
  if (_bp) return { backend: _tf.getBackend() };
  _tf = await import(TFJS);
  // @tensorflow/tfjs-backend-wasm@3.19.0 was tried and fails inside
  // basic-pitch's graph ("Unknown dtype undefined"), so without WebGL the
  // fallback is TF.js's plain-JS CPU backend.
  const order = prefer === 'wasm' ? ['cpu'] : ['webgl', 'cpu'];
  for (const b of order) {
    try {
      if (await _tf.setBackend(b)) { await _tf.ready(); break; }
    } catch (e) { console.warn(`tfjs backend ${b} unavailable: ${e}`); }
  }
  const mod = await import(`${BASIC_PITCH}/+esm`);
  _bp = { mod, model: new mod.BasicPitch(`${BASIC_PITCH}/model/model.json`) };
  return { backend: _tf.getBackend() };
}

export async function transcribe(audio22k, onProgress) {
  const { mod, model } = _bp;
  const frames = [], onsets = [], contours = [];
  await model.evaluateModel(audio22k, (f, o, c) => { frames.push(...f); onsets.push(...o); contours.push(...c); }, p => onProgress && onProgress(p));
  const notes = mod.noteFramesToTime(mod.addPitchBendsToNoteEvents(contours,
    mod.outputToNotesPoly(frames, onsets, ONSET_THRESHOLD, FRAME_THRESHOLD, 5)));
  return notes.map(n => ({ start: n.startTimeSeconds, end: n.startTimeSeconds + n.durationSeconds, midi: n.pitchMidi, amp: n.amplitude, bends: n.pitchBends || [] }));
}

function mergeSamePitch(events) {
  const out = []; const last = new Map();
  for (const e of [...events].sort((a, b) => a.start - b.start)) {
    const j = last.get(e.midi);
    if (j != null && e.start <= out[j].end + 0.03) { out[j].end = Math.max(out[j].end, e.end); out[j].amp = Math.max(out[j].amp, e.amp); continue; }
    last.set(e.midi, out.length); out.push({ ...e });
  }
  return out;
}

function bendCents(b) {
  if (!b || !b.length) return 0;
  return (b.reduce((s, x) => s + x, 0) / b.length - BEND_ZERO) * 100 / 3;
}

// Drop notes the lead-vocal stream already carries (same pitch within a
// semitone, overlapping in time) so the voice is not doubled on piano.
function notInLead(n, lead) {
  return !lead.some(l => Math.abs(l.midi - n.midi) <= 1 && n.start < l.end && n.end > l.start
    && (Math.min(n.end, l.end) - Math.max(n.start, l.start)) > 0.5 * (n.end - n.start));
}

// Bass and upper notes get their own amplitude floor: on a full mix a quiet
// bass line sits far below the loudest chord, and one global floor (what the
// native encoder uses per separated stem) erased it.
export function noteStages(raw, grid, lead = []) {
  const events = mergeSamePitch(raw);
  const mk = (name, inst, all, stem) => {
    const peak = Math.max(...all.map(e => e.amp), 1e-9);
    const list = all.filter(e => e.amp >= AMP_FLOOR * peak && e.end - e.start >= 0.05 && notInLead(e, lead));
    const st = { name: `notes.${name}`, fields: { inst }, src: 'basic-pitch-js@1.0.1', stem, warns: [], lines: [] };
    for (const e of list) {
      const pos = position(e.start, grid);
      const cents = e.midi * 100 + bendCents(e.bends);
      const rel = clamp(e.amp / peak, 0, 1);
      const vel = Math.round(clamp(20 + 107 * Math.sqrt(rel), 1, 127));
      const mark = rel >= 0.8 ? '' : ` ?${rel.toFixed(2)}`;
      st.lines.push(`${pos}  ${centsToName(Math.round(cents))}  ${duration(pos, e.end - e.start, grid)} ${vel}${mark}`);
    }
    st.conf = list.length ? list.reduce((s, e) => s + Math.min(e.amp / peak, 1), 0) / list.length : 0;
    st.ok = st.lines.length > 0;
    st.warns.push('full mix, no source separation: every instrument (and the voice) lands in these notes');
    st.warns.push(`${list.length} of ${all.length} notes kept (amplitude floor ${AMP_FLOOR.toFixed(2)} of this split's peak, same-pitch merged, lead-vocal duplicates removed)`);
    return st;
  };
  const sBass = mk('bass', 'bass.electric', events.filter(e => e.midi < BASS_SPLIT), null);
  sBass.warns.push(`split by pitch: MIDI < ${BASS_SPLIT}`);
  // No `meta stem`: nothing was separated, so the renderer/scorer files it by
  // its inst family (keys -> piano part). Tagging it stem=other instead was
  // tried: song scores moved -2..+1 on the four clips, i.e. a wash -- without
  // separation any part assignment is a guess.
  const sKeys = mk('keys', 'keys.piano', events.filter(e => e.midi >= BASS_SPLIT), null);
  return [sBass, sKeys];
}
