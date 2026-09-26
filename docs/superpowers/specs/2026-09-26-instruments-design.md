# Compare, sampled rendering and instrument inventory — design

Date: 2026-09-26. Status: approved in conversation, iterating. Revised the same day after a comparison spike (§0).
Roadmap: Milestone 3 ("every instrument and every harmony as code"), step 1 of 3.
Parent spec: `2026-07-31-soundcode-design.md` §4.3.6 (track declarations, `inst=` vocabulary).

## Goal

A rebuilt song should use the **right instrument types**, across every common instrument and the most commonly used synth sounds: bass sounds like a bass, drums like a kit (or the right drum machine), piano like a piano, and pads, strings, guitars, brass and synth leads where the song has them. It does not have to match the exact sound of the record. The encoder records every instrument type it hears, with when it plays and how sure it is.

Long-term bar this step builds toward:
- identify every word and every instrument type in a song;
- tell apart distinct voices and distinct instruments.

## 0. Findings that shaped this revision

A throwaway spike rendered a 30 s clip of *The River* through a GM SoundFont (instruments picked by stem name, pitch corrected by hand). The user judged it an improvement over the mock, with piano partly working and other parts poor. Per-stem comparison against the original's stems showed that the main problem is **invented notes**, not instrument choice:

| Stem | Level, original → render | Note F1 | Onset F1 | Finding |
|---|---|---|---|---|
| bass | −59.9 → −26.7 dB | 0.00 | 0.24 | The bass stem is near-silent for the first 17 s; the encoder transcribed bleed |
| piano | −36.1 → −22.0 dB | 0.05 | 0.12 | The original has clean chords every ~2 s; the render is a smear of extra overlapping notes |
| guitar | −41.5 → −34.4 dB | 0.07 | 0.29 | A sustained texture was chopped into short fragments |
| lead_vocals | −31.5 → −82.3 dB | 0.03 | 0.23 | The melody is mostly missing, and the re-separated "voice oohs" were not seen as vocals |
| other | −83.0 → −39.5 dB | 0.08 | 0.24 | A silent stem received transcribed bleed |
| drums | −32.1 → −32.0 dB | — | 0.80 | Timing is right; only the sound is generic |

What follows from these findings:
1. Build a measuring tool first.
2. Gate transcription by absolute loudness.
3. Keep fewer, better notes.
4. Match each stream's level to its source stem.
5. Compare rendered streams directly, rather than re-separating the render.

The spike also confirmed that `tinysoundfont` renders offline (A4 piano → 441 Hz), so no FluidSynth fallback is needed.

## Scope

This step is built in this order:
1. **Part A — `soundcode compare` and the sampled renderer.** These form the measuring baseline.
2. **Part B — encoder fixes** found by `compare`: a loudness gate, stricter note filtering, level metadata, and the +33 cent pitch fix.
3. **Part C — the instrument inventory** in the encoder.

Not included, each with a later step:

| Item | Later step |
|---|---|
| Distinct instruments of the same type (two guitars) | Milestone 3, step 2 |
| Matching the record's exact timbre | Milestone 3, step 3 (DawDreamer/sfizz) |
| Realistic singing (DiffSinger) | Milestone 2 |
| Distinct voices and lyrics | Milestone 2 |

## A1. `soundcode compare`

New module `src/soundcode/compare.py`.

**CLI.**
- `soundcode compare <original-audio> <song.sc> [--engine sf2|mock] [-o DIR]`, where `DIR` defaults to `out/compare/<name>/`.
- A plain rendered or generated `.wav` may be passed instead of the `.sc`, e.g. a future ACE-Step cover. That path separates it and says in the report that stems came from re-separation.

**Pairing.**
- Original stems come from `out/stems/<name>/` when its `manifest.json` source matches. Otherwise `separate()` runs.
- Each `.sc` stream is rendered alone via `render_streams` and paired with its source stem:
  - When present, the stream's `meta stem=<name>` gives the source stem (Part B writes it).
  - Otherwise the stream name gives it: `notes.bass` → `bass`, `notes.vox` → `lead_vocals`, `perc.drums` → `drums`, and `notes.<x>` → `<x>` when `<x>` is a stem name.
  - Remaining streams pair with `other`.
- Streams from the same stem are summed before comparing.

**Metrics per stem:**

| Metric | How |
|---|---|
| Level | RMS dB of the original, the render, and the difference |
| Notes | basic-pitch on both, then `mir_eval.transcription` F1: onset ±50 ms, pitch ±50 c, offsets ignored. Also an octave-agnostic F1. Drums are skipped |
| Harmony | Chroma (CQT) cosine similarity per bar (bars from the `:grid`, else 2 s blocks), plus the mean |
| Rhythm | Onset F1 (`mir_eval.onset`, ±70 ms) |
| Energy | Correlation of the RMS envelopes |
| Sound | CLAP embedding cosine, when msclap is installed; otherwise omitted with a note |

Whole-mix metrics:
- the same chroma, onset, energy and CLAP scores on the full mixes;
- the tempo of each, from the beat tracker.

**Outputs in `DIR`:**
- `report.json`: every number above. Silent stems give `null` rather than NaN.
- `report.html`: a score table per stem, audio players (original stem and rendered stream), and the images below.
- `<stem>.png` for each stem: original and render spectrograms and chromagrams, aligned in time. This is the image Claude reads when asked why something sounds wrong.
- `bars.png`: a heatmap of chroma similarity with stems as rows and bars as columns, showing where the render diverges.

**Server.** `soundcode serve` serves `out/compare/**` (html, png, json) under `/compare/` and links each report from the track list.

**CLI summary.** A per-stem table, one line per stem, as in §0, plus the report path. The exit code is 0.

## A2. Sampled renderer

New module `src/soundcode/render_sf.py`. The existing `render.py` stays as the `mock` engine.

**CLI.**
- `soundcode render song.sc [-o out.wav] [--engine sf2|mock]`. The default is `sf2`.
- The default output is `<name>.render.wav` for `sf2`, and stays `<name>.mock.wav` for `mock`.
- The server tags `*.render.wav` as kind `render`.

**SoundFont.**
- GeneralUser GS (free license) is downloaded on first use to `models/soundfonts/GeneralUser-GS.sf2`, with a checksum.
- `$SOUNDCODE_SOUNDFONT` overrides it with any `.sf2`.
- If the file is missing while offline, the error is one line naming both options, and the exit code is 2.

**Engine.** `tinysoundfont`, rendered offline: events are sorted by time and audio is generated between events. Offline rendering was confirmed by the spike (§0).

**Per-stream output.** `render_streams(doc) -> dict[str, np.ndarray]` renders each stream separately, before mixing. `render(doc)` mixes those. `compare` uses the per-stream renders.

**Mapping.**
- Each note stream gets its own channel. More than 15 pitched streams share channels by program.
- The program comes from `inst=`. With no `inst=`, it comes from the stream-name family: `notes.bass` → bass, `notes.vox` → voice oohs, `notes.other` → piano. `unknown` → Acoustic Grand Piano.
- Drums go to channel 10 with this voice map:

| Voice | GM kit note |
|---|---|
| kick | 36 |
| snare | 38 |
| hat | 42 |
| hat.open | 46 |
| crash | 49 |
| ride | 51 |
| tom.hi / tom.mid / tom.lo | 50 / 47 / 43 |
| anything else | 39 (clap) |

**Notes.**
- Onset, duration and velocity come from `expand()` (`Note.start`, `Note.dur`, `Note.vel`).
- Pitch comes from `Note.cents`:
  - On polyphonic streams it is rounded to the nearest semitone.
  - On monophonic streams (`voice.*`, `synth.lead`, or no overlapping notes), the cents offset plus the `:contour` vibrato/scoop/fall are sent as channel pitch bend, with the bend range set to ±2 semitones.

**Mix.**
- Level: when a stream has `meta level=<dB>` (Part B), its render is scaled so its RMS matches that level. Otherwise the gain per family from `render.py`'s `_GAIN` is used.
- Pan per family reuses `render.py`'s `_PAN`.
- The section gain arc reuses `render.section_gains`.
- The final render is peak-normalised as in `render.render`.

## B. Encoder fixes found by `compare`

**B1. Absolute loudness gate.**
- Before transcribing a stem, compute its RMS per 2 s block.
- A block is gated (produces no notes) when its RMS is below −50 dBFS, or more than 35 dB below the full mix's RMS in the same block.
- A stem whose blocks are all gated gets `# :notes.<x> omitted — stem silent` instead of invented notes.
- Both thresholds are named constants, tuned with `compare`.

**B2. Stricter note filtering** in `stage_notes_poly`:
- basic-pitch's `onset_threshold` goes from 0.5 to 0.6, `frame_threshold` from 0.3 to 0.4, and `minimum_note_length` from 58 ms to 80 ms.
- Overlapping notes of the same pitch merge into one.
- The amplitude floor rises from 0.30 to 0.40 of the stem's peak.
- These are starting values; the plan tunes them on the five test songs by note F1 in `compare`.

**B3. Level and provenance metadata.** Each note and percussion stream gets these, both read by the renderer and by `compare`:
- `meta stem=<source stem>`;
- `meta level=<dB>`, the source stem's RMS in the song's active (non-gated) blocks.

**B4. Pitch fix.** `encode.py:439-441` converts basic-pitch bend bins at face value. basic-pitch reports +1 bin (+33 c) on perfectly tuned tones, so the zero point is corrected. A permanent test (`tests/test_transcription_calibration.py`) checks:
- Synthetic tones at 110, 261.63 and 440 Hz transcribe within ±10 c of true pitch.
- A tone +30 c sharp still reads sharp (between +15 and +45 c).

## C. Instrument inventory (encoder)

New module `src/soundcode/instruments.py`. It runs after separation.

**Vocabulary: every common instrument and the common synth sounds.** The coarse §4.3.6 list becomes the *family* level of a two-level taxonomy, `family.instrument[.variant]`, which the `inst=` dotted refinement already allows. The table lives in `src/soundcode/data/instruments.toml`. It is the single source for the encoder, the renderer and `compare`. It covers:

- **All 128 General MIDI melodic programs**, grouped into their 16 GM families:
  - piano, chromatic percussion, organ, guitar, bass, strings, ensemble, brass, reed, pipe;
  - synth lead, synth pad, synth effects, ethnic, percussive, sound effects.

  Examples: `keys.piano.grand`, `keys.ep.rhodes`, `keys.clav`, `mallet.vibraphone`, `organ.drawbar`, `gtr.acoustic.nylon`, `gtr.electric.clean`, `gtr.electric.overdrive`, `gtr.electric.distortion`, `bass.electric.finger`, `bass.fretless`, `bass.upright`, `strings.violin`, `strings.cello`, `strings.ensemble`, `strings.pizzicato`, `strings.harp`, `brass.trumpet`, `brass.section`, `winds.sax.alto`, `winds.flute`, `winds.clarinet`, `ethnic.sitar`, `voice.choir`.

- **Drum kits:**
  - acoustic: `drums.kit.standard`, `.room`, `.power`, `.jazz`, `.brush`, `.orchestral`;
  - machines: `drums.machine.808`, `.909`, `.707`, `.linndrum`, `.electronic`;
  - hand percussion: `perc.hand` (congas, bongos, shaker, tambourine).

- **Common modern synth sounds**, beyond GM's synth programs:

  | Group | Sounds |
  |---|---|
  | Bass | `synth.bass.sub`, `.808`, `.reese`, `.acid` (303), `.fm`, `.wobble`, `.pluck` |
  | Lead | `synth.lead.supersaw`, `.saw`, `.square`, `.sine`, `.pluck`, `.arp` |
  | Pad | `synth.pad.warm`, `.strings`, `.choir`, `.ambient`, `.sweep` |
  | Keys | `synth.keys.fm` (DX7-style), `synth.stab.chord`, `synth.stab.brass`, `synth.stab.hoover` |
  | FX | `fx.riser`, `fx.downlifter`, `fx.impact`, `fx.noise` |

- **Voices:** `voice.lead`, `voice.backing`, `voice.choir`, `voice.speech`, `voice.rap`.
- Plus `other` and `unknown`.

Each entry carries:
- its family;
- two or three text prompts for zero-shot classifiers;
- a render target.

**Render targets.**
- Every entry maps to a GM program, or to a GM drum kit (bank 128 presets: Standard, Room, Power, Electronic, TR-808, Jazz, Brush, Orchestra), so everything renders today.
- Modern synth sounds use the nearest GM program for now, e.g. `synth.lead.supersaw` → Lead 2 (sawtooth), `synth.bass.808` → Synth Bass 1 with a long release, `synth.pad.warm` → Pad 2 (warm).
- An optional `patch=` slot per entry is reserved for Milestone 3 step 3, where Surge XT / Vital patches via DawDreamer give those sounds their real character.

**Two-level classification.**
- The family is chosen first, e.g. `synth.lead` vs `gtr`.
- Then the instrument within the family is chosen.
- When the fine label's margin is too small, the `.sc` carries the family only (e.g. `inst=synth.lead`) with the fine scores in `tags{}`.
- A confident family plus an uncertain variant is still a useful, honest answer.

**Classifier interface.** `Tagger.scores(audio: np.ndarray, sr: int) -> dict[str, float]` returns a score for every vocabulary entry. Three implementations:
- `ClapTagger` (msclap; zero-shot over the prompts);
- `MuLanTagger` (MuQ-MuLan; zero-shot, CC BY-NC weights);
- `EssentiaTagger` (the MTG-Jamendo instrument model, with its labels mapped into the vocabulary).

Audio LLMs (MOSS-Music, Qwen3-Omni) are not taggers here. They score about 31% on NSynth instruments. They may later cross-check the inventory as a second opinion.

The default is chosen by the bake-off (Evaluation, item 1).

**Scanning.**
- Each stem is cut into 5 s windows with a 2.5 s hop.
- A window more than 40 dB below the stem's loudest window is skipped as silence.
- The stem name restricts the candidates:

| Stem | Candidates |
|---|---|
| `bass` | `bass.*`, `synth.bass.*` |
| `drums` | `drums.* perc.*` |
| `guitar` | `gtr.*`, plus `ethnic.*` plucked strings |
| `piano` | `keys.*`, `mallet.*`, `organ.*`, `synth.keys.*` |
| `lead_vocals` / `backing_vocals` | `voice.*` |
| `other` | everything except `voice.*`, `drums.*`, `perc.*`, `bass.*`, `synth.bass.*` |

**Spans.**
- Per window, the top label wins when its margin over the runner-up is at least 0.10. Otherwise the window is `unknown`, and its top scores are kept as evidence.
- Consecutive windows with the same label merge into a span.
- Span edges snap to the nearest bar line in the `:grid`.
- Span confidence is the mean top score.
- A second label is recorded alongside the main one when its mean score in the span is at least 0.25. This is the "also heard" evidence for instruments playing at the same time.

**Output in `.sc`:**

1. A new `:instruments` stream, which is the inventory. It has one line per (stem, span):

   ```
   :instruments
   meta    src=clap  conf=0.71
   other   bars 1-16   synth.pad ?0.72   | strings 0.31
   other   bars 17-32  strings ?0.66
   bass    bars 1-40   bass.electric ?0.91
   ```

   Confidence ≥ 0.80 carries no `?` marker, following the existing encoder convention.

2. Note streams split by instrument:
   - Notes from a stem go to a stream named after the instrument of the span they start in, e.g. `:notes.pad inst=synth.pad`, `:notes.strings inst=strings`.
   - Notes in an `unknown` span stay in `:notes.<stem> inst=unknown`, with a `tags{}` line holding the top scores.
   - Name clashes (two stems both labelled `synth.pad`) get the stem appended, e.g. `:notes.pad.other`.
   - Drums keep `:perc.drums` and gain `inst=drums.kit` or `drums.machine`.

3. The parser already preserves unknown streams. `check` lists `:instruments` like any other stream.

**Failure.** If no tagger can load, the inventory stage is omitted with a `# :instruments omitted — reason` line. Note streams then fall back to today's stem-named streams with `inst=` seeded from the stem name (e.g. `bass` → `bass.electric`). The fail-soft rule is from spec §5.

## Evaluation and acceptance

0. **Baseline, then improvement, measured by `compare`.**
   - Run `compare` on the five test songs (30 s clips) and the full *River*, before and after Part B.
   - Record both tables in the README.
   - Required after Part B:
     - every stem's level is within ±3 dB of the original;
     - a stem more than 40 dB below the mix produces no notes (the *River* bass for 0–17 s, and "other");
     - mean note F1 over the pitched stems improves on the baseline for every song.
1. **Classifier bake-off.**
   - Data: BabySlakh (20 tracks, labelled stems), in a gitignored `data/`.
   - Metric: family-level and fine-level accuracy of each tagger's (CLAP, MuQ-MuLan, Essentia) per-stem top label. Slakh labels are GM programs, so both levels are scored directly.
   - Target: ≥ 80% at family level. Fine-level accuracy is recorded, with no target yet. The higher-scoring tagger becomes the default.
   - If neither meets the target, the better one still ships and the scores are recorded in the README.
   - Script: `scripts/tagger_bakeoff.py`.
2. **Re-encode the five test songs.** Each `.sc` has:
   - an `:instruments` stream;
   - note streams with `inst=`;
   - notes sitting exactly at +33 c dropping from about 60% to under 10%.
3. **Render.** Each song renders with `--engine sf2` and appears in the server as `render`.
4. **Listening.** The user A/Bs the original against the render. The bar is right instrument types.

## Testing

Unit tests, with no models or SoundFont needed:
- `compare` stream→stem pairing (meta and name fallback);
- metric functions on synthetic signals (identical input gives F1 1.0 and cosine 1.0; silence gives null);
- `report.json` has no NaN;
- the loudness gate on a synthetic stem that is silent then loud;
- same-pitch note merging;
- the `meta stem=`/`level=` emission, and level matching in the renderer;
- taxonomy integrity: all 128 GM programs present, every entry has a family, prompts and a render target, and names are unique;
- stem→candidate restriction;
- window→span merging and bar snapping;
- margin→`unknown`;
- "also heard" threshold;
- `:instruments` emission;
- note-stream splitting and name clashes;
- inst→program mapping and stream-name fallback;
- drum voice map;
- note→MIDI event conversion (rounding vs. pitch bend);
- CLI engine selection and missing-SoundFont error.

Tests with real dependencies, skipped when they are absent:
- the tagger smoke test (msclap/essentia);
- a SoundFont render of a C-major scale (non-silent, correct length, pitch of each note within ±20 c by autocorrelation);
- the pitch calibration test (basic-pitch);
- a `compare` smoke run on a synthetic song whose `.sc` renders back exactly, with every stem scoring note F1 ≥ 0.9.
