# Instrument inventory and sampled rendering — design

Date: 2026-09-26. Status: approved in conversation, iterating.
Roadmap: Milestone 3 ("every instrument and every harmony as code"), step 1 of 3.
Parent spec: `2026-07-31-soundcode-design.md` §4.3.6 (track declarations, `inst=` vocabulary).

## Goal

A rebuilt song should use the **right instrument types**: bass sounds like a bass, drums like a kit, piano like a piano, and pads, strings and guitars where the song has them. It does not have to match the exact sound of the record. The encoder records every instrument type it hears, with when it plays and how sure it is.

Long-term bar this step builds toward:
- identify every word and every instrument type in a song;
- tell apart distinct voices and distinct instruments.

## Scope

Included in this step:
1. An instrument inventory in the encoder.
2. A sampled renderer.
3. The basic-pitch +33 cent fix.

Not included, each with a later step:

| Item | Later step |
|---|---|
| Distinct instruments of the same type (two guitars) | Milestone 3, step 2 |
| Matching the record's exact timbre | Milestone 3, step 3 (DawDreamer/sfizz) |
| Realistic singing (DiffSinger) | Milestone 2 |
| Distinct voices and lyrics | Milestone 2 |

## 1. Instrument inventory (encoder)

New module `src/soundcode/instruments.py`. It runs after separation.

**Vocabulary.** The §4.3.6 list:
- `drums.kit drums.machine perc.hand`
- `bass.electric bass.synth bass.upright`
- `gtr.clean gtr.crunch gtr.acoustic`
- `keys.piano keys.ep keys.organ`
- `synth.lead synth.pad strings brass winds`
- `voice.lead voice.backing voice.speech`
- `fx other unknown`

Each entry carries:
- text prompts for the classifier;
- a General MIDI program (or a drum-kit flag);
- its family (`bass`, `gtr`, `keys`, `synth`, `strings`, `brass`, `winds`, `drums`, `voice`, `other`).

The table is the single source for both the encoder and the renderer.

**Classifier interface.** `Tagger.scores(audio: np.ndarray, sr: int) -> dict[str, float]` returns a score for every vocabulary entry. Two implementations:
- `ClapTagger` (msclap; zero-shot over the prompts);
- `EssentiaTagger` (the MTG-Jamendo instrument model, with its labels mapped into the vocabulary).

The default is chosen by the bake-off (§4).

**Scanning.**
- Each stem is cut into 5 s windows with a 2.5 s hop.
- A window more than 40 dB below the stem's loudest window is skipped as silence.
- The stem name restricts the candidates:

| Stem | Candidates |
|---|---|
| `bass` | `bass.*` |
| `drums` | `drums.* perc.hand` |
| `guitar` | `gtr.*` |
| `piano` | `keys.*` |
| `lead_vocals` / `backing_vocals` | `voice.*` |
| `other` | everything except `voice.*`, `drums.*`, `bass.*` |

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

## 2. Sampled renderer

New module `src/soundcode/render_sf.py`. The existing `render.py` stays as the `mock` engine.

**CLI.**
- `soundcode render song.sc [-o out.wav] [--engine sf2|mock]`. The default is `sf2`.
- The default output is `<name>.render.wav` for `sf2`, and stays `<name>.mock.wav` for `mock`.
- The server tags `*.render.wav` as kind `render`.

**SoundFont.**
- GeneralUser GS (free license) is downloaded on first use to `models/soundfonts/GeneralUser-GS.sf2`, with a checksum.
- `$SOUNDCODE_SOUNDFONT` overrides it with any `.sf2`.
- If the file is missing while offline, the error is one line naming both options, and the exit code is 2.

**Engine.** `tinysoundfont`, rendered offline: events are sorted by time and audio is generated between events. Task 1 of the plan is a spike confirming offline rendering. If it fails, FluidSynth (pyfluidsynth) replaces it behind the same function, and nothing else changes.

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
- Pan and gain per family reuse `render.py`'s `_PAN`/`_GAIN`.
- The section gain arc reuses `render.section_gains`.
- The final render is peak-normalised as in `render.render`.

## 3. Pitch fix

`encode.py:439-441` converts basic-pitch bend bins at face value. basic-pitch reports +1 bin (+33 c) on perfectly tuned tones, so the zero point is corrected. A permanent test (`tests/test_transcription_calibration.py`) checks:
- Synthetic tones at 110, 261.63 and 440 Hz transcribe within ±10 c of true pitch.
- A tone +30 c sharp still reads sharp (between +15 and +45 c).

## 4. Evaluation and acceptance

1. **Classifier bake-off.**
   - Data: BabySlakh (20 tracks, labelled stems), in a gitignored `data/`.
   - Metric: family-level accuracy of each tagger's per-stem top label.
   - Target: ≥ 80%. The higher-scoring tagger becomes the default.
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
- vocabulary table integrity;
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
- the pitch calibration test (basic-pitch).
