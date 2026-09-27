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
3. **Part C — the tsumugi transcription backbone and instrument inventory** (revised: adopt an existing analyser, then add more analysers that vote).

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
- basic-pitch's `onset_threshold` goes from 0.5 to 0.6 and `frame_threshold` from 0.3 to 0.4. `minimum_note_length` stays at its default of 127.7 ms, which is already stricter than the 80 ms first proposed.
- Overlapping notes of the same pitch merge into one.
- The amplitude floor rises from 0.30 to 0.40 of the stem's peak.
- These are starting values; the plan tunes them on the five test songs by note F1 in `compare`.

**B3. Level and provenance metadata.** Each note and percussion stream gets these, both read by the renderer and by `compare`:
- `meta stem=<source stem>`;
- `meta level=<dB>`, the source stem's RMS in the song's active (non-gated) blocks.

**B4. Pitch fix.** `encode.py:439-441` converts basic-pitch bend bins at face value. basic-pitch reports +1 bin (+33 c) on perfectly tuned tones, so the zero point is corrected. A permanent test (`tests/test_transcription_calibration.py`) checks:
- Synthetic tones at 110, 261.63 and 440 Hz transcribe within ±10 c of true pitch.
- A tone +30 c sharp still reads sharp (between +15 and +45 c).

## C. Transcription backbone: tsumugi (revised 2026-09-26)

**Decision.** The user chose to adopt an existing open-source analyser rather than build the tagger ensemble first. Research (`docs/research/2026-09-26-open-song-analyzers.md`) picked **tsumugi** (anime-song/tsumugi, MIT, code and weights). On the River clip:
- the separator's "guitar" stem came out as piano 0.73 / e-piano 0.27, which matches the user's "piano and clapping";
- the drum stem produced hand-clap hits;
- each run took about 5 s per 30 s on MPS.

tsumugi replaces basic-pitch, band-split drum onsets and zero-shot tagging. It is the first analyser in the pipeline; later analysers (ADT_STR for drums, MuScriptor as a mix-level second opinion) plug in beside it and vote.

The larger taxonomy below (all 128 GM programs, drum machines, modern synth sounds) remains the vocabulary target. tsumugi's 36 classes map into it.

**Install.**
- `scripts/install_tsumugi.sh` clones tsumugi into `external/tsumugi` (gitignored), pinned to the tested commit `020edc1`, then runs `uv sync --locked --python 3.11`.
- Checkpoints download from Hugging Face on first use (about 55 MB each).
- `$SOUNDCODE_TSUMUGI` overrides the path.

It runs in its own venv as a subprocess, as ACE-Step does, because it pins its own PyTorch.

**Wrapper: new module `src/soundcode/tsumugi.py`.**
- `transcribe_stem(stem_wav, stem_name, out_dir) -> Path` runs `python -m instrument_agnostic_amt.amt.cli.infer --audio … --output-midi … --type <T> --device mps --disable-tqdm`. The model type by stem:

  | Stem | Model type |
  |---|---|
  | piano | `default` |
  | guitar | `guitar_v1_5` |
  | bass | `bass_v2` |
  | other | `other_v1_5` |
  | drums | `drums_v1_5` |
  | lead_vocals / vocals | `vocal_harmony_v1_5` |

- `refine(stem_wav, midi, stem_name, out_dir) -> dict` runs `python -m instrument_agnostic_amt.instrument_refinement.cli.infer --mode single --stem-name <stem> --output-json …` and returns class → probability.
- `transcribe_mix(mix_wav, out_dir) -> Path` runs the unrestricted `default` model on the full mix, for cross-checking.
- The velocity model (`instrument_agnostic_amt.velocity.cli.infer_velocity`) runs on the stems directory when available, because dynamics matter for a near-original rebuild. Without it, every note has velocity 100.
- If the tool is missing or any call fails, it raises `TsumugiError`. The encoder then falls back to the current basic-pitch path and writes a `meta warn`.

**Instrument inventory: several sources voting.** For each stem that passes the loudness gate:
1. **Refinement under the stem's own prior** gives a class and probability, e.g. guitar stem → `distorted_guitar 0.97`.
2. **The mix-level unrestricted run** attributes notes to classes with no stem prior. For the stem's notes, the notes that match mix notes (onset ±50 ms, pitch ±1 semitone) vote for the mix note's family.
3. **The stem name** is a prior, not a decision.

Decision:
- If at least 60% of matched mix votes name a family different from the stem's prior, the stem is **reassigned**. Refinement then re-runs with that family's prior; for River, the guitar stem becomes piano 0.73.
- The `:instruments` line records the reassignment with `meta warn="guitar stem reassigned to keys by mix-level vote 0.xx"`.
- Confidence is the agreement: the product of the refinement probability and the mix-vote share, or the refinement probability alone when fewer than 5 notes matched. Below 0.5 it is written with `?`.

**Output in `.sc`.**
- The `:instruments` stream, per stem: `<stem> bars A-B <inst> ?conf | <runner-up> p`, as in the earlier design.
- One note stream per resulting instrument: `:notes.<short> inst=<vocab>`, plus `meta stem=<stem> level=<dB> src=tsumugi:<type>@020edc1 conf=<c>`.
- Note positions use `bar:beat` with 3 decimal beats instead of the current half-beat rounding: a near-original rebuild must keep the performed timing. Durations are in beats to 3 decimals.
- Streams with fewer than 3 notes, or under 2% of their stem's notes, are dropped as bleed (`# … omitted — bleed (N notes)`).
- Drums: `drums_v1_5` GM pitches map to voice names. `gm.DRUM_NOTES` grows to cover the GM kit: `clap` 39, `stick` 37, `hat.pedal` 44, `hat.open` 46, `tom.floor` 41/43, `crash` 49/57, `ride` 51/59, `cowbell` 56, `tamb` 54, and the rest. The renderer maps them back.
- Vocals: the vocal stem is transcribed (`:notes.vox`, `meta stem=lead_vocals`) so the singing work has melody notes, but renders still leave vocals out unless `--with-vocals`.

**Taxonomy mapping.** `src/soundcode/data/tsumugi_classes.json` maps each of tsumugi's 36 classes to our `family.instrument` name and a GM program, e.g. `electric_piano` → `keys.ep` / 4, `distorted_guitar` → `gtr.electric.distortion` / 30, `melody` → `voice.lead`. `gm.target_for` resolves the fine-grained `inst=` through this table before falling back to the family.

**Unchanged:** the loudness gate (a silent stem skips tsumugi entirely), `:grid`, `:struct`, `:harmony`, `:text`.

## Evaluation and acceptance

0. **Baseline, then improvement, measured by `compare`.**
   - Run `compare` on the five test songs (30 s clips) and the full *River*, before and after Part B.
   - Record both tables in the README.
   - Required after Part B:
     - every stem's level is within ±3 dB of the original;
     - a stem more than 40 dB below the mix produces no notes (the *River* bass for 0–17 s, and "other");
     - mean note F1 over the pitched stems improves on the baseline for every song.
1. **tsumugi vs the Plan 1 encoder** on the five clips (`compare`, `after2` baseline):
   - mean note F1 over active stems improves on every song;
   - River's inventory is keys (piano / e-piano) plus drums with claps, with no guitar stream;
   - the listening checkpoint: "sounds like the song's instruments".
2. **Classifier bake-off** (deferred; superseded by item 1 unless tsumugi's labels fail on the test songs).
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
