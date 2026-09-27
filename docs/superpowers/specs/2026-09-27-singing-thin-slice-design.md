# Singing, thin slice — design (Milestone 2, step 1)

Date: 2026-09-27. Status: approved direction in conversation ("go, thin slice first"), iterating.
Roadmap: Milestone 2 ("the voice as code"). North star: a `.sc` from which the song is rebuilt **very close to the original**, singing included.
Evidence: `docs/research/2026-09-26-singing-spike.md`. There, DiffSinger (ONNX, headless) → Seed-VC (singing model, MPS) with the separated lead stem as the voice reference scored 40 cents median pitch error and 0.90 speaker-embedding cosine to the original singer. The ceiling is 10 cents / 0.96.

## Goal

`soundcode render song.sc --with-vocals` sings the lead vocal from the `.sc` in the original singer's voice, and mixes it with the instrument render. `soundcode compare` scores the sung vocal on pitch and voice likeness.

## Scope

In this slice:
1. **`:contour.vox`**: the lead vocal's f0 curve, written into the `.sc`. This is the main pitch lever: the note list alone caps pitch at about 30 cents.
2. **Lyrics as performed**: `:text.vox` word onsets with 3-decimal beats (not half-beats) and word durations, from the existing ASR.
3. **Vocal synthesis**: DiffSinger (ONNX bank) sings notes + lyrics + f0, then Seed-VC converts to the original singer's timbre. It sits behind `render --with-vocals`.
4. **Vocal metrics in `compare`**: pitch error in cents and a voice-similarity score.

Later steps, each measured by `compare`:

| Item | Step |
|---|---|
| Lyric correction against LRCLIB, forced alignment, Qwen3-ASR | Milestone 2, step 2 |
| Singer diarization (pyannote): lead / backing / harmony voices | Milestone 2, step 3 |
| SoulX-Singer as a one-model alternative (head-to-head) | Milestone 2, step 4 |
| Backing vocals sung | after diarization |

## 1. `:contour.vox`: the f0 curve

- **Source.** The encoder's lead-vocal stem (`lead_vocals`, or the unsplit `vocals` when `encoder_stems` fell back). It goes through **torchcrepe** (already installed): model `full`, hop 10 ms, periodicity threshold 0.5 for voicing, then a 30 ms median filter.
- **Resolution.** It is written at **50 Hz (20 ms)**. That keeps vibrato (4–7 Hz) and scoops, and costs about 1,500 values per 30 s of singing.
- **Format.** A stream the parser already preserves, with one statement per voiced phrase:

  ```
  :contour.vox  rate=50
  meta    src=torchcrepe:full  conf=0.83  stem=lead_vocals
  f0  @12.340  6912 6915 6920 6925 6921 …
  f0  @14.020  6710 6712 …
  ```

  - Values are absolute pitch in cents (MIDI × 100, rounded to integers), one per 20 ms from the stated start time.
  - A phrase ends at the first unvoiced gap of 60 ms or more. A shorter gap is filled by linear interpolation, and so is a lone unvoiced frame.
  - Times are seconds (`@`), because f0 is performance data, not metric data.
  - `conf` is the mean periodicity over voiced frames.
- **Gate.** The lead stem's loudness gate (Plan 1) applies: no contour for a gated-silent vocal.
- **Size check.** A 30 s River contour is about 10 KB of text. That is acceptable, and the stream is optional for any consumer that ignores it.

## 2. Lyrics as performed

- `stage_lyrics` keeps faster-whisper, but word positions use `tsumugi_sc.position` (3-decimal beats, `@seconds` before the downbeat) instead of half-beat rounding. It also writes each word's duration (`0.420b`, or `s` for `@` words), from the ASR word end.
- Words whose onsets collide after the ASR are nudged 20 ms apart, not 120 ms.
- The stream stays `:text.vox lang=<detected> align=word`.

## 3. Vocal synthesis

**New module `src/soundcode/sing.py`**, with three stages:

1. **Score.** From the `.sc`: the lead vocal note stream (`inst=voice.lead`, or `stem=lead_vocals`/`vocals`), `:text.vox` and `:contour.vox`, it produces a phoneme sequence with durations, plus a frame-level f0 curve.
   - Phonemes come from **CMUdict** (`cmudict` package), mapped to the bank's ARPAbet. The bank's own dictionary covers too few words.
   - A syllable is split at its vowel. Onset consonants take 70 ms *before* the note onset, the vowel fills the note, and codas take 70 ms. `SP` fills gaps, and `AP` (breath) takes the last 300 ms of gaps longer than 450 ms.
   - f0 is the `:contour.vox` value where present. Elsewhere it is the note pitch with 30 ms portamento. The bank's own pitch predictor is **never** used: it drifts 80 cents.
2. **DiffSinger.** An OpenUtau ONNX bank driven headless with `onnxruntime` (already installed): linguistic → variance (breathiness/voicing/tension) → acoustic (`steps=20`, `depth=0.6`) → NSF-HiFiGAN vocoder at 44.1 kHz. Bank: **Azure Cobalt** (CC BY-SA 4.0); vocoder `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02` (CC BY-NC-SA; fine for private research). Both live on the external drive under `models/diffsinger/`, and `$SOUNDCODE_DIFFSINGER` overrides the location.
3. **Seed-VC.** Converts to the original singer's timbre.
   - It lives in its own checkout and venv, `external/seed-vc` (GPL-3.0, run as a subprocess), pinned to a commit. The three patches from the spike are kept as `scripts/seedvc.patch`: the dac import, MPS float32, and the soundfile save.
   - Checkpoints go to the external drive via `HF_HUB_CACHE`.
   - Flags: `--f0-condition True --auto-f0-adjust False --diffusion-steps 30 --inference-cfg-rate 0.7`. The spike used 50 steps; 30 is tried first for speed, and 50 is used if the metrics drop.
   - `scripts/install_seedvc.sh` sets it up.

**Voice reference.** `render --with-vocals --voice-ref <wav>`. By default it uses `out/stems/<@source stem>/lead_vocals.wav`, falling back to `vocals.wav`. With neither present, it fails with a one-line error naming `--voice-ref`. Seed-VC uses the first 25 s of the reference.

**Caching.** A sung vocal is cached under `out/sing/<sha1 of the vocal streams + reference path + engine settings>.wav`. Seed-VC takes about 5 min per 30 s on this Mac, so a re-render or a `compare` reuses the file.

**Mixing.** The sung vocal becomes one more stream in `render_sf.mix`, level-matched to the vocal stream's `meta level`, just as instruments are. `render` without `--with-vocals` is unchanged.

**Errors.** A missing bank, Seed-VC or reference is a `SingError` with a one-line message and exit 2. `render --with-vocals` never silently drops the vocal.

## 4. Vocal metrics in `compare`

`compare --with-vocals` renders the lead vocal through `sing.py` (from cache when available) and adds two numbers to the `lead_vocals` row:
- `pitch_cents`: median absolute cents error over frames voiced in both, via torchcrepe on both;
- `voice_sim`: resemblyzer speaker-embedding cosine to the original lead stem. It is added to the main venv, about 20 MB.

The other metrics (level, onset, energy, chroma) apply to the sung vocal as to any stem. Note F1 stays reported.

## Evaluation and acceptance

- **River 30 s.** `render --with-vocals` produces the full rebuild, and `compare --with-vocals` shows:
  - `pitch_cents` ≤ 30, down from 40 without the contour;
  - `voice_sim` ≥ 0.88.
- **Pitch lever.** The same River render *without* `:contour.vox` (the stream deleted) scores worse on `pitch_cents`, which proves the contour carries the gain.
- **Listening checkpoint.** Play the original, then the full rebuild (instruments + sung vocal).
- The other four clips render with vocals and none crashes. Numbers are recorded, with no bar (lyrics accuracy is step 2).

## Testing

Unit tests, with no models needed:
- contour extraction on a synthetic vibrato tone (the recovered f0 follows the modulation, and phrase splitting at gaps is correct);
- `:contour.vox` round-trip through the parser;
- word positions and durations;
- G2P and phoneme timing on a two-word example;
- the f0 curve prefers contour over notes, and uses portamento between notes;
- the cache key changes when the notes, lyrics, contour or reference change;
- `--voice-ref` resolution and its error;
- the metric functions on synthetic signals.

Tests with real dependencies, skipped when they are absent:
- a DiffSinger render of 2 bars (non-silent, the right length, f0 within 50 cents of the input curve);
- a Seed-VC conversion of 3 s (runs, and the output length matches).
