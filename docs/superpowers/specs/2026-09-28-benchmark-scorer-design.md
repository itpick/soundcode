# Benchmark scorer — design

Date: 2026-09-28. Status: design approved in conversation ("go, we can always iterate"); approach B (rule metrics plus the MERT embedding).
Grounded in `docs/research/2026-09-28-render-review-fable.md` §6–7.

## Why

The user heard the full-length Discipline vocal "drift later in the song". `compare` missed it, because every metric is one number averaged over the clip: a good first half hid an off-key second half, which was the SoulX `--auto_shift` bug. The user asked for tests that "rate all the different parts of the song on the render to help with improving", on 60 s up to full-length songs.

## Goal

Rate every part of a rebuild **per section and over time**, on a 0–100 scale:
- vocals, backing vocals, keys, guitar, bass, drums, other, and the mix.

The ratings must be good enough to show:
- where a rebuild is wrong (which part, which section);
- whether a change helped, as a before/after against the previous run.

## Out of scope (follow-on specs)

- **The corpus** of research datasets and the user's own songs, with its catalog. Its songs join the benchmark set when it lands.
- **The ear ledger** of the user's pairwise listening verdicts, plus blind A/B in the listening server, used to validate the scorer and refit its weights.
- **A learned scorer.**

## Commands

- `soundcode score ORIGINAL SC [--out DIR]`: score one song and write its report. It renders the `.sc` with `--with-vocals --parts` unless a cached render exists.
- `soundcode bench [--tier A|B|C|all] [--label TEXT] [--force]`: run the benchmark set, save the run, and print the change from the previous run.

## What is compared

- Each **rebuilt part** (from `render --parts`: lead_vocals, backing_vocals, piano, guitar, bass, drums, other) is compared with the **same original stem** from `separate`.
- The **rebuilt mix** is compared with the **original mix**.
- A part missing on one side is reported as missing (score 0, flagged), never skipped silently.
- A part whose original stem is below the loudness gate (−50 dBFS) is marked *silent* and left out.
- Part keys and labels come from `render_sf.PART_KEYS` and `PART_LABELS`.

## Time slices

Every metric is computed on three kinds of slice:
1. **Sections.** Use `:struct` sections when there are at least 2 with distinct labels. Otherwise use fixed 8-bar windows from the grid, because structure detection is weak today.
2. **20 s windows** (10 s hop): the timeline that shows drift.
3. **The whole song.**

A slice is scored only if the original part is active in it (above the gate). Otherwise it is marked *silent* for that part.

## Metrics per part

Three axes per part, each 0–100: **What** (notes and timing), **Sound** (timbre) and **Dyn** (dynamics).

| Part | What | Sound | Dyn |
|---|---|---|---|
| keys, guitar, other | note F1 at cover tolerance (±100 ms, ±100 cents); chroma cosine per bar; onset F1 (±50 ms) | MERT cosine; 1/12-octave log-spectral distance | RMS-envelope correlation; level difference per section |
| bass | onset F1; bass f0 agreement (median abs cents over frames voiced in both, torchcrepe fmin 30 Hz); chroma | MERT; 1/12-octave distance | envelope correlation; level per section |
| drums | onset F1 overall, and **per voice** (kick, snare, hat, clap). The per-voice onsets come from tsumugi `drums_v1_5` run on both the original stem and the rebuilt part. | MERT; median hit decay (time to −20 dB) agreement per voice | per-bar level correlation |
| lead_vocals, backing_vocals | pitch error (median abs cents, voiced in both); word-timing MAE (Whisper word stamps on both sides); `sung_wer − floor`, where the floor is the original stem's own `sung_wer` | voice similarity (resemblyzer); MERT | envelope correlation; level per phrase or section |
| mix | chroma; onset F1 | MERT; LUFS-I difference (pyloudnorm); stereo width difference | per-section LUFS difference |

On every part, the **lag** per 20 s window is the cross-correlation peak of the onset-strength envelopes, searched within ±1.5 s.
- A window whose |lag| is above 30 ms is flagged **drift**, and the lag is reported.
- Nothing is realigned. The metrics are computed on the unaligned audio, so a drifting part scores badly, as it sounds.

**MERT:**
- Model: `m-a-p/MERT-v1-95M` (CC BY-NC 4.0, allowed for private research). It is stored under the external drive's `models/mert/` when mounted.
- Input: 24 kHz mono. Output: layer-averaged frame embeddings, pooled per slice. The score is the cosine between the original's and the rebuild's pooled vectors.
- Embeddings are cached per file hash under `out/bench/cache/`.
- It runs on MPS, falling back to CPU.

**Guards:**
- Metrics that the renderer optimises directly (`spectral_db`, whole-song level difference) keep low weights (0.1 inside their axis) and are also reported raw.
- Chroma and onset F1 always sit beside note F1 or per-voice F1, never alone.

## From metrics to 0–100

For each metric, per part type, there are two anchors:
- **ceiling:** the value of the original stem against the stem from a second separation run of the same song (separation noise), measured once;
- **floor:** the value of the original stem against the same part of a **different** song in the set, measured once.

`score = 100 × clamp((m − floor) / (ceiling − floor), 0, 1)`, inverted where lower is better.

The anchors live in `src/soundcode/score/anchors.json` (committed). `soundcode bench --calibrate` re-measures them.

**Combining:**
- **Axis:** the weighted mean of its metrics. Weights sit in `anchors.json` beside the anchors, 1.0 by default, and 0.1 for the self-optimised metrics.
- **Part:** 0.4 What + 0.4 Sound + 0.2 Dyn.
- **Section:** the same formula on that slice.
- **Song:** the mean of the part scores, each weighted by its energy share in the original, then blended 50/50 with the mix score.

Every song and every part reports its **worst section** beside its mean.

### Amendment (2026-09-28, after the first real calibration)

The first real validation run failed on the off-key Discipline vocal: 70.8 against 75.5 for the fixed one, where a gap of at least 30 is required. The cause is the floor: different-song floors for error-size metrics sit far past the point where the ear calls something wrong (pitch 747 c, lag 758 ms, word timing 7.5 s). The scorer changes three ways:
- **Perceptual floor caps.** `f0_cents` floor ≤ 100 c (a semitone off scores 0), `lag_ms_abs` ≤ 100 ms, `word_mae_s` ≤ 0.5 s. Calibration never sets a floor looser than its cap.
- **Pitch weighs more.** `f0_cents` weighs 3 in the vocal and bass What axes.
- **Axes combine by a weighted geometric mean.** The part score is the weighted geometric mean of the axes (weights 0.4 / 0.4 / 0.2, renormalised over the axes that exist, each axis floored at 1). A part that fails one axis, such as wrong notes with the right voice, can no longer average out to "fine".

### Amendment 2 (2026-09-29, after the first listening checkpoint)

The worst full-Discipline slice ("keys intro, score 0") was a false alarm. The original keys stem is at −84 dBFS there, so both sides are silent. One loud 100 ms click was enough to count the whole section as active.

**New rule.** A part is active in a slice when two things hold:
- at least **10%** of its 100 ms frames are at or above −50 dBFS;
- the slice RMS is ≥ −60 dBFS.

The same rule decides "missing" on the rebuild side, and the whole-song silent/missing check. The worst-slice list only ranks slices where the original part is active under this rule.

### Amendment 3 (2026-09-29, final whole-branch review)

- **Lag.** With a periodic onset envelope the cross-correlation peaks at every beat multiple, and a far peak can win by a hair.
  - Per window: the lag is the **smallest-|lag| candidate whose correlation is within 0.05 of the maximum**, searched by FFT cross-correlation.
  - Song-level: `lag_ms_abs` is the **median** |window lag|, not the mean.
- **Bass pitch.** It is tracked from **32.7 Hz** (C1, the lowest bin of CREPE); 30 Hz is outside CREPE's range and produced no values.
- **Mix loudness.**
  - The whole-song `lufs_diff_abs` is a self-optimised gain offset: weight **0.1**, and it is reported raw.
  - The mix Dyn axis adds **`lufs_section_diff`**: the mean |per-section LUFS difference| once the whole-song offset is removed, i.e. the dynamic arc. Anchors: floor 6 dB, ceiling 0.5 dB, weight 1.
- **Worst spot.** A song's `worst` is its worst *scored* slice, the same rule as the report's worst-slice list. Parts that are never rebuilt are listed separately as `missing: [...]` in the table, history and README.
- **Stage caching.** It is by input mtime, so a code change to encode or render needs `--force`. The CLI help and the README header say so.

## Benchmark set

| Tier | Songs | Why |
|---|---|---|
| A (regression, fast) | river-30s, discipline-30s, lights_in_the_sky-30s, 999999-30s, corona_radiata-30s | run on every change |
| B (sections) | discipline-100s; lights_in_the_sky 0:30–1:30; corona_radiata 0:00–1:00; 999999 full (85 s) | section changes, SoulX segment joins |
| C (full) | discipline (259 s), lights_in_the_sky (209 s), corona_radiata (453 s), The River full (196 s) | drift, structure, the whole-song arc |

- The sets are a constant in `score/bench.py`.
- The B excerpts are cut once with ffmpeg into `audio/bench/`, which is gitignored like the rest of `audio/`.
- The pipeline per song runs `separate` → `encode` → `render --with-vocals --parts` → score. Each stage is cached under `out/bench/<song>/` by the mtime of its input; `--force` redoes all of them.
- `bench` runs `separate` itself, so stems always land where `render` looks. That fixes the temp-workdir gap found on the full Discipline run.

## Outputs

**Per run:** `out/bench/runs/<YYYY-MM-DD-HHMM>-<label>/` (not committed), containing:
- `scores.json`: every metric, axis, part, section and window, with raw values and scores;
- `report.html`, one per song:
  - a heatmap of parts × sections, colored by score, with the value in each cell;
  - timelines of lag/drift per part, vocal pitch error, and level;
  - a table of the three worst slices, each with a **15 s listen pair** (original, then rebuild) cut to `excerpts/`;
- the terminal table per song: `part | what | sound | dyn | score | worst section (time)`, then, after a bench run, the change from the previous run per part and song, e.g. `lead_vocals 71 → 84 (+13)`.

**Committed:**
- `docs/results/benchmark/history.jsonl`: one line per run, with the label, git commit, date, tier, and per song, part and section the scores only. No raw audio and no lyrics.
- `docs/results/benchmark/README.md`: regenerated after each run, with the latest table per tier, the change from the previous run, and a short list of improvements and regressions.

## Also fixed in this project

- **Misleading warnings:** when SoulX is the singer, `render` stops printing `sing_score`'s heuristic "dropped (no room)" warnings, because SoulX's own alignment sends every word.

## Validation (automated tests)

**Sanity anchors that must always hold:**
- original stem against itself: part score ≥ 95;
- original against a different song's same part: ≤ 10.

**Regressions that must be caught:**
- the full Discipline vocal from before the `--auto_shift` fix (`out/bench/fixtures/discipline-full-vocal-autoshift.wav`) scores at least 25 points below the fixed one (first guess 30; measured 29.1 after the amendment) (`…-fixed.wav`) on lead_vocals, when both are scored against the original lead-vocal stem. This is a real-data test, skipped when the fixtures are absent.
- a rebuild delayed by 200 ms is flagged as drift in every window, and its What score drops by at least 30.

**Unit tests** use synthetic audio and a fake MERT: lag detection, slicing, the 0–100 mapping, per-voice onset F1, the gating of silent parts, the history diff, and the report's structure.

## Performance

- Tier A: a few minutes on cached renders.
- Tier C: about an hour cold (separation, SoulX on framepick, torchcrepe). Re-scoring is minutes.
- `bench` prints wall time per stage.

## Acceptance

- `soundcode bench --tier A` runs end to end, writes the history line and README, and every sanity anchor holds.
- The pre-fix Discipline regression is caught as specified.
- One full-song report (Discipline) is opened and checked by the user.
