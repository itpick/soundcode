# compare results after Plan 1 encoder fixes

Date: 2026-09-26. Baseline: `2026-09-26-compare-baseline.md` (same clips, same command).

Encoder changes measured here:
- +33 c pitch fix;
- loudness gate on notes and drums (−50 dBFS absolute, 35 dB under the mix, 2 s blocks);
- stricter basic-pitch thresholds (onset 0.6, frame 0.4, amplitude floor 0.40) and same-pitch merging;
- `meta stem=`/`level=` with level-matched rendering;
- the `:grid` now states the single tempo that notes are positioned with (a contradictory per-window tempo curve used to drift renders by up to 2.4 s);
- no pyin fallback on gated stems.

## Summary

Mean note F1 over the stems that are really present in the original (original stem louder than about −60 dBFS). This excludes metric noise from silent stems, where basic-pitch transcribes bleed on both sides.

| Song | Baseline | After | |
|---|---|---|---|
| 999999 | 0.17 | 0.12 | worse (ambient, sparse; stricter filter drops soft real notes) |
| corona_radiata | 0.07 | 0.09 | better |
| discipline | 0.05 | 0.12 | better |
| lights_in_the_sky | 0.09 | 0.18 | better |
| river | 0.03 | 0.14 | better |

River piano in detail:

| Metric | Baseline | After |
|---|---|---|
| Note F1 | 0.00 | 0.29 |
| Chroma | 0.55 | 0.94 |
| Onset F1 | 0.07 | 0.55 |
| Energy r | −0.07 | 0.74 |

Drums onset F1 is 0.97–1.00 where drums exist.

## Acceptance (spec Evaluation, item 0)

- **Levels within ±3 dB:** met for most stems. Four stems are 3.1–5.6 dB quiet (999999 lead, corona guitar and piano, discipline piano): rendered notes are shorter than the original's sustained sound, so their average level is lower.
- **No notes on stems more than 40 dB under the mix:** met. The River bass and `other`, and all silent drums/guitar/piano stems, are now omitted with `# … omitted — stem silent`.
- **Mean note F1 above baseline on every song:** met on 4 of 5. 999999 did not improve. One tuning try (amplitude floor back to 0.30) gave no gain: 999999 other 0.13 → 0.14, bass 0.23 → 0.22.

## What still diverges (for Plan 2)

- **Identity, not timing or level.** The user reports that River is piano and clapping. The render still has a guitar-like part (the `guitar` stem, which PANNs confidently tags "Guitar 0.59"), and the claps become kick/snare/hat. We need confident instrument identification from several agreeing tools, plus drum-voice classes that include claps.
- **Transcription quality.** Note F1 of 0.1–0.3 on real parts is the ceiling of basic-pitch on separated stems. An existing multi-instrument transcriber with instrument labels is being researched (`docs/research/2026-09-26-open-song-analyzers.md`).
- **Vocals.** Renders now leave vocal streams out by default (`--with-vocals` includes them); the singing voice is Milestone 2.

## Raw tables (after)

### 999999-30s

```
stem            orig dB rend dB   Δ dB  noteF1  anyOct  chroma onsetF1  energy
lead_vocals       -46.5   -49.8   -3.3    0.00    0.00    0.84    0.00    0.73
backing_vocals    -72.8       —      —    0.00    0.00       —       —       —
drums             -69.1       —      —       —       —       —    0.00       —
bass              -30.9   -33.9   -3.0    0.23    0.26    0.92    0.18    0.48
guitar            -74.3       —      —    0.00    0.00       —       —       —
piano             -71.5       —      —    0.00    0.00       —       —       —
other             -33.6   -34.5   -0.9    0.13    0.13    0.83    0.22    0.19
```
### corona_radiata-30s

```
stem            orig dB rend dB   Δ dB  noteF1  anyOct  chroma onsetF1  energy
lead_vocals       -34.0   -35.0   -1.1    0.15    0.15    0.79    0.23    0.42
backing_vocals        —       —      —       —       —       —       —       —
drums             -86.1       —      —       —       —       —       —       —
bass              -21.1   -21.2   -0.0    0.08    0.17    0.89    0.14    0.50
guitar            -30.4   -36.0   -5.6    0.10    0.13    0.89    0.14    0.07
piano             -56.9   -60.1   -3.2    0.00    0.00    0.98    0.00    0.49
other             -24.9   -25.5   -0.7    0.13    0.15    0.92    0.17    0.18
```
### discipline-30s

```
stem            orig dB rend dB   Δ dB  noteF1  anyOct  chroma onsetF1  energy
lead_vocals       -21.9   -23.8   -1.9    0.20    0.20    0.89    0.26    0.65
backing_vocals    -35.7       —      —    0.00    0.00       —    0.00       —
drums             -14.5   -17.2   -2.7       —       —    0.98    1.00   -0.12
bass              -16.4   -18.4   -2.0    0.15    0.23    0.92    0.37    0.47
guitar            -20.5   -21.7   -1.3    0.23    0.29    0.95    0.33    0.30
piano             -31.4   -34.9   -3.4    0.03    0.03    0.82    0.11    0.77
other             -74.9       —      —    0.00    0.00       —       —       —
```
### lights_in_the_sky-30s

```
stem            orig dB rend dB   Δ dB  noteF1  anyOct  chroma onsetF1  energy
lead_vocals       -39.4   -42.0   -2.5    0.34    0.34    0.91    0.49    0.70
backing_vocals    -71.1       —      —    0.00    0.00       —       —       —
drums             -87.6       —      —       —       —       —       —       —
bass              -85.0       —      —    0.00    0.00       —       —       —
guitar            -74.0       —      —    0.00    0.00       —       —       —
piano             -25.7   -26.8   -1.0    0.10    0.10    0.89    0.71    0.59
other             -33.8   -32.5   +1.3    0.11    0.11    0.83    0.60    0.51
```
### river-30s

```
stem            orig dB rend dB   Δ dB  noteF1  anyOct  chroma onsetF1  energy
lead_vocals       -31.5   -32.3   -0.8    0.30    0.32    0.83    0.26    0.37
backing_vocals    -43.6       —      —    0.00    0.00       —    0.00       —
drums             -32.0   -34.7   -2.7       —       —    0.96    0.97    0.19
bass              -58.9       —      —    0.00    0.00       —    0.00       —
guitar            -41.4   -41.5   -0.1    0.13    0.16    0.88    0.34    0.31
piano             -36.3   -36.6   -0.3    0.29    0.37    0.94    0.55    0.74
other             -83.6       —      —    0.00    0.00       —       —       —
```
