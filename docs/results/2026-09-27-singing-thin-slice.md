# Singing thin slice — results

Date: 2026-09-27. Pipeline:
- encoder: tsumugi backbone + `:contour.vox` (torchcrepe, weighted-argmax decoder, 20 ms) + lyrics at performed timing (faster-whisper base);
- `render --with-vocals`: DiffSinger (Azure Cobalt ONNX, explicit f0) → Seed-VC (singing model, 30 steps) with the separated lead stem as the voice reference, run on the Framework's RTX 5070 (`SOUNDCODE_SEEDVC_HOST=framepick`).

Scores come from `compare --with-vocals` on the lead vocal:
- **pitch**: median |cents| over frames voiced in both;
- **voice**: resemblyzer speaker-embedding cosine to the original singer;
- note F1, chroma, onset F1 and level difference.

## River: the pitch-curve lever

| | pitch error | voice similarity | note F1 | chroma | onset F1 |
|---|---|---|---|---|---|
| with `:contour.vox` | **19 c** | **0.93** | **0.42** | **0.98** | 0.53 |
| without (stream deleted) | 56 c | 0.89 | 0.19 | 0.91 | 0.46 |
| singing spike (notes only, 2026-09-26) | 40 c | 0.90 | — | — | — |

Acceptance is met: ≤ 30 c and ≥ 0.88. The contour carries the gain: without it, pitch error nearly triples. The ceiling (Seed-VC converting the real stem to itself) is about 10 c and 0.96.

## All five clips

| clip | pitch | voice | level Δ | note F1 | chroma | notes |
|---|---|---|---|---|---|---|
| river | 19 c | 0.93 | +0.1 dB | 0.42 | 0.98 | |
| discipline | 20 c | 0.91 | −0.3 dB | 0.35 | 0.93 | |
| lights_in_the_sky | 20 c | 0.86 | 0.0 dB | 0.29 | 0.95 | |
| 999999 | 1 c* | 0.80 | +1.1 dB | 0.12 | 0.87 | *few frames voiced in both: not meaningful |
| corona_radiata | 21 c | 0.61 | −2.7 dB | 0.11 | 0.86 | choir-like drone vocal; one singer's reference clip does not describe it |

None crashed. Each 30 s clip takes about 7 min end to end: separation and encode about 5 min, Seed-VC on the RTX 5070 about 2 min, versus about 5–6 min on the M1.

## Bugs found by real runs and fixed

1. torchcrepe's default Viterbi decoder is random (about 9 c between identical calls, even seeded). The contour and the pitch metric use `weighted_argmax`, which is deterministic and more accurate on a test tone.
2. resemblyzer silently returned nothing: webrtcvad imports `pkg_resources`, which setuptools 80+ removed. It is pinned to `setuptools<80`, and the render/sing dependencies are declared in `pyproject.toml`.
3. The sung vocal came out +3 to +20 dB loud, because Seed-VC's faint noise floor counted as singing. Level matching now measures only blocks within 30 dB of the voice's loudest block, in the renderer and in `compare`.

## Follow-ups

- **Voice.** The DiffSinger bank is a feminine hobbyist voice. A bank closer to each singer, or SoulX-Singer (M2 step 4), should lift voice similarity toward the 0.96 ceiling.
- **Lyrics** are whisper-base words: no LRCLIB correction yet (M2 step 2). Intelligibility is unscored.
- **Backing vocals** and diarization (M2 step 3).
- Seed-VC reloads its models on every call (about 1.5 of the 2 min). A persistent server on the Framework would cut that.
- The user's listening verdict on the River rebuild (`~/Downloads/river-30s-rebuild-with-vocals.wav`) is still to be recorded.
