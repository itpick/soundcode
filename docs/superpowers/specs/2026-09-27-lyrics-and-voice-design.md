# Correct words, clearer singing — design (Milestone 2, steps 2 + 4)

Date: 2026-09-27. Status: approved in conversation ("go, can always iterate").
User feedback on the sung River rebuild: "the singing i don't think has correct words and it doesn't sound great."

The evidence points to two separate problems:
- **The words are wrong in the `.sc`.** whisper-base misheard three lines of the chorus (for example "whiter" as "wider"). whisper-small on the same stem gets these right.
- **The singing loses words.** ASR on our sung vocal hears only fragments of the lyric. The causes are DiffSinger's English, my fixed 70 ms consonant timing, and the Seed-VC conversion.

## Goal

The `.sc` carries the **correct words**, timed to the performance, with their source recorded. The sung rebuild is **intelligible** and sounds better, judged by two new scores and by ear.

## 1. Correct words

- **Better ASR.** `stage_lyrics` uses faster-whisper `large-v3-turbo` (CPU int8 on the Mac) by default; `$SOUNDCODE_ASR_MODEL` overrides it. Model files download into `/Volumes/ExFAT 2/infinity-engine/models/whisper/` when the drive is mounted, and into the default cache otherwise.
- **Published lyrics.** A new `lyrics.py` looks up **LRCLIB**, which is free, needs no key and has synced lyrics.
  - The lookup uses the song's title and artist:
    - `encode --title --artist` (a new `--artist` flag);
    - otherwise parsed from the file name (`Artist - Title`, `Title-Artist`, …).
  - It queries `/api/get` with the duration and falls back to `/api/search`, picking the result whose duration is closest.
  - It sends a descriptive User-Agent, and caches responses under `out/lyrics/<artist>-<title>.json`.
  - Synced lines give a time prior. Only lines inside the clip (`@offset` … `@offset + @duration`) are used.
- **Reconcile** ASR words against the reference words: align the normalised tokens with `difflib.SequenceMatcher`.
  - Matched runs take the reference word with the ASR timing.
  - For substitutions, the reference word wins unless the ASR probability is ≥ 0.9 *and* the reference line's synced time is more than 2 s away. When the two disagree, the event carries `alt="<other>"`.
  - A reference word the ASR missed gets a time interpolated between its neighbours, bounded by its LRC line time, and is marked `?0.50`.
  - An ASR word with no reference counterpart (an ad-lib) is kept with `?conf`.
  - `:text.vox` gets `meta src=lrclib+<asr-model>`.
  - Without a reference (no lookup, or offline) it behaves exactly as today with the better ASR, plus `meta warn="no reference lyrics"`.
- **Metric: `lyric_wer`**, the word error rate of the `.sc` lyrics against the reference words inside the clip. It is reported by `compare` when a reference is cached.

## 2. Clearer, better singing

**Intelligibility metric: `sung_wer`**, reported by `compare --with-vocals`. whisper-small transcribes the sung vocal, and its words are scored against the `.sc` lyrics.

Two candidates are compared head-to-head on River, and on discipline where lyrics exist:
1. **The improved chain.** DiffSinger with the bank's own duration model (`dsdur`), which predicts phoneme durations within each word's span instead of my fixed 70 ms consonants, then Seed-VC at 50 diffusion steps on the Framework's GPU.
2. **SoulX-Singer** (Apache-2.0), a zero-shot SVS model that takes lyrics, a melody (MIDI notes, or an f0 curve from `:contour.vox`) and the singer's reference clip, and outputs singing in that voice. One model, no conversion step. It runs on the Framework (RTX 5070, 8 GB) in its own env. `scripts/install_soulx_remote.sh` sets it up, and `soulx.py` adapts `.sc` → SoulX metadata and runs it over ssh, the same way as `seedvc.py`.

**Selection.** `render --with-vocals --singer diffsinger|soulx` (default: the winner). The winner is chosen by `sung_wer` first, then `voice_sim`, then `pitch_cents`, and confirmed by ear.

## Acceptance (River)

- `lyric_wer` ≤ 0.10, down from about 0.25 with whisper-base. the three misheard lines come out correct.
- The winning singer's `sung_wer` is at least 30% lower than today's chain (measured first as the baseline), with `voice_sim` ≥ 0.88 and `pitch_cents` ≤ 30.
- Listening checkpoint: the original, then the winner, then the old chain.

## Out of scope

Singer diarization (M2 step 3), backing vocals, AcoustID fingerprinting (title/artist come from flags or the file name for now), and a different DiffSinger bank.
