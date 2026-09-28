# Correct words + clearer singing — results (River, 30 s)

Date: 2026-09-27. The trigger was user feedback on the sung rebuild: "the singing i don't think has correct words and it doesn't sound great".

## Words

| | `.sc` lyrics vs published (LRCLIB) |
|---|---|
| before (whisper-base) | ≈ 0.25 WER: three lines of the chorus misheard |
| after (large-v3-turbo + LRCLIB reconcile) | **0.00**: the three previously misheard lines are correct |

LRCLIB has one error of its own: one word the singer sings as "meeting". The ASR heard it the same way this time, so nothing disagreed. A second lyrics source or a phonetic tie-break is a follow-up.

## Singing: head-to-head on the corrected `.sc`

| lead vocal | SoulX-Singer | DiffSinger (dsdur) → Seed-VC 50 steps | before (heard by the user) |
|---|---|---|---|
| voice similarity (resemblyzer) | **0.96** | 0.93 | 0.93 |
| pitch error | **15 c** | 19 c | 19 c |
| note F1 | **0.64** | 0.48 | 0.42 |
| onset F1 | **0.73** | 0.51 | 0.53 |
| energy correlation | **0.66** | 0.21 | — |
| spectral_db | **1.3** | 3.4 | — |
| sung_wer (intelligibility; lower is better) | 0.31 | **0.16** | 0.32 |
| time per 30 s (on framepick, RTX 5070) | **~50 s** | ~2.5 min | ~2.5 min |

**Chosen by ear: SoulX-Singer.** The user: "the one before this one sounded better" (SoulX played before DiffSinger). It also wins every metric except intelligibility. The plan ranked singers by `sung_wer` first; that is overridden by the listening verdict, and recorded as a ruling. `sing.DEFAULT_SINGER = "soulx"`, with DiffSinger→Seed-VC as the automatic fallback.

Acceptance:
- `lyric_wer` ≤ 0.10: **met** (0.00).
- Intelligibility ≥ 30% better than before: **met by the DiffSinger chain** (0.32 → 0.16, −50%). **SoulX, the chosen default, is level** (0.31). Improving SoulX's word clarity is the next task.
- `voice_sim` ≥ 0.88 and `pitch_cents` ≤ 30: met by both.

## Bugs found by real runs and fixed

1. Reconciled lyrics put reference-only "oh oh oh oh" at 30.1–31.7 s, past the 30 s clip, and both singers failed on "phoneme timing overflowed". Reference words after the last heard word are dropped, and the score builder drops words past the end.
2. SoulX-Singer's pinned torch 2.2 has no Blackwell (sm_120) kernels, which gave a CUDA device-side assert on the RTX 5070. It now uses a torch 2.9.1 cu128 build.
3. torchaudio 2.9 needs torchcodec + FFmpeg, which the Framework lacks. SoulX loads audio with soundfile (`scripts/soulx.patch`).
4. NLTK's cmudict download failed TLS on NixOS: `SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt`.
5. Twice, a playback checkpoint played the DiffSinger fallback in place of a failed SoulX run. The checkpoint script now plays only when SoulX really sang.

## Follow-ups

- SoulX intelligibility (0.31): try `--control score` (MIDI notes) vs `melody`, a longer or cleaner prompt clip, and explicit syllable-per-note alignment.
- A second lyrics source, or phonetic arbitration, where LRCLIB is wrong.
- Song identification from audio (AcoustID/MusicBrainz) when the file name lacks title and artist.
