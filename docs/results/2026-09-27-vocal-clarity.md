# Vocal word clarity (SoulX-Singer) — results

Date: 2026-09-27. Branch `vocal-clarity`. Song: River, 30 s clip. The reference voice is the separated lead vocal.

`sung_wer` is whisper-small's word error rate on the sung vocal against the `.sc` lyrics.
It now counts only the words the singer is given; River's four "oh"s past the clip end used to count as misses.
**The original vocal scores 0.09 on it.** That is the realistic floor, because whisper-small also mishears the real singer ("eating" → "meeting").

## What changed

1. **Word-driven alignment** (e635644). Every lyric word gets a note. Note-driven assignment had dropped 11 of 61 words.
2. **A fixed seed** (`soulx.SEED = 0`). SoulX is a diffusion model: unseeded repeats of one setting ranged from 0.20 to 0.41, which hid every real effect. Seeded repeats are identical.
3. **A 12 s voice prompt** (was 8 s).
4. **Held notes.** A word's last note is held while `:contour.vox` is still voiced. The singer is voicing for 23 s of the clip; 5.8 s of it was sent to SoulX as rest, and is now 4.3 s.
5. **The SoulX settings** (control, prompt, seed, alignment version) are now part of the vocal cache key.

## Seeded sweep (seeds 0 and 1)

| Setting | sung_wer | voice_sim | pitch |
|---|---|---|---|
| 8 s prompt | 0.26 / 0.26 | 0.97 | 2–3 c |
| **12 s prompt** (default) | **0.18 / 0.25** | 0.96–0.97 | 18–19 c |
| 12 s, cfg 5 | 0.28 / 0.26 * | 0.97 | 18–19 c |
| 12 s, cfg 2 | 0.26 / 0.41 * | 0.96–0.97 | 18–19 c |
| 12 s, 64 steps | 0.23 / 0.31 * | 0.97 | 18–19 c |
| 12 s + held notes | 0.23 / 0.25 | 0.96–0.97 | 18 c |

\* Scored with the older metric, which also counted the four "oh"s. Those numbers run a few points high (the 12 s default was 0.23 / 0.30 on the older metric).

**Pitch.** The 8 s runs follow the `.sc` contour almost exactly (2 c). The 12 s runs have no offset and no lag, but ±18 c of spread around the note, which is expression from the longer prompt. Both are inside the 30 c target.

## Remaining errors

The misses are now scattered single words, and differ between seeds. Only "mercy" (heard as "murmurs") fails in every render, and "it is" vs "it's" counts as two errors.

Held notes are within seed noise of no-hold on `sung_wer`. **By ear the unheld 12 s render sounded better, so held notes were reverted.**
The files to compare are in `~/Downloads/vocal-clarity/`:
- `1-original-vocal`
- `2-soulx-12s-prompt`
- `3-soulx-12s-held-notes`
- `0-soulx-before-8s-prompt`
