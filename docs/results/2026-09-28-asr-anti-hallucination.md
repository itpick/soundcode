# Whisper anti-hallucination settings — lyrics check

Date: 2026-09-28. Branch `demo-site`.

## What changed

`stage_lyrics` now:
- transcribes with `condition_on_previous_text=False`;
- drops segments with `no_speech_prob > 0.6`;
- drops a transcript that is only a common hallucination ("thank you", …) when any word's probability is below 0.5.

## Why

The demo-site build found Whisper "hearing" **"Thank you."** on two instrumental clips (999,999 and Corona Radiata). SoulX then sang it over a choir line.

## Check

This is a pipeline-wide ASR change, so both lyric clips were re-encoded before and after it:

| Clip | Words | `lyric_wer` vs LRCLIB | Lyrics source |
|---|---|---|---|
| Discipline 30 s | 31 → 31 | **1.81 → 0.00** | `whisper` → `lrclib+large-v3-turbo` (the ASR now reconciles with the published lyrics) |
| River 30 s | 61 → 56 | 0.00 → 0.08 | `lrclib+…` both |

- **River lost `life oh oh oh oh`** at the clip end. The four "oh"s start past the clip end, and the singer already dropped them. **"life" is a real loss:** the last reference word after the last heard word is now treated as untimeable.
- **The instrumental clips** now get no `:text.vox`, with a warn. Their vocal-stem choir line is rendered as its instrument, not sung.

## Follow-up

Keep a reference word whose LRC line starts inside the clip even when ASR stops hearing words before it. That recovers River's "life".
