# Production matching and the song's own drums — design

Date: 2026-09-27. Status: approved direction in conversation ("go with 1 then 2").
North star: rebuild the song **very close to the original**. The render today is dry and generic: every part plays through one General MIDI SoundFont, with no EQ, reverb, stereo image or dynamics of the record. This step measures those properties per stem in the encoder, writes them into the `.sc`, and applies them in the renderer. It also plays drums with hits cut from the song's own drum stem.

## Scope

1. **Production profile per stem (`fx`)**, measured from the original stem and applied to the rendered part:
   - tone (EQ curve);
   - room (reverb decay and wet level);
   - stereo width and pan;
   - dynamics (crest factor → compression).
2. **The song's own drum kit**: one-shot samples of each drum voice (kick, snare, clap, hat, …), cut from the separated drum stem at tsumugi's hit times, and played at the rebuilt pattern.
3. **A new `compare` metric, `spectral_db`**: the mean absolute difference in dB between the original stem's and the render's long-term spectra (1/3-octave). It measures whether a part *sounds* like the original, which note F1 cannot.

Not in this step:
- better sample libraries and synth patches (sfizz, DawDreamer);
- piano pedal and guitar bends;
- the ACE-Step polish pass.

## 1. The `fx` profile

**Measured** by the encoder per stem that produced a note or perc stream. The measurement uses only the gate-active blocks (Plan 1 loudness gate), on the stereo stem:

| Field | Measure | Stored as |
|---|---|---|
| `eq` | Long-term average spectrum in 31 one-third-octave bands, 20 Hz–20 kHz, relative to its own mean, in dB | 31 integers |
| `rt60` | Reverb decay: the median slope of the energy decay after isolated note offsets (onsets from the stream's notes), extrapolated to −60 dB, clamped to 0.1–4 s | seconds |
| `wet` | Direct-to-reverberant proxy: energy in the 50–400 ms tail after offsets, relative to the note body, mapped to 0–1 | 0.00–1.00 |
| `width` | Side/mid energy ratio | 0.00–1.00 |
| `pan` | (R − L)/(R + L) energy | −1.00 to 1.00 |
| `crest` | Peak-to-RMS over active blocks | dB |

It is written as one statement in the stream, which the parser already keeps:

```
:notes.piano inst=keys.piano
meta    src=tsumugi:default@020edc1  conf=0.97
meta    stem=piano  level=-36.3dB
fx      eq=-12,-8,-5,-2,0,1,2,2,1,0,-1,-1,-2,-2,-3,-4,-4,-5,-6,-7,-8,-10,-12,-14,-17,-20,-24,-28,-33,-38,-44  rt60=0.62s  wet=0.18  width=0.35  pan=-0.10  crest=14.2dB
```

**Applied** in `render_sf.render_streams`, per stream, after synthesis and before level matching:
1. **EQ.** Measure the rendered part's own 31-band spectrum and apply the difference (target − rendered) as a smooth gain curve. This is a zero-phase STFT-domain filter, with gains interpolated between band centres and clamped to ±15 dB.
2. **Reverb.** `pedalboard.Reverb`: `room_size` is derived from `rt60` by a fixed monotone map, `wet_level = wet`, and `dry_level = 1 − wet/2`.
3. **Compression.** Only when the render's crest factor exceeds the target by more than 3 dB: a `pedalboard.Compressor` whose ratio is chosen to close the gap (clamped 1.5–6:1).
4. **Width and pan.** Mid/side scaling to the target width; constant-power pan.

Level matching (already built) runs last, so loudness stays exactly as before.

Streams without an `fx` line render exactly as today, so old files are unaffected. `--no-fx` on `render` disables the stage for A/B checks.

## 2. The song's own drum kit

**Built** by the encoder's drums stage when tsumugi produced `:perc.drums`:
- For each drum voice, take the hit onsets and choose up to 4 **isolated** hits: no other hit within 80 ms before or 150 ms after. Prefer the median-energy ones, so the samples are typical rather than extreme.
- Cut each from the stereo drum stem, from onset − 5 ms to the next hit or +0.6 s (whichever is first), with a 5 ms fade-in and a 20 ms fade-out.
- Write them as `<work>/kit/<voice>_<k>.wav` (44.1 kHz, float).
- A voice with no isolated hit gets no sample, and falls back to the GM kit sound.

**Referenced** in the `.sc` by the drum stream: `meta kit=<path>`. The path is relative to the `.sc` file when possible, otherwise absolute. This is the first `.sc` reference to audio files; the design already anticipated it ("`:audio.*`, stems as first-class"). A missing kit folder is a `meta warn` at render time and a GM-kit fallback, never a failure.

**Played** by a new drum path in `render_sf`, used when `meta kit` resolves:
- each hit plays one of its voice's samples, round-robin so repeats don't sound machine-gunned;
- the sample is scaled by `(vel/127)^1.5` relative to its own cut level;
- the hit is placed at the note time.
- Voices without a sample use the GM kit on a separate channel, mixed in.

The `fx` profile then applies to the drum stream like any other.

**Honesty note.** The drum samples are audio from the original recording, so this part of the rebuild is partly *sampled*, not purely synthesized from code. It is the faithful-rebuild path the roadmap asked for ("re-render = both"). The `.sc` still carries every hit as code, and deleting `meta kit` gives the pure-code render.

## 3. `spectral_db` in `compare`

For each stem, `spectral_db` is the mean absolute dB difference between the two long-term 1/3-octave spectra, each normalised to its own mean. Only bands where the original is within 50 dB of its loudest band count. A silent side gives `null`. It is added to the table, `report.json` and the HTML.

## Evaluation and acceptance

- **River and discipline, with fx vs `--no-fx`:** `spectral_db` improves (drops) on every active stem, and the drums improve most once the kit is used.
- **Drum kit:** River's kit folder has at least `clap` and one more voice. A render with the kit vs without has lower drum `spectral_db`.
- **No regressions:** level diff, note F1 and onset F1 stay within ±0.02 of the no-fx render. Neither fx nor the kit may change timing or notes.
- **Listening checkpoint:** River, then discipline — original, then the new render (with vocals for River).

## Testing

Unit tests, with no models needed:
- band spectrum of a synthetic tilt;
- the EQ moves a render's spectrum toward the target (`spectral_db` drops);
- rt60 of a synthetic exponential decay;
- width and pan of synthetic stereo signals;
- the crest-triggered compressor;
- `fx` line round-trip through the parser;
- a render without `fx` is byte-identical to before;
- isolated-hit selection on a synthetic hit train;
- kit cut boundaries and fades;
- round-robin playback;
- the missing-kit fallback;
- the `spectral_db` metric.

A real test (skipped when absent): River's drum kit built from its stem, with `clap` present.
