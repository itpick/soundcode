# Render review: where the rebuild loses fidelity, and what to do about it

Date: 2026-09-28. Read-only review of `soundcode` (branch `demo-site`) by Fable 5.1.
Ground truth used: the code under `src/soundcode/`, the results in `docs/results/*`, the three sample
files `out/sc/lv/river-30s.sc`, `out/sc/main/discipline-30s.sc`, `out/sc/main/lights_in_the_sky-30s.sc`,
their kits, and the stems in `out/stems/<song>/`. Numbers marked *(measured here)* come from small
read-only scripts run in the scratchpad against those files; everything else cites a file:line or a
results doc. Model/licence claims for tools not in the repo are from `docs/research/*` and my own
knowledge; the ones I could not confirm from the repo are marked *(verify)*.

Constraints honoured throughout: open weights only, non-commercial OK, Apple-Silicon Mac + `framepick`
(RTX 5070 8 GB), `.sc` stays the source of truth.

---

## 0. The renderer in one paragraph (what the code actually does)

`render_sf.render_streams` (`src/soundcode/render_sf.py:149-178`) expands the `.sc` to flat notes, then per
stream: picks one General MIDI program by `inst=` family (`gm.py:14-27`, `gm.py:66-76`), turns notes into
note-on/off events with the pitch **rounded to the semitone** except for `voice` streams
(`render_sf.py:39-56`, `gm.py:55`), synthesises through tinysoundfont + GeneralUser GS on one channel
(`render_sf.py:102-129`), applies the stream's `fx` line (31-band static EQ match, Freeverb, crest limiter,
mid/side width, pan — `fx.py:164-191`), then scales the whole stream to one `meta level` dB
(`render_sf.py:172-176`). Drums with `meta kit=` play one-shots cut from the song's own drum stem
(`kit.py`), GM kit for missing voices. `mix()` sums, multiplies each `:struct` section by its `energy`
value as a linear gain (`render_sf.py:224-227` → `render.py:183-195`), then peak-normalises to 0.89
(`render_sf.py:228-230`). The lead vocal is sung by SoulX-Singer on `framepick` from words + notes +
`:contour.vox` f0 (`soulx.py`), loaded back at 24 kHz, given the lead stem's `fx`, and level-matched
(`render_sf.py:272-290`). Nothing is rendered for backing vocals, `other`, textures, or the residual.

---

## 1. Where the rebuild loses the most fidelity today, per part

Scores below are from `docs/results/2026-09-27-production-match.md` (run C: fx + kit) and
`2026-09-27-lyrics-and-voice.md` / `2026-09-27-vocal-clarity.md` unless stated.

### 1.1 Piano / keys — biggest loss: dynamics and sustain, then timbre

- **Velocity is absolute loudness, not dynamics.** tsumugi's velocity model output is written straight
  into the file (`tsumugi_sc.py:53-62`, `encode.py:669`). *(measured here)* River `notes.piano.piano`
  velocities p10/50/90 = 29/36/55, River `notes.piano` (the keys on the "guitar" stem) 30/32/36, every
  vocal note in all three songs is exactly 29, while Discipline's bass sits at 91/123/126. Velocity tracks
  the stem's dB level (River keys −36/−41 dB → ~30; Discipline bass −16 dB → 123): loudness is encoded
  twice (velocity and `meta level`) and *within-part* dynamics hardly at all. Through a GM SoundFont a
  velocity-29 piano is the softest, dullest sample layer; the level match (`render_sf.py:172-176`) then
  lifts it ~20 dB and the EQ match (`fx.py:140-157`, clamp ±15 dB) tries to put the missing brightness
  back with a static filter. Result: an EQ'd soft-layer piano, and no accents.
- **Sustain and pedal.** Durations come from tsumugi note offsets with no pedal model; the renderer sends
  plain note-off at `start + dur` (`render_sf.py:55`) and no CC64. *(measured here)* River `notes.piano`
  durations cluster at 0.20–0.24 s (23 notes) and 0.30–0.50 s (96); the chords in `notes.piano.piano`
  are ~1.9 s (a bar). GM piano release is short, so 0.46 s notes read as staccato where the record
  is pedalled. Note F1 stays 0.22–0.43 (`compare-tsumugi.md`), energy correlation 0.8–0.9 (good), but
  the "smear" of the real instrument is missing.
- **Timbre.** GeneralUser GS "Acoustic Grand" for every `keys.*` except the mapped EP/clav/organ
  (`data/tsumugi_classes.json`). The static 31-band EQ gets `spectral_db` to 2.1 dB (River, Discipline)
  but cannot make one piano sound like another across the keyboard (per-note spectra differ).
- **Two piano streams from two stems** (River `notes.piano` + `notes.piano.piano`, Lights
  `notes.piano` + `notes.piano.other`) both on the same GM program, polyphony *(measured here)* 3.7 + 4.4
  voices: eight simultaneous piano voices for what is probably piano + pad/EP. The inventory's runner-up
  (`electric_piano 0.28`) is never used by the renderer.

### 1.2 Bass — loss: identity and articulation; on River, the stem itself

- River's bass stem is piano bleed at −58.9 dB (`out/stems/river-30s/bass.wav`, 0 % of frames above
  −45 dB *(measured here)*), yet it still yields a `notes.bass inst=synth.bass` stream of 23 notes at
  velocity ~30 (`river-30s.sc:253-280`). `spectral_db` 10.7 dB, note F1 0.11. That stream should be
  omitted (the loudness gate passes because the *peak* 100 ms floor is −50 dB, `encode.py:64-89`).
- Discipline bass (real, −16 dB): note F1 0.39, onset F1 0.79, `spectral_db` 3.1 dB, energy corr 0.72.
  Rendered as GM "Finger Bass" (program 33) with semitone-rounded pitch and no slides, no pick/fret
  noise, no distortion (NIN bass is a driven synth/bass). No `:contour.bass` exists although torchcrepe
  on a monophonic bass stem is cheap and the vocal path already proves the value of a contour (pitch
  error 56 c → 19 c, `singing-thin-slice.md`).
- Durations again cluster at 0.22–0.24 s *(measured here: 59 of 117)* — plausible 8ths at 123 BPM, but
  the render is a string of plucks with no legato.

### 1.3 Guitar — the single worst pitched part: envelope and distortion

- Discipline `notes.electric.guitar inst=gtr.electric.distortion`: note F1 0.22, **onset F1 0.39,
  energy correlation 0.25** — the lowest on any real part. *(measured here)* 66 of 138 note durations
  are 0.20–0.24 s; the original is a sustained distorted wall (stem sounding 100 % of frames). The
  render is GM "Distortion Guitar" plucks that decay, so the envelope shape is wrong even where the
  pitches are right.
- Distortion is not measurable or reproducible in the current `fx` model: EQ + reverb + limiter cannot
  create the harmonic density of an amp (`fx.py:164-191` has no saturation stage). `spectral_db`
  reaches 1.0 dB only because the static EQ curve was matched — a good example of a metric the renderer
  optimises directly (see §7).
- No chord voicings/strums beyond what tsumugi's offsets give; no palm-mute/open articulation; no bends.

### 1.4 Synths / other / textures — not represented at all

- `other` is gated silent on River and Discipline, but on 999999 (`other` −33.6 dB, note F1 0.11,
  energy corr −0.05) and Lights (`other` −33.8 dB, reassigned to keys) it carries pads/ambience that a
  note list cannot hold. `:texture.*` (spec §4.3.9) is not implemented; `stage_mix` never writes
  `by_section` or `!auto`; nothing renders `!auto` (roadmap "known bugs").
- The residual (`residual.wav`) is −51 dB on River but **−34.6 dB and sounding 100 % of the time on
  Discipline** *(measured here)* — only 23 dB under the mix. For NIN material the residual is the
  room/glue/noise bed, and it is discarded.

### 1.5 Drums — the best part, but choppy: every one-shot is truncated

- Onset F1 0.97–0.98, `spectral_db` 0.4–0.9 dB with the kit (`production-match.md`). Good.
- **Every kit sample is cut at ~0.24 s with its tail still at −14 to −26 dB below peak** *(measured
  here on all 32 samples in the two kits)*. Cause: `kit.build` cuts each hit "to the next hit or
  +0.6 s, whichever is first" (`kit.py:46-47`), and with a hi-hat every eighth (median gap 232 ms
  *(measured here)*) the next hit is always ~240 ms away. `MAX_S = 0.6` is never reached; the 20 ms
  fade-out chops snare/clap/tom decays and every open hat. Discipline's `crash` has no sample and
  falls back to GM. The `rt60 0.30 / wet 0.10` defaults do not put the tails back.
- Velocity layers: samples are chosen nearest the median velocity (`kit.py:30-32`) and scaled by
  `(vel/127)^1.5` (`kit.py:92`); a loud snare is a soft snare turned up. Round-robin of ≤4 helps.
- The GM fallback voices (`gm.DRUM_DEFAULT` clap for unknown) and the `stick`/`hat` co-onsets
  (`river-30s.sc:59-64`) suggest tsumugi labels sidestick vs closed hat inconsistently; not
  measurable with the current metrics (no per-voice onset F1).

### 1.6 Lead vocal — best measured part; the ceiling now is words and naturalness

- SoulX: voice_sim 0.96–0.97, pitch 15–19 c, note F1 0.64, onset F1 0.73, `spectral_db` 1.3,
  `sung_wer` 0.18–0.25 vs 0.09 for the original (`vocal-clarity.md`). Chosen by ear.
- Remaining losses: (a) intelligibility gap 0.09 → 0.18–0.25; (b) SoulX output is 24 kHz
  (`soulx.py:32`) so nothing above 12 kHz — the lead-stem EQ can't lift empty bands
  (`fx.py:148`); (c) the prompt is the *first* ~12 s of the lead stem (`soulx.py:168-178`), not the
  cleanest 12 s; (d) one seed, one take — seed variance is ±0.1 WER (`vocal-clarity.md`), and the
  spec's take-selection loop (§7.3) is not implemented; (e) breaths, consonant timing and phrase
  dynamics come from SoulX, not the record; velocity is 29 everywhere so no dynamic hint exists.
- corona_radiata (choir-like) 0.61 voice_sim and 999999 (0.80) show the one-prompt model fails on
  layered/processed vocals.

### 1.7 Backing vocals — 100 % loss where they exist

- `encoder_stems` keeps only the lead (`encode.py:208-220`); no `:notes.bvox`, `:text.bvox`,
  `:contour.bvox`. River backing −43.5 dB sounding 31 % of frames, Discipline −35.7 dB sounding 37 %
  *(measured here)*. Every results table shows `backing_vocals 0.00`. The SoulX chain would work as-is
  with the backing stem as prompt (`singing-spike.md` §"next steps" 5).

### 1.8 Mix / master — no bus, wrong dynamic arc, generic reverb

- No master processing: sum → section gain → peak normalise (`render_sf.py:204-231`). `:mix`
  (`lufs_int` = RMS dBFS, `stereo_width`, `centroid`, `rolloff`) is written (`encode.py:754-785`)
  but never read by the renderer. Loudness, bus compression, master EQ, and stereo image per band are
  not matched.
- **Dynamic arc:** one `level` per stream for the whole clip; `section_gains` multiplies by
  `:struct energy` (a linear RMS ratio, `encode.py:329-335`) on top of already level-matched parts
  (`render.py:191-194`). With per-note velocity flat, nothing carries crescendos inside a section.
  Energy correlation on real parts is 0.5–0.9; this is the term that will collapse on full songs (§6).
- Reverb is `pedalboard.Reverb` (Freeverb) with `room_size` from a fixed map (`fx.py:160-161`,
  `179-181`); the "room" is only measured after isolated notes (≥0.5 s gap, `fx.py:94-98`) and
  otherwise defaults to 0.3 s / 0.10. Busy parts therefore get a generic small room regardless of the
  record. Width for mono renders is faked with an 11 ms Haas roll (`fx.py:186-187`) — comb filtering;
  River's keys stream asks for `width=0.98 pan=0.70`.
- Same-key retriggers: tinysoundfont `noteoff` on one channel kills every voice on that key
  (`render_sf.py:119-122`), so overlapping repeated notes in the two piano streams cut each other.

---

## 2. Concrete improvements to the current renderer, ranked by audible gain ÷ effort

| # | Change | Where | Gain | Effort | Acceptance (compare + ear) |
|---|---|---|---|---|---|
| 1 | **Drum one-shot tails.** Cut to `MAX_S` (raise to 1.0 s; 2 s for crash/ride/open hat), not to the next hit. Prefer hits followed by a ≥0.6 s gap when the song has any (River bars 10–15 are snare-only); otherwise extrapolate: fit the decay slope of the last 60 ms before the next hit and extend with matched-band noise, or cut from a per-voice separated stem (LarsNet, DrumSep — roadmap alternatives). Store the chosen velocity per sample and pick the nearest layer at play time. | `kit.py:46-47`, `kit.py:15-16`, `kit.py:77-93` | high (drums are 1/3 of these mixes) | small | drums `spectral_db` ≤ now, energy corr up (Discipline 0.91 →), no onset-F1 change; ear: "not choppy", open hats ring. |
| 2 | **Decouple velocity from loudness.** In the encoder, rescale each stream's velocities so p50 ≈ 80–90 and the p10–p90 spread is kept (or compute per-note velocity from the stem's own RMS around the onset relative to the stream's peak, as `stage_notes_poly` already does at `encode.py:611`). `meta level` keeps loudness. | `tsumugi_sc.py:53-62`, `encode.py:669` | high for keys/vocals/bass on quiet stems | tiny | `--no-fx` `spectral_db` drops (brighter layers); log the mean |EQ gain| the match applies and require it to fall; ear: accents audible. |
| 3 | **Sustain, legato and pedal.** For `keys.*`: send CC64 (sustain) while ≥2 notes overlap, or extend each note to the next onset of the same pitch class (cap 2 bars) when the stem's band energy at the note's pitch has not dropped >12 dB (score-informed offset check on the stem CQT). For `gtr.*`/`synth.pad`/`strings`: "hold until the next chord change". Write the result to the file (`dur`) so the `.sc` stays the truth. | `render_sf.py:39-56`, `tsumugi_sc.py:58-62`, new encoder pass | high for guitar (energy corr 0.25) and keys | medium | Discipline guitar energy corr ≥ 0.6, onset F1 not lower; River piano `spectral_db` ≤ now; ear: sustained wall, pedalled piano. |
| 4 | **Per-bar level automation per stream.** Encoder measures the stem's RMS per bar (or per beat, smoothed) relative to `meta level` and writes `!auto <stream>.gain bar:beat dB` lines (the grammar already has `!auto`, `parser.py:254`); renderer applies a gain envelope per stream before fx; drop the `:struct energy` linear gain. | `encode.py:661`, `render_sf.py:172-176`, `render_sf.py:224-227` | high; required for anything >30 s | small–medium | energy corr ≥ 0.9 on every real stem; per-section level diff ≤ ±1 dB (§6 metrics). |
| 5 | **Better instruments per family via SFZ.** sfizz (BSD-2, archived but works; `pysfizz` *(verify wheel)*) or DawDreamer (GPLv3) hosting sfizz VST3: Salamander Grand (CC BY 3.0) for `keys.piano`; VSCO 2 CE (CC0) strings/brass/winds; Karoryfer basses (CC BY 4.0 *(verify)*); keep GeneralUser GS as fallback. Add `render=` hint per stream for the preset. | `gm.py`, new `render_sfz.py`, `render_sf.py:162-166` | high for piano-led songs (3 of 5 test clips) | medium | River/Lights piano `spectral_db` with `--no-fx` drops below the GM value; ear A/B. |
| 6 | **Distortion for `gtr.electric.distortion` / driven bass.** Render a clean guitar (SFZ or GM clean) then pass it through an open amp model: Neural Amp Modeler core (MIT) with free NAM captures, or pedalboard's `Distortion`+`Convolution` cab IR as the cheap version. Encoder adds a `dist=` scalar (spectral flatness / harmonic-to-noise vs a clean template) to `fx`. | `fx.py:164-191`, `encode.py:122-141` | high on NIN material | medium | Discipline guitar/bass `spectral_db` with a *fine* (1/12-oct) spectrum ≤ 3 dB; ear. |
| 7 | **Reverb from the record, not Freeverb.** Measure per-band RT60 (EDR in 6 octave bands) at isolated offsets; synthesise an exponentially decaying noise IR per band (pedalboard `Convolution`) instead of `Reverb(room_size)`. Replace the Haas width trick with an all-pass decorrelator. Apply fx per `:struct` section (measure per section in the encoder). | `fx.py:71-90`, `fx.py:160-191`, `encode.py:136-141` | medium | small each | `spectral_db` unchanged, energy corr up on reverberant stems (Lights piano rt60 1.4 s); ear: "same room". |
| 8 | **Master bus.** Match the sum to the original mix: 31-band EQ on the full mix, integrated loudness via pyloudnorm (MIT) to the measured LUFS (fix `lufs_int` to real LUFS), per-band stereo width from `:mix stereo_width`, gentle bus compressor from the mix's short-term loudness spread, peak-limit last. | `render_sf.py:204-231`, `encode.py:754-785` | medium ("glue") | small | full-mix `spectral_db` ≤ 1.5 dB, LUFS diff ≤ 0.5 dB, width per band ≤ 0.1; ear. |
| 9 | **Vocal takes and prompt selection.** Render 2–3 SoulX seeds, choose by `sung_wer` + pitch error (spec §7.3 retake); choose the prompt window with the best lead/backing energy ratio and torchcrepe periodicity rather than the first phrase; A/B `--control score`. | `soulx.py:168-178`, `sing.py:57-95` | medium | small | `sung_wer` ≤ 0.15 on River (floor 0.09); voice_sim ≥ 0.96 kept. |
| 10 | **Backing vocals.** Encode `:notes.bvox`/`:text.bvox`/`:contour.bvox` from the backing stem (same stages), sing with SoulX using the backing stem as prompt, apply the backing stem's fx/level. | `encode.py:208-220`, `sing_score.vocal_stream` | high where present (2 of 3 sample songs) | medium | backing row no longer `0.00`: chroma ≥ 0.85, voice_sim ≥ 0.85; ear. |
| 11 | **Texture/residual bed.** `:texture.residual` (and `:texture.other` when notes are sparse): per-bar 1/3-octave band energies from the residual/other stem; render as band-shaped noise with those envelopes (deterministic), later a generative bed (§3.3). | `separate.py`, new `texture.py` | medium on NIN/ambient, nil on River | small | 999999/corona `other` energy corr ≥ 0.5; full-mix `spectral_db` drops. |
| 12 | Voice stealing: one tinysoundfont channel per stream, and skip the note-off when a later same-key note is still sounding. | `render_sf.py:102-129` | small | tiny | no metric; removes clicks/cuts. |
| 13 | Omit bleed streams by *stream* loudness, not stem peak: River `notes.bass` at −52.8 dB / 23 notes should be `omitted — bleed`. | `encode.py:64-89`, `tsumugi_sc.drop_bleed` | small (removes a wrong part) | tiny | River bass row `—`. |

Things I would **not** spend effort on now: timing humanisation (the file already carries performed
onsets at 3-decimal beats, ≈2 ms; `tsumugi_sc.py:40-47`), microtonal cents for instruments (tsumugi
emits integer MIDI), and `env{}` ADSR blocks (samplers ignore them; sustain is the issue, not attack).

---

## 3. Different ways to render

Each entry: what it is, what it would take here, what it would sound like, and how it keeps `.sc` as
the source of truth. Fit: **5070** = fits the 8 GB RTX 5070; **Mac** = runs on Apple Silicon.

### 3.1 Neural / DDSP timbre transfer per stem, conditioned on the `.sc` notes

- **DDSP per-song instrument (monophonic parts: bass, lead lines, single-note synths).**
  Train a tiny DDSP harmonic+noise autoencoder on the song's own stem (DDSP is data-efficient; a full
  song's bass stem is enough for a usable model, 30 s is marginal). At render time drive it with
  f0 + loudness frames synthesised from `:notes.bass` (+ portamento, or `:contour.bass` once encoded)
  and the velocity/`!auto` envelope. Tools: `magenta/ddsp` (Apache-2.0, TensorFlow — painful on Mac,
  fine on framepick), `acids-ircam/ddsp_pytorch` (PyTorch, MPS-capable; licence *(verify)*),
  DDSP-SVC (MIT) for the vocoder side. Fit: 5070 and Mac.
  Sound: the record's bass timbre with our notes; "the bass player re-recorded the line".
  Truth: the model is a sidecar like `meta kit=` (`meta model=<song>.bass.ddsp`); notes stay in the
  file; delete the meta and the sampler path renders. Honest label: song-sampled timbre.
- **Polyphonic parts (piano, guitar, pads): score-informed NMF resynthesis.** With the `.sc` notes
  known, run score-informed NMF on the stem's CQT/STFT (librosa `decompose.nmf`, BSD) with one
  template per pitch initialised from the notes' harmonic series; the learned templates are the
  instrument's per-note spectra, the activations are constrained by the notes. Render = templates ×
  activations built from the (possibly edited) `.sc`, inverse STFT with the stem's phase where the note
  is unchanged and Griffin-Lim/phase-vocoder elsewhere. No training, CPU/Mac, deterministic.
  Sound: very close to the stem for unedited notes; slightly phasey on edited notes.
  Truth: templates are timbre-only sidecar data; every event comes from the `.sc`.
- **"Auto-SFZ from the stem"** (the drum-kit idea generalised): cut isolated single notes from the
  piano/guitar stem at tsumugi onsets, one per pitch where available, pitch-shift neighbours ±3
  semitones (librosa/rubberband) to fill gaps, write an SFZ with velocity layers from the measured
  RMS, play through sfizz. Fit: Mac. Sound: the record's piano on sparse passages; artefacts where no
  isolated note exists (dense chords). Truth: same as the kit.

### 3.2 MIDI-to-audio neural synthesis

- **TokenSynth** (`KyungsuKim42/tokensynth`, MIT, ICASSP 2025): polyphonic single-instrument audio
  from MIDI + a timbre reference (audio clip or CLAP text). Exactly "play `:notes.piano` with the sound
  of the piano stem". DAC codec + transformer; CPU supported (slow), CUDA recommended → 5070 *(fit
  unverified but likely; small model)*. Best first neural experiment: `.sc` stream → MIDI (symusic /
  pretty_midi) → TokenSynth with the stem as reference → per-stream wav into the existing fx/level
  chain. Sound: neural-codec quality (some fizz), timbre near the stem, note accuracy depends on the
  model. Truth: input is the `.sc` notes; reference audio is timbre only.
- **MIDI-VALLE** (Apache-2.0 code, CC BY 4.0 models): expressive *piano* only; codec LM; inference
  should fit 8 GB *(verify)*. Worth a try for the piano-led clips.
- **MIDI-DDSP** (Magenta, archived, TF): dead on Mac; skip. Break-the-Beat / Anysynth / P-MUSE: no
  code (`open-synthesis.md` §5d).
- **DiffSynth-Music** (Apache-2.0; adapters on ACE-Step 1.5): its *prosody* template takes a
  pitch+timing-only vocal (our SoulX or even a sine render) and generates natural vocals; *vocals*
  template generates accompaniment from a vocal; *beats* locks tempo to a click from `:grid`. Fit:
  CUDA documented with low-VRAM offload; 8 GB plausible *(verify)*. Truth: the control signal is a
  render of the `.sc`; verify by re-encode.

### 3.3 Generative models conditioned on the `.sc` — polish or restyle pass

| Model | Licence | Fit | Conditioning we can feed from `.sc` | Role here |
|---|---|---|---|---|
| ACE-Step 1.5 (2B turbo + 0.6B LM) | MIT | Mac (MLX/MPS), 5070 with INT8 (≤6 GB, spec §7.1) | `src_audio` = our rebuild + `audio_cover_strength`; `reference_audio` = original mix/stem for timbre; BPM/key/meter/duration; lyrics with section tags; `repaint` per `:struct` section; `lego` add-a-track | **Polish pass** at low strength (0.2–0.4): keeps our timing/notes, adds realism and glue; `repaint` a section that drifted. Also the planned Path B. |
| DiffSynth-Music | Apache-2.0 | 5070 *(verify)*, Mac untested | click track, prosody vocal, vocal/accompaniment stems | Cleaner control than cover strength; the best "faithful but generated" vocal and bed candidate. |
| YuE2-3B | code Apache-2.0, weights CC BY-NC 4.0 | Mac via mlx-Yue (fits 32 GB); official needs 24 GB CUDA (not the 5070) | supplied/edited ABC score (melody + chords from `:notes.vox` + `:harmony`), sections, lyrics, cover mode | Restyle with melody and chords locked; sung vocals included. |
| MusicGen-melody (1.5B) | code MIT, weights CC BY-NC 4.0 | 5070 (fp16 ~3 GB), Mac with patches | chromagram of our render as melody; text; 30 s windows | Instrumental restyle bed; weak timing adherence. |
| Stable Audio Open 1.0 + MuseControlLite adapters | SAO: Stability Community Licence; MuseControlLite MIT | 5070 (1.2B fp16), Mac slow | melody / rhythm / dynamics curves extracted from our render; ≤47 s; in/out-painting | **Textures, pads, drones** (999999, corona) — the parts a note list cannot hold; dynamics curve from `:texture` bands. |
| Stable Audio 3 Small (433M) | Stability Community Licence | Mac (CPU/MLX/CoreML), 5070 | text + `init_audio` with noise level | Cheap audio-to-audio polish of a texture bed. |
| MuLaCover | code Apache-2.0, weights CC BY-NC 4.0 | Linux + NVIDIA, tested on B300; 8 GB unlikely | melody/chord/drum MIDI + lyrics — a straight `.sc` export | Rented-GPU experiment only (roadmap Phase 4). |
| JASCO | weights CC BY-NC 4.0 | 5070 (400M/1B) | chords + melody salience + drum stem; 10 s clips | Research toy; the 10 s window kills it. |
| Magenta RealTime 2 | weights CC BY 4.0 | Mac (MLX) | text, audio examples, MIDI note control | Live textures; not faithful. |

How a generative pass keeps `.sc` as truth: treat it as a *post-process whose output must re-encode to
the same file*. Gate every generated section by the round-trip metrics (note F1 at cover tolerances
±100 ms/±100 c vs the `.sc` expanded notes, lyric timing MAE, chroma per bar); a section that drifts is
repainted or falls back to the deterministic render. Record in the output's provenance which sections
are generated and at what strength. Never write generated content back into the `.sc`.

### 3.4 "Resynthesis from stems" hybrid (the roadmap's `:audio.<trk>`, Phase 3)

Keep the separated stems next to the `.sc` (`<song>.stems/`, referenced by `meta audio=<stem>`, like
`meta kit=`). The renderer uses a **fidelity ladder per part**: original stem (unedited part) →
song-sampled timbre (kit / auto-SFZ / NMF templates / DDSP) → generic samples (SFZ / GM) → mock. A part
whose events were edited drops one rung automatically (or the NMF path resynthesises just the changed
notes over the untouched stem). This gives the near-original rebuild the north star asks for *today*,
and makes "change one part and re-render" seamless. Cost: small (`render_sf` reads `meta audio`,
`compare` reports the rung used). Honesty: the demo must label which parts are code-rendered; `compare`
should score the pure-code render separately (`--max-rung code`).

### 3.5 Residual / texture channel

Three rungs, all keeping `%residual` as a declared hook (spec §1.2):
1. `:texture.residual` band-energy rows per bar (text, honest, tiny) rendered as shaped noise
   (§2 #11).
2. A generated bed from the rows (Stable Audio Open + MuseControlLite dynamics curve, or ACE-Step
   `lego` on top of the rebuild) — "air" that follows the measured envelope.
3. A learned residual: encode `residual.wav` with an open neural codec (DAC, MIT; EnCodec, MIT) to a
   binary sidecar `%residual dac:<file>`. Closest to the original, but this is a codec and the spec
   calls compression a non-goal; keep it as an opt-in rung for the "very close" listening target and
   never as the default.

---

## 4. Encoding gaps that most limit rendering

1. **Velocity semantics** (§1.1): write relative dynamics; loudness lives in `meta level` +
   per-bar `!auto` gain. `tsumugi_sc.py:53-62`.
2. **Offsets / sustain / pedal**: score-informed offset check against the stem; `ped` events or CC64
   inference for `keys.piano`; "hold to next chord" for pads/distorted guitar. Durations clustering at
   one value *(measured here)* say offsets are the weakest field in the file.
3. **Per-bar level automation and real `:mix by_section`** (`encode.py:754-785` writes neither; LUFS
   is RMS dBFS). Needed before any 60 s+ evaluation (§6).
4. **Instrument contours**: `:contour.bass` (torchcrepe, monophonic) for slides; per-note `bend{}`
   for guitar from the same f0 track when a stream is monophonic in a window.
5. **Timbre evidence the renderer can act on**: the spec's `timbre{}` scalars (centroid, attack,
   harmonicity, `dist`, stereo) are never emitted (`encode.py` has no timbre stage); the inventory's
   runner-up class is written but unused. A preset picker (GM/SFZ/synth preset by nearest timbre)
   needs these. Add a `dist=` and `comp=` field to `fx`.
6. **fx per section, and richer fx**: per-band RT60, delay/echo detection (envelope autocorrelation),
   compression estimate (short-term vs long-term loudness spread), saturation. Measure inside each
   `:struct` section (`encode.py:136-141` caches one measurement per stem).
7. **Backing vocals** (§1.7): the whole vocal stage on `backing_vocals`, plus a harmony-vs-double
   flag (interval to the lead).
8. **Drum kit metadata**: velocity per sample, open/closed hat by decay length, and cut length; a
   per-voice onset F1 metric needs a drum transcription of the render too (tsumugi on the render).
9. **Grid honesty for long songs**: `stage_grid` writes one tempo and synthetic anchors computed from
   it (`encode.py:279-294`), and downbeat = first beat; `beat_this` downbeats → real anchors every bar
   or every 4 bars (roadmap Phase 1). Times stay exact either way, but bar-scoped automation, sections,
   harmony and `compare`'s per-bar blocks are wrong when the tempo drifts.
10. **Structure**: `stage_struct` caps at 6 sections and labels positionally (`encode.py:309,325`);
    `vocal` flag = energy > 0.45 (`encode.py:334`). Section-scoped fx/level/prompt selection all
    depend on this; SongFormer / allin1 / beat_this-novelty with the vocal stem's activity for the flag.
11. **Bleed streams**: gate by the *stream's* level (River `notes.bass` −52.8 dB).
12. **Lyrics**: `align=syllable` (or at least per-word phoneme string `ph=`) so SoulX/DiffSinger don't
    re-derive; breaths (`AP`) from the contour's unvoiced gaps; the LRCLIB synced-line times as a ±1 s
    prior for alignment on long songs (currently only used for the clip offset, `lyrics.py:108-121`).

---

## 5. Prioritised roadmap: the next five steps

Each step: change → `compare` acceptance → listening check. Start on the 30 s clips (fast loop), confirm
on Tier B/C material (§6) once step 2 lands.

1. **Drums that ring + velocities that mean dynamics** (§2 #1, #2, #13, #12). Two encoder changes,
   one kit change, one renderer channel fix.
   Accept: River/Discipline drums `spectral_db` ≤ 0.9/0.4 dB kept, drum energy corr ≥ 0.85 both; keys
   `--no-fx` `spectral_db` drops ≥ 1 dB on River and Lights; River bass row becomes `—`.
   Ear: River and Discipline, original then render; drums should stop sounding gated.
2. **Dynamic arc: per-bar level automation + sustain/pedal** (§2 #3, #4).
   Accept: energy corr ≥ 0.9 on every stem with a real part in all five clips; Discipline guitar energy
   corr ≥ 0.6 and onset F1 ≥ 0.39 kept; per-section level diff ≤ ±1 dB on `discipline-100s`.
   Ear: Discipline 100 s — the intro-to-verse lift and the guitar sustain.
3. **Master bus + reverb/width from the record** (§2 #7, #8).
   Accept: full-mix `spectral_db` ≤ 1.5 dB, LUFS diff ≤ 0.5 dB, per-band width diff ≤ 0.1; no stem
   metric regresses by more than 0.02.
   Ear: Lights (reverberant piano) and River.
4. **Instrument identity: SFZ piano/bass + amp model for distorted guitar; backing vocals** (§2 #5,
   #6, #10). Introduce the fidelity ladder (`meta audio=`/`render=`) so `compare` reports the rung.
   Accept: River/Lights piano `--no-fx` `spectral_db` ≤ GM value −1 dB; Discipline guitar 1/12-octave
   spectral distance ≤ 3 dB; backing row ≥ 0.85 chroma, ≥ 0.85 voice_sim where a backing stem exists.
   Ear: all three.
5. **First neural/generative experiments, measured the same way**: TokenSynth on River piano with
   the piano stem as reference; ACE-Step 1.5 cover at strength 0.2/0.4/0.6 on the Discipline rebuild
   with `reference_audio` = original; MuseControlLite on 999999 `other`.
   Accept: re-encode of the output vs the `.sc` keeps note F1 at cover tolerance ≥ 0.8 of the
   deterministic render's value and lyric timing MAE ≤ 80 ms; `spectral_db`/MERT-cosine (§7) improve.
   Ear: pick by the part scorer's ranking and confirm.

---

## 6. Longer material: what breaks with length, and a benchmark set

Full-length CC-licensed sources exist in `audio/nin/`: `999999.mp3` (85 s), `lights_in_the_sky.mp3`
(209 s), `discipline.mp3` (259 s) plus `discipline-100s.wav`, `corona_radiata.mp3` (453 s).

### 6.1 What degrades or breaks with length

| Stage | Behaviour on long files | Where | Consequence |
|---|---|---|---|
| Separation | Chunked by audio-separator/demucs; full River 196 s took 573 s on the M1 (README) → corona ≈ 20+ min; on framepick minutes. Memory fine. Demucs random shifts vary ±1 dB. | `separate.py` | Time only; run on framepick. |
| Grid | One tempo for the whole song, downbeat = first detected beat, anchors every 16 bars *computed from that tempo* (so they pin nothing). Real tempo drift or a rubato intro shifts every `bar:beat` label but not the times. | `encode.py:279-294`, `timebase.py` | Bars in `:harmony`, `:struct`, `compare` blocks and any per-bar automation misalign progressively; renders stay on time. Needs beat_this downbeats as real anchors. |
| Structure | `k = min(6, duration // 12)` sections, positional labels; `vocal` = energy > 0.45. | `encode.py:309-335` | Corona (453 s) gets ≤ 6 sections of ~75 s; section-scoped fx/level/prompt logic has nothing to hang on; `section_gains` scales by a meaningless energy. |
| Notes / drums (tsumugi) | ~5 s per 30 s per stem on MPS → ~75 s per stem for corona; velocity model likewise. Chunking inside tsumugi unknown *(verify note continuity at chunk edges: a held drone may be split)*. | `tsumugi.py` | Time only, plus a possible chunk-boundary artefact on drones. |
| Level | One `meta level` per stream over all active blocks. | `encode.py:661`, `render_sf.py:172-176` | The single biggest loss with length: a part that is quiet in verses and loud in choruses is rendered flat. Fix = §2 #4. |
| fx | One measurement per stem over the whole song. | `encode.py:136-141` | Averages a clean intro with a distorted chorus; rt60 improves (more isolated notes). Measure per section. |
| Kit | Isolated-hit search over the whole song → better samples. Truncation (§1.5) unchanged. | `kit.py` | Neutral to positive. |
| Vocal f0 contour | torchcrepe `full` on CPU at 100 Hz: ~real time → 7–8 min for corona; contour lines ≈ 50 values/s → a 4-min song adds ~250 lines. | `contour.py:70-87`, `encode.py:788-814` | Time; file size fine. |
| Lyrics | Whisper on the whole vocal stem with no `:struct` gating (the spec's hallucination defence, §5 stage 7, is not implemented); LRCLIB reconciliation works on the full text with `best_offset`; synced line times unused for alignment. | `encode.py:840-887`, `lyrics.py:108-199` | Hallucinated words in long instrumental gaps; word timing drift inside long verses. Gate by the contour's voiced spans (already in the file) and use LRCLIB line times as a prior. |
| SoulX | `MAX_SEG_S = 15` → ~30 segments for corona, one call; one 12 s prompt from the first phrase; fixed seed; `--auto_shift` once per call; ssh timeout 1800 s; ~50 s per 30 s on the 5070 → ~13 min for corona (a 10-min song would hit the timeout). | `soulx.py:33,168-178,257,263` | Timbre consistent (same prompt/seed); possible loudness/tone steps at segment joins (measure per-segment RMS/centroid continuity); raise the timeout or batch segments. |
| Render | ~160 MB per stream at 453 s; `eq_match` STFT ~1 GB transient; per-event 512-sample loop is fine. | `render_sf.py` | Fine on 32 GB. |
| compare | Global metrics only; basic-pitch and torchcrepe on both sides at full length (~10 min for corona); onset F1 window 70 ms is lag-sensitive; spectrogram PNGs unreadable at 4 min; energy corr dominated by section loudness. | `compare.py` | Needs per-section rows, a lag/drift check, and a per-stage timing log. |
| Listening | A 4-minute A/B is not judgeable by ear in one pass. | — | Checkpoint on 15–20 s excerpts per section, original then render, chosen by the scorer's worst sections. |

### 6.2 Benchmark set

| Tier | Item | Length | Why | Vocal? |
|---|---|---|---|---|
| A (regression, fast) | river-30s, discipline-30s, lights-30s, 999999-30s, corona-30s | 30 s | Existing tables; every change must not regress these. | River, Discipline, Lights |
| B (sections) | `discipline-100s.wav`; lights_in_the_sky 60 s (verse→chorus); corona 60 s; 999999 full (85 s) | 60–100 s | First section transition, per-bar level, SoulX multi-segment joins, `:struct` with ≥2 real sections. | yes / yes / choir / sparse |
| C (full) | discipline (259 s), lights_in_the_sky (209 s), corona_radiata (453 s), The River (196 s, private) | full | Drift, structure, memory/time, whole-song dynamic arc; Discipline has official multitracks (spec §10.2) → separation error can be measured apart from render error. | yes |

Per song, per part, report (all existing unless marked new): level diff, note F1 (+octave-agnostic),
chroma, onset F1, energy corr, `spectral_db`, and for vocals pitch cents / voice_sim / `sung_wer`;
**new:** the same table per `:struct` section (or fixed 8-bar windows when structure is weak), a
**lag/drift** value per 30 s window (cross-correlation of original vs rendered stem envelopes; flag
> 30 ms), **per-section level diff**, **SoulX segment-join continuity** (RMS and centroid step at each
join), and a per-stage **wall time and peak RSS** line (encode: separate / tsumugi / contour / lyrics;
render: streams / fx / vocal). Acceptance for long material is always "worst section ≥ threshold", not
the mean.

---

## 7. A per-part benchmark scorer (0–100)

Goal: one number per part (vocals, backing, keys, bass, drums, guitar, other, overall mix) and per section
that predicts "sounds like the original" by ear, to rank render variants automatically.

### 7.1 Metrics per part

Group every part's metrics into three axes; the part score is the weighted mean of the axes.

| Part | **What** (notes/timing) | **Sound** (timbre) | **Dynamics** (loud when) |
|---|---|---|---|
| Keys, guitar, other | note F1 at cover tolerance (±100 ms/±100 c; the current ±50/±50 punishes reverb), chroma per bar, onset F1 | `spectral_db` (keep, but see pitfalls), **MERT-v1-95M frame-embedding cosine per bar** (CC BY-NC 4.0) or **CLAP audio-audio cosine** (laion-clap CC0 / msclap MIT), 1/12-octave log-spectral distance per bar (new, finer than the 31-band match) | energy corr, per-section level diff, sounding-time fraction diff (sustain coverage) |
| Bass | onset F1, bass f0 agreement (torchcrepe fmin 30 Hz, median cents over frames voiced in both), chroma | `spectral_db`, MERT cosine | energy corr, level per section |
| Drums | onset F1; **per-voice onset F1** (run tsumugi `drums_v1_5` on the render as well; kick/snare/hat/clap separately) | `spectral_db`, MERT cosine; decay-length agreement per voice (new, cheap: median hit decay to −20 dB) | energy corr, per-bar level |
| Lead / backing vocal | pitch cents, note F1, onset F1, `sung_wer`, **word-timing MAE** (whisper word stamps on both sides) | voice_sim (resemblyzer), `spectral_db`, MERT cosine; naturalness: **SingMOS** (open singing-MOS predictor, *verify licence*) or **Audiobox Aesthetics** PQ (Meta; weights CC BY-NC 4.0) | energy corr, per-phrase level diff |
| Overall mix | chroma per bar, onset F1 on the mix, round-trip note F1 (re-encode the render and diff against the `.sc`, spec §10.3) | full-mix `spectral_db`, per-band stereo width diff, CLAP/MERT cosine, Audiobox PQ/CE, LUFS-I diff (pyloudnorm) | per-section LUFS diff, energy corr |

Not recommended: ViSQOL/PESQ-style intrusive metrics (need sample-aligned identical signals; a
re-performed part scores near the floor regardless of quality); single-pair FAD (statistically
meaningless, spec §10.4 — use FAD only across the whole benchmark set with fadtk, MIT, CLAP/MERT
embeddings).

### 7.2 Mapping to 0–100

For each metric m, fix two anchors per part type from the data we already have:
- **ceiling** `m_hi`: the metric between the original stem and a "perfect" render — the stem versus
  itself after a re-separation round trip (captures separation noise), and for vocals Seed-VC's
  self-conversion (10 c / 0.96, `singing-spike.md`) and `sung_wer` 0.09 on the original;
- **floor** `m_lo`: the metric against a clearly wrong render — the GM `--no-fx` mock, or the same
  stem from a *different* song.

Score = 100 × clamp((m − m_lo) / (m_hi − m_lo), 0, 1) (inverted for "lower is better"). Axis score =
mean of its metrics; part score = 0.4·What + 0.4·Sound + 0.2·Dynamics (start there; refit from the
ear ledger, §7.4). Per-section scores use the same formula on section slices; the song score reports
the mean **and the worst section**. Overall = energy-share-weighted mean of part scores (a part that is
−45 dB in the original barely counts) blended 50/50 with the mix axis. Print the table as
`part | what | sound | dyn | score | worst section`.

### 7.3 Pitfalls

- **Metrics the renderer optimises directly are not evidence.** `spectral_db` is exactly what the
  31-band EQ match minimises (`fx.py:140-157`), and the level diff is what level matching zeroes; both
  hit ~1 dB while a GM piano still sounds like a GM piano. Keep them as sanity checks, weight them
  low, and put the *sound* weight on embeddings (MERT/CLAP) and the finer 1/12-octave distance.
- **Chroma is gameable** by any pad holding the chord; onset F1 is gameable by dense hats. Always pair
  them with note F1 and per-voice drum F1.
- **Note F1 uses basic-pitch on both sides** (`compare.py:265-274`): a correct drone scores ~0
  (`compare-tsumugi.md`, corona); a reverberant render loses onsets at ±50 ms. Use cover tolerances and
  add the f0/chroma path for sustained parts.
- **Bleed in the reference**: River's bass stem is piano bleed; any metric against it is noise. Gate
  parts by reference stem level (the `—` rows already do this) and weight by energy share; for
  Discipline use the official multitracks as reference to see separation error separately.
- **Level/tempo misalignment on long songs**: a constant lag > 50 ms kills onset F1 and MR-spectral
  distances; measure the lag per window first (§6.2) and report it rather than silently aligning.
  Global energy corr is dominated by section loudness — compute it per section too.
- **Vocal metrics have seed noise** (±0.1 WER): score the mean of 2 seeds or the chosen take, and say
  which. Whisper-small mishears the original too (floor 0.09); report `sung_wer − floor`.
- **Embedding models have their own bias**: MERT/CLAP are near-insensitive to fine timing and can
  reward a generative restyle that drifted; they only count once the *What* axis passes.
- **Section boundaries are weak** (`stage_struct`): until beat_this/structure lands, use fixed 8-bar
  windows for per-section scores so the scorer is not hostage to `:struct`.

### 7.4 Validating the scorer against the user's ear

- Keep an **ear ledger** (`docs/results/ear-ledger.md`): every listening checkpoint already produces a
  verdict; record `(song, section, part, variant A, variant B, preferred, note)`. Existing entries to
  seed it: SoulX > DiffSinger→Seed-VC (River), unheld > held notes (River), fx+kit > no-fx (River,
  Discipline, pending), "some of that piano is working".
- Metric: **pairwise agreement** — the share of ledger pairs where the scorer ranks the preferred
  variant higher (and Kendall τ on any ranked sets). Trust the scorer for automatic take/variant
  selection only above ~80 % on ≥ 20 pairs; refit the axis weights by logistic regression on the
  pairs once there are ~40.
- **Sanity anchors** that must always hold: original vs itself = 100; original vs a different song's
  stem ≤ 10; the known regression from the 4 s-reverb bug (`production-match.md`, onset F1 0.85 → 0.02)
  must drop the drums/keys score by ≥ 30 points.
- Blind A/B with the listening server (`serve`): present the two variants unlabeled per section, log
  the click; that makes the ledger cheap to grow at every checkpoint.

---

## Appendix: facts checked, and what I could not check

- Checked in code: everything cited with `file:line` above.
- Measured here (read-only scripts in the scratchpad): velocity/duration/polyphony distributions of
  the three `.sc` files; kit sample lengths and tail levels; stem RMS and sounding fractions.
- Not run: any render, SoulX, tsumugi, or framepick job.
- Unverified claims (marked): pysfizz wheel on Python 3.11 arm64; TokenSynth/MIDI-VALLE/DiffSynth-Music
  VRAM on 8 GB; `acids-ircam/ddsp_pytorch` and SingMOS licences; tsumugi chunking behaviour on long
  files; whether the 0.22–0.24 s duration cluster is a tsumugi minimum-length artefact or performed
  eighths (Discipline bass plausibly is; River vocal at a constant 0.5 beat is suspicious).
