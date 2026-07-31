# soundcode — Design Spec (v0.3)

Date: 2026-07-31
Status: draft for review
Living format reference: `examples/signal-lost.v3.sc`
Prior art in-repo: `examples/signal-lost.sc` (v0.1), `examples/signal-lost.v2.sc` (v0.2)

---

## 1. What soundcode is

soundcode transcribes a real recorded song into a human- and LLM-readable code
representation (a `.sc` file), and then regenerates a new audio rendering from
that code that sounds as close to the original as possible.

The quality bar is an **excellent cover version**: same song, same words, same
groove, same arrangement, very similar production character, different
performance. "Almost the same." A listener should say *"that's clearly the same
song, produced the same way"* — not *"that's the same recording."*

### 1.1 Goals

- G1. A `.sc` text format that can hold what a song *is* — structure, harmony,
  notes, lyrics, groove, timbre, mix character — for any musical tradition
  (metric or free-time, 12-TET or microtonal, sung, rapped, or instrumental).
- G2. A fully local encoder pipeline (`audio → .sc`) built from existing MIR
  tools, that is **honest about its own uncertainty** in the file it writes.
- G3. A fully local decoder path (`.sc → audio`) using an LLM "decode bridge"
  plus an open music generation model, on an M1 Max 32 GB with no CUDA.
- G4. `.sc` is pleasant to read, edit, and diff — for humans in an editor and
  for LLMs in a context window. Editing the file and re-decoding is the
  intended creative loop.
- G5. An automated quality signal: round-trip self-consistency
  (`encode → decode → re-encode → compare`) plus perceptual similarity,
  measurable without human listening on every iteration.

### 1.2 Non-goals (explicit)

- **Compression is a non-goal.** `.sc` is not a codec. Bitrate, file size, and
  bits-per-second are never evaluation criteria and never appear in reports.
  (The v0.1 example's kbps footer was a mistake and is gone.)
- **Lossless / waveform reconstruction is a non-goal.** Exact audio identity is
  out of scope by design; the target is a faithful cover.
- **Real-time operation is a non-goal.** Encoding and decoding are batch jobs.
- **The learned acoustic-residual layer is deferred.** `%residual none` remains
  a forward-compatibility hook only. Nothing in this spec designs it.
- **Training models is a non-goal for v1.** We compose existing tools; LoRA
  fine-tuning of the decoder is a listed future option, not part of v1.

---

## 2. Design pressures found in adversarial review of v0.2

These drove every change in v0.3. Summarized here so the rationale survives.

1. **The decoder can't eat most of v0.2's precision — unless we make it.**
   ACE-Step's conditioning surface is: a natural-language caption/tags, a
   lyrics block with `[Verse]`/`[Chorus]`-style markers, duration, and (in
   ACE-Step 1.5) BPM / key / time-signature metadata — plus audio-reference
   modes. There is **no note-level, MIDI, or melody-token input** in either
   version (verified, §7.1). So `dev+22ms`, cents, envelopes, and contour data
   are unreachable through text conditioning. The only route by which
   note-level `.sc` content reaches the decoder is **audio**: render the `.sc`
   symbolically to a rough "mock" and use it as the source for ACE-Step's
   remix/audio2audio mode. This makes the mock renderer a first-class decoder
   component, not a test utility — and it retroactively justifies keeping
   note-level detail in the format.
2. **Silent wrongness.** Every v0.2 value is asserted flatly, but the encoder
   stack is fallible in known, measurable ways (chords ~80% on a good day,
   sung-lyric alignment noticeably worse than speech, structure boundaries
   ±1 bar). A representation that states a wrong chord confidently poisons the
   bridge, the mock render, and every downstream edit. v0.3 makes confidence
   and provenance first-class syntax (§4.4).
3. **Timbre prose was fiction.** `~"distorted saw, lowpass ~400Hz"` is
   authorable by a human, not emittable by an encoder. v0.3 splits timbre into
   (a) a controlled `inst=` class, (b) measured `timbre{...}` scalars,
   (c) classifier `tags{...}` with scores, and (d) the freeform gloss, which is
   explicitly *authored* (by a human or the encode-time LLM pass) (§4.3.6).
4. **Patterns pretended to be exact.** Real performances never repeat exactly;
   inferred `%pat` bindings and `like` reuse are lossy factorizations. v0.3
   keeps them (they are what makes a 4-minute file readable and
   token-affordable) but requires a `~0.NN` similarity annotation whenever the
   binding discards per-instance nuance (§4.5). The encoder always produces
   explicit events first; factoring is a separate, optional, honest pass.
5. **Things that only break at 4 minutes.** Beat-tracker drift accumulating
   over hundreds of bars (fix: repeated `anchor` lines); sections that repeat
   *with variation* (fix: per-section character in `:struct` + lossy `like`);
   dynamic arc invisible in a global LUFS number (fix: `:mix by_section`);
   file length drowning the LLM bridge (fix: factoring + section-scoped
   reading); Whisper hallucinating lyrics in long instrumental stretches
   (fix: `:struct` vocal/inst flags gate the lyric stream).
6. **Grammar sloppiness.** `|` meant "event separator" in note streams and
   "field separator" in `:mix`. v0.3: `|` separates items on a line, full
   stop; multi-field scalar lines use commas. Quoted strings may span lines.
   `@offset` pins the file's t=0 to the source audio.

---

## 3. Environment pins (verified 2026-07)

These are constraints, not suggestions. Sources and verification notes in §12.

| Concern | Pin | Why |
|---|---|---|
| Python | **3.11.x venv** (system 3.14 unusable) | basic-pitch has no 3.12 support (open PR); WhisperX requires >=3.10,<3.14; Essentia arm64 wheels pin to `2.1b6.dev1389` (cp310–cp313) |
| Torch device | CPU for Demucs/WhisperX; MPS where it works; MLX for ACE-Step 1.5 LM | Demucs MPS is verified broken (complex FFT ops); CTranslate2 (WhisperX) has no MPS |
| Separation | `demucs` 4.1.0 from the maintained `adefossez/demucs` fork (PyPI), models `htdemucs` / `htdemucs_ft` | facebookresearch repo archived 2025-01 |
| Beats/downbeats | `beat-this` (PyPI 1.1.0, CPJKU) | madmom PyPI is a 2018 sdist that won't build on py>=3.10 |
| Chords | madmom **git master** CNN+CRF; fallback Essentia `ChordsDetection` | quality vs. install friction tradeoff, both recorded |
| Structure | `allin1` (via `all-in-one-fix`) if it installs; fallback beat_this + novelty segmentation | NATTEN/madmom dep stack is fragile, no MPS |
| Note transcription | `basic-pitch` per-stem; `YourMT3+` optional for multi-instrument | MT3 proper is unrunnable locally (dead T5X/JAX stack) |
| F0 | `torchcrepe` (f0 + periodicity = built-in confidence); `librosa.pyin` baseline | TF-CREPE drags the TF-on-arm64 headache in |
| Lyrics | `whisperx` (3.8.x), `--device cpu --compute_type int8`, run on the **separated vocal stem** | no MPS; sung-vocal alignment degrades (unbenchmarked upstream) — confidence markers required |
| Mix features | `librosa` + `essentia==2.1b6.dev1389` | wheel availability |
| Decoder | **ACE-Step 1.5** (MIT), 2B-turbo DiT + 0.6B LM on Mac; ACE-Step v1 3.5B as alternate | §7 |
| Bridge LLM | ollama `qwen3.5:27b` (quality) / `qwen3.5:9b` (fast), JSON-schema `format`, non-streaming | 32 GB fit; strongest structured-output family |
| Eval | `mir_eval` 0.8.2, `museval` 0.4.1, `msclap` (CLAP), `fadtk` (corpus-level FAD only) | §10 |

---

## 4. The `.sc` v0.3 data model

### 4.1 Invariant design rules (carried from v0.2, unchanged)

1. **Time**: absolute seconds are canonical (`@7.500`). `bar:beat` is sugar
   resolved through the optional `:grid` stream. Free-time music omits `:grid`
   and writes `@seconds` throughout.
2. **Pitch**: integer cents from the declared reference are canonical
   (`5708c`). Note names (`A3`, `A3+8c`) are sugar resolved through the
   optional `:tuning` stream. Raw frequency (`~220.0Hz`) always valid.
   Microtonal, maqam, gamelan, scoops: representable without lying.
3. **Streams are optional, namespaced, and independently versioned.** A song
   is whatever subset of streams applies. Unknown stream namespaces must be
   preserved by parsers (parse → hold as opaque lines → re-emit), so new
   stream types never invalidate old tooling.
4. Both overlays are pure sugar: the parser normalizes to seconds + cents and
   re-emits the preferred notation on write. Text → model → text round-trips
   byte-stable when nothing changed.

### 4.2 File anatomy

```
pragmas      %sc, %profile, %residual
header       @key value        (title/artist/source/offset/license/duration/sr,
                                @style, @mix prose)
streams      :grid :tuning :struct :harmony
             :perc.<trk> :notes.<trk> :text.<trk> :contour.<trk>
             :texture.<trk> :mix
```

Header directives of note:

- `@offset <secs>` — **new in v0.3, always present.** Where this file's t=0
  sits in the source audio (leading silence, excerpt start). Removes ambiguity
  when comparing encodes of the same recording trimmed differently.
- `@style`, `@mix` — prose consumed by the decode bridge. On encoded files
  these are **written by the encode-time LLM describe pass** from measured
  evidence (`timbre{}` blocks, `:mix` scalars) and are freely human-editable.
  They are presentation; the measured evidence is the source of truth.

### 4.3 Stream reference

Every stream may begin with `meta` lines (§4.4). "Omitted when" is normative:
an encoder must not emit a stream it has no evidence for.

#### 4.3.1 `:grid` — pulse
- `meter @t N/D` (repeatable; mid-song meter changes are more meter lines)
- `anchor bar N @t` (repeatable) — pins a bar line to absolute seconds.
  **Encoded files must anchor at least every 16 bars and at every `:struct`
  boundary** so beat-tracking drift cannot accumulate on long songs.
- `tempo @t BPM` (repeatable) — tempo is a curve; points interpolate linearly.
- Omitted when: no perceivable pulse (rubato solo, ambient, field recording).
  Everything downstream then uses `@seconds` and second-valued durations.

#### 4.3.2 `:tuning` — pitch-name overlay
- `ref <name> = <freq>Hz`; `temperament 12tet` or an explicit cent table
  (`degrees 0c 204c 355c ...` for maqam/gamelan/historical).
- Omitted when: unpitched/atonal material; then only raw cents/Hz appear.

#### 4.3.3 `:struct` — sections, decoder-facing
- Entry: `<label> <barrange|@t-@t> <vocal|inst> energy=<0-1> [desc="..."]`
- `vocal`/`inst` flag and `energy` (section LUFS relative to song max) are
  **required on encoded files** — the bridge turns them into the decoder's
  lyric section tags and `[inst]` markers, and they gate WhisperX output
  (no lyrics emitted inside `inst` sections → kills hallucinated words).
- `desc` optional, bridge-authored, one line.
- Omitted when: never, in practice — a single `all` section is legal minimum.

#### 4.3.4 `:harmony`
- `tonal_center <pc> <mode>`; bar-bound chord lists
  `bars 1-4 Am | F | C | G`; `like`/`x` reuse.
- Chords carry doubt inline: `F ?0.58 alt=Dm`.
- Cross-check contract: the encoder compares chord roots against
  `:notes.bass`; disagreement is recorded as
  `meta warn="bar 22: bass implies Dm, chord model says F (0.58)"` — never
  silently resolved.
- Omitted when: non-harmonic music (percussion works, noise, some drone).

#### 4.3.5 `:perc.<trk>` / `:notes.<trk>` — event streams
- Event: `<time> <atom> [dur] [vel] [ann...]`
  - atom: pitch (notes), voice name (`kick`, `snare`, `tom1`… for perc),
    `.` = current chord root resolved from `:harmony`.
  - dur: `0.5b` (beats, needs grid) or `0.234s` (always valid).
  - vel: 0–127, mapped from stem-relative level at encode time.
  - annotations: `dev±Nms` (micro-timing vs grid — the groove),
    `env{a …, d …, s …, r …}`, `?0.NN`, `alt=<atom>`.
- Omitted when: the track has no discrete onsets (use `:texture.<trk>`).

#### 4.3.6 Track declarations (applies to `:perc/:notes/:texture`)
```
:notes.bass  inst=bass.synth
  timbre{ centroid 410Hz, rolloff_95 1.9kHz, attack 6ms, harmonicity 0.71,
          dist 0.64, stereo 0.03 }
  tags{ "synth bass" 0.83, "electric bass" 0.11 }
  ~"distorted saw, lowpass ~400Hz, slight pitch drift, mono"
```
- `inst=` from a small controlled vocabulary with dotted refinement:
  `drums.kit drums.machine perc.hand bass.electric bass.synth bass.upright
  gtr.clean gtr.crunch gtr.acoustic keys.piano keys.ep keys.organ synth.lead
  synth.pad strings brass winds voice.lead voice.backing voice.speech
  fx other unknown`. `unknown` is legal and expected — when classification
  confidence is low the encoder writes `inst=unknown ?0.4` plus the measured
  evidence and lets a human or the bridge decide.
- `timbre{}`: measured scalars (spectral centroid, rolloff, attack, stereo
  width, distortion estimate, etc. — open key set, units mandatory).
- `tags{}`: audio-tagging classifier output with scores (zero-shot CLAP over
  the instrument vocabulary in v1).
- `~"gloss"`: freeform prose. Authored, not evidence.
- Identification pipeline: Demucs stem name seeds the class (vocals/drums/bass
  are near-free); the `other` stem is sub-clustered and CLAP-classified; when
  scores are flat, `inst=unknown` + evidence. Nothing here is ever guessed
  silently.

#### 4.3.7 `:text.<trk>` — lyrics/speech
- Declaration: `:text.vox lang=en align=word|syllable`.
  `align=word` is what forced alignment actually produces and is the encoder
  default; `align=syllable` is authoring-grade (the control track uses it).
- Event: `<time> "token"` with optional `?0.NN alt="…"` (ASR doubt).
- Rap/spoken word: keep `:text`, drop `:notes` for that track — prosody can
  live in `:contour` without scalar pitch.
- Omitted when: instrumental music (and inside `inst` sections).

#### 4.3.8 `:contour.<trk>` — continuous F0 behaviour
- Sparse verb events at times where pitch departs from the notated value:
  `vib{rate 5.4Hz, depth ±31c, onset 0.22s}`, `scoop{from -68c, dur 90ms}`,
  `fall{to -140c, dur 380ms}`, `bend{to +180c, dur 220ms}`,
  `gliss{to <pitch>, dur …}`. Verb set is open; parsers preserve unknown verbs.
- Source: torchcrepe F0 + periodicity over the vocal/lead stem; events emitted
  only where |F0 − notated| exceeds threshold for >40ms. Periodicity below
  floor ⇒ no claim (not a `?`-marked guess — silence).
- Omitted when: nothing departs from notated pitch (most keyboard music).

#### 4.3.9 `:texture.<trk>` — onset-free material
- Band-energy trajectories + statistical descriptors for drones, ambience,
  noise beds: `bands @t low L, mid M, high H` rows plus `timbre{}`.
  Deliberately coarse in v1; exists so such tracks are *representable*, with
  fidelity carried by the decoder's style conditioning.

#### 4.3.10 `:mix` — production character
- Scalars (comma-separated fields): `lufs_int`, `lufs_range`,
  `stereo_width low …, mid …, high …`, `rt60 …`, `comp_est …`,
  `by_section intro -12.1, verse -9.8, chorus -7.6` (**new in v0.3** — the
  dynamic arc a 4-minute song lives or dies by).
- `!auto <target> @t v -> @t v` automation ramps. Consumed by the mock
  renderer (Path B, §7.3) and by the bridge as prose hints; targets are
  dotted names (`pad.cutoff`, `vox.reverb_send`, `master.drive`), open set.

### 4.4 Uncertainty, confidence, provenance

Three mechanisms, smallest sufficient set:

1. **Stream `meta` lines** (first lines of a stream):
   - `meta src=<tool>:<model> [stem=<name>] conf=<0-1> [conf_floor=<0-1>]`
   - `meta warn="<free text>"` — encoder-detected conflicts and anomalies.
   - `src=authored conf=1.00` for human-written files.
2. **Value-level doubt**: any value may be followed by `?0.NN` (confidence)
   and `alt=<value>` (runner-up hypothesis).
   **Silence is a claim**: an unmarked value asserts conf ≥ the stream's
   `conf_floor` (default 0.80). Encoders must mark, not round up.
3. **Binding similarity**: `~0.NN` on pattern/`like` bindings (§4.5).

What deliberately does *not* go in the file: per-stage tool versions, run
timestamps, full posteriors. Those live in the work directory
(`<name>.scw/provenance.json`, §8.3). The `.sc` carries what a reader needs to
calibrate trust; the sidecar carries what a debugger needs.

Confidence sources per stage: torchcrepe periodicity (F0), basic-pitch
posteriors (notes), WhisperX word scores (lyrics), chord model marginals
(harmony), beat_this activation strength (grid), CLAP score margins (inst
tags). Stages without native confidence (Demucs) get stream-level conf from
proxy checks (§5).

### 4.5 Patterns and reuse — honest factoring

- `%pat name { events }` defines a pattern scoped to the current stream;
  `bars A-B name` binds it. `bars A-B like C-D [xN] [transpose N] [vel ±N]`
  reuses an earlier range.
- **Bare binding = exact**: expanding it reproduces the events byte-for-byte
  (after transpose/vel modifiers).
- **`~0.NN` binding = lossy**: the factoring pass merged near-identical bars;
  per-instance deviation was discarded; `0.NN` is the mean normalized
  similarity of the merged instances. The pattern body keeps the **median**
  `dev`/vel per hit, so the groove survives factoring even though per-bar
  spread does not.
- Pipeline contract: **the encoder emits explicit events only.** `sc compact`
  (a deterministic post-pass, §8) introduces `%pat`/`like` with tolerances
  (onset ±25ms, pitch ±50c, vel ±12; below similarity 0.90 it refuses to
  factor and leaves events explicit). `sc expand` inverts exactly for bare
  bindings and to the stored median form for `~` bindings. Decode and eval
  always run on expanded form, so factoring can never change what the decoder
  hears — only what the reader reads.
- Default: `soundcode encode` runs compact automatically for audio >90s
  (readability and bridge token budget), off below that. `--no-compact`
  always available. (Open question Q3.)

### 4.6 Grammar (normative sketch)

Line-oriented; a physical line is one of the forms below. `#` starts a
comment only as the first non-whitespace character. Indented lines following
a `:` declaration continue that declaration. A quoted string may span lines;
continuation is joined with single spaces. Inside `%pat { … }` braces,
newlines and `|` both separate events. `|` separates items on a line —
nothing else, anywhere. Blocks `name{ k v, k v, … }` use commas.

```ebnf
file        = { line } ;
line        = pragma | header | stream_decl | meta | patdef | binding
            | statement | event_line | auto | comment | blank ;

pragma      = "%sc" version | "%profile" name | "%residual" spec ;
header      = "@" key ( string | value ) ;

stream_decl = ":" dotted_name { field } [ gloss ]
              { INDENT ( block | field | gloss ) } ;
field       = key "=" value [ doubt ] ;
block       = name "{" kv { "," kv } "}" ;
gloss       = "~" string ;
meta        = "meta" { key "=" value } | "meta" "warn" "=" string ;

patdef      = "%pat" name "{" event { ( "|" | NEWLINE ) event } "}" ;
binding     = "bars" range ( patname | chords | "like" range { mod } )
              [ "~" float ] ;
chords      = chord { "|" chord } ;            (* :harmony only *)
mod         = "x" int | "transpose" int | "vel" signed ;

event_line  = event { "|" event } ;
event       = time [ atom [ dur ] [ vel ] | string ] { ann } ;
              (* atom-less events are legal where the stream schema says so:
                 :contour events are time + verb blocks only *)
time        = bar ":" beat | "@" secs | beat ;  (* bare beat inside %pat *)
atom        = pitch | voicename | "." ;
pitch       = notename [ cents ] | int "c" | "~" float "Hz" ;
dur         = num "b" | num "s" ;
vel         = int ;                              (* 0..127 *)
ann         = "dev" signed "ms" | doubt | "alt" "=" value | block ;
doubt       = "?" float ;

statement   = word { valuegroup { "," valuegroup } } ;
              (* stream-schema statements: meter/anchor/tempo/ref/
                 temperament/tonal_center/struct entries/:mix scalars *)
auto        = "!auto" dotted_name time num "->" time num ;
```

The grammar is two-level by intent: this lexical layer is generic; each
stream namespace defines which statements/atoms are meaningful (schema layer).
Parsers must preserve and re-emit lines they lex but don't understand,
attached to their stream — that is what makes stream versioning safe.

`examples/signal-lost.v3.sc` is the golden file for this grammar; the parser
milestone (M1) includes byte-stable re-emission of it.

---

## 5. Encoder pipeline

`soundcode encode song.wav` runs a DAG of stages; every stage writes JSON to
the work dir (§8.3), consumes upstream JSON, and declares confidence. Any
stage may fail soft: its stream is omitted and a `meta warn` recorded in
adjacent streams where relevant. Order below is topological.

| # | Stage | Tool (pinned) | Input → Output | Confidence source | Failure mode & fallback |
|---|---|---|---|---|---|
| 1 | ingest | ffmpeg | any audio → 44.1k stereo wav, `@offset` from silence trim | n/a | unreadable input → hard fail |
| 2 | separate | demucs 4.1.0 `htdemucs_ft` (CPU, `--segment` for RAM) | mix → vocals/drums/bass/other stems | proxy: leakage estimate (stem cross-correlation); no native conf | heavily mono/lo-fi sources separate poorly → stream-level conf drops; worst case encode from the mix with `meta warn` |
| 3 | grid | beat_this 1.1.0 (CPU) | mix → beats, downbeats → tempo curve, anchors every ≤16 bars | activation strength | no stable pulse detected → omit `:grid` entirely (free-time path), everything downstream in `@seconds` |
| 4 | structure | allin1 (`all-in-one-fix`), CPU | mix (+stems) → section boundaries/labels/energy | model conf | **fragile install (NATTEN/madmom). Plan B (must ship): beat_this downbeats + librosa novelty/recurrence segmentation + energy profile; labels then come from the bridge LLM reading per-section stats.** boundaries snap to nearest downbeat when `:grid` exists |
| 5 | harmony | madmom git CNN+CRF; fallback Essentia `ChordsDetection` | mix → per-beat chords → per-bar with `?`/`alt=` | CRF marginals / template distance | madmom build failure → Essentia, stream `conf` capped at 0.7; key/mode via Essentia `KeyExtractor` |
| 6 | notes | basic-pitch per melodic stem (bass, other; vocals via 6b) | stem → note events (onset, pitch+bend, vel from stem RMS) | model posteriors | dense polyphony in `other` → keep only notes above posterior floor, mark the rest as coverage gap in stream conf; optional YourMT3+ pass behind a flag |
| 6b | vocal notes + contour | torchcrepe on vocal stem (+pyin cross-check); note segmentation from F0 + onsets | stem → `:notes.vox` + `:contour.vox` | periodicity | low periodicity (screams, whispers) → `:notes.vox` omitted, `:text` + `:contour` only |
| 6c | drums | onset detect + spectral template match on drum stem (kick/snare/hat/tom/crash/ride); `dev` = onset − nearest grid line | stem → `:perc.drums` | template margin | electronic/nonstandard kits → voices fall back to `perc1..N` names with `timbre{}` evidence |
| 7 | lyrics | whisperx 3.8.x CPU int8 **on the vocal stem**, gated by `:struct` vocal flags | stem → word events with scores | WhisperX word conf | melisma/sustained notes misalign (known, unbenchmarked upstream) → words get `?`; whole-section garbage (avg conf < 0.4) → drop section's words + `meta warn` |
| 8 | timbre/inst | librosa + essentia scalars per stem/cluster; zero-shot CLAP (msclap) over inst vocabulary | stems → `inst=`, `timbre{}`, `tags{}` | CLAP margin | flat scores → `inst=unknown` + evidence |
| 9 | mixfeat | librosa/essentia + pyloudnorm-class measures | mix + stems → `:mix` scalars, `by_section`, coarse `!auto` from tracked band envelopes | n/a (measurements) | none expected |
| 10 | cross-check | internal | bass roots vs chords; vocal notes vs key; struct vs energy | n/a | disagreements → `meta warn` lines, confidence demotions; never silent resolution |
| 11 | describe | bridge LLM (§6) | all streams → `@style`, `@mix` prose, `:struct desc`, track glosses | n/a (authored) | LLM unavailable → template prose from `tags{}`/`timbre{}`, marked `src=template` |
| 12 | emit + compact | scfmt | model → `.sc` (explicit) → factored `.sc` | n/a | compact refuses below similarity floor (§4.5) |

Poison-propagation notes (why the DAG is shaped this way):
- **Grid is the most dangerous stage** — a half-tempo or off-by-one-downbeat
  error corrupts every `bar:beat`, every `dev`, and the factoring pass.
  Mitigations: tempo curve + anchors make errors local; a grid sanity check
  (median onset deviation of drum hits vs grid > 40ms ⇒ demote grid conf and
  re-run beat_this with halved/doubled prior); worst case, drop to `@seconds`.
- **Separation quality bounds stages 6–8**; the leakage proxy conf is written
  so downstream readers know whether "pad" notes might be bleed.
- **Structure gates lyrics** (no ASR inside `inst` sections), which is the
  main defense against Whisper hallucination on long instrumentals.

---

## 6. The LLM decode bridge

The bridge reads a `.sc` (expanded form) and emits a **conditioning bundle**
for the decoder. It generates no audio.

- Runtime: ollama, non-streaming, with a JSON schema passed via the `format`
  parameter (grammar-constrained decoding — never trust freeform output).
- Models: `qwen3.5:27b` (~17 GB, default), `qwen3.5:9b` (~6.6 GB,
  `--fast`, leaves RAM headroom to run alongside the audio stack). Both fit
  32 GB; keep context modest on the 27b. (Verified availability/fit; JSON
  reputation is family track record, not a benchmark.)
- Input: the `.sc` header + `:struct` + `:harmony` + `:text.*` + track
  declarations + `:mix` — i.e. the bridge does *not* need note events, which
  keeps even 4-minute songs inside a modest context.
- Output schema (versioned, `bridge/schema.py`):

```json
{
  "caption": "…natural-language music caption…",
  "bpm": 128, "key_scale": "A minor", "time_signature": "4",
  "duration_s": 30.0,
  "vocal_language": "en", "instrumental": false,
  "lyrics": "[Verse 1]\nI was a signal…\n\n[Chorus]\nHold the line…",
  "negative": "…optional…",
  "section_map": [ {"label": "intro", "start": 0.0, "end": 7.5,
                    "lyric_tag": "[inst]"}, … ],
  "notes_for_human": ["…anything the bridge was unsure about…"]
}
```

- Deterministic fields (`bpm` from the `:grid` tempo median, `key_scale` from
  `:harmony tonal_center`, `time_signature` from `meter`, `duration_s`,
  `section_map` from `:struct`, lyric text/ordering from `:text.*`) are
  computed by *code* and injected — the LLM writes only the caption, lyric
  formatting, negative prompt, and prose. LLMs are not allowed to be the
  source of truth for numbers the file already contains.
- Confidence-aware captioning: the bridge is prompted to hedge or omit claims
  whose stream conf is low (e.g., don't assert "in F major" over
  `?0.55`-grade harmony), and to fold `meta warn` content into
  `notes_for_human`.
- The same bridge (different schema) powers encode stage 11 (describe) and
  the human-facing round-trip report summary.

---

## 7. Decoder

### 7.1 Verified ACE-Step facts (2026-07)

**ACE-Step v1 (3.5B)** — repo `ace-step/ACE-Step`:
- Conditioning: comma-separated tags/descriptive text; lyrics with
  `[verse]`/`[chorus]`/`[bridge]` markers; duration (≤ ~4 min, `-1` random);
  seed; guidance scale; infer steps (27/60 benchmarks); scheduler.
- Modes: text2music, **audio2audio with `ref_audio_strength` 0–1 (default
  0.5)**, retake (variance), repaint (time segment), edit (`only_lyrics` /
  `remix`), extend. LoRAs: chinese-rap, Lyric2Vocal, Text2Samples.
- Mac: tested on M2 Max, `--bf16 false` required; **RTF 1.03× at 60 steps /
  2.27× at 27 steps on M2 Max** (≈ 1 min audio per minute). ~8 GB min VRAM
  path via `--cpu_offload`. Checkpoint auto-downloads (~several GB;
  exact size not published — unverified).
- **Absent: any melody/MIDI/note conditioning, any BPM/key parameter, any
  control over where sections fall in time.**

**ACE-Step 1.5** — repo `ace-step/ACE-Step-1.5`, MIT, released 2026-01-28:
- Architecture: Qwen3-based LM (0.6B / 1.7B / 4B) generating 5 Hz semantic
  codes + DiT (2B turbo, ~4.7 GB bf16; 4B XL, ~9 GB) + VAE.
- Conditioning: "Music Caption" natural-language field; lyrics with
  `[Verse 1]`/`[Chorus]` tags; **Instrumental checkbox**; vocal language;
  **duration 10–600 s**; **BPM 30–300**; **Key Scale** (e.g. "Am");
  **Time Signature** (2/3/4/6). LM "thinking" mode expands tags/metadata.
- Modes: Simple, Custom (text2music), **Remix — "transform existing audio
  while maintaining its melodic structure", strength 0–1 (higher = closer to
  original structure)**, Repaint (time segment), Extract, Lego (add a track),
  Complete; reference-audio field for style/timbre guidance; "Convert to
  Codes" (5 Hz semantic codes from source audio) + Transcribe; **Auto LRC**
  (lyric timestamps on output — useful for our alignment verification).
- Inference params: steps default 8 (turbo 1–20, base 1–200), CFG 7.0 (base
  only), seed(s), shift, ode/sde, INT8 auto-quant (<20 GB GPUs), LM
  temperature/CFG/top-p, negative prompt.
- API: `uv run acestep` (gradio, :7860), `uv run acestep-api` (REST, :8001),
  `cli.py`; macOS start scripts provided; Mac backend = **MLX for the LM,
  MPS for the DiT**.
- Mac fit (community-verified, not official): on M1 Max 32 GB use the
  **0.6B LM** — the 1.7B LM under MLX peaked ~42 GiB and crashed; 2B-turbo
  DiT with INT8 ≈ ≤6 GB. **No official Apple-Silicon performance table
  exists — unverified; community reports ≈10–20× slower than A100.**
  Still section-repaint-friendly because turbo needs only 8 steps.
- Precise section timing control: **absent** in both versions. Sections
  emerge from the lyric tags; only Repaint gives explicit time-region
  control. Remix inherits timing from the source audio.

### 7.2 Decoder selection

**Primary: ACE-Step 1.5, 2B-turbo DiT + 0.6B LM, via its REST API.**
It is the only verified local option that accepts BPM/key/time-signature
directly, has a melody-preserving Remix mode, and has a Mac-native path.

Fallbacks, in order:
1. **ACE-Step v1 3.5B** — better documented, known-working M-series numbers
   (RTF ≈1× at 60 steps); loses metadata conditioning; audio2audio replaces
   Remix. Use if 1.5's Mac port proves unstable.
2. **MusicGen (audiocraft) `melody` variant** — chromagram melody
   conditioning, instrumental only, ~30 s windows; the fallback for
   melodic-fidelity experiments on instrumental material if both ACE-Step
   paths fail on MPS. (Interface detail deferred until/unless needed.)
3. **Stable Audio Open** — textures/ambience only; not a song decoder.
4. **YuE** — rented-CUDA experiment only; out of local scope.

### 7.3 Two decode paths

**Path A — text-only bridge (baseline).** Bundle → Custom/text2music:
caption, lyrics (+`[inst]` markers per `section_map`), BPM, key, time
signature, duration, seed. *Known ceiling:* melody, exact groove, and
section placement are uncontrolled; this path yields "same words, same
style, same tempo — different tune." Path A exists as the floor and as the
ablation baseline.

**Path B — mock-render remix (fidelity path, primary).**
1. `soundcode render song.sc → mock.wav`: a deterministic, MIDI-quality
   symbolic rendering (simple synths per `inst` class, drum samples, sung
   syllables as formant-ish placeholders or plain vocal-less lead line —
   v1 keeps this deliberately crude; it must be *timed and pitched* right,
   not pretty). Honors `:grid`, notes with cents, `dev`, `:mix by_section`
   energy, `!auto` ramps it can map.
2. ACE-Step 1.5 **Remix** with source = `mock.wav`, strength ≈ 0.6–0.8,
   plus the full Path-A conditioning bundle.
3. Result inherits melody/groove/section timing from the mock, and timbre/
   production from the caption. This is the route by which every note-level
   `.sc` detail reaches the output.

Risk R1 (§11): Remix behavior on synthetic mock input is unbenchmarked;
strength sweep is an explicit milestone (M8).

**Alignment verification loop (both paths).** After generation:
re-run stages 3+4 (+7 if vocal) on the output; compare section boundaries,
BPM, and lyric timing (Auto LRC cross-check) against `section_map`. If a
section boundary deviates > 2 s or a section's lyrics are misplaced:
Path B → Repaint that time segment (turbo, 8 steps, cheap); Path A → retake
with new seed (bounded retries, default 3), keep the best-scoring take by
the §10.4 composite.

### 7.4 Long songs (4 min)

- ACE-Step 1.5 generates up to 600 s in one call — chunking is not
  structurally required. Practical Mac-throughput and coherence may still
  favor section-wise work; the supported unit is **generate full-length
  once, then Repaint per flawed section** (boundaries from `:struct`).
- Path B scales naturally: the mock is full-length and carries the timing.
- Encode side at 4 min: anchors every ≤16 bars (drift), compact pass on
  (file size/readability), bridge reads structure-level streams only
  (context budget) — all specified above.

---

## 8. CLI

### 8.1 Commands

```
soundcode <song.wav>                  # encode → decode → report (the demo path)
soundcode encode <audio> [-o song.sc] [--no-compact] [--stages LIST]
                         [--fast] [--keep-work]
soundcode decode <song.sc> [-o cover.wav] [--path remix|text] [--strength F]
                         [--seed N] [--retries N] [--section LABEL]
soundcode render <song.sc> [-o mock.wav]        # deterministic symbolic mock
soundcode diff   <a.sc> <b.sc> [--json]         # stream-by-stream scorecard
soundcode eval roundtrip <audio> [--report out.json]
soundcode eval transcription <audio> <truth.midi>
soundcode eval separation <mix.wav> --stems <dir>
soundcode expand|compact <song.sc>              # factoring, both directions
soundcode inspect <song.sc> [stream] [--expand]
```

Defaults: `--path remix`, `--strength 0.7`, compact auto-on >90 s,
bridge model `qwen3.5:27b` (`--fast` ⇒ 9b + htdemucs non-ft + turbo steps).

### 8.2 Exit contract

`encode` succeeds if it emits a parseable `.sc` with ≥ `:struct` and `:mix`;
every omitted stream is listed on stderr with its reason (mirrors of the
`meta warn` records). `decode` fails hard only if the decoder API is
unreachable; alignment-verification failures after retries produce output +
a nonzero "quality" exit code + the report.

### 8.3 Work directory

`<name>.scw/` (kept with `--keep-work`, always kept on failure):
`ingest.wav`, `stems/*.wav`, `features/*.json` (one per stage, schema-
versioned), `provenance.json` (tool versions, run times, full confidences),
`bridge/bundle.json`, `mock.wav`, `gen/take-*.wav` + per-take scores,
`report.json`.

---

## 9. Module boundaries

Sized so each unit is independently testable; no module reaches across a
boundary except through the typed artifacts named above (SongModel,
stage JSON, bundle JSON).

```
soundcode/
  cli.py               # arg parsing, orchestration only
  scfmt/               # THE FORMAT — zero audio deps, pure text↔model
    model.py           #   SongModel dataclasses (canonical secs+cents)
    parse.py  emit.py  #   text → model → text (byte-stable golden tests)
    resolve.py         #   grid/tuning sugar resolution both directions
    factor.py          #   compact/expand with similarity accounting
    diff.py            #   stream-by-stream scorecard (§10.3)
  encode/
    pipeline.py        #   DAG runner, soft-fail + provenance
    separate.py grid.py structure.py harmony.py notes.py drums.py
    vocal.py timbre.py mixfeat.py crosscheck.py
  bridge/
    schema.py prompt.py ollama.py     # bundle schema, prompting, client
    describe.py                       # encode-time prose (stage 11)
  decode/
    acestep.py         #   REST client for ACE-Step 1.5 (+ v1 shim)
    plan.py            #   path A/B planning, section_map handling
    verify.py          #   post-gen alignment loop, retake/repaint policy
  rendermock/
    synth.py drums.py voice.py mixdown.py      # Path B mock renderer
  evalx/
    transcription.py separation.py roundtrip.py perceptual.py report.py
tests/                 # unit per module; golden/ holds .sc fixtures
examples/              # signal-lost.v3.sc is the living reference
docs/superpowers/specs/
```

Test seams worth naming: `scfmt` has no audio dependencies at all (fast CI);
every `encode/*` stage is testable from a fixture JSON + short fixture wav;
`decode/verify.py` is testable against synthetic misaligned audio without
running ACE-Step; `rendermock` output is itself a corpus item (Control).

---

## 10. Evaluation

Four legs; the first three are fully automated.

### 10.1 Transcription accuracy (Mini corpus: MAESTRO v3, 30 s)
- Ground truth: Disklavier-captured MIDI, ~3 ms alignment (verified).
- Metric: `mir_eval.transcription` note F1 — standard tolerances 50 ms onset
  / 50 cents; report onset-only and onset+offset variants, plus velocity-
  aware F1.
- Provisional target: onset-only F1 ≥ 0.85 on solo piano (basic-pitch on
  clean solo instrument; flagged provisional, calibrate at M4).

### 10.2 Separation sanity (Real corpus: NIN "Discipline", 30 s)
- Ground truth: official multitracks (Internet Archive mirror of the
  official release, CC BY-NC-SA 3.0 — verified live), bounced to the
  Demucs 4-stem taxonomy.
- Metric: `museval` BSSEval SDR per stem (SI-SDR reported alongside, never
  mixed into the same comparison).
- Purpose: bounds-checking stage 2 so stem-derived confidences are
  calibrated — not a leaderboard.

### 10.3 Round-trip self-consistency (all corpus items) — the main signal
Design: `encode(A) → sc1`; `decode(sc1) → B`; `encode(B) → sc2`;
`soundcode diff sc1 sc2` on **expanded** form, after a single global time
offset alignment (search ±2 s maximizing struct overlap; both files carry
`@offset`, residual skew from the decoder is expected).

Per-stream scores (each in [0,1], reported as a vector — never only a scalar):

| Stream | Score |
|---|---|
| `:struct` | boundary F1 @ ±2 s + label agreement (label match via bridge-normalized vocabulary) |
| `:grid` | tempo-curve MAPE sampled at 1 s; downbeat phase agreement |
| `:harmony` | `mir_eval.chord` weighted accuracy sampled per beat |
| `:notes.*` | note F1 at **cover tolerances**: onset ±100 ms, pitch ±100 cents (a cover, not a transcription check) |
| `:perc.*` | onset F1 per voice class @ ±50 ms |
| `:text.*` | WER + aligned word-timing MAE |
| track decls | `inst` class agreement; `timbre{}` scalar relative deltas |
| `:mix` | LUFS/width/`by_section` deltas mapped to [0,1] |

**Anti-gaming guard** (self-consistency is trivially maximized by encoding
nothing): every report pairs the consistency vector with
(a) **coverage** — fraction of source audio energy attributable to emitted
streams (stem energy accounted for by event/texture streams), and
(b) **fidelity** — CLAP audio-audio cosine similarity between A and B
(msclap embeddings; laion-clap from git as the cross-check encoder).
The headline triple is **(fidelity, consistency, coverage)**; a change is a
win only if it does not degrade the other two.

### 10.4 Perceptual + human
- CLAP audio-audio and audio-text (caption) similarity per song. **FAD is
  corpus-level only** — single-pair FAD is statistically meaningless
  (verified: Gaussian-fit instability, sample-size bias; the per-song mode
  in fadtk scores one song against a reference *set*). FAD (fadtk,
  CLAP/VGGish embeddings) enters at MUSDB18 scale-out, comparing the
  distribution of decodes to the distribution of originals.
- Human A/B: fixed protocol — 10 s excerpts, original vs decode, questions:
  "same song?" (y/n), "same production?" (1–5), "which is the original?"
  (forced choice; near-50% confusion is *not* the goal — recognizability
  as the same song is). Run at M9/M10 gates.
- Composite take-selection score for decode retries: weighted
  (CLAP-to-original, alignment error, lyric WER via WhisperX-on-output).

### 10.5 Corpus

| Tier | Item | Ground truth | License note |
|---|---|---|---|
| Control | "Signal Lost" — rendered by `soundcode render` from the v3 file | exact (the file *is* the truth) | CC0, ours |
| Mini | 30 s MAESTRO v3 piano | aligned MIDI | CC BY-NC-SA 4.0 |
| Real | 30 s NIN "Discipline" (*The Slip*) | official multitracks | CC BY-NC-SA 3.0 |
| Real-long | full "Discipline" (~4 min) | same | same |
| Scale-out | MUSDB18-HQ | isolated stems | research license |

All noncommercial-research compatible. Corpus audio is never committed to
git; a fetch script + checksums are.

---

## 11. Milestones (ordered; each independently verifiable)

- **M0 Environment** — py3.11 venv; every §3 pin installs; smoke-import
  script passes; ACE-Step 1.5 launches on the machine and generates 10 s
  from a trivial prompt. *Verify: script exit 0 + audible output.*
- **M1 Format core** — `scfmt` parse/emit/resolve/diff; golden test:
  `signal-lost.v3.sc` → model → text is byte-stable; sugar resolution
  round-trips seconds↔bar:beat and cents↔names. *Verify: golden tests.*
- **M2 Mock renderer + Control corpus** — `soundcode render` produces the
  Control wav from the v3 file. *Verify: rendered audio matches the file's
  own note events when re-analyzed (onset/pitch spot-check ≥0.95 F1).*
- **M3 Grid/struct/harmony encode on Control** — near-exact recovery
  expected (truth known). *Verify: tempo within 0.5 BPM, sections exact,
  chords ≥ 15/16 bars.*
- **M4 Notes on Mini** — basic-pitch stage + `eval transcription` vs
  MAESTRO MIDI. *Verify: F1 reported; target calibrated and pinned here.*
- **M5 Separation + full encode on Real-30s** — SDR vs NIN stems; complete
  `.sc` with confidences and warns. *Verify: SDR report + parseable file +
  every low-conf value marked.*
- **M6 Bridge** — `.sc` → bundle JSON, schema-validated, deterministic
  fields computed by code. *Verify: schema validation + fixture snapshots
  on all corpus `.sc` files.*
- **M7 Decode Path A on Control** — text2music from the bridge bundle;
  baseline CLAP + alignment scores recorded. *Verify: report exists;
  lyrics WER < 0.3 on output.*
- **M8 Decode Path B** — mock-remix with strength sweep {0.4…0.9}; A/B vs
  Path A on Control + Real-30s. *Verify: Path B beats Path A on melody
  (note F1 of re-encode vs sc1) without losing CLAP fidelity — this is the
  go/no-go for the fidelity thesis (R1).*
- **M9 Round-trip harness** — `eval roundtrip` produces the
  (fidelity, consistency, coverage) triple on all three short corpus items;
  first human A/B. *Verify: report generated end-to-end by one command.*
- **M10 Real-long** — full 4-min encode/decode; anchors, compact, repaint
  loop exercised. *Verify: round-trip report + no stream silently absent.*
- **M11 Scale-out** — MUSDB18-HQ batch; corpus-level FAD; per-stage failure
  statistics feed back into confidence calibration. *Verify: batch report.*

---

## 12. Risks and open questions

### Risks

- **R1 (highest): Path B is unproven.** Remix "maintains melodic structure"
  per its docs, but behavior on a *synthetic MIDI-quality mock* as source is
  unbenchmarked anywhere. If it clings to mock timbre at useful strengths or
  discards melody at low ones, the fidelity thesis needs rework (options:
  strength scheduling, mock-quality investment, v1 audio2audio, LoRA).
  M8 is deliberately early-ish and decisive.
- **R2: ACE-Step 1.5 on M1 Max is thinly documented.** No official Apple
  performance table; community reports ≈10–20× slower than A100 and a hard
  requirement to use the 0.6B LM on 32 GB (1.7B peaked ~42 GiB under MLX).
  Fallback chain in §7.2 is load-bearing, not decorative.
- **R3: Sung-lyric alignment.** WhisperX degrades on melisma/sustain (no
  first-party benchmark). Mitigations: vocal-stem input, struct gating,
  `?`/`alt=` doubt, WER measured on Control where truth is known.
- **R4: Fragile deps.** allin1 (NATTEN, stale), madmom (git-only, nominal
  maintenance), laion-clap (install pain). Every one has a named fallback in
  §3/§5; pins live in one lockfile; M0 exists to surface breakage first.
- **R5: Grid errors are systemic** (half/double tempo, downbeat phase).
  Sanity check + free-time fallback specified in §5; Control and Mini make
  the failure measurable before Real audio hides it.
- **R6: License hygiene.** MAESTRO and NIN corpus items are CC BY-NC-SA —
  fine for this research use; outputs derived from them inherit NC-SA and
  must not ship in demos without that notice. Never commit corpus audio.
  Never put third-party lyrics in specs/examples (the Control lyrics are
  original to this project — keep it that way).

### Open questions (need a human decision)

- **Q1. Bless Path B (mock-render remix) as the primary decode path?**
  Spec assumes yes; it's the only route note-level fidelity can travel.
  The alternative posture — Path A only, accept "same words/style, new
  tune" — would shrink the mock renderer to a test utility.
- **Q2. ACE-Step 1.5 vs v1 as the primary target?** Spec picks 1.5
  (metadata conditioning + Remix + Mac path + MIT) despite thinner docs;
  v1 is better-benchmarked on M-series. Flip only if M0/M7 show 1.5
  instability.
- **Q3. Compact-by-default threshold.** Auto-factoring above 90 s trades
  literal per-bar nuance in the *file* (never in decode, which expands) for
  readability and bridge token budget. Accept, always-on, or opt-in?
- **Q4. `align=word` as encoder default** (syllable only for authored
  files)? Syllable-izing ASR output adds a failure mode for marginal gain.
- **Q5. Store bridge-authored `@style`/`@mix`/`desc` prose in the `.sc`?**
  Spec says yes (cached, editable, provenance in sidecar). The purist
  alternative regenerates prose at decode time from evidence only.
- **Q6. Is (fidelity, consistency, coverage) the right headline triple,**
  and what initial weights should take-selection use? Provisional weights
  ship at M7; calibration against human A/B lands at M9.

---

## Appendix A — verification ledger

Verified by direct source check (2026-07): ACE-Step v1 README (modes,
params, M2 Max RTF table, `--bf16 false`, audio2audio `ref_audio_strength`
0–1 default 0.5); ACE-Step 1.5 repo + Gradio guide (model sizes, caption/
lyrics/BPM/key/time-sig/duration 10–600 s fields, Remix/Repaint/Extract/
Lego/Complete, REST/gradio entry points, MIT license, INT8/compile flags);
demucs fork 4.1.0 + MPS breakage; beat_this on PyPI; madmom PyPI-vs-git
status; basic-pitch 3.12 gap; WhisperX version bounds + CPU-only on Mac;
torchcrepe/pyin; Essentia wheel matrix; mir_eval 0.8.2 tolerances; museval;
fadtk single-pair caveat; MAESTRO alignment/license; NIN stems archive +
license; qwen3.5/gemma4 availability and RAM fit.

Explicitly **unverified** (marked where used): exact ACE-Step checkpoint
download sizes; official Apple-Silicon perf for 1.5 (community numbers
only); Remix behavior on synthetic mock sources (R1); WhisperX sung-vocal
accuracy (no first-party benchmark); MPS support for beat_this/torchcrepe/
allin1 (CPU is the verified path); basic-pitch drum-content behavior
(inferred from model design); per-model LLM JSON quality (family
reputation, not benchmarked).
