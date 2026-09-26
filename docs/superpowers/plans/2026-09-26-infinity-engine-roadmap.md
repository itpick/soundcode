# Infinity Engine roadmap (soundcode)

Date: 2026-09-26. Source: Fable 5.1 repo review + model/tool research.
The project will eventually be renamed **Infinity Engine**; the rename is deferred.

## Milestones

1. **Clean separation: vocals from music, and music from vocals.** Split any song into separate stems: lead vocal, backing vocals, drums, bass, and guitar/piano/other. Each stem should be clean enough to play on its own, and summing them should come back within about 1 dB of the original. *(Start of Phase 1; stem references in Phase 3.)*
2. **The voice as code: pitch, notes, expression and lyrics.** Track the melody as notes, plus the pitch curve (vibrato, slides, fall-offs), timing and dynamics. Every lyric word is timed and checked against lyrics found online. Handle lead, backing and harmony voices. Done when rendering only the vocal line is recognisable, lyrics are at least 95% correct, and each word is timed to within about 80 ms. *(Vocal part of Phase 1; Phase 2.)*
3. **Every instrument and every harmony as code.** Give each instrument its own track (notes, drum hits, a basic description of its sound). Capture key, chords with type/extensions/inversions, bass line, vocal harmony parts relative to the lead, sections, tempo changes and time signature. Done when the synth render is recognisably the same song and re-analysing it gives back the same `.sc` within tolerance. *(Rest of Phase 1; Phase 0 format tooling.)*
4. **Rebuild the song, two ways.** Faithful: rebuild from the `.sc` file plus stems, and change any part. Generative: send the `.sc` file to ACE-Step (or MuLaCover on a rented GPU) for a new version that keeps structure, melody and lyrics. Done when generated covers keep the melody and lyric timing on re-analysis, and a stem rebuild with one part changed sounds seamless. *(Phases 3, 4, 6.)*
5. **Infinity Engine: write new music.** A written description → a full `.sc` file from a language model (checked automatically, errors sent back to fix) → rendered or generated audio. Also "same vibe, new song" from an analysed song. Done when at least 8 of 10 varied descriptions produce valid, listenable songs with no hand edits. *(Phases 5, 7; rename.)*

Order: 1 first → 2 and 3 in parallel → 4 → 5.

## Decisions

| Question | Decision |
|---|---|
| Rename timing | Later, not during Phase 0 |
| Use | Research and private only. Non-commercial (NC) model licenses are acceptable. |
| Copyrighted lyrics | May be stored in encoded `.sc` files and work dirs, private only. `out/` and `audio/` stay gitignored, the server stays bound to 127.0.0.1, and the GitHub repo stays private. Committed fixtures and examples contain no third-party lyrics. |
| Meaning of "re-render" | Both: a generative cover/remix (ACE-Step) **and** a deterministic rebuild from stems + automation |

## Model stack (open, free, Mac-local first; NC allowed; no paid APIs)

Full detail: [open codifying](../../research/2026-09-26-open-codifying.md) · [open synthesis](../../research/2026-09-26-open-synthesis.md)

### Codifying (audio → `.sc`)

| Job | Default | Alternatives / experiments |
|---|---|---|
| Separation | audio-separator: BS-RoFormer vocals → karaoke RoFormer lead/backing → Demucs `htdemucs_6s` | Mel-Band RoFormer, SCNet, DrumSep for drum kit pieces |
| Beats, key, chords, structure, lead melody | **SheetSage2** on the full mix (CC BY-NC) | `beat_this` 1.1, SongFormer, ChordMini + Essentia |
| Drums | DrumSep → ADTOF | onset templates (current) |
| Piano | Transkun | basic-pitch |
| Bass / other notes | basic-pitch | MuScriptor, YourMT3+ |
| Vocal pitch curve | torchcrepe | SwiftF0, penn |
| Lyrics from audio | Qwen3-ASR 1.7B + Qwen3-ForcedAligner (`mlx-qwen3-asr`) | WhisperX; MOSS-Music as a second opinion |
| Web lyrics | AcoustID → MusicBrainz → LRCLIB (free), then forced alignment on the vocal stem | — |
| Listening helper (captions, descriptions, Q&A) | MOSS-Music-8B, 8-bit MLX | Qwen3-Omni (MLX). Never used for key, chords or tempo; dedicated models are far more accurate |
| `.sc` authoring LLM | **Qwen3.8-27B** 4-bit MLX, constrained by a Lark grammar through Outlines (`mlxlm`) | Qwen3.6-35B-A3B for fast repair loops, gpt-oss-20b for checker triage, Gemma 4 31B; later QLoRA with `mlx_lm.lora` |
| Interop | symusic (MIDI/ABC), music21 (MusicXML), JAMS + mir_eval (annotations, metrics) | MidiTok, abcMIDI, MuseScore 4.6 CLI, LilyPond |
| Eval data | RWC-Popular, MUSDB18-HQ (+ lyrics), Slakh2100, JamendoLyrics, Harmonix | Lakh/MidiCaps rendered as a `.sc` training corpus |

### Synthesis (`.sc` → audio)

| Job | Default | Alternatives / experiments |
|---|---|---|
| Singing from notes + lyrics | **DiffSinger** (openvpi `.ds` input: phonemes, durations, notes, explicit f0 so vibrato/slides carry through) | SoulX-Singer (CUDA) |
| Re-sing in another voice | Seed-VC (Apple Silicon supported, zero-shot) | YingMusic-SVC; RVC/Applio when a voice must be trained |
| Faithful instruments | tinysoundfont + GeneralUser GS; sfizz + free SFZ (Salamander, VPO, VSCO 2 CE); DawDreamer hosting Surge XT / Vital / Dexed; pedalboard for `:mix` | TokenSynth (neural) |
| Generative full song | **ACE-Step 1.5** (installed; MIT; tempo/key/meter, cover, repaint) + DiffSynth-Music controls | MuLaCover (melody/chord/drum MIDI; rented GPU), YuE2 (editable ABC; MLX ports) |
| Instrumental generation | ACE-Step lego/instrumental | MuseControlLite, Magenta RealTime 2, JASCO |

The research agents' version and release claims should be verified when each tool is adopted; items flagged *(verify)* in the research files especially.
Known pins: librosa `<1.0` while on Python 3.11. torchaudio forced alignment was removed in 2.9.

## Lyrics reconciliation

1. Identify the song: AcoustID/Chromaprint → MusicBrainz (title, artist, duration).
2. Fetch LRCLIB synced and plain lyrics (free, no key).
3. Forced-align the web text to the vocal stem. LRCLIB line times serve as a ±1 s prior.
4. Where the alignment score is low, fall back to ASR words (rapidfuzz/jiwer alignment, voting per word across sources). Unresolved words carry `?conf` / `alt=`.
5. An open LLM (local) only chooses between candidates (ordering, section labels, casing) and never writes lyrics.
6. Record provenance per line: `src=lrclib|asr|aligned`.

## Phases

**Phase 0: Foundation.**
- `testpaths = ["tests"]`; align the pyproject extras with the code.
- `emit.py` (Document → text); the encoder builds a `Document`.
- `check --strict` with line-numbered errors.
- Parser fixes: single-chord bindings, text `dur`, header-after-stream, bare beats.
- Renderer fixes: `@t-@t` struct ranges, beat durations on absolute events, render `!auto`.
- `expand`/`compact`/`inspect`, plus JSON export.

**Phase 1: Encoder correctness.**
- Fix the +33 c pitch bias (`encode.py:439`) with a tone-calibration test.
- Downbeats via beat_this; anchors at section boundaries; honest grid confidence.
- Monophonic vocal F0 → `:notes.vox` + `:contour.vox`.
- Deduplicate drum onsets; emit `inst=`, `tonal_center`, real LUFS.
- Work dir with `provenance.json`.

**Phase 2: Lyrics (parallel with Phase 1).**
- Format additions: a `:lyrics` stream, text `dur`, and `line=`/`word=` references.
- Modules: `lyrics/asr.py`, `lyrics/lookup.py`, `lyrics/align.py`, and the `soundcode lyrics` command.

**Phase 3: Format v0.4.**
- `:audio.<trk>` stem and sample references, with deterministic stem rendering (the second re-render path).
- `:prompt` generation settings block.
- `groove`.
- `:texture` support.

**Phase 4: Decode.**
- Deterministic bridge bundle → ACE-Step `GenerationParams`.
- `soundcode decode`, plus a strength sweep through the server.
- MuLaCover MIDI-conditioned experiment on a rented GPU.

**Phase 5: Compose.** `soundcode compose "<brief>"`: the LLM emits `.sc`, `check --strict` errors are fed back, with bounded retries.

**Phase 6: Evaluation.**
- Per-stream diff and a round-trip score.
- Re-encode generated output and repaint drifted sections.

**Phase 7: Scale.** Full-length songs and batch corpus runs.

## Known bugs (from review)

- Every encoded note is +33 c sharp (`encode.py:439-441`).
- Downbeat = first beat (`encode.py:128`).
- `x:5.00` and negative beats (`encode.py:302-307, 433-437`).
- `:grid conf=1.00` is always emitted, and the first tempo point is off by 3:2.
- Vocals are transcribed polyphonically, which produces octave stacks.
- `vocal` flag = mix energy (`encode.py:192`).
- `lufs_int` is RMS dBFS.
- Duplicate drum onsets.
- Parser holes (`parser.py:127-129, 184-187, 204, 262-268`).
- `section_gains` crashes on `@t` ranges (`render.py:186`).
- Beat durations on `@seconds` events assume 120 BPM (`expand.py:234`).
- `!auto` is never rendered.
- Per-sample Python lowpass.
- The server runs demucs synchronously, and upload rstrip truncates binaries.
