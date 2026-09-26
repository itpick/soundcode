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

## Model stack (Mac-local first; NC allowed)

| Job | Default | Alternatives / experiments |
|---|---|---|
| Separation | audio-separator (BS-RoFormer vocals) + Demucs 4.1 `htdemucs_ft` (MPS) | Mel-Band RoFormer, SCNet |
| Beats and downbeats | `beat_this` 1.1 | SheetSage2, madmom (git) |
| Structure | SongFormer | SheetSage2; novelty fallback snapped to downbeats |
| Chords and key | **SheetSage2** (NC; one model covers beats, key, chords, structure and melody) | ChordMini/BTC + Essentia key |
| Notes | basic-pitch (bass/other); penn or torchcrepe for monophonic vocals | YourMT3+, VocalParse |
| Timbre / description | msclap tags + MOSS-Music 8B (MLX) captions | Gemini audio-in |
| Lyrics ASR | Qwen3-ASR 1.7B + Qwen3-ForcedAligner (`mlx-qwen3-asr`) | WhisperX; MOSS-Music as a second opinion |
| Web lyrics | AcoustID → MusicBrainz → LRCLIB, then `ctc-forced-aligner` on the vocal stem | Musixmatch, AudD |
| `.sc` authoring LLM | Claude Opus 5.5 / Fable 5.1 (structured output) | Local Qwen3.5-27B (MLX) |
| Audio decoder | ACE-Step 1.5 (remix/cover/repaint) | MuLaCover (NC; melody+chord MIDI conditioning; needs CUDA, so rent a GPU), YuE2 (NC, CUDA), ElevenLabs Music API |

The research agent's version and release claims should be verified when each tool is adopted.
Known pins: librosa `<1.0` while on Python 3.11. torchaudio forced alignment was removed in 2.9.

## Lyrics reconciliation

1. Identify the song: AcoustID/Chromaprint → MusicBrainz (title, artist, duration).
2. Fetch LRCLIB synced and plain lyrics. Musixmatch is an optional fallback.
3. Forced-align the web text to the vocal stem. LRCLIB line times serve as a ±1 s prior.
4. Where the alignment score is low, fall back to ASR words (rapidfuzz/jiwer alignment, voting per word across sources). Unresolved words carry `?conf` / `alt=`.
5. An LLM only chooses between candidates (ordering, section labels, casing) and never writes lyrics.
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
