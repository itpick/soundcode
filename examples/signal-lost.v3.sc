# =============================================================================
#  soundcode  v0.3   —   .sc
# =============================================================================
#
#  WHAT CHANGED FROM v0.2  (v0.2's three structural rules are unchanged:
#  seconds are canonical time, cents are canonical pitch, streams are an
#  optional namespaced subset. v0.3 makes the format honest about being
#  MACHINE-PRODUCED from messy audio, and useful to a real decoder.)
#
#   1. CONFIDENCE IS FIRST-CLASS. Every stream may open with `meta` lines
#      declaring its source tool and overall confidence. Any value may carry
#      a trailing `?0.NN` marker when the encoder is unsure, and `alt=` may
#      offer the runner-up hypothesis. A wrong chord stated confidently is
#      worse than "Am ?0.55 alt=C". Absence of `?` means conf >= the
#      stream's `conf_floor` (default 0.80) — silence is a claim.
#
#   2. TIMBRE IS MEASURED, NOT ASSERTED. Track declarations carry an
#      `inst=` class from a small controlled vocabulary, a `timbre{...}`
#      block of measured spectral facts, and `tags{...}` classifier output
#      with scores. The freeform ~"..." gloss remains, but it is authored
#      (by a human or by the encode-time LLM pass) — it is presentation,
#      not evidence.
#
#   3. PATTERNS ADMIT THEY ARE LOSSY. `%pat` and `like` bindings may carry
#      `~0.NN` similarity. Bare bindings are exact (byte-identical events);
#      `~` bindings mean "factored within tolerance, per-instance nuance
#      discarded". The encoder emits explicit events; a separate factoring
#      pass introduces patterns and must annotate what it threw away.
#
#   4. STRUCTURE CARRIES DECODER-FACING CHARACTER. Each :struct section
#      gets vocal/instrumental flags, relative energy, and an optional
#      one-line desc. This is what the decode bridge actually turns into
#      conditioning; on 4-minute songs it is the difference between
#      "a chorus" and "the third chorus, half-time, stripped to voice".
#
#   5. THE GRID RE-ANCHORS. Multiple `anchor` lines pin bars to seconds so
#      beat-tracking drift cannot accumulate across a full-length song.
#
#   6. `|` MEANS ONE THING. It separates events on a line. Nothing else.
#      Multi-field scalar lines (as in :mix) use commas.
#
# -----------------------------------------------------------------------------
#  NOTATION
#
#   time        5:1.0        bar 5 beat 1      (needs :grid)
#               @7.500       absolute seconds  (always valid)
#   duration    0.5b         beats             (needs :grid)
#               0.234s       seconds           (always valid)
#   pitch       A3           note name         (needs :tuning)
#               A3+8c        name + cents offset
#               5708c        raw cents from reference
#               ~220.0Hz     raw frequency
#   velocity    0-127        (mapped from stem-relative level at encode)
#
#   :name       stream declaration; kv fields and blocks may follow
#   meta        stream metadata line (src, conf, warn) — first lines only
#   %pat        named pattern, scoped to the current stream
#   !auto       parameter automation between two time points
#   ?0.NN       confidence marker on the preceding value
#   alt=X       runner-up hypothesis for the preceding value
#   ~0.NN       similarity on a pattern/like binding (lossy factoring)
#
# =============================================================================

%sc        0.3
%profile   metric-tonal
%residual  none
#          ^ v1 emits pure symbols. A future learned acoustic residual slots
#            in below this line without changing anything above it.

@title     "Signal Lost"
@artist    "soundcode control track"
@source    control/signal-lost-render.wav
@offset    0.000
#          ^ where this file's t=0 sits in the source audio (leading silence
#            or excerpt start). Always present; usually zero.
@license   CC0-1.0
@duration  30.000
@sr        44100

# @style and @mix are prose for the decode bridge. In an encoded file they
# are written by the encode-time LLM describe pass from the measured
# evidence below (timbre blocks, :mix scalars) and are freely human-
# editable. This is a control track, so they are authored.

@style     "mid-tempo industrial pop, 128 bpm, A minor, dry compressed
            drums, distorted saw bass, clean electric piano pad, close-mic
            male vocal with doubled female backing, moderate plate reverb"

@mix       "drums forward and center, bass mono below 120Hz, vocal center
            with 12ms slapback, pad wide stereo, light sidechain duck on
            pad from kick"


# --- grid --------------------------------------------------------------------
# Tempo is a CURVE. Points interpolate linearly. Multiple anchors pin bars
# to absolute seconds so drift cannot accumulate; on encoded files an
# anchor is emitted at least every 16 bars and at every section boundary.
# Omit this entire stream for free-time music.

:grid
meta    src=authored  conf=1.00
meter   @0.000   4/4
anchor  bar 1    @0.000
anchor  bar 13   @22.500
tempo   @0.000   128.00
tempo   @26.250  128.00
tempo   @30.000  127.10
#       ^ slight ritard through the final bar.


# --- tuning ------------------------------------------------------------------
# Omit for atonal/unpitched material. Replace `12tet` with an explicit
# cent table for maqam, gamelan, or historical temperaments.

:tuning
ref          A4 = 440.0Hz
temperament  12tet


# --- struct ------------------------------------------------------------------
# vocal|inst flag and energy (0-1, relative to song max) are required on
# encoded files; desc is optional and bridge-authored. The decode bridge
# maps these directly onto the decoder's lyric section tags, so getting
# them wrong is audible — hence they may carry ? markers like anything else.

:struct
meta    src=authored  conf=1.00
intro   1-4     inst   energy=0.42  desc="drums+bass+pad, no voice"
verse   5-12    vocal  energy=0.61  desc="close male vocal enters, pad steady"
chorus  13-16   vocal  energy=0.88  desc="full band, backing vocals, drive up"


# --- harmony -----------------------------------------------------------------

:harmony
meta    src=authored  conf=1.00
tonal_center  A  aeolian
bars 1-4    Am | F | C | G
bars 5-12   like 1-4  x2
bars 13-16  F  | C | G | Am

# On an encoded file this stream looks like:
#   meta  src=chordrec:btc  conf=0.81
#   bars 21-24   Am | F ?0.58 alt=Dm | C | G
# and the encoder cross-checks it against :notes.bass roots. Disagreement
# is recorded, not hidden:
#   meta  warn="bar 22: bass implies Dm, chord model says F (0.58)"


# --- perc.drums --------------------------------------------------------------

:perc.drums  inst=drums.kit
  timbre{ attack 4ms, decay_50 180ms, crest 9.8dB, gated true }
  ~"dry compressed kit, tight gated snare, closed hats, no room"
meta    src=authored  conf=1.00

%pat basic {
  1.0 kick 104 dev+0ms  | 1.5 hat 58 dev+11ms | 2.0 snare 118 dev+6ms
  2.5 hat  55 dev+13ms  | 3.0 kick 98 dev-3ms | 3.25 kick 74 dev+2ms
  3.5 hat  60 dev+12ms  | 4.0 snare 120 dev+8ms | 4.5 hat 56 dev+14ms
}
#   dev = micro-timing deviation from the grid, in ms. The hats sitting
#   consistently late IS the feel. When the factoring pass merges bars into
#   a pattern it keeps the MEDIAN dev per hit; the per-bar spread it
#   discarded is what the ~similarity number below is being honest about.

%pat sparse { 1.0 kick 100 | 2.0 snare 112 | 3.0 kick 96 | 4.0 snare 114 }
%pat fill   { 1.0 kick 104 | 2.0 snare 118 | 3.0 tom1 106 | 3.5 tom1 101
              4.0 tom2 110 | 4.5 tom2 104 }

bars 1-2   sparse
bars 3     basic
bars 4     fill
bars 5-11  basic ~0.96
bars 12    fill
bars 13-15 basic ~0.94
bars 16    fill

3:1.0   crash 112
13:1.0  crash 121
16:1.0  crash 118


# --- notes.bass --------------------------------------------------------------

:notes.bass  inst=bass.synth
  timbre{ centroid 410Hz, rolloff_95 1.9kHz, attack 6ms, harmonicity 0.71,
          dist 0.64, stereo 0.03 }
  tags{ "synth bass" 0.83, "electric bass" 0.11 }
  ~"distorted saw, lowpass ~400Hz, slight pitch drift, mono"
meta    src=authored  conf=1.00

%pat root8 { 1.0 . 0.5b 108 | 1.5 . 0.5b 92 | 2.5 . 0.5b 96 | 3.0 . 1.0b 104 }
#   `.` resolves to the root of the current bar's chord from :harmony,
#   one octave below the notated register.

bars 1-12   root8

13:1.0  F1-6c  1.5b 112 | 13:2.5 F1-4c 0.5b 94 | 13:3.0 F1-7c 1.0b 106
14:1.0  C2-3c  1.5b 110 | 14:2.5 C2-2c 0.5b 92 | 14:3.0 E2-5c 1.0b 100
15:1.0  G1-8c  1.5b 112 | 15:2.5 G1-6c 0.5b 94 | 15:3.0 B1-4c 1.0b 104
16:1.0  A1-9c  3.0b 116 | 16:4.0 A1-11c 1.0b 98
#   The consistent flat drift is the analog oscillator, not a transcription
#   error. Cents-canonical pitch can hold it.


# --- notes.pad ---------------------------------------------------------------

:notes.pad  inst=keys.ep
  timbre{ centroid 1.4kHz, attack 18ms, release_est 900ms, stereo 0.78,
          chorus_est 0.6 }
  tags{ "electric piano" 0.77, "synth pad" 0.19 }
  ~"clean electric piano through chorus, wide stereo, long release"
meta    src=authored  conf=1.00

1:1.0  A3 4.0b 62 | 1:1.0 C4 4.0b 58 | 1:1.0 E4 4.0b 60
2:1.0  F3 4.0b 62 | 2:1.0 A3 4.0b 58 | 2:1.0 C4 4.0b 60
3:1.0  C4 4.0b 64 | 3:1.0 E4 4.0b 60 | 3:1.0 G4 4.0b 62
4:1.0  G3 4.0b 64 | 4:1.0 B3 4.0b 60 | 4:1.0 D4 4.0b 62

bars 5-12   like 1-4  x2
bars 13-16  like 1-4  vel -8
#   Bare `like` = exact reuse. A factored, inexact reuse must say so:
#   `bars 13-16  like 1-4  vel -8  ~0.91`.


# --- notes.vox ---------------------------------------------------------------

:notes.vox  inst=voice.lead  gender=male  range=A3-F4
  timbre{ centroid 2.6kHz, breathiness 0.22, dist 0.18 }
  ~"male, close-mic, slight tube distortion, 12ms slapback, center"
meta    src=authored  conf=1.00

# verse
5:1.0   A3+6c  0.5b  96 dev+22ms
5:2.0   C4+4c  0.5b  98 dev+18ms
5:3.0   C4+3c  0.5b  97 dev+15ms
5:4.0   D4+9c  0.5b 101 dev+24ms
5:4.5   D4+7c  0.5b  99 dev+19ms
6:1.0   C4+5c  0.5b  96 dev+16ms
6:2.0   A3+2c  0.5b  94 dev+21ms
6:3.0   A3+4c  1.5b  99 dev+17ms  env{a 21ms, d 110ms, s 0.68, r 240ms}
6:4.5   G3-3c  0.5b  92 dev+12ms

7:1.0   A3+5c  0.5b  97 dev+20ms
7:2.0   C4+3c  0.5b  99 dev+17ms
7:3.0   C4+2c  0.5b  97 dev+14ms
7:4.0   E4+11c 0.5b 104 dev+26ms
7:4.5   D4+8c  0.5b 100 dev+18ms
8:1.0   C4+4c  0.5b  98 dev+15ms
8:2.0   A3+1c  0.5b  95 dev+19ms
8:3.0   A3+3c  1.5b 100 dev+16ms
8:4.5   G3-4c  0.5b  93 dev+11ms

9:1.0   C4+6c  0.5b 100 dev+23ms
9:2.0   D4+8c  0.5b 102 dev+19ms
9:2.5   D4+6c  0.5b 100 dev+16ms
9:3.0   E4+12c 0.5b 105 dev+27ms
9:4.0   D4+7c  1.0b 101 dev+20ms
10:1.0  C4+4c  0.5b  99 dev+17ms
10:1.5  C4+3c  0.5b  97 dev+14ms
10:2.0  A3+2c  2.0b 103 dev+18ms  env{a 19ms, d 130ms, s 0.72, r 310ms}

11:1.0  C4+5c  0.5b 101 dev+21ms
11:2.0  C4+4c  0.5b  99 dev+18ms
11:3.0  D4+7c  0.5b 102 dev+15ms
11:4.0  E4+13c 1.0b 106 dev+25ms
12:1.0  D4+9c  0.5b 102 dev+19ms
12:2.0  C4+5c  0.5b 100 dev+16ms
12:2.5  A3+3c  2.5b 104 dev+20ms

# chorus — pushes noticeably harder and sits later
13:1.0  F4+14c 0.5b 112 dev+31ms
13:2.0  E4+11c 0.5b 110 dev+27ms
13:3.0  C4+8c  2.0b 114 dev+24ms  env{a 12ms, d 80ms, s 0.79, r 420ms}
14:1.0  E4+12c 0.5b 112 dev+29ms
14:2.0  D4+10c 0.5b 110 dev+26ms
14:3.0  C4+7c  2.0b 113 dev+23ms
15:1.0  D4+11c 0.5b 111 dev+28ms
15:1.5  D4+9c  0.5b 109 dev+25ms
15:2.0  E4+13c 0.5b 112 dev+30ms
15:3.0  D4+10c 0.5b 110 dev+26ms
15:4.0  B3+6c  1.0b 108 dev+22ms
16:1.0  A3+4c  3.5b 106 dev+18ms

# On an encoded file, a doubtful note carries its doubt:
#   9:3.0  E4+12c 0.5b 105 ?0.42 alt=G4
# and a note whose onset the aligner could not pin gets `dev?`.


# --- text.vox ----------------------------------------------------------------
# Lyrics in their OWN stream, keyed to time. `align=word` is what forced
# alignment actually produces; `align=syllable` is authoring-grade. Rap and
# spoken word keep this stream and drop :notes.vox entirely.

:text.vox  lang=en  align=syllable
meta    src=authored  conf=1.00

5:1.0  "I" | 5:2.0 "was" | 5:3.0 "a" | 5:4.0 "sig" | 5:4.5 "nal"
6:1.0  "in" | 6:2.0 "the" | 6:3.0 "stat" | 6:4.5 "ic"
7:1.0  "you" | 7:2.0 "were" | 7:3.0 "the" | 7:4.0 "hand" | 7:4.5 "that"
8:1.0  "turned" | 8:2.0 "the" | 8:3.0 "di" | 8:4.5 "al"
9:1.0  "now" | 9:2.0 "ev" | 9:2.5 "ery" | 9:3.0 "word" | 9:4.0 "I"
10:1.0 "ev" | 10:1.5 "er" | 10:2.0 "said"
11:1.0 "comes" | 11:2.0 "back" | 11:3.0 "to" | 11:4.0 "me"
12:1.0 "as" | 12:2.0 "num" | 12:2.5 "bers"
13:1.0 "Hold" | 13:2.0 "the" | 13:3.0 "line"
14:1.0 "hold" | 14:2.0 "the" | 14:3.0 "line"
15:1.0 "no" | 15:1.5 "thing" | 15:2.0 "here" | 15:3.0 "is" | 15:4.0 "lost"
16:1.0 "ohh"

# Encoded files carry ASR doubt the same way as everything else:
#   21:3.0 "dial" ?0.51 alt="denial"


# --- contour.vox -------------------------------------------------------------
# Continuous F0 behaviour that discrete notes cannot express. Written
# sparsely — only where pitch departs from the notated value. This stream
# is what makes glissando, maqam, blues bends and throat singing
# representable at all.

:contour.vox
meta    src=authored  conf=1.00
6:3.0   vib{rate 5.4Hz, depth ±31c, onset 0.22s}
10:2.0  vib{rate 5.1Hz, depth ±28c, onset 0.31s}
13:3.0  scoop{from -68c, dur 90ms} vib{rate 5.8Hz, depth ±42c, onset 0.18s}
16:1.0  vib{rate 4.9Hz, depth ±37c, onset 0.25s} fall{to -140c, dur 380ms}


# --- notes.bvox --------------------------------------------------------------

:notes.bvox  inst=voice.backing  gender=female
  ~"doubled and detuned ±8 cents, wide, heavy plate reverb"
meta    src=authored  conf=1.00

13:1.0  A4 4.0b 74 | 14:1.0 G4 4.0b 74 | 15:1.0 G4 4.0b 76 | 16:1.0 A4 4.0b 78
13:3.0  C5 2.0b 82 | 14:3.0 C5 2.0b 82

:text.bvox  lang=en  align=syllable
meta    src=authored  conf=1.00
13:1.0 "ah" | 14:1.0 "ah" | 15:1.0 "ah" | 16:1.0 "ah"
13:3.0 "line" | 14:3.0 "line"


# --- mix ---------------------------------------------------------------------
# Scalar lines are `name field value, field value, ...` — commas, never `|`.
# `by_section` rows give the coarse dynamic arc a 30s clip doesn't need but
# a 4-minute song lives or dies by.

:mix
meta    src=authored  conf=1.00
lufs_int      -8.4
lufs_range     4.1
stereo_width  low 0.02, mid 0.61, high 0.88
rt60          low 1.10s, mid 0.82s, high 0.41s
comp_est      ratio 4.2:1, attack 8ms, release 120ms
by_section    intro -12.1, verse -9.8, chorus -7.6

!auto  pad.cutoff       @0.000  0.25 -> @7.500  0.85
!auto  pad.cutoff       @7.500  0.60 -> @22.500 0.60
!auto  vox.reverb_send  @7.500  0.18 -> @22.500 0.28
!auto  vox.reverb_send  @22.500 0.42 -> @30.000 0.55
!auto  master.drive     @22.500 0.30 -> @30.000 0.48


# =============================================================================
#  STREAMS NOT USED BY THIS SONG (documentation only):
#
#    :texture.<trk>   band-energy trajectories for material with no discrete
#                     onsets — ambient, drone, noise, field recording.
#    :residual        Layer-1 learned acoustic tokens. Binary; lives in the
#                     .scz container, never inline in .sc.
#
#  A song is whatever subset of streams applies to it. Solo piano needs
#  four. A harsh-noise track needs :mix and :texture and nothing else.
# =============================================================================
