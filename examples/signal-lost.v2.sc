# =============================================================================
#  soundcode  v0.2   —   .sc
# =============================================================================
#
#  WHAT CHANGED FROM v0.1  (all three are structural; syntax is unchanged)
#
#   1. TIME is seconds. The bar:beat grid is an optional overlay.
#      Canonical timestamps are absolute seconds. If a :grid stream is
#      present you may WRITE bar:beat and the parser resolves it. Music
#      with no pulse simply omits :grid and writes @seconds throughout.
#
#   2. PITCH is cents from a declared reference. Note names are an overlay.
#      Canonical pitch is an integer cent value. If a :tuning stream is
#      present you may WRITE note names, optionally with a cent offset.
#      Microtonal, maqam, gamelan, and vocal scoops stop being lies.
#
#   3. STREAMS are optional, namespaced and versioned. There is no single
#      schema every song must fit. A song is whatever subset of streams
#      applies to it. Adding stream types later never invalidates old files.
#
#  Both overlays are pure sugar: the parser normalizes to seconds+cents,
#  and re-emits your preferred notation on write. Round-trip stable.
#
# -----------------------------------------------------------------------------
#  NOTATION
#
#   time        5:1.0        bar 5 beat 1      (needs :grid)
#               @7.500       absolute seconds  (always valid)
#   duration    0.5b         beats             (needs :grid)
#               0.234s       seconds           (always valid)
#   pitch       A3           note name         (needs :tuning)
#               A3+8c        name + cents
#               5708c        raw cents from reference
#               ~220.0Hz     raw frequency
#   velocity    0-127
#
#   :name       stream declaration.  ~"..." after it is a freeform tag.
#   %pat        named pattern, scoped to the current stream
#   !auto       parameter automation between two time points
#
# =============================================================================

%sc        0.2
%profile   metric-tonal
%residual  none
#          ^ v1 emits pure symbols. Setting this to `semanticodec:0.31kbps`
#            later adds a Layer-1 acoustic residual. Nothing above this line
#            changes when that happens — that is the whole point of the split.

@title     "Signal Lost"
@artist    "soundcode control track"
@source    control/signal-lost-render.wav
@license   CC0-1.0
@duration  30.000
@sr        44100

@style     "mid-tempo industrial pop, dry compressed drums, distorted saw
            bass, clean electric piano pad, close-mic male vocal with
            doubled female backing, moderate plate reverb"

@mix       "drums forward and center, bass mono below 120Hz, vocal center
            with 12ms slapback, pad wide stereo, light sidechain duck on
            pad from kick"


# --- grid --------------------------------------------------------------------
# Tempo is a CURVE, not a number. Points are linearly interpolated.
# Omit this entire stream for free-time music.

:grid
meter   @0.000   4/4
anchor  bar 1  @0.000
tempo   @0.000   128.00
tempo   @26.250  128.00
tempo   @30.000  127.10
#       ^ slight ritard through the final bar. A single @tempo scalar
#         cannot express this, and its absence is audible as stiffness.


# --- tuning ------------------------------------------------------------------
# Omit for atonal/unpitched material. Replace `12tet` with an explicit
# cent table for maqam, gamelan, or historical temperaments.

:tuning
ref          A4 = 440.0Hz
temperament  12tet


# --- struct ------------------------------------------------------------------

:struct
intro   1-4
verse   5-12
chorus  13-16


# --- harmony -----------------------------------------------------------------

:harmony
tonal_center  A  aeolian
bars 1-4    Am | F | C | G
bars 5-12   like 1-4  x2
bars 13-16  F  | C | G | Am


# --- perc.drums --------------------------------------------------------------

:perc.drums  ~"dry compressed kit, tight gated snare, closed hats, no room"

%pat basic {
  1.0 kick 104 dev+0ms  | 1.5 hat 58 dev+11ms | 2.0 snare 118 dev+6ms
  2.5 hat  55 dev+13ms  | 3.0 kick 98 dev-3ms | 3.25 kick 74 dev+2ms
  3.5 hat  60 dev+12ms  | 4.0 snare 120 dev+8ms | 4.5 hat 56 dev+14ms
}
#   dev = micro-timing deviation from the grid, in ms. This is the single
#   cheapest thing that separates "a person played this" from "MIDI".
#   Note the hats consistently sit late — that IS the feel.

%pat sparse { 1.0 kick 100 | 2.0 snare 112 | 3.0 kick 96 | 4.0 snare 114 }
%pat fill   { 1.0 kick 104 | 2.0 snare 118 | 3.0 tom1 106 | 3.5 tom1 101
              4.0 tom2 110 | 4.5 tom2 104 }

bars 1-2   sparse
bars 3     basic
bars 4     fill
bars 5-11  basic
bars 12    fill
bars 13-15 basic
bars 16    fill

3:1.0   crash 112
13:1.0  crash 121
16:1.0  crash 118


# --- notes.bass --------------------------------------------------------------

:notes.bass  ~"distorted saw, lowpass ~400Hz, slight pitch drift, mono"

%pat root8 { 1.0 . 0.5b 108 | 1.5 . 0.5b 92 | 2.5 . 0.5b 96 | 3.0 . 1.0b 104 }
#   `.` resolves to the root of the current bar's chord from :harmony,
#   one octave below the notated register. Transposing the song is a
#   one-line edit, and the encoder can cross-check transcribed bass
#   against detected chords — disagreement means one stage is wrong.

bars 1-12   root8

13:1.0  F1-6c  1.5b 112 | 13:2.5 F1-4c 0.5b 94 | 13:3.0 F1-7c 1.0b 106
14:1.0  C2-3c  1.5b 110 | 14:2.5 C2-2c 0.5b 92 | 14:3.0 E2-5c 1.0b 100
15:1.0  G1-8c  1.5b 112 | 15:2.5 G1-6c 0.5b 94 | 15:3.0 B1-4c 1.0b 104
16:1.0  A1-9c  3.0b 116 | 16:4.0 A1-11c 1.0b 98
#   The consistent flat drift is the analog oscillator, not a transcription
#   error. v0.1 rounded this away; v0.2 can hold it.


# --- notes.pad ---------------------------------------------------------------

:notes.pad  ~"clean electric piano through chorus, wide stereo, long release"

1:1.0  A3 4.0b 62 | 1:1.0 C4 4.0b 58 | 1:1.0 E4 4.0b 60
2:1.0  F3 4.0b 62 | 2:1.0 A3 4.0b 58 | 2:1.0 C4 4.0b 60
3:1.0  C4 4.0b 64 | 3:1.0 E4 4.0b 60 | 3:1.0 G4 4.0b 62
4:1.0  G3 4.0b 64 | 4:1.0 B3 4.0b 60 | 4:1.0 D4 4.0b 62

bars 5-12   like 1-4  x2
bars 13-16  like 1-4  vel -8


# --- notes.vox ---------------------------------------------------------------

:notes.vox  ~"male, close-mic, slight tube distortion, 12ms slapback, center"

# verse
5:1.0   A3+6c  0.5b  96 dev+22ms
5:2.0   C4+4c  0.5b  98 dev+18ms
5:3.0   C4+3c  0.5b  97 dev+15ms
5:4.0   D4+9c  0.5b 101 dev+24ms
5:4.5   D4+7c  0.5b  99 dev+19ms
6:1.0   C4+5c  0.5b  96 dev+16ms
6:2.0   A3+2c  0.5b  94 dev+21ms
6:3.0   A3+4c  1.5b  99 dev+17ms  env{a 21ms d 110ms s 0.68 r 240ms}
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
10:2.0  A3+2c  2.0b 103 dev+18ms  env{a 19ms d 130ms s 0.72 r 310ms}

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
13:3.0  C4+8c  2.0b 114 dev+24ms  env{a 12ms d 80ms s 0.79 r 420ms}
14:1.0  E4+12c 0.5b 112 dev+29ms
14:2.0  D4+10c 0.5b 110 dev+26ms
14:3.0  C4+7c  2.0b 113 dev+23ms
15:1.0  D4+11c 0.5b 111 dev+28ms
15:1.5  D4+9c  0.5b 109 dev+25ms
15:2.0  E4+13c 0.5b 112 dev+30ms
15:3.0  D4+10c 0.5b 110 dev+26ms
15:4.0  B3+6c  1.0b 108 dev+22ms
16:1.0  A3+4c  3.5b 106 dev+18ms


# --- text.vox ----------------------------------------------------------------
# Lyrics live in their OWN stream, keyed to note onsets rather than
# interleaved with them. Rap and spoken word keep this stream and drop
# :notes.vox entirely — prosody without scalar pitch.

:text.vox  lang=en  align=syllable

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


# --- contour.vox -------------------------------------------------------------
# Continuous F0 behaviour that discrete notes cannot express. Written
# sparsely — only where the pitch departs from the notated value.
# This stream is what makes glissando, maqam, blues bends and throat
# singing representable at all.

:contour.vox
6:3.0   vib{rate 5.4Hz, depth ±31c, onset 0.22s}
10:2.0  vib{rate 5.1Hz, depth ±28c, onset 0.31s}
13:3.0  scoop{from -68c, dur 90ms} vib{rate 5.8Hz, depth ±42c, onset 0.18s}
16:1.0  vib{rate 4.9Hz, depth ±37c, onset 0.25s} fall{to -140c, dur 380ms}


# --- notes.bvox --------------------------------------------------------------

:notes.bvox  ~"female, doubled and detuned ±8 cents, wide, heavy plate reverb"

13:1.0  A4 4.0b 74 | 14:1.0 G4 4.0b 74 | 15:1.0 G4 4.0b 76 | 16:1.0 A4 4.0b 78
13:3.0  C5 2.0b 82 | 14:3.0 C5 2.0b 82

:text.bvox  lang=en  align=syllable
13:1.0 "ah" | 14:1.0 "ah" | 15:1.0 "ah" | 16:1.0 "ah"
13:3.0 "line" | 14:3.0 "line"


# --- mix ---------------------------------------------------------------------

:mix
lufs_int      -8.4
lufs_range     4.1
stereo_width  low 0.02 | mid 0.61 | high 0.88
rt60          low 1.10s | mid 0.82s | high 0.41s
comp_est      ratio 4.2:1  attack 8ms  release 120ms

!auto  pad.cutoff       @0.000  0.25 -> @7.500  0.85
!auto  pad.cutoff       @7.500  0.60 -> @22.500 0.60
!auto  vox.reverb_send  @7.500  0.18 -> @22.500 0.28
!auto  vox.reverb_send  @22.500 0.42 -> @30.000 0.55
!auto  master.drive     @22.500 0.30 -> @30.000 0.48


# =============================================================================
#  STREAMS NOT USED BY THIS SONG (declared here only as documentation):
#
#    :texture.<trk>   spectral + statistical descriptors for material with
#                     no discrete onsets — ambient, drone, noise, field
#                     recording. Encodes as band-energy trajectories rather
#                     than events.
#    :residual        Layer-1 learned acoustic tokens. Binary; lives in the
#                     .scz container, never inline in .sc.
#
#  A song is whatever subset of streams applies to it. Solo piano needs
#  four. A harsh-noise track needs :mix and :residual and nothing else.
#  Coverage gaps cost compression and editability — never the ability
#  to encode the song.
# =============================================================================
