# =============================================================================
#  soundcode  v0.1   —   .sc  "song code"
# =============================================================================
#
#  GRAMMAR REFERENCE
#
#    @key value              header directive (global, one per line)
#    %pat name { ... }       named reusable pattern, scoped to current track
#    %residual mode          acoustic-residual layer (v1: none)
#    #name  first-last       section, bar range inclusive
#    :name  ~"timbre"        track declaration + freeform timbre tag
#    bars A-B  pat           bind a pattern to a bar range
#    !auto target  a -> b    parameter automation between two time points
#
#  TIME       bar:beat   1-indexed, beat is fractional  (9:2.5 = bar 9, beat 2.5)
#             inside a %pat, bare beats are relative to that bar
#
#  PITCH      scientific notation, C4 = middle C = MIDI 60
#             drum tracks use named voices instead of pitches
#
#  EVENT      <time> <pitch|voice> [dur] [vel] ["syllable"]
#             dur in beats, vel 0-127, both optional (inherit track default)
#             `-` as syllable = melisma, continues the previous word
#             `|` separates events on one line, purely cosmetic
#
#  Anything after `#` at line start is a comment. Inline `#` is NOT a comment
#  (bar ranges use it), so keep comments on their own lines.
#
# =============================================================================

@title      "Signal Lost"
@artist     "soundcode control track"
@source     control/signal-lost-render.wav
@license    CC0-1.0
@duration   30.000
@sr         44100
@tempo      128
@meter      4/4
@key        Am

@style      "mid-tempo industrial pop, dry compressed drums, distorted
             saw bass, clean electric piano pad, close-mic male vocal
             with doubled female backing, moderate plate reverb"

@mix        "drums forward and center, bass mono below 120Hz, vocal
             center with 12ms slapback, pad wide stereo, light
             sidechain duck on pad from kick"

# v1 emits pure symbols — no learned acoustic tokens.
# Setting this to `semanticodec:0.31kbps` later adds the residual layer
# without changing anything above it.
%residual   none


# --- structure ---------------------------------------------------------------

#intro    1-4
#verse    5-12
#chorus   13-16


# --- harmony -----------------------------------------------------------------
# One chord symbol per bar. Consumed by the decode bridge for the ACE-Step
# prompt and used by the encoder to sanity-check transcribed bass notes.

:harmony
bars 1-4    Am | F | C | G
bars 5-8    Am | F | C | G
bars 9-12   Am | F | C | G
bars 13-16  F  | C | G | Am


# --- drums -------------------------------------------------------------------

:drums  ~"dry compressed kit, tight gated snare, closed hats, no room"

%pat basic {
  1.0 kick  104 | 1.5 hat 58 | 2.0 snare 118 | 2.5 hat  55
  3.0 kick   98 | 3.25 kick 74 | 3.5 hat 60 | 4.0 snare 120 | 4.5 hat 56
}

%pat sparse {
  1.0 kick  100 | 2.0 snare 112
  3.0 kick   96 | 4.0 snare 114
}

%pat fill {
  1.0 kick  104 | 1.5 hat 58 | 2.0 snare 118
  3.0 tom1  106 | 3.5 tom1 101 | 4.0 tom2 110 | 4.5 tom2 104
}

bars 1-2    sparse
bars 3      basic
bars 4      fill
bars 5-7    basic
bars 8      basic
bars 9-11   basic
bars 12     fill
bars 13-15  basic
bars 16     fill

# one-off accents layered on top of the patterns above
3:1.0   crash 112
13:1.0  crash 121
16:1.0  crash 118


# --- bass --------------------------------------------------------------------

:bass  ~"distorted saw, heavy lowpass ~400Hz, slight pitch drift, mono"

%pat root8 {
  1.0 . 0.5 108 | 1.5 . 0.5 92 | 2.5 . 0.5 96 | 3.0 . 1.0 104
}

# `.` in a pattern means "the root of this bar's chord, one octave below
# the notated register" — resolved against :harmony at parse time.

bars 1-2    root8
bars 3-4    root8
bars 5-12   root8

# chorus bass is written out explicitly — it departs from the pattern
13:1.0  F1  1.5 112 | 13:2.5 F1 0.5 94 | 13:3.0 F1 1.0 106
14:1.0  C2  1.5 110 | 14:2.5 C2 0.5 92 | 14:3.0 E2 1.0 100
15:1.0  G1  1.5 112 | 15:2.5 G1 0.5 94 | 15:3.0 B1 1.0 104
16:1.0  A1  3.0 116 | 16:4.0 A1 1.0 98


# --- pad ---------------------------------------------------------------------

:pad  ~"clean electric piano through chorus, wide stereo, long release"

# whole-bar triads, one per chord change
1:1.0   A3 4.0 62 | 1:1.0 C4 4.0 58 | 1:1.0 E4 4.0 60
2:1.0   F3 4.0 62 | 2:1.0 A3 4.0 58 | 2:1.0 C4 4.0 60
3:1.0   C4 4.0 64 | 3:1.0 E4 4.0 60 | 3:1.0 G4 4.0 62
4:1.0   G3 4.0 64 | 4:1.0 B3 4.0 60 | 4:1.0 D4 4.0 62

bars 5-12   like 1-4
bars 13-16  like 1-4  transpose 0  vel -8

# `like` reuses an earlier bar range's events, remapped onto the current
# harmony. `transpose` in semitones, `vel` as a signed offset.


# --- lead vocal --------------------------------------------------------------

:vox  ~"male, close-mic, slight tube distortion, 12ms slapback, center"

# verse — bars 5-12
5:1.0    A3 0.5  96 "I"
5:2.0    C4 0.5  98 "was"
5:3.0    C4 0.5  97 "a"
5:4.0    D4 0.5 101 "sig"
5:4.5    D4 0.5  99 "nal"
6:1.0    C4 0.5  96 "in"
6:2.0    A3 0.5  94 "the"
6:3.0    A3 1.5  99 "stat"
6:4.5    G3 0.5  92 "ic"

7:1.0    A3 0.5  97 "you"
7:2.0    C4 0.5  99 "were"
7:3.0    C4 0.5  97 "the"
7:4.0    E4 0.5 104 "hand"
7:4.5    D4 0.5 100 "that"
8:1.0    C4 0.5  98 "turned"
8:2.0    A3 0.5  95 "the"
8:3.0    A3 1.5 100 "di"
8:4.5    G3 0.5  93 "al"

9:1.0    C4 0.5 100 "now"
9:2.0    D4 0.5 102 "ev"
9:2.5    D4 0.5 100 "ery"
9:3.0    E4 0.5 105 "word"
9:4.0    D4 1.0 101 "I"
10:1.0   C4 0.5  99 "ev"
10:1.5   C4 0.5  97 "er"
10:2.0   A3 2.0 103 "said"

11:1.0   C4 0.5 101 "comes"
11:2.0   C4 0.5  99 "back"
11:3.0   D4 0.5 102 "to"
11:4.0   E4 1.0 106 "me"
12:1.0   D4 0.5 102 "as"
12:2.0   C4 0.5 100 "num"
12:2.5   A3 2.5 104 "bers"
12:4.5   A3 0.5  88 -

# chorus — bars 13-16
13:1.0   F4 0.5 112 "Hold"
13:2.0   E4 0.5 110 "the"
13:3.0   C4 2.0 114 "line"

14:1.0   E4 0.5 112 "hold"
14:2.0   D4 0.5 110 "the"
14:3.0   C4 2.0 113 "line"

15:1.0   D4 0.5 111 "no"
15:1.5   D4 0.5 109 "thing"
15:2.0   E4 0.5 112 "here"
15:3.0   D4 0.5 110 "is"
15:4.0   B3 1.0 108 "lost"

16:1.0   A3 3.5 106 "ohh"
16:2.0   .  -   -   -


# --- backing vocals ----------------------------------------------------------

:bvox  ~"female, doubled and detuned +/-8 cents, wide, heavy plate reverb"

# sustained pad-like vowels under the chorus only
13:1.0   A4 4.0 74 "ah"
14:1.0   G4 4.0 74 "ah"
15:1.0   G4 4.0 76 "ah"
16:1.0   A4 4.0 78 "ah"

# answer-phrase doubling the chorus hook an octave up
13:3.0   C5 2.0 82 "line"
14:3.0   C5 2.0 82 "line"


# --- automation --------------------------------------------------------------

!auto  pad.cutoff      1:1.0  0.25 -> 4:4.0  0.85
!auto  pad.cutoff      5:1.0  0.60 -> 12:4.0 0.60
!auto  vox.reverb_send 5:1.0  0.18 -> 12:4.0 0.28
!auto  vox.reverb_send 13:1.0 0.42 -> 16:4.0 0.55
!auto  master.drive    13:1.0 0.30 -> 16:4.0 0.48


# =============================================================================
#  30.000s  @  128bpm  ·  16 bars  ·  6 tracks
#  target: ~2.5 KB raw  →  ~0.67 kbps   (vs 128kbps MP3 = ~480 KB)
# =============================================================================
