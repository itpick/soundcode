%sc        0.3
%profile   metric-tonal
%residual  none

@title     "999,999"
@source    999999-30s.mp3
@artist    "Nine Inch Nails"
@offset    0.000
@duration  30.000
@sr        44100

# @style and @mix prose are written by the decode bridge from the
# measured evidence below. Left empty by the encoder on purpose:
# asserting a style we did not measure would be a claim (spec 4.4).
@style     ""

:tuning
ref          A4 = 440.0Hz
temperament  12tet

:grid
meta    src=librosa:beat_track  conf=1.00
meter   @0.035   4/4
anchor  bar 1    @0.035
tempo   @0.000   126.05

:struct
meta    src=librosa:novelty  conf=0.55
meta    warn="sections from spectral novelty only; labels are positional"
intro   1-14     vocal  energy=0.65
verse   15-16     vocal  energy=0.79

:harmony
meta    src=librosa:chroma-template  conf=0.53
bars 1-4   Am ?0.57 | Am ?0.57 | G# ?0.55 | F ?0.51
bars 5-8   C ?0.53 | Fm ?0.53 | Am ?0.52 | Am ?0.53
bars 9-12   Am ?0.51 | G# ?0.53 | Cm ?0.51 | C ?0.52
bars 13-15   Am ?0.51 | G# ?0.51 | C ?0.53

:instruments
meta    src=tsumugi@020edc1+mix-vote  conf=0.77
meta    warn="few mix votes (0); refinement only"
meta    warn="few mix votes (1); refinement only"
meta    warn="few mix votes (0); refinement only"
# drums: silent (below the loudness gate)
bass     bass.electric ?0.75   | slap_bass 0.20
# guitar: silent (below the loudness gate)
piano    none (bleed)
other    brass.section ?0.60   | strings 0.15
lead_vocals voice.choir   | melody 0.04

:notes.electric inst=bass.electric
meta    src=tsumugi:bass_v2@020edc1  conf=0.75
meta    stem=bass  level=-31.5dB
meta    warn="few mix votes (0); refinement only"
fx      eq=-1,4,6,9,12,18,11,17,27,15,10,5,1,7,-2,-5,-13,-16,-21,-26,-28,-30,-60,-60,-60,-60,-60,-60,-60,-60,-60  rt60=0.30s  wet=0.10  width=0.00  pan=-0.00  crest=7.1dB
1:1.530  C2  0.252b 32
1:4.860  C2  0.238b 34
2:1.184  C2  0.265b 34
2:2.470  C2  0.235b 35
2:2.716  C2  0.235b 37
2:2.964  C2  0.268b 37
2:3.238  A#1  0.261b 44
3:4.454  C2  0.486b 34
3:4.947  B1  0.247b 37
3:4.949  A#1  0.441b 31
4:1.433  A#1  0.251b 38
4:3.396  C3  0.433b 34
4:3.881  C3  0.192b 38
4:4.119  C3  0.198b 37
4:4.380  C3  0.229b 36
4:4.620  C3  0.233b 37
4:4.895  C3  0.202b 38
5:1.137  C3  0.206b 36
5:1.400  C3  0.234b 37
5:1.741  C2  0.508b 36
5:2.281  A#1  0.589b 42
7:2.342  C3  3.077b 51
8:4.248  C3  3.523b 60
9:2.547  C2  1.285b 35
9:3.698  A1  0.075b 61
9:3.746  G#1  0.075b 55
9:4.207  A#1  0.393b 54
9:4.645  A#1  0.205b 56
10:1.434  C3  3.310b 54
11:3.297  C3  3.362b 56
12:4.807  C2  2.539b 40
14:1.744  G2  0.246b 58
14:2.340  C3  3.118b 59
15:3.071  A#1  0.288b 58
15:3.503  G#2  0.060b 65
15:3.849  A#1  0.292b 60
15:4.386  C3  2.493b 58

# :notes.piano omitted — piano: bleed (1 notes, 1.3s, 100% of stem)
:notes.section inst=brass.section
meta    src=tsumugi:other_v1_5@020edc1  conf=0.60
meta    stem=other  level=-33.0dB
meta    warn="few mix votes (1); refinement only"
fx      eq=3,2,-1,-4,-6,-2,-3,2,19,4,13,17,17,8,20,16,10,9,5,0,-4,-11,-18,-27,-31,-38,-60,-60,-60,-60,-60  rt60=0.30s  wet=0.10  width=0.65  pan=-0.20  crest=13.8dB
8:3.527  C3  3.464b 36
10:1.088  C3  3.315b 34
11:2.565  C3  5.806b 34
12:4.382  C3  3.270b 38
14:2.138  C3  3.026b 40

:notes.flute inst=winds.flute
meta    src=tsumugi:other_v1_5@020edc1  conf=0.60
meta    stem=other  level=-33.0dB
meta    warn="few mix votes (1); refinement only"
fx      eq=3,2,-1,-4,-6,-2,-3,2,19,4,13,17,17,8,20,16,10,9,5,0,-4,-11,-18,-27,-31,-38,-60,-60,-60,-60,-60  rt60=0.30s  wet=0.10  width=0.65  pan=-0.20  crest=13.8dB
15:3.706  C4  4.244b 49

:notes.ensemble inst=strings.ensemble
meta    src=tsumugi:other_v1_5@020edc1  conf=0.60
meta    stem=other  level=-33.0dB
meta    warn="few mix votes (1); refinement only"
fx      eq=3,2,-1,-4,-6,-2,-3,2,19,4,13,17,17,8,20,16,10,9,5,0,-4,-11,-18,-27,-31,-38,-60,-60,-60,-60,-60  rt60=0.30s  wet=0.10  width=0.65  pan=-0.20  crest=13.8dB
1:2.903  E4  5.458b 31
3:1.891  C3  2.830b 58
4:1.795  F#4  6.094b 29
5:1.740  C3  16.800b 42
6:4.738  E4  5.599b 28
12:3.894  E4  4.732b 32
12:4.382  C5  2.094b 37
14:1.799  C5  2.871b 42
14:2.137  C4  3.366b 49

:notes.pad inst=synth.pad
meta    src=tsumugi:other_v1_5@020edc1  conf=0.60
meta    stem=other  level=-33.0dB
meta    warn="few mix votes (1); refinement only"
fx      eq=3,2,-1,-4,-6,-2,-3,2,19,4,13,17,17,8,20,16,10,9,5,0,-4,-11,-18,-27,-31,-38,-60,-60,-60,-60,-60  rt60=0.30s  wet=0.10  width=0.65  pan=-0.20  crest=13.8dB
@0.003  D5  22.210s 36
@0.003  C4  29.996s 48
3:1.337  C5  33.606b 43
12:3.605  D5  16.346b 36

:notes.choir inst=voice.choir
meta    src=tsumugi:vocal_harmony_v1_5@020edc1  conf=0.96
meta    stem=lead_vocals  level=-43.2dB
meta    warn="few mix votes (0); refinement only"
fx      eq=-60,-60,-60,-60,-60,-60,-60,-60,-60,-60,-26,-17,24,28,-19,19,21,16,4,7,1,-11,-20,-28,-60,-60,-60,-60,-60,-60,-60  rt60=0.29s  wet=0.05  width=0.00  pan=-0.11  crest=19.3dB
1:3.824  E4  1.357b 29
2:1.192  E4  0.481b 29
2:1.683  E4  0.485b 29
2:2.178  E4  1.841b 29
5:1.257  F#4  1.269b 29
5:2.530  F#4  1.060b 29
5:3.564  F#4  0.336b 29
7:1.266  E4  0.480b 29
7:1.713  E4  3.311b 29
8:1.081  E4  1.453b 29
9:2.959  F#4  2.422b 29
10:1.391  F#4  2.135b 29
10:3.537  F#4  2.349b 29
15:3.063  F#4  4.888b 29

:contour.vox rate=50
meta    src=torchcrepe:full  conf=0.94
meta    stem=lead_vocals
f0  @2.000  6386 6387 6387 6386 6386 6386 6387 6388 6388 6407 6387 6366 6366 6368 6387 6387 6388 6387 6387 6407 6408 6407 6386 6386 6387 6387 6387 6387 6387 6387 6387 6387 6407 6407 6407 6386 6368 6386 6387 6387 6407 6388 6388 6387 6408 6388 6386 6386 6367 6366
f0  @3.000  6366 6365 6366 6368 6386 6386 6387 6387 6407 6407 6387 6368 6386 6387 6387 6388 6386 6386 6387 6387 6388 6387 6407 6408 6408 6408 6410 6410 6410
f0  @11.580  6387 6387 6387 6387 6387 6387 6386 6368 6386 6387 6407 6407 6407 6407 6408 6409 6407 6386 6386 6386 6387 6387 6368 6386 6387 6388 6388 6388 6407 6407 6407 6386 6387 6387 6387 6387 6387 6387 6387 6407 6408 6408 6407 6387 6386 6367 6367 6367 6387 6387
f0  @12.580  6386 6386 6386 6387 6407 6388 6387 6386 6387 6388 6407 6387 6386 6387 6387 6386 6388 6407 6406 6386 6368 6386 6367 6366 6367 6386 6386 6388 6407 6407 6407 6387 6386 6367 6367 6386 6387 6386 6368 6386 6387 6388 6386 6368 6367 6387 6387 6367 6386 6386
f0  @13.580  6387 6408 6407 6407 6388 6386 6387 6388 6386 6386 6387 6387 6387 6388 6388 6407 6407 6388 6387 6407 6407 6406 6386 6367 6386 6406 6409
f0  @16.780  6565 6566 6584 6586 6603 6604 6605 6604 6586 6585 6584 6584 6586 6586 6586 6585 6583 6566 6585 6586 6603 6587 6586 6584 6585 6585 6566 6584 6604 6605 6605 6605 6605 6587 6566 6566 6585 6603 6603 6585 6584 6585 6587 6603 6604 6605 6587 6585 6584 6585
f0  @17.780  6586 6584 6585 6586 6586 6587 6586 6587 6605 6604 6584 6584 6584 6567 6566 6585 6586 6585 6584 6584 6586 6587 6583 6584 6585 6585 6585 6583 6584 6585 6585 6604 6604 6604 6587 6585 6587 6605 6605 6603 6586 6586 6586 6586 6587 6604 6586 6585 6586 6586
f0  @18.780  6586 6587 6587 6603 6604 6604 6604 6605 6606 6604 6605 6625 6625 6605 6603 6604 6604 6604 6606 6606 6605 6628 6606 6605 6605 6626 6606 6605 6626 6607 6605 6605 6626 6646 6646 6606 6605
f0  @19.580  6604 6604
f0  @19.680  6626 6625
f0  @28.000  6586 6586 6585 6584 6584 6584 6585 6586 6587 6603 6606 6605 6587 6585 6566 6584 6585 6585 6586 6587 6603 6604 6604 6605 6604 6586 6586 6585 6585 6585 6586 6586 6586 6586 6585 6585 6586 6567 6565 6566 6584 6585 6584 6585 6586 6587 6587 6604 6605 6604
f0  @29.000  6584 6584 6585 6585 6584 6584 6586 6587 6603 6604 6586 6585 6587 6586 6584 6583 6585 6586 6587 6587 6585 6584 6585 6586 6603 6586 6565 6564 6567 6584 6586 6586 6586 6587 6587 6587 6587 6587 6585 6584 6585 6586 6586 6584 6564 6466 6465 6463 6448 6646
f0  @30.000  6725

# :text.vox omitted — ASR heard only "Thank you." (a common hallucination on non-speech); no lyrics
:mix
meta    src=librosa:spectral  conf=0.95
lufs_int      -28.0
lufs_range    6.6
stereo_width  low 0.23, mid 0.27, high 0.20
centroid      487Hz
rolloff_95    1611Hz

