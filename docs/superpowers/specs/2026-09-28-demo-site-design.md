# Demo site: sound to code — design

Date: 2026-09-28. Status: design approved in conversation ("go, can always iterate").

The user asked for a GitHub page, prepared but **not deployed**. It shows examples going from each 30 s original clip to the rebuild made from sound code, and shows the true compression of turning the sound into code, with and without gzip.

## Content policy

The page uses only the four Nine Inch Nails clips from *The Slip* (2008): Discipline, Lights in the Sky, 999999 and Corona Radiata.
- NIN released *The Slip* under **Creative Commons BY-NC-SA 3.0**. The page credits it, links the license, and states that the rebuilds are non-commercial derivatives shared under the same license.
- River (Jordan Feliz) is copyrighted, so it stays private and never appears in `site/`. The build script has no option to add it.

## The page (`site/index.html`)

**Hero**
- One sentence: a song goes in, text code comes out, and the song is rebuilt from that code.
- The headline totals across the four clips: original WAV MB → `.sc` KB → gzipped `.sc` KB, with the ratios.

**One card per song**
- **Players:** "Original (30 s)" and "Rebuilt from code", next to each other; stacked on phones.
- **Size strip.** Horizontal bars on a log scale, each with bytes and the ratio to the WAV:
  1. the original as 16-bit stereo 44.1 kHz WAV, the uncompressed baseline;
  2. the original MP3, the real file we started from;
  3. the `.sc` as raw text;
  4. the `.sc` gzipped (level 9).
- **Borrowed from the original.** The rebuild is not code alone, and the card says so:
  - the drum kit folder (hits cut from the song's drum stem), when the `.sc` has `meta kit=`;
  - the singer's voice sample SoulX clones from (`PROMPT_S` seconds at 24 kHz, 16-bit mono), when the rebuild has vocals.

  The card shows a second, honest ratio: WAV ÷ (gzipped `.sc` + borrowed audio).
- **The code.** A `<details>` block with the full `.sc` text, monospace and scrollable, plus a download link.
- **What's in it.** Tempo, key, the instrument list (from `:instruments`/`inst=`), and the word count of `:text.vox`.

**Footer**
- The CC credit and license link.
- "Private research project; rebuilds made with open models" and the list of models (audio-separator, tsumugi, torchcrepe, faster-whisper, GeneralUser GS, SoulX-Singer).
- The build date.

**Look:** plain HTML/CSS/JS, one file plus `data.json` and `media/`. Light and dark themes via color tokens. No external scripts; a Google Font at most.

## Build (`scripts/build_site.py`)

Songs are a constant list: slug, title and source clip under `audio/test/`. Every song renders `--with-vocals`; a clip with no vocal (999999 is instrumental) renders instruments-only through the existing NoVocalError path, and then gets `voice_bytes = 0`.

For each song:
1. **Code.**
   - Encode: `soundcode encode <clip> --title --artist "Nine Inch Nails" -o out/site/<slug>.sc`.
   - Render: `soundcode render --with-vocals -o out/site/<slug>.wav`.
   - Both are skipped when the output is newer than its input, and `--force` redoes them. These call the existing CLI; the site script adds no audio logic.
2. **Media.**
   - Copy the original MP3 to `site/media/<slug>-original.mp3`.
   - Encode the rebuild to `site/media/<slug>-rebuild.mp3`: ffmpeg, falling back to lameenc, at 192 kb/s, the same as the originals.
   - Copy the `.sc` to `site/media/<slug>.sc`.
3. **Measure.** Everything is computed from the files; no number is typed by hand.
   - `wav_bytes = frames × 2 ch × 2 bytes + 44` from the decoded original.
   - `mp3_bytes` is the file size.
   - `sc_bytes` and `sc_gz_bytes` come from `gzip.compress(level=9)`.
   - `kit_bytes` is the sum of the kit folder's files.
   - `voice_bytes = PROMPT_S × 24000 × 2`, taken from `soulx.PROMPT_S`/`soulx.SR`, but only when the `.sc` has sung words in `:text.vox`; otherwise 0.
   - Ratios are rounded to whole numbers.
   - Summary fields come from the `.sc` via the parser: tempo, key, instruments, words.
4. Write `site/data.json`: the songs, their sizes and ratios, the totals and the build date.

`index.html` fetches `data.json` and renders the cards. The measurement lives in one place, the script, and the page only displays it.

## Not deployed

- `site/` is committed to the repo (media included; all CC-licensed). Nothing is pushed or published.
- Preview it locally with `python -m http.server -d site`.
- A `site/README.md` records how to deploy later. GitHub Pages on a *private* repo needs a paid plan, so the free route is pushing `site/` alone to a public repo and enabling Pages there.

## Testing

Unit tests (`tests/test_site.py`), with no models needed:
- sizes and ratios for a synthetic `.sc` and WAV;
- kit bytes summed and voice bytes zero when the `.sc` has no vocal;
- the summary fields read from a small `.sc`;
- `data.json` has the shape the page reads;
- a guard that every song's source clip is one of the four CC clips (River can never be added by mistake).

Manual checks:
- a real build of all four songs;
- open the page locally at desktop and phone widths, and check that both players play.
