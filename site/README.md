# Demo site

This is the demo page for sound-to-code: it plays each original clip next to the rebuild made
from its `.sc` code, and shows how small that code is. It is **not deployed** — everything here
lives only in this repo, viewed locally.

## Build

```
.venv/bin/python scripts/build_site.py
```

This encodes and renders the clips, makes the MP3s, and writes `data.json` and `media/`.

## Preview

```
python -m http.server -d site
```

Then open the printed URL. Opening `index.html` directly (`file://`) will not work — the page
fetches `data.json`, which browsers block for local files without a server.

## Deploy later

GitHub Pages on a *private* repo needs a paid plan. The free route: copy `site/` into a public
repo and enable Pages from its root.

## Licensing

The page uses only the CC BY-NC-SA Slip clips; no other songs are included.
