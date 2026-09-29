# Demo site

This is the demo page for sound-to-code. It plays each original clip next to the rebuild made from its `.sc` code, and shows how small that code is.

**Deployed** with GitHub Pages at https://itpick.github.io/soundcode/ by `.github/workflows/pages.yml`, which publishes only this `site/` folder on every push to `main` that changes it. The repo and the page are public. Search engines are asked not to index the page (`robots.txt`); delete that file to allow indexing.

## Build

`build()` needs each clip's separated stems in `out/stems/<slug>/` first -- it reads the
original parts from there, and a sung song's voice reference comes from `lead_vocals.wav`
in that folder. `encode` does not create it, so run `separate` on every clip before building:

```
.venv/bin/python -m soundcode.cli separate audio/test/discipline-30s.mp3
.venv/bin/python -m soundcode.cli separate audio/test/lights_in_the_sky-30s.mp3
.venv/bin/python -m soundcode.cli separate audio/test/999999-30s.mp3
.venv/bin/python -m soundcode.cli separate audio/test/corona_radiata-30s.mp3
.venv/bin/python scripts/build_site.py
```

This encodes and renders the clips, makes the MP3s, and writes `data.json` and `media/`.

## Preview

```
python -m http.server -d site
```

Then open the printed URL. Opening `index.html` directly (`file://`) will not work — the page
fetches `data.json`, which browsers block for local files without a server.

## Alternative: private and free (Cloudflare Pages + Cloudflare Access)

Use this instead of GitHub Pages if the page should be private. GitHub can't do that for free: access-controlled Pages needs GitHub Enterprise Cloud with an organization, and on every other plan a Pages site is public even from a private repo. The free route is Cloudflare. Pages hosts the site, and Access (free for up to 50 users) puts a login in front of it.

1. **Create the Pages project.**
   - Make a free Cloudflare account.
   - In *Workers & Pages → Create → Pages → Connect to Git*, pick `itpick/soundcode` and grant access to just this repo.
   - Build settings:
     - Framework preset: **None**
     - Build command: *(leave empty)*
     - Build output directory: **`site`**
     - Production branch: `main`

   Every push to `main` redeploys. The site is only the committed `site/` folder, so run `scripts/build_site.py` and commit before pushing.
2. **Lock it down before sharing the URL.**
   - Go to *Zero Trust → Access → Applications → Add → Self-hosted*.
   - Application domain: the `*.pages.dev` hostname, **and** `*.<project>.pages.dev`, which covers preview deploys.
   - Add a policy: Action **Allow**, Include **Emails**, listing the people who have repo access.
   - Login methods: One-time PIN (email). You can also add GitHub as an identity provider.
   - The allow-list is kept by hand; it is not synced with the repo's collaborators.
3. **Check it.** Open the URL in a private window: you should get the Cloudflare login, not the page.

What's already set up here:
- `_headers` sets `noindex`, a strict same-origin Content-Security-Policy, and caching for `media/`.
- `robots.txt` disallows crawling.
- Each file is well under Pages' 25 MiB per-file limit, and the whole site is about 25 MB.

## Licensing

The page uses only the CC BY-NC-SA Slip clips; no other songs are included.
