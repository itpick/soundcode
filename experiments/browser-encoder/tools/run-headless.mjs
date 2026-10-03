// Drive the page headlessly with Playwright and encode a list of clips.
//
//   PLAYWRIGHT=/path/to/node_modules/playwright CHROME=/path/to/chrome \
//   node tools/run-headless.mjs --url http://127.0.0.1:8000/ --out OUTDIR clip1.mp3 clip2.mp3 ...
//
// Options: --backend auto|wasm  --model <hf id>  --no-lrclib  --headless-new
//          --gpu (pass --enable-unsafe-webgpu)  --meta DIR (read @title/@artist
//          from DIR/<clip>.sc, standing in for a user typing them).
// Clips are served to the page from their absolute paths through a route, so
// the audio never has to be copied into the repo.

import fs from 'node:fs';
import path from 'node:path';

const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf(k); if (i < 0) return d; const v = argv[i + 1]; argv.splice(i, 2); return v; };
const flag = k => { const i = argv.indexOf(k); if (i < 0) return false; argv.splice(i, 1); return true; };
const url = opt('--url', 'http://127.0.0.1:8000/');
const outDir = opt('--out', 'out');
const backend = opt('--backend', 'auto');
const model = opt('--model', undefined);
const metaDir = opt('--meta', undefined);
const tag = opt('--tag', '');
const extra = JSON.parse(opt('--opts', '{}'));
const noLrclib = flag('--no-lrclib');
const headlessNew = flag('--headless-new');
const gpu = flag('--gpu');
const clips = argv;

const { chromium } = await import(path.join(process.env.PLAYWRIGHT, 'index.mjs'));
const args = [];
if (headlessNew) args.push('--headless=new');
if (gpu) args.push('--enable-unsafe-webgpu');
const browser = await chromium.launch({ executablePath: process.env.CHROME, args });
const page = await browser.newPage();
page.setDefaultTimeout(0);
page.on('console', m => { const t = m.text(); if (!t.includes('GL Driver')) console.log(`[page] ${t}`); });
page.on('pageerror', e => console.log(`[pageerror] ${e}`));
await page.route('**/__clips/**', async route => {
  const name = decodeURIComponent(new URL(route.request().url()).pathname.split('/__clips/')[1]);
  const file = clips.find(c => path.basename(c) === name);
  if (!file) return route.fulfill({ status: 404 });
  await route.fulfill({ status: 200, body: fs.readFileSync(file), contentType: 'audio/mpeg' });
});
await page.goto(url);
await page.waitForFunction(() => window.scReady === true);
const env = await page.evaluate(async () => {
  let adapter = null;
  if (navigator.gpu) { try { const a = await navigator.gpu.requestAdapter(); adapter = a ? `${a.info?.vendor}/${a.info?.architecture}` : null; } catch { /* none */ } }
  return { ua: navigator.userAgent, isolated: self.crossOriginIsolated, cores: navigator.hardwareConcurrency, webgpu: !!navigator.gpu, adapter };
});
console.log('env', JSON.stringify(env));
fs.mkdirSync(outDir, { recursive: true });

const all = [];
for (const clip of clips) {
  const name = path.basename(clip);
  const stem = name.replace(/\.[^.]+$/, '');
  let title, artist;
  if (metaDir && fs.existsSync(path.join(metaDir, `${stem}.sc`))) {
    const txt = fs.readFileSync(path.join(metaDir, `${stem}.sc`), 'utf8');
    title = txt.match(/^@title\s+"([^"]*)"/m)?.[1];
    artist = txt.match(/^@artist\s+"([^"]*)"/m)?.[1];
  }
  const before = await page.evaluate(() => performance.getEntriesByType('resource').length);
  const t0 = Date.now();
  const res = await page.evaluate(async ([n, o]) => {
    const r = await window.scEncodeUrl(`__clips/${encodeURIComponent(n)}`, o);
    return { sc: r.sc, stages: r.stages, backends: r.backends, summary: r.summary, downloads: r.downloads };
  }, [name, { title, artist, backend, lrclib: !noLrclib, ...(model ? { whisperModel: model } : {}), ...extra }]);
  const wall = (Date.now() - t0) / 1000;
  fs.writeFileSync(path.join(outDir, `${stem}.sc`), res.sc);
  const rec = { clip: name, wall_s: wall, env, backends: res.backends, summary: res.summary, stages: res.stages, downloads_cumulative: res.downloads, resources_before: before };
  fs.writeFileSync(path.join(outDir, `${stem}${tag}.json`), JSON.stringify(rec, null, 1));
  console.log(`done ${name} in ${wall.toFixed(1)} s`);
  all.push(rec);
}
await browser.close();
