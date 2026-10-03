import { encode } from './pipeline.js';

const $ = id => document.getElementById(id);
let file = null, lastSc = null, lastName = 'out.sc';

const logEl = $('log');
function log(m) { logEl.textContent += m + '\n'; logEl.scrollTop = logEl.scrollHeight; console.log(m); }

function pick(f) {
  if (!f) return;
  file = f;
  $('drop').textContent = `${f.name} (${(f.size / 1e6).toFixed(1)} MB)`;
  $('go').disabled = false;
  logEl.textContent = 'Ready.\n';
}
$('drop').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', e => pick(e.target.files[0]));
$('drop').addEventListener('dragover', e => { e.preventDefault(); $('drop').classList.add('over'); });
$('drop').addEventListener('dragleave', () => $('drop').classList.remove('over'));
$('drop').addEventListener('drop', e => { e.preventDefault(); $('drop').classList.remove('over'); pick(e.dataTransfer.files[0]); });

function showTimings(res) {
  const rows = res.stages.map(s => `<tr><td>${s.name}</td><td class="n">${(s.ms / 1000).toFixed(2)} s</td><td class="${s.ok ? 'ok' : 'bad'}">${s.ok ? 'ok' : 'failed'}</td></tr>`);
  const total = res.stages.reduce((a, s) => a + s.ms, 0);
  rows.push(`<tr><th>total</th><th class="n">${(total / 1000).toFixed(2)} s</th><th></th></tr>`);
  $('timings').innerHTML = `<tbody>${rows.join('')}</tbody>`;
}

async function run(bytes, name, opts) {
  const res = await encode(bytes, { fileName: name, log, ...opts });
  lastSc = res.sc; lastName = name.replace(/\.[^.]+$/, '') + '.sc';
  $('out').value = res.sc;
  $('dl').disabled = false;
  showTimings(res);
  return res;
}

$('go').addEventListener('click', async () => {
  $('go').disabled = true;
  logEl.textContent = '';
  try {
    await run(await file.arrayBuffer(), file.name, {
      title: $('title').value.trim() || undefined, artist: $('artist').value.trim() || undefined,
      backend: $('backend').value, whisperModel: $('model').value, lrclib: $('lrclib').checked,
    });
  } catch (e) { log(`failed: ${e}`); }
  $('go').disabled = false;
});

$('dl').addEventListener('click', () => {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([lastSc], { type: 'text/plain' }));
  a.download = lastName; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

// Headless entry point (tools/run-headless.mjs): encode a same-origin URL.
window.scEncodeUrl = async (url, opts = {}) => {
  logEl.textContent = '';
  const bytes = await (await fetch(url)).arrayBuffer();
  const name = opts.fileName || decodeURIComponent(url.split('/').pop());
  return run(bytes, name, opts);
};
window.scReady = true;
