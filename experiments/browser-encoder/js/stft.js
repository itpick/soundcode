// A small radix-2 STFT (Hann window, magnitude only). Used for the spectral
// scalars in :mix and the band-flux drum onsets; essentia's per-frame calls
// cross the JS/WASM boundary twice per frame, which is slower for these.

function fftInPlace(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
        const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
}

// Returns {mags: Float32Array[] (nFrames x (nfft/2+1)), hop, nfft, sr}.
// Frames are centred (librosa's default), padding with zeros.
export function stft(x, sr, nfft = 2048, hop = 512) {
  const win = new Float32Array(nfft);
  for (let i = 0; i < nfft; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / nfft);
  const nFrames = 1 + Math.floor(x.length / hop);
  const half = nfft / 2;
  const mags = new Array(nFrames);
  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  for (let f = 0; f < nFrames; f++) {
    const c = f * hop - half;
    for (let i = 0; i < nfft; i++) {
      const k = c + i;
      re[i] = (k >= 0 && k < x.length ? x[k] : 0) * win[i];
      im[i] = 0;
    }
    fftInPlace(re, im);
    const m = new Float32Array(half + 1);
    for (let i = 0; i <= half; i++) m[i] = Math.hypot(re[i], im[i]);
    mags[f] = m;
  }
  return { mags, hop, nfft, sr };
}

// "Poor man's separation": keep the time-frequency bins where left and right
// agree (centre-panned material, which in most mixes is the lead vocal, plus
// kick, snare and bass), band-limited to the voice range. Not a model;
// a cheap pre-filter for Whisper and the melody tracker.
export function centreVocal(L, R, sr, { nfft = 2048, hop = 512, power = 6, lo = 120, hi = 7000 } = {}) {
  const win = new Float64Array(nfft);
  for (let i = 0; i < nfft; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / nfft);
  const out = new Float32Array(L.length), norm = new Float32Array(L.length);
  const lr = new Float64Array(nfft), li = new Float64Array(nfft), rr = new Float64Array(nfft), ri = new Float64Array(nfft);
  const b0 = Math.floor(lo * nfft / sr), b1 = Math.ceil(hi * nfft / sr);
  for (let start = -nfft / 2; start < L.length; start += hop) {
    for (let i = 0; i < nfft; i++) {
      const k = start + i, ok = k >= 0 && k < L.length;
      lr[i] = ok ? L[k] * win[i] : 0; rr[i] = ok ? R[k] * win[i] : 0; li[i] = 0; ri[i] = 0;
    }
    fftInPlace(lr, li); fftInPlace(rr, ri);
    for (let b = 0; b < nfft; b++) {
      const f = b <= nfft / 2 ? b : nfft - b;
      let m = 0;
      if (f >= b0 && f <= b1) {
        const cr = lr[b] * rr[b] + li[b] * ri[b], ci = li[b] * rr[b] - lr[b] * ri[b];
        const e = lr[b] ** 2 + li[b] ** 2 + rr[b] ** 2 + ri[b] ** 2;
        const sim = e > 1e-12 ? 2 * Math.hypot(cr, ci) / e : 0;     // 1 = identical L and R
        m = Math.pow(sim, power);
      }
      lr[b] = 0.5 * (lr[b] + rr[b]) * m; li[b] = 0.5 * (li[b] + ri[b]) * m;
    }
    // inverse FFT via conjugation
    for (let i = 0; i < nfft; i++) li[i] = -li[i];
    fftInPlace(lr, li);
    for (let i = 0; i < nfft; i++) {
      const k = start + i;
      if (k >= 0 && k < L.length) { out[k] += (lr[i] / nfft) * win[i]; norm[k] += win[i] * win[i]; }
    }
  }
  for (let i = 0; i < out.length; i++) if (norm[i] > 1e-6) out[i] /= norm[i];
  return out;
}

export function binFreq(i, nfft, sr) { return i * sr / nfft; }
