// Decode with the Web Audio API and resample with an OfflineAudioContext.

export async function decode(arrayBuffer) {
  // decodeAudioData resamples to the context rate; 44.1 kHz is what the
  // essentia algorithms (and the native encoder) expect.
  const ctx = new OfflineAudioContext(2, 1, 44100);
  const buf = await ctx.decodeAudioData(arrayBuffer.slice(0));
  return buf; // AudioBuffer @ 44100
}

export function channels(buf) {
  const L = buf.getChannelData(0);
  const R = buf.numberOfChannels > 1 ? buf.getChannelData(1) : L;
  const mono = new Float32Array(L.length);
  for (let i = 0; i < L.length; i++) mono[i] = 0.5 * (L[i] + R[i]);
  return { L, R, mono, sr: buf.sampleRate, stereo: buf.numberOfChannels > 1 };
}

// Mono resample through the browser's own resampler.
export async function resampleMono(mono, srIn, srOut) {
  if (srIn === srOut) return mono;
  const n = Math.ceil(mono.length * srOut / srIn);
  const ctx = new OfflineAudioContext(1, n, srOut);
  const src = ctx.createBufferSource();
  const b = ctx.createBuffer(1, mono.length, srIn);
  b.copyToChannel(mono, 0);
  src.buffer = b;
  src.connect(ctx.destination);
  src.start();
  const out = await ctx.startRendering();
  return out.getChannelData(0);
}
