// Praat analysis of a recording of "Lana" -> JSON for the logo.
import { readFileSync, writeFileSync } from 'node:fs';
const [pkg, wav, out] = process.argv.slice(2);
const { createPraatWasm } = await import(pkg + '/js/praat-wasm.mjs');
const P = await createPraatWasm(pkg + '/dist/praat.wasm');
const buf = readFileSync(wav);
const snd = P.readAudio(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), 'lana.wav');
const matrix = (obj, cmd, ...a) => {
  const m = P.call(obj, cmd, ...a);
  P.call(m, 'Save as matrix text file...', '/m.txt');
  // Same layout the page's parseMatrix reads: x header on line 3, y on 4, rows after.
  const lines = new TextDecoder().decode(P.getFile('/m.txt')).split('\n');
  const [xmin, xmax, nx, dx, x1] = lines[2].trim().split(/\s+/).map(Number);
  const ny = Number(lines[3].trim().split(/\s+/)[2]);
  return { x1, dx, nx, rows: lines.slice(4, 4 + ny).map(l => l.trim().split(/\s+/).map(Number)) };
};
const pitch = matrix(P.call(snd, 'To Pitch...', 0.005, 75, 400), 'To Matrix');
const form = P.call(snd, 'To Formant (burg)...', 0.005, 5, 5500, 0.025, 50);
const F = [1, 2, 3].map(n => matrix(form, 'To Matrix...', n));
const inten = matrix(P.call(snd, 'To Intensity...', 75, 0.005, 'yes'), 'Down to Matrix');
// samples straight from the WAV (16-bit PCM from espeak)
const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
let off = 12, rate = 0, data = null;
while (off < buf.length - 8) {
  const id = buf.toString('ascii', off, off + 4), size = dv.getUint32(off + 4, true);
  if (id === 'fmt ') rate = dv.getUint32(off + 12, true);
  if (id === 'data') { data = new Int16Array(buf.buffer.slice(buf.byteOffset + off + 8, buf.byteOffset + off + 8 + size)); break; }
  off += 8 + size + (size & 1);
}
const voiced = t => { const i = Math.round((t - pitch.x1) / pitch.dx); return (pitch.rows[0]?.[i] || 0) > 0; };
const frames = [];
for (let i = 0; i < F[0].nx; i++) {
  const t = F[0].x1 + i * F[0].dx;
  frames.push({ t, voiced: voiced(t), f: F.map(m => m.rows[0][i]) });
}
writeFileSync(out, JSON.stringify({ rate, samples: Array.from(data, v => v / 32768), frames,
  intensity: { x1: inten.x1, dx: inten.dx, v: inten.rows[0] } }));
console.log(wav, 'rate', rate, 'samples', data.length, 'frames', frames.length, 'voiced', frames.filter(f => f.voiced).length);
