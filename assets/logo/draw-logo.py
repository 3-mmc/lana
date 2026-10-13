# Draw the Lana icon from Praat's analysis of the word "Lana".
import json, math, sys
src, out, layout = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(src))
N = 1024
BG, PANEL, INK = '#1b1b1d', '#252528', '#e7e7ea'
FC = ['#ef6f5e', '#45b0a4', '#a98ada']       # app's F1 F2 F3 (graphite theme)
s, rate = d['samples'], d['rate']
fr = [f for f in d['frames'] if f['voiced'] and all(x == x and x > 0 for x in f['f'])]
t0 = fr[0]['t'] - 0.03; t1 = fr[-1]['t'] + 0.03       # crop to the voiced word
i0, i1 = max(0, int(t0 * rate)), min(len(s), int(t1 * rate))
s = s[i0:i1]; peak = max(abs(v) for v in s) or 1
M = 150                                    # inner margin
W = N - 2 * M
def env(cols):
    out = []
    for c in range(cols):
        a, b = c * len(s) // cols, (c + 1) * len(s) // cols
        seg = s[a:b] or [0]
        out.append((min(seg) / peak, max(seg) / peak))
    return out
def wave_path(y_mid, half, cols=160):
    e = env(cols); xs = [M + W * (c + .5) / cols for c in range(cols)]
    top = ' '.join(f'{x:.1f},{y_mid - half * hi:.1f}' for x, (lo, hi) in zip(xs, e))
    bot = ' '.join(f'{x:.1f},{y_mid - half * lo:.1f}' for x, (lo, hi) in reversed(list(zip(xs, e))))
    return f'<polygon points="{top} {bot}" fill="{INK}" opacity="0.92"/>'
def dots(y_top, y_bot, fmax, step, r):
    out = []
    for k, f in enumerate(fr[::step]):
        x = M + W * (f['t'] - t0) / (t1 - t0)
        for n in (2, 1, 0):                # F3 under F2 under F1
            y = y_bot - (y_bot - y_top) * min(f['f'][n], fmax) / fmax
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{FC[n]}"/>')
    return '\n'.join(out)
tile = f'<rect x="24" y="24" width="{N-48}" height="{N-48}" rx="210" fill="{BG}" stroke="#4a4a52" stroke-width="10"/>'
if layout == 'overlay':          # dots drawn over the waveform, Praat-scale 0-3.2 kHz
    body = wave_path(N * .5, 250) + dots(M + 40, N - M - 40, 3200, 3, 13)
elif layout == 'editor':         # Praat's editor: waveform above, formant track below
    body = (f'<rect x="{M-40}" y="{M+200}" width="{W+80}" height="{N-2*M-210}" rx="40" fill="{PANEL}"/>'
            + wave_path(M + 90, 95) + dots(M + 250, N - M - 40, 3200, 3, 12))
elif layout == 'final':          # editor layout, tuned to survive small sizes
    body = (f'<rect x="{M-40}" y="{M+215}" width="{W+80}" height="{N-2*M-225}" rx="44" fill="{PANEL}"/>'
            + wave_path(M + 95, 105, cols=72) + dots(M + 270, N - M - 45, 3100, 3, 15))
elif layout == 'bold':           # fewer, larger dots: survives 32 px
    body = wave_path(N * .5, 270) + dots(M + 30, N - M - 30, 3000, 6, 22)
# Apple's icon grid: the tile is ~824 of 1024, centred. Other platforms are
# fine with the same margin, and one source keeps every size consistent.
k = 824 / (N - 48)
open(out, 'w').write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {N} {N}">'
                     f'<g transform="translate({N/2} {N/2}) scale({k:.4f}) translate({-N/2} {-N/2})">{tile}{body}</g></svg>')
