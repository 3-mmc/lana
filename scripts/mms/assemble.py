"""Build a language folder the way the app will: hard links to the shared
model.onnx and mms-base.bin, adapter.bin rebuilt from Meta's
adapter.<lang>.safetensors using adapters.json, tokens.txt from vocab.json."""
import json, os, sys, numpy as np
from safetensors.numpy import load_file
PKG, lang, out = sys.argv[1:4]
m = json.load(open(f'{PKG}/adapters.json'))
st = load_file(os.path.expanduser(f'~/models/mms-1b-all/adapter.{lang}.safetensors'))
os.makedirs(out, exist_ok=True)
for f in ('model.onnx', 'mms-base.bin'):
    if not os.path.exists(f'{out}/{f}'):
        os.link(f'{PKG}/{f}', f'{out}/{f}')
buf = bytearray(m['adapter_bytes'])
for t in m['tensors']:
    a = st[t['source']].astype(np.float32)
    if t['transpose']: a = a.T
    if t['pad'] == 'cols0':
        a = np.concatenate([a, np.zeros((a.shape[0], m['head'] - a.shape[1]), np.float32)], 1)
    elif t['pad'] == 'fill-1e4':
        a = np.concatenate([a, np.full(m['head'] - a.shape[0], -1e4, np.float32)])
    assert list(a.shape) == t['shape'], (t['name'], a.shape, t['shape'])
    b = np.ascontiguousarray(a).tobytes(); buf[t['offset']:t['offset'] + len(b)] = b
open(f'{out}/adapter.bin', 'wb').write(buf)
vocab = json.load(open(os.path.expanduser('~/models/mms-1b-all/vocab.json'), encoding='utf-8'))[lang]
with open(f'{out}/tokens.txt', 'w', encoding='utf-8') as f:
    for s, i in sorted(vocab.items(), key=lambda kv: kv[1]):
        f.write(f"{'|' if s == ' ' else s} {i}\n")
print(lang, 'ok', len(vocab), 'symbols')
