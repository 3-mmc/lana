"""Export MMS-1b-all with one language adapter to ONNX in sherpa-onnx's
Omnilingual layout: raw 16 kHz samples in (normalised by sherpa-onnx itself,
as the MMS processor does), CTC logits out, blank = 0, every weight in its own
file. build.py then quantises and splits it. (onnxruntime's quantize_dynamic
is not used: on the 1B model it needed over 8 GB and was killed.)

usage: export.py <language code> <out dir>; needs transformers, torch, onnx"""
import gc, json, os, sys, time
import torch, onnx
from transformers import Wav2Vec2ForCTC

SRC = os.path.expanduser('~/models/mms-1b-all')
lang, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        return self.m(x).logits

t = time.time()
m = Wav2Vec2ForCTC.from_pretrained(SRC, target_lang=lang, ignore_mismatched_sizes=True).eval()
vocab = json.load(open(f'{SRC}/vocab.json', encoding='utf-8'))[lang]
# tokens.txt in sherpa-onnx's format: "<symbol> <id>"; MMS writes "|" between words.
inv = sorted(vocab.items(), key=lambda kv: kv[1])
with open(f'{out}/tokens.txt', 'w', encoding='utf-8') as f:
    for s, i in inv:
        f.write(f"{'|' if s == ' ' else s} {i}\n")
print('loaded', lang, 'vocab', len(vocab), f'{time.time()-t:.0f}s', flush=True)

fp32 = f'{out}/fp32.onnx'
with torch.no_grad():
    torch.onnx.export(Wrap(m), (torch.zeros(1, 16000),), fp32, input_names=['x'], output_names=['logits'],
                      dynamic_axes={'x': {0: 'N', 1: 'samples'}, 'logits': {0: 'N', 1: 'frames'}},
                      opset_version=17, dynamo=False, external_data=True)
del m; gc.collect()
print('exported', f'{time.time()-t:.0f}s', flush=True)

print('done: run build.py on', out)
