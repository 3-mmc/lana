"""Make the wide variant of Lana's MMS graph, for alphabets over 512 symbols.

The shared graph pads every language's output layer to 512 symbols, which is
enough for 1,194 of MMS's 1,198 adapters. Mandarin (4,495), Cantonese (2,299),
Japanese (2,268) and Korean (1,330) need more. This writes a second graph whose
output layer has WIDE symbols, and the adapter layout that goes with it: the
same tensors in the same order, the two output tensors larger, and every offset
after them moved along. Only the 1.4 MB graph and the layout differ; the shared
weights in mms-base.bin are the same file.

usage: widen.py <model.onnx> <adapters.json> <WIDE> <out model> <out adapters.json>
"""
import json, sys
import onnx
from onnx import TensorProto

model_in, map_in, wide, model_out, map_out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
m = json.load(open(map_in))
g = onnx.load(model_in, load_external_data=False)
inits = {t.name: t for t in g.graph.initializer}

offset, tensors = 0, []
for t in sorted(m['tensors'], key=lambda t: t['offset']):
    t = dict(t)
    if t['source'] == 'lm_head.weight':
        t['shape'] = [t['shape'][0], wide]
    elif t['source'] == 'lm_head.bias':
        t['shape'] = [wide]
    n = 1
    for d in t['shape']:
        n *= d
    t['bytes'], t['offset'] = n * 4, offset
    offset += t['bytes']
    tensors.append(t)
    init = inits[t['name']]
    assert init.data_location == TensorProto.EXTERNAL
    del init.dims[:]
    init.dims.extend(t['shape'])
    info = {e.key: e for e in init.external_data}
    assert info['location'].value == 'adapter.bin', info['location'].value
    info['offset'].value, info['length'].value = str(t['offset']), str(t['bytes'])

onnx.save(g, model_out)
json.dump({'head': wide, 'tensors': tensors, 'adapter_bytes': offset}, open(map_out, 'w'), indent=1)
print(f'head {wide}: adapter.bin {offset} bytes, graph {model_out}')
