"""Turn an fp32 ONNX export of MMS-1b-all (one weight per external file) into
Lana's MMS package, one tensor at a time so memory stays small:

  model.onnx     the graph, identical for every language
  mms-base.bin   shared weights; MatMul weights quantised to int8
  adapter.bin    this language's adapter layers and output head, fp32
  adapters.json  where each adapter tensor sits in adapter.bin, and which
                 tensor of Meta's adapter.<lang>.safetensors it comes from,
                 so the app can build adapter.bin for any language itself

The output head is padded to HEAD rows (bias -1e4 on the padding), so its
shape, and with it the graph, does not depend on the language's alphabet.

usage: build.py <fp32 export dir> <adapter.safetensors> <out dir>
"""
import json, os, sys
import numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto
from onnx.external_data_helper import set_external_data
from safetensors.numpy import load_file

HEAD = 512
SRC, ADAPTER, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
g = onnx.load(os.path.join(SRC, 'fp32.onnx'), load_external_data=False)
inits = {t.name: t for t in g.graph.initializer}
NP = {TensorProto.FLOAT: np.float32, TensorProto.INT64: np.int64, TensorProto.INT32: np.int32}

def read(t):
    """One initializer's values, from its own external file or inline."""
    if t.data_location != TensorProto.EXTERNAL:
        return numpy_helper.to_array(t)
    info = {e.key: e.value for e in t.external_data}
    arr = np.fromfile(os.path.join(SRC, info['location']), dtype=NP[t.data_type],
                      offset=int(info.get('offset', 0)),
                      count=int(np.prod(t.dims)) if t.dims else 1)
    return arr.reshape(tuple(t.dims))

consumers = {}
for n in g.graph.node:
    for i in n.input:
        consumers.setdefault(i, []).append(n)
def is_adapter(name):
    users = consumers.get(name, [])
    return 'adapter_layer' in name or 'lm_head' in name or any(
        'adapter_layer' in u.name or 'lm_head' in u.name for u in users)

st = load_file(ADAPTER)

# --- quantise the shared MatMuls -------------------------------------------
new_nodes, quant = [], {}
for n in g.graph.node:
    w = n.input[1] if n.op_type == 'MatMul' and len(n.input) == 2 else None
    if w in inits and len(inits[w].dims) == 2 and not is_adapter(w) and np.prod(inits[w].dims) >= 65536:
        p = n.name.strip('/').replace('/', '_')
        a, out = n.input[0], n.output[0]
        new_nodes += [
            helper.make_node('DynamicQuantizeLinear', [a], [p + '_aq', p + '_as', p + '_azp'], name=p + '_dq'),
            helper.make_node('MatMulInteger', [p + '_aq', w + '_q', p + '_azp', w + '_zp'], [p + '_mmi'], name=p + '_mmi'),
            helper.make_node('Cast', [p + '_mmi'], [p + '_f'], to=TensorProto.FLOAT, name=p + '_cast'),
            helper.make_node('Mul', [p + '_as', w + '_scale'], [p + '_s'], name=p + '_scale'),
            helper.make_node('Mul', [p + '_f', p + '_s'], [out], name=p + '_out'),
        ]
        quant[w] = True
    else:
        new_nodes.append(n)
del g.graph.node[:]
g.graph.node.extend(new_nodes)

# --- write tensors one by one --------------------------------------------
base = open(os.path.join(OUT, 'mms-base.bin'), 'wb')
adap = open(os.path.join(OUT, 'adapter.bin'), 'wb')
mapping, new_inits = [], []

def put(f, location, name, arr, dtype):
    arr = np.ascontiguousarray(arr)
    t = TensorProto(name=name, data_type=dtype, dims=list(arr.shape))
    off = f.tell()
    f.write(arr.tobytes())
    for k, v in (('location', location), ('offset', str(off)), ('length', str(arr.nbytes))):
        e = t.external_data.add(); e.key, e.value = k, v
    t.data_location = TensorProto.EXTERNAL
    new_inits.append(t)
    return off

def source_of(arr):
    """The adapter tensor this initializer was made from, and whether the
    exporter transposed it (a Linear weight becomes a MatMul by its transpose)."""
    for k, v in st.items():
        if v.shape == arr.shape and np.allclose(v, arr):
            return k, False
        if v.ndim == 2 and v.T.shape == arr.shape and np.allclose(v.T, arr):
            return k, True
    raise SystemExit(f'no adapter tensor matches shape {arr.shape}')

for t in list(g.graph.initializer):
    arr = read(t)
    if t.name in quant:
        scale = np.float32(max(np.abs(arr).max(), 1e-8) / 127.0)
        q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
        put(base, 'mms-base.bin', t.name + '_q', q, TensorProto.INT8)
        new_inits.append(numpy_helper.from_array(np.array(scale, np.float32), t.name + '_scale'))
        new_inits.append(numpy_helper.from_array(np.array(0, np.int8), t.name + '_zp'))
    elif is_adapter(t.name):
        key, tr = source_of(arr)
        pad = None
        if key == 'lm_head.weight':      # stored as [1280, V]: pad the columns
            arr = np.concatenate([arr, np.zeros((arr.shape[0], HEAD - arr.shape[1]), np.float32)], 1); pad = 'cols0'
        elif key == 'lm_head.bias':
            arr = np.concatenate([arr, np.full(HEAD - arr.shape[0], -1e4, np.float32)]); pad = 'fill-1e4'
        off = put(adap, 'adapter.bin', t.name, arr.astype(np.float32), TensorProto.FLOAT)
        mapping.append({'name': t.name, 'source': key, 'transpose': tr, 'pad': pad,
                        'shape': list(arr.shape), 'offset': off, 'bytes': arr.nbytes})
    elif t.data_location == TensorProto.EXTERNAL or arr.nbytes > 1024:
        put(base, 'mms-base.bin', t.name, arr, t.data_type)
    else:
        new_inits.append(t)
base.close(); adap.close()
del g.graph.initializer[:]
g.graph.initializer.extend(new_inits)

# The alphabet size must not be baked into the graph anywhere.
del g.graph.value_info[:]
for o in g.graph.output:
    o.type.tensor_type.shape.dim[2].ClearField('dim_value')
    o.type.tensor_type.shape.dim[2].dim_param = 'vocab'
if not any(o.domain in ('', 'ai.onnx') and o.version >= 17 for o in g.opset_import):
    raise SystemExit('unexpected opset')
onnx.save(g, os.path.join(OUT, 'model.onnx'))
json.dump({'head': HEAD, 'tensors': mapping, 'adapter_bytes': os.path.getsize(os.path.join(OUT, 'adapter.bin'))},
          open(os.path.join(OUT, 'adapters.json'), 'w'), indent=1)
print('quantised', len(quant), 'MatMuls; adapter tensors', len(mapping),
      {f: os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)})
