# MMS for Lana

How the shared part of Lana's MMS support is made. The app downloads it once
and builds each language from Meta's own `adapter.<lang>.safetensors`
(`src-tauri/src/local/mms.rs`).

1. `export.py dar <dir>`: MMS-1b-all with the Dargwa adapter to fp32 ONNX,
   raw audio in, CTC logits out (the layout sherpa-onnx runs as Omnilingual).
2. `build.py <dir> adapter.dar.safetensors <pkg>`: quantises the shared MatMuls
   to int8 one tensor at a time (peak memory well under 4 GB), keeps the
   adapters fp32 in `adapter.bin`, pads the output layer to 512 symbols so one
   graph serves every language, and writes `adapters.json`, the map the app
   uses to build any language's `adapter.bin`.
3. `assemble.py <pkg> <lang> <out>`: what the app does, in Python, for testing.

The published files are `model.onnx`, `mms-base.bin` and `adapters.json`.

Measured on 40 Common Voice 27.0 test clips per language, character error
rate: Dargwa 10.1 % (the original MMS on a GPU: 10.0 %), Ossetian 12.1 %
(11.7 %).

MMS is © Meta, CC BY-NC 4.0; so are these converted weights. Non-commercial
use only.
