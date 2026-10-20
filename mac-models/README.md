# Native Mac models in Lana 0.2.1

Lana's normal ONNX models use int8 on the Mac CPU. Whisper uses its existing
whisper.cpp Metal backend in Automatic mode. The generic ONNX-to-CoreML
MLProgram route is disabled: the M3/macOS 14.6 probe aborts while compiling
MatMul shapes `[1,1,1] × [1024,1024]` (or `[1280,1280]`).

Optional downloads in model settings add native acceleration:

| Model | Runtime | Minimum macOS | Extra weights |
| --- | --- | --- | --- |
| Parakeet v3 | FluidAudio / CoreML, CPU and Neural Engine | 14 | ~484 MB |
| Omnilingual 300M | speech-swift / MLX Metal, 8-bit, v2 | 15 | ~360 MB |
| Omnilingual 1B | speech-swift / MLX Metal, 8-bit, v2 | 15 | ~1,055 MB |

The original int8 models stay installed for CPU mode, correction/search,
and fallback. Downloading acceleration is optional. Existing generic FP32
downloads are not selected on macOS. They can still be used on other OSes.

## Packaging

On an Apple Silicon build Mac with Swift 6.2+ / Xcode and the Metal toolchain:

```sh
bash scripts/build-mac-models.sh
cd src-tauri
cargo tauri build
```

The packages pin the FluidAudio and speech-swift source revisions and MLX Swift 0.31.4. The build
also compiles and includes `mlx.metallib` and copies SwiftPM resource bundles.
It then tests the copied helpers with the build directories hidden. MLX runs
a CPU operation for this packaging check; hosted runners may lack a GPU.
The app's macOS Tauri config bundles the output under `Resources/mac-models`.
No Python, Swift toolchain, external inference server or user-installed runtime
is needed on the destination Mac. Omnilingual's higher deployment target does
not raise Lana's deployment target: its helper is only offered on macOS 15+.

Weights are fetched by Lana's checksum-verified Hugging Face downloader into
the chosen model folder. Helpers load only local files; MLX is explicitly in
offline mode and FluidAudio uses `loadLocal`. Each helper stays alive for one
job. Workers receive normalized 16 kHz Float32 audio for Omnilingual or raw
16 kHz Float32 for Parakeet. Omnilingual returns logits, preserving Lana's
alphabet restriction, confidence, alternatives and model frame timings. Its
v2 vocabulary comes from its own tokenizer, never the older ONNX vocabulary;
pad/blank is remapped from 1 to Lana's index 0. Parakeet returns native word
timings. Correction/search continue using the original ONNX model.

An error or native abort kills only the helper. Lana clears partial results
and reruns the job with the original int8 CPU model, with a warning. Full
worker stderr is copied into benchmark stderr, and the speed-test JSON keeps
the full stderr and exit signal. The native worker's peak memory is reported
separately; combined `peak_mb` is the conservative sum of process high-water
marks rather than a measurement of their simultaneous peak.

## Verification on a Mac

1. Use Automatic mode with only the base downloads: all ONNX models should
   report CPU, Whisper should run with Metal. No CoreML compiler crash.
2. Download Parakeet's Mac acceleration. On macOS 14+, test transcription,
   word timings, partial results, cancellation and long recordings.
3. On macOS 15+, download each Omnilingual Mac acceleration. Check the same
   recordings in CPU/Automatic modes, language alphabet restrictions,
   confidence/alternatives, correction and search. The 300M native weights
   are v2, so text need not be identical to the original ONNX 300M.
4. Run the speed test. It includes CPU/Automatic for Whisper and every
   installed native variant. Check runtime labels, memory and `stderr`.
5. For a fallback check in a development checkout, set `LANA_MAC_MODELS` to
   a folder containing a copy of the helpers with one helper replaced by an
   executable that exits nonzero. The job must finish on CPU with a warning.
6. To check the bundled MLX shaders directly on a real macOS 15+ Mac:
   `/Applications/Lana.app/Contents/Resources/mac-models/omnilingual/lana-omnilingual --self-test-metal`

The Swift runners require a Mac build and hardware validation. Linux tests
exercise worker framing, bounds, diagnostics and the existing recognisers;
they do not establish Mac acceleration speed or recognition parity.

Upstreams: https://github.com/FluidInference/FluidAudio (Apache-2.0),
https://github.com/soniqo/speech-swift (Apache-2.0), https://github.com/ml-explore/mlx-swift (MIT).
