# lana-desktop — plan

Native builds of [lana-web](https://github.com/3-mmc/lana-web) for **Apple Silicon macOS,
Windows 11 and Linux**. Same interface and feature set as the web version: spectrogram, pitch,
formant and intensity panes, word/phone/speaker tiers, pluggable recognition providers, and
TextGrid export. On top of that, two things only a native app can do well:

1. **Find the models already on the machine and use them**, and
2. **Get a linguist with no command-line experience from "downloaded the app" to "a TextGrid
   with IPA in it"** without editing a JSON file, setting an environment variable or running a
   server.

GPL-3.0-or-later, inherited from praat-wasm through lana-web.

## Layout

```
lana-desktop/
  web/          git submodule -> 3-mmc/lana-web (the UI; do not fork it here)
  src-tauri/    Rust shell: window, embedded server, broker, model manager   (to come)
  PLAN.md
```

`web/` is a submodule, not a copy. The interface has one source of truth. A change the desktop
app needs goes into lana-web behind feature detection, and the submodule pointer gets bumped.
A vendored copy would drift on the first commit.

## The seam: keep lana-web's `/api/*` contract

lana-web already feature-detects a backend. When `GET /api/providers` answers, the page uses the
broker. When it does not, the page falls back to a typed endpoint. `serve.py` implements that
contract in 505 lines: `providers`, `transcribe`, `progress`, `partial`, `cancel`, `recluster`,
plus `normalise()`, which folds every response into the cue shape.

The desktop shell **re-implements that contract in Rust and serves the unchanged `index.html`
from an embedded `127.0.0.1:<random port>` server** that the webview loads. The result:

- the frontend needs zero changes to run natively. Desktop-only UI (model manager, key entry) is
  added to lana-web behind `GET /api/capabilities`, so the browser build never shows it;
- workers, wasm and same-origin fetches behave exactly as they do in the browser. praat-wasm
  needs no `SharedArrayBuffer`/cross-origin isolation, which was checked, so all three webviews
  can run it;
- `serve.py` remains the reference implementation. The Rust broker must pass the same requests.

## Stack: Tauri 2 (recommended over Electron)

| | Tauri 2 | Electron |
|---|---|---|
| Installer size | ~10 MB + 24 MB praat-wasm + runtimes | ~120 MB + the same |
| Webview | WKWebView / WebView2 / WebKitGTK | bundled Chromium |
| Backend language | Rust, which lets us link whisper.cpp, ONNX Runtime and the OS keychain directly | Node, which needs a Python or C++ sidecar for inference anyway |
| Risk | WebKitGTK on Linux is the weakest webview, so canvas-heavy code must be tested there | none on rendering |

The deciding factor is that the local inference runtimes (below) are C/C++ libraries with good
Rust bindings. With Electron we would ship a second native process regardless.

## Providers, unchanged from the web build

- **Remote OpenAI-shaped APIs** (OpenAI, Groq, Fireworks, Together, …). These are edited in a
  Settings screen instead of `providers.json`. **API keys go in the OS keychain**
  (Keychain / Windows Credential Manager / Secret Service, via the `keyring` crate), never in a
  file and never sent to the webview. This keeps the guarantee `serve.py` gives with
  `api_key_env`, minus the environment variable.
- **Any OpenAI-compatible server** by URL, which includes llama-swap, as on this machine.

## Local models, the new part

Three tiers, in order of how much a user has to know:

### 1. Built-in runtimes + a curated catalogue (the default path for most linguists)

The app ships its own inference and offers a short, opinionated list with plain-language labels,
download sizes and what each one fills:

| Tier | Model | Runtime | Notes |
|---|---|---|---|
| words | whisper (tiny → large-v3-turbo, ggml) | whisper.cpp (`whisper-rs`): Metal on Apple Silicon, Vulkan/CUDA elsewhere, CPU fallback | word timestamps from DTW |
| phones | wav2vec2/XLS-R phoneme CTC (e.g. `facebook/wav2vec2-xlsr-53-espeak-cv-ft`; PhoneticXeus if an ONNX export works) | ONNX Runtime (`ort`): CoreML / DirectML / CUDA EPs | the CTC spike → phone-boundary logic is ported from ipasub `engine.transcribe` |
| speakers | sherpa-onnx segmentation + embedding | ONNX Runtime | same models as `~/lab/diarize`, CPU-only, runs beside the others |

Model cards show the licence in plain words. MMS is CC-BY-NC, so it is flagged
"non-commercial use only", not hidden.

Downloads go to the platform app-data dir, are resumable, are checked against a SHA-256 pinned in
the catalogue, and show disk space before starting.

### 2. Discovery of models already installed

On first run and on demand ("Look for models on this computer"), scan known locations
**read-only** and offer what is found next to the catalogue, marked "found on this computer":

- whisper.cpp `ggml-*.bin` in common dirs (`~/.cache/whisper`, MacWhisper / Buzz / Whisper
  Transcription app folders, `~/models`);
- the HuggingFace cache (`$HF_HOME` or `~/.cache/huggingface/hub`): `openai/whisper-*`,
  `Systran/faster-whisper-*`, `facebook/wav2vec2-*phoneme*` / `*espeak*`, `facebook/mms-*`. These
  are usable only where a runtime above can load the format. Anything else is listed greyed with
  the reason, not silently dropped;
- sherpa-onnx model dirs.

Never copy or move a found model. Reference it in place, and fall back cleanly if it disappears.

### 3. Discovery of running local servers

Probe localhost once for OpenAI-compatible `/v1/models`: llama-swap `:9292`, speaches/
faster-whisper-server `:8000`, whisper.cpp server `:8080`, LM Studio `:1234`. Add any that expose
audio models as providers.
**Lesson from lana-web:** any GET through llama-swap's `/upstream/<id>/` *starts* that service
and can evict a resident GPU model. Discovery must hit only the gateway's root `/v1/models`,
never an upstream.

### Friendliness rules for non-technical users

- First-run wizard: "Which language are your recordings in?" leads to a recommended model pair
  and one download button. Language codes are picked from a named list. The app maps to 639-1 or
  639-3 per model, a mismatch that already bit lana-web (mms `kat` vs whisper `ka`).
- Hardware check up front: RAM/VRAM/Apple-Silicon detection selects the largest model that will
  actually fit and says why.
- Errors in sentences ("This model needs about 3 GB of free memory; you have 1.2 GB"), never a
  status code.
- Everything works offline once models are downloaded, and the app says so.

## Builds and where they happen

- **Windows 11**: build natively on Windows from `D:\Projekte\lana-desktop` (MSVC + WebView2).
- **Linux**: build in WSL. Point `CARGO_TARGET_DIR` at ext4 (`~/.cache/lana-desktop-target`),
  because a cargo target dir on drvfs is very slow.
- **Apple Silicon**: GitHub Actions `macos-14` (arm64) runner, unless a Mac is available locally.
  Signing and notarisation need an Apple Developer ID. Without one, Gatekeeper makes users
  right-click → Open, which is exactly the friction this project is meant to remove.
- CI matrix for all three in `.github/workflows/`, producing `.dmg`, `.msi`/`.exe` (NSIS) and
  `.AppImage`/`.deb`.

## Milestones

1. **Shell**: Tauri app serving `web/` + praat-wasm from the embedded server, with no broker.
   This proves praat-wasm, canvases and workers in all three webviews, WebKitGTK especially.
2. **Broker port**: Rust `/api/*` matching `serve.py`, with providers from a settings file and
   keys in the keychain. Diff-test against `serve.py` with the same requests.
3. **whisper.cpp built in** + catalogue download for the word tier.
4. **Phone tier** via ONNX Runtime + CTC spike timing, checked against ipasub output on the
   German and Georgian FLEURS clips used for the lana-web README.
5. **Discovery** of installed models and running servers.
6. **Speaker tier** via sherpa-onnx; first-run wizard; installers + CI.

## Open questions

- Is a Mac available for local testing, or is CI the only Apple Silicon build/test path?
- Apple Developer ID for signing: yes or no?
- Does PhoneticXeus export cleanly to ONNX? If not, the phone tier ships wav2vec2-espeak first.
