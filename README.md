# Lana — desktop

**L**inguistic **ANA**lysis. Lana is built on [Praat](https://www.fon.hum.uva.nl/praat/) (Paul Boersma & David Weenink) but is not part of the Praat project or endorsed by its authors.

Lana's browser interface (in [`web/`](web/), runnable on its own with `web/serve.py`) as a
desktop app for macOS (Apple Silicon), Windows 11 and Linux: Praat's analysis with word, phone and speaker tiers filled by speech
recognition, exported as a TextGrid. Design and milestones are in [PLAN.md](PLAN.md).

**Built-in recognition:** Whisper models (base, small, large-v3 turbo) can be downloaded from
⚙ Connection and run inside the app with whisper.cpp, offline, with word timings from DTW. Models
go to the app's data directory and are checked against HuggingFace's published SHA-256.

**Status: milestone 2.** A native window with the web interface and praat-wasm analysis,
and a Rust port of `serve.py`'s recognition broker. Providers come from `providers.json` in the
app's config directory (same format as `web/serve.py`'s; `LANA_PROVIDERS` overrides the path). A
provider with `"keychain": true` takes its API key from the OS keychain (service `lana`,
account = provider id). Recognisers and keys are managed from ⚙ Connection → *Recognisers in
this app*. Where there is no keychain (Linux without a Secret Service, e.g. WSL) a key can be
saved in the settings file instead, after the user explicitly agrees.

`scripts/parity-test.py` sends the same requests to `web/serve.py` and the Rust broker through a
mock recogniser and fails on any difference that is not listed as deliberate.

## Build

Prerequisites: [Rust](https://rustup.rs), the Tauri CLI (`cargo install tauri-cli --locked`),
the [Tauri system dependencies](https://v2.tauri.app/start/prerequisites/) for your platform,
and, for the built-in speech recognition (whisper.cpp), **cmake** and **libclang**:

- **macOS:** `xcode-select --install` provides libclang. Install cmake from
  [cmake.org](https://cmake.org/download/) or with `brew install cmake`.
- **Linux (Debian/Ubuntu):** `sudo apt install cmake libclang-dev`
- **Windows:** Visual Studio Build Tools ships cmake (add its `CMake\bin` to `PATH`). For libclang,
  set `LIBCLANG_PATH` to a folder containing `libclang.dll` (from an LLVM install, or the
  `libclang` wheel on PyPI). Do not set `WHISPER_DONT_GENERATE_BINDINGS`: the bundled bindings
  are generated on Linux and their struct layouts do not compile on Windows.

```sh
git clone https://github.com/3-mmc/lana
scripts/fetch-praat-wasm.sh              # 24 MB, once, into vendor/
cd src-tauri
cargo run                                # development
cargo run -- path/to/recording.wav       # open a recording at start
cargo tauri build                        # installer for this platform
```

On macOS without an Apple Developer ID the build is ad-hoc signed. A copy built on the Mac
itself opens normally. A downloaded copy must be allowed once under System Settings → Privacy &
Security → **Open Anyway**.

## Licence

GPL-3.0-or-later, as required by praat-wasm.
