# Lana — desktop

**L**inguistic **ANA**lysis. Lana is built on [Praat](https://www.fon.hum.uva.nl/praat/) (Paul Boersma & David Weenink) but is not part of the Praat project or endorsed by its authors.

[lana-web](https://github.com/3-mmc/lana-web) as a desktop app for macOS (Apple Silicon),
Windows 11 and Linux: Praat's analysis with word, phone and speaker tiers filled by speech
recognition, exported as a TextGrid. Design and milestones are in [PLAN.md](PLAN.md).

**Status: milestone 2.** A native window with the lana-web interface and praat-wasm analysis,
and a Rust port of `serve.py`'s recognition broker. Providers come from `providers.json` in the
app's config directory (same format as lana-web's; `LANA_PROVIDERS` overrides the path). A
provider with `"keychain": true` takes its API key from the OS keychain (service `lana`,
account = provider id). Recognisers and keys are managed from ⚙ Connection → *Recognisers in
this app*. Where there is no keychain (Linux without a Secret Service, e.g. WSL) a key can be
saved in the settings file instead, after the user explicitly agrees.

`scripts/parity-test.py` sends the same requests to `web/serve.py` and the Rust broker through a
mock recogniser and fails on any difference that is not listed as deliberate.

## Build

Prerequisites: [Rust](https://rustup.rs), the Tauri CLI (`cargo install tauri-cli --locked`),
and the [Tauri system dependencies](https://v2.tauri.app/start/prerequisites/) for your platform.

```sh
git clone --recursive <this repo>        # web/ is a submodule
scripts/fetch-praat-wasm.sh              # 24 MB, once, into vendor/
cd src-tauri
cargo run                                # development
cargo tauri build                        # installer for this platform
```

On macOS without an Apple Developer ID the build is ad-hoc signed. A copy built on the Mac
itself opens normally. A downloaded copy must be allowed once under System Settings → Privacy &
Security → **Open Anyway**.

## Licence

GPL-3.0-or-later, as required by praat-wasm.
