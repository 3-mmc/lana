# Lana — desktop

**L**inguistic **ANA**lysis. Lana is built on [Praat](https://www.fon.hum.uva.nl/praat/) (Paul Boersma & David Weenink) but is not part of the Praat project or endorsed by its authors.

[lana-web](https://github.com/3-mmc/lana-web) as a desktop app for macOS (Apple Silicon),
Windows 11 and Linux: Praat's analysis with word, phone and speaker tiers filled by speech
recognition, exported as a TextGrid. Design and milestones are in [PLAN.md](PLAN.md).

**Status: milestone 1.** A native window with the lana-web interface and praat-wasm analysis.
Recognition works only against an endpoint you enter yourself, until the Rust broker lands.

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
