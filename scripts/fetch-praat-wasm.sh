#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Download praat-wasm into ./vendor (gitignored), as serve.py does in the web build.
# Works on macOS, Linux, and Windows (Git Bash; Windows 11 ships curl and tar).
set -euo pipefail
VERSION=6.4.6200   # keep in step with VERSION in web/serve.py
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/vendor"
if [[ -f "$DEST/js/worker-client.mjs" ]]; then
    echo "praat-wasm already in $DEST"; exit 0
fi
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
curl -fsSL "https://registry.npmjs.org/praat-wasm/-/praat-wasm-$VERSION.tgz" -o "$TMP/p.tgz"
tar xzf "$TMP/p.tgz" -C "$TMP"
rm -rf "$DEST"; mv "$TMP/package" "$DEST"
echo "praat-wasm $VERSION ready in $DEST"
