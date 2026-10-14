#!/usr/bin/env bash
# Assemble the openlana.app static site into one folder:
#
#   /            site/        the landing page
#   /app/        web/         the browser build, unchanged; /app/?demo is the live demo
#   /app/vendor/              praat-wasm from npm, which web/ keeps out of git
#
#   scripts/build-site.sh [out-dir]      (default: _site)
#
# The Pages workflow runs this; run it locally and serve the result with
# `python3 -m http.server -d _site` to see exactly what will be published.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-"$ROOT/_site"}
# Keep in step with VERSION in web/serve.py.
PRAAT_WASM=6.4.6200

rm -rf "$OUT"
mkdir -p "$OUT/app"

cp -r "$ROOT/site/." "$OUT/"
cp "$ROOT/web/index.html" "$ROOT/web/languages.json" "$ROOT/web/LICENSE" "$OUT/app/"
cp -r "$ROOT/web/demo" "$OUT/app/demo"
rm -f "$OUT/app/demo/record.py"

if [[ -d "${PRAAT_WASM_DIR:-}" ]]; then
  cp -r "$PRAAT_WASM_DIR/." "$OUT/app/vendor/"
else
  mkdir -p "$OUT/app/vendor"
  curl -fsSL "https://registry.npmjs.org/praat-wasm/-/praat-wasm-$PRAAT_WASM.tgz" \
    | tar xz -C "$OUT/app/vendor" --strip-components=1 package
fi

# GitHub Pages runs Jekyll on anything without this, and Jekyll drops _-prefixed paths.
touch "$OUT/.nojekyll"
echo "site assembled in $OUT ($(du -sh "$OUT" | cut -f1))"
