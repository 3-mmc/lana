#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Build the bundled native runners on the release Mac. Users need neither
# Swift nor Python. Omnilingual/MLX is macOS 15+, Parakeet/CoreML is 14+.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/src-tauri/mac-models"
[[ "$(uname -s)" == Darwin ]] || { echo 'Native Mac runners must be built on macOS.' >&2; exit 1; }
mkdir -p "$DEST"

for family in parakeet omnilingual; do
    package="$ROOT/mac-models/$family"
    # Share the small protocol implementation without a third Swift package.
    cmp "$ROOT/mac-models/WorkerIO.swift" "$package/Sources/WorkerIO.swift"
    if [[ "$family" == parakeet ]]; then deployment=14.0; else deployment=15.0; fi
    MACOSX_DEPLOYMENT_TARGET="$deployment" swift build --package-path "$package" -c release --product "lana-$family"
    output="$(swift build --package-path "$package" -c release --show-bin-path)"
    mkdir -p "$DEST/$family"
    cp "$output/lana-$family" "$DEST/$family/"
    # SwiftPM resources may include vocab/frontend resources and Metal data.
    for resource in "$output"/*.bundle "$output"/*.dylib; do
        [[ ! -e "$resource" ]] || cp -R "$resource" "$DEST/$family/"
    done
    # Tauri's Rust executable does not trigger Xcode's Swift embedding phase.
    # Copy any Swift compatibility libraries needed by these child binaries.
    xcrun swift-stdlib-tool --copy --platform macosx \
        --scan-executable "$DEST/$family/lana-$family" --destination "$DEST/$family"
    if ! otool -l "$DEST/$family/lana-$family" | grep -Fq 'path @executable_path (offset'; then
        install_name_tool -add_rpath @executable_path "$DEST/$family/lana-$family"
    fi
    if [[ "$family" == omnilingual ]]; then
        # SwiftPM alone does not compile MLX's Metal kernels. Use the build
        # script from the pinned speech-swift dependency, on our build tree.
        BUILD_DIR="$package/.build" MACOSX_DEPLOYMENT_TARGET=15.0 \
            bash "$package/.build/checkouts/speech-swift/scripts/build_mlx_metallib.sh" release
        cp "$output/mlx.metallib" "$DEST/$family/"
        test -s "$DEST/$family/mlx.metallib"
    fi
    # These execute as child processes from Resources; sign them before
    # Tauri signs the enclosing app. No hardened-runtime JIT is needed.
    for library in "$DEST/$family/"*.dylib; do
        [[ ! -f "$library" ]] || codesign --force --sign - "$library"
    done
    codesign --force --sign - "$DEST/$family/lana-$family"
    # Prove that the copied runner can load its dependencies/resources without
    # SwiftPM's fallback paths into the build checkout. GPU validation remains
    # a separate real-Mac test; hosted runners may not expose a Metal device.
    mv "$package/.build" "$package/.build-self-test"
    if (cd /tmp && "$DEST/$family/lana-$family" --self-test); then
        mv "$package/.build-self-test" "$package/.build"
    else
        mv "$package/.build-self-test" "$package/.build"
        echo "Bundled $family runner self-test failed" >&2
        exit 1
    fi
done

# Keep the third-party licences with the shipped code.
cp "$ROOT/mac-models/README.md" "$DEST/README.md"
for family in parakeet omnilingual; do
    for dependency in "$ROOT/mac-models/$family/.build/checkouts/"*; do
        for licence in "$dependency"/LICENSE "$dependency"/LICENSE.txt "$dependency"/LICENSE.md; do
            [[ ! -f "$licence" ]] || cp "$licence" "$DEST/$family/$(basename "$dependency")-LICENSE.txt"
        done
    done
done
