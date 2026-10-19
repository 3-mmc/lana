#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Download the ONNX Runtime library the built-in models run on into
# src-tauri/onnxruntime (gitignored), which the app bundles as a resource and
# a development build or lana-bench finds there too (local/runtime.rs).
#
#   macOS (Apple Silicon)  1.30.0, with the CoreML provider
#   Windows (x64)          1.24.4 with DirectML (the last DirectML build), and
#                          DirectML.dll 1.15.4, from NuGet
#   Linux (x64)            1.30.0, CPU; CUDA comes with the in-app GPU pack
#
# Every download is checked against its SHA-256. Works on macOS, Linux, and
# Windows (Git Bash, whose tar is GNU tar and cannot read the NuGet zips:
# unzip, or Windows' own tar.exe). `$1` picks the platform (macos, windows,
# linux) instead of this computer's.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/src-tauri/onnxruntime"

platform="${1:-}"
if [[ -z "$platform" ]]; then
    case "$(uname -s)" in
        Darwin) platform=macos ;;
        MINGW*|MSYS*|CYGWIN*) platform=windows ;;
        *) platform=linux ;;
    esac
fi

GH=https://github.com/microsoft/onnxruntime/releases/download
NUGET=https://api.nuget.org/v3-flatcontainer
case "$platform" in
    macos)
        want="libonnxruntime.dylib"
        files=("$GH/v1.30.0/onnxruntime-osx-arm64-1.30.0.tgz 6ebb5062a934537c352937821f9fe9718e7de1a2db1122a93dd363ffd53a7012") ;;
    windows)
        want="onnxruntime.dll"
        files=("$NUGET/microsoft.ml.onnxruntime.directml/1.24.4/microsoft.ml.onnxruntime.directml.1.24.4.nupkg 57e9f11b73437bef7a309496135d4c1f96b1a8e9ddba60013fa27bfc1d788681"
               "$NUGET/microsoft.ai.directml/1.15.4/microsoft.ai.directml.1.15.4.nupkg 4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9") ;;
    linux)
        want="libonnxruntime.so"
        files=("$GH/v1.30.0/onnxruntime-linux-x64-1.30.0.tgz a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd") ;;
    *) echo "unknown platform: $platform" >&2; exit 1 ;;
esac

if [[ -f "$DEST/$want" && -f "$DEST/platform" && "$(cat "$DEST/platform")" == "$platform" ]]; then
    echo "ONNX Runtime for $platform already in $DEST"; exit 0
fi

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
rm -rf "$DEST"; mkdir -p "$DEST"
for entry in "${files[@]}"; do
    url="${entry% *}"; sum="${entry##* }"; name="$(basename "$url")"
    curl -fsSL "$url" -o "$TMP/$name"
    got="$(sha256 "$TMP/$name")"
    if [[ "$got" != "$sum" ]]; then
        echo "checksum mismatch for $name: $got" >&2; exit 1
    fi
    mkdir -p "$TMP/x/$name"
    case "$name" in
        *.nupkg)
            if command -v unzip >/dev/null; then unzip -q "$TMP/$name" -d "$TMP/x/$name"
            else /c/Windows/System32/tar.exe -xf "$TMP/$name" -C "$TMP/x/$name"; fi ;;
        *) tar xzf "$TMP/$name" -C "$TMP/x/$name" ;;
    esac
done

# Only the library itself (the real file, not the version symlinks) and its
# licence notices.
case "$platform" in
    macos)
        src="$TMP/x/onnxruntime-osx-arm64-1.30.0.tgz/onnxruntime-osx-arm64-1.30.0"
        cp "$src/lib/libonnxruntime.1.30.0.dylib" "$DEST/libonnxruntime.dylib"
        cp "$src/LICENSE" "$src/ThirdPartyNotices.txt" "$DEST/" ;;
    windows)
        ort="$TMP/x/microsoft.ml.onnxruntime.directml.1.24.4.nupkg"
        dml="$TMP/x/microsoft.ai.directml.1.15.4.nupkg"
        cp "$ort/runtimes/win-x64/native/onnxruntime.dll" "$ort/runtimes/win-x64/native/onnxruntime_providers_shared.dll" "$DEST/"
        cp "$ort/LICENSE" "$DEST/LICENSE"
        cp "$ort/ThirdPartyNotices.txt" "$DEST/ThirdPartyNotices.txt"
        cp "$dml/bin/x64-win/DirectML.dll" "$DEST/"
        cp "$dml/LICENSE.txt" "$DEST/DirectML-LICENSE.txt"
        cp "$dml/ThirdPartyNotices.txt" "$DEST/DirectML-ThirdPartyNotices.txt" ;;
    linux)
        src="$TMP/x/onnxruntime-linux-x64-1.30.0.tgz/onnxruntime-linux-x64-1.30.0"
        cp "$src/lib/libonnxruntime.so.1.30.0" "$DEST/libonnxruntime.so"
        cp "$src/LICENSE" "$src/ThirdPartyNotices.txt" "$DEST/" ;;
esac
echo "$platform" > "$DEST/platform"
echo "ONNX Runtime for $platform ready in $DEST"
