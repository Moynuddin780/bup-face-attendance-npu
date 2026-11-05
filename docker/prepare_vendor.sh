#!/usr/bin/env bash
# Copy the host's DEEPX runtime (DXRT) into the Docker build context.
#
# The container talks to the host's dxrtd service and kernel driver, so its libdxrt must be
# the SAME build as the host's — vendoring the installed files guarantees that (x86_64
# libdxrt-bin package or an arm64 dx_rt source install alike).
#
#   docker/prepare_vendor.sh            (run before `docker compose build`)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PREFIX="${DXRT_PREFIX:-/usr/local}"
OUT="$HERE/vendor/dxrt"

[ -f "$PREFIX/include/dxrt/dxrt_cxx_api.h" ] || [ -f "$PREFIX/include/dxrt/dxrt_api.h" ] || { echo "DXRT headers not found under $PREFIX/include/dxrt" >&2; exit 1; }
ls "$PREFIX"/lib/libdxrt.so* >/dev/null 2>&1 || { echo "libdxrt not found under $PREFIX/lib" >&2; exit 1; }

rm -rf "$OUT"
mkdir -p "$OUT/include" "$OUT/lib/cmake" "$OUT/bin"
cp -a "$PREFIX/include/dxrt" "$OUT/include/"
cp -a "$PREFIX"/lib/libdxrt.so* "$OUT/lib/"
# only the ONNX Runtime libdxrt actually loads (a host may keep several versions side by side)
if [ -e "$PREFIX/lib/libonnxruntime.so.1" ]; then
    cp -a "$PREFIX/lib/libonnxruntime.so" "$PREFIX/lib/libonnxruntime.so.1" "$OUT/lib/" 2>/dev/null || true
    cp -a "$(readlink -f "$PREFIX/lib/libonnxruntime.so.1")" "$OUT/lib/"
else
    echo "note: no libonnxruntime (DXRT built without ORT)"
fi
# a source-built libdxrt carries ~40 MB of debug info: keep the image small (same code, same ABI)
if command -v strip >/dev/null 2>&1; then
    find "$OUT/lib" -maxdepth 1 -type f -name '*.so*' -exec strip --strip-debug {} +
fi
cp -a "$PREFIX/lib/cmake/dxrt" "$OUT/lib/cmake/"
for b in dxrt-cli parse_model; do   # these are symlinks (dxcli, dxparse): copy the real binaries
    if [ -x "$PREFIX/bin/$b" ]; then cp -L "$PREFIX/bin/$b" "$OUT/bin/$b"; fi
done
VER="$(dxrt-cli -v 2>/dev/null || true)"
echo "${VER%%$'\n'*} vendored into $OUT ($(du -sh "$OUT" | cut -f1))"
