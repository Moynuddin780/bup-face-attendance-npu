#!/usr/bin/env bash
# BUP Face Attendance (C++ NPU backend) — one-time host setup.
#   1. checks the DEEPX runtime + NPU (the only external prerequisite)
#   2. checks C++ build dependencies (GStreamer, OpenCV, cmake)
#   3. creates .venv and installs Python deps
#   4. downloads the two NPU models into models/
#   5. builds the C++ engine (bup_engine.so + bupctl) and installs it next to app.py
#   6. creates .env with a fresh secret key
# Re-running is safe (every step is idempotent).
#
# Env knobs: PYTHON=python3.12  WITH_DEV=0 (skip test/benchmark deps)  DXRT_PREFIX=/usr/local
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$APP_DIR"
PY="${PYTHON:-python3}"
VENV="$APP_DIR/.venv"
WITH_DEV="${WITH_DEV:-1}"
DXRT_PREFIX="${DXRT_PREFIX:-/usr/local}"
MODEL_BASE="https://sdk.deepx.ai/modelzoo/dxnn/2_4_0"
MODELS=(scrfd-10g_640x640.dxnn arcface_iresnet100_112x112_ms1m.dxnn)

log() { echo "[setup] $*"; }
die() { echo "[setup] ERROR: $*" >&2; exit 1; }

# ── 1. DEEPX runtime + NPU ────────────────────────────────────────────────
log "1/6 DEEPX runtime"
command -v dxrt-cli >/dev/null 2>&1 || die "dxrt-cli not found — install the DEEPX runtime (libdxrt-bin or dx_rt)"
DXRT_STATUS="$(dxrt-cli -s 2>&1 || true)"
echo "$DXRT_STATUS" | grep -q "Device 0" || { echo "$DXRT_STATUS"; die "no DX-M1 device visible"; }
echo "$DXRT_STATUS" | grep -E "DXRT v|Device 0|RT Driver|FW version" | sed 's/^/[setup]   /'
[ -f "$DXRT_PREFIX/include/dxrt/dxrt_cxx_api.h" ] || [ -f "$DXRT_PREFIX/include/dxrt/dxrt_api.h" ] \
    || die "DXRT C++ headers not found in $DXRT_PREFIX/include/dxrt"
[ -f "$DXRT_PREFIX/lib/cmake/dxrt/dxrtConfig.cmake" ] || die "dxrtConfig.cmake not found in $DXRT_PREFIX/lib/cmake/dxrt"

# ── 2. build dependencies ─────────────────────────────────────────────────
log "2/6 build dependencies"
command -v cmake >/dev/null 2>&1 || die "cmake missing (apt install cmake build-essential)"
for pc in gstreamer-1.0 gstreamer-app-1.0 opencv4; do
    pkg-config --exists "$pc" || die "missing dev package for $pc — apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libopencv-dev"
done
"$PY" -c "import sysconfig,os,sys; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()['include'],'Python.h')) else 1)" \
    || die "Python headers missing (apt install python3-dev)"
for el in rtspsrc avdec_h264 avdec_h265; do
    gst-inspect-1.0 "$el" >/dev/null 2>&1 || log "  WARN: GStreamer element '$el' missing (apt install gstreamer1.0-plugins-good gstreamer1.0-libav) — FFmpeg fallback will be used"
done

# ── 3. Python venv ────────────────────────────────────────────────────────
log "3/6 Python venv ($VENV)"
# a venv copied from another machine keeps that machine's absolute paths in bin/* scripts
if [ -x "$VENV/bin/python" ] && ! "$VENV/bin/pip" --version >/dev/null 2>&1; then
    log "  $VENV was created on another machine (broken script paths) — recreating"
    rm -rf "$VENV"
fi
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r requirements.txt
if [ "$WITH_DEV" = "1" ]; then
    "$VENV/bin/pip" install -q -r requirements-dev.txt
    # dx_engine is only needed by the baseline comparison (original Python pipeline).
    # It must be the host's DXRT version: libdxrt-bin wheel, else the dx_rt source tree.
    DXRT_VER="$(dxrt-cli -v 2>/dev/null | sed -n 's/^DXRT v\([0-9.]*\).*/\1/p' | head -1)"
    HAVE_VER="$("$VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("dx-engine"))' 2>/dev/null || true)"
    PYTAG="cp$("$VENV/bin/python" -c 'import sys;print(f"{sys.version_info.major}{sys.version_info.minor}")')"
    WHL="$(ls /usr/share/libdxrt-bin/python/dx_engine-*-"$PYTAG"-*.whl 2>/dev/null | head -1 || true)"
    PYPKG="${DXRT_PYTHON_PACKAGE:-}"
    if [ -z "$PYPKG" ]; then   # dx-all-suite source install: <suite>/dx-runtime/dx_rt/python_package
        d="$APP_DIR"
        while [ "$d" != "/" ]; do
            [ -f "$d/dx-runtime/dx_rt/python_package/pyproject.toml" ] && { PYPKG="$d/dx-runtime/dx_rt/python_package"; break; }
            d="$(dirname "$d")"
        done
    fi
    if [ -n "$DXRT_VER" ] && [ "$HAVE_VER" = "$DXRT_VER" ]; then
        log "  dx_engine $HAVE_VER present (matches DXRT)"
    elif [ -n "$WHL" ]; then
        "$VENV/bin/pip" install -q --force-reinstall --no-deps "$WHL"
    elif [ -n "$PYPKG" ]; then
        log "  building dx_engine from $PYPKG (against the installed libdxrt)"
        "$VENV/bin/pip" install -q --force-reinstall --no-deps "$PYPKG"
    else
        log "  NOTE: no dx_engine for $PYTAG — baseline comparison tests will be skipped"
    fi
fi

# ── 4. NPU models ─────────────────────────────────────────────────────────
log "4/6 NPU models"
mkdir -p models
for m in "${MODELS[@]}"; do
    if [ -s "models/$m" ]; then
        log "  $m present ($(du -h "models/$m" | cut -f1))"
    else
        log "  downloading $m"
        curl -fSL --retry 3 -o "models/$m.part" "$MODEL_BASE/$m"
        mv "models/$m.part" "models/$m"
    fi
done
# the recognition model in use is built here, not downloaded (InsightFace w600k_r50 -> DX-COM 2.4.0)
if [ -s models/arcface_w600k_r50_112x112.dxnn ]; then
    log "  arcface_w600k_r50_112x112.dxnn present ($(du -h models/arcface_w600k_r50_112x112.dxnn | cut -f1))"
else
    log "  WARN: models/arcface_w600k_r50_112x112.dxnn missing — copy it from the deployed server or build it"
    log "        with tools/model_convert (README.md there); until then set REC_MODEL_PATH=models/arcface_iresnet100_112x112_ms1m.dxnn"
fi

# ── 5. C++ engine ─────────────────────────────────────────────────────────
log "5/6 building bup_engine (C++14, DXRT + OpenCV + GStreamer)"
PYBIND11_DIR="$("$VENV/bin/python" -m pybind11 --cmakedir)"
# a CMake cache from another checkout/machine refuses to configure: start clean
if [ -f build/CMakeCache.txt ] && ! grep -qx "CMAKE_HOME_DIRECTORY:INTERNAL=$APP_DIR/cpp" build/CMakeCache.txt; then
    rm -rf build
fi
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release \
      -DPython3_EXECUTABLE="$VENV/bin/python" -Dpybind11_DIR="$PYBIND11_DIR" \
      -DCMAKE_PREFIX_PATH="$DXRT_PREFIX" >/dev/null
cmake --build build -j"$(nproc)"
cp -f build/bup_engine*.so "$APP_DIR/"
mkdir -p bin && cp -f build/bupctl bin/bupctl
"$VENV/bin/python" -c "import bup_engine; print('[setup]   bup_engine', bup_engine.__version__, '| GStreamer:', bup_engine.has_gstreamer())"

# ── 6. .env ───────────────────────────────────────────────────────────────
log "6/6 configuration"
if [ ! -f .env ]; then
    cp .env.example .env
    KEY="$("$VENV/bin/python" -c 'import secrets; print(secrets.token_hex(32))')"
    sed -i "s/^BUP_SECRET_KEY=.*/BUP_SECRET_KEY=$KEY/" .env
    log "  .env created with a new secret key"
else
    log "  .env exists (kept)"
fi
# docker compose runs the container as this user, so files in ./data, ./db, ... stay yours
grep -q '^BUP_UID=' .env || printf 'BUP_UID=%s\nBUP_GID=%s\n' "$(id -u)" "$(id -g)" >> .env
mkdir -p db
log "done — start with ./run.sh"
