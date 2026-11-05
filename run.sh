#!/usr/bin/env bash
# Start BUP Face Attendance on this host (no Docker).
#
#   ./run.sh          production: gunicorn, 1 worker x $GUNICORN_THREADS threads
#   ./run.sh dev      Flask development server (python app.py)
#
# Configuration comes from .env (see .env.example). Run ./setup.sh once first.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$APP_DIR"
# several classes need far more than the usual 1024 open files (GStreamer + VA-API + DXRT per camera)
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# venv: local .venv / venv, else the dx-all-suite shared runtime venv (if this dir lives inside it)
VENV=""
for v in "$APP_DIR/.venv" "$APP_DIR/venv"; do
    [ -x "$v/bin/python" ] && { VENV="$v"; break; }
done
if [ -z "$VENV" ]; then
    d="$APP_DIR"
    while [ "$d" != "/" ]; do
        [ -x "$d/dx-runtime/venv-dx-runtime/bin/python" ] && { VENV="$d/dx-runtime/venv-dx-runtime"; break; }
        d="$(dirname "$d")"
    done
fi
[ -n "$VENV" ] || { echo "No Python venv found — run ./setup.sh first" >&2; exit 1; }

# .env provides defaults; variables already set in the environment win (BUP_PORT=5077 ./run.sh)
if [ -f .env ]; then
    while IFS='=' read -r key value; do
        case "$key" in ''|\#*) continue ;; esac
        [ -z "${!key+x}" ] && export "$key=$value"
    done < .env
fi
# the live DB is db/bup_db.json (shared with the Docker deployment); ./bup_db.json only seeds it
if [ -z "${BUP_DB_FILE:-}" ] && [ -f "$APP_DIR/db/bup_db.json" ]; then
    export BUP_DB_FILE="$APP_DIR/db/bup_db.json"
fi
export DET_MODEL_PATH="${DET_MODEL_PATH:-models/scrfd-10g_640x640.dxnn}"
export REC_MODEL_PATH="${REC_MODEL_PATH:-models/arcface_iresnet100_112x112_ms1m.dxnn}"
export DXRT_DYNAMIC_CPU_THREAD="${DXRT_DYNAMIC_CPU_THREAD:-ON}"   # SCRFD's CPU tail task

# guards: fail early with the fix instead of a stack trace
for m in "$DET_MODEL_PATH" "$REC_MODEL_PATH"; do
    [ -s "$m" ] || { echo "Model $m missing — run ./setup.sh (downloads it)" >&2; exit 1; }
done
ls bup_engine*.so >/dev/null 2>&1 || { echo "C++ engine not built — run ./setup.sh" >&2; exit 1; }
dxrt-cli -s >/dev/null 2>&1 || echo "WARN: DX-M1 NPU not visible (dxrt-cli -s) — inference will fail" >&2

echo "BUP Face Attendance (DeepX DX-M1, C++ engine) -> http://${BUP_HOST:-0.0.0.0}:${BUP_PORT:-5000}"
case "${1:-serve}" in
    serve)
        if [ -x "$VENV/bin/gunicorn" ]; then
            exec "$VENV/bin/gunicorn" --workers 1 --worker-class gthread --threads "${GUNICORN_THREADS:-16}" \
                --bind "${BUP_HOST:-0.0.0.0}:${BUP_PORT:-5000}" --timeout 120 --graceful-timeout 20 \
                --access-logfile - app:app
        fi
        echo "gunicorn not installed in $VENV — falling back to the Flask server" >&2
        exec "$VENV/bin/python" app.py
        ;;
    dev)
        exec "$VENV/bin/python" app.py
        ;;
    *)
        echo "usage: $0 [serve|dev]" >&2
        exit 2
        ;;
esac
