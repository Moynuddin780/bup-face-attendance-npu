#!/bin/sh
# Container start: seed the DB volume on first run, check models and the NPU, then exec CMD.
set -e
cd /app

mkdir -p "$(dirname "$BUP_DB_FILE")" data attendance_history cloud_backup uploads
if [ ! -f "$BUP_DB_FILE" ]; then
    cp /app/seed/bup_db.json "$BUP_DB_FILE"
    echo "[entrypoint] seeded $BUP_DB_FILE"
fi

for m in "$DET_MODEL_PATH" "$REC_MODEL_PATH"; do
    case "$m" in /*) f="$m" ;; *) f="/app/$m" ;; esac
    if [ ! -s "$f" ]; then
        echo "[entrypoint] ERROR: model $f missing — mount the models directory (./models:/app/models:ro)" >&2
        exit 1
    fi
done

if ! dxrt-cli -s >/dev/null 2>&1; then
    echo "[entrypoint] WARN: dxrt-cli -s failed — pass /dev/dxrt0 and make sure dxrtd runs on the host" >&2
fi
if [ -z "$BUP_SECRET_KEY" ] || [ "$BUP_SECRET_KEY" = "change-me" ]; then
    echo "[entrypoint] WARN: BUP_SECRET_KEY is not set — set it in .env" >&2
fi

exec "$@"
