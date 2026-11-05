#!/usr/bin/env bash
# Local RTSP "cameras" for testing without real IP cameras.
#
#   tools/rtsp_server.sh start   -> mediamtx on :18554 + 4 looping ffmpeg publishers
#   tools/rtsp_server.sh stop
#
# Paths follow the Hikvision ISAPI scheme so the sub-stream rewrite is exercised:
#   rtsp://127.0.0.1:18554/Streaming/Channels/101  camera1 main (1920x1080)
#   rtsp://127.0.0.1:18554/Streaming/Channels/102  camera1 sub  (704x576, anamorphic)
#   rtsp://127.0.0.1:18554/Streaming/Channels/201  camera2 main
#   rtsp://127.0.0.1:18554/Streaming/Channels/202  camera2 sub
# Media comes from tools/make_test_streams.py (tests/media/).
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
BIN_DIR="$APP_DIR/tools/.bin"
MEDIA="${MEDIA_DIR:-$APP_DIR/tests/media}"
RUN_DIR="${RTSP_RUN_DIR:-$APP_DIR/tools/.run}"
PORT="${RTSP_TEST_PORT:-18554}"
MTX_VERSION="v1.21.1"

start() {
    mkdir -p "$BIN_DIR" "$RUN_DIR"
    if [ ! -x "$BIN_DIR/mediamtx" ]; then
        local arch; arch="$(uname -m)"; [ "$arch" = "x86_64" ] && arch=amd64; [ "$arch" = "aarch64" ] && arch=arm64
        curl -fsSL "https://github.com/bluenviron/mediamtx/releases/download/${MTX_VERSION}/mediamtx_${MTX_VERSION}_linux_${arch}.tar.gz" \
            | tar -xz -C "$BIN_DIR" mediamtx
    fi
    for f in camera1_main camera1_sub camera2_main camera2_sub; do
        [ -s "$MEDIA/$f.mp4" ] || { echo "missing $MEDIA/$f.mp4 — run tools/make_test_streams.py" >&2; exit 1; }
    done
    stop >/dev/null 2>&1 || true

    # every protocol except RTSP off; RTP/RTCP on non-default ports (no clash with other stacks);
    # without a config file mediamtx serves no paths, so accept any publisher path
    cat >"$RUN_DIR/mediamtx.yml" <<YML
rtspAddress: :$PORT
rtpAddress: :18000
rtcpAddress: :18001
rtmp: no
hls: no
webrtc: no
srt: no
api: no
metrics: no
pprof: no
playback: no
paths:
  all_others:
YML
    # run inside RUN_DIR: mediamtx drops auto-generated files (e.g. TLS certs) in its cwd
    (cd "$RUN_DIR" && exec "$BIN_DIR/mediamtx" "$RUN_DIR/mediamtx.yml") >"$RUN_DIR/mediamtx.log" 2>&1 &
    echo $! >"$RUN_DIR/mediamtx.pid"
    for _ in $(seq 1 50); do (echo >/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null && break; sleep 0.1; done

    publish() {  # <file> <channel>
        ffmpeg -loglevel error -re -stream_loop -1 -i "$MEDIA/$1.mp4" -c copy -f rtsp -rtsp_transport tcp \
            "rtsp://127.0.0.1:$PORT/Streaming/Channels/$2" >"$RUN_DIR/pub_$2.log" 2>&1 &
        echo $! >"$RUN_DIR/pub_$2.pid"
    }
    publish camera1_main 101
    publish camera1_sub 102
    publish camera2_main 201
    publish camera2_sub 202
    sleep 1.5
    echo "RTSP test cameras on rtsp://127.0.0.1:$PORT/Streaming/Channels/{101,102,201,202}"
}

stop() {
    for p in "$RUN_DIR"/*.pid; do
        [ -f "$p" ] || continue
        kill "$(cat "$p")" 2>/dev/null || true
        rm -f "$p"
    done
    echo "RTSP test cameras stopped"
}

case "${1:-}" in
    start) start ;;
    stop) stop ;;
    *) echo "usage: $0 start|stop" >&2; exit 2 ;;
esac
