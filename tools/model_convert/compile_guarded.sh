#!/usr/bin/env bash
# Compile with DX-COM at low priority, stopped before the disk or the memory runs out
# (a full root filesystem crashes the attendance app).
#   compile_guarded.sh <dxcom python> <onnx> <config.json> <out dir> [dxcom options]
PY="$1"; shift
HERE="$(cd "$(dirname "$0")" && pwd)"
nice -n 15 ionice -c 3 "$PY" "$HERE/dxcom_lowram.py" -m "$1" -c "$2" -o "$3" "${@:4}" &
PID=$!
min_disk=999999; min_mem=999999; peak_rss=0
while kill -0 $PID 2>/dev/null; do
  disk=$(df --output=avail -m / | tail -1 | tr -d ' ')
  mem=$(awk '/MemAvailable/ {a=$2} /SwapFree/ {s=$2} END {print int((a+s)/1024)}' /proc/meminfo)
  rss=$(ps -o rss= --ppid $PID -p $PID 2>/dev/null | awk '{s+=$1} END {print int(s/1024)}')
  [ "$disk" -lt "$min_disk" ] && min_disk=$disk
  [ "$mem" -lt "$min_mem" ] && min_mem=$mem
  [ "${rss:-0}" -gt "$peak_rss" ] && peak_rss=$rss
  if [ "$disk" -lt 1200 ] || [ "$mem" -lt 500 ]; then
    echo "GUARD: stopping the compile (disk ${disk} MB free, memory+swap ${mem} MB available)"
    pkill -TERM -P $PID; kill -TERM $PID; sleep 5; pkill -KILL -P $PID; kill -KILL $PID 2>/dev/null
    wait $PID; echo "guard: min_disk=${min_disk}MB min_mem=${min_mem}MB peak_rss=${peak_rss}MB"; exit 3
  fi
  sleep 2
done
wait $PID; rc=$?
echo "guard: rc=$rc min_disk=${min_disk}MB min_mem=${min_mem}MB peak_rss=${peak_rss}MB"
exit $rc
