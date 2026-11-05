# BUP Face Attendance — DeepX DX-M1 NPU edition (Flask + C++ engine)
#
# Build context = project root:   docker/prepare_vendor.sh && docker compose build
# The host keeps the NPU kernel driver and the dxrtd service; the image carries the
# matching DXRT user-space library (vendored from the host by prepare_vendor.sh).

ARG UBUNTU=24.04

# ── stage 1: build the C++ engine ────────────────────────────────────────────
FROM ubuntu:${UBUNTU} AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config python3 python3-dev python3-venv \
        libopencv-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    && rm -rf /var/lib/apt/lists/*
COPY docker/vendor/dxrt/include/ /usr/local/include/
COPY docker/vendor/dxrt/lib/ /usr/local/lib/
RUN ldconfig

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

COPY cpp /src/cpp
RUN cmake -S /src/cpp -B /src/build -DCMAKE_BUILD_TYPE=Release \
        -DPython3_EXECUTABLE=/opt/venv/bin/python \
        -Dpybind11_DIR="$(/opt/venv/bin/python -m pybind11 --cmakedir)" \
        -DCMAKE_PREFIX_PATH=/usr/local \
    && cmake --build /src/build -j"$(nproc)"

# ── stage 2: runtime ─────────────────────────────────────────────────────────
FROM ubuntu:${UBUNTU} AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 ca-certificates curl \
        libopencv-core406t64 libopencv-imgproc406t64 libopencv-imgcodecs406t64 libopencv-videoio406t64 \
        gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav \
        intel-media-va-driver \
    && rm -rf /var/lib/apt/lists/*
COPY docker/vendor/dxrt/lib/ /usr/local/lib/
COPY docker/vendor/dxrt/bin/ /usr/local/bin/
RUN ldconfig

COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY app.py camera_control.py fix_paths.py migrate_data.py requirements.txt ./
COPY templates ./templates
COPY static ./static
COPY --from=build /src/build/bup_engine*.so /app/
COPY --from=build /src/build/bupctl /usr/local/bin/bupctl
# first-run seed for the DB volume (the entrypoint copies it only if /app/db is empty)
COPY bup_db.json /app/seed/bup_db.json
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && mkdir -p /app/db /app/data /app/attendance_history /app/cloud_backup /app/uploads /app/models

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    BUP_DB_FILE=/app/db/bup_db.json \
    DET_MODEL_PATH=models/scrfd-10g_640x640.dxnn \
    REC_MODEL_PATH=models/arcface_w600k_r50_112x112.dxnn \
    STREAM_PROFILE=sub \
    RTSP_TRANSPORT=tcp \
    SUBSTREAM_ASPECT=auto \
    NPU_BOUND=auto \
    DXRT_DYNAMIC_CPU_THREAD=ON \
    BUP_HOST=0.0.0.0 \
    BUP_PORT=5000 \
    GUNICORN_THREADS=16

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${BUP_PORT}/" -o /dev/null || exit 1
ENTRYPOINT ["/entrypoint.sh"]
# ONE worker: the live session and the NPU engine are process-wide; threads serve the pages
CMD ["sh", "-c", "exec gunicorn --workers 1 --worker-class gthread --threads ${GUNICORN_THREADS} --bind ${BUP_HOST}:${BUP_PORT} --timeout 120 --graceful-timeout 20 --access-logfile - app:app"]
