<p align="center">
  <img src="docs/media/banner.svg" alt="BUP Face Attendance — DEEPX DX-M1 NPU edition" width="100%">
</p>

<p align="center">
<b>Hands-free classroom attendance for Bangladesh University of Professionals.</b><br>
IP-CCTV cameras mark students present automatically — detection and recognition run in a C++ engine on the <b>DEEPX DX-M1 NPU</b>.
</p>

<p align="center">
  <img alt="C++ 14" src="https://img.shields.io/badge/C%2B%2B-14-00599C?style=flat-square&logo=cplusplus&logoColor=white">
  <img alt="Flask gunicorn" src="https://img.shields.io/badge/Flask-gunicorn-000000?style=flat-square&logo=flask&logoColor=white">
  <img alt="Docker ready" src="https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white">
  <img alt="NPU DEEPX DX-M1" src="https://img.shields.io/badge/NPU-DEEPX%20DX--M1-0A7E6C?style=flat-square">
  <img alt="rank-1 accuracy 99.0%" src="https://img.shields.io/badge/rank--1%20accuracy-99.0%25-2EA44F?style=flat-square">
  <img alt="tests 30+" src="https://img.shields.io/badge/tests-30%2B-6F42C1?style=flat-square">
</p>

<p align="center">
  <img src="docs/media/demo.gif" alt="Live classroom demo: every face is detected and recognised on the NPU (faces blurred for privacy)" width="760">
</p>

<p align="center">
  <a href="docs/media/demo.mp4"><b>▶ Full demo video</b></a> &nbsp;·&nbsp; <a href="#screenshots">Screenshots</a> &nbsp;·&nbsp; <a href="#architecture">Architecture</a> &nbsp;·&nbsp; <a href="#quick-start">Quick start</a> &nbsp;·&nbsp; <a href="#documentation">Documentation</a>
</p>

---

## ✨ Highlights

| Highlight | Details |
|---|---|
| 🎯 **99.0 % rank-1 identification** | up from 92.8 % in the original Python pipeline (318 enrolled photos, leave-one-out) — and 0 % wrong students marked |
| ⚡ **1.7× faster, 3.1× less CPU** | the per-frame work runs in C++ on the DX-M1 NPU and reads the cameras' sub-streams, which cost about 10× less to decode |
| 🧠 **Detect → recognise → track** | SCRFD-10G detection, ArcFace embeddings, cosine matching and an attendance tracker per camera; detection of frame *t+1* overlaps recognition of frame *t* |
| 🏫 **Built for real classrooms** | role-based dashboards (super admin, department admin, teacher), a Live Monitor of every running class, CSV/JSON export, cloud backup and manual override |
| 🔁 **Resilient** | running classes are saved every 5 s and resume by themselves after a crash, deploy or reboot; cameras reconnect on their own |
| 📦 **Ships as Docker** | one image and a compose file, `.env` configuration, healthcheck and restart policy |

> **About this edition.** The application is kept exactly as it was — all 25 templates and every route, the database schema and roles, the attendance rules and the saved record format, the live overlay. Only the per-frame work moved into the C++ engine.

<a id="screenshots"></a>

## 🖼 Screenshots

<table>
  <tr>
    <td width="50%"><img src="docs/media/screenshots/01-live-recognition.jpg" alt="Live face recognition in a classroom"><br><sub><b>Live recognition</b> — every face in the room is detected and matched in real time</sub></td>
    <td width="50%"><img src="docs/media/screenshots/02-multi-student-tracking.jpg" alt="Several students tracked at the same time"><br><sub><b>Many students at once</b> — dozens of faces tracked in every frame</sub></td>
  </tr>
</table>

<sub>Frames from the demo video (<a href="docs/media/demo.mp4">docs/media/demo.mp4</a> — open it for the full 28-second run). Student faces and names are blurred for privacy.</sub>

<a id="architecture"></a>

## 🏗 Architecture at a glance

```mermaid
flowchart LR
  CAM["IP-CCTV cameras<br/>RTSP sub-stream"] --> CAP
  subgraph ENGINE["bup_engine — C++ · pybind11 · DXRT"]
    CAP["capture<br/>GStreamer · CPU decode"] --> DET["SCRFD-10G<br/>face detection · NPU"]
    DET --> REC["ArcFace<br/>embedding · NPU"]
    REC --> MATCH["cosine match<br/>+ attendance tracker"]
  end
  MATCH --> OVL["live overlay<br/>JPEG"]
  MATCH --> APP["Flask app<br/>routes · database · roles"]
  APP --> UI["dashboards<br/>admin · department · teacher"]
  OVL --> UI
```

## 🧰 Tech stack

| Layer | Technology |
|---|---|
| **Engine** | C++14 · pybind11 · DEEPX DXRT C++ API · GStreamer · OpenCV |
| **Models** | SCRFD-10G (detection) · ArcFace (recognition), compiled to `.dxnn` for the DX-M1 NPU |
| **Application** | Flask + gunicorn · 25 server-side templates · JSON database with automatic backups · role-based access |
| **Deployment** | Docker + Compose (host IPC/PID namespaces for DXRT) · `.env` configuration |
| **Quality** | pytest — tracker parity on 40 000 random events, UI parity, end-to-end sessions through Flask and Docker · benchmark and accuracy tools |

<a id="quick-start"></a>

## 🚀 Quick start

```bash
./setup.sh                      # checks DXRT + NPU, builds bup_engine, writes .env
./run.sh                        # gunicorn → http://<host>:5000

# or with Docker
docker/prepare_vendor.sh && docker compose up -d --build
```

Details, prerequisites and configuration are in the documentation below.

<a id="documentation"></a>

## 📚 Documentation

## What changed — and what did not

| Unchanged (verified by tests) | New |
|---|---|
| 25 templates + static files — byte-identical | `bup_engine` C++ module: camera capture, NPU inference, matching, tracker, live overlay |
| Every route/endpoint/method; every page renders the same HTML | Cameras read on the **sub-stream** (URL rewritten at connect time; DB untouched) |
| DB schema, roles, CRUD, history, CSV/JSON export, cloud backup | NPU work submitted asynchronously and spread over the 3 NPU cores |
| Attendance rules: 60 s = Present, 2 s gap rule per camera, manual override, `SIM_THRESHOLD` 0.45 | Recognition accuracy fixes (letterbox, SCRFD decode, alignment, normalization) |
| Live page look: boxes, names, seconds, "Unknown", stats bar | Docker image + compose, gunicorn, `.env` config, atomic DB writes |

### Which part moved to C++

```
Python (Flask, unchanged)                     C++  bup_engine (pybind11, DXRT C++ API)
─────────────────────────                     ────────────────────────────────────────────────
routes, templates, DB, roles     ──start──►   per camera:
build_embeddings_for_class                      capture   GStreamer rtspsrc → avdec_h264 (CPU) → newest frame
/live_stats /live_list /manual_mark  ◄─────     detect    letterbox → SCRFD-10G (NPU, RunAsync) → decode + NMS
/video_feed  ◄─ JPEG (encoded once) ─────       recognize 5-pt align → ArcFace-R100 (NPU, all faces in parallel)
                                                          → cosine match → AttendanceTracker → overlay
                                              compositor: 2-camera view + stats bar → cached JPEG
```

The code is **C++14** and follows dx_app's **IFactory** pattern: `ScrfdFactory` and `ArcFaceFactory` are built from vendored dx_app processors (`cpp/third_party/dxapp`, DEEPX sources — not included in this repository, see its `VENDORED.md`). Detect and recognize are pipelined per camera, so SCRFD on frame *t+1* overlaps ArcFace on frame *t*.

### Accuracy fixes

Measured on the 318 enrolled photos, leave-one-out, using the app's own enrollment and matching math:

| | Original Python | v2 C++ |
|---|---|---|
| Face detected | 97.5 % | 97.8 % |
| Rank-1 identification | 92.8 % | **99.0 %** |
| Correct student marked (score ≥ 0.45) | 80.8 % | **96.4 %** |
| Wrong student marked (score ≥ 0.45) | 0 % | 0 % |
| Mean genuine similarity | 0.620 | 0.754 |

What was fixed:
- **Squashed frames:** a 16:9 frame used to be squashed into 640×640. It is now letterboxed.
- **SCRFD outputs:** they are bound by tensor *name*, where they used to be bound by position.
- **Anchor offset:** the anchor centres had an offset of +0.5 stride, which is now removed.
- **Enrollment averaging:** the embeddings are now L2-normalized *before* averaging. The old `normed_embedding` was never normalized.
- **Alignment:** the 5-point alignment now uses the InsightFace (Umeyama) similarity transform.
- **Stale cache:** the embedding cache now carries a fingerprint (models, pipeline, roster, photos). Deleted students no longer linger. Old caches are rebuilt automatically on the first session of each course, in about 10–20 s.

### Performance

Benchmark on identical input: two RTSP cameras, 20 fps, 4–5 faces per frame, 30 s (`tools/bench.py`).

| Mode | Decoded → processed | Processed fps / camera | CPU cores used |
|---|---|---|---|
| Original Python, main stream | 1920×1080 → 1280×720 | 8.7 | 1.29 |
| v2 C++, main stream | 1920×1080 → 1280×720 | 14.5 (1.7×) | 0.72 |
| **v2 C++, sub-stream** | **704×576 → 1024×576** | **14.4 (1.7×)** | **0.42 (3.1× less)** |

> These numbers were measured while another application (VisionX edge) was using the same NPU. It pinned all 3 cores, so absolute fps is lower than on a dedicated NPU. Sub-stream decoding costs about **10× less CPU per frame** than a 2–4 MP main stream: 1.7 ms vs 17.9 ms per frame, measured on this laptop. The saving grows with the camera resolution and the number of rooms.

---

## Quick start — on the host

```bash
./setup.sh          # checks DXRT + NPU, creates .venv, downloads models, builds bup_engine, writes .env
./run.sh            # gunicorn (1 worker, 16 threads)  →  http://<host>:5000
./run.sh dev        # Flask dev server
```

Sign in with the default `superadmin` account and **change its password immediately** — the initial credentials are created on first start.

Prerequisites:
- the DEEPX runtime (`dxrt-cli -s` shows the device)
- `cmake`, `g++`, `python3-dev`
- `libopencv-dev`, `libgstreamer1.0-dev`, `libgstreamer-plugins-base1.0-dev`
- `gstreamer1.0-plugins-{good,bad}`, `gstreamer1.0-libav`

## Quick start — Docker (deployment)

```bash
./setup.sh                      # once: models + .env (secret key, your UID/GID)
docker/prepare_vendor.sh        # once per host DXRT version: copies the host's DXRT library into the build
docker compose up -d --build
docker compose logs -f
```

What the compose file sets up:
- **`network_mode: host`** reaches the cameras and dxrtd's abstract IPC socket.
- **`pid: host`** is required. libdxrt checks `/proc` for the `dxrtd` process, and without it every inference fails with *"dxrt service is not running"*.
- **`ipc: host`** is required with DXRT 3.3.x, which talks to dxrtd over SysV message queues and `/dev/shm`. Without it the first attendance session hangs at *"Initializing DeepX InferenceEngine"*.
- **`/dev/dxrt0`** is passed through, and the container runs as your UID.
- **Volumes:** `data/`, `attendance_history/`, `cloud_backup/`, `uploads/`, `db/`, and `models/` (read-only).
- **First start:** `db/bup_db.json` is seeded from `./bup_db.json`. From then on `db/` is the live DB.
- **Robustness:** a healthcheck, and `restart: unless-stopped`.

---

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `BUP_SECRET_KEY` | generated by setup.sh | Flask session key (keep it stable and secret) |
| `STREAM_PROFILE` | `sub` | `sub` = sub-stream, `main` = full resolution |
| `RTSP_TRANSPORT` | `tcp` | `tcp` (no packet-loss smearing) \| `udp` \| `udp+tcp` |
| `SUBSTREAM_ASPECT` | `auto` | Fixes squashed D1 sub-streams: `auto` \| `off` \| `16:9` |
| `NPU_BOUND` | `auto` | NPU cores: `auto` \| `all` \| `0` \| `1` \| `2` |
| `DET_MODEL_PATH` / `REC_MODEL_PATH` | `models/…` | SCRFD-10G / ArcFace w600k ResNet-50 `.dxnn` (built by `tools/model_convert`, not downloaded) |
| `SIM_THRESHOLD` | per model | match threshold; default 0.42 for w600k_r50, 0.45 for the earlier iResNet100-MS1M |
| `BUP_HOST` / `BUP_PORT` / `GUNICORN_THREADS` | `0.0.0.0` / `5000` / `16` | Web server |
| `DXRT_DYNAMIC_CPU_THREAD` | `ON` | SCRFD has a CPU tail task — multi-threaded |
| `FLASK_DEBUG` | `0` | Never `1` in production (the Werkzeug debugger allows remote code execution) |

`MIN_PRESENT_SEC`, `FRAME_W` and `FRAME_H` stay as constants in `app.py`, as before; `SIM_THRESHOLD` follows the recognition model (`MODEL_SIM_THRESHOLDS`).

## Cameras and the sub-stream

- **URL rewriting:** the DB keeps the main-stream URLs, and the engine rewrites them at connect time.
  - Dahua: `…subtype=0` becomes `subtype=1`.
  - Hikvision: `/Streaming/Channels/X01` becomes `X02`, and `/h264/chN/main/` becomes `/sub/`.
  - Other vendors are used as stored, and a warning is logged.
- **Squashed D1 streams:** a Dahua D1 sub-stream (704×576) carries the 16:9 scene squashed horizontally, which hurts face recognition. With `SUBSTREAM_ASPECT=auto`, the engine reads the true aspect ratio from one main-stream frame at connect time and re-stretches every sub-stream frame, for example to 1024×576.
- **Recommended camera setting:** set the sub-stream to **1280×720 (or 704×576), H.264, 15–20 fps, ≥ 1 Mbps**. Faces from far rows need pixels. If recognition of the back rows is weak, use `STREAM_PROFILE=main` for that deployment.
- **Checking a camera without the web app:**
  ```bash
  bin/bupctl stream "rtsp://user:pass@10.10.4.2:554/cam/realmonitor?channel=1&subtype=0" --seconds 10
  # Docker: docker exec bup-attendance bupctl stream "<url>"
  ```
- **Diagnostics:** `/api/stream_info` shows, per camera:
  - backend, decoded size and processing size
  - decode fps and inference fps
  - detect/recognize latency
  - reconnect and error counters

## Operations

**Live Monitor** (super admin, sidebar): the live picture of every running class on one page, with room,
course, teacher, attendance so far and each camera's frame rate; a class can be enlarged or ended from
there. Each tile fetches the same JPEG its teacher sees, one at a time (1 picture/s by default, none for
tiles out of view), so the page holds no stream connections open.

```bash
bin/bupctl detect data/<student>/img1.jpg --out /tmp/det.jpg --crops /tmp/crops   # NPU check on a photo
bin/bupctl bench  data/<student>/img1.jpg --iters 50                              # detect + embed latency
dxrt-cli -s                                                                        # NPU status
```

**Sharing the NPU with another application.**
- dxrtd accepts at most 3 different core bindings on one device. If another process (for example VisionX edge) pins NPU_0, NPU_1 and NPU_2, NPU_ALL is refused.
- With `NPU_BOUND=auto` the engine then opens one engine per core and spreads the work itself, and logs a warning.
- On a dedicated NPU it uses NPU_ALL.

**Several classes at once, and restarts.**
- One camera stream is read once, however many running classes use it. Cameras allow only a few viewers, so a second connection made them drop each other.
- GPU decoders are kept across camera reconnects and reused by later classes. GStreamer 1.24's `vapostproc` never frees ~17 file descriptors per instance. Rebuilding it on every reconnect of a Wi-Fi camera used up the worker's 1024 descriptors within 5–10 minutes, and the worker crashed with every class in it (2026-10-12).
- The open-file limit is 65536 (`docker-compose.yml`, `run.sh`, and the engine itself). `/api/stream_info` shows `open_files` to super admins. The session start/stop log lines show it too.
- A camera's frame rate is what its network delivers. On 2026-10-12, with three classes running, the wired cameras gave a steady 20/25 fps. The three cameras behind Wi-Fi (RTT 0.1–1.5 s, up to 20 % loss) swung between 1 and 34 fps. Fixes: wire them (PoE), or make their stream fit the link on Buildings & Cameras. 1080p H.265 at 1024–1536 kbps (H.264 at 2048) with an I-frame every second fits. `/api/stream_info` shows each camera's `stream_kbps`, `stream_gaps` and `network_rtt_ms`.
- Data lost on the way is never decoded into grey, smeared pictures: frames are skipped to the next key frame (`frames_skipped`).
- Running classes are saved to `db/live_sessions.json` every 5 s. After a crash, deploy or reboot they continue by themselves with the time measured so far ("Resumed at …" on the live page). A class that cannot continue goes to its teacher's review-and-save page instead (`RESUME_MAX_GAP_S`).

---

## Tests and benchmark

```bash
.venv/bin/python -m pytest tests/ -v          # 30+ tests: tracker parity, NPU accuracy, RTSP sessions, Flask UI parity, Docker
tools/rtsp_server.sh start                     # local RTSP "cameras" (mediamtx) built from enrolled photos
.venv/bin/python tools/bench.py --all          # original vs C++ main vs C++ sub → bench_report.md
```

What the tests cover:
- **Tracker parity:** the C++ `AttendanceTracker` against the original Python class, on 40 000 random events.
- **UI parity:** every page for super admin, dept admin and teacher, rendered by the original and by the new app against the same DB, must produce identical HTML.
- **Sessions end to end:** full attendance sessions through Flask and through the Docker container, run as a non-root user.

## Known behaviour

- **Dual-camera time counting.** The tracker adds time per camera, as the original did. A student seen by both cameras accumulates time about 2× faster, for example 42 s in a 25 s session. If you want wall-clock time, this is a one-line change in `AttendanceTracker::update` (C++).

## Layout

```
app.py                  Flask app (routes/UI unchanged; hot path → bup_engine)
cpp/                    C++ engine: include/bup, src, tools/bupctl.cpp, third_party/dxapp (DEEPX sources, not in this repo — see VENDORED.md)
templates/ static/      unchanged
data/ attendance_history/ cloud_backup/ uploads/ db/          runtime data (private, git-ignored)
models/                 SCRFD-10G + ArcFace .dxnn (not in this repo — see models/README.md)
docker/ docker-compose.yml .dockerignore     deployment
tests/ tools/           tests, RTSP test cameras, benchmark, accuracy evaluation
setup.sh run.sh start.sh .env.example requirements*.txt
```

---

## 👤 Author

**Moynuddin Al Masum** — AI Engineer (Computer Vision) at SmartData Technologies

[![Portfolio](https://img.shields.io/badge/Portfolio-moynuddin780.github.io-0F6B54?style=flat-square)](https://moynuddin780.github.io)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Moynuddin%20Al%20Masum-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/moynuddin-al-masum-683102375)
[![GitHub](https://img.shields.io/badge/GitHub-Moynuddin780-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Moynuddin780)

## 📄 Licence and third-party notice

No open-source licence is granted yet — **all rights reserved**. Please get in touch before reusing the code.

The DEEPX runtime (DXRT), the DEEPX `dx_app` sample sources and the compiled `.dxnn` models are **not** part of this repository and remain subject to DEEPX's and the model authors' own terms.
