"""Live view at the camera's frame rate: decoding on the Intel iGPU (VA-API) when available, the
picture independent of the NPU (faces drawn from the latest results), CPU decoding as fallback."""
import json
import os
import subprocess
import time

import numpy as np
import pytest

from conftest import APP_DIR

MEDIA = APP_DIR / "tests" / "media"
PORT = int(os.environ.get("RTSP_TEST_PORT", "18554"))
BASE = f"rtsp://127.0.0.1:{PORT}/Streaming/Channels"


@pytest.fixture(scope="module")
def streams():
    if not (MEDIA / "streams.json").exists():
        subprocess.run([str(APP_DIR / ".venv" / "bin" / "python"), str(APP_DIR / "tools" / "make_test_streams.py")],
                       check=True)
    subprocess.run(["bash", str(APP_DIR / "tools" / "rtsp_server.sh"), "start"], check=True)
    yield json.loads((MEDIA / "streams.json").read_text())
    subprocess.run(["bash", str(APP_DIR / "tools" / "rtsp_server.sh"), "stop"], check=False)


def _run(engine_module, face_engine, hw_decode, seconds=6.0):
    meta = [{"student_id": f"S{i}", "name": f"Student {i}"} for i in range(4)]
    emb = np.random.default_rng(0).standard_normal((4, 512)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    sess = engine_module.AttendanceSession(
        face_engine, engine_module.AttendanceTracker(meta, 60), emb, meta,
        [{"id": "camera1", "url": f"{BASE}/101", "profile": "main", "fixed_url": True, "max_w": 0, "max_h": 0},
         {"id": "camera2", "url": f"{BASE}/202", "profile": "sub", "fixed_url": True, "max_w": 0, "max_h": 0,
          "aspect": "16:9"}],
        jpeg_quality=80, hw_decode=hw_decode, gpu_max_width=1920)
    sess.start()
    try:
        deadline = time.time() + 30
        while time.time() < deadline and not all(sess.camera(c).get_info()["feed_fps"] > 0 for c in sess.camera_ids()):
            time.sleep(0.5)
        time.sleep(2)
        pictures, last, t0 = 0, None, time.monotonic()
        while time.monotonic() - t0 < seconds:   # what the browser gets: polled like _gen_feed
            jpg = sess.mjpeg_frame()
            if jpg != last:
                pictures, last = pictures + 1, jpg
            time.sleep(1 / 30)
        rate = pictures / (time.monotonic() - t0)
        return rate, {c: sess.camera(c).get_info() for c in sess.camera_ids()}, last
    finally:
        sess.stop()


def test_gpu_decoding_at_camera_rate(streams, engine_module, face_engine):
    if not engine_module.gpu_decoding_available():
        pytest.skip("no VA-API GPU decoding on this machine")
    rate, cams, _ = _run(engine_module, face_engine, hw_decode=True)
    src = streams["fps"]   # the test cameras send 20 fps
    for cid, info in cams.items():
        assert info["decoder"] == "gpu" and "VA-API" in info["backend"], (cid, info)
        assert info["feed_fps"] >= 0.85 * src, (cid, info)
    assert cams["camera1"]["stream_size"] == "1920x1080" and cams["camera2"]["stream_size"] == "704x576"
    assert rate >= 0.85 * src, rate   # the live view shows every camera frame


def test_gpu_scales_wide_streams(streams, engine_module, face_engine):
    if not engine_module.gpu_decoding_available():
        pytest.skip("no VA-API GPU decoding on this machine")
    meta = [{"student_id": "S0", "name": "S"}]
    sess = engine_module.AttendanceSession(
        face_engine, engine_module.AttendanceTracker(meta, 60), np.ones((1, 512), np.float32) / np.sqrt(512), meta,
        [{"id": "camera1", "url": f"{BASE}/101", "profile": "main", "fixed_url": True, "max_w": 0, "max_h": 0}],
        hw_decode=True, gpu_max_width=1280)
    sess.start()
    try:
        deadline = time.time() + 30
        while time.time() < deadline and not sess.camera("camera1").get_info()["frame_size"]:
            time.sleep(0.5)
        info = sess.camera("camera1").get_info()
    finally:
        sess.stop()
    assert info["stream_size"] == "1920x1080" and info["frame_size"] == "1280x720", info   # scaled on the GPU


def test_cpu_decoding_fallback(streams, engine_module, face_engine):
    rate, cams, jpg = _run(engine_module, face_engine, hw_decode=False)
    for cid, info in cams.items():
        assert info["decoder"] == "cpu" and "VA-API" not in info["backend"], (cid, info)
        assert info["feed_fps"] >= 0.85 * streams["fps"], (cid, info)
    import cv2
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 800 and img.shape[0] > 2 * 400   # two cameras stacked (+ stats strip)
