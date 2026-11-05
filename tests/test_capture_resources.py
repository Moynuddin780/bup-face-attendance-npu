"""Long-running classes must not use up the process's file descriptors. GStreamer 1.24's vapostproc
never frees ~17 descriptors per instance: rebuilt on every reconnect of a flaky camera it ran the
worker out of descriptors within minutes and every running class ended at once (2026-10-12).
Cameras shared by several rooms must be read once, not once per class."""
import gc
import os
import subprocess
import time

import numpy as np
import pytest

from conftest import APP_DIR

PORT = int(os.environ.get("RTSP_TEST_PORT", "18554"))
BASE = f"rtsp://127.0.0.1:{PORT}/Streaming/Channels"
META = [{"student_id": "S0", "name": "S"}]
EMB = np.ones((1, 512), np.float32) / np.sqrt(512)


@pytest.fixture(scope="module")
def rtsp():
    subprocess.run(["bash", str(APP_DIR / "tools" / "rtsp_server.sh"), "start"], check=True, capture_output=True)
    yield
    subprocess.run(["bash", str(APP_DIR / "tools" / "rtsp_server.sh"), "stop"], check=False, capture_output=True)


def _session(engine_module, face_engine, cams, profile="main"):
    return engine_module.AttendanceSession(face_engine, engine_module.AttendanceTracker(META, min_time=60), EMB, META,
                                           cams, stream_profile=profile)


def _fds(engine_module):
    return engine_module.open_files()[0]


def _wait_frames(sess, cam="camera1", n=20, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        if sess.camera(cam).get_info()["frames_decoded"] >= n:
            return True
        time.sleep(0.5)
    return False


def test_open_file_limit_is_raised(engine_module):
    n, limit = engine_module.open_files()
    assert n > 0 and (limit == -1 or limit >= 4096)


def test_reconnects_do_not_leak_descriptors(rtsp, engine_module, face_engine):
    s = _session(engine_module, face_engine, [("camera1", f"{BASE}/101"), ("camera2", f"{BASE}/201")])
    s.start()
    try:
        assert _wait_frames(s) and _wait_frames(s, "camera2")
        before = _fds(engine_module)
        for k in range(4):   # every camera drops and reconnects
            subprocess.run(["bash", str(APP_DIR / "tools" / "rtsp_server.sh"), "start"], check=True, capture_output=True)
            end = time.time() + 30
            while time.time() < end and min(s.camera(c).get_info()["reconnects"] for c in ("camera1", "camera2")) <= k:
                time.sleep(0.5)
            got = s.camera("camera1").get_info()["frames_decoded"]
            assert _wait_frames(s, n=got + 10), s.camera("camera1").get_info()
        info = {c: s.camera(c).get_info() for c in ("camera1", "camera2")}
        assert all(i["reconnects"] >= 4 for i in info.values()), info
        assert _fds(engine_module) - before <= 4, (before, _fds(engine_module))   # was +17 per camera per reconnect
    finally:
        s.stop()


def test_classes_in_a_row_do_not_leak_descriptors(rtsp, engine_module, face_engine):
    cams = [("camera1", f"{BASE}/101"), ("camera2", f"{BASE}/201")]
    counts = []
    for _ in range(4):
        s = _session(engine_module, face_engine, cams)
        s.start()
        assert _wait_frames(s, n=10)
        s.stop()
        del s
        gc.collect()
        counts.append(_fds(engine_module))
    assert counts[-1] - counts[0] <= 4, counts   # was +34 per class (two new GPU post-processors)


def test_one_camera_shared_by_two_classes(rtsp, engine_module, face_engine):
    cams = [("camera1", f"{BASE}/101"), ("camera2", f"{BASE}/201")]
    a = _session(engine_module, face_engine, cams)
    a.start()
    b = None
    try:
        assert _wait_frames(a, n=20)
        before = _fds(engine_module)
        b = _session(engine_module, face_engine, cams)
        b.start()
        time.sleep(2)
        ia, ib = a.camera("camera1").get_info(), b.camera("camera1").get_info()
        # one capture: the second class sees the same decoded stream at once, no second connection
        assert ib["frames_decoded"] >= ia["frames_decoded"] - 2 and ib["frames_decoded"] > 20, (ia, ib)
        assert ib["cam_id"] == "camera1" and _fds(engine_module) - before <= 6
        assert _wait_frames(b, n=ib["frames_decoded"] + 10)
        assert b.camera("camera1").get_info()["frames_processed"] > 0   # each class runs its own recognition
        a.stop()
        n = b.camera("camera1").get_info()["frames_decoded"]
        assert _wait_frames(b, n=n + 20), "the other class's camera stopped with the first class"
    finally:
        a.stop()
        if b is not None:
            b.stop()


def _publish(name, path):
    """ffmpeg pushing tests/media/<name>.mp4 to rtsp://.../<path> in a loop (the camera)."""
    return subprocess.Popen(["ffmpeg", "-loglevel", "error", "-re", "-stream_loop", "-1", "-i",
                             str(APP_DIR / "tests" / "media" / f"{name}.mp4"), "-c", "copy", "-f", "rtsp",
                             "-rtsp_transport", "tcp", f"rtsp://127.0.0.1:{PORT}/{path}"])


def test_camera_changing_its_picture_size_gets_a_new_gpu_decoder(rtsp, engine_module, face_engine):
    """A camera that comes back with another picture size (Hikvision .101 did, before four
    crashes on 2026-10-12): the GPU decoder is not renegotiated in place — the connection ends and
    the next one gets a fresh decoder — and the picture continues at the new size."""
    if not engine_module.gpu_decoding_available():
        pytest.skip("no VA-API GPU decoding on this machine")
    pub = _publish("camera1_main", "sizechange")   # 1920x1080
    s = _session(engine_module, face_engine, [{"id": "camera1", "url": f"rtsp://127.0.0.1:{PORT}/sizechange",
                                               "profile": "main", "fixed_url": True, "max_w": 0, "max_h": 0}])
    s.start()
    try:
        assert _wait_frames(s)
        info = s.camera("camera1").get_info()
        assert info["decoder"] == "gpu" and info["stream_size"] == "1920x1080", info
        pub.terminate(); pub.wait()
        pub = _publish("camera1_sub", "sizechange")   # the same camera, now 704x576
        end = time.time() + 45
        while time.time() < end and s.camera("camera1").get_info()["stream_size"] != "704x576":
            time.sleep(0.5)
        info = s.camera("camera1").get_info()
        assert info["stream_size"] == "704x576" and info["decoder"] == "gpu", info
        n = info["frames_decoded"]
        time.sleep(3)
        assert s.camera("camera1").get_info()["frames_decoded"] > n + 20   # decoding goes on
    finally:
        s.stop()
        pub.terminate(); pub.wait()
