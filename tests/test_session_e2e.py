"""End-to-end AttendanceSession over RTSP (spec §9 criterion 4): two cameras, sub-streams,
CPU decode, NPU cascade, tracker, MJPEG. Uses the local RTSP cameras from tools/."""
import glob
import json
import os
import subprocess
import time

import numpy as np
import pytest

from conftest import APP_DIR

MEDIA = APP_DIR / "tests" / "media"
ARTIFACTS = APP_DIR / "tests" / "artifacts"
PORT = int(os.environ.get("RTSP_TEST_PORT", "18554"))
BASE = f"rtsp://127.0.0.1:{PORT}/Streaming/Channels"
CLASS = "ICE4204"


@pytest.fixture(scope="module")
def streams():
    if not (MEDIA / "streams.json").exists():
        subprocess.run([str(APP_DIR / ".venv" / "bin" / "python"), str(APP_DIR / "tools" / "make_test_streams.py")],
                       check=True)
    subprocess.run(["bash", str(APP_DIR / "tools" / "rtsp_server.sh"), "start"], check=True)
    yield json.loads((MEDIA / "streams.json").read_text())
    subprocess.run(["bash", str(APP_DIR / "tools" / "rtsp_server.sh"), "stop"], check=False)


@pytest.fixture(scope="module")
def klass(face_engine, engine_module):
    """Class gallery exactly like build_embeddings_for_class: mean of the photos' normed
    embeddings, L2-normalized, one row per student in DB order."""
    db = json.loads((APP_DIR / "bup_db.json").read_text())
    c = db["courses"][CLASS]
    students = [s for s in db["students"].values()
                if s["dept_code"] == c["dept_code"] and s.get("batch") == c.get("batch")
                and s.get("section") == c.get("section")]
    rows, meta = [], []
    for s in students:
        folder = APP_DIR / "data" / os.path.basename(s["images_folder"])
        embs = []
        for p in sorted(glob.glob(str(folder / "*"))):
            img = engine_module.imread(p)
            faces = face_engine.get(img) if img is not None else []
            if faces:
                f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                embs.append(f.normed_embedding)
        if embs:
            m = np.mean(np.vstack(embs), axis=0)
            rows.append(m / np.linalg.norm(m))
            meta.append({"student_id": s["student_id"], "name": s["name"]})
    return np.vstack(rows).astype(np.float32), meta


def _run(engine_module, face_engine, klass, profile, seconds, min_time):
    embeds, meta = klass
    tracker = engine_module.AttendanceTracker(meta, min_time=min_time)
    sess = engine_module.AttendanceSession(
        face_engine, tracker, embeds, meta,
        [("camera1", f"{BASE}/101"), ("camera2", f"{BASE}/201")],   # main-stream URLs, as in the DB
        stream_profile=profile, rtsp_transport="tcp")
    sess.start()
    jpeg = None
    t_end = time.time() + seconds
    while time.time() < t_end:
        time.sleep(0.5)
        jpeg = sess.mjpeg_frame()
    info = {cid: sess.camera(cid).get_info() for cid in sess.camera_ids()}
    fps = sess.fps()
    t0 = time.time()
    sess.stop()
    stop_s = time.time() - t0
    return tracker, info, jpeg, fps, stop_s


def _decode(jpeg):
    import cv2
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def test_two_camera_substream_session(streams, engine_module, face_engine, klass):
    tracker, info, jpeg, fps, stop_s = _run(engine_module, face_engine, klass, "sub", 25, min_time=5)
    print(json.dumps(info, indent=1), "fps", fps)
    for cid, ch in (("camera1", "102"), ("camera2", "202")):
        i = info[cid]
        assert i["backend"].startswith("GStreamer"), i
        assert i["url"].endswith(f"/{ch}")                 # DB main URL rewritten to the sub-stream
        assert i["frame_size"] == "704x576"                # decoded sub-stream
        assert i["processing_size"] == "1024x576"          # anamorphic D1 corrected to 16:9
        assert i["infer_fps"] >= 5, i
        assert i["errors"] == 0, i
        # ByteTrack: the faces are tracked and recognised per track, so ArcFace runs far less
        # often than once per face per frame (re-checks ~1/s, strangers ~2.5/s)
        assert i["tracks"] >= 3 and i["identified_tracks"] >= 3, i
        assert 0 < i["embeddings_computed"] < 0.35 * i["frames_processed"] * i["faces_last_frame"], i
    assert fps >= 5
    assert stop_s < 5.0

    rows = {r["student_id"]: r for r in tracker.summary()}
    in_video = set(streams["cameras"]["camera1"]) | set(streams["cameras"]["camera2"])
    enrolled_in_video = in_video - set(streams["not_enrolled"])
    for sid in enrolled_in_video:
        assert rows[sid]["detection_count"] > 0, rows[sid]
        assert rows[sid]["present"], rows[sid]              # >= min_time seen
    for sid, r in rows.items():
        if sid not in in_video:                             # nobody else may be credited
            assert r["detection_count"] == 0 and r["total_time"] == 0.0, r
    assert tracker.live_stats()["present"] == len(enrolled_in_video)

    img = _decode(jpeg)
    assert img.shape[1] == 800 and img.shape[0] > 2 * 400   # two cameras stacked top to bottom (+ stats strip)
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "live_feed_substream.jpg").write_bytes(jpeg)


def test_main_stream_profile_still_works(streams, engine_module, face_engine, klass):
    tracker, info, jpeg, fps, _ = _run(engine_module, face_engine, klass, "main", 12, min_time=5)
    for cid, ch in (("camera1", "101"), ("camera2", "201")):
        assert info[cid]["url"].endswith(f"/{ch}")
        assert info[cid]["frame_size"] == "1920x1080"
        assert info[cid]["errors"] == 0
    assert sum(r["detection_count"] > 0 for r in tracker.summary()) >= 5
    (ARTIFACTS / "live_feed_mainstream.jpg").write_bytes(jpeg)
