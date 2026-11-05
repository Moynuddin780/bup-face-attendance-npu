"""Recorded-video test sources: in the VIDEO_TEST_FLOORS rooms (FST floor 11) a camera can play an
uploaded recording instead of its RTSP stream. Upload in pieces (resumable), the file checked
with GStreamer, the class session decoding it in real time and looped, records marked as tests;
every other room stays RTSP only."""
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

from conftest import APP_DIR
from test_app_smoke import TEACHER, _make_copy

MEDIA = APP_DIR / "tests" / "media"
ROOM = "/buildings/FST/floor/11/room/1101"


@pytest.fixture(scope="module")
def video_app(tmp_path_factory):
    if not (MEDIA / "streams.json").exists():
        pytest.skip("test media missing — run tools/make_test_streams.py")
    root = _make_copy(tmp_path_factory.mktemp("videos") / "app", APP_DIR / "app.py", with_engine=True, data_mode="copy")
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("bup_video_app", root / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.app.config["TESTING"] = True
    yield mod
    mod._stop_session()


def _login(mod, user, pw):
    c = mod.app.test_client()
    assert c.post("/", data={"username": user, "password": pw}).status_code == 302
    return c


def _upload(c, room, cam, path, chunk=None, name=None):
    data = path.read_bytes()
    r = c.post(f"{room}/video/{cam}/start", json={"name": name or path.name, "size": len(data)})
    if r.status_code != 200:
        return r
    st = r.get_json()
    step = chunk or st["chunk"]
    off = 0
    while off < len(data):
        r = c.post(f"{room}/video/{cam}/chunk?upload_id={st['upload_id']}&offset={off}", data=data[off:off + step],
                   content_type="application/octet-stream")
        if r.status_code != 200:
            return r
        off += step
    return r


def test_only_the_test_floor_takes_videos(video_app):
    admin = _login(video_app, "superadmin", "bup@2024")
    for room in ("/buildings/FST/floor/4/room/401", "/buildings/FBS/floor/11/room/1101", "/buildings/FST/floor/10/room/1001"):
        assert admin.post(f"{room}/video/camera1/start", json={"name": "a.mp4", "size": 10}).status_code == 404
        assert "Recorded video" not in admin.get(room).get_data(as_text=True)
    assert admin.post(f"{ROOM}/video/camera3/start", json={"name": "a.mp4", "size": 10}).status_code == 404
    page = admin.get(ROOM).get_data(as_text=True)
    assert "Recorded video" in page and 'name="camera1_source"' in page
    # teachers cannot upload
    t = _login(video_app, *TEACHER)
    assert t.post(f"{ROOM}/video/camera1/start", json={"name": "a.mp4", "size": 10}).status_code in (302, 403)


def test_upload_is_checked(video_app, tmp_path):
    admin = _login(video_app, "superadmin", "bup@2024")
    r = admin.post(f"{ROOM}/video/camera1/start", json={"name": "notes.txt", "size": 10})
    assert r.status_code == 400 and "not a supported video type" in r.get_json()["error"]
    r = admin.post(f"{ROOM}/video/camera1/start", json={"name": "big.mp4", "size": (video_app.TEST_VIDEO_MAX_MB + 1) * 2**20})
    assert r.status_code == 413
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"this is not a video" * 1000)
    r = _upload(admin, ROOM, "camera1", fake)
    assert r.status_code == 400 and "cannot be played" in r.get_json()["error"]
    assert not list(Path(video_app.TEST_VIDEO_DIR).glob("*.mp4"))
    # a piece sent at the wrong offset is answered with where to continue
    st = admin.post(f"{ROOM}/video/camera1/start", json={"name": "x.mp4", "size": 100}).get_json()
    r = admin.post(f"{ROOM}/video/camera1/chunk?upload_id={st['upload_id']}&offset=50", data=b"x" * 10,
                   content_type="application/octet-stream")
    assert r.status_code == 409 and r.get_json()["resume"] == 0
    # someone else's upload id is refused
    other = _login(video_app, "faizur_ict", "admin123")
    r = other.post(f"{ROOM}/video/camera1/chunk?upload_id={st['upload_id']}&offset=0", data=b"x" * 10,
                   content_type="application/octet-stream")
    assert r.status_code == 404


def test_recorded_video_class(video_app):
    mod = video_app
    admin = _login(mod, "superadmin", "bup@2024")
    streams = json.loads((MEDIA / "streams.json").read_text())

    # upload camera1's recording in small pieces; camera2 keeps its (unreachable) RTSP camera
    r = _upload(admin, ROOM, "camera1", MEDIA / "camera1_sub.mp4", chunk=300_000)
    assert r.status_code == 200, r.get_json()
    v = r.get_json()["video"]
    assert (v["width"], v["height"], round(v["fps"]), v["codec"]) == (704, 576, streams["fps"], "H.264")
    assert v["duration"] == pytest.approx(streams["seconds"], abs=1) and not v["missing"]
    assert admin.get(v["thumb"]).mimetype == "image/jpeg"
    entry = mod.load_db()["buildings"]["FST"]["floors"]["11"]["rooms"]["1101"]["test_videos"]["camera1"]
    assert entry["active"] and entry["name"] == "camera1_sub.mp4"

    # saving the room keeps camera1 on its video (its RTSP URL stays stored), camera2 on RTSP
    room = mod.load_db()["buildings"]["FST"]["floors"]["11"]["rooms"]["1101"]
    r = admin.post(ROOM, data={"camera1": room["camera1"], "camera1_source": "video",
                               "camera2": room["camera2"], "camera2_source": "rtsp"}, follow_redirects=True)
    assert "recorded video camera1_sub.mp4" in r.get_data(as_text=True)
    room = mod.load_db()["buildings"]["FST"]["floors"]["11"]["rooms"]["1101"]
    assert room["camera1"].startswith("rtsp://") and room["test_videos"]["camera1"]["active"]
    assert "Recorded video" in admin.get("/buildings").get_data(as_text=True)

    # the class: camera1 plays the recording, camera2 (10.10.11.3) is unreachable and is skipped
    t = _login(mod, *TEACHER)
    r = t.post("/take_attendance/start", data={"course_id": "ICE4204", "faculty_code": "FST", "floor": "11",
                                                "room": "1101"}, follow_redirects=True)
    body = r.get_data(as_text=True)
    assert r.request.path == "/live_session", body[-1500:]
    assert "Recorded video test" in body and "Camera 2" in body   # badge + the offline camera2 notice
    try:
        deadline = time.time() + 40
        while time.time() < deadline:
            cams = t.get("/api/stream_info").get_json()["cameras"]
            c1 = cams["camera1"]
            if c1["frames_processed"] > 40 and c1["identified_tracks"] >= 3:
                break
            time.sleep(1)
        assert c1["backend"].startswith("GStreamer video file") and c1["url"].startswith("video: "), c1
        assert c1["frame_size"] == "704x576" and c1["feed_fps"] >= 0.8 * streams["fps"], c1
        assert c1["identified_tracks"] >= 3, c1
        rows = {x["student_id"]: x for x in t.get("/live_list").get_json()}
        seen = [sid for sid in streams["cameras"]["camera1"] if sid not in streams["not_enrolled"]]
        assert all(rows[sid]["detection_count"] > 0 for sid in seen), {sid: rows[sid] for sid in seen}
    finally:
        r = t.post("/end_session", follow_redirects=True)
    assert mod._sessions == {} and "recorded video" in r.get_data(as_text=True)
    r = t.post("/confirm_attendance", data={}, follow_redirects=True)
    assert r.request.path == "/history" and "Test" in r.get_data(as_text=True)
    rec = sorted(Path(mod.HISTORY_DIR).glob("attendance_ICE4204_*.json"))[-1]
    assert json.loads(rec.read_text())["source"] == "Recorded video (test)"

    # back to the RTSP camera: the video stays uploaded but is not used
    admin.post(ROOM, data={"camera1": room["camera1"], "camera1_source": "rtsp", "camera2": room["camera2"]})
    room = mod.load_db()["buildings"]["FST"]["floors"]["11"]["rooms"]["1101"]
    assert not room["test_videos"]["camera1"]["active"]
    assert all(video is None for _, _, video in mod._room_sources("FST", "11", room))

    # remove it: file and thumbnail gone
    path = Path(mod.TEST_VIDEO_DIR) / room["test_videos"]["camera1"]["file"]
    assert path.exists() and path.with_name(path.name + ".jpg").exists()
    assert admin.post(f"{ROOM}/video/camera1/delete").get_json()["ok"]
    assert not path.exists() and not path.with_name(path.name + ".jpg").exists()
    assert "camera1" not in mod.load_db()["buildings"]["FST"]["floors"]["11"]["rooms"]["1101"]["test_videos"]
