"""Live Monitor: the super admin sees every running class's live picture on one page."""
import importlib.util
import json
import sys

import pytest

from conftest import APP_DIR
from test_app_smoke import TEACHER, _h, _make_copy


class _Cam:
    def __init__(self, info):
        self.info = info

    def get_info(self):
        return self.info


class _Engine:
    def mjpeg_frame(self):
        return b"\xff\xd8fake-jpeg\xff\xd9"


class _Tracker:
    def live_stats(self):
        return {"total": 43, "present": 21, "absent": 22, "rate": 48.8, "duration": 446.0}


@pytest.fixture(scope="module")
def app_copy(tmp_path_factory):
    root = _make_copy(tmp_path_factory.mktemp("monitor") / "app", APP_DIR / "app.py", with_engine=True, data_mode="symlink")
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("bup_monitor_app", root / "app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bup_monitor_app"] = mod
    spec.loader.exec_module(mod)
    mod.app.config["TESTING"] = True

    def login(user, pw):
        c = mod.app.test_client()
        assert c.post("/", data={"username": user, "password": pw}).status_code == 302
        return c
    return mod, login


def _running_class(mod, key="FST/5/501", test_video=False):
    faculty, floor, room = key.split("/")
    s = mod._new_session(key, TEACHER[0], "Test Teacher", "ICE4204", faculty, floor, room, "A")
    s.update(active=True, starting=False, engine=_Engine(), tracker=_Tracker(), test_video=test_video, cameras={
        "camera1": _Cam({"url": "rtsp://***@192.168.68.101:554/Streaming/Channels/101", "has_frame": True,
                         "feed_fps": 24.6, "infer_fps": 8.1, "stream_size": "1920x1080", "reconnects": 0, "decoder": "gpu"}),
        "camera2": _Cam({"url": "rtsp://***@192.168.68.103:554/ch01.264", "has_frame": False,
                         "feed_fps": 0, "infer_fps": 0, "stream_size": "", "reconnects": 3, "decoder": "cpu"})})
    with mod._sessions_lock:
        mod._sessions[key] = s
    return s


def test_super_admin_only(app_copy):
    mod, login = app_copy
    for user, pw in (("faizur_ict", "admin123"), TEACHER):
        c = login(user, pw)
        for path in ("/live_monitor", "/live_monitor/status", "/live_monitor/frame/FST/5/501"):
            r = c.get(path)
            assert r.status_code == 302 and "/dashboard" in r.headers["Location"], (user, path)


def test_every_running_class_with_its_picture(app_copy):
    mod, login = app_copy
    sa = login("superadmin", "bup@2024")
    empty = sa.get("/live_monitor").get_data(as_text=True)
    assert "No class is running right now" in empty and 'id="k-classes">0<' in empty
    _running_class(mod, "FST/5/501")
    _running_class(mod, "FST/11/1101", test_video=True)
    try:
        page = sa.get("/live_monitor").get_data(as_text=True)
        assert 'id="k-classes">2<' in page and 'id="k-cams">4<' in page and "42 / 86" in page
        assert 'class="nav-live">2<' in page                       # the sidebar shows the running classes
        st = sa.get("/live_monitor/status").get_json()
        assert [c["key"] for c in st["classes"]] == ["FST/5/501", "FST/11/1101"]   # building, floor, room order
        c = st["classes"][0]
        assert (c["room"], c["course_id"], c["teacher_name"], c["present"], c["total"]) == ("501", "ICE4204", "Test Teacher", 21, 43)
        assert [(k["source"], k["connected"], k["gpu"]) for k in c["cameras"]] == \
            [("192.168.68.101:554", True, True), ("192.168.68.103:554", False, False)]
        assert "***" not in json.dumps(st) and "rtsp://" not in json.dumps(st)   # no camera credentials
        r = sa.get("/live_monitor/frame/FST/5/501")
        assert r.status_code == 200 and r.mimetype == "image/jpeg" and r.data.startswith(b"\xff\xd8")
        assert r.headers["Cache-Control"] == "no-store"
        assert sa.get("/live_monitor/frame/FST/9/999").status_code == 404
    finally:
        with mod._sessions_lock:
            mod._sessions.clear()


def test_unknown_page_is_a_404_for_a_logged_in_admin(app_copy):
    mod, login = app_copy
    r = login("superadmin", "bup@2024").get("/favicon.ico")   # no endpoint: the sidebar must still render
    assert r.status_code == 404 and "Page Not Found" in r.get_data(as_text=True)
