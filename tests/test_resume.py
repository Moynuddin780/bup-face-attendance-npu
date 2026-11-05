"""Classes survive a server restart: the app process is killed in the middle of a class (as when the
worker crashed on 2026-10-12) and started again — the class continues by itself with the time
measured so far. A class that cannot continue goes to the teacher's review page instead."""
import json
import os
import signal
import socket
import subprocess
import sys
import time

import pytest
import requests

from conftest import APP_DIR
from test_app_smoke import TEACHER, _make_copy

PY = sys.executable


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class App:
    def __init__(self, root):
        self.root, self.port, self.proc = root, _free_port(), None
        self.url = f"http://127.0.0.1:{self.port}"

    def start(self):
        env = {**os.environ, "BUP_PORT": str(self.port), "BUP_HOST": "127.0.0.1"}
        self.log = open(self.root / f"app_{int(time.time())}.log", "w")
        self.proc = subprocess.Popen([PY, "app.py"], cwd=self.root, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        end = time.time() + 60
        while time.time() < end:
            try:
                requests.get(self.url + "/", timeout=1)
                return
            except requests.RequestException:
                time.sleep(0.5)
        raise RuntimeError("app did not start")

    def kill(self):   # a crash: no atexit, no clean shutdown
        self.proc.send_signal(signal.SIGKILL)
        self.proc.wait(10)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    subprocess.run(["bash", str(APP_DIR / "tools" / "rtsp_server.sh"), "start"], check=True, capture_output=True)
    root = _make_copy(tmp_path_factory.mktemp("resume") / "app", APP_DIR / "app.py", with_engine=True, data_mode="copy")
    a = App(root)
    yield a
    a.stop()
    subprocess.run(["bash", str(APP_DIR / "tools" / "rtsp_server.sh"), "stop"], check=False, capture_output=True)


def _login(app):
    c = requests.Session()
    r = c.post(app.url + "/", data={"username": TEACHER[0], "password": TEACHER[1]}, allow_redirects=False)
    assert r.status_code == 302
    return c


def _state(app):
    p = app.root / "live_sessions.json"
    return json.loads(p.read_text()) if p.exists() else None


def _seen(c, app):
    return {r["student_id"]: r for r in c.get(app.url + "/live_list").json() if r["total_time"] > 0}


def test_class_continues_after_a_crash(app):
    app.start()
    c = _login(app)
    r = c.post(app.url + "/take_attendance/start",
               data={"course_id": "ICE4204", "faculty_code": "FST", "floor": "4", "room": "401"})
    assert r.url.endswith("/live_session"), r.url
    end = time.time() + 90   # until a snapshot holds measured time for three students
    saved = {}
    while time.time() < end and len(saved) < 3:
        time.sleep(1)
        st = _state(app) or {"sessions": []}
        saved = {t["student_id"]: t["total_time"] for x in st["sessions"] for t in x["tracker"] if t["total_time"] > 0}
    assert len(saved) >= 3 and [x["key"] for x in st["sessions"]] == ["FST/4/401"], st

    app.kill()
    app.start()
    end = time.time() + 90
    page = ""
    while time.time() < end:
        r = c.get(app.url + "/live_session")
        page = r.text
        if r.url.endswith("/live_session") and "Resumed at" in page:
            break
        time.sleep(1)
    assert "Resumed at" in page, page[-1500:]
    after = _seen(c, app)
    for sid, t in saved.items():   # the time measured until the last snapshot is kept, and keeps growing
        assert after[sid]["total_time"] >= round(t, 1), (sid, t, after.get(sid))
    end = time.time() + 30
    while time.time() < end and sum(x["total_time"] for x in _seen(c, app).values()) <= sum(x["total_time"] for x in after.values()):
        time.sleep(1)
    assert sum(x["total_time"] for x in _seen(c, app).values()) > sum(x["total_time"] for x in after.values())

    # ended normally: nothing left to resume
    r = c.post(app.url + "/end_session")
    assert r.url.endswith("/confirm_attendance")
    assert _state(app)["sessions"] == []


def test_class_down_too_long_goes_to_review(app):
    """A snapshot older than RESUME_MAX_GAP_S is not resumed; its results wait for the teacher."""
    app.stop()
    snap = {"saved": time.time() - 4 * 3600, "pending": {}, "sessions": [{
        "key": "FST/4/401", "teacher": TEACHER[0], "teacher_name": "Test Teacher", "course_id": "ICE4204",
        "faculty": "FST", "floor": "4", "room": "401", "section": None, "started": time.time() - 5 * 3600,
        "test_video": False, "resumes": 0,
        "meta": [{"student_id": "2054901001", "name": "A"}, {"student_id": "2054901003", "name": "B"}],
        "tracker": [{"student_id": "2054901001", "total_time": 75.0, "detection_count": 300, "first_seen": None,
                     "manual_override": None},
                    {"student_id": "2054901003", "total_time": 5.0, "detection_count": 20, "first_seen": None,
                     "manual_override": None}]}]}
    (app.root / "live_sessions.json").write_text(json.dumps(snap))
    app.start()
    time.sleep(3)
    c = _login(app)
    r = c.get(app.url + "/live_session")
    assert r.url.endswith("/confirm_attendance") and "interrupted by a server restart" in r.text, r.text[-1500:]
    rows = {row["student_id"]: row for row in c.get(app.url + "/live_list").json()} or None
    assert rows is None   # no class runs
    assert "1.2 min" in r.text   # the time measured before the restart (75 s) is on the review page
    assert _state(app)["sessions"] == [] and _state(app)["pending"] == {}
    app.stop()
