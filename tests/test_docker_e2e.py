"""The Docker image end to end (spec §9 criterion 6): a container (non-root, host network +
host PID, /dev/dxrt0) runs a full attendance session against the local RTSP cameras."""
import glob
import json
import os
import shutil
import subprocess
import time

import pytest
import requests

from conftest import APP_DIR
from test_app_smoke import CLASS, TEACHER, _prepare_db

IMAGE = os.environ.get("BUP_IMAGE", "bup-attendance-npu:2.0")
PORT = 5055
BASE = f"http://127.0.0.1:{PORT}"


def _docker(*args, check=True):
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check)


@pytest.fixture(scope="module")
def container(tmp_path_factory):
    if shutil.which("docker") is None or _docker("image", "inspect", IMAGE, check=False).returncode:
        pytest.skip(f"docker image {IMAGE} not built (docker compose build)")
    root = tmp_path_factory.mktemp("docker")
    for d in ("db", "data", "attendance_history", "cloud_backup", "uploads"):
        (root / d).mkdir()
    db = _prepare_db(APP_DIR / "bup_db.json", root / "db" / "bup_db.json")
    c = db["courses"][CLASS]
    for s in db["students"].values():
        if (s["dept_code"], s.get("batch"), s.get("section")) == (c["dept_code"], c["batch"], c["section"]):
            name = os.path.basename(s["images_folder"])
            shutil.copytree(APP_DIR / "data" / name, root / "data" / name)

    subprocess.run(["bash", str(APP_DIR / "tools" / "rtsp_server.sh"), "start"], check=True)
    _docker("rm", "-f", "bup-e2e", check=False)
    vols = [x for d in ("db", "data", "attendance_history", "cloud_backup", "uploads")
            for x in ("-v", f"{root / d}:/app/{d}")]
    _docker("run", "-d", "--name", "bup-e2e", "--network", "host", "--pid", "host", "--ipc", "host",
            "--device", "/dev/dxrt0", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
            "-e", f"BUP_PORT={PORT}", "-e", "BUP_SECRET_KEY=e2e-test-key",
            *vols, "-v", f"{APP_DIR / 'models'}:/app/models:ro", IMAGE)
    for _ in range(60):
        try:
            if requests.get(BASE + "/", timeout=2).status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(1)
    yield root
    print(_docker("logs", "bup-e2e", check=False).stdout[-4000:])
    _docker("rm", "-f", "bup-e2e", check=False)
    subprocess.run(["bash", str(APP_DIR / "tools" / "rtsp_server.sh"), "stop"], check=False)


def test_attendance_session_in_container(container):
    s = requests.Session()
    r = s.post(BASE + "/", data={"username": TEACHER[0], "password": TEACHER[1]}, allow_redirects=False)
    assert r.status_code == 302
    r = s.post(BASE + "/take_attendance/start", allow_redirects=False,
               data={"course_id": CLASS, "faculty_code": "FST", "floor": "4", "room": "401"})
    assert r.status_code == 302 and r.headers["Location"].endswith("/live_session"), r.text[:300]

    time.sleep(20)
    stats = s.get(BASE + "/live_stats").json()
    assert stats["total"] == 8
    rows = s.get(BASE + "/live_list").json()
    assert sum(x["detection_count"] > 0 for x in rows) >= 6, rows
    info = s.get(BASE + "/api/stream_info").json()
    assert all(v["backend"].startswith("GStreamer") and v["processing_size"] == "1024x576"
               and v["errors"] == 0 for v in info["cameras"].values()), info

    with s.get(BASE + "/video_feed", stream=True, timeout=10) as feed:
        head = next(feed.iter_content(chunk_size=64))
    assert head.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")

    assert s.post(BASE + "/end_session", allow_redirects=False).status_code == 302
    assert s.post(BASE + "/confirm_attendance", data={}, allow_redirects=False).status_code == 302
    saved = glob.glob(str(container / "attendance_history" / f"attendance_{CLASS}_*.json"))
    assert saved and json.load(open(saved[0]))["total"] == 8
    assert glob.glob(str(container / "attendance_history" / f"attendance_{CLASS}_*.csv"))
    # the NPU embedding cache was written by the (non-root) container
    assert glob.glob(str(container / "data" / f"embeddings_{CLASS}*.npz"))   # per section: embeddings_<course>_A.npz
