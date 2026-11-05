"""Flask app integration: every page renders for every role, every URL of the original app is
still served, and a full attendance session runs through the C++ engine."""
import glob
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

from conftest import APP_DIR

PY = sys.executable
RTSP = "rtsp://127.0.0.1:{}/Streaming/Channels".format(os.environ.get("RTSP_TEST_PORT", "18554"))
TEACHER = ("t_test", "test123")
CLASS = "ICE4204"


def _h(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


def _prepare_db(src_db, dst_db):
    """Test copy of the real DB: a known teacher for the class, room FST-401 pointing at
    the local RTSP cameras (main-stream URLs, exactly like the stored Dahua ones)."""
    db = json.loads(open(src_db, encoding="utf-8").read())
    db["teachers"][TEACHER[0]] = {"password": _h(TEACHER[1]), "name": "Test Teacher", "email": "t@test",
                                  "faculty_code": "FST", "dept_code": "ICT", "courses": [CLASS]}
    db["dept_admins"]["faizur_ict"]["password"] = _h("admin123")
    db["buildings"]["FST"]["floors"]["4"]["rooms"]["401"] = {"camera1": f"{RTSP}/101", "camera2": f"{RTSP}/201"}
    json.dump(db, open(dst_db, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return db


def _make_copy(root, app_py, with_engine, data_mode):
    """App dir copy: app.py + templates/static + DB; data either symlinked (read-only
    page rendering) or a real copy of the class's photo folders (the session writes caches)."""
    root.mkdir(parents=True)
    shutil.copy(app_py, root / "app.py")
    if (app_py.parent / "camera_control.py").exists():   # the app's own modules (absent in the original)
        shutil.copy(app_py.parent / "camera_control.py", root / "camera_control.py")
    for d in ("templates", "static"):
        shutil.copytree(APP_DIR / d, root / d)
    for d in ("attendance_history", "cloud_backup", "uploads"):   # never the uploaded test videos (GBs)
        shutil.copytree(APP_DIR / d, root / d, ignore=shutil.ignore_patterns("test_videos"))
    db = _prepare_db(APP_DIR / "bup_db.json", root / "bup_db.json")
    if data_mode == "symlink":
        os.symlink(APP_DIR / "data", root / "data")
    else:
        (root / "data").mkdir()
        c = db["courses"][CLASS]
        for s in db["students"].values():
            if (s["dept_code"], s.get("batch"), s.get("section")) == (c["dept_code"], c["batch"], c["section"]):
                name = os.path.basename(s["images_folder"])
                shutil.copytree(APP_DIR / "data" / name, root / "data" / name)
    if with_engine:
        for so in glob.glob(str(APP_DIR / "bup_engine*.so")):
            shutil.copy(so, root)
        os.symlink(APP_DIR / "models", root / "models")
    return root


PAGES = {
    "super_admin": ["/dashboard", "/admins", "/admins/add", "/teachers", "/courses", "/students",
                    "/buildings", "/history", "/profile", "/teachers/edit/nazneen_ict",
                    "/buildings/FST/floor/4/room/402", f"/students/2054901004/summary"],
    "dept_admin": ["/dashboard", "/teachers", "/teachers/add", "/courses", "/courses/add", "/students",
                   "/students/add", "/students/2054901004/images", "/history", "/profile"],
    "teacher": ["/dashboard", "/take_attendance", "/history", "/profile", "/live_session",
                "/confirm_attendance", "/live_stats", "/live_list"],
}
LOGINS = {"super_admin": ["superadmin", "bup@2024"], "dept_admin": ["faizur_ict", "admin123"],
          "teacher": list(TEACHER)}


def _render(app_dir, tmp_path):
    req = tmp_path / f"req_{app_dir.name}.json"
    req.write_text(json.dumps({"logins": LOGINS, "pages": PAGES}))
    r = subprocess.run([PY, str(APP_DIR / "tests" / "_render_pages.py"), str(app_dir), str(req)],
                       capture_output=True, text=True, timeout=300,
                       env={**os.environ, "NPU_BOUND": os.environ.get("NPU_BOUND", "auto")})
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout)


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("render")
    return _render(_make_copy(tmp / "new", APP_DIR / "app.py", with_engine=True, data_mode="symlink"), tmp)


def test_original_routes_still_served(rendered):
    """Every URL of the original app (baseline/original_routes.json) is still served with its
    methods, so old links and bookmarks keep working. The pages themselves were rebuilt since."""
    have = {}
    for rule, _endpoint, methods in rendered["rules"]:
        have.setdefault(rule, set()).update(methods)
    original = json.loads((APP_DIR / "baseline" / "original_routes.json").read_text())["routes"]
    assert len(original) == 39
    missing = [(rule, methods) for rule, methods in original if not set(methods) <= have.get(rule, set())]
    assert not missing, missing


def test_every_page_renders(rendered):
    for role, pages in rendered["pages"].items():
        assert pages["__login__"]["status"] == 302, (role, pages["__login__"])
        for url, p in pages.items():
            if url == "__login__":
                continue
            assert p["status"] in (200, 302), (role, url, p["status"])   # 302: e.g. no live class to show
            assert "Internal Server Error" not in p["body"], (role, url)


# ── full attendance session through Flask ─────────────────────────────────────

@pytest.fixture(scope="module")
def live_app(tmp_path_factory):
    media = APP_DIR / "tests" / "media"
    if not (media / "streams.json").exists():
        subprocess.run([PY, str(APP_DIR / "tools" / "make_test_streams.py")], check=True)
    subprocess.run(["bash", str(APP_DIR / "tools" / "rtsp_server.sh"), "start"], check=True)
    root = _make_copy(tmp_path_factory.mktemp("live") / "app", APP_DIR / "app.py", with_engine=True,
                      data_mode="copy")
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("bup_live_app", root / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.app.config["TESTING"] = True
    yield mod, root
    mod._stop_session()
    subprocess.run(["bash", str(APP_DIR / "tools" / "rtsp_server.sh"), "stop"], check=False)


def test_attendance_session_end_to_end(live_app):
    mod, root = live_app
    c = mod.app.test_client()
    assert c.post("/", data={"username": TEACHER[0], "password": TEACHER[1]}).status_code == 302
    r = c.post("/take_attendance/start", data={"course_id": CLASS, "faculty_code": "FST", "floor": "4", "room": "401"})
    assert r.status_code == 302 and r.headers["Location"].endswith("/live_session"), r.get_data(as_text=True)[:500]
    assert c.get("/live_session").status_code == 200

    time.sleep(15)
    stats = c.get("/live_stats").get_json()
    assert set(stats) == {"total", "present", "absent", "rate", "duration"} and stats["total"] == 8
    rows = {x["student_id"]: x for x in c.get("/live_list").get_json()}
    streams = json.loads((APP_DIR / "tests" / "media" / "streams.json").read_text())
    seen = (set(streams["cameras"]["camera1"]) | set(streams["cameras"]["camera2"])) - set(streams["not_enrolled"])
    assert all(rows[s]["total_time"] > 0 for s in seen), rows
    assert all(rows[s]["detection_count"] == 0 for s in rows if s not in seen), rows

    info = c.get("/api/stream_info").get_json()
    assert info["preferred_backend"] == "GStreamer"
    assert all(v["backend"].startswith("GStreamer") and v["frame_size"] == "704x576"
               for v in info["cameras"].values()), info

    resp = c.get("/video_feed", buffered=False)
    first = next(iter(resp.response))
    resp.close()
    assert first.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\xff\xd8")

    assert c.post("/manual_mark", json={"student_id": "2054901002", "status": "present"}).get_json() == {"ok": True}
    r = c.post("/end_session")
    assert r.status_code == 302 and r.headers["Location"].endswith("/confirm_attendance")
    page = c.get("/confirm_attendance")
    assert page.status_code == 200 and "Rihan" in page.get_data(as_text=True)
    r = c.post("/confirm_attendance", data={})
    assert r.status_code == 302

    saved = sorted(glob.glob(str(root / "attendance_history" / f"attendance_{CLASS}_*.json")))
    rec = json.load(open(saved[-1]))
    assert list(rec) == ["filename", "course_id", "course_name", "dept_code", "faculty_code", "batch", "section",
                         "teacher", "teacher_name", "date", "time", "total", "present", "absent", "rate",
                         "gpu_mode", "records"]
    # rows pass through the Flask cookie session (sort_keys JSON) in the original app too,
    # so their key ORDER is alphabetical in both versions — compare the key set
    assert set(rec["records"][0]) == {"student_id", "name", "total_time", "present", "status",
                                      "detection_count", "manual_override"}
    by_id = {x["student_id"]: x for x in rec["records"]}
    assert by_id["2054901002"]["present"] is True and by_id["2054901002"]["manual_override"] == "present"
    assert os.path.exists(saved[-1][:-5] + ".csv")
    assert os.path.exists(root / "cloud_backup" / os.path.basename(saved[-1]))


def test_embedding_cache_tracks_roster(live_app):
    """F7: the cache is rebuilt when the class roster changes (the original kept serving
    deleted students until someone uploaded a photo)."""
    mod, root = live_app
    embeds, meta = mod.build_embeddings_for_class(CLASS)
    n = len(meta)
    cache = root / "data" / f"embeddings_{CLASS}.npz"
    assert "fingerprint" in np.load(cache, allow_pickle=True).files
    db = mod.load_db()
    victim = meta[-1]["student_id"]
    db["students"].pop(victim)
    mod.save_db(db)
    embeds2, meta2 = mod.build_embeddings_for_class(CLASS)
    assert len(meta2) == n - 1 and victim not in {m["student_id"] for m in meta2}
    assert np.allclose(np.linalg.norm(embeds2, axis=1), 1.0, atol=1e-4)


def test_legacy_cache_is_rebuilt(live_app):
    """An npz from the old Python pipeline (no fingerprint, different embeddings) is not trusted."""
    mod, root = live_app
    cache = root / "data" / f"embeddings_{CLASS}.npz"
    np.savez(cache, embeds=np.zeros((1, 512), np.float32), meta=np.array([{"student_id": "x", "name": "x"}], dtype=object))
    embeds, meta = mod.build_embeddings_for_class(CLASS)
    assert len(meta) > 1 and "x" not in {m["student_id"] for m in meta}


def test_save_db_is_atomic(live_app):
    mod, root = live_app
    db = mod.load_db()
    mod.save_db(db)
    assert json.load(open(root / "bup_db.json", encoding="utf-8")) == db
    assert not list(root.glob("bup_db.json.*tmp*"))
