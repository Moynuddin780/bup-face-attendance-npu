"""Courses cover a whole batch: the teacher gets the course for every section, the section is
chosen when a class starts, and admins can edit a course (name, batch, teacher)."""
import glob
import importlib.util
import json
import re
import sys

import pytest

from conftest import APP_DIR
from test_app_smoke import TEACHER, _h, _make_copy


@pytest.fixture(scope="module")
def app_copy(tmp_path_factory):
    root = _make_copy(tmp_path_factory.mktemp("courses") / "app", APP_DIR / "app.py", with_engine=True, data_mode="copy")
    db = json.loads((root / "bup_db.json").read_text(encoding="utf-8"))
    db["teachers"]["mainul_ict"]["password"] = _h("teach789")
    import os, shutil   # section B of batch 2020 too (the copy holds the ICE4204 = section A photos)
    for s in db["students"].values():
        name = os.path.basename(s["images_folder"])
        if (s["dept_code"], s["batch"]) == ("ICT", "2020") and not (root / "data" / name).exists():
            shutil.copytree(APP_DIR / "data" / name, root / "data" / name, ignore=shutil.ignore_patterns("embedding.npz", ".thumbs"))
    (root / "bup_db.json").write_text(json.dumps(db, indent=2, ensure_ascii=False), encoding="utf-8")
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("bup_courses_app", root / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.app.config["TESTING"] = True

    def login(user, pw):
        c = mod.app.test_client()
        assert c.post("/", data={"username": user, "password": pw}).status_code == 302
        return c
    return mod, root, login


def test_add_course_has_no_section(app_copy):
    mod, root, login = app_copy
    admin = login("faizur_ict", "admin123")
    assert 'name="section"' not in admin.get("/courses/add").get_data(as_text=True)
    r = admin.post("/courses/add", data={"course_id": "ice9001", "name": "Test Course", "batch": "2020",
                                         "teacher_username": "mainul_ict"})
    assert r.status_code == 302
    db = mod.load_db()
    assert db["courses"]["ICE9001"] == {"name": "Test Course", "faculty_code": "FST", "dept_code": "ICT",
                                        "batch": "2020", "teacher_username": "mainul_ict"}
    assert "ICE9001" in db["teachers"]["mainul_ict"]["courses"]
    page = admin.get("/courses").get_data(as_text=True)
    row = re.search(r"<b>ICE9001</b>.*?</tr>", page, re.S).group(0)
    assert "Section A" in row and "Section B" in row and "/courses/edit/ICE9001" in row


def test_section_chosen_when_the_class_starts(app_copy):
    mod, root, login = app_copy
    t = login("mainul_ict", "teach789")
    page = t.post("/take_attendance/select_location", data={"course_id": "ICE9001"}).get_data(as_text=True)
    assert re.search(r'name="section" value="A"\s*/>', page)          # nothing preselected: A or B?
    assert 'value="B"' in page and 'value="ALL"' in page
    legacy = login(*TEACHER).post("/take_attendance/select_location", data={"course_id": "ICE4204"}).get_data(as_text=True)
    assert re.search(r'name="section" value="A" checked', legacy)     # course made for section A before
    r = t.post("/take_attendance/start", data={"course_id": "ICE9001", "faculty_code": "FST", "floor": "4",
                                               "room": "401", "section": "C"}, follow_redirects=True)
    assert "has no section C" in r.get_data(as_text=True) and mod._sessions == {}


def test_roster_follows_the_section(app_copy):
    mod, root, login = app_copy
    db = mod.load_db()
    count = lambda sec: sum(1 for s in db["students"].values() if (s["dept_code"], s["batch"], s["section"]) == ("ICT", "2020", sec))
    _, a = mod.build_embeddings_for_class("ICE9001", "A")
    _, b = mod.build_embeddings_for_class("ICE9001", "B")
    _, every = mod.build_embeddings_for_class("ICE9001")
    assert len(a) == count("A") and len(b) == count("B") and len(every) == len(a) + len(b)
    assert {"embeddings_ICE9001_A.npz", "embeddings_ICE9001_B.npz", "embeddings_ICE9001.npz"} <= \
        {p.split("/")[-1] for p in glob.glob(str(root / "data" / "embeddings_ICE9001*.npz"))}


def test_saved_record_has_the_class_section(app_copy):
    mod, root, login = app_copy
    t = login("mainul_ict", "teach789")
    with t.session_transaction() as s:
        s["pending_summary"] = [{"student_id": "x", "name": "X", "total_time": 70, "present": True, "status": "Present",
                                 "detection_count": 3, "manual_override": None}]
        s["pending_course"], s["pending_section"] = "ICE9001", "B"
    assert "Section B" in t.get("/confirm_attendance").get_data(as_text=True)
    assert t.post("/confirm_attendance", data={}).status_code == 302
    rec = json.load(open(sorted(glob.glob(str(root / "attendance_history" / "attendance_ICE9001_*.json")))[-1]))
    assert rec["section"] == "B" and rec["batch"] == "2020"


def test_edit_course_reassigns_teacher(app_copy):
    mod, root, login = app_copy
    admin = login("faizur_ict", "admin123")
    page = admin.get("/courses/edit/ICE9001").get_data(as_text=True)
    assert 'value="ICE9001" disabled' in page and 'value="mainul_ict" selected' in page
    r = admin.post("/courses/edit/ICE9001", data={"name": "Renamed", "batch": "2020", "teacher_username": "nazneen_ict"},
                   follow_redirects=True)
    assert "assigned to" in r.get_data(as_text=True)
    db = mod.load_db()
    assert db["courses"]["ICE9001"]["name"] == "Renamed" and db["courses"]["ICE9001"]["teacher_username"] == "nazneen_ict"
    assert "ICE9001" not in db["teachers"]["mainul_ict"]["courses"] and "ICE9001" in db["teachers"]["nazneen_ict"]["courses"]
    # a course made per section before: editing it drops the section, the course covers the batch
    admin.post("/courses/edit/ICE4204", data={"name": "Cloud Computing", "batch": "2020", "teacher_username": "nazneen_ict"})
    assert "section" not in mod.load_db()["courses"]["ICE4204"]


def test_edit_and_delete_limited_to_own_department(app_copy):
    mod, root, login = app_copy
    db = mod.load_db()
    db["courses"]["CSE100"] = {"name": "Other", "faculty_code": "FST", "dept_code": "CSE", "batch": "2024", "teacher_username": None}
    mod.save_db(db)
    admin = login("faizur_ict", "admin123")
    assert admin.get("/courses/edit/CSE100").status_code == 404
    assert admin.get("/courses/delete/CSE100").status_code == 404 and "CSE100" in mod.load_db()["courses"]
    assert admin.get("/courses/delete/ICE9001").status_code == 302
    db = mod.load_db()
    assert "ICE9001" not in db["courses"] and "ICE9001" not in db["teachers"]["nazneen_ict"]["courses"]
