"""Teacher/admin page changes of 2026-10-12: minutes instead of seconds, the Present | Absent
pills, one class card per section (and a joint card), batch-wise courses and history, history
delete for department admins only, adjunct faculty, and profile photos."""
import importlib.util
import io
import json
import re
import sys

import pytest

from conftest import APP_DIR
from test_app_smoke import TEACHER, _h, _make_copy


@pytest.fixture(scope="module")
def app_copy(tmp_path_factory):
    root = _make_copy(tmp_path_factory.mktemp("ui") / "app", APP_DIR / "app.py", with_engine=True, data_mode="symlink")
    db = json.loads((root / "bup_db.json").read_text(encoding="utf-8"))
    db["teachers"]["nazneen_ict"]["password"] = _h("naz123")
    (root / "bup_db.json").write_text(json.dumps(db, indent=2, ensure_ascii=False), encoding="utf-8")
    hist = root / "attendance_history"
    for p in hist.glob("*"):
        if p.is_file():
            p.unlink()

    def record(fname, course, batch, section, teacher, date):
        rows = [{"student_id": "1", "name": "A", "total_time": 75.0, "detection_count": 9, "present": True, "status": "Present"},
                {"student_id": "2", "name": "B", "total_time": 9.1, "detection_count": 1, "present": False, "status": "Absent"}]
        (hist / f"{fname}.json").write_text(json.dumps({
            "filename": fname, "course_id": course, "course_name": course + " name", "dept_code": "ICT", "faculty_code": "FST",
            "batch": batch, "section": section, "teacher": teacher, "teacher_name": teacher, "date": date, "time": "10:00:00",
            "total": 2, "present": 1, "absent": 1, "rate": 50.0, "records": rows}))
    record("attendance_ICE4204_A", "ICE4204", "2020", "A", TEACHER[0], "2026-10-01")
    record("attendance_ICE4204_B", "ICE4204", "2020", "B", TEACHER[0], "2026-10-02")
    record("attendance_ICE4201_A", "ICE4201", "2022", "A", "nazneen_ict", "2026-10-03")

    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("bup_ui_app", root / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.app.config["TESTING"] = True

    def login(user, pw):
        c = mod.app.test_client()
        assert c.post("/", data={"username": user, "password": pw}).status_code == 302
        return c
    return mod, root, login


def test_minutes_text(app_copy):
    mod, _, _ = app_copy
    assert [mod.minutes_text(s) for s in (0, 5, 6, 59.9, 60, 90, 240, 446, None)] == \
        ["0 min", "0 min", "0.1 min", "0.9 min", "1 min", "1.5 min", "4 min", "7.4 min", "0 min"]


def test_one_card_per_section_and_a_joint_card(app_copy):
    mod, _, login = app_copy
    t = login(*TEACHER)
    page = t.get("/take_attendance").get_data(as_text=True)
    forms = re.findall(r'name="course_id" value="ICE4204"/>\s*<input type="hidden" name="section" value="(\w*)"', page)
    assert forms == ["A", "B", "ALL"], forms
    assert "Sections A + B" in page and "joint class" in page
    # the chosen section is not asked again on the room page
    room = t.post("/take_attendance/select_location", data={"course_id": "ICE4204", "section": "B"}).get_data(as_text=True)
    assert re.search(r'<input type="radio" name="section" value="B" checked hidden/>', room)
    assert 'value="A"' not in room and 'value="ALL"' not in room
    joint = t.post("/take_attendance/select_location", data={"course_id": "ICE4204", "section": "ALL"}).get_data(as_text=True)
    assert 'name="section" value="ALL" checked hidden' in joint and "joint class" in joint
    bad = t.post("/take_attendance/select_location", data={"course_id": "ICE4204", "section": "Z"})
    assert bad.status_code == 302


def test_review_page_pills_and_minutes(app_copy):
    mod, _, login = app_copy
    t = login(*TEACHER)
    with t.session_transaction() as s:
        s["pending_summary"] = [
            {"student_id": "1", "name": "Alpha", "total_time": 75.0, "detection_count": 9, "present": True, "status": "Present"},
            {"student_id": "2", "name": "Beta", "total_time": 9.1, "detection_count": 167, "present": False, "status": "Absent"}]
        s["pending_course"] = "ICE4204"
        s["pending_section"] = "A"
    page = t.get("/confirm_attendance").get_data(as_text=True)
    assert page.count('class="att-seg"') == 2 and "<select" not in page
    assert 'name="status_1" value="Present"' in page and 'name="status_2" value="Absent"' in page
    assert "1.2 min" in page and "0.1 min" in page and "On camera for 1 min or more" in page
    assert "9.1s" not in page


def test_courses_grouped_by_batch(app_copy):
    mod, _, login = app_copy
    page = login("faizur_ict", "admin123").get("/courses").get_data(as_text=True)
    order = re.findall(r'id="batch-(\w+)"', page)
    assert order == sorted(order, reverse=True) and {"2020", "2022"} <= set(order)
    b2020 = page[page.index('id="batch-2020"'):]
    b2020 = b2020[:b2020.index('</table>')]
    assert "ICE4204" in b2020 and "ICE4201" not in b2020


def test_history_access_and_delete(app_copy):
    mod, root, login = app_copy
    teacher = login(*TEACHER)
    page = teacher.get("/history").get_data(as_text=True)
    assert "attendance_ICE4204_A" in page and "attendance_ICE4201_A" not in page   # own classes only
    assert "/delete" not in page                                                   # no delete for teachers
    assert teacher.get("/history/attendance_ICE4201_A").status_code == 403         # another teacher's record
    assert teacher.get("/history/attendance_ICE4201_A/download").status_code == 403
    assert teacher.post("/history/attendance_ICE4204_A/delete").status_code == 302  # role_required -> dashboard
    assert (root / "attendance_history" / "attendance_ICE4204_A.json").exists()

    admin = login("faizur_ict", "admin123")
    page = admin.get("/history").get_data(as_text=True)
    assert 'data-batch="2020"' in page and 'data-section="B"' in page and "/history/attendance_ICE4204_B/delete" in page
    assert page.index('data-course="ICE4201"') < page.index('data-course="ICE4204"')   # course-wise cards
    r = admin.post("/history/attendance_ICE4204_B/delete", data={"next": "//evil.example/x"})
    assert r.status_code == 302 and r.headers["Location"].endswith("/history")
    assert not (root / "attendance_history" / "attendance_ICE4204_B.json").exists()
    kept = list((root / "attendance_history" / ".deleted").glob("attendance_ICE4204_B__deleted_*.json"))
    assert len(kept) == 1 and json.loads(kept[0].read_text())["deleted_by"] == "faizur_ict"
    assert admin.get("/history/attendance_ICE4204_B").status_code == 404
    supera = login("superadmin", "bup@2024")
    assert supera.post("/history/attendance_ICE4204_A/delete").status_code == 302
    assert (root / "attendance_history" / "attendance_ICE4204_A.json").exists()   # department admins only


def test_adjunct_faculty(app_copy):
    mod, _, login = app_copy
    admin = login("faizur_ict", "admin123")
    form = {"name": "Dr. Guest", "username": "guest_du", "password": "pw123456", "email": "",
            "affiliation": "external", "university": ""}
    page = admin.post("/teachers/add", data=form).get_data(as_text=True)
    assert "Enter the university" in page and 'value="guest_du"' in page        # kept what was typed
    form["university"] = "  University   of Dhaka "
    assert admin.post("/teachers/add", data=form).status_code == 302
    t = mod.load_db()["teachers"]["guest_du"]
    assert (t["affiliation"], t["university"]) == ("external", "University of Dhaka")
    page = admin.get("/teachers").get_data(as_text=True)
    adj = page[page.index("Adjunct Faculty"):]
    assert "Dr. Guest" in adj and "University of Dhaka" in adj
    assert "Dr. Guest" not in page[page.index("BUP Faculty"):page.index("Adjunct Faculty")]
    # back to BUP from the edit page
    assert admin.post("/teachers/edit/guest_du", data={"name": "Dr. Guest", "email": "", "affiliation": "bup"}).status_code == 302
    assert mod.load_db()["teachers"]["guest_du"]["affiliation"] == "bup"


def test_profile_photo(app_copy):
    mod, root, login = app_copy
    from PIL import Image
    t = login(*TEACHER)
    buf = io.BytesIO()
    Image.new("RGB", (640, 400), (200, 30, 30)).save(buf, "PNG")
    buf.seek(0)
    r = t.post("/profile", data={"action": "photo", "photo": (buf, "me.png")}, content_type="multipart/form-data")
    assert r.status_code == 302
    photo = mod.load_db()["teachers"][TEACHER[0]]["photo"]
    img = Image.open(root / "uploads" / "avatars" / photo)
    assert img.size == (320, 320) and img.format == "JPEG"
    page = t.get("/dashboard").get_data(as_text=True)
    assert f"/avatar/{photo}" in page                                    # sidebar + top bar
    assert t.get(f"/avatar/{photo}").status_code == 200
    bad = t.post("/profile", data={"action": "photo", "photo": (io.BytesIO(b"not an image"), "x.jpg")},
                 content_type="multipart/form-data", follow_redirects=True).get_data(as_text=True)
    assert "Could not use that photo" in bad
    assert t.post("/profile", data={"action": "photo_remove"}).status_code == 302
    assert "photo" not in mod.load_db()["teachers"][TEACHER[0]]
    assert not (root / "uploads" / "avatars" / photo).exists()
    # the password form still works without the new field
    assert t.post("/profile", data={"old_password": "wrong", "new_password": "x12345"}, follow_redirects=True).status_code == 200
