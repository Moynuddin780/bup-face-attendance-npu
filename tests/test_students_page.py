"""Students page: an overview of batches / sections first, then one class at a time — a dept
admin picks batch + section of their department, a super admin department + batch + section."""
import importlib.util
import re
import sys

import pytest

from conftest import APP_DIR
from test_app_smoke import _make_copy


@pytest.fixture(scope="module")
def pages(tmp_path_factory):
    root = _make_copy(tmp_path_factory.mktemp("students") / "app", APP_DIR / "app.py", with_engine=True, data_mode="copy")
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("bup_students_app", root / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.app.config["TESTING"] = True
    db = mod.load_db()   # a second department, so the department choice matters
    db["students"]["CSE001"] = {"student_id": "CSE001", "name": "Cse Student", "faculty_code": "FST", "dept_code": "CSE",
                                "batch": "2024", "section": "A", "images_folder": "", "image_count": 0}
    db["students"]["2054901004"]["extra_sections"] = [{"batch": "2020", "section": "B"}]
    mod.save_db(db)

    def login(user, pw):
        c = mod.app.test_client()
        assert c.post("/", data={"username": user, "password": pw}).status_code == 302
        return c
    return mod, login("superadmin", "bup@2024"), login("faizur_ict", "admin123")


def _rows(html):
    return re.findall(r"<tr>\s*<td><code>([^<]+)</code>", html)


def test_overview_first(pages):
    mod, sa, _ = pages
    html = sa.get("/students").get_data(as_text=True)
    assert not _rows(html) and 'id="stbl"' not in html
    assert "Batch 2020" in html and "Batch 2021" in html and "CSE – Computer Science" in html
    assert '/students?dept=ICT&amp;batch=2020&amp;section=A' in html


def test_super_admin_picks_department_batch_section(pages):
    mod, sa, _ = pages
    html = sa.get("/students?dept=ICT&batch=2020&section=A").get_data(as_text=True)
    ids = _rows(html)
    db = mod.load_db()
    want = sorted(k for k, s in db["students"].items() if (s["dept_code"], s["batch"], s["section"]) == ("ICT", "2020", "A"))
    assert sorted(ids) == want and len(ids) == 8
    assert "ICT · Batch 2020 · Section A" in html
    b = _rows(sa.get("/students?dept=ICT&batch=2020&section=B").get_data(as_text=True))
    assert "2054901004" in b                        # attends section B too (extra_sections)
    assert _rows(sa.get("/students?dept=CSE&batch=2024&section=A").get_data(as_text=True)) == ["CSE001"]


def test_dept_admin_sees_only_own_department(pages):
    mod, _, da = pages
    html = da.get("/students").get_data(as_text=True)
    assert "CSE – Computer Science" not in html and "Batch 2020" in html
    assert 'name="dept" id="f-dept" value="ICT"' in html       # department fixed, not a choice
    assert _rows(da.get("/students?dept=CSE&batch=2024&section=A").get_data(as_text=True)) == []
    assert _rows(da.get("/students?q=CSE001").get_data(as_text=True)) == []     # outside their department


def test_search_across_batches(pages):
    mod, sa, _ = pages
    assert _rows(sa.get("/students?q=rihan").get_data(as_text=True)) == ["2054901004"]
    assert "No student matches" in sa.get("/students?q=zzzz").get_data(as_text=True)


def test_add_prefilled_and_delete_returns_to_class(pages):
    mod, sa, _ = pages
    html = sa.get("/students/add?dept=CSE&batch=2024&section=A").get_data(as_text=True)
    assert 'value="2024"' in html and '<option value="FST" selected>' in html and '"CSE"' in html
    r = sa.get("/students/delete/CSE001")
    assert r.status_code == 302 and r.headers["Location"].endswith("/students?dept=CSE&batch=2024&section=A")
