"""Student enrollment: a face template is built and saved the moment photos are added (web upload
or bulk import), HEIC photos are accepted, and the class gallery reuses the saved templates."""
import glob
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from conftest import APP_DIR
from test_app_smoke import _make_copy

PY = sys.executable
ADMIN = ("faizur_ict", "admin123")   # ICT dept admin; _prepare_db sets this password
SRC_STUDENT = APP_DIR / "data" / "2054901004_Rihan"


@pytest.fixture(scope="module")
def enroll_app(tmp_path_factory):
    root = _make_copy(tmp_path_factory.mktemp("enroll") / "app", APP_DIR / "app.py", with_engine=True,
                      data_mode="copy")
    (root / "tools").mkdir()
    shutil.copy(APP_DIR / "tools" / "import_students.py", root / "tools")
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("bup_enroll_app", root / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.app.config["TESTING"] = True
    c = mod.app.test_client()
    assert c.post("/", data={"username": ADMIN[0], "password": ADMIN[1]}).status_code == 302
    return mod, root, c


def _photos(n):
    return sorted(glob.glob(str(SRC_STUDENT / "*.jp*g")))[:n]


def _heic_bytes(jpg):
    from PIL import Image
    import pillow_heif
    pillow_heif.register_heif_opener()
    buf = io.BytesIO()
    with Image.open(jpg) as im:
        im.save(buf, format="HEIF")
    return buf.getvalue()


def test_upload_saves_face_template(enroll_app):
    mod, root, c = enroll_app
    r = c.post("/students/add", data={"student_id": "T0001", "name": "Test Student", "faculty_code": "FST",
                                      "dept_code": "ICT", "batch": "2099", "section": "A"})
    assert r.status_code == 302
    files = [(open(p, "rb"), os.path.basename(p)) for p in _photos(3)]
    files.append((io.BytesIO(b"%PDF-1.4"), "cv.pdf"))
    r = c.post("/students/T0001/images", data={"images": files}, content_type="multipart/form-data",
               follow_redirects=True)
    body = r.get_data(as_text=True)
    assert "Face template saved from 3 of 3 photos" in body and "cv.pdf" in body
    s = mod.load_db()["students"]["T0001"]
    assert s["image_count"] == 3 and s["embedding"]["faces"] == 3 and s["embedding"]["photos"] == 3
    folder = mod._resolve_student_folder(s)
    d = np.load(os.path.join(folder, mod.STUDENT_EMB_FILE))
    assert d["embedding"].shape == (1, 512) and abs(np.linalg.norm(d["embedding"]) - 1) < 1e-4


def test_heic_upload_is_converted(enroll_app):
    mod, root, c = enroll_app
    heic = _heic_bytes(_photos(4)[3])
    r = c.post("/students/T0001/images", data={"images": [(io.BytesIO(heic), "IMG_0001.HEIC")]},
               content_type="multipart/form-data", follow_redirects=True)
    assert "Face template saved from 4 of 4 photos" in r.get_data(as_text=True)
    folder = mod._resolve_student_folder(mod.load_db()["students"]["T0001"])
    assert sorted(os.path.basename(p) for p in mod._student_photos(folder)) == ["img1.jpeg", "img2.jpeg",
                                                                                 "img3.jpeg", "img4.jpg"]


def test_heic_named_jpg_upload_is_converted(enroll_app):
    mod, root, c = enroll_app
    heic = _heic_bytes(_photos(5)[4])
    r = c.post("/students/T0001/images", data={"images": [(io.BytesIO(heic), "photo.jpg")]},
               content_type="multipart/form-data", follow_redirects=True)
    assert "Face template saved from 5 of 5 photos" in r.get_data(as_text=True)
    folder = mod._resolve_student_folder(mod.load_db()["students"]["T0001"])
    with open(os.path.join(folder, "img5.jpg"), "rb") as f:
        assert f.read(3) == b"\xff\xd8\xff"                                  # stored as real JPEG
    assert c.get("/students/T0001/images/delete/img5.jpg").status_code == 302


def test_delete_photo_updates_template(enroll_app):
    mod, root, c = enroll_app
    r = c.get("/students/T0001/images/delete/img2.jpeg", follow_redirects=True)
    assert "Face template updated from 3 of 3 photos" in r.get_data(as_text=True)
    assert mod.load_db()["students"]["T0001"]["embedding"]["photos"] == 3


def test_class_gallery_reuses_saved_templates(enroll_app):
    mod, root, c = enroll_app
    db = mod.load_db()
    db["courses"]["TST101"] = {"name": "Test", "faculty_code": "FST", "dept_code": "ICT", "batch": "2099",
                               "teacher_username": None}   # a course covers the whole batch
    mod.save_db(db)
    folder = mod._resolve_student_folder(db["students"]["T0001"])
    stamp = os.path.getmtime(os.path.join(folder, mod.STUDENT_EMB_FILE))
    embeds, meta = mod.build_embeddings_for_class("TST101", "A")
    assert [m["student_id"] for m in meta] == ["T0001"]
    assert os.path.getmtime(os.path.join(folder, mod.STUDENT_EMB_FILE)) == stamp   # not rebuilt
    assert mod.build_embeddings_for_class("TST101", "B") == (None, None)           # a section-B class: not on it
    db = mod.load_db()
    db["students"]["T0001"]["extra_sections"] = [{"batch": "2099", "section": "B"}]
    mod.save_db(db)
    embeds_b, meta_b = mod.build_embeddings_for_class("TST101", "B")
    assert [m["student_id"] for m in meta_b] == ["T0001"] and np.allclose(embeds_b, embeds)


def test_student_photo_route(enroll_app):
    mod, root, c = enroll_app
    assert c.get("/student_img/../bup_db.json").status_code == 404
    folder = os.path.basename(mod._resolve_student_folder(mod.load_db()["students"]["T0001"]))
    full = c.get(f"/student_img/{folder}/img1.jpeg")
    thumb = c.get(f"/student_img/{folder}/img1.jpeg?w=240")
    assert full.status_code == thumb.status_code == 200
    assert thumb.mimetype == "image/jpeg" and len(thumb.data) < len(full.data)
    page = c.get("/students?dept=ICT&batch=2099&section=A").get_data(as_text=True)
    assert "Ready · 3/3" in page


def test_bulk_import(enroll_app, tmp_path):
    mod, root, c = enroll_app
    src = tmp_path / "Section_A"
    a = src / "99000001_ alice  example "
    b = src / "Bob Example-99000002"
    empty = src / "99000003_Carol sectionA"
    for d in (a, b, empty):
        d.mkdir(parents=True)
    for i, p in enumerate(_photos(3)):
        shutil.copy(p, a / f"photo {i}.JPG")
    shutil.copy(_photos(1)[0], a / "same again.jpg")                       # duplicate content
    (a / "IMG_9.HEIC").write_bytes(_heic_bytes(_photos(4)[3]))
    (a / "renamed.jpg").write_bytes(_heic_bytes(_photos(5)[4]))            # HEIC under a .jpg name
    (a / "notes.tex").write_text("x")
    for p in _photos(2):
        shutil.copy(p, b / os.path.basename(p))
    other = tmp_path / "Section_B" / "99000001_Alice Example"
    other.mkdir(parents=True)
    shutil.copy(_photos(1)[0], other / "x.jpg")

    env = {**os.environ, "BUP_DB_FILE": str(root / "bup_db.json")}
    tool = str(root / "tools" / "import_students.py")
    for path, section in ((src, "A"), (other.parent, "B"), (src, "A")):   # 3rd run: nothing new
        r = subprocess.run([PY, tool, str(path), "--batch", "2098", "--section", section],
                           capture_output=True, text=True, env=env, timeout=600)
        assert r.returncode == 0, r.stderr[-2000:]
    db = mod.load_db()
    alice, bob, carol = (db["students"][k] for k in ("99000001", "99000002", "99000003"))
    assert (alice["name"], bob["name"], carol["name"]) == ("Alice Example", "Bob Example", "Carol")
    assert alice["batch"] == "2098" and alice["section"] == "A"
    assert alice["extra_sections"] == [{"batch": "2098", "section": "B"}]
    assert alice["image_count"] == 5 and alice["embedding"]["faces"] == 5      # 3 JPG + 2 HEIC, dup dropped
    assert bob["embedding"]["faces"] == 2 and carol["image_count"] == 0
    folder = mod._resolve_student_folder(alice)
    linked = [p for p in mod._student_photos(folder) if os.stat(p).st_nlink > 1]
    assert len(linked) == 3                                                    # hard links, no copies
    assert os.path.exists(os.path.join(folder, mod.STUDENT_EMB_FILE))
    assert glob.glob(str(root / "bup_db.json.bak-*"))
    assert "+ section 2098/B" in r.stdout or "updated" in r.stdout
