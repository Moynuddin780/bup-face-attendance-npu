"""FaceEngine on the real NPU (spec §9 criterion 3): contract of the returned faces,
and recognition accuracy on the enrolled photos versus the ORIGINAL Python pipeline."""
import glob
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import APP_DIR, ORIG_APP

TOOLS = APP_DIR / "tools"
sys.path.insert(0, str(TOOLS))
from recog_metrics import evaluate  # noqa: E402

SAMPLE_PHOTOS = sorted(glob.glob(str(APP_DIR / "data" / "2054901002_Aslam_Sikder" / "*.jpeg")))[:3]


def test_face_contract(face_engine, engine_module):
    img = engine_module.imread(SAMPLE_PHOTOS[0])
    assert img is not None and img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8
    faces = face_engine.get(img)
    assert len(faces) >= 1
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    assert f.bbox.shape == (4,) and f.bbox.dtype == np.float32
    assert f.bbox[2] > f.bbox[0] and f.bbox[3] > f.bbox[1]
    assert f.kps.shape == (5, 2)
    assert 0.5 <= f.det_score <= 1.0
    emb = f.normed_embedding
    assert emb.shape == (512,) and emb.dtype == np.float32
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-3        # fix F4: truly normalized
    # every landmark lies inside (a slightly grown) face box
    x1, y1, x2, y2 = f.bbox
    m = 0.1 * (x2 - x1)
    assert all(x1 - m <= x <= x2 + m and y1 - m <= y <= y2 + m for x, y in f.kps)


def test_no_face_on_blank(face_engine):
    assert face_engine.get(np.zeros((720, 1280, 3), np.uint8)) == []


def test_same_person_closer_than_other(face_engine, engine_module):
    other = sorted(glob.glob(str(APP_DIR / "data" / "2054901004_Rihan" / "*.jp*g")))[0]
    a, b = (face_engine.get(engine_module.imread(p))[0].normed_embedding for p in SAMPLE_PHOTOS[:2])
    c = face_engine.get(engine_module.imread(other))[0].normed_embedding
    assert float(a @ b) > float(a @ c) + 0.1


def test_engine_info(face_engine):
    info = face_engine.info()
    assert info["det_input_size"] == (640, 640) and info["rec_input_size"] == (112, 112)
    assert info["embedding_dim"] == 512
    assert info["npu_bound_det"] == "all" or set(info["npu_bound_det"].split(",")) <= {"0", "1", "2"}


@pytest.fixture(scope="module")
def embeddings(tmp_path_factory):
    """Embed all enrolled photos with both pipelines (separate processes)."""
    if not ORIG_APP.exists():
        pytest.skip("original app.py not available for the baseline")
    try:
        import dx_engine  # noqa: F401
        import cv2  # noqa: F401
    except ImportError:
        pytest.skip("baseline needs dx_engine + opencv (setup.sh WITH_DEV=1)")
    out = tmp_path_factory.mktemp("recog")
    res = {}
    for pipe in ("original", "cpp"):
        npz = out / f"{pipe}.npz"
        r = subprocess.run([sys.executable, str(TOOLS / "eval_recognition.py"), "--pipeline", pipe,
                            "--out", str(npz), "--orig-app", str(ORIG_APP)],
                           capture_output=True, text=True, timeout=1800)
        assert r.returncode == 0, r.stderr[-2000:]
        print(r.stdout.strip())
        res[pipe] = evaluate(str(npz))
        print(pipe, res[pipe])
    return res


def test_detection_rate(embeddings):
    assert embeddings["cpp"]["detect_rate"] >= 0.90
    assert embeddings["cpp"]["detect_rate"] >= embeddings["original"]["detect_rate"] - 0.01


def test_rank1_not_worse_than_original(embeddings):
    assert embeddings["cpp"]["rank1"] >= embeddings["original"]["rank1"]


def test_attendance_decisions_not_worse(embeddings):
    cpp, orig = embeddings["cpp"], embeddings["original"]
    assert cpp["accept@0.45"] >= orig["accept@0.45"]
    assert cpp["false@0.45"] <= orig["false@0.45"]
