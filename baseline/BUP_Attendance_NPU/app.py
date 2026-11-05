#!/usr/bin/env python3
"""
BUP Face Attendance System - Full University Management
Bangladesh University of Professionals
Multi-role: Super Admin → Department Admin → Teacher
"""

import os, glob, json, shutil, hashlib, threading, time, signal, sys
from datetime import datetime
from collections import defaultdict
from functools import wraps

import cv2
import numpy as np
from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, jsonify, Response, send_file, abort)
from werkzeug.utils import secure_filename

# ─────────────────────────────────────────────
#  NPU Setup — DeepX DX M1 ONLY
#  NO CPU/GPU FALLBACK — NPU is REQUIRED
# ─────────────────────────────────────────────
try:
    import dx_engine as dx
    DX_AVAILABLE = True
    print("  ✅ DeepX dx_engine loaded successfully")
except ImportError:
    print("=" * 70)
    print("  ❌ FATAL: DeepX dx_engine NOT FOUND!")
    print("  " + "-" * 66)
    print("  This system requires the DeepX DX-M1 NPU runtime.")
    print("")
    print("  Fix steps:")
    print("    1. Activate the DeepX environment first:")
    print("       source ~/workspace/dx-all-suite/dx-runtime/venv-dx-runtime/bin/activate")
    print("    2. Then run this app:")
    print("       python3 app.py")
    print("")
    print("  The dx_engine library is provided by the DeepX SDK and")
    print("  is NOT available on PyPI. It must be loaded from the")
    print("  DeepX virtual environment.")
    print("=" * 70)
    sys.exit(1)

GPU_MODE = "DeepX DX-M1 NPU"   # Shown in UI (reusing var name for template compat)
USE_GPU  = True                  # True = hardware accelerated (NPU in this case)

# ── DeepX NPU Model Paths ──
# Place these .dxnn files in the project root directory
DET_MODEL_FILE = "SCRFD10G.dxnn"                # Face detection model
REC_MODEL_FILE = "arcface_iresnet100_ms1m.dxnn"  # Face recognition model

# ── ArcFace 5-point alignment reference ──
ARC_FACE_SRC = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]
], dtype=np.float32)


def _nms(boxes, scores, iou_threshold):
    """Non-maximum suppression for SCRFD face detection."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


class DeepXFacePipeline:
    """
    Face Detection + Recognition pipeline running on DeepX DX-M1 NPU.
    Uses SCRFD10G for detection and ArcFace iResNet100 for recognition.
    Replaces InsightFace + onnxruntime-gpu entirely.
    """

    def __init__(self, det_path, rec_path):
        print("\n  🔧 Initializing DeepX InferenceEngine on DX-M1 NPU...")
        t0 = time.time()
        self.det_engine = dx.InferenceEngine(det_path)
        self.rec_engine = dx.InferenceEngine(rec_path)
        elapsed = time.time() - t0
        print(f"  ✅ DeepX NPU engines loaded ({elapsed:.1f}s)")
        print(f"     Detection model: {os.path.basename(det_path)}")
        print(f"     Recognition model: {os.path.basename(rec_path)}")

    def get(self, img):
        """Detect faces and compute embeddings. Returns list of FaceObj."""
        faces = []
        original_h, original_w = img.shape[:2]

        # 1. Detection — resize to 640x640 for SCRFD
        det_input = cv2.resize(img, (640, 640))
        det_input = cv2.cvtColor(det_input, cv2.COLOR_BGR2RGB)
        det_input_expanded = np.ascontiguousarray(np.expand_dims(det_input, axis=0))

        try:
            det_outputs = self.det_engine.run(det_input_expanded)
        except Exception:
            return faces

        if not isinstance(det_outputs, list):
            det_outputs = [det_outputs]

        bboxes, kpss = self._decode_scrfd(det_outputs, original_w, original_h, 640, conf_thresh=0.5)

        for bbox, kps in zip(bboxes, kpss):
            # 2. Alignment — ArcFace 5-point warp
            aligned_face = self._align_face(img, kps)
            if aligned_face is None:
                continue

            rec_input = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            rec_input_expanded = np.ascontiguousarray(np.expand_dims(rec_input, axis=0))

            # 3. Recognition — compute embedding on NPU
            try:
                emb_output = self.rec_engine.run(rec_input_expanded)
                emb_output = emb_output[0] if isinstance(emb_output, list) else emb_output
            except Exception:
                continue

            class FaceObj:
                pass

            face = FaceObj()
            face.bbox = bbox[:4]
            face.det_score = bbox[4]
            face.normed_embedding = emb_output.flatten()
            faces.append(face)

        return faces

    def _decode_scrfd(self, outputs, img_w, img_h, input_size=640, conf_thresh=0.5):
        """Decode SCRFD multi-stride detection outputs."""
        strides = [8, 16, 32]
        scores_list, bboxes_list, kpss_list = [], [], []

        for i, stride in enumerate(strides):
            score = np.squeeze(outputs[i])
            bbox = np.squeeze(outputs[i + 3])
            kps = np.squeeze(outputs[i + 6])

            if score.max() > 1.0 or score.min() < 0.0:
                score = 1.0 / (1.0 + np.exp(-np.clip(score, -10, 10)))

            feat_h, feat_w = input_size // stride, input_size // stride
            yy, xx = np.mgrid[:feat_h, :feat_w]
            centers = (np.vstack([xx.ravel(), yy.ravel()]).T + 0.5) * stride
            centers = np.repeat(centers, 2, axis=0)

            mask = score > conf_thresh
            score, bbox, kps, centers = score[mask], bbox[mask], kps[mask], centers[mask]

            if len(score) == 0:
                continue

            decoded_bboxes = np.zeros_like(bbox)
            decoded_bboxes[:, 0] = centers[:, 0] - bbox[:, 0] * stride
            decoded_bboxes[:, 1] = centers[:, 1] - bbox[:, 1] * stride
            decoded_bboxes[:, 2] = centers[:, 0] + bbox[:, 2] * stride
            decoded_bboxes[:, 3] = centers[:, 1] + bbox[:, 3] * stride

            decoded_kps = np.zeros_like(kps)
            for k in range(5):
                decoded_kps[:, k * 2] = centers[:, 0] + kps[:, k * 2] * stride
                decoded_kps[:, k * 2 + 1] = centers[:, 1] + kps[:, k * 2 + 1] * stride

            scores_list.append(score)
            bboxes_list.append(decoded_bboxes)
            kpss_list.append(decoded_kps)

        if not scores_list:
            return [], []

        scores = np.concatenate(scores_list)
        bboxes = np.concatenate(bboxes_list)
        kpss = np.concatenate(kpss_list)

        x_scale, y_scale = img_w / input_size, img_h / input_size
        bboxes[:, 0::2] *= x_scale
        bboxes[:, 1::2] *= y_scale
        for k in range(5):
            kpss[:, k * 2] *= x_scale
            kpss[:, k * 2 + 1] *= y_scale

        keep = _nms(bboxes, scores, iou_threshold=0.4)

        final_bboxes, final_kpss = [], []
        for idx in keep:
            b = bboxes[idx].tolist()
            b.append(scores[idx])
            final_bboxes.append(b)
            final_kpss.append(kpss[idx].reshape((5, 2)))

        return final_bboxes, final_kpss

    def _align_face(self, img, kps):
        """5-point ArcFace alignment via affine warp."""
        M, _ = cv2.estimateAffinePartial2D(kps, ARC_FACE_SRC, method=cv2.LMEDS)
        if M is not None:
            return cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
        return None


INSIGHTFACE_AVAILABLE = False  # Not used — DeepX replaces InsightFace

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# ─────────────────────────────────────────────
#  GStreamer support detection
# ─────────────────────────────────────────────
HAS_GSTREAMER = False
try:
    _test_cap = cv2.VideoCapture("videotestsrc ! autovideosink", cv2.CAP_GSTREAMER)
    if _test_cap.isOpened():
        HAS_GSTREAMER = True
    _test_cap.release()
except Exception:
    pass

if not HAS_GSTREAMER:
    # Simpler check via build info
    try:
        build_info = cv2.getBuildInformation()
        if 'GStreamer:                   YES' in build_info:
            HAS_GSTREAMER = True
    except Exception:
        pass

STREAM_BACKEND = "GStreamer" if HAS_GSTREAMER else "FFmpeg"


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
HISTORY_DIR     = os.path.join(BASE_DIR, "attendance_history")
BACKUP_DIR      = os.path.join(BASE_DIR, "cloud_backup")
UPLOADS_DIR     = os.path.join(BASE_DIR, "uploads")
DB_FILE         = os.path.join(BASE_DIR, "bup_db.json")

SIM_THRESHOLD   = 0.45
MIN_PRESENT_SEC = 60

# ── DeepX DX-M1 NPU Settings ──
DET_SIZE         = (640, 640)    # SCRFD detection input size
FRAME_W          = 1280          # Processing width
FRAME_H          = 720           # Processing height
PROCESS_INTERVAL = 0.01          # NPU inference interval
NPU_WARMUP_RUNS  = 2             # Warm-up inference runs on startup

ALLOWED_IMG_EXT = {'jpg','jpeg','png','bmp','webp'}

def _resolve_student_folder(student):
    """Always return the correct absolute path for a student's image folder,
    regardless of what absolute path is stored in the DB (handles machine migration)."""
    sid  = student.get('student_id', '')
    name = student.get('name', '')
    # Preferred: use stored path if it exists
    stored = student.get('images_folder', '')
    if stored and os.path.isdir(stored):
        return stored
    # Fallback: reconstruct from DATA_DIR + stored basename
    if stored:
        basename = os.path.basename(stored)
        candidate = os.path.join(DATA_DIR, basename)
        if os.path.isdir(candidate):
            return candidate
    # Last resort: sid_name pattern
    safe_name = name.replace(' ', '_')
    candidate2 = os.path.join(DATA_DIR, f"{sid}_{safe_name}")
    if os.path.isdir(candidate2):
        return candidate2
    # Create fresh folder
    folder = os.path.join(DATA_DIR, f"{sid}_{safe_name}")
    os.makedirs(folder, exist_ok=True)
    return folder



for d in [DATA_DIR, HISTORY_DIR, BACKUP_DIR, UPLOADS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────
#  BUP University Structure
# ─────────────────────────────────────────────
BUP_STRUCTURE = {
    "FST": {
        "name": "Faculty of Science & Technology",
        "building": "FST Tower",
        "departments": {
            "ICT": {"name": "Information and Communication Technology", "floors": [4,5,6,7]},
            "ES":  {"name": "Environmental Science", "floors": [1,2,3]},
            "CSE": {"name": "Computer Science and Engineering", "floors": [8,9,10,11]},
        }
    },
    "FBS": {
        "name": "Faculty of Business Studies",
        "building": "FBS Tower",
        "departments": {
            "MGT":    {"name": "Business Administration – Management Studies", "floors": [1,2]},
            "FIN":    {"name": "Business Administration in Finance & Banking", "floors": [3,4]},
            "MKT":    {"name": "Business Administration in Marketing", "floors": [5,6]},
            "AIS":    {"name": "Business Administration in Accounting & Information Systems", "floors": [7,8]},
            "BBA":    {"name": "Business Administration – General", "floors": [9,10,11]},
        }
    },
    "FSSS": {
        "name": "Faculty of Security & Strategic Studies",
        "building": "FSSS Tower",
        "departments": {
            "IR":   {"name": "International Relations", "floors": [1,2,3]},
            "LAW":  {"name": "Law", "floors": [4,5,6]},
            "MCJ":  {"name": "Mass Communication & Journalism", "floors": [7,8,9]},
            "PCHR": {"name": "Peace, Conflict & Human Rights", "floors": [10,11,12]},
        }
    },
    "FASS": {
        "name": "Faculty of Arts & Social Sciences",
        "building": "FASS Tower",
        "departments": {
            "ECON": {"name": "Economics", "floors": [1,2]},
            "DMR":  {"name": "Disaster Management and Resilience", "floors": [3,4]},
            "ENG":  {"name": "English", "floors": [5,6]},
            "PA":   {"name": "Public Administration", "floors": [7,8]},
            "SOC":  {"name": "Sociology", "floors": [9,10]},
            "DS":   {"name": "Development Studies", "floors": [11,12]},
        }
    }
}

def get_rooms_for_floor(floor_num):
    """Return 4 room numbers for a floor (e.g. floor 5 → 501-504)"""
    return [floor_num*100 + i for i in range(1,5)]

def get_all_buildings():
    return {fac: info["building"] for fac, info in BUP_STRUCTURE.items()}

def get_floors_for_faculty(faculty_code):
    floors = set()
    for dept in BUP_STRUCTURE[faculty_code]["departments"].values():
        floors.update(dept["floors"])
    return sorted(floors)

def get_dept_for_floor(faculty_code, floor):
    for dcode, dinfo in BUP_STRUCTURE[faculty_code]["departments"].items():
        if floor in dinfo["floors"]:
            return dcode, dinfo["name"]
    return None, None

# ─────────────────────────────────────────────
#  Database helpers (JSON file-based)
# ─────────────────────────────────────────────
def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return _default_db()

def save_db(db):
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

def _hash(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def _default_db():
    db = {
        "super_admin": {"username": "superadmin", "password": _hash("bup@2024"), "name": "Super Admin"},
        "dept_admins": {},   # {username: {password, name, dept_code, faculty_code}}
        "teachers": {},      # {username: {password, name, dept_code, faculty_code, courses}}
        "students": {},      # {student_id: {name, dept_code, faculty_code, batch, section, images_folder}}
        "courses": {},       # {course_id: {name, dept_code, faculty_code, batch, section, teacher_username}}
        "buildings": {},     # {faculty_code: {floors: {floor_num: {rooms: {room_num: {cam1, cam2}}}}}}
        "attendance": []     # list of records
    }
    # Seed building/floor/room/camera structure
    for fac_code, fac_info in BUP_STRUCTURE.items():
        db["buildings"][fac_code] = {"floors": {}}
        for floor in get_floors_for_faculty(fac_code):
            rooms = {}
            for room_num in get_rooms_for_floor(floor):
                rooms[str(room_num)] = {
                    "camera1": f"rtsp://admin:password@10.10.{floor}.{room_num % 100 + 1}:554/cam/realmonitor?channel=1&subtype=0",
                    "camera2": f"rtsp://admin:password@10.10.{floor}.{room_num % 100 + 2}:554/cam/realmonitor?channel=1&subtype=0"
                }
            db["buildings"][fac_code]["floors"][str(floor)] = {"rooms": rooms}
    return db

# ─────────────────────────────────────────────
#  Flask App
# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "BUP_SECRET_KEY_2024_FACE_ATTENDANCE"
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

@app.context_processor
def inject_globals():
    return dict(gpu_mode=GPU_MODE, use_gpu=USE_GPU, stream_backend=STREAM_BACKEND)

# ─────────────────────────────────────────────
#  Auth decorators
# ─────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            flash('Please log in first.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if session.get('role') not in roles:
                flash('Access denied.', 'danger')
                return redirect(url_for('dashboard'))
            return f(*args, **kwargs)
        return decorated
    return decorator

# ─────────────────────────────────────────────
#  Face Recognition Engine
# ─────────────────────────────────────────────
face_engine = None

def get_face_engine():
    global face_engine
    if face_engine is None:
        det_path = os.path.join(BASE_DIR, DET_MODEL_FILE)
        rec_path = os.path.join(BASE_DIR, REC_MODEL_FILE)

        if not os.path.exists(det_path):
            print(f"  ❌ Detection model not found: {det_path}")
            print(f"     Place {DET_MODEL_FILE} in the project root directory.")
            return None
        if not os.path.exists(rec_path):
            print(f"  ❌ Recognition model not found: {rec_path}")
            print(f"     Place {REC_MODEL_FILE} in the project root directory.")
            return None

        try:
            pipeline = DeepXFacePipeline(det_path, rec_path)

            # NPU Warm-up
            print("  🔥 Warming up NPU...")
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            for i in range(NPU_WARMUP_RUNS):
                _ = pipeline.get(dummy)
            print("  ✅ DeepX DX-M1 NPU ready!")

            face_engine = pipeline
        except Exception as e:
            print(f"  ❌ DeepX NPU init error: {e}")
            print(f"     Ensure DeepX environment is activated:")
            print(f"     source ~/workspace/dx-all-suite/dx-runtime/venv-dx-runtime/bin/activate")
            raise
    return face_engine


def l2_norm(v, axis=1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    n = np.where(n == 0, 1e-9, n)
    return v / n


def _get_embedding(face):
    """Safely get face embedding — handles numpy array truth value issue."""
    emb = getattr(face, 'normed_embedding', None)
    if emb is not None:
        return emb
    emb = getattr(face, 'embedding', None)
    if emb is not None:
        return emb
    return None

def compute_embedding(img):
    fa = get_face_engine()
    if fa is None:
        return None
    faces = fa.get(img)
    if not faces:
        return None
    face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    emb = _get_embedding(face)
    return emb

def build_embeddings_for_class(course_id):
    """Build/cache face embeddings for all students in a course."""
    db = load_db()
    course = db["courses"].get(course_id)
    if not course:
        return None, None
    students = [s for sid, s in db["students"].items()
                if s["dept_code"] == course["dept_code"]
                and s.get("batch") == course.get("batch")
                and s.get("section") == course.get("section")]

    cache_file = os.path.join(DATA_DIR, f"embeddings_{course_id}.npz")
    # Rebuild if student images changed
    rebuild = not os.path.exists(cache_file)

    if not rebuild:
        try:
            d = np.load(cache_file, allow_pickle=True)
            return d["embeds"], d["meta"].tolist()
        except Exception:
            rebuild = True

    fa = get_face_engine()
    if fa is None:
        return None, None

    db_vecs, meta = [], []
    print(f"  🧠 Building GPU embeddings for {len(students)} students...")
    t0 = time.time()

    for idx, s in enumerate(students):
        folder = _resolve_student_folder(s)
        if not folder or not os.path.isdir(folder):
            continue
        img_paths = []
        for ext in ALLOWED_IMG_EXT:
            img_paths += glob.glob(os.path.join(folder, f"*.{ext}"))
        embs = []
        # GPU batch: load all images then process
        images = []
        for p in img_paths:
            img = cv2.imread(p)
            if img is not None:
                images.append(img)

        # Process images on GPU
        for img in images:
            e = compute_embedding(img)
            if e is not None:
                embs.append(e)

        if not embs:
            continue
        E = np.vstack(embs)
        mean_emb = l2_norm(np.mean(E, axis=0, keepdims=True))
        db_vecs.append(mean_emb)
        meta.append({
            "student_id": s["student_id"],
            "name": s["name"]
        })

    elapsed = time.time() - t0
    print(f"  ✅ Embeddings built: {len(db_vecs)} students in {elapsed:.1f}s (GPU)")
    if not db_vecs:
        return None, None
    db_vecs = np.vstack(db_vecs).astype(np.float32)
    np.savez(cache_file, embeds=db_vecs, meta=np.array(meta, dtype=object))
    return db_vecs, meta

# ─────────────────────────────────────────────
#  Live Session State (global)
# ─────────────────────────────────────────────
_session = {
    "active": False,
    "course_id": None,
    "teacher": None,
    "faculty": None,
    "floor": None,
    "room": None,
    "cameras": {},       # {cam_id: RTSPCapture}
    "frames": {},        # {cam_id: frame_ndarray}
    "frame_locks": {},
    "tracker": None,
    "threads": [],
    "shutdown": False,
    "db_embeds": None,
    "db_meta": None,
    "fps": 0
}
_session_lock = threading.Lock()


class AttendanceTracker:
    def __init__(self, meta_list, min_time=MIN_PRESENT_SEC):
        self.min_time = min_time
        self.session_start = time.time()
        self.students = {}
        for m in meta_list:
            sid = m["student_id"]
            self.students[sid] = {
                "name": m["name"],
                "total_time": 0.0,
                "last_seen": {},
                "first_seen": None,
                "detection_count": 0,
                "manual_override": None  # "present" | "absent" | None
            }

    def update(self, student_id, camera_id, now):
        s = self.students.get(student_id)
        if not s:
            return
        last = s["last_seen"].get(camera_id, 0)
        if last > 0:
            diff = now - last
            if diff < 2.0:
                s["total_time"] += diff
        s["last_seen"][camera_id] = now
        if s["first_seen"] is None:
            s["first_seen"] = now
        s["detection_count"] += 1

    def set_manual(self, student_id, status):
        if student_id in self.students:
            self.students[student_id]["manual_override"] = status

    def summary(self):
        rows = []
        for sid, s in self.students.items():
            if s["manual_override"]:
                present = s["manual_override"] == "present"
            else:
                present = s["total_time"] >= self.min_time
            rows.append({
                "student_id": sid,
                "name": s["name"],
                "total_time": round(s["total_time"], 1),
                "present": present,
                "status": "Present" if present else "Absent",
                "detection_count": s["detection_count"],
                "manual_override": s["manual_override"]
            })
        return rows

    def live_stats(self):
        total = len(self.students)
        present = sum(1 for s in self.students.values()
                      if (s["manual_override"] == "present") or
                         (s["manual_override"] is None and s["total_time"] >= self.min_time))
        return {
            "total": total, "present": present, "absent": total - present,
            "rate": round(present/total*100 if total else 0, 1),
            "duration": round(time.time() - self.session_start, 0)
        }


class RTSPCapture:
    """
    RTSP Camera Capture with GStreamer pipeline (preferred) or FFmpeg fallback.

    GStreamer advantages over FFmpeg for RTSP:
      - Ultra-low latency (<500ms vs 2-5s)
      - UDP transport (less overhead than TCP)
      - Hardware-accelerated decoding (VA-API, NVDEC)
      - Real-time frame dropping (no stale buffer buildup)
      - Better reconnection handling
    """

    def __init__(self, cam_id, url):
        self.cam_id = cam_id
        self.url = url
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self._backend = None
        self._fail_count = 0
        self._max_fails = 10

    def _build_gst_pipeline(self):
        """Build an optimized GStreamer pipeline for RTSP with low latency."""
        return (
            f'rtspsrc location="{self.url}" '
            f'  latency=100 '           # 100ms target latency
            f'  buffer-mode=auto '
            f'  drop-on-latency=true '   # Drop frames when behind
            f'  protocols=udp '          # UDP first (lower latency)
            f'  retry=5 '               # Auto-reconnect attempts
            f'  timeout=5000000 '        # 5s connection timeout
            f'! queue '
            f'  max-size-buffers=1 '     # Only keep latest frame
            f'  leaky=downstream '       # Drop old frames, keep newest
            f'! rtph264depay '
            f'! h264parse '
            f'! decodebin '
            f'! videoconvert '
            f'! videoscale '
            f'! video/x-raw,format=BGR '
            f'! appsink '
            f'  drop=true '             # Drop frames if app is slow
            f'  max-buffers=1 '         # Buffer only 1 frame
            f'  sync=false '            # Don't sync to clock (real-time)
            f'  emit-signals=false'
        )

    def _build_gst_pipeline_h265(self):
        """Fallback GStreamer pipeline for H.265/HEVC cameras."""
        return (
            f'rtspsrc location="{self.url}" '
            f'  latency=100 buffer-mode=auto '
            f'  drop-on-latency=true protocols=udp '
            f'! queue max-size-buffers=1 leaky=downstream '
            f'! rtph265depay '
            f'! h265parse '
            f'! decodebin '
            f'! videoconvert '
            f'! video/x-raw,format=BGR '
            f'! appsink drop=true max-buffers=1 sync=false'
        )

    def _build_gst_pipeline_auto(self):
        """Auto-detect codec GStreamer pipeline (most compatible)."""
        return (
            f'rtspsrc location="{self.url}" '
            f'  latency=150 buffer-mode=auto '
            f'  drop-on-latency=true protocols=udp+tcp '
            f'! decodebin '
            f'! videoconvert '
            f'! video/x-raw,format=BGR '
            f'! appsink drop=true max-buffers=1 sync=false'
        )

    def _open_gstreamer(self):
        """Try multiple GStreamer pipelines in order of preference."""
        pipelines = [
            ("H.264", self._build_gst_pipeline()),
            ("Auto",  self._build_gst_pipeline_auto()),
            ("H.265", self._build_gst_pipeline_h265()),
        ]
        for name, pipeline in pipelines:
            try:
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        self._backend = f"GStreamer ({name})"
                        return cap
                cap.release()
            except Exception:
                pass
        return None

    def _open_ffmpeg(self):
        """Fallback: open with FFmpeg backend."""
        try:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Reduce FFmpeg latency as much as possible
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                self._backend = "FFmpeg"
                return cap
            cap.release()
        except Exception:
            pass
        return None

    def _open_camera(self):
        """Open camera: try GStreamer first, then FFmpeg."""
        if HAS_GSTREAMER:
            cap = self._open_gstreamer()
            if cap:
                print(f"  📹 Camera {self.cam_id}: Connected via {self._backend}")
                return cap

        cap = self._open_ffmpeg()
        if cap:
            print(f"  📹 Camera {self.cam_id}: Connected via {self._backend}")
            return cap

        return None

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()
        print(f"  🎥 Camera {self.cam_id}: Starting ({STREAM_BACKEND} preferred)...")

    def _run(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.cap = self._open_camera()
                if self.cap is None:
                    self._fail_count += 1
                    wait = min(3 * self._fail_count, 30)  # Exponential backoff, max 30s
                    print(f"  ⚠️  Camera {self.cam_id}: Connection failed (attempt {self._fail_count}), retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                self._fail_count = 0  # Reset on success

            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
            else:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
                time.sleep(1)
            time.sleep(PROCESS_INTERVAL)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return (True, self.frame.copy())
            return (False, None)

    def stop(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass

    def get_info(self):
        """Return connection info for debugging/UI."""
        return {
            "cam_id": self.cam_id,
            "backend": self._backend or "Not connected",
            "has_frame": self.frame is not None,
            "fail_count": self._fail_count,
        }


def _process_camera(cam_id):
    """GPU-accelerated camera processing with optimized face recognition."""
    global _session
    db_embeds = _session["db_embeds"]
    db_meta   = _session["db_meta"]
    tracker   = _session["tracker"]
    cap       = _session["cameras"][cam_id]
    fa        = get_face_engine()

    # Pre-compute contiguous GPU-ready array once
    db_np     = np.ascontiguousarray(db_embeds, dtype=np.float32)

    frame_count = 0
    fps_start = time.time()

    while not _session["shutdown"]:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        now = time.time()
        frame_count += 1

        # Resize to processing resolution
        if frame.shape[1] > FRAME_W or frame.shape[0] > FRAME_H:
            frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_LINEAR)

        # GPU face detection + recognition
        if fa is not None:
            faces = fa.get(frame)

            if faces:
                # Batch all embeddings at once for GPU efficiency
                embeddings = []
                valid_faces = []
                for face in faces:
                    emb = _get_embedding(face)
                    if emb is not None:
                        embeddings.append(emb)
                        valid_faces.append(face)

                if embeddings:
                    # Vectorized similarity — all faces vs all students in one GPU op
                    emb_matrix = l2_norm(np.array(embeddings, dtype=np.float32))
                    sim_matrix = np.dot(emb_matrix, db_np.T)  # [num_faces x num_students]

                    for i, face in enumerate(valid_faces):
                        sims = sim_matrix[i]
                        best = int(np.argmax(sims))
                        score = float(sims[best])
                        x1, y1, x2, y2 = map(int, face.bbox)

                        if score >= SIM_THRESHOLD:
                            sid = db_meta[best]["student_id"]
                            tracker.update(sid, cam_id, now)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                            t = tracker.students[sid]["total_time"]
                            label = f"{db_meta[best]['name']} {t:.0f}s"
                            cv2.putText(frame, label,
                                        (x1, max(15, y1 - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2)
                        else:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 220), 2)
                            cv2.putText(frame, "Unknown",
                                        (x1, max(15, y1 - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 2)

        # FPS counter — update every second for live display
        fps_elapsed = time.time() - fps_start
        if fps_elapsed >= 1.0:
            _session["fps"] = round(frame_count / fps_elapsed, 1)
            frame_count = 0
            fps_start = time.time()

        with _session["frame_locks"].get(cam_id, threading.Lock()):
            _session["frames"][cam_id] = frame

        time.sleep(PROCESS_INTERVAL)


def _gen_feed():
    while _session["active"]:
        frames = {}
        for cid in list(_session["frames"].keys()):
            lk = _session["frame_locks"].get(cid)
            if lk:
                with lk:
                    f = _session["frames"].get(cid)
                    if f is not None:
                        frames[cid] = f.copy()
        if not frames:
            blank = np.zeros((480,640,3), np.uint8)
            cv2.putText(blank,"Waiting for camera...",(30,240),cv2.FONT_HERSHEY_SIMPLEX,1,(0,220,220),2)
            combined = blank
        elif len(frames) == 1:
            combined = list(frames.values())[0]
            if combined.shape[1] > 960:
                combined = cv2.resize(combined,(960,540))
        else:
            fs = list(frames.values())
            f1 = cv2.resize(fs[0],(480,540))
            f2 = cv2.resize(fs[1],(480,540))
            cv2.putText(f1,"Camera 1",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.putText(f2,"Camera 2",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            combined = np.hstack([f1,f2])
        # Stats bar
        if _session["tracker"]:
            s = _session["tracker"].live_stats()
            h = combined.shape[0]
            cv2.rectangle(combined,(0,h-45),(combined.shape[1],h),(0,0,0),-1)
            fps_val = _session.get("fps", 0)
            cv2.putText(combined,
                f"Present: {s['present']}/{s['total']}  |  Rate: {s['rate']}%  |  Time: {s['duration']:.0f}s  |  {GPU_MODE} | {STREAM_BACKEND} | {fps_val} FPS",
                (10,h-15),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
        _, buf = cv2.imencode('.jpg', combined)
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        time.sleep(0.033)

