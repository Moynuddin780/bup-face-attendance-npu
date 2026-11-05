#!/usr/bin/env python3
"""
BUP Face Attendance System - Full University Management
Bangladesh University of Professionals
Multi-role: Super Admin → Department Admin → Teacher
"""

import os, glob, json, re, shutil, hashlib, threading, time, signal, sys
from datetime import datetime
from collections import defaultdict
from functools import wraps

import numpy as np
import camera_control
from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, jsonify, Response, send_file, send_from_directory, abort)
from werkzeug.utils import secure_filename
from werkzeug.security import safe_join

# ─────────────────────────────────────────────
#  NPU Setup — DeepX DX M1 ONLY
#  NO CPU/GPU FALLBACK — NPU is REQUIRED
#  Camera decoding, face detection + recognition, matching, attendance timing and
#  the live video overlay run in the C++ engine `bup_engine` (DXRT C++ API).
# ─────────────────────────────────────────────
try:
    import bup_engine
    print(f"  ✅ DeepX C++ engine loaded (bup_engine {bup_engine.__version__})")
except ImportError as _e:
    print("=" * 70)
    print("  ❌ FATAL: bup_engine (C++ NPU engine) NOT FOUND!")
    print("  " + "-" * 66)
    print(f"  {_e}")
    print("")
    print("  Fix steps:")
    print("    1. Build it once (checks the DeepX runtime, builds the C++ engine):")
    print("       ./setup.sh")
    print("    2. Then run this app:")
    print("       ./run.sh")
    print("=" * 70)
    sys.exit(1)

GPU_MODE = "DeepX DX-M1 NPU"   # Shown in UI (reusing var name for template compat)
USE_GPU  = True                  # True = hardware accelerated (NPU in this case)

# ── DeepX NPU Model Paths (relative to the project directory, or absolute) ──
DET_MODEL_FILE = os.environ.get("DET_MODEL_PATH", "models/scrfd-10g_640x640.dxnn")                # Face detection model
# Face recognition: InsightFace ArcFace ResNet-50 trained on WebFace600K (buffalo_l's w600k_r50), compiled
# for the NPU by tools/model_convert (DX-COM 2.4.0). On the enrolled photos it puts a person who is not
# enrolled on a student 0.0 % of the time at its threshold, where the earlier MS1M ResNet-100 did 5 %.
REC_MODEL_FILE = os.environ.get("REC_MODEL_PATH", "models/arcface_w600k_r50_112x112.dxnn")
NPU_BOUND      = os.environ.get("NPU_BOUND", "auto")   # NPU core binding: auto | all | 0 | 1 | 2


INSIGHTFACE_AVAILABLE = False  # Not used — DeepX replaces InsightFace

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# ─────────────────────────────────────────────
#  GStreamer support detection
# ─────────────────────────────────────────────
HAS_GSTREAMER = bup_engine.has_gstreamer()   # the C++ capture uses system GStreamer (rtspsrc + appsink)
STREAM_BACKEND = "GStreamer" if HAS_GSTREAMER else "FFmpeg"


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
HISTORY_DIR     = os.path.join(BASE_DIR, "attendance_history")
BACKUP_DIR      = os.path.join(BASE_DIR, "cloud_backup")
UPLOADS_DIR     = os.path.join(BASE_DIR, "uploads")
DB_FILE         = os.environ.get("BUP_DB_FILE") or os.path.join(BASE_DIR, "bup_db.json")

# Cosine-similarity threshold for "this face is that student", per recognition model: each model
# spreads its scores differently. Measured on the enrolled photos with tools/model_convert/report.py
# (the lowest threshold that keeps people who are not enrolled at or below 1 % of probes, rounded
# up); SIM_THRESHOLD in the environment overrides it.
MODEL_SIM_THRESHOLDS = {
    "arcface_w600k_r50_112x112": 0.42,         # strangers <= 1 % from 0.405; right student 99.5 %
    "arcface_iresnet100_112x112_ms1m": 0.45,   # the earlier model (strangers 5 % at 0.45)
}
_rec_stem = os.path.splitext(os.path.basename(REC_MODEL_FILE))[0]
SIM_THRESHOLD   = float(os.environ.get("SIM_THRESHOLD") or MODEL_SIM_THRESHOLDS.get(_rec_stem, 0.45))
MIN_PRESENT_SEC = 60

# ── DeepX DX-M1 NPU Settings ──
FRAME_W          = 1280          # Processing width  (larger camera frames are scaled down to fit)
FRAME_H          = 720           # Processing height
NPU_WARMUP_RUNS  = 2             # Warm-up inference runs on startup
FLASK_DEBUG      = os.environ.get("FLASK_DEBUG", "0") == "1"
# Live view JPEG quality: 80 is ~3x smaller than 95 and looks the same in the browser, which keeps
# the MJPEG feed usable on phones over Wi-Fi. Recognition always runs on the raw frames.
LIVE_JPEG_QUALITY = int(os.environ.get("LIVE_JPEG_QUALITY", "80"))
# Live view frame rate: the feed shows every camera frame up to this rate (cameras send 20-30 fps).
LIVE_FEED_FPS     = float(os.environ.get("LIVE_FEED_FPS", "30"))
# GPU (Intel VA-API) video decoding: "auto" uses it when the iGPU is reachable, "off" = CPU only.
# Streams wider than GPU_MAX_WIDTH are scaled down on the GPU (same shape) before the CPU sees them.
HW_DECODE         = os.environ.get("HW_DECODE", "auto").lower() != "off"
GPU_MAX_WIDTH     = int(os.environ.get("GPU_MAX_WIDTH", "1920"))
# Playout buffer (ms) for cameras whose frames arrive in Wi-Fi bursts: frames are released at
# their own timestamps, evenly. 0 = off (some cameras' timestamps do not suit it, e.g. Uniview).
LIVE_SMOOTH_MS    = int(os.environ.get("LIVE_SMOOTH_MS", "0"))
# Face tracking (ByteTrack): a recognised face keeps its name on its track; ArcFace re-checks it
# every TRACK_REVERIFY_S, a hidden face keeps its track for TRACK_LOST_S.
TRACK_LOST_S      = float(os.environ.get("TRACK_LOST_S", "2.0"))
TRACK_REVERIFY_S  = float(os.environ.get("TRACK_REVERIFY_S", "1.0"))
MAX_EMBED_PER_FRAME = int(os.environ.get("MAX_EMBED_PER_FRAME", "12"))
# Recorded-video testing: in these "FACULTY/FLOOR" floors a room camera can play an uploaded
# recording instead of its RTSP stream (inference tests). Everywhere else cameras are RTSP only.
VIDEO_TEST_FLOORS = {x.strip().upper() for x in os.environ.get("VIDEO_TEST_FLOORS", "FST/11").split(",") if x.strip()}
TEST_VIDEO_DIR    = os.path.join(UPLOADS_DIR, "test_videos")
TEST_VIDEO_MAX_MB = int(os.environ.get("TEST_VIDEO_MAX_MB", "2048"))
TEST_VIDEO_MIN_FREE_MB = int(os.environ.get("TEST_VIDEO_MIN_FREE_MB", "1024"))   # never fill the disk
TEST_VIDEO_EXTS   = {'.mp4', '.m4v', '.mov', '.mkv', '.avi', '.ts', '.mts', '.m2ts', '.webm', '.flv',
                     '.3gp', '.mpg', '.mpeg', '.wmv'}
UPLOAD_CHUNK_MAX  = 16 * 1024 * 1024   # the browser sends videos in 8 MB pieces

# ── Camera streams ──
STREAM_PROFILE   = os.environ.get("STREAM_PROFILE", "sub")      # sub = sub-stream (cheap CPU decode) | main
RTSP_TRANSPORT   = os.environ.get("RTSP_TRANSPORT", "tcp")      # tcp | udp | udp+tcp
SUBSTREAM_ASPECT = os.environ.get("SUBSTREAM_ASPECT", "auto")   # auto | off | 16:9  (un-squash D1 sub-streams)

ALLOWED_IMG_EXT = {'jpg','jpeg','png','bmp','webp'}
CONVERT_IMG_EXT  = {'heic','heif'}   # phone photos OpenCV cannot decode: stored as JPEG
CONVERT_MAX_SIDE = 2048              # converted photos are capped here (faces stay far above 112 px)
STUDENT_EMB_FILE = "embedding.npz"   # per-student face template, kept next to the photos

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

def _student_photos(folder):
    """A student's photo files, sorted. Extensions match case-insensitively (IMG_1.JPG counts)."""
    if not folder or not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if '.' in f and f.rsplit('.', 1)[1].lower() in ALLOWED_IMG_EXT
                  and os.path.isfile(os.path.join(folder, f)))

def _student_sections(student):
    """(batch, section) pairs a student attends: the home section plus extra_sections
    (a student who also takes classes with another section of the same department)."""
    pairs = [(student.get('batch'), student.get('section'))]
    pairs += [(x.get('batch'), x.get('section')) for x in student.get('extra_sections') or []]
    return pairs

def _in_class(student, course, section=None):
    """Is the student on this course's roster? A course covers its whole batch; a class session
    is usually held for one section of it (section=None: every section)."""
    if student.get('dept_code') != course.get('dept_code'):
        return False
    pairs = _student_sections(student)
    if section:
        return (course.get('batch'), section) in pairs
    return any(b == course.get('batch') for b, _ in pairs)

def _batch_sections(db, dept, batch):
    """{section: number of students} of one department's batch, in section order."""
    out = {}
    for st in db['students'].values():
        if st.get('dept_code') != dept:
            continue
        for b, x in _student_sections(st):
            if b == batch and x:
                out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))

def _next_photo_path(folder, ext):
    """img<N>.<ext> with the first N not used by any photo in the folder."""
    used = {os.path.basename(p).rsplit('.', 1)[0].lower() for p in _student_photos(folder)}
    n = 1
    while f"img{n}" in used:
        n += 1
    return os.path.join(folder, f"img{n}.{ext}")

HEIF_BRANDS = (b'heic', b'heix', b'hevc', b'hevx', b'heim', b'heis', b'mif1', b'msf1')

def is_heif(head):
    """True for a HEIF/HEIC container (iPhone photos) — phones and chat apps often save these
    under a .jpg name, which OpenCV then cannot read. `head` = the file's first 12 bytes."""
    return len(head) >= 12 and head[4:8] == b'ftyp' and head[8:12] in HEIF_BRANDS

def convert_photo_to_jpeg(src, dst):
    """HEIC/HEIF phone photo (path or file object) -> JPEG at dst. EXIF rotation is applied and
    the long side capped at CONVERT_MAX_SIDE. Raises on an unreadable file."""
    from PIL import Image, ImageOps
    import pillow_heif
    pillow_heif.register_heif_opener()
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        if max(im.size) > CONVERT_MAX_SIDE:
            im.thumbnail((CONVERT_MAX_SIDE, CONVERT_MAX_SIDE), Image.LANCZOS)
        im.save(dst, 'JPEG', quality=92)



AVATAR_DIR = os.path.join(UPLOADS_DIR, "avatars")   # account photos (teachers, admins)
AVATAR_SIDE = 320

def save_avatar(file_storage, role, uname):
    """An uploaded profile photo -> AVATAR_DIR/<role>_<user>_<time>.jpg: EXIF rotation applied,
    centre-cropped square, AVATAR_SIDE px, re-encoded as JPEG (so only a clean image is ever
    served). Any image Pillow reads, iPhone HEIC included. Returns the file name; raises ValueError."""
    from PIL import Image, ImageOps, UnidentifiedImageError
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    try:
        with Image.open(file_storage.stream) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            side = min(im.size)
            if side < 64:
                raise ValueError('the photo is too small (at least 64×64 pixels)')
            left, top = (im.width - side) // 2, (im.height - side) // 2
            im = im.crop((left, top, left + side, top + side)).resize((AVATAR_SIDE, AVATAR_SIDE), Image.LANCZOS)
            fname = f"{role}_{secure_filename(uname) or 'user'}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            os.makedirs(AVATAR_DIR, exist_ok=True)
            im.save(os.path.join(AVATAR_DIR, fname), 'JPEG', quality=90)
            return fname
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError('not an image file Pillow can read') from e

def remove_avatar_file(fname):
    if fname:
        p = os.path.join(AVATAR_DIR, secure_filename(fname))
        if os.path.isfile(p):
            os.remove(p)

for d in [DATA_DIR, HISTORY_DIR, BACKUP_DIR, UPLOADS_DIR, TEST_VIDEO_DIR, os.path.dirname(DB_FILE)]:
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

_db_lock = threading.Lock()

def save_db(db):
    """Atomic write (temp file + fsync + rename) under a lock: concurrent requests or a
    crash mid-write can never leave a truncated bup_db.json."""
    with _db_lock:
        tmp = f"{DB_FILE}.{os.getpid()}.tmp"
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(db, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, DB_FILE)

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
app.secret_key = os.environ.get("BUP_SECRET_KEY") or "BUP_SECRET_KEY_2024_FACE_ATTENDANCE"
if not os.environ.get("BUP_SECRET_KEY"):
    print("  ⚠️  BUP_SECRET_KEY not set — using the built-in key (set it in .env before deploying)")
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
# the session cookie is not sent with POSTs from other sites (deletes and edits are POST forms)
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

@app.context_processor
def inject_globals():
    return dict(gpu_mode=GPU_MODE, use_gpu=USE_GPU, stream_backend=STREAM_BACKEND,
                my_avatar=url_for('avatar', fname=session['photo']) if session.get('photo') else None,
                running_classes=len(_sessions) if session.get('role') == 'super_admin' else 0)

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

_face_engine_lock = threading.Lock()

def get_face_engine():
    global face_engine
    if face_engine is None:
        with _face_engine_lock:
            if face_engine is not None:
                return face_engine
            det_path = os.path.join(BASE_DIR, DET_MODEL_FILE)
            rec_path = os.path.join(BASE_DIR, REC_MODEL_FILE)

            if not os.path.exists(det_path):
                print(f"  ❌ Detection model not found: {det_path}")
                print(f"     Run ./setup.sh (downloads the models) or set DET_MODEL_PATH.")
                return None
            if not os.path.exists(rec_path):
                print(f"  ❌ Recognition model not found: {rec_path}")
                print(f"     Run ./setup.sh (downloads the models) or set REC_MODEL_PATH.")
                return None

            try:
                print("\n  🔧 Initializing DeepX InferenceEngine on DX-M1 NPU (C++)...")
                t0 = time.time()
                pipeline = bup_engine.FaceEngine(det_path, rec_path, npu_bound=NPU_BOUND)
                print(f"  ✅ DeepX NPU engines loaded ({time.time() - t0:.1f}s)")
                print(f"     Detection model: {os.path.basename(det_path)}")
                print(f"     Recognition model: {os.path.basename(rec_path)}")
                print(f"     Match threshold: {SIM_THRESHOLD}")

                # NPU Warm-up
                print("  🔥 Warming up NPU...")
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                for i in range(NPU_WARMUP_RUNS):
                    _ = pipeline.get(dummy)
                print("  ✅ DeepX DX-M1 NPU ready!")

                face_engine = pipeline
            except Exception as e:
                print(f"  ❌ DeepX NPU init error: {e}")
                print(f"     Check the NPU with: dxrt-cli -s")
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

EMBED_PIPELINE_VERSION = "bup_engine-2/scrfd-letterbox/umeyama-align/arcface-l2"

def _embedding_fingerprint(students):
    """Everything a class's cached embeddings depend on: NPU pipeline version, model files,
    the roster and every photo (name, size, mtime). Any change -> the cache is rebuilt."""
    h = hashlib.sha1(EMBED_PIPELINE_VERSION.encode())
    for model in (DET_MODEL_FILE, REC_MODEL_FILE):
        full = os.path.join(BASE_DIR, model)
        h.update(f"{os.path.basename(full)}:{os.path.getsize(full) if os.path.exists(full) else -1}".encode())
    for s in students:
        h.update(f"|{s['student_id']}|{s['name']}".encode())
        for p in _student_photos(_resolve_student_folder(s)):
            st = os.stat(p)
            h.update(f"{os.path.basename(p)}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()

def _photos_fingerprint(photos):
    """What one student's face template depends on: pipeline version, models, and the photos."""
    h = hashlib.sha1(EMBED_PIPELINE_VERSION.encode())
    for model in (DET_MODEL_FILE, REC_MODEL_FILE):
        full = os.path.join(BASE_DIR, model)
        h.update(f"{os.path.basename(full)}:{os.path.getsize(full) if os.path.exists(full) else -1}".encode())
    for p in photos:
        st = os.stat(p)
        h.update(f"|{os.path.basename(p)}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()

_embed_lock = threading.Lock()

def _load_student_embedding(cache, fingerprint):
    try:
        d = np.load(cache, allow_pickle=False)
        if str(d["fingerprint"]) != fingerprint:
            return None
        vec = d["embedding"].astype(np.float32)
        info = {"photos": int(d["photos"]), "faces": int(d["faces"]), "updated": str(d["updated"]), "cached": True}
        return (vec if vec.size else None), info
    except Exception:
        return None

def student_embedding(student, force=False):
    """A student's face template: the L2-normalised mean of the ArcFace embeddings of their
    photos (largest face per photo), the same math the class gallery always used. Saved in
    <photo folder>/embedding.npz and recomputed only when the photos, models or pipeline
    change, so adding photos costs NPU time once and starting a session costs none.
    Returns (vector (1, 512) or None, info {photos, faces, updated, cached})."""
    folder = _resolve_student_folder(student)
    photos = _student_photos(folder)
    fingerprint = _photos_fingerprint(photos)
    cache = os.path.join(folder, STUDENT_EMB_FILE)
    with _embed_lock:   # one builder at a time; a waiting caller then finds the fresh file
        if not force:
            hit = _load_student_embedding(cache, fingerprint)
            if hit is not None:
                return hit
        info = {"photos": len(photos), "faces": 0,
                "updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "cached": False}
        if not photos:
            if os.path.exists(cache):
                os.remove(cache)
            return None, info
        fa = get_face_engine()
        if fa is None:
            return None, dict(info, error="NPU engine unavailable")
        embs = []
        for p in photos:
            img = bup_engine.imread(p)
            if img is None:
                continue
            e = compute_embedding(img)
            if e is not None:
                embs.append(np.asarray(e, dtype=np.float32).reshape(-1))
        info["faces"] = len(embs)
        vec = (l2_norm(np.mean(np.vstack(embs), axis=0, keepdims=True)).astype(np.float32)
               if embs else np.zeros((0, 512), np.float32))
        tmp = f"{cache}.{os.getpid()}.tmp"
        with open(tmp, 'wb') as f:
            np.savez(f, embedding=vec, fingerprint=np.array(fingerprint), photos=info["photos"],
                     faces=info["faces"], updated=np.array(info["updated"]))
        os.replace(tmp, cache)
        return (vec if embs else None), info

def _embedding_summary(info):
    """The part of a student_embedding() info that is kept in the DB (students list badges)."""
    return {k: info[k] for k in ("photos", "faces", "updated")}

def refresh_student_embedding(sid):
    """Recompute one student's face template now and record photo count + result in the DB."""
    student = load_db()['students'].get(sid)
    if not student:
        return None
    _, info = student_embedding(student)
    db = load_db()   # re-read: the NPU work above takes seconds
    if sid in db['students']:
        db['students'][sid]['image_count'] = info['photos']
        db['students'][sid]['embedding'] = _embedding_summary(info)
        save_db(db)
    return info

def build_embeddings_for_class(course_id, section=None):
    """Build/cache face embeddings for all students in a course."""
    db = load_db()
    course = db["courses"].get(course_id)
    if not course:
        return None, None
    students = [s for sid, s in db["students"].items() if _in_class(s, course, section)]

    cache_file = os.path.join(DATA_DIR, f"embeddings_{course_id}{'_' + section if section else ''}.npz")
    fingerprint = _embedding_fingerprint(students)
    # Rebuild if student images, the roster or the NPU pipeline changed
    rebuild = not os.path.exists(cache_file)

    if not rebuild:
        try:
            d = np.load(cache_file, allow_pickle=True)
            if "fingerprint" in d.files and str(d["fingerprint"]) == fingerprint:
                return d["embeds"], d["meta"].tolist()
            rebuild = True
        except Exception:
            rebuild = True

    # Per-student templates are normally ready (computed when the photos were added); only
    # missing or outdated ones cost NPU time here.
    db_vecs, meta, fresh = [], [], {}
    print(f"  🧠 Loading face templates for {len(students)} students...")
    t0 = time.time()

    for s in students:
        vec, info = student_embedding(s)
        if not info.get("cached"):
            fresh[s["student_id"]] = info
        if vec is None:
            continue
        db_vecs.append(vec)
        meta.append({
            "student_id": s["student_id"],
            "name": s["name"]
        })

    if fresh:   # keep the students list badges in step with templates built here
        db = load_db()
        for sid, info in fresh.items():
            if sid in db["students"]:
                db["students"][sid]["image_count"] = info["photos"]
                db["students"][sid]["embedding"] = _embedding_summary(info)
        save_db(db)
    elapsed = time.time() - t0
    print(f"  ✅ Face templates ready: {len(db_vecs)} students in {elapsed:.1f}s "
          f"({len(fresh)} computed on the NPU, {len(students) - len(fresh)} from cache)")
    if not db_vecs:
        return None, None
    db_vecs = np.vstack(db_vecs).astype(np.float32)
    np.savez(cache_file, embeds=db_vecs, meta=np.array(meta, dtype=object), fingerprint=np.array(fingerprint))
    return db_vecs, meta

# ─────────────────────────────────────────────
#  Live Sessions — one per classroom
# ─────────────────────────────────────────────
# Every room runs its own session (cameras, NPU pipeline, tracker). A room is locked from the
# moment a teacher starts a class there until that class ends: no other teacher can select it or
# start a class in it, and a teacher runs one class at a time. An admin can end a stuck session.
_sessions = {}                      # room key "FST/5/501" -> session dict (see _new_session)
_sessions_lock = threading.Lock()


def _room_key(faculty, floor, room):
    return f"{faculty}/{floor}/{room}"


def _new_session(key, teacher, teacher_name, course_id, faculty, floor, room, section=None):
    return {
        "key": key, "active": False, "starting": True, "section": section,   # None = every section
        "course_id": course_id, "teacher": teacher, "teacher_name": teacher_name,
        "faculty": faculty, "floor": floor, "room": room,
        "started": time.time(),
        "cameras": {},       # {cam_id: bup_engine.CameraHandle}  (.get_info())
        "engine": None,      # bup_engine.AttendanceSession
        "tracker": None,
        "shutdown": False,
        "db_embeds": None,
        "db_meta": None,
        "ended_by": None,    # set when an admin ends the session
        "test_video": False, # a camera plays a recorded video (test room)
        "resumes": 0,        # times this class was resumed after a server restart
        "resumed_at": None,
    }


def _session_of(teacher):
    """The teacher's running (or starting) class, if any."""
    with _sessions_lock:
        return next((x for x in _sessions.values() if x["teacher"] == teacher), None)


def _busy_rooms(faculty=None):
    """{room key: who/what occupies it} — for the room picker and the admin pages."""
    db = load_db()
    with _sessions_lock:
        items = list(_sessions.values())
    out = {}
    for x in items:
        if faculty and x["faculty"] != faculty:
            continue
        out[x["key"]] = {
            "teacher": x["teacher"], "teacher_name": x["teacher_name"], "course_id": x["course_id"],
            "course_name": db["courses"].get(x["course_id"], {}).get("name", ""),
            "section": _section_label(x.get("section")),
            "since": datetime.fromtimestamp(x["started"]).strftime('%H:%M'),
            "starting": x["starting"],
        }
    return out


def _section_label(section):
    return f"Section {section}" if section else "All sections"


def _stop_session(sess=None):
    """Stop one session (releases its room), or every session when called without one
    (gunicorn / docker stop)."""
    if sess is None:
        with _sessions_lock:
            targets = list(_sessions.values())
        for x in targets:
            _stop_session(x)
        return
    sess["shutdown"] = True
    eng = sess.get("engine")
    if eng is not None:
        try: eng.stop()          # joins capture / NPU worker threads
        except Exception: pass
    sess["active"] = False
    with _sessions_lock:
        if _sessions.get(sess["key"]) is sess:
            del _sessions[sess["key"]]
    _save_live_state()   # an ended class is not resumed (no-op while the server shuts down)


# ─────────────────────────────────────────────
#  Classes survive a server restart
# ─────────────────────────────────────────────
# Every running class (room, teacher, course, each student's measured time) is snapshotted to
# LIVE_STATE_FILE every few seconds. When the worker process restarts — a crash, `docker compose up`
# after a deploy, a reboot — the classes are started again with the time measured so far. A class
# that cannot be resumed (room or course gone, down too long, crashing again and again) goes to its
# teacher's review-and-save page instead, so its attendance is never lost.
LIVE_STATE_FILE   = os.path.join(os.path.dirname(os.path.abspath(DB_FILE)), "live_sessions.json")
LIVE_STATE_EVERY_S = 5    # a crash loses at most this much of a class
RESUME_MAX_GAP_S  = float(os.environ.get("RESUME_MAX_GAP_S", "1800"))   # longer down: hand over, don't resume
RESUME_MAX_TIMES  = 3                                                  # more: a crash loop, hand over
_live_state_lock = threading.Lock()
_live_state = {"shutting_down": False, "written_empty": False,
               "carry": []}   # snapshot entries not resumed yet: kept in every save until handled


def _save_live_state():
    """Snapshot the running classes (and results waiting for review) to LIVE_STATE_FILE."""
    if _live_state["shutting_down"]:
        return
    with _sessions_lock:
        running = [x for x in _sessions.values() if x["active"] and x.get("tracker") is not None]
    rows = []
    for x in running:
        try:
            state = x["tracker"].export_state()
        except Exception:
            continue
        row = {k: x.get(k) for k in ("key", "teacher", "teacher_name", "course_id", "faculty", "floor", "room",
                                     "section", "started", "test_video", "resumes")}
        row["meta"] = [{"student_id": m["student_id"], "name": m["name"]} for m in (x.get("db_meta") or [])]
        row["tracker"] = state
        rows.append(row)
    keys = {r["key"] for r in rows}
    rows += [e for e in list(_live_state["carry"]) if e["key"] not in keys]
    pending = dict(_admin_ended)
    empty = not rows and not pending
    with _live_state_lock:
        if empty and _live_state["written_empty"]:
            return   # nothing running: no writes to the SD card
        tmp = LIVE_STATE_FILE + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"saved": time.time(), "sessions": rows, "pending": pending}, f, ensure_ascii=False)
            os.replace(tmp, LIVE_STATE_FILE)
            _live_state["written_empty"] = empty
        except OSError as e:
            print(f"  ⚠️  could not save the live classes: {e}")


def _live_state_loop():
    while not _live_state["shutting_down"]:
        time.sleep(LIVE_STATE_EVERY_S)
        try:
            _save_live_state()
        except Exception as e:   # never let the snapshot thread die
            print(f"  ⚠️  live class snapshot failed: {e}")


def _hand_over(entry, why):
    """A class that cannot be resumed: its results wait on the teacher's review-and-save page."""
    t = AttendanceTracker(entry.get("meta") or [], session_start=entry.get("started"))
    t.restore_state(entry.get("tracker") or [])
    summary = t.summary()
    if summary:
        _admin_ended[entry["teacher"]] = {
            "summary": summary, "course": entry["course_id"], "faculty": entry["faculty"], "floor": entry["floor"],
            "room": entry["room"], "section": entry.get("section"), "test_video": entry.get("test_video", False),
            "notice": f'Your class in Room {entry["room"]} ({entry["course_id"]}) was interrupted by a server restart '
                      f'and could not continue ({why}). The attendance measured until then is below — review and save it.'}
    print(f"  ⚠️  class {entry['course_id']} in {entry['key']} not resumed ({why}) — results kept for {entry['teacher']}")


def _resume_one(entry, gap):
    if gap > RESUME_MAX_GAP_S:
        return _hand_over(entry, f"the server was down for {gap / 60:.0f} minutes")
    if entry.get("resumes", 0) >= RESUME_MAX_TIMES:
        return _hand_over(entry, "it was interrupted several times in a row")
    db = load_db()
    if entry["course_id"] not in db["courses"]:
        return _hand_over(entry, "the course no longer exists")
    room_data = db["buildings"].get(entry["faculty"], {}).get("floors", {}).get(entry["floor"], {}).get("rooms", {}).get(entry["room"])
    sources = _room_sources(entry["faculty"], entry["floor"], room_data) if room_data else []
    if not sources:
        return _hand_over(entry, "the room has no cameras any more")
    with _sessions_lock:
        if entry["key"] in _sessions or any(x["teacher"] == entry["teacher"] for x in _sessions.values()):
            busy = True
        else:
            busy = False
            sess = _new_session(entry["key"], entry["teacher"], entry["teacher_name"], entry["course_id"],
                                entry["faculty"], entry["floor"], entry["room"], entry.get("section"))
            sess["started"] = entry["started"]
            sess["resumes"] = entry.get("resumes", 0) + 1
            sess["resumed_at"] = time.time()
            _sessions[entry["key"]] = sess
    if busy:
        return _hand_over(entry, "the room or the teacher is in another class")
    try:
        embeds, meta = build_embeddings_for_class(entry["course_id"], entry.get("section"))
        if embeds is None or not meta:
            raise RuntimeError("no student face data")
        tracker = AttendanceTracker(meta, session_start=entry["started"])
        tracker.restore_state(entry.get("tracker") or [])
        _launch_engine(sess, room_data, sources, embeds, meta, tracker)
    except Exception as e:
        _stop_session(sess)
        return _hand_over(entry, str(e))
    print(f"  ✅ class {entry['course_id']} in {entry['key']} ({entry['teacher_name']}) resumed after a server restart")


def _resume_live_sessions():
    try:
        with open(LIVE_STATE_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return
    _admin_ended.update(data.get("pending") or {})
    gap = time.time() - float(data.get("saved") or 0)
    entries = [e for e in data.get("sessions") or [] if e.get("key") and e.get("teacher")]
    _live_state["carry"] = list(entries)
    for entry in entries:
        try:
            _resume_one(entry, gap)
        except Exception as e:
            print(f"  ⚠️  resuming {entry.get('key')} failed: {e}")
            try: _hand_over(entry, str(e))
            except Exception: pass
        _live_state["carry"].remove(entry)
    _save_live_state()


def _live_state_main():
    _resume_live_sessions()   # first: a snapshot must not overwrite the classes still to resume
    _live_state_loop()


class AttendanceTracker(bup_engine.AttendanceTracker):
    """Attendance timing, implemented in C++ (bup_engine) with the original rules and outputs:
    a sighting adds the gap since the previous sighting on the same camera when it is < 2 s,
    a student is Present after min_time seconds, and a teacher's manual override wins.
    summary() / live_stats() / set_manual() return exactly what the Python class returned."""

    def __init__(self, meta_list, min_time=MIN_PRESENT_SEC, session_start=None):
        super().__init__(meta_list, min_time, session_start)


def _gen_feed(sess):
    """MJPEG stream for the live page. The C++ engine composes the camera views and the stats
    bar and encodes each JPEG once, shared by every open viewer."""
    last = None
    period = 1.0 / max(1.0, LIVE_FEED_FPS)
    due = time.monotonic()
    while sess["active"]:
        eng = sess.get("engine")
        if eng is None:
            break
        frame = eng.mjpeg_frame()   # the newest frame of every camera + the latest faces
        if frame != last:   # only new pictures go over the network (phones on Wi-Fi)
            last = frame
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        # paced to LIVE_FEED_FPS: the time spent composing counts against the period
        due += period
        delay = due - time.monotonic()
        if delay < -period:
            due = time.monotonic()
        else:
            time.sleep(max(0.0, delay))

# ─────────────────────────────────────────────
#  Routes – Auth
# ─────────────────────────────────────────────
@app.route('/', methods=['GET','POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        uname = request.form.get('username','').strip()
        pw    = request.form.get('password','')
        db    = load_db()
        # Super admin
        if uname == db["super_admin"]["username"] and db["super_admin"]["password"] == _hash(pw):
            session.update({'logged_in':True,'username':uname,'role':'super_admin','name':db["super_admin"]["name"],'dept':None,'faculty':None,'photo':db["super_admin"].get("photo")})
            return redirect(url_for('dashboard'))
        # Dept admin
        if uname in db["dept_admins"]:
            a = db["dept_admins"][uname]
            if a["password"] == _hash(pw):
                session.update({'logged_in':True,'username':uname,'role':'dept_admin','name':a["name"],'dept':a["dept_code"],'faculty':a["faculty_code"],'photo':a.get("photo")})
                return redirect(url_for('dashboard'))
        # Teacher
        if uname in db["teachers"]:
            t = db["teachers"][uname]
            if t["password"] == _hash(pw):
                session.update({'logged_in':True,'username':uname,'role':'teacher','name':t["name"],'dept':t["dept_code"],'faculty':t["faculty_code"],'photo':t.get("photo")})
                return redirect(url_for('dashboard'))
        flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ─────────────────────────────────────────────
#  Dashboard
# ─────────────────────────────────────────────
@app.route('/dashboard')
@login_required
def dashboard():
    db = load_db()
    role = session['role']
    stats = {}
    if role == 'super_admin':
        stats = {
            'faculties': len(BUP_STRUCTURE),
            'dept_admins': len(db['dept_admins']),
            'teachers': len(db['teachers']),
            'students': len(db['students']),
            'courses': len(db['courses']),
        }
    elif role == 'dept_admin':
        dept = session['dept']
        stats = {
            'teachers': sum(1 for t in db['teachers'].values() if t['dept_code']==dept),
            'students': sum(1 for s in db['students'].values() if s['dept_code']==dept),
            'courses':  sum(1 for c in db['courses'].values()  if c['dept_code']==dept),
        }
    elif role == 'teacher':
        uname = session['username']
        my_courses = [cid for cid,c in db['courses'].items() if c.get('teacher_username')==uname]
        stats = {'my_courses': len(my_courses)}
    recent = _get_recent_attendance(5, session.get('dept'), role)
    return render_template('dashboard.html', stats=stats, recent=recent,
                           bup=BUP_STRUCTURE, gpu_mode=GPU_MODE, use_gpu=USE_GPU)


def _normalize_record(d, fname=''):
    """Normalize old and new format attendance JSON records into a flat dict."""
    if not isinstance(d, dict):
        return d
    # If it already has top-level 'records' key (new format), just ensure filename
    if 'records' in d:
        d.setdefault('filename', fname.replace('.json',''))
        return d
    # Old format: has 'metadata' and 'attendance_records'
    meta = d.get('metadata', {})
    rows = d.get('attendance_records', [])
    # Normalize each row
    norm_rows = []
    for r in rows:
        norm_rows.append({
            'student_id':      r.get('student_id', r.get('id', '—')),
            'name':            r.get('student_name', r.get('name', '—')),
            'total_time':      r.get('total_time', r.get('total_detection_time', 0)),
            'detection_count': r.get('detection_count', 0),
            'present':         r.get('present', r.get('status','Absent') == 'Present'),
            'status':          r.get('status', 'Present' if r.get('present') else 'Absent'),
            'manual_override': r.get('manual_override', None),
        })
    present_count = sum(1 for r in norm_rows if r['present'])
    total = len(norm_rows)
    return {
        'filename':     d.get('session_id', fname.replace('.json','')),
        'course_id':    d.get('class_name', meta.get('class_name', d.get('class_name','—'))).replace(' | ','_').replace(' ','_'),
        'course_name':  d.get('class_name', meta.get('class_name','—')),
        'dept_code':    d.get('dept_code', ''),
        'faculty_code': d.get('faculty_code', ''),
        'batch':        d.get('batch', meta.get('batch','—')),
        'section':      d.get('section', meta.get('section','—')),
        'teacher':      d.get('metadata',{}).get('teacher', d.get('teacher','—')),
        'teacher_name': meta.get('teacher', d.get('teacher_name', d.get('teacher','—'))),
        'date':         meta.get('date', d.get('date','—')),
        'time':         meta.get('time', d.get('time','—')),
        'total':        meta.get('total_students', total),
        'present':      meta.get('present_count', present_count),
        'absent':       meta.get('absent_count', total - present_count),
        'rate':         float(meta.get('attendance_rate','0').replace('%','')) if isinstance(meta.get('attendance_rate'), str) else meta.get('attendance_rate', round(present_count/total*100 if total else 0, 1)),
        'gpu_mode':     meta.get('optimization_mode', d.get('gpu_mode', 'CPU')),
        'records':      norm_rows,
    }

def _get_recent_attendance(limit=10, dept=None, role=None):
    records = []
    files = sorted(glob.glob(os.path.join(HISTORY_DIR,"*.json")), reverse=True)[:limit*3]
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
            if dept and role != 'super_admin' and d.get('dept_code') != dept:
                continue
            rec = _normalize_record(d, os.path.basename(f))
            if not isinstance(rec, dict) or not _may_see_record(rec):   # a teacher: their own classes
                continue
            records.append(rec)
            if len(records) >= limit:
                break
        except Exception:
            pass
    return records

# ─────────────────────────────────────────────
#  Super Admin – Manage Dept Admins
# ─────────────────────────────────────────────
@app.route('/admins')
@login_required
@role_required('super_admin')
def list_admins():
    db = load_db()
    return render_template('admins.html', admins=db['dept_admins'], bup=BUP_STRUCTURE)

@app.route('/admins/add', methods=['GET','POST'])
@login_required
@role_required('super_admin')
def add_admin():
    db = load_db()
    if request.method == 'POST':
        uname    = request.form['username'].strip()
        pw       = request.form['password']
        name     = request.form['name'].strip()
        faculty  = request.form['faculty_code']
        dept     = request.form['dept_code']
        if uname in db['dept_admins'] or uname in db['teachers'] or uname == db['super_admin']['username']:
            flash('Username already exists.','danger')
        else:
            db['dept_admins'][uname] = {'password':_hash(pw),'name':name,'faculty_code':faculty,'dept_code':dept}
            save_db(db)
            flash(f'Department Admin "{name}" created.','success')
            return redirect(url_for('list_admins'))
    return render_template('add_admin.html', bup=BUP_STRUCTURE)

@app.route('/admins/delete/<uname>')
@login_required
@role_required('super_admin')
def delete_admin(uname):
    db = load_db()
    db['dept_admins'].pop(uname, None)
    save_db(db)
    flash('Admin deleted.','info')
    return redirect(url_for('list_admins'))

# ─────────────────────────────────────────────
#  Dept Admin – Manage Teachers
# ─────────────────────────────────────────────
def _affiliation_from_form(form):
    """('bup', '') or ('external', university) from the teacher form; ValueError when incomplete."""
    if form.get('affiliation') == 'external':
        uni = ' '.join(form.get('university', '').split())
        if not uni:
            raise ValueError('Enter the university of the adjunct faculty member.')
        return 'external', uni[:120]
    return 'bup', ''

@app.route('/teachers')
@login_required
@role_required('super_admin','dept_admin')
def list_teachers():
    db = load_db()
    dept = session.get('dept')
    teachers = {u:t for u,t in db['teachers'].items() if (dept is None or t['dept_code']==dept)}
    # BUP's own faculty first, then the adjunct faculty from other universities
    bup_t = {u: t for u, t in teachers.items() if t.get('affiliation', 'bup') != 'external'}
    adjunct = {u: t for u, t in teachers.items() if t.get('affiliation') == 'external'}
    return render_template('teachers.html', teachers=teachers, bup_teachers=bup_t, adjunct=adjunct, bup=BUP_STRUCTURE)

@app.route('/teachers/add', methods=['GET','POST'])
@login_required
@role_required('super_admin','dept_admin')
def add_teacher():
    db = load_db()
    if request.method == 'POST':
        uname   = request.form['username'].strip()
        pw      = request.form['password']
        name    = request.form['name'].strip()
        email   = request.form.get('email','').strip()
        faculty = request.form.get('faculty_code') or session.get('faculty')
        dept    = request.form.get('dept_code')    or session.get('dept')
        try:
            affiliation, university = _affiliation_from_form(request.form)
        except ValueError as e:
            flash(str(e), 'danger')
            return render_template('add_teacher.html', bup=BUP_STRUCTURE, form=request.form,
                                   fixed_faculty=session.get('faculty'), fixed_dept=session.get('dept'))
        if uname in db['teachers'] or uname in db['dept_admins'] or uname == db['super_admin']['username']:
            flash('Username already exists.','danger')
        else:
            db['teachers'][uname] = {
                'password': _hash(pw), 'name': name, 'email': email,
                'faculty_code': faculty, 'dept_code': dept, 'courses': [],
                'affiliation': affiliation, 'university': university,
            }
            save_db(db)
            flash(f'{"Adjunct faculty" if affiliation == "external" else "Teacher"} "{name}" added.','success')
            return redirect(url_for('list_teachers'))
    return render_template('add_teacher.html', bup=BUP_STRUCTURE, form=request.form,
                           fixed_faculty=session.get('faculty'), fixed_dept=session.get('dept'))

@app.route('/teachers/delete/<uname>')
@login_required
@role_required('super_admin','dept_admin')
def delete_teacher(uname):
    db = load_db()
    if session['role'] == 'dept_admin' and db['teachers'].get(uname,{}).get('dept_code') != session['dept']:
        abort(403)
    db['teachers'].pop(uname, None)
    save_db(db)
    flash('Teacher deleted.','info')
    return redirect(url_for('list_teachers'))

@app.route('/teachers/edit/<uname>', methods=['GET','POST'])
@login_required
@role_required('super_admin','dept_admin')
def edit_teacher(uname):
    db = load_db()
    teacher = db['teachers'].get(uname)
    if not teacher:
        abort(404)
    if session['role'] == 'dept_admin' and teacher.get('dept_code') != session['dept']:
        abort(403)
    if request.method == 'POST':
        try:
            teacher['affiliation'], teacher['university'] = _affiliation_from_form(request.form)
        except ValueError as e:
            flash(str(e), 'danger')
            return render_template('edit_teacher.html', teacher=teacher, uname=uname)
        teacher['name']  = request.form['name'].strip()
        teacher['email'] = request.form.get('email','').strip()
        if request.form.get('password'):
            teacher['password'] = _hash(request.form['password'])
        save_db(db)
        flash('Teacher updated.','success')
        return redirect(url_for('list_teachers'))
    return render_template('edit_teacher.html', teacher=teacher, uname=uname)

# ─────────────────────────────────────────────
#  Courses management
# ─────────────────────────────────────────────
@app.route('/courses')
@login_required
def list_courses():
    db = load_db()
    dept = session.get('dept')
    role = session['role']
    uname = session['username']
    if role == 'teacher':
        courses = {cid:c for cid,c in db['courses'].items() if c.get('teacher_username')==uname}
    elif role == 'dept_admin':
        courses = {cid:c for cid,c in db['courses'].items() if c.get('dept_code')==dept}
    else:
        courses = db['courses']
    teachers = db['teachers']
    sections = {cid: _batch_sections(db, c.get('dept_code'), c.get('batch')) for cid, c in courses.items()}
    # batch by batch, newest batch first; courses in course-code order inside a batch
    by_batch = {}
    for cid, c in sorted(courses.items()):
        by_batch.setdefault(str(c.get('batch') or '—'), []).append((cid, c))
    batches = (sorted(((b, l) for b, l in by_batch.items() if b != '—'), reverse=True)
               + [(b, l) for b, l in by_batch.items() if b == '—'])
    return render_template('courses.html', courses=courses, batches=batches, teachers=teachers, sections=sections,
                           bup=BUP_STRUCTURE)

@app.route('/courses/add', methods=['GET','POST'])
@login_required
@role_required('super_admin','dept_admin')
def add_course():
    db = load_db()
    if request.method == 'POST':
        cid      = request.form['course_id'].strip().upper()
        name     = request.form['name'].strip()
        faculty  = request.form.get('faculty_code') or session.get('faculty')
        dept     = request.form.get('dept_code')    or session.get('dept')
        batch    = request.form['batch'].strip()
        teacher  = request.form.get('teacher_username','').strip() or None
        if cid in db['courses']:
            flash('Course ID already exists.','danger')
        else:
            # one course = the whole batch: its teacher takes it with every section
            db['courses'][cid] = {
                'name': name, 'faculty_code': faculty, 'dept_code': dept,
                'batch': batch, 'teacher_username': teacher
            }
            _assign_course_teacher(db, cid, None, teacher)
            save_db(db)
            flash(f'Course "{name}" added.','success')
            return redirect(url_for('list_courses'))
    return render_template('add_course.html', bup=BUP_STRUCTURE,
                           dept_teachers=_dept_teachers(db),
                           fixed_faculty=session.get('faculty'), fixed_dept=session.get('dept'))

def _dept_teachers(db, dept=None):
    """Teachers a course can be given to: the admin's department (or the course's)."""
    dept = session.get('dept') or dept
    return {u: t for u, t in db['teachers'].items() if not dept or t.get('dept_code') == dept}

def _assign_course_teacher(db, cid, old, new):
    """Keep each teacher's course list in step with course.teacher_username."""
    if old and old != new and old in db['teachers']:
        lst = db['teachers'][old].get('courses', [])
        if cid in lst:
            lst.remove(cid)
    if new and new in db['teachers']:
        lst = db['teachers'][new].setdefault('courses', [])
        if cid not in lst:
            lst.append(cid)

def _course_or_404(db, cid):
    course = db['courses'].get(cid)
    if not course or (session.get('dept') and course.get('dept_code') != session.get('dept')):
        abort(404)   # dept admins only reach their own department's courses
    return course

@app.route('/courses/edit/<cid>', methods=['GET','POST'])
@login_required
@role_required('super_admin','dept_admin')
def edit_course(cid):
    db = load_db()
    course = _course_or_404(db, cid)
    if request.method == 'POST':
        old_teacher = course.get('teacher_username')
        teacher = request.form.get('teacher_username', '').strip() or None
        if teacher and teacher not in db['teachers']:
            flash('Unknown teacher.', 'danger')
            return redirect(url_for('edit_course', cid=cid))
        course['name'] = request.form.get('name', course.get('name', '')).strip() or course.get('name', '')
        course['batch'] = request.form.get('batch', course.get('batch', '')).strip() or course.get('batch', '')
        if not session.get('dept'):   # super admin may move a course to another department
            course['faculty_code'] = request.form.get('faculty_code') or course.get('faculty_code')
            course['dept_code'] = request.form.get('dept_code') or course.get('dept_code')
        course['teacher_username'] = teacher
        course.pop('section', None)   # courses cover every section of their batch now
        _assign_course_teacher(db, cid, old_teacher, teacher)
        save_db(db)
        who = db['teachers'][teacher]['name'] if teacher else 'nobody (unassigned)'
        flash(f'Course {cid} updated — assigned to {who} for every section of batch {course["batch"]}.', 'success')
        return redirect(url_for('list_courses'))
    return render_template('edit_course.html', cid=cid, course=course, bup=BUP_STRUCTURE,
                           dept_teachers=_dept_teachers(db, course.get('dept_code')),
                           sections=_batch_sections(db, course.get('dept_code'), course.get('batch')),
                           fixed_faculty=session.get('faculty'), fixed_dept=session.get('dept'))

@app.route('/courses/delete/<cid>')
@login_required
@role_required('super_admin','dept_admin')
def delete_course(cid):
    db = load_db()
    course = _course_or_404(db, cid)
    _assign_course_teacher(db, cid, course.get('teacher_username'), None)
    db['courses'].pop(cid, None)
    save_db(db)
    flash('Course deleted.','info')
    return redirect(url_for('list_courses'))

# ─────────────────────────────────────────────
#  Students management
# ─────────────────────────────────────────────
@app.route('/students')
@login_required
@role_required('super_admin','dept_admin')
def list_students():
    """Students are shown one class at a time: a dept admin picks batch + section of their
    department, a super admin picks department + batch + section. Before that, an overview of
    every batch / section (student and face-template counts). `q` searches by name or ID."""
    db = load_db()
    fixed_dept = session.get('dept')   # dept admins only ever see their own department
    scope = [s for s in db['students'].values() if fixed_dept is None or s.get('dept_code') == fixed_dept]

    def batch_key(b):
        return (-int(b), '') if str(b).isdigit() else (0, str(b))

    # overview: department -> batch -> section -> counts (a student in two sections counts in both)
    tree = {}
    for st in scope:
        ready = bool((st.get('embedding') or {}).get('faces'))
        for b, x in _student_sections(st):
            node = tree.setdefault(st.get('dept_code') or '?', {}).setdefault(str(b), {}).setdefault(
                str(x), {'total': 0, 'ready': 0, 'no_photos': 0})
            node['total'] += 1
            node['ready'] += ready
            node['no_photos'] += not st.get('image_count')
    dept_order = [d for f in BUP_STRUCTURE.values() for d in f['departments']]
    tree = {d: {b: dict(sorted(tree[d][b].items())) for b in sorted(tree[d], key=batch_key)}
            for d in sorted(tree, key=lambda d: dept_order.index(d) if d in dept_order else len(dept_order))}
    dept_names = {d: info['name'] for f in BUP_STRUCTURE.values() for d, info in f['departments'].items()}

    dept = fixed_dept or request.args.get('dept', '').strip()
    batch = request.args.get('batch', '').strip()
    section = request.args.get('section', '').strip().upper()
    q = request.args.get('q', '').strip()
    if q:
        ql = q.lower()
        rows = [st for st in scope if ql in st.get('name', '').lower() or ql in str(st.get('student_id', '')).lower()]
    elif dept and batch and section:
        rows = [st for st in scope if st.get('dept_code') == dept and (batch, section) in _student_sections(st)]
    else:
        rows = None   # overview only
    students = None
    if rows is not None:
        rows.sort(key=lambda st: (st.get('dept_code') or '', batch_key(str(st.get('batch') or '')),
                                  st.get('section') or '', str(st.get('student_id'))))
        students = {st['student_id']: dict(st, sections=_student_sections(st)) for st in rows}
    return render_template('students.html', students=students, tree=tree, dept_names=dept_names,
                           sel={'dept': dept, 'batch': batch, 'section': section, 'q': q},
                           fixed_dept=fixed_dept, total=len(scope), bup=BUP_STRUCTURE)

@app.route('/students/add', methods=['GET','POST'])
@login_required
@role_required('super_admin','dept_admin')
def add_student():
    db = load_db()
    if request.method == 'POST':
        sid     = request.form['student_id'].strip()
        name    = request.form['name'].strip()
        faculty = request.form.get('faculty_code') or session.get('faculty')
        dept    = request.form.get('dept_code')    or session.get('dept')
        batch   = request.form['batch'].strip()
        section = request.form['section'].strip().upper()
        if sid in db['students']:
            flash('Student ID already exists.','danger')
        else:
            folder = os.path.join(DATA_DIR, f"{sid}_{name.replace(' ','_')}")
            os.makedirs(folder, exist_ok=True)
            db['students'][sid] = {
                'student_id': sid, 'name': name, 'faculty_code': faculty,
                'dept_code': dept, 'batch': batch, 'section': section,
                'images_folder': folder, 'image_count': 0
            }
            save_db(db)
            flash(f'Student "{name}" added. Now upload their photos.','success')
            return redirect(url_for('upload_student_images', sid=sid))
    # coming from a class list: that department / batch / section is filled in
    pre = {k: request.args.get(k, '').strip() for k in ('dept', 'batch', 'section')}
    pre['faculty'] = next((f for f, info in BUP_STRUCTURE.items() if pre['dept'] in info['departments']), '')
    return render_template('add_student.html', bup=BUP_STRUCTURE, pre=pre,
                           fixed_faculty=session.get('faculty'), fixed_dept=session.get('dept'))

@app.route('/students/delete/<sid>')
@login_required
@role_required('super_admin','dept_admin')
def delete_student(sid):
    db = load_db()
    s = db['students'].pop(sid, None)
    if s:
        # the stored path may come from another machine: resolve it under DATA_DIR
        folder = os.path.realpath(_resolve_student_folder(s))
        if os.path.dirname(folder) == os.path.realpath(DATA_DIR):
            shutil.rmtree(folder, ignore_errors=True)
    save_db(db)
    flash('Student deleted.','info')
    if s:   # back to the class the student was listed in
        return redirect(url_for('list_students', dept=s.get('dept_code'), batch=s.get('batch'), section=s.get('section')))
    return redirect(url_for('list_students'))

@app.route('/students/<sid>/images', methods=['GET','POST'])
@login_required
@role_required('super_admin','dept_admin')
def upload_student_images(sid):
    db = load_db()
    student = db['students'].get(sid)
    if not student:
        abort(404)
    folder = _resolve_student_folder(student)
    # Update DB with correct path for this machine
    if folder != student.get('images_folder'):
        db['students'][sid]['images_folder'] = folder
        save_db(db)
    os.makedirs(folder, exist_ok=True)
    if request.method == 'POST':
        files = request.files.getlist('images')
        saved, rejected = 0, []
        for f in files:
            if f and f.filename:
                ext = f.filename.rsplit('.',1)[-1].lower() if '.' in f.filename else ''
                head = f.stream.read(12)
                f.stream.seek(0)
                if ext in ALLOWED_IMG_EXT and not is_heif(head):
                    f.save(_next_photo_path(folder, ext))
                    saved += 1
                elif ext in CONVERT_IMG_EXT or ext in ALLOWED_IMG_EXT:   # HEIC, or HEIC named .jpg
                    dst = _next_photo_path(folder, 'jpg')
                    try:
                        convert_photo_to_jpeg(f.stream, dst)
                        saved += 1
                    except Exception:
                        if os.path.exists(dst):
                            os.remove(dst)
                        rejected.append(f.filename)
                else:
                    rejected.append(f.filename)
        # The face template is rebuilt right away, so a class session never waits for it.
        info = refresh_student_embedding(sid)
        msg = f'{saved} photo(s) uploaded.'
        if rejected:
            msg += f' Skipped (not a supported image): {", ".join(rejected[:5])}{" …" if len(rejected) > 5 else ""}.'
        if info and info.get('error'):
            flash(f'{msg} Face template not updated: {info["error"]}.', 'warning')
        elif info and info['photos'] and not info['faces']:
            flash(f'{msg} No face was found in any photo — upload clear, front-facing photos.', 'warning')
        elif info and info['photos']:
            flash(f'{msg} Face template saved from {info["faces"]} of {info["photos"]} photos.', 'success')
        else:
            flash(msg, 'success' if saved else 'warning')
        return redirect(url_for('upload_student_images', sid=sid))
    student = load_db()['students'].get(sid, student)
    return render_template('student_images.html', student=student, sid=sid,
                           images=_student_photos(folder), data_dir=DATA_DIR)

@app.route('/students/<sid>/images/delete/<fname>')
@login_required
@role_required('super_admin','dept_admin')
def delete_student_image(sid, fname):
    db = load_db()
    s = db['students'].get(sid)
    if not s:
        abort(404)
    folder2 = _resolve_student_folder(s)
    fpath = os.path.join(folder2, secure_filename(fname))
    if os.path.exists(fpath):
        os.remove(fpath)
        stem = os.path.basename(fpath)
        for t in glob.glob(os.path.join(folder2, '.thumbs', glob.escape(stem) + '.*.jpg')):
            os.remove(t)
        info = refresh_student_embedding(sid)
        if info and info['photos'] and not info.get('error'):
            flash(f'Image deleted. Face template updated from {info["faces"]} of {info["photos"]} photos.', 'info')
        else:
            flash('Image deleted.', 'info')
    return redirect(url_for('upload_student_images', sid=sid))

# ─────────────────────────────────────────────
#  Building / Camera management
# ─────────────────────────────────────────────
@app.route('/buildings')
@login_required
@role_required('super_admin','dept_admin')
def buildings():
    db = load_db()
    faculty = session.get('faculty')
    bldg = {fac:db['buildings'][fac] for fac in (
        [faculty] if faculty else db['buildings'].keys()
    ) if fac in db['buildings']}
    return render_template('buildings.html', buildings=bldg, bup=BUP_STRUCTURE, busy=_busy_rooms(faculty),
                           video_floors=VIDEO_TEST_FLOORS)

def minutes_text(seconds):
    """Time in tenths of a minute, counted up (0.1 min = 6 s): "0 min", "0.1 min" ... "1 min",
    "1.5 min", "4 min" — the same text the live video shows (bup::minutes_text)."""
    try:
        tenths = int(float(seconds or 0) / 6.0) if float(seconds or 0) > 0 else 0
    except (TypeError, ValueError):
        tenths = 0
    return f"{tenths // 10} min" if tenths % 10 == 0 else f"{tenths // 10}.{tenths % 10} min"

app.add_template_filter(minutes_text, 'mins')

@app.template_filter('camhost')
def camhost(url):
    """Camera address for display — never the credentials inside the RTSP URL."""
    t = camera_control.target_from_rtsp(url or '')
    return t.host or '—'

@app.route('/buildings/<faculty_code>/floor/<floor>/room/<room>', methods=['GET','POST'])
@login_required
@role_required('super_admin','dept_admin')
def edit_room(faculty_code, floor, room):
    db = load_db()
    room_data = db['buildings'].get(faculty_code,{}).get('floors',{}).get(floor,{}).get('rooms',{}).get(room)
    if not room_data:
        abort(404)
    busy = _busy_rooms().get(_room_key(faculty_code, floor, room))
    if request.method == 'POST':
        if busy:
            flash(f'Room {room} has a class running ({busy["teacher_name"]}, {busy["course_id"]}) — '
                  f'camera settings can be changed after it ends.', 'danger')
            return redirect(url_for('edit_room', faculty_code=faculty_code, floor=floor, room=room))
        settings = room_data.setdefault('camera_settings', {})
        video_room = _video_test_room(faculty_code, floor)
        videos = room_data.get('test_videos') or {}
        notes, problems = [], []
        for cam in ('camera1', 'camera2'):
            url = request.form.get(cam, '').strip()
            room_data[cam] = url
            stream = request.form.get(f'{cam}_stream', '')
            res = request.form.get(f'{cam}_res', '')
            fps = request.form.get(f'{cam}_fps', '')
            bitrate = request.form.get(f'{cam}_bitrate', '')
            gop = request.form.get(f'{cam}_gop', '')
            label = cam.replace('camera', 'Camera ')
            if video_room and cam in videos:
                # test rooms: the camera plays its recorded video, or its RTSP stream again
                videos[cam]['active'] = request.form.get(f'{cam}_source') == 'video'
                if videos[cam]['active']:
                    notes.append(f'{label}: recorded video {videos[cam].get("name", "")}')
                    continue   # the RTSP URL and its stream settings are kept for later, untouched
            if not url or stream not in ('main', 'sub'):
                settings.pop(cam, None)   # deployment default (STREAM_PROFILE)
                continue
            entry, msg, err = _apply_camera_stream(url, stream, res, fps, bitrate, gop)
            settings[cam] = entry
            if msg:
                notes.append(f'{label}: {msg}')
            if err:
                problems.append(f'{label}: {err}')
        save_db(db)
        if problems:
            flash('Saved, but: ' + ' · '.join(problems), 'warning')
        flash('Camera settings saved. ' + ' · '.join(notes), 'success')
        return redirect(url_for('buildings'))
    return render_template('edit_room.html', room=room_data, faculty_code=faculty_code,
                           floor=floor, room_num=room, bup=BUP_STRUCTURE, busy=busy,
                           settings=room_data.get('camera_settings') or {},
                           default_profile=STREAM_PROFILE, default_size=f"{FRAME_W}×{FRAME_H}",
                           video_test=_video_test_room(faculty_code, floor),
                           videos={c: _video_entry_public(v) for c, v in (room_data.get('test_videos') or {}).items()},
                           video_max_mb=TEST_VIDEO_MAX_MB, video_exts=sorted(TEST_VIDEO_EXTS))

def _parse_res(res):
    m = re.match(r'^\s*(\d{2,5})\s*[x×]\s*(\d{2,5})\s*$', res or '')
    return (int(m.group(1)), int(m.group(2))) if m else None

def _apply_camera_stream(url, stream, res, fps='', bitrate='', gop=''):
    """Read the camera, switch the chosen stream to the chosen resolution and frame rate on the
    camera itself, and return (settings entry, note, error). The entry records the stream URI the
    camera reports, so the engine reads exactly that stream."""
    want = _parse_res(res)
    want_fps = int(fps) if str(fps).strip().isdigit() and int(fps) > 0 else None
    want_br = int(bitrate) if str(bitrate).strip().isdigit() and 64 <= int(bitrate) <= 16384 else None
    want_gop = int(gop) if str(gop).strip().isdigit() and 1 <= int(gop) <= 400 else None
    entry = {"stream": stream, "checked": datetime.now().strftime('%Y-%m-%d %H:%M')}
    if want:
        entry["width"], entry["height"] = want
    info, result = camera_control.configure_stream(url, stream, *(want or (None, None)), fps=want_fps,
                                                   bitrate_kbps=want_br, gop=want_gop)
    if not info["ok"]:
        return entry, f'{stream} stream saved', f'camera not reachable, resolution not applied ({info["error"]})'
    s = next((x for x in info["streams"] if x["id"] == stream), None)
    if s is None:
        return entry, '', f'the camera has no {stream} stream'
    err, note = '', f'{stream} stream'
    if result["ok"] and not result.get("unchanged"):
        note += ' switched on the camera to'
    elif not result["ok"]:
        err = result["error"]
    note += f' {s["width"]}×{s["height"]} {s["codec"]} {round(s.get("fps") or 0)} fps'
    if s.get("bitrate_kbps"):
        note += f' {s["bitrate_kbps"]} kbps'
    if s.get("gop"):
        note += f' I-frame every {s["gop"]}'
    main = next((x for x in info["streams"] if x["id"] == "main"), None)
    entry.update({"width": s["width"], "height": s["height"], "codec": s["codec"], "fps": s["fps"],
                  "bitrate_kbps": s.get("bitrate_kbps") or None, "gop": s.get("gop") or None,
                  "uri": s.get("uri") or "", "protocol": info["protocol"],
                  "model": " ".join(x for x in (info["vendor"], info["model"]) if x),
                  "scene": [main["width"], main["height"]] if main else None})
    return entry, note, err

@app.route('/api/camera_info', methods=['POST'])
@login_required
@role_required('super_admin','dept_admin')
def camera_info():
    """The camera's streams and the resolutions it offers, read from the camera (ISAPI / ONVIF)."""
    url = (request.get_json(silent=True) or {}).get('url', '').strip()
    info = camera_control.public_info(camera_control.read_camera(url))
    if not info["ok"] and url:
        # no settings from the web port (often only the RTSP port is forwarded): does the stream
        # itself open with this URL's user and password?
        info["rtsp"] = camera_control.rtsp_check(url)
    return jsonify(info)

# ─────────────────────────────────────────────
#  Recorded test videos (VIDEO_TEST_FLOORS rooms only)
# ─────────────────────────────────────────────
def _video_test_room(faculty, floor):
    return f"{faculty}/{floor}".upper() in VIDEO_TEST_FLOORS

def _test_video_path(entry):
    """Absolute path of a room camera's uploaded video, or None when there is none on disk."""
    name = os.path.basename((entry or {}).get('file') or '')
    path = os.path.join(TEST_VIDEO_DIR, name) if name else None
    return path if path and os.path.isfile(path) else None

def _remove_test_video(entry):
    name = os.path.basename((entry or {}).get('file') or '')
    for p in ([os.path.join(TEST_VIDEO_DIR, name), os.path.join(TEST_VIDEO_DIR, name + '.jpg')] if name else []):
        try: os.remove(p)
        except OSError: pass

def _room_sources(faculty, floor, room_data):
    """[(cam_id, rtsp_url, video_path)] in room order. In a video-test room a camera set to its
    recorded video plays that file (video_path); every other camera reads its RTSP URL."""
    videos = (room_data.get('test_videos') or {}) if _video_test_room(faculty, floor) else {}
    out = []
    for cam_id, url in room_data.items():
        if not (cam_id.startswith('camera') and isinstance(url, str)):
            continue
        v = videos.get(cam_id) or {}
        path = _test_video_path(v) if v.get('active') else None
        if path or url.strip():
            out.append((cam_id, url.strip(), path))
    return out

def _video_room_or_404(faculty_code, floor, room, cam):
    if cam not in ('camera1', 'camera2') or not _video_test_room(faculty_code, floor):
        abort(404)
    db = load_db()
    room_data = db['buildings'].get(faculty_code, {}).get('floors', {}).get(floor, {}).get('rooms', {}).get(room)
    if room_data is None:
        abort(404)
    return db, room_data

def _upload_part(upload_id):
    return os.path.join(TEST_VIDEO_DIR, f".upload_{upload_id}.part")

_video_uploads = {}   # upload id -> {room, cam, name, size, user}: uploads in progress (one worker process)
_video_uploads_lock = threading.Lock()

def _video_entry_public(entry):
    if not entry:
        return None
    keep = ('name', 'size', 'width', 'height', 'fps', 'duration', 'codec', 'uploaded', 'active')
    out = {k: entry.get(k) for k in keep}
    out['thumb'] = url_for('test_video_thumb', name=entry['file']) if entry.get('file') else None
    out['missing'] = _test_video_path(entry) is None
    return out

@app.route('/buildings/<faculty_code>/floor/<floor>/room/<room>/video/<cam>/start', methods=['POST'])
@login_required
@role_required('super_admin','dept_admin')
def test_video_start(faculty_code, floor, room, cam):
    """Begin a (chunked) upload of a recorded video for one camera of a video-test room."""
    _video_room_or_404(faculty_code, floor, room, cam)
    if _busy_rooms().get(_room_key(faculty_code, floor, room)):
        return jsonify(ok=False, error='A class is running in this room — upload after it ends.'), 409
    data = request.get_json(silent=True) or {}
    name = secure_filename(str(data.get('name') or '')) or 'video.mp4'
    size = int(data.get('size') or 0)
    ext = os.path.splitext(name)[1].lower()
    if ext not in TEST_VIDEO_EXTS:
        return jsonify(ok=False, error=f'{ext or "This file"} is not a supported video type '
                                       f'({", ".join(sorted(TEST_VIDEO_EXTS))}).'), 400
    if size <= 0:
        return jsonify(ok=False, error='The file is empty.'), 400
    if size > TEST_VIDEO_MAX_MB * 1024 * 1024:
        return jsonify(ok=False, error=f'The video is {size / 2**20:.0f} MB — the limit is {TEST_VIDEO_MAX_MB} MB.'), 413
    # forget half-finished uploads older than six hours
    for old in glob.glob(os.path.join(TEST_VIDEO_DIR, '.upload_*.part')):
        try:
            if time.time() - os.path.getmtime(old) > 6 * 3600: os.remove(old)
        except OSError: pass
    free = shutil.disk_usage(TEST_VIDEO_DIR).free
    if free - size < TEST_VIDEO_MIN_FREE_MB * 1024 * 1024:
        return jsonify(ok=False, error=f'Not enough disk space on the server: {free / 2**30:.1f} GB free, the video '
                                       f'needs {size / 2**30:.2f} GB and {TEST_VIDEO_MIN_FREE_MB / 1024:.0f} GB must stay free.'), 507
    upload_id = hashlib.sha1(os.urandom(16)).hexdigest()[:20]
    open(_upload_part(upload_id), 'wb').close()
    with _video_uploads_lock:
        _video_uploads[upload_id] = {"room": _room_key(faculty_code, floor, room), "cam": cam,
                                     "name": name, "size": size, "user": session.get('username')}
    return jsonify(ok=True, upload_id=upload_id, chunk=8 * 1024 * 1024)

@app.route('/buildings/<faculty_code>/floor/<floor>/room/<room>/video/<cam>/chunk', methods=['POST'])
@login_required
@role_required('super_admin','dept_admin')
def test_video_chunk(faculty_code, floor, room, cam):
    """One piece of the upload (raw body) at ?offset=; the last piece installs the video."""
    upload_id = request.args.get('upload_id', '')
    with _video_uploads_lock:
        info = _video_uploads.get(upload_id)
    if (not info or info["room"] != _room_key(faculty_code, floor, room) or info["cam"] != cam
            or info["user"] != session.get('username')):
        return jsonify(ok=False, error='Unknown upload — start again.'), 404
    part = _upload_part(upload_id)
    if not os.path.isfile(part):
        return jsonify(ok=False, error='The upload expired — start again.'), 410
    offset = request.args.get('offset', type=int)
    have = os.path.getsize(part)
    if offset != have:   # a retried / out-of-order piece: tell the browser where to continue
        return jsonify(ok=False, resume=have, error='offset mismatch'), 409
    if (request.content_length or 0) > UPLOAD_CHUNK_MAX:
        return jsonify(ok=False, error='Piece too large.'), 413
    with open(part, 'ab') as f:
        while True:
            buf = request.stream.read(1024 * 1024)
            if not buf:
                break
            f.write(buf)
    have = os.path.getsize(part)
    if have > info["size"]:
        os.remove(part)
        with _video_uploads_lock:
            _video_uploads.pop(upload_id, None)
        return jsonify(ok=False, error='More data than announced — start again.'), 400
    if have < info["size"]:
        return jsonify(ok=True, received=have)

    # complete: check that GStreamer can play it, then install it for this camera
    with _video_uploads_lock:
        _video_uploads.pop(upload_id, None)
    ext = os.path.splitext(info["name"])[1].lower()
    fname = secure_filename(f"{faculty_code}_{floor}_{room}_{cam}_{datetime.now().strftime('%Y%m%d-%H%M%S')}{ext}")
    final = os.path.join(TEST_VIDEO_DIR, fname)
    os.replace(part, final)
    probe = bup_engine.probe_video(final)
    if not probe["ok"]:
        os.remove(final)
        return jsonify(ok=False, error=f'This file cannot be played: {probe["error"]}'), 400
    if probe.get("thumbnail"):
        with open(final + '.jpg', 'wb') as f:
            f.write(probe["thumbnail"])
    db, room_data = _video_room_or_404(faculty_code, floor, room, cam)
    videos = room_data.setdefault('test_videos', {})
    old = videos.get(cam)
    videos[cam] = {"file": fname, "name": info["name"], "size": have, "width": probe["width"],
                   "height": probe["height"], "fps": round(probe["fps"], 2), "duration": round(probe["duration"], 1),
                   "codec": probe["codec"], "uploaded": datetime.now().strftime('%Y-%m-%d %H:%M'), "active": True}
    save_db(db)
    if old and old.get('file') != fname:
        _remove_test_video(old)
    return jsonify(ok=True, video=_video_entry_public(videos[cam]))

@app.route('/buildings/<faculty_code>/floor/<floor>/room/<room>/video/<cam>/delete', methods=['POST'])
@login_required
@role_required('super_admin','dept_admin')
def test_video_delete(faculty_code, floor, room, cam):
    db, room_data = _video_room_or_404(faculty_code, floor, room, cam)
    if _busy_rooms().get(_room_key(faculty_code, floor, room)):
        return jsonify(ok=False, error='A class is running in this room — remove the video after it ends.'), 409
    old = (room_data.get('test_videos') or {}).pop(cam, None)
    save_db(db)
    _remove_test_video(old)
    return jsonify(ok=True)

@app.route('/test_videos/thumb/<name>')
@login_required
@role_required('super_admin','dept_admin')
def test_video_thumb(name):
    path = os.path.join(TEST_VIDEO_DIR, os.path.basename(name) + '.jpg')
    if not os.path.isfile(path):
        abort(404)
    return send_file(path, mimetype='image/jpeg', max_age=3600)

# ─────────────────────────────────────────────
#  Teacher – Start Attendance Session
# ─────────────────────────────────────────────
@app.route('/take_attendance')
@login_required
@role_required('teacher')
def take_attendance():
    if _claim_admin_ended():
        return redirect(url_for('confirm_attendance'))
    db   = load_db()
    uname = session['username']
    teacher = db['teachers'].get(uname, {})
    my_courses = {cid: db['courses'][cid] for cid in teacher.get('courses',[]) if cid in db['courses']}
    running = _session_of(uname)
    # one card per class the teacher can start: each section of the course's batch on its own,
    # and all of them together (a joint class) when the batch has more than one
    classes = []
    for cid, c in sorted(my_courses.items(), key=lambda kv: (str(kv[1].get('batch', '')), kv[0])):
        sections = _batch_sections(db, c.get('dept_code'), c.get('batch'))
        if not sections:
            classes.append({"cid": cid, "course": c, "section": "", "label": "", "students": 0})
            continue
        for x, n in sections.items():
            classes.append({"cid": cid, "course": c, "section": x, "badge": x, "label": f"Section {x}", "students": n})
        if len(sections) > 1:
            classes.append({"cid": cid, "course": c, "section": "ALL", "joint": True, "badge": "+".join(sections),
                            "label": "Sections " + " + ".join(sections), "students": sum(sections.values())})
    return render_template('take_attendance.html', classes=classes, running=running)

@app.route('/take_attendance/select_location', methods=['POST'])
@login_required
@role_required('teacher')
def select_location():
    course_id = request.form.get('course_id')
    db = load_db()
    if course_id not in db['courses']:
        flash('Invalid course.','danger')
        return redirect(url_for('take_attendance'))
    mine = _session_of(session['username'])
    if mine:
        flash(f'You already have a class running in Room {mine["room"]} — end it before starting another.', 'warning')
        return redirect(url_for('live_session'))
    session['att_course'] = course_id
    faculty = db['courses'][course_id]['faculty_code']
    bldg_data = db['buildings'].get(faculty, {})
    busy = _busy_rooms(faculty)
    floors_info = {}
    for floor_str, fdata in bldg_data.get('floors',{}).items():
        floor_int = int(floor_str)
        dcode, dname = get_dept_for_floor(faculty, floor_int)
        # room numbers and camera counts only: camera URLs carry passwords and stay server-side
        floors_info[floor_str] = {'rooms': {r: {'cams': len(_room_sources(faculty, floor_str, rd))} for r, rd in fdata['rooms'].items()},
                                  'dept_code': dcode, 'dept_name': dname,
                                  'busy': {r: {k: b[k] for k in ("teacher_name", "course_id", "course_name", "section", "since")}
                                           for r in fdata['rooms']
                                           for b in [busy.get(_room_key(faculty, floor_str, r))] if b}}
    course = db['courses'][course_id]
    sections = _batch_sections(db, course.get('dept_code'), course.get('batch'))
    # the class card chosen on Take Attendance names its section (or ALL for a joint class);
    # without one (an old bookmark): one section needs no choice, else the course's old section
    chosen = request.form.get('section', '').strip().upper()
    if chosen and chosen != 'ALL' and chosen not in sections:
        flash(f'Batch {course.get("batch")} has no section {chosen}.', 'danger')
        return redirect(url_for('take_attendance'))
    default = chosen or (next(iter(sections)) if len(sections) == 1 else (course.get('section') if course.get('section') in sections else ''))
    return render_template('select_location.html', course_id=course_id,
                           course=course, sections=sections, default_section=default, fixed_section=bool(chosen),
                           batch_total=sum(1 for st in db['students'].values() if _in_class(st, course)),
                           faculty_code=faculty, floors_info=floors_info,
                           bup=BUP_STRUCTURE)

@app.route('/api/room_status/<faculty>')
@login_required
def room_status(faculty):
    """{floor: {room: who occupies it}} for one building — the room picker refreshes its locks."""
    out = {}
    for key, b in _busy_rooms(faculty).items():
        _, floor, room = key.split('/', 2)
        out.setdefault(floor, {})[room] = {k: b[k] for k in ("teacher_name", "course_id", "course_name", "section", "since")}
    return jsonify(out)

def _room_cameras(room_data):
    """[(cam_id, url)] of the room's configured cameras, in room order."""
    return [(k, v.strip()) for k, v in room_data.items()
            if k.startswith('camera') and isinstance(v, str) and v.strip()]

def _unreachable_cameras(cameras, timeout=4.0):
    """{cam_id: host} of the cameras whose RTSP port does not answer (all probed at once)."""
    targets = {cam_id: camera_control.target_from_rtsp(url) for cam_id, url in cameras}
    found = {}

    def probe(cam_id, t):
        found[cam_id] = bool(t.host) and camera_control.probe_ports(t.host, [t.rtsp_port], timeout)[t.rtsp_port] is not None
    threads = [threading.Thread(target=probe, args=item, daemon=True) for item in targets.items()]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout + 1)
    return {cam_id: t.host or '?' for cam_id, t in targets.items() if not found.get(cam_id)}

def _camera_spec(cam_id, url, cfg):
    """Engine settings for one room camera. When a stream was chosen in Buildings & Cameras,
    the engine reads exactly that stream (the URI the camera reported for it) and decodes and
    runs inference at that stream's own resolution. Without a choice the deployment-wide
    STREAM_PROFILE / FRAME_W x FRAME_H rules apply, as before."""
    if not cfg or cfg.get('stream') not in ('main', 'sub'):
        return {"id": cam_id, "url": url}
    spec = {"id": cam_id, "url": url, "profile": cfg['stream'], "max_w": 0, "max_h": 0}
    if cfg.get('uri'):
        spec["url"] = camera_control.stream_url(url, cfg['uri'])
        spec["fixed_url"] = True
    if cfg.get('codec'):
        spec["codec"] = cfg['codec']   # H264 / H265 as the camera reported: no wrong-codec session on connect
    scene = cfg.get('scene')
    if cfg['stream'] == 'sub' and scene and scene[0] and scene[1]:
        spec["aspect"] = f"{scene[0]}:{scene[1]}"   # sub-streams often squash the main stream's scene
    return spec

def _launch_engine(sess, room_data, sources, embeds, meta, tracker):
    """Start a class's C++ engine: per camera GStreamer capture (VA-API decode) -> SCRFD on NPU ->
    ByteTrack -> ArcFace on NPU per track -> match -> tracker -> overlay; threads live in bup_engine."""
    sess["db_embeds"], sess["db_meta"] = embeds, meta
    sess["tracker"] = tracker
    settings = room_data.get('camera_settings') or {}
    sess["test_video"] = any(video for _, _, video in sources)
    eng = bup_engine.AttendanceSession(
        get_face_engine(), tracker, embeds, meta,
        [{"id": cam_id, "url": "", "file": video} if video else _camera_spec(cam_id, url, settings.get(cam_id))
         for cam_id, url, video in sources],
        stream_profile=STREAM_PROFILE, rtsp_transport=RTSP_TRANSPORT,
        substream_aspect=SUBSTREAM_ASPECT, sim_threshold=SIM_THRESHOLD,
        frame_w=FRAME_W, frame_h=FRAME_H, mode_label=GPU_MODE, backend_label=STREAM_BACKEND,
        jpeg_quality=LIVE_JPEG_QUALITY, hw_decode=HW_DECODE, gpu_max_width=GPU_MAX_WIDTH,
        smooth_ms=LIVE_SMOOTH_MS, track_lost_s=TRACK_LOST_S, reverify_s=TRACK_REVERIFY_S,
        max_embed_per_frame=MAX_EMBED_PER_FRAME)
    sess["engine"]  = eng
    sess["cameras"] = {cam_id: eng.camera(cam_id) for cam_id in eng.camera_ids()}
    eng.start()
    sess["active"], sess["starting"] = True, False
    _save_live_state()

@app.route('/take_attendance/start', methods=['POST'])
@login_required
@role_required('teacher')
def start_attendance():
    teacher    = session['username']
    course_id  = request.form.get('course_id') or session.get('att_course')
    faculty    = request.form.get('faculty_code')
    floor      = request.form.get('floor')
    room       = request.form.get('room')
    db         = load_db()
    course     = db['courses'].get(course_id)
    if not course:
        flash('Course not found.','danger')
        return redirect(url_for('take_attendance'))
    room_data  = db['buildings'].get(faculty,{}).get('floors',{}).get(floor,{}).get('rooms',{}).get(room)
    if not room_data:
        flash('Room not found in database.','danger')
        return redirect(url_for('take_attendance'))
    sources = _room_sources(faculty, floor, room_data)   # RTSP cameras, or recorded videos in test rooms
    cameras = [(cam_id, url) for cam_id, url, video in sources if not video]
    if not sources:
        flash(f'Room {room} has no camera configured — ask an administrator.','danger')
        return redirect(url_for('take_attendance'))
    sections = _batch_sections(db, course.get('dept_code'), course.get('batch'))
    section = request.form.get('section', '').strip().upper()
    if section == 'ALL':
        section = None
    elif not section:   # not chosen (old clients): the only section, else the course's old one, else all
        section = next(iter(sections)) if len(sections) == 1 else (course.get('section') if course.get('section') in sections else None)
    elif section not in sections:
        flash(f'Batch {course.get("batch")} has no section {section}.', 'danger')
        return redirect(url_for('take_attendance'))

    mine = _session_of(teacher)
    if mine:
        flash(f'You already have a class running in Room {mine["room"]} — end it before starting another.', 'warning')
        return redirect(url_for('live_session'))
    offline = _unreachable_cameras(cameras)   # a few seconds at most, before the room is locked
    if len(offline) == len(sources):
        flash(f'Room {room}: the attendance server cannot reach {"either camera" if len(cameras) > 1 else "the camera"} '
              f'({", ".join(offline.values())}) — no answer on the RTSP port. Check the cameras\' network '
              f'(the server must be on the same network or have a route to it), or choose another room.', 'danger')
        return redirect(url_for('take_attendance'))

    key = _room_key(faculty, floor, room)
    with _sessions_lock:   # check and lock the room in one step: two teachers can never both get it
        mine = next((x for x in _sessions.values() if x["teacher"] == teacher), None)
        if mine:
            flash(f'You already have a class running in Room {mine["room"]} — end it before starting another.', 'warning')
            return redirect(url_for('live_session'))
        other = _sessions.get(key)
        if other:
            flash(f'Room {room} is in use: {other["teacher_name"]} is taking {other["course_id"]} there '
                  f'(since {datetime.fromtimestamp(other["started"]).strftime("%H:%M")}). Choose another room.', 'danger')
            return redirect(url_for('take_attendance'))
        sess = _new_session(key, teacher, session.get('name', teacher), course_id, faculty, floor, room, section)
        _sessions[key] = sess

    try:
        embeds, meta = build_embeddings_for_class(course_id, section)
        if embeds is None or not meta:
            _stop_session(sess)
            flash('No student face data found for this course. Please upload student photos first.','warning')
            return redirect(url_for('take_attendance'))
        _launch_engine(sess, room_data, sources, embeds, meta, AttendanceTracker(meta))
    except Exception as e:
        _stop_session(sess)
        flash(f'Could not start the session: {e}', 'danger')
        return redirect(url_for('take_attendance'))
    if sess["shutdown"]:   # an admin ended it while it was starting
        _stop_session(sess)
        return redirect(url_for('take_attendance'))
    for cam_id, host in offline.items():
        flash(f'{cam_id.replace("camera", "Camera ")} ({host}) is not reachable from the server — the class runs with '
              f'the other camera; it joins automatically if it comes back.', 'warning')
    return redirect(url_for('live_session'))

# docker stop / gunicorn shutdown: one last snapshot (the classes resume in the next process),
# then release the cameras and NPU cleanly
import atexit

def _on_shutdown():
    _save_live_state()
    _live_state["shutting_down"] = True
    _stop_session()

atexit.register(_on_shutdown)

# A session an admin ended: its results wait here until the teacher next opens a page,
# then go through the normal review-and-save step (confirm_attendance).
_admin_ended = {}   # teacher username -> pending summary + notice

def _claim_admin_ended():
    pending = _admin_ended.pop(session.get('username'), None)
    if not pending:
        return False
    _save_live_state()
    for k in ('summary', 'course', 'faculty', 'floor', 'room', 'section', 'test_video'):
        session[f'pending_{k}'] = pending.get(k)
    flash(pending['notice'], 'warning')
    return True

def _my_live_session():
    sess = _session_of(session.get('username'))
    return sess if sess and sess["active"] else None

@app.route('/live_session')
@login_required
@role_required('teacher')
def live_session():
    sess = _my_live_session()
    if not sess:
        if _claim_admin_ended():
            return redirect(url_for('confirm_attendance'))
        return redirect(url_for('take_attendance'))
    db = load_db()
    course = db['courses'].get(sess["course_id"],{})
    return render_template('live_session.html', section=_section_label(sess.get("section")),
                           course=course, course_id=sess["course_id"],
                           floor=sess["floor"], room=sess["room"],
                           faculty=sess["faculty"], test_video=sess.get("test_video", False),
                           resumed_at=datetime.fromtimestamp(sess["resumed_at"]).strftime('%H:%M') if sess.get("resumed_at") else None)

@app.route('/video_feed')
@login_required
def video_feed():
    sess = _my_live_session()
    return Response(_gen_feed(sess) if sess else iter(()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_stats')
@login_required
def live_stats():
    sess = _my_live_session()
    if not sess or not sess["tracker"]:
        return jsonify({})
    return jsonify(sess["tracker"].live_stats())

@app.route('/live_list')
@login_required
def live_list():
    sess = _my_live_session()
    if not sess or not sess["tracker"]:
        return jsonify([])
    return jsonify(sess["tracker"].summary())

@app.route('/manual_mark', methods=['POST'])
@login_required
@role_required('teacher')
def manual_mark():
    data = request.get_json()
    sid    = data.get('student_id')
    status = data.get('status')  # "present" or "absent"
    sess = _my_live_session()
    if sess and sess["tracker"] and sid and status in ('present','absent'):
        sess["tracker"].set_manual(sid, status)
        return jsonify({'ok': True})
    return jsonify({'ok': False}), 400

@app.route('/end_session', methods=['POST'])
@login_required
@role_required('teacher')
def end_session():
    sess = _my_live_session()
    if not sess:
        return redirect(url_for('take_attendance'))
    summary = sess["tracker"].summary() if sess["tracker"] else []
    session['pending_summary'] = summary
    session['pending_course']  = sess["course_id"]
    session['pending_faculty'] = sess["faculty"]
    session['pending_floor']   = sess["floor"]
    session['pending_room']    = sess["room"]
    session['pending_section'] = sess.get("section")
    session['pending_test_video'] = sess.get("test_video", False)
    _stop_session(sess)   # the room is free again
    return redirect(url_for('confirm_attendance'))

@app.route('/sessions/end', methods=['POST'])
@login_required
@role_required('super_admin','dept_admin')
def admin_end_session():
    """End a running class from the admin side (e.g. a teacher left without ending it). Its
    attendance is not lost: the teacher reviews and saves it on their next page."""
    key = request.form.get('room_key', '')
    with _sessions_lock:
        sess = _sessions.get(key)
    if sess and (session.get('faculty') in (None, sess["faculty"])):
        summary = sess["tracker"].summary() if sess["tracker"] else []
        if summary:
            _admin_ended[sess["teacher"]] = {
                "summary": summary, "course": sess["course_id"], "faculty": sess["faculty"],
                "floor": sess["floor"], "room": sess["room"], "section": sess.get("section"),
                "test_video": sess.get("test_video", False),
                "notice": f'Your class in Room {sess["room"]} was ended by {session.get("name", "an administrator")}. '
                          f'Review the attendance below and save it.'}
        _stop_session(sess)
        flash(f'Class in Room {sess["room"]} ({sess["teacher_name"]}, {sess["course_id"]}) ended. '
              f'The room is free again.', 'info')
    else:
        flash('That class has already ended.', 'info')
    return redirect(request.referrer or url_for('buildings'))

@app.route('/confirm_attendance', methods=['GET','POST'])
@login_required
@role_required('teacher')
def confirm_attendance():
    summary  = session.get('pending_summary', [])
    course_id= session.get('pending_course')
    if not summary or not course_id:
        return redirect(url_for('take_attendance'))
    db = load_db()
    course = db['courses'].get(course_id,{})

    if request.method == 'POST':
        # Apply any last-minute overrides from the form
        for row in summary:
            sid = row['student_id']
            val = request.form.get(f'status_{sid}')
            if val in ('Present','Absent'):
                row['present'] = (val == 'Present')
                row['status']  = val
        # Save
        _save_attendance_record(summary, course_id, course, db, session.get('pending_section'),
                                test_video=session.get('pending_test_video', False))
        session.pop('pending_summary', None)
        session.pop('pending_course', None)
        session.pop('pending_section', None)
        session.pop('pending_test_video', None)
        flash('Attendance saved successfully!','success')
        return redirect(url_for('attendance_history'))

    return render_template('confirm_attendance.html', summary=summary, course=course, course_id=course_id,
                           min_present=MIN_PRESENT_SEC,
                           section=_section_label(session.get('pending_section')),
                           test_video=session.get('pending_test_video', False))

def _save_attendance_record(summary, course_id, course, db, section=None, test_video=False):
    now   = datetime.now()
    ts    = now.strftime('%Y%m%d_%H%M%S')
    fname = f"attendance_{course_id}_{ts}"
    present = sum(1 for r in summary if r['present'])
    record = {
        "filename": fname,
        "course_id": course_id,
        "course_name": course.get('name',''),
        "dept_code": course.get('dept_code',''),
        "faculty_code": course.get('faculty_code',''),
        "batch": course.get('batch',''),
        # the section this class was held for ("A", "B"); "A+B" for a joint class of the batch
        "section": section or '+'.join(_batch_sections(db, course.get('dept_code'), course.get('batch'))) or course.get('section', ''),
        "teacher": session['username'],
        "teacher_name": session['name'],
        "date": now.strftime('%Y-%m-%d'),
        "time": now.strftime('%H:%M:%S'),
        "total": len(summary),
        "present": present,
        "absent": len(summary) - present,
        "rate": round(present/len(summary)*100 if summary else 0, 1),
        "gpu_mode": GPU_MODE,
        "records": summary
    }
    if test_video:   # run on an uploaded recording, not the room's live cameras
        record["source"] = "Recorded video (test)"
    jpath = os.path.join(HISTORY_DIR, fname+'.json')
    with open(jpath,'w',encoding='utf-8') as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    bpath = os.path.join(BACKUP_DIR, fname+'.json')
    shutil.copy2(jpath, bpath)
    # Also save CSV
    if PANDAS_AVAILABLE:
        import pandas as pd
        pd.DataFrame(summary).to_csv(os.path.join(HISTORY_DIR, fname+'.csv'), index=False)

# ─────────────────────────────────────────────
#  Attendance History
# ─────────────────────────────────────────────
def _may_see_record(rec):
    """A teacher sees their own classes, a department admin their department's, the super admin all."""
    role = session.get('role')
    if role == 'super_admin':
        return True
    if role == 'dept_admin':
        return rec.get('dept_code') == session.get('dept')
    return rec.get('teacher') == session.get('username')

def _record_path(fname):
    safe = secure_filename(fname)
    if not safe.endswith('.json'):
        safe += '.json'
    return safe, os.path.join(HISTORY_DIR, safe)

def _load_record(fname):
    """(safe file name, path, normalized record) of a record the user may open; 404 / 403 otherwise."""
    safe, fpath = _record_path(fname)
    if not os.path.exists(fpath):
        abort(404)
    with open(fpath) as f:
        record = _normalize_record(json.load(f), safe)
    if not _may_see_record(record):
        abort(403)
    return safe, fpath, record

@app.route('/history')
@login_required
def attendance_history():
    dept  = session.get('dept')
    role  = session['role']
    uname = session['username']
    records = []
    files = sorted(glob.glob(os.path.join(HISTORY_DIR,"*.json")), reverse=True)
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
            if role == 'teacher' and d.get('teacher') != uname:
                continue
            if role == 'dept_admin' and d.get('dept_code') != dept:
                continue
            records.append(_normalize_record(d, os.path.basename(f)))
        except Exception:
            pass
    records = [r for r in records if isinstance(r, dict)]
    for r in records:
        r.setdefault('course_id', '—')
    batch_counts = {}
    for r in records:
        if r.get('batch'):
            batch_counts[str(r['batch'])] = batch_counts.get(str(r['batch']), 0) + 1
    return render_template('history.html', records=records,
                           batches=sorted(batch_counts.items(), reverse=True),
                           can_delete=session['role'] == 'dept_admin')

@app.route('/history/<fname>')
@login_required
def view_record(fname):
    _, _, record = _load_record(fname)
    return render_template('view_record.html', record=record, can_delete=session['role'] == 'dept_admin')

@app.route('/history/<fname>/edit', methods=['GET','POST'])
@login_required
def edit_record(fname):
    safe, fpath, record = _load_record(fname)
    if request.method == 'POST':
        for row in record['records']:
            sid = row['student_id']
            val = request.form.get(f'status_{sid}')
            if val in ('Present','Absent'):
                row['present'] = (val=='Present')
                row['status']  = val
        record['present'] = sum(1 for r in record['records'] if r['present'])
        record['absent']  = len(record['records']) - record['present']
        record['rate']    = round(record['present']/len(record['records'])*100 if record['records'] else 0,1)
        with open(fpath,'w') as f:
            json.dump(record, f, indent=2)
        flash('Attendance record updated.','success')
        return redirect(url_for('view_record', fname=fname.replace('.json','')))
    return render_template('edit_record.html', record=record, fname=fname)

@app.route('/history/<fname>/download')
@login_required
def download_record(fname):
    safe, fpath, record = _load_record(fname)
    csv_path = os.path.join(HISTORY_DIR, safe.replace('.json','')+'.csv')
    if os.path.exists(csv_path):
        return send_file(csv_path, as_attachment=True)
    # Generate on-the-fly
    import io
    lines = ['student_id,name,status,total_time,detection_count']
    for r in record.get('records',[]):
        lines.append(f"{r['student_id']},{r['name']},{r['status']},{r.get('total_time',0)},{r.get('detection_count',0)}")
    buf = io.BytesIO('\n'.join(lines).encode())
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=safe.replace('.json','')+'.csv',
                     mimetype='text/csv')

@app.route('/history/<fname>/delete', methods=['POST'])
@login_required
@role_required('dept_admin')
def delete_record(fname):
    """Department admins only, for their own department. The record leaves the history but is
    kept in attendance_history/.deleted/ (with who deleted it and when) — a mistake can be undone
    by moving the files back."""
    safe, fpath, record = _load_record(fname)
    trash = os.path.join(HISTORY_DIR, '.deleted')
    os.makedirs(trash, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    record['deleted_by'] = session['username']
    record['deleted_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(os.path.join(trash, f"{safe[:-5]}__deleted_{stamp}.json"), 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    os.remove(fpath)
    csv_path = fpath[:-5] + '.csv'
    if os.path.exists(csv_path):
        shutil.move(csv_path, os.path.join(trash, f"{safe[:-5]}__deleted_{stamp}.csv"))
    flash(f'Attendance record deleted: {record.get("course_id", "")} on {record.get("date", "")} '
          f'{record.get("time", "")} (Batch {record.get("batch", "—")} · Section {record.get("section", "—")}).', 'info')
    nxt = request.form.get('next', '')
    return redirect(nxt if nxt.startswith('/') and not nxt.startswith('//') else url_for('attendance_history'))

# ─────────────────────────────────────────────
#  Profile / Password change
# ─────────────────────────────────────────────
def _my_account(db):
    role, uname = session['role'], session['username']
    if role == 'super_admin':
        return db['super_admin']
    return db['dept_admins' if role == 'dept_admin' else 'teachers'].get(uname)

@app.route('/profile', methods=['GET','POST'])
@login_required
def profile():
    db = load_db()
    role  = session['role']
    uname = session['username']
    account = _my_account(db) or {}
    session['photo'] = account.get('photo')   # sessions from before photos existed
    action = request.form.get('action', 'password')
    if request.method == 'POST' and action in ('photo', 'photo_remove'):
        old = account.get('photo')
        if action == 'photo':
            f = request.files.get('photo')
            if not f or not f.filename:
                flash('Choose a photo first.', 'warning')
                return redirect(url_for('profile'))
            try:
                account['photo'] = save_avatar(f, role, uname)
            except ValueError as e:
                flash(f'Could not use that photo: {e}.', 'danger')
                return redirect(url_for('profile'))
        else:
            account.pop('photo', None)
        save_db(db)
        if old and old != account.get('photo'):
            remove_avatar_file(old)
        session['photo'] = account.get('photo')
        flash('Profile photo updated.' if action == 'photo' else 'Profile photo removed.', 'success')
        return redirect(url_for('profile'))
    if request.method == 'POST':
        old_pw = request.form.get('old_password','')
        new_pw = request.form.get('new_password','')
        if not new_pw:
            flash('New password cannot be empty.','danger')
            return redirect(url_for('profile'))
        if role == 'super_admin':
            if db['super_admin']['password'] != _hash(old_pw):
                flash('Current password incorrect.','danger')
                return redirect(url_for('profile'))
            db['super_admin']['password'] = _hash(new_pw)
        elif role == 'dept_admin':
            if db['dept_admins'][uname]['password'] != _hash(old_pw):
                flash('Current password incorrect.','danger')
                return redirect(url_for('profile'))
            db['dept_admins'][uname]['password'] = _hash(new_pw)
        elif role == 'teacher':
            if db['teachers'][uname]['password'] != _hash(old_pw):
                flash('Current password incorrect.','danger')
                return redirect(url_for('profile'))
            db['teachers'][uname]['password'] = _hash(new_pw)
        save_db(db)
        flash('Password changed.','success')
    return render_template('profile.html', account=account)

@app.route('/avatar/<fname>')
@login_required
def avatar(fname):
    return send_from_directory(AVATAR_DIR, secure_filename(fname), max_age=86400)

# ─────────────────────────────────────────────
#  Static image serve for student photos
# ─────────────────────────────────────────────
@app.route('/student_img/<path:rel_path>')
@login_required
def student_img(rel_path):
    # rel_path is <student_folder>/<file> under DATA_DIR; safe_join refuses "../" escapes
    full = safe_join(DATA_DIR, rel_path)
    if not full or not os.path.isfile(full):
        # Fallback: search all student folders
        full = None
        fname = os.path.basename(rel_path)
        for root, dirs, files in os.walk(DATA_DIR):
            if fname in files:
                full = os.path.join(root, fname)
                break
        if not full:
            abort(404)
    width = request.args.get('w', type=int)
    if width:   # ?w=240: cached thumbnail (phone photos are up to 30+ MP)
        try:
            return send_file(_thumbnail(full, max(64, min(width, 1024))), mimetype='image/jpeg', max_age=86400)
        except Exception:
            pass   # not decodable by Pillow: fall back to the original file
    return send_file(full)

def _thumbnail(path, width):
    """JPEG thumbnail of a student photo in <photo folder>/.thumbs, regenerated when the photo changes."""
    from PIL import Image, ImageOps
    folder, name = os.path.split(path)
    thumb = os.path.join(folder, '.thumbs', f"{name}.{width}.jpg")
    if os.path.exists(thumb) and os.path.getmtime(thumb) >= os.path.getmtime(path):
        return thumb
    os.makedirs(os.path.dirname(thumb), exist_ok=True)
    with Image.open(path) as im:
        im.draft('RGB', (width * 2, width * 2))   # JPEG: decode at reduced scale, much faster
        im = ImageOps.exif_transpose(im).convert('RGB')
        im.thumbnail((width, width * 2), Image.LANCZOS)
        tmp = f"{thumb}.{os.getpid()}.tmp"
        im.save(tmp, 'JPEG', quality=85)
    os.replace(tmp, thumb)
    return thumb

# ─────────────────────────────────────────────
#  API – teacher course allocation (AJAX)
# ─────────────────────────────────────────────
@app.route('/api/allocate_course', methods=['POST'])
@login_required
@role_required('super_admin','dept_admin')
def allocate_course():
    db = load_db()
    data = request.get_json()
    course_id = data.get('course_id')
    teacher_uname = data.get('teacher_username')
    if course_id not in db['courses']:
        return jsonify({'ok':False,'msg':'Course not found'}),404
    if teacher_uname not in db['teachers']:
        return jsonify({'ok':False,'msg':'Teacher not found'}),404
    old_teacher = db['courses'][course_id].get('teacher_username')
    if old_teacher and old_teacher in db['teachers']:
        try: db['teachers'][old_teacher]['courses'].remove(course_id)
        except: pass
    db['courses'][course_id]['teacher_username'] = teacher_uname
    if course_id not in db['teachers'][teacher_uname].get('courses',[]):
        db['teachers'][teacher_uname].setdefault('courses',[]).append(course_id)
    save_db(db)
    return jsonify({'ok':True})

# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
#  Student Attendance Summary
# ─────────────────────────────────────────────
@app.route('/students/<sid>/summary')
@login_required
@role_required('super_admin','dept_admin')
def student_summary(sid):
    db = load_db()
    student = db['students'].get(sid)
    if not student:
        abort(404)
    # Find all attendance records for this student
    records = []
    files = sorted(glob.glob(os.path.join(HISTORY_DIR,"*.json")), reverse=True)
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
            if d.get('dept_code') != student.get('dept_code'):
                continue
            for r in d.get('records', []):
                if r['student_id'] == sid:
                    records.append({
                        'date': d['date'],
                        'time': d['time'],
                        'course_id': d['course_id'],
                        'course_name': d['course_name'],
                        'status': r['status'],
                        'present': r['present'],
                        'total_time': r.get('total_time', 0)
                    })
        except Exception:
            pass
    total_classes = len(records)
    present_classes = sum(1 for r in records if r['present'])
    attendance_rate = round(present_classes / total_classes * 100 if total_classes else 0, 1)
    return render_template('student_summary.html',
                           student=student, sid=sid,
                           records=records,
                           total_classes=total_classes,
                           present_classes=present_classes,
                           attendance_rate=attendance_rate)


# ─────────────────────────────────────────────
#  Error Handlers
# ─────────────────────────────────────────────
@app.errorhandler(403)
def forbidden(e):
    return render_template('error.html', code=403, message="Access Denied",
                           detail="You don't have permission to access this page."), 403

@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', code=404, message="Page Not Found",
                           detail="The page you're looking for doesn't exist."), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', code=500, message="Internal Server Error",
                           detail=str(e)), 500


# ─────────────────────────────────────────────
#  Camera/Stream Info API
# ─────────────────────────────────────────────
@app.route('/api/stream_info')
@login_required
def stream_info():
    """Return current stream backend info (for debugging)."""
    info = {
        "preferred_backend": STREAM_BACKEND,
        "gstreamer_available": HAS_GSTREAMER,
        "gpu_mode": GPU_MODE,
        "cameras": {}
    }
    if session.get('role') == 'super_admin':
        n, limit = bup_engine.open_files()
        info["open_files"], info["open_files_limit"] = n, limit
    mine = _my_live_session()
    if mine:
        for cam_id, cap in mine.get("cameras", {}).items():
            info["cameras"][cam_id] = cap.get_info()
    if session.get('role') in ('super_admin', 'dept_admin'):
        with _sessions_lock:
            running = list(_sessions.values())
        info["sessions"] = {x["key"]: {"teacher": x["teacher"], "course_id": x["course_id"], "active": x["active"],
                                       "cameras": {c: cap.get_info() for c, cap in x.get("cameras", {}).items()}}
                            for x in running if session.get('faculty') in (None, x["faculty"])}
    return jsonify(info)

# ─────────────────────────────────────────────
#  Live Monitor (super admin): every running class's live picture on one page
# ─────────────────────────────────────────────
def _monitor_entry(x, db):
    """What the monitor shows of one running class (no camera credentials)."""
    stats = {}
    if x.get("tracker") is not None:
        try:
            stats = x["tracker"].live_stats()
        except Exception:
            stats = {}
    cams = []
    for cam_id, cap in sorted(x.get("cameras", {}).items()):
        try:
            i = cap.get_info()
        except Exception:
            continue
        src = (i.get("url") or "").split("@")[-1].split("/")[0] or i.get("backend", "")
        if x.get("test_video") and not i.get("url"):
            src = "recorded video"
        cams.append({"id": cam_id, "source": src, "connected": bool(i.get("has_frame")),
                     "feed_fps": i.get("feed_fps") or 0, "infer_fps": i.get("infer_fps") or 0,
                     "size": i.get("stream_size") or i.get("frame_size") or "", "reconnects": i.get("reconnects") or 0,
                     "gpu": i.get("decoder") == "gpu"})
    bldg = BUP_STRUCTURE.get(x["faculty"], {}).get("building", x["faculty"])
    course = db["courses"].get(x["course_id"], {})
    return {"key": x["key"], "building": bldg, "faculty": x["faculty"], "floor": x["floor"], "room": x["room"],
            "course_id": x["course_id"], "course_name": course.get("name", ""), "batch": course.get("batch", ""),
            "section": _section_label(x.get("section")), "teacher_name": x["teacher_name"],
            "since": datetime.fromtimestamp(x["started"]).strftime('%H:%M'), "starting": bool(x.get("starting")),
            "active": bool(x.get("active")), "test_video": bool(x.get("test_video")),
            "present": stats.get("present", 0), "total": stats.get("total", 0), "rate": stats.get("rate", 0),
            "duration": stats.get("duration", 0), "cameras": cams}

def _monitor_entries():
    db = load_db()
    with _sessions_lock:
        running = list(_sessions.values())
    out = [_monitor_entry(x, db) for x in running]
    out.sort(key=lambda e: (e["faculty"], int(e["floor"]) if str(e["floor"]).isdigit() else 0, str(e["room"])))
    return out

@app.route('/live_monitor')
@login_required
@role_required('super_admin')
def live_monitor():
    return render_template('live_monitor.html', classes=_monitor_entries())

@app.route('/live_monitor/status')
@login_required
@role_required('super_admin')
def live_monitor_status():
    return jsonify({"classes": _monitor_entries(), "time": datetime.now().strftime('%H:%M:%S')})

@app.route('/live_monitor/frame/<path:key>')
@login_required
@role_required('super_admin')
def live_monitor_frame(key):
    """The newest picture of one class — the same JPEG its teacher sees (composed and encoded once,
    shared). One short request per picture: a page with many classes never holds connections
    open (browsers allow 6 per server, the app has 16 worker threads)."""
    with _sessions_lock:
        x = _sessions.get(key)
    eng = x.get("engine") if x and x.get("active") else None
    if eng is None:
        abort(404)
    resp = Response(eng.mjpeg_frame(), mimetype='image/jpeg')
    resp.headers['Cache-Control'] = 'no-store'
    return resp

@app.route('/api/system_usage')
@login_required
@role_required('super_admin')
def system_usage():
    """Live NPU-core and CPU load for the super-admin sidebar (sampled every second in C++)."""
    return jsonify(bup_engine.system_usage())

# keep the running classes on disk, and resume the ones a restart interrupted — in the web app only:
# a command-line tool that imports this module (tools/import_students.py) sets BUP_TOOL=1, so it never
# resumes (and so doubles) the classes the running app is already serving
if os.environ.get("BUP_TOOL") != "1":
    threading.Thread(target=_live_state_main, name="live-classes", daemon=True).start()

def _warm_face_templates():
    """A new recognition model (or detector, or pipeline version) makes every student's cached face
    template stale. Rebuild them in the background, one student at a time, so that the first class
    of each course does not wait a minute for its students' photos. A class that starts meanwhile
    builds its own students (same lock, same cache files) and the warm-up skips them."""
    time.sleep(30)   # after the resumed classes are running again
    try:
        students = list(load_db()['students'].items())
    except Exception:
        return
    stale = []
    for sid, s in students:
        folder = _resolve_student_folder(s)
        photos = _student_photos(folder)
        if photos and _load_student_embedding(os.path.join(folder, STUDENT_EMB_FILE), _photos_fingerprint(photos)) is None:
            stale.append(sid)
    if not stale:
        return
    print(f"  🔄 rebuilding the face templates of {len(stale)} students for "
          f"{os.path.basename(REC_MODEL_FILE)} (background)", flush=True)
    t0, done = time.time(), 0
    for sid in stale:
        try:
            refresh_student_embedding(sid)
            done += 1
        except Exception as e:   # one bad folder must not stop the rest
            print(f"  ⚠️  face template of {sid}: {e}", flush=True)
        time.sleep(0.2)          # live classes share the NPU
    print(f"  ✅ face templates rebuilt: {done}/{len(stale)} students in {time.time() - t0:.0f}s", flush=True)

if os.environ.get("WARM_FACE_TEMPLATES") == "1":   # the deployment sets it (docker-compose.yml)
    threading.Thread(target=_warm_face_templates, name="face-templates", daemon=True).start()

if __name__ == '__main__':
    # Ensure DB exists
    if not os.path.exists(DB_FILE):
        save_db(_default_db())
        print("✅ Fresh database created.")
        print("   Default Super Admin → username: superadmin | password: bup@2024")
    # debug stays OFF unless FLASK_DEBUG=1: the Werkzeug debugger on 0.0.0.0 is remote code
    # execution, and its reloader would load the NPU engine twice. Production: ./run.sh (gunicorn).
    app.run(debug=FLASK_DEBUG, use_reloader=False, host=os.environ.get("BUP_HOST", "0.0.0.0"),
            port=int(os.environ.get("BUP_PORT", "5000")), threaded=True)