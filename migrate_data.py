#!/usr/bin/env python3
"""
migrate_data.py
Run this once to import your existing student folders + attendance history
from the old project into the new BUP Attendance system.

Usage:
  python migrate_data.py --old_project "path/to/Face Attendance WEB Implementation_4"
"""

import os, sys, json, shutil, glob, argparse, hashlib

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
HIST_DIR  = os.path.join(BASE_DIR, "attendance_history")
DB_FILE   = os.path.join(BASE_DIR, "bup_db.json")

def _hash(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE) as f:
            return json.load(f)
    print("❌ bup_db.json not found. Run the app once first (python app.py) then re-run migration.")
    sys.exit(1)

def save_db(db):
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

def migrate(old_path):
    db = load_db()
    print(f"📂 Migrating from: {old_path}")

    # ── 1. Migrate student image folders ──────────────────────────
    old_data = os.path.join(old_path, "data")
    if os.path.isdir(old_data):
        for class_folder in os.listdir(old_data):
            class_path = os.path.join(old_data, class_folder)
            if not os.path.isdir(class_path):
                continue
            print(f"\n  📁 Class folder: {class_folder}")
            # Parse class folder name e.g. BICE21_Section_A_OODP
            parts = class_folder.split('_')
            for student_folder in os.listdir(class_path):
                spath = os.path.join(class_path, student_folder)
                if not os.path.isdir(spath):
                    continue
                # Parse: "002_Tamim" → id=002, name=Tamim
                fp = student_folder.split('_', 1)
                if len(fp) < 2:
                    continue
                raw_id, raw_name = fp[0], fp[1]
                student_id = raw_id.lstrip('0') or raw_id
                name = raw_name.replace('_', ' ')

                # Destination folder
                dest = os.path.join(DATA_DIR, f"{student_id}_{raw_name}")
                os.makedirs(dest, exist_ok=True)

                # Copy images
                imgs = glob.glob(os.path.join(spath, "*.jpg")) + \
                       glob.glob(os.path.join(spath, "*.jpeg")) + \
                       glob.glob(os.path.join(spath, "*.png"))
                for i, img in enumerate(imgs, 1):
                    ext = img.rsplit('.', 1)[-1]
                    dst = os.path.join(dest, f"img{i}.{ext}")
                    if not os.path.exists(dst):
                        shutil.copy2(img, dst)

                # Add to DB if not exists
                if student_id not in db['students']:
                    db['students'][student_id] = {
                        'student_id': student_id,
                        'name': name,
                        'faculty_code': 'FST',
                        'dept_code': 'ICT',
                        'batch': '2021',
                        'section': 'A',
                        'images_folder': dest,
                        'image_count': len(imgs)
                    }
                    print(f"    ✅ Imported student: {name} (ID: {student_id}) – {len(imgs)} photos")
                else:
                    # Update folder path and count
                    db['students'][student_id]['images_folder'] = dest
                    db['students'][student_id]['image_count'] = len(imgs)
                    print(f"    🔄 Updated: {name} – {len(imgs)} photos")
    else:
        print("  ⚠️  No 'data' folder found in old project.")

    # ── 2. Copy attendance history ─────────────────────────────────
    old_hist = os.path.join(old_path, "attendance_history")
    if os.path.isdir(old_hist):
        copied = 0
        for f in glob.glob(os.path.join(old_hist, "*.json")):
            dst = os.path.join(HIST_DIR, os.path.basename(f))
            if not os.path.exists(dst):
                shutil.copy2(f, dst)
                copied += 1
        for f in glob.glob(os.path.join(old_hist, "*.csv")):
            dst = os.path.join(HIST_DIR, os.path.basename(f))
            if not os.path.exists(dst):
                shutil.copy2(f, dst)
                copied += 1
        print(f"\n  📊 Copied {copied} attendance history files.")

    # ── 3. Copy .npz embeddings ────────────────────────────────────
    for npz in glob.glob(os.path.join(old_path, "*.npz")):
        dst = os.path.join(DATA_DIR, os.path.basename(npz))
        if not os.path.exists(dst):
            shutil.copy2(npz, dst)
            print(f"  🧠 Copied embeddings: {os.path.basename(npz)}")

    save_db(db)
    print(f"\n✅ Migration complete! {len(db['students'])} students in database.")
    print("   → Now run: python app.py")
    print("   → Login at http://localhost:5000 with superadmin / bup@2024")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_project', required=True, help='Path to old project folder')
    args = parser.parse_args()
    migrate(args.old_project)
