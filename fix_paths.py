#!/usr/bin/env python3
"""
fix_paths.py
Run this once on your machine to fix all student image folder paths in the database.
This is needed because the paths were generated on a different machine.

Usage:
    python3 fix_paths.py
"""
import os, json, glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_FILE  = os.path.join(BASE_DIR, "bup_db.json")

def fix():
    if not os.path.exists(DB_FILE):
        print("❌ bup_db.json not found. Run python3 app.py first.")
        return

    with open(DB_FILE) as f:
        db = json.load(f)

    fixed = 0
    missing = 0

    for sid, s in db['students'].items():
        name = s.get('name', '')
        safe_name = name.replace(' ', '_')
        stored = s.get('images_folder', '')

        # Check if stored path works
        if stored and os.path.isdir(stored):
            continue  # already fine

        # Try basename in DATA_DIR
        found = None
        if stored:
            basename = os.path.basename(stored)
            candidate = os.path.join(DATA_DIR, basename)
            if os.path.isdir(candidate):
                found = candidate

        # Try sid_name pattern
        if not found:
            candidate2 = os.path.join(DATA_DIR, f"{sid}_{safe_name}")
            if os.path.isdir(candidate2):
                found = candidate2

        # Try scanning DATA_DIR for any folder starting with sid_
        if not found:
            for d in os.listdir(DATA_DIR):
                if d.startswith(f"{sid}_") and os.path.isdir(os.path.join(DATA_DIR, d)):
                    found = os.path.join(DATA_DIR, d)
                    break

        if found:
            db['students'][sid]['images_folder'] = found
            imgs = []
            for ext in ['jpg','jpeg','png','bmp','webp']:
                imgs += glob.glob(os.path.join(found, f'*.{ext}'))
            db['students'][sid]['image_count'] = len(imgs)
            fixed += 1
            print(f"  ✅ Fixed: {sid} {name} → {found} ({len(imgs)} photos)")
        else:
            # Create empty folder
            new_folder = os.path.join(DATA_DIR, f"{sid}_{safe_name}")
            os.makedirs(new_folder, exist_ok=True)
            db['students'][sid]['images_folder'] = new_folder
            db['students'][sid]['image_count'] = 0
            missing += 1
            print(f"  📁 Created empty folder: {sid} {name} → {new_folder}")

    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! Fixed {fixed} paths, created {missing} new folders.")
    print("   Now restart the app: python3 app.py")

if __name__ == '__main__':
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"Fixing student image folder paths...\n")
    fix()
