#!/usr/bin/env python3
"""Bulk-import students of one section from a folder of per-student photo folders.

    .venv/bin/python tools/import_students.py "<section folder>" --batch 2026 --section A [--dept ICT]
    .venv/bin/python tools/import_students.py --refresh-all          # (re)build every student's face template

Each sub-folder of <section folder> is one student, named "<student id>_<name>" ("<id> <name>",
"<id>-<name>" and "<name>-<id>" are accepted too; a trailing "sectionB" is dropped).

  * Photos are hard-linked into data/<id>_<name>/ as img1.jpg, img2.jpg, ... — no extra disk space,
    and the source folder stays untouched (copied when a link is impossible). HEIC/HEIF photos are
    converted to JPEG (also when named .jpg); other files (PDF, LaTeX, ...) are skipped. Re-running is safe: photos already
    imported (same content) are not added twice.
  * A student already enrolled in another section of the same department is enrolled in this
    section too (extra_sections) instead of being duplicated.
  * Every imported student's face template (data/<folder>/embedding.npz) is computed on the NPU right
    away, exactly as when photos are uploaded in the web app.

Writes the live DB (db/bup_db.json, or $BUP_DB_FILE); a backup is written next to it first. The
running app (host or Docker) reads the DB on every request, so no restart is needed.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.environ.get("BUP_DB_FILE") and os.path.isfile(os.path.join(APP_DIR, "db", "bup_db.json")):
    os.environ["BUP_DB_FILE"] = os.path.join(APP_DIR, "db", "bup_db.json")
os.environ.setdefault("BUP_SECRET_KEY", "import-tool")   # the tool serves no pages
os.environ["BUP_TOOL"] = "1"   # no live-class resume in this process: the running app serves the classes
sys.path.insert(0, APP_DIR)
os.chdir(APP_DIR)
import app as bup  # noqa: E402  (the web app's own DB, folder and NPU helpers)

MANIFEST = ".imported.json"   # source-photo sha1 -> stored file name (keeps re-runs idempotent)
ID_FIRST = re.compile(r"^\s*(\d{6,})\s*[-_\s]*\s*(.*?)\s*$")
ID_LAST = re.compile(r"^\s*(.*?)\s*[-_\s]+\s*(\d{6,})\s*$")


def parse_folder_name(folder):
    """'26549012004_ Sifat Ahmed' -> ('26549012004', 'Sifat Ahmed'); None when there is no ID."""
    m = ID_FIRST.match(folder)
    if m and m.group(2):
        sid, name = m.groups()
    else:
        m = ID_LAST.match(folder)
        if not m:
            return None
        name, sid = m.groups()
    name = re.sub(r"\s*section\s*[a-z]\s*$", "", name, flags=re.I)      # "Dip Alen sectionB"
    name = re.sub(r"\s+", " ", name.replace("_", " ")).strip(" -_.")
    if name.isupper():                                                  # "FARDINA TABASSUM"
        name = name.title()
    name = " ".join(w.capitalize() if w.islower() and len(w) > 1 else w for w in name.split(" "))
    return sid, name


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def place_photos(src_dir, folder, dry_run):
    """Link/convert the photos of src_dir into folder. Returns counters."""
    stats = {"linked": 0, "copied": 0, "converted": 0, "duplicate": 0, "skipped": [], "failed": []}
    manifest_path = os.path.join(folder, MANIFEST)
    manifest = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    known = set(manifest)
    if not dry_run:
        for p in bup._student_photos(folder):   # photos uploaded through the web app count too
            known.add(sha1_file(p))
    for name in sorted(os.listdir(src_dir)):
        src = os.path.join(src_dir, name)
        if not os.path.isfile(src):
            continue
        ext = name.rsplit(".", 1)[1].lower() if "." in name else ""
        if ext not in bup.ALLOWED_IMG_EXT and ext not in bup.CONVERT_IMG_EXT:
            stats["skipped"].append(name)
            continue
        digest = sha1_file(src)
        if digest in known:
            stats["duplicate"] += 1
            continue
        known.add(digest)
        with open(src, "rb") as f:
            heif = bup.is_heif(f.read(12))   # iPhone photos are often HEIC saved as .jpg
        if dry_run:
            stats["converted" if heif or ext in bup.CONVERT_IMG_EXT else "linked"] += 1
            continue
        if ext in bup.ALLOWED_IMG_EXT and not heif:
            dst = bup._next_photo_path(folder, ext)
            try:
                os.link(src, dst)
                stats["linked"] += 1
            except OSError:   # another filesystem, or not permitted: copy instead
                shutil.copy2(src, dst)
                stats["copied"] += 1
        else:
            dst = bup._next_photo_path(folder, "jpg")
            try:
                bup.convert_photo_to_jpeg(src, dst)
                stats["converted"] += 1
            except Exception as e:
                if os.path.exists(dst):
                    os.remove(dst)
                stats["failed"].append(f"{name} ({e})")
                continue
        manifest[digest] = os.path.basename(dst)
    if not dry_run and manifest:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)
    return stats


def import_section(src_root, batch, section, dept, faculty, dry_run, embed):
    rows, problems = [], []
    for entry in sorted(os.listdir(src_root)):
        src_dir = os.path.join(src_root, entry)
        if not os.path.isdir(src_dir):
            continue
        parsed = parse_folder_name(entry)
        if not parsed:
            problems.append(f"{entry!r}: no student ID in the folder name — skipped")
            continue
        sid, name = parsed
        db = bup.load_db()
        existing = db["students"].get(sid)
        if existing and existing.get("dept_code") != dept:
            problems.append(f"{sid} {name}: already enrolled in {existing.get('dept_code')} — skipped")
            continue
        if existing:
            record = existing
            if (batch, section) in bup._student_sections(existing):
                action = "updated"
            else:
                action = f"+ section {batch}/{section}"
        else:
            record = {
                "student_id": sid, "name": name, "faculty_code": faculty, "dept_code": dept,
                "batch": batch, "section": section,
                "images_folder": os.path.join(bup.DATA_DIR, f"{sid}_{name.replace(' ', '_')}"),
                "image_count": 0,
            }
            action = "added"
        folder = bup._resolve_student_folder(record) if not dry_run else record.get("images_folder", "")
        stats = place_photos(src_dir, folder, dry_run)

        info = {"photos": len(bup._student_photos(folder)) if not dry_run else 0, "faces": 0}
        if not dry_run and embed:
            _, info = bup.student_embedding(dict(record, images_folder=folder))
        if not dry_run:
            db = bup.load_db()   # re-read: other writers (the web app) may have saved meanwhile
            rec = db["students"].get(sid) or record
            if action.startswith("+"):
                rec.setdefault("extra_sections", []).append({"batch": batch, "section": section})
            rec["images_folder"] = folder
            rec["image_count"] = len(bup._student_photos(folder))
            if embed:
                rec["embedding"] = bup._embedding_summary(info)
            db["students"][sid] = rec
            bup.save_db(db)

        new = stats["linked"] + stats["copied"] + stats["converted"]
        note = []
        if stats["converted"]:
            note.append(f"{stats['converted']} HEIC→JPG")
        if stats["duplicate"]:
            note.append(f"{stats['duplicate']} already there")
        if stats["skipped"]:
            note.append(f"skipped {len(stats['skipped'])} non-image")
        if stats["failed"]:
            problems.append(f"{sid} {name}: could not convert {', '.join(stats['failed'])}")
        face = "" if dry_run or not embed else (
            f"template {info['faces']}/{info['photos']}" if info.get("faces") else
            ("NO FACE FOUND" if info.get("photos") else "no photos"))
        rows.append((sid, record.get("name", name), action, new, face, ", ".join(note)))
        print(f"  {sid:<12} {record.get('name', name)[:30]:<30} {action:<16} +{new:<3} {face:<16} {', '.join(note)}",
              flush=True)
        if not dry_run and embed and info.get("photos") and not info.get("faces"):
            problems.append(f"{sid} {name}: no face found in any photo")
        if not dry_run and not info.get("photos"):
            problems.append(f"{sid} {name}: no photos")
    return rows, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?", help="section folder: one sub-folder per student")
    ap.add_argument("--batch", help="batch (year), e.g. 2026")
    ap.add_argument("--section", help="section letter, e.g. A")
    ap.add_argument("--dept", default="ICT", help="department code (default ICT)")
    ap.add_argument("--faculty", help="faculty code (default: the department's faculty)")
    ap.add_argument("--dry-run", action="store_true", help="show what would happen, change nothing")
    ap.add_argument("--no-embed", action="store_true", help="skip the NPU face templates (built at the next session)")
    ap.add_argument("--refresh-all", action="store_true", help="build missing/outdated face templates of every student")
    args = ap.parse_args()

    if not args.dry_run:
        backup = f"{bup.DB_FILE}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(bup.DB_FILE, backup)
        print(f"DB: {bup.DB_FILE}  (backup: {os.path.basename(backup)})")

    if args.refresh_all:
        t0 = time.time()
        sids = list(bup.load_db()["students"])
        for i, sid in enumerate(sids, 1):
            info = bup.refresh_student_embedding(sid)
            if info:
                print(f"  [{i}/{len(sids)}] {sid:<12} template {info['faces']}/{info['photos']}"
                      f"{' (cached)' if info.get('cached') else ''}", flush=True)
        print(f"done in {time.time() - t0:.0f}s")
        return 0

    if not (args.src and args.batch and args.section):
        ap.error("src, --batch and --section are required (or use --refresh-all)")
    dept = args.dept.upper()
    faculty = args.faculty or next((f for f, info in bup.BUP_STRUCTURE.items() if dept in info["departments"]), None)
    if not faculty:
        ap.error(f"unknown department {dept}")
    section = args.section.strip().upper()
    print(f"Importing {args.src} -> {faculty}/{dept} batch {args.batch} section {section}"
          f"{' (dry run)' if args.dry_run else ''}")
    t0 = time.time()
    rows, problems = import_section(args.src, args.batch.strip(), section, dept, faculty,
                                    args.dry_run, not args.no_embed)
    print(f"{len(rows)} students in {time.time() - t0:.0f}s")
    for p in problems:
        print(f"  ! {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
