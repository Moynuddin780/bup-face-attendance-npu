#!/usr/bin/env python3
"""Aligned 112x112 face crops of the enrolled student photos, made exactly as the engine makes them.

The largest face of every photo is found by the app's detector (bup_engine.FaceEngine) and warped
to the ArcFace 5-point template with the same Umeyama similarity as cpp/src/face_align.cpp. The
crops serve two purposes:
  - calibration images for DX-COM (INT8 quantization of a recognition model)
  - one identical input set for comparing recognition models (embed.py / report.py)

  .venv/bin/python tools/model_convert/crops.py --out /path/to/crops

Writes <out>/<student folder>/<n>.png (BGR) and <out>/index.npz {labels, paths, files}.
"""
import argparse
import glob
import os
import sys

import numpy as np

APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, APP_DIR)

ARCFACE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)
IMG_EXT = ("jpg", "jpeg", "png", "bmp", "webp")


def umeyama(src, dst):
    """2x3 similarity transform src -> dst (least squares), as bup::umeyama_similarity."""
    n = len(src)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s, d = src - mu_s, dst - mu_d
    a = d.T @ s / n
    var_s = (s ** 2).sum() / n
    u, w, vt = np.linalg.svd(a)
    dd = np.eye(2)
    if np.linalg.det(a) < 0:
        dd[1, 1] = -1.0
    r = u @ dd @ vt
    scale = (w[0] * dd[0, 0] + w[1] * dd[1, 1]) / var_s if var_s > 0 else 1.0
    t = mu_d - scale * (r @ mu_s)
    return np.hstack([scale * r, t[:, None]])


def align(img, kps, cv2):
    m = umeyama(np.asarray(kps, dtype=np.float64).reshape(5, 2), ARCFACE_112)
    return cv2.warpAffine(img, m, (112, 112), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))


def student_photos(data_dir):
    for folder in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if not os.path.isdir(folder):
            continue
        paths = []
        for ext in IMG_EXT:
            paths += glob.glob(os.path.join(folder, f"*.{ext}")) + glob.glob(os.path.join(folder, f"*.{ext.upper()}"))
        for p in sorted(set(paths)):
            yield os.path.basename(folder), p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.path.join(APP_DIR, "data"))
    ap.add_argument("--det", default=os.path.join(APP_DIR, "models", "scrfd-10g_640x640.dxnn"))
    ap.add_argument("--rec", default=os.path.join(APP_DIR, "models", "arcface_iresnet100_112x112_ms1m.dxnn"),
                    help="any recognition model (the engine needs one; only detection is used)")
    args = ap.parse_args()

    import cv2
    import bup_engine
    fe = bup_engine.FaceEngine(args.det, args.rec)
    labels, paths, files = [], [], []
    seen = 0
    for label, p in student_photos(args.data):
        seen += 1
        img = bup_engine.imread(p)
        if img is None:
            continue
        faces = fe.detect(img)
        if not faces:
            continue
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        crop = align(img, f.kps, cv2)
        os.makedirs(os.path.join(args.out, label), exist_ok=True)
        rel = os.path.join(label, f"{len(files)}.png")
        cv2.imwrite(os.path.join(args.out, rel), crop)
        labels.append(label)
        paths.append(os.path.relpath(p, args.data))
        files.append(rel)
    np.savez(os.path.join(args.out, "index.npz"), labels=np.array(labels), paths=np.array(paths),
             files=np.array(files))
    print(f"{seen} photos, {len(files)} face crops, {len(set(labels))} students -> {args.out}")


if __name__ == "__main__":
    main()
