#!/usr/bin/env python3
"""Embed every enrolled student photo with one pipeline and save the result.

  --pipeline cpp       the new C++ engine (bup_engine.FaceEngine)
  --pipeline original  the ORIGINAL app.py code (_nms + DeepXFacePipeline), extracted
                       verbatim with `ast` and run on dx_engine + cv2 — the baseline

Both follow the app's own enrollment rule (compute_embedding): largest face per
photo, its `normed_embedding`. Metrics are computed by recog_metrics.py.
Run each pipeline in its own process (the baseline needs pip-OpenCV + dx_engine,
the engine links system OpenCV + libdxrt).
"""
import argparse
import ast
import glob
import os
import sys
import time

import numpy as np

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOWED_IMG_EXT = {"jpg", "jpeg", "png", "bmp", "webp"}
DET = os.path.join(APP_DIR, "models", "scrfd-10g_640x640.dxnn")
REC = os.path.join(APP_DIR, "models", "arcface_iresnet100_112x112_ms1m.dxnn")


def student_images(data_dir):
    for folder in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if not os.path.isdir(folder):
            continue
        paths = []
        for ext in ALLOWED_IMG_EXT:
            paths += glob.glob(os.path.join(folder, f"*.{ext}"))
        for p in sorted(paths):
            yield os.path.basename(folder), p


def load_original_pipeline(orig_app, npu_bound):
    """Exec _nms, ARC_FACE_SRC and DeepXFacePipeline from the original app.py."""
    import cv2
    import dx_engine

    src = open(orig_app, encoding="utf-8").read()
    tree = ast.parse(src)
    keep = [n for n in tree.body
            if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in ("_nms", "DeepXFacePipeline"))
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "ARC_FACE_SRC" for t in n.targets))]

    class _DxShim:
        """The original always opens NPU_ALL. On a shared NPU (another process pins
        cores) dxrtd refuses that, so only the core binding is injected here —
        the pipeline code itself runs unmodified."""
        def __getattr__(self, name):
            return getattr(dx_engine, name)

        @staticmethod
        def InferenceEngine(path):  # noqa: N802 - mirrors dx_engine API
            last = None
            for b in ([npu_bound] if npu_bound != "auto" else ["NPU_ALL", "NPU_0", "NPU_1", "NPU_2"]):
                opt = dx_engine.InferenceOption()
                opt.bound_option = getattr(dx_engine.InferenceOption.BOUND_OPTION, b)
                try:
                    return dx_engine.InferenceEngine(path, opt)
                except Exception as e:  # noqa: BLE001 - try the next binding
                    last = e
            raise last

    ns = {"np": np, "cv2": cv2, "dx": _DxShim(), "time": time, "os": os}
    exec(compile(ast.Module(body=keep, type_ignores=[]), orig_app, "exec"), ns)
    return ns["DeepXFacePipeline"](DET, REC), cv2.imread


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", choices=["cpp", "original"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.path.join(APP_DIR, "data"))
    ap.add_argument("--orig-app", default=os.environ.get(
        "BUP_ORIG_APP", os.path.join(APP_DIR, "baseline", "BUP_Attendance_NPU", "app.py")))
    ap.add_argument("--npu-bound", default=os.environ.get("NPU_BOUND", "auto"))
    ap.add_argument("--rec", default=REC, help="recognition model (.dxnn) for --pipeline cpp")
    args = ap.parse_args()

    if args.pipeline == "cpp":
        sys.path.insert(0, APP_DIR)
        import bup_engine
        fa = bup_engine.FaceEngine(DET, args.rec, npu_bound=args.npu_bound)
        imread = bup_engine.imread
    else:
        fa, imread = load_original_pipeline(args.orig_app, args.npu_bound)

    labels, paths, embs, scores, n_faces = [], [], [], [], []
    t0 = time.time()
    for label, p in student_images(args.data):
        img = imread(p)
        faces = fa.get(img) if img is not None else []
        labels.append(label)
        paths.append(os.path.relpath(p, args.data))
        n_faces.append(len(faces))
        if faces:
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            embs.append(np.asarray(face.normed_embedding, dtype=np.float32).ravel())
            scores.append(float(face.det_score))
        else:
            embs.append(np.full(512, np.nan, dtype=np.float32))
            scores.append(float("nan"))
    elapsed = time.time() - t0
    np.savez(args.out, embs=np.vstack(embs), labels=np.array(labels), paths=np.array(paths),
             det_scores=np.array(scores), n_faces=np.array(n_faces), seconds=elapsed)
    found = sum(1 for n in n_faces if n > 0)
    print(f"{args.pipeline}: {len(paths)} photos, face found in {found}, {elapsed:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
