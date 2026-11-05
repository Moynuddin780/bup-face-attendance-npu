#!/usr/bin/env python3
"""Embed the aligned crops of crops.py with one recognition model.

  --dxnn MODEL.dxnn   on the NPU (dx_engine; the app's .venv). Input: RGB uint8 112x112, as the app feeds it.
  --onnx MODEL.onnx   in FP32 on the CPU (onnxruntime; the DX-COM venv). Input: RGB, (x - 127.5) / 127.5,
                      the InsightFace convention — the reference a compiled model is measured against.

Writes an npz {embs, labels, paths, files, seconds} that recog_metrics.py and report.py read.
"""
import argparse
import os
import time

import numpy as np


def l2(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1e-9, n)


def dxnn_runner(path):
    import dx_engine
    ie = dx_engine.InferenceEngine(path)

    def run(rgb):
        out = ie.run(np.ascontiguousarray(rgb[None]))
        return np.asarray(out[0] if isinstance(out, list) else out, dtype=np.float32).ravel()
    return run


def onnx_runner(path):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    def run(rgb):
        x = ((rgb.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        return np.asarray(sess.run(None, {name: x})[0], dtype=np.float32).ravel()
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", required=True)
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dxnn")
    g.add_argument("--onnx")
    args = ap.parse_args()

    import cv2
    run = dxnn_runner(args.dxnn) if args.dxnn else onnx_runner(args.onnx)
    idx = np.load(os.path.join(args.crops, "index.npz"))
    embs = []
    t0 = time.time()
    for rel in idx["files"]:
        bgr = cv2.imread(os.path.join(args.crops, str(rel)))
        embs.append(run(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    secs = time.time() - t0
    np.savez(args.out, embs=l2(np.vstack(embs)), labels=idx["labels"], paths=idx["paths"], files=idx["files"],
             seconds=secs)
    print(f"{len(embs)} crops, {secs / max(1, len(embs)) * 1000:.1f} ms/crop -> {args.out}")


if __name__ == "__main__":
    main()
