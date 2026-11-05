#!/usr/bin/env python3
"""Build a synthetic two-camera classroom from enrolled students' photos.

For each camera it writes
  <out>/<cam>_main.mp4  1920x1080 @20 fps H.264 (4 Mbps)  - like a 2 MP camera main stream
  <out>/<cam>_sub.mp4    704x576  @20 fps H.264 (512 kbps) - like a Dahua D1 sub-stream:
                         the SAME 16:9 scene stored anamorphically (horizontally squashed)
and <out>/streams.json describing who appears where (the e2e test's ground truth).

Faces are cropped from the enrolled photos with the NPU face detector, pasted at
classroom-like sizes and drift slowly so the encoder produces real motion.
Needs the dev venv (opencv-python-headless) plus bup_engine.
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_DIR)
import bup_engine  # noqa: E402

W, H, FPS, SECONDS = 1920, 1080, 20, 20
SUB_W, SUB_H = 704, 576

# course ICE4204 (Cloud Computing, ICT 2020-A). 2054901002/2054901010 are the same
# person enrolled twice, so they are left out to keep the ground truth unambiguous.
CAMERAS = {
    "camera1": ["2054901001", "2054901003", "2054901004", "2054901005", "52"],
    "camera2": ["2054901004", "2054901005", "2054901006", "2054901011"],
}
NOT_ENROLLED = {"52"}  # Rakin, ICT 2021-A: must show up as "Unknown"


def face_tile(engine, folder, face_px):
    """Square head crop from the first photo with a face, scaled so the face box is face_px wide."""
    for name in sorted(os.listdir(folder)):
        img = bup_engine.imread(os.path.join(folder, name))
        if img is None:
            continue
        faces = engine.get(img)
        if not faces:
            continue
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        x1, y1, x2, y2 = f.bbox
        fw = x2 - x1
        side = int(fw * 1.9)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 + 0.05 * fw
        sx, sy = int(cx - side / 2), int(cy - side / 2)
        pad = side
        big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        crop = big[sy + pad:sy + pad + side, sx + pad:sx + pad + side]
        scale = face_px / fw
        return cv2.resize(crop, (int(side * scale), int(side * scale)), interpolation=cv2.INTER_AREA)
    raise RuntimeError(f"no face found in {folder}")


def render(tiles, t):
    """Classroom-ish background + tiles in a row, each drifting slowly."""
    frame = np.empty((H, W, 3), np.uint8)
    grad = np.linspace(70, 140, H, dtype=np.uint8)[:, None]
    frame[:] = np.dstack([grad + 10, grad + 5, grad])  # warm wall
    cv2.rectangle(frame, (0, int(H * 0.78)), (W, H), (50, 60, 75), -1)  # desks
    n = len(tiles)
    for i, tile in enumerate(tiles):
        th, tw = tile.shape[:2]
        cx = int((i + 0.5) * W / n + 40 * np.sin(0.6 * t + i))
        cy = int(H * 0.45 + 25 * np.cos(0.4 * t + 2 * i))
        x, y = max(0, min(W - tw, cx - tw // 2)), max(0, min(H - th, cy - th // 2))
        frame[y:y + th, x:x + tw] = tile
    return frame


def encode(frames_iter, path, size, bitrate):
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-vf", f"scale={size[0]}:{size[1]}",
           "-c:v", "libx264", "-preset", "veryfast", "-profile:v", "main", "-bf", "0",
           "-g", str(2 * FPS), "-b:v", bitrate, "-maxrate", bitrate, "-bufsize", bitrate,
           "-pix_fmt", "yuv420p", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames_iter:
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(APP_DIR, "tests", "media"))
    ap.add_argument("--data", default=os.path.join(APP_DIR, "data"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    engine = bup_engine.FaceEngine(os.path.join(APP_DIR, "models", "scrfd-10g_640x640.dxnn"),
                                   os.path.join(APP_DIR, "models", "arcface_iresnet100_112x112_ms1m.dxnn"))
    folders = {d.split("_", 1)[0]: os.path.join(args.data, d) for d in os.listdir(args.data)
               if os.path.isdir(os.path.join(args.data, d))}
    # front row ~190 px faces, back row ~130 px on the 1080p main stream
    sizes = [190, 160, 170, 140, 130]
    meta = {"fps": FPS, "seconds": SECONDS, "main": [W, H], "sub": [SUB_W, SUB_H],
            "course": "ICE4204", "not_enrolled": sorted(NOT_ENROLLED), "cameras": {}}
    for cam, sids in CAMERAS.items():
        tiles = [face_tile(engine, folders[sid], sizes[i % len(sizes)]) for i, sid in enumerate(sids)]
        n = FPS * SECONDS

        def frames():  # rendered on the fly: 400 x 1080p frames would be ~2.5 GB in RAM
            return (render(tiles, k / FPS) for k in range(n))

        encode(frames(), os.path.join(args.out, f"{cam}_main.mp4"), (W, H), "4M")
        encode(frames(), os.path.join(args.out, f"{cam}_sub.mp4"), (SUB_W, SUB_H), "512k")
        cv2.imwrite(os.path.join(args.out, f"{cam}_preview.jpg"), render(tiles, 0.0))
        meta["cameras"][cam] = sids
        print(f"{cam}: {n} frames, students {sids}")
    with open(os.path.join(args.out, "streams.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
