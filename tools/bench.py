#!/usr/bin/env python3
"""Before/after benchmark on identical input (the local RTSP test cameras).

  A  original   ORIGINAL app.py code (RTSPCapture + DeepXFacePipeline + _process_camera,
                extracted verbatim) on the MAIN stream, pip-OpenCV capture, Python loop
  B  cpp-main   C++ engine, main stream (1920x1080 decode)
  C  cpp-sub    C++ engine, sub-stream (704x576 decode, aspect-corrected)

Each mode runs 2 cameras for --seconds in its own process; we report per-camera processed
fps (unique camera frames through the full pipeline), decode fps, CPU cores used by the
process, and pipeline latency. Writes bench_report.json + bench_report.md.

  tools/bench.py --all            (needs tools/rtsp_server.sh start)
"""
import argparse
import ast
import json
import os
import subprocess
import sys
import threading
import time

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DET = os.path.join(APP_DIR, "models", "scrfd-10g_640x640.dxnn")
REC = os.path.join(APP_DIR, "models", "arcface_iresnet100_112x112_ms1m.dxnn")
PORT = int(os.environ.get("RTSP_TEST_PORT", "18554"))
CAMS = [("camera1", f"rtsp://127.0.0.1:{PORT}/Streaming/Channels/101"),
        ("camera2", f"rtsp://127.0.0.1:{PORT}/Streaming/Channels/201")]
CLASS = "ICE4204"


def cpu_seconds():
    with open("/proc/self/stat") as f:
        parts = f.read().rsplit(")", 1)[1].split()
    return (int(parts[11]) + int(parts[12])) / os.sysconf("SC_CLK_TCK")


def class_gallery(get_faces, imread):
    """The app's enrollment math for course ICE4204 (mean of normed embeddings, L2)."""
    import glob
    import numpy as np
    db = json.load(open(os.path.join(APP_DIR, "bup_db.json"), encoding="utf-8"))
    c = db["courses"][CLASS]
    rows, meta = [], []
    for s in db["students"].values():
        if (s["dept_code"], s.get("batch"), s.get("section")) != (c["dept_code"], c["batch"], c["section"]):
            continue
        embs = []
        for p in sorted(glob.glob(os.path.join(APP_DIR, "data", os.path.basename(s["images_folder"]), "*"))):
            img = imread(p)
            faces = get_faces(img) if img is not None else []
            if faces:
                f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                e = np.asarray(f.normed_embedding, np.float32).ravel()
                embs.append(e)
        if embs:
            m = np.mean(np.vstack(embs), axis=0)
            rows.append(m / np.linalg.norm(m))
            meta.append({"student_id": s["student_id"], "name": s["name"]})
    return np.vstack(rows).astype(np.float32), meta


# ── A: the original Python pipeline ──────────────────────────────────────────
def run_original(seconds, orig_app):
    import cv2
    import dx_engine
    import numpy as np

    src = open(orig_app, encoding="utf-8").read()
    names = {"_nms", "DeepXFacePipeline", "AttendanceTracker", "RTSPCapture", "_process_camera",
             "l2_norm", "_get_embedding"}
    consts = {"ARC_FACE_SRC", "SIM_THRESHOLD", "MIN_PRESENT_SEC", "FRAME_W", "FRAME_H", "PROCESS_INTERVAL"}
    body = [n for n in ast.parse(src).body
            if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in consts for t in n.targets))]

    bounds = iter(["NPU_0", "NPU_1"])  # the original opens NPU_ALL; the shared NPU refuses it

    class Dx:
        def __getattr__(self, k):
            return getattr(dx_engine, k)

        @staticmethod
        def InferenceEngine(path):  # noqa: N802
            try:
                return dx_engine.InferenceEngine(path)
            except Exception:  # noqa: BLE001 - NPU_ALL refused on a shared device
                opt = dx_engine.InferenceOption()
                opt.bound_option = getattr(dx_engine.InferenceOption.BOUND_OPTION, next(bounds))
                return dx_engine.InferenceEngine(path, opt)

    ns = {"np": np, "cv2": cv2, "dx": Dx(), "time": time, "os": os, "threading": threading,
          "HAS_GSTREAMER": False, "STREAM_BACKEND": "FFmpeg", "print": lambda *a, **k: None}
    exec(compile(ast.Module(body=body, type_ignores=[]), orig_app, "exec"), ns)
    fa = ns["DeepXFacePipeline"](DET, REC)
    embeds, meta = class_gallery(fa.get, cv2.imread)

    # instrumentation (no behaviour change): count unique frames and time the NPU pipeline
    stats = {cid: {"calls": 0, "unique": 0, "pipe_ms": 0.0, "faces": 0} for cid, _ in CAMS}
    orig_read = ns["RTSPCapture"].read
    last_obj = {}

    def read(self):
        with self.lock:
            obj = self.frame
        ok, fr = orig_read(self)
        if ok and last_obj.get(self.cam_id) is not obj:
            last_obj[self.cam_id] = obj
            stats[self.cam_id]["unique"] += 1
        return ok, fr
    ns["RTSPCapture"].read = read

    tl = threading.local()
    orig_get = fa.get

    def get(img):
        t0 = time.perf_counter()
        faces = orig_get(img)
        s = stats[tl.cam]
        s["calls"] += 1
        s["pipe_ms"] += (time.perf_counter() - t0) * 1000
        s["faces"] += len(faces)
        return faces
    fa.get = get

    session = {"shutdown": False, "tracker": ns["AttendanceTracker"](meta), "db_embeds": embeds,
               "db_meta": meta, "cameras": {}, "frame_locks": {}, "frames": {}, "fps": 0}
    ns["_session"] = session
    ns["get_face_engine"] = lambda: fa
    for cid, url in CAMS:
        cap = ns["RTSPCapture"](cid, url)
        cap.start()
        session["cameras"][cid] = cap
        session["frame_locks"][cid] = threading.Lock()

    def worker(cid):
        tl.cam = cid
        ns["_process_camera"](cid)
    threads = [threading.Thread(target=worker, args=(cid,), daemon=True) for cid, _ in CAMS]
    for t in threads:
        t.start()

    time.sleep(5)  # warm-up / connect
    s0 = {k: dict(v) for k, v in stats.items()}
    c0, w0 = cpu_seconds(), time.time()
    time.sleep(seconds)
    c1, w1 = cpu_seconds(), time.time()
    s1 = {k: dict(v) for k, v in stats.items()}
    session["shutdown"] = True
    for t in threads:
        t.join(timeout=5)
    for cap in session["cameras"].values():
        cap.stop()
    wall = w1 - w0
    cams = {}
    for cid, _ in CAMS:
        d = {k: s1[cid][k] - s0[cid][k] for k in s1[cid]}
        cams[cid] = {"processed_fps": round(d["unique"] / wall, 2), "loop_fps": round(d["calls"] / wall, 2),
                     "pipeline_ms": round(d["pipe_ms"] / max(d["calls"], 1), 1),
                     "faces_per_frame": round(d["faces"] / max(d["calls"], 1), 2),
                     "frame_size": "1920x1080 -> 1280x720 (squashed if not 16:9)"}
    seen = sum(r["detection_count"] > 0 for r in session["tracker"].summary())
    return {"mode": "original", "stream": "main", "cpu_cores": round((c1 - c0) / wall, 2),
            "students_seen": seen, "cameras": cams}


# ── B / C: the C++ engine ────────────────────────────────────────────────────
def fit(size, max_w=1280, max_h=720):
    """Processing size: the session fits frames inside FRAME_W x FRAME_H keeping the aspect."""
    w, h = (int(x) for x in size.split("x"))
    s = min(1.0, max_w / w, max_h / h)
    return f"{round(w * s)}x{round(h * s)}"


def run_cpp(seconds, profile):
    sys.path.insert(0, APP_DIR)
    import bup_engine
    engine = bup_engine.FaceEngine(DET, REC)
    embeds, meta = class_gallery(engine.get, bup_engine.imread)
    tracker = bup_engine.AttendanceTracker(meta, min_time=60)
    sess = bup_engine.AttendanceSession(engine, tracker, embeds, meta, CAMS, stream_profile=profile)
    sess.start()
    time.sleep(5)
    i0 = {cid: sess.camera(cid).get_info() for cid, _ in CAMS}
    c0, w0 = cpu_seconds(), time.time()
    time.sleep(seconds)
    c1, w1 = cpu_seconds(), time.time()
    i1 = {cid: sess.camera(cid).get_info() for cid, _ in CAMS}
    sess.stop()
    wall = w1 - w0
    cams = {}
    for cid, _ in CAMS:
        a, b = i0[cid], i1[cid]
        cams[cid] = {"processed_fps": round((b["frames_processed"] - a["frames_processed"]) / wall, 2),
                     "decode_fps": round((b["frames_decoded"] - a["frames_decoded"]) / wall, 2),
                     "detect_ms": b["detect_ms_avg"], "recognize_ms": b["recognize_ms_avg"],
                     "faces_per_frame": b["faces_last_frame"],
                     "frame_size": f'{b["frame_size"]} -> {fit(b["processing_size"])}', "backend": b["backend"]}
    seen = sum(r["detection_count"] > 0 for r in tracker.summary())
    return {"mode": f"cpp-{profile}", "stream": profile, "cpu_cores": round((c1 - c0) / wall, 2),
            "students_seen": seen, "npu_bound": engine.info()["npu_bound_rec"], "cameras": cams}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["original", "cpp-main", "cpp-sub"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--orig-app", default=os.environ.get(
        "BUP_ORIG_APP", os.path.join(APP_DIR, "baseline", "BUP_Attendance_NPU", "app.py")))
    ap.add_argument("--out", default=os.path.join(APP_DIR, "bench_report"))
    args = ap.parse_args()

    if args.mode:
        res = run_original(args.seconds, args.orig_app) if args.mode == "original" else \
            run_cpp(args.seconds, args.mode.split("-")[1])
        print("BENCH_RESULT " + json.dumps(res), flush=True)
        # The original's capture threads are fire-and-forget daemons still blocked in
        # cv2 / dx_engine calls; interpreter teardown under them aborts. Results are out.
        os._exit(0)

    results = []
    for mode in ("original", "cpp-main", "cpp-sub"):
        r = subprocess.run([sys.executable, __file__, "--mode", mode, "--seconds", str(args.seconds),
                            "--orig-app", args.orig_app], capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if l.startswith("BENCH_RESULT ")), None)
        if r.returncode != 0 or not line:
            sys.exit(f"{mode} failed:\n{r.stderr[-3000:]}")
        results.append(json.loads(line[len("BENCH_RESULT "):]))
        print(json.dumps(results[-1]))
    contended = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True).stdout
    report = {"seconds": args.seconds, "npu_shared_with": [n for n in contended.split() if "edge" in n],
              "results": results}
    json.dump(report, open(args.out + ".json", "w"), indent=2)

    def fps(r):
        return sum(c["processed_fps"] for c in r["cameras"].values()) / len(r["cameras"])
    base = results[0]
    md = ["| Mode | Stream decoded -> processed | Processed fps / camera | CPU cores (process) | Students seen |",
          "|---|---|---|---|---|"]
    for r in results:
        c = next(iter(r["cameras"].values()))
        md.append(f"| {r['mode']} | {c['frame_size']} | {fps(r):.1f} ({fps(r) / fps(base):.1f}x) | "
                  f"{r['cpu_cores']:.2f} | {r['students_seen']} |")
    note = ("\n> NPU shared with `" + ", ".join(report["npu_shared_with"]) + "` during this run "
            "(NPU_ALL refused, per-core bindings) — absolute numbers are lower than on a dedicated NPU.\n"
            if report["npu_shared_with"] else "\n")
    open(args.out + ".md", "w").write("\n".join(md) + "\n" + note)
    print("\n".join(md) + note)


if __name__ == "__main__":
    main()
