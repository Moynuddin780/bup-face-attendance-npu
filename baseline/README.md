# Original Python baseline

`BUP_Attendance_NPU/app.py` is the **original Python project's code, verbatim** — lines 1–965 of
its 1890-line `app.py`: imports and settings, `DeepXFacePipeline` (SCRFD-10G + ArcFace on the NPU
through `dx_engine`, with `_nms`, `ARC_FACE_SRC`), the DB helpers, `build_embeddings_for_class`,
`AttendanceTracker`, `RTSPCapture`, `_process_camera` and `_gen_feed`. Nothing in it was edited.

It is what the C++ engine is measured against:

| Test / tool | Uses from the baseline |
|---|---|
| `tests/test_tracker_parity.py` | `AttendanceTracker`, `MIN_PRESENT_SEC` — identical attendance results |
| `tests/test_face_engine.py`, `tools/eval_recognition.py` | `DeepXFacePipeline` — recognition accuracy on the enrolled photos |
| `tools/bench.py --mode original` | `RTSPCapture`, `DeepXFacePipeline`, `_process_camera` — speed / CPU before vs after |

`original_routes.json` lists every route of the original app; `tests/test_app_smoke.py` checks
that the current app still serves all of them (old links and bookmarks keep working).

## Where it came from

The original project folder was not copied to this machine, so the baseline was reconstructed from the
porting session log (2026-09-21): lines 1–965 of the original `app.py` are reproduced verbatim. Where the same
lines were recorded twice, the two copies agree exactly. The rest of the original file — the Flask page routes
(lines 966–1890) — is not here: the log holds only parts of it, and the pages have since been rebuilt (rooms,
sessions per room, students, courses, mobile layout), so a page-by-page comparison with the original no longer
applies. The original templates are not available either.

To compare against a complete copy of the original project instead, point the tests at it:
`BUP_ORIG_APP=/path/to/BUP_Attendance_NPU/app.py pytest tests/`.
