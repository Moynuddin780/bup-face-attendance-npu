"""Shared pytest fixtures for the BUP Attendance C++ backend."""
import ast
import os
import sys
import types
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))  # bup_engine*.so and app.py live in the app root

# The original Python project is the parity/accuracy baseline: its NPU pipeline and attendance
# tracker, verbatim, ship in baseline/ (see baseline/README.md). BUP_ORIG_APP points elsewhere,
# e.g. at a complete copy of the original project.
ORIG_APP = Path(os.environ.get(
    "BUP_ORIG_APP",
    APP_DIR / "baseline" / "BUP_Attendance_NPU" / "app.py"))

DET_MODEL = APP_DIR / "models" / "scrfd-10g_640x640.dxnn"
REC_MODEL = APP_DIR / "models" / "arcface_iresnet100_112x112_ms1m.dxnn"


def load_original_symbols(names, extra_globals=None):
    """Exec only the named top-level defs/classes of the ORIGINAL app.py.

    Importing the original module would start Flask, probe GStreamer and demand
    dx_engine; extracting the exact source of the classes under test keeps the
    baseline byte-for-byte the original code without those side effects.
    """
    if not ORIG_APP.exists():
        pytest.skip(f"original app.py not found at {ORIG_APP} (set BUP_ORIG_APP)")
    src = ORIG_APP.read_text(encoding="utf-8")
    tree = ast.parse(src)
    wanted = [n for n in tree.body
              if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Assign))
              and (getattr(n, "name", None) in names
                   or (isinstance(n, ast.Assign)
                       and any(getattr(t, "id", None) in names for t in n.targets)))]
    missing = set(names) - {getattr(n, "name", None) for n in wanted} - {
        t.id for n in wanted if isinstance(n, ast.Assign) for t in n.targets}
    assert not missing, f"original app.py lacks {missing}"
    module = ast.Module(body=wanted, type_ignores=[])
    ns = {"__name__": "orig_app_extract"}
    import time, numpy as np  # noqa: E401 - names the original code expects
    ns.update(time=time, np=np, os=os)
    ns.update(extra_globals or {})
    exec(compile(module, str(ORIG_APP), "exec"), ns)
    return types.SimpleNamespace(**{n: ns[n] for n in names})


@pytest.fixture(scope="session")
def engine_module():
    import bup_engine  # noqa: WPS433 - built by setup.sh into APP_DIR
    return bup_engine


@pytest.fixture(scope="session")
def face_engine(engine_module):
    if not DET_MODEL.exists() or not REC_MODEL.exists():
        pytest.skip("models missing — run setup.sh")
    return engine_module.FaceEngine(str(DET_MODEL), str(REC_MODEL))
