"""The C++ AttendanceTracker must reproduce the original Python class exactly
(spec §4 parity rule): same rows, same order, same rounding, same override rules."""
import random

import pytest

from conftest import load_original_symbols


def _meta(n):
    return [{"student_id": f"20549010{i:02d}", "name": f"Student {i}"} for i in range(n)]


def _drive(trackers, rng, n_events, sids, cams):
    now = 1_700_000_000.0
    for _ in range(n_events):
        r = rng.random()
        if r < 0.03:
            sid = rng.choice(sids + ["unknown_sid"])
            status = rng.choice(["present", "absent", None, ""])
            for t in trackers:
                t.set_manual(sid, status)
            continue
        # mostly small steps (continuous presence), sometimes gaps > 2 s,
        # occasionally a backwards clock step (NTP) which the original also accepts
        step = rng.choice([0.05, 0.1, 0.2, 0.5, 1.0, 1.9, 2.0, 2.5, 7.0, -0.3])
        now += step
        sid = rng.choice(sids + ["not_in_class"])
        cam = rng.choice(cams)
        for t in trackers:
            t.update(sid, cam, now)


@pytest.mark.parametrize("seed", range(8))
def test_summary_and_live_stats_match_original(engine_module, seed):
    orig = load_original_symbols(["MIN_PRESENT_SEC", "AttendanceTracker"])
    rng = random.Random(seed)
    meta = _meta(rng.randint(1, 40))
    min_time = rng.choice([orig.MIN_PRESENT_SEC, 5, 30])
    py_t = orig.AttendanceTracker(meta, min_time=min_time)
    cc_t = engine_module.AttendanceTracker(meta, min_time=min_time)
    sids = [m["student_id"] for m in meta]
    _drive([py_t, cc_t], rng, 5000, sids, ["camera1", "camera2"])

    assert cc_t.summary() == py_t.summary()

    py_live, cc_live = py_t.live_stats(), cc_t.live_stats()
    assert abs(cc_live.pop("duration") - py_live.pop("duration")) <= 1.0
    assert cc_live == py_live
    # types matter: the summary is stored in the Flask cookie session and in JSON records
    for a, b in zip(cc_t.summary(), py_t.summary()):
        assert {k: type(v) for k, v in a.items()} == {k: type(v) for k, v in b.items()}


def test_empty_class(engine_module):
    orig = load_original_symbols(["MIN_PRESENT_SEC", "AttendanceTracker"])
    py_t, cc_t = orig.AttendanceTracker([]), engine_module.AttendanceTracker([])
    assert cc_t.summary() == py_t.summary() == []
    py_live, cc_live = py_t.live_stats(), cc_t.live_stats()
    py_live.pop("duration"); cc_live.pop("duration")
    assert cc_live == py_live and type(cc_live["rate"]) is type(py_live["rate"])


def test_total_time_accessor(engine_module):
    t = engine_module.AttendanceTracker([{"student_id": "a", "name": "A"}], min_time=60)
    t.update("a", "camera1", 100.0)
    t.update("a", "camera1", 101.5)
    assert t.total_time("a") == pytest.approx(1.5)
    assert t.total_time("zzz") == 0.0
