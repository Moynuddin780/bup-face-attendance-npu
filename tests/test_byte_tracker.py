"""ByteTrack face tracker (bup_engine.ByteTracker = the session's per-camera tracker): identities
must survive motion, weak (turned / occluded) detections, short disappearances and crossings."""
import pytest

from conftest import APP_DIR  # noqa: F401  (puts the engine on sys.path)

bup_engine = pytest.importorskip("bup_engine")

DT = 0.08   # ~12.5 NPU frames per second


def box(cx, cy, w=60, h=75, score=0.9):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score)


def ids_by_det(tracks):
    return {t["det"]: t["id"] for t in tracks if t["det"] >= 0}


def test_ids_stay_with_moving_faces():
    bt = bup_engine.ByteTracker()
    seen = {0: set(), 1: set()}
    for k in range(60):
        dets = [box(100 + 4 * k, 200), box(600 - 3 * k, 260)]
        m = ids_by_det(bt.update(dets, k * DT))
        for d in (0, 1):
            seen[d].add(m[d])
    assert len(seen[0]) == 1 and len(seen[1]) == 1 and seen[0] != seen[1]
    # the Kalman velocity follows the motion (pixels per second)
    vx = {t["det"]: t["velocity"][0] for t in bt.update([box(100 + 4 * 60, 200), box(600 - 3 * 60, 260)], 60 * DT)}
    assert vx[0] == pytest.approx(4 / DT, rel=0.25) and vx[1] == pytest.approx(-3 / DT, rel=0.25)


def test_weak_detection_keeps_the_track():
    """A face that turns away scores below the confident threshold: ByteTrack's second association
    keeps it on its track instead of dropping it."""
    bt = bup_engine.ByteTracker()
    for k in range(5):
        tid = ids_by_det(bt.update([box(300, 300)], k * DT))[0]
    for k in range(5, 15):
        m = ids_by_det(bt.update([box(300 + k, 300, score=0.35)], k * DT))
        assert m == {0: tid}
    assert ids_by_det(bt.update([box(316, 300)], 15 * DT)) == {0: tid}
    # a weak detection never starts a track of its own
    assert all(t["id"] == tid for t in bt.update([box(316, 300), box(900, 300, score=0.4)], 16 * DT))


def test_lost_face_gets_its_track_back():
    bt = bup_engine.ByteTracker(lost_keep_s=2.0)
    for k in range(5):
        tid = ids_by_det(bt.update([box(400, 300)], k * DT))[0]
    for k in range(5, 15):                         # hidden for 0.8 s
        out = bt.update([], k * DT)
        assert [t["id"] for t in out] == [tid] and out[0]["lost"]
    assert ids_by_det(bt.update([box(402, 301)], 15 * DT)) == {0: tid}


def test_track_is_forgotten_after_lost_keep():
    bt = bup_engine.ByteTracker(lost_keep_s=1.0)
    for k in range(5):
        tid = ids_by_det(bt.update([box(400, 300)], k * DT))[0]
    assert bt.update([], 5 * DT + 1.5) == []
    new = bt.update([box(400, 300)], 5 * DT + 1.6)
    assert new[0]["id"] != tid


def test_crossing_faces_keep_their_ids():
    bt = bup_engine.ByteTracker()
    first = None
    for k in range(40):
        a = box(100 + 10 * k, 300)      # left -> right
        b = box(500 - 10 * k, 300)      # right -> left, they pass each other around k = 20
        m = ids_by_det(bt.update([a, b], k * DT))
        if first is None:
            first = m
        if 0 in m:
            assert m[0] == first[0], k
        if 1 in m:
            assert m[1] == first[1], k


def test_one_frame_blip_is_not_a_track():
    bt = bup_engine.ByteTracker()
    bt.update([box(100, 100)], 0.0)
    bt.update([box(100, 100), box(700, 400)], DT)    # false positive for one frame
    out = bt.update([box(100, 100)], 2 * DT)
    assert len(out) == 1


def test_linear_assignment_prefers_overlap():
    bt = bup_engine.ByteTracker()
    bt.update([box(100, 100), box(160, 100)], 0.0)   # two neighbours, boxes overlap a little
    out = bt.update([box(163, 100), box(103, 100)], DT)   # listed in the other order
    m = {t["id"]: t["box"][0] for t in out}
    assert m[1] < m[2]                                  # id 1 stays the left face
