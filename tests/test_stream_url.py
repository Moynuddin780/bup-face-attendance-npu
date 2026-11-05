"""Sub-stream URL rewriting (spec §5). The DB keeps main-stream URLs; the engine
switches profile at connect time."""
import pytest


@pytest.mark.parametrize("url,expected", [
    # Dahua (the scheme stored in bup_db.json)
    ("rtsp://admin:pw@10.10.4.6:554/cam/realmonitor?channel=1&subtype=0",
     "rtsp://admin:pw@10.10.4.6:554/cam/realmonitor?channel=1&subtype=1"),
    ("rtsp://h/cam/realmonitor?subtype=0&channel=2",
     "rtsp://h/cam/realmonitor?subtype=1&channel=2"),
    ("rtsp://h/cam/realmonitor?channel=1",
     "rtsp://h/cam/realmonitor?channel=1&subtype=1"),
    # Hikvision ISAPI style
    ("rtsp://h:554/Streaming/Channels/101", "rtsp://h:554/Streaming/Channels/102"),
    ("rtsp://h/Streaming/Channels/201?transportmode=unicast",
     "rtsp://h/Streaming/Channels/202?transportmode=unicast"),
    ("rtsp://h/streaming/channels/1601", "rtsp://h/streaming/channels/1602"),
    # Hikvision legacy
    ("rtsp://h/h264/ch1/main/av_stream", "rtsp://h/h264/ch1/sub/av_stream"),
])
def test_sub_profile_rewrites_known_vendors(engine_module, url, expected):
    assert engine_module.resolve_stream_url(url, "sub") == expected


@pytest.mark.parametrize("url", [
    "rtsp://h/cam/realmonitor?channel=1&subtype=0",
    "rtsp://h/Streaming/Channels/101",
])
def test_main_profile_keeps_url(engine_module, url):
    assert engine_module.resolve_stream_url(url, "main") == url


def test_unknown_vendor_is_left_alone(engine_module):
    assert engine_module.resolve_stream_url("rtsp://h/live/stream", "sub") == "rtsp://h/live/stream"


def test_already_sub_is_idempotent(engine_module):
    u = "rtsp://h/cam/realmonitor?channel=1&subtype=1"
    assert engine_module.resolve_stream_url(u, "sub") == u
    assert engine_module.resolve_stream_url("rtsp://h/Streaming/Channels/102", "sub") \
        == "rtsp://h/Streaming/Channels/102"


def test_bad_profile_rejected(engine_module):
    with pytest.raises(ValueError):
        engine_module.resolve_stream_url("rtsp://h/x", "4k")
