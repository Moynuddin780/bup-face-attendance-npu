"""rtsp_check: a camera reached through port forwarding often has only its RTSP port open. The room
page must then tell "the stream works" from "wrong password" (it said "Unreachable" for both)."""
import hashlib
import re
import socket
import threading

import pytest

import camera_control as cc

USER, PASSWORD, REALM, NONCE = "admin", "s3cret", "88263f8e04e6", "abc123"


def _md5(s):
    return hashlib.md5(s.encode()).hexdigest()


class FakeRtspCamera:
    """DESCRIBE only, Digest auth like Uniview / Hikvision, one stream at /media/video1."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(5)
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._client, args=(conn,), daemon=True).start()

    def _client(self, conn):
        buf = b""
        with conn:
            while True:
                while b"\r\n\r\n" not in buf:
                    d = conn.recv(4096)
                    if not d:
                        return
                    buf += d
                head, _, buf = buf.partition(b"\r\n\r\n")
                lines = head.decode().split("\r\n")
                method, uri, _ = lines[0].split(" ")
                cseq = next(l.split(":", 1)[1].strip() for l in lines if l.lower().startswith("cseq"))
                auth = next((l.split(":", 1)[1].strip() for l in lines if l.lower().startswith("authorization")), "")
                if "@" in uri:   # credentials never go into the request line
                    conn.sendall(f"RTSP/1.0 400 Bad Request\r\nCSeq: {cseq}\r\n\r\n".encode())
                    continue
                p = dict(re.findall(r'(\w+)="([^"]*)"', auth))
                good = p.get("response") == _md5(f"{_md5(f'{USER}:{REALM}:{PASSWORD}')}:{NONCE}:{_md5(f'{method}:{uri}')}")
                if not good:
                    conn.sendall(f'RTSP/1.0 401 Unauthorized\r\nCSeq: {cseq}\r\nWWW-Authenticate: Digest realm="{REALM}", '
                                 f'nonce="{NONCE}", stale="FALSE"\r\n\r\n'.encode())
                    continue
                if not uri.endswith("/media/video1"):
                    conn.sendall(f"RTSP/1.0 404 Not Found\r\nCSeq: {cseq}\r\n\r\n".encode())
                    continue
                sdp = "v=0\r\nm=video 0 RTP/AVP 96\r\na=rtpmap:96 H265/90000\r\nm=audio 0 RTP/AVP 0\r\na=rtpmap:0 PCMU/8000\r\n"
                conn.sendall(f"RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nContent-Type: application/sdp\r\n"
                             f"Content-Length: {len(sdp)}\r\n\r\n{sdp}".encode())


@pytest.fixture(scope="module")
def cam():
    return FakeRtspCamera()


def test_stream_opens_with_the_right_password(cam):
    r = cc.rtsp_check(f"rtsp://{USER}:{PASSWORD}@127.0.0.1:{cam.port}/media/video1", timeout=5)
    assert r == {"ok": True, "status": 200, "codecs": ["H265"], "error": ""}


def test_wrong_password_and_wrong_path_are_named(cam):
    r = cc.rtsp_check(f"rtsp://{USER}:nope@127.0.0.1:{cam.port}/media/video1", timeout=5)
    assert not r["ok"] and r["status"] == 401 and "user name / password" in r["error"]
    r = cc.rtsp_check(f"rtsp://127.0.0.1:{cam.port}/media/video1", timeout=5)
    assert r["status"] == 401 and "needs a user name and password" in r["error"]
    r = cc.rtsp_check(f"rtsp://{USER}:{PASSWORD}@127.0.0.1:{cam.port}/Streaming/Channels/101", timeout=5)
    assert r["status"] == 404 and "no stream at this path" in r["error"]


def test_closed_port(cam):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()   # nothing listens here
    r = cc.rtsp_check(f"rtsp://{USER}:{PASSWORD}@127.0.0.1:{port}/media/video1", timeout=3)
    assert not r["ok"] and r["status"] == 0 and "cannot connect" in r["error"]
