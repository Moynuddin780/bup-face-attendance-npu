"""Buildings & Cameras: read a camera's streams and set a stream's resolution on the camera
(Hikvision ISAPI with digest auth, ONVIF Media2 with WS-Security — both faked here), and the
per-room classroom lock: one class per room and per teacher, rooms locked until the class ends."""
import base64
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from conftest import APP_DIR
from test_app_smoke import RTSP, TEACHER, _h, _make_copy

sys.path.insert(0, str(APP_DIR))
import camera_control as cc  # noqa: E402

USER, PASSWORD = "admin", "s3cret:pw"
md5 = lambda s: hashlib.md5(s.encode()).hexdigest()


# ── fake cameras ──────────────────────────────────────────────────────────────────────────────
class FakeHikvision(BaseHTTPRequestHandler):
    streams = {"101": [2560, 1440], "102": [640, 480]}
    rates = {"101": 2000, "102": 2000}   # 1/100 fps
    bitrates = {"101": 4096, "102": 4096}
    gops = {"101": 50, "102": 50}
    opts = {"101": ("2560,1920,1280", "1440,1080,720"), "102": ("1280,640,640", "720,480,360")}
    puts = []

    def log_message(self, *a):
        pass

    def _send(self, code, body, headers=()):
        data = body.encode()
        self.send_response(code)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self):
        h = self.headers.get("Authorization", "")
        if not h.startswith("Digest "):
            return False
        p = dict(re.findall(r'(\w+)="?([^",]*)"?', h[7:]))
        ha1, ha2 = md5(f"{p.get('username')}:IP Camera:{PASSWORD}"), md5(f"{self.command}:{p.get('uri')}")
        return p.get("username") == USER and p.get("response") == md5(f"{ha1}:{p.get('nonce')}:{p.get('nc')}:{p.get('cnonce')}:auth:{ha2}")

    def _handle(self, body=""):
        if not self._authorized():
            return self._send(401, "<ResponseStatus><statusString>Unauthorized ISAPI</statusString></ResponseStatus>",
                              [("WWW-Authenticate", 'Digest realm="IP Camera", qop="auth", nonce="n0nce", algorithm=MD5')])
        ns = 'xmlns="http://www.hikvision.com/ver20/XMLSchema"'
        if self.path == "/ISAPI/System/deviceInfo":
            return self._send(200, f"<DeviceInfo {ns}><manufacturer>hikvision</manufacturer><model>DS-FAKE</model></DeviceInfo>")
        m = re.match(r"^/ISAPI/Streaming/channels/(\d+)(/capabilities)?$", self.path)
        if not m or m.group(1) not in self.streams:
            return self._send(404, "<ResponseStatus><statusCode>4</statusCode></ResponseStatus>")
        cid = m.group(1)
        if self.command == "PUT":
            w = int(re.search(r"<videoResolutionWidth>(\d+)<", body).group(1))
            h = int(re.search(r"<videoResolutionHeight>(\d+)<", body).group(1))
            self.rates[cid] = int(re.search(r"<maxFrameRate>(\d+)<", body).group(1))
            self.bitrates[cid] = int(re.search(r"<vbrUpperCap>(\d+)<", body).group(1))
            self.gops[cid] = int(re.search(r"<GovLength>(\d+)<", body).group(1))
            self.streams[cid] = [w, h]
            self.puts.append((cid, w, h))
            return self._send(200, f"<ResponseStatus {ns}><statusCode>1</statusCode><statusString>OK</statusString></ResponseStatus>")
        w, h = self.streams[cid]
        if m.group(2):
            ow, oh = self.opts[cid]
            return self._send(200, f'<StreamingChannel {ns}><Video><videoResolutionWidth opt="{ow}">{w}</videoResolutionWidth>'
                                   f'<videoResolutionHeight opt="{oh}">{h}</videoResolutionHeight>'
                                   f'<maxFrameRate opt="2500,2200,2000,1500,50">{self.rates[cid]}</maxFrameRate></Video></StreamingChannel>')
        return self._send(200, f"<StreamingChannel {ns}><id>{cid}</id><Video><videoCodecType>H.264</videoCodecType>"
                               f"<videoResolutionWidth>{w}</videoResolutionWidth><videoResolutionHeight>{h}</videoResolutionHeight>"
                               f"<videoQualityControlType>VBR</videoQualityControlType><vbrUpperCap>{self.bitrates[cid]}</vbrUpperCap>"
                               f"<maxFrameRate>{self.rates[cid]}</maxFrameRate><GovLength>{self.gops[cid]}</GovLength></Video></StreamingChannel>")

    def do_GET(self):
        self._handle()

    def do_PUT(self):
        self._handle(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode())


class FakeOnvif(BaseHTTPRequestHandler):
    enc = {"enc0": [2560, 1440], "enc1": [640, 480]}
    rate = {"enc0": 20, "enc1": 20}
    options = {"enc0": [[2560, 1440], [1920, 1080], [1280, 720]], "enc1": [[704, 576], [640, 480], [352, 288]]}
    sets = []

    def log_message(self, *a):
        pass

    def _send(self, code, body):
        data = ('<?xml version="1.0"?><SOAP-ENV:Envelope xmlns:SOAP-ENV="http://www.w3.org/2003/05/soap-envelope" '
                'xmlns:tt="http://www.onvif.org/ver10/schema" xmlns:tr2="http://www.onvif.org/ver20/media/wsdl" '
                'xmlns:tds="http://www.onvif.org/ver10/device/wsdl"><SOAP-ENV:Body>' + body +
                '</SOAP-ENV:Body></SOAP-ENV:Envelope>').encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/soap+xml")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _wsse_ok(self, b):
        try:
            nonce = base64.b64decode(re.search(r"<Nonce[^>]*>([^<]+)<", b).group(1))
            created = re.search(r"<Created[^>]*>([^<]+)<", b).group(1)
            digest = re.search(r"<Password[^>]*>([^<]+)<", b).group(1)
            user = re.search(r"<Username>([^<]+)<", b).group(1)
        except AttributeError:
            return False
        want = base64.b64encode(hashlib.sha1(nonce + created.encode() + PASSWORD.encode()).digest()).decode()
        return user == USER and digest == want

    def do_GET(self):
        self._send(404, "")

    def do_POST(self):
        b = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
        op = re.search(r"<s:Body><(\w+)", b).group(1)
        if op == "GetSystemDateAndTime":
            t = time.gmtime(time.time() + 3600)   # camera clock one hour ahead
            return self._send(200, "<tds:GetSystemDateAndTimeResponse><tds:SystemDateAndTime><tt:UTCDateTime><tt:Time>"
                                   f"<tt:Hour>{t.tm_hour}</tt:Hour><tt:Minute>{t.tm_min}</tt:Minute><tt:Second>{t.tm_sec}</tt:Second>"
                                   f"</tt:Time><tt:Date><tt:Year>{t.tm_year}</tt:Year><tt:Month>{t.tm_mon}</tt:Month>"
                                   f"<tt:Day>{t.tm_mday}</tt:Day></tt:Date></tt:UTCDateTime></tds:SystemDateAndTime>"
                                   "</tds:GetSystemDateAndTimeResponse>")
        if not self._wsse_ok(b):
            return self._send(400, "<SOAP-ENV:Fault><SOAP-ENV:Reason><SOAP-ENV:Text>ter:NotAuthorized</SOAP-ENV:Text>"
                                   "</SOAP-ENV:Reason></SOAP-ENV:Fault>")
        if op == "GetServices":   # Dahua-style https XAddr: reached over plain http on the same port
            return self._send(200, "<tds:GetServicesResponse><tds:Service><tds:Namespace>http://www.onvif.org/ver20/media/wsdl"
                                   "</tds:Namespace><tds:XAddr>https://127.0.0.1/onvif/media2_service</tds:XAddr></tds:Service>"
                                   "</tds:GetServicesResponse>")
        if op == "GetDeviceInformation":
            return self._send(200, "<tds:GetDeviceInformationResponse><tds:Manufacturer>Dahua</tds:Manufacturer>"
                                   "<tds:Model>IPC-FAKE</tds:Model></tds:GetDeviceInformationResponse>")
        if op == "GetProfiles":
            prof = "".join(f'<tr2:Profiles token="prof{i}" fixed="true"><tt:Name>Profile{i}</tt:Name><tr2:Configurations>'
                           f'<tr2:VideoEncoder token="enc{i}" GovLength="50" Profile="Main"><tt:Name>enc{i}</tt:Name>'
                           f'<tt:UseCount>1</tt:UseCount><tt:Encoding>H265</tt:Encoding><tt:Resolution><tt:Width>{w}</tt:Width>'
                           f'<tt:Height>{h}</tt:Height></tt:Resolution><tt:RateControl ConstantBitRate="false">'
                           f'<tt:FrameRateLimit>{self.rate["enc%d" % i]}</tt:FrameRateLimit><tt:BitrateLimit>2048</tt:BitrateLimit></tt:RateControl>'
                           f'<tt:Quality>4</tt:Quality></tr2:VideoEncoder></tr2:Configurations></tr2:Profiles>'
                           for i, (w, h) in enumerate([self.enc["enc0"], self.enc["enc1"]]))
            return self._send(200, f"<tr2:GetProfilesResponse>{prof}</tr2:GetProfilesResponse>")
        if op == "GetVideoEncoderConfigurationOptions":
            tok = re.search(r"<ConfigurationToken>(\w+)<", b).group(1)
            res = "".join(f"<tt:ResolutionsAvailable><tt:Width>{w}</tt:Width><tt:Height>{h}</tt:Height></tt:ResolutionsAvailable>"
                          for w, h in self.options[tok])
            return self._send(200, "<tr2:GetVideoEncoderConfigurationOptionsResponse><tr2:Options><tt:Encoding>H264</tt:Encoding>"
                                   "<tt:ResolutionsAvailable><tt:Width>320</tt:Width><tt:Height>240</tt:Height></tt:ResolutionsAvailable>"
                                   f"</tr2:Options><tr2:Options><tt:Encoding>H265</tt:Encoding>{res}"
                                   "<tt:FrameRatesSupported>30 25 20 15</tt:FrameRatesSupported></tr2:Options>"
                                   "</tr2:GetVideoEncoderConfigurationOptionsResponse>")
        if op == "GetStreamUri":
            sub = re.search(r"<ProfileToken>prof(\d)<", b).group(1)
            return self._send(200, "<tr2:GetStreamUriResponse><tr2:Uri>rtsp://192.0.2.1:554/cam/realmonitor??channel=1&amp;"
                                   f"subtype={sub}&amp;unicast=true</tr2:Uri></tr2:GetStreamUriResponse>")
        if op == "GetVideoEncoderConfigurations":
            tok = re.search(r"<ConfigurationToken>(\w+)<", b).group(1)
            w, h = self.enc[tok]
            return self._send(200, f'<tr2:GetVideoEncoderConfigurationsResponse><tr2:Configurations token="{tok}" GovLength="50" '
                                   f'Profile="Main"><tt:Name>{tok}</tt:Name><tt:UseCount>1</tt:UseCount><tt:Encoding>H265</tt:Encoding>'
                                   f'<tt:Resolution><tt:Width>{w}</tt:Width><tt:Height>{h}</tt:Height></tt:Resolution>'
                                   f'<tt:RateControl ConstantBitRate="false"><tt:FrameRateLimit>{self.rate[tok]}</tt:FrameRateLimit>'
                                   f'<tt:BitrateLimit>2048</tt:BitrateLimit></tt:RateControl><tt:Quality>4</tt:Quality>'
                                   f'</tr2:Configurations></tr2:GetVideoEncoderConfigurationsResponse>')
        if op == "SetVideoEncoderConfiguration":
            tok = re.search(r'<Configuration token="(\w+)"', b).group(1)
            w = int(re.search(r"<tt:Width>(\d+)<", b).group(1))
            h = int(re.search(r"<tt:Height>(\d+)<", b).group(1))
            self.enc[tok] = [w, h]
            self.rate[tok] = int(float(re.search(r"<tt:FrameRateLimit>([\d.]+)<", b).group(1)))
            self.sets.append((tok, w, h, 'GovLength="50"' in b))
            type(self).last_set = b
            return self._send(200, "<tr2:SetVideoEncoderConfigurationResponse/>")
        return self._send(400, "<SOAP-ENV:Fault><SOAP-ENV:Reason><SOAP-ENV:Text>unknown</SOAP-ENV:Text></SOAP-ENV:Reason></SOAP-ENV:Fault>")


@pytest.fixture(scope="module")
def fake_cameras():
    servers = {}
    for name, handler in (("isapi", FakeHikvision), ("onvif", FakeOnvif)):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers[name] = srv.server_address[1]
    yield servers


def test_isapi_read_and_set(fake_cameras):
    port = fake_cameras["isapi"]
    url = f"rtsp://{USER}:{PASSWORD.replace(':', '%3A')}@127.0.0.1:554/Streaming/Channels/101"
    info = cc.read_camera(url, timeout=5, http_port=port)
    assert info["ok"] and info["protocol"] == "isapi" and info["model"] == "DS-FAKE", info
    main, sub = info["streams"]
    assert (main["id"], main["width"], main["height"], main["codec"], main["fps"]) == ("main", 2560, 1440, "H264", 20.0)
    assert main["resolutions"] == [(2560, 1440), (1920, 1080), (1280, 720)]
    assert sub["resolutions"] == [(1280, 720), (640, 480), (640, 360)]
    assert sub["uri"] == "rtsp://127.0.0.1:554/Streaming/Channels/102"
    assert sub["fps"] == 20.0 and sub["fps_options"] == [25, 22, 20, 15]   # 0.5 fps option dropped
    info, r = cc.configure_stream(url, "sub", fps=25, timeout=5, http_port=port)   # frame rate only
    assert r["ok"] and FakeHikvision.rates["102"] == 2500 and FakeHikvision.streams["102"] == [640, 480]
    assert cc.configure_stream(url, "sub", fps=24, timeout=5, http_port=port)[1]["ok"] is False
    assert cc.set_resolution(url, "sub", 1280, 720, timeout=5, http_port=port)["ok"]
    assert FakeHikvision.puts[-1] == ("102", 1280, 720)
    # a Wi-Fi camera's stream made to fit its link: bitrate ceiling and an I-frame every second
    info, r = cc.configure_stream(url, "main", bitrate_kbps=2048, gop=20, timeout=5, http_port=port)
    assert r["ok"] and (FakeHikvision.bitrates["101"], FakeHikvision.gops["101"]) == (2048, 20), r
    main = cc.read_camera(url, timeout=5, http_port=port)["streams"][0]
    assert (main["bitrate_kbps"], main["gop"], main["width"]) == (2048, 20, 2560)
    assert cc.configure_stream(url, "main", bitrate_kbps=2048, gop=20, timeout=5, http_port=port)[1].get("unchanged")
    assert cc.set_resolution(url, "sub", 1280, 720, timeout=5, http_port=port).get("unchanged")
    bad = cc.set_resolution(url, "sub", 999, 999, timeout=5, http_port=port)
    assert not bad["ok"] and "does not offer" in bad["error"]
    wrong = cc.read_camera(url.replace("s3cret%3Apw", "nope"), timeout=5, http_port=port)
    assert not wrong["ok"] and "refused" in wrong["error"]


def test_onvif_read_and_set(fake_cameras):
    port = fake_cameras["onvif"]
    url = f"rtsp://{USER}:{PASSWORD.replace(':', '%3A')}@127.0.0.1:554/cam/realmonitor?channel=1&subtype=0"
    info = cc.read_camera(url, timeout=5, http_port=port)
    assert info["ok"] and info["protocol"] == "onvif" and (info["vendor"], info["model"]) == ("Dahua", "IPC-FAKE"), info
    main, sub = info["streams"]
    assert (main["width"], main["height"], main["codec"]) == (2560, 1440, "H265")
    assert sub["resolutions"] == [(704, 576), (640, 480), (352, 288)]       # the H265 options, not the H264 ones
    assert sub["fps"] == 20 and sub["fps_options"] == [30, 25, 20, 15]
    info, r = cc.configure_stream(url, "sub", 704, 576, fps=25, timeout=5, http_port=port)   # both in one request
    assert r["ok"] and FakeOnvif.enc["enc1"] == [704, 576] and FakeOnvif.rate["enc1"] == 25
    assert cc.stream_url(url, sub["uri"]) == \
        f"rtsp://{USER}:s3cret%3Apw@127.0.0.1:554/cam/realmonitor?channel=1&subtype=1&unicast=true"
    r = cc.set_resolution(url, "main", 1920, 1080, timeout=5, http_port=port)
    assert r["ok"], r
    assert FakeOnvif.sets[-1] == ("enc0", 1920, 1080, True)                  # GovLength kept
    main = cc.read_camera(url, timeout=5, http_port=port)["streams"][0]
    assert (main["bitrate_kbps"], main["gop"]) == (2048, 50)
    r = cc.configure_stream(url, "main", bitrate_kbps=1536, gop=25, timeout=5, http_port=port)[1]
    assert r["ok"] and 'GovLength="25"' in FakeOnvif.last_set and "<tt:BitrateLimit>1536<" in FakeOnvif.last_set, r
    assert "<tt:Width>1920<" in FakeOnvif.last_set and "<tt:FrameRateLimit>20<" in FakeOnvif.last_set   # the rest kept
    assert cc.read_camera(url, timeout=5, http_port=port)["streams"][0]["width"] == 1920


def test_unreachable_camera_fails_fast():
    t0 = time.time()
    info = cc.read_camera("rtsp://a:b@127.0.0.1:9/x", http_port=9)          # nothing listens on port 9
    assert not info["ok"] and "cannot reach 127.0.0.1 at all" in info["error"] and time.time() - t0 < 7


def test_rtsp_only_camera_is_explained():
    import socket
    rtsp = socket.socket()
    rtsp.bind(("127.0.0.1", 0))
    rtsp.listen()
    try:
        info = cc.read_camera(f"rtsp://a:b@127.0.0.1:{rtsp.getsockname()[1]}/x", http_port=9)
        assert not info["ok"] and "answers on its RTSP port" in info["error"], info
    finally:
        rtsp.close()


# ── classroom locks and per-camera streams, with the local RTSP test cameras ─────────────────
OTHER = ("prottasha_ict", "teach456")    # teaches ICE4205: same students (ICT 2020 A) as ICE4204


@pytest.fixture(scope="module")
def rooms_app(tmp_path_factory):
    subprocess.run(["bash", str(APP_DIR / "tools" / "rtsp_server.sh"), "start"], check=True)
    root = _make_copy(tmp_path_factory.mktemp("rooms") / "app", APP_DIR / "app.py", with_engine=True, data_mode="copy")
    db = json.loads((root / "bup_db.json").read_text(encoding="utf-8"))
    db["teachers"][OTHER[0]]["password"] = _h(OTHER[1])
    rooms = db["buildings"]["FST"]["floors"]["4"]["rooms"]
    rooms["402"] = {"camera1": f"{RTSP}/101", "camera2": f"{RTSP}/201"}
    # room 403: camera1 = the sub-stream the camera reported (squashed 704x576 of a 16:9 scene),
    # camera2 = main stream at full resolution (no 1280x720 cap)
    rooms["403"] = {"camera1": f"{RTSP}/101", "camera2": f"{RTSP}/201", "camera_settings": {
        "camera1": {"stream": "sub", "width": 704, "height": 576, "uri": f"{RTSP}/102", "scene": [1920, 1080]},
        "camera2": {"stream": "main", "width": 1920, "height": 1080, "uri": f"{RTSP}/201", "scene": [1920, 1080]}}}
    (root / "bup_db.json").write_text(json.dumps(db, indent=2, ensure_ascii=False), encoding="utf-8")
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("bup_rooms_app", root / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.app.config["TESTING"] = True
    yield mod
    mod._stop_session()
    subprocess.run(["bash", str(APP_DIR / "tools" / "rtsp_server.sh"), "stop"], check=False)


def _login(mod, user, pw):
    c = mod.app.test_client()
    assert c.post("/", data={"username": user, "password": pw}).status_code == 302
    return c


def _start(c, course, room):
    return c.post("/take_attendance/start", data={"course_id": course, "faculty_code": "FST", "floor": "4", "room": room},
                  follow_redirects=True)


def test_room_lock_two_teachers(rooms_app):
    mod = rooms_app
    a, b = _login(mod, *TEACHER), _login(mod, *OTHER)
    admin = _login(mod, "superadmin", "bup@2024")
    r = _start(a, "ICE4204", "401")
    assert r.request.path == "/live_session"

    # B: room 401 shows as in use and cannot be started; another room can
    page = b.post("/take_attendance/select_location", data={"course_id": "ICE4205"}).get_data(as_text=True)
    floors = json.loads(re.search(r"const floorsInfo = (\{.*?\});\n", page).group(1))
    assert floors["4"]["busy"]["401"]["teacher_name"] == "Test Teacher" and "rtsp" not in page
    assert b.get("/api/room_status/FST").get_json()["4"]["401"]["course_id"] == "ICE4204"
    r = _start(b, "ICE4205", "401")
    assert r.request.path == "/take_attendance" and "Room 401 is in use" in r.get_data(as_text=True)
    assert list(mod._sessions) == ["FST/4/401"]
    r = _start(b, "ICE4205", "402")
    assert r.request.path == "/live_session" and sorted(mod._sessions) == ["FST/4/401", "FST/4/402"]

    # one class per teacher; each teacher sees only their own class
    r = _start(a, "ICE4204", "404")
    assert r.request.path == "/live_session" and "already have a class running in Room 401" in r.get_data(as_text=True)
    time.sleep(6)
    assert a.get("/live_stats").get_json()["total"] == 8 and b.get("/live_stats").get_json()["total"] == 8
    assert {s["course_id"] for s in admin.get("/api/stream_info").get_json()["sessions"].values()} == {"ICE4204", "ICE4205"}

    # room settings are locked while a class runs
    r = admin.post("/buildings/FST/floor/4/room/401", data={"camera1": "rtsp://x/1", "camera2": ""}, follow_redirects=True)
    assert "has a class running" in r.get_data(as_text=True)
    assert mod.load_db()["buildings"]["FST"]["floors"]["4"]["rooms"]["401"]["camera1"].startswith(RTSP)
    assert "In session" in admin.get("/buildings").get_data(as_text=True)

    # admin ends B's class: room free, B reviews and saves the attendance on the next page
    r = admin.post("/sessions/end", data={"room_key": "FST/4/402"}, follow_redirects=True)
    assert "Room 402" in r.get_data(as_text=True) and list(mod._sessions) == ["FST/4/401"]
    assert b.get("/live_stats").get_json() == {}
    r = b.get("/live_session", follow_redirects=True)
    assert r.request.path == "/confirm_attendance" and "ended by" in r.get_data(as_text=True)

    # A ends normally: every room free again
    assert a.post("/end_session").status_code == 302 and mod._sessions == {}
    page = b.post("/take_attendance/select_location", data={"course_id": "ICE4205"}).get_data(as_text=True)
    assert json.loads(re.search(r"const floorsInfo = (\{.*?\});\n", page).group(1))["4"]["busy"] == {}


def test_room_with_unreachable_cameras_is_refused(rooms_app):
    mod = rooms_app
    a = _login(mod, *TEACHER)
    t0 = time.time()
    r = _start(a, "ICE4204", "404")            # the default 10.10.4.x camera addresses: no route from here
    body = r.get_data(as_text=True)
    assert r.request.path == "/take_attendance" and "cannot reach either camera" in body, body[-2000:]
    assert mod._sessions == {} and time.time() - t0 < 8


def test_room_uses_chosen_stream_and_resolution(rooms_app):
    mod = rooms_app
    a = _login(mod, *TEACHER)
    assert _start(a, "ICE4204", "403").request.path == "/live_session"
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            cams = a.get("/api/stream_info").get_json()["cameras"]
            if all(v["frame_size"] for v in cams.values()) and all(v["frames_processed"] for v in cams.values()):
                break
            time.sleep(1)
        c1, c2 = cams["camera1"], cams["camera2"]
        assert c1["url"].endswith("/102") and not c1["url_rewritten"]          # the reported sub-stream URI
        assert c1["frame_size"] == "704x576" and c1["inference_size"] == "1024x576"    # un-squashed, not capped
        assert c2["url"].endswith("/201") and c2["frame_size"] == c2["inference_size"] == "1920x1080"
        assert c2["frames_processed"] > 0 and c2["faces_last_frame"] >= 0
    finally:
        a.post("/end_session")
    assert mod._sessions == {}
