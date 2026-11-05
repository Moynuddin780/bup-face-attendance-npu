"""Camera-side stream settings: read each stream's codec / resolution / frame rate and the
resolutions the camera offers, and change a stream's resolution — so the admin never has to log in
to the camera's own web page.

Two protocols, picked automatically per camera:
  * Hikvision ISAPI  (/ISAPI/Streaming/channels/N01 = main, N02 = sub), HTTP digest auth;
  * ONVIF            (Media2, Media1 as a fallback) — Dahua, Uniview, TP-Link, most other brands.
    Profile 1 is the main stream, profile 2 the sub-stream. WS-Security UsernameToken auth with the
    camera's clock offset applied; HTTP digest/basic is answered too when a camera asks for it.
The camera address and credentials come from its RTSP URL (rtsp://user:password@host:554/...).

Ported from the CCPLC backend (camera_control.cpp), where it was verified against Hikvision,
Dahua, Uniview and generic ONVIF cameras.
"""
import base64
import calendar
import hashlib
import http.client
import os
import re
import socket
import ssl
import threading
import time
from urllib.parse import unquote, urlsplit

STREAM_IDS = ("main", "sub", "third")


# ── just enough XML: elements by local name whatever their namespace prefix ─────────────────
def _find_start(x, name, start=0):
    for m in re.finditer(r"<([A-Za-z_][\w.\-]*:)?([\w.\-]+)(?=[\s/>])", x[start:]):
        if m.group(2) == name:
            return start + m.start()
    return -1


def _element_at(x, start, name):
    tag_end = x.find(">", start)
    if tag_end < 0:
        return ""
    if x[tag_end - 1] == "/":
        return x[start:tag_end + 1]
    depth, p = 1, tag_end + 1
    tag_re = re.compile(r"<(/?)([A-Za-z_][\w.\-]*:)?([\w.\-]+)[^>]*?(/?)>", re.S)
    while depth > 0:
        m = tag_re.search(x, p)
        if not m:
            return ""
        if m.group(3) == name:
            if m.group(1):
                depth -= 1
            elif not m.group(4):
                depth += 1
        p = m.end()
    return x[start:p]


def xml_all(x, name):
    out, p = [], _find_start(x, name)
    while p >= 0:
        el = _element_at(x, p, name)
        if not el:
            break
        out.append(el)
        p = _find_start(x, name, p + len(el))
    return out


def xml_first(x, name):
    p = _find_start(x, name)
    return _element_at(x, p, name) if p >= 0 else ""


def _unescape(s):
    for a, b in (("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&apos;", "'"), ("&amp;", "&")):
        s = s.replace(a, b)
    return s


def xml_escape(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def xml_text(x, name):
    el = xml_first(x, name)
    o, c = el.find(">"), el.rfind("</")
    if o < 0 or c <= o:
        return ""
    return _unescape(el[o + 1:c].strip())


def xml_attr(element, name):
    tag = element[:element.find(">")] if ">" in element else element
    m = re.search(r'(?:^|[\s])' + re.escape(name) + r'=(["\'])(.*?)\1', tag, re.S)
    return _unescape(m.group(2)) if m else ""


def xml_set_text(x, name, value):
    p = _find_start(x, name)
    if p < 0:
        return None
    o = x.find(">", p)
    c = x.find("</", o)
    if o < 0 or c < 0 or x[o - 1] == "/":
        return None
    return x[:o + 1] + xml_escape(value) + x[c:]


def _to_int(s, fallback=0):
    try:
        return int(float(s)) if s not in (None, "") else fallback
    except ValueError:
        return fallback


def _norm_codec(c):
    lc = (c or "").lower()
    if "265" in lc or "hevc" in lc:
        return "H265"
    if "264" in lc:
        return "H264"
    if "jpeg" in lc or "mjpg" in lc:
        return "MJPEG"
    return c or ""


# ── RTSP URL → camera address ────────────────────────────────────────────────────────────────
class Target:
    def __init__(self, host="", user="", password="", http_port=80, isapi_channel=1, rtsp_port=554):
        self.host, self.user, self.password = host, user, password
        self.http_port, self.isapi_channel, self.rtsp_port = http_port, isapi_channel, rtsp_port


def target_from_rtsp(url):
    t = Target()
    rest = url.split("://", 1)[1] if "://" in url else url
    cut = min([i for i in (rest.find("/"), rest.find("?")) if i >= 0], default=len(rest))
    authority, path = rest[:cut], rest[cut:]
    if "@" in authority:
        userinfo, authority = authority.rsplit("@", 1)
        user, _, password = userinfo.partition(":")
        t.user, t.password = unquote(user), unquote(password)
    if authority.startswith("["):
        t.host = authority[1:authority.find("]")]
        port = authority[authority.find("]") + 2:] if "]:" in authority else ""
    else:
        t.host, _, port = authority.partition(":")
    if port.isdigit():
        t.rtsp_port = int(port)
    lp = path.lower()
    m = re.search(r"/channels/(\d+)", lp) or re.search(r"/ch(\d+)", lp)   # Hikvision 101 / ch1
    if m:
        d = int(m.group(1))
        if d > 0:
            t.isapi_channel = d // 100 if d >= 100 else d
    return t


def stream_url(stored_url, camera_uri):
    """The camera-reported stream URI (no credentials), reached through the host, port and
    credentials of the stored RTSP URL — the address the admin knows to work."""
    if not camera_uri:
        return stored_url
    s, c = urlsplit(stored_url), urlsplit(camera_uri)
    path = c.path or "/"
    query = c.query.lstrip("?")   # Dahua reports ".../realmonitor??channel=1&subtype=1..."
    if query:
        path += "?" + query
    return f"{s.scheme or 'rtsp'}://{s.netloc}{path}"


HTTPS_PORTS = (443, 8443)
# Where cameras serve their web / ONVIF API when not on the RTSP URL's default port 80:
# HTTPS-only cameras, alternative HTTP ports, TP-Link (2020) and some ONVIF stacks (8899).
WEB_PORTS = (80, 443, 8080, 2020, 8899, 8443)


def _timeout_for(rtt, base=5.0):
    """HTTP timeout scaled to the camera's network round trip: a camera behind a congested
    Wi-Fi hop needs seconds per request."""
    return min(max(base, base + 8 * rtt), 45.0)


def link_timeout(host, port=80, base=5.0):
    """Timeout for one web port from a TCP handshake (≈ one round trip). None when nothing answers."""
    rtt = probe_ports(host, [port], base).get(port)
    return None if rtt is None else _timeout_for(rtt, base)


def probe_ports(host, ports, timeout=5.0):
    """{port: TCP handshake seconds or None}, all ports tried at once."""
    out = {}

    def one(port):
        t0 = time.monotonic()
        try:
            with socket.create_connection((host, port), timeout=timeout):
                out[port] = time.monotonic() - t0
        except OSError:
            out[port] = None
    threads = [threading.Thread(target=one, args=(p,), daemon=True) for p in dict.fromkeys(ports)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout + 1)
    return {p: out.get(p) for p in ports}


def _local_address(host):
    """This server's own address on the way to host (for the error message)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((host, 9))
            return s.getsockname()[0]
    except OSError:
        return "this server"


def _reach(t, http_port):
    """Pick the camera's web port and a timeout for it, or explain why there is none.
    Returns (port, timeout, error)."""
    ports = [http_port] + [p for p in WEB_PORTS if p != http_port] if http_port == 80 else [http_port]
    found = probe_ports(t.host, ports + [t.rtsp_port])
    for p in ports:
        if found.get(p) is not None:
            return p, _timeout_for(found[p]), ""
    me = _local_address(t.host)
    if found.get(t.rtsp_port) is None:
        return None, None, (f"the attendance server ({me}) cannot reach {t.host} at all — no answer on its RTSP port "
                            f"{t.rtsp_port} or web port {http_port}. If the camera works from another computer, "
                            f"this server is not on the camera's network (no route / VLAN); class sessions in this "
                            f"room cannot get video either")
    return None, None, (f"{t.host} answers on its RTSP port {t.rtsp_port}, but its web / ONVIF service is not "
                        f"reachable (tried ports {', '.join(map(str, ports))}) — enable ONVIF / HTTP on the camera to "
                        f"read and change its resolution here; choosing main / sub stream still works")


# ── HTTP with digest / basic auth ────────────────────────────────────────────────────────────
def digest_authorization(challenge, method, uri, user, password, nc, cnonce):
    p = {k.lower(): (v1 if v1 else v2) for k, v1, v2 in
         re.findall(r'(\w+)=(?:"([^"]*)"|([^,\s]*))', challenge.split(" ", 1)[1] if " " in challenge else "")}
    realm, nonce, algorithm = p.get("realm", ""), p.get("nonce", ""), p.get("algorithm", "")
    qop_auth = "auth" in [q.strip() for q in p.get("qop", "").lower().split(",")]
    md5 = lambda s: hashlib.md5(s.encode()).hexdigest()
    ha1 = md5(f"{user}:{realm}:{password}")
    if algorithm.lower() == "md5-sess":
        ha1 = md5(f"{ha1}:{nonce}:{cnonce}")
    ha2 = md5(f"{method}:{uri}")
    ncs = f"{nc:08x}"
    response = md5(f"{ha1}:{nonce}:{ncs}:{cnonce}:auth:{ha2}") if qop_auth else md5(f"{ha1}:{nonce}:{ha2}")
    h = f'Digest username="{user}", realm="{realm}", nonce="{nonce}", uri="{uri}", response="{response}"'
    if algorithm:
        h += f", algorithm={algorithm}"
    if qop_auth:
        h += f', qop=auth, nc={ncs}, cnonce="{cnonce}"'
    if "opaque" in p:
        h += f', opaque="{p["opaque"]}"'
    return h


class _Resp:
    def __init__(self, status=0, body="", error=""):
        self.status, self.body, self.error = status, body, error


class _Http:
    def __init__(self, host, port, user, password, timeout):
        self.host, self.port, self.user, self.password, self.timeout = host, port, user, password, timeout
        self.conn = None
        self.challenge = self.last_challenge = ""
        self.nc = 0

    def request(self, method, path, body="", content_type=""):
        r = self._send(method, path, body, content_type)
        if r.status == 401 and self.last_challenge:   # (re)authenticate once: new or stale nonce
            self.challenge, self.nc = self.last_challenge, 0
            r = self._send(method, path, body, content_type)
        return r

    def _send(self, method, path, body, content_type):
        headers = {}
        if self.challenge:
            if self.challenge.lower().startswith("basic"):
                headers["Authorization"] = "Basic " + base64.b64encode(f"{self.user}:{self.password}".encode()).decode()
            else:
                self.nc += 1
                cnonce = os.urandom(8).hex()
                headers["Authorization"] = digest_authorization(self.challenge, method, path, self.user,
                                                                self.password, self.nc, cnonce)
        if content_type:
            headers["Content-Type"] = content_type
        data = body.encode() if body else None
        for attempt in (0, 1):   # a kept-alive connection the camera closed: reconnect once
            try:
                if self.conn is None:
                    if self.port in HTTPS_PORTS:   # camera certificates are self-signed
                        self.conn = http.client.HTTPSConnection(self.host, self.port, timeout=self.timeout,
                                                                context=ssl._create_unverified_context())
                    else:
                        self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
                self.conn.request(method, path, body=data, headers=headers)
                res = self.conn.getresponse()
                text = res.read().decode("utf-8", "replace")
                break
            except (http.client.HTTPException, OSError) as e:
                if self.conn is not None:
                    self.conn.close()
                self.conn = None
                if attempt or isinstance(e, socket.timeout):
                    return _Resp(error=str(e) or type(e).__name__)
        r = _Resp(res.status, text)
        self.last_challenge = ""
        if r.status == 401:   # prefer Digest over Basic when the camera offers both
            for k, v in res.getheaders():
                if k.lower() != "www-authenticate":
                    continue
                if v.lower().startswith("digest"):
                    self.last_challenge = v
                elif not self.last_challenge and v.lower().startswith("basic"):
                    self.last_challenge = v
            if self.last_challenge == self.challenge and self.challenge and "stale=true" not in self.challenge.lower():
                self.last_challenge = ""   # same challenge refused again: wrong credentials, do not loop
        return r


def _split_xaddr(xaddr, fallback_port):
    """'http://host:port/path' -> (port, path); https XAddrs (Dahua) are reached over plain http."""
    if "://" not in xaddr:
        return fallback_port, xaddr if xaddr.startswith("/") else "/" + xaddr
    u = urlsplit(xaddr)
    port = fallback_port if u.scheme.lower() == "https" else (u.port or fallback_port)
    return port, (u.path or "/") + ("?" + u.query if u.query else "")


def _strip_credentials(uri):
    u = urlsplit(uri)
    if "@" not in u.netloc:
        return uri
    return uri.replace(u.netloc, u.netloc.rsplit("@", 1)[1], 1)


def _resolutions_in(x):
    out = []
    for r in xml_all(x, "ResolutionsAvailable"):
        res = (_to_int(xml_text(r, "Width")), _to_int(xml_text(r, "Height")))
        if res[0] > 0 and res[1] > 0 and res not in out:
            out.append(res)
    return sorted(out, key=lambda r: -r[0] * r[1])


def _fps_list(values):
    """Whole frame rates >= 1, highest first."""
    out = set()
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f >= 1:
            out.add(int(round(f)))
    return sorted(out, reverse=True)


def isapi_resolutions(caps_xml):
    """Paired opt lists of videoResolutionWidth / videoResolutionHeight in /capabilities."""
    ws = [_to_int(v) for v in xml_attr(xml_first(caps_xml, "videoResolutionWidth"), "opt").split(",")]
    hs = [_to_int(v) for v in xml_attr(xml_first(caps_xml, "videoResolutionHeight"), "opt").split(",")]
    out = []
    for r in zip(ws, hs):
        if r[0] > 0 and r[1] > 0 and r not in out:
            out.append(r)
    return sorted(out, key=lambda r: -r[0] * r[1])


_proto_lock = threading.Lock()
_proto = {}   # host -> protocol that worked last time (skips the ISAPI probe on ONVIF cameras)


def _new_info():
    return {"ok": False, "protocol": "", "vendor": "", "model": "", "streams": [], "error": ""}


# ── Hikvision ISAPI ──────────────────────────────────────────────────────────────────────────
class _Isapi:
    def __init__(self, t, timeout):
        self.t = t
        self.http = _Http(t.host, t.http_port, t.user, t.password, timeout)

    def detect(self, info):
        """1 = ISAPI camera, 0 = not ISAPI, -1 = ISAPI but refused, -2 = no web service."""
        r = self.http.request("GET", "/ISAPI/System/deviceInfo")
        if r.error:
            info["error"] = f"the camera's web service (port {self.t.http_port}) does not answer: {r.error}"
            return -2
        if r.status == 401 and "ISAPI" in r.body:
            info["protocol"] = "isapi"
            info["error"] = "Hikvision ISAPI refused the user name / password from the RTSP URL"
            return -1
        if r.status != 200 or _find_start(r.body, "DeviceInfo") < 0:
            return 0
        info["protocol"] = "isapi"
        info["vendor"] = xml_text(r.body, "manufacturer") or "Hikvision"
        info["model"] = xml_text(r.body, "model")
        return 1

    @staticmethod
    def _path(cid):
        return f"/ISAPI/Streaming/channels/{cid}"

    def read_streams(self, info):
        for k in range(2):
            cid = self.t.isapi_channel * 100 + k + 1
            r = self.http.request("GET", self._path(cid))
            if r.status != 200:
                continue
            quality = xml_text(r.body, "videoQualityControlType").lower()
            s = {"id": STREAM_IDS[k], "name": f"channel {cid}", "token": str(cid),
                 "codec": _norm_codec(xml_text(r.body, "videoCodecType")),
                 "width": _to_int(xml_text(r.body, "videoResolutionWidth")),
                 "height": _to_int(xml_text(r.body, "videoResolutionHeight")),
                 "fps": _to_int(xml_text(r.body, "maxFrameRate")) / 100.0,
                 "bitrate_kbps": _to_int(xml_text(r.body, "constantBitRate" if quality == "cbr" else "vbrUpperCap")),
                 "gop": _to_int(xml_text(r.body, "GovLength")),
                 "resolutions": [], "uri": f"rtsp://{self.t.host}:554/Streaming/Channels/{cid}"}
            caps = self.http.request("GET", self._path(cid) + "/capabilities")
            if caps.status == 200:
                s["resolutions"] = isapi_resolutions(caps.body)
                opt = xml_attr(xml_first(caps.body, "maxFrameRate"), "opt")
                s["fps_options"] = _fps_list(_to_int(v) / 100.0 for v in opt.split(",") if v.strip())
            info["streams"].append(s)
        info["ok"] = bool(info["streams"])
        if not info["ok"]:
            info["error"] = f"the camera did not report its streams (ISAPI channel {self.t.isapi_channel}01)"

    def set_resolution(self, stream, w, h, known=None, fps=None, bitrate=None, gop=None):
        cid = self.t.isapi_channel * 100 + (2 if stream == "sub" else 1)
        r = self.http.request("GET", self._path(cid))
        if r.status != 200:
            return {"ok": False, "error": f"could not read the camera's {stream} stream settings (HTTP {r.status})"}
        body = xml_set_text(r.body, "videoResolutionWidth", w)
        body = body and xml_set_text(body, "videoResolutionHeight", h)
        if body and fps:
            body = xml_set_text(body, "maxFrameRate", int(fps) * 100)   # ISAPI counts 1/100 fps
        if body and bitrate:   # whichever rate control the camera uses: the VBR ceiling and the CBR rate
            for tag in ("vbrUpperCap", "constantBitRate"):
                body = xml_set_text(body, tag, int(bitrate)) or body
        if body and gop:
            body = xml_set_text(body, "GovLength", int(gop)) or body
        if not body:
            return {"ok": False, "error": "the camera's stream settings have no resolution / frame-rate field"}
        r = self.http.request("PUT", self._path(cid), body, "application/xml")
        code = _to_int(xml_text(r.body, "statusCode"), -1)
        if r.status == 200 and code in (1, -1):
            return {"ok": True, "note": "the camera restarts this stream with the new resolution"}
        if code == 7:
            return {"ok": True, "note": "the camera needs a reboot to apply it"}
        why = xml_text(r.body, "subStatusCode") or xml_text(r.body, "statusString") or r.error
        return {"ok": False, "error": f"the camera refused the change (HTTP {r.status}{', ' + why if why else ''})"}


# ── ONVIF ────────────────────────────────────────────────────────────────────────────────────
class _Onvif:
    TT = "http://www.onvif.org/ver10/schema"

    def __init__(self, t, timeout):
        self.t, self.timeout = t, timeout
        self.clock_offset = 0
        self.media1 = self.media2 = (0, "")
        self.used_media2 = False
        self.clients = {}

    def _call(self, port, path, body, auth=True):
        http = self.clients.get(port)
        if http is None:
            http = self.clients[port] = _Http(self.t.host, port, self.t.user, self.t.password, self.timeout)
        header = ""
        if auth and self.t.user:
            nonce = os.urandom(16)
            created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + self.clock_offset))
            digest = base64.b64encode(hashlib.sha1(nonce + created.encode() + self.t.password.encode()).digest()).decode()
            header = ('<s:Header><Security s:mustUnderstand="1" xmlns="http://docs.oasis-open.org/wss/2004/01/'
                      'oasis-200401-wss-wssecurity-secext-1.0.xsd"><UsernameToken><Username>' + xml_escape(self.t.user) +
                      '</Username><Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-'
                      'token-profile-1.0#PasswordDigest">' + digest + '</Password><Nonce EncodingType="http://docs.'
                      'oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">' +
                      base64.b64encode(nonce).decode() + '</Nonce><Created xmlns="http://docs.oasis-open.org/wss/2004'
                      '/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">' + created +
                      '</Created></UsernameToken></Security></s:Header>')
        envelope = ('<?xml version="1.0" encoding="UTF-8"?><s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">'
                    + header + "<s:Body>" + body + "</s:Body></s:Envelope>")
        return http.request("POST", path, envelope, "application/soap+xml; charset=utf-8")

    @staticmethod
    def _fault(r):
        if r.error:
            return "the camera did not answer: " + r.error
        if r.status == 200:
            return ""
        low = r.body.lower()
        if r.status == 401 or "notauthorized" in low or "not authorized" in low or "failedauthentication" in low:
            return "ONVIF refused the user name / password from the RTSP URL (check them, and the camera's clock)"
        reason = xml_text(r.body, "Text")
        return f"the camera answered HTTP {r.status}" + (f": {reason}" if reason else "")

    def connect(self, info):
        dev = "/onvif/device_service"
        # The WS-Security timestamp must match the camera's clock: learn the offset first (no auth).
        tr = self._call(self.t.http_port, dev, '<GetSystemDateAndTime xmlns="http://www.onvif.org/ver10/device/wsdl"/>', False)
        if tr.error:
            info["error"] = f"the camera's web service (port {self.t.http_port}) does not answer: {tr.error}"
            return False
        if tr.status != 200:
            info["error"] = ("ONVIF is disabled on this camera — enable it in the camera's network settings"
                             if "ONVIF" in tr.body and "disabled" in tr.body.lower() else
                             f"the camera supports neither Hikvision ISAPI nor ONVIF (HTTP {tr.status})")
            return False
        utc = xml_first(tr.body, "UTCDateTime")
        if utc:
            y = _to_int(xml_text(utc, "Year"))
            if y > 2000:
                cam = calendar.timegm((y, _to_int(xml_text(utc, "Month")), _to_int(xml_text(utc, "Day")),
                                       _to_int(xml_text(utc, "Hour")), _to_int(xml_text(utc, "Minute")),
                                       _to_int(xml_text(utc, "Second")), 0, 0, 0))
                self.clock_offset = cam - time.time()
        svc = self._call(self.t.http_port, dev, '<GetServices xmlns="http://www.onvif.org/ver10/device/wsdl">'
                                                '<IncludeCapability>false</IncludeCapability></GetServices>')
        fault = self._fault(svc)
        if fault:
            info["error"] = fault
            return False
        for s in xml_all(svc.body, "Service"):
            ns, addr = xml_text(s, "Namespace"), xml_text(s, "XAddr")
            if "ver20/media/wsdl" in ns:
                self.media2 = _split_xaddr(addr, self.t.http_port)
            elif "ver10/media/wsdl" in ns:
                self.media1 = _split_xaddr(addr, self.t.http_port)
        if not self.media1[1] and not self.media2[1]:   # older cameras: GetCapabilities
            caps = self._call(self.t.http_port, dev, '<GetCapabilities xmlns="http://www.onvif.org/ver10/device/wsdl">'
                                                     '<Category>Media</Category></GetCapabilities>')
            media = xml_first(caps.body, "Media")
            if media:
                self.media1 = _split_xaddr(xml_text(media, "XAddr"), self.t.http_port)
        if not self.media1[1] and not self.media2[1]:
            info["error"] = "the camera has no ONVIF media service"
            return False
        di = self._call(self.t.http_port, dev, '<GetDeviceInformation xmlns="http://www.onvif.org/ver10/device/wsdl"/>')
        info["protocol"] = "onvif"
        info["vendor"] = xml_text(di.body, "Manufacturer")
        info["model"] = xml_text(di.body, "Model")
        return True

    def read_streams(self, info):
        if self.media2[1]:
            self._read_media2(info)
        if not info["streams"] and self.media1[1]:
            self._read_media1(info)
        info["ok"] = bool(info["streams"])
        if not info["ok"] and not info["error"]:
            info["error"] = "the camera reported no video encoder profiles"

    def _read_media2(self, info):
        port, path = self.media2
        pr = self._call(port, path, '<GetProfiles xmlns="http://www.onvif.org/ver20/media/wsdl"><Type>VideoEncoder</Type></GetProfiles>')
        fault = self._fault(pr)
        if fault:
            info["error"] = fault
            return
        for p in xml_all(pr.body, "Profiles"):
            enc = xml_first(p, "VideoEncoder")
            if not enc or len(info["streams"]) >= 3:
                continue
            encoding = xml_text(enc, "Encoding")
            s = {"id": STREAM_IDS[len(info["streams"])], "token": xml_attr(p, "token"), "name": xml_text(p, "Name"),
                 "encoder_token": xml_attr(enc, "token"), "codec": _norm_codec(encoding),
                 "width": _to_int(xml_text(enc, "Width")), "height": _to_int(xml_text(enc, "Height")),
                 "fps": _to_int(xml_text(enc, "FrameRateLimit")), "bitrate_kbps": _to_int(xml_text(enc, "BitrateLimit")),
                 "gop": _to_int(xml_attr(enc, "GovLength"))}
            op = self._call(port, path, '<GetVideoEncoderConfigurationOptions xmlns="http://www.onvif.org/ver20/media/wsdl">'
                                        '<ConfigurationToken>' + xml_escape(s["encoder_token"]) + '</ConfigurationToken>'
                                        '<ProfileToken>' + xml_escape(s["token"]) + '</ProfileToken>'
                                        '</GetVideoEncoderConfigurationOptions>')
            chosen = ""
            for o in xml_all(op.body, "Options"):
                same = xml_text(o, "Encoding").lower() == encoding.lower()
                if not chosen or same:
                    chosen = o
                if same:
                    break
            s["resolutions"] = _resolutions_in(chosen)
            s["fps_options"] = _fps_list((xml_text(chosen, "FrameRatesSupported") or xml_attr(chosen, "FrameRatesSupported")).split())
            su = self._call(port, path, '<GetStreamUri xmlns="http://www.onvif.org/ver20/media/wsdl"><Protocol>RtspUnicast'
                                        '</Protocol><ProfileToken>' + xml_escape(s["token"]) + '</ProfileToken></GetStreamUri>')
            s["uri"] = _strip_credentials(xml_text(su.body, "Uri"))
            info["streams"].append(s)
        self.used_media2 = bool(info["streams"])

    def _read_media1(self, info):
        port, path = self.media1
        pr = self._call(port, path, '<GetProfiles xmlns="http://www.onvif.org/ver10/media/wsdl"/>')
        fault = self._fault(pr)
        if fault:
            info["error"] = fault
            return
        for p in xml_all(pr.body, "Profiles"):
            enc = xml_first(p, "VideoEncoderConfiguration")
            if not enc or len(info["streams"]) >= 3:
                continue
            s = {"id": STREAM_IDS[len(info["streams"])], "token": xml_attr(p, "token"), "name": xml_text(p, "Name"),
                 "encoder_token": xml_attr(enc, "token"), "codec": _norm_codec(xml_text(enc, "Encoding")),
                 "width": _to_int(xml_text(enc, "Width")), "height": _to_int(xml_text(enc, "Height")),
                 "fps": _to_int(xml_text(enc, "FrameRateLimit")), "bitrate_kbps": _to_int(xml_text(enc, "BitrateLimit")),
                 "gop": _to_int(xml_text(xml_first(enc, "H264") or xml_first(enc, "MPEG4") or "", "GovLength"))}
            op = self._call(port, path, '<GetVideoEncoderConfigurationOptions xmlns="http://www.onvif.org/ver10/media/wsdl">'
                                        '<ConfigurationToken>' + xml_escape(s["encoder_token"]) + '</ConfigurationToken>'
                                        '<ProfileToken>' + xml_escape(s["token"]) + '</ProfileToken>'
                                        '</GetVideoEncoderConfigurationOptions>')
            block = xml_first(op.body, {"MJPEG": "JPEG", "H265": "H265"}.get(s["codec"], "H264"))
            s["resolutions"] = _resolutions_in(block or op.body)
            rng = xml_first(block or op.body, "FrameRateRange")
            lo, hi = _to_int(xml_text(rng, "Min"), 1), _to_int(xml_text(rng, "Max"), 0)
            s["fps_options"] = list(range(hi, max(1, lo) - 1, -1)) if hi >= 1 else []
            su = self._call(port, path, '<GetStreamUri xmlns="http://www.onvif.org/ver10/media/wsdl"><StreamSetup>'
                                        '<Stream xmlns="http://www.onvif.org/ver10/schema">RTP-Unicast</Stream>'
                                        '<Transport xmlns="http://www.onvif.org/ver10/schema"><Protocol>RTSP</Protocol>'
                                        '</Transport></StreamSetup><ProfileToken>' + xml_escape(s["token"]) +
                                        '</ProfileToken></GetStreamUri>')
            s["uri"] = _strip_credentials(xml_text(su.body, "Uri"))
            info["streams"].append(s)

    @staticmethod
    def _tt(name, value):
        return f"<tt:{name}>{xml_escape(value)}</tt:{name}>"

    def _multicast(self, cfg):
        m = xml_first(cfg, "Multicast")
        if not m:
            return ""
        addr = xml_first(m, "Address")
        out = "<tt:Multicast><tt:Address>" + self._tt("Type", xml_text(addr, "Type"))
        v4, v6 = xml_text(addr, "IPv4Address"), xml_text(addr, "IPv6Address")
        if v4 or not v6:
            out += self._tt("IPv4Address", v4 or "0.0.0.0")
        if v6:
            out += self._tt("IPv6Address", v6)
        return (out + "</tt:Address>" + self._tt("Port", xml_text(m, "Port")) + self._tt("TTL", xml_text(m, "TTL")) +
                self._tt("AutoStart", xml_text(m, "AutoStart")) + "</tt:Multicast>")

    def _finish(self, r):
        fault = self._fault(r)
        if fault:
            return {"ok": False, "error": "the camera refused the change — " + fault}
        return {"ok": True, "note": "the camera restarts this stream with the new resolution"}

    def set_resolution(self, stream, w, h, known=None, fps=None, bitrate=None, gop=None):
        s = known
        if s is None or not s.get("encoder_token"):
            info = _new_info()
            self.read_streams(info)
            s = next((x for x in info["streams"] if x["id"] == stream), None)
        if s is None:
            return {"ok": False, "error": f"the camera has no {stream} stream"}
        return (self._set_media2(s["encoder_token"], w, h, fps, bitrate, gop) if self.used_media2
                else self._set_media1(s["encoder_token"], w, h, fps, bitrate, gop))

    def _set_media2(self, token, w, h, fps=None, bitrate=None, gop=None):
        port, path = self.media2
        g = self._call(port, path, '<GetVideoEncoderConfigurations xmlns="http://www.onvif.org/ver20/media/wsdl">'
                                   '<ConfigurationToken>' + xml_escape(token) + '</ConfigurationToken></GetVideoEncoderConfigurations>')
        cfg = xml_first(g.body, "Configurations")
        if not cfg:
            return {"ok": False, "error": "could not read the encoder configuration — " + self._fault(g)}
        attrs = f' token="{xml_escape(token)}"'
        for a in ("GovLength", "Profile"):
            v = str(int(gop)) if a == "GovLength" and gop else xml_attr(cfg, a)
            if v:
                attrs += f' {a}="{xml_escape(v)}"'
        body = (f'<SetVideoEncoderConfiguration xmlns="http://www.onvif.org/ver20/media/wsdl" xmlns:tt="{self.TT}">'
                f'<Configuration{attrs}>' + self._tt("Name", xml_text(cfg, "Name")) + self._tt("UseCount", xml_text(cfg, "UseCount")) +
                self._tt("Encoding", xml_text(cfg, "Encoding")) + "<tt:Resolution>" + self._tt("Width", w) +
                self._tt("Height", h) + "</tt:Resolution>")
        rc = xml_first(cfg, "RateControl")
        if rc:
            cbr = xml_attr(rc, "ConstantBitRate")
            body += ("<tt:RateControl" + (f' ConstantBitRate="{cbr}"' if cbr else "") + ">" +
                     self._tt("FrameRateLimit", fps or xml_text(rc, "FrameRateLimit")) +
                     self._tt("BitrateLimit", bitrate or xml_text(rc, "BitrateLimit")) + "</tt:RateControl>")
        body += self._multicast(cfg) + self._tt("Quality", xml_text(cfg, "Quality")) + "</Configuration></SetVideoEncoderConfiguration>"
        return self._finish(self._call(port, path, body))

    def _set_media1(self, token, w, h, fps=None, bitrate=None, gop=None):
        port, path = self.media1
        g = self._call(port, path, '<GetVideoEncoderConfiguration xmlns="http://www.onvif.org/ver10/media/wsdl">'
                                   '<ConfigurationToken>' + xml_escape(token) + '</ConfigurationToken></GetVideoEncoderConfiguration>')
        cfg = xml_first(g.body, "Configuration")
        if not cfg:
            return {"ok": False, "error": "could not read the encoder configuration — " + self._fault(g)}
        body = (f'<SetVideoEncoderConfiguration xmlns="http://www.onvif.org/ver10/media/wsdl" xmlns:tt="{self.TT}">'
                f'<Configuration token="{xml_escape(token)}">' + self._tt("Name", xml_text(cfg, "Name")) +
                self._tt("UseCount", xml_text(cfg, "UseCount")) + self._tt("Encoding", xml_text(cfg, "Encoding")) +
                "<tt:Resolution>" + self._tt("Width", w) + self._tt("Height", h) + "</tt:Resolution>" +
                self._tt("Quality", xml_text(cfg, "Quality")))
        rc = xml_first(cfg, "RateControl")
        if rc:
            body += ("<tt:RateControl>" + self._tt("FrameRateLimit", fps or xml_text(rc, "FrameRateLimit")) +
                     self._tt("EncodingInterval", xml_text(rc, "EncodingInterval")) +
                     self._tt("BitrateLimit", bitrate or xml_text(rc, "BitrateLimit")) + "</tt:RateControl>")
        m4 = xml_first(cfg, "MPEG4")
        if m4:
            body += "<tt:MPEG4>" + self._tt("GovLength", gop or xml_text(m4, "GovLength")) + self._tt("Mpeg4Profile", xml_text(m4, "Mpeg4Profile")) + "</tt:MPEG4>"
        h264 = xml_first(cfg, "H264")
        if h264:
            body += "<tt:H264>" + self._tt("GovLength", gop or xml_text(h264, "GovLength")) + self._tt("H264Profile", xml_text(h264, "H264Profile")) + "</tt:H264>"
        body += (self._multicast(cfg) + self._tt("SessionTimeout", xml_text(cfg, "SessionTimeout")) +
                 "</Configuration><ForcePersistence>true</ForcePersistence></SetVideoEncoderConfiguration>")
        return self._finish(self._call(port, path, body))


# ── public API ───────────────────────────────────────────────────────────────────────────────
def _open(rtsp_url, timeout, http_port):
    """(info, protocol handler). The handler keeps its authenticated connection, so a follow-up
    resolution change costs no extra round trips — slow camera links take seconds per request."""
    info = _new_info()
    t = target_from_rtsp(rtsp_url or "")
    t.http_port = http_port
    if not t.host:
        info["error"] = "enter the camera's RTSP URL first"
        return info, None
    if timeout is None:
        t.http_port, timeout, err = _reach(t, http_port)
        if err:
            info["error"] = err
            return info, None
    info["web_port"] = t.http_port
    with _proto_lock:
        known = _proto.get(t.host)
    if known != "onvif":
        isapi = _Isapi(t, timeout)
        found = isapi.detect(info)
        if found == -2:
            return info, None
        if found != 0:
            if found == 1:
                isapi.read_streams(info)
            if info["ok"]:
                with _proto_lock:
                    _proto[t.host] = "isapi"
            return info, isapi
    info = _new_info()
    onvif = _Onvif(t, timeout)
    if not onvif.connect(info):
        return info, None
    onvif.read_streams(info)
    info["web_port"] = t.http_port
    if info["ok"]:
        with _proto_lock:
            _proto[t.host] = "onvif"
    return info, onvif


def rtsp_check(rtsp_url, timeout=6.0):
    """Log in to the RTSP stream itself (DESCRIBE, Digest or Basic with the URL's user and password).
    The settings check above needs the camera's web port; a camera reached through port forwarding
    often has only its RTSP port open, and then this is the only thing that can tell "the stream
    works" from "wrong password" (401) or "wrong path" (404).
    {ok, status, codecs, error}"""
    t = target_from_rtsp(rtsp_url or "")
    out = {"ok": False, "status": 0, "codecs": [], "error": ""}
    if not t.host:
        out["error"] = "enter the camera's RTSP URL first"
        return out
    rest = rtsp_url.split("://", 1)[1]
    authority_end = min([i for i in (rest.find("/"), rest.find("?")) if i >= 0], default=len(rest))
    host_port = rest[:authority_end].rsplit("@", 1)[-1]
    uri = f"rtsp://{host_port}{rest[authority_end:] or '/'}"   # what the camera sees: no credentials

    def request(sock, cseq, auth=None):
        req = f"DESCRIBE {uri} RTSP/1.0\r\nCSeq: {cseq}\r\nAccept: application/sdp\r\nUser-Agent: BUP-Attendance\r\n"
        if auth:
            req += f"Authorization: {auth}\r\n"
        sock.sendall((req + "\r\n").encode())
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
        head, _, body = data.partition(b"\r\n\r\n")
        lines = head.decode(errors="replace").split("\r\n")
        status = _to_int(lines[0].split(" ")[1]) if len(lines[0].split(" ")) > 1 else 0
        headers = {}
        for ln in lines[1:]:
            k, _, v = ln.partition(":")
            headers.setdefault(k.strip().lower(), []).append(v.strip())
        want = _to_int((headers.get("content-length") or ["0"])[0])
        while len(body) < want:
            chunk = sock.recv(4096)
            if not chunk:
                break
            body += chunk
        return status, headers, body.decode(errors="replace")

    try:
        with socket.create_connection((t.host, t.rtsp_port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            status, headers, body = request(sock, 1)
            if status == 401 and (t.user or t.password):
                challenges = headers.get("www-authenticate", [])
                digest = next((c for c in challenges if c.lower().startswith("digest")), None)
                if digest:
                    auth = digest_authorization(digest, "DESCRIBE", uri, t.user, t.password, 1, os.urandom(8).hex())
                else:
                    auth = "Basic " + base64.b64encode(f"{t.user}:{t.password}".encode()).decode()
                status, headers, body = request(sock, 2, auth)
    except (socket.timeout, TimeoutError):
        out["error"] = f"the RTSP port {t.rtsp_port} of {t.host} does not answer"
        return out
    except OSError as e:
        out["error"] = f"cannot connect to {t.host}:{t.rtsp_port} ({e.strerror or e})"
        return out
    out["status"] = status
    out["codecs"] = sorted({m.upper() for m in re.findall(r"a=rtpmap:\d+\s+([A-Za-z0-9]+)/", body)} - {"PCMA", "PCMU", "MPEG4-GENERIC"})
    if status == 200:
        out["ok"] = True
    elif status == 401:
        out["error"] = ("the camera refused the user name / password in the URL (RTSP 401)" if t.user or t.password
                        else "the camera needs a user name and password in the URL (RTSP 401)")
    elif status == 404:
        out["error"] = "the camera has no stream at this path (RTSP 404) — check the part after the port"
    else:
        out["error"] = f"the camera answered RTSP {status or 'nothing'}"
    return out


def read_camera(rtsp_url, timeout=None, http_port=80):
    """{ok, protocol, vendor, model, streams: [{id, name, codec, width, height, fps, bitrate_kbps,
    resolutions: [(w, h), ...], uri}], error}. http_port: the camera's web port."""
    return _open(rtsp_url, timeout, http_port)[0]


def public_info(info):
    """read_camera() output for the browser: protocol handles dropped, resolutions as lists."""
    out = {k: info[k] for k in ("ok", "protocol", "vendor", "model", "error")}
    out["web_port"] = info.get("web_port")
    out["streams"] = [{k: s.get(k) for k in ("id", "name", "codec", "width", "height", "fps", "bitrate_kbps", "uri",
                                             "fps_options", "gop")}
                      | {"resolutions": [list(r) for r in s.get("resolutions", [])]} for s in info["streams"]]
    return out


def configure_stream(rtsp_url, stream, width=None, height=None, timeout=None, http_port=80, fps=None,
                     bitrate_kbps=None, gop=None):
    """Read the camera and, when a resolution, frame rate, bitrate (kbit/s ceiling) or I-frame interval
    (GOP, frames) is given, switch that stream to it on the camera (one request). Returns (info,
    result): info as read_camera() with the new values applied, result {ok, error | note, unchanged?}."""
    if stream not in ("main", "sub"):
        return _new_info(), {"ok": False, "error": "stream must be main or sub"}
    info, handler = _open(rtsp_url, timeout, http_port)
    if not info["ok"]:
        return info, {"ok": False, "error": info["error"] or "cannot read the camera's stream settings"}
    s = next((x for x in info["streams"] if x["id"] == stream), None)
    if s is None:
        return info, {"ok": False, "error": f"the camera has no {stream} stream"}
    new_res = bool(width and height) and (s["width"], s["height"]) != (width, height)
    new_fps = bool(fps) and round(s.get("fps") or 0) != int(fps)
    new_br = bool(bitrate_kbps) and (s.get("bitrate_kbps") or 0) != int(bitrate_kbps)
    new_gop = bool(gop) and (s.get("gop") or 0) != int(gop)
    if not new_res and not new_fps and not new_br and not new_gop:
        return info, {"ok": True, "note": "already set", "unchanged": True}
    if new_res and s["resolutions"] and (width, height) not in s["resolutions"]:
        return info, {"ok": False, "error": f"the camera does not offer {width}×{height} on its {stream} stream"}
    if new_fps and s.get("fps_options") and int(fps) not in s["fps_options"]:
        return info, {"ok": False, "error": f"the camera does not offer {fps} fps on its {stream} stream"}
    w, h = (width, height) if new_res else (s["width"], s["height"])
    r = handler.set_resolution(stream, w, h, s, int(fps) if new_fps else None,
                               int(bitrate_kbps) if new_br else None, int(gop) if new_gop else None)
    if r["ok"]:
        s["width"], s["height"] = w, h
        if new_fps:
            s["fps"] = int(fps)
        if new_br:
            s["bitrate_kbps"] = int(bitrate_kbps)
        if new_gop:
            s["gop"] = int(gop)
        r["note"] = "the camera restarts this stream with the new settings"
    return info, r


def set_resolution(rtsp_url, stream, width, height, timeout=None, http_port=80):
    """Change one stream's resolution on the camera. {ok, error | note}"""
    return configure_stream(rtsp_url, stream, width, height, timeout, http_port)[1]
