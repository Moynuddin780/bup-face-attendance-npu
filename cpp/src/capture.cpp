#include "bup/capture.hpp"

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <fcntl.h>
#include <netdb.h>
#include <dirent.h>
#include <poll.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "bup/common.hpp"
#include "bup/stream_url.hpp"

namespace bup {

namespace {

std::once_flag g_gst_once;
bool g_gst_ok = false;

void ensure_gst() {
    std::call_once(g_gst_once, [] {
        GError* err = nullptr;
        g_gst_ok = gst_init_check(nullptr, nullptr, &err) == TRUE;
        if (err) {
            log(LogLevel::Warn, std::string("GStreamer init failed: ") + err->message);
            g_error_free(err);
        }
    });
}

bool has_element(const char* name) {
    GstElementFactory* f = gst_element_factory_find(name);
    if (!f) return false;
    gst_object_unref(f);
    return true;
}

/// Pops one ERROR/EOS message; returns true (and a description) if the pipeline is dead.
/// *codec_mismatch tells a decoder chain that does not fit the stream (rtph264depay on an H.265
/// camera ends in "not-linked") from a network/RTSP failure (404, timeout, reset): a codec
/// mismatch is cured by another pipeline, a network failure is not.
bool pipeline_failed(GstElement* pipeline, std::string* why, bool* codec_mismatch = nullptr) {
    GstBus* bus = gst_element_get_bus(pipeline);
    GstMessage* msg = gst_bus_pop_filtered(bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    gst_object_unref(bus);
    if (!msg) return false;
    if (why || codec_mismatch) {
        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
            GError* err = nullptr;
            gchar* dbg = nullptr;
            gst_message_parse_error(msg, &err, &dbg);
            const std::string text = std::string(err ? err->message : "error") + " " + (dbg ? dbg : "");
            if (why) *why = err ? err->message : "error";
            if (codec_mismatch) {
                *codec_mismatch = text.find("not-linked") != std::string::npos ||
                                  text.find("not-negotiated") != std::string::npos ||
                                  text.find("GstVa") != std::string::npos ||      // VA decoder / post-processor
                                  text.find("vapostproc") != std::string::npos ||
                                  text.find("no suitable plugins") != std::string::npos ||
                                  text.find("missing a plug-in") != std::string::npos;
            }
            if (err) g_error_free(err);
            g_free(dbg);
        } else {
            if (why) *why = "end of stream";
            if (codec_mismatch) *codec_mismatch = false;
        }
    }
    gst_message_unref(msg);
    return true;
}

/// rtspsrc "select-stream": SETUP only the video track. Cameras also offer audio and ONVIF
/// metadata tracks; the pipeline never reads them, yet each one costs a SETUP round trip and,
/// on cameras with a tiny viewer limit, an extra stream slot.
gboolean select_video_stream(GstElement*, guint, GstCaps* caps, gpointer) {
    if (!caps || gst_caps_is_empty(caps)) return TRUE;
    const gchar* media = gst_structure_get_string(gst_caps_get_structure(caps, 0), "media");
    return !media || std::strcmp(media, "video") == 0;
}

/// Camera-reported codec ("H265", "hevc", "H.264", ...) -> the candidate pipeline name.
std::string codec_pipeline_name(std::string codec) {
    std::string c;
    for (char ch : codec) {
        if (std::isalnum(static_cast<unsigned char>(ch))) c += static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    if (c == "H264" || c == "AVC") return "H.264";
    if (c == "H265" || c == "HEVC") return "H.265";
    return std::string();
}

/// Duration of one TCP handshake with the camera's RTSP port, i.e. about one network round
/// trip. -1 when the port cannot be reached within timeout_ms.
double probe_rtt_ms(const std::string& url, int timeout_ms) {
    std::string auth = url;
    const auto scheme = auth.find("://");
    if (scheme != std::string::npos) auth = auth.substr(scheme + 3);
    auth = auth.substr(0, auth.find_first_of("/?#"));
    const auto at = auth.rfind('@');
    if (at != std::string::npos) auth = auth.substr(at + 1);  // drop user:password
    std::string host = auth, port = "554";
    if (!auth.empty() && auth[0] == '[') {  // [IPv6]:port
        const auto close = auth.find(']');
        host = auth.substr(1, close == std::string::npos ? std::string::npos : close - 1);
        if (close != std::string::npos && close + 2 < auth.size() && auth[close + 1] == ':') {
            port = auth.substr(close + 2);
        }
    } else {
        const auto colon = auth.rfind(':');
        if (colon != std::string::npos) {
            host = auth.substr(0, colon);
            port = auth.substr(colon + 1);
        }
    }
    if (host.empty()) return -1.0;

    addrinfo hints{};
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || !res) return -1.0;
    double rtt = -1.0;
    const int fd = socket(res->ai_family, res->ai_socktype | SOCK_CLOEXEC, res->ai_protocol);
    if (fd >= 0) {
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
        const double t0 = mono_ms();
        int rc = connect(fd, res->ai_addr, res->ai_addrlen);
        if (rc != 0 && errno == EINPROGRESS) {
            pollfd pfd{fd, POLLOUT, 0};
            rc = poll(&pfd, 1, timeout_ms) == 1 ? 0 : -1;
            int err = 0;
            socklen_t len = sizeof(err);
            if (rc == 0 && (getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len) != 0 || err != 0)) rc = -1;
        }
        if (rc == 0) rtt = mono_ms() - t0;
        close(fd);
    }
    freeaddrinfo(res);
    return rtt;
}

// Link-scaled limits (rtt in ms). An RTSP connect is ~6 requests (OPTIONS, DESCRIBE, the
// authenticated DESCRIBE, SETUP, PLAY) plus the wait for a key frame.
double connect_deadline_ms(double rtt) { return std::min(std::max(10000.0, 10000.0 + 8.0 * rtt), 90000.0); }
long long socket_timeout_us(double rtt) {   // a Wi-Fi stall of a few seconds must not end the connection
    return static_cast<long long>(std::min(std::max(12000.0, 8.0 * rtt), 60000.0) * 1000.0);
}
int jitter_latency_ms(int floor_ms, double rtt) {
    return std::max(floor_ms, static_cast<int>(std::min(2.0 * rtt, 2000.0)));
}
double stall_timeout_s(double floor_s, double rtt) { return std::max(floor_s, std::min(3.0 + 4.0 * rtt / 1000.0, 60.0)); }

std::string rtt_note(double rtt) {
    if (rtt < 200.0) return std::string();
    return " (slow network: RTT " + std::to_string(static_cast<int>(rtt)) + " ms)";
}

/// One appsink sample -> an owned BGR image (CPU pipelines deliver BGR, GPU ones BGRx).
bool sample_to_bgr(GstSample* sample, cv::Mat& frame) {
    bool ok = false;
    GstCaps* caps = gst_sample_get_caps(sample);
    GstBuffer* buf = gst_sample_get_buffer(sample);
    GstVideoInfo vi;
    if (caps && buf && gst_video_info_from_caps(&vi, caps)) {
        GstVideoFrame vf;   // honours the buffer's own plane offsets / strides (GstVideoMeta)
        if (gst_video_frame_map(&vf, &vi, buf, GST_MAP_READ)) {
            const int w = GST_VIDEO_FRAME_WIDTH(&vf), h = GST_VIDEO_FRAME_HEIGHT(&vf);
            const auto plane = [&](int i, int rows, int cols, int type) {
                return cv::Mat(rows, cols, type, GST_VIDEO_FRAME_PLANE_DATA(&vf, i),
                               static_cast<std::size_t>(GST_VIDEO_FRAME_PLANE_STRIDE(&vf, i)));
            };
            if (w > 0 && h > 0) {
                switch (GST_VIDEO_FRAME_FORMAT(&vf)) {
                    case GST_VIDEO_FORMAT_BGR:   // CPU pipelines (videoconvert): own the pixels
                        plane(0, h, w, CV_8UC3).copyTo(frame);
                        ok = true;
                        break;
                    case GST_VIDEO_FORMAT_NV12:
                        cv::cvtColorTwoPlane(plane(0, h, w, CV_8UC1), plane(1, h / 2, w / 2, CV_8UC2), frame,
                                             cv::COLOR_YUV2BGR_NV12);
                        ok = true;
                        break;
                    case GST_VIDEO_FORMAT_BGRx:  // GPU (vapostproc)
                    case GST_VIDEO_FORMAT_BGRA:
                        cv::cvtColor(plane(0, h, w, CV_8UC4), frame, cv::COLOR_BGRA2BGR);
                        ok = true;
                        break;
                    default:
                        break;
                }
            }
            gst_video_frame_unmap(&vf);
        }
    }
    return ok;
}

std::string file_name(const std::string& path) {
    const auto slash = path.find_last_of('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

/// The video decoder decodebin picked, as a codec name ("H.264", "H.265", ...).
std::string video_codec_of(GstElement* decodebin) {
    std::string codec;
    GstIterator* it = gst_bin_iterate_recurse(GST_BIN(decodebin));
    GValue item = G_VALUE_INIT;
    while (codec.empty() && gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
        GstElement* el = GST_ELEMENT(g_value_get_object(&item));
        GstElementFactory* f = gst_element_get_factory(el);
        const gchar* klass = f ? gst_element_factory_get_metadata(f, GST_ELEMENT_METADATA_KLASS) : nullptr;
        if (klass && std::strstr(klass, "Decoder") && std::strstr(klass, "Video")) {
            if (GstPad* pad = gst_element_get_static_pad(el, "sink")) {
                if (GstCaps* caps = gst_pad_get_current_caps(pad)) {
                    const GstStructure* st = gst_caps_get_structure(caps, 0);
                    const std::string name = gst_structure_get_name(st);
                    int ver = 0;
                    gst_structure_get_int(st, "mpegversion", &ver);
                    if (name == "video/x-h264") codec = "H.264";
                    else if (name == "video/x-h265") codec = "H.265";
                    else if (name == "video/mpeg") codec = ver == 4 ? "MPEG-4" : "MPEG-" + std::to_string(ver);
                    else if (name == "video/x-vp8") codec = "VP8";
                    else if (name == "video/x-vp9") codec = "VP9";
                    else if (name == "video/x-av1") codec = "AV1";
                    else if (name == "image/jpeg") codec = "MJPEG";
                    gst_caps_unref(caps);
                }
                gst_object_unref(pad);
            }
            if (codec.empty()) codec = GST_OBJECT_NAME(f);
        }
        g_value_reset(&item);
    }
    g_value_unset(&item);
    gst_iterator_free(it);
    return codec;
}

}  // namespace

VideoProbe probe_video(const std::string& path, double timeout_s) {
    VideoProbe r;
    ensure_gst();
    if (!g_gst_ok) {
        r.error = "GStreamer is not available";
        return r;
    }
    GError* err = nullptr;
    GstElement* p = gst_parse_launch("filesrc name=src ! decodebin name=dec force-sw-decoders=true ! videoconvert ! "
                                     "video/x-raw,format=BGR ! appsink name=sink sync=false max-buffers=1", &err);
    if (err) {
        r.error = err->message;
        g_error_free(err);
        if (p) gst_object_unref(p);
        return r;
    }
    GstElement* src = gst_bin_get_by_name(GST_BIN(p), "src");
    g_object_set(src, "location", path.c_str(), nullptr);
    gst_object_unref(src);
    GstElement* sink = gst_bin_get_by_name(GST_BIN(p), "sink");
    GstElement* dec = gst_bin_get_by_name(GST_BIN(p), "dec");
    gst_element_set_state(p, GST_STATE_PAUSED);
    GstSample* sample = nullptr;
    const double deadline = mono_ms() + timeout_s * 1000.0;
    while (!sample && mono_ms() < deadline) {   // preroll: the first decoded frame
        sample = gst_app_sink_try_pull_preroll(GST_APP_SINK(sink), 200 * GST_MSECOND);
        if (sample) break;
        std::string why;
        if (pipeline_failed(p, &why)) {
            r.error = why == "end of stream" ? "the file has no video track" : why;
            break;
        }
    }
    if (sample) {
        if (GstCaps* caps = gst_sample_get_caps(sample)) {
            const GstStructure* st = gst_caps_get_structure(caps, 0);
            gst_structure_get_int(st, "width", &r.width);
            gst_structure_get_int(st, "height", &r.height);
            int num = 0, den = 1;
            if (gst_structure_get_fraction(st, "framerate", &num, &den) && num > 0 && den > 0) {
                r.fps = static_cast<double>(num) / den;
            }
        }
        sample_to_bgr(sample, r.first);
        gst_sample_unref(sample);
        gint64 dur = 0;
        if (gst_element_query_duration(p, GST_FORMAT_TIME, &dur) && dur > 0) r.duration_s = dur / 1e9;
        r.codec = video_codec_of(dec);
        r.ok = r.width > 0 && r.height > 0 && !r.first.empty();
        if (!r.ok) r.error = "no picture could be decoded";
    } else if (r.error.empty()) {
        r.error = "not a playable video (no frame within " + std::to_string(static_cast<int>(timeout_s)) + " s)";
    }
    gst_element_set_state(p, GST_STATE_NULL);
    gst_object_unref(dec);
    gst_object_unref(sink);
    gst_object_unref(p);
    return r;
}

bool gstreamer_available() {
    ensure_gst();
    return g_gst_ok && has_element("rtspsrc") && has_element("appsink") && has_element("videoconvert");
}

bool gpu_decoding_available() {
    static std::once_flag once;
    static bool ok = false;
    std::call_once(once, [] {
        if (!gstreamer_available() || !has_element("vapostproc") || !has_element("vah264dec")) {
            log(LogLevel::Info, "GPU decoding (VA-API) not available: GStreamer VA elements missing — CPU decoding");
            return;
        }
        // a tiny pipeline through the GPU post-processor proves the driver and the render node
        GError* err = nullptr;
        GstElement* p = gst_parse_launch("videotestsrc num-buffers=3 ! video/x-raw,width=320,height=240 ! vapostproc ! "
                                         "video/x-raw,format=BGRx,width=160,height=120 ! fakesink", &err);
        if (err) g_error_free(err);
        if (!p) return;
        gst_element_set_state(p, GST_STATE_PLAYING);
        GstBus* bus = gst_element_get_bus(p);
        GstMessage* m = gst_bus_timed_pop_filtered(bus, 5 * GST_SECOND,
                                                   static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
        ok = m && GST_MESSAGE_TYPE(m) == GST_MESSAGE_EOS;
        if (m) gst_message_unref(m);
        gst_object_unref(bus);
        gst_element_set_state(p, GST_STATE_NULL);
        gst_object_unref(p);
        log(LogLevel::Info, ok ? "GPU decoding (VA-API) available" : "GPU decoding (VA-API) self-test failed — CPU decoding");
    });
    return ok;
}

struct RtspCapture::GstSource {
    GstElement* pipeline{nullptr};
    GstAppSink* sink{nullptr};       ///< decoded frames (for a GPU receiver: its decoder's sink)
    GstElement* filter{nullptr};     ///< VA pipelines: output caps of the GPU post-processor
    std::string name;
    int max_width{0};                ///< VA pipelines: scale wider streams on the GPU to this width
    std::atomic<int> native_w{0}, native_h{0};
    // GPU receiver (rtspsrc -> depay -> parse -> appsink "es"): feeds the persistent decoder
    GstSource* dec{nullptr};         ///< the decoder it feeds (owned by RtspCapture::va_dec_)
    GstElement* feed{nullptr};       ///< that decoder's appsrc (own ref)
    bool key_seen{false};            ///< streaming thread only: a key frame was passed on
    std::atomic<std::uint64_t> bytes{0}, pushed{0}, gaps{0}, skipped{0};   ///< receiver statistics
    std::atomic<bool> size_changed{false};   ///< the camera now sends another picture size (see on_es_sample)
    std::atomic<int> new_w{0}, new_h{0};
    ~GstSource() {
        if (pipeline) {   // first: no more streaming-thread callbacks into `feed`
            gst_element_set_state(pipeline, GST_STATE_NULL);
            gst_object_unref(pipeline);
        }
        if (feed) gst_object_unref(feed);
        if (filter) gst_object_unref(filter);
        if (sink) gst_object_unref(sink);
    }
};

RtspCapture::RtspCapture(CaptureConfig cfg) : cfg_(std::move(cfg)) {
    tag_ = "Camera " + cfg_.cam_id;
    if (!cfg_.file.empty()) {   // recorded video: no URL, no stream rules, no aspect probe
        tag_ += " (" + file_name(cfg_.file) + ")";
        url_ = cfg_.file;
        aspect_resolved_ = true;
        info_.cam_id = cfg_.cam_id;
        info_.profile = "video";
        info_.url_masked = "video: " + file_name(cfg_.file);
        return;
    }
    bool vendor_known = cfg_.fixed_url;
    url_ = cfg_.fixed_url ? cfg_.url : resolve_stream_url(cfg_.url, cfg_.profile, &rewritten_, &vendor_known);
    hint_ = codec_pipeline_name(cfg_.codec_hint);
    info_.cam_id = cfg_.cam_id;
    info_.profile = cfg_.profile;
    info_.url_rewritten = rewritten_;
    info_.url_masked = mask_credentials(url_);
    {   // the camera's address in every log line: several rooms' sessions log at once
        std::string host = url_.substr(url_.find("://") == std::string::npos ? 0 : url_.find("://") + 3);
        host = host.substr(0, host.find_first_of("/?#"));
        host = host.substr(host.rfind('@') == std::string::npos ? 0 : host.rfind('@') + 1);
        if (!host.empty()) tag_ += " (" + host + ")";
    }
    if (cfg_.profile == "sub" && !vendor_known) {
        log(LogLevel::Warn, tag_ + ": no sub-stream rule for " + info_.url_masked +
                                " — using the URL as stored");
    }
}

RtspCapture::~RtspCapture() { halt(); }

void RtspCapture::start() {
    std::lock_guard<std::mutex> life(life_mu_);
    if (users_++ > 0) {
        log(LogLevel::Info, tag_ + ": shared with another class (one connection to the camera)");
        return;
    }
    {   // a restarted capture shows nothing from its previous run
        std::lock_guard<std::mutex> lk(mu_);
        latest_.bgr.release();
        info_.has_frame = false;
    }
    running_ = true;
    thread_ = std::thread(&RtspCapture::run, this);
    log(LogLevel::Info, tag_ + ": starting " +
                            (cfg_.file.empty() ? "(" + cfg_.profile + "-stream, " + cfg_.transport + ") " : std::string()) +
                            info_.url_masked);
}

void RtspCapture::stop() {
    std::lock_guard<std::mutex> life(life_mu_);
    if (users_ == 0 || --users_ > 0) return;
    running_ = false;
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void RtspCapture::halt() {
    std::lock_guard<std::mutex> life(life_mu_);
    users_ = 0;
    running_ = false;
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    // the GPU decoders go to the next capture, not away (see parked_decoders)
    std::lock_guard<std::mutex> lk(parked_mu());
    for (auto& d : va_dec_) {
        while (GstSample* x = gst_app_sink_try_pull_sample(d.second->sink, 0)) gst_sample_unref(x);
        parked_decoders().emplace(d.first + "@" + std::to_string(cfg_.gpu_max_width), std::move(d.second));
    }
    va_dec_.clear();
}

void RtspCapture::sleep_interruptible(double seconds) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait_for(lk, std::chrono::duration<double>(seconds), [this] { return !running_; });
}

bool RtspCapture::latest(FramePacket& out) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (latest_.bgr.empty()) return false;
    out = latest_;  // shares the pixel buffer; publish() always installs a new Mat
    return true;
}

// VA pipelines: the decoded size reaches the GPU post-processor -> pick its output: BGRx, scaled on
// the GPU to max_width when the stream is wider. GStreamer 1.24's vapostproc aborts the process
// (a bogus multi-terabyte g_malloc) for NV12 output and for BGRx widths that are not a multiple of
// 32 — tested on this host — so the width is rounded down to a multiple of 32 (720 -> 704: 2 %).
GstPadProbeReturn RtspCapture::on_decoded_caps(GstPad*, GstPadProbeInfo* info, gpointer self) {
    auto* src = static_cast<GstSource*>(self);
    GstEvent* ev = GST_PAD_PROBE_INFO_EVENT(info);
    if (GST_EVENT_TYPE(ev) != GST_EVENT_CAPS) return GST_PAD_PROBE_OK;
    GstCaps* caps = nullptr;
    gst_event_parse_caps(ev, &caps);
    const GstStructure* st = gst_caps_get_structure(caps, 0);
    int w = 0, h = 0;
    gst_structure_get_int(st, "width", &w);
    gst_structure_get_int(st, "height", &h);
    if (w <= 0 || h <= 0) return GST_PAD_PROBE_OK;
    src->native_w = w;
    src->native_h = h;
    const int cap = src->max_width > 0 ? std::min(w, src->max_width) : w;
    const int tw = std::max(32, cap & ~31);
    const int th = tw == w ? h : std::max(2, static_cast<int>(std::lround(static_cast<double>(h) * tw / w / 2.0)) * 2);
    GstCaps* out = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "BGRx", "width", G_TYPE_INT, tw,
                                       "height", G_TYPE_INT, th, nullptr);
    g_object_set(src->filter, "caps", out, nullptr);
    gst_caps_unref(out);
    return GST_PAD_PROBE_OK;
}

bool RtspCapture::wait_next(std::uint64_t after_seq, FramePacket& out, double timeout_s) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait_for(lk, std::chrono::duration<double>(timeout_s),
                 [&] { return !running_ || latest_.seq > after_seq; });
    if (latest_.seq <= after_seq || latest_.bgr.empty()) return false;
    out = latest_;  // shares the pixel buffer; publish() always installs a new Mat
    return true;
}

CaptureInfo RtspCapture::info() const {
    std::lock_guard<std::mutex> lk(mu_);
    return info_;
}

void RtspCapture::publish(cv::Mat&& frame) {
    const double now_ms = mono_ms();
    const int dec_w = frame.cols, dec_h = frame.rows;
    if (scene_aspect_ > 0.0 && dec_h > 0) {
        const int want_w = static_cast<int>(std::lround(dec_h * scene_aspect_));
        if (std::abs(want_w - dec_w) > 0.03 * want_w) {  // re-stretch width, keep vertical detail
            cv::Mat fixed;
            cv::resize(frame, fixed, cv::Size(want_w, dec_h), 0, 0, cv::INTER_LINEAR);
            frame = std::move(fixed);
        }
    }
    std::lock_guard<std::mutex> lk(mu_);
    latest_.bgr = std::move(frame);
    latest_.seq += 1;
    latest_.ts = wall_time();
    info_.has_frame = true;
    info_.frames += 1;
    info_.width = dec_w;
    info_.height = dec_h;
    info_.out_width = latest_.bgr.cols;
    info_.out_height = latest_.bgr.rows;
    if (fps_window_frames_ == 0) fps_window_start_ = now_ms;
    fps_window_frames_ += 1;
    if (now_ms - fps_window_start_ >= 1000.0) {
        info_.decode_fps = (fps_window_frames_ - 1) * 1000.0 / (now_ms - fps_window_start_);
        fps_window_frames_ = 0;
    }
    cv_.notify_all();
}

std::unique_ptr<RtspCapture::GstSource> RtspCapture::open_gstreamer(const std::string& url, cv::Mat& first,
                                                                    const std::string& prefer,
                                                                    std::vector<std::string>* rejected, bool gpu) {
    std::vector<std::string> local_rejected;
    if (!rejected) rejected = &local_rejected;
    net_failed_ = false;
    const std::string sock_us = std::to_string(socket_timeout_us(rtt_ms_));
    const std::string src = "rtspsrc name=src latency=" + std::to_string(jitter_latency_ms(std::max(cfg_.latency_ms, cfg_.smooth_ms), rtt_ms_)) +
                            " protocols=" + cfg_.transport +
                            " tcp-timeout=" + sock_us + " timeout=" + sock_us;
    // drop-on-latency stays off: over a congested Wi-Fi link the packets arrive in bursts, and a
    // jitterbuffer that drops the late ones cuts holes into the video (grey, smeared frames)
    const std::string threads =
        cfg_.decoder_threads > 0 ? " max-threads=" + std::to_string(cfg_.decoder_threads) : "";
    // appsink keeps only the newest decoded frame; upstream is never blocked by us. With a playout
    // buffer (smooth_ms) frames are handed over at their timestamps instead of in network bursts.
    const std::string sync = cfg_.smooth_ms > 0 ? " sync=true" : " sync=false";
    const std::string sink =
        "videoconvert ! video/x-raw,format=BGR ! appsink name=sink drop=true max-buffers=1" + sync;
    std::vector<std::pair<std::string, std::string>> candidates = {
        {"H.264", src + " ! rtph264depay ! h264parse ! avdec_h264 output-corrupt=false" + threads + " ! " + sink},
        {"H.265", src + " ! rtph265depay ! h265parse ! avdec_h265 output-corrupt=false" + threads + " ! " + sink},
        {"Auto", src + " ! application/x-rtp,media=video ! decodebin force-sw-decoders=true ! " + sink},
    };
    if (gpu && cfg_.hw_decode && gpu_decoding_available()) {
        // GPU first: decoding a 2-6 MP H.264/H.265 camera at 25 fps is beyond this CPU, the iGPU does it
        // for almost nothing. A VA failure falls back to the CPU pipeline of the same codec.
        if (cfg_.smooth_ms > 0) {   // playout buffer: one pipeline, synced to the camera's timestamps
            const std::string va_sink = " ! vapostproc name=post ! capsfilter name=filt caps=video/x-raw,format=BGRx"
                                        " ! appsink name=sink drop=true max-buffers=1" + sync;
            candidates.insert(candidates.begin(), {
                {"VA-H.264", src + " ! rtph264depay ! h264parse ! vah264dec" + va_sink},
                {"VA-H.265", src + " ! rtph265depay ! h265parse ! vah265dec" + va_sink},
            });
        } else {   // receiver only: the decoder behind it outlives reconnects (see va_decoder)
            const std::string es = " ! appsink name=es sync=false max-buffers=120 drop=false";
            candidates.insert(candidates.begin(), {
                {"VA-H.264", src + " ! rtph264depay ! h264parse config-interval=-1 ! "
                                   "video/x-h264,stream-format=byte-stream,alignment=au" + es},
                {"VA-H.265", src + " ! rtph265depay ! h265parse config-interval=-1 ! "
                                   "video/x-h265,stream-format=byte-stream,alignment=au" + es},
            });
        }
    }
    // A codec this camera already refused (its decoder chain ended "not-linked") is not tried
    // again: every try is a full RTSP session, and cameras with a small viewer limit answer the
    // next session with 404 until the previous one is gone.
    const auto refused = [&](const std::string& name) {
        return std::find(rejected->begin(), rejected->end(), name) != rejected->end();
    };
    const auto kept = std::remove_if(candidates.begin(), candidates.end(),
                                     [&](const std::pair<std::string, std::string>& c) { return refused(c.first); });
    if (kept != candidates.begin()) candidates.erase(kept, candidates.end());
    if (!prefer.empty()) {  // the codec that worked before (or the camera's reported codec) goes first,
                            // its GPU pipeline ahead of its CPU one
        const auto codec_of = [](const std::string& n) { return n.rfind("VA-", 0) == 0 ? n.substr(3) : n; };
        std::stable_partition(candidates.begin(), candidates.end(),
                              [&](const std::pair<std::string, std::string>& c) { return codec_of(c.first) == codec_of(prefer); });
    }

    bool auto_only = false;  // a silent timeout: skip the guessed codecs, generic decoding is the last try
    for (const auto& c : candidates) {
        if (!running_) return nullptr;
        if (auto_only && c.first != "Auto") continue;
        GError* err = nullptr;
        GstElement* p = gst_parse_launch(c.second.c_str(), &err);
        if (err) {  // missing plugin (e.g. no gst-libav): try the next pipeline
            g_error_free(err);
            if (p) gst_object_unref(p);
            continue;
        }
        if (!p) continue;
        auto s = std::make_unique<GstSource>();
        s->pipeline = p;
        s->name = c.first;
        GstElement* rtsp = gst_bin_get_by_name(GST_BIN(p), "src");
        g_object_set(rtsp, "location", url.c_str(), nullptr);  // no quoting issues with & or @
        if (g_signal_lookup("select-stream", G_OBJECT_TYPE(rtsp))) {
            g_signal_connect(rtsp, "select-stream", G_CALLBACK(select_video_stream), nullptr);
        }
        gst_object_unref(rtsp);
        if (GstElement* es = gst_bin_get_by_name(GST_BIN(p), "es")) {   // GPU receiver -> persistent decoder
            s->dec = va_decoder(c.first.substr(3));
            if (!s->dec) {
                gst_object_unref(es);
                continue;
            }
            s->feed = gst_bin_get_by_name(GST_BIN(s->dec->pipeline), "src");
            s->sink = GST_APP_SINK(gst_object_ref(s->dec->sink));
            while (GstSample* stale = gst_app_sink_try_pull_sample(s->sink, 0)) gst_sample_unref(stale);  // pre-reconnect frame
            GstAppSinkCallbacks cb{};
            cb.new_sample = &RtspCapture::on_es_sample;
            gst_app_sink_set_callbacks(GST_APP_SINK(es), &cb, s.get(), nullptr);
            gst_object_unref(es);
        } else {
            s->sink = GST_APP_SINK(gst_bin_get_by_name(GST_BIN(p), "sink"));
            attach_gpu_scaler(*s);
        }
        if (gst_element_set_state(p, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) continue;

        const double deadline = mono_ms() + connect_deadline_ms(rtt_ms_);  // connect + first decoded frame
        std::string why;
        bool mismatch = false;
        bool failed = false;
        while (running_ && mono_ms() < deadline) {
            if (read_gstreamer(*s, first)) return s;
            if (pipeline_failed(p, &why, &mismatch)) {
                failed = true;
                break;
            }
            if (s->dec && pipeline_failed(s->dec->pipeline, &why)) {   // the GPU cannot decode this stream
                failed = mismatch = true;
                const std::string codec = c.first.substr(3);
                s.reset();
                va_dec_.erase(codec);
                break;
            }
            if (s->dec && s->size_changed) {   // a reused decoder, another picture size: a fresh one next time
                log(LogLevel::Info, tag_ + ": stream is " + std::to_string(s->new_w.load()) + "x" +
                                        std::to_string(s->new_h.load()) + " now, starting a new GPU decoder");
                const std::string codec = c.first.substr(3);
                s.reset();
                va_dec_.erase(codec);
                net_failed_ = true;
                return nullptr;
            }
        }
        if (!running_) return nullptr;
        if (!failed && s && s->dec && s->pushed > 0) {
            failed = mismatch = true;   // the stream arrived but the GPU decoded nothing from it
        }
        if (failed && mismatch) {
            rejected->push_back(c.first);  // `s` is torn down (TEARDOWN) before the next pipeline starts
            log(LogLevel::Info, tag_ + ": " + c.first + " pipeline does not fit the stream, trying the next");
        } else if (failed) {
            net_failed_ = true;
            return nullptr;  // network/RTSP failure (404, reset, auth): another codec cannot help
        } else if (!prefer.empty() && prefer != "Auto") {
            // the codec is known (the camera reported it, or it worked before): no frame in time is the
            // network, not the codec — try the same pipeline again rather than the slower CPU fallback
            net_failed_ = true;
            return nullptr;
        } else {
            auto_only = true;  // no frame before the deadline
        }
    }
    return nullptr;
}

std::multimap<std::string, std::unique_ptr<RtspCapture::GstSource>>& RtspCapture::parked_decoders() {
    static auto* parked = new std::multimap<std::string, std::unique_ptr<GstSource>>();   // never torn down at exit
    return *parked;
}

std::mutex& RtspCapture::parked_mu() {
    static auto* mu = new std::mutex();
    return *mu;
}

RtspCapture::GstSource* RtspCapture::va_decoder(const std::string& codec) {
    auto it = va_dec_.find(codec);
    if (it != va_dec_.end()) return it->second.get();
    // parked by codec and GPU output width: the width is applied when a stream's size first arrives,
    // and a next stream of the same size brings no new size to apply another width with
    const std::string park_key = codec + "@" + std::to_string(cfg_.gpu_max_width);
    {   // a decoder a finished class left behind: the next stream renegotiates it
        std::unique_ptr<GstSource> d;
        {
            std::lock_guard<std::mutex> lk(parked_mu());
            auto& parked = parked_decoders();
            auto p = parked.find(park_key);
            if (p != parked.end()) {
                d = std::move(p->second);
                parked.erase(p);
            }
        }
        if (d && !pipeline_failed(d->pipeline, nullptr)) {
            GstSource* raw = d.get();
            va_dec_[codec] = std::move(d);
            return raw;
        }
    }
    const std::string lc = codec == "H.265" ? "h265" : "h264";
    // appsrc: arrival timestamps (do-timestamp), never blocks the receiver; on overload the oldest
    // data goes (decoding resumes cleanly at the next key frame)
    const std::string desc = "appsrc name=src is-live=true format=time do-timestamp=true max-bytes=16777216 leaky-type=downstream"
                             " ! " + lc + "parse ! va" + lc + "dec ! vapostproc name=post ! capsfilter name=filt caps=video/x-raw,format=BGRx"
                             " ! appsink name=sink drop=true max-buffers=1 sync=false";
    GError* err = nullptr;
    GstElement* p = gst_parse_launch(desc.c_str(), &err);
    if (err) {
        log(LogLevel::Warn, tag_ + ": GPU " + codec + " decoder unavailable: " + err->message);
        g_error_free(err);
        if (p) gst_object_unref(p);
        return nullptr;
    }
    if (!p) return nullptr;
    auto d = std::make_unique<GstSource>();
    d->pipeline = p;
    d->name = "VA-" + codec;
    d->sink = GST_APP_SINK(gst_bin_get_by_name(GST_BIN(p), "sink"));
    attach_gpu_scaler(*d);
    if (gst_element_set_state(p, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) return nullptr;
    GstSource* raw = d.get();
    va_dec_[codec] = std::move(d);
    return raw;
}

// Receiver streaming thread: one access unit (H.264/H.265 byte-stream) -> the decoder. After every
// (re)connect it starts at a key frame and is marked DISCONT; the receiver's timestamps restart
// with each connection, so they are dropped and the decoder's appsrc stamps arrival time.
GstFlowReturn RtspCapture::on_es_sample(GstAppSink* es, gpointer self) {
    auto* s = static_cast<GstSource*>(self);
    GstSample* sample = gst_app_sink_pull_sample(es);
    if (!sample) return GST_FLOW_OK;
    GstBuffer* buf = gst_sample_get_buffer(sample);
    if (buf) s->bytes += gst_buffer_get_size(buf);
    // Data lost on the way (the depayloader marks the next frame DISCONT): every frame up to the next
    // key frame refers to what is missing and would decode grey and smeared — skip them. rtph265depay
    // has no wait-for-keyframe in GStreamer 1.24, so this gate does it for both codecs.
    if (buf && s->key_seen && GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_DISCONT)) {
        s->key_seen = false;
        s->gaps += 1;
    }
    if (buf && !s->key_seen && GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_DELTA_UNIT)) s->skipped += 1;
    if (buf && s->feed && (s->key_seen || !GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_DELTA_UNIT))) {
        s->pushed += 1;
        GstBuffer* out = gst_buffer_copy(buf);   // shares the data
        if (!s->key_seen) {
            GST_BUFFER_FLAG_SET(out, GST_BUFFER_FLAG_DISCONT);
            s->key_seen = true;
        }
        GST_BUFFER_PTS(out) = GST_BUFFER_DTS(out) = GST_CLOCK_TIME_NONE;
        GstAppSrc* src = GST_APP_SRC(s->feed);
        GstCaps* caps = gst_sample_get_caps(sample);
        GstCaps* cur = gst_app_src_get_caps(src);
        bool feed_it = true;
        if (caps && !cur) {
            gst_app_src_set_caps(src, caps);   // a new decoder: its first stream
        } else if (caps && cur && !gst_caps_is_equal(cur, caps)) {
            // The decoder already decodes a stream. Renegotiating a running VA decoder (and its
            // post-processor) to another picture size crashed GStreamer 1.24 (SIGSEGV), so a size
            // change ends this connection and the next one gets a fresh decoder. Same size, other
            // details (frame rate, level): nothing to renegotiate — the decoder's parser reads the
            // parameter sets in the stream itself.
            int cw = 0, ch = 0, nw = 0, nh = 0;
            gst_structure_get_int(gst_caps_get_structure(cur, 0), "width", &cw);
            gst_structure_get_int(gst_caps_get_structure(cur, 0), "height", &ch);
            gst_structure_get_int(gst_caps_get_structure(caps, 0), "width", &nw);
            gst_structure_get_int(gst_caps_get_structure(caps, 0), "height", &nh);
            if (nw > 0 && nh > 0 && (nw != cw || nh != ch)) {
                s->new_w = nw;
                s->new_h = nh;
                s->size_changed = true;
                feed_it = false;
            }
        }
        if (cur) gst_caps_unref(cur);
        if (feed_it) {
            gst_app_src_push_buffer(src, out);   // takes `out`
        } else {
            gst_buffer_unref(out);
        }
    }
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

void RtspCapture::attach_gpu_scaler(GstSource& s) {
    GstElement* post = gst_bin_get_by_name(GST_BIN(s.pipeline), "post");
    if (!post) return;   // CPU pipeline
    s.filter = gst_bin_get_by_name(GST_BIN(s.pipeline), "filt");
    s.max_width = cfg_.gpu_max_width;
    GstPad* pad = gst_element_get_static_pad(post, "sink");
    gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM, &RtspCapture::on_decoded_caps, &s, nullptr);
    gst_object_unref(pad);
    gst_object_unref(post);
}

std::unique_ptr<RtspCapture::GstSource> RtspCapture::open_file(cv::Mat& first) {
    // sync=true: frames leave at the file's own timestamps, i.e. in real time like a camera, and
    // decoding runs only as fast as that (decodebin picks the VA decoder when one fits the codec)
    const std::string tail = " ! appsink name=sink drop=true max-buffers=1 sync=true";
    std::vector<std::pair<std::string, std::string>> candidates;
    if (cfg_.hw_decode && gpu_decoding_available()) {
        candidates.push_back({"VA", "filesrc name=src ! decodebin ! vapostproc name=post ! "
                                    "capsfilter name=filt caps=video/x-raw,format=BGRx" + tail});
    }
    candidates.push_back({"CPU", "filesrc name=src ! decodebin force-sw-decoders=true ! videoconvert ! "
                                 "video/x-raw,format=BGR" + tail});
    for (const auto& c : candidates) {
        if (!running_) return nullptr;
        GError* err = nullptr;
        GstElement* p = gst_parse_launch(c.second.c_str(), &err);
        if (err) {
            g_error_free(err);
            if (p) gst_object_unref(p);
            continue;
        }
        if (!p) continue;
        auto s = std::make_unique<GstSource>();
        s->pipeline = p;
        s->name = c.first;
        GstElement* src = gst_bin_get_by_name(GST_BIN(p), "src");
        g_object_set(src, "location", cfg_.file.c_str(), nullptr);
        gst_object_unref(src);
        s->sink = GST_APP_SINK(gst_bin_get_by_name(GST_BIN(p), "sink"));
        attach_gpu_scaler(*s);
        if (gst_element_set_state(p, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) continue;
        const double deadline = mono_ms() + 15000.0;
        std::string why;
        while (running_ && mono_ms() < deadline) {
            if (read_gstreamer(*s, first)) return s;
            if (pipeline_failed(p, &why)) break;
        }
        if (!running_) return nullptr;
        log(LogLevel::Info, tag_ + ": " + c.first + " video pipeline failed" +
                                (why.empty() ? std::string() : " (" + why + ")") + ", trying the next");
    }
    return nullptr;
}

void RtspCapture::run_file() {
    const std::string name = file_name(cfg_.file);
    int fail = 0;
    while (running_) {
        cv::Mat first;
        std::unique_ptr<GstSource> gst = gstreamer_available() ? open_file(first) : nullptr;
        if (!gst) {
            if (!running_) break;
            ++fail;
            {
                std::lock_guard<std::mutex> lk(mu_);
                info_.fail_count = fail;
                info_.backend = "Video not playable";
            }
            const double wait = std::min(3.0 * fail, 30.0);
            log(LogLevel::Warn, tag_ + ": cannot play the recorded video " + name + " (attempt " +
                                    std::to_string(fail) + "), retrying in " + std::to_string(static_cast<int>(wait)) + "s...");
            sleep_interruptible(wait);
            continue;
        }
        fail = 0;
        const bool va = gst->name == "VA";
        const std::string backend = va ? "GStreamer video file (VA-API GPU)" : "GStreamer video file (CPU)";
        publish(std::move(first));
        int w = 0, h = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            info_.fail_count = 0;
            info_.backend = backend;
            info_.gpu = va;
            w = info_.width;
            h = info_.height;
            info_.native_width = gst->native_w > 0 ? gst->native_w.load() : w;
            info_.native_height = gst->native_h > 0 ? gst->native_h.load() : h;
        }
        log(LogLevel::Info, tag_ + ": playing recorded video " + name + " via " + backend + " " +
                                std::to_string(w) + "x" + std::to_string(h) + " (looped)");

        std::string why;
        double last_frame_ms = mono_ms();
        while (running_) {
            cv::Mat f;
            if (read_gstreamer(*gst, f)) {
                publish(std::move(f));
                last_frame_ms = mono_ms();
                continue;
            }
            GstBus* bus = gst_element_get_bus(gst->pipeline);
            GstMessage* msg = gst_bus_pop_filtered(bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
            gst_object_unref(bus);
            if (msg && GST_MESSAGE_TYPE(msg) == GST_MESSAGE_EOS) {
                gst_message_unref(msg);
                // the end of the recording: start it again, like a camera that never stops
                if (!gst_element_seek_simple(gst->pipeline, GST_FORMAT_TIME,
                                             static_cast<GstSeekFlags>(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_KEY_UNIT), 0)) {
                    why = "cannot rewind";
                    break;
                }
                last_frame_ms = mono_ms();
                continue;
            }
            if (msg) {
                GError* err = nullptr;
                gst_message_parse_error(msg, &err, nullptr);
                why = err ? err->message : "error";
                if (err) g_error_free(err);
                gst_message_unref(msg);
                break;
            }
            if (mono_ms() - last_frame_ms > 10000.0) {
                why = "no picture for 10 s";
                break;
            }
        }
        gst.reset();
        if (!running_) break;
        {
            std::lock_guard<std::mutex> lk(mu_);
            info_.reconnects += 1;
        }
        log(LogLevel::Warn, tag_ + ": recorded video stopped (" + why + "), restarting");
        sleep_interruptible(1.0);
    }
}

void RtspCapture::resolve_scene_aspect() {
    aspect_resolved_ = true;
    if (cfg_.profile != "sub" || cfg_.aspect == "off") return;
    const auto colon = cfg_.aspect.find(':');
    if (colon != std::string::npos) {  // explicit "W:H"
        const double aw = std::atof(cfg_.aspect.substr(0, colon).c_str());
        const double ah = std::atof(cfg_.aspect.substr(colon + 1).c_str());
        if (aw > 0 && ah > 0) scene_aspect_ = aw / ah;
    } else if (cfg_.aspect == "auto" && rewritten_ && gstreamer_available()) {
        // One decoded main-stream frame tells the true scene geometry (one-off cost).
        cv::Mat probe;
        // CPU decoding: one frame is all it needs, and no GPU decoder is made for a one-off look
        auto main_src = open_gstreamer(cfg_.url, probe, std::string(), nullptr, /*gpu=*/false);
        if (main_src && !probe.empty()) scene_aspect_ = static_cast<double>(probe.cols) / probe.rows;
        if (!main_src) {
            log(LogLevel::Warn, tag_ + ": main stream not reachable for the aspect probe — "
                                "sub-stream frames used as decoded (set SUBSTREAM_ASPECT=16:9 to force)");
        }
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        info_.scene_aspect = scene_aspect_;
    }
    if (scene_aspect_ > 0.0) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.4f", scene_aspect_);
        log(LogLevel::Info, tag_ + ": scene aspect " + buf +
                                " — sub-stream frames are corrected to square pixels when needed");
    }
}

bool RtspCapture::read_gstreamer(GstSource& src, cv::Mat& frame) {
    GstSample* sample = gst_app_sink_try_pull_sample(src.sink, 500 * GST_MSECOND);
    if (!sample) return false;
    const bool ok = sample_to_bgr(sample, frame);
    gst_sample_unref(sample);
    return ok;
}

void RtspCapture::run() {
    if (!cfg_.file.empty()) {
        run_file();
        return;
    }
    int fail = 0;
    while (running_) {
        std::unique_ptr<GstSource> gst;
        cv::VideoCapture ff;
        std::string backend;
        cv::Mat first;
        // one TCP handshake ~ one round trip: size every timeout to this camera's link. The
        // estimate rises at once and decays slowly, since such links swing between ms and s.
        const double rtt = probe_rtt_ms(url_, 15000);
        const bool reachable = rtt >= 0.0;
        if (reachable) {
            rtt_ms_ = std::max(rtt, 0.5 * rtt_ms_);
            std::lock_guard<std::mutex> lk(mu_);
            info_.rtt_ms = rtt;
        }
        if (reachable && gstreamer_available()) {
            gst = open_gstreamer(url_, first, codec_.empty() ? hint_ : codec_, &rejected_);
            if (gst) {
                if (gst->name != "Auto") codec_ = gst->name;   // a generic fallback is not a codec to prefer
                rejected_.clear();
                const bool va = gst->name.rfind("VA-", 0) == 0;
                backend = va ? "GStreamer (VA-API GPU, " + gst->name.substr(3) + ")" : "GStreamer (" + gst->name + ")";
                std::lock_guard<std::mutex> lk(mu_);
                info_.gpu = va;
            }
        }
        // FFmpeg only when GStreamer has no pipeline for this camera — not after a network failure,
        // which it cannot do better (and it decodes on the CPU without the corrupt-frame gate)
        if (reachable && !gst && running_ && !(gstreamer_available() && net_failed_)) {
            if (cfg_.transport.find("tcp") != std::string::npos) {
                setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp", 0);
            }
            const int ff_ms = static_cast<int>(socket_timeout_us(rtt_ms_) / 1000);
            ff.open(url_, cv::CAP_FFMPEG,
                    {cv::CAP_PROP_OPEN_TIMEOUT_MSEC, ff_ms, cv::CAP_PROP_READ_TIMEOUT_MSEC, ff_ms});
            if (ff.isOpened()) {
                ff.set(cv::CAP_PROP_BUFFERSIZE, 1);
                backend = "FFmpeg";
            }
        }
        if (!gst && !ff.isOpened()) {
            if (!running_) break;
            ++fail;
            {
                std::lock_guard<std::mutex> lk(mu_);
                info_.fail_count = fail;
            }
            // Failed tries are cheap when the camera answers (a refused SETUP takes ~100 ms), and a
            // viewer slot freed by the camera is only noticed at the next try: poll quickly. A port
            // that does not answer at all keeps the original back-off (3 s x attempts, max 30 s).
            const double wait = reachable ? std::min(1.0 + fail, 5.0) : std::min(3.0 * fail, 30.0);
            log(LogLevel::Warn, tag_ + ": Connection failed (attempt " +
                                    std::to_string(fail) + (reachable ? rtt_note(rtt_ms_) : ", RTSP port unreachable") +
                                    "), retrying in " + std::to_string(static_cast<int>(wait)) + "s...");
            sleep_interruptible(wait);
            continue;
        }

        fail = 0;
        if (!aspect_resolved_) resolve_scene_aspect();  // once per capture (main-stream probe)
        if (!first.empty()) publish(std::move(first));
        int w = 0, h = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            info_.fail_count = 0;
            info_.backend = backend;
            if (!gst) info_.gpu = false;
            w = info_.width;
            h = info_.height;
            // the camera's own size: before GPU scaling (VA) or as decoded (CPU)
            const GstSource* sized = gst && gst->dec ? gst->dec : gst.get();
            info_.native_width = sized && sized->native_w > 0 ? sized->native_w.load() : w;
            info_.native_height = sized && sized->native_h > 0 ? sized->native_h.load() : h;
        }
        log(LogLevel::Info, tag_ + ": Connected via " + backend +
                                (gst ? " " + std::to_string(w) + "x" + std::to_string(h) : std::string()) +
                                rtt_note(rtt_ms_));
        const double stall_ms = stall_timeout_s(cfg_.stall_timeout_s, rtt_ms_) * 1000.0;

        double last_frame_ms = mono_ms();
        double stat_ms = last_frame_ms;
        std::uint64_t stat_bytes = gst ? gst->bytes.load() : 0;
        std::string why;
        while (running_) {
            if (gst && gst->dec && mono_ms() - stat_ms >= 1000.0) {   // what the camera's link delivers
                const double now = mono_ms();
                const std::uint64_t b = gst->bytes.load();
                std::lock_guard<std::mutex> lk(mu_);
                info_.stream_kbps = (b - stat_bytes) * 8.0 / (now - stat_ms);
                info_.stream_gaps = gaps_total_ + gst->gaps.load();
                info_.frames_skipped = skipped_total_ + gst->skipped.load();
                stat_ms = now;
                stat_bytes = b;
            }
            cv::Mat f;
            const bool got = gst ? read_gstreamer(*gst, f) : (ff.read(f) && !f.empty());
            if (got) {
                publish(std::move(f));
                last_frame_ms = mono_ms();
                continue;
            }
            if (!gst) { why = "read failed"; break; }  // FFmpeg: reconnect like the original
            if (pipeline_failed(gst->pipeline, &why)) break;
            if (gst->dec && pipeline_failed(gst->dec->pipeline, &why)) {   // rebuilt on the next connect
                why = "GPU decoder: " + why;
                const std::string codec = gst->name.substr(3);
                gst.reset();
                va_dec_.erase(codec);
                break;
            }
            if (gst->dec && gst->size_changed) {   // see on_es_sample: a fresh decoder for the new size
                why = "the camera changed its picture size to " + std::to_string(gst->new_w.load()) + "x" +
                      std::to_string(gst->new_h.load());
                const std::string codec = gst->name.substr(3);
                gaps_total_ += gst->gaps.load();
                skipped_total_ += gst->skipped.load();
                gst.reset();
                va_dec_.erase(codec);
                break;
            }
            if (mono_ms() - last_frame_ms > stall_ms) { why = "stalled"; break; }
        }
        if (gst) {   // keep the totals across reconnects
            gaps_total_ += gst->gaps.load();
            skipped_total_ += gst->skipped.load();
        }
        gst.reset();
        ff.release();
        if (!running_) break;
        {
            std::lock_guard<std::mutex> lk(mu_);
            info_.reconnects += 1;
        }
        log(LogLevel::Warn, tag_ + ": stream lost (" + why + "), reconnecting");
        sleep_interruptible(1.0);
    }
}

std::shared_ptr<RtspCapture> acquire_capture(const CaptureConfig& cfg) {
    static std::mutex mu;
    static std::unordered_map<std::string, std::weak_ptr<RtspCapture>> live;
    // the same stream, read the same way (a sub-stream and the main stream of one camera are two)
    const std::string where = !cfg.file.empty() ? "file:" + cfg.file
                              : cfg.fixed_url   ? cfg.url
                                                : resolve_stream_url(cfg.url, cfg.profile, nullptr, nullptr);
    const std::string key = where + "|" + cfg.profile + "|" + cfg.aspect + "|" + cfg.transport + "|" +
                            (cfg.hw_decode ? "gpu" : "cpu") + "|" + std::to_string(cfg.gpu_max_width) + "|" +
                            std::to_string(cfg.smooth_ms);
    std::lock_guard<std::mutex> lk(mu);
    for (auto it = live.begin(); it != live.end();) it = it->second.expired() ? live.erase(it) : std::next(it);
    if (auto cap = live[key].lock()) return cap;
    auto cap = std::make_shared<RtspCapture>(cfg);
    live[key] = cap;
    return cap;
}

int open_fd_count() {
    int n = 0;
    if (DIR* d = opendir("/proc/self/fd")) {
        while (const dirent* e = readdir(d)) n += e->d_name[0] != '.';
        closedir(d);
        --n;   // the directory itself
    }
    return n;
}

int open_fd_limit() {
    rlimit r{};
    return getrlimit(RLIMIT_NOFILE, &r) == 0 && r.rlim_cur != RLIM_INFINITY ? static_cast<int>(r.rlim_cur) : -1;
}

}  // namespace bup
