// pybind11 module `bup_engine` — the Python face of the C++ attendance engine.
//
// Python-visible API mirrors the original app.py objects so Flask routes stay
// unchanged:
//   AttendanceTracker(meta, min_time=60): update / set_manual / summary / live_stats
//   FaceEngine(det, rec).get(img) -> [Face(bbox, det_score, kps, normed_embedding)]
//   AttendanceSession(...): start / stop / mjpeg_frame / camera(cam_id).get_info()
// Every call that can block (NPU inference, joins, JPEG encode) releases the GIL.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <dlfcn.h>
#include <execinfo.h>
#include <pthread.h>
#include <signal.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "bup/byte_tracker.hpp"
#include "bup/capture.hpp"
#include "bup/common.hpp"
#include "bup/face_engine.hpp"
#include "bup/session.hpp"
#include "bup/stream_url.hpp"
#include "bup/system_monitor.hpp"
#include "bup/tracker.hpp"

namespace py = pybind11;
using namespace bup;

namespace {

using U8Array = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
using F32Array = py::array_t<float, py::array::c_style | py::array::forcecast>;

/// Wrap (no copy) an HxWx3 uint8 numpy image. The array must outlive the Mat.
cv::Mat mat_view(const U8Array& img) {
    if (img.ndim() != 3 || img.shape(2) != 3) {
        throw py::value_error("expected an HxWx3 uint8 BGR image");
    }
    return cv::Mat(static_cast<int>(img.shape(0)), static_cast<int>(img.shape(1)), CV_8UC3,
                   const_cast<uint8_t*>(img.data()));
}

py::object mat_to_numpy(const cv::Mat& m) {
    if (m.empty()) return py::none();
    U8Array out({m.rows, m.cols, m.channels()});
    const std::size_t row = static_cast<std::size_t>(m.cols) * m.elemSize();
    for (int y = 0; y < m.rows; ++y) std::memcpy(out.mutable_data(y), m.ptr(y), row);
    return std::move(out);
}

std::vector<StudentMeta> meta_from_py(const py::iterable& meta) {
    std::vector<StudentMeta> out;
    for (const py::handle& h : meta) {
        py::dict d = py::reinterpret_borrow<py::dict>(h);
        out.push_back({py::str(d["student_id"]), py::str(d["name"])});
    }
    return out;
}

/// Python's own round(): banker's rounding + int/float result types identical to app.py.
/// (Looked up per call — a static py::object would be destroyed after interpreter shutdown.)
py::object builtin_round(py::object value, int ndigits) {
    return py::module_::import("builtins").attr("round")(value, ndigits);
}

/// Python-side view of one camera of a session: `_session["cameras"][cam_id].get_info()`
struct CameraHandle {
    std::shared_ptr<AttendanceSession> session;
    std::string cam_id;
};

py::dict camera_info_dict(const CameraStatus& st) {
    const CaptureInfo& c = st.capture;
    py::dict d;
    // keys of the original RTSPCapture.get_info()
    d["cam_id"] = c.cam_id;
    d["backend"] = c.backend;
    d["has_frame"] = c.has_frame;
    d["fail_count"] = c.fail_count;
    // additional diagnostics
    d["stream_profile"] = c.profile;
    d["url"] = c.url_masked;
    d["url_rewritten"] = c.url_rewritten;
    d["frame_size"] = c.width > 0 ? py::object(py::str(std::to_string(c.width) + "x" + std::to_string(c.height)))
                                  : py::object(py::none());
    d["processing_size"] = c.out_width > 0
        ? py::object(py::str(std::to_string(c.out_width) + "x" + std::to_string(c.out_height)))
        : py::object(py::none());
    d["scene_aspect"] = c.scene_aspect > 0 ? py::object(builtin_round(py::float_(c.scene_aspect), 4))
                                           : py::object(py::none());
    d["decode_fps"] = builtin_round(py::float_(c.decode_fps), 1);
    d["infer_fps"] = st.infer_fps;
    d["frames_decoded"] = c.frames;
    d["frames_processed"] = st.processed;
    d["reconnects"] = c.reconnects;
    d["network_rtt_ms"] = c.rtt_ms >= 0 ? py::object(builtin_round(py::float_(c.rtt_ms), 1)) : py::object(py::none());
    d["errors"] = st.errors;
    d["detect_ms_avg"] = builtin_round(py::float_(st.det_ms_avg), 2);
    d["recognize_ms_avg"] = builtin_round(py::float_(st.rec_ms_avg), 2);
    d["faces_last_frame"] = st.last_faces;
    d["stream_size"] = c.native_width > 0
        ? py::object(py::str(std::to_string(c.native_width) + "x" + std::to_string(c.native_height)))
        : py::object(py::none());
    d["decoder"] = c.has_frame ? py::object(py::str(c.gpu ? "gpu" : "cpu")) : py::object(py::none());
    d["feed_fps"] = builtin_round(py::float_(c.decode_fps), 1);
    d["stream_kbps"] = builtin_round(py::float_(c.stream_kbps), 0);
    d["stream_gaps"] = c.stream_gaps;
    d["frames_skipped"] = c.frames_skipped;
        d["tracks"] = st.tracks;
    d["identified_tracks"] = st.identified;
    d["embeddings_computed"] = st.embeds;
    d["inference_size"] = st.infer_w > 0 ? py::object(py::str(std::to_string(st.infer_w) + "x" + std::to_string(st.infer_h)))
                                         : py::object(py::none());
    return d;
}

}  // namespace

namespace {

// A crash (SIGSEGV, SIGBUS, SIGABRT, SIGFPE, SIGILL) is written to stderr — the container log —
// before the process dies as before: the signal, the faulting address, the thread's name
// (GStreamer names its streaming threads after the element) and the native stack as
// <library>+<offset> (symbol). Core dumps stay off (1.3 GB each filled the disk); this is what
// tells which library crashed and where (addr2line -e <library> <offset>).
void write_err(const char* s, int n) {
    while (n > 0) {
        const ssize_t w = ::write(STDERR_FILENO, s, static_cast<size_t>(n));
        if (w <= 0) return;
        s += w;
        n -= static_cast<int>(w);
    }
}

void on_fatal_signal(int sig, siginfo_t* si, void*) {
    char line[512];
    char thread[32] = "?";
    pthread_getname_np(pthread_self(), thread, sizeof(thread));
    int n = std::snprintf(line, sizeof(line), "\n[bup_engine] FATAL: signal %d (%s) at address %p, thread '%s' (tid %ld)\n",
                          sig, sig == SIGSEGV ? "SIGSEGV" : sig == SIGBUS ? "SIGBUS" : sig == SIGABRT ? "SIGABRT"
                               : sig == SIGFPE ? "SIGFPE" : "SIGILL",
                          si ? si->si_addr : nullptr, thread, static_cast<long>(gettid()));
    write_err(line, n);
    void* frames[64];
    const int k = backtrace(frames, 64);
    for (int i = 0; i < k; ++i) {
        Dl_info info{};
        if (dladdr(frames[i], &info) && info.dli_fname) {
            n = std::snprintf(line, sizeof(line), "[bup_engine]   #%-2d %s+0x%lx (%s)\n", i, info.dli_fname,
                              static_cast<unsigned long>(static_cast<char*>(frames[i]) - static_cast<char*>(info.dli_fbase)),
                              info.dli_sname ? info.dli_sname : "?");
        } else {
            n = std::snprintf(line, sizeof(line), "[bup_engine]   #%-2d %p\n", i, frames[i]);
        }
        write_err(line, n);
    }
    ::signal(sig, SIG_DFL);   // die by the same signal (gunicorn restarts the worker; live classes resume)
    ::raise(sig);
}

void install_crash_report() {
    static bool done = false;
    if (done) return;
    done = true;
    void* warm[1];
    backtrace(warm, 1);   // loads libgcc now, not inside the signal handler
    struct sigaction sa{};
    sa.sa_sigaction = &on_fatal_signal;
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
    sigemptyset(&sa.sa_mask);
    for (int sig : {SIGSEGV, SIGBUS, SIGABRT, SIGFPE, SIGILL}) {
        struct sigaction old{};
        sigaction(sig, nullptr, &old);
        // keep a handler someone installed on purpose (e.g. Python's faulthandler)
        if (old.sa_handler == SIG_DFL) sigaction(sig, &sa, nullptr);
    }
}

}  // namespace

PYBIND11_MODULE(bup_engine, m) {
    m.doc() = "BUP Face Attendance — C++ NPU engine (DEEPX DX-M1)";
    m.attr("__version__") = BUP_ENGINE_VERSION;

    // Open-file limit: the usual soft default (1024) is too small for several classes at once
    // (GStreamer, VA-API and DXRT each hold dozens of descriptors per camera); raise it towards the
    // hard limit. Running out of descriptors aborts the whole process inside GLib.
    {
        rlimit r{};
        const rlim_t want = 65536;
        if (getrlimit(RLIMIT_NOFILE, &r) == 0 && r.rlim_cur != RLIM_INFINITY && r.rlim_cur < want) {
            r.rlim_cur = r.rlim_max == RLIM_INFINITY ? want : std::min(want, r.rlim_max);
            setrlimit(RLIMIT_NOFILE, &r);
        }
    }
    install_crash_report();
    m.def("open_files", [] { return py::make_tuple(open_fd_count(), open_fd_limit()); },
          "(open file descriptors of this process, their limit)");

    m.def("gpu_decoding_available", &gpu_decoding_available,
          "True when Intel VA-API decoding works here (GStreamer va plugin, iHD driver, /dev/dri access)");
    m.def("probe_video",
          [](const std::string& path, int thumb_width) {
              VideoProbe r;
              {
                  py::gil_scoped_release nogil;
                  r = probe_video(path);
              }
              py::dict d;
              d["ok"] = r.ok;
              d["error"] = r.error;
              d["width"] = r.width;
              d["height"] = r.height;
              d["fps"] = r.fps;
              d["duration"] = r.duration_s;
              d["codec"] = r.codec;
              d["thumbnail"] = py::none();
              if (r.ok && thumb_width > 0) {   // JPEG of the first frame for the room page
                  cv::Mat small;
                  const double s = std::min(1.0, static_cast<double>(thumb_width) / r.first.cols);
                  cv::resize(r.first, small, cv::Size(), s, s, cv::INTER_AREA);
                  std::vector<uchar> jpg;
                  cv::imencode(".jpg", small, jpg, {cv::IMWRITE_JPEG_QUALITY, 85});
                  d["thumbnail"] = py::bytes(reinterpret_cast<const char*>(jpg.data()), jpg.size());
              }
              return d;
          },
          py::arg("path"), py::arg("thumb_width") = 480,
          "What a video file holds: {ok, error, width, height, fps, duration, codec, thumbnail (JPEG bytes)}");
    m.def("has_gstreamer", &gstreamer_available,
          "True when GStreamer with rtspsrc/appsink is usable for camera capture");

    m.def("system_usage",
          [] {
              const SystemUsage u = system_usage();
              py::dict d;
              d["npu_cores"] = u.npu_core_pct.empty() ? py::object(py::none()) : py::cast(u.npu_core_pct);
              d["npu_note"] = u.npu_note;
              d["cpu"] = u.cpu_pct < 0 ? py::object(py::none()) : py::object(py::float_(u.cpu_pct));
              d["cpu_cores"] = u.cpu_core_pct;
              d["age_s"] = u.age_s < 0 ? py::object(py::none()) : py::object(py::float_(u.age_s));
              return d;
          },
          "Latest NPU-core and CPU busy % (sampled every second in the background; never blocks)");

    m.def("resolve_stream_url",
          [](const std::string& url, const std::string& profile) {
              try {
                  return resolve_stream_url(url, profile);
              } catch (const std::invalid_argument& e) {
                  throw py::value_error(e.what());
              }
          },
          py::arg("url"), py::arg("profile") = "sub");

    m.def("imread",
          [](const std::string& path) -> py::object {
              cv::Mat img;
              {
                  py::gil_scoped_release nogil;
                  img = cv::imread(path, cv::IMREAD_COLOR);  // same flags as cv2.imread default
              }
              return mat_to_numpy(img);
          },
          py::arg("path"), "cv2.imread replacement: BGR uint8 array, or None if unreadable");

    // ── AttendanceTracker ────────────────────────────────────────────────
    py::class_<AttendanceTracker, std::shared_ptr<AttendanceTracker>>(m, "AttendanceTracker")
        .def(py::init([](const py::iterable& meta, double min_time, const py::object& session_start) {
                 const double start = session_start.is_none() ? wall_time() : py::cast<double>(session_start);
                 return std::make_shared<AttendanceTracker>(meta_from_py(meta), min_time, start);
             }),
             py::arg("meta_list"), py::arg("min_time") = 60.0, py::arg("session_start") = py::none())
        .def("update", &AttendanceTracker::update, py::arg("student_id"), py::arg("camera_id"), py::arg("now"))
        .def("set_manual",
             [](AttendanceTracker& t, const std::string& sid, const py::object& status) {
                 if (status.is_none()) {
                     t.set_manual(sid, false, std::string());
                 } else {
                     t.set_manual(sid, true, py::str(status));
                 }
             },
             py::arg("student_id"), py::arg("status"))
        .def("summary",
             [](const AttendanceTracker& t) {
                 py::list rows;
                 for (const auto& r : t.rows()) {
                     py::dict d;
                     d["student_id"] = r.student_id;
                     d["name"] = r.name;
                     d["total_time"] = builtin_round(py::float_(r.total_time), 1);
                     d["present"] = r.present;
                     d["status"] = r.present ? "Present" : "Absent";
                     d["detection_count"] = r.detection_count;
                     d["manual_override"] = r.has_override ? py::object(py::str(r.manual_override))
                                                           : py::object(py::none());
                     rows.append(d);
                 }
                 return rows;
             })
        .def("live_stats",
             [](const AttendanceTracker& t) {
                 const LiveCounts c = t.live();
                 py::dict d;
                 d["total"] = c.total;
                 d["present"] = c.present;
                 d["absent"] = c.total - c.present;
                 d["rate"] = c.total ? builtin_round(py::float_(static_cast<double>(c.present) /
                                                                static_cast<double>(c.total) * 100.0), 1)
                                     : builtin_round(py::int_(0), 1);
                 d["duration"] = builtin_round(py::float_(wall_time() - c.session_start), 0);
                 return d;
             })
        .def("total_time", &AttendanceTracker::total_time, py::arg("student_id"))
        .def_property_readonly("min_time", &AttendanceTracker::min_time)
        .def_property_readonly("session_start", &AttendanceTracker::session_start)
        .def("export_state",
             [](const AttendanceTracker& t) {
                 py::list out;
                 for (const StudentState& st : t.export_state()) {
                     py::dict d;
                     d["student_id"] = st.student_id;
                     d["total_time"] = st.total_time;
                     d["detection_count"] = st.detection_count;
                     d["first_seen"] = st.has_first_seen ? py::object(py::float_(st.first_seen)) : py::object(py::none());
                     d["manual_override"] = st.has_override ? py::object(py::str(st.manual_override)) : py::object(py::none());
                     out.append(d);
                 }
                 return out;
             },
             "JSON-ready snapshot of every student's measured time (restore_state puts it back)")
        .def("restore_state",
             [](AttendanceTracker& t, const py::iterable& rows) {
                 for (const py::handle& h : rows) {
                     const py::dict d = py::reinterpret_borrow<py::dict>(h);
                     StudentState st;
                     st.student_id = py::str(d["student_id"]);
                     st.total_time = py::cast<double>(d["total_time"]);
                     st.detection_count = py::cast<long long>(d["detection_count"]);
                     if (d.contains("first_seen") && !d["first_seen"].is_none()) {
                         st.has_first_seen = true;
                         st.first_seen = py::cast<double>(d["first_seen"]);
                     }
                     if (d.contains("manual_override") && !d["manual_override"].is_none()) {
                         st.has_override = true;
                         st.manual_override = py::str(d["manual_override"]);
                     }
                     t.restore(st);
                 }
             },
             py::arg("rows"));

    // ── Face / FaceEngine ────────────────────────────────────────────────
    py::class_<Face>(m, "Face")
        .def_property_readonly("bbox", [](const Face& f) {
            return py::array_t<float>(4, f.bbox.data());
        })
        .def_readonly("det_score", &Face::det_score)
        .def_property_readonly("kps", [](const Face& f) {
            py::array_t<float> a({5, 2});
            for (int i = 0; i < 5; ++i) {
                a.mutable_at(i, 0) = f.kps[i].x;
                a.mutable_at(i, 1) = f.kps[i].y;
            }
            return a;
        })
        .def_property_readonly("normed_embedding", [](const Face& f) -> py::object {
            if (f.embedding.empty()) return py::none();
            return py::array_t<float>(static_cast<py::ssize_t>(f.embedding.size()), f.embedding.data());
        })
        .def_property_readonly("embedding", [](const Face& f) -> py::object {
            if (f.embedding.empty()) return py::none();
            return py::array_t<float>(static_cast<py::ssize_t>(f.embedding.size()), f.embedding.data());
        })
        .def("__repr__", [](const Face& f) {
            return "<Face score=" + std::to_string(f.det_score) + " box=[" + std::to_string(f.bbox[0]) + ", " +
                   std::to_string(f.bbox[1]) + ", " + std::to_string(f.bbox[2]) + ", " +
                   std::to_string(f.bbox[3]) + "]>";
        });

    py::class_<FaceEngine, std::shared_ptr<FaceEngine>>(m, "FaceEngine")
        .def(py::init([](const std::string& det, const std::string& rec, float score_th, float nms_th,
                         const std::string& npu_bound, float low_th) {
                 FaceEngineConfig cfg;
                 cfg.det_model_path = det;
                 cfg.rec_model_path = rec;
                 cfg.det_score_threshold = score_th;
                 cfg.det_nms_threshold = nms_th;
                 cfg.npu_bound = npu_bound;
                 cfg.det_low_threshold = low_th;
                 py::gil_scoped_release nogil;
                 return std::make_shared<FaceEngine>(cfg);
             }),
             py::arg("det_model"), py::arg("rec_model"), py::arg("det_score_threshold") = 0.5f,
             py::arg("nms_threshold") = 0.4f, py::arg("npu_bound") = "auto", py::arg("det_low_threshold") = 0.3f)
        .def("get",
             [](FaceEngine& e, const U8Array& img) {
                 const cv::Mat view = mat_view(img);
                 py::gil_scoped_release nogil;
                 return e.get(view);
             },
             py::arg("img"), "detect + embed; faces carry L2-normalized `normed_embedding`")
        .def("detect",
             [](FaceEngine& e, const U8Array& img, py::object min_score) {
                 const cv::Mat view = mat_view(img);
                 const float th = min_score.is_none() ? e.config().det_score_threshold : py::cast<float>(min_score);
                 py::gil_scoped_release nogil;
                 return e.detect(view, th);
             },
             py::arg("img"), py::arg("min_score") = py::none(),
             "faces >= det_score_threshold, or >= min_score (down to det_low_threshold) for tracking")
        .def_property_readonly("embedding_dim", &FaceEngine::embedding_dim)
        .def_property_readonly("det_input_size", [](const FaceEngine& e) {
            return py::make_tuple(e.det_input_width(), e.det_input_height());
        })
        .def_property_readonly("rec_input_size", [](const FaceEngine& e) {
            return py::make_tuple(e.rec_input_width(), e.rec_input_height());
        })
        .def("info", [](const FaceEngine& e) {
            py::dict d;
            d["det_model"] = e.config().det_model_path;
            d["rec_model"] = e.config().rec_model_path;
            d["det_model_name"] = e.det_model_name();
            d["rec_model_name"] = e.rec_model_name();
            d["det_input_size"] = py::make_tuple(e.det_input_width(), e.det_input_height());
            d["rec_input_size"] = py::make_tuple(e.rec_input_width(), e.rec_input_height());
            d["embedding_dim"] = e.embedding_dim();
            d["npu_bound_det"] = e.det_bound();
            d["npu_bound_rec"] = e.rec_bound();
            return d;
        })
        .def("stats", [](const FaceEngine& e) {
            const EngineStats s = e.stats();
            py::dict d;
            d["det_calls"] = s.det_calls;
            d["rec_calls"] = s.rec_calls;
            d["rec_faces"] = s.rec_faces;
            d["det_ms_avg"] = s.det_calls ? s.det_npu_ms / s.det_calls : 0.0;
            d["rec_ms_per_batch_avg"] = s.rec_calls ? s.rec_npu_ms / s.rec_calls : 0.0;
            d["rec_ms_per_face_avg"] = s.rec_faces ? s.rec_npu_ms / s.rec_faces : 0.0;
            return d;
        });

    // ── ByteTracker (the session's per-camera face tracker; exposed for tests / tools) ──
    py::class_<ByteTracker>(m, "ByteTracker")
        .def(py::init([](float high, float low, float new_track, double lost_keep_s) {
                 TrackerParams p;
                 p.high_thresh = high;
                 p.low_thresh = low;
                 p.new_track_thresh = new_track;
                 p.lost_keep_s = lost_keep_s;
                 return ByteTracker(p);
             }),
             py::arg("high_thresh") = 0.5f, py::arg("low_thresh") = 0.3f, py::arg("new_track_thresh") = 0.5f,
             py::arg("lost_keep_s") = 2.0)
        .def("update",
             [](ByteTracker& bt, const py::iterable& dets, double t) {
                 // dets: iterable of (x1, y1, x2, y2, score)
                 std::vector<TrackDet> v;
                 for (const py::handle& h : dets) {
                     const py::sequence q = py::reinterpret_borrow<py::sequence>(h);
                     if (q.size() != 5) throw py::value_error("each detection is (x1, y1, x2, y2, score)");
                     TrackDet d;
                     for (int k = 0; k < 4; ++k) d.tlbr[k] = py::cast<float>(q[k]);
                     d.score = py::cast<float>(q[4]);
                     v.push_back(d);
                 }
                 py::list out;
                 for (const TrackOut& o : bt.update(v, t)) {
                     py::dict d;
                     d["id"] = o.id;
                     d["box"] = py::make_tuple(o.tlbr[0], o.tlbr[1], o.tlbr[2], o.tlbr[3]);
                     d["velocity"] = py::make_tuple(o.vx, o.vy);
                     d["det"] = o.det;
                     d["score"] = o.score;
                     d["activated"] = o.activated;
                     d["lost"] = o.lost;
                     d["hits"] = o.hits;
                     out.append(d);
                 }
                 return out;
             },
             py::arg("dets"), py::arg("t"))
        .def("reset", &ByteTracker::reset);

    // ── AttendanceSession ────────────────────────────────────────────────
    py::class_<CameraHandle>(m, "CameraHandle")
        .def("get_info", [](const CameraHandle& h) { return camera_info_dict(h.session->camera_status(h.cam_id)); })
        .def_readonly("cam_id", &CameraHandle::cam_id);

    py::class_<AttendanceSession, std::shared_ptr<AttendanceSession>>(m, "AttendanceSession")
        .def(py::init([](std::shared_ptr<FaceEngine> engine, std::shared_ptr<AttendanceTracker> tracker,
                         const F32Array& embeds, const py::iterable& meta, const py::iterable& cameras,
                         const std::string& stream_profile, const std::string& rtsp_transport,
                         const std::string& substream_aspect, float sim_threshold, int frame_w, int frame_h, const std::string& mode_label,
                         const std::string& backend_label, int jpeg_quality, bool hw_decode, int gpu_max_width,
                         int smooth_ms, double track_lost_s, double reverify_s, int max_embed_per_frame) {
                 if (embeds.ndim() != 2) throw py::value_error("embeds must be a 2-D float32 array (N x D)");
                 cv::Mat e(static_cast<int>(embeds.shape(0)), static_cast<int>(embeds.shape(1)), CV_32F,
                           const_cast<float*>(embeds.data()));
                 SessionConfig cfg;
                 // room order; each item is a (cam_id, url) pair, or a dict with "id", "url" and the
                 // optional per-camera "profile", "aspect", "fixed_url", "codec", "max_w", "max_h" (CameraSpec)
                 for (const py::handle& c : cameras) {
                     CameraSpec spec;
                     if (py::isinstance<py::dict>(c)) {
                         const py::dict d = py::reinterpret_borrow<py::dict>(c);
                         spec.id = py::str(d["id"]);
                         spec.url = py::str(d["url"]);
                         if (d.contains("profile")) spec.profile = py::str(d["profile"]);
                         if (d.contains("aspect")) spec.aspect = py::str(d["aspect"]);
                         if (d.contains("fixed_url")) spec.fixed_url = py::cast<bool>(d["fixed_url"]);
                         if (d.contains("codec") && !d["codec"].is_none()) spec.codec = py::str(d["codec"]);
                         if (d.contains("max_w")) spec.max_w = py::cast<int>(d["max_w"]);
                         if (d.contains("max_h")) spec.max_h = py::cast<int>(d["max_h"]);
                         if (d.contains("file") && !d["file"].is_none()) spec.file = py::str(d["file"]);
                     } else {
                         const py::sequence pair = py::reinterpret_borrow<py::sequence>(c);
                         spec.id = py::str(pair[0]);
                         spec.url = py::str(pair[1]);
                     }
                     if (!spec.file.empty()) {
                         spec.profile = "main";   // a recording: no sub-stream rules, no aspect probe
                         spec.aspect = "off";
                         spec.max_w = spec.max_w < 0 ? 0 : spec.max_w;
                         spec.max_h = spec.max_h < 0 ? 0 : spec.max_h;
                     }
                     if (!spec.profile.empty() && spec.profile != "sub" && spec.profile != "main") {
                         throw py::value_error("camera profile must be 'sub' or 'main'");
                     }
                     cfg.cameras.push_back(std::move(spec));
                 }
                 if (stream_profile != "sub" && stream_profile != "main") {
                     throw py::value_error("stream_profile must be 'sub' or 'main'");
                 }
                 cfg.stream_profile = stream_profile;
                 cfg.rtsp_transport = rtsp_transport;
                 cfg.substream_aspect = substream_aspect;
                 cfg.sim_threshold = sim_threshold;
                 cfg.frame_w = frame_w;
                 cfg.frame_h = frame_h;
                 cfg.mode_label = mode_label;
                 cfg.backend_label = backend_label;
                 if (jpeg_quality < 1 || jpeg_quality > 100) throw py::value_error("jpeg_quality must be 1..100");
                 cfg.jpeg_quality = jpeg_quality;
                 cfg.hw_decode = hw_decode;
                 cfg.gpu_max_width = gpu_max_width;
                 cfg.smooth_ms = std::max(0, smooth_ms);
                 if (track_lost_s < 0 || reverify_s <= 0 || max_embed_per_frame < 1) {
                     throw py::value_error("track_lost_s >= 0, reverify_s > 0 and max_embed_per_frame >= 1 are required");
                 }
                 cfg.track_lost_s = track_lost_s;
                 cfg.reverify_s = reverify_s;
                 cfg.max_embed_per_frame = max_embed_per_frame;
                 return std::make_shared<AttendanceSession>(std::move(engine), std::move(tracker), e.clone(),
                                                            meta_from_py(meta), cfg);
             }),
             py::arg("engine"), py::arg("tracker"), py::arg("embeds"), py::arg("meta"), py::arg("cameras"),
             py::arg("stream_profile") = "sub", py::arg("rtsp_transport") = "tcp",
             py::arg("substream_aspect") = "auto", py::arg("sim_threshold") = 0.45f, py::arg("frame_w") = 1280, py::arg("frame_h") = 720,
             py::arg("mode_label") = "DeepX DX-M1 NPU", py::arg("backend_label") = "GStreamer",
             py::arg("jpeg_quality") = 95, py::arg("hw_decode") = true, py::arg("gpu_max_width") = 1920,
             py::arg("smooth_ms") = 0, py::arg("track_lost_s") = 2.0, py::arg("reverify_s") = 1.0,
             py::arg("max_embed_per_frame") = 12)
        .def("start", &AttendanceSession::start, py::call_guard<py::gil_scoped_release>())
        .def("stop", &AttendanceSession::stop, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("running", &AttendanceSession::running)
        .def("mjpeg_frame",
             [](AttendanceSession& s) {
                 std::vector<uchar> jpg;
                 {
                     py::gil_scoped_release nogil;
                     jpg = s.mjpeg_frame();
                 }
                 return py::bytes(reinterpret_cast<const char*>(jpg.data()), jpg.size());
             })
        .def("camera_ids", &AttendanceSession::camera_ids)
        .def("camera",
             [](std::shared_ptr<AttendanceSession> s, const std::string& cam_id) {
                 s->camera_status(cam_id);  // throws for an unknown id
                 return CameraHandle{std::move(s), cam_id};
             },
             py::arg("cam_id"))
        .def("fps", &AttendanceSession::fps);
}
