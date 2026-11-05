#include "bup/session.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <condition_variable>
#include <deque>
#include <functional>
#include <limits>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "bup/byte_tracker.hpp"
#include "bup/common.hpp"
#include "bup/overlay.hpp"

namespace bup {

namespace {

struct Detected {
    cv::Mat frame;             ///< processing-resolution frame (owned)
    double ts{0.0};            ///< wall-clock receive time -> tracker "now"
    std::vector<Face> faces;
};

constexpr std::size_t kHandoffDepth = 2;
constexpr double kBoxesMaxAgeMs = 1500.0;   ///< live view: older face results are not drawn (person left)
constexpr double kLostDrawS = 0.5;          ///< a recognised face hidden this long still shows (predicted box)
constexpr double kMaxLeadS = 0.3;           ///< live view moves boxes along their track at most this far
constexpr float kStrongMargin = 0.12f;      ///< one embedding this far above the threshold names a track at once
constexpr float kVerifySlack = 0.10f;       ///< re-check: still the same student down to threshold - slack
constexpr double kRetryUnknownS = 0.4;      ///< an unrecognised track (stranger, turned away) is retried this often

/// What a track is known to be. Lives as long as the track (tracked or lost).
struct TrackIdentity {
    int student{-1};          ///< row of the class gallery, -1 = not recognised (yet)
    float score{0.0f};        ///< similarity of `feat` to that student
    std::vector<float> feat;  ///< the track's embeddings aggregated (unit length)
    int n_emb{0};
    int mismatches{0};        ///< re-checks in a row that looked like another student
    double next_embed{0.0};   ///< when this track wants ArcFace again
};

/// Running mean of the first embeddings, then an exponential moving average (recent views count
/// more), kept at unit length so it can be matched like a single embedding.
void blend(TrackIdentity& id, const std::vector<float>& e) {
    if (id.feat.size() != e.size()) {
        id.feat = e;
        id.n_emb = 1;
        return;
    }
    const float w = id.n_emb < 10 ? 1.0f / static_cast<float>(id.n_emb + 1) : 0.1f;
    double norm = 0.0;
    for (std::size_t k = 0; k < e.size(); ++k) {
        id.feat[k] = (1.0f - w) * id.feat[k] + w * e[k];
        norm += static_cast<double>(id.feat[k]) * id.feat[k];
    }
    const float inv = norm > 0.0 ? static_cast<float>(1.0 / std::sqrt(norm)) : 0.0f;
    for (float& v : id.feat) v *= inv;
    id.n_emb += 1;
}

}  // namespace

struct AttendanceSession::Camera {
    std::string id;
    std::string note;   ///< live view caption ("Recorded video")
    int max_w{0};   ///< processing cap (0 = the stream's own size)
    int max_h{0};
    int infer_w{0}; ///< last processed frame size (guarded by out_mu)
    int infer_h{0};
    std::shared_ptr<RtspCapture> cap;   ///< shared with other sessions reading the same stream
    std::thread det_thr;
    std::thread rec_thr;

    // recognize thread only
    ByteTracker bt;
    std::unordered_map<int, TrackIdentity> ids;   ///< by track id

    // detect -> recognize hand-off
    std::mutex q_mu;
    std::condition_variable q_cv;
    std::deque<Detected> q;

    // published result + stats (guarded by out_mu): the latest faces, drawn by the live view on
    // the newest camera frame, so the picture runs at the camera's rate, not the NPU's
    mutable std::mutex out_mu;
    std::vector<OverlayBox> boxes;
    int box_w{0};
    int box_h{0};
    double boxes_ms{0.0};
    double boxes_ts{0.0};   ///< receive time of the frame the boxes belong to
    std::uint64_t out_seq{0};
    bool has_fps{false};
    double fps{0.0};
    double fps_start_ms{0.0};
    int fps_count{0};
    std::uint64_t processed{0};
    double det_ms_sum{0.0};
    std::uint64_t det_n{0};
    double rec_ms_sum{0.0};
    std::uint64_t rec_n{0};
    std::size_t last_faces{0};
    std::size_t tracks{0};
    std::size_t identified{0};
    std::uint64_t embeds{0};
    std::atomic<std::uint64_t> errors{0};
    LogThrottle err_log{10.0};
};

AttendanceSession::AttendanceSession(std::shared_ptr<FaceEngine> engine,
                                     std::shared_ptr<AttendanceTracker> tracker,
                                     cv::Mat class_embeddings, std::vector<StudentMeta> meta,
                                     SessionConfig cfg)
    : engine_(std::move(engine)),
      tracker_(std::move(tracker)),
      meta_(std::move(meta)),
      cfg_(std::move(cfg)) {
    if (!engine_ || !tracker_) throw std::invalid_argument("engine and tracker are required");
    if (!class_embeddings.empty()) {
        if (class_embeddings.type() != CV_32F || class_embeddings.dims != 2) {
            throw std::invalid_argument("class embeddings must be a 2-D float32 matrix");
        }
        if (class_embeddings.cols != engine_->embedding_dim()) {
            throw std::invalid_argument("class embeddings have " + std::to_string(class_embeddings.cols) +
                                        " dims, model produces " + std::to_string(engine_->embedding_dim()) +
                                        " — rebuild the embedding cache");
        }
        if (static_cast<std::size_t>(class_embeddings.rows) != meta_.size()) {
            throw std::invalid_argument("class embeddings rows != meta entries");
        }
        class_embs_ = class_embeddings.clone();
    }
    for (const auto& c : cfg_.cameras) {
        auto cam = std::make_unique<Camera>();
        cam->id = c.id;
        cam->max_w = c.max_w < 0 ? cfg_.frame_w : c.max_w;
        cam->max_h = c.max_h < 0 ? cfg_.frame_h : c.max_h;
        TrackerParams tp;
        tp.high_thresh = engine_->config().det_score_threshold;
        tp.new_track_thresh = tp.high_thresh;
        tp.low_thresh = std::min(tp.high_thresh, engine_->config().det_low_threshold);
        tp.lost_keep_s = cfg_.track_lost_s;
        cam->bt = ByteTracker(tp);
        CaptureConfig cc;
        cc.cam_id = c.id;
        cc.url = c.url;
        cc.profile = c.profile.empty() ? cfg_.stream_profile : c.profile;
        cc.transport = cfg_.rtsp_transport;
        cc.aspect = c.aspect.empty() ? cfg_.substream_aspect : c.aspect;
        cc.fixed_url = c.fixed_url;
        cc.codec_hint = c.codec;
        cc.hw_decode = cfg_.hw_decode;
        cc.gpu_max_width = cfg_.gpu_max_width;
        cc.smooth_ms = cfg_.smooth_ms;
        cc.file = c.file;
        cam->note = c.file.empty() ? std::string() : "Recorded video";
        cam->cap = acquire_capture(cc);
        cams_.push_back(std::move(cam));
    }
}

AttendanceSession::~AttendanceSession() { stop(); }

void AttendanceSession::start() {
    std::lock_guard<std::mutex> lk(lifecycle_mu_);
    if (running_) return;
    running_ = true;
    for (auto& cam : cams_) {
        cam->cap->start();
        Camera* c = cam.get();
        c->det_thr = std::thread([this, c] { detect_loop(*c); });
        c->rec_thr = std::thread([this, c] { recognize_loop(*c); });
    }
    log(LogLevel::Info, "session started: " + std::to_string(cams_.size()) + " camera(s), " +
                            std::to_string(meta_.size()) + " enrolled, " + cfg_.stream_profile + "-stream (open files " +
                            std::to_string(open_fd_count()) + "/" + std::to_string(open_fd_limit()) + ")");
}

void AttendanceSession::stop() {
    std::lock_guard<std::mutex> lk(lifecycle_mu_);
    const bool was_running = running_.exchange(false);
    for (auto& cam : cams_) {
        if (was_running) cam->cap->stop();   // counted: a capture another class still uses keeps running
        cam->q_cv.notify_all();
    }
    for (auto& cam : cams_) {
        if (cam->det_thr.joinable()) cam->det_thr.join();
        if (cam->rec_thr.joinable()) cam->rec_thr.join();
    }
    if (was_running) {
        log(LogLevel::Info, "session stopped (open files " + std::to_string(open_fd_count()) + "/" +
                                std::to_string(open_fd_limit()) + ")");
    }
}

void AttendanceSession::detect_loop(Camera& cam) {
    std::uint64_t seq = 0;
    while (running_) {
        FramePacket pkt;
        if (!cam.cap->wait_next(seq, pkt, 0.5)) continue;
        seq = pkt.seq;

        Detected d;
        d.ts = pkt.ts;
        if (cam.max_w > 0 && cam.max_h > 0 && (pkt.bgr.cols > cam.max_w || pkt.bgr.rows > cam.max_h)) {
            // fit inside the camera's processing cap keeping the aspect ratio (never squash faces)
            const double s = std::min(static_cast<double>(cam.max_w) / pkt.bgr.cols,
                                      static_cast<double>(cam.max_h) / pkt.bgr.rows);
            cv::resize(pkt.bgr, d.frame,
                       cv::Size(static_cast<int>(std::lround(pkt.bgr.cols * s)),
                                static_cast<int>(std::lround(pkt.bgr.rows * s))),
                       0, 0, cv::INTER_LINEAR);
        } else {
            d.frame = pkt.bgr.clone();  // the capture's buffer is shared: draw on our own copy
        }

        const double t0 = mono_ms();
        try {
            d.faces = engine_->detect(d.frame, engine_->config().det_low_threshold);   // weak ones feed the tracker
        } catch (const std::exception& e) {
            cam.errors += 1;
            if (cam.err_log.allow()) log(LogLevel::Error, "Camera " + cam.id + ": detection failed: " + e.what());
        }
        {
            std::lock_guard<std::mutex> lk(cam.out_mu);
            cam.det_ms_sum += mono_ms() - t0;
            cam.det_n += 1;
            cam.infer_w = d.frame.cols;
            cam.infer_h = d.frame.rows;
        }
        {
            std::lock_guard<std::mutex> lk(cam.q_mu);
            if (cam.q.size() >= kHandoffDepth) cam.q.pop_front();  // keep latency bounded
            cam.q.push_back(std::move(d));
        }
        cam.q_cv.notify_one();
    }
}

int AttendanceSession::best_match(const std::vector<float>& emb, float& score) const {
    score = -std::numeric_limits<float>::infinity();
    int best = -1;
    for (int r = 0; r < class_embs_.rows; ++r) {
        const float* row = class_embs_.ptr<float>(r);
        float dot = 0.0f;
        for (int k = 0; k < class_embs_.cols; ++k) dot += row[k] * emb[k];
        if (dot > score) {  // first maximum wins, like np.argmax
            score = dot;
            best = r;
        }
    }
    return best;
}

float AttendanceSession::similarity(const std::vector<float>& emb, int student) const {
    const float* row = class_embs_.ptr<float>(student);
    float dot = 0.0f;
    for (int k = 0; k < class_embs_.cols; ++k) dot += row[k] * emb[k];
    return dot;
}

void AttendanceSession::identify(Camera& cam, int track_id, const std::vector<float>& emb, double now) const {
    const float thr = cfg_.sim_threshold;
    TrackIdentity& id = cam.ids[track_id];
    if (id.student >= 0) {
        // re-check: is the face on this track still that student?
        if (similarity(emb, id.student) >= thr - kVerifySlack) {
            id.mismatches = 0;
            blend(id, emb);
            id.score = similarity(id.feat, id.student);
            id.next_embed = now + cfg_.reverify_s;
            return;
        }
        float s = 0.0f;
        const int b = best_match(emb, s);
        if (b < 0 || b == id.student || s < thr || ++id.mismatches < 2) {
            // a poor view (turned, blurred), or the first doubt: look again soon, keep the name
            id.next_embed = now + (id.mismatches ? 0.0 : cfg_.reverify_s / 2.0);
            return;
        }
        // twice in a row clearly someone else: the tracker swapped faces in a crowd — start over
        id = TrackIdentity();
    }

    // not recognised yet: aggregate the track's views, match the aggregate
    blend(id, emb);
    float s = 0.0f;
    const int b = best_match(id.feat, s);
    const bool sure = b >= 0 && s >= thr && (id.n_emb >= 2 || s >= thr + kStrongMargin);
    if (!sure) {
        id.next_embed = now + (id.n_emb < 3 ? 0.0 : kRetryUnknownS);
        return;
    }
    // one face per student per camera: the better match keeps the name, the other looks again
    for (auto& kv : cam.ids) {
        if (kv.first == track_id || kv.second.student != b) continue;
        if (kv.second.score >= s) {
            id.next_embed = now + kRetryUnknownS;
            return;
        }
        kv.second = TrackIdentity();
    }
    id.student = b;
    id.score = s;
    id.mismatches = 0;
    id.next_embed = now + cfg_.reverify_s;
}

void AttendanceSession::recognize_loop(Camera& cam) {
    const int dim = engine_->embedding_dim();
    const float high = engine_->config().det_score_threshold;
    const std::size_t budget = static_cast<std::size_t>(std::max(1, cfg_.max_embed_per_frame));
    while (running_) {
        Detected d;
        {
            std::unique_lock<std::mutex> lk(cam.q_mu);
            cam.q_cv.wait_for(lk, std::chrono::milliseconds(500), [&] { return !running_ || !cam.q.empty(); });
            if (!running_) break;
            if (cam.q.empty()) continue;
            d = std::move(cam.q.front());
            cam.q.pop_front();
        }

        // 1) ByteTrack over every face of the frame, the weak ones included
        std::vector<TrackDet> tds;
        tds.reserve(d.faces.size());
        std::size_t n_faces = 0;
        for (const Face& f : d.faces) {
            tds.push_back({f.bbox, f.det_score});
            if (f.det_score >= high) ++n_faces;
        }
        const std::vector<TrackOut> tracks = cam.bt.update(tds, d.ts);
        {   // identities end with their tracks
            std::unordered_map<int, TrackIdentity> alive;
            for (const TrackOut& t : tracks) {
                auto it = cam.ids.find(t.id);
                if (it != cam.ids.end()) alive.emplace(t.id, std::move(it->second));
            }
            cam.ids.swap(alive);
        }

        // 2) ArcFace only where it is needed: confirmed tracks seen clearly now that are not
        //    recognised yet (first), or are due for their periodic re-check
        struct Pick { std::size_t ti; bool known; double due; };
        std::vector<Pick> picks;
        for (std::size_t i = 0; i < tracks.size(); ++i) {
            const TrackOut& t = tracks[i];
            if (t.det < 0 || !t.activated || d.faces[static_cast<std::size_t>(t.det)].det_score < high) continue;
            const TrackIdentity& id = cam.ids[t.id];
            if (d.ts >= id.next_embed) picks.push_back({i, id.student >= 0, id.next_embed});
        }
        std::sort(picks.begin(), picks.end(), [](const Pick& a, const Pick& b) {
            return a.known != b.known ? !a.known : a.due < b.due;
        });
        if (picks.size() > budget) picks.resize(budget);
        std::vector<Face> batch;
        batch.reserve(picks.size());
        for (const Pick& p : picks) batch.push_back(d.faces[static_cast<std::size_t>(tracks[p.ti].det)]);
        double rec_ms = -1.0;
        if (!batch.empty()) {
            const double t0 = mono_ms();
            try {
                engine_->embed(d.frame, batch);
            } catch (const std::exception& e) {
                batch.clear();   // those tracks are simply tried again on the next frame
                cam.errors += 1;
                if (cam.err_log.allow()) log(LogLevel::Error, "Camera " + cam.id + ": recognition failed: " + e.what());
            }
            rec_ms = mono_ms() - t0;
        }
        for (std::size_t k = 0; k < batch.size(); ++k) {
            if (static_cast<int>(batch[k].embedding.size()) != dim) continue;
            identify(cam, tracks[picks[k].ti].id, batch[k].embedding, d.ts);
        }

        // 3) attendance for every recognised face seen in this frame, and the live boxes
        std::vector<OverlayBox> boxes;
        std::size_t n_tracks = 0, n_known = 0;
        for (const TrackOut& t : tracks) {
            auto it = cam.ids.find(t.id);
            const int student = it == cam.ids.end() ? -1 : it->second.student;
            const bool seen = t.det >= 0;
            if (!t.lost) {
                ++n_tracks;
                if (student >= 0) ++n_known;
            }
            if (seen && student >= 0) {
                tracker_->update(meta_[static_cast<std::size_t>(student)].student_id, cam.id, d.ts);
            }
            if (!seen && !(t.activated && student >= 0 && d.ts - t.last_seen <= kLostDrawS)) continue;
            OverlayBox b;
            b.bbox = t.tlbr;
            b.vx = t.vx;
            b.vy = t.vy;
            if (student >= 0) {
                const StudentMeta& m = meta_[static_cast<std::size_t>(student)];
                b.text = match_label(m.name, tracker_->total_time(m.student_id));
                b.known = true;
            } else {
                b.text = "Unknown";
            }
            boxes.push_back(std::move(b));
        }

        const double now_ms = mono_ms();
        std::lock_guard<std::mutex> lk(cam.out_mu);
        cam.boxes = std::move(boxes);
        cam.box_w = d.frame.cols;
        cam.box_h = d.frame.rows;
        cam.boxes_ms = now_ms;
        cam.boxes_ts = d.ts;
        cam.out_seq += 1;
        cam.processed += 1;
        cam.last_faces = n_faces;
        cam.tracks = n_tracks;
        cam.identified = n_known;
        cam.embeds += batch.size();
        if (rec_ms >= 0) {
            cam.rec_ms_sum += rec_ms;
            cam.rec_n += 1;
        }
        // original: fps = round(frame_count / elapsed, 1), refreshed every >= 1 s
        if (cam.fps_count == 0 && cam.fps_start_ms == 0.0) cam.fps_start_ms = now_ms;
        cam.fps_count += 1;
        const double elapsed = (now_ms - cam.fps_start_ms) / 1000.0;
        if (elapsed >= 1.0) {
            cam.fps = std::round(cam.fps_count / elapsed * 10.0) / 10.0;
            cam.has_fps = true;
            cam.fps_count = 0;
            cam.fps_start_ms = now_ms;
        }
    }
}

double AttendanceSession::fps() const {
    double sum = 0.0;
    int n = 0;
    for (const auto& cam : cams_) {
        std::lock_guard<std::mutex> lk(cam->out_mu);
        if (cam->has_fps) {
            sum += cam->fps;
            ++n;
        }
    }
    return n ? sum / n : 0.0;
}

std::vector<uchar> AttendanceSession::mjpeg_frame() {
    std::lock_guard<std::mutex> rlk(render_mu_);
    std::vector<FeedTile> tiles;
    std::vector<std::uint64_t> key;
    std::vector<std::string> labels;
    bool has_fps = false;
    double fps_sum = 0.0;
    int fps_n = 0;
    const double now_ms = mono_ms();
    for (const auto& cam : cams_) {
        // the newest camera frame (camera rate) + the latest faces (NPU rate)
        FeedTile t;
        t.note = cam->note;
        FramePacket pkt;
        if (cam->cap->latest(pkt)) t.frame = pkt.bgr;
        key.push_back(pkt.seq);
        const CaptureInfo ci = cam->cap->info();
        if (ci.native_width > 0) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%dx%d %.0f fps", ci.native_width, ci.native_height, ci.decode_fps);
            labels.push_back(buf);
        } else {
            labels.push_back("--");
        }
        std::lock_guard<std::mutex> lk(cam->out_mu);
        if (now_ms - cam->boxes_ms <= kBoxesMaxAgeMs) {
            t.boxes = cam->boxes;
            t.box_w = cam->box_w;
            t.box_h = cam->box_h;
            key.push_back(cam->out_seq);
            // the faces were found on an older frame than the one shown: move each box along its
            // track to this frame's time, so a walking student's name stays on their face
            const double lead = std::min(kMaxLeadS, std::max(0.0, pkt.ts - cam->boxes_ts));
            if (!t.frame.empty() && lead > 0.0) {
                for (OverlayBox& b : t.boxes) {
                    const float dx = static_cast<float>(b.vx * lead), dy = static_cast<float>(b.vy * lead);
                    b.bbox = {b.bbox[0] + dx, b.bbox[1] + dy, b.bbox[2] + dx, b.bbox[3] + dy};
                }
            }
        }
        if (cam->has_fps) {
            has_fps = true;
            fps_sum += cam->fps;
            ++fps_n;
        }
        tiles.push_back(std::move(t));
    }

    const LiveCounts lc = tracker_->live();
    StatsBarValues sv;
    sv.present = lc.present;
    sv.total = lc.total;
    sv.rate = lc.total ? static_cast<double>(lc.present) / static_cast<double>(lc.total) * 100.0 : 0.0;
    sv.duration = std::nearbyint(wall_time() - lc.session_start);  // round(x, 0): half-to-even
    sv.mode_label = cfg_.mode_label;
    sv.backend_label = cfg_.backend_label;
    sv.has_fps = has_fps;
    sv.fps = fps_n ? std::round(fps_sum / fps_n * 10.0) / 10.0 : 0.0;
    sv.camera_sizes = labels;

    key.push_back(tracker_->version());
    key.push_back(static_cast<std::uint64_t>(sv.duration));
    for (const std::string& l : labels) key.push_back(std::hash<std::string>()(l));
    if (key == cached_key_ && !cached_jpeg_.empty()) return cached_jpeg_;

    cv::Mat combined = compose_feed(tiles);
    draw_stats_bar(combined, stats_text(sv));
    std::vector<uchar> jpg;
    cv::imencode(".jpg", combined, jpg, {cv::IMWRITE_JPEG_QUALITY, cfg_.jpeg_quality});
    cached_key_ = std::move(key);
    cached_jpeg_ = jpg;
    return jpg;
}

std::vector<std::string> AttendanceSession::camera_ids() const {
    std::vector<std::string> ids;
    for (const auto& cam : cams_) ids.push_back(cam->id);
    return ids;
}

CameraStatus AttendanceSession::camera_status(const std::string& cam_id) const {
    for (const auto& cam : cams_) {
        if (cam->id != cam_id) continue;
        CameraStatus st;
        st.capture = cam->cap->info();
        st.capture.cam_id = cam->id;   // a shared capture carries the id of the session that opened it
        std::lock_guard<std::mutex> lk(cam->out_mu);
        st.infer_fps = cam->has_fps ? cam->fps : 0.0;
        st.processed = cam->processed;
        st.errors = cam->errors.load();
        st.det_ms_avg = cam->det_n ? cam->det_ms_sum / cam->det_n : 0.0;
        st.rec_ms_avg = cam->rec_n ? cam->rec_ms_sum / cam->rec_n : 0.0;
        st.last_faces = cam->last_faces;
        st.infer_w = cam->infer_w;
        st.infer_h = cam->infer_h;
        st.tracks = cam->tracks;
        st.identified = cam->identified;
        st.embeds = cam->embeds;
        return st;
    }
    throw std::out_of_range("unknown camera '" + cam_id + "'");
}

}  // namespace bup
