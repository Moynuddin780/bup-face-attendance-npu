#ifndef BUP_SESSION_HPP
#define BUP_SESSION_HPP

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "bup/capture.hpp"
#include "bup/face_engine.hpp"
#include "bup/tracker.hpp"

namespace bup {

/// One camera of a room. Empty / zero fields fall back to the session-wide SessionConfig values.
struct CameraSpec {
    std::string id;
    std::string url;
    std::string profile;     ///< "main" | "sub" ("" = SessionConfig::stream_profile)
    std::string aspect;      ///< "auto" | "off" | "W:H" ("" = SessionConfig::substream_aspect)
    bool fixed_url{false};   ///< url already points at the chosen stream: never rewritten
    std::string codec;       ///< codec the camera reported for it ("H264", "H265"; "" = unknown): tried first
    int max_w{-1};           ///< processing cap; 0 = none (process at the stream's own size), -1 = frame_w/h
    int max_h{-1};
    std::string file;        ///< a recorded video played instead of the camera (test rooms); url unused
};

struct SessionConfig {
    std::vector<CameraSpec> cameras;   ///< in room order
    std::string stream_profile{"sub"};
    std::string rtsp_transport{"tcp"};
    std::string substream_aspect{"auto"};  ///< see CaptureConfig::aspect
    float sim_threshold{0.45f};        ///< SIM_THRESHOLD
    int frame_w{1280};                 ///< FRAME_W: larger frames are downscaled to fit
    int frame_h{720};                  ///< FRAME_H
    std::string mode_label{"DeepX DX-M1 NPU"};
    std::string backend_label{"GStreamer"};
    int jpeg_quality{95};              ///< cv2.imencode default
    bool hw_decode{true};              ///< GPU (VA-API) decoding when available
    int gpu_max_width{1920};           ///< see CaptureConfig::gpu_max_width
    int smooth_ms{0};                  ///< see CaptureConfig::smooth_ms
    // face tracking (ByteTrack) + recognition per track
    double track_lost_s{2.0};          ///< a hidden face keeps its track (and name) this long
    double reverify_s{1.0};            ///< a recognised track is re-checked against its student this often
    int max_embed_per_frame{12};       ///< ArcFace budget per frame and camera (unknown tracks first)
};

struct CameraStatus {
    CaptureInfo capture;
    double infer_fps{0.0};
    std::uint64_t processed{0};
    std::uint64_t errors{0};
    double det_ms_avg{0.0};            ///< per frame, detect stage (preprocess + NPU + decode)
    double rec_ms_avg{0.0};            ///< per frame with faces, recognize stage (align + NPU batch)
    std::size_t last_faces{0};
    int infer_w{0};                    ///< size of the frames detection / recognition run on
    int infer_h{0};
    std::size_t tracks{0};             ///< faces being tracked now
    std::size_t identified{0};         ///< ... of them recognised as a student of the class
    std::uint64_t embeds{0};           ///< ArcFace embeddings computed so far
};

/**
 * One live attendance session (the C++ replacement of RTSPCapture threads,
 * _process_camera and _gen_feed).
 *
 * Per camera, a CascadeRunner of three threads:
 *   capture   (RtspCapture)  newest decoded frame, sequence-numbered (GStreamer, VA-API decode)
 *   detect    SCRFD on NPU   -> hand-off queue (depth 2, oldest dropped)
 *   recognize ByteTrack over all faces -> ArcFace on NPU for the tracks that need it ->
 *             embedding match per track -> tracker.update for every recognised track -> overlay
 * so SCRFD on frame t+1 overlaps ArcFace on frame t, and both cameras feed the
 * NPU cores concurrently.
 *
 * Recognition follows the track, not the single frame: a track's embeddings are aggregated
 * until they match one student (twice, or once clearly); from then on the track keeps that
 * name while the face is turned, blurred or briefly hidden, is re-checked every reverify_s,
 * and gives up the name if the face on it turns out to be someone else. A student is held by
 * at most one track per camera. The MJPEG composite is rendered on demand and cached
 * until a camera frame or the stats change, so N browser tabs cost one encode.
 */
class AttendanceSession {
public:
    AttendanceSession(std::shared_ptr<FaceEngine> engine, std::shared_ptr<AttendanceTracker> tracker,
                      cv::Mat class_embeddings, std::vector<StudentMeta> meta, SessionConfig cfg);
    ~AttendanceSession();
    AttendanceSession(const AttendanceSession&) = delete;
    AttendanceSession& operator=(const AttendanceSession&) = delete;

    void start();
    void stop();  ///< idempotent; joins every thread
    bool running() const { return running_; }

    std::vector<uchar> mjpeg_frame();
    std::vector<std::string> camera_ids() const;
    CameraStatus camera_status(const std::string& cam_id) const;
    double fps() const;  ///< mean processed fps over cameras that reported one

private:
    struct Camera;
    void detect_loop(Camera& cam);
    void recognize_loop(Camera& cam);
    int best_match(const std::vector<float>& emb, float& score) const;
    float similarity(const std::vector<float>& emb, int student) const;
    void identify(Camera& cam, int track_id, const std::vector<float>& emb, double now) const;

    std::shared_ptr<FaceEngine> engine_;
    std::shared_ptr<AttendanceTracker> tracker_;
    cv::Mat class_embs_;  ///< N x D, CV_32F, rows L2-normalized (np.savez cache)
    std::vector<StudentMeta> meta_;
    SessionConfig cfg_;
    std::vector<std::unique_ptr<Camera>> cams_;
    std::atomic<bool> running_{false};
    std::mutex lifecycle_mu_;

    std::mutex render_mu_;
    std::vector<std::uint64_t> cached_key_;
    std::vector<uchar> cached_jpeg_;
};

}  // namespace bup

#endif  // BUP_SESSION_HPP
