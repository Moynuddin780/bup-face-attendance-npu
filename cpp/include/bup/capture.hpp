#ifndef BUP_CAPTURE_HPP
#define BUP_CAPTURE_HPP

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <opencv2/core.hpp>

namespace bup {

bool gstreamer_available();
/// Intel VA-API decoding (vah264dec / vah265dec + vapostproc) works here: checked once with a
/// tiny pipeline through the GPU post-processor. False without /dev/dri access or the iHD driver.
bool gpu_decoding_available();

/// What a recorded video file holds (read with GStreamer, CPU decoding of the first frame).
struct VideoProbe {
    bool ok{false};
    std::string error;
    int width{0};
    int height{0};
    double fps{0.0};          ///< 0 = variable / unknown
    double duration_s{0.0};   ///< 0 = unknown
    std::string codec;        ///< "H.264", "H.265", "MPEG-4", "VP9", ... or the decoder's name
    cv::Mat first;            ///< first decoded frame (BGR)
};
VideoProbe probe_video(const std::string& path, double timeout_s = 15.0);

struct CaptureConfig {
    std::string cam_id;
    std::string url;                 ///< as stored in the DB (main stream)
    std::string profile{"sub"};      ///< "sub" | "main" (see resolve_stream_url)
    std::string transport{"tcp"};    ///< rtspsrc protocols: tcp | udp | udp+tcp
    int latency_ms{100};             ///< floor; raised on slow links (see RtspCapture)
    double stall_timeout_s{10.0};    ///< no frame for this long -> reconnect (floor, as latency_ms). A
                                     ///< reconnect costs a Wi-Fi camera 10-20 s of video; a stall rarely lasts that long
    int decoder_threads{0};          ///< avdec max-threads (0 = auto)
    /// Sub-stream pixel-aspect correction: "auto" (learn the scene aspect from one main-stream
    /// frame), "off", or "W:H" (e.g. "16:9"). Dahua D1 sub-streams (704x576) carry a 16:9 scene
    /// squashed horizontally; ArcFace needs undistorted faces, so frames are re-stretched.
    std::string aspect{"auto"};
    /// url already is the stream to read (e.g. the URI the camera reported for its sub-stream):
    /// no vendor rewriting; profile then only tells whether the aspect correction applies.
    bool fixed_url{false};
    /// Codec the camera reported for this stream ("H264", "H265", "hevc", ...; "" = unknown). Its
    /// pipeline is tried first, so the first connect does not spend a camera session on a guess.
    std::string codec_hint;
    /// GPU decoding (VA-API) when available: the CPU then only converts the decoded picture.
    bool hw_decode{true};
    /// Streams wider than this are scaled on the GPU to this width (same shape), so a 4-6 MP
    /// camera costs the CPU a 1920-px frame, not a 2560/3200-px one. 0 = native size.
    int gpu_max_width{1920};
    /// Playout buffer (ms): frames are released at their own timestamps after this delay, so a
    /// camera whose packets arrive in bursts (Wi-Fi jitter) still gives an even frame rate.
    /// 0 = every frame as soon as it is decoded.
    int smooth_ms{0};
    /// A local video file to play instead of the camera (test rooms): decoded like a camera
    /// stream (GPU when available), released in real time at the file's own frame rate, and
    /// looped, so the session sees a camera that never ends. url is then ignored.
    std::string file;
};

struct FramePacket {
    cv::Mat bgr;
    std::uint64_t seq{0};
    double ts{0.0};                  ///< wall-clock time the frame was received
};

struct CaptureInfo {
    std::string cam_id;
    std::string backend{"Not connected"};
    bool has_frame{false};
    int fail_count{0};
    int width{0};
    int height{0};
    double decode_fps{0.0};
    std::uint64_t frames{0};
    int reconnects{0};
    std::string profile;
    bool url_rewritten{false};
    std::string url_masked;
    double scene_aspect{0.0};        ///< target W/H after correction (0 = none)
    int out_width{0};                ///< frame size handed to the pipeline (after correction)
    int out_height{0};
    double rtt_ms{-1.0};             ///< TCP handshake time to the camera (-1 = not measured)
    int native_width{0};             ///< the camera's own stream size (before any GPU scaling)
    int native_height{0};
    bool gpu{false};                 ///< decoded on the GPU (VA-API)
    double stream_kbps{0.0};         ///< video data received from the camera (GPU receiver), kbit/s
    std::uint64_t stream_gaps{0};    ///< times data was lost on the way (frames skipped to the next key frame)
    std::uint64_t frames_skipped{0}; ///< compressed frames not decoded because of such a gap
};

/**
 * RTSP camera reader: GStreamer appsink with Intel VA-API (GPU) decoding when available, then
 * software (CPU) decoding, FFmpeg as the last resort. Only the newest frame is kept
 * (appsink drop=true max-buffers=1); consumers wait on a sequence number, so no
 * frame is ever processed twice. A silent stream is treated as a failure after
 * stall_timeout_s and reconnected; a camera that answers is retried after 2..5 s, one whose
 * RTSP port is silent with the original back-off (3 s x attempts, max 30 s).
 *
 * Before each connect the camera's network round trip is measured (one TCP handshake with
 * its RTSP port). A camera behind a congested Wi-Fi/mesh hop can need seconds per RTSP
 * request, so the connect deadline, socket timeouts, jitter-buffer latency and stall timeout
 * grow with it; an unreachable port fails the attempt at once. The codec that worked last is
 * tried first on reconnect, which saves a full RTSP handshake per attempt.
 *
 * GPU decoding is split in two pipelines: the RTSP receiver (rtspsrc -> depay -> parse), rebuilt on
 * every (re)connect, feeds a decoder (appsrc -> vah26xdec -> vapostproc -> appsink) that lives as
 * long as the capture. GStreamer 1.24's vapostproc never frees ~17 file descriptors per instance,
 * so a decoder rebuilt on each reconnect of a flaky Wi-Fi camera ran the process out of file
 * descriptors within minutes and killed every running class.
 *
 * start()/stop() are counted: a capture shared by several sessions (acquire_capture) runs until
 * its last user stops it.
 */
class RtspCapture {
public:
    explicit RtspCapture(CaptureConfig cfg);
    ~RtspCapture();
    RtspCapture(const RtspCapture&) = delete;
    RtspCapture& operator=(const RtspCapture&) = delete;

    void start();  ///< counted: the first start() starts the reader thread
    void stop();   ///< counted: the last stop() joins the reader thread (<= ~0.5 s for GStreamer)

    /// Wait for a frame with seq > after_seq. False on timeout or stop.
    bool wait_next(std::uint64_t after_seq, FramePacket& out, double timeout_s);
    /// The newest frame without waiting (live view). False before the first frame.
    bool latest(FramePacket& out) const;
    CaptureInfo info() const;
    const std::string& resolved_url() const { return url_; }

private:
    struct GstSource;
    void run();
    void run_file();
    std::unique_ptr<GstSource> open_file(cv::Mat& first);
    void attach_gpu_scaler(GstSource& s);
    GstSource* va_decoder(const std::string& codec);   ///< the persistent GPU decoder for "H.264" / "H.265"
    /// GPU decoders of finished captures, reused by later ones (a new vapostproc costs ~17 descriptors
    /// for good; reusing one for another camera costs ~1)
    static std::multimap<std::string, std::unique_ptr<GstSource>>& parked_decoders();
    static std::mutex& parked_mu();
    static GstFlowReturn on_es_sample(GstAppSink* es, gpointer self);
    void halt();
    std::unique_ptr<GstSource> open_gstreamer(const std::string& url, cv::Mat& first,
                                              const std::string& prefer = std::string(),
                                              std::vector<std::string>* rejected = nullptr, bool gpu = true);
    bool read_gstreamer(GstSource& src, cv::Mat& frame);
    static GstPadProbeReturn on_decoded_caps(GstPad* pad, GstPadProbeInfo* info, gpointer self);
    void resolve_scene_aspect();
    void publish(cv::Mat&& frame);
    void sleep_interruptible(double seconds);

    CaptureConfig cfg_;
    std::string url_;
    bool rewritten_{false};
    double scene_aspect_{0.0};
    bool aspect_resolved_{false};
    double rtt_ms_{0.0};             ///< link round trip used for the timeouts (capture thread)
    std::string codec_;              ///< pipeline that connected last ("H.264", "H.265", "Auto")
    std::string hint_;               ///< pipeline for the codec the camera reported (cfg_.codec_hint)
    std::vector<std::string> rejected_;  ///< pipelines that ended "not-linked" on this camera (capture thread)
    std::map<std::string, std::unique_ptr<GstSource>> va_dec_;  ///< GPU decoders by codec (capture thread)
    std::string tag_;                ///< "Camera camera1 (192.168.68.103)" for the logs
    bool net_failed_{false};         ///< open_gstreamer: the last attempt failed on the network, not the codec
    std::uint64_t gaps_total_{0}, skipped_total_{0};   ///< receiver statistics of earlier connections
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::mutex life_mu_;
    int users_{0};                   ///< start() calls not yet matched by stop()

    mutable std::mutex mu_;
    std::condition_variable cv_;
    FramePacket latest_;
    CaptureInfo info_;
    double fps_window_start_{0.0};
    std::uint64_t fps_window_frames_{0};
};

/// The capture for this camera stream, shared by every session that reads the same stream (one
/// camera serving two rooms): one RTSP connection and one decoder for all of them. Cameras allow
/// only a few viewers, and a second connection to the same stream made them drop each other.
std::shared_ptr<RtspCapture> acquire_capture(const CaptureConfig& cfg);

/// Open file descriptors of this process, and its limit (RLIMIT_NOFILE soft).
int open_fd_count();
int open_fd_limit();

}  // namespace bup

#endif  // BUP_CAPTURE_HPP
