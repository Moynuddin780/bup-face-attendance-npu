#ifndef BUP_BYTE_TRACKER_HPP
#define BUP_BYTE_TRACKER_HPP

/**
 * ByteTrack (Zhang et al., ECCV 2022) multi-object tracker for the faces of one camera.
 *
 * Every detection is used, not only the confident ones:
 *   1. confident detections (score >= high_thresh) are matched to all tracks — tracked and
 *      recently lost — by IoU with the Kalman-predicted boxes (fused with the score);
 *   2. the remaining tracks get a second chance with the weak detections
 *      (low_thresh <= score < high_thresh): turned, blurred or half-occluded faces keep their track;
 *   3. unmatched confident detections start new tracks (confirmed by a second hit), and a
 *      track nobody matches is kept "lost" for lost_keep_s so a face that was briefly hidden
 *      gets its old track (and its identity) back.
 *
 * The Kalman filter is the constant-velocity xyah filter of ByteTrack/DeepSORT, with the
 * step scaled by the real time between frames, because the NPU's frame rate varies.
 * Not thread-safe: one tracker per camera, driven by that camera's worker thread.
 */

#include <array>
#include <memory>
#include <vector>

namespace bup {

struct TrackDet {
    std::array<float, 4> tlbr{};   ///< x1, y1, x2, y2
    float score{0.0f};
};

struct TrackerParams {
    float high_thresh{0.5f};             ///< first association; new tracks need at least new_track_thresh
    float low_thresh{0.3f};              ///< weaker detections only extend existing tracks
    float new_track_thresh{0.5f};
    float match_thresh{0.8f};            ///< max 1 - IoU*score, first association
    float second_match_thresh{0.5f};     ///< max 1 - IoU, weak detections
    float unconfirmed_match_thresh{0.7f};
    double lost_keep_s{2.0};             ///< how long a lost track waits to be found again
    double step_hz{25.0};                ///< Kalman time unit (velocities are per 1/step_hz s)
};

struct TrackOut {
    int id{0};
    std::array<float, 4> tlbr{};   ///< Kalman-filtered box (predicted when not matched now)
    float vx{0.0f};                ///< centre velocity, pixels per second
    float vy{0.0f};
    float score{0.0f};             ///< score of the detection matched in this update
    int det{-1};                   ///< index into the update's detections, -1 = not matched now
    bool activated{false};         ///< confirmed (seen in two frames, or in the first frame)
    bool lost{false};
    double last_seen{0.0};         ///< time of the last matched detection
    int hits{0};                   ///< matched detections so far
};

class ByteTracker {
public:
    explicit ByteTracker(TrackerParams p = TrackerParams());
    ~ByteTracker();
    ByteTracker(ByteTracker&&) noexcept;
    ByteTracker& operator=(ByteTracker&&) noexcept;
    ByteTracker(const ByteTracker&) = delete;
    ByteTracker& operator=(const ByteTracker&) = delete;

    /// One frame at time t (seconds, any monotonic clock). Returns every live track — the
    /// ones matched in this frame (det >= 0) and the lost ones still waiting (lost == true).
    std::vector<TrackOut> update(const std::vector<TrackDet>& dets, double t);
    void reset();
    const TrackerParams& params() const { return p_; }

private:
    struct Track;
    TrackerParams p_;
    std::vector<std::unique_ptr<Track>> tracks_;   ///< tracked (confirmed or not) and lost
    int next_id_{1};
    long frame_{0};
    double last_t_{-1.0};
};

/// Minimum-cost assignment with a cost limit (lapjv's extend_cost semantics): pairs costing
/// more than `limit` are never made. Returns, per row, the matched column or -1.
std::vector<int> linear_assignment(const std::vector<std::vector<float>>& cost, float limit);

}  // namespace bup

#endif  // BUP_BYTE_TRACKER_HPP
