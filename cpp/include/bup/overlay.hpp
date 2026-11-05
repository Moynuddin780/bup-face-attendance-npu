#ifndef BUP_OVERLAY_HPP
#define BUP_OVERLAY_HPP

/**
 * Pixel-for-pixel ports of the original drawing code in app.py:
 *   _process_camera  -> draw_match / draw_unknown
 *   _gen_feed        -> compose_feed / draw_stats_bar / stats_text
 * Same colours, fonts, scales, thicknesses, positions and text formats, so the
 * live page looks exactly as before.
 */

#include <array>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace bup {

/// One face to draw on the live view: green "<name> <seconds>s" or red "Unknown". bbox is in the
/// coordinates of the frame the face pipeline processed (FeedTile::box_w x box_h).
struct OverlayBox {
    std::array<float, 4> bbox{};
    std::string text;
    bool known{false};
    float vx{0.0f};   ///< tracked motion, pixels/s: the live view moves the box to the newer camera frame
    float vy{0.0f};
};

/// "<name> <seconds>s" — seconds formatted like Python f"{t:.0f}".
/// "0 min", "0.1 min" ... "1 min", "1.5 min": time in tenths of a minute, counted up.
std::string minutes_text(double seconds);
std::string match_label(const std::string& name, double total_time);

/// One camera of the live view: its newest frame (camera frame rate) and the latest results of
/// the face pipeline (NPU rate), drawn on top. The two rates are independent, so the picture
/// never waits for the NPU.
struct FeedTile {
    cv::Mat frame;
    std::vector<OverlayBox> boxes;
    int box_w{0};
    int box_h{0};
    std::string note;   ///< shown after the camera's name, e.g. "Recorded video"
};

/// "Waiting for camera..." / the cameras stacked top to bottom at one width, each undistorted.
cv::Mat compose_feed(const std::vector<FeedTile>& tiles);

struct StatsBarValues {
    std::size_t present{0};
    std::size_t total{0};
    double rate{0.0};            ///< ignored when total == 0 (Python prints the int 0)
    double duration{0.0};        ///< already rounded to whole seconds
    std::string mode_label;      ///< GPU_MODE, e.g. "DeepX DX-M1 NPU"
    std::string backend_label;   ///< STREAM_BACKEND, e.g. "GStreamer"
    bool has_fps{false};         ///< false -> Python prints the initial int 0
    double fps{0.0};
    std::vector<std::string> camera_sizes;  ///< per camera, room order: "2560x1440 25 fps" (stream size, feed rate), "--" before the first frame
};

std::string stats_text(const StatsBarValues& v);
void draw_stats_bar(cv::Mat& img, const std::string& text);

}  // namespace bup

#endif  // BUP_OVERLAY_HPP
