#include "bup/overlay.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <opencv2/imgproc.hpp>

namespace bup {

namespace {

// Labels are drawn on the live-view tile itself (kFeedWidth wide), so every camera shows them at
// the same size whatever its resolution.
constexpr int kFeedWidth = 800;
constexpr double kLabelScale = 0.62;

void draw_label(cv::Mat& tile, const std::string& text, const cv::Point& box_top_left, const cv::Scalar& color) {
    const int thick = 2;
    int base = 0;
    const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, kLabelScale, thick, &base);
    // above the box; inside its top edge when the face touches the top of the picture
    const int y = box_top_left.y - 6 >= ts.height + 2 ? box_top_left.y - 6 : box_top_left.y + ts.height + 6;
    cv::putText(tile, text, cv::Point(std::max(0, box_top_left.x), y), cv::FONT_HERSHEY_SIMPLEX, kLabelScale, color, thick);
}

}  // namespace

std::string minutes_text(double seconds) {
    // tenths of a minute, counted up (0.1 min = 6 s): 0 min, 0.1 min ... 1 min, 1.5 min, 4 min
    const long tenths = seconds > 0 ? static_cast<long>(seconds / 6.0) : 0;
    char out[32];
    if (tenths % 10 == 0) {
        std::snprintf(out, sizeof(out), "%ld min", tenths / 10);
    } else {
        std::snprintf(out, sizeof(out), "%ld.%ld min", tenths / 10, tenths % 10);
    }
    return out;
}

std::string match_label(const std::string& name, double total_time) {
    return name + " " + minutes_text(total_time);
}

cv::Mat compose_feed(const std::vector<FeedTile>& tiles) {
    std::vector<cv::Mat> parts;
    for (const FeedTile& t : tiles) {
        if (t.frame.empty()) continue;
        const cv::Mat& f = t.frame;
        const int h = std::max(1, static_cast<int>(std::lround(static_cast<double>(f.rows) * kFeedWidth / f.cols)));
        cv::Mat tile;
        cv::resize(f, tile, cv::Size(kFeedWidth, h), 0, 0, cv::INTER_LINEAR);
        // the latest faces, mapped from the processing frame onto the tile (same shape)
        if (t.box_w > 0 && t.box_h > 0) {
            const double sx = static_cast<double>(tile.cols) / t.box_w, sy = static_cast<double>(tile.rows) / t.box_h;
            for (const OverlayBox& b : t.boxes) {
                const cv::Scalar color = b.known ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 0, 220);
                const cv::Point p1(static_cast<int>(b.bbox[0] * sx), static_cast<int>(b.bbox[1] * sy));
                const cv::Point p2(static_cast<int>(b.bbox[2] * sx), static_cast<int>(b.bbox[3] * sy));
                cv::rectangle(tile, p1, p2, color, 2);
                draw_label(tile, b.text, p1, color);
            }
        }
        if (tiles.size() > 1 || !t.note.empty()) {
            std::string name = tiles.size() > 1 ? "Camera " + std::to_string(&t - tiles.data() + 1) : std::string();
            if (!t.note.empty()) name += (name.empty() ? "" : " - ") + t.note;
            cv::putText(tile, name, cv::Point(12, 34), cv::FONT_HERSHEY_SIMPLEX, 0.85, cv::Scalar(0, 0, 0), 5);
            cv::putText(tile, name, cv::Point(12, 34), cv::FONT_HERSHEY_SIMPLEX, 0.85, cv::Scalar(0, 255, 0), 2);
        }
        if (!parts.empty()) parts.push_back(cv::Mat(4, kFeedWidth, CV_8UC3, cv::Scalar(20, 20, 20)));
        parts.push_back(tile);
    }
    cv::Mat combined;
    if (parts.empty()) {
        combined = cv::Mat::zeros(450, kFeedWidth, CV_8UC3);
        cv::putText(combined, "Waiting for camera...", cv::Point(30, 240), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(0, 220, 220), 2);
    } else {
        cv::vconcat(parts, combined);
    }
    return combined;
}

std::string stats_text(const StatsBarValues& v) {
    char rate[32], fps[32];
    if (v.total == 0) {
        std::snprintf(rate, sizeof(rate), "0");
    } else {
        std::snprintf(rate, sizeof(rate), "%.1f", v.rate);  // repr(round(x, 1))
    }
    if (v.has_fps) {
        std::snprintf(fps, sizeof(fps), "%.1f", v.fps);
    } else {
        std::snprintf(fps, sizeof(fps), "0");
    }
    // what the teacher needs at a glance: who is present, for how long, how fast the NPU runs, and
    // at which resolution each camera delivers its picture
    std::string text = "Present: " + std::to_string(v.present) + "/" + std::to_string(v.total) + "  |  Time: " + minutes_text(v.duration);
    (void)fps;   // the per-camera feed rate below replaces the overall pipeline rate
    for (std::size_t i = 0; i < v.camera_sizes.size(); ++i) {
        text += "  |  Cam " + std::to_string(i + 1) + ": " + v.camera_sizes[i];
    }
    (void)rate;
    return text;
}

void draw_stats_bar(cv::Mat& img, const std::string& text) {
    // its own strip under the cameras: the bar never hides part of a camera picture
    cv::Mat with_strip;
    cv::copyMakeBorder(img, with_strip, 0, 45, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    img = with_strip;
    const int h = img.rows;
    double scale = 0.55;   // shrink to fit narrow views (a single 640-px camera)
    int base = 0;
    while (scale > 0.3 && cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, 2, &base).width > img.cols - 20) scale -= 0.05;
    cv::putText(img, text, cv::Point(10, h - 15), cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255, 255, 255), 2);
}

}  // namespace bup
