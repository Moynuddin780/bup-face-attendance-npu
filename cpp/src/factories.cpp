#include "bup/factories.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace bup {

const std::vector<std::string>& scrfd_output_names() {
    static const std::vector<std::string> names = {
        "score_8", "score_16", "score_32", "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32"};
    return names;
}

NamedScrfdPostprocessor::NamedScrfdPostprocessor(int input_w, int input_h, float score_threshold,
                                                 float nms_threshold, bool is_ort_configured)
    : input_w_(input_w),
      input_h_(input_h),
      impl_(input_w, input_h, score_threshold, nms_threshold, is_ort_configured) {}

std::vector<dxapp::FaceDetectionResult> NamedScrfdPostprocessor::process(
    const dxrt::TensorPtrs& outputs, const dxapp::PreprocessContext& ctx) {
    // Bind by name and check each tensor's anchor count, so a recompiled .dxnn with a
    // different output order fails loudly instead of decoding scores as boxes.
    static const int strides[3] = {8, 16, 32};
    static const int widths[3] = {1, 4, 10};  // score, bbox, kps values per anchor
    dxrt::TensorPtrs ordered;
    ordered.reserve(9);
    const auto& names = scrfd_output_names();
    for (std::size_t i = 0; i < names.size(); ++i) {
        auto it = std::find_if(outputs.begin(), outputs.end(),
                               [&](const dxrt::TensorPtr& t) { return t && t->name() == names[i]; });
        if (it == outputs.end()) {
            throw std::runtime_error("SCRFD output tensor '" + names[i] +
                                     "' not found — incompatible detection model");
        }
        const int stride = strides[i % 3];
        const std::size_t anchors = static_cast<std::size_t>(input_w_ / stride) * (input_h_ / stride) * 2;
        const std::size_t expect = anchors * widths[i / 3] * sizeof(float);
        if ((*it)->size_in_bytes() != expect) {
            throw std::runtime_error("SCRFD output '" + names[i] + "' has " +
                                     std::to_string((*it)->size_in_bytes()) + " bytes, expected " +
                                     std::to_string(expect));
        }
        ordered.push_back(*it);
    }

    const std::vector<SCRFDResult> raw = impl_.postprocess(ordered);

    std::vector<dxapp::FaceDetectionResult> faces;
    faces.reserve(raw.size());
    const float s = ctx.scale > 0 ? ctx.scale : 1.0f;
    for (const auto& r : raw) {
        dxapp::FaceDetectionResult f;
        f.confidence = r.confidence;
        // letterbox inverse (dx_app detail::scaleBox / scaleKeypoint, letterbox branch)
        f.box = {(r.box[0] - ctx.pad_x) / s, (r.box[1] - ctx.pad_y) / s,
                 (r.box[2] - ctx.pad_x) / s, (r.box[3] - ctx.pad_y) / s};
        for (std::size_t k = 0; k + 1 < r.landmarks.size(); k += 2) {
            f.landmarks.emplace_back((r.landmarks[k] - ctx.pad_x) / s,
                                     (r.landmarks[k + 1] - ctx.pad_y) / s, 1.0f);
        }
        faces.push_back(std::move(f));
    }
    return faces;
}

cv::Mat FaceBoxVisualizer::draw(const cv::Mat& frame,
                                const std::vector<dxapp::FaceDetectionResult>& results,
                                const dxapp::PreprocessContext& /*ctx*/) {
    cv::Mat out = frame.clone();
    for (const auto& f : results) {
        if (f.box.size() < 4) continue;
        const cv::Point p1(static_cast<int>(f.box[0]), static_cast<int>(f.box[1]));
        const cv::Point p2(static_cast<int>(f.box[2]), static_cast<int>(f.box[3]));
        cv::rectangle(out, p1, p2, cv::Scalar(0, 220, 0), thickness_);
        for (const auto& kp : f.landmarks) {
            cv::circle(out, cv::Point(static_cast<int>(kp.x), static_cast<int>(kp.y)), 2,
                       cv::Scalar(0, 255, 255), -1);
        }
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.2f", f.confidence);
        cv::putText(out, buf, cv::Point(p1.x, std::max(15, p1.y - 8)), cv::FONT_HERSHEY_SIMPLEX,
                    font_scale_, cv::Scalar(0, 220, 0), thickness_);
    }
    return out;
}

void FaceBoxVisualizer::setParameters(int line_thickness, double font_scale, float /*alpha*/) {
    thickness_ = line_thickness;
    font_scale_ = font_scale;
}

cv::Mat AlignedCropVisualizer::draw(const cv::Mat& frame,
                                    const std::vector<dxapp::EmbeddingResult>& results,
                                    const dxapp::PreprocessContext& /*ctx*/) {
    cv::Mat out = frame.clone();
    if (!results.empty()) {
        double norm = 0.0;
        for (float v : results[0].embedding) norm += static_cast<double>(v) * v;
        char buf[48];
        std::snprintf(buf, sizeof(buf), "d=%d |e|=%.2f", results[0].dimension, std::sqrt(norm));
        cv::putText(out, buf, cv::Point(2, out.rows - 4), cv::FONT_HERSHEY_SIMPLEX, font_scale_,
                    cv::Scalar(0, 255, 255), 1);
    }
    return out;
}

void AlignedCropVisualizer::setParameters(int /*line_thickness*/, double font_scale, float /*alpha*/) {
    font_scale_ = font_scale;
}

}  // namespace bup
