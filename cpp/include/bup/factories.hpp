#ifndef BUP_FACTORIES_HPP
#define BUP_FACTORIES_HPP

/**
 * dx_app IFactory implementations for the two NPU models used by the attendance
 * pipeline. Model-specific pre/post-processing is built from the vendored dx_app
 * components (third_party/dxapp) exactly as dx_app's own SCRFD10G and
 * ArcFace-iResNet100 examples do, plus two corrections for this deployment:
 *   - SCRFD outputs are bound by tensor NAME (dx_app decodes outputs[0..8] by position)
 *   - SCRFD letterbox pads with 0 (InsightFace reference) instead of YOLO's 114
 */

#include <string>
#include <vector>

#include "common/base/i_factory.hpp"
#include "common/processors/embedding_postprocessor.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/scrfd_face_postprocessor.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"

namespace bup {

/// SCRFD output tensor names, in the order dx_app's SCRFDPostProcess expects.
const std::vector<std::string>& scrfd_output_names();

/// dx_app SCRFD decode + NMS, with outputs bound by name and boxes/landmarks mapped
/// back through the letterbox (pad, scale) into source-image pixels.
class NamedScrfdPostprocessor : public dxapp::IPostprocessor<dxapp::FaceDetectionResult> {
public:
    NamedScrfdPostprocessor(int input_w, int input_h, float score_threshold, float nms_threshold,
                            bool is_ort_configured);
    std::vector<dxapp::FaceDetectionResult> process(const dxrt::TensorPtrs& outputs,
                                                    const dxapp::PreprocessContext& ctx) override;
    std::string getModelName() const override { return "SCRFD"; }

private:
    int input_w_;
    int input_h_;
    SCRFDPostProcess impl_;
};

/// Boxes + 5 landmarks + score (bupctl --out).
class FaceBoxVisualizer : public dxapp::IVisualizer<dxapp::FaceDetectionResult> {
public:
    cv::Mat draw(const cv::Mat& frame, const std::vector<dxapp::FaceDetectionResult>& results,
                 const dxapp::PreprocessContext& ctx) override;
    void setParameters(int line_thickness, double font_scale, float alpha) override;

private:
    int thickness_{2};
    double font_scale_{0.5};
};

/// Annotates an aligned ArcFace crop with its embedding size/norm (bupctl --crops).
class AlignedCropVisualizer : public dxapp::IVisualizer<dxapp::EmbeddingResult> {
public:
    cv::Mat draw(const cv::Mat& frame, const std::vector<dxapp::EmbeddingResult>& results,
                 const dxapp::PreprocessContext& ctx) override;
    void setParameters(int line_thickness, double font_scale, float alpha) override;

private:
    double font_scale_{0.3};
};

class ScrfdFactory : public dxapp::IFaceDetectionFactory {
public:
    ScrfdFactory(float score_threshold, float nms_threshold)
        : score_threshold_(score_threshold), nms_threshold_(nms_threshold) {}

    dxapp::PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<dxapp::DetectionPreprocessor>(input_width, input_height,
                                                              cv::COLOR_BGR2RGB, /*pad_value=*/0);
    }
    dxapp::PostprocessorPtr<dxapp::FaceDetectionResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured) override {
        return std::make_unique<NamedScrfdPostprocessor>(input_width, input_height, score_threshold_,
                                                         nms_threshold_, is_ort_configured);
    }
    dxapp::VisualizerPtr<dxapp::FaceDetectionResult> createVisualizer() override {
        return std::make_unique<FaceBoxVisualizer>();
    }
    std::string getModelName() const override { return "SCRFD10G"; }
    std::string getTaskType() const override { return "face_detection"; }

private:
    float score_threshold_;
    float nms_threshold_;
};

class ArcFaceFactory : public dxapp::IEmbeddingFactory {
public:
    dxapp::PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        // input is an already-aligned crop: resize is a no-op at 112x112, BGR->RGB
        return std::make_unique<dxapp::SimpleResizePreprocessor>(input_width, input_height,
                                                                 cv::COLOR_BGR2RGB);
    }
    dxapp::PostprocessorPtr<dxapp::EmbeddingResult> createPostprocessor(int input_width,
                                                                       int input_height) override {
        return std::make_unique<dxapp::GenericEmbeddingPostprocessor>(input_width, input_height,
                                                                      /*normalize=*/true);
    }
    dxapp::VisualizerPtr<dxapp::EmbeddingResult> createVisualizer() override {
        return std::make_unique<AlignedCropVisualizer>();
    }
    std::string getModelName() const override { return "ArcFace-iResNet100-MS1M"; }
    std::string getTaskType() const override { return "embedding"; }
};

}  // namespace bup

#endif  // BUP_FACTORIES_HPP
