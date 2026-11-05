#ifndef BUP_FACE_ENGINE_HPP
#define BUP_FACE_ENGINE_HPP

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// DXRT >= 3.4 ships the single-header C++ API (dxrt_api.h became an obsolete alias of it);
// DXRT <= 3.3 only has dxrt_api.h. Both expose the same dxrt:: classes used here.
#if __has_include(<dxrt/dxrt_cxx_api.h>)
#include <dxrt/dxrt_cxx_api.h>
#else
#include <dxrt/dxrt_api.h>
#endif
#include <opencv2/core.hpp>

#include "bup/face_align.hpp"
#include "bup/factories.hpp"

namespace bup {

struct Face {
    std::array<float, 4> bbox{};   ///< x1, y1, x2, y2 in source-image pixels
    float det_score{0.0f};
    Landmarks5 kps{};              ///< 5-point landmarks in source-image pixels
    std::vector<float> embedding;  ///< L2-normalized ArcFace feature; empty until embed()
};

struct FaceEngineConfig {
    std::string det_model_path;
    std::string rec_model_path;
    float det_score_threshold{0.5f};  ///< original app: conf_thresh=0.5
    /// SCRFD also decodes faces down to this score for the tracker's second association
    /// (detect(img, min_score)); detect(img) / get(img) still return only >= det_score_threshold.
    float det_low_threshold{0.3f};
    float det_nms_threshold{0.4f};    ///< original app: NMS IoU 0.4
    /// NPU core binding: "auto" | "all" | "0" | "1" | "2" | "01" | "12" | "02".
    /// "auto" = one NPU_ALL engine per model (DXRT spreads jobs over the 3 cores). When dxrtd
    /// refuses NPU_ALL — it does while another process holds 3 distinct core bindings (DXRT
    /// BoundManager allows at most 3 bound types per device) — it opens one engine per core
    /// instead and spreads the work over that pool itself.
    std::string npu_bound{"auto"};
};

struct EngineStats {
    std::uint64_t det_calls{0};
    std::uint64_t rec_calls{0};      ///< embed() calls (one per frame with faces)
    std::uint64_t rec_faces{0};      ///< embeddings produced
    double det_npu_ms{0.0};          ///< summed submit->result wall time of detection
    double rec_npu_ms{0.0};
};

/**
 * SCRFD-10G detection + ArcFace-iResNet100 recognition on the DX-M1 NPU.
 *
 * Thread-safe: any number of camera workers may call detect()/embed() at once.
 * Every face of a frame is submitted with RunAsync before any Wait, so the faces are
 * recognised in parallel on the NPU cores instead of one after another; detection calls
 * from different cameras likewise land on different cores.
 */
class FaceEngine {
public:
    explicit FaceEngine(FaceEngineConfig cfg);
    ~FaceEngine();
    FaceEngine(const FaceEngine&) = delete;
    FaceEngine& operator=(const FaceEngine&) = delete;

    std::vector<Face> detect(const cv::Mat& bgr);                   ///< faces >= det_score_threshold
    std::vector<Face> detect(const cv::Mat& bgr, float min_score);  ///< faces >= min_score (none below min(det_score_threshold, det_low_threshold))
    void embed(const cv::Mat& bgr, std::vector<Face>& faces);
    std::vector<Face> get(const cv::Mat& bgr);  ///< detect + embed (enrollment / tools)

    int det_input_width() const { return det_w_; }
    int det_input_height() const { return det_h_; }
    int rec_input_width() const { return rec_w_; }
    int rec_input_height() const { return rec_h_; }
    int embedding_dim() const { return emb_dim_; }
    const FaceEngineConfig& config() const { return cfg_; }
    std::string det_model_name() const { return det_factory_.getModelName(); }
    /// the recognition model file without folder and extension (models are swapped by REC_MODEL_PATH)
    std::string rec_model_name() const {
        std::string f = cfg_.rec_model_path.substr(cfg_.rec_model_path.find_last_of('/') + 1);
        const auto dot = f.rfind(".dxnn");
        return dot == std::string::npos ? f : f.substr(0, dot);
    }
    std::string det_bound() const { return join(det_bounds_); }  ///< bindings granted, e.g. "all" or "0,1,2"
    std::string rec_bound() const { return join(rec_bounds_); }
    EngineStats stats() const;

private:
    using EnginePtr = std::unique_ptr<dxrt::InferenceEngine>;
    static std::vector<EnginePtr> open_pool(const std::string& path, const std::string& what,
                                            const std::string& bound, std::vector<std::string>& granted);
    static void parse_nhwc_uint8(dxrt::InferenceEngine& ie, const std::string& what, int& w, int& h,
                                 std::size_t& bytes);
    static void copy_input(const cv::Mat& pre, std::vector<uint8_t>& buf, const char* what);
    static std::string join(const std::vector<std::string>& v);

    FaceEngineConfig cfg_;
    ScrfdFactory det_factory_;
    ArcFaceFactory rec_factory_;
    std::vector<std::string> det_bounds_;
    std::vector<std::string> rec_bounds_;
    std::vector<EnginePtr> det_pool_;
    std::vector<EnginePtr> rec_pool_;
    std::atomic<std::uint32_t> det_rr_{0};
    std::atomic<std::uint32_t> rec_rr_{0};
    dxapp::PreprocessorPtr det_pre_;
    dxapp::PreprocessorPtr rec_pre_;
    dxapp::PostprocessorPtr<dxapp::FaceDetectionResult> det_post_;
    dxapp::PostprocessorPtr<dxapp::EmbeddingResult> rec_post_;
    int det_w_{0}, det_h_{0}, rec_w_{0}, rec_h_{0}, emb_dim_{0};
    std::size_t det_in_bytes_{0}, rec_in_bytes_{0};

    mutable std::mutex stats_mu_;
    EngineStats stats_;
};

}  // namespace bup

#endif  // BUP_FACE_ENGINE_HPP
