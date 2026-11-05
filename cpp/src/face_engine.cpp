#include "bup/face_engine.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "bup/common.hpp"

namespace bup {

namespace {

std::uint32_t bound_option(const std::string& name) {
    using B = dxrt::InferenceOption;
    if (name == "all") return B::NPU_ALL;
    if (name == "0") return B::NPU_0;
    if (name == "1") return B::NPU_1;
    if (name == "2") return B::NPU_2;
    if (name == "01") return B::NPU_01;
    if (name == "12") return B::NPU_12;
    if (name == "02") return B::NPU_02;
    throw std::invalid_argument("npu_bound must be auto|all|0|1|2|01|12|02, got '" + name + "'");
}

/// dxrtd refused to register the task (bound-type limit) — worth trying another binding.
bool is_bound_refusal(const dxrt::Exception& e) {
    return std::string(e.what()).find("task init") != std::string::npos;
}

}  // namespace

std::string FaceEngine::join(const std::vector<std::string>& v) {
    std::string s;
    for (const auto& x : v) s += (s.empty() ? "" : ",") + x;
    return s;
}

std::vector<FaceEngine::EnginePtr> FaceEngine::open_pool(const std::string& path, const std::string& what,
                                                         const std::string& bound,
                                                         std::vector<std::string>& granted) {
    std::vector<EnginePtr> pool;
    auto open = [&](const std::string& b) {
        dxrt::InferenceOption opt;  // ORT stays enabled (SCRFD has a CPU tail task)
        opt.boundOption = bound_option(b);
        pool.push_back(std::make_unique<dxrt::InferenceEngine>(path, opt));
        granted.push_back(b);
    };
    if (bound != "auto") {
        open(bound);
        return pool;
    }
    std::string refusal;
    try {
        open("all");
        return pool;
    } catch (const dxrt::Exception& e) {
        if (!is_bound_refusal(e)) throw;
        refusal = e.what();
    }
    // Shared NPU: one engine per core that dxrtd accepts, work spread by us.
    for (const char* core : {"0", "1", "2"}) {
        try {
            open(core);
        } catch (const dxrt::Exception& e) {
            if (!is_bound_refusal(e)) throw;
        }
    }
    if (pool.empty()) throw std::runtime_error(what + " model: dxrtd accepted no NPU binding (" + refusal + ")");
    log(LogLevel::Warn, what + " model: NPU_ALL refused by dxrtd (another process holds 3 core bindings); "
                              "using a per-core engine pool on core(s) " + join(granted));
    return pool;
}

FaceEngine::FaceEngine(FaceEngineConfig cfg)
    : cfg_(std::move(cfg)),
      det_factory_(std::min(cfg_.det_score_threshold, cfg_.det_low_threshold), cfg_.det_nms_threshold) {
    if (cfg_.npu_bound != "auto") bound_option(cfg_.npu_bound);  // validate early
    det_pool_ = open_pool(cfg_.det_model_path, "detection", cfg_.npu_bound, det_bounds_);
    rec_pool_ = open_pool(cfg_.rec_model_path, "recognition", cfg_.npu_bound, rec_bounds_);

    // DXNN models self-describe their input — never hardcode 640 / 112.
    parse_nhwc_uint8(*det_pool_.front(), "detection", det_w_, det_h_, det_in_bytes_);
    parse_nhwc_uint8(*rec_pool_.front(), "recognition", rec_w_, rec_h_, rec_in_bytes_);

    const auto names = det_pool_.front()->GetOutputTensorNames();
    for (const auto& want : scrfd_output_names()) {
        if (std::find(names.begin(), names.end(), want) == names.end()) {
            throw std::runtime_error("detection model lacks output '" + want +
                                     "' — expected an SCRFD .dxnn (score_/bbox_/kps_ x 8/16/32)");
        }
    }
    const auto rec_out = rec_pool_.front()->GetOutputs();
    if (rec_out.size() != 1) throw std::runtime_error("recognition model must have exactly 1 output");
    emb_dim_ = 1;
    for (auto d : rec_out.front().shape()) emb_dim_ *= static_cast<int>(d);

    det_pre_ = det_factory_.createPreprocessor(det_w_, det_h_);
    det_post_ = det_factory_.createPostprocessor(det_w_, det_h_, det_pool_.front()->IsOrtConfigured());
    rec_pre_ = rec_factory_.createPreprocessor(rec_w_, rec_h_);
    rec_post_ = rec_factory_.createPostprocessor(rec_w_, rec_h_);

    log(LogLevel::Info, det_factory_.getModelName() + " " + std::to_string(det_w_) + "x" +
                            std::to_string(det_h_) + " + ArcFace " + rec_model_name() + " " +
                            std::to_string(rec_w_) + "x" + std::to_string(rec_h_) + " -> " +
                            std::to_string(emb_dim_) + "-d, loaded on DX-M1 (NPU bound det=" + det_bound() +
                            " rec=" + rec_bound() + ")");
}

FaceEngine::~FaceEngine() = default;

void FaceEngine::parse_nhwc_uint8(dxrt::InferenceEngine& ie, const std::string& what, int& w, int& h,
                                  std::size_t& bytes) {
    const auto inputs = ie.GetInputs();
    if (inputs.size() != 1) throw std::runtime_error(what + " model must have exactly 1 input");
    const auto& in = inputs.front();
    const auto& s = in.shape();
    if (s.size() != 4 || s[3] != 3) {
        throw std::runtime_error(what + " model input is not NHWC x3 (got rank " +
                                 std::to_string(s.size()) + ")");
    }
    if (in.type() != dxrt::DataType::UINT8) {
        throw std::runtime_error(what + " model input must be UINT8 (got " +
                                 dxrt::DataTypeToString(in.type()) + ")");
    }
    h = static_cast<int>(s[1]);
    w = static_cast<int>(s[2]);
    bytes = static_cast<std::size_t>(ie.GetInputSize());
    if (bytes != static_cast<std::size_t>(w) * h * 3) {
        throw std::runtime_error(what + " model input size " + std::to_string(bytes) +
                                 " != H*W*3 (row-padded inputs are not supported)");
    }
}

void FaceEngine::copy_input(const cv::Mat& pre, std::vector<uint8_t>& buf, const char* what) {
    const std::size_t n = pre.total() * pre.elemSize();
    if (pre.depth() != CV_8U || n != buf.size()) {
        throw std::runtime_error(std::string(what) + " preprocess produced " + std::to_string(n) +
                                 " bytes, model expects " + std::to_string(buf.size()));
    }
    if (pre.isContinuous()) {
        std::memcpy(buf.data(), pre.data, n);
    } else {
        const std::size_t row = static_cast<std::size_t>(pre.cols) * pre.elemSize();
        for (int y = 0; y < pre.rows; ++y) std::memcpy(buf.data() + y * row, pre.ptr(y), row);
    }
}

std::vector<Face> FaceEngine::detect(const cv::Mat& bgr) { return detect(bgr, cfg_.det_score_threshold); }

std::vector<Face> FaceEngine::detect(const cv::Mat& bgr, float min_score) {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::invalid_argument("detect() expects a non-empty 8-bit BGR image");
    }
    dxapp::PreprocessContext ctx;
    cv::Mat pre;
    det_pre_->process(bgr, pre, ctx);
    std::vector<uint8_t> input(det_in_bytes_);
    copy_input(pre, input, "SCRFD");

    dxrt::InferenceEngine& ie = *det_pool_[det_rr_++ % det_pool_.size()];
    const double t0 = mono_ms();
    const int job = ie.RunAsync(input.data());
    dxrt::TensorPtrs outs = ie.Wait(job);  // owned copy of the outputs
    const double t1 = mono_ms();

    const auto dets = det_post_->process(outs, ctx);
    std::vector<Face> faces;
    faces.reserve(dets.size());
    for (const auto& d : dets) {
        if (d.box.size() < 4 || d.landmarks.size() < 5 || d.confidence < min_score) continue;
        Face f;
        f.bbox = {d.box[0], d.box[1], d.box[2], d.box[3]};
        f.det_score = d.confidence;
        for (int k = 0; k < 5; ++k) f.kps[k] = cv::Point2f(d.landmarks[k].x, d.landmarks[k].y);
        faces.push_back(std::move(f));
    }
    {
        std::lock_guard<std::mutex> lk(stats_mu_);
        stats_.det_calls += 1;
        stats_.det_npu_ms += t1 - t0;
    }
    return faces;
}

void FaceEngine::embed(const cv::Mat& bgr, std::vector<Face>& faces) {
    if (faces.empty()) return;
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::invalid_argument("embed() expects a non-empty 8-bit BGR image");
    }
    const std::size_t n = faces.size();
    // input buffers must outlive their NPU jobs: allocated once, waited on below
    std::vector<std::vector<uint8_t>> inputs(n, std::vector<uint8_t>(rec_in_bytes_));
    dxapp::PreprocessContext ctx;
    for (std::size_t i = 0; i < n; ++i) {
        // align on the full-resolution source image (not the detector's 640 letterbox)
        const cv::Mat crop = align_face(bgr, faces[i].kps, rec_w_, rec_h_);
        cv::Mat pre;
        rec_pre_->process(crop, pre, ctx);
        copy_input(pre, inputs[i], "ArcFace");
    }

    // Submit every face before waiting on any: DXRT runs them concurrently on the cores
    // (NPU_ALL engine), or we spread them round-robin over the per-core pool.
    struct Job { dxrt::InferenceEngine* ie; int id; };
    std::vector<Job> jobs;
    jobs.reserve(n);
    const std::uint32_t base = rec_rr_.fetch_add(static_cast<std::uint32_t>(n));
    const double t0 = mono_ms();
    std::vector<dxrt::TensorPtrs> outs(n);
    try {
        for (std::size_t i = 0; i < n; ++i) {
            dxrt::InferenceEngine* ie = rec_pool_[(base + i) % rec_pool_.size()].get();
            jobs.push_back({ie, ie->RunAsync(inputs[i].data())});
        }
        for (std::size_t i = 0; i < n; ++i) outs[i] = jobs[i].ie->Wait(jobs[i].id);
    } catch (...) {
        // never let a submitted job outlive its input buffer
        for (std::size_t i = 0; i < jobs.size(); ++i) {
            if (outs[i].empty()) {
                try { jobs[i].ie->Wait(jobs[i].id); } catch (...) { /* already failing */ }
            }
        }
        throw;
    }
    const double t1 = mono_ms();

    for (std::size_t i = 0; i < n; ++i) {
        auto res = rec_post_->process(outs[i], ctx);
        if (res.empty()) throw std::runtime_error("ArcFace produced no embedding");
        faces[i].embedding = std::move(res.front().embedding);
    }
    {
        std::lock_guard<std::mutex> lk(stats_mu_);
        stats_.rec_calls += 1;
        stats_.rec_faces += n;
        stats_.rec_npu_ms += t1 - t0;
    }
}

std::vector<Face> FaceEngine::get(const cv::Mat& bgr) {
    std::vector<Face> faces = detect(bgr);
    embed(bgr, faces);
    return faces;
}

EngineStats FaceEngine::stats() const {
    std::lock_guard<std::mutex> lk(stats_mu_);
    return stats_;
}

}  // namespace bup
