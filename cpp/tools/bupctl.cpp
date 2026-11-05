// bupctl — operator CLI for the BUP attendance engine (works inside the Docker image too).
//
//   bupctl detect <image> [--out annotated.jpg] [--crops DIR]   faces + embeddings on the NPU
//   bupctl bench  <image> [--iters N]                           detect+embed latency
//   bupctl stream <rtsp-url> [--profile sub|main] [--codec H264|H265] [--transport tcp] [--seconds S]
//                                                               camera connectivity / decode fps
// Models: --det/--rec, else $DET_MODEL_PATH/$REC_MODEL_PATH, else ./models/...

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "bup/capture.hpp"
#include "bup/common.hpp"
#include "bup/face_engine.hpp"
#include "bup/stream_url.hpp"

using namespace bup;

namespace {

int usage() {
    std::cerr << "usage:\n"
                 "  bupctl detect <image> [--out annotated.jpg] [--crops DIR] [--det M] [--rec M] [--bound auto|all|0|1|2]\n"
                 "  bupctl bench  <image> [--iters N] [--det M] [--rec M]\n"
                 "  bupctl stream <rtsp-url> [--profile sub|main] [--transport tcp|udp|udp+tcp] [--aspect auto|off|16:9] [--codec H264|H265] [--seconds S]\n";
    return 2;
}

std::map<std::string, std::string> parse_opts(int argc, char** argv, int from) {
    std::map<std::string, std::string> o;
    for (int i = from; i + 1 < argc; i += 2) o[argv[i]] = argv[i + 1];
    return o;
}

std::string opt(const std::map<std::string, std::string>& o, const std::string& k, const char* env,
                const std::string& def) {
    auto it = o.find(k);
    if (it != o.end()) return it->second;
    const char* e = env ? std::getenv(env) : nullptr;
    return e && *e ? std::string(e) : def;
}

FaceEngineConfig engine_cfg(const std::map<std::string, std::string>& o) {
    FaceEngineConfig c;
    c.det_model_path = opt(o, "--det", "DET_MODEL_PATH", "models/scrfd-10g_640x640.dxnn");
    c.rec_model_path = opt(o, "--rec", "REC_MODEL_PATH", "models/arcface_iresnet100_112x112_ms1m.dxnn");
    c.npu_bound = opt(o, "--bound", "NPU_BOUND", "auto");
    return c;
}

int cmd_detect(const std::string& path, const std::map<std::string, std::string>& o) {
    const cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) { std::cerr << "cannot read " << path << "\n"; return 1; }
    FaceEngine engine(engine_cfg(o));
    const auto faces = engine.get(img);
    std::printf("%s: %dx%d, %zu face(s)\n", path.c_str(), img.cols, img.rows, faces.size());
    std::vector<dxapp::FaceDetectionResult> vis;
    for (std::size_t i = 0; i < faces.size(); ++i) {
        const Face& f = faces[i];
        double norm = 0;
        for (float v : f.embedding) norm += static_cast<double>(v) * v;
        std::printf("  [%zu] score=%.3f box=(%.0f,%.0f,%.0f,%.0f) emb_dim=%zu |emb|=%.4f\n", i, f.det_score,
                    f.bbox[0], f.bbox[1], f.bbox[2], f.bbox[3], f.embedding.size(), std::sqrt(norm));
        dxapp::FaceDetectionResult r;
        r.box = {f.bbox[0], f.bbox[1], f.bbox[2], f.bbox[3]};
        r.confidence = f.det_score;
        for (const auto& p : f.kps) r.landmarks.emplace_back(p.x, p.y, 1.0f);
        vis.push_back(std::move(r));
    }
    ScrfdFactory det_factory(0.5f, 0.4f);
    ArcFaceFactory rec_factory;
    if (o.count("--out")) {
        auto viz = det_factory.createVisualizer();
        cv::imwrite(o.at("--out"), viz->draw(img, vis, dxapp::PreprocessContext()));
        std::printf("  annotated -> %s\n", o.at("--out").c_str());
    }
    if (o.count("--crops")) {
        auto viz = rec_factory.createVisualizer();
        for (std::size_t i = 0; i < faces.size(); ++i) {
            dxapp::EmbeddingResult er;
            er.embedding = faces[i].embedding;
            er.dimension = static_cast<int>(faces[i].embedding.size());
            const cv::Mat crop = align_face(img, faces[i].kps, engine.rec_input_width(), engine.rec_input_height());
            const std::string out = o.at("--crops") + "/crop_" + std::to_string(i) + ".jpg";
            cv::imwrite(out, viz->draw(crop, {er}, dxapp::PreprocessContext()));
        }
        std::printf("  %zu aligned crop(s) -> %s/\n", faces.size(), o.at("--crops").c_str());
    }
    return faces.empty() ? 3 : 0;
}

int cmd_bench(const std::string& path, const std::map<std::string, std::string>& o) {
    const cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) { std::cerr << "cannot read " << path << "\n"; return 1; }
    const int iters = std::atoi(opt(o, "--iters", nullptr, "50").c_str());
    FaceEngine engine(engine_cfg(o));
    engine.get(img);  // warm-up
    double det_ms = 0, rec_ms = 0;
    std::size_t faces_n = 0;
    for (int i = 0; i < iters; ++i) {
        const double t0 = mono_ms();
        auto faces = engine.detect(img);
        const double t1 = mono_ms();
        engine.embed(img, faces);
        const double t2 = mono_ms();
        det_ms += t1 - t0;
        rec_ms += t2 - t1;
        faces_n = faces.size();
    }
    std::printf("%d iters, %zu face(s)/frame: detect %.2f ms, embed(batch) %.2f ms, total %.2f ms (%.1f fps)\n",
                iters, faces_n, det_ms / iters, rec_ms / iters, (det_ms + rec_ms) / iters,
                1000.0 * iters / (det_ms + rec_ms));
    return 0;
}

int cmd_stream(const std::string& url, const std::map<std::string, std::string>& o) {
    CaptureConfig c;
    c.cam_id = "cli";
    c.url = url;
    c.profile = opt(o, "--profile", "STREAM_PROFILE", "sub");
    c.transport = opt(o, "--transport", "RTSP_TRANSPORT", "tcp");
    c.aspect = opt(o, "--aspect", "SUBSTREAM_ASPECT", "auto");
    c.codec_hint = opt(o, "--codec", nullptr, "");
    const double seconds = std::atof(opt(o, "--seconds", nullptr, "10").c_str());
    std::printf("GStreamer available: %s\n", gstreamer_available() ? "yes" : "no");
    RtspCapture cap(c);
    std::printf("resolved url: %s\n", mask_credentials(cap.resolved_url()).c_str());
    cap.start();
    const double end = mono_ms() + seconds * 1000.0;
    while (mono_ms() < end) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        const CaptureInfo i = cap.info();
        std::printf("  backend=%s decoded=%dx%d processed=%dx%d decode_fps=%.1f frames=%llu fails=%d reconnects=%d\n",
                    i.backend.c_str(), i.width, i.height, i.out_width, i.out_height, i.decode_fps,
                    static_cast<unsigned long long>(i.frames), i.fail_count, i.reconnects);
    }
    cap.stop();
    return cap.info().frames > 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) return usage();
    const std::string cmd = argv[1], arg = argv[2];
    const auto o = parse_opts(argc, argv, 3);
    try {
        if (cmd == "detect") return cmd_detect(arg, o);
        if (cmd == "bench") return cmd_bench(arg, o);
        if (cmd == "stream") return cmd_stream(arg, o);
    } catch (const std::exception& e) {
        std::cerr << "bupctl: " << e.what() << "\n";
        return 1;
    }
    return usage();
}
