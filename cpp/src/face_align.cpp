#include "bup/face_align.hpp"

#include <opencv2/imgproc.hpp>

namespace bup {

const Landmarks5& arcface_template_112() {
    static const Landmarks5 ref = {{
        {38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f},
        {41.5493f, 92.3655f}, {70.7299f, 92.2041f}}};
    return ref;
}

cv::Mat umeyama_similarity(const Landmarks5& src, const Landmarks5& dst) {
    const int n = static_cast<int>(src.size());
    cv::Matx21d mu_s(0, 0), mu_d(0, 0);
    for (int i = 0; i < n; ++i) {
        mu_s += cv::Matx21d(src[i].x, src[i].y);
        mu_d += cv::Matx21d(dst[i].x, dst[i].y);
    }
    mu_s *= 1.0 / n;
    mu_d *= 1.0 / n;

    // covariance A = (1/n) * sum(dst_i' * src_i'^T) and source variance
    cv::Matx22d A = cv::Matx22d::zeros();
    double var_s = 0.0;
    for (int i = 0; i < n; ++i) {
        const cv::Matx21d s(src[i].x - mu_s(0), src[i].y - mu_s(1));
        const cv::Matx21d d(dst[i].x - mu_d(0), dst[i].y - mu_d(1));
        A += d * s.t();
        var_s += s(0) * s(0) + s(1) * s(1);
    }
    A *= 1.0 / n;
    var_s /= n;

    cv::Mat w, u, vt;
    cv::SVD::compute(cv::Mat(A), w, u, vt);
    // reflection guard: force det(R) = +1
    cv::Matx22d D = cv::Matx22d::eye();
    if (cv::determinant(cv::Mat(A)) < 0) D(1, 1) = -1.0;
    const cv::Matx22d U(u), Vt(vt);
    const cv::Matx22d R = U * D * Vt;
    const double scale = var_s > 0 ? (w.at<double>(0) * D(0, 0) + w.at<double>(1) * D(1, 1)) / var_s : 1.0;
    const cv::Matx21d t = mu_d - scale * (R * mu_s);

    cv::Mat M(2, 3, CV_64F);
    M.at<double>(0, 0) = scale * R(0, 0);
    M.at<double>(0, 1) = scale * R(0, 1);
    M.at<double>(0, 2) = t(0);
    M.at<double>(1, 0) = scale * R(1, 0);
    M.at<double>(1, 1) = scale * R(1, 1);
    M.at<double>(1, 2) = t(1);
    return M;
}

cv::Mat align_face(const cv::Mat& bgr, const Landmarks5& kps, int out_w, int out_h) {
    Landmarks5 dst = arcface_template_112();
    if (out_w != 112 || out_h != 112) {  // InsightFace: scale the template to the crop size
        const float sx = out_w / 112.0f, sy = out_h / 112.0f;
        for (auto& p : dst) { p.x *= sx; p.y *= sy; }
    }
    const cv::Mat M = umeyama_similarity(kps, dst);
    cv::Mat out;
    cv::warpAffine(bgr, out, M, cv::Size(out_w, out_h), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                   cv::Scalar(0, 0, 0));
    return out;
}

}  // namespace bup
