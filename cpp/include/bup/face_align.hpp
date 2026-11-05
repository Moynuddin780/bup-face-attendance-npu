#ifndef BUP_FACE_ALIGN_HPP
#define BUP_FACE_ALIGN_HPP

#include <array>

#include <opencv2/core.hpp>

namespace bup {

using Landmarks5 = std::array<cv::Point2f, 5>;  ///< left eye, right eye, nose, mouth left, mouth right

/// ArcFace canonical landmark positions in a 112x112 crop (InsightFace `arcface_dst`,
/// identical to ARC_FACE_SRC in the original app.py).
const Landmarks5& arcface_template_112();

/**
 * Least-squares similarity transform (rotation + uniform scale + translation, no
 * reflection) mapping src -> dst — Umeyama 1991, the estimator InsightFace's
 * norm_crop uses (skimage SimilarityTransform). Returns a 2x3 CV_64F matrix.
 */
cv::Mat umeyama_similarity(const Landmarks5& src, const Landmarks5& dst);

/// Warp the face so its landmarks land on the ArcFace template; out_w x out_h BGR,
/// border filled with 0 (as the original warpAffine call).
cv::Mat align_face(const cv::Mat& bgr, const Landmarks5& kps, int out_w = 112, int out_h = 112);

}  // namespace bup

#endif  // BUP_FACE_ALIGN_HPP
