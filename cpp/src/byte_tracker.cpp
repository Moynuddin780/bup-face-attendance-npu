#include "bup/byte_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/core.hpp>

namespace bup {

namespace {

using Vec8 = cv::Matx<double, 8, 1>;
using Mat8 = cv::Matx<double, 8, 8>;
using Vec4 = cv::Matx<double, 4, 1>;
using Mat4 = cv::Matx<double, 4, 4>;
using Mat84 = cv::Matx<double, 8, 4>;

// ByteTrack / DeepSORT kalman_filter.py: noise relative to the box height
constexpr double kStdPos = 1.0 / 20.0;
constexpr double kStdVel = 1.0 / 160.0;

Vec4 to_xyah(const std::array<float, 4>& b) {
    const double w = std::max(1.0f, b[2] - b[0]);
    const double h = std::max(1.0f, b[3] - b[1]);
    return Vec4(b[0] + w / 2.0, b[1] + h / 2.0, w / h, h);
}

float iou(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    const float iw = std::min(a[2], b[2]) - std::max(a[0], b[0]);
    const float ih = std::min(a[3], b[3]) - std::max(a[1], b[1]);
    if (iw <= 0.0f || ih <= 0.0f) return 0.0f;
    const float inter = iw * ih;
    const float uni = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

}  // namespace

struct ByteTracker::Track {
    enum class State { Tracked, Lost, Removed };
    int id{0};
    Vec8 mean;
    Mat8 cov;
    State state{State::Tracked};
    bool activated{false};
    float score{0.0f};
    int det{-1};
    int hits{0};
    long start_frame{0};
    double last_seen{0.0};

    std::array<float, 4> tlbr() const {
        const double h = mean(3), w = mean(2) * h;
        return {static_cast<float>(mean(0) - w / 2.0), static_cast<float>(mean(1) - h / 2.0),
                static_cast<float>(mean(0) + w / 2.0), static_cast<float>(mean(1) + h / 2.0)};
    }

    void initiate(const std::array<float, 4>& box) {
        const Vec4 m = to_xyah(box);
        mean = Vec8::zeros();
        for (int i = 0; i < 4; ++i) mean(i) = m(i);
        const double h = m(3);
        const double std[8] = {2 * kStdPos * h, 2 * kStdPos * h, 1e-2, 2 * kStdPos * h,
                               10 * kStdVel * h, 10 * kStdVel * h, 1e-5, 10 * kStdVel * h};
        cov = Mat8::zeros();
        for (int i = 0; i < 8; ++i) cov(i, i) = std[i] * std[i];
    }

    /// constant velocity over `steps` time units; the process noise grows with the elapsed time
    void predict(double steps) {
        if (state != State::Tracked) mean(7) = 0.0;   // ByteTrack: a lost box stops changing size
        const double h = mean(3);
        const double std[8] = {kStdPos * h, kStdPos * h, 1e-2, kStdPos * h,
                               kStdVel * h, kStdVel * h, 1e-5, kStdVel * h};
        Mat8 F = Mat8::eye();
        for (int i = 0; i < 4; ++i) F(i, i + 4) = steps;
        Mat8 Q = Mat8::zeros();
        for (int i = 0; i < 8; ++i) Q(i, i) = std[i] * std[i] * steps;
        mean = F * mean;
        cov = F * cov * F.t() + Q;
        if (mean(3) < 1.0) mean(3) = 1.0;
    }

    void correct(const std::array<float, 4>& box) {
        const double h = mean(3);
        const double std[4] = {kStdPos * h, kStdPos * h, 1e-1, kStdPos * h};
        Mat4 S;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) S(r, c) = cov(r, c) + (r == c ? std[r] * std[r] : 0.0);
        Mat84 PHt;
        for (int r = 0; r < 8; ++r)
            for (int c = 0; c < 4; ++c) PHt(r, c) = cov(r, c);
        const Mat84 K = PHt * S.inv(cv::DECOMP_CHOLESKY);
        const Vec4 z = to_xyah(box);
        Vec4 innov;
        for (int i = 0; i < 4; ++i) innov(i) = z(i) - mean(i);
        mean += K * innov;
        cov -= K * S * K.t();
    }
};

ByteTracker::ByteTracker(TrackerParams p) : p_(p) {}
ByteTracker::~ByteTracker() = default;
ByteTracker::ByteTracker(ByteTracker&&) noexcept = default;
ByteTracker& ByteTracker::operator=(ByteTracker&&) noexcept = default;

void ByteTracker::reset() {
    tracks_.clear();
    next_id_ = 1;
    frame_ = 0;
    last_t_ = -1.0;
}

std::vector<TrackOut> ByteTracker::update(const std::vector<TrackDet>& dets, double t) {
    using State = Track::State;
    ++frame_;
    // Kalman step in units of 1/step_hz s: the NPU rate varies, the motion model must not
    const double steps = last_t_ < 0.0 ? 1.0 : std::min(50.0, std::max(0.05, (t - last_t_) * p_.step_hz));
    last_t_ = t;

    std::vector<int> high, low;
    for (int i = 0; i < static_cast<int>(dets.size()); ++i) {
        if (dets[i].score >= p_.high_thresh) high.push_back(i);
        else if (dets[i].score >= p_.low_thresh) low.push_back(i);
    }

    std::vector<Track*> pool, unconfirmed;
    for (auto& tr : tracks_) {
        tr->det = -1;
        tr->predict(steps);
        if (tr->state == State::Tracked && !tr->activated) unconfirmed.push_back(tr.get());
        else pool.push_back(tr.get());
    }

    auto apply = [&](Track* tr, int d) {
        tr->correct(dets[d].tlbr);
        tr->state = State::Tracked;
        tr->activated = true;   // matched again: confirmed (a lost track is re-activated)
        tr->score = dets[d].score;
        tr->det = d;
        tr->hits += 1;
        tr->last_seen = t;
    };
    auto cost_matrix = [&](const std::vector<Track*>& trs, const std::vector<int>& ds, bool fuse) {
        std::vector<std::vector<float>> c(trs.size(), std::vector<float>(ds.size(), 1.0f));
        for (std::size_t r = 0; r < trs.size(); ++r) {
            const auto box = trs[r]->tlbr();
            for (std::size_t k = 0; k < ds.size(); ++k) {
                const float sim = iou(box, dets[ds[k]].tlbr);
                c[r][k] = 1.0f - (fuse ? sim * dets[ds[k]].score : sim);
            }
        }
        return c;
    };

    // 1) confident detections <-> tracked + lost tracks
    std::vector<bool> high_used(high.size(), false);
    std::vector<Track*> rest;
    {
        const auto m = linear_assignment(cost_matrix(pool, high, true), p_.match_thresh);
        for (std::size_t r = 0; r < pool.size(); ++r) {
            if (m[r] >= 0) {
                apply(pool[r], high[m[r]]);
                high_used[m[r]] = true;
            } else {
                rest.push_back(pool[r]);
            }
        }
    }
    // 2) weak detections <-> the tracked ones still unmatched
    {
        std::vector<Track*> still;
        for (Track* tr : rest)
            if (tr->state == State::Tracked) still.push_back(tr);
        const auto m = linear_assignment(cost_matrix(still, low, false), p_.second_match_thresh);
        for (std::size_t r = 0; r < still.size(); ++r) {
            if (m[r] >= 0) apply(still[r], low[m[r]]);
            else still[r]->state = State::Lost;
        }
    }
    // 3) new (unconfirmed) tracks <-> the confident detections left
    std::vector<int> left;
    for (std::size_t k = 0; k < high.size(); ++k)
        if (!high_used[k]) left.push_back(high[k]);
    std::vector<bool> left_used(left.size(), false);
    {
        const auto m = linear_assignment(cost_matrix(unconfirmed, left, true), p_.unconfirmed_match_thresh);
        for (std::size_t r = 0; r < unconfirmed.size(); ++r) {
            if (m[r] >= 0) {
                apply(unconfirmed[r], left[m[r]]);
                left_used[m[r]] = true;
            } else {
                unconfirmed[r]->state = State::Removed;   // a one-frame blip
            }
        }
    }
    // 4) start tracks for the confident detections nobody claimed
    for (std::size_t k = 0; k < left.size(); ++k) {
        const int d = left[k];
        if (left_used[k] || dets[d].score < p_.new_track_thresh) continue;
        auto tr = std::unique_ptr<Track>(new Track());
        tr->id = next_id_++;
        tr->initiate(dets[d].tlbr);
        tr->activated = frame_ == 1;
        tr->score = dets[d].score;
        tr->det = d;
        tr->hits = 1;
        tr->start_frame = frame_;
        tr->last_seen = t;
        tracks_.push_back(std::move(tr));
    }
    // 5) forget tracks lost for too long; drop a lost/tracked pair that became the same box
    for (auto& tr : tracks_)
        if (tr->state == State::Lost && t - tr->last_seen > p_.lost_keep_s) tr->state = State::Removed;
    for (auto& a : tracks_) {
        if (a->state != State::Tracked) continue;
        for (auto& b : tracks_) {
            if (b->state != State::Lost) continue;
            if (iou(a->tlbr(), b->tlbr()) <= 0.85f) continue;
            // keep the longer-lived track (ByteTrack remove_duplicate_stracks)
            if (frame_ - a->start_frame > frame_ - b->start_frame) b->state = State::Removed;
            else a->state = State::Removed;
            if (a->state == State::Removed) break;
        }
    }
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [](const std::unique_ptr<Track>& tr) { return tr->state == State::Removed; }),
                  tracks_.end());

    std::vector<TrackOut> out;
    out.reserve(tracks_.size());
    for (const auto& tr : tracks_) {
        TrackOut o;
        o.id = tr->id;
        o.tlbr = tr->tlbr();
        o.vx = static_cast<float>(tr->mean(4) * p_.step_hz);
        o.vy = static_cast<float>(tr->mean(5) * p_.step_hz);
        o.score = tr->det >= 0 ? tr->score : 0.0f;
        o.det = tr->det;
        o.activated = tr->activated;
        o.lost = tr->state == State::Lost;
        o.last_seen = tr->last_seen;
        o.hits = tr->hits;
        out.push_back(o);
    }
    return out;
}

std::vector<int> linear_assignment(const std::vector<std::vector<float>>& cost, float limit) {
    const int n = static_cast<int>(cost.size());
    const int m = n ? static_cast<int>(cost[0].size()) : 0;
    std::vector<int> row_match(n, -1);
    if (n == 0 || m == 0) return row_match;

    // Square (n+m) problem: every row may instead take its own "unmatched" column, and every
    // column its own "unmatched" row, at limit/2 each — so a real pair is only chosen when it
    // costs less than leaving both unmatched (lapjv extend_cost with cost_limit).
    const int N = n + m;
    const double big = 1e6, half = limit / 2.0;
    std::vector<std::vector<double>> a(N + 1, std::vector<double>(N + 1, big));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j)
            if (cost[i][j] <= limit) a[i + 1][j + 1] = cost[i][j];
        a[i + 1][m + i + 1] = half;
    }
    for (int j = 0; j < m; ++j) {
        a[n + j + 1][j + 1] = half;
        for (int i = 0; i < n; ++i) a[n + j + 1][m + i + 1] = 0.0;
    }

    // Hungarian algorithm with potentials, O(N^3)
    std::vector<double> u(N + 1, 0.0), v(N + 1, 0.0);
    std::vector<int> p(N + 1, 0), way(N + 1, 0);
    for (int i = 1; i <= N; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(N + 1, std::numeric_limits<double>::infinity());
        std::vector<char> used(N + 1, 0);
        do {
            used[j0] = 1;
            const int i0 = p[j0];
            double delta = std::numeric_limits<double>::infinity();
            int j1 = 0;
            for (int j = 1; j <= N; ++j) {
                if (used[j]) continue;
                const double cur = a[i0][j] - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= N; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }
    for (int j = 1; j <= m; ++j) {
        const int i = p[j] - 1;
        if (i >= 0 && i < n && cost[i][j - 1] <= limit) row_match[i] = j - 1;
    }
    return row_match;
}

}  // namespace bup
