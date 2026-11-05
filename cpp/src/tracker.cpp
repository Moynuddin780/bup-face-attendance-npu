#include "bup/tracker.hpp"

namespace bup {

AttendanceTracker::AttendanceTracker(const std::vector<StudentMeta>& meta, double min_time,
                                     double session_start)
    : min_time_(min_time), session_start_(session_start) {
    for (const auto& m : meta) {
        Student s;
        s.name = m.name;
        // Python dict semantics: a repeated id keeps its first position, last value wins
        if (students_.find(m.student_id) == students_.end()) order_.push_back(m.student_id);
        students_[m.student_id] = std::move(s);
    }
}

void AttendanceTracker::update(const std::string& student_id, const std::string& camera_id,
                               double now) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = students_.find(student_id);
    if (it == students_.end()) return;
    Student& s = it->second;
    auto ls = s.last_seen.find(camera_id);
    const double last = ls == s.last_seen.end() ? 0.0 : ls->second;
    if (last > 0) {
        const double diff = now - last;
        if (diff < 2.0) s.total_time += diff;
    }
    s.last_seen[camera_id] = now;
    if (!s.has_first_seen) {
        s.has_first_seen = true;
        s.first_seen = now;
    }
    s.detection_count += 1;
    ++version_;
}

void AttendanceTracker::set_manual(const std::string& student_id, bool has_status,
                                   const std::string& status) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = students_.find(student_id);
    if (it == students_.end()) return;
    it->second.has_override = has_status;
    it->second.manual_override = has_status ? status : std::string();
    ++version_;
}

std::vector<TrackerRow> AttendanceTracker::rows() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<TrackerRow> out;
    out.reserve(order_.size());
    for (const auto& sid : order_) {
        const Student& s = students_.at(sid);
        TrackerRow r;
        r.student_id = sid;
        r.name = s.name;
        r.total_time = s.total_time;
        r.present = override_truthy(s) ? (s.manual_override == "present") : (s.total_time >= min_time_);
        r.detection_count = s.detection_count;
        r.has_override = s.has_override;
        r.manual_override = s.manual_override;
        out.push_back(std::move(r));
    }
    return out;
}

LiveCounts AttendanceTracker::live() const {
    std::lock_guard<std::mutex> lk(mu_);
    LiveCounts c;
    c.total = order_.size();
    c.session_start = session_start_;
    for (const auto& sid : order_) {
        const Student& s = students_.at(sid);
        const bool forced = s.has_override && s.manual_override == "present";
        const bool timed = !s.has_override && s.total_time >= min_time_;
        if (forced || timed) ++c.present;
    }
    return c;
}

double AttendanceTracker::total_time(const std::string& student_id) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = students_.find(student_id);
    return it == students_.end() ? 0.0 : it->second.total_time;
}

std::vector<StudentState> AttendanceTracker::export_state() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<StudentState> out;
    out.reserve(order_.size());
    for (const auto& sid : order_) {
        const Student& s = students_.at(sid);
        StudentState st;
        st.student_id = sid;
        st.total_time = s.total_time;
        st.detection_count = s.detection_count;
        st.has_first_seen = s.has_first_seen;
        st.first_seen = s.first_seen;
        st.has_override = s.has_override;
        st.manual_override = s.manual_override;
        out.push_back(std::move(st));
    }
    return out;
}

void AttendanceTracker::restore(const StudentState& st) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = students_.find(st.student_id);
    if (it == students_.end()) return;
    Student& s = it->second;
    s.total_time = st.total_time;
    s.detection_count = st.detection_count;
    s.has_first_seen = st.has_first_seen;
    s.first_seen = st.first_seen;
    s.has_override = st.has_override;
    s.manual_override = st.has_override ? st.manual_override : std::string();
    s.last_seen.clear();
    ++version_;
}

std::uint64_t AttendanceTracker::version() const {
    std::lock_guard<std::mutex> lk(mu_);
    return version_;
}

}  // namespace bup
