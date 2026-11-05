#ifndef BUP_TRACKER_HPP
#define BUP_TRACKER_HPP

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace bup {

struct StudentMeta {
    std::string student_id;
    std::string name;
};

/// One row of AttendanceTracker.summary(), before Python-side rounding.
struct TrackerRow {
    std::string student_id;
    std::string name;
    double total_time{0.0};
    bool present{false};
    long long detection_count{0};
    bool has_override{false};       // false -> manual_override is None
    std::string manual_override;
};

/// What a class has measured for one student: a crash-safe snapshot (export_state / restore).
struct StudentState {
    std::string student_id;
    double total_time{0.0};
    long long detection_count{0};
    bool has_first_seen{false};
    double first_seen{0.0};
    bool has_override{false};
    std::string manual_override;
};

struct LiveCounts {
    std::size_t total{0};
    std::size_t present{0};
    double session_start{0.0};
};

/**
 * 1:1 port of the original Python AttendanceTracker (app.py).
 *
 *  - update(): a sighting adds (now - last_seen[camera]) to total_time when the gap
 *    is < 2.0 s (per camera), counts the detection and records first_seen.
 *  - summary: manual override (truthy string) wins, else total_time >= min_time.
 *  - live stats: "present" override, or no override (None) and total_time >= min_time.
 *    (The two rules differ for an empty-string override exactly as in the original.)
 *
 * Rows keep the order of the meta list (Python dict insertion order). All methods
 * are thread-safe: camera workers update while Flask threads read.
 */
class AttendanceTracker {
public:
    AttendanceTracker(const std::vector<StudentMeta>& meta, double min_time, double session_start);

    void update(const std::string& student_id, const std::string& camera_id, double now);
    void set_manual(const std::string& student_id, bool has_status, const std::string& status);

    std::vector<TrackerRow> rows() const;
    LiveCounts live() const;
    double total_time(const std::string& student_id) const;  ///< 0 for unknown ids
    std::uint64_t version() const;                            ///< bumps on every change
    double min_time() const { return min_time_; }
    double session_start() const { return session_start_; }

    std::vector<StudentState> export_state() const;           ///< in meta order
    /// Put a snapshot back (a class resumed after a restart). Unknown ids are ignored; the next
    /// sighting on each camera starts a new gap, so the restart's downtime is never counted.
    void restore(const StudentState& st);

private:
    struct Student {
        std::string name;
        double total_time{0.0};
        std::unordered_map<std::string, double> last_seen;
        bool has_first_seen{false};
        double first_seen{0.0};
        long long detection_count{0};
        bool has_override{false};
        std::string manual_override;
    };

    static bool override_truthy(const Student& s) { return s.has_override && !s.manual_override.empty(); }

    mutable std::mutex mu_;
    double min_time_;
    double session_start_;
    std::vector<std::string> order_;
    std::unordered_map<std::string, Student> students_;
    std::uint64_t version_{0};
};

}  // namespace bup

#endif  // BUP_TRACKER_HPP
