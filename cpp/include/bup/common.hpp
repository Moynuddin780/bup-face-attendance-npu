#ifndef BUP_COMMON_HPP
#define BUP_COMMON_HPP

#include <atomic>
#include <chrono>
#include <string>

namespace bup {

/// Seconds since the Unix epoch — the same clock as Python's time.time(), so
/// tracker timestamps and session_start stay comparable with the original code.
inline double wall_time() {
    using namespace std::chrono;
    return duration<double>(system_clock::now().time_since_epoch()).count();
}

/// Monotonic milliseconds for latency / fps measurement.
inline double mono_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

enum class LogLevel { Info, Warn, Error };

/// One line to stderr, prefixed "[bup_engine]". Safe to call from any thread.
void log(LogLevel level, const std::string& msg);

/// Lets one message through per `interval_s` — used for per-frame error paths so a
/// broken camera cannot flood the log (the original swallowed these silently).
class LogThrottle {
public:
    explicit LogThrottle(double interval_s) : interval_ms_(interval_s * 1000.0) {}
    bool allow();

private:
    double interval_ms_;
    std::atomic<double> last_ms_{-1e18};
};

/// rtsp://user:pass@host/... -> rtsp://***@host/... (never log camera credentials).
std::string mask_credentials(const std::string& url);

}  // namespace bup

#endif  // BUP_COMMON_HPP
