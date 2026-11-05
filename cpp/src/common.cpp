#include "bup/common.hpp"

#include <cstdio>
#include <regex>

namespace bup {

void log(LogLevel level, const std::string& msg) {
    const char* tag = level == LogLevel::Error ? "ERROR" : level == LogLevel::Warn ? "WARN" : "INFO";
    // single fprintf call -> lines from different threads do not interleave
    std::fprintf(stderr, "[bup_engine] [%s] %s\n", tag, msg.c_str());
    std::fflush(stderr);
}

bool LogThrottle::allow() {
    const double now = mono_ms();
    double last = last_ms_.load();
    while (now - last >= interval_ms_) {
        if (last_ms_.compare_exchange_weak(last, now)) return true;
    }
    return false;
}

std::string mask_credentials(const std::string& url) {
    static const std::regex creds(R"(^([a-zA-Z][a-zA-Z0-9+.-]*://)[^/@]*@)");
    return std::regex_replace(url, creds, "$1***@");
}

}  // namespace bup
