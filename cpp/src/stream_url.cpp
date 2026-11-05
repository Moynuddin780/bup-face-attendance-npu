#include "bup/stream_url.hpp"

#include <regex>
#include <stdexcept>

namespace bup {

namespace {

bool rewrite_dahua(const std::string& url, std::string& out) {
    if (url.find("/cam/realmonitor") == std::string::npos) return false;
    static const std::regex subtype(R"(([?&]subtype=)(\d+))");
    std::smatch m;
    if (std::regex_search(url, m, subtype)) {
        if (m[2].str() != "0") {  // already a sub/third stream: respect it
            out = url;
            return true;
        }
        out = m.prefix().str() + m[1].str() + "1" + m.suffix().str();
        return true;
    }
    out = url + (url.find('?') == std::string::npos ? "?" : "&") + "subtype=1";
    return true;
}

bool rewrite_hikvision(const std::string& url, std::string& out) {
    // ISAPI: /Streaming/Channels/101 (channel 1 main) -> 102 (channel 1 sub)
    static const std::regex isapi(R"((/[Ss]treaming/[Cc]hannels/)(\d+?)01(?=[/?#]|$))");
    std::smatch m;
    if (std::regex_search(url, m, isapi)) {
        out = m.prefix().str() + m[1].str() + m[2].str() + "02" + m.suffix().str();
        return true;
    }
    // Legacy: /h264/ch1/main/av_stream -> /h264/ch1/sub/av_stream
    static const std::regex legacy(R"((/ch\d+/)main(/))", std::regex::icase);
    if (std::regex_search(url, m, legacy)) {
        out = m.prefix().str() + m[1].str() + "sub" + m[2].str() + m.suffix().str();
        return true;
    }
    // Recognised Hikvision URL that already names a non-main stream (e.g. .../102)
    static const std::regex isapi_any(R"(/[Ss]treaming/[Cc]hannels/\d+)");
    static const std::regex legacy_sub(R"(/ch\d+/sub/)", std::regex::icase);
    if (std::regex_search(url, isapi_any) || std::regex_search(url, legacy_sub)) {
        out = url;
        return true;
    }
    return false;
}

}  // namespace

std::string resolve_stream_url(const std::string& url, const std::string& profile,
                               bool* rewritten, bool* vendor_known) {
    if (profile != "main" && profile != "sub") {
        throw std::invalid_argument("stream profile must be 'main' or 'sub', got '" + profile + "'");
    }
    if (rewritten) *rewritten = false;
    if (vendor_known) *vendor_known = false;
    if (profile == "main") return url;

    std::string out;
    if (rewrite_dahua(url, out) || rewrite_hikvision(url, out)) {
        if (rewritten) *rewritten = (out != url);
        if (vendor_known) *vendor_known = true;
        return out;
    }
    return url;
}

}  // namespace bup
