#ifndef BUP_STREAM_URL_HPP
#define BUP_STREAM_URL_HPP

#include <string>

namespace bup {

/**
 * Map a camera URL (as stored in bup_db.json, i.e. the main stream) to the
 * requested stream profile.
 *
 *  profile "main" -> URL unchanged
 *  profile "sub"  -> Dahua   .../cam/realmonitor?...subtype=0      -> subtype=1
 *                             (subtype missing -> "&subtype=1" appended)
 *                    Hikvision .../Streaming/Channels/<ch>01        -> <ch>02
 *                              .../h264/ch<N>/main/av_stream        -> /sub/
 *                    anything else                                  -> unchanged
 *
 * @param rewritten     optional out-flag: true when the URL was changed
 * @param vendor_known  optional out-flag: true when a Dahua/Hikvision rule recognised
 *                      the URL (even if it already pointed at a sub-stream)
 * @throws std::invalid_argument for a profile other than "main"/"sub"
 */
std::string resolve_stream_url(const std::string& url, const std::string& profile,
                               bool* rewritten = nullptr, bool* vendor_known = nullptr);

}  // namespace bup

#endif  // BUP_STREAM_URL_HPP
