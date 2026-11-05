#ifndef BUP_SYSTEM_MONITOR_HPP
#define BUP_SYSTEM_MONITOR_HPP

#include <string>
#include <vector>

namespace bup {

/// Host load shown in the super-admin sidebar, sampled once per second on a background thread.
struct SystemUsage {
    std::vector<double> npu_core_pct;   ///< busy % per NPU core (DX-M1: 3); empty when unavailable
    std::string npu_note;               ///< why npu_core_pct is empty
    double cpu_pct{-1.0};               ///< all CPUs; -1 until two /proc/stat samples exist
    std::vector<double> cpu_core_pct;   ///< per logical CPU
    double age_s{-1.0};                 ///< seconds since the last sample (-1 = none yet)
};

/// Latest sample. The sampler starts on the first call; the call itself never blocks on the
/// NPU service (a dxrtd that stops answering only makes the sample age).
SystemUsage system_usage();

}  // namespace bup

#endif  // BUP_SYSTEM_MONITOR_HPP
