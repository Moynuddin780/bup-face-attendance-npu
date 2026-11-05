#include "bup/system_monitor.hpp"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <utility>

#include "bup/common.hpp"

// NPU core usage comes from dxrtd (the numbers dxtop shows). DXRT <= 3.3 ships the IPC client
// headers; without them only the CPU part is reported.
#if __has_include(<dxrt/ipc_wrapper/ipc_client_wrapper.h>)
#include <dxrt/ipc_wrapper/ipc_client_wrapper.h>
#include <dxrt/service_util.h>
#define BUP_HAVE_DXRT_IPC 1
#endif

namespace bup {

namespace {

constexpr int kNpuCores = 3;  // DX-M1; dxrtd keeps one usage timer per core

struct CpuTimes {
    unsigned long long idle{0};
    unsigned long long total{0};
};

/// [0] = all CPUs, [1..] = one entry per logical CPU.
std::vector<CpuTimes> read_proc_stat() {
    std::vector<CpuTimes> out;
    std::ifstream f("/proc/stat");
    std::string line;
    while (std::getline(f, line) && line.compare(0, 3, "cpu") == 0) {
        std::istringstream ss(line);
        std::string name;
        ss >> name;
        unsigned long long v[8] = {0};  // user nice system idle iowait irq softirq steal
        for (auto& x : v) ss >> x;
        CpuTimes t;
        t.idle = v[3] + v[4];
        for (auto x : v) t.total += x;  // guest time is already counted in user/nice
        out.push_back(t);
    }
    return out;
}

double busy_pct(const CpuTimes& a, const CpuTimes& b) {
    if (b.total <= a.total) return 0.0;
    const double dt = static_cast<double>(b.total - a.total);
    const double di = static_cast<double>(b.idle >= a.idle ? b.idle - a.idle : 0);
    return std::min(100.0, std::max(0.0, 100.0 * (1.0 - di / dt)));
}

class Sampler {
public:
    // Leaked on purpose: the detached thread may still run while static objects are destroyed.
    static Sampler& instance() {
        static Sampler* s = new Sampler();
        return *s;
    }

    SystemUsage snapshot() {
        std::lock_guard<std::mutex> lk(mu_);
        SystemUsage u = latest_;
        u.age_s = last_ms_ < 0.0 ? -1.0 : (mono_ms() - last_ms_) / 1000.0;
        return u;
    }

private:
    Sampler() { std::thread(&Sampler::run, this).detach(); }

    void run() {
        auto prev = read_proc_stat();
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            SystemUsage u;
            auto cur = read_proc_stat();
            if (!cur.empty() && cur.size() == prev.size()) {
                u.cpu_pct = busy_pct(prev[0], cur[0]);
                for (std::size_t i = 1; i < cur.size(); ++i) u.cpu_core_pct.push_back(busy_pct(prev[i], cur[i]));
            }
            prev = std::move(cur);
            sample_npu(u);
            std::lock_guard<std::mutex> lk(mu_);
            latest_ = std::move(u);
            last_ms_ = mono_ms();
        }
    }

#ifdef BUP_HAVE_DXRT_IPC
    void sample_npu(SystemUsage& u) {
        if (!dxrt::isDxrtServiceRunning()) {
            u.npu_note = "dxrtd not running";
            ipc_.reset();
            return;
        }
        if (!ipc_) {
            // dxrtd answers on one shared queue, addressed by message type. This process already
            // receives there as pid (inference completions) and pid + MAX_PID (memory sync), so
            // the usage queries get a type of their own and can never take those replies.
            ipc_.reset(new dxrt::IPCClientWrapper(dxrt::IPCDefaultType(),
                                                  getpid() + 2 * dxrt::IPCClientWrapper::MAX_PID));
            if (ipc_->Initialize(false) != 0) {
                ipc_.reset();
                u.npu_note = "dxrtd IPC unavailable";
                return;
            }
        }
        for (int core = 0; core < kNpuCores; ++core) {
            dxrt::IPCClientMessage req;
            dxrt::IPCServerMessage rsp;
            req.code = dxrt::REQUEST_CODE::GET_USAGE;
            req.deviceId = 0;
            req.data = static_cast<uint64_t>(core);
            req.pid = getpid();
            if (ipc_->SendToServer(rsp, req) != 0 || rsp.result != 0) {
                u.npu_core_pct.clear();
                u.npu_note = "NPU usage query failed";
                return;
            }
            u.npu_core_pct.push_back(std::min(100.0, static_cast<double>(rsp.data) / 10.0));  // busy fraction x 1000
        }
    }

    std::unique_ptr<dxrt::IPCClientWrapper> ipc_;
#else
    void sample_npu(SystemUsage& u) { u.npu_note = "NPU usage not available with this DXRT version"; }
#endif

    std::mutex mu_;
    SystemUsage latest_;
    double last_ms_{-1.0};
};

}  // namespace

SystemUsage system_usage() { return Sampler::instance().snapshot(); }

}  // namespace bup
