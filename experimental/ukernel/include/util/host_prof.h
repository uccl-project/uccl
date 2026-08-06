#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace UKernel {
namespace CCL {

// Host-side orchestration profiler (UK_CCL_HOST_PROF=1). Aggregates
// where the per-op host cost goes — enqueue dispatch, signal drain,
// transport/put completion drain, device completion drain — and prints
// one summary at executor teardown. Purpose: confirm whether the shim is
// host-bound (per-op dispatch slower than the GPU's per-tile work) and
// where the host time is spent at high rank counts.
struct HostProf {
  static bool enabled() {
    static bool const e = std::getenv("UK_CCL_HOST_PROF") != nullptr;
    return e;
  }

  // Accumulated microseconds and op counts per stage.
  static inline std::atomic<uint64_t> enq_us{0}, enq_ops{0};
  static inline std::atomic<uint64_t> sig_us{0}, sig_ops{0};
  static inline std::atomic<uint64_t> tpt_us{0}, tpt_ops{0};
  static inline std::atomic<uint64_t> dev_us{0}, dev_ops{0};

  struct Scope {
    uint64_t t0;
    std::atomic<uint64_t>* us;
    Scope(std::atomic<uint64_t>& acc) : us(&acc) {
      t0 = enabled() ? tick() : 0;
    }
    ~Scope() {
      if (t0) us->fetch_add(tick() - t0, std::memory_order_relaxed);
    }
    static uint64_t tick() {
      return static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now().time_since_epoch())
              .count());
    }
  };

  static void print() {
    if (!enabled()) return;
    auto us = [](std::atomic<uint64_t>& a) { return a.load() / 1000.0; };
    auto op = [](std::atomic<uint64_t>& a) { return a.load(); };
    std::fprintf(
        stderr,
        "[hostprof] enq %.1fus/%lluops (%.2fus/op) sig %.1fus/%llu "
        "(%.2fus/sig) tpt %.1fus/%llu (%.2fus/put) dev %.1fus/%llu "
        "(%.2fus/dev)\n",
        us(enq_us), (unsigned long long)op(enq_ops),
        op(enq_ops) ? us(enq_us) / op(enq_ops) : 0.0, us(sig_us),
        (unsigned long long)op(sig_ops),
        op(sig_ops) ? us(sig_us) / op(sig_ops) : 0.0, us(tpt_us),
        (unsigned long long)op(tpt_ops),
        op(tpt_ops) ? us(tpt_us) / op(tpt_ops) : 0.0, us(dev_us),
        (unsigned long long)op(dev_ops),
        op(dev_ops) ? us(dev_us) / op(dev_ops) : 0.0);
  }
};

}  // namespace CCL
}  // namespace UKernel
