#pragma once

#include <infiniband/verbs.h>
#include <cstdlib>

namespace uccl {
namespace cc {

// Get link bandwidth in bytes/sec.
// Priority: env_var override → ibv_query_port auto-detect → 400 Gbps default.
inline double get_link_bandwidth_bps(ibv_context* ctx,
                                     char const* env_var = nullptr,
                                     uint8_t port_num = 1) {
  static constexpr double kDefault = 400.0 * 1e9 / 8.0;

  if (env_var) {
    if (auto* val = std::getenv(env_var)) {
      double gbps = std::atof(val);
      if (gbps > 0) return gbps * 1e9 / 8.0;
    }
  }

  if (ctx) {
    ibv_port_attr attr{};
    if (ibv_query_port(ctx, port_num, &attr) == 0) {
      static constexpr int kSpeeds[] = {2500,  5000,  10000, 10000,
                                        14000, 25000, 50000, 100000};
      static constexpr int kWidths[] = {1, 4, 8, 12, 2};
      auto firstBit = [](int val, int max) {
        for (int i = 0; i < max; i++)
          if (val & (1 << i)) return i;
        return max;
      };
      int spd = kSpeeds[firstBit(attr.active_speed, 7)];
      int wid = kWidths[firstBit(attr.active_width, 4)];
      return static_cast<double>(spd) * wid * 1e6 / 8.0;
    }
  }

  return kDefault;
}

}  // namespace cc
}  // namespace uccl
