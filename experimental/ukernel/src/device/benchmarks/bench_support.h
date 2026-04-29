#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

inline uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

inline void print_latency(std::vector<uint64_t>& v) {
  std::sort(v.begin(), v.end());
  auto p = [&](double q) {
    size_t i = static_cast<size_t>(q * v.size());
    if (i >= v.size()) i = v.size() - 1;
    return v[i] / 1e3;
  };

  printf("Latency (us): min %.2f | p50 %.2f | p90 %.2f | p99 %.2f | max %.2f\n",
         v.front() / 1e3, p(0.5), p(0.9), p(0.99), v.back() / 1e3);
}
