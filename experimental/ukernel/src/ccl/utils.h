#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace UKernel {
namespace CCL {

constexpr uint32_t kNoOp = 0xFFFFFFFFu;

inline void* byte_offset(void* base, size_t offset) {
  return static_cast<void*>(static_cast<char*>(base) + offset);
}

inline void const* byte_offset(void const* base, size_t offset) {
  return static_cast<void const*>(static_cast<char const*>(base) + offset);
}

inline void add_dep(std::vector<uint32_t>& deps, uint32_t dep) {
  if (dep == kNoOp) return;
  for (uint32_t existing : deps) {
    if (existing == dep) return;
  }
  deps.push_back(dep);
}

inline size_t ceil_div(size_t a, size_t b) {
  return b == 0 ? 0 : (a + b - 1) / b;
}

}  // namespace CCL
}  // namespace UKernel
