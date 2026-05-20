#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace UKernel {
namespace CCL {

inline void validate_span(char const* what, size_t offset, size_t bytes,
                          size_t capacity) {
  if (offset > capacity || bytes > capacity - offset) {
    throw std::invalid_argument(std::string(what) + " out of range");
  }
}

inline void* byte_offset(void* base, size_t offset) {
  return static_cast<void*>(static_cast<char*>(base) + offset);
}

inline void const* byte_offset(void const* base, size_t offset) {
  return static_cast<void const*>(static_cast<char const*>(base) + offset);
}

inline void add_dep(std::vector<uint32_t>& deps, uint32_t dep) {
  constexpr uint32_t kNoOp = 0xFFFFFFFFu;
  if (dep == kNoOp) return;
  for (uint32_t existing : deps) {
    if (existing == dep) return;
  }
  deps.push_back(dep);
}

}  // namespace CCL
}  // namespace UKernel
