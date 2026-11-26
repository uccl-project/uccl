#include "router_core.hpp"
#include <cstdint>
#include <limits>

namespace uccl_lite::cpu_router {
namespace {
void hostSpan(MemorySpan const& span) {
  if (span.kind == MemoryKind::Gpu || (!span.address && span.bytes))
    throw std::invalid_argument("process requires CPU-visible spans");
}
void independentOutput(MemorySpan const& input, MemorySpan const& output) {
  auto a = reinterpret_cast<uintptr_t>(input.address);
  auto b = reinterpret_cast<uintptr_t>(output.address);
  if (input.bytes && output.bytes &&
      (a <= b ? b - a < input.bytes : a - b < output.bytes))
    throw std::invalid_argument("pack/deduplicate output overlaps input");
}
size_t validate(MemorySpan const* sources, size_t count,
                MemorySpan const& output) {
  hostSpan(output);
  size_t bytes = 0;
  for (size_t i = 0; i < count; ++i) {
    hostSpan(sources[i]);
    independentOutput(sources[i], output);
    if (sources[i].bytes > std::numeric_limits<size_t>::max() - bytes)
      throw std::overflow_error("process input size overflow");
    bytes += sources[i].bytes;
  }
  return bytes;
}
}  // namespace
size_t PackBytes::operator()(MemorySpan const* sources, size_t count,
                             MemorySpan output) const {
  size_t bytes = validate(sources, count, output);
  if (bytes > output.bytes)
    throw std::invalid_argument("pack output too small");
  size_t offset = 0;
  for (size_t i = 0; i < count; ++i) {
    if (sources[i].bytes)
      std::memcpy(static_cast<char*>(output.address) + offset,
                  sources[i].address, sources[i].bytes);
    offset += sources[i].bytes;
  }
  return bytes;
}
size_t DeduplicateRecords::operator()(MemorySpan const* sources, size_t count,
                                      MemorySpan output) const {
  validate(sources, count, output);
  if (!recordBytes) throw std::invalid_argument("zero record width");
  for (size_t s = 0; s < count; ++s)
    if (sources[s].bytes % recordBytes)
      throw std::invalid_argument("partial deduplicate record");
  size_t used = 0;
  for (size_t s = 0; s < count; ++s)
    for (size_t i = 0; i < sources[s].bytes; i += recordBytes) {
      char const* record = static_cast<char const*>(sources[s].address) + i;
      bool found = false;
      for (size_t j = 0; j < used; j += recordBytes)
        if (!std::memcmp(static_cast<char const*>(output.address) + j, record,
                         recordBytes)) {
          found = true;
          break;
        }
      if (!found) {
        if (recordBytes > output.bytes - used)
          throw std::invalid_argument("deduplicate output too small");
        std::memcpy(static_cast<char*>(output.address) + used, record,
                    recordBytes);
        used += recordBytes;
      }
    }
  return used;
}
}  // namespace uccl_lite::cpu_router
