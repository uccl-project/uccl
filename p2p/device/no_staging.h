#pragma once
#include <cstddef>
#include <cstdint>
class Endpoint;
struct FifoItem;
namespace uccl::device {
// Empty state on devices that transfer user buffers directly. The engine's
// compile-time policy discards the staging entry points entirely.
class NetworkStaging {
 public:
  explicit NetworkStaging(Endpoint&) {}
  void request_stop() {}
  void shutdown() {}
  bool read(uint64_t, void*, size_t, FifoItem const&);
  bool write(uint64_t, void*, size_t, FifoItem const&);
  bool advertise(uint64_t, void*, size_t, char*);
};
}  // namespace uccl::device
