#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
namespace uccl_lite::cpu_router {
using Generation = uint32_t;
struct alignas(128) GenerationSignal {
  Generation value = 0;
};
struct SignalRef {
  GenerationSignal* word;
  Generation acquire() const {
    return __atomic_load_n(&word->value, __ATOMIC_ACQUIRE);
  }
  void publish(Generation g) const {
    __atomic_store_n(&word->value, g, __ATOMIC_RELEASE);
  }
  void wait(Generation g) const;
};
// One publisher, up to 64 independent readers. Generations never wrap.
struct PacketOwnershipView {
  SignalRef ready, released;
  GenerationSignal* readers;
  uint64_t readerMask;
  void acquire(Generation previous) const {
    if (previous) released.wait(previous);
  }
  void publish(Generation g) const { ready.publish(g); }
  void consume(unsigned reader, Generation g) const {
    if (reader >= 64 || !(readerMask & (1ULL << reader)))
      throw std::invalid_argument("unbound reader");
    SignalRef{readers + reader}.publish(g);
  }
  void release(Generation g) const {
    for (unsigned i = 0; i < 64; ++i)
      if (readerMask & (1ULL << i)) SignalRef{readers + i}.wait(g);
    released.publish(g);
  }
};
struct PacketOwnership {
  GenerationSignal ready, released, readers[64];
  PacketOwnershipView view(uint64_t mask) {
    return {{&ready}, {&released}, readers, mask};
  }
};
inline Generation nextGeneration(Generation old, size_t extent = 1) {
  if (!extent || extent > UINT32_MAX - old)
    throw std::overflow_error("generation exhausted");
  return old + extent;
}
}  // namespace uccl_lite::cpu_router
