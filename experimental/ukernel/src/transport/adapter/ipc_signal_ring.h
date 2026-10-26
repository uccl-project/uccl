#pragma once

// Shared-memory signal ring layout, shared between host code and
// device kernels: a fused PutSignal (TaskType::CollPut) writes the tag
// into the peer's ring directly from the GPU, through the same channel
// the host SignalBackend uses. Receivers always poll from the CPU.

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace UKernel {
namespace Transport {

inline constexpr size_t kSignalRingSize = 4096;  // power of two

struct SignalSlot {
  std::atomic<bool> ready{false};
  uint64_t tag{0};
};

struct PeerSignalRing {
  SignalSlot slots[kSignalRingSize];
  std::atomic<uint64_t> write_idx{0};
  std::atomic<uint64_t> read_idx{0};
};

// POD mirror used by device kernels; layout must match exactly.
struct SignalSlotDevice {
  bool ready;
  uint8_t _pad[7];
  uint64_t tag;
};
struct PeerSignalRingDevice {
  SignalSlotDevice slots[kSignalRingSize];
  uint64_t write_idx;
  uint64_t read_idx;
};

static_assert(sizeof(SignalSlotDevice) == sizeof(SignalSlot),
              "SignalSlot layout mismatch");
static_assert(offsetof(SignalSlotDevice, tag) == offsetof(SignalSlot, tag),
              "SignalSlot.tag offset mismatch");
static_assert(sizeof(PeerSignalRingDevice) == sizeof(PeerSignalRing),
              "PeerSignalRing layout mismatch");
static_assert(offsetof(PeerSignalRingDevice, write_idx) ==
                  offsetof(PeerSignalRing, write_idx),
              "PeerSignalRing.write_idx offset mismatch");
static_assert(offsetof(PeerSignalRingDevice, read_idx) ==
                  offsetof(PeerSignalRing, read_idx),
              "PeerSignalRing.read_idx offset mismatch");

}  // namespace Transport
}  // namespace UKernel
