/*
 * Device-side helper for RdmaFusedRing.
 *
 * The layout must match RdmaFusedRing::DeviceHandle in rdma_fused_proxy.cc.
 * A kernel calls rdma_fused_ring_push() after finishing a fused reduce+copy
 * to notify the CCL proxy that the corresponding Cmd is ready to be posted
 * over RDMA.
 */
#pragma once

#include <cstdint>

namespace UKernel {
namespace Device {

struct RdmaFusedRingDeviceHandle {
  uint64_t* indices;
  uint64_t* ready;
  uint64_t* head;  // device-owned producer counter
  int size;
};

__device__ __forceinline__ uint64_t rdma_fused_ring_push(
    RdmaFusedRingDeviceHandle* h, uint64_t index) {
  // Claim a slot. The host never writes head, so a simple atomic add is
  // safe even with multiple device producers.
  uint64_t slot = atomicAdd(reinterpret_cast<unsigned long long*>(h->head), 1ull);

  // Backpressure: wait until the host has consumed this slot.
  uint64_t* rp = h->ready + (slot % static_cast<uint64_t>(h->size));
  while (*(volatile uint64_t*)rp != 0) {
    // Host clears ready after consuming; spin without waiting on a CUDA
    // event. This is producer backpressure, not a GPU wait-sync on RDMA
    // completion.
#ifdef __HIP_PLATFORM_AMD__
    __builtin_amdgcn_s_sleep(2);
#else
    __nanosleep(200);
#endif
  }

  h->indices[slot % static_cast<uint64_t>(h->size)] = index;
  __threadfence_system();
  *(volatile uint64_t*)rp = 1;
  return slot;
}

}  // namespace Device
}  // namespace UKernel
