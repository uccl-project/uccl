#ifndef RING_BUFFER_CUH
#define RING_BUFFER_CUH

#include "common.hpp"  // For TransferCmd, QUEUE_SIZE
#include "copy_ring.hpp"
#include <atomic>
#include <cuda.h>

// Command structure for each transfer
struct TransferCmd {
  uint64_t cmd;
  uint32_t dst_rank;  // remote node id (MPI-style)
  uint32_t dst_gpu;   // GPU id on remote node
  void* src_ptr;      // device pointer to data
  uint64_t bytes;     // transfer size
};

enum class FlowDirection { HostToDevice, DeviceToHost };

__device__ __forceinline__ uint64_t ld_volatile(uint64_t* ptr) {
  uint64_t ans;
  asm volatile("ld.volatile.global.u64 %0, [%1];"
               : "=l"(ans)
               : "l"(ptr)
               : "memory");
  return ans;
}

// Host-pinned lock-free ring buffer
template <typename T, FlowDirection Dir, uint32_t Capacity>
struct alignas(128) RingBuffer {
  uint64_t head;
  uint64_t tail;
  T buf[Capacity];
  uint64_t cycle_accum;
  uint64_t op_count;
  uint64_t cycle_start;
  uint64_t cycle_end;
  uint32_t capacity = Capacity;

  __host__ __device__ static constexpr uint32_t mask() { return Capacity - 1; }

  __host__ __device__ __forceinline__ bool full() const {
    return head - tail == Capacity;
  }

  __host__ __device__ __forceinline__ bool empty() const {
    return head == tail;
  }

  __host__ __device__ __forceinline__ void set_buffer(int idx, T entry) {
    buf[idx & mask()] = entry;
  }

  __host__ __device__ __forceinline__ bool push(T const& item) {
    if (full()) return false;
    buf[head & mask()] = item;
    head++;
    return true;
  }

  __host__ __device__ __forceinline__ T get_entry(int idx) const {
    return buf[idx & mask()];
  }

  __host__ __device__ __forceinline__ void commit_with_head(int new_head) {
#if __CUDA_ARCH__
    if constexpr (Dir == FlowDirection::DeviceToHost) __threadfence_system();
#else
    if constexpr (Dir == FlowDirection::DeviceToHost)
      std::atomic_thread_fence(std::memory_order_release);
#endif
    head = new_head;
  }

  __host__ __device__ __forceinline__ bool pop(T& out) {
    if (empty()) return false;

#if __CUDA_ARCH__
    if constexpr (Dir == FlowDirection::HostToDevice) __threadfence();
#endif
    out = buf[tail & mask()];
    tail++;
    return true;
  }

  __host__ __device__ __forceinline__ uint64_t volatile_tail() {
#if __CUDA_ARCH__
    return ld_volatile(&tail);
#else  // host compilation
    // Ordinary C++: read through a `volatile` alias to prevent the
    // compiler from reordering or eliding the load.
    return *reinterpret_cast<volatile uint64_t const*>(&tail);
    /*  an alternative with explicit atomics:
        return __atomic_load_n(&tail, __ATOMIC_ACQUIRE);
    */
#endif
  }
};

typedef RingBuffer<TransferCmd, FlowDirection::DeviceToHost, kQueueSize>
    DeviceToHostCmdBuffer;
typedef RingBuffer<CopyTask, FlowDirection::HostToDevice, COPY_RING_CAP>
    HostToDeviceNVlinkBuffer;

#endif  // RING_BUFFER_CUH