#ifndef RING_BUFFER_CUH
#define RING_BUFFER_CUH

#include "common.hpp"
#include "util/gpu_rt.h"
#include <infiniband/verbs.h>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64)
#include <cassert>
#include <immintrin.h>
#endif
#ifndef COPY_RING_CAP
#define COPY_RING_CAP 4096
#endif

enum class CmdType : uint64_t { EMPTY = 0, WRITE, ATOMIC, QUIET, BARRIER };

// Command structure for each transfer
struct TransferCmd {
  // NOTE(MaoZiming): cmd is used to identify the command type and needs to be
  // set in order for proxy to process the command.
  CmdType cmd_type;
  uint64_t cmd;
  uint32_t dst_rank;  // remote node id (MPI-style)
  uint32_t dst_gpu;   // GPU id on remote node
  void* src_ptr;      // device pointer to data
  uint64_t bytes;     // transfer size

  uint64_t req_rptr;
  uint64_t req_lptr;
  int warp_id;
  int expert_idx;
  int lane_id;
  int message_idx;
  bool is_atomic;
  int value;
  bool is_combine;
  int low_latency_buffer_idx;

  uint64_t atomic_offset;
  uint64_t atomic_val;
};

struct CopyTask {
  uint64_t wr_id;
  int dst_dev;
  void* src_ptr;
  void* dst_ptr;
  size_t bytes;
};

enum class FlowDirection { HostToDevice, DeviceToHost, HostToHost };

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
#include <atomic>
#define HOST_ACQUIRE() std::atomic_thread_fence(std::memory_order_acquire)
#define HOST_RELEASE() std::atomic_thread_fence(std::memory_order_release)
#else
#define HOST_ACQUIRE()
#define HOST_RELEASE()
#endif

__device__ __forceinline__ uint64_t ld_volatile(uint64_t* ptr) {
#if defined(__CUDA_ARCH__)
  uint64_t ans;
  asm volatile("ld.volatile.global.u64 %0, [%1];"
               : "=l"(ans)
               : "l"(ptr)
               : "memory");
  return ans;
#elif defined(__HIP_DEVICE_COMPILE__)
  return __builtin_nontemporal_load(ptr);
#else
  return *((volatile uint64_t const*)ptr);
#endif
}

template <typename T, FlowDirection Dir, uint32_t Capacity>
struct alignas(128) RingBuffer {
  uint64_t head = 0;
  uint64_t tail = 0;
  T buf[Capacity];
  uint64_t cycle_accum = 0;
  uint64_t op_count = 0;
  uint64_t cycle_start = 0;
  uint64_t cycle_end = 0;
  uint32_t capacity = Capacity;

  static constexpr size_t kNumWords = (Capacity + 63) / 64;
  uint64_t ack_mask[kNumWords] = {};
  static constexpr int kFenceBatch = 8;
  inline void mark_acked(size_t idx) noexcept {
    assert(idx < Capacity && "mark_acked: idx out of range");
    const size_t word = idx >> 6;  // idx / 64
    const size_t bit = idx & 63;   // idx % 64
    ack_mask[word] |= (1ull << bit);
  }

  inline void clear_acked(size_t idx) noexcept {
    assert(idx < Capacity && "clear_acked: idx out of range");
    const size_t word = idx >> 6;
    const size_t bit = idx & 63;
    ack_mask[word] &= ~(1ull << bit);
  }

  inline bool is_acked(size_t idx) const noexcept {
    assert(idx < Capacity && "is_acked: idx out of range");
    const size_t word = idx >> 6;
    const size_t bit = idx & 63;
    return (ack_mask[word] >> bit) & 1ull;
  }

  // Return next index >= start_idx whose bit is UNSET (0)
  inline size_t next_unacked(size_t start_idx) const noexcept {
    if (start_idx >= Capacity) return Capacity;
    size_t word = start_idx >> 6;
    size_t bit = start_idx & 63;

    size_t remaining = Capacity - start_idx;
    size_t span = remaining < (64 - bit) ? remaining : (64 - bit);

    // Mask the slice in this word to only the valid range
    uint64_t slice = (ack_mask[word] >> bit);
    uint64_t range_mask = (span == 64) ? ~0ull : ((1ull << span) - 1);
    slice &= range_mask;

    // If there is any 0 in the slice, return its position
    uint64_t inv = (~slice) & range_mask;
    if (inv) return start_idx + __builtin_ctzll(inv);

    // Scan subsequent full words, but clamp the last word
    for (++word; word < kNumWords; ++word) {
      size_t base = word << 6;
      size_t valid =
          Capacity > base ? (size_t)std::min<size_t>(64, Capacity - base) : 0;
      if (!valid) break;

      uint64_t m = (valid == 64) ? ~0ull : ((1ull << valid) - 1);
      uint64_t blk = ack_mask[word] & m;
      if (blk != m) {
        uint64_t inv2 = (~blk) & m;
        return base + __builtin_ctzll(inv2);
      }
    }
    return Capacity;
  }

  inline void clear_acked_range(size_t start, size_t end) noexcept {
    if (start >= end) return;
    if (start >= Capacity) return;
    assert(end <= Capacity);

    size_t start_word = start >> 6;
    size_t end_word = (end - 1) >> 6;
    if (start_word == end_word) {
      uint64_t mask = ((~0ull >> (64 - (end - start))) << (start & 63));
      ack_mask[start_word] &= ~mask;
      return;
    }
    if (start & 63) {
      ack_mask[start_word] &= ~((~0ull) << (start & 63));
      ++start_word;
    }
    for (size_t w = start_word; w < end_word; ++w) ack_mask[w] = 0;
    if ((end & 63) != 0) {
      uint64_t mask = (~0ull) >> (64 - (end & 63));
      ack_mask[end_word] &= ~mask;
    } else {
      ack_mask[end_word] = 0;
    }
  }

  inline uint64_t advance_tail_from_mask() noexcept {
    size_t local = (size_t)(tail % Capacity);

    // First pass: [local, Capacity)
    size_t next0 = next_unacked(local);
    if (next0 == Capacity) {
      // Everything from local..end is acked; maybe we can wrap
      if (local != 0) {
        // Consume [local, Capacity)
        clear_acked_range(local, Capacity);
        tail += (Capacity - local);

        // Second pass: [0, new_local)
        size_t wrap0 = next_unacked(0);
        if (wrap0 > 0 && wrap0 <= local) {
          clear_acked_range(0, wrap0);
          tail += wrap0;
        }
      }
    } else if (next0 > local) {
      // Consume [local, next0)
      clear_acked_range(local, next0);
      tail += (next0 - local);
    }

    // Publish with release semantics
    cpu_volatile_store_tail(tail);
    return tail;
  }

  RingBuffer() {
    for (uint32_t i = 0; i < Capacity; i++) {
      buf[i] = {};
    }
  }

  /* TODO(MaoZiming) to refactor */
  struct ibv_qp* ack_qp = nullptr;
  void* ctx = nullptr;
  ibv_mr* ack_mr = nullptr;
  uint64_t ack_buf[RECEIVER_BATCH_SIZE] = {0};

  inline void cpu_volatile_store_tail(uint64_t new_tail) {
    __atomic_store_n(&tail, new_tail, __ATOMIC_RELEASE);
  }

  inline uint64_t volatile_load_cmd(int idx) const {
    return __atomic_load_n(&buf[idx & mask()].cmd, __ATOMIC_ACQUIRE);
  }

  inline T& load_cmd_entry(int idx) { return buf[idx & mask()]; }

  inline void volatile_store_cmd(int idx, uint64_t val) {
    __atomic_store_n(&buf[idx & mask()].cmd, val, __ATOMIC_RELEASE);
  }

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
    commit_with_head(head + 1);
    return true;
  }

  __host__ __forceinline__ bool pushN(T const* items, int n) {
    if (n <= 0) return true;
    uint64_t h = head;
    uint64_t t = tail;
    uint64_t free_slots = capacity - (h - t);
    if (n > static_cast<int>(free_slots)) return false;

    for (int i = 0; i < n; ++i) buf[(h + i) & mask()] = items[i];

    commit_with_head(h + n);
    return true;
  }

  __host__ __device__ __forceinline__ T get_entry(int idx) const {
    return buf[idx & mask()];
  }

  __host__ __device__ __forceinline__ void commit_with_head(int new_head) {
#if __CUDA_ARCH__ || __HIP_DEVICE_COMPILE__
    if constexpr (Dir == FlowDirection::DeviceToHost) __threadfence_system();
#else
    if constexpr (Dir == FlowDirection::DeviceToHost)
      std::atomic_thread_fence(std::memory_order_release);
    if constexpr (Dir == FlowDirection::HostToHost) HOST_RELEASE();
#endif
    head = new_head;
  }

  __host__ __device__ __forceinline__ bool pop(T& out) {
    if (empty()) return false;

#if __CUDA_ARCH__ || __HIP_DEVICE_COMPILE__
    if constexpr (Dir == FlowDirection::HostToDevice) __threadfence();
#else
    if constexpr (Dir == FlowDirection::HostToHost) HOST_ACQUIRE();
#endif
    out = buf[tail & mask()];
    tail++;
    return true;
  }

  __host__ __device__ __forceinline__ int popN(T* out, int n) {
    if (n <= 0) return 0;
    uint64_t t = tail;
    uint64_t h = head;
    uint64_t avail = h - t;
    if (avail == 0) return 0;
    int cnt = (n < static_cast<int>(avail)) ? n : static_cast<int>(avail);
#if __CUDA_ARCH__ || __HIP_DEVICE_COMPILE__
    if constexpr (Dir == FlowDirection::HostToDevice) __threadfence();
#else
    if constexpr (Dir == FlowDirection::HostToHost) HOST_ACQUIRE();
#endif
    for (int i = 0; i < cnt; ++i) out[i] = buf[(t + i) & mask()];
    tail = t + cnt;
    return cnt;
  }

  __host__ __device__ __forceinline__ uint64_t volatile_tail() {
#if __CUDA_ARCH__ || __HIP_DEVICE_COMPILE__
    return ld_volatile(&tail);
#else
    return *reinterpret_cast<volatile uint64_t const*>(&tail);
#endif
  }

  __host__ __device__ __forceinline__ uint64_t volatile_head() {
    uint64_t val;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ld_volatile(&head);
#elif defined(__x86_64__)
    asm volatile("movq %1, %0" : "=r"(val) : "m"(head) : "memory");
#elif defined(__aarch64__)
    asm volatile("ldr %0, [%1]" : "=r"(val) : "r"(&head) : "memory");
#else
#error "Unsupported architecture"
#endif
    return val;
  }

  __host__ __device__ inline bool atomic_set_and_commit(
      const T& item, uint64_t* out_slot = nullptr) {
    uint64_t slot;
    while (true) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      uint64_t h = ld_volatile(&head);
      uint64_t t = ld_volatile(&tail);
      if (h - t == Capacity) {
        __nanosleep(64);
        continue;
      }
      unsigned long long prev =
          atomicCAS((unsigned long long*)&head, (unsigned long long)h,
                    (unsigned long long)(h + 1));
      if (prev == h) {
        slot = h;
        break;
      }
#else
      uint64_t h = __atomic_load_n(&head, __ATOMIC_RELAXED);
      uint64_t t = __atomic_load_n(&tail, __ATOMIC_RELAXED);
      if (h - t == Capacity) {
        cpu_relax();
        continue;
      }
      uint64_t expected = h;
      if (__atomic_compare_exchange_n(&head, &expected, h + 1, true,
                                      __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
        slot = h;
        break;
      }
#endif
    }
    uint32_t idx = (uint32_t)slot & mask();

    T tmp = item;
    auto saved_cmd = tmp.cmd;
    tmp.cmd = 0;
    buf[idx] = tmp;

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if constexpr (Dir == FlowDirection::DeviceToHost)
      __threadfence_system();
    else
      __threadfence();
#else
    std::atomic_thread_fence(std::memory_order_release);
#endif

    buf[idx].cmd = saved_cmd;
    if (out_slot) *out_slot = slot;
    return true;
  }
};

typedef RingBuffer<TransferCmd, FlowDirection::DeviceToHost, kQueueSize>
    DeviceToHostCmdBuffer;
typedef RingBuffer<CopyTask, FlowDirection::HostToDevice, COPY_RING_CAP>
    HostToDeviceNVlinkBuffer;
typedef RingBuffer<CopyTask, FlowDirection::HostToHost, COPY_RING_CAP>
    CopyRingBuffer;

static inline uintptr_t alloc_cmd_ring() {
  void* raw = nullptr;
  auto err = cudaMallocHost(&raw, sizeof(DeviceToHostCmdBuffer));
  if (err != cudaSuccess || raw == nullptr) {
    throw std::runtime_error("cudaMallocHost(DeviceToHostCmdBuffer) failed");
  }
  auto* rb = static_cast<DeviceToHostCmdBuffer*>(raw);
  new (rb) DeviceToHostCmdBuffer{};
  return reinterpret_cast<uintptr_t>(rb);
}

static inline void free_cmd_ring(uintptr_t addr) {
  if (!addr) return;
  auto* rb = reinterpret_cast<DeviceToHostCmdBuffer*>(addr);
  rb->~DeviceToHostCmdBuffer();
  auto err = cudaFreeHost(static_cast<void*>(rb));
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaFreeHost(DeviceToHostCmdBuffer) failed");
  }
}

#endif  // RING_BUFFER_CUH