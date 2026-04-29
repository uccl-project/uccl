#pragma once

#include <cstddef>
#include <cstdint>

namespace uccl {

static constexpr int kQueueSize = 2048;
static constexpr int kChannelPerProxy = 8;
static constexpr int kNumProxyThs = 4;
static constexpr int kMaxInflightNormal = 512;
static constexpr int kWriteAddrShiftNormal = 2;
static constexpr int kPrintCycleInterval = 1000000000;

enum class CmdType : uint8_t {
  EMPTY = 0,
  WRITE = 1,
  ATOMIC = 2,
  QUIET = 3,
  BARRIER = 4,
  PUT_VALUE = 5,
};

__device__ __host__ __forceinline__ CmdType make_cmd_type(
    CmdType base, bool is_combine, bool low_latency) {
  uint8_t v = static_cast<uint8_t>(base);
  if (is_combine) v |= (1u << 6);
  if (low_latency) v |= (1u << 7);
  return static_cast<CmdType>(v);
}

__device__ __host__ __forceinline__ CmdType get_base_cmd(CmdType c) {
  return static_cast<CmdType>(static_cast<uint8_t>(c) & 0x3fu);
}

#pragma pack(push, 1)
struct TransferCmd {
  CmdType cmd_type;
  uint8_t dst_rank;
  union {
    struct {
      uint32_t atomic_val : 8;
      uint32_t bytes : 24;
    };
    uint32_t bytes_and_val;
  };
  uint32_t req_rptr;
  union {
    uint32_t req_lptr;
    int value;
  };
  union {
    uint16_t expert_idx;
    uint16_t atomic_offset;
  };
};
#pragma pack(pop)

static_assert(sizeof(TransferCmd) == 16, "TransferCmd must be 128 bits");

__device__ __forceinline__ uint64_t ld_volatile_u64(uint64_t* ptr) {
  uint64_t ans;
  asm volatile("ld.volatile.global.u64 %0, [%1];"
               : "=l"(ans)
               : "l"(ptr)
               : "memory");
  return ans;
}

__device__ __forceinline__ uint64_t ld_cv_u64(uint64_t const* ptr) {
  uint64_t ans;
  asm volatile("ld.global.cv.u64 %0, [%1];"
               : "=l"(ans)
               : "l"(ptr)
               : "memory");
  return ans;
}

struct alignas(128) DeviceToHostCmdBuffer {
  uint64_t head = 0;
  uint64_t tail = 0;
  TransferCmd buf[kQueueSize];
  uint64_t cycle_accum = 0;
  uint64_t op_count = 0;
  uint64_t cycle_start = 0;
  uint64_t cycle_end = 0;
  uint32_t capacity = kQueueSize;

  __device__ __forceinline__ uint64_t volatile_tail() {
    return ld_volatile_u64(&tail);
  }

  __device__ __forceinline__ uint64_t volatile_head() {
    return ld_volatile_u64(&head);
  }

  __device__ static constexpr uint32_t mask() { return kQueueSize - 1; }

  __device__ __forceinline__ bool atomic_set_and_commit(
      TransferCmd const& item, uint64_t* out_slot = nullptr) {
    uint64_t slot;
    while (true) {
      uint64_t h = ld_volatile_u64(&head);
      uint64_t t = ld_volatile_u64(&tail);
      if (h - t == kQueueSize) {
        __nanosleep(64);
        continue;
      }
      unsigned long long prev =
          atomicCAS(reinterpret_cast<unsigned long long*>(&head),
                    static_cast<unsigned long long>(h),
                    static_cast<unsigned long long>(h + 1));
      if (prev == h) {
        slot = h;
        break;
      }
    }

    uint32_t idx = static_cast<uint32_t>(slot) & mask();
    TransferCmd tmp = item;
    auto saved_cmd = tmp.cmd_type;
    tmp.cmd_type = CmdType::EMPTY;
    buf[idx] = tmp;
    __threadfence_system();
    buf[idx].cmd_type = saved_cmd;
    if (out_slot) *out_slot = slot;
    return true;
  }
};

__device__ __forceinline__ void trap() {
  asm volatile("trap;");
}

template <bool use_normal_mode = false>
__device__ __forceinline__ void nvshmemi_ibgda_put_nbi_warp(
    uint64_t req_rptr, uint64_t req_lptr, size_t bytes, int dst_rank,
    int expert_idx, int lane_id, int, uint64_t const* d2h_channel_addrs,
    int num_d2h_channel_addrs, bool is_combine, int low_latency_buffer_idx = 0,
    uint64_t atomic_offset = 0, uint64_t atomic_val = 0, int num_tokens = 1) {
  if (lane_id != 0) return;
  int thread_idx = (expert_idx % num_d2h_channel_addrs) % kNumProxyThs;
  int per_thread_d2h_channel_idx =
      (expert_idx % num_d2h_channel_addrs) / kNumProxyThs;
  EP_DEVICE_ASSERT(per_thread_d2h_channel_idx < kChannelPerProxy);
  int d2h_channel_idx = thread_idx * kChannelPerProxy + per_thread_d2h_channel_idx;
  EP_DEVICE_ASSERT(d2h_channel_idx < num_d2h_channel_addrs);

  auto* h = reinterpret_cast<DeviceToHostCmdBuffer*>(
      static_cast<uintptr_t>(d2h_channel_addrs[d2h_channel_idx]));
#ifdef EP_UCCL_DEVICE_TRACE
  if (expert_idx < 2048) {
    printf("[UCCL_DEVICE_TRACE] WRITE before commit ch=%d ring=%p dst=%d bytes=%lu lptr_off=%lu rptr_off=%lu head=%lu tail=%lu\n",
           d2h_channel_idx, h, dst_rank, static_cast<unsigned long>(bytes),
           static_cast<unsigned long>(req_lptr),
           static_cast<unsigned long>(req_rptr),
           static_cast<unsigned long>(h->volatile_head()),
           static_cast<unsigned long>(h->volatile_tail()));
  }
#endif
  constexpr int kWriteAddrShift = kWriteAddrShiftNormal;
  constexpr unsigned long long kWriteAddrAlignMask =
      (1ull << kWriteAddrShift) - 1ull;
  EP_DEVICE_ASSERT((req_rptr & kWriteAddrAlignMask) == 0);
  EP_DEVICE_ASSERT((req_lptr & kWriteAddrAlignMask) == 0);
  EP_DEVICE_ASSERT((bytes >> 24) == 0);

  TransferCmd cmd{};
  cmd.cmd_type = make_cmd_type(CmdType::WRITE, is_combine, low_latency_buffer_idx);
  cmd.req_rptr = static_cast<uint32_t>(req_rptr >> kWriteAddrShift);
  cmd.req_lptr = static_cast<uint32_t>(req_lptr >> kWriteAddrShift);
  cmd.bytes = static_cast<uint32_t>(bytes);
  cmd.dst_rank = static_cast<uint8_t>(dst_rank);
  if constexpr (use_normal_mode) {
    EP_DEVICE_ASSERT((atomic_offset >> 16) == 0);
    EP_DEVICE_ASSERT((atomic_val >> 8) == 0);
    cmd.atomic_offset = static_cast<uint16_t>(atomic_offset);
    cmd.atomic_val = static_cast<uint32_t>(atomic_val);
  } else {
    cmd.expert_idx = static_cast<uint16_t>(expert_idx);
    EP_DEVICE_ASSERT(num_tokens > 0 && num_tokens <= 255);
    cmd.atomic_val = static_cast<uint8_t>(num_tokens);
  }
  uint64_t slot = 0;
  h->atomic_set_and_commit(cmd, &slot);
#ifdef EP_UCCL_DEVICE_TRACE
  if (slot < 8) {
    printf("[UCCL_DEVICE_TRACE] WRITE ch=%d ring=%p slot=%lu dst=%d bytes=%lu lptr=%lu rptr=%lu head=%lu tail=%lu\n",
           d2h_channel_idx, h, static_cast<unsigned long>(slot), dst_rank,
           static_cast<unsigned long>(bytes),
           static_cast<unsigned long>(cmd.req_lptr),
           static_cast<unsigned long>(cmd.req_rptr),
           static_cast<unsigned long>(h->volatile_head()),
           static_cast<unsigned long>(h->volatile_tail()));
  }
#endif
}

__device__ __forceinline__ void nvshmemi_ibgda_put_value_nbi(
    uint64_t rptr_offset, uint64_t value, size_t bytes, int dst_rank,
    int channel_idx, const uint64_t* d2h_channel_addrs,
    int num_d2h_channel_addrs) {
  if (num_d2h_channel_addrs <= 0) return;
  int d2h_channel_idx = channel_idx % num_d2h_channel_addrs;
  auto* h = reinterpret_cast<DeviceToHostCmdBuffer*>(
      static_cast<uintptr_t>(d2h_channel_addrs[d2h_channel_idx]));
  TransferCmd cmd{};
  cmd.cmd_type = CmdType::PUT_VALUE;
  cmd.dst_rank = static_cast<uint8_t>(dst_rank);
  cmd.bytes_and_val = static_cast<uint32_t>(value);
  EP_DEVICE_ASSERT((rptr_offset & ((1ull << kWriteAddrShiftNormal) - 1ull)) == 0);
  EP_DEVICE_ASSERT(bytes > 0 && bytes <= sizeof(uint64_t));
  cmd.req_rptr = static_cast<uint32_t>(rptr_offset >> kWriteAddrShiftNormal);
  cmd.req_lptr = static_cast<uint32_t>(value >> 32);
  cmd.atomic_offset = static_cast<uint16_t>(bytes);
  uint64_t slot = 0;
  h->atomic_set_and_commit(cmd, &slot);
#ifdef EP_UCCL_DEVICE_TRACE
  if (slot < 8) {
    printf("[UCCL_DEVICE_TRACE] PUT_VALUE ch=%d ring=%p slot=%lu dst=%d bytes=%lu rptr=%lu value=%lu head=%lu tail=%lu\n",
           d2h_channel_idx, h, static_cast<unsigned long>(slot), dst_rank,
           static_cast<unsigned long>(bytes),
           static_cast<unsigned long>(cmd.req_rptr),
           static_cast<unsigned long>(value),
           static_cast<unsigned long>(h->volatile_head()),
           static_cast<unsigned long>(h->volatile_tail()));
  }
#endif
}

__device__ __forceinline__ void wait_until_cmd_consumed(
    DeviceToHostCmdBuffer* h, uint64_t slot, int nvl_rank = -1,
    CmdType cmd_type = CmdType::EMPTY, int label = -1) {
  auto last_print = clock64();
  while (true) {
    uint64_t cur_tail = h->volatile_tail();
    if (cur_tail > slot) break;
    if ((clock64() - last_print) > kPrintCycleInterval) {
#ifdef EP_UCCL_DEVICE_TRACE
      printf("[wait_until_cmd_consumed nvl:%d cmd:%d label:%d] waiting head=%lu tail=%lu slot=%lu\n",
             nvl_rank, static_cast<int>(cmd_type), label,
             static_cast<unsigned long>(h->volatile_head()),
             static_cast<unsigned long>(cur_tail),
             static_cast<unsigned long>(slot));
#endif
      last_print = clock64();
    }
  }
}

__device__ __forceinline__ void nvshmemi_ibgda_quiet(
    uint64_t const* d2h_channel_addrs, int num_d2h_channel_addrs,
    int nvl_rank = -1, int label = -1) {
  EP_DEVICE_ASSERT(num_d2h_channel_addrs % kChannelPerProxy == 0);
  EP_DEVICE_ASSERT(num_d2h_channel_addrs / kChannelPerProxy == kNumProxyThs);
  uint64_t slots[kNumProxyThs * kChannelPerProxy];
  int num_posted = 0;
  for (int d2h_channel_idx = 0; d2h_channel_idx < num_d2h_channel_addrs;
       ++d2h_channel_idx) {
    auto* h = reinterpret_cast<DeviceToHostCmdBuffer*>(
        static_cast<uintptr_t>(d2h_channel_addrs[d2h_channel_idx]));
    while (true) {
      uint64_t cur_head = h->volatile_head();
      uint64_t cur_tail = h->volatile_tail();
      if (cur_head - cur_tail < 1) {
        TransferCmd cmd{};
        cmd.cmd_type = CmdType::QUIET;
        uint64_t slot = cur_head;
        h->atomic_set_and_commit(cmd, &slot);
#ifdef EP_UCCL_DEVICE_TRACE
        if (slot < 8) {
          printf("[UCCL_DEVICE_TRACE] QUIET ch=%d ring=%p slot=%lu head=%lu tail=%lu\n",
                 d2h_channel_idx, h, static_cast<unsigned long>(slot),
                 static_cast<unsigned long>(h->volatile_head()),
                 static_cast<unsigned long>(h->volatile_tail()));
        }
#endif
        slots[num_posted++] = slot;
        break;
      }
    }
  }

  for (int i = 0; i < num_posted; ++i) {
    auto* h = reinterpret_cast<DeviceToHostCmdBuffer*>(
        static_cast<uintptr_t>(d2h_channel_addrs[i]));
    wait_until_cmd_consumed(h, slots[i], nvl_rank, CmdType::QUIET, label);
  }
}

#ifdef EP_UCCL_DEVICE_TRACE
__device__ __forceinline__ void debug_post_marker(
    uint64_t const* d2h_channel_addrs, int num_d2h_channel_addrs, int label) {
  if (num_d2h_channel_addrs <= 0) return;
  auto* h = reinterpret_cast<DeviceToHostCmdBuffer*>(
      static_cast<uintptr_t>(d2h_channel_addrs[0]));
  TransferCmd cmd{};
  cmd.cmd_type = CmdType::QUIET;
  uint64_t slot = 0;
  h->atomic_set_and_commit(cmd, &slot);
  printf("[UCCL_DEVICE_TRACE] MARK label=%d ring=%p slot=%lu head=%lu tail=%lu\n",
         label, h, static_cast<unsigned long>(slot),
         static_cast<unsigned long>(h->volatile_head()),
         static_cast<unsigned long>(h->volatile_tail()));
}
#endif

}  // namespace uccl
