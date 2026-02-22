#ifndef COMMON_HPP
#define COMMON_HPP

#include "util/gpu_rt.h"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <stdio.h>
#include <unistd.h>

// #define SOFTWARE_ORDERING
#define MAX_IB_DEVS 32
// #define MEASURE_PER_OP_LATENCY
// #define MEASURE_PER_VERB_LATENCY

// Barrier type selection (can be overridden at compile time)
#ifndef USE_SENDER_BARRIER
#ifdef EFA
#define USE_RECEIVER_BARRIER
#endif
#endif

#ifdef EFA
#define EFA_QP_LOW_LATENCY_SERVICE_LEVEL 8
extern bool use_ll_sl;
#endif

#define USE_MSCCLPP_FIFO_BACKEND
// #define USE_SUBSET_BARRIER

// Intel RDMA NIC support
#ifdef INTEL_RDMA_NIC
// Use DMA-BUF for GPU memory registration (avoids nvidia_peermem dependency).
// Falls back to ibv_reg_mr_iova2 at runtime if DMA-BUF is unsupported.
#define USE_DMABUF
// Use pinned host memory for the atomic buffer instead of cudaMalloc.
// Required for NICs without nvidia_peermem (e.g. Intel irdma) so that
// ibv_reg_mr succeeds, and allows CPU proxy threads to do std::atomic ops.
#define ATOMICS_USE_HOST_MEMORY
#endif

#define kAtomicBufferSize 81960
#define kQueueSize 2048
#define kQueueMask (kQueueSize - 1)
// This is the highest we can get due to the number of bits we allocate in the
// imm for reordering buffer sequence tracking.
#define kMaxInflightLowLatency 32
#define kMaxInflightNormal 8
#define kChannelPerProxy 8
#define kNumProxyThs 4
// NCCL EFA plugin default: 8 MB mimicing (512KB*16)
// NCCL IB net.cc default: 2 MB (128KB*16)
#define kMaxInflightBytes SIZE_MAX
#define kBatchSize 32
#define kIterations 40000
#define kTestNumGpuThPerBlock 1
#define kObjectSize 7168  // 7 KB
// #define kObjectSize 10752  // 10.5 KB
// #define kObjectSize 14336  // 14 KB
#define kMaxOutstandingSends 2048  // = max_send_wr, max_recv_wr, cq_depth / 2
#define kMaxOutstandingRecvs 2048
#define kSenderAckQueueDepth 2048
#define kWarmupOps 10000
// TODO(MaoZiming): I tried to fit more bits, but this eats into offset and
// values.
#define kReorderingBufferSize 16  // Right now only 4 bits.
#define kRemoteBufferSize (kBatchSize * kNumProxyThs * kObjectSize * 100)
#define MAIN_THREAD_CPU_IDX 31
#define MAX_NUM_GPUS 8
#define RECEIVER_BATCH_SIZE 16
#define kAtomicWrTag 0xa70a000000000000ULL
#define kAtomicMask 0x0000FFFFFFFFFFFFULL
#define kBarrierWrTag 0xbaba000000000000ULL
#define kBarrierMask 0x0000FFFFFFFFFFFFULL
#define kPrintCycleInterval 100000000000ULL
#define MAX_RETRIES 100
#define RETRY_DELAY_MS 50
#define QKEY 0x11111111u
#define kLargeAtomicValue 33550000
#define kMaxSendAtomicValue 16383

// P2P enable flags (once per GPU pair)
extern std::once_flag peer_ok_flag[MAX_NUM_GPUS][MAX_NUM_GPUS];
bool pin_thread_to_cpu(int cpu);
bool pin_thread_to_numa(int numa_node);
bool pin_thread_unique(int numa_node, int local_rank, int thread_idx,
                       int threads_per_rank);
void cpu_relax();
int get_num_max_nvl_peers();

void maybe_enable_peer_access(int src_dev, int dst_dev);

uint64_t make_wr_id(uint32_t tag, uint32_t slot);
uint32_t wr_tag(uint64_t wrid);
uint32_t wr_slot(uint64_t wrid);

extern thread_local std::atomic<size_t> current_inflight_bytes;

// C++11 guarantees that a static local variable's initializer runs exactly
// once, even under concurrent access — the compiler emits a guard variable and
// lock. This eliminates both the data race and the redundant getenv calls.
static inline size_t get_max_inflight_bytes() {
  static size_t val = []() -> size_t {
    char const* env = getenv("UCCL_IB_MAX_INFLIGHT_BYTES");
    return env ? static_cast<size_t>(atoi(env)) : kMaxInflightBytes;
  }();
  return val;
}

static inline uint32_t get_max_inflight_low_latency() {
  static uint32_t val = []() -> uint32_t {
    char const* env = getenv("UCCL_IB_MAX_INFLIGHT_LOW_LATENCY");
    return env ? static_cast<uint32_t>(atoi(env)) : kMaxInflightLowLatency;
  }();
  return val;
}

static inline uint32_t get_max_inflight_normal() {
  static uint32_t val = []() -> uint32_t {
    char const* env = getenv("UCCL_IB_MAX_INFLIGHT_NORMAL");
    return env ? static_cast<uint32_t>(atoi(env)) : kMaxInflightNormal;
  }();
  return val;
}

#endif  // COMMON_HPP
