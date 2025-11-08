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

#define MAX_IB_DEVS 32
#define MEASURE_PER_OP_LATENCY
#define MEASURE_PER_VERB_LATENCY

// Barrier type selection (can be overridden at compile time)
#ifndef USE_SENDER_BARRIER
#ifdef EFA
#define USE_RECEIVER_BARRIER
#endif
#endif

#define USE_MSCCLPP_FIFO_BACKEND
// #define USE_SUBSET_BARRIER
#define kAtomicBufferSize 81960
#define kQueueSize 2048
#define kQueueMask (kQueueSize - 1)
// This is the highest we can get due to the number of bits we allocate in the
// imm for reordering buffer sequence tracking.
#define kMaxInflightLowLatency 32
#define kMaxInflightNormal 8
#define kBatchSize 32
#define kIterations 40000
#define kNumThBlocks 4
#define kNumThPerBlock 1
#define kObjectSize 7168  // 7 KB
// #define kObjectSize 10752  // 10.5 KB
// #define kObjectSize 14336  // 14 KB
#define kMaxOutstandingSends 2048
#define kMaxOutstandingRecvs 2048 * 2
#define kSenderAckQueueDepth 2048 * 2
#define kWarmupOps 10000
#define kChannelPerProxy 8
// TODO(MaoZiming): I tried to fit more bits, but this eats into offset and
// values.
#define kReorderingBufferSize 16  // Right now only 4 bits.
#define kRemoteBufferSize (kBatchSize * kNumThBlocks * kObjectSize * 100)
#define MAIN_THREAD_CPU_IDX 31
#define MAX_NUM_GPUS 8
#define RECEIVER_BATCH_SIZE 16
#define kAtomicWrTag 0xa70a000000000000ULL
#define kAtomicMask 0x0000FFFFFFFFFFFFULL
#define kBarrierWrTag 0xbaba000000000000ULL
#define kBarrierMask 0x0000FFFFFFFFFFFFULL
#define kPrintCycleInterval 100000000000ULL
// Base TCP port for Proxy barrier rendezvous (rank0 server)
#define TCP_PORT 18515
#define MAX_RETRIES 100
#define RETRY_DELAY_MS 50
#define QKEY 0x11111111u
// #define kLargeAtomicValue 33554352
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

#endif  // COMMON_HPP
