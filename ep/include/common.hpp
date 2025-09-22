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

#define MEASURE_PER_OP_LATENCY
#define MEASURE_PER_VERB_LATENCY
#define kAtomicBufferSize 8196
#define kQueueSize 1024
#define kQueueMask (kQueueSize - 1)
#define kMaxInflight 256
#define kBatchSize 32
#define kIterations 40000
#define kNumThBlocks 4
#define kNumThPerBlock 1
#define kObjectSize 10752  // 10.5 KB
#define kMaxOutstandingSends 2048
#define kMaxOutstandingRecvs 2048 * 2
#define kSenderAckQueueDepth 2048 * 2
#define kWarmupOps 10000
#define kRemoteBufferSize (kBatchSize * kNumThBlocks * kObjectSize * 100)
#define MAIN_THREAD_CPU_IDX 31
#define MAX_NUM_GPUS 8
#define RECEIVER_BATCH_SIZE 16
#define NVLINK_SM_PER_PROCESS 1
#define kAtomicWrTag 0xa70a000000000000ULL
#define kAtomicMask 0x0000FFFFFFFFFFFFULL
#define kPrintCycleInterval 1000000000ULL
// P2P enable flags (once per GPU pair)
extern std::once_flag peer_ok_flag[MAX_NUM_GPUS][MAX_NUM_GPUS];
bool pin_thread_to_cpu(int cpu);
void cpu_relax();
int get_num_max_nvl_peers();

void maybe_enable_peer_access(int src_dev, int dst_dev);

uint64_t make_wr_id(uint32_t tag, uint32_t slot);
uint32_t wr_tag(uint64_t wrid);
uint32_t wr_slot(uint64_t wrid);

#endif  // COMMON_HPP