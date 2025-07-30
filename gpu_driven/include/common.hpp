#ifndef COMMON_HPP
#define COMMON_HPP

#include "util/cuda.h"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>

// #define REMOTE_PERSISTENT_KERNEL
#define USE_GRACE_HOPPER
#define MEASURE_PER_OP_LATENCY
#define ASSUME_WR_IN_ORDER
#define ENABLE_PROXY_CUDA_MEMCPY
#define SYNCHRONOUS_COMPLETION
#define RDMA_BATCH_TOKENS
#define kQueueSize 1024
#define kQueueMask (kQueueSize - 1)
#define kMaxInflight 64
#define kBatchSize 32
#define kIterations 1000000
#define kNumThBlocks 6
#define kNumThPerBlock 1
#ifdef SYNCHRONOUS_COMPLETION
#define kRemoteNVLinkBatchSize \
  16  // Immediately synchronize stream for latency.
#else
#define kRemoteNVLinkBatchSize 512
#endif
#define kObjectSize 10752  // 10.5 KB
#define kMaxOutstandingSends 2048
#define kMaxOutstandingRecvs 2048
#define kSignalledEvery 1
#define kSenderAckQueueDepth 1024
#define kNumPollingThreads 0  // Rely on CPU proxy to poll.
#define kPollingThreadStartPort kNumThBlocks * 2
#define kWarmupOps 10000
#define kRemoteBufferSize kBatchSize* kNumThBlocks* kObjectSize * 100
#define MAIN_THREAD_CPU_IDX 31
#define NUM_GPUS 1
#define RECEIVER_BATCH_SIZE 16
#ifdef SYNCHRONOUS_COMPLETION
#define NVLINK_SM_PER_PROCESS \
  1  // Total number of SMs used is NVLINK_SM_PER_PROCESS * kNumThBlocks
#else
#define NVLINK_SM_PER_PROCESS 2
#endif

bool pin_thread_to_cpu(int cpu);
void cpu_relax();

#endif  // COMMON_HPP