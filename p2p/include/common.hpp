#ifndef COMMON_HPP
#define COMMON_HPP

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

// #define DEBUG_PRINT
// CUDA error checking macro
#define CHECK_CUDA(call)                                            \
  do {                                                              \
    cudaError_t _e = (call);                                        \
    if (_e != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(_e));                              \
      std::exit(EXIT_FAILURE);                                      \
    }                                                               \
  } while (0)

#define cudaCheckErrors(msg)                                  \
  do {                                                        \
    cudaError_t __err = cudaGetLastError();                   \
    if (__err != cudaSuccess) {                               \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, \
              cudaGetErrorString(__err), __FILE__, __LINE__); \
      fprintf(stderr, "*** FAILED - ABORTING\n");             \
      exit(1);                                                \
    }                                                         \
  } while (0)

// #define REMOTE_PERSISTENT_KERNEL
#define MEASURE_PER_OP_LATENCY
#define ENABLE_WRITE_WITH_IMMEDIATE
#define ASSUME_WR_IN_ORDER
#define NUMA_AWARE_SCHEDULING
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
#define NUM_GPUS 8
#define RECEIVER_BATCH_SIZE 16
#ifdef SYNCHRONOUS_COMPLETION
#define NVLINK_SM_PER_PROCESS \
  1  // Total number of SMs used is NVLINK_SM_PER_PROCESS * kNumThBlocks
#else
#define NVLINK_SM_PER_PROCESS 2
#endif
// #define SEPARATE_POLLING

bool pin_thread_to_cpu(int cpu);

#endif  // COMMON_HPP