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

#define MEASURE_PER_OP_LATENCY
#define ENABLE_WRITE_WITH_IMMEDIATE
#define ASSUME_WR_IN_ORDER
#define NUMA_AWARE_SCHEDULING
#define ENABLE_PROXY_CUDA_MEMCPY
#define kQueueSize 1024
#define kQueueMask (kQueueSize - 1)
#define kMaxInflight 32
#define kBatchSize 16
#define kIterations 1000000
#define kNumThBlocks 4
#define kNumThPerBlock 1
#define kRemoteNVLinkBatchSize 512
#define kObjectSize 8192  // 8 KB
#define kMaxOutstandingSends 1024
#define kMaxOutstandingRecvs 1024
#define kSignalledEvery 1
#define kNumPollingThreads 0  // Rely on CPU proxy to poll.
#define kPollingThreadStartPort kNumThBlocks * 2
#define kWarmupOps 10000
#define kRemoteBufferSize kBatchSize* kNumThBlocks* kObjectSize * 100
#define MAIN_THREAD_CPU_IDX 31
#define NUM_GPUS 8
#define RECEIVER_BATCH_SIZE 16
#define NVLINK_SM_PER_PROCESS 2
// #define SEPARATE_POLLING
// Command structure for each transfer
struct TransferCmd {
  uint64_t cmd;
  uint32_t dst_rank;  // remote node id (MPI-style)
  uint32_t dst_gpu;   // GPU id on remote node
  void* src_ptr;      // device pointer to data
  uint64_t bytes;     // transfer size
};

bool pin_thread_to_cpu(int cpu);

#endif  // COMMON_HPP