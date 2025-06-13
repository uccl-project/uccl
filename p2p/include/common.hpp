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
#define ASSUME_WR_IN_ORDER
#define kQueueSize 32
#define kQueueMask (kQueueSize - 1)
#define kBatchSize 4
#define kIterations 100000
#define kNumThBlocks 8
#define kNumThPerBlock 1
#define kObjectSize 8192  // 8 KB
#define kMaxOutstandingSends kNumThBlocks* kBatchSize
#define kSignalledEvery 1
#define kNumPollingThreads 0  // Rely on CPU proxy to poll.
#define kPollingThreadStartPort kNumThBlocks * 2
#define kWarmupOps 10000
// #define SEPARATE_POLLING
// Command structure for each transfer
struct TransferCmd {
  uint64_t cmd;
  uint32_t dst_rank;  // remote node id (MPI-style)
  uint32_t dst_gpu;   // GPU id on remote node
  void* src_ptr;      // device pointer to data
  uint64_t bytes;     // transfer size
};

// Ring buffer queue size and mask (must be a power of 2)
constexpr uint32_t QUEUE_SIZE = 1024;
constexpr uint32_t QUEUE_MASK = QUEUE_SIZE - 1;

bool pin_thread_to_cpu(int cpu);

#endif  // COMMON_HPP