#pragma once

#include <cuda_runtime.h>
#include <driver_types.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// An A100 SM can have at most 32 thread blocks.
static const uint32_t THREAD_BLOCKS = 4;
// NCCL uses 512 for current gen, and 256 for previous gen.
static const uint32_t THREADS_PER_BLOCK = 512;
// Supporting 128 * 9KB = 1.152MB net chunk size.
static const uint32_t MAX_COPIES = 128;

typedef struct {
  uint64_t dst[MAX_COPIES];
  uint64_t src[MAX_COPIES];
  uint32_t len[MAX_COPIES];
} copy_param_t;

static const uint32_t param_size = sizeof(copy_param_t);
// https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/:
// CUDA 12.1 supports upto 32764 as kernel parameters.
static_assert(param_size <= 32764, "param_size must be <= 32764");

// Launch wrapper (exposed with C linkage) that is callable from a .cc file.
void launchScatteredMemcpy(uint32_t num_copies, copy_param_t const* p);
// 1 vs 128 MAX_COPIES -> 2.6 vs 3.6 us async launch time.
void launchScatteredMemcpyAsync(uint32_t num_copies, copy_param_t const* params,
                                cudaStream_t stream);
// 0.56 us for pollScatteredMemcpy.
int pollScatteredMemcpy(cudaStream_t stream);

void launchPersistentScatteredMemcpy(int iterations, cudaStream_t streamk);

#ifdef __cplusplus
}
#endif
