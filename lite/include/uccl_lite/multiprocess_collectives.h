#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Single-node, one-GPU-per-process BF16 collectives. The identifier must be
// identical at every rank and unique for each communicator lifetime.
void* uccl_lite_mp_create(char const* identifier, int ranks, int rank,
                          size_t max_bytes);
// Dedicated 8-rank, 64-KiB GPU-packet RS handle. No CPU packet worker or
// unrelated collective backend is prepared.
void* uccl_lite_mp_create_rs_ready(char const* identifier, int ranks, int rank);
// operation: 1=AllGather, 2=ReduceScatter(SUM), 3=AllReduce(SUM). count is
// the per-rank input element count for AG, per-rank output element count for
// RS, and input/output element count for AR.
// Enqueues on the caller's CUDA stream and returns before completion.
int uccl_lite_mp_enqueue(void* handle, int operation, void const* input,
                         void* output, size_t count, void* stream);
// Integration-only variant: the caller must serialize all submissions and
// insert CUDA event dependencies when changing streams. Preserves the strict
// single-stream contract of uccl_lite_mp_enqueue for existing callers.
int uccl_lite_mp_enqueue_ordered(void* handle, int operation, void const* input,
                                 void* output, size_t count, void* stream);
// Small fixed-shape BF16 AG+GEMM: local A[1,4096], gathered A[8,4096],
// weight B[128,4096], output C[8,128]. Uses the packet AG communicator.
int uccl_lite_mp_enqueue_ag_gemm_128x4096(void* handle, void const* input,
                                          void* gathered, void const* weight,
                                          void* gemm_output, void* stream);
// Small GPU-packet RS only: upload the BF16-rounded sum of two independent
// producer tensors without materializing their sum. Returns -1 if this
// communicator/size uses any other RS backend.
int uccl_lite_mp_enqueue_rs_sum2(void* handle, void const* input,
                                 void const* auxiliary, void* output,
                                 size_t shard_count, void* stream);
// Dedicated-stream GPU packet RS for exactly eight 4096-element BF16 rows.
// The producer sets ready[d]=1 with system-scope release after row d is valid.
int uccl_lite_mp_enqueue_rs_ready_sum2(void* handle, void const* input,
                                       void const* auxiliary, void* output,
                                       uint32_t const* ready, void* stream);
int uccl_lite_mp_destroy(void* handle);
char const* uccl_lite_mp_last_error(void);

#ifdef __cplusplus
}
#endif
