#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <nccl.h>

extern "C" ncclResult_t uccl_lite_nccl_reduce_scatter_sum2(
    const void* input, const void* auxiliary, void* output, size_t shard_count,
    ncclComm_t comm, cudaStream_t stream);
extern "C" ncclResult_t uccl_lite_nccl_all_gather_gemm_128x4096(
    const void* input, void* gathered, const void* weight, void* gemm_output,
    ncclComm_t comm, cudaStream_t stream);
extern "C" ncclResult_t uccl_lite_nccl_prepare_rs_ready(ncclComm_t comm);
extern "C" ncclResult_t uccl_lite_nccl_reduce_scatter_ready_sum2(
    const void* input, const void* auxiliary, void* output,
    const uint32_t* ready, ncclComm_t comm, cudaStream_t stream);
