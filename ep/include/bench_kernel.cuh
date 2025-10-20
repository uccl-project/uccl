#ifndef BENCH_KERNEL_CUH
#define BENCH_KERNEL_CUH

#include "common.hpp"
#include "fifo_device.hpp"
#include "ring_buffer.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

// Ring buffer based kernel
__global__ void gpu_issue_batched_commands(DeviceToHostCmdBuffer* rbs);

// Launcher function for ring buffer-based kernel
cudaError_t launch_gpu_issue_batched_commands_shim(int blocks,
                                                   int threads_per_block,
                                                   size_t shmem_bytes,
                                                   cudaStream_t stream,
                                                   DeviceToHostCmdBuffer* rbs);

// FIFO based kernel
__global__ void gpu_issue_batched_commands_fifo(
    mscclpp::FifoDeviceHandle* fifos, uint64_t* cycle_start_out,
    uint64_t* cycle_end_out, uint64_t* cycle_accum_out, uint32_t* op_count_out);

// Launcher function for FIFO-based kernel
cudaError_t launch_gpu_issue_batched_commands_fifo(
    int blocks, int threads_per_block, size_t shmem_bytes, cudaStream_t stream,
    mscclpp::FifoDeviceHandle* d_fifos, uint64_t* cycle_start = nullptr,
    uint64_t* cycle_end = nullptr, uint64_t* cycle_accum = nullptr,
    uint32_t* op_count = nullptr);

#endif  // BENCH_KERNEL_CUH

