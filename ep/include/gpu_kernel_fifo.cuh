#pragma once
#include "fifo_device.hpp"
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void gpu_issue_batched_commands_fifo(
    mscclpp::FifoDeviceHandle* fifos, uint64_t* cycle_start_out,
    uint64_t* cycle_end_out, uint64_t* cycle_accum_out, uint32_t* op_count_out);

// Launcher function - declared here, implemented in .cu file
cudaError_t launch_gpu_issue_batched_commands_fifo(
    int blocks, int threads_per_block, size_t shmem_bytes, cudaStream_t stream,
    mscclpp::FifoDeviceHandle* d_fifos, uint64_t* cycle_start = nullptr,
    uint64_t* cycle_end = nullptr, uint64_t* cycle_accum = nullptr,
    uint32_t* op_count = nullptr);
