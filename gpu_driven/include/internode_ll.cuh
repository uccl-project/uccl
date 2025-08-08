#pragma once

#include <cstddef>
#include <cstdint>  // int64_t
#include <vector>
#include <cuda_runtime_api.h>
// #include <torch/extension.h>

namespace uccl {
namespace internode_ll {

// Dummy host launcher declaration
void dispatch(void* packed_recv_x, void* packed_recv_x_scales,
              int* packed_recv_src_info, int64_t* packed_recv_layout_range,
              int* packed_recv_count, int* cumulative_local_expert_recv_stats,
              int64_t* dispatch_wait_recv_cost_stats, void* rdma_recv_x,
              int* rdma_recv_count, void* rdma_x, void const* x,
              int64_t const* topk_idx, int* next_clean, int num_next_clean_int,
              int num_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
              int num_topk, int num_experts, int rank, int num_ranks,
              bool use_fp8, bool round_scale, bool use_ue8m0, void* workspace,
              int num_device_sms, cudaStream_t stream, int phases);

void combine(void* combined_x, void* rdma_recv_x, int* rdma_recv_flag,
             void* rdma_send_x, void const* x, int64_t const* topk_idx,
             float const* topk_weights, int const* src_info,
             int64_t const* layout_range, int64_t* combine_wait_recv_cost_stats,
             int* next_clean, int num_next_clean_int, int num_combined_tokens,
             int hidden, int num_max_dispatch_tokens_per_rank, int num_topk,
             int num_experts, int rank, int num_ranks, bool use_logfmt,
             void* workspace, int num_device_sms, cudaStream_t stream,
             int phases, bool zero_copy);

int init(std::vector<uint8_t> const& root_unique_id_val, int rank,
         int num_ranks, bool low_latency_mode);

void* alloc(size_t size, size_t alignment);
void barrier();
}  // namespace internode_ll
}  // namespace uccl