#pragma once

#include <cstddef>
#include <cstdint>  // int64_t
#include <vector>
#include <cuda_runtime_api.h>
// #include <torch/extension.h>

namespace uccl {
namespace internode_ll {
void clean_low_latency_buffer(int* clean_0, int num_clean_int_0, int* clean_1,
                              int num_clean_int_1, cudaStream_t stream);
void clean_low_latency_buffer_internode(int64_t* clean_0, int num_clean_int_0,
                                        int64_t* clean_1, int num_clean_int_1,
                                        cudaStream_t stream);
// Dummy host launcher declaration
void dispatch(void* packed_recv_x, void* packed_recv_x_scales,
              int* packed_recv_src_info, int64_t* packed_recv_layout_range,
              int* packed_recv_count, int* cumulative_local_expert_recv_stats,
              int64_t* dispatch_wait_recv_cost_stats, void* rdma_recv_x,
              int* rdma_recv_count, void* rdma_x, void const* x,
              int64_t const* topk_idx, int* next_clean,
              int64_t* next_clean_second, int num_next_clean_int,
              int num_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
              int num_topk, int num_experts, int rank, int num_ranks,
              bool use_fp8, bool round_scale, bool use_ue8m0, void* workspace,
              int num_device_sms, cudaStream_t stream, int phases,
              uint64_t const* d2h_channel_addrs, int num_d2h_channel_addrs,
              int max_nvl_peers, int low_latency_buffer_idx,
              void** ipc_rdma_base_ptrs = nullptr,
              void* rdma_buffer_ptr = nullptr,
              void* atomic_buffer_ptr = nullptr,
              int64_t* rdma_recv_count_internode = nullptr);

void combine(void* combined_x, void* rdma_recv_x, int* rdma_recv_flag,
             void* rdma_send_x, void const* x, int64_t const* topk_idx,
             float const* topk_weights, int const* src_info,
             int64_t const* layout_range, int64_t* combine_wait_recv_cost_stats,
             int* next_clean, int64_t* next_clean_second,
             int num_next_clean_int, int num_combined_tokens, int hidden,
             int num_max_dispatch_tokens_per_rank, int num_topk,
             int num_experts, int rank, int num_ranks, bool use_logfmt,
             void* workspace, int num_device_sms, cudaStream_t stream,
             int phases, bool zero_copy, uint64_t const* d2h_channel_addrs,
             int num_d2h_channel_addrs, int max_nvl_peers,
             int low_latency_buffer_idx, void** ipc_rdma_base_ptrs = nullptr,
             void* rdma_buffer_ptr = nullptr, void* atomic_buffer_ptr = nullptr,
             int64_t* rdma_recv_flag_internode = nullptr);
}  // namespace internode_ll
}  // namespace uccl