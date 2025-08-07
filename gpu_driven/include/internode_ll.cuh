#pragma once

#include <cstdint>  // int64_t
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

}  // namespace internode_ll
}  // namespace uccl