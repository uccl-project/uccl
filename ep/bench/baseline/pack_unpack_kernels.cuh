#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

namespace moe_pack_unpack {

// Pack MoE data into per-rank buffers on GPU
// Args:
//   x: (num_tokens, hidden_dim) - input tokens
//   topk_idx: (num_tokens, experts_per_token) - expert indices
//   topk_weights: (num_tokens, experts_per_token) - routing weights
//   buffers: List of world_size output buffers (uint8)
//   per_rank_offsets: (world_size,) - current write offset for each rank buffer
//   num_experts: total number of experts
//   num_local_experts: experts per rank
//   world_size: number of ranks
//
// Returns:
//   per_rank_bytes: (world_size,) - total bytes written to each rank
void pack_moe_data_cuda(
    const torch::Tensor& x,                    // (num_tokens, hidden_dim)
    const torch::Tensor& topk_idx,             // (num_tokens, experts_per_token)
    const torch::Tensor& topk_weights,         // (num_tokens, experts_per_token)
    const std::vector<torch::Tensor>& buffers, // world_size buffers
    int num_experts,
    int num_local_experts,
    int world_size,
    torch::Tensor& per_rank_bytes              // (world_size,) output
);

// Unpack MoE data from per-rank buffers on GPU
// Args:
//   buffers: List of world_size input buffers (uint8)
//   per_rank_recv_bytes: (world_size,) - bytes to read from each rank buffer
//   num_local_experts: experts on this rank
//   hidden_dim: hidden dimension
//   world_size: number of ranks
//   x_dtype: data type for x
//   idx_dtype: data type for indices
//   weight_dtype: data type for weights
//
// Returns:
//   tuple of (recv_x, recv_topk_idx, recv_topk_weights, recv_expert_counts)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
unpack_moe_data_cuda(
    const std::vector<torch::Tensor>& buffers,   // world_size buffers
    const torch::Tensor& per_rank_recv_bytes,    // (world_size,)
    int num_local_experts,
    int hidden_dim,
    int world_size,
    torch::ScalarType x_dtype,
    torch::ScalarType idx_dtype,
    torch::ScalarType weight_dtype
);

} // namespace moe_pack_unpack
