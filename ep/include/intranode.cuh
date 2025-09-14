#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace uccl {
namespace intranode {

void notify_dispatch(int const* num_tokens_per_rank,
                     int* moe_recv_counter_mapped, int num_ranks,
                     int const* num_tokens_per_expert,
                     int* moe_recv_expert_counter_mapped, int num_experts,
                     int num_tokens, bool const* is_token_in_rank,
                     int* channel_prefix_matrix, int* rank_prefix_matrix_copy,
                     int num_memset_int, int expert_alignment,
                     void** buffer_ptrs, int** barrier_signal_ptrs, int rank,
                     cudaStream_t stream, int num_channels);

void cached_notify_dispatch(int const* rank_prefix_matrix, int num_memset_int,
                            void** buffer_ptrs, int** barrier_signal_ptrs,
                            int rank, int num_ranks, cudaStream_t stream);

void dispatch(void* recv_x, float* recv_x_scales, int* recv_src_idx,
              int64_t* recv_topk_idx, float* recv_topk_weights,
              int* recv_channel_offset, int* send_head, void const* x,
              float const* x_scales, int64_t const* topk_idx,
              float const* topk_weights, bool const* is_token_in_rank,
              int const* channel_prefix_matrix, int num_tokens,
              int num_worst_tokens, int hidden_int4, int num_topk,
              int num_experts, int num_scales, int scale_token_stride,
              int scale_hidden_stride, void** buffer_ptrs, int rank,
              int num_ranks, cudaStream_t stream, int num_sms,
              int num_max_send_tokens, int num_recv_buffer_tokens);

void cached_notify_combine(void** buffer_ptrs, int* send_head, int num_channels,
                           int num_recv_tokens, int num_memset_int,
                           int** barrier_signal_ptrs, int rank, int num_ranks,
                           cudaStream_t stream);

void combine(cudaDataType_t type, void* recv_x, float* recv_topk_weights,
             void const* x, float const* topk_weights, void const* bias_0,
             void const* bias_1, int const* src_idx,
             int const* rank_prefix_matrix, int const* channel_prefix_matrix,
             int* send_head, int num_tokens, int num_recv_tokens, int hidden,
             int num_topk, void** buffer_ptrs, int rank, int num_ranks,
             cudaStream_t stream, int num_sms, int num_max_send_tokens,
             int num_recv_buffer_tokens);

}  // namespace intranode
}  // namespace uccl