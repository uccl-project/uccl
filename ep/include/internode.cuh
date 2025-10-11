#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <cuda_runtime_api.h>

namespace uccl {
namespace internode {

// extern nvshmem_team_t cpu_rdma_team;

struct SourceMeta;

int get_source_meta_bytes();

__host__ __device__ __forceinline__ int get_num_bytes_per_token(
    int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights);

__host__ __device__ __forceinline__ std::pair<int, int> get_rdma_clean_meta(
    int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights,
    int num_rdma_ranks, int num_rdma_recv_buffer_tokens, int num_channels);

__host__ __device__ __forceinline__ std::pair<int, int> get_nvl_clean_meta(
    int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights,
    int num_rdma_ranks, int num_nvl_ranks, int num_nvl_recv_buffer_tokens,
    int num_channels, bool is_dispatch);

void notify_dispatch(
    int const* num_tokens_per_rank, int* moe_recv_counter_mapped, int num_ranks,
    int const* num_tokens_per_rdma_rank, int* moe_recv_rdma_counter_mapped,
    int const* num_tokens_per_expert, int* moe_recv_expert_counter_mapped,
    int num_experts, bool const* is_token_in_rank, int num_tokens,
    int num_channels, int hidden_int4, int num_scales, int num_topk,
    int expert_alignment, int* rdma_channel_prefix_matrix,
    int* recv_rdma_rank_prefix_sum, int* gbl_channel_prefix_matrix,
    int* recv_gbl_rank_prefix_sum, void* rdma_buffer_ptr,
    int num_max_rdma_chunked_recv_tokens, void** buffer_ptrs,
    int num_max_nvl_chunked_recv_tokens, int** barrier_signal_ptrs, int rank,
    cudaStream_t stream, int64_t num_rdma_bytes, int64_t num_nvl_bytes,
    bool low_latency_mode, uint64_t const* ring_addrs, int num_ring_addrs,
    void* atomic_buffer_ptr);

void cached_notify(int hidden_int4, int num_scales, int num_topk_idx,
                   int num_topk_weights, int num_ranks, int num_channels,
                   int num_combined_tokens, int* combined_rdma_head,
                   int const* rdma_channel_prefix_matrix,
                   int const* rdma_rank_prefix_sum, int* combined_nvl_head,
                   void* rdma_buffer_ptr, int num_max_rdma_chunked_recv_tokens,
                   void** buffer_ptrs, int num_max_nvl_chunked_recv_tokens,
                   int** barrier_signal_ptrs, int rank, cudaStream_t stream,
                   int64_t num_rdma_bytes, int64_t num_nvl_bytes,
                   bool is_cached_dispatch, bool low_latency_mode,
                   uint64_t const* ring_addrs, int num_ring_addrs,
                   void* atomic_buffer_ptr);

void dispatch(void* recv_x, float* recv_x_scales, int64_t* recv_topk_idx,
              float* recv_topk_weights, void* recv_src_meta, void const* x,
              float const* x_scales, int64_t const* topk_idx,
              float const* topk_weights, int* send_rdma_head,
              int* send_nvl_head, int* recv_rdma_channel_prefix_matrix,
              int* recv_gbl_channel_prefix_matrix,
              int const* rdma_channel_prefix_matrix,
              int const* recv_rdma_rank_prefix_sum,
              int const* gbl_channel_prefix_matrix,
              int const* recv_gbl_rank_prefix_sum, bool const* is_token_in_rank,
              int num_tokens, int hidden_int4, int num_scales, int num_topk,
              int num_experts, int scale_token_stride, int scale_hidden_stride,
              void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens,
              int num_max_rdma_chunked_recv_tokens, void** buffer_ptrs,
              int num_max_nvl_chunked_send_tokens,
              int num_max_nvl_chunked_recv_tokens, int rank, int num_ranks,
              bool is_cached_dispatch, cudaStream_t stream, int num_channels,
              bool low_latency_mode, uint64_t const* ring_addrs,
              int num_ring_addrs, void* atomic_buffer_ptr);

void combine(cudaDataType_t type, void* combined_x,
             float* combined_topk_weights,
             bool const* is_combined_token_in_rank, void const* x,
             float const* topk_weights, void const* bias_0, void const* bias_1,
             int const* combined_rdma_head, int const* combined_nvl_head,
             void const* src_meta, int const* rdma_channel_prefix_matrix,
             int const* rdma_rank_prefix_sum,
             int const* gbl_channel_prefix_matrix, int num_tokens,
             int num_combined_tokens, int hidden, int num_topk,
             void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens,
             int num_max_rdma_chunked_recv_tokens, void** buffer_ptrs,
             int num_max_nvl_chunked_send_tokens,
             int num_max_nvl_chunked_recv_tokens, int rank, int num_ranks,
             cudaStream_t stream, int num_channels, bool low_latency_mode,
             uint64_t const* ring_addrs, int num_ring_addrs,
             void* atomic_buffer_ptr);

}  // namespace internode
}  // namespace uccl