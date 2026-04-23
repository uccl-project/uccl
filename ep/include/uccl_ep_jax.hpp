// ============================================================================
// JAX FFI bridge for UCCL-EP
// ----------------------------------------------------------------------------
// This header decouples the JAX FFI handler implementations (in
// ``ep/src/uccl_ep_jax.cc``) from the full ``uccl::Buffer`` class
// definition (in ``ep/src/uccl_ep.cc``).
//
// uccl_ep.cc populates ``g_buffer_bridge`` with 8 thin trampolines that
// forward to the corresponding ``Buffer`` methods. uccl_ep_jax.cc
// treats ``Buffer*`` as an opaque pointer (``void*``) and dispatches
// through the bridge, so the two translation units share only this
// small plain-C interface.
// ============================================================================

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <nanobind/nanobind.h>

namespace uccl_jax_ffi {

namespace nb = nanobind;

// Opaque handle for ``uccl::Buffer*``. The real type lives inside
// uccl_ep.cc's anonymous namespace; we interact with it only through
// ``BufferBridge``.
using OpaqueBuffer = void*;

// All 8 ``Buffer`` methods we need are funneled through this struct of
// function pointers, populated once by uccl_ep.cc at module init.
struct BufferBridge {
  void (*low_latency_dispatch)(
      OpaqueBuffer self,
      std::uintptr_t x_ptr, int x_rows, int x_cols,
      std::uintptr_t topk_idx_ptr, int topk_rows, int topk_cols,
      std::uintptr_t packed_recv_x_ptr,
      std::uintptr_t packed_recv_x_scales_ptr,
      std::uintptr_t packed_recv_count_ptr,
      std::uintptr_t packed_recv_src_info_ptr,
      std::uintptr_t packed_recv_layout_range_ptr,
      std::uintptr_t cumulative_local_expert_recv_stats_ptr,
      std::uintptr_t dispatch_wait_recv_cost_stats_ptr,
      std::uintptr_t compute_stream_ptr,
      int num_max_dispatch_tokens_per_rank, int num_experts,
      bool use_fp8, bool round_scale, bool use_ue8m0,
      bool is_async, bool return_recv_hook);

  void (*low_latency_combine)(
      OpaqueBuffer self,
      std::uintptr_t x_ptr, int x_dim0, int x_dim1, int x_dim2,
      std::uintptr_t topk_idx_ptr, int topk_rows, int topk_cols,
      std::uintptr_t topk_weights_ptr,
      std::uintptr_t src_info_ptr, int src_info_dim0, int src_info_dim1,
      std::uintptr_t layout_range_ptr,
      int layout_range_dim0, int layout_range_dim1,
      std::uintptr_t combine_wait_recv_cost_stats_ptr,
      std::uintptr_t compute_stream_ptr,
      int num_max_dispatch_tokens_per_rank, int num_experts,
      bool use_logfmt, bool zero_copy, bool is_async,
      bool return_recv_hook, std::uintptr_t out_ptr);

  void (*get_dispatch_layout)(
      OpaqueBuffer self,
      std::uintptr_t topk_idx_ptr, int num_tokens, int num_topk,
      int num_experts, std::uintptr_t num_tokens_per_rank_ptr,
      std::uintptr_t num_tokens_per_rdma_rank_ptr,
      std::uintptr_t num_tokens_per_expert_ptr,
      std::uintptr_t is_token_in_rank_ptr,
      bool async, bool allocate_on_comm_stream,
      std::uintptr_t compute_stream_ptr);

  // ``intranode_prepare`` returns (num_recv_tokens, per-expert list,
  // event) in the Python binding; the FFI handlers don't need those
  // return values, so the bridge returns void.
  void (*intranode_prepare)(
      OpaqueBuffer self,
      std::uintptr_t num_tokens_per_rank_ptr,
      std::uintptr_t is_token_in_rank_ptr,
      std::uintptr_t num_tokens_per_expert_ptr, int num_tokens,
      int num_experts, std::uintptr_t rank_prefix_matrix_ptr,
      std::uintptr_t channel_prefix_matrix_ptr, int expert_alignment,
      int num_worst_tokens,
      // Config fields, inlined so we don't leak uccl::Config:
      int cfg_num_sms, int cfg_nvl_send, int cfg_nvl_recv,
      int cfg_rdma_send, int cfg_rdma_recv,
      bool async, bool allocate_on_comm_stream,
      std::uintptr_t compute_stream_ptr);

  void (*intranode_dispatch)(
      OpaqueBuffer self,
      std::uintptr_t x_ptr, int num_tokens, int hidden, int x_element_size,
      std::uintptr_t x_scales_ptr, int num_scales, int scale_token_stride,
      int scale_hidden_stride, std::uintptr_t topk_idx_ptr, int num_topk,
      std::uintptr_t topk_weights_ptr, std::uintptr_t is_token_in_rank_ptr,
      std::uintptr_t rank_prefix_matrix_ptr,
      std::uintptr_t channel_prefix_matrix_ptr, int num_experts,
      int num_worst_tokens, bool cached_mode,
      int cfg_num_sms, int cfg_nvl_send, int cfg_nvl_recv,
      int cfg_rdma_send, int cfg_rdma_recv,
      int num_recv_tokens, std::uintptr_t recv_x_ptr,
      std::uintptr_t recv_x_scales_ptr, std::uintptr_t recv_topk_idx_ptr,
      std::uintptr_t recv_topk_weights_ptr,
      std::uintptr_t recv_channel_prefix_matrix_ptr,
      std::uintptr_t recv_src_idx_ptr, std::uintptr_t send_head_ptr,
      bool async, bool allocate_on_comm_stream,
      std::uintptr_t compute_stream_ptr);

  void (*intranode_combine)(
      OpaqueBuffer self,
      std::uintptr_t x_ptr, int num_tokens, int hidden, int x_dtype_code,
      int x_element_size, std::uintptr_t topk_weights_ptr, int num_topk,
      std::uintptr_t bias_0_ptr, std::uintptr_t bias_1_ptr,
      std::uintptr_t src_idx_ptr, int num_recv_tokens,
      std::uintptr_t rank_prefix_matrix_ptr,
      std::uintptr_t channel_prefix_matrix_ptr,
      std::uintptr_t send_head_ptr,
      int cfg_num_sms, int cfg_nvl_send, int cfg_nvl_recv,
      int cfg_rdma_send, int cfg_rdma_recv,
      std::uintptr_t recv_x_ptr, std::uintptr_t recv_topk_weights_ptr,
      bool async, bool allocate_on_comm_stream,
      std::uintptr_t compute_stream_ptr);

  void (*internode_prepare)(
      OpaqueBuffer self,
      std::uintptr_t num_tokens_per_rank_ptr,
      std::uintptr_t num_tokens_per_rdma_rank_ptr,
      std::uintptr_t num_tokens_per_expert_ptr,
      std::uintptr_t is_token_in_rank_ptr, int num_tokens,
      int hidden, int x_element_size, int num_scales,
      int num_topk, int num_experts, int expert_alignment,
      int num_worst_tokens,
      int cfg_num_sms, int cfg_nvl_send, int cfg_nvl_recv,
      int cfg_rdma_send, int cfg_rdma_recv,
      std::uintptr_t rdma_channel_prefix_matrix_ptr,
      std::uintptr_t recv_rdma_rank_prefix_sum_ptr,
      std::uintptr_t gbl_channel_prefix_matrix_ptr,
      std::uintptr_t recv_gbl_rank_prefix_sum_ptr,
      bool async, bool allocate_on_comm_stream,
      std::uintptr_t compute_stream_ptr);

  void (*internode_dispatch)(
      OpaqueBuffer self,
      std::uintptr_t x_ptr, int num_tokens, int hidden, int x_element_size,
      std::uintptr_t x_scales_ptr, int num_scales, int scale_token_stride,
      int scale_hidden_stride, std::uintptr_t topk_idx_ptr, int num_topk,
      std::uintptr_t topk_weights_ptr,
      std::uintptr_t is_token_in_rank_ptr,
      std::uintptr_t rdma_channel_prefix_matrix_ptr,
      std::uintptr_t recv_rdma_rank_prefix_sum_ptr,
      std::uintptr_t gbl_channel_prefix_matrix_ptr,
      std::uintptr_t recv_gbl_rank_prefix_sum_ptr, int num_experts,
      int num_worst_tokens, bool cached_mode, int num_rdma_recv_tokens,
      int cfg_num_sms, int cfg_nvl_send, int cfg_nvl_recv,
      int cfg_rdma_send, int cfg_rdma_recv,
      std::uintptr_t recv_x_ptr,
      std::uintptr_t recv_x_scales_ptr, std::uintptr_t recv_topk_idx_ptr,
      std::uintptr_t recv_topk_weights_ptr,
      std::uintptr_t recv_src_meta_ptr,
      std::uintptr_t recv_rdma_channel_prefix_matrix_ptr,
      std::uintptr_t recv_gbl_channel_prefix_matrix_ptr,
      std::uintptr_t send_rdma_head_ptr,
      std::uintptr_t send_nvl_head_ptr, bool async,
      bool allocate_on_comm_stream,
      std::uintptr_t compute_stream_ptr);

  void (*internode_combine)(
      OpaqueBuffer self,
      std::uintptr_t x_ptr, int num_tokens, int hidden, int x_dtype_code,
      int x_element_size, std::uintptr_t topk_weights_ptr, int num_topk,
      std::uintptr_t bias_0_ptr, std::uintptr_t bias_1_ptr,
      std::uintptr_t src_meta_ptr, int num_combined_tokens,
      std::uintptr_t is_combined_token_in_rank_ptr,
      std::uintptr_t rdma_channel_prefix_matrix_ptr,
      std::uintptr_t rdma_rank_prefix_sum_ptr,
      std::uintptr_t gbl_channel_prefix_matrix_ptr,
      std::uintptr_t combined_rdma_head_ptr,
      std::uintptr_t combined_nvl_head_ptr,
      int cfg_num_sms, int cfg_nvl_send, int cfg_nvl_recv,
      int cfg_rdma_send, int cfg_rdma_recv,
      std::uintptr_t combined_x_ptr,
      std::uintptr_t combined_topk_weights_ptr,
      bool async, bool allocate_on_comm_stream,
      std::uintptr_t compute_stream_ptr);
};

// Populated once by uccl_ep.cc (via ``install_buffer_bridge``); read
// only from inside the FFI handlers.
BufferBridge const& get_buffer_bridge();
void install_buffer_bridge(BufferBridge const& bridge);

// Per-device ``Buffer*`` registry used by the FFI handlers.
OpaqueBuffer get_buffer_for_device(int device_index);
void register_buffer_for_device(int device_index, OpaqueBuffer buffer);
void unregister_buffer_for_device(int device_index);

// The 8 XLA legacy custom-call handlers.
using XlaCustomCallStatus = void;
extern "C" {

void uccl_moe_low_latency_dispatch_ffi(
    cudaStream_t stream, void** buffers, char const* opaque, size_t opaque_len,
    XlaCustomCallStatus* status);
void uccl_moe_low_latency_combine_ffi(
    cudaStream_t stream, void** buffers, char const* opaque, size_t opaque_len,
    XlaCustomCallStatus* status);
void uccl_moe_dispatch_ffi(
    cudaStream_t stream, void** buffers, char const* opaque, size_t opaque_len,
    XlaCustomCallStatus* status);
void uccl_moe_combine_ffi(
    cudaStream_t stream, void** buffers, char const* opaque, size_t opaque_len,
    XlaCustomCallStatus* status);
void uccl_moe_cached_dispatch_ffi(
    cudaStream_t stream, void** buffers, char const* opaque, size_t opaque_len,
    XlaCustomCallStatus* status);
void uccl_moe_internode_dispatch_ffi(
    cudaStream_t stream, void** buffers, char const* opaque, size_t opaque_len,
    XlaCustomCallStatus* status);
void uccl_moe_internode_cached_dispatch_ffi(
    cudaStream_t stream, void** buffers, char const* opaque, size_t opaque_len,
    XlaCustomCallStatus* status);
void uccl_moe_internode_combine_ffi(
    cudaStream_t stream, void** buffers, char const* opaque, size_t opaque_len,
    XlaCustomCallStatus* status);

}  // extern "C"

// Attach the nanobind pieces (``register_jax_ffi_buffer``,
// ``unregister_jax_ffi_buffer``, ``get_jax_ffi_targets``) to the
// ``uccl.ep`` module. Called from uccl_ep.cc's ``NB_MODULE`` body.
void register_jax_bindings(nb::module_& m);

}  // namespace uccl_jax_ffi
