// ============================================================================
// JAX FFI bridge for UCCL-EP -- translation unit #2
// ----------------------------------------------------------------------------
// This file contains every JAX-specific C++ component of the ``uccl.ep``
// module. It is compiled as a separate translation unit from
// ``uccl_ep.cc`` (which owns the ``NB_MODULE(ep, m)`` top-level
// definition and the out-of-line ``Buffer`` method bodies).
//
// ``Buffer`` is declared in ``ep/include/uccl_ep.hpp``; both TUs
// include it and the FFI handlers call ``Buffer`` methods directly.
//
// uccl_ep.cc reaches this file through one hook declared in
// ``ep/include/uccl_ep_jax.hpp``:
//
//   * ``uccl_jax_ffi::register_jax_bindings(m)``: called once during
//     ``NB_MODULE(ep, m)`` to install ``register_jax_ffi_buffer`` /
//     ``unregister_jax_ffi_buffer`` / ``get_jax_ffi_targets`` onto the
//     module.
// ============================================================================

#include "uccl_ep_jax.hpp"

#include <mutex>
#include <optional>
#include <unordered_map>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "uccl_ep.hpp"

namespace uccl_jax_ffi {

namespace nb = nanobind;

// ---------------------------------------------------------------------------
// Per-device Buffer registry
// ---------------------------------------------------------------------------

namespace {
std::mutex g_jax_ffi_mu;
std::unordered_map<int, Buffer*> g_jax_ffi_buffers;
}  // namespace

Buffer* get_buffer_for_device(int device_index) {
  std::lock_guard<std::mutex> lk(g_jax_ffi_mu);
  auto it = g_jax_ffi_buffers.find(device_index);
  if (it == g_jax_ffi_buffers.end()) return nullptr;
  return it->second;
}

void register_buffer_for_device(int device_index, Buffer* buffer) {
  std::lock_guard<std::mutex> lk(g_jax_ffi_mu);
  g_jax_ffi_buffers[device_index] = buffer;
}

void unregister_buffer_for_device(int device_index) {
  std::lock_guard<std::mutex> lk(g_jax_ffi_mu);
  g_jax_ffi_buffers.erase(device_index);
}

static Buffer* current_buffer_or_die() {
  int dev = 0;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) return nullptr;
  return get_buffer_for_device(dev);
}

// ============================================================================
// XLA custom-call handlers
// ----------------------------------------------------------------------------
// ABI: api_version=0 (legacy custom call)
//   void handler(cudaStream_t stream, void** buffers,
//                const char* opaque, size_t opaque_len,
//                XlaCustomCallStatus* status);
//
// The opaque buffer is a tightly-packed C struct; the Python side
// (see ``ep/jax_wrapper/uccl_ep_jax/primitive/_calls.py``) packs int32
// fields into the ``legacy_backend_config`` attribute whose layout
// must stay byte-identical to the matching struct below.
// ============================================================================

// ---------------------------------------------------------------------------
// moe_low_latency_dispatch
// ---------------------------------------------------------------------------
struct MoELowLatencyDispatchOpaque {
  int32_t num_tokens;
  int32_t hidden;
  int32_t num_topk;
  int32_t num_max_dispatch_tokens_per_rank;
  int32_t num_experts;
  int32_t use_fp8;
  int32_t round_scale;
  int32_t use_ue8m0;
};

extern "C" void uccl_moe_low_latency_dispatch_ffi(
    cudaStream_t stream, void** buffers, char const* opaque, size_t opaque_len,
    XlaCustomCallStatus* /*status*/) {
  if (opaque_len != sizeof(MoELowLatencyDispatchOpaque)) return;
  auto const& p = *reinterpret_cast<MoELowLatencyDispatchOpaque const*>(opaque);
  auto* buf = current_buffer_or_die();
  if (!buf) return;

  void* x = buffers[0];
  void* topk_idx = buffers[1];
  void* recv_x = buffers[2];
  void* recv_count = buffers[3];
  void* recv_src_info = buffers[4];
  void* recv_layout_range = buffers[5];
  void* recv_x_scales_storage = buffers[6];

  auto compute_stream_ptr = reinterpret_cast<std::uintptr_t>(stream);
  buf->low_latency_dispatch(
      reinterpret_cast<std::uintptr_t>(x), p.num_tokens, p.hidden,
      reinterpret_cast<std::uintptr_t>(topk_idx), p.num_tokens, p.num_topk,
      reinterpret_cast<std::uintptr_t>(recv_x),
      p.use_fp8 ? reinterpret_cast<std::uintptr_t>(recv_x_scales_storage) : 0,
      reinterpret_cast<std::uintptr_t>(recv_count),
      reinterpret_cast<std::uintptr_t>(recv_src_info),
      reinterpret_cast<std::uintptr_t>(recv_layout_range),
      /*cumulative_local_expert_recv_stats_ptr=*/0,
      /*dispatch_wait_recv_cost_stats_ptr=*/0, compute_stream_ptr,
      p.num_max_dispatch_tokens_per_rank, p.num_experts,
      static_cast<bool>(p.use_fp8), static_cast<bool>(p.round_scale),
      static_cast<bool>(p.use_ue8m0), /*is_async=*/false,
      /*return_recv_hook=*/false);
}

// ---------------------------------------------------------------------------
// moe_low_latency_combine
// ---------------------------------------------------------------------------
struct MoELowLatencyCombineOpaque {
  int32_t x_dim0;
  int32_t x_dim1;
  int32_t x_dim2;
  int32_t num_tokens;
  int32_t num_topk;
  int32_t num_max_dispatch_tokens_per_rank;
  int32_t num_experts;
  int32_t use_logfmt;
  int32_t zero_copy;
  int32_t src_info_dim0;
  int32_t src_info_dim1;
  int32_t layout_range_dim0;
  int32_t layout_range_dim1;
};

extern "C" void uccl_moe_low_latency_combine_ffi(
    cudaStream_t stream, void** buffers, char const* opaque, size_t opaque_len,
    XlaCustomCallStatus* /*status*/) {
  if (opaque_len != sizeof(MoELowLatencyCombineOpaque)) return;
  auto const& p = *reinterpret_cast<MoELowLatencyCombineOpaque const*>(opaque);
  auto* buf = current_buffer_or_die();
  if (!buf) return;

  void* x = buffers[0];
  void* topk_idx = buffers[1];
  void* topk_weights = buffers[2];
  void* src_info = buffers[3];
  void* layout_range = buffers[4];
  void* combined_x = buffers[5];

  auto compute_stream_ptr = reinterpret_cast<std::uintptr_t>(stream);
  buf->low_latency_combine(
      reinterpret_cast<std::uintptr_t>(x), p.x_dim0, p.x_dim1, p.x_dim2,
      reinterpret_cast<std::uintptr_t>(topk_idx), p.num_tokens, p.num_topk,
      reinterpret_cast<std::uintptr_t>(topk_weights),
      reinterpret_cast<std::uintptr_t>(src_info), p.src_info_dim0,
      p.src_info_dim1, reinterpret_cast<std::uintptr_t>(layout_range),
      p.layout_range_dim0, p.layout_range_dim1,
      /*combine_wait_recv_cost_stats_ptr=*/0, compute_stream_ptr,
      p.num_max_dispatch_tokens_per_rank, p.num_experts,
      static_cast<bool>(p.use_logfmt), static_cast<bool>(p.zero_copy),
      /*is_async=*/false, /*return_recv_hook=*/false,
      reinterpret_cast<std::uintptr_t>(combined_x));
}

// ---------------------------------------------------------------------------
// moe_dispatch (intranode)
// ---------------------------------------------------------------------------
struct MoEDispatchOpaque {
  int32_t num_tokens;
  int32_t hidden;
  int32_t x_element_size;
  int32_t num_scales;
  int32_t scale_token_stride;
  int32_t scale_hidden_stride;
  int32_t num_topk;
  int32_t num_experts;
  int32_t expert_alignment;
  int32_t num_worst_tokens;
  int32_t num_sms;
  int32_t num_max_nvl_chunked_send_tokens;
  int32_t num_max_nvl_chunked_recv_tokens;
  int32_t num_max_rdma_chunked_send_tokens;
  int32_t num_max_rdma_chunked_recv_tokens;
  int32_t has_x_scales;
};

extern "C" void uccl_moe_dispatch_ffi(
    cudaStream_t stream, void** buffers, char const* opaque, size_t opaque_len,
    XlaCustomCallStatus* /*status*/) {
  if (opaque_len != sizeof(MoEDispatchOpaque)) return;
  auto const& p = *reinterpret_cast<MoEDispatchOpaque const*>(opaque);
  auto* buf = current_buffer_or_die();
  if (!buf) return;

  void* x = buffers[0];
  void* x_scales = buffers[1];
  void* topk_idx = buffers[2];
  void* topk_weights = buffers[3];
  void* recv_x = buffers[4];
  void* recv_x_scales = buffers[5];
  void* recv_topk_idx = buffers[6];
  void* recv_topk_weights = buffers[7];
  void* is_token_in_rank = buffers[8];
  void* num_tokens_per_rank = buffers[9];
  void* num_tokens_per_expert = buffers[10];
  void* rank_prefix_matrix = buffers[11];
  void* channel_prefix_matrix = buffers[12];
  void* recv_channel_prefix_matrix = buffers[13];
  void* recv_src_idx = buffers[14];
  void* send_head = buffers[15];

  auto compute_stream_ptr = reinterpret_cast<std::uintptr_t>(stream);
  uccl::Config config(p.num_sms, p.num_max_nvl_chunked_send_tokens,
                      p.num_max_nvl_chunked_recv_tokens,
                      p.num_max_rdma_chunked_send_tokens,
                      p.num_max_rdma_chunked_recv_tokens);
  std::optional<EventHandle> prev;

  // 1) layout
  buf->get_dispatch_layout(
      reinterpret_cast<std::uintptr_t>(topk_idx), p.num_tokens, p.num_topk,
      p.num_experts, reinterpret_cast<std::uintptr_t>(num_tokens_per_rank), 0,
      reinterpret_cast<std::uintptr_t>(num_tokens_per_expert),
      reinterpret_cast<std::uintptr_t>(is_token_in_rank), prev, false, false,
      compute_stream_ptr);

  // 2) intranode_prepare (num_worst_tokens > 0 => no host sync)
  buf->intranode_prepare(
      reinterpret_cast<std::uintptr_t>(num_tokens_per_rank),
      reinterpret_cast<std::uintptr_t>(is_token_in_rank),
      reinterpret_cast<std::uintptr_t>(num_tokens_per_expert), p.num_tokens,
      p.num_experts, reinterpret_cast<std::uintptr_t>(rank_prefix_matrix),
      reinterpret_cast<std::uintptr_t>(channel_prefix_matrix),
      p.expert_alignment, p.num_worst_tokens, config, prev, false, false,
      compute_stream_ptr);

  // 3) intranode_dispatch
  buf->intranode_dispatch(
      reinterpret_cast<std::uintptr_t>(x), p.num_tokens, p.hidden,
      p.x_element_size,
      p.has_x_scales ? reinterpret_cast<std::uintptr_t>(x_scales) : 0,
      p.num_scales, p.scale_token_stride, p.scale_hidden_stride,
      reinterpret_cast<std::uintptr_t>(topk_idx), p.num_topk,
      reinterpret_cast<std::uintptr_t>(topk_weights),
      reinterpret_cast<std::uintptr_t>(is_token_in_rank),
      reinterpret_cast<std::uintptr_t>(rank_prefix_matrix),
      reinterpret_cast<std::uintptr_t>(channel_prefix_matrix), p.num_experts,
      p.num_worst_tokens, /*cached_mode=*/false, config, p.num_worst_tokens,
      reinterpret_cast<std::uintptr_t>(recv_x),
      p.has_x_scales ? reinterpret_cast<std::uintptr_t>(recv_x_scales) : 0,
      reinterpret_cast<std::uintptr_t>(recv_topk_idx),
      reinterpret_cast<std::uintptr_t>(recv_topk_weights),
      reinterpret_cast<std::uintptr_t>(recv_channel_prefix_matrix),
      reinterpret_cast<std::uintptr_t>(recv_src_idx),
      reinterpret_cast<std::uintptr_t>(send_head), prev, false, false,
      compute_stream_ptr);
}

// ---------------------------------------------------------------------------
// moe_cached_dispatch (intranode, replay via cached handle)
// ---------------------------------------------------------------------------
struct MoECachedDispatchOpaque {
  int32_t num_tokens;
  int32_t hidden;
  int32_t x_element_size;
  int32_t num_scales;
  int32_t scale_token_stride;
  int32_t scale_hidden_stride;
  int32_t num_recv_tokens;
  int32_t num_sms;
  int32_t num_max_nvl_chunked_send_tokens;
  int32_t num_max_nvl_chunked_recv_tokens;
  int32_t num_max_rdma_chunked_send_tokens;
  int32_t num_max_rdma_chunked_recv_tokens;
  int32_t has_x_scales;
};

extern "C" void uccl_moe_cached_dispatch_ffi(
    cudaStream_t stream, void** buffers, char const* opaque, size_t opaque_len,
    XlaCustomCallStatus* /*status*/) {
  if (opaque_len != sizeof(MoECachedDispatchOpaque)) return;
  auto const& p = *reinterpret_cast<MoECachedDispatchOpaque const*>(opaque);
  auto* buf = current_buffer_or_die();
  if (!buf) return;

  void* x = buffers[0];
  void* x_scales = buffers[1];
  void* is_token_in_rank = buffers[2];
  void* rank_prefix_matrix = buffers[3];
  void* channel_prefix_matrix = buffers[4];
  void* recv_channel_prefix_matrix = buffers[5];
  void* recv_src_idx = buffers[6];
  void* send_head = buffers[7];
  void* recv_x = buffers[8];
  void* recv_x_scales = buffers[9];

  auto compute_stream_ptr = reinterpret_cast<std::uintptr_t>(stream);
  uccl::Config config(p.num_sms, p.num_max_nvl_chunked_send_tokens,
                      p.num_max_nvl_chunked_recv_tokens,
                      p.num_max_rdma_chunked_send_tokens,
                      p.num_max_rdma_chunked_recv_tokens);
  std::optional<EventHandle> prev;

  buf->intranode_dispatch(
      reinterpret_cast<std::uintptr_t>(x), p.num_tokens, p.hidden,
      p.x_element_size,
      p.has_x_scales ? reinterpret_cast<std::uintptr_t>(x_scales) : 0,
      p.num_scales, p.scale_token_stride, p.scale_hidden_stride,
      /*topk_idx_ptr=*/0, /*num_topk=*/0, /*topk_weights_ptr=*/0,
      reinterpret_cast<std::uintptr_t>(is_token_in_rank),
      reinterpret_cast<std::uintptr_t>(rank_prefix_matrix),
      reinterpret_cast<std::uintptr_t>(channel_prefix_matrix),
      /*num_experts=*/0, /*num_worst_tokens=*/0, /*cached_mode=*/true, config,
      p.num_recv_tokens, reinterpret_cast<std::uintptr_t>(recv_x),
      p.has_x_scales ? reinterpret_cast<std::uintptr_t>(recv_x_scales) : 0,
      /*recv_topk_idx_ptr=*/0, /*recv_topk_weights_ptr=*/0,
      reinterpret_cast<std::uintptr_t>(recv_channel_prefix_matrix),
      reinterpret_cast<std::uintptr_t>(recv_src_idx),
      reinterpret_cast<std::uintptr_t>(send_head), prev, false, false,
      compute_stream_ptr);
}

// ---------------------------------------------------------------------------
// moe_internode_cached_dispatch (replay via cached handle)
// ---------------------------------------------------------------------------
struct MoEInternodeCachedDispatchOpaque {
  int32_t num_tokens;
  int32_t hidden;
  int32_t x_element_size;
  int32_t num_scales;
  int32_t scale_token_stride;
  int32_t scale_hidden_stride;
  int32_t num_recv_tokens;
  int32_t num_rdma_recv_tokens;
  int32_t num_experts;
  int32_t num_sms;
  int32_t num_max_nvl_chunked_send_tokens;
  int32_t num_max_nvl_chunked_recv_tokens;
  int32_t num_max_rdma_chunked_send_tokens;
  int32_t num_max_rdma_chunked_recv_tokens;
  int32_t has_x_scales;
};

extern "C" void uccl_moe_internode_cached_dispatch_ffi(
    cudaStream_t stream, void** buffers, char const* opaque, size_t opaque_len,
    XlaCustomCallStatus* /*status*/) {
  if (opaque_len != sizeof(MoEInternodeCachedDispatchOpaque)) return;
  auto const& p =
      *reinterpret_cast<MoEInternodeCachedDispatchOpaque const*>(opaque);
  auto* buf = current_buffer_or_die();
  if (!buf) return;

  void* x = buffers[0];
  void* x_scales = buffers[1];
  void* is_token_in_rank = buffers[2];
  void* rdma_channel_prefix_matrix = buffers[3];
  void* gbl_channel_prefix_matrix = buffers[4];
  void* recv_rdma_rank_prefix_sum = buffers[5];
  void* recv_gbl_rank_prefix_sum = buffers[6];
  void* recv_x = buffers[7];
  void* recv_x_scales = buffers[8];

  auto compute_stream_ptr = reinterpret_cast<std::uintptr_t>(stream);
  uccl::Config config(p.num_sms, p.num_max_nvl_chunked_send_tokens,
                      p.num_max_nvl_chunked_recv_tokens,
                      p.num_max_rdma_chunked_send_tokens,
                      p.num_max_rdma_chunked_recv_tokens);
  std::optional<EventHandle> prev;

  buf->internode_dispatch(
      reinterpret_cast<std::uintptr_t>(x), p.num_tokens, p.hidden,
      p.x_element_size,
      p.has_x_scales ? reinterpret_cast<std::uintptr_t>(x_scales) : 0,
      p.num_scales, p.scale_token_stride, p.scale_hidden_stride,
      /*topk_idx_ptr=*/0, /*num_topk=*/0, /*topk_weights_ptr=*/0,
      reinterpret_cast<std::uintptr_t>(is_token_in_rank),
      reinterpret_cast<std::uintptr_t>(rdma_channel_prefix_matrix),
      reinterpret_cast<std::uintptr_t>(recv_rdma_rank_prefix_sum),
      reinterpret_cast<std::uintptr_t>(gbl_channel_prefix_matrix),
      reinterpret_cast<std::uintptr_t>(recv_gbl_rank_prefix_sum),
      p.num_experts, /*num_worst_tokens=*/0, /*cached_mode=*/true,
      p.num_rdma_recv_tokens, config,
      reinterpret_cast<std::uintptr_t>(recv_x),
      p.has_x_scales ? reinterpret_cast<std::uintptr_t>(recv_x_scales) : 0,
      /*recv_topk_idx_ptr=*/0, /*recv_topk_weights_ptr=*/0,
      /*recv_src_meta_ptr=*/0,
      /*recv_rdma_channel_prefix_matrix_ptr=*/0,
      /*recv_gbl_channel_prefix_matrix_ptr=*/0,
      /*send_rdma_head_ptr=*/0, /*send_nvl_head_ptr=*/0, prev, false, false,
      compute_stream_ptr);
}

// ---------------------------------------------------------------------------
// moe_internode_dispatch
// ---------------------------------------------------------------------------
struct MoEInternodeDispatchOpaque {
  int32_t num_tokens;
  int32_t hidden;
  int32_t x_element_size;
  int32_t num_scales;
  int32_t scale_token_stride;
  int32_t scale_hidden_stride;
  int32_t num_topk;
  int32_t num_experts;
  int32_t expert_alignment;
  int32_t num_worst_tokens;
  int32_t source_meta_bytes;
  int32_t num_sms;
  int32_t num_max_nvl_chunked_send_tokens;
  int32_t num_max_nvl_chunked_recv_tokens;
  int32_t num_max_rdma_chunked_send_tokens;
  int32_t num_max_rdma_chunked_recv_tokens;
  int32_t has_x_scales;
};

extern "C" void uccl_moe_internode_dispatch_ffi(
    cudaStream_t stream, void** buffers, char const* opaque, size_t opaque_len,
    XlaCustomCallStatus* /*status*/) {
  if (opaque_len != sizeof(MoEInternodeDispatchOpaque)) return;
  auto const& p = *reinterpret_cast<MoEInternodeDispatchOpaque const*>(opaque);
  auto* buf = current_buffer_or_die();
  if (!buf) return;

  // Inputs (4)
  void* x = buffers[0];
  void* x_scales = buffers[1];
  void* topk_idx = buffers[2];
  void* topk_weights = buffers[3];
  // Outputs (17)
  void* recv_x = buffers[4];
  void* recv_x_scales = buffers[5];
  void* recv_topk_idx = buffers[6];
  void* recv_topk_weights = buffers[7];
  void* is_token_in_rank = buffers[8];
  void* num_tokens_per_rank = buffers[9];
  void* num_tokens_per_rdma_rank = buffers[10];
  void* num_tokens_per_expert = buffers[11];
  void* rdma_channel_prefix_matrix = buffers[12];
  void* recv_rdma_rank_prefix_sum = buffers[13];
  void* gbl_channel_prefix_matrix = buffers[14];
  void* recv_gbl_rank_prefix_sum = buffers[15];
  void* recv_src_meta = buffers[16];
  void* recv_rdma_channel_prefix_matrix = buffers[17];
  void* recv_gbl_channel_prefix_matrix = buffers[18];
  void* send_rdma_head = buffers[19];
  void* send_nvl_head = buffers[20];

  auto compute_stream_ptr = reinterpret_cast<std::uintptr_t>(stream);
  uccl::Config config(p.num_sms, p.num_max_nvl_chunked_send_tokens,
                      p.num_max_nvl_chunked_recv_tokens,
                      p.num_max_rdma_chunked_send_tokens,
                      p.num_max_rdma_chunked_recv_tokens);
  std::optional<EventHandle> prev;

  // 1) Global layout (needs num_tokens_per_rdma_rank for the internode path).
  buf->get_dispatch_layout(
      reinterpret_cast<std::uintptr_t>(topk_idx), p.num_tokens, p.num_topk,
      p.num_experts, reinterpret_cast<std::uintptr_t>(num_tokens_per_rank),
      reinterpret_cast<std::uintptr_t>(num_tokens_per_rdma_rank),
      reinterpret_cast<std::uintptr_t>(num_tokens_per_expert),
      reinterpret_cast<std::uintptr_t>(is_token_in_rank), prev, false, false,
      compute_stream_ptr);

  // 2) internode_prepare with num_worst_tokens > 0 (no host sync).
  buf->internode_prepare(
      reinterpret_cast<std::uintptr_t>(num_tokens_per_rank),
      reinterpret_cast<std::uintptr_t>(num_tokens_per_rdma_rank),
      reinterpret_cast<std::uintptr_t>(num_tokens_per_expert),
      reinterpret_cast<std::uintptr_t>(is_token_in_rank), p.num_tokens,
      p.hidden, p.x_element_size, p.num_scales, p.num_topk, p.num_experts,
      p.expert_alignment, p.num_worst_tokens, config,
      reinterpret_cast<std::uintptr_t>(rdma_channel_prefix_matrix),
      reinterpret_cast<std::uintptr_t>(recv_rdma_rank_prefix_sum),
      reinterpret_cast<std::uintptr_t>(gbl_channel_prefix_matrix),
      reinterpret_cast<std::uintptr_t>(recv_gbl_rank_prefix_sum), prev, false,
      false, compute_stream_ptr);

  // 3) internode_dispatch.
  buf->internode_dispatch(
      reinterpret_cast<std::uintptr_t>(x), p.num_tokens, p.hidden,
      p.x_element_size,
      p.has_x_scales ? reinterpret_cast<std::uintptr_t>(x_scales) : 0,
      p.num_scales, p.scale_token_stride, p.scale_hidden_stride,
      reinterpret_cast<std::uintptr_t>(topk_idx), p.num_topk,
      reinterpret_cast<std::uintptr_t>(topk_weights),
      reinterpret_cast<std::uintptr_t>(is_token_in_rank),
      reinterpret_cast<std::uintptr_t>(rdma_channel_prefix_matrix),
      reinterpret_cast<std::uintptr_t>(recv_rdma_rank_prefix_sum),
      reinterpret_cast<std::uintptr_t>(gbl_channel_prefix_matrix),
      reinterpret_cast<std::uintptr_t>(recv_gbl_rank_prefix_sum), p.num_experts,
      p.num_worst_tokens, /*cached_mode=*/false, p.num_worst_tokens, config,
      reinterpret_cast<std::uintptr_t>(recv_x),
      p.has_x_scales ? reinterpret_cast<std::uintptr_t>(recv_x_scales) : 0,
      reinterpret_cast<std::uintptr_t>(recv_topk_idx),
      reinterpret_cast<std::uintptr_t>(recv_topk_weights),
      reinterpret_cast<std::uintptr_t>(recv_src_meta),
      reinterpret_cast<std::uintptr_t>(recv_rdma_channel_prefix_matrix),
      reinterpret_cast<std::uintptr_t>(recv_gbl_channel_prefix_matrix),
      reinterpret_cast<std::uintptr_t>(send_rdma_head),
      reinterpret_cast<std::uintptr_t>(send_nvl_head), prev, false, false,
      compute_stream_ptr);
}

// ---------------------------------------------------------------------------
// moe_internode_combine
// ---------------------------------------------------------------------------
struct MoEInternodeCombineOpaque {
  int32_t num_tokens;
  int32_t hidden;
  int32_t x_dtype_code;
  int32_t x_element_size;
  int32_t num_topk;
  int32_t num_combined_tokens;
  int32_t has_topk_weights;
  int32_t has_bias_0;
  int32_t has_bias_1;
  int32_t num_sms;
  int32_t num_max_nvl_chunked_send_tokens;
  int32_t num_max_nvl_chunked_recv_tokens;
  int32_t num_max_rdma_chunked_send_tokens;
  int32_t num_max_rdma_chunked_recv_tokens;
};

extern "C" void uccl_moe_internode_combine_ffi(
    cudaStream_t stream, void** buffers, char const* opaque, size_t opaque_len,
    XlaCustomCallStatus* /*status*/) {
  if (opaque_len != sizeof(MoEInternodeCombineOpaque)) return;
  auto const& p = *reinterpret_cast<MoEInternodeCombineOpaque const*>(opaque);
  auto* buf = current_buffer_or_die();
  if (!buf) return;

  void* x = buffers[0];
  void* topk_weights = buffers[1];
  void* bias_0 = buffers[2];
  void* bias_1 = buffers[3];
  void* is_tir = buffers[4];
  void* rdma_channel_prefix_matrix = buffers[5];
  void* recv_rdma_rank_prefix_sum = buffers[6];
  void* gbl_channel_prefix_matrix = buffers[7];
  void* src_meta = buffers[8];
  void* send_rdma_head = buffers[9];
  void* send_nvl_head = buffers[10];
  void* combined_x = buffers[11];
  void* combined_topk_weights = buffers[12];

  auto compute_stream_ptr = reinterpret_cast<std::uintptr_t>(stream);
  uccl::Config config(p.num_sms, p.num_max_nvl_chunked_send_tokens,
                      p.num_max_nvl_chunked_recv_tokens,
                      p.num_max_rdma_chunked_send_tokens,
                      p.num_max_rdma_chunked_recv_tokens);
  std::optional<EventHandle> prev;

  buf->internode_combine(
      reinterpret_cast<std::uintptr_t>(x), p.num_tokens, p.hidden,
      p.x_dtype_code, p.x_element_size,
      p.has_topk_weights ? reinterpret_cast<std::uintptr_t>(topk_weights) : 0,
      p.num_topk,
      p.has_bias_0 ? reinterpret_cast<std::uintptr_t>(bias_0) : 0,
      p.has_bias_1 ? reinterpret_cast<std::uintptr_t>(bias_1) : 0,
      reinterpret_cast<std::uintptr_t>(src_meta), p.num_combined_tokens,
      reinterpret_cast<std::uintptr_t>(is_tir),
      reinterpret_cast<std::uintptr_t>(rdma_channel_prefix_matrix),
      reinterpret_cast<std::uintptr_t>(recv_rdma_rank_prefix_sum),
      reinterpret_cast<std::uintptr_t>(gbl_channel_prefix_matrix),
      reinterpret_cast<std::uintptr_t>(send_rdma_head),
      reinterpret_cast<std::uintptr_t>(send_nvl_head), config,
      reinterpret_cast<std::uintptr_t>(combined_x),
      p.has_topk_weights
          ? reinterpret_cast<std::uintptr_t>(combined_topk_weights)
          : 0,
      prev, false, false, compute_stream_ptr);
}

// ---------------------------------------------------------------------------
// moe_combine (intranode)
// ---------------------------------------------------------------------------
struct MoECombineOpaque {
  int32_t num_tokens;
  int32_t hidden;
  int32_t x_dtype_code;
  int32_t x_element_size;
  int32_t num_topk;
  int32_t num_recv_tokens;
  int32_t has_topk_weights;
  int32_t has_bias_0;
  int32_t has_bias_1;
  int32_t num_sms;
  int32_t num_max_nvl_chunked_send_tokens;
  int32_t num_max_nvl_chunked_recv_tokens;
  int32_t num_max_rdma_chunked_send_tokens;
  int32_t num_max_rdma_chunked_recv_tokens;
};

extern "C" void uccl_moe_combine_ffi(
    cudaStream_t stream, void** buffers, char const* opaque, size_t opaque_len,
    XlaCustomCallStatus* /*status*/) {
  if (opaque_len != sizeof(MoECombineOpaque)) return;
  auto const& p = *reinterpret_cast<MoECombineOpaque const*>(opaque);
  auto* buf = current_buffer_or_die();
  if (!buf) return;

  void* x = buffers[0];
  void* topk_weights = buffers[1];
  void* bias_0 = buffers[2];
  void* bias_1 = buffers[3];
  void* src_idx = buffers[4];
  void* rank_prefix_matrix = buffers[5];
  void* channel_prefix_matrix = buffers[6];
  void* send_head = buffers[7];
  void* combined_x = buffers[8];
  void* combined_topk_weights = buffers[9];

  auto compute_stream_ptr = reinterpret_cast<std::uintptr_t>(stream);
  uccl::Config config(p.num_sms, p.num_max_nvl_chunked_send_tokens,
                      p.num_max_nvl_chunked_recv_tokens,
                      p.num_max_rdma_chunked_send_tokens,
                      p.num_max_rdma_chunked_recv_tokens);
  std::optional<EventHandle> prev;

  buf->intranode_combine(
      reinterpret_cast<std::uintptr_t>(x), p.num_tokens, p.hidden,
      p.x_dtype_code, p.x_element_size,
      p.has_topk_weights ? reinterpret_cast<std::uintptr_t>(topk_weights) : 0,
      p.num_topk,
      p.has_bias_0 ? reinterpret_cast<std::uintptr_t>(bias_0) : 0,
      p.has_bias_1 ? reinterpret_cast<std::uintptr_t>(bias_1) : 0,
      reinterpret_cast<std::uintptr_t>(src_idx), p.num_recv_tokens,
      reinterpret_cast<std::uintptr_t>(rank_prefix_matrix),
      reinterpret_cast<std::uintptr_t>(channel_prefix_matrix),
      reinterpret_cast<std::uintptr_t>(send_head), config,
      reinterpret_cast<std::uintptr_t>(combined_x),
      p.has_topk_weights
          ? reinterpret_cast<std::uintptr_t>(combined_topk_weights)
          : 0,
      prev, false, false, compute_stream_ptr);
}

// ---------------------------------------------------------------------------
// Nanobind bindings for the JAX plumbing, wired in from uccl_ep.cc
// ---------------------------------------------------------------------------

void register_jax_bindings(nb::module_& m) {
  m.def(
      "register_jax_ffi_buffer",
      [](int device_index, Buffer& buf) {
        register_buffer_for_device(device_index, &buf);
      },
      nb::arg("device_index"), nb::arg("buffer"),
      "Register a uccl.ep.Buffer for the given CUDA device so JAX FFI "
      "handlers can look it up at call time.");
  m.def(
      "unregister_jax_ffi_buffer",
      [](int device_index) { unregister_buffer_for_device(device_index); },
      nb::arg("device_index"));
  m.def("get_jax_ffi_targets", []() {
    auto make_capsule = [](void* fn) {
      return nb::capsule(fn, "xla._CUSTOM_CALL_TARGET");
    };
    nb::dict d;
    d["uccl_moe_low_latency_dispatch"] = make_capsule(
        reinterpret_cast<void*>(&uccl_moe_low_latency_dispatch_ffi));
    d["uccl_moe_low_latency_combine"] = make_capsule(
        reinterpret_cast<void*>(&uccl_moe_low_latency_combine_ffi));
    d["uccl_moe_dispatch"] =
        make_capsule(reinterpret_cast<void*>(&uccl_moe_dispatch_ffi));
    d["uccl_moe_combine"] =
        make_capsule(reinterpret_cast<void*>(&uccl_moe_combine_ffi));
    d["uccl_moe_internode_dispatch"] =
        make_capsule(reinterpret_cast<void*>(&uccl_moe_internode_dispatch_ffi));
    d["uccl_moe_internode_combine"] =
        make_capsule(reinterpret_cast<void*>(&uccl_moe_internode_combine_ffi));
    d["uccl_moe_cached_dispatch"] =
        make_capsule(reinterpret_cast<void*>(&uccl_moe_cached_dispatch_ffi));
    d["uccl_moe_internode_cached_dispatch"] = make_capsule(
        reinterpret_cast<void*>(&uccl_moe_internode_cached_dispatch_ffi));
    return d;
  });
}

}  // namespace uccl_jax_ffi
