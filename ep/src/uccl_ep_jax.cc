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

#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

// XLA Typed FFI: all custom calls below use
// ``XLA_FFI_DEFINE_HANDLER_SYMBOL`` (API_VERSION_TYPED_FFI). This is the
// only custom-call execution path currently supported by XLA:GPU (the
// legacy void**-buffers ABI was removed from ``CustomCallThunk``).
#include "xla/ffi/api/ffi.h"

#include "uccl_ep.hpp"

namespace uccl_jax_ffi {

namespace nb = nanobind;
namespace ffi = xla::ffi;

// Short aliases for the two Buffer flavors we use everywhere. ``AnyBuffer``
// keeps the dtype opaque (we pass raw ``uintptr_t`` into kernels that handle
// fp8/bf16/fp16/fp32/i32/i64/u8/bool uniformly), which keeps every handler
// binding compact.
using AnyBuf = ffi::AnyBuffer;
using RetAnyBuf = ffi::Result<ffi::AnyBuffer>;

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
// XLA custom-call handlers (Typed FFI)
// ----------------------------------------------------------------------------
// Every handler below is defined with ``XLA_FFI_DEFINE_HANDLER_SYMBOL`` and
// registered with ``jax.ffi.register_ffi_target(..., api_version=1)``, which
// corresponds to XLA's ``API_VERSION_TYPED_FFI``. Recent XLA:GPU
// (Nov-2025+) removed execution support for the legacy ``API_VERSION_ORIGINAL``
// and ``API_VERSION_STATUS_RETURNING`` ABIs, so this is the only shape
// that still works.
//
// Binding conventions used here:
//   * ``ffi::PlatformStream<cudaStream_t>`` is bound via ``Ctx<>`` to get
//     the CUDA stream XLA has scheduled the call on.
//   * Every input is bound as ``ffi::AnyBuffer`` (dtype-erased) and every
//     output as ``ffi::Result<ffi::AnyBuffer>``. We use ``untyped_data()`` +
//     ``dimensions()`` on both. This keeps the bindings short and matches
//     the underlying kernels, which take raw ``uintptr_t`` pointers.
//   * Python-side integer parameters (shapes, topo, bools, etc.) are
//     passed as typed ``Attr<int32_t>("name")`` attributes. The Python
//     wrappers in ``_calls.py`` must pass the exact same names / types
//     via ``jax.ffi.ffi_call(..., custom_call_api_version=4)``.
// ============================================================================

namespace {

// Convenience helpers to avoid littering the handlers with reinterpret_casts.
inline std::uintptr_t uptr(void* p) {
  return reinterpret_cast<std::uintptr_t>(p);
}
inline std::uintptr_t uptr(AnyBuf const& b) { return uptr(b.untyped_data()); }
// ``ffi::Result::operator->`` is non-const (returns ``T*``), so we take the
// wrapper by value. It's a trivial struct that stores the underlying buffer
// by value; copying it is cheap and lets us call member functions on it.
inline std::uintptr_t uptr(RetAnyBuf b) { return uptr(b->untyped_data()); }
inline std::uintptr_t uptr_if(bool cond, AnyBuf const& b) {
  return cond ? uptr(b) : 0;
}
inline std::uintptr_t uptr_if(bool cond, RetAnyBuf b) {
  return cond ? uptr(b) : 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// moe_low_latency_dispatch
// ---------------------------------------------------------------------------
static ffi::Error Impl_MoeLowLatencyDispatch(
    cudaStream_t stream,
    // Inputs
    AnyBuf x, AnyBuf topk_idx,
    // Outputs
    RetAnyBuf recv_x, RetAnyBuf recv_count, RetAnyBuf recv_src_info,
    RetAnyBuf recv_layout_range, RetAnyBuf recv_x_scales_storage,
    // Attrs
    int32_t num_tokens, int32_t hidden, int32_t num_topk,
    int32_t num_max_dispatch_tokens_per_rank, int32_t num_experts,
    int32_t use_fp8, int32_t round_scale, int32_t use_ue8m0) {
  auto* buf = current_buffer_or_die();
  if (!buf) {
    return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                      "uccl.ep: no Buffer registered for current CUDA device");
  }
  auto compute_stream_ptr = reinterpret_cast<std::uintptr_t>(stream);
  buf->low_latency_dispatch(
      uptr(x), num_tokens, hidden, uptr(topk_idx), num_tokens, num_topk,
      uptr(recv_x),
      use_fp8 ? uptr(recv_x_scales_storage) : 0,
      uptr(recv_count), uptr(recv_src_info), uptr(recv_layout_range),
      /*cumulative_local_expert_recv_stats_ptr=*/0,
      /*dispatch_wait_recv_cost_stats_ptr=*/0, compute_stream_ptr,
      num_max_dispatch_tokens_per_rank, num_experts,
      static_cast<bool>(use_fp8), static_cast<bool>(round_scale),
      static_cast<bool>(use_ue8m0), /*is_async=*/false,
      /*return_recv_hook=*/false);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Handler_MoeLowLatencyDispatch, Impl_MoeLowLatencyDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<AnyBuf>().Arg<AnyBuf>()
        .Ret<AnyBuf>().Ret<AnyBuf>().Ret<AnyBuf>().Ret<AnyBuf>().Ret<AnyBuf>()
        .Attr<int32_t>("num_tokens")
        .Attr<int32_t>("hidden")
        .Attr<int32_t>("num_topk")
        .Attr<int32_t>("num_max_dispatch_tokens_per_rank")
        .Attr<int32_t>("num_experts")
        .Attr<int32_t>("use_fp8")
        .Attr<int32_t>("round_scale")
        .Attr<int32_t>("use_ue8m0"));

// ---------------------------------------------------------------------------
// moe_low_latency_combine
// ---------------------------------------------------------------------------
static ffi::Error Impl_MoeLowLatencyCombine(
    cudaStream_t stream,
    // Inputs
    AnyBuf x, AnyBuf topk_idx, AnyBuf topk_weights, AnyBuf src_info,
    AnyBuf layout_range,
    // Outputs
    RetAnyBuf combined_x,
    // Attrs
    int32_t x_dim0, int32_t x_dim1, int32_t x_dim2, int32_t num_tokens,
    int32_t num_topk, int32_t num_max_dispatch_tokens_per_rank,
    int32_t num_experts, int32_t use_logfmt, int32_t zero_copy,
    int32_t src_info_dim0, int32_t src_info_dim1, int32_t layout_range_dim0,
    int32_t layout_range_dim1) {
  auto* buf = current_buffer_or_die();
  if (!buf) {
    return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                      "uccl.ep: no Buffer registered for current CUDA device");
  }
  auto compute_stream_ptr = reinterpret_cast<std::uintptr_t>(stream);
  buf->low_latency_combine(
      uptr(x), x_dim0, x_dim1, x_dim2,
      uptr(topk_idx), num_tokens, num_topk,
      uptr(topk_weights), uptr(src_info), src_info_dim0, src_info_dim1,
      uptr(layout_range), layout_range_dim0, layout_range_dim1,
      /*combine_wait_recv_cost_stats_ptr=*/0, compute_stream_ptr,
      num_max_dispatch_tokens_per_rank, num_experts,
      static_cast<bool>(use_logfmt), static_cast<bool>(zero_copy),
      /*is_async=*/false, /*return_recv_hook=*/false, uptr(combined_x));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Handler_MoeLowLatencyCombine, Impl_MoeLowLatencyCombine,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<AnyBuf>().Arg<AnyBuf>().Arg<AnyBuf>().Arg<AnyBuf>().Arg<AnyBuf>()
        .Ret<AnyBuf>()
        .Attr<int32_t>("x_dim0")
        .Attr<int32_t>("x_dim1")
        .Attr<int32_t>("x_dim2")
        .Attr<int32_t>("num_tokens")
        .Attr<int32_t>("num_topk")
        .Attr<int32_t>("num_max_dispatch_tokens_per_rank")
        .Attr<int32_t>("num_experts")
        .Attr<int32_t>("use_logfmt")
        .Attr<int32_t>("zero_copy")
        .Attr<int32_t>("src_info_dim0")
        .Attr<int32_t>("src_info_dim1")
        .Attr<int32_t>("layout_range_dim0")
        .Attr<int32_t>("layout_range_dim1"));

// ---------------------------------------------------------------------------
// moe_dispatch (intranode)
// ---------------------------------------------------------------------------
static ffi::Error Impl_MoeDispatch(
    cudaStream_t stream,
    // Inputs
    AnyBuf x, AnyBuf x_scales, AnyBuf topk_idx, AnyBuf topk_weights,
    // Outputs
    RetAnyBuf recv_x, RetAnyBuf recv_x_scales, RetAnyBuf recv_topk_idx,
    RetAnyBuf recv_topk_weights, RetAnyBuf is_token_in_rank,
    RetAnyBuf num_tokens_per_rank, RetAnyBuf num_tokens_per_expert,
    RetAnyBuf rank_prefix_matrix, RetAnyBuf channel_prefix_matrix,
    RetAnyBuf recv_channel_prefix_matrix, RetAnyBuf recv_src_idx,
    RetAnyBuf send_head,
    // Attrs
    int32_t num_tokens, int32_t hidden, int32_t x_element_size,
    int32_t num_scales, int32_t scale_token_stride, int32_t scale_hidden_stride,
    int32_t num_topk, int32_t num_experts, int32_t expert_alignment,
    int32_t num_worst_tokens, int32_t num_sms,
    int32_t num_max_nvl_chunked_send_tokens,
    int32_t num_max_nvl_chunked_recv_tokens,
    int32_t num_max_rdma_chunked_send_tokens,
    int32_t num_max_rdma_chunked_recv_tokens, int32_t has_x_scales) {
  auto* buf = current_buffer_or_die();
  if (!buf) {
    return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                      "uccl.ep: no Buffer registered for current CUDA device");
  }
  auto compute_stream_ptr = reinterpret_cast<std::uintptr_t>(stream);
  uccl::Config config(num_sms, num_max_nvl_chunked_send_tokens,
                      num_max_nvl_chunked_recv_tokens,
                      num_max_rdma_chunked_send_tokens,
                      num_max_rdma_chunked_recv_tokens);
  std::optional<EventHandle> prev;

  // 1) layout
  buf->get_dispatch_layout(
      uptr(topk_idx), num_tokens, num_topk, num_experts,
      uptr(num_tokens_per_rank), 0, uptr(num_tokens_per_expert),
      uptr(is_token_in_rank), prev, false, false, compute_stream_ptr);

  // 2) intranode_prepare
  buf->intranode_prepare(
      uptr(num_tokens_per_rank), uptr(is_token_in_rank),
      uptr(num_tokens_per_expert), num_tokens, num_experts,
      uptr(rank_prefix_matrix), uptr(channel_prefix_matrix), expert_alignment,
      num_worst_tokens, config, prev, false, false, compute_stream_ptr);

  // 3) intranode_dispatch
  buf->intranode_dispatch(
      uptr(x), num_tokens, hidden, x_element_size,
      uptr_if(has_x_scales, x_scales), num_scales, scale_token_stride,
      scale_hidden_stride, uptr(topk_idx), num_topk, uptr(topk_weights),
      uptr(is_token_in_rank), uptr(rank_prefix_matrix),
      uptr(channel_prefix_matrix), num_experts, num_worst_tokens,
      /*cached_mode=*/false, config, num_worst_tokens, uptr(recv_x),
      uptr_if(has_x_scales, recv_x_scales), uptr(recv_topk_idx),
      uptr(recv_topk_weights), uptr(recv_channel_prefix_matrix),
      uptr(recv_src_idx), uptr(send_head), prev, false, false,
      compute_stream_ptr);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Handler_MoeDispatch, Impl_MoeDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<AnyBuf>().Arg<AnyBuf>().Arg<AnyBuf>().Arg<AnyBuf>()
        .Ret<AnyBuf>().Ret<AnyBuf>().Ret<AnyBuf>().Ret<AnyBuf>()
        .Ret<AnyBuf>().Ret<AnyBuf>().Ret<AnyBuf>().Ret<AnyBuf>()
        .Ret<AnyBuf>().Ret<AnyBuf>().Ret<AnyBuf>().Ret<AnyBuf>()
        .Attr<int32_t>("num_tokens")
        .Attr<int32_t>("hidden")
        .Attr<int32_t>("x_element_size")
        .Attr<int32_t>("num_scales")
        .Attr<int32_t>("scale_token_stride")
        .Attr<int32_t>("scale_hidden_stride")
        .Attr<int32_t>("num_topk")
        .Attr<int32_t>("num_experts")
        .Attr<int32_t>("expert_alignment")
        .Attr<int32_t>("num_worst_tokens")
        .Attr<int32_t>("num_sms")
        .Attr<int32_t>("num_max_nvl_chunked_send_tokens")
        .Attr<int32_t>("num_max_nvl_chunked_recv_tokens")
        .Attr<int32_t>("num_max_rdma_chunked_send_tokens")
        .Attr<int32_t>("num_max_rdma_chunked_recv_tokens")
        .Attr<int32_t>("has_x_scales"));

// ---------------------------------------------------------------------------
// moe_cached_dispatch (intranode, replay via cached handle)
// ---------------------------------------------------------------------------
static ffi::Error Impl_MoeCachedDispatch(
    cudaStream_t stream,
    // Inputs
    AnyBuf x, AnyBuf x_scales, AnyBuf is_token_in_rank,
    AnyBuf rank_prefix_matrix, AnyBuf channel_prefix_matrix,
    AnyBuf recv_channel_prefix_matrix, AnyBuf recv_src_idx, AnyBuf send_head,
    // Outputs
    RetAnyBuf recv_x, RetAnyBuf recv_x_scales,
    // Attrs
    int32_t num_tokens, int32_t hidden, int32_t x_element_size,
    int32_t num_scales, int32_t scale_token_stride, int32_t scale_hidden_stride,
    int32_t num_recv_tokens, int32_t num_sms,
    int32_t num_max_nvl_chunked_send_tokens,
    int32_t num_max_nvl_chunked_recv_tokens,
    int32_t num_max_rdma_chunked_send_tokens,
    int32_t num_max_rdma_chunked_recv_tokens, int32_t has_x_scales) {
  auto* buf = current_buffer_or_die();
  if (!buf) {
    return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                      "uccl.ep: no Buffer registered for current CUDA device");
  }
  auto compute_stream_ptr = reinterpret_cast<std::uintptr_t>(stream);
  uccl::Config config(num_sms, num_max_nvl_chunked_send_tokens,
                      num_max_nvl_chunked_recv_tokens,
                      num_max_rdma_chunked_send_tokens,
                      num_max_rdma_chunked_recv_tokens);
  std::optional<EventHandle> prev;

  buf->intranode_dispatch(
      uptr(x), num_tokens, hidden, x_element_size,
      uptr_if(has_x_scales, x_scales), num_scales, scale_token_stride,
      scale_hidden_stride,
      /*topk_idx_ptr=*/0, /*num_topk=*/0, /*topk_weights_ptr=*/0,
      uptr(is_token_in_rank), uptr(rank_prefix_matrix),
      uptr(channel_prefix_matrix), /*num_experts=*/0, /*num_worst_tokens=*/0,
      /*cached_mode=*/true, config, num_recv_tokens, uptr(recv_x),
      uptr_if(has_x_scales, recv_x_scales),
      /*recv_topk_idx_ptr=*/0, /*recv_topk_weights_ptr=*/0,
      uptr(recv_channel_prefix_matrix), uptr(recv_src_idx), uptr(send_head),
      prev, false, false, compute_stream_ptr);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Handler_MoeCachedDispatch, Impl_MoeCachedDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<AnyBuf>().Arg<AnyBuf>().Arg<AnyBuf>().Arg<AnyBuf>()
        .Arg<AnyBuf>().Arg<AnyBuf>().Arg<AnyBuf>().Arg<AnyBuf>()
        .Ret<AnyBuf>().Ret<AnyBuf>()
        .Attr<int32_t>("num_tokens")
        .Attr<int32_t>("hidden")
        .Attr<int32_t>("x_element_size")
        .Attr<int32_t>("num_scales")
        .Attr<int32_t>("scale_token_stride")
        .Attr<int32_t>("scale_hidden_stride")
        .Attr<int32_t>("num_recv_tokens")
        .Attr<int32_t>("num_sms")
        .Attr<int32_t>("num_max_nvl_chunked_send_tokens")
        .Attr<int32_t>("num_max_nvl_chunked_recv_tokens")
        .Attr<int32_t>("num_max_rdma_chunked_send_tokens")
        .Attr<int32_t>("num_max_rdma_chunked_recv_tokens")
        .Attr<int32_t>("has_x_scales"));

// ---------------------------------------------------------------------------
// moe_internode_cached_dispatch
// ---------------------------------------------------------------------------
static ffi::Error Impl_MoeInternodeCachedDispatch(
    cudaStream_t stream,
    // Inputs
    AnyBuf x, AnyBuf x_scales, AnyBuf is_token_in_rank,
    AnyBuf rdma_channel_prefix_matrix, AnyBuf gbl_channel_prefix_matrix,
    AnyBuf recv_rdma_rank_prefix_sum, AnyBuf recv_gbl_rank_prefix_sum,
    // Outputs
    RetAnyBuf recv_x, RetAnyBuf recv_x_scales,
    // Attrs
    int32_t num_tokens, int32_t hidden, int32_t x_element_size,
    int32_t num_scales, int32_t scale_token_stride, int32_t scale_hidden_stride,
    int32_t num_recv_tokens, int32_t num_rdma_recv_tokens, int32_t num_experts,
    int32_t num_sms, int32_t num_max_nvl_chunked_send_tokens,
    int32_t num_max_nvl_chunked_recv_tokens,
    int32_t num_max_rdma_chunked_send_tokens,
    int32_t num_max_rdma_chunked_recv_tokens, int32_t has_x_scales) {
  auto* buf = current_buffer_or_die();
  if (!buf) {
    return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                      "uccl.ep: no Buffer registered for current CUDA device");
  }
  auto compute_stream_ptr = reinterpret_cast<std::uintptr_t>(stream);
  uccl::Config config(num_sms, num_max_nvl_chunked_send_tokens,
                      num_max_nvl_chunked_recv_tokens,
                      num_max_rdma_chunked_send_tokens,
                      num_max_rdma_chunked_recv_tokens);
  std::optional<EventHandle> prev;

  buf->internode_dispatch(
      uptr(x), num_tokens, hidden, x_element_size,
      uptr_if(has_x_scales, x_scales), num_scales, scale_token_stride,
      scale_hidden_stride,
      /*topk_idx_ptr=*/0, /*num_topk=*/0, /*topk_weights_ptr=*/0,
      uptr(is_token_in_rank), uptr(rdma_channel_prefix_matrix),
      uptr(recv_rdma_rank_prefix_sum), uptr(gbl_channel_prefix_matrix),
      uptr(recv_gbl_rank_prefix_sum), num_experts, /*num_worst_tokens=*/0,
      /*cached_mode=*/true, num_rdma_recv_tokens, config, uptr(recv_x),
      uptr_if(has_x_scales, recv_x_scales),
      /*recv_topk_idx_ptr=*/0, /*recv_topk_weights_ptr=*/0,
      /*recv_src_meta_ptr=*/0,
      /*recv_rdma_channel_prefix_matrix_ptr=*/0,
      /*recv_gbl_channel_prefix_matrix_ptr=*/0,
      /*send_rdma_head_ptr=*/0, /*send_nvl_head_ptr=*/0, prev, false, false,
      compute_stream_ptr);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Handler_MoeInternodeCachedDispatch, Impl_MoeInternodeCachedDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<AnyBuf>().Arg<AnyBuf>().Arg<AnyBuf>().Arg<AnyBuf>()
        .Arg<AnyBuf>().Arg<AnyBuf>().Arg<AnyBuf>()
        .Ret<AnyBuf>().Ret<AnyBuf>()
        .Attr<int32_t>("num_tokens")
        .Attr<int32_t>("hidden")
        .Attr<int32_t>("x_element_size")
        .Attr<int32_t>("num_scales")
        .Attr<int32_t>("scale_token_stride")
        .Attr<int32_t>("scale_hidden_stride")
        .Attr<int32_t>("num_recv_tokens")
        .Attr<int32_t>("num_rdma_recv_tokens")
        .Attr<int32_t>("num_experts")
        .Attr<int32_t>("num_sms")
        .Attr<int32_t>("num_max_nvl_chunked_send_tokens")
        .Attr<int32_t>("num_max_nvl_chunked_recv_tokens")
        .Attr<int32_t>("num_max_rdma_chunked_send_tokens")
        .Attr<int32_t>("num_max_rdma_chunked_recv_tokens")
        .Attr<int32_t>("has_x_scales"));

// ---------------------------------------------------------------------------
// moe_internode_dispatch
// ---------------------------------------------------------------------------
static ffi::Error Impl_MoeInternodeDispatch(
    cudaStream_t stream,
    // Inputs (4)
    AnyBuf x, AnyBuf x_scales, AnyBuf topk_idx, AnyBuf topk_weights,
    // Outputs (17)
    RetAnyBuf recv_x, RetAnyBuf recv_x_scales, RetAnyBuf recv_topk_idx,
    RetAnyBuf recv_topk_weights, RetAnyBuf is_token_in_rank,
    RetAnyBuf num_tokens_per_rank, RetAnyBuf num_tokens_per_rdma_rank,
    RetAnyBuf num_tokens_per_expert, RetAnyBuf rdma_channel_prefix_matrix,
    RetAnyBuf recv_rdma_rank_prefix_sum, RetAnyBuf gbl_channel_prefix_matrix,
    RetAnyBuf recv_gbl_rank_prefix_sum, RetAnyBuf recv_src_meta,
    RetAnyBuf recv_rdma_channel_prefix_matrix,
    RetAnyBuf recv_gbl_channel_prefix_matrix, RetAnyBuf send_rdma_head,
    RetAnyBuf send_nvl_head,
    // Attrs
    int32_t num_tokens, int32_t hidden, int32_t x_element_size,
    int32_t num_scales, int32_t scale_token_stride, int32_t scale_hidden_stride,
    int32_t num_topk, int32_t num_experts, int32_t expert_alignment,
    int32_t num_worst_tokens, int32_t source_meta_bytes, int32_t num_sms,
    int32_t num_max_nvl_chunked_send_tokens,
    int32_t num_max_nvl_chunked_recv_tokens,
    int32_t num_max_rdma_chunked_send_tokens,
    int32_t num_max_rdma_chunked_recv_tokens, int32_t has_x_scales) {
  auto* buf = current_buffer_or_die();
  if (!buf) {
    return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                      "uccl.ep: no Buffer registered for current CUDA device");
  }
  auto compute_stream_ptr = reinterpret_cast<std::uintptr_t>(stream);
  uccl::Config config(num_sms, num_max_nvl_chunked_send_tokens,
                      num_max_nvl_chunked_recv_tokens,
                      num_max_rdma_chunked_send_tokens,
                      num_max_rdma_chunked_recv_tokens);
  std::optional<EventHandle> prev;

  // 1) Global layout
  buf->get_dispatch_layout(
      uptr(topk_idx), num_tokens, num_topk, num_experts,
      uptr(num_tokens_per_rank), uptr(num_tokens_per_rdma_rank),
      uptr(num_tokens_per_expert), uptr(is_token_in_rank), prev, false, false,
      compute_stream_ptr);

  // 2) internode_prepare with num_worst_tokens > 0 (no host sync).
  buf->internode_prepare(
      uptr(num_tokens_per_rank), uptr(num_tokens_per_rdma_rank),
      uptr(num_tokens_per_expert), uptr(is_token_in_rank), num_tokens, hidden,
      x_element_size, num_scales, num_topk, num_experts, expert_alignment,
      num_worst_tokens, config, uptr(rdma_channel_prefix_matrix),
      uptr(recv_rdma_rank_prefix_sum), uptr(gbl_channel_prefix_matrix),
      uptr(recv_gbl_rank_prefix_sum), prev, false, false, compute_stream_ptr);

  // 3) internode_dispatch
  buf->internode_dispatch(
      uptr(x), num_tokens, hidden, x_element_size,
      uptr_if(has_x_scales, x_scales), num_scales, scale_token_stride,
      scale_hidden_stride, uptr(topk_idx), num_topk, uptr(topk_weights),
      uptr(is_token_in_rank), uptr(rdma_channel_prefix_matrix),
      uptr(recv_rdma_rank_prefix_sum), uptr(gbl_channel_prefix_matrix),
      uptr(recv_gbl_rank_prefix_sum), num_experts, num_worst_tokens,
      /*cached_mode=*/false, num_worst_tokens, config, uptr(recv_x),
      uptr_if(has_x_scales, recv_x_scales), uptr(recv_topk_idx),
      uptr(recv_topk_weights), uptr(recv_src_meta),
      uptr(recv_rdma_channel_prefix_matrix),
      uptr(recv_gbl_channel_prefix_matrix), uptr(send_rdma_head),
      uptr(send_nvl_head), prev, false, false, compute_stream_ptr);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Handler_MoeInternodeDispatch, Impl_MoeInternodeDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<AnyBuf>().Arg<AnyBuf>().Arg<AnyBuf>().Arg<AnyBuf>()
        .Ret<AnyBuf>().Ret<AnyBuf>().Ret<AnyBuf>().Ret<AnyBuf>()
        .Ret<AnyBuf>().Ret<AnyBuf>().Ret<AnyBuf>().Ret<AnyBuf>()
        .Ret<AnyBuf>().Ret<AnyBuf>().Ret<AnyBuf>().Ret<AnyBuf>()
        .Ret<AnyBuf>().Ret<AnyBuf>().Ret<AnyBuf>().Ret<AnyBuf>()
        .Ret<AnyBuf>()
        .Attr<int32_t>("num_tokens")
        .Attr<int32_t>("hidden")
        .Attr<int32_t>("x_element_size")
        .Attr<int32_t>("num_scales")
        .Attr<int32_t>("scale_token_stride")
        .Attr<int32_t>("scale_hidden_stride")
        .Attr<int32_t>("num_topk")
        .Attr<int32_t>("num_experts")
        .Attr<int32_t>("expert_alignment")
        .Attr<int32_t>("num_worst_tokens")
        .Attr<int32_t>("source_meta_bytes")
        .Attr<int32_t>("num_sms")
        .Attr<int32_t>("num_max_nvl_chunked_send_tokens")
        .Attr<int32_t>("num_max_nvl_chunked_recv_tokens")
        .Attr<int32_t>("num_max_rdma_chunked_send_tokens")
        .Attr<int32_t>("num_max_rdma_chunked_recv_tokens")
        .Attr<int32_t>("has_x_scales"));

// ---------------------------------------------------------------------------
// moe_internode_combine
// ---------------------------------------------------------------------------
static ffi::Error Impl_MoeInternodeCombine(
    cudaStream_t stream,
    // Inputs
    AnyBuf x, AnyBuf topk_weights, AnyBuf bias_0, AnyBuf bias_1,
    AnyBuf is_tir, AnyBuf rdma_channel_prefix_matrix,
    AnyBuf recv_rdma_rank_prefix_sum, AnyBuf gbl_channel_prefix_matrix,
    AnyBuf src_meta, AnyBuf send_rdma_head, AnyBuf send_nvl_head,
    // Outputs
    RetAnyBuf combined_x, RetAnyBuf combined_topk_weights,
    // Attrs
    int32_t num_tokens, int32_t hidden, int32_t x_dtype_code,
    int32_t x_element_size, int32_t num_topk, int32_t num_combined_tokens,
    int32_t has_topk_weights, int32_t has_bias_0, int32_t has_bias_1,
    int32_t num_sms, int32_t num_max_nvl_chunked_send_tokens,
    int32_t num_max_nvl_chunked_recv_tokens,
    int32_t num_max_rdma_chunked_send_tokens,
    int32_t num_max_rdma_chunked_recv_tokens) {
  auto* buf = current_buffer_or_die();
  if (!buf) {
    return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                      "uccl.ep: no Buffer registered for current CUDA device");
  }
  auto compute_stream_ptr = reinterpret_cast<std::uintptr_t>(stream);
  uccl::Config config(num_sms, num_max_nvl_chunked_send_tokens,
                      num_max_nvl_chunked_recv_tokens,
                      num_max_rdma_chunked_send_tokens,
                      num_max_rdma_chunked_recv_tokens);
  std::optional<EventHandle> prev;

  buf->internode_combine(
      uptr(x), num_tokens, hidden, x_dtype_code, x_element_size,
      uptr_if(has_topk_weights, topk_weights), num_topk,
      uptr_if(has_bias_0, bias_0), uptr_if(has_bias_1, bias_1),
      uptr(src_meta), num_combined_tokens, uptr(is_tir),
      uptr(rdma_channel_prefix_matrix), uptr(recv_rdma_rank_prefix_sum),
      uptr(gbl_channel_prefix_matrix), uptr(send_rdma_head),
      uptr(send_nvl_head), config, uptr(combined_x),
      uptr_if(has_topk_weights, combined_topk_weights), prev, false, false,
      compute_stream_ptr);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Handler_MoeInternodeCombine, Impl_MoeInternodeCombine,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<AnyBuf>().Arg<AnyBuf>().Arg<AnyBuf>().Arg<AnyBuf>()
        .Arg<AnyBuf>().Arg<AnyBuf>().Arg<AnyBuf>().Arg<AnyBuf>()
        .Arg<AnyBuf>().Arg<AnyBuf>().Arg<AnyBuf>()
        .Ret<AnyBuf>().Ret<AnyBuf>()
        .Attr<int32_t>("num_tokens")
        .Attr<int32_t>("hidden")
        .Attr<int32_t>("x_dtype_code")
        .Attr<int32_t>("x_element_size")
        .Attr<int32_t>("num_topk")
        .Attr<int32_t>("num_combined_tokens")
        .Attr<int32_t>("has_topk_weights")
        .Attr<int32_t>("has_bias_0")
        .Attr<int32_t>("has_bias_1")
        .Attr<int32_t>("num_sms")
        .Attr<int32_t>("num_max_nvl_chunked_send_tokens")
        .Attr<int32_t>("num_max_nvl_chunked_recv_tokens")
        .Attr<int32_t>("num_max_rdma_chunked_send_tokens")
        .Attr<int32_t>("num_max_rdma_chunked_recv_tokens"));

// ---------------------------------------------------------------------------
// moe_combine (intranode)
// ---------------------------------------------------------------------------
static ffi::Error Impl_MoeCombine(
    cudaStream_t stream,
    // Inputs
    AnyBuf x, AnyBuf topk_weights, AnyBuf bias_0, AnyBuf bias_1, AnyBuf src_idx,
    AnyBuf rank_prefix_matrix, AnyBuf channel_prefix_matrix, AnyBuf send_head,
    // Outputs
    RetAnyBuf combined_x, RetAnyBuf combined_topk_weights,
    // Attrs
    int32_t num_tokens, int32_t hidden, int32_t x_dtype_code,
    int32_t x_element_size, int32_t num_topk, int32_t num_recv_tokens,
    int32_t has_topk_weights, int32_t has_bias_0, int32_t has_bias_1,
    int32_t num_sms, int32_t num_max_nvl_chunked_send_tokens,
    int32_t num_max_nvl_chunked_recv_tokens,
    int32_t num_max_rdma_chunked_send_tokens,
    int32_t num_max_rdma_chunked_recv_tokens) {
  auto* buf = current_buffer_or_die();
  if (!buf) {
    return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                      "uccl.ep: no Buffer registered for current CUDA device");
  }
  auto compute_stream_ptr = reinterpret_cast<std::uintptr_t>(stream);
  uccl::Config config(num_sms, num_max_nvl_chunked_send_tokens,
                      num_max_nvl_chunked_recv_tokens,
                      num_max_rdma_chunked_send_tokens,
                      num_max_rdma_chunked_recv_tokens);
  std::optional<EventHandle> prev;

  buf->intranode_combine(
      uptr(x), num_tokens, hidden, x_dtype_code, x_element_size,
      uptr_if(has_topk_weights, topk_weights), num_topk,
      uptr_if(has_bias_0, bias_0), uptr_if(has_bias_1, bias_1),
      uptr(src_idx), num_recv_tokens, uptr(rank_prefix_matrix),
      uptr(channel_prefix_matrix), uptr(send_head), config, uptr(combined_x),
      uptr_if(has_topk_weights, combined_topk_weights), prev, false, false,
      compute_stream_ptr);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Handler_MoeCombine, Impl_MoeCombine,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<AnyBuf>().Arg<AnyBuf>().Arg<AnyBuf>().Arg<AnyBuf>()
        .Arg<AnyBuf>().Arg<AnyBuf>().Arg<AnyBuf>().Arg<AnyBuf>()
        .Ret<AnyBuf>().Ret<AnyBuf>()
        .Attr<int32_t>("num_tokens")
        .Attr<int32_t>("hidden")
        .Attr<int32_t>("x_dtype_code")
        .Attr<int32_t>("x_element_size")
        .Attr<int32_t>("num_topk")
        .Attr<int32_t>("num_recv_tokens")
        .Attr<int32_t>("has_topk_weights")
        .Attr<int32_t>("has_bias_0")
        .Attr<int32_t>("has_bias_1")
        .Attr<int32_t>("num_sms")
        .Attr<int32_t>("num_max_nvl_chunked_send_tokens")
        .Attr<int32_t>("num_max_nvl_chunked_recv_tokens")
        .Attr<int32_t>("num_max_rdma_chunked_send_tokens")
        .Attr<int32_t>("num_max_rdma_chunked_recv_tokens"));

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
    // Each Handler_* symbol below is an ``XLA_FFI_Handler*`` function
    // (see ``XLA_FFI_DEFINE_HANDLER_SYMBOL`` above), which is what
    // ``jax.ffi.register_ffi_target(..., api_version=1)`` expects when
    // it unwraps an ``xla._CUSTOM_CALL_TARGET`` capsule.
    auto make_capsule = [](void* fn) {
      return nb::capsule(fn, "xla._CUSTOM_CALL_TARGET");
    };
    nb::dict d;
    d["uccl_moe_low_latency_dispatch"] =
        make_capsule(reinterpret_cast<void*>(&Handler_MoeLowLatencyDispatch));
    d["uccl_moe_low_latency_combine"] =
        make_capsule(reinterpret_cast<void*>(&Handler_MoeLowLatencyCombine));
    d["uccl_moe_dispatch"] =
        make_capsule(reinterpret_cast<void*>(&Handler_MoeDispatch));
    d["uccl_moe_combine"] =
        make_capsule(reinterpret_cast<void*>(&Handler_MoeCombine));
    d["uccl_moe_internode_dispatch"] =
        make_capsule(reinterpret_cast<void*>(&Handler_MoeInternodeDispatch));
    d["uccl_moe_internode_combine"] =
        make_capsule(reinterpret_cast<void*>(&Handler_MoeInternodeCombine));
    d["uccl_moe_cached_dispatch"] =
        make_capsule(reinterpret_cast<void*>(&Handler_MoeCachedDispatch));
    d["uccl_moe_internode_cached_dispatch"] = make_capsule(
        reinterpret_cast<void*>(&Handler_MoeInternodeCachedDispatch));
    return d;
  });
}

}  // namespace uccl_jax_ffi
