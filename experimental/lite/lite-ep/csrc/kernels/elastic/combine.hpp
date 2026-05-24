#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <nccl.h>

namespace deep_ep::elastic {

class CombineRuntime final : public jit::LaunchRuntime<CombineRuntime> {
 public:
  struct Args {
    // Templated arguments
    bool is_scaleup_nvlink;
    bool use_expanded_layout, allow_multiple_reduction;
    int num_scaleup_warps, num_forward_warps;
    int num_scaleout_ranks, num_scaleup_ranks;
    int hidden;
    int num_max_tokens_per_rank;
    int num_experts;
    int num_topk;
    int num_qps;
    int64_t num_timeout_cycles;

    // Parameters
    nv_bfloat16* x;
    float* topk_weights;
    int* src_metadata;
    int* psum_num_recv_tokens_per_scaleup_rank;
    int* token_metadata_at_forward;
    int* channel_linked_list;
    ncclDevComm_t nccl_dev_comm;
    ncclWindow_t nccl_window;
    void* buffer;
    void* workspace;
    int scaleout_rank_idx, scaleup_rank_idx;
    int num_reduced_tokens;
    uint64_t const* uccl_d2h_channel_addrs;
    int uccl_num_d2h_channel_addrs;
    uint64_t* uccl_signal_shadow;
    uint64_t uccl_shared_per_rank_bytes;
    int uccl_intranode_local_world_size;
    int uccl_intranode_my_local_rank;
    int uccl_intranode_node_idx;

    jit::LaunchArgs launch_args;
  };

  static std::string generate_impl(Args const& args) {
    std::string header_name, func_name;
    if (args.num_scaleout_ranks == 1) {
      header_name = "combine";
      func_name = fmt::format(
          "combine_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
          args.is_scaleup_nvlink, args.use_expanded_layout,
          args.allow_multiple_reduction, args.launch_args.grid_dim.first,
          args.launch_args.num_threads / 32,
          args.num_scaleup_ranks * args.num_scaleout_ranks, args.hidden,
          args.num_max_tokens_per_rank, args.num_experts, args.num_topk,
          args.num_qps, args.num_timeout_cycles);
    } else {
      header_name = "hybrid_combine";
      func_name = fmt::format(
          "hybrid_combine_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, "
          "{}>",
          args.use_expanded_layout, args.allow_multiple_reduction,
          args.launch_args.grid_dim.first, args.num_scaleup_warps,
          args.num_forward_warps, args.num_scaleout_ranks,
          args.num_scaleup_ranks, args.hidden, args.num_max_tokens_per_rank,
          args.num_experts, args.num_topk, args.num_qps,
          args.num_timeout_cycles);
    }
    return fmt::format(R"(
#include <deep_ep/impls/{}.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{});
}}
)",
                       header_name, func_name);
  }

  static void launch_impl(jit::KernelHandle const& kernel,
                          jit::LaunchConfigHandle const& config, Args args) {
    if (args.num_scaleout_ranks == 1) {
      EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
          kernel, config, args.x, args.topk_weights, args.src_metadata,
          args.psum_num_recv_tokens_per_scaleup_rank, args.nccl_dev_comm,
          args.nccl_window, args.buffer, args.workspace, args.scaleup_rank_idx,
          args.num_reduced_tokens, args.uccl_d2h_channel_addrs,
          args.uccl_num_d2h_channel_addrs, args.uccl_signal_shadow,
          args.uccl_shared_per_rank_bytes, args.uccl_intranode_local_world_size,
          args.uccl_intranode_my_local_rank, args.uccl_intranode_node_idx));
    } else {
      EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
          kernel, config, args.x, args.topk_weights, args.src_metadata,
          args.psum_num_recv_tokens_per_scaleup_rank,
          args.token_metadata_at_forward, args.channel_linked_list,
          args.nccl_dev_comm, args.nccl_window, args.buffer, args.workspace,
          args.scaleout_rank_idx, args.scaleup_rank_idx,
          args.num_reduced_tokens, args.uccl_d2h_channel_addrs,
          args.uccl_num_d2h_channel_addrs, args.uccl_signal_shadow));
    }
  }
};

static layout::TokenLayout get_combine_token_layout(int const& hidden,
                                                    int const& elem_size,
                                                    int const& num_topk) {
  return layout::TokenLayout(hidden * elem_size, 0, num_topk, false);
}

static void* launch_combine(
    void* x, void* topk_weights, int* src_metadata,
    int* psum_num_recv_tokens_per_scaleup_rank, int* token_metadata_at_forward,
    int* channel_linked_list, ncclDevComm_t const& nccl_dev_comm,
    ncclWindow_t const& nccl_window, void* buffer, void* workspace,
    int const& num_reduced_tokens, int const& num_max_tokens_per_rank,
    int const& hidden, int const& num_experts, int const& num_topk,
    int const& num_qps, int64_t const& num_timeout_cycles,
    int const& num_scaleout_ranks, int const& num_scaleup_ranks,
    int const& scaleout_rank_idx, int const& scaleup_rank_idx,
    bool const& is_scaleup_nvlink, int const& num_sms,
    int const& num_smem_bytes, int const& num_channels,
    bool const& use_expanded_layout, bool const& allow_multiple_reduction,
    uint64_t const* uccl_d2h_channel_addrs,
    int const& uccl_num_d2h_channel_addrs, uint64_t* uccl_signal_shadow,
    uint64_t const& uccl_shared_per_rank_bytes,
    int const& uccl_intranode_local_world_size,
    int const& uccl_intranode_my_local_rank, int const& uccl_intranode_node_idx,
    at::cuda::CUDAStream const& stream) {
  // Maximize shared memory utilization
  auto const token_layout =
      get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);
  auto num_warps =
      std::min(num_smem_bytes / token_layout.get_num_bytes<true>(), 32);

  // Decide warps
  int num_scaleup_warps = 0, num_forward_warps = 0;
  if (num_scaleout_ranks > 1) {
    EP_HOST_ASSERT(num_channels % num_sms == 0 and
                   "Invalid number of channels or SMs, you may use a different "
                   "SM count than dispatch");
    EP_HOST_ASSERT(num_channels / num_sms <= 16);

    num_scaleup_warps = num_forward_warps = num_channels / num_sms;
    num_warps = num_scaleup_warps + num_forward_warps;
    EP_HOST_ASSERT(
        num_warps * token_layout.get_num_bytes<true>() <= num_smem_bytes and
        "Invalid combine SM count, please try to match your dispatch config");
  }

  // Generate, build and launch
  auto const num_threads = num_warps * 32;
  const CombineRuntime::Args args = {
      .is_scaleup_nvlink = is_scaleup_nvlink,
      .use_expanded_layout = use_expanded_layout,
      .allow_multiple_reduction = allow_multiple_reduction,
      .num_scaleup_warps = num_scaleup_warps,
      .num_forward_warps = num_forward_warps,
      .num_scaleout_ranks = num_scaleout_ranks,
      .num_scaleup_ranks = num_scaleup_ranks,
      .hidden = hidden,
      .num_max_tokens_per_rank = num_max_tokens_per_rank,
      .num_experts = num_experts,
      .num_topk = num_topk,
      .num_qps = num_qps,
      .num_timeout_cycles = num_timeout_cycles,
      .x = static_cast<nv_bfloat16*>(x),
      .topk_weights = static_cast<float*>(topk_weights),
      .src_metadata = src_metadata,
      .psum_num_recv_tokens_per_scaleup_rank =
          psum_num_recv_tokens_per_scaleup_rank,
      .token_metadata_at_forward = token_metadata_at_forward,
      .channel_linked_list = channel_linked_list,
      .nccl_dev_comm = nccl_dev_comm,
      .nccl_window = nccl_window,
      .buffer = buffer,
      .workspace = workspace,
      .scaleout_rank_idx = scaleout_rank_idx,
      .scaleup_rank_idx = scaleup_rank_idx,
      .num_reduced_tokens = num_reduced_tokens,
      .uccl_d2h_channel_addrs = uccl_d2h_channel_addrs,
      .uccl_num_d2h_channel_addrs = uccl_num_d2h_channel_addrs,
      .uccl_signal_shadow = uccl_signal_shadow,
      .uccl_shared_per_rank_bytes = uccl_shared_per_rank_bytes,
      .uccl_intranode_local_world_size = uccl_intranode_local_world_size,
      .uccl_intranode_my_local_rank = uccl_intranode_my_local_rank,
      .uccl_intranode_node_idx = uccl_intranode_node_idx,
      // NOTES: make cluster dim 2 to overlap with clustered computation kernels
      .launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes,
                                     2 - (num_sms % 2), true)};
  auto const code = CombineRuntime::generate(args);
  auto const runtime = jit::compiler->build("combine", code);
  CombineRuntime::launch(runtime, args, stream);

  // Return the buffer to be reduced
  if (num_scaleout_ranks == 1) return buffer;

  // For hybrid mode, we have to skip the scale-up buffer
  bool const is_scaleup_buffer_rank_layout =
      allow_multiple_reduction ? (num_scaleup_ranks <= num_topk) : false;
  auto const scaleup_buffer = layout::BufferLayout<false>(
      token_layout,
      is_scaleup_buffer_rank_layout ? num_scaleup_ranks : num_topk,
      num_scaleout_ranks * num_max_tokens_per_rank, buffer);
  return scaleup_buffer.get_buffer_end_ptr();
}

class CombineReduceEpilogueRuntime final
    : public jit::LaunchRuntime<CombineReduceEpilogueRuntime> {
 public:
  struct Args {
    // Templated arguments
    bool use_expanded_layout, allow_multiple_reduction;
    int num_scaleout_ranks, num_scaleup_ranks;
    int hidden;
    int num_max_tokens_per_rank;
    int num_experts, num_topk;

    // Parameters
    nv_bfloat16* combined_x;
    float* combined_topk_weights;
    topk_idx_t* combined_topk_idx;
    void* reduce_buffer;
    void* bias_0;
    void* bias_1;
    int num_combined_tokens;
    int scaleout_rank_idx, scaleup_rank_idx;

    jit::LaunchArgs launch_args;
  };

  static std::string generate_impl(Args const& args) {
    return fmt::format(
        R"(
#include <deep_ep/impls/combine_reduce_epilogue.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&combine_reduce_epilogue_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",
        args.use_expanded_layout, args.allow_multiple_reduction,
        args.launch_args.grid_dim.first, args.launch_args.num_threads / 32,
        args.num_scaleout_ranks, args.num_scaleup_ranks, args.hidden,
        args.num_max_tokens_per_rank, args.num_experts, args.num_topk);
  }

  static void launch_impl(jit::KernelHandle const& kernel,
                          jit::LaunchConfigHandle const& config, Args args) {
    EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
        kernel, config, args.combined_x, args.combined_topk_weights,
        args.combined_topk_idx, args.reduce_buffer, args.bias_0, args.bias_1,
        args.num_combined_tokens, args.scaleout_rank_idx,
        args.scaleup_rank_idx));
  }
};

static void launch_combine_reduce_epilogue(
    void* combined_x, float* combined_topk_weights,
    topk_idx_t* combined_topk_idx, int const& num_combined_tokens,
    int const& num_max_tokens_per_rank, int const& hidden,
    int const& num_experts, int const& num_topk, void* reduce_buffer,
    void* bias_0, void* bias_1, int const& num_scaleout_ranks,
    int const& num_scaleup_ranks, int const& scaleout_rank_idx,
    int const& scaleup_rank_idx, int const& num_sms, int const& num_smem_bytes,
    bool const& use_expanded_layout, bool const& allow_multiple_reduction,
    at::cuda::CUDAStream const& stream) {
  // Maximize shared memory utilization
  // Too many warps may cause performance degrade, so we limit into 1024
  auto const token_layout =
      layout::TokenLayout(hidden * sizeof(nv_bfloat16), 0, 0, false);
  auto const num_warps =
      std::min<int>(num_smem_bytes / token_layout.get_num_bytes<false>(), 32);
  auto const num_threads = num_warps * 32;

  // Generate, build and launch
  const CombineReduceEpilogueRuntime::Args args = {
      .use_expanded_layout = use_expanded_layout,
      .allow_multiple_reduction = allow_multiple_reduction,
      .num_scaleout_ranks = num_scaleout_ranks,
      .num_scaleup_ranks = num_scaleup_ranks,
      .hidden = hidden,
      .num_max_tokens_per_rank = num_max_tokens_per_rank,
      .num_experts = num_experts,
      .num_topk = num_topk,
      .combined_x = static_cast<nv_bfloat16*>(combined_x),
      .combined_topk_weights = combined_topk_weights,
      .combined_topk_idx = combined_topk_idx,
      .reduce_buffer = reduce_buffer,
      .bias_0 = bias_0,
      .bias_1 = bias_1,
      .num_combined_tokens = num_combined_tokens,
      .scaleout_rank_idx = scaleout_rank_idx,
      .scaleup_rank_idx = scaleup_rank_idx,
      .launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes, 1,
                                     false, true)};
  auto const code = CombineReduceEpilogueRuntime::generate(args);
  auto const runtime = jit::compiler->build("combine_reduce_epilogue", code);
  CombineReduceEpilogueRuntime::launch(runtime, args, stream);
}

}  // namespace deep_ep::elastic
