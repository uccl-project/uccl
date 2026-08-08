#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <nccl.h>
#include <nccl_device.h>

namespace deep_ep::elastic {

class BarrierRuntime final : public jit::LaunchRuntime<BarrierRuntime> {
 public:
  struct Args {
    // Templated arguments
    bool is_scaleup_nvlink;
    int num_scaleout_ranks, num_scaleup_ranks;
    int64_t num_timeout_cycles;

    // Parameters
    ncclDevComm_t nccl_dev_comm;
    ncclWindow_t nccl_window;
    void* workspace;
    int scaleout_rank_idx, scaleup_rank_idx;
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
    return fmt::format(R"(
#include <deep_ep/impls/barrier.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&barrier_impl<{}, {}, {}, {}, {}, {}>);
}}
)",
                       args.is_scaleup_nvlink, args.launch_args.grid_dim.first,
                       args.launch_args.num_threads, args.num_scaleout_ranks,
                       args.num_scaleup_ranks, args.num_timeout_cycles);
  }

  static void launch_impl(jit::KernelHandle const& kernel,
                          jit::LaunchConfigHandle const& config, Args args) {
    EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
        kernel, config, args.nccl_dev_comm, args.nccl_window, args.workspace,
        args.scaleout_rank_idx, args.scaleup_rank_idx,
        args.uccl_d2h_channel_addrs, args.uccl_num_d2h_channel_addrs,
        args.uccl_signal_shadow, args.uccl_shared_per_rank_bytes,
        args.uccl_intranode_local_world_size, args.uccl_intranode_my_local_rank,
        args.uccl_intranode_node_idx));
  }
};

static void launch_barrier(
    ncclDevComm_t const& nccl_dev_comm, ncclWindow_t const& nccl_window,
    void* workspace, int const& scaleout_rank_idx, int const& scaleup_rank_idx,
    int const& num_scaleout_ranks, int const& num_scaleup_ranks,
    int64_t const& num_timeout_cycles, bool const& is_scaleup_nvlink,
    uint64_t const* uccl_d2h_channel_addrs,
    int const& uccl_num_d2h_channel_addrs, uint64_t* uccl_signal_shadow,
    uint64_t const& uccl_shared_per_rank_bytes,
    int const& uccl_intranode_local_world_size,
    int const& uccl_intranode_my_local_rank, int const& uccl_intranode_node_idx,
    at::cuda::CUDAStream const& stream) {
  // Number of threads equals to the number of ranks
  constexpr auto kNumThreads = 512;

  // Generate, build and launch
  // NOTES: only the hybrid kernel needs 2 SMs
  auto const num_sms = num_scaleout_ranks > 1 ? 2 : 1;
  const BarrierRuntime::Args args = {
      .is_scaleup_nvlink = is_scaleup_nvlink,
      .num_scaleout_ranks = num_scaleout_ranks,
      .num_scaleup_ranks = num_scaleup_ranks,
      .num_timeout_cycles = num_timeout_cycles,
      .nccl_dev_comm = nccl_dev_comm,
      .nccl_window = nccl_window,
      .workspace = workspace,
      .scaleout_rank_idx = scaleout_rank_idx,
      .scaleup_rank_idx = scaleup_rank_idx,
      .uccl_d2h_channel_addrs = uccl_d2h_channel_addrs,
      .uccl_num_d2h_channel_addrs = uccl_num_d2h_channel_addrs,
      .uccl_signal_shadow = uccl_signal_shadow,
      .uccl_shared_per_rank_bytes = uccl_shared_per_rank_bytes,
      .uccl_intranode_local_world_size = uccl_intranode_local_world_size,
      .uccl_intranode_my_local_rank = uccl_intranode_my_local_rank,
      .uccl_intranode_node_idx = uccl_intranode_node_idx,
      .launch_args = jit::LaunchArgs(num_sms, kNumThreads, 0, 1, true)};
  auto const code = BarrierRuntime::generate(args);
  auto const runtime = jit::compiler->build("barrier", code);
  BarrierRuntime::launch(runtime, args, stream);
}

}  // namespace deep_ep::elastic
