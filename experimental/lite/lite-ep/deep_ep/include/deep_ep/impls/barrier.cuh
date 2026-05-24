#pragma once

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/ptx.cuh>

namespace deep_ep::elastic {

template <bool kIsScaleupNVLink, int kNumSMs, int kNumThreads,
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int64_t kNumTimeoutCycles>
__global__ void __launch_bounds__(kNumThreads, 1)
    barrier_impl(const ncclDevComm_t nccl_dev_comm,
                 const ncclWindow_t nccl_window, void* workspace,
                 int const scaleout_rank_idx, int const scaleup_rank_idx,
                 uint64_t const* uccl_d2h_channel_addrs,
                 int const uccl_num_d2h_channel_addrs,
                 uint64_t* uccl_signal_shadow,
                 const uint64_t uccl_shared_per_rank_bytes,
                 int const uccl_intranode_local_world_size,
                 int const uccl_intranode_my_local_rank,
                 int const uccl_intranode_node_idx) {
  auto const sm_idx = static_cast<int>(blockIdx.x),
             thread_idx = static_cast<int>(threadIdx.x);

  // Barrier only uses the first part of workspace, so making `num_experts` as 0
  // is fine
  auto const workspace_layout = layout::WorkspaceLayout(
      workspace, kNumScaleoutRanks, kNumScaleupRanks, 0);
  auto const rank_idx = scaleout_rank_idx * kNumScaleupRanks + scaleup_rank_idx;
  auto const gin = handle::NCCLGin(
      nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_GPU, workspace,
      uccl_d2h_channel_addrs, uccl_num_d2h_channel_addrs, rank_idx,
      uccl_signal_shadow, scaleout_rank_idx, scaleup_rank_idx,
      uccl_shared_per_rank_bytes, uccl_intranode_local_world_size,
      uccl_intranode_my_local_rank, uccl_intranode_node_idx);
  comm::gpu_barrier<kIsScaleupNVLink, kNumScaleoutRanks, kNumScaleupRanks,
                    kNumSMs, kNumThreads, comm::kFlushAllAllocatedQPs,
                    kNumTimeoutCycles, comm::kKernelBarrierTag, false, false,
                    false>(gin, workspace_layout, scaleout_rank_idx,
                           scaleup_rank_idx, sm_idx, thread_idx);
}

}  // namespace deep_ep::elastic
