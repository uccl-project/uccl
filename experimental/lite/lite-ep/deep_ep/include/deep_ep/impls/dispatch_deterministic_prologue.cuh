#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <cooperative_groups.h>

namespace deep_ep::elastic {

// TODO: support scale-out
template <int kNumSMs, int kNumWarps, int kNumScaleupRanks,
          int kNumMaxTokensPerRank, int kNumExperts, int kNumTopk,
          int kNumThreads = kNumWarps * 32>
__global__ void __launch_bounds__(kNumThreads, 1)
    dispatch_deterministic_prologue_impl(topk_idx_t* topk_idx,
                                         int* rank_count_buffer,
                                         int* dst_buffer_slot_idx,
                                         int const num_tokens,
                                         int const scaleup_rank_idx) {
  constexpr int kNumExpertsPerRank = kNumExperts / kNumScaleupRanks;
  EP_STATIC_ASSERT(kNumExperts % kNumScaleupRanks == 0,
                   "Invalid number of experts or ranks");

  // Utils
  auto const sm_idx = static_cast<int>(blockIdx.x),
             thread_idx = static_cast<int>(threadIdx.x);
  auto const warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();
  auto const global_warp_idx = sm_idx * kNumWarps + warp_idx;

  // Token region the current warp is responsible for
  auto const num_tokens_per_warp =
      math::ceil_div(num_tokens, kNumSMs * kNumWarps);
  auto const start_token_idx = global_warp_idx * num_tokens_per_warp;
  auto const end_token_idx =
      min(start_token_idx + num_tokens_per_warp, num_tokens);

  // Group configs
  // NOTES: Group refers to the tokens that each warp handles concurrently
  constexpr int kNumTokensPerGroup = 32 / kNumTopk;
  auto const token_idx_offset = lane_idx / kNumTopk;
  unsigned const token_mask = ((1u << kNumTopk) - 1)
                              << (token_idx_offset * kNumTopk);
  EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k");

  // Shared memory for reduction
  // NOTES: Each warp owns separate shared memory region for separate sum.
  extern __shared__ int8_t smem[];
  auto const rank_count_global_psum = math::advance_ptr<int>(smem, 0);
  auto const rank_count_warp_sum = math::advance_ptr<int>(
      rank_count_global_psum,
      (kNumScaleupRanks + warp_idx * kNumScaleupRanks) * sizeof(int));
  auto const rank_count_warp_psum = math::advance_ptr<int>(
      rank_count_warp_sum, kNumWarps * kNumScaleupRanks * sizeof(int));

  // Initialize to zero before reduce
  for (int i = thread_idx; i < kNumScaleupRanks * (1 + 2 * kNumWarps);
       i += kNumThreads)
    reinterpret_cast<int*>(smem)[i] = 0;
  __syncthreads();

  // Util functions
  auto const map_expert_to_rank_idx = [&](int const& expert_idx) {
    return expert_idx >= 0 ? expert_idx / kNumExpertsPerRank : -1;
  };
  auto const is_unique = [&](int const& rank_idx) {
    return ((ptx::match(rank_idx) & token_mask) >> lane_idx) == 1;
  };
  auto const count_ones_before = [&](unsigned const& mask, int const& bit_idx) {
    return __popc(mask & ((1u << bit_idx) - 1));
  };
  auto const get_other_rank_count_warp_sum = [&](int const& other_warp_idx) {
    // NOTES: pass negative num_bytes to advance pointer
    return math::advance_ptr<int>(
        rank_count_warp_sum,
        (other_warp_idx - warp_idx) * kNumScaleupRanks * sizeof(int));
  };

  // Each warp scan the tokens separately
  for (int i = start_token_idx; i < end_token_idx; i += kNumTokensPerGroup) {
    auto const token_idx = i + token_idx_offset;
    auto const is_active_thread =
        lane_idx < kNumTopk * kNumTokensPerGroup and token_idx < end_token_idx;
    int const expert_idx =
        is_active_thread
            ? static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx))
            : -1;
    auto const rank_idx = map_expert_to_rank_idx(expert_idx);

    // Avoid duplicate messages to a single rank
    auto const deduped_rank_idx = is_unique(rank_idx) ? rank_idx : -1;
    auto const rank_idx_mask = ptx::match(deduped_rank_idx);

    // Let the one with the largest lane index send the count
    if ((rank_idx_mask >> lane_idx) == 1 and deduped_rank_idx >= 0)
      rank_count_warp_sum[deduped_rank_idx] += __popc(rank_idx_mask);
  }
  __syncthreads();

  // Get block sum and store to global
  for (int rank_idx = thread_idx; rank_idx < kNumScaleupRanks;
       rank_idx += kNumThreads) {
    int rank_count_block_sum = 0;
    for (int i = 0; i < kNumWarps; i++)
      rank_count_block_sum += get_other_rank_count_warp_sum(i)[rank_idx];
    rank_count_buffer[sm_idx * kNumScaleupRanks + rank_idx] =
        rank_count_block_sum;
  }
  cooperative_groups::this_grid().sync();

  // Get the prefix sum before the current SM
  for (int rank_idx = lane_idx; rank_idx < kNumScaleupRanks; rank_idx += 32) {
    int rank_count = 0;
    for (int i = warp_idx; i < sm_idx; i += kNumWarps)
      rank_count += rank_count_buffer[i * kNumScaleupRanks + rank_idx];
    atomicAdd_block(rank_count_global_psum + rank_idx, rank_count);
  }
  __syncthreads();

  // Get each warp's prefix sum
  for (int rank_idx = lane_idx; rank_idx < kNumScaleupRanks; rank_idx += 32) {
    int rank_count = rank_count_global_psum[rank_idx];
    for (int i = 0; i < warp_idx; i++)
      rank_count += get_other_rank_count_warp_sum(i)[rank_idx];
    rank_count_warp_psum[rank_idx] = rank_count;
  }
  __syncwarp();

  // Each warp scan the tokens separately
  for (int i = start_token_idx; i < end_token_idx; i += kNumTokensPerGroup) {
    auto const token_idx = i + token_idx_offset;
    auto const is_active_thread =
        lane_idx < kNumTopk * kNumTokensPerGroup and token_idx < end_token_idx;
    auto const expert_idx =
        is_active_thread
            ? static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx))
            : -1;
    auto const rank_idx = map_expert_to_rank_idx(expert_idx);

    // Avoid duplicate messages to a single rank
    auto const deduped_rank_idx = is_unique(rank_idx) ? rank_idx : -1;
    auto const rank_idx_mask = ptx::match(deduped_rank_idx);

    // Store to target buffer
    auto const stored_dst_slot_idx =
        deduped_rank_idx >= 0 ? rank_count_warp_psum[deduped_rank_idx] +
                                    count_ones_before(rank_idx_mask, lane_idx)
                              : -1;
    auto const value =
        stored_dst_slot_idx >= 0
            ? scaleup_rank_idx * kNumMaxTokensPerRank + stored_dst_slot_idx
            : -1;
    if (is_active_thread) dst_buffer_slot_idx[i * kNumTopk + lane_idx] = value;

    // Let the one with the largest lane index send the count
    if ((rank_idx_mask >> lane_idx) == 1 and deduped_rank_idx >= 0)
      rank_count_warp_psum[deduped_rank_idx] += __popc(rank_idx_mask);
    __syncwarp();
  }
}

}  // namespace deep_ep::elastic
