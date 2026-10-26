#include "ep_configs.cuh"
#include "ep_launch.cuh"
#include "ep_utils.cuh"
#include "exception.cuh"
#include "layout.hpp"

namespace uccl {
namespace layout {

constexpr int kMaxHistogramRanks = 128;
constexpr int kMaxHistogramExperts = 1024;

__global__ void clear_dispatch_layout_counts(int* num_tokens_per_expert,
                                             int* num_tokens_per_rank,
                                             int* num_tokens_per_rdma_rank,
                                             int num_experts, int num_ranks) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num_experts) num_tokens_per_expert[index] = 0;
  if (index < num_ranks) num_tokens_per_rank[index] = 0;
  if (num_tokens_per_rdma_rank && index < num_ranks / NUM_MAX_NVL_PEERS)
    num_tokens_per_rdma_rank[index] = 0;
}

__global__ void get_dispatch_layout_histogram(int64_t const* topk_idx,
                                              int* num_tokens_per_expert,
                                              int* num_tokens_per_rank,
                                              int* num_tokens_per_rdma_rank,
                                              bool* is_token_in_rank,
                                              int num_tokens, int num_topk,
                                              int num_experts, int num_ranks) {
  extern __shared__ int partial_counts[];
  int num_rdma_ranks =
      num_tokens_per_rdma_rank ? num_ranks / NUM_MAX_NVL_PEERS : 0;
  int num_counts = num_experts + num_ranks + num_rdma_ranks;
  for (int i = threadIdx.x; i < num_counts; i += blockDim.x)
    partial_counts[i] = 0;
  __syncthreads();
  int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (token_idx < num_tokens) {
    uint64_t rank_mask[(kMaxHistogramRanks + 63) / 64] = {};
    int num_experts_per_rank = num_experts / num_ranks;
    for (int j = 0; j < num_topk; ++j) {
      int expert_idx = static_cast<int>(topk_idx[token_idx * num_topk + j]);
      if (expert_idx >= 0 && expert_idx < num_experts) {
        atomicAdd(partial_counts + expert_idx, 1);
        int rank_idx = expert_idx / num_experts_per_rank;
        rank_mask[rank_idx / 64] |= uint64_t{1} << (rank_idx % 64);
      }
    }
    for (int rank_idx = 0; rank_idx < num_ranks; ++rank_idx) {
      bool present = (rank_mask[rank_idx / 64] >> (rank_idx % 64)) & 1;
      is_token_in_rank[token_idx * num_ranks + rank_idx] = present;
      if (present) atomicAdd(partial_counts + num_experts + rank_idx, 1);
    }
    for (int rdma_rank_idx = 0; rdma_rank_idx < num_rdma_ranks;
         ++rdma_rank_idx) {
      bool present = false;
      for (int peer = 0; peer < NUM_MAX_NVL_PEERS; ++peer) {
        int rank_idx = rdma_rank_idx * NUM_MAX_NVL_PEERS + peer;
        present |= (rank_mask[rank_idx / 64] >> (rank_idx % 64)) & 1;
      }
      if (present)
        atomicAdd(partial_counts + num_experts + num_ranks + rdma_rank_idx, 1);
    }
  }
  __syncthreads();
  for (int index = threadIdx.x; index < num_counts; index += blockDim.x) {
    int count = partial_counts[index];
    if (count == 0) continue;
    if (index < num_experts) {
      atomicAdd(num_tokens_per_expert + index, count);
    } else if (index < num_experts + num_ranks) {
      atomicAdd(num_tokens_per_rank + index - num_experts, count);
    } else {
      atomicAdd(num_tokens_per_rdma_rank + index - num_experts - num_ranks,
                count);
    }
  }
}

template <int kNumThreads, int kNumExpertsPerSM, int kNumRanksPerSM>
__global__ void get_dispatch_layout(int64_t const* topk_idx,
                                    int* num_tokens_per_rank,
                                    int* num_tokens_per_rdma_rank,
                                    int* num_tokens_per_expert,
                                    bool* is_token_in_rank, int num_tokens,
                                    int num_topk, int num_ranks,
                                    int num_experts) {
  auto sm_id = static_cast<int>(blockIdx.x);
  auto thread_id = static_cast<int>(threadIdx.x);

  // Count expert statistics
  __shared__ int num_tokens_per_expert_per_thread[kNumThreads]
                                                 [kNumExpertsPerSM];
  int expert_begin_idx = sm_id * kNumExpertsPerSM,
      expert_end_idx = min(expert_begin_idx + kNumExpertsPerSM, num_experts);
  if (expert_begin_idx < expert_end_idx) {
// Per-thread count
#pragma unroll
    for (int i = 0; i < kNumExpertsPerSM; ++i)
      num_tokens_per_expert_per_thread[thread_id][i] = 0;
#pragma unroll
    for (int i = thread_id; i < num_tokens; i += kNumThreads) {
      auto shifted_topk_idx = topk_idx + i * num_topk;
#pragma unroll
      for (int j = 0, expert_idx; j < num_topk; ++j) {
        expert_idx = static_cast<int>(shifted_topk_idx[j]);
        if (expert_begin_idx <= expert_idx and expert_idx < expert_end_idx)
          ++num_tokens_per_expert_per_thread[thread_id]
                                            [expert_idx - expert_begin_idx];
      }
    }
    __syncthreads();

    // Sum up
    EP_STATIC_ASSERT(kNumExpertsPerSM <= kNumThreads,
                     "Too many experts per SM");
    int lane = thread_id % WARP_SIZE;
    for (int expert = thread_id / WARP_SIZE;
         expert_begin_idx + expert < expert_end_idx;
         expert += kNumThreads / WARP_SIZE) {
      int sum = 0;
#pragma unroll
      for (int i = lane; i < kNumThreads; i += WARP_SIZE)
        sum += num_tokens_per_expert_per_thread[i][expert];
      sum = warp_reduce_sum(sum);
      if (lane == 0) num_tokens_per_expert[expert_begin_idx + expert] = sum;
    }
    return;
  }

  if (num_tokens_per_rdma_rank != nullptr)
    EP_DEVICE_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0 and
                     num_ranks > NUM_MAX_NVL_PEERS);

  // Count rank statistics
  constexpr int kNumRDMARanksPerSM = kNumRanksPerSM / NUM_MAX_NVL_PEERS;
  __shared__ int num_tokens_per_rank_per_thread[kNumThreads][kNumRanksPerSM];
  __shared__ int num_tokens_per_rdma_rank_per_thread[kNumThreads]
                                                    [kNumRDMARanksPerSM];
  auto sm_begin = (num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM;
  int rank_begin_idx = (sm_id - sm_begin) * kNumRanksPerSM,
      rank_end_idx = min(rank_begin_idx + kNumRanksPerSM, num_ranks);
  int rdma_rank_begin_idx = rank_begin_idx / NUM_MAX_NVL_PEERS,
      rdma_rank_end_idx = rank_end_idx / NUM_MAX_NVL_PEERS;
  if (rank_begin_idx < rank_end_idx) {
    auto const num_expert_per_rank = num_experts / num_ranks;
    auto expert_begin = rank_begin_idx * num_expert_per_rank;
    auto expert_end = rank_end_idx * num_expert_per_rank;

// Per-thread count
#pragma unroll
    for (int i = 0; i < kNumRanksPerSM; ++i)
      num_tokens_per_rank_per_thread[thread_id][i] = 0;
#pragma unroll
    for (int i = 0; i < kNumRDMARanksPerSM; ++i)
      num_tokens_per_rdma_rank_per_thread[thread_id][i] = 0;
#pragma unroll
    for (int i = thread_id; i < num_tokens; i += kNumThreads) {
      auto shifted_topk_idx = topk_idx + i * num_topk;
      int is_in_rank[kNumRanksPerSM] = {0},
          is_in_rdma_rank[kNumRDMARanksPerSM] = {0};
#pragma unroll
      for (int j = 0, expert_idx, rank_idx; j < num_topk; ++j) {
        expert_idx = static_cast<int>(shifted_topk_idx[j]);
        if (expert_begin <= expert_idx and expert_idx < expert_end) {
          // Count single rank
          rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx;
          is_in_rank[rank_idx]++,
              is_in_rdma_rank[rank_idx / NUM_MAX_NVL_PEERS]++;
        }
      }

      auto shifted_is_token_in_rank = is_token_in_rank + i * num_ranks;
#pragma unroll
      for (int j = 0; j + rank_begin_idx < rank_end_idx; ++j) {
        shifted_is_token_in_rank[j + rank_begin_idx] = (is_in_rank[j] > 0);
        num_tokens_per_rank_per_thread[thread_id][j] += (is_in_rank[j] > 0);
      }

#pragma unroll
      for (int j = 0; j + rdma_rank_begin_idx < rdma_rank_end_idx; ++j)
        num_tokens_per_rdma_rank_per_thread[thread_id][j] +=
            (is_in_rdma_rank[j] > 0);
    }
    __syncthreads();

    // Sum up
    EP_STATIC_ASSERT(kNumRanksPerSM <= kNumThreads, "Too many ranks per SM");
    int lane = thread_id % WARP_SIZE;
    for (int rank = thread_id / WARP_SIZE; rank_begin_idx + rank < rank_end_idx;
         rank += kNumThreads / WARP_SIZE) {
      int sum = 0;
#pragma unroll
      for (int i = lane; i < kNumThreads; i += WARP_SIZE)
        sum += num_tokens_per_rank_per_thread[i][rank];
      sum = warp_reduce_sum(sum);
      if (lane == 0) num_tokens_per_rank[rank_begin_idx + rank] = sum;
    }

    if (num_tokens_per_rdma_rank != nullptr) {
      for (int rank = thread_id / WARP_SIZE;
           rdma_rank_begin_idx + rank < rdma_rank_end_idx;
           rank += kNumThreads / WARP_SIZE) {
        int sum = 0;
#pragma unroll
        for (int i = lane; i < kNumThreads; i += WARP_SIZE)
          sum += num_tokens_per_rdma_rank_per_thread[i][rank];
        sum = warp_reduce_sum(sum);
        if (lane == 0)
          num_tokens_per_rdma_rank[rdma_rank_begin_idx + rank] = sum;
      }
    }
  }
}

void get_dispatch_layout(int64_t const* topk_idx, int* num_tokens_per_rank,
                         int* num_tokens_per_rdma_rank,
                         int* num_tokens_per_expert, bool* is_token_in_rank,
                         int num_tokens, int num_topk, int num_ranks,
                         int num_experts, cudaStream_t stream) {
  constexpr int kNumThreads = 256, kNumExpertsPerSM = 4, kNumRanksPerSM = 8;
  // The token histogram amortizes its second launch for prefill batches
  // with at least eight routing slots per token.
  // These bounds keep the histogram below 5 KiB and rank membership in two
  // words.
  bool valid_rdma_layout =
      num_tokens_per_rdma_rank == nullptr ||
      (num_ranks > NUM_MAX_NVL_PEERS && num_ranks % NUM_MAX_NVL_PEERS == 0);
  if (NUM_MAX_NVL_PEERS == 8 && num_tokens >= 4096 && num_topk >= 8 &&
      num_experts >= 256 && num_experts <= kMaxHistogramExperts &&
      num_ranks >= 8 && num_ranks <= kMaxHistogramRanks &&
      num_experts % num_ranks == 0 && valid_rdma_layout) {
    int num_rdma_ranks =
        num_tokens_per_rdma_rank ? num_ranks / NUM_MAX_NVL_PEERS : 0;
    clear_dispatch_layout_counts<<<(num_experts + kNumThreads - 1) /
                                       kNumThreads,
                                   kNumThreads, 0, stream>>>(
        num_tokens_per_expert, num_tokens_per_rank, num_tokens_per_rdma_rank,
        num_experts, num_ranks);
    CUDA_CHECK(cudaGetLastError());
    get_dispatch_layout_histogram<<<
        (num_tokens + kNumThreads - 1) / kNumThreads, kNumThreads,
        (num_experts + num_ranks + num_rdma_ranks) * sizeof(int), stream>>>(
        topk_idx, num_tokens_per_expert, num_tokens_per_rank,
        num_tokens_per_rdma_rank, is_token_in_rank, num_tokens, num_topk,
        num_experts, num_ranks);
    CUDA_CHECK(cudaGetLastError());
    return;
  }
  int num_sms = ((num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM) +
                (num_ranks + kNumRanksPerSM - 1) / kNumRanksPerSM;
  EP_STATIC_ASSERT(kNumRanksPerSM % NUM_MAX_NVL_PEERS == 0,
                   "Invalid number of ranks per SM");

  SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
  LAUNCH_KERNEL(
      &cfg,
      (get_dispatch_layout<kNumThreads, kNumExpertsPerSM, kNumRanksPerSM>),
      topk_idx, num_tokens_per_rank, num_tokens_per_rdma_rank,
      num_tokens_per_expert, is_token_in_rank, num_tokens, num_topk, num_ranks,
      num_experts);
}

}  // namespace layout
}  // namespace uccl