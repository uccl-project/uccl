// Per-rank deduplicated low-latency dispatch and reduce-before-send combine.
//
// The default LL kernels send one message per (token, expert). When several of
// a token's top-k experts live on the same rank, the same bytes cross the
// network several times (about 4 copies at EP2 with top-8). Here a token goes
// once to each destination rank and carries its top-k ids and weights; the
// receiver expands it into the usual per-expert packed layout, so the expert
// GEMM input is unchanged. Combine mirrors this: the expert rank sums
// w_k * y_k over its local experts and sends one row per (token, rank); the
// source rank adds one partial per destination rank.
//
// Ordering: messages to a rank are spread over C proxy channels by slot
// (slot % C). Each channel ends with its own count (dispatch) or flag
// (combine) on the same channel, which the proxy posts on the same QP after
// the channel's writes, so a receiver that has seen all C signals of a source
// rank has all of its data.
#include "common.hpp"
#include "ep_configs.cuh"
#include "ep_launch.cuh"
#include "ep_runtime.cuh"
#include "ep_util.hpp"
#include "ep_utils.cuh"
#include "internode_ll.cuh"
#include "uccl_ibgda.cuh"
#include <algorithm>
#include <cstdlib>
#include <string>

// The kernels assume 32-lane warps (32-bit ballot masks, launches of
// kNumMaxWarpGroups * WARP_SIZE threads) and the NVIDIA FP8 conversions, so
// ROCm builds get host entry points that reject the call (end of this file).
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#endif

namespace uccl {
namespace internode_ll_dedup {

constexpr int kMaxChannels = 32;

int num_channels(int num_d2h_channel_addrs, int num_experts, int num_ranks) {
  int c = std::min(num_d2h_channel_addrs, num_experts / num_ranks);
  return std::max(1, std::min(c, kMaxChannels));
}

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)

constexpr int kNumMaxWarpGroups = 32;
constexpr int kMaxTopK = 8;
// Message header: int4 {src token, 0, 0, 0} | int16 expert ids[8] |
// float weights[8].
constexpr int kHeaderBytes = 4 * sizeof(int4);

// Dedup counters live in the upper half of the LL workspace so they never
// overlap the default kernels' counters. All of them return to zero at the
// end of every call.
__host__ __device__ inline int* dedup_workspace(void* workspace) {
  return static_cast<int*>(workspace) + NUM_WORKSPACE_BYTES / 2 / sizeof(int);
}

// Cooperative launch without the thread-block cluster attribute that
// SETUP_LAUNCH_CONFIG adds; these kernels use grid sync but no clusters.
#define DEDUP_LAUNCH_CONFIG(num_sms, num_threads, stream)                     \
  cudaLaunchConfig_t cfg = {(num_sms), (num_threads), 0, stream, nullptr, 0}; \
  cudaLaunchAttribute attr[1];                                                \
  attr[0].id = cudaLaunchAttributeCooperative;                                \
  attr[0].val.cooperative = 1;                                                \
  cfg.attrs = attr;                                                           \
  cfg.numAttrs = 1

__device__ __forceinline__ int find_expert(int4 const& ids4, int expert) {
  auto const ids = reinterpret_cast<int16_t const*>(&ids4);
  int hit = -1;
#pragma unroll
  for (int k = 0; k < kMaxTopK; ++k)
    if (ids[k] == expert) hit = k;
  return hit;
}

// Coherent load of a staged message the same warp wrote before a __syncwarp
// (the non-coherent path gives no such guarantee).
__device__ __forceinline__ int4 ld_staged(int4 const* ptr) { return *ptr; }

template <bool kUseFP8, bool kUseUE8M0, int kHidden>
__global__ __launch_bounds__(1024, 1) void dispatch(
    void* packed_recv_x, void* packed_recv_x_scales, int* packed_recv_src_info,
    int64_t* packed_recv_layout_range, int* packed_recv_count, int4* dedup_hdr,
    int* dedup_pos, int* dedup_count, void* rdma_recv_x, int* rdma_recv_count,
    int64_t* rdma_recv_count_internode, void* rdma_x, void const* x,
    int64_t const* topk_idx, float const* topk_weights, int* workspace,
    int* next_clean, int64_t* next_clean_second, int num_next_clean_int,
    int num_tokens, int num_max_dispatch_tokens_per_rank, int num_topk,
    int num_experts, int rank, int num_ranks, int num_channels,
    int num_warp_groups, int num_warps_per_group, bool round_scale, int phases,
    uint64_t const* d2h_channel_addrs, int num_d2h_channel_addrs,
    int max_nvl_peers, int low_latency_buffer_idx, void** ipc_rdma_base_ptrs,
    void* rdma_buffer_ptr, void* atomic_buffer_ptr, bool warp_per_token,
    uint8_t* ce_stage, int* ce_counts) {
  auto const sm_id = static_cast<int>(blockIdx.x);
  auto const num_sms = static_cast<int>(gridDim.x);
  auto const thread_id = static_cast<int>(threadIdx.x);
  auto const warp_id = thread_id / WARP_SIZE, lane_id = get_lane_id();
  auto const num_warps = num_warp_groups * num_warps_per_group;
  auto const num_local_experts = num_experts / num_ranks;
  auto const warp_group_id = warp_id / num_warps_per_group;
  auto const sub_warp_id = warp_id % num_warps_per_group;
  auto const C = num_channels;
  auto const num_recv_per_expert = num_ranks * num_max_dispatch_tokens_per_rank;
  // Workspace: combine finish counters [num_ranks * C] and clean flag, then
  // per LL buffer the slot counters [num_ranks] and a finished-blocks count.
  // A slot counter ends at the number of messages to its rank, so the count
  // senders read it once every block has issued its tokens.
  int* slot_counter =
      workspace + num_ranks * C + 1 + low_latency_buffer_idx * (num_ranks + 1);
  int* blocks_done = slot_counter + num_ranks;

  using scale_t = std::conditional_t<kUseUE8M0, uint8_t, float>;
  using packed_t = std::conditional_t<kUseUE8M0, uint32_t, float>;
  using vec_t = typename std::conditional<kUseFP8, int2, int4>::type;
  constexpr int kNumPerChannels = 128;
  constexpr int kNumScales = kHidden / kNumPerChannels;
  constexpr int kHiddenBytes =
      kHidden * (kUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));
  constexpr int kHiddenInt4 = kHiddenBytes / sizeof(int4);
  constexpr int kMsgBytes =
      kHeaderBytes + (kUseFP8 ? (kHidden + kNumScales * sizeof(float))
                              : (kHidden * sizeof(nv_bfloat16)));
  constexpr int kMsgInt4 = kMsgBytes / sizeof(int4);
  EP_STATIC_ASSERT(kMsgBytes % sizeof(int4) == 0, "Invalid message size");

  __shared__ int s_dst_rank[2][kMaxTopK];
  __shared__ int s_num_recv[kNumMaxWarpGroups];
  __shared__ int s_begin[kNumMaxWarpGroups];
  __shared__ int s_warp_hits[kNumMaxWarpGroups][kNumMaxWarpGroups];

  if ((phases & LOW_LATENCY_SEND_PHASE) == 0) goto DEDUP_DISPATCH_RECV;

  if (warp_id < num_warps - 1) {
    // Token warps stage each token once (header + FP8 cast) and send it once
    // per distinct destination rank. Two schedules: a whole block per token
    // (the FP8 cast spread over the block, one token in flight per SM), or
    // with `warp_per_token` one token per warp (SM-major numbering, up to one
    // token per warp in flight), which the host picks when the per-token
    // latency, not the link, bounds the send (NVLink peers, many tokens/SM).
    constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
    EP_STATIC_ASSERT(kHidden % (WARP_SIZE * kNumElemsPerRead) == 0,
                     "Invalid hidden");
    EP_STATIC_ASSERT(kMaxTopK <= WARP_SIZE, "Invalid top-k");
    int const num_token_warps = num_warps - 1;
    int const num_threads = num_token_warps * WARP_SIZE;
    int const hidden_bf16_int4 = kHidden / kNumElemsPerRead;
    if (sm_id == 0) {
      for (int i = thread_id; i < num_next_clean_int; i += num_threads) {
        next_clean[i] = 0;
        next_clean_second[i] = 0;
      }
      for (int i = thread_id; i < num_local_experts; i += num_threads)
        packed_recv_count[i] = 0;
    }
    if (warp_per_token) {
      for (int token_idx = warp_id * num_sms + sm_id; token_idx < num_tokens;
           token_idx += num_token_warps * num_sms) {
        auto const x_int4 =
            static_cast<int4 const*>(x) + token_idx * hidden_bf16_int4;
        auto const msg = static_cast<uint8_t*>(rdma_x) +
                         static_cast<size_t>(token_idx) * kMsgBytes;
        auto const msg_vec = reinterpret_cast<vec_t*>(msg + kHeaderBytes);
        auto const msg_scales =
            reinterpret_cast<float*>(msg + kHeaderBytes + kHiddenBytes);

        int const e = lane_id < num_topk
                          ? static_cast<int>(__ldg(
                                topk_idx + token_idx * num_topk + lane_id))
                          : -1;
        int const dst_rank = e >= 0 ? e / num_local_experts : -1;
        if (lane_id < kMaxTopK) {
          float const w =
              e >= 0 ? __ldg(topk_weights + token_idx * num_topk + lane_id)
                     : 0.f;
          reinterpret_cast<int16_t*>(msg + sizeof(int4))[lane_id] =
              static_cast<int16_t>(e);
          reinterpret_cast<float*>(msg + 2 * sizeof(int4))[lane_id] = w;
        }
        if (lane_id == 0)
          *reinterpret_cast<int4*>(msg) = make_int4(token_idx, 0, 0, 0);

        for (int i = lane_id; i < hidden_bf16_int4; i += WARP_SIZE) {
          auto int4_value = __ldg(x_int4 + i);
          if constexpr (kUseFP8) {
            auto bf16_values = reinterpret_cast<nv_bfloat16*>(&int4_value);
            float fp32_values[kNumElemsPerRead];
            float amax = kFP8Margin, scale, scale_inv;
#pragma unroll
            for (int j = 0; j < kNumElemsPerRead; ++j) {
              fp32_values[j] = static_cast<float>(bf16_values[j]);
              amax = fmaxf(amax, fabsf(fp32_values[j]));
            }
            amax = warp_reduce_max<16>(amax);
            calculate_fp8_scales(amax, scale, scale_inv, round_scale);
            if (lane_id == 0 or lane_id == 16)
              msg_scales[i * kNumElemsPerRead / 128] = scale_inv;
            vec_t int2_value;
            auto fp8x2_values =
                reinterpret_cast<__nv_fp8x2_storage_t*>(&int2_value);
#pragma unroll
            for (int j = 0; j < kNumElemsPerRead; j += 2) {
              float2 fp32x2 = {fp32_values[j] * scale,
                               fp32_values[j + 1] * scale};
              fp8x2_values[j / 2] =
                  __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, __NV_E4M3);
            }
            msg_vec[i] = int2_value;
          } else {
            msg_vec[i] = *reinterpret_cast<vec_t*>(&int4_value);
          }
        }
        // The staged message is read back by other lanes of this warp below
        __syncwarp();

        // One send per distinct destination rank, in top-k order
        bool leader = dst_rank >= 0;
#pragma unroll
        for (int k = 0; k < kMaxTopK; ++k) {
          int const other = __shfl_sync(WARP_MASK, dst_rank, k);
          if (k < lane_id and other == dst_rank) leader = false;
        }
        for (unsigned mask = __ballot_sync(WARP_MASK, leader); mask;
             mask &= mask - 1) {
          int const r = __shfl_sync(WARP_MASK, dst_rank, __ffs(mask) - 1);
          int slot = lane_id == 0 ? atomicAdd(slot_counter + r, 1) : 0;
          slot = __shfl_sync(WARP_MASK, slot, 0);
          int const channel = slot % C;
          auto const dst_ptr =
              reinterpret_cast<uint64_t>(rdma_recv_x) +
              (static_cast<size_t>(rank) * num_max_dispatch_tokens_per_rank +
               slot) *
                  kMsgBytes;
          // Copy-engine mode: stage per destination locally; the host
          // copies each destination's run to the peer after this launch.
          auto const dst_p2p_ptr =
              ce_stage and r != rank
                  ? reinterpret_cast<uint64_t>(
                        ce_stage + (static_cast<size_t>(r) *
                                        num_max_dispatch_tokens_per_rank +
                                    slot) *
                                       kMsgBytes)
              : ipc_rdma_base_ptrs
                  ? uccl::get_ipc_p2p_ptr(dst_ptr, ipc_rdma_base_ptrs, rank, r,
                                          max_nvl_peers, 0)
                  : 0;
          if (dst_p2p_ptr == 0) {
            __threadfence_system();
            uccl::nvshmemi_ibgda_put_nbi_warp(
                dst_ptr - reinterpret_cast<uint64_t>(rdma_buffer_ptr),
                reinterpret_cast<uint64_t>(msg) -
                    reinterpret_cast<uint64_t>(rdma_buffer_ptr),
                kMsgBytes, r, /*channel=*/channel, lane_id, slot,
                d2h_channel_addrs, num_d2h_channel_addrs, false,
                low_latency_buffer_idx);
          } else {
            UNROLLED_WARP_COPY(
                8, lane_id, kMsgInt4, reinterpret_cast<int4*>(dst_p2p_ptr),
                reinterpret_cast<int4 const*>(msg), ld_staged, st_na_global);
          }
          __syncwarp();
        }
      }
    } else {
      int iter = 0;
      for (int token_idx = sm_id; token_idx < num_tokens;
           token_idx += num_sms, ++iter) {
        int const buf = iter & 1;
        auto const x_int4 =
            static_cast<int4 const*>(x) + token_idx * hidden_bf16_int4;
        auto const msg = static_cast<uint8_t*>(rdma_x) +
                         static_cast<size_t>(token_idx) * kMsgBytes;
        auto const msg_vec = reinterpret_cast<vec_t*>(msg + kHeaderBytes);
        auto const msg_scales =
            reinterpret_cast<float*>(msg + kHeaderBytes + kHiddenBytes);

        if (warp_id < kMaxTopK and lane_id == 0) {
          int const e = warp_id < num_topk
                            ? static_cast<int>(__ldg(
                                  topk_idx + token_idx * num_topk + warp_id))
                            : -1;
          float const w =
              e >= 0 ? __ldg(topk_weights + token_idx * num_topk + warp_id)
                     : 0.f;
          reinterpret_cast<int16_t*>(msg + sizeof(int4))[warp_id] =
              static_cast<int16_t>(e);
          reinterpret_cast<float*>(msg + 2 * sizeof(int4))[warp_id] = w;
          s_dst_rank[buf][warp_id] = e >= 0 ? e / num_local_experts : -1;
        }
        if (thread_id == 0)
          *reinterpret_cast<int4*>(msg) = make_int4(token_idx, 0, 0, 0);

        for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
          auto int4_value = __ldg(x_int4 + i);
          if constexpr (kUseFP8) {
            auto bf16_values = reinterpret_cast<nv_bfloat16*>(&int4_value);
            float fp32_values[kNumElemsPerRead];
            float amax = kFP8Margin, scale, scale_inv;
#pragma unroll
            for (int j = 0; j < kNumElemsPerRead; ++j) {
              fp32_values[j] = static_cast<float>(bf16_values[j]);
              amax = fmaxf(amax, fabsf(fp32_values[j]));
            }
            amax = warp_reduce_max<16>(amax);
            calculate_fp8_scales(amax, scale, scale_inv, round_scale);
            if (lane_id == 0 or lane_id == 16)
              msg_scales[i * kNumElemsPerRead / 128] = scale_inv;
            vec_t int2_value;
            auto fp8x2_values =
                reinterpret_cast<__nv_fp8x2_storage_t*>(&int2_value);
#pragma unroll
            for (int j = 0; j < kNumElemsPerRead; j += 2) {
              float2 fp32x2 = {fp32_values[j] * scale,
                               fp32_values[j + 1] * scale};
              fp8x2_values[j / 2] =
                  __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, __NV_E4M3);
            }
            msg_vec[i] = int2_value;
          } else {
            msg_vec[i] = *reinterpret_cast<vec_t*>(&int4_value);
          }
        }
        sync_barrier_1(num_threads);

        if (warp_id < num_topk) {
          int const dst_rank = s_dst_rank[buf][warp_id];
          bool leader = dst_rank >= 0;
          for (int k = 0; k < warp_id; ++k)
            leader &= s_dst_rank[buf][k] != dst_rank;
          if (leader) {
            int slot = lane_id == 0 ? atomicAdd(slot_counter + dst_rank, 1) : 0;
            slot = __shfl_sync(WARP_MASK, slot, 0);
            int const channel = slot % C;
            auto const dst_ptr =
                reinterpret_cast<uint64_t>(rdma_recv_x) +
                (static_cast<size_t>(rank) * num_max_dispatch_tokens_per_rank +
                 slot) *
                    kMsgBytes;
            auto const dst_p2p_ptr =
                ce_stage and dst_rank != rank
                    ? reinterpret_cast<uint64_t>(
                          ce_stage + (static_cast<size_t>(dst_rank) *
                                          num_max_dispatch_tokens_per_rank +
                                      slot) *
                                         kMsgBytes)
                : ipc_rdma_base_ptrs
                    ? uccl::get_ipc_p2p_ptr(dst_ptr, ipc_rdma_base_ptrs, rank,
                                            dst_rank, max_nvl_peers, 0)
                    : 0;
            if (dst_p2p_ptr == 0) {
              __threadfence_system();
              uccl::nvshmemi_ibgda_put_nbi_warp(
                  dst_ptr - reinterpret_cast<uint64_t>(rdma_buffer_ptr),
                  reinterpret_cast<uint64_t>(msg) -
                      reinterpret_cast<uint64_t>(rdma_buffer_ptr),
                  kMsgBytes, dst_rank, /*channel=*/channel, lane_id, slot,
                  d2h_channel_addrs, num_d2h_channel_addrs, false,
                  low_latency_buffer_idx);
            } else {
              UNROLLED_WARP_COPY(8, lane_id, kMsgInt4,
                                 reinterpret_cast<int4*>(dst_p2p_ptr),
                                 reinterpret_cast<int4 const*>(msg),
                                 ld_nc_global, st_na_global);
            }
            __syncwarp();
          }
        }
      }
    }
    // Every message of this block is issued (and block 0 has cleaned)
    sync_barrier_1(num_threads);
    if (thread_id == 0) atomic_add_release_global(blocks_done, 1);
  } else {
    // Last warp: once every block has issued its tokens, the slot counters
    // hold the per-rank message counts; send each channel's count after the
    // channel's messages (same proxy channel, so the same QP order).
    for (int u = sm_id; u < num_ranks * C; u += num_sms) {
      int const dst_rank = u / C, channel = u % C;
      if (lane_id == 0) {
        while (ld_acquire_global(blocks_done) != num_sms)
          ;
        int const count = ld_acquire_global(slot_counter + dst_rank);
        int const num_sent = count / C + (channel < count % C ? 1 : 0);
        auto const dst_ptr =
            reinterpret_cast<uint64_t>(rdma_recv_count + rank * C + channel);
        auto const dst_ptr_internode = reinterpret_cast<uint64_t>(
            rdma_recv_count_internode + rank * C + channel);
        auto const dst_p2p_ptr =
            ipc_rdma_base_ptrs
                ? uccl::get_ipc_p2p_ptr(dst_ptr, ipc_rdma_base_ptrs, rank,
                                        dst_rank, max_nvl_peers, 0)
                : 0;
        if (ce_counts and dst_rank != rank) {
          // Copied to the peer after this launch, behind the data
          ce_counts[dst_rank * C + channel] = -num_sent - 1;
        } else if (dst_p2p_ptr == 0) {
          uccl::nvshmemi_ibgda_amo_nonfetch_add(
              dst_ptr_internode, reinterpret_cast<uint64_t>(atomic_buffer_ptr),
              -num_sent - 1, dst_rank, /*channel=*/channel, false,
              d2h_channel_addrs, num_d2h_channel_addrs, false,
              low_latency_buffer_idx);
        } else {
          st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr),
                                -num_sent - 1);
        }
      }
      __syncwarp();
    }
  }

DEDUP_DISPATCH_RECV:
  if ((phases & LOW_LATENCY_RECV_PHASE) == 0) return;

  // Send-and-recv launches need every send (and the `packed_recv_count`
  // reset) to finish before receiving; a separate recv launch gets this from
  // the kernel boundary.
  if (phases & LOW_LATENCY_SEND_PHASE) cg::this_grid().sync();
  if (sm_id == 0 and thread_id <= num_ranks) slot_counter[thread_id] = 0;

  {
    int const unit = sm_id * num_warp_groups + warp_group_id;
    if (unit >= num_experts) return;
    int const src_rank = unit / num_local_experts;
    int const local_expert_idx = unit % num_local_experts;
    int const global_expert_idx = rank * num_local_experts + local_expert_idx;
    bool const same_node = src_rank / max_nvl_peers == rank / max_nvl_peers;
    auto const src_base = static_cast<uint8_t const*>(rdma_recv_x) +
                          static_cast<size_t>(src_rank) *
                              num_max_dispatch_tokens_per_rank * kMsgBytes;

    // Wait for every channel's count from `src_rank`
    if (sub_warp_id == 0) {
      int num = 0;
      if (lane_id < C) {
        int value = 0;
        if (same_node) {
          while ((value = ld_acquire_sys_global(rdma_recv_count + src_rank * C +
                                                lane_id)) == 0)
            ;
        } else {
          while (
              (value = static_cast<int>(
                   ld_acquire_sys_global(reinterpret_cast<uint64_t const*>(
                       rdma_recv_count_internode + src_rank * C + lane_id)))) ==
              0)
            ;
        }
        num = -value - 1;
      }
      num = warp_reduce_sum(num);
      if (lane_id == 0) s_num_recv[warp_group_id] = num;
    }
    sync_barrier<true>(warp_group_id + 2, num_warps_per_group * WARP_SIZE);
    int const num_recv = s_num_recv[warp_group_id];

    // Pass 1: tokens from `src_rank` routed to this expert, per warp
    int hits = 0;
    for (int i0 = sub_warp_id * WARP_SIZE; i0 < num_recv;
         i0 += num_warps_per_group * WARP_SIZE) {
      int const i = i0 + lane_id;
      bool hit = false;
      if (i < num_recv)
        hit = find_expert(ld_nc_global(reinterpret_cast<int4 const*>(
                                           src_base + i * kMsgBytes) +
                                       1),
                          global_expert_idx) >= 0;
      hits += __popc(__ballot_sync(WARP_MASK, hit));
    }
    if (lane_id == 0) s_warp_hits[warp_group_id][sub_warp_id] = hits;
    sync_barrier<true>(warp_group_id + 2, num_warps_per_group * WARP_SIZE);
    int warp_offset = 0, num_hits = 0;
    for (int w = 0; w < num_warps_per_group; ++w) {
      int const v = s_warp_hits[warp_group_id][w];
      warp_offset += w < sub_warp_id ? v : 0;
      num_hits += v;
    }
    if (sub_warp_id == 0 and lane_id == 0) {
      int const begin =
          atomicAdd(packed_recv_count + local_expert_idx, num_hits);
      s_begin[warp_group_id] = begin;
      packed_recv_layout_range[local_expert_idx * num_ranks + src_rank] =
          pack2<int, int64_t>(num_hits, begin);
      if (local_expert_idx == 0) dedup_count[src_rank] = num_recv;
    }
    sync_barrier<true>(warp_group_id + 2, num_warps_per_group * WARP_SIZE);

    // Pass 2: expand into the per-expert packed layout
    auto const recv_x_int4 = static_cast<int4*>(packed_recv_x) +
                             static_cast<size_t>(local_expert_idx) *
                                 num_recv_per_expert * kHiddenInt4;
    auto const recv_src_info =
        packed_recv_src_info + local_expert_idx * num_recv_per_expert;
    auto const num_aligned_scales =
        align<int>(kNumScales, sizeof(float) / sizeof(scale_t));
    auto const recv_x_scales = static_cast<scale_t*>(packed_recv_x_scales) +
                               static_cast<size_t>(local_expert_idx) *
                                   num_recv_per_expert * num_aligned_scales;
    int pos = s_begin[warp_group_id] + warp_offset;
    for (int i0 = sub_warp_id * WARP_SIZE; i0 < num_recv;
         i0 += num_warps_per_group * WARP_SIZE) {
      int const i = i0 + lane_id;
      int slot_k = -1;
      if (i < num_recv)
        slot_k = find_expert(
            ld_nc_global(
                reinterpret_cast<int4 const*>(src_base + i * kMsgBytes) + 1),
            global_expert_idx);
      auto mask = __ballot_sync(WARP_MASK, slot_k >= 0);
      while (mask) {
        int const l = __ffs(mask) - 1;
        mask &= mask - 1;
        int const token = i0 + l;
        int const k = __shfl_sync(WARP_MASK, slot_k, l);
        auto const msg = src_base + static_cast<size_t>(token) * kMsgBytes;
        if (lane_id == 0) {
          recv_src_info[pos] = ld_nc_global(reinterpret_cast<int const*>(msg));
          dedup_pos[(src_rank * num_max_dispatch_tokens_per_rank + token) *
                        kMaxTopK +
                    k] = local_expert_idx * num_recv_per_expert + pos;
        }
        auto const src_data = reinterpret_cast<int4 const*>(msg + kHeaderBytes);
        UNROLLED_WARP_COPY(7, lane_id, kHiddenInt4,
                           recv_x_int4 + static_cast<size_t>(pos) * kHiddenInt4,
                           src_data, ld_nc_global, st_na_global);
        if constexpr (kUseFP8) {
          auto const src_scales =
              reinterpret_cast<float const*>(msg + kHeaderBytes + kHiddenBytes);
          auto const num_elems_per_pack =
              static_cast<int>(sizeof(packed_t) / sizeof(scale_t));
          auto const pack_stride = num_recv_per_expert * num_elems_per_pack;
          for (int s = lane_id; s < kNumScales; s += WARP_SIZE) {
            auto const pack_idx = s / num_elems_per_pack;
            auto const elem_idx = s % num_elems_per_pack;
            recv_x_scales[pos * num_elems_per_pack + pack_idx * pack_stride +
                          elem_idx] =
                extract_required_scale_format<kUseUE8M0>(
                    ld_nc_global(src_scales + s));
          }
        }
        ++pos;
      }
    }

    // One unit per source rank keeps the headers for combine
    if (local_expert_idx == 0) {
      auto const dst = dedup_hdr + static_cast<size_t>(src_rank) *
                                       num_max_dispatch_tokens_per_rank * 4;
      for (int i = sub_warp_id * WARP_SIZE + lane_id; i < num_recv * 4;
           i += num_warps_per_group * WARP_SIZE)
        dst[i] = ld_nc_global(
            reinterpret_cast<int4 const*>(src_base + (i / 4) * kMsgBytes) +
            (i % 4));
    }
  }
}

void dispatch(void* packed_recv_x, void* packed_recv_x_scales,
              int* packed_recv_src_info, int64_t* packed_recv_layout_range,
              int* packed_recv_count, void* dedup_hdr, int* dedup_pos,
              int* dedup_count, void* rdma_recv_x, int* rdma_recv_count,
              int64_t* rdma_recv_count_internode, void* rdma_x, void const* x,
              int64_t const* topk_idx, float const* topk_weights,
              int* next_clean, int64_t* next_clean_second,
              int num_next_clean_int, int num_tokens, int hidden,
              int num_max_dispatch_tokens_per_rank, int num_topk,
              int num_experts, int rank, int num_ranks, bool use_fp8,
              bool round_scale, bool use_ue8m0, void* workspace,
              int num_device_sms, cudaStream_t stream, int phases,
              uint64_t const* d2h_channel_addrs, int num_d2h_channel_addrs,
              int max_nvl_peers, int low_latency_buffer_idx,
              void** ipc_rdma_base_ptrs, void* rdma_buffer_ptr,
              void* atomic_buffer_ptr, uint8_t* ce_stage, int* ce_counts) {
  int const num_warp_groups = ceil_div(num_experts, num_device_sms);
  int const num_warps_per_group = kNumMaxWarpGroups / num_warp_groups;
  auto const num_warps = num_warp_groups * num_warps_per_group;
  auto const num_sms = ceil_div(num_experts, num_warp_groups);
  int const C = num_channels(num_d2h_channel_addrs, num_experts, num_ranks);
  EP_HOST_ASSERT(num_topk <= kMaxTopK and num_topk + 1 <= num_warps);
  EP_HOST_ASSERT(num_experts / num_ranks >= 2 and
                 "dedup messages need at least two local experts of space");
  EP_HOST_ASSERT(num_experts < 32768 and num_ranks * C <= num_experts);
  EP_HOST_ASSERT(num_warps_per_group >= 1 and num_warp_groups < 14);
  if (use_ue8m0)
    EP_HOST_ASSERT(round_scale and "UE8M0 SF requires `round_scale=True`");
  int* ws = dedup_workspace(workspace);
  // One token per warp only pays when every peer is an NVLink peer (the SMs
  // move the data, so per-token latency bounds the send) and each SM has
  // enough tokens that the per-warp FP8 cast of a whole row is hidden.
  // UCCL_EP_DEDUP_SEND=warp|block overrides.
  bool warp_per_token = ipc_rdma_base_ptrs != nullptr and
                        num_ranks <= max_nvl_peers and
                        static_cast<int64_t>(num_tokens) * 256 >=
                            static_cast<int64_t>(hidden) * num_sms;
  static int const forced = [] {  // read once: getenv is slow on a slow host
    char const* mode = std::getenv("UCCL_EP_DEDUP_SEND");
    return mode == nullptr ? -1 : std::string(mode) == "warp" ? 1 : 0;
  }();
  if (forced >= 0) warp_per_token = forced == 1;

#define DEDUP_DISPATCH_LAUNCH_CASE(hidden)                                     \
  {                                                                            \
    auto dispatch_func = dispatch<false, false, hidden>;                       \
    if (use_fp8 and not use_ue8m0)                                             \
      dispatch_func = dispatch<true, false, hidden>;                           \
    if (use_fp8 and use_ue8m0) dispatch_func = dispatch<true, true, hidden>;   \
    LAUNCH_KERNEL(                                                             \
        &cfg, dispatch_func, packed_recv_x, packed_recv_x_scales,              \
        packed_recv_src_info, packed_recv_layout_range, packed_recv_count,     \
        static_cast<int4*>(dedup_hdr), dedup_pos, dedup_count, rdma_recv_x,    \
        rdma_recv_count, rdma_recv_count_internode, rdma_x, x, topk_idx,       \
        topk_weights, ws, next_clean, next_clean_second, num_next_clean_int,   \
        num_tokens, num_max_dispatch_tokens_per_rank, num_topk, num_experts,   \
        rank, num_ranks, C, num_warp_groups, num_warps_per_group, round_scale, \
        phases, d2h_channel_addrs, num_d2h_channel_addrs, max_nvl_peers,       \
        low_latency_buffer_idx, ipc_rdma_base_ptrs, rdma_buffer_ptr,           \
        atomic_buffer_ptr, warp_per_token, ce_stage, ce_counts);               \
  }                                                                            \
  break

  DEDUP_LAUNCH_CONFIG(num_sms, num_warps * WARP_SIZE, stream);
  SWITCH_HIDDEN(DEDUP_DISPATCH_LAUNCH_CASE);
#undef DEDUP_DISPATCH_LAUNCH_CASE
}

// Combine compression for rows sent over RDMA: a per-row window code.
// The row's mean nonzero exponent fixes a window of 7 exponents
// [mean - 4, mean + 2]; exponents inside it become 3-bit codes, the rest
// (either side) are escapes (code 7, raw exponent byte in a per-lane list).
// Exact. A row goes raw only when packing would not be smaller. (A window
// from the row maximum breaks on outlier rows; a mean over all exponents
// breaks on rows with exact zeros.)
// Message: int4 {flags | base << 8, escapes, bytes, 0} | sign+mantissa byte
// per element (lane-major uint2 per 8 elements) | code words (3 per 4
// iterations per lane) | escape bytes (per lane, in element order).
constexpr int kZipHeader = sizeof(int4);
constexpr int kZipRaw = 0, kZipPacked = 1;

template <int kHidden>
struct ZipLayout {
  static constexpr int kIters = kHidden / 8 / WARP_SIZE;  // int4 per lane
  static constexpr int kGroups = (kIters + 3) / 4;        // 4 iterations each
  static constexpr int kWords = kGroups * 3;              // code words per lane
  static constexpr int kCodeOffset = kZipHeader + kHidden;
  static constexpr int kEscOffset = kCodeOffset + kWords * WARP_SIZE * 4;
  static constexpr int kStride = kZipHeader + kHidden * 2;   // a raw row fits
  static constexpr int kMaxEsc = kStride - 16 - kEscOffset;  // packed < raw
  static_assert(kHidden % (8 * WARP_SIZE) == 0,
                "row must be whole int4 per lane");
};

__device__ __forceinline__ int warp_exclusive_sum_int(int v, int& total) {
  int x = v;
#pragma unroll
  for (int d = 1; d < WARP_SIZE; d <<= 1) {
    int const y = __shfl_up_sync(WARP_MASK, x, d);
    if (get_lane_id() >= d) x += y;
  }
  total = __shfl_sync(WARP_MASK, x, WARP_SIZE - 1);
  return x - v;
}

__device__ __forceinline__ int count_escapes24(uint32_t c24) {
  return __popc(c24 & (c24 >> 1) & (c24 >> 2) & 0x249249u);
}

// Encode the BF16 row at `raw` (lane-major int4: index it * 32 + lane) into
// `msg`; returns the message bytes, or -1 if the row should go raw.
template <int kHidden>
__device__ int zip_encode_row(int4 const* raw, uint8_t* msg, int lane_id) {
  using L = ZipLayout<kHidden>;
  int esum = 0, ecnt = 0;
  for (int it = 0; it < L::kIters; ++it) {
    int4 const v = raw[it * WARP_SIZE + lane_id];
    uint32_t const w[4] = {
        static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y),
        static_cast<uint32_t>(v.z), static_cast<uint32_t>(v.w)};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int const lo = (w[j] >> 7) & 0xFF, hi = (w[j] >> 23) & 0xFF;
      esum += lo + hi;
      ecnt += (lo != 0) + (hi != 0);
    }
  }
#pragma unroll
  for (int d = WARP_SIZE / 2; d > 0; d >>= 1) {
    esum += __shfl_xor_sync(WARP_MASK, esum, d);
    ecnt += __shfl_xor_sync(WARP_MASK, ecnt, d);
  }
  int const base =
      ecnt == 0 ? 0 : min(max((esum + ecnt / 2) / ecnt - 4, 0), 255 - 6);

  auto const sm_out = reinterpret_cast<uint2*>(msg + kZipHeader);
  auto const code_out = reinterpret_cast<uint32_t*>(msg + L::kCodeOffset);
  int nesc = 0;
  for (int g = 0; g < L::kGroups; ++g) {
    uint32_t c[4] = {0, 0, 0, 0};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      int const it = g * 4 + q;
      if (it < L::kIters) {
        int4 const v = raw[it * WARP_SIZE + lane_id];
        auto const h = reinterpret_cast<uint16_t const*>(&v);
        uint32_t sm[2] = {0, 0}, c24 = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          int const e = (h[k] >> 7) & 0xFF;
          uint32_t const code = static_cast<uint32_t>(e - base) <= 6u
                                    ? static_cast<uint32_t>(e - base)
                                    : 7u;
          c24 |= code << (3 * k);
          sm[k / 4] |=
              static_cast<uint32_t>(((h[k] >> 8) & 0x80) | (h[k] & 0x7F))
              << (8 * (k % 4));
        }
        nesc += count_escapes24(c24);
        sm_out[it * WARP_SIZE + lane_id] = make_uint2(sm[0], sm[1]);
        c[q] = c24;
      }
    }
    code_out[(g * 3 + 0) * WARP_SIZE + lane_id] = c[0] | (c[1] << 24);
    code_out[(g * 3 + 1) * WARP_SIZE + lane_id] = (c[1] >> 8) | (c[2] << 16);
    code_out[(g * 3 + 2) * WARP_SIZE + lane_id] = (c[2] >> 16) | (c[3] << 8);
  }
  int total;
  int const off = warp_exclusive_sum_int(nesc, total);
  if (total > L::kMaxEsc) return -1;
  if (nesc > 0) {
    uint8_t* esc = msg + L::kEscOffset + off;
    for (int it = 0; it < L::kIters; ++it) {
      int4 const v = raw[it * WARP_SIZE + lane_id];
      auto const h = reinterpret_cast<uint16_t const*>(&v);
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        int const e = (h[k] >> 7) & 0xFF;
        if (static_cast<uint32_t>(e - base) > 6u)
          *esc++ = static_cast<uint8_t>(e);
      }
    }
  }
  int const bytes = (L::kEscOffset + total + 15) & ~15;
  if (lane_id == 0)
    *reinterpret_cast<int4*>(msg) =
        make_int4(kZipPacked | (base << 8), total, bytes, 0);
  return bytes;
}

// One received row, raw or packed, as seen by one lane
struct ZipRow {
  uint8_t const* msg;
  uint8_t const* esc;  // this lane's next escape byte
  int base;
  bool packed;
};

template <int kHidden>
__device__ __forceinline__ ZipRow zip_open_row(uint8_t const* msg) {
  using L = ZipLayout<kHidden>;
  int4 const h = ld_nc_global(reinterpret_cast<int4 const*>(msg));
  ZipRow r{msg, nullptr, (h.x >> 8) & 0xFF, (h.x & 0xFF) == kZipPacked};
  if (r.packed) {
    auto const words = reinterpret_cast<uint32_t const*>(msg + L::kCodeOffset);
    int nesc = 0;
    for (int g = 0; g < L::kGroups; ++g) {
      uint32_t const w0 =
          ld_nc_global(words + (g * 3 + 0) * WARP_SIZE + get_lane_id());
      uint32_t const w1 =
          ld_nc_global(words + (g * 3 + 1) * WARP_SIZE + get_lane_id());
      uint32_t const w2 =
          ld_nc_global(words + (g * 3 + 2) * WARP_SIZE + get_lane_id());
      nesc += count_escapes24(w0 & 0xFFFFFF) +
              count_escapes24((w0 >> 24) | ((w1 & 0xFFFF) << 8)) +
              count_escapes24((w1 >> 16) | ((w2 & 0xFF) << 16)) +
              count_escapes24(w2 >> 8);
    }
    int total;
    r.esc = msg + L::kEscOffset + warp_exclusive_sum_int(nesc, total);
  }
  return r;
}

// Add rows `it0 .. it0 + 3` of one received row (lane-major) into `acc`
template <int kHidden>
__device__ __forceinline__ void zip_accumulate_group(ZipRow& r, int g,
                                                     float (&acc)[4][8],
                                                     int lane_id) {
  using L = ZipLayout<kHidden>;
  if (not r.packed) {
    auto const data = reinterpret_cast<int4 const*>(r.msg + kZipHeader);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      int const it = g * 4 + q;
      if (it < L::kIters) {
        int4 const v = ld_nc_global(data + it * WARP_SIZE + lane_id);
        auto const b = reinterpret_cast<nv_bfloat16 const*>(&v);
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[q][k] += static_cast<float>(b[k]);
      }
    }
    return;
  }
  auto const words = reinterpret_cast<uint32_t const*>(r.msg + L::kCodeOffset);
  uint32_t const w0 = ld_nc_global(words + (g * 3 + 0) * WARP_SIZE + lane_id);
  uint32_t const w1 = ld_nc_global(words + (g * 3 + 1) * WARP_SIZE + lane_id);
  uint32_t const w2 = ld_nc_global(words + (g * 3 + 2) * WARP_SIZE + lane_id);
  uint32_t const c[4] = {w0 & 0xFFFFFF, (w0 >> 24) | ((w1 & 0xFFFF) << 8),
                         (w1 >> 16) | ((w2 & 0xFF) << 16), w2 >> 8};
  auto const sm_in = reinterpret_cast<uint2 const*>(r.msg + kZipHeader);
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    int const it = g * 4 + q;
    if (it < L::kIters) {
      uint2 const smv = ld_nc_global(sm_in + it * WARP_SIZE + lane_id);
      uint32_t const sm[2] = {smv.x, smv.y};
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        uint32_t const code = (c[q] >> (3 * k)) & 7;
        uint32_t const e =
            code == 7 ? *r.esc++ : static_cast<uint32_t>(r.base) + code;
        uint32_t const b = (sm[k / 4] >> (8 * (k % 4))) & 0xFF;
        uint32_t const bits = ((b & 0x80) << 8) | (e << 7) | (b & 0x7F);
        acc[q][k] += __uint_as_float(bits << 16);
      }
    }
  }
}

template <int kHidden, bool kCompress>
__global__ __launch_bounds__(1024, 1) void combine(
    void* combined_x, void* rdma_recv_x, int* rdma_recv_flag,
    int64_t* rdma_recv_flag_internode, void* rdma_send_x, void const* x,
    int64_t const* topk_idx, int4 const* dedup_hdr, int const* dedup_pos,
    int const* dedup_count, int* workspace, int* next_clean,
    int64_t* next_clean_second, int num_next_clean_int, int num_combined_tokens,
    int num_max_dispatch_tokens_per_rank, int num_topk, int num_experts,
    int rank, int num_ranks, int num_channels, int phases,
    uint64_t const* d2h_channel_addrs, int num_d2h_channel_addrs,
    int max_nvl_peers, int low_latency_buffer_idx, void** ipc_rdma_base_ptrs,
    void* rdma_buffer_ptr, void* atomic_buffer_ptr) {
  auto const sm_id = static_cast<int>(blockIdx.x);
  auto const num_sms = static_cast<int>(gridDim.x);
  auto const thread_id = static_cast<int>(threadIdx.x);
  auto const warp_id = thread_id / WARP_SIZE, lane_id = get_lane_id();
  auto const num_warps = static_cast<int>(blockDim.x) / WARP_SIZE;
  // SM-major warp numbering: consecutive rows land on different SMs, so a
  // few hundred rows still spread over the whole grid.
  auto const global_warp = warp_id * num_sms + sm_id;
  auto const num_global_warps = num_sms * num_warps;
  auto const num_local_experts = num_experts / num_ranks;
  auto const C = num_channels;
  int* finish_counter = workspace;              // [num_ranks * C]
  int* clean_flag = workspace + num_ranks * C;  // [1]

  constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
  constexpr int kRowBytes = kHidden * sizeof(nv_bfloat16);
  constexpr int kRowInt4 = kRowBytes / sizeof(int4);
  constexpr int kUnroll = 4;
  using Zip = ZipLayout<kHidden>;
  // Receive slots carry a header when compressing (raw rows fit, too)
  constexpr int kSlotBytes = kCompress ? Zip::kStride : kRowBytes;
  constexpr int kDataOffset = kCompress ? kZipHeader : 0;
  constexpr int kChunkInt4 = WARP_SIZE * kUnroll;  // one unrolled pass
  constexpr int kChunks = (kRowInt4 + kChunkInt4 - 1) / kChunkInt4;
  // Rows to NVLink peers (and to this rank) are split into chunks
  auto const row_chunks = [&](int r) {
    return (ipc_rdma_base_ptrs != nullptr and
            uccl::get_ipc_p2p_ptr(reinterpret_cast<uint64_t>(rdma_recv_x),
                                  ipc_rdma_base_ptrs, rank, r, max_nvl_peers,
                                  0) != 0)
               ? kChunks
               : 1;
  };

  if ((phases & LOW_LATENCY_SEND_PHASE) == 0) goto DEDUP_COMBINE_RECV;

  if (sm_id == 0 and warp_id == 0) {
    for (int i = lane_id; i < num_next_clean_int; i += WARP_SIZE) {
      next_clean[i] = 0;
      next_clean_second[i] = 0;
    }
    __syncwarp();
    if (lane_id == 0) atomic_add_release_global(clean_flag, num_ranks * C);
  }

  // One warp per received (source rank, token) row, or per chunk of
  // kChunkInt4 when the row goes to an NVLink peer (those rows are bound by
  // per-warp store latency, and big rows at small batches leave most warps
  // idle). Weighted sum over this rank's experts of the token, sent back to
  // the source rank. Rows of all source ranks form one index space.
  for (int flat = global_warp, num_items = -1;; flat += num_global_warps) {
    if (num_items < 0) {
      num_items = 0;
      for (int r = 0; r < num_ranks; ++r)
        num_items += __ldg(dedup_count + r) * row_chunks(r);
    }
    if (flat >= num_items) break;
    {
      int src_rank = 0, i = flat;
      for (int n;
           i >= (n = __ldg(dedup_count + src_rank) * row_chunks(src_rank));
           ++src_rank)
        i -= n;
      int const chunks = row_chunks(src_rank);
      int const chunk = i % chunks;
      i /= chunks;
      size_t const row =
          static_cast<size_t>(src_rank) * num_max_dispatch_tokens_per_rank + i;
      int4 const h0 = __ldg(dedup_hdr + row * 4);
      int4 const h1 = __ldg(dedup_hdr + row * 4 + 1);
      int4 const h2 = __ldg(dedup_hdr + row * 4 + 2);
      int4 const h3 = __ldg(dedup_hdr + row * 4 + 3);
      auto const ids = reinterpret_cast<int16_t const*>(&h1);
      float weights[kMaxTopK];
      *reinterpret_cast<int4*>(weights) = h2;
      *reinterpret_cast<int4*>(weights + 4) = h3;
      int src_pos[kMaxTopK];
      float src_w[kMaxTopK];
      int num_src = 0;
#pragma unroll
      for (int k = 0; k < kMaxTopK; ++k) {
        int const e = ids[k];
        if (e >= rank * num_local_experts and
            e < (rank + 1) * num_local_experts) {
          src_pos[num_src] = __ldg(dedup_pos + row * kMaxTopK + k);
          src_w[num_src] = weights[k];
          ++num_src;
        }
      }
      int const token_idx = h0.x;
      int const channel = i % C;
      auto const dst_ptr =
          reinterpret_cast<uint64_t>(rdma_recv_x) +
          (static_cast<size_t>(rank) * num_max_dispatch_tokens_per_rank +
           token_idx) *
              kSlotBytes;
      auto const dst_p2p_ptr =
          ipc_rdma_base_ptrs
              ? uccl::get_ipc_p2p_ptr(dst_ptr, ipc_rdma_base_ptrs, rank,
                                      src_rank, max_nvl_peers, 0)
              : 0;
      auto const stage = static_cast<uint8_t*>(rdma_send_x) + row * kSlotBytes;
      auto const out_slot =
          dst_p2p_ptr != 0 ? reinterpret_cast<uint8_t*>(dst_p2p_ptr) : stage;
      auto const out_row = reinterpret_cast<int4*>(out_slot + kDataOffset);
      if constexpr (kCompress) {
        if (lane_id == 0 and chunk == 0)
          *reinterpret_cast<int4*>(out_slot) =
              make_int4(kZipRaw, 0, Zip::kStride, 0);
      }
      // kUnroll independent loads per source row in flight per lane; a
      // chunked row covers one pass only
      int const base_begin = lane_id + (chunks > 1 ? chunk * kChunkInt4 : 0);
      int const base_end =
          chunks > 1 ? min(kRowInt4, (chunk + 1) * kChunkInt4) : kRowInt4;
      for (int base = base_begin; base < base_end;
           base += WARP_SIZE * kUnroll) {
        float acc[kUnroll][kNumElemsPerInt4] = {};
        for (int k = 0; k < num_src; ++k) {
          auto const src = static_cast<int4 const*>(x) +
                           static_cast<size_t>(src_pos[k]) * kRowInt4;
          int4 v[kUnroll];
#pragma unroll
          for (int u = 0; u < kUnroll; ++u) {
            int const idx = base + u * WARP_SIZE;
            v[u] = idx < kRowInt4 ? ld_nc_global(src + idx)
                                  : make_int4(0, 0, 0, 0);
          }
#pragma unroll
          for (int u = 0; u < kUnroll; ++u) {
            auto const v_bf16 = reinterpret_cast<nv_bfloat16 const*>(&v[u]);
#pragma unroll
            for (int j = 0; j < kNumElemsPerInt4; ++j)
              acc[u][j] += static_cast<float>(v_bf16[j]) * src_w[k];
          }
        }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          int const idx = base + u * WARP_SIZE;
          if (idx < kRowInt4) {
            int4 packed;
            auto const packed_bf16 = reinterpret_cast<nv_bfloat16*>(&packed);
#pragma unroll
            for (int j = 0; j < kNumElemsPerInt4; ++j)
              packed_bf16[j] = static_cast<nv_bfloat16>(acc[u][j]);
            out_row[idx] = packed;
          }
        }
      }
      __syncwarp();
      if (dst_p2p_ptr == 0) {
        // RDMA rows: compress into a second staging slot; NVLink and local
        // rows stay raw (their cost is SM stores, not bytes)
        uint8_t const* send_msg = stage;
        int send_bytes = kSlotBytes;
        if constexpr (kCompress) {
          auto const zstage = static_cast<uint8_t*>(rdma_send_x) +
                              (static_cast<size_t>(num_ranks) *
                                   num_max_dispatch_tokens_per_rank +
                               row) *
                                  kSlotBytes;
          int const bytes = zip_encode_row<kHidden>(out_row, zstage, lane_id);
          if (bytes > 0) send_msg = zstage, send_bytes = bytes;
          __syncwarp();
        }
        __threadfence_system();
        uccl::nvshmemi_ibgda_put_nbi_warp(
            dst_ptr - reinterpret_cast<uint64_t>(rdma_buffer_ptr),
            reinterpret_cast<uint64_t>(send_msg) -
                reinterpret_cast<uint64_t>(rdma_buffer_ptr),
            send_bytes, src_rank, /*channel=*/channel, lane_id, token_idx,
            d2h_channel_addrs, num_d2h_channel_addrs, true,
            low_latency_buffer_idx);
      }
      __syncwarp();
      if (lane_id == 0)
        atomic_add_release_global(finish_counter + src_rank * C + channel, 1);
    }
  }

  // One flag per (source rank, channel), after the channel's rows
  for (int u = global_warp; u < num_ranks * C; u += num_global_warps) {
    if (lane_id == 0) {
      int const dst_rank = u / C, channel = u % C;
      int const num_recv = __ldg(dedup_count + dst_rank);
      int const num_sent = (num_recv / C + (channel < num_recv % C ? 1 : 0)) *
                           row_chunks(dst_rank);
      while (ld_acquire_global(finish_counter + u) != num_sent)
        ;
      while (ld_acquire_global(clean_flag) == 0)
        ;
      auto const dst_ptr =
          reinterpret_cast<uint64_t>(rdma_recv_flag + rank * C + channel);
      auto const dst_ptr_internode = reinterpret_cast<uint64_t>(
          rdma_recv_flag_internode + rank * C + channel);
      auto const dst_p2p_ptr =
          ipc_rdma_base_ptrs
              ? uccl::get_ipc_p2p_ptr(dst_ptr, ipc_rdma_base_ptrs, rank,
                                      dst_rank, max_nvl_peers, 0)
              : 0;
      if (dst_p2p_ptr != 0) {
        st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), 1);
      } else {
        uccl::nvshmemi_ibgda_amo_nonfetch_add(
            dst_ptr_internode, reinterpret_cast<uint64_t>(atomic_buffer_ptr), 1,
            dst_rank, /*channel=*/channel, false, d2h_channel_addrs,
            num_d2h_channel_addrs, true, low_latency_buffer_idx);
      }
      finish_counter[u] = 0;
      atomic_add_release_global(clean_flag, -1);
    }
    __syncwarp();
  }

DEDUP_COMBINE_RECV:
  if ((phases & LOW_LATENCY_RECV_PHASE) == 0) return;

  for (int u = global_warp; u < num_ranks * C; u += num_global_warps) {
    if (lane_id == 0) {
      int const src_rank = u / C, channel = u % C;
      if (src_rank / max_nvl_peers == rank / max_nvl_peers) {
        while (ld_acquire_sys_global(rdma_recv_flag + src_rank * C + channel) ==
               0)
          ;
      } else {
        while (ld_acquire_sys_global(reinterpret_cast<uint64_t const*>(
                   rdma_recv_flag_internode + src_rank * C + channel)) == 0)
          ;
      }
    }
    __syncwarp();
  }
  cg::this_grid().sync();

  // One warp per own token: add one partial per destination rank
  for (int token_idx = global_warp; token_idx < num_combined_tokens;
       token_idx += num_global_warps) {
    uint64_t rank_mask = 0;
    for (int k = 0; k < num_topk; ++k) {
      auto const e =
          static_cast<int>(__ldg(topk_idx + token_idx * num_topk + k));
      if (e >= 0) rank_mask |= 1ull << (e / num_local_experts);
    }
    auto const out_row = static_cast<int4*>(combined_x) +
                         static_cast<size_t>(token_idx) * kRowInt4;
    if constexpr (kCompress) {
      // Same lane mapping and summation order as the raw path, so the
      // result is bit-identical to the uncompressed combine
      ZipRow rows[kMaxTopK];
      int num_rows = 0;
      for (uint64_t m = rank_mask; m; m &= m - 1) {
        int const r = __ffsll(static_cast<long long>(m)) - 1;
        rows[num_rows++] = zip_open_row<kHidden>(
            static_cast<uint8_t const*>(rdma_recv_x) +
            (static_cast<size_t>(r) * num_max_dispatch_tokens_per_rank +
             token_idx) *
                kSlotBytes);
      }
      for (int g = 0; g < Zip::kGroups; ++g) {
        float acc[4][8] = {};
        for (int j = 0; j < num_rows; ++j)
          zip_accumulate_group<kHidden>(rows[j], g, acc, lane_id);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          int const it = g * 4 + q;
          if (it < Zip::kIters) {
            int4 packed;
            auto const packed_bf16 = reinterpret_cast<nv_bfloat16*>(&packed);
#pragma unroll
            for (int k = 0; k < 8; ++k)
              packed_bf16[k] = static_cast<nv_bfloat16>(acc[q][k]);
            out_row[it * WARP_SIZE + lane_id] = packed;
          }
        }
      }
      continue;
    }
    for (int base = lane_id; base < kRowInt4; base += WARP_SIZE * kUnroll) {
      float acc[kUnroll][kNumElemsPerInt4] = {};
      for (uint64_t m = rank_mask; m; m &= m - 1) {
        int const r = __ffsll(static_cast<long long>(m)) - 1;
        auto const src = reinterpret_cast<int4 const*>(
            static_cast<uint8_t const*>(rdma_recv_x) +
            (static_cast<size_t>(r) * num_max_dispatch_tokens_per_rank +
             token_idx) *
                kSlotBytes);
        int4 v[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          int const idx = base + u * WARP_SIZE;
          v[u] =
              idx < kRowInt4 ? ld_nc_global(src + idx) : make_int4(0, 0, 0, 0);
        }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          auto const v_bf16 = reinterpret_cast<nv_bfloat16 const*>(&v[u]);
#pragma unroll
          for (int j = 0; j < kNumElemsPerInt4; ++j)
            acc[u][j] += static_cast<float>(v_bf16[j]);
        }
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        int const idx = base + u * WARP_SIZE;
        if (idx < kRowInt4) {
          int4 packed;
          auto const packed_bf16 = reinterpret_cast<nv_bfloat16*>(&packed);
#pragma unroll
          for (int j = 0; j < kNumElemsPerInt4; ++j)
            packed_bf16[j] = static_cast<nv_bfloat16>(acc[u][j]);
          out_row[idx] = packed;
        }
      }
    }
  }
}

void combine(void* combined_x, void* rdma_recv_x, int* rdma_recv_flag,
             int64_t* rdma_recv_flag_internode, void* rdma_send_x,
             void const* x, int64_t const* topk_idx, void const* dedup_hdr,
             int const* dedup_pos, int const* dedup_count, int* next_clean,
             int64_t* next_clean_second, int num_next_clean_int,
             int num_combined_tokens, int hidden,
             int num_max_dispatch_tokens_per_rank, int num_topk,
             int num_experts, int rank, int num_ranks, void* workspace,
             int num_device_sms, cudaStream_t stream, int phases,
             uint64_t const* d2h_channel_addrs, int num_d2h_channel_addrs,
             int max_nvl_peers, int low_latency_buffer_idx,
             void** ipc_rdma_base_ptrs, void* rdma_buffer_ptr,
             void* atomic_buffer_ptr, bool compress) {
  int const num_warp_groups = ceil_div(num_experts, num_device_sms);
  auto const num_sms = ceil_div(num_experts, num_warp_groups);
  int const C = num_channels(num_d2h_channel_addrs, num_experts, num_ranks);
  EP_HOST_ASSERT(num_topk <= kMaxTopK and num_ranks <= 64);
  int* ws = dedup_workspace(workspace);

#define DEDUP_COMBINE_LAUNCH_CASE(hidden)                                      \
  {                                                                            \
    auto combine_func =                                                        \
        compress ? combine<hidden, true> : combine<hidden, false>;             \
    LAUNCH_KERNEL(&cfg, combine_func, combined_x, rdma_recv_x, rdma_recv_flag, \
                  rdma_recv_flag_internode, rdma_send_x, x, topk_idx,          \
                  static_cast<int4 const*>(dedup_hdr), dedup_pos, dedup_count, \
                  ws, next_clean, next_clean_second, num_next_clean_int,       \
                  num_combined_tokens, num_max_dispatch_tokens_per_rank,       \
                  num_topk, num_experts, rank, num_ranks, C, phases,           \
                  d2h_channel_addrs, num_d2h_channel_addrs, max_nvl_peers,     \
                  low_latency_buffer_idx, ipc_rdma_base_ptrs, rdma_buffer_ptr, \
                  atomic_buffer_ptr);                                          \
  }                                                                            \
  break

  DEDUP_LAUNCH_CONFIG(num_sms, kNumMaxWarpGroups * WARP_SIZE, stream);
  SWITCH_HIDDEN(DEDUP_COMBINE_LAUNCH_CASE);
#undef DEDUP_COMBINE_LAUNCH_CASE
}

#else  // ROCm

void dispatch(void*, void*, int*, int64_t*, int*, void*, int*, int*, void*,
              int*, int64_t*, void*, void const*, int64_t const*, float const*,
              int*, int64_t*, int, int, int, int, int, int, int, int, bool,
              bool, bool, void*, int, cudaStream_t, int, uint64_t const*, int,
              int, int, void**, void*, void*, uint8_t*, int*) {
  throw EPException("EP", __FILE__, __LINE__,
                    "low_latency_dispatch_dedup is not supported on ROCm; "
                    "use low_latency_dispatch");
}

void combine(void*, void*, int*, int64_t*, void*, void const*, int64_t const*,
             void const*, int const*, int const*, int*, int64_t*, int, int, int,
             int, int, int, int, int, void*, int, cudaStream_t, int,
             uint64_t const*, int, int, int, void**, void*, void*, bool) {
  throw EPException("EP", __FILE__, __LINE__,
                    "low_latency_combine_dedup is not supported on ROCm; "
                    "use low_latency_combine");
}

#endif

}  // namespace internode_ll_dedup
}  // namespace uccl
