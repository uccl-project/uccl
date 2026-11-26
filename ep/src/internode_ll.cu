#include "common.hpp"
#include "ep_configs.cuh"
#include "ep_launch.cuh"
#include "ep_runtime.cuh"
#include "ep_util.hpp"
#include "ep_utils.cuh"
#include "internode_ll.cuh"
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
#define UCCL_EP_ENABLE_FP4 1
#include "fp4_utils.cuh"
#endif
#include "uccl_transfer_device.cuh"
#include <iostream>
#include <vector>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
namespace uccl {
namespace internode_ll {

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
constexpr int kNumMaxWarpGroups = 16;
#else
constexpr int kNumMaxWarpGroups = 32;
#endif

template <int kNumThreads, int kNumRanks>
__launch_bounds__(kNumThreads, 1) __global__
    void clean_low_latency_buffer(int* clean_0, int num_clean_int_0,
                                  int* clean_1, int num_clean_int_1,
                                  int** barrier_signal_ptrs, int rank) {
  // Peers can still be issuing IPC writes to this rank's signaling buffer.
  // Synchronize before and after cleaning so those writes are not wiped out.
  if constexpr (kNumRanks > 1) {
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
  }

  // Clean
  auto thread_id = static_cast<int>(threadIdx.x);
#pragma unroll
  for (int i = thread_id; i < num_clean_int_0; i += kNumThreads) clean_0[i] = 0;
#pragma unroll
  for (int i = thread_id; i < num_clean_int_1; i += kNumThreads) clean_1[i] = 0;

  if constexpr (kNumRanks > 1) {
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
  }
}

void clean_low_latency_buffer(int* clean_0, int num_clean_int_0, int* clean_1,
                              int num_clean_int_1, int** barrier_signal_ptrs,
                              int rank, int num_ranks, cudaStream_t stream) {
  constexpr int kNumThreads = 256;

  SETUP_LAUNCH_CONFIG(1, kNumThreads, stream);

#define CLEAN_LL_LAUNCH_CASE(ranks)                                            \
  LAUNCH_KERNEL(&cfg, (clean_low_latency_buffer<kNumThreads, ranks>), clean_0, \
                num_clean_int_0, clean_1, num_clean_int_1,                     \
                barrier_signal_ptrs, rank);                                    \
  break
  SWITCH_RANKS(CLEAN_LL_LAUNCH_CASE);
#undef CLEAN_LL_LAUNCH_CASE
}

// kFP4: 0 = off, 1 = NVFP4, 2 = MXFP4 (see fp4_utils.cuh).  FP4 messages are
// [int4 header | hidden/2 bytes of E2M1 | one scale byte per block | pad to
// 16].  Header .x is the source token index; for NVFP4 header .y holds the
// per-token FP32 decode scale.
template <bool kUseFP8, bool kUseUE8M0, int kHidden, bool kUseAggressiveAtomic,
          int kFP4 = 0, bool kUseMXFP8 = false>
__global__ __launch_bounds__(1024, 1) void dispatch(
    void* packed_recv_x, void* packed_recv_x_scales, int* packed_recv_src_info,
    int64_t* packed_recv_layout_range, int* packed_recv_count,
    int* cumulative_local_expert_recv_stats,
    int64_t* dispatch_wait_recv_cost_stats, void* rdma_recv_x,
    int* rdma_recv_count, void* rdma_x, void const* x, int64_t const* topk_idx,
    int* atomic_counter_per_expert, int* atomic_finish_counter_per_expert,
    int* atomic_send_counter_per_expert, int* next_clean,
    int64_t* next_clean_second, int num_next_clean_int, int num_tokens,
    int num_max_dispatch_tokens_per_rank, int num_topk, int num_experts,
    int rank, int num_ranks, int num_warp_groups, int num_warps_per_group,
    bool round_scale, int phases, uint64_t const* d2h_channel_addrs,
    int num_d2h_channel_addrs, int max_nvl_peers, int low_latency_buffer_idx,
    void** ipc_rdma_base_ptrs = nullptr, void* rdma_buffer_ptr = nullptr,
    void* atomic_buffer_ptr = nullptr,
    int64_t* rdma_recv_count_internode = nullptr,
    int* grid_sync_barrier_ptr = nullptr, float* packed_recv_x_gs = nullptr) {
  auto const sm_id = static_cast<int>(blockIdx.x);
  auto const thread_id = static_cast<int>(threadIdx.x);
  auto const warp_id = thread_id / WARP_SIZE, lane_id = get_lane_id();
  auto const num_sms = static_cast<int>(gridDim.x);
  auto const num_warps = num_warp_groups * num_warps_per_group;
  auto const num_local_experts = num_experts / num_ranks;
  auto const warp_group_id = warp_id / num_warps_per_group;
  auto const sub_warp_id = warp_id % num_warps_per_group;
  auto const responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

  // May extract UE8M0 from the scales
  using scale_t = std::conditional_t<kUseUE8M0, uint8_t, float>;
  using packed_t = std::conditional_t<kUseUE8M0, uint32_t, float>;
  EP_STATIC_ASSERT(sizeof(packed_t) % sizeof(scale_t) == 0,
                   "Invalid vector length");

  constexpr bool kIsFP4 = kFP4 != 0;
  EP_STATIC_ASSERT(not(kIsFP4 and kUseFP8), "FP4 and FP8 are exclusive");
  EP_STATIC_ASSERT(!kUseMXFP8 || (kUseFP8 && !kUseUE8M0), "Invalid MXFP8 mode");
  constexpr int kScaleBlockSize =
      kUseMXFP8 || kFP4 == 2 ? 32 : (kFP4 == 1 ? 16 : 128);
  constexpr int num_scales = kHidden / kScaleBlockSize;
  constexpr int kFP4NumScales = kIsFP4 ? num_scales : 0;
  EP_STATIC_ASSERT(not kIsFP4 or kHidden % 32 == 0, "Invalid FP4 hidden");
#ifndef UCCL_EP_ENABLE_FP4
  EP_STATIC_ASSERT(not kIsFP4, "FP4 dispatch is CUDA-only");
#endif

  size_t const hidden_bytes =
      kIsFP4 ? kHidden / 2
             : kHidden *
                   (kUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));
  size_t const hidden_int4 = hidden_bytes / sizeof(int4);

  // Message package: hidden data, FP8 scales, index at source
  // NOTES: currently we have 3 reserved int fields for future use (FP4 uses
  // the second one for the NVFP4 token scale)
  using vec_t = typename std::conditional<kUseFP8, int2, int4>::type;
  size_t const num_bytes_per_msg =
      sizeof(int4) +
      (kIsFP4 ? (kHidden / 2 + kFP4NumScales + 15) / 16 * 16
              : (kUseFP8 ? (kHidden + num_scales * (kUseMXFP8 ? sizeof(uint8_t)
                                                              : sizeof(float)))
                         : (kHidden * sizeof(nv_bfloat16))));
  size_t const num_int4_per_msg = num_bytes_per_msg / sizeof(int4);
  EP_DEVICE_ASSERT(num_bytes_per_msg % sizeof(int4) == 0);

  // Expert counts
  __shared__ int shared_num_tokens_sent_per_expert[kNumMaxWarpGroups];
  // Per-warp partial token amax for NVFP4
  __shared__ float shared_fp4_token_amax[kNumMaxWarpGroups];

#ifdef PER_EXPERT_BATCHING
  // Global counter slots used for batching sends to each top-k destination.
  constexpr int kNumMaxTopK = 16;
  __shared__ int shared_send_slots[kNumMaxTopK];
  __shared__ int shared_dst_experts[kNumMaxTopK];
#endif

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  // initialize barrier
  amd::barrier_init(1);
  // `barrier_init` is a plain shared-memory store executed by every thread.
  // Without a block-wide synchronization a late thread can reset the counter
  // after other warps have already arrived, so those warps wait for a
  // generation flip that can no longer happen and the block deadlocks before
  // sending any token.
  __syncthreads();
#endif

  // Sending phase
  if ((phases & LOW_LATENCY_SEND_PHASE) == 0) goto LOW_LATENCY_DISPATCH_RECV;

  // There are 2 kinds of warps in this part:
  // 1. The first-kind warps for FP8 cast and sending top-k tokens
  // 2. The last warp for reading `topk_idx` and count for per-expert
  // information
  if (warp_id < num_warps - 1) {
    constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
    EP_STATIC_ASSERT(kHidden % (WARP_SIZE * kNumElemsPerRead) == 0,
                     "Invalid hidden");
    constexpr int kLanesPerScale = kScaleBlockSize / kNumElemsPerRead;
    EP_STATIC_ASSERT(kScaleBlockSize % kNumElemsPerRead == 0 &&
                         WARP_SIZE % kLanesPerScale == 0,
                     "Invalid scale block vectorization");
    auto const num_threads = (num_warps - 1) * WARP_SIZE;
    size_t const hidden_bf16_int4 = kHidden / kNumElemsPerRead;

    for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
      auto const x_int4 =
          static_cast<int4 const*>(x) + token_idx * hidden_bf16_int4;
      auto const rdma_x_src_idx = reinterpret_cast<int*>(
          static_cast<uint8_t*>(rdma_x) + token_idx * num_bytes_per_msg);
      auto const rdma_x_vec = reinterpret_cast<vec_t*>(
          reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));
      auto const rdma_x_scales = reinterpret_cast<float*>(
          reinterpret_cast<uint8_t*>(rdma_x_vec) + hidden_bytes);

      // Overlap top-k index read and source token index writes
      auto dst_expert_idx =
          warp_id < num_topk ? static_cast<int>(__ldg(
                                   topk_idx + token_idx * num_topk + warp_id))
                             : -1;
      thread_id == 0 ? (*rdma_x_src_idx = token_idx) : 0;

#ifdef UCCL_EP_ENABLE_FP4
      // NVFP4: per-token amax -> FP32 decode scale, written into header .y.
      // The shared slots are re-written only after the barrier that ends the
      // cast loop below, which every reader has passed.
      float fp4_gs = 1.0f;
      if constexpr (kFP4 == fp4::kNVFP4) {
        float tamax = 0.0f;
        for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
          auto v = __ldg(x_int4 + i);
          auto b = reinterpret_cast<nv_bfloat16 const*>(&v);
#pragma unroll
          for (int j = 0; j < kNumElemsPerRead; ++j)
            tamax = fmaxf(tamax, fabsf(static_cast<float>(b[j])));
        }
        tamax = warp_reduce_max(tamax);
        if (lane_id == 0) shared_fp4_token_amax[warp_id] = tamax;
        sync_barrier_1(num_threads);
        tamax = 0.0f;
        for (int w = 0; w < num_warps - 1; ++w)
          tamax = fmaxf(tamax, shared_fp4_token_amax[w]);
        fp4_gs = fp4::nv_token_scale(tamax);
        if (thread_id == 0) rdma_x_src_idx[1] = __float_as_int(fp4_gs);
      }
      auto const fp4_data = reinterpret_cast<uint32_t*>(
          reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));
      auto const fp4_scales =
          reinterpret_cast<uint8_t*>(fp4_data) + kHidden / 2;
#endif

#ifdef PER_EXPERT_BATCHING
      // Allocate per-expert send slots for top-k destinations.
      // Each warp (warp_id < num_topk) reserves one slot for its destination
      // expert.
      if (warp_id < num_topk && lane_id == 0) {
        shared_dst_experts[warp_id] = dst_expert_idx;
        if (dst_expert_idx >= 0) {
          shared_send_slots[warp_id] =
              atomicAdd(atomic_send_counter_per_expert + dst_expert_idx, 1);
        } else {
          shared_send_slots[warp_id] = -1;
        }
      }
      // Sync to make shared_send_slots visible to all threads
      sync_barrier_1((num_warps - 1) * WARP_SIZE);
#endif

// FP8 cast
#pragma unroll
      for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
        // Read
        auto int4_value = __ldg(x_int4 + i);

        if constexpr (kUseFP8) {
          // Calculate local amax
          auto bf16_values = reinterpret_cast<nv_bfloat16*>(&int4_value);
          float fp32_values[kNumElemsPerRead];
          float amax = kUseMXFP8 ? 0.0f : kFP8Margin;
          float scale, scale_inv;
#pragma unroll
          for (int j = 0; j < kNumElemsPerRead; ++j) {
            fp32_values[j] = static_cast<float>(bf16_values[j]);
            amax = fmaxf(amax, fabsf(fp32_values[j]));
          }

          // Reduce amax and scale

          amax = warp_reduce_max<kLanesPerScale>(amax);
#ifdef UCCL_EP_ENABLE_FP4
          if constexpr (kUseMXFP8) {
            auto const code = mx::mxfp8_scale(amax);
            scale = mx::ue8m0_to_inv_f32(code);
            if (lane_id % kLanesPerScale == 0)
              reinterpret_cast<uint8_t*>(rdma_x_scales)[i / kLanesPerScale] =
                  code;
          } else
#endif
          {
            calculate_fp8_scales(amax, scale, scale_inv, round_scale);
            if (lane_id % kLanesPerScale == 0)
              rdma_x_scales[i / kLanesPerScale] = scale_inv;
          }

          // Cast into send buffer
          vec_t int2_value;
          auto fp8x2_values =
              reinterpret_cast<__nv_fp8x2_storage_t*>(&int2_value);
#pragma unroll
          for (int j = 0; j < kNumElemsPerRead; j += 2) {
            float2 fp32x2 = {fp32_values[j] * scale,
                             fp32_values[j + 1] * scale};
#ifdef UCCL_EP_ENABLE_FP4
            if constexpr (kUseMXFP8)
              fp8x2_values[j / 2] = mx::e4m3x2_rn(fp32x2.x, fp32x2.y);
            else
#endif
              fp8x2_values[j / 2] =
                  __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE,
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#if defined(__gfx942__)
                                           __HIP_E4M3_FNUZ
#else
                                           __HIP_E4M3
#endif
#else
                                         __NV_E4M3
#endif
                  );
          }
          rdma_x_vec[i] = int2_value;
#ifdef UCCL_EP_ENABLE_FP4
        } else if constexpr (kIsFP4) {
          // One int4 = 8 BF16 -> 8 E2M1 nibbles.  A block is 2 (NVFP4) or 4
          // (MXFP4) consecutive int4s, i.e. aligned groups of lanes, since
          // `num_threads` is a multiple of the warp size.
          auto bf16_values = reinterpret_cast<nv_bfloat16*>(&int4_value);
          float fp32_values[kNumElemsPerRead];
          float amax = 0.0f;
#pragma unroll
          for (int j = 0; j < kNumElemsPerRead; ++j) {
            fp32_values[j] = static_cast<float>(bf16_values[j]);
            amax = fmaxf(amax, fabsf(fp32_values[j]));
          }
          amax = warp_reduce_max<kLanesPerScale>(amax);
          float inv_scale;
          uint8_t scale_byte;
          if constexpr (kFP4 == fp4::kNVFP4)
            scale_byte = fp4::nv_block_scale(amax, fp4_gs, inv_scale);
          else
            scale_byte = fp4::mx_block_scale(amax, inv_scale);
#pragma unroll
          for (int j = 0; j < kNumElemsPerRead; ++j)
            fp32_values[j] *= inv_scale;
          fp4_data[i] = fp4::pack8_e2m1(fp32_values);
          if (lane_id % kLanesPerScale == 0)
            fp4_scales[i / kLanesPerScale] = scale_byte;
          (void)fp4_gs;
#endif
        } else {
          // Reinterpret-cast is for C++14 compatibility
          rdma_x_vec[i] = *reinterpret_cast<vec_t*>(&int4_value);
        }
      }
      sync_barrier_1(num_threads);

      // Issue RDMA sends
      if (dst_expert_idx >= 0) {
        int slot_idx =
            lane_id == 0
                ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1)
                : 0;
        slot_idx = __shfl_sync(WARP_MASK, slot_idx, 0);
        auto const dst_rank = dst_expert_idx / num_local_experts;
        auto const dst_expert_local_idx = dst_expert_idx % num_local_experts;
        auto const src_ptr = reinterpret_cast<uint64_t>(rdma_x_src_idx);
        auto const dst_ptr =
            reinterpret_cast<uint64_t>(rdma_recv_x) +
            static_cast<size_t>(dst_expert_local_idx) * num_ranks *
                num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
            rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
            slot_idx * num_bytes_per_msg;
        auto const dst_p2p_ptr =
            ipc_rdma_base_ptrs
                ? uccl::get_ipc_p2p_ptr(dst_ptr, ipc_rdma_base_ptrs, rank,
                                        dst_rank, max_nvl_peers, 0)
                : 0;
        if (dst_p2p_ptr == 0) {
#ifdef PER_EXPERT_BATCHING
          // For inter-node send path, copy temp data to the per-expert batch
          // buffer, then issue a batched RDMA send.
          // TODO: This has an extra temp->per-expert copy in the FP8 path.
          // FP8 output is written to the temp buffer first, then copied here.
          auto const slot_idx = shared_send_slots[warp_id];
          auto const batch_buf_offset =
              num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
          auto const batch_buf_ptr =
              static_cast<uint8_t*>(rdma_x) + batch_buf_offset +
              (dst_expert_idx * num_max_dispatch_tokens_per_rank + slot_idx) *
                  num_bytes_per_msg;
          auto const* src_int4_ptr =
              reinterpret_cast<int4 const*>(rdma_x_src_idx);
          auto* batch_buf_int4_ptr = reinterpret_cast<int4*>(batch_buf_ptr);
          UNROLLED_WARP_COPY(8, lane_id, num_int4_per_msg, batch_buf_int4_ptr,
                             src_int4_ptr, ld_nc_global, st_na_global);
#else
          // Legacy path: directly issue one RDMA send per token.
          __threadfence_system();
          uccl::put_nbi_warp(
              dst_ptr - reinterpret_cast<uint64_t>(rdma_buffer_ptr),
              src_ptr - reinterpret_cast<uint64_t>(rdma_buffer_ptr),
              num_bytes_per_msg, dst_rank,
              /*warp_id=*/dst_expert_local_idx,  // NOTE(Yang): for selecting
                                                 // rb.
              lane_id, slot_idx, d2h_channel_addrs, num_d2h_channel_addrs,
              false, low_latency_buffer_idx);
#endif
        } else {
          // Intra-node: use direct memory copy via IPC
          auto const* src_int4_ptr = reinterpret_cast<int4 const*>(src_ptr);
          auto* dst_int4_ptr = reinterpret_cast<int4*>(dst_p2p_ptr);
          UNROLLED_WARP_COPY(8, lane_id, num_int4_per_msg, dst_int4_ptr,
                             src_int4_ptr, ld_nc_global, st_na_global);
        }
        // Increase counter after finishing
        __syncwarp();
        lane_id == 0 ? atomic_add_release_global<kUseAggressiveAtomic>(
                           atomic_finish_counter_per_expert + dst_expert_idx, 1)
                     : 0;
      }
    }
  } else if (warp_id == num_warps - 1) {
    // NOTE(MaoZiming): These checks are RDMA specific.
    EP_DEVICE_ASSERT(num_sms > 1);
    if (sm_id == 0) {
      // The first SM is also responsible for cleaning the next buffer
#pragma unroll
      for (int i = lane_id; i < num_next_clean_int; i += WARP_SIZE) {
        next_clean[i] = 0;
        next_clean_second[i] = 0;
      }
      // Notify before executing `int_p`
      __syncwarp();
#pragma unroll
      for (int i = lane_id; i < num_experts; i += WARP_SIZE)
        atomic_add_release_global<kUseAggressiveAtomic>(
            atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG);
    }
    // This SM should be responsible for some destination experts, read
    // `topk_idx` for them
    int expert_count[kNumMaxWarpGroups] = {0};
    auto const expert_begin_idx = sm_id * num_warp_groups;
    auto const expert_end_idx =
        min(expert_begin_idx + num_warp_groups, num_experts);

// Per lane count
#pragma unroll 8
    for (int i = lane_id; i < num_tokens * num_topk; i += WARP_SIZE) {
      auto idx = static_cast<int>(__ldg(topk_idx + i));
      if (idx >= expert_begin_idx and idx < expert_end_idx)
        expert_count[idx - expert_begin_idx]++;
    }

// Warp reduce
#pragma unroll
    for (int i = expert_begin_idx; i < expert_end_idx; ++i) {
      auto sum = warp_reduce_sum(expert_count[i - expert_begin_idx]);
      if (lane_id == 0) {
        shared_num_tokens_sent_per_expert[i - expert_begin_idx] = sum;
        atomic_add_release_global<kUseAggressiveAtomic>(
            atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG - sum);
      }
    }
  }
  __syncthreads();

#ifdef PER_EXPERT_BATCHING
  // Grid-wide sync before batch-send.
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  // Each reset counter is used once per kernel so every CTA can observe zero.
  amd::grid_sync_then_zero(grid_sync_barrier_ptr + 1, num_sms);
#else
  cg::this_grid().sync();
#endif

  // Batch RDMA send phase - send entire expert buffer in ONE RDMA call
  // Each warp group handles one expert (only first sub_warp does the send)
  if (responsible_expert_idx < num_experts && sub_warp_id == 0) {
    auto const dst_rank = responsible_expert_idx / num_local_experts;
    auto const dst_expert_local_idx =
        responsible_expert_idx % num_local_experts;

    // Check if this destination is inter-node (needs RDMA batch send)
    // IPC destinations were already sent in the token loop
    auto const test_dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x);
    auto const dst_p2p_ptr =
        ipc_rdma_base_ptrs
            ? uccl::get_ipc_p2p_ptr(test_dst_ptr, ipc_rdma_base_ptrs, rank,
                                    dst_rank, max_nvl_peers, 0)
            : 0;

    if (dst_p2p_ptr == 0) {
      // Inter-node: batch send ALL tokens for this expert in ONE call
      auto const num_tokens_to_send =
          atomic_send_counter_per_expert[responsible_expert_idx];

      if (num_tokens_to_send > 0) {
        auto const batch_buf_offset =
            num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
        // Source: start of this expert's batch buffer (contiguous)
        auto const batch_buf_ptr =
            static_cast<uint8_t*>(rdma_x) + batch_buf_offset +
            responsible_expert_idx * num_max_dispatch_tokens_per_rank *
                num_bytes_per_msg;
        auto const src_ptr = reinterpret_cast<uint64_t>(batch_buf_ptr);
        // Destination: start of this expert's recv buffer on remote rank
        auto const dst_ptr =
            reinterpret_cast<uint64_t>(rdma_recv_x) +
            static_cast<size_t>(dst_expert_local_idx) * num_ranks *
                num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
            rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
        // Total bytes: all tokens for this expert
        auto const total_bytes = num_tokens_to_send * num_bytes_per_msg;

        __threadfence_system();

        uccl::put_nbi_warp(
            dst_ptr - reinterpret_cast<uint64_t>(rdma_buffer_ptr),
            src_ptr - reinterpret_cast<uint64_t>(rdma_buffer_ptr), total_bytes,
            dst_rank,
            /*warp_id=*/dst_expert_local_idx,  // NOTE(Yang): for selecting rb.
            lane_id, /*slot=*/0, d2h_channel_addrs, num_d2h_channel_addrs,
            false, low_latency_buffer_idx, 0, 0, num_tokens_to_send);
      }
    }
    // IPC: already sent in the token loop, nothing to do here
  }

  __threadfence_system();  // Ensure batch sends are visible before count sends
#endif

  // Issue count sends
  if (responsible_expert_idx < num_experts and sub_warp_id == 0 and
      lane_id == 0) {
    auto const dst_rank = responsible_expert_idx / num_local_experts;
    auto const dst_expert_local_idx =
        responsible_expert_idx % num_local_experts;
    auto const num_tokens_sent =
        shared_num_tokens_sent_per_expert[responsible_expert_idx -
                                          sm_id * num_warp_groups];
    // Wait local sends issued and send expert counts
    while (ld_acquire_global<kUseAggressiveAtomic>(
               atomic_finish_counter_per_expert + responsible_expert_idx) !=
           FINISHED_SUM_TAG * 2)
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
      __builtin_amdgcn_s_sleep(1);
#else
      ;
#endif

    auto dst_ptr = reinterpret_cast<uint64_t>(
        rdma_recv_count + dst_expert_local_idx * num_ranks + rank);
    auto dst_ptr_internode = reinterpret_cast<uint64_t>(
        rdma_recv_count_internode + dst_expert_local_idx * num_ranks + rank);
    // Try to use IPC for intra-node atomic operations
    auto const dst_p2p_ptr =
        ipc_rdma_base_ptrs
            ? uccl::get_ipc_p2p_ptr(dst_ptr, ipc_rdma_base_ptrs, rank, dst_rank,
                                    max_nvl_peers, 0)
            : 0;
    if (dst_p2p_ptr == 0) {
      // Inter-node or no IPC: use RDMA atomic
      uccl::atomic_nonfetch_add(
          dst_ptr_internode, reinterpret_cast<uint64_t>(atomic_buffer_ptr),
          -num_tokens_sent - 1, dst_rank,
          /*warp_id=*/dst_expert_local_idx,  // NOTE(Yang): for selecting rb.
          false, d2h_channel_addrs, num_d2h_channel_addrs, false,
          low_latency_buffer_idx);
    } else {
      // Intra-node: use direct atomic operation
      st_release_sys_global<kUseAggressiveAtomic>(
          reinterpret_cast<int*>(dst_p2p_ptr), -num_tokens_sent - 1);
    }
    // Clean workspace for next use
    atomic_counter_per_expert[responsible_expert_idx] = 0;
    atomic_finish_counter_per_expert[responsible_expert_idx] = 0;
#ifdef PER_EXPERT_BATCHING
    atomic_send_counter_per_expert[responsible_expert_idx] = 0;
#endif
    // Clean `packed_recv_count`
    if (dst_rank == 0) packed_recv_count[dst_expert_local_idx] = 0;
  }
  __syncwarp();

// Receiving phase
LOW_LATENCY_DISPATCH_RECV:
  if ((phases & LOW_LATENCY_RECV_PHASE) == 0) {
    return;
  }

  // For send-and-recv kernels, we need a grid sync for making
  // `packed_recv_count` visible
  if (phases & LOW_LATENCY_SEND_PHASE)
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    amd::grid_sync_then_zero(grid_sync_barrier_ptr, num_sms);
#else
    cg::this_grid().sync();
#endif

  // Receiving and packing
  if (responsible_expert_idx < num_experts) {
    auto const src_rank = responsible_expert_idx / num_local_experts;
    auto const local_expert_idx = responsible_expert_idx % num_local_experts;
    auto const rdma_recv_x_uint8 =
        static_cast<uint8_t*>(rdma_recv_x) +
        static_cast<size_t>(local_expert_idx) * num_ranks *
            num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
        src_rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
    auto const recv_x_int4 = static_cast<int4*>(packed_recv_x) +
                             local_expert_idx * num_ranks *
                                 num_max_dispatch_tokens_per_rank * hidden_int4;
    auto const recv_src_info =
        packed_recv_src_info +
        local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
    auto const recv_range =
        packed_recv_layout_range + local_expert_idx * num_ranks;
    auto const num_aligned_scales =
        align<int>(num_scales, sizeof(float) / sizeof(scale_t));
    auto const expert_row = static_cast<size_t>(local_expert_idx) * num_ranks *
                            num_max_dispatch_tokens_per_rank;

    // Shared between sub-warps in warp groups
    __shared__ int shared_num_recv_tokens[kNumMaxWarpGroups],
        shared_recv_token_begin_idx[kNumMaxWarpGroups];

    // Wait tokens to arrive
    // NOTES: using sub-warp 1 to overlap with sub-warp 0
    int num_recv_tokens_internode = 0, num_recv_tokens_ipc = 0,
        num_recv_tokens = 0, recv_token_begin_idx = 0;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    EP_DEVICE_ASSERT(num_warps_per_group > 1);
#else
    EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 15);
#endif
    if (sub_warp_id == 1 and lane_id == 0) {
      auto start_time = clock64();
      while ((src_rank / max_nvl_peers == rank / max_nvl_peers) &&
             (num_recv_tokens_ipc = ld_acquire_sys_global<kUseAggressiveAtomic>(
                  rdma_recv_count + local_expert_idx * num_ranks + src_rank)) ==
                 0)
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        __builtin_amdgcn_s_sleep(1);
#else
        ;
#endif

      while ((src_rank / max_nvl_peers != rank / max_nvl_peers) &&
             (num_recv_tokens_internode =
                  static_cast<int>(ld_acquire_sys_global<kUseAggressiveAtomic>(
                      reinterpret_cast<uint64_t const*>(
                          rdma_recv_count_internode +
                          local_expert_idx * num_ranks + src_rank)))) == 0)
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        __builtin_amdgcn_s_sleep(1);
#else
        ;
#endif

      if (src_rank / max_nvl_peers == rank / max_nvl_peers) {
        if (ld_acquire_sys_global<kUseAggressiveAtomic>(
                reinterpret_cast<uint64_t const*>(rdma_recv_count_internode +
                                                  local_expert_idx * num_ranks +
                                                  src_rank)) != 0) {
          printf(
              "Same node but rdma_recv_count_internode is not zero! src_rank: "
              "%d, rank: %d, max_nvl_peers: %d\n",
              src_rank, rank, max_nvl_peers);
          EP_DEVICE_ASSERT(false);
        }
      }
      if (src_rank / max_nvl_peers != rank / max_nvl_peers) {
        if (ld_acquire_sys_global<kUseAggressiveAtomic>(
                rdma_recv_count + local_expert_idx * num_ranks + src_rank) !=
            0) {
          printf(
              "Different node but rdma_recv_count is not zero! src_rank: %d, "
              "rank: %d, max_nvl_peers: %d\n",
              src_rank, rank, max_nvl_peers);
          EP_DEVICE_ASSERT(false);
        }
      }
      auto wait_recv_cost = clock64() - start_time;
      num_recv_tokens_internode =
          num_recv_tokens_internode != 0 ? -num_recv_tokens_internode - 1 : 0;
      num_recv_tokens_ipc =
          num_recv_tokens_ipc != 0 ? -num_recv_tokens_ipc - 1 : 0;
      // printf(
      //     "num_recv_tokens_internode: %d, num_recv_tokens_ipc: %d, src_rank:"
      //     "%d, rank: %d, max_nvl_peers: %d, responsible_expert_idx: %d,"
      //     "num_experts: %d, num_local_experts: %d\n",
      //     num_recv_tokens_internode, num_recv_tokens_ipc, src_rank, rank,
      //     max_nvl_peers, responsible_expert_idx, num_experts,
      //     num_local_experts);
      num_recv_tokens = num_recv_tokens_internode + num_recv_tokens_ipc;
      recv_token_begin_idx =
          atomicAdd(packed_recv_count + local_expert_idx, num_recv_tokens);
      shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
      shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
      recv_range[src_rank] =
          pack2<int, int64_t>(num_recv_tokens, recv_token_begin_idx);
      // Add stats for diagnosis
      if (cumulative_local_expert_recv_stats != nullptr)
        atomicAdd(cumulative_local_expert_recv_stats + local_expert_idx,
                  num_recv_tokens);
      if (dispatch_wait_recv_cost_stats != nullptr)
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      dispatch_wait_recv_cost_stats + src_rank),
                  wait_recv_cost);
    }
    sync_barrier<true>(warp_group_id + 2, num_warps_per_group * WARP_SIZE);

    num_recv_tokens = shared_num_recv_tokens[warp_group_id];
    recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];

    // Copy tokens
    if constexpr (kUseFP8 && !kUseMXFP8) {
      EP_STATIC_ASSERT(num_scales <= 2 * WARP_SIZE, "Legacy FP8 scale copy");
    }
    for (int i = sub_warp_id; i < num_recv_tokens; i += num_warps_per_group) {
      // Copy source info
      auto const src_src_idx =
          reinterpret_cast<int*>(rdma_recv_x_uint8 + i * num_bytes_per_msg);
      if (lane_id == 0)
        recv_src_info[recv_token_begin_idx + i] = ld_nc_global(src_src_idx);
      __syncwarp();

      // Copy data
      // NOTES: only 2 load iterations for 7K hidden with 7 unrolls
      auto const src_data = reinterpret_cast<int4*>(
          reinterpret_cast<uint8_t*>(src_src_idx) + sizeof(int4));
      auto const dst_data =
          recv_x_int4 + (recv_token_begin_idx + i) * hidden_int4;
      UNROLLED_WARP_COPY(7, lane_id, hidden_int4, dst_data, src_data,
                         ld_nc_global, st_na_global);

#ifdef UCCL_EP_ENABLE_FP4
      // FP4: row-major uint8 scales `[E, R * T, kFP4NumScales]` and, for
      // NVFP4, FP32 token scales `[E, R * T]`
      if constexpr (kIsFP4) {
        EP_STATIC_ASSERT(kFP4NumScales % sizeof(int4) == 0,
                         "FP4 scales must be int4 aligned");
        auto const token_idx =
            local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank +
            recv_token_begin_idx + i;
        auto const src_fp4_scales = reinterpret_cast<int4 const*>(
            reinterpret_cast<uint8_t*>(src_data) + hidden_bytes);
        auto const dst_fp4_scales = reinterpret_cast<int4*>(
            static_cast<uint8_t*>(packed_recv_x_scales) +
            static_cast<size_t>(token_idx) * kFP4NumScales);
        for (int k = lane_id;
             k < kFP4NumScales / static_cast<int>(sizeof(int4)); k += WARP_SIZE)
          st_na_global(dst_fp4_scales + k, ld_nc_global(src_fp4_scales + k));
        if constexpr (kFP4 == fp4::kNVFP4) {
          if (lane_id == 0)
            packed_recv_x_gs[token_idx] =
                __int_as_float(ld_nc_global(src_src_idx + 1));
        }
      }
#endif

      // Copy MXFP8 scales as contiguous rows. The legacy FP8 layout is
      // transposed for column-major consumers.
      if constexpr (kUseMXFP8) {
        EP_STATIC_ASSERT(num_scales % sizeof(int4) == 0, "Unaligned MXFP8 row");
        auto const src_scales = reinterpret_cast<int4 const*>(
            reinterpret_cast<uint8_t*>(src_data) + hidden_bytes);
        auto const dst_scales = reinterpret_cast<int4*>(
            static_cast<uint8_t*>(packed_recv_x_scales) +
            (expert_row + recv_token_begin_idx + i) * num_scales);
        for (int k = lane_id; k < num_scales / static_cast<int>(sizeof(int4));
             k += WARP_SIZE)
          st_na_global(dst_scales + k, ld_nc_global(src_scales + k));
      } else if constexpr (kUseFP8) {
        auto const recv_x_scales = static_cast<scale_t*>(packed_recv_x_scales) +
                                   expert_row * num_aligned_scales;
        // Equivalent CuTe layout:
        //   (num_tokens, (num_packed, num_elems_per_pack)):(num_elems_per_pack,
        //   (num_tokens * num_elems_per_pack, 1))
        auto const src_scales = reinterpret_cast<float*>(
            reinterpret_cast<uint8_t*>(src_data) + hidden_bytes);
        auto const num_elems_per_pack =
            static_cast<int>(sizeof(packed_t) / sizeof(scale_t));
        auto const token_idx = recv_token_begin_idx + i;
        auto const token_stride = num_elems_per_pack;
        auto const pack_stride =
            num_ranks * num_max_dispatch_tokens_per_rank * num_elems_per_pack;
        if (lane_id < num_scales) {
          auto const pack_idx = lane_id / num_elems_per_pack;
          auto const elem_idx = lane_id % num_elems_per_pack;
          auto scale = extract_required_scale_format<kUseUE8M0>(
              ld_nc_global(src_scales + lane_id));
          recv_x_scales[token_idx * token_stride + pack_idx * pack_stride +
                        elem_idx] = scale;
        }
        if (lane_id + WARP_SIZE < num_scales) {
          auto const pack_idx = (lane_id + WARP_SIZE) / num_elems_per_pack;
          auto const elem_idx = (lane_id + WARP_SIZE) % num_elems_per_pack;
          auto scale = extract_required_scale_format<kUseUE8M0>(
              ld_nc_global(src_scales + lane_id + WARP_SIZE));
          recv_x_scales[token_idx * token_stride + pack_idx * pack_stride +
                        elem_idx] = scale;
        }
      }
    }
    // if (blockIdx.x == 0 && threadIdx.x == 0)
    //   printf("[dispatch] RECV finished\n");
  }
}

void dispatch(void* packed_recv_x, void* packed_recv_x_scales,
              int* packed_recv_src_info, int64_t* packed_recv_layout_range,
              int* packed_recv_count, int* cumulative_local_expert_recv_stats,
              int64_t* dispatch_wait_recv_cost_stats, void* rdma_recv_x,
              int* rdma_recv_count, void* rdma_x, void const* x,
              int64_t const* topk_idx, int* next_clean,
              int64_t* next_clean_second, int num_next_clean_int,
              int num_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
              int num_topk, int num_experts, int rank, int num_ranks,
              bool use_fp8, bool round_scale, bool use_ue8m0, void* workspace,
              int num_device_sms, cudaStream_t stream, int phases,
              uint64_t const* d2h_channel_addrs, int num_d2h_channel_addrs,
              int max_nvl_peers, int low_latency_buffer_idx,
              void** ipc_rdma_base_ptrs, void* rdma_buffer_ptr,
              void* atomic_buffer_ptr, int64_t* rdma_recv_count_internode,
              int fp4_mode, float* packed_recv_x_gs, bool use_mxfp8) {
  constexpr int kNumMaxTopK = 16;
  int const num_warp_groups = ceil_div(num_experts, num_device_sms);
  int const num_warps_per_group = kNumMaxWarpGroups / num_warp_groups;
  EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0);
  EP_HOST_ASSERT(num_topk + 1 <= num_warp_groups * num_warps_per_group);

  auto const num_warps = num_warp_groups * num_warps_per_group;
  auto const num_sms = ceil_div(num_experts, num_warp_groups);
  EP_HOST_ASSERT(num_topk <= kNumMaxTopK);

  // Workspace checks
  auto atomic_counter_per_expert = static_cast<int*>(workspace);
  auto atomic_finish_counter_per_expert =
      atomic_counter_per_expert + num_experts;
#ifdef PER_EXPERT_BATCHING
  auto atomic_send_counter_per_expert =
      atomic_finish_counter_per_expert + num_experts;
  auto grid_sync_barrier_ptr = atomic_send_counter_per_expert + num_experts;
  EP_HOST_ASSERT((num_experts * 3 + 2) * sizeof(int) <= NUM_WORKSPACE_BYTES);
#else
  auto atomic_send_counter_per_expert =
      atomic_finish_counter_per_expert + num_experts;  // Unused in legacy path.
  auto grid_sync_barrier_ptr = atomic_finish_counter_per_expert + num_experts;
  EP_HOST_ASSERT((num_experts * 2 + 1) * sizeof(int) <= NUM_WORKSPACE_BYTES);
#endif

  // FP8 checks
  if (use_ue8m0)
    EP_HOST_ASSERT(round_scale and "UE8M0 SF requires `round_scale=True`");

  if (use_mxfp8) {
#ifndef UCCL_EP_ENABLE_FP4
    EP_HOST_ASSERT(false and "MXFP8 dispatch is CUDA-only");
#endif
    EP_HOST_ASSERT(use_fp8 && !use_ue8m0 && !round_scale && fp4_mode == 0);
    EP_HOST_ASSERT(hidden % 512 == 0 && packed_recv_x_scales != nullptr);
  }

  // FP4 checks
  EP_HOST_ASSERT(fp4_mode >= 0 and fp4_mode <= 2);
  if (fp4_mode != 0) {
#ifndef UCCL_EP_ENABLE_FP4
    EP_HOST_ASSERT(false and "FP4 dispatch is CUDA-only");
#endif
    EP_HOST_ASSERT(not use_fp8 and "FP4 and FP8 are exclusive");
    EP_HOST_ASSERT(hidden % 512 == 0);
    EP_HOST_ASSERT(packed_recv_x_scales != nullptr);
    if (fp4_mode == 1) EP_HOST_ASSERT(packed_recv_x_gs != nullptr);
  }

  static bool const aggressive_atomic_enabled = get_aggressive_atomic_enabled();

#define DISPATCH_LAUNCH_CASE(hidden)                                         \
  {                                                                          \
    auto dispatch_func = aggressive_atomic_enabled                           \
                             ? dispatch<false, false, hidden, true>          \
                             : dispatch<false, false, hidden, false>;        \
    if (use_fp8 and not use_ue8m0)                                           \
      dispatch_func = aggressive_atomic_enabled                              \
                          ? dispatch<true, false, hidden, true>              \
                          : dispatch<true, false, hidden, false>;            \
    if (use_fp8 and use_ue8m0)                                               \
      dispatch_func = aggressive_atomic_enabled                              \
                          ? dispatch<true, true, hidden, true>               \
                          : dispatch<true, true, hidden, false>;             \
    DISPATCH_BLOCK_SCALED_CASES(hidden);                                     \
    LAUNCH_KERNEL(                                                           \
        &cfg, dispatch_func, packed_recv_x, packed_recv_x_scales,            \
        packed_recv_src_info, packed_recv_layout_range, packed_recv_count,   \
        cumulative_local_expert_recv_stats, dispatch_wait_recv_cost_stats,   \
        rdma_recv_x, rdma_recv_count, rdma_x, x, topk_idx,                   \
        atomic_counter_per_expert, atomic_finish_counter_per_expert,         \
        atomic_send_counter_per_expert, next_clean, next_clean_second,       \
        num_next_clean_int, num_tokens, num_max_dispatch_tokens_per_rank,    \
        num_topk, num_experts, rank, num_ranks, num_warp_groups,             \
        num_warps_per_group, round_scale, phases, d2h_channel_addrs,         \
        num_d2h_channel_addrs, max_nvl_peers, low_latency_buffer_idx,        \
        ipc_rdma_base_ptrs, rdma_buffer_ptr, atomic_buffer_ptr,              \
        rdma_recv_count_internode, grid_sync_barrier_ptr, packed_recv_x_gs); \
  }                                                                          \
  break
#ifdef UCCL_EP_ENABLE_FP4
#define DISPATCH_BLOCK_SCALED_CASES(hidden)                              \
  if (use_mxfp8)                                                         \
    dispatch_func = aggressive_atomic_enabled                            \
                        ? dispatch<true, false, hidden, true, 0, true>   \
                        : dispatch<true, false, hidden, false, 0, true>; \
  if (fp4_mode == 1)                                                     \
    dispatch_func = aggressive_atomic_enabled                            \
                        ? dispatch<false, false, hidden, true, 1>        \
                        : dispatch<false, false, hidden, false, 1>;      \
  if (fp4_mode == 2)                                                     \
  dispatch_func = aggressive_atomic_enabled                              \
                      ? dispatch<false, false, hidden, true, 2>          \
                      : dispatch<false, false, hidden, false, 2>
#else
#define DISPATCH_BLOCK_SCALED_CASES(hidden) (void)0
#endif
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  EP_HOST_ASSERT(num_warps * WARP_SIZE <= MAX_NTHREADS);
#endif

  SETUP_LAUNCH_CONFIG(num_sms, num_warps * WARP_SIZE, stream);
  SWITCH_HIDDEN(DISPATCH_LAUNCH_CASE);
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("[dispatch] kernel launch error: %s\n", cudaGetErrorString(err));
    fflush(stdout);
  }

#undef DISPATCH_LAUNCH_CASE
#undef DISPATCH_BLOCK_SCALED_CASES
}

#ifdef UCCL_EP_ENABLE_FP4
// FP4 combine message (one row of kHidden elements, 16-byte aligned):
//   [kHidden/2 B E2M1 data][block scales][NVFP4 only: FP32 token scale]
// using the dispatch packing (fp4_utils.cuh).  One warp quantizes one row;
// each lane owns 32 consecutive elements per pass, so a pass covers 1024
// elements.  The host requires hidden % 512 == 0, which keeps every 4-byte
// scale word either fully inside or fully outside the row.
template <int kFP4, int kHidden>
__device__ __forceinline__ void combine_quant_fp4_row(int4 const* x,
                                                      uint8_t* dst,
                                                      int lane_id) {
  constexpr int kBlock = kFP4 == fp4::kNVFP4 ? 16 : 32;
  constexpr int kNumScales = kHidden / kBlock;
  constexpr int kNumPasses = (kHidden + 32 * WARP_SIZE - 1) / (32 * WARP_SIZE);
  EP_STATIC_ASSERT(kHidden % 32 == 0, "Invalid hidden for FP4");
  uint8_t* scales = dst + kHidden / 2;

  float gs = 1.0f;
  if constexpr (kFP4 == fp4::kNVFP4) {
    float a = 0.0f;
    for (int i = lane_id; i < kHidden / 8; i += WARP_SIZE) {
      auto v = ld_nc_global(x + i);
      auto const b = reinterpret_cast<nv_bfloat16 const*>(&v);
#pragma unroll
      for (int j = 0; j < 8; ++j) a = fmaxf(a, fabsf(__bfloat162float(b[j])));
    }
    gs = fp4::nv_token_scale(warp_reduce_max(a));
    if (lane_id == 0) *reinterpret_cast<float*>(scales + kNumScales) = gs;
  }

#pragma unroll
  for (int pass = 0; pass < kNumPasses; ++pass) {
    int const e0 = (pass * WARP_SIZE + lane_id) * 32;
    bool const valid = e0 < kHidden;
    float f[32];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      int4 v = valid ? ld_nc_global(x + e0 / 8 + k) : make_int4(0, 0, 0, 0);
      auto const b = reinterpret_cast<nv_bfloat16 const*>(&v);
#pragma unroll
      for (int j = 0; j < 8; ++j) f[k * 8 + j] = __bfloat162float(b[j]);
    }
    uint32_t w[4];
    uint32_t sc = 0;
#pragma unroll
    for (int h = 0; h < 32 / kBlock; ++h) {
      float amax = 0.0f;
#pragma unroll
      for (int j = 0; j < kBlock; ++j)
        amax = fmaxf(amax, fabsf(f[h * kBlock + j]));
      float inv;
      uint8_t const s = kFP4 == fp4::kNVFP4 ? fp4::nv_block_scale(amax, gs, inv)
                                            : fp4::mx_block_scale(amax, inv);
      sc |= static_cast<uint32_t>(s) << (8 * h);
#pragma unroll
      for (int k = 0; k < kBlock / 8; ++k) {
        float v8[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) v8[j] = f[h * kBlock + k * 8 + j] * inv;
        w[h * (kBlock / 8) + k] = fp4::pack8_e2m1(v8);
      }
    }
    if (valid)
      *reinterpret_cast<int4*>(dst + e0 / 2) =
          make_int4(static_cast<int>(w[0]), static_cast<int>(w[1]),
                    static_cast<int>(w[2]), static_cast<int>(w[3]));
    // Gather the per-lane scale bytes into 4-byte words before storing.
    uint32_t word;
    int src;
    if constexpr (kFP4 == fp4::kMXFP4) {
      src = (lane_id & 7) * 4;
      word = 0;
#pragma unroll
      for (int k = 0; k < 4; ++k)
        word |= (__shfl_sync(WARP_MASK, sc, src + k) & 0xffu) << (8 * k);
    } else {
      src = (lane_id & 15) * 2;
      word = (__shfl_sync(WARP_MASK, sc, src) & 0xffffu) |
             (__shfl_sync(WARP_MASK, sc, src + 1) << 16);
    }
    constexpr int kNumWordLanes = kFP4 == fp4::kMXFP4 ? 8 : 16;
    if (lane_id < kNumWordLanes and (pass * WARP_SIZE + src) * 32 < kHidden)
      *reinterpret_cast<uint32_t*>(scales + pass * (32 * WARP_SIZE / kBlock) +
                                   lane_id * 4) = word;
  }
}
#endif

template <bool kUseLogFMT, int kHidden, int kNumMaxTopk,
          bool kUseAggressiveAtomic, int kFP4 = 0>
__global__ __launch_bounds__(1024, 1) void combine(
    void* combined_x, void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
    void const* x, int64_t const* topk_idx, float const* topk_weights,
    int const* src_info, int64_t const* layout_range,
    int64_t* combine_wait_recv_cost_stats, int* next_clean,
    int64_t* next_clean_second, int num_next_clean_int, int* atomic_clean_flag,
    int num_combined_tokens, int hidden, int num_topk,
    int num_max_dispatch_tokens_per_rank, int num_experts, int rank,
    int num_ranks, int num_warp_groups, int num_warps_per_group, int phases,
    bool zero_copy, uint64_t const* d2h_channel_addrs,
    int num_d2h_channel_addrs, int max_nvl_peers, int low_latency_buffer_idx,
    void** ipc_rdma_base_ptrs = nullptr, void* rdma_buffer_ptr = nullptr,
    void* atomic_buffer_ptr = nullptr,
    int64_t* rdma_recv_flag_internode = nullptr,
    int* grid_sync_barrier_ptr = nullptr) {
  auto const sm_id = static_cast<int>(blockIdx.x);
  auto const num_sms = static_cast<int>(gridDim.x);
  auto const thread_id = static_cast<int>(threadIdx.x);
  auto const num_threads = static_cast<int>(blockDim.x);
  auto const warp_id = thread_id / WARP_SIZE, lane_id = get_lane_id();
  auto const num_local_experts = num_experts / num_ranks;
  auto const warp_group_id = warp_id / num_warps_per_group;
  auto const sub_warp_id = warp_id % num_warps_per_group;
  auto const responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;
  // Data type staffs
  constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
  constexpr int64_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;
  constexpr int kNumUnrolls = 4;
  constexpr int hidden_bf16_int4_pad =
      align(static_cast<int>(hidden_bf16_int4), WARP_SIZE * kNumUnrolls);
  EP_STATIC_ASSERT(hidden_bf16_int4 % kNumUnrolls == 0, "Invalid hidden");
  EP_STATIC_ASSERT(kNumUnrolls == 1 or kNumUnrolls == 2 or kNumUnrolls == 4,
                   "Invalid unrolling factors");

  // Message package
  constexpr size_t num_bytes_per_slot = kHidden * sizeof(nv_bfloat16);
  EP_STATIC_ASSERT(num_bytes_per_slot % sizeof(int4) == 0,
                   "Invalid vectorization");

  // FP4: the message is written at the start of the BF16-sized slot, so the
  // buffer layout is unchanged; only `kFP4MsgBytes` are copied or put.
  constexpr bool kIsFP4 = kFP4 != 0;
#ifdef UCCL_EP_ENABLE_FP4
  constexpr int kFP4Block = kFP4 == 1 ? 16 : 32;
  constexpr int kFP4NumScales = kHidden / kFP4Block;
  constexpr size_t kFP4MsgBytes =
      kIsFP4 ? fp4::row_bytes(kFP4, kHidden, true) : 0;
  EP_STATIC_ASSERT(not(kIsFP4 and kUseLogFMT), "FP4 excludes LogFMT");
  EP_STATIC_ASSERT(kFP4MsgBytes <= num_bytes_per_slot, "FP4 message too large");
#else
  EP_STATIC_ASSERT(not kIsFP4, "FP4 combine is CUDA-only");
#endif

  // Sending phase
  if ((phases & LOW_LATENCY_SEND_PHASE) == 0) goto LOW_LATENCY_COMBINE_RECV;

  // Clean up next buffer
  if (sm_id == 0 and warp_group_id == 0 and sub_warp_id == 0) {
#pragma unroll
    for (int i = lane_id; i < num_next_clean_int; i += WARP_SIZE) {
      next_clean[i] = 0;
      next_clean_second[i] = 0;
    }

    // Notify before executing `int_p`
    __syncwarp();
    if (lane_id == 0)
      atomic_add_release_global<kUseAggressiveAtomic>(atomic_clean_flag,
                                                      num_experts);
  }

  // Issue RDMA sends
  if (responsible_expert_idx < num_experts) {
    auto const dst_rank = responsible_expert_idx / num_local_experts;
    auto const local_expert_idx = responsible_expert_idx % num_local_experts;
    auto const global_expert_idx = rank * num_local_experts + local_expert_idx;
    auto const layout =
        __ldg(layout_range + local_expert_idx * num_ranks + dst_rank);
    auto const local_x =
        static_cast<int4 const*>(x) + local_expert_idx * num_ranks *
                                          num_max_dispatch_tokens_per_rank *
                                          hidden_bf16_int4;
    auto const local_src_info = src_info + local_expert_idx * num_ranks *
                                               num_max_dispatch_tokens_per_rank;
    auto const rdma_send_x_vec = static_cast<uint8_t*>(rdma_send_x) +
                                 local_expert_idx * num_ranks *
                                     num_max_dispatch_tokens_per_rank *
                                     num_bytes_per_slot;

    // Unpack layout
    int offset, num_tokens_to_send;
    unpack2(layout, num_tokens_to_send, offset);

#if defined(__NVCC__) && !defined(DISABLE_SM90_FEATURES)
    // TMA stuffs
    constexpr int kNumTMABufferBytes = sizeof(int4) * WARP_SIZE * kNumUnrolls;
    constexpr int kNumStages = 3;
    constexpr int kNumPrefetch = 1;
    EP_STATIC_ASSERT(kNumStages == 3 and kNumPrefetch == 1, "Invalid stages");

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_ptr =
        smem_buffer + warp_id * kNumStages * (kNumTMABufferBytes + 16);
    uint32_t tma_phase[kNumStages] = {};
    auto tma_buffer = PatternVisitor([=](int const& i) {
      return reinterpret_cast<int4*>(smem_ptr + i * (kNumTMABufferBytes + 16));
    });
    auto tma_mbarrier = PatternVisitor([=](int const& i) {
      return reinterpret_cast<uint64_t*>(
          smem_ptr + i * (kNumTMABufferBytes + 16) + kNumTMABufferBytes);
    });
    EP_STATIC_ASSERT(kNumUnrolls * kNumStages <= 12,
                     "TMA buffer size exceed limit");

    // Initialize m-barriers
    if (lane_id < kNumStages) {
      mbarrier_init(tma_mbarrier[lane_id], 1);
      fence_view_async_shared();
      fence_barrier_init();
    }
    __syncwarp();

    constexpr int kNumIters = hidden_bf16_int4_pad / (WARP_SIZE * kNumUnrolls);
    auto tma_load_and_arrive = [&](int const& stage_idx, int4 const* gmem_ptr,
                                   int const& num_bytes) {
      tma_load_1d(tma_buffer[stage_idx], gmem_ptr, tma_mbarrier[stage_idx],
                  num_bytes);
      mbarrier_arrive_and_expect_tx(tma_mbarrier[stage_idx], num_bytes);
    };
    auto get_num_tma_bytes = [&](int const& offset_int4) {
      return min(
          kNumTMABufferBytes,
          static_cast<int>((hidden_bf16_int4 - offset_int4) * sizeof(int4)));
    };
#endif

    // Issue RDMA send
    for (int token_idx = offset + sub_warp_id;
         token_idx < offset + num_tokens_to_send;
         token_idx += num_warps_per_group) {
      auto const x_int4 = local_x + token_idx * hidden_bf16_int4;
      auto const rdma_send_type_row = reinterpret_cast<int*>(
          rdma_send_x_vec + token_idx * num_bytes_per_slot);
      auto const rdma_send_x_vec_row =
          reinterpret_cast<uint8_t*>(rdma_send_type_row);

      auto const src_idx =
          __shfl_sync(WARP_MASK, __ldg(local_src_info + token_idx), 0);
      auto const buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);
      auto const dst_ptr =
          reinterpret_cast<uint64_t>(rdma_recv_x) +
          (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) *
              num_bytes_per_slot;
      // TODO(yihan): Mark here for future debugging check.
      // ORIGINAL CODE: Calculate absolute destination address using local
      // rdma_recv_x base auto const dst_ptr =
      //     reinterpret_cast<uint64_t>(rdma_recv_x) +
      //     (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) *
      //         num_bytes_per_slot;

      // Use IPC for intra-node P2P mapping when available
      auto const dst_p2p_ptr =
          ipc_rdma_base_ptrs
              ? uccl::get_ipc_p2p_ptr(dst_ptr, ipc_rdma_base_ptrs, rank,
                                      dst_rank, max_nvl_peers, 0)
              : 0;

#ifdef UCCL_EP_ENABLE_FP4
      if constexpr (kIsFP4) {
        // Quantize straight into the peer's slot (IPC) or the send buffer.
        combine_quant_fp4_row<kFP4, kHidden>(
            x_int4,
            dst_p2p_ptr == 0 ? reinterpret_cast<uint8_t*>(buf_ptr)
                             : reinterpret_cast<uint8_t*>(dst_p2p_ptr),
            lane_id);
        __syncwarp();
      } else
#endif
          if (not zero_copy or dst_p2p_ptr != 0) {
        // Read from `cpy_src_int4_ptr` and copy into `cpy_dst_int4_ptr`
        auto const cpy_src_int4_ptr =
            zero_copy ? reinterpret_cast<int4*>(buf_ptr) : x_int4;
        auto const cpy_dst_int4_ptr =
            dst_p2p_ptr == 0 ? reinterpret_cast<int4*>(buf_ptr)
                             : reinterpret_cast<int4*>(dst_p2p_ptr);

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        // TODO:  Simulated cast
        EP_DEVICE_ASSERT(not kUseLogFMT);
        UNROLLED_WARP_COPY(7, lane_id, hidden_bf16_int4, cpy_dst_int4_ptr,
                           cpy_src_int4_ptr, ld_nc_global, st_na_global);

#elif defined(DISABLE_SM90_FEATURES)
        // Non-SM90 NVIDIA path: simple warp copy (no TMA available)
        UNROLLED_WARP_COPY(7, lane_id, hidden_bf16_int4, cpy_dst_int4_ptr,
                           cpy_src_int4_ptr, ld_nc_global, st_na_global);

#else
        // Prefetch
        if (elect_one_sync(lane_id))
          tma_load_and_arrive(0, cpy_src_int4_ptr, get_num_tma_bytes(0));
        __syncwarp();

#pragma unroll
        for (int i = lane_id * kNumUnrolls, iter_idx = 0;
             i < hidden_bf16_int4_pad;
             i += WARP_SIZE * kNumUnrolls, ++iter_idx) {
          // Read
          int4 int4_values[kNumUnrolls] = {make_int4(0, 0, 0, 0)};
          auto uint32_values = reinterpret_cast<uint32_t*>(int4_values);

          // Load the next iteration
          // TODO: try `elect_one_sync`
          int const& stage_idx = iter_idx % kNumStages;
          int const& next_stage_idx = (iter_idx + 1) % kNumStages;
          tma_store_wait<kNumStages - kNumPrefetch - 1>();
          if (iter_idx + 1 < kNumIters and elect_one_sync(lane_id)) {
            auto const& offset_int4 = i + WARP_SIZE * kNumUnrolls;
            tma_load_and_arrive(next_stage_idx, cpy_src_int4_ptr + offset_int4,
                                get_num_tma_bytes(offset_int4));
          }
          __syncwarp();

          // Wait the current TMA arrival
          mbarrier_wait(tma_mbarrier[stage_idx], tma_phase[stage_idx]);
          auto const& uint32_buffer = reinterpret_cast<uint32_t*>(
              tma_buffer[stage_idx] + lane_id * kNumUnrolls);

          // Simulated cast
          if constexpr (kUseLogFMT) {
            constexpr float kThreshold = 1;
            constexpr float kMinClip = 32;  // `== log_2(2 ^ (2 ^ 5))`
            constexpr int kNumBits = 10;
            constexpr int kNumValues = 1 << (kNumBits - 1);
            EP_STATIC_ASSERT(kHidden % (kNumElemsPerInt4 * WARP_SIZE) == 0 and
                                 kNumElemsPerInt4 == 8,
                             "Invalid hidden");

            // Local log amax
            float log_abs_values[kNumElemsPerInt4 * kNumUnrolls], log_amax,
                log_amin, amax;
            auto log_aminmax = [&](int const& j, float const& value) {
              log_abs_values[j] = log2f_approx(fabsf(value));
              amax = j == 0 ? value : fmaxf(amax, fabsf(value));
              log_amax = j == 0 ? log_abs_values[j]
                                : fmaxf(log_amax, log_abs_values[j]);
              log_amin = value != 0
                             ? (j == 0 ? log_abs_values[j]
                                       : fminf(log_amin, log_abs_values[j]))
                             : log_amin;
            };
#pragma unroll
            for (int k = 0; k < kNumUnrolls * 4; ++k) {
              uint32_values[k] = uint32_buffer[k ^ (lane_id * kNumUnrolls / 8)];
              auto bf162_values =
                  *reinterpret_cast<__nv_bfloat162*>(uint32_values + k);
              auto float2_values = __bfloat1622float2(bf162_values);
              log_aminmax(k * 2, float2_values.x);
              log_aminmax(k * 2 + 1, float2_values.y);
            }

            // Reduce per 128 channels
            amax = warp_reduce_max<(16 / kNumUnrolls)>(amax);
            log_amax = warp_reduce_max<(16 / kNumUnrolls)>(log_amax);
            log_amin = fmaxf(warp_reduce_min<(16 / kNumUnrolls)>(log_amin),
                             log_amax - kMinClip);

            auto const step =
                (log_amax - log_amin) / static_cast<float>(kNumValues - 2);
            auto const step_inv = 1.0f / step;
            auto const rounding =
                2.0f -
                log2f_approx((1.0f + exp2f_approx(step)) * 0.5f) * step_inv;

            // Use LogFMT only with `amax <= kThreshold` (maybe not all
            // quarter-warps)
            if (amax <= kThreshold and log_amin < log_amax) {
              // Transform
              auto transform = [=](float const& log_abs_value) -> nv_bfloat16 {
                auto const encoded =
                    floorf((log_abs_value - log_amin) * step_inv + rounding);
                auto const decoded =
                    exp2f_approx((encoded - 1) * step + log_amin);
                return decoded;
              };
#pragma unroll
              for (int k = 0; k < kNumUnrolls * 4; ++k) {
                auto bf162_pack =
                    __nv_bfloat162(transform(log_abs_values[k * 2]),
                                   transform(log_abs_values[k * 2 + 1]));
                auto uint32_pack = *reinterpret_cast<uint32_t*>(&bf162_pack);
                uint32_buffer[k ^ (lane_id * kNumUnrolls / 8)] =
                    (uint32_values[k] & 0x80008000) | uint32_pack;
              }
            }
            tma_store_fence();
          }
          __syncwarp();

          // Store
          if (elect_one_sync(lane_id))
            tma_store_1d(tma_buffer[stage_idx], cpy_dst_int4_ptr + i,
                         get_num_tma_bytes(i));
          __syncwarp();
        }
#endif
      }

#if defined(__NVCC__) && !defined(DISABLE_SM90_FEATURES)
      // Flush all stores
      tma_store_wait();
      __syncwarp();
#endif

      // Issue RDMA only if we couldn't use IPC
      // NOTES: for zero-copy mode, we assume the data is already in the send
      // buffer
      if (dst_p2p_ptr == 0) {
        __threadfence_system();
        put_nbi_warp(
            dst_ptr - reinterpret_cast<uint64_t>(rdma_buffer_ptr),
            buf_ptr - reinterpret_cast<uint64_t>(rdma_buffer_ptr),
#ifdef UCCL_EP_ENABLE_FP4
            kIsFP4 ? kFP4MsgBytes : hidden * sizeof(nv_bfloat16),
#else
            hidden * sizeof(nv_bfloat16),
#endif
            dst_rank,
            /*warp_id=*/global_expert_idx,  // NOTE(Yang): for selecting rb.
            // NOTE(Ziming): this is global_expert_idx because destination is
            // indexed by global_expert_idx
            lane_id, token_idx - offset, d2h_channel_addrs,
            num_d2h_channel_addrs, true, low_latency_buffer_idx);
      }
    }

    // Put the finishing flag
    EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 16);
    sync_barrier<true>(warp_group_id + 1, num_warps_per_group * WARP_SIZE);
    if (sub_warp_id == 1 and lane_id == 0) {
      while (ld_acquire_global<kUseAggressiveAtomic>(atomic_clean_flag) == 0)
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        __builtin_amdgcn_s_sleep(1);
#else
        ;
#endif
      // Calculate offset from data buffer to flag buffer (similar to dispatch
      // phase) rdma_recv_flag_internode corresponds to
      // combine_rdma_recv_flag_buffer We need to calculate the offset from
      // rdma_recv_x (data buffer) to the flag buffer Try to use IPC for
      // intra-node atomic operations
      auto dst_ptr =
          reinterpret_cast<uint64_t>(rdma_recv_flag + global_expert_idx);
      auto dst_ptr_internode = reinterpret_cast<uint64_t>(
          rdma_recv_flag_internode + global_expert_idx);
      auto dst_p2p_ptr =
          ipc_rdma_base_ptrs
              ? uccl::get_ipc_p2p_ptr(dst_ptr, ipc_rdma_base_ptrs, rank,
                                      dst_rank, max_nvl_peers, 0)
              : 0;
      if (dst_p2p_ptr != 0) {
        // Intra-node: use direct atomic operation
        st_release_sys_global<kUseAggressiveAtomic>(
            reinterpret_cast<int*>(dst_p2p_ptr), 1);
      } else {
        // Inter-node or no IPC: use RDMA atomic
        // NOTE(MaoZiming): Without GPU-initiated RDMA, we can only use atomic
        // add
        // Pass offset to CPU proxy for atomic operation (similar to dispatch
        // phase)
        uccl::atomic_nonfetch_add(
            dst_ptr_internode, reinterpret_cast<uint64_t>(atomic_buffer_ptr),
            num_tokens_to_send /* Will be changed to 1 in the proxy */,
            dst_rank,
            /*warp_id=*/global_expert_idx,  // NOTE(Yang): for selecting rb.
            false, d2h_channel_addrs, num_d2h_channel_addrs, true,
            low_latency_buffer_idx);
      }
      atomic_add_release_global<kUseAggressiveAtomic>(atomic_clean_flag, -1);
    }
    __syncwarp();
  }

// Receiving phase
LOW_LATENCY_COMBINE_RECV:
  if ((phases & LOW_LATENCY_RECV_PHASE) == 0) {
    // if (blockIdx.x == 0 && threadIdx.x == 0)
    //   printf("[combine] SEND finished\n");
    return;
  }
  // Wait all ranks to arrive
  if (responsible_expert_idx < num_experts) {
    EP_DEVICE_ASSERT(num_warps_per_group > 1);
    if (sub_warp_id == 0 and lane_id == 0) {
      auto const src_rank = responsible_expert_idx / num_local_experts;
      auto start_time = clock64();
      while ((src_rank / max_nvl_peers == rank / max_nvl_peers) &&
             ld_acquire_sys_global<kUseAggressiveAtomic>(
                 rdma_recv_flag + responsible_expert_idx) == 0)
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        __builtin_amdgcn_s_sleep(1);
#else
        ;
#endif

      while ((src_rank / max_nvl_peers != rank / max_nvl_peers) &&
             ld_acquire_sys_global<kUseAggressiveAtomic>(
                 reinterpret_cast<uint64_t const*>(
                     rdma_recv_flag_internode + responsible_expert_idx)) == 0)
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        __builtin_amdgcn_s_sleep(1);
#else
        ;
#endif

      if (src_rank / max_nvl_peers == rank / max_nvl_peers) {
        if (ld_acquire_sys_global<kUseAggressiveAtomic>(
                reinterpret_cast<uint64_t const*>(
                    rdma_recv_flag_internode + responsible_expert_idx)) != 0) {
          printf(
              "Same node but rdma_recv_flag_internode is not zero! src_rank: "
              "%d, rank: %d, max_nvl_peers: %d\n",
              src_rank, rank, max_nvl_peers);
          EP_DEVICE_ASSERT(false);
        }
      }
      if (src_rank / max_nvl_peers != rank / max_nvl_peers) {
        if (ld_acquire_sys_global<kUseAggressiveAtomic>(
                rdma_recv_flag + responsible_expert_idx) != 0) {
          printf(
              "Different node but rdma_recv_flag is not zero! src_rank: %d, "
              "rank: %d, max_nvl_peers: %d\n",
              src_rank, rank, max_nvl_peers);
          EP_DEVICE_ASSERT(false);
        }
      }

      auto wait_recv_cost = clock64() - start_time;
      if (combine_wait_recv_cost_stats != nullptr) {
        auto const& src_rank = responsible_expert_idx / num_local_experts;
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      combine_wait_recv_cost_stats + src_rank),
                  wait_recv_cost);
      }
    }
  }
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  amd::grid_sync_then_zero(grid_sync_barrier_ptr, num_sms);
#else
  cg::this_grid().sync();
#endif

  // Reduce tokens
  EP_DEVICE_ASSERT(num_topk <= WARP_SIZE);
  EP_STATIC_ASSERT(kHidden % (WARP_SIZE * kNumElemsPerInt4) == 0,
                   "Invalid vectorization");
  for (int hidden_idx = thread_id; hidden_idx < hidden_bf16_int4;
       hidden_idx += num_threads) {
    for (int token_idx = sm_id; token_idx < num_combined_tokens;
         token_idx += num_sms) {
      // Read top-k indices and weights
      int reg_topk_idx[kNumMaxTopk];
      float reg_topk_weights[kNumMaxTopk];
#pragma unroll
      for (int i = 0; i < num_topk; ++i) {
        reg_topk_idx[i] =
            static_cast<int>(__ldg(topk_idx + token_idx * num_topk + i));
        reg_topk_weights[i] = __ldg(topk_weights + token_idx * num_topk + i);
      }

      float combined_values[kNumElemsPerInt4] = {0.0f};
#pragma unroll
      for (int i = 0; i < num_topk; ++i)
        if (reg_topk_idx[i] >= 0) {
          // Read from sources
          auto rdma_buffer_type = reinterpret_cast<int const*>(
              static_cast<uint8_t*>(rdma_recv_x) +
              (reg_topk_idx[i] * num_max_dispatch_tokens_per_rank + token_idx) *
                  num_bytes_per_slot);
          auto rdma_buffer_row =
              reinterpret_cast<uint8_t const*>(rdma_buffer_type);

#ifdef UCCL_EP_ENABLE_FP4
          if constexpr (kIsFP4) {
            // 8 elements = one 4-byte data word; all 8 share a block scale.
            auto const w = static_cast<uint32_t>(ld_nc_global(
                reinterpret_cast<int const*>(rdma_buffer_row) + hidden_idx));
            auto const sb = ld_nc_global(rdma_buffer_row + kHidden / 2 +
                                         hidden_idx * 8 / kFP4Block);
            float scale;
            if constexpr (kFP4 == 1)
              scale = fp4::e4m3_to_float(sb) *
                      ld_nc_global(reinterpret_cast<float const*>(
                          rdma_buffer_row + kHidden / 2 + kFP4NumScales));
            else
              scale = fp4::mx_decode_scale(sb);
            scale *= reg_topk_weights[i];
#pragma unroll
            for (int j = 0; j < kNumElemsPerInt4; ++j)
              combined_values[j] += fp4::e2m1_decode(w >> (4 * j)) * scale;
            continue;
          }
#endif
          // Reduce
          auto x_vec = ld_nc_global(
              reinterpret_cast<int4 const*>(rdma_buffer_row) + hidden_idx);
          auto const x_bf16 = reinterpret_cast<nv_bfloat16*>(&x_vec);
#pragma unroll
          for (int j = 0; j < kNumElemsPerInt4; ++j)
            combined_values[j] +=
                static_cast<float>(x_bf16[j]) * reg_topk_weights[i];
        }

      // Write results
      int4& combined_int4 = *reinterpret_cast<int4*>(combined_values);
      auto combined_bf16 = reinterpret_cast<nv_bfloat16*>(&combined_values);
#pragma unroll
      for (int j = 0; j < kNumElemsPerInt4; ++j)
        combined_bf16[j] = static_cast<nv_bfloat16>(combined_values[j]);
      (static_cast<int4*>(combined_x) +
       token_idx * hidden_bf16_int4)[hidden_idx] = combined_int4;
    }

    // if (blockIdx.x == 0 && threadIdx.x == 0)
    //   printf("[combine] RECV finished\n");
  }
}

void combine(void* combined_x, void* rdma_recv_x, int* rdma_recv_flag,
             void* rdma_send_x, void const* x, int64_t const* topk_idx,
             float const* topk_weights, int const* src_info,
             int64_t const* layout_range, int64_t* combine_wait_recv_cost_stats,
             int* next_clean, int64_t* next_clean_second,
             int num_next_clean_int, int num_combined_tokens, int hidden,
             int num_max_dispatch_tokens_per_rank, int num_topk,
             int num_experts, int rank, int num_ranks, bool use_logfmt,
             void* workspace, int num_device_sms, cudaStream_t stream,
             int phases, bool zero_copy, uint64_t const* d2h_channel_addrs,
             int num_d2h_channel_addrs, int max_nvl_peers,
             int low_latency_buffer_idx, void** ipc_rdma_base_ptrs,
             void* rdma_buffer_ptr, void* atomic_buffer_ptr,
             int64_t* rdma_recv_flag_internode, int fp4_mode) {
  constexpr int kNumMaxTopk = 16;
  int const num_warp_groups = ceil_div(num_experts, num_device_sms);
  int const num_warps_per_group = kNumMaxWarpGroups / num_warp_groups;
  EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0);

  auto const num_warps = num_warp_groups * num_warps_per_group;
  auto const num_sms = ceil_div(num_experts, num_warp_groups);

  // Check workspace
  auto atomic_clean_flag = static_cast<int*>(workspace);
  auto grid_sync_barrier_ptr = atomic_clean_flag + 1;
  EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
  EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

  // Online cast cannot use zero-copy
  EP_HOST_ASSERT(not(zero_copy and use_logfmt));
  EP_HOST_ASSERT(fp4_mode >= 0 and fp4_mode <= 2);
  if (fp4_mode != 0) {
#ifndef UCCL_EP_ENABLE_FP4
    EP_HOST_ASSERT(false and "FP4 combine is CUDA-only");
#endif
    EP_HOST_ASSERT(not use_logfmt and not zero_copy);
    EP_HOST_ASSERT(hidden % 512 == 0);
  }

  constexpr int kNumTMABytesPerWarp = 12 * (512 + 16);
  int num_warps_launch = num_warps;
  int num_warps_per_group_launch = num_warps_per_group;
#if defined(__NVCC__) && !defined(DISABLE_SM90_FEATURES)
  int device_id = 0;
  if (cudaGetDevice(&device_id) == cudaSuccess) {
    int max_smem = 0;
    if (cudaDeviceGetAttribute(&max_smem,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               device_id) == cudaSuccess &&
        max_smem > 0) {
      size_t const requested_smem = kNumTMABytesPerWarp * num_warps;
      if (requested_smem > static_cast<size_t>(max_smem)) {
        int const max_warps_by_smem = max_smem / kNumTMABytesPerWarp;
        num_warps_launch =
            (max_warps_by_smem / num_warp_groups) * num_warp_groups;
        if (num_warps_launch < num_warp_groups)
          num_warps_launch = num_warp_groups;
        if (num_warps_launch >= 2 * num_warp_groups) {
          num_warps_per_group_launch = num_warps_launch / num_warp_groups;
        } else {
          num_warps_launch = num_warps;
          num_warps_per_group_launch = num_warps_per_group;
        }
      }
    }
  }
#endif
  int const smem_size = kNumTMABytesPerWarp * num_warps_launch;
  // printf("Combine launched\n");

  static bool const aggressive_atomic_enabled = get_aggressive_atomic_enabled();

#define COMBINE_LAUNCH_CASE(hidden)                                          \
  {                                                                          \
    auto combine_func =                                                      \
        aggressive_atomic_enabled                                            \
            ? (use_logfmt ? combine<true, hidden, kNumMaxTopk, true>         \
                          : combine<false, hidden, kNumMaxTopk, true>)       \
            : (use_logfmt ? combine<true, hidden, kNumMaxTopk, false>        \
                          : combine<false, hidden, kNumMaxTopk, false>);     \
    COMBINE_FP4_CASES(hidden);                                               \
    SET_SHARED_MEMORY_FOR_TMA(combine_func);                                 \
    LAUNCH_KERNEL(                                                           \
        &cfg, combine_func, combined_x, rdma_recv_x, rdma_recv_flag,         \
        rdma_send_x, x, topk_idx, topk_weights, src_info, layout_range,      \
        combine_wait_recv_cost_stats, next_clean, next_clean_second,         \
        num_next_clean_int, atomic_clean_flag, num_combined_tokens, hidden,  \
        num_topk, num_max_dispatch_tokens_per_rank, num_experts, rank,       \
        num_ranks, num_warp_groups, num_warps_per_group_launch, phases,      \
        zero_copy, d2h_channel_addrs, num_d2h_channel_addrs, max_nvl_peers,  \
        low_latency_buffer_idx, ipc_rdma_base_ptrs, rdma_buffer_ptr,         \
        atomic_buffer_ptr, rdma_recv_flag_internode, grid_sync_barrier_ptr); \
  }                                                                          \
  break

#ifdef UCCL_EP_ENABLE_FP4
#define COMBINE_FP4_CASES(hidden)                                       \
  if (fp4_mode == 1)                                                    \
    combine_func = aggressive_atomic_enabled                            \
                       ? combine<false, hidden, kNumMaxTopk, true, 1>   \
                       : combine<false, hidden, kNumMaxTopk, false, 1>; \
  if (fp4_mode == 2)                                                    \
  combine_func = aggressive_atomic_enabled                              \
                     ? combine<false, hidden, kNumMaxTopk, true, 2>     \
                     : combine<false, hidden, kNumMaxTopk, false, 2>
#else
#define COMBINE_FP4_CASES(hidden) (void)0
#endif
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  EP_HOST_ASSERT(num_warps_launch * WARP_SIZE <= MAX_NTHREADS);
#endif

  SETUP_LAUNCH_CONFIG(num_sms, num_warps_launch * WARP_SIZE, stream);
  SWITCH_HIDDEN(COMBINE_LAUNCH_CASE);
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("[combine] kernel launch error: %s\n", cudaGetErrorString(err));
    fflush(stdout);
  }
#undef COMBINE_LAUNCH_CASE
#undef COMBINE_FP4_CASES
}

}  // namespace internode_ll
}  // namespace uccl
