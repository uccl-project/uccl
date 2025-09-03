#include "ep_configs.cuh"
#include "ep_launch.cuh"
#include "ep_runtime.cuh"
#include "ep_util.hpp"
#include "ep_utils.cuh"
#include "internode_ll.cuh"
#include "uccl_ibgda.cuh"
#include <iostream>
#include <vector>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
namespace uccl {
namespace internode_ll {

template <bool kUseFP8, bool kUseUE8M0, int kHidden>
__global__ __launch_bounds__(1024, 1) void dispatch(
    void* packed_recv_x, void* packed_recv_x_scales, int* packed_recv_src_info,
    int64_t* packed_recv_layout_range, int* packed_recv_count,
    int* cumulative_local_expert_recv_stats,
    int64_t* dispatch_wait_recv_cost_stats, void* rdma_recv_x,
    int* rdma_recv_count, void* rdma_x, void const* x, int64_t const* topk_idx,
    int* atomic_counter_per_expert, int* atomic_finish_counter_per_expert,
    int* next_clean, int num_next_clean_int, int num_tokens,
    int num_max_dispatch_tokens_per_rank, int num_topk, int num_experts,
    int rank, int num_ranks, int num_warp_groups, int num_warps_per_group,
    bool round_scale, int phases, uint64_t const* ring_addrs,
    int num_ring_addrs, int max_nvl_peers, void** ipc_base_ptrs = nullptr,
    void* atomic_buffer_ptr = nullptr) {
  auto const sm_id = static_cast<int>(blockIdx.x);
  auto const thread_id = static_cast<int>(threadIdx.x);
  auto const warp_id = thread_id / 32, lane_id = get_lane_id();
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

  // FP8 staffs
  constexpr int kNumPerChannels = 128;
  int const num_scales = kHidden / kNumPerChannels;
  size_t const hidden_bytes =
      kHidden * (kUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));
  size_t const hidden_int4 = hidden_bytes / sizeof(int4);

  // Message package: hidden data, FP8 scales, index at source
  // NOTES: currently we have 3 reserved int fields for future use
  using vec_t = typename std::conditional<kUseFP8, int2, int4>::type;
  size_t const num_bytes_per_msg =
      sizeof(int4) + (kUseFP8 ? (kHidden + num_scales * sizeof(float))
                              : (kHidden * sizeof(nv_bfloat16)));
  size_t const num_int4_per_msg = num_bytes_per_msg / sizeof(int4);
  EP_DEVICE_ASSERT(num_bytes_per_msg % sizeof(int4) == 0);

  // Expert counts
  constexpr int kNumMaxWarpGroups = 32;
  __shared__ int shared_num_tokens_sent_per_expert[kNumMaxWarpGroups];

  // Sending phase
  if ((phases & LOW_LATENCY_SEND_PHASE) == 0) goto LOW_LATENCY_DISPATCH_RECV;

  // There are 2 kinds of warps in this part:
  // 1. The first-kind warps for FP8 cast and sending top-k tokens
  // 2. The last warp for reading `topk_idx` and count for per-expert
  // information
  if (warp_id < num_warps - 1) {
    constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
    EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerRead) == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumElemsPerRead * 32 % kNumPerChannels == 0,
                     "Invalid vectorization");
    auto const num_threads = (num_warps - 1) * 32;
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

// FP8 cast
#pragma unroll
      for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
        // Read
        auto int4_value = __ldg(x_int4 + i);

        if constexpr (kUseFP8) {
          // Calculate local amax
          auto bf16_values = reinterpret_cast<nv_bfloat16*>(&int4_value);
          float fp32_values[kNumElemsPerRead];
          float amax = kFP8Margin, scale, scale_inv;
#pragma unroll
          for (int j = 0; j < kNumElemsPerRead; ++j) {
            fp32_values[j] = static_cast<float>(bf16_values[j]);
            amax = fmaxf(amax, fabsf(fp32_values[j]));
          }

          // Reduce amax and scale
          EP_STATIC_ASSERT(kNumElemsPerRead * 32 / kNumPerChannels == 2,
                           "Invalid vectorization");
          amax = warp_reduce_max<16>(amax);
          calculate_fp8_scales(amax, scale, scale_inv, round_scale);
          if (lane_id == 0 or lane_id == 16)
            rdma_x_scales[i * kNumElemsPerRead / 128] = scale_inv;

          // Cast into send buffer
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
          rdma_x_vec[i] = int2_value;
        } else {
          // Reinterpret-cast is for C++14 compatibility
          rdma_x_vec[i] = *reinterpret_cast<vec_t*>(&int4_value);
        }
      }
      asm volatile("bar.sync 1, %0;" ::"r"(num_threads));

      // Issue IBGDA sends
      if (dst_expert_idx >= 0) {
        int slot_idx =
            lane_id == 0
                ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1)
                : 0;
        slot_idx = __shfl_sync(0xffffffff, slot_idx, 0);
        auto const dst_rank = dst_expert_idx / num_local_experts;
        auto const dst_expert_local_idx = dst_expert_idx % num_local_experts;
        auto const src_ptr = reinterpret_cast<uint64_t>(rdma_x_src_idx);
        // Pass offset relative to dispatch_rdma_recv_data_buffer (rdma_recv_x)
        // CPU proxy will add the base offset of dispatch_rdma_recv_data_buffer
        // within rdma_buffer
        auto const dst_offset =
            dst_expert_local_idx * num_ranks *
                num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
            rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
            slot_idx * num_bytes_per_msg;

        // Try to use IPC for intra-node communication

        auto const dst_p2p_ptr =
            ipc_base_ptrs
                ? uccl::get_ipc_p2p_ptr(
                      reinterpret_cast<void*>(
                          reinterpret_cast<char*>(rdma_recv_x) + dst_offset),
                      ipc_base_ptrs, rank, dst_rank, max_nvl_peers, 0)
                : nullptr;

        if (dst_p2p_ptr != nullptr) {
          // Intra-node: use direct memory copy via IPC
          auto const* src_int4_ptr = reinterpret_cast<int4 const*>(src_ptr);
          auto* dst_int4_ptr = reinterpret_cast<int4*>(dst_p2p_ptr);
          UNROLLED_WARP_COPY(8, lane_id, num_int4_per_msg, dst_int4_ptr,
                             src_int4_ptr, ld_nc_global, st_na_global);
        } else {
          // Inter-node or no IPC: use IBGDA
          uccl::nvshmemi_ibgda_put_nbi_warp(
              dst_offset, src_ptr, num_bytes_per_msg, dst_rank,
              sm_id,  // NOTE(MaoZiming): use sm_id for rb.
              lane_id, slot_idx, ring_addrs, num_ring_addrs, false);
        }
        // Increase counter after finishing
        __syncwarp();
        lane_id == 0 ? atomic_add_release_global(
                           atomic_finish_counter_per_expert + dst_expert_idx, 1)
                     : 0;
      }
    }
  } else if (warp_id == num_warps - 1) {
    // EP_DEVICE_ASSERT(num_sms > 1);
    // NOTE(MaoZiming): These checks are ibgda specific.
    if (sm_id == 0) {
#ifdef false
      // The first SM is also responsible for checking QPs
      if (num_ranks > 1 && gridDim.x > 1) {
        // The first SM is also responsible for checking QPs (multi-rank,
        // multi-SM only)
        EP_DEVICE_ASSERT(uccl::ibgda_get_state() != nullptr);
        EP_DEVICE_ASSERT(uccl::ibgda_get_state()->num_rc_per_pe >=
                         num_local_experts);
      }
#endif

// The first SM is also responsible for cleaning the next buffer
#pragma unroll
      for (int i = lane_id; i < num_next_clean_int; i += 32) next_clean[i] = 0;

      // Notify before executing `int_p`
      __syncwarp();
#pragma unroll
      for (int i = lane_id; i < num_experts; i += 32)
        atomic_add_release_global(atomic_finish_counter_per_expert + i,
                                  FINISHED_SUM_TAG);
    }

    // This SM should be responsible for some destination experts, read
    // `topk_idx` for them
    int expert_count[kNumMaxWarpGroups] = {0};
    auto const expert_begin_idx = sm_id * num_warp_groups;
    auto const expert_end_idx =
        min(expert_begin_idx + num_warp_groups, num_experts);

// Per lane count
#pragma unroll 8
    for (int i = lane_id; i < num_tokens * num_topk; i += 32) {
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
        atomic_add_release_global(atomic_finish_counter_per_expert + i,
                                  FINISHED_SUM_TAG - sum);
      }
    }
  }
  __syncthreads();

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
    while (ld_acquire_global(atomic_finish_counter_per_expert +
                             responsible_expert_idx) != FINISHED_SUM_TAG * 2)
      ;
    // TODO(yihan): Mark here for future debugging check.
    // Calculate offset within LowLatencyLayout buffer for CPU proxy translation
    // Calculate offset relative to dispatch_rdma_recv_data_buffer (rdma_recv_x)
    // CPU proxy will translate this to the correct
    // dispatch_rdma_recv_count_buffer address

    // TODO: Ensure 8-byte alignment for 64-bit atomic operations
    // Calculate index, multiply by 8 instead of 4, then add to base
    // NOTE: index by source rank.
    auto count_index = dst_expert_local_idx * num_ranks + rank;
    auto aligned_count_addr = reinterpret_cast<uint64_t>(rdma_recv_count) +
                              count_index * sizeof(uint64_t);
    auto dst_offset =
        aligned_count_addr - reinterpret_cast<uint64_t>(atomic_buffer_ptr);

    // Try to use IPC for intra-node atomic operations
    auto const dst_p2p_ptr =
        // NOTE(Ziming): it seems there is a mismatch from aligned_count_addr
        // before.
        ipc_base_ptrs ? uccl::get_ipc_p2p_ptr(
                            // reinterpret_cast<void*>(
                            // rdma_recv_count +
                            // dst_expert_local_idx * num_ranks + rank),
                            reinterpret_cast<void*>(aligned_count_addr),
                            ipc_base_ptrs, rank, dst_rank, max_nvl_peers, 0)
                      : nullptr;

    printf("get_ipc_p2p_ptr=0x%" PRIxPTR ", dst_offset=%" PRId64
           ", src_rank=%d, dst_rank=%d, ranks_per_node=%d, count_index: %d, "
           "expert_idx=%d\n",
           (uintptr_t)dst_p2p_ptr, dst_offset, rank, dst_rank, max_nvl_peers,
           count_index, responsible_expert_idx);
    if (dst_p2p_ptr != nullptr) {
      // Intra-node: use direct atomic operation
      st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr),
                            -num_tokens_sent - 1);
    } else {
      // Inter-node or no IPC: use IBGDA atomic
      uccl::nvshmemi_ibgda_amo_nonfetch_add(
          dst_offset, -num_tokens_sent - 1, dst_rank, sm_id,
          dst_expert_local_idx, false, ring_addrs, num_ring_addrs);
    }
    // Clean workspace for next use
    atomic_counter_per_expert[responsible_expert_idx] = 0;
    atomic_finish_counter_per_expert[responsible_expert_idx] = 0;

    // Clean `packed_recv_count`
    if (dst_rank == 0) packed_recv_count[dst_expert_local_idx] = 0;
  }
  __syncwarp();

// Receiving phase
LOW_LATENCY_DISPATCH_RECV:
  if ((phases & LOW_LATENCY_RECV_PHASE) == 0) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
      printf("[dispatch] SEND finished\n");
    return;
  }

  // For send-and-recv kernels, we need a grid sync for making
  // `packed_recv_count` visible
  if (phases & LOW_LATENCY_SEND_PHASE) cg::this_grid().sync();

  // Receiving and packing
  if (responsible_expert_idx < num_experts) {
    auto const src_rank = responsible_expert_idx / num_local_experts;
    auto const local_expert_idx = responsible_expert_idx % num_local_experts;
    auto const rdma_recv_x_uint8 =
        static_cast<uint8_t*>(rdma_recv_x) +
        local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank *
            num_bytes_per_msg +
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
    auto const recv_x_scales = static_cast<scale_t*>(packed_recv_x_scales) +
                               local_expert_idx * num_ranks *
                                   num_max_dispatch_tokens_per_rank *
                                   num_aligned_scales;

    // Shared between sub-warps in warp groups
    __shared__ int shared_num_recv_tokens[kNumMaxWarpGroups],
        shared_recv_token_begin_idx[kNumMaxWarpGroups];

    // Wait tokens to arrive
    // NOTES: using sub-warp 1 to overlap with sub-warp 0
    int num_recv_tokens, recv_token_begin_idx;
    EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 15);
    if (sub_warp_id == 1 and lane_id == 0) {
      auto start_time = clock64();
      // TODO: Use same 8-byte aligned address calculation as sender
      // Fixed: Calculate index, multiply by 8, then cast to int* for
      // compatibility
      auto count_index = local_expert_idx * num_ranks + src_rank;
      auto aligned_count_addr = reinterpret_cast<uint64_t>(rdma_recv_count) +
                                count_index * sizeof(uint64_t);
      int* count_addr = reinterpret_cast<int*>(aligned_count_addr);

      printf("Before waiting for tokens, count_addr: %p\n", (void*)count_addr);
      while ((num_recv_tokens = ld_acquire_sys_global(count_addr)) == 0)
        ;
      auto wait_recv_cost = clock64() - start_time;
      num_recv_tokens = -num_recv_tokens - 1;
      printf(
          "[RECV_COUNT_DECODED] Decoded token count: %d (from received value "
          "%d), count_addr; %p\n",
          num_recv_tokens, -num_recv_tokens - 1, (void*)count_addr);
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
    asm volatile("bar.sync %0, %1;" ::"r"(warp_group_id + 2),
                 "r"(num_warps_per_group * 32));
    num_recv_tokens = shared_num_recv_tokens[warp_group_id];
    recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];

    // Copy tokens
    EP_DEVICE_ASSERT(num_scales <= 64);
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

      // Copy scales
      if constexpr (kUseFP8) {
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
        if (lane_id + 32 < num_scales) {
          auto const pack_idx = (lane_id + 32) / num_elems_per_pack;
          auto const elem_idx = (lane_id + 32) % num_elems_per_pack;
          auto scale = extract_required_scale_format<kUseUE8M0>(
              ld_nc_global(src_scales + lane_id + 32));
          recv_x_scales[token_idx * token_stride + pack_idx * pack_stride +
                        elem_idx] = scale;
        }
      }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0)
      printf("[dispatch] RECV finished\n");
  }
}

void dispatch(void* packed_recv_x, void* packed_recv_x_scales,
              int* packed_recv_src_info, int64_t* packed_recv_layout_range,
              int* packed_recv_count, int* cumulative_local_expert_recv_stats,
              int64_t* dispatch_wait_recv_cost_stats, void* rdma_recv_x,
              int* rdma_recv_count, void* rdma_x, void const* x,
              int64_t const* topk_idx, int* next_clean, int num_next_clean_int,
              int num_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
              int num_topk, int num_experts, int rank, int num_ranks,
              bool use_fp8, bool round_scale, bool use_ue8m0, void* workspace,
              int num_device_sms, cudaStream_t stream, int phases,
              uint64_t const* ring_addrs, int num_ring_addrs, int max_nvl_peers,
              void** ipc_base_ptrs, void* atomic_buffer_ptr) {
  constexpr int kNumMaxTopK = 9;
  int const num_warp_groups = ceil_div(num_experts, num_device_sms);
  int const num_warps_per_group = 32 / num_warp_groups;
  EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0);
  EP_HOST_ASSERT(kNumMaxTopK + 1 <= num_warp_groups * num_warps_per_group);

  auto const num_warps = num_warp_groups * num_warps_per_group;
  auto const num_sms = ceil_div(num_experts, num_warp_groups);
  EP_HOST_ASSERT(num_topk <= kNumMaxTopK);

  // Workspace checks
  auto atomic_counter_per_expert = static_cast<int*>(workspace);
  auto atomic_finish_counter_per_expert =
      atomic_counter_per_expert + num_experts;
  EP_HOST_ASSERT(num_experts * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);

  // FP8 checks
  if (use_ue8m0)
    EP_HOST_ASSERT(round_scale and "UE8M0 SF requires `round_scale=True`");

#define DISPATCH_LAUNCH_CASE(hidden)                                           \
  {                                                                            \
    auto dispatch_func = dispatch<false, false, hidden>;                       \
    if (use_fp8 and not use_ue8m0)                                             \
      dispatch_func = dispatch<true, false, hidden>;                           \
    if (use_fp8 and use_ue8m0) dispatch_func = dispatch<true, true, hidden>;   \
    LAUNCH_KERNEL(&cfg, dispatch_func, packed_recv_x, packed_recv_x_scales,    \
                  packed_recv_src_info, packed_recv_layout_range,              \
                  packed_recv_count, cumulative_local_expert_recv_stats,       \
                  dispatch_wait_recv_cost_stats, rdma_recv_x, rdma_recv_count, \
                  rdma_x, x, topk_idx, atomic_counter_per_expert,              \
                  atomic_finish_counter_per_expert, next_clean,                \
                  num_next_clean_int, num_tokens,                              \
                  num_max_dispatch_tokens_per_rank, num_topk, num_experts,     \
                  rank, num_ranks, num_warp_groups, num_warps_per_group,       \
                  round_scale, phases, ring_addrs, num_ring_addrs,             \
                  max_nvl_peers, ipc_base_ptrs, atomic_buffer_ptr);            \
  }                                                                            \
  break

  SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
  SWITCH_HIDDEN(DISPATCH_LAUNCH_CASE);
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("[dispatch] kernel launch error: %s\n", cudaGetErrorString(err));
    fflush(stdout);
  }

#undef DISPATCH_LAUNCH_CASE
}

template <bool kUseLogFMT, int kHidden, int kNumMaxTopk>
__global__ __launch_bounds__(1024, 1) void combine(
    void* combined_x, void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
    void const* x, int64_t const* topk_idx, float const* topk_weights,
    int const* src_info, int64_t const* layout_range,
    int64_t* combine_wait_recv_cost_stats, int* next_clean,
    int num_next_clean_int, int* atomic_clean_flag, int num_combined_tokens,
    int hidden, int num_topk, int num_max_dispatch_tokens_per_rank,
    int num_experts, int rank, int num_ranks, int num_warp_groups,
    int num_warps_per_group, int phases, bool zero_copy,
    uint64_t const* ring_addrs, int num_ring_addrs, int max_nvl_peers,
    void** ipc_base_ptrs = nullptr, void* rdma_buffer_ptr = nullptr,
    void* atomic_buffer_ptr = nullptr) {
  auto const sm_id = static_cast<int>(blockIdx.x);
  auto const num_sms = static_cast<int>(gridDim.x);
  auto const thread_id = static_cast<int>(threadIdx.x);
  auto const num_threads = static_cast<int>(blockDim.x);
  auto const warp_id = thread_id / 32, lane_id = get_lane_id();
  auto const num_local_experts = num_experts / num_ranks;
  auto const warp_group_id = warp_id / num_warps_per_group;
  auto const sub_warp_id = warp_id % num_warps_per_group;
  auto const responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;
  // Data type staffs
  constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
  constexpr int64_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;
  constexpr int kNumUnrolls = 4;
  constexpr int hidden_bf16_int4_pad =
      align(static_cast<int>(hidden_bf16_int4), 32 * kNumUnrolls);
  EP_STATIC_ASSERT(hidden_bf16_int4 % kNumUnrolls == 0, "Invalid hidden");
  EP_STATIC_ASSERT(kNumUnrolls == 1 or kNumUnrolls == 2 or kNumUnrolls == 4,
                   "Invalid unrolling factors");

  // Message package
  constexpr size_t num_bytes_per_slot = kHidden * sizeof(nv_bfloat16);
  EP_STATIC_ASSERT(num_bytes_per_slot % sizeof(int4) == 0,
                   "Invalid vectorization");

  // Sending phase
  if ((phases & LOW_LATENCY_SEND_PHASE) == 0) goto LOW_LATENCY_COMBINE_RECV;

  // Clean up next buffer
  if (sm_id == 0 and warp_group_id == 0 and sub_warp_id == 0) {
#pragma unroll
    for (int i = lane_id; i < num_next_clean_int; i += 32) next_clean[i] = 0;

    // Notify before executing `int_p`
    __syncwarp();
    if (lane_id == 0) atomic_add_release_global(atomic_clean_flag, num_experts);
  }

  // Issue IBGDA sends
  if (responsible_expert_idx < num_experts) {
    auto const dst_rank = responsible_expert_idx / num_local_experts;
    auto const local_expert_idx = responsible_expert_idx % num_local_experts;
    auto const global_expert_idx = rank * num_local_experts + local_expert_idx;
    auto const layout =
        __ldg(layout_range + local_expert_idx * num_ranks + dst_rank);
    // auto const local_x =
    //     static_cast<int4 const*>(x) + local_expert_idx * num_ranks *
    //                                       num_max_dispatch_tokens_per_rank *
    //                                       hidden_bf16_int4;
    auto const local_src_info = src_info + local_expert_idx * num_ranks *
                                               num_max_dispatch_tokens_per_rank;

    // NOTE(NEW): !!! Check logic. This differs from DeepEP.
    int const slice_len = num_max_dispatch_tokens_per_rank;
    auto const rdma_send_x_vec =
        static_cast<uint8_t*>(rdma_send_x) +
        ((size_t)local_expert_idx * (size_t)num_ranks + (size_t)dst_rank) *
            (size_t)slice_len * (size_t)num_bytes_per_slot;

    // Unpack layout
    int offset, num_tokens_to_send;
    unpack2(layout, num_tokens_to_send, offset);

    // TMA stuffs
    constexpr int kNumTMABufferBytes = sizeof(int4) * 32 * kNumUnrolls;
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

    constexpr int kNumIters = hidden_bf16_int4_pad / (32 * kNumUnrolls);
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

    // Issue IBGDA send
    for (int token_idx = offset + sub_warp_id;
         token_idx < offset + num_tokens_to_send;
         token_idx += num_warps_per_group) {
      // NOTE(NEW): !!! Check logic. This differs from DeepEP.
      // const auto x_int4 = local_x + token_idx * hidden_bf16_int4;
      auto const rdma_send_type_row = reinterpret_cast<int*>(
          rdma_send_x_vec + token_idx * num_bytes_per_slot);
      auto const rdma_send_x_vec_row =
          reinterpret_cast<uint8_t*>(rdma_send_type_row);

      // Copy directly to local rank, or copy to buffer and issue RDMA
      // NOTE: New
      auto const src_idx = __shfl_sync(
          0xffffffff, __ldg(local_src_info + dst_rank * slice_len + token_idx),
          0);

      if (src_idx < 0 || src_idx >= num_ranks * slice_len) {
        // TODO: sometimes src_idx will be greater. Not sure why.
        if (lane_id == 0) {
          printf("BAD src_idx=%d total=%d dst_rank=%d token_idx=%d offset=%d\n",
                 src_idx, num_ranks * slice_len, dst_rank, token_idx, offset);
        }
        continue;
      }
      auto const x_int4 =
          static_cast<int4 const*>(x) + src_idx * hidden_bf16_int4;
      auto const buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);

      // TODO(yihan): Mark here for future debugging check.
      // ORIGINAL CODE: Calculate absolute destination address using local
      // rdma_recv_x base auto const dst_ptr =
      //     reinterpret_cast<uint64_t>(rdma_recv_x) +
      //     (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) *
      //         num_bytes_per_slot;

      // NEW APPROACH: Calculate offset within LowLatencyLayout buffer for CPU
      // proxy translation
      uint64_t dst_offset = (static_cast<uint64_t>(global_expert_idx) *
                                 num_max_dispatch_tokens_per_rank +
                             static_cast<uint64_t>(src_idx)) *
                            static_cast<uint64_t>(num_bytes_per_slot);

      // Use IPC for intra-node P2P mapping when available
      auto const dst_p2p_ptr =
          ipc_base_ptrs
              ? uccl::get_ipc_p2p_ptr(
                    reinterpret_cast<void*>(
                        reinterpret_cast<char*>(rdma_recv_x) + dst_offset),
                    ipc_base_ptrs, rank, dst_rank, max_nvl_peers, 0)
              : nullptr;

      // NOTE: Otherwise overflow.
      uint64_t dst_off = (reinterpret_cast<uint64_t>(rdma_recv_x) -
                          reinterpret_cast<uint64_t>(rdma_buffer_ptr)) +
                         dst_offset;

      if (not zero_copy or dst_p2p_ptr != nullptr) {
        // Read from `cpy_src_int4_ptr` and copy into `cpy_dst_int4_ptr`
        auto const cpy_src_int4_ptr =
            zero_copy ? reinterpret_cast<int4*>(buf_ptr) : x_int4;
        auto const cpy_dst_int4_ptr =
            dst_p2p_ptr == nullptr ? reinterpret_cast<int4*>(buf_ptr)
                                   : reinterpret_cast<int4*>(dst_p2p_ptr);

        // Prefetch
        if (elect_one_sync(lane_id))
          tma_load_and_arrive(0, cpy_src_int4_ptr, get_num_tma_bytes(0));
        __syncwarp();

#pragma unroll
        for (int i = lane_id * kNumUnrolls, iter_idx = 0;
             i < hidden_bf16_int4_pad; i += 32 * kNumUnrolls, ++iter_idx) {
          // Read
          int4 int4_values[kNumUnrolls] = {0};
          auto uint32_values = reinterpret_cast<uint32_t*>(int4_values);

          // Load the next iteration
          // TODO: try `elect_one_sync`
          int const& stage_idx = iter_idx % kNumStages;
          int const& next_stage_idx = (iter_idx + 1) % kNumStages;
          tma_store_wait<kNumStages - kNumPrefetch - 1>();
          if (iter_idx + 1 < kNumIters and elect_one_sync(lane_id)) {
            auto const& offset_int4 = i + 32 * kNumUnrolls;
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
            EP_STATIC_ASSERT(kHidden % (kNumElemsPerInt4 * 32) == 0 and
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
      }
      // Flush all stores
      tma_store_wait();
      __syncwarp();

      // Issue RDMA only if we couldn't use IPC
      // NOTES: for zero-copy mode, we assume the data is already in the send
      // buffer
      if (dst_p2p_ptr == nullptr) {
        nvshmemi_ibgda_put_nbi_warp(
            dst_off, buf_ptr, hidden * sizeof(nv_bfloat16), dst_rank,
            sm_id,  // NOTE(MaoZiming): use sm_id for rb
            lane_id, token_idx - offset, ring_addrs, num_ring_addrs, true);
      }
    }

    // Put the finishing flag
    EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 16);
    asm volatile("bar.sync %0, %1;" ::"r"(warp_group_id + 1),
                 "r"(num_warps_per_group * 32));
    if (sub_warp_id == 1 and lane_id == 0) {
      while (ld_acquire_global(atomic_clean_flag) == 0)
        ;
      // Calculate offset from data buffer to flag buffer (similar to dispatch
      // phase) rdma_recv_flag corresponds to combine_rdma_recv_flag_buffer We
      // need to calculate the offset from rdma_recv_x (data buffer) to the flag
      // buffer
      // Try to use IPC for intra-node atomic operations
      auto const dst_p2p_ptr_flag =
          ipc_base_ptrs
              ? uccl::get_ipc_p2p_ptr(
                    reinterpret_cast<void*>(rdma_recv_flag + global_expert_idx),
                    ipc_base_ptrs, rank, dst_rank, max_nvl_peers, 0)
              : nullptr;
      uint64_t off_send = (uintptr_t)(rdma_recv_flag + global_expert_idx) -
                          (uintptr_t)(atomic_buffer_ptr);
      printf("Combine dst_p2p_ptr_flag=0x%" PRIxPTR ", off_send=%" PRId64
             ", src_rank=%d, dst_rank=%d, ranks_per_node=%d, "
             "global_expert_idx: %d\n",
             (uintptr_t)dst_p2p_ptr_flag, off_send, rank, dst_rank,
             max_nvl_peers, global_expert_idx);

      printf("[SEND] set flag idx=%d byte_off=%llu base=%p ptr=%p\n",
             global_expert_idx, (unsigned long long)off_send, rdma_recv_flag,
             rdma_recv_flag + global_expert_idx);
      if (dst_p2p_ptr_flag != nullptr) {
        // Intra-node: use direct atomic operation
        st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr_flag), 1);
      } else {
        // Inter-node or no IPC: use IBGDA atomic
        // NOTE(MaoZiming): Without ibgda, we can only use atomic add
        // Pass offset to CPU proxy for atomic operation (similar to dispatch
        // phase)
        nvshmemi_ibgda_amo_nonfetch_add(off_send, 1, dst_rank, sm_id,
                                        local_expert_idx, false, ring_addrs,
                                        num_ring_addrs);
      }
      atomic_add_release_global(atomic_clean_flag, -1);
    }
    __syncwarp();
  }

// Receiving phase
LOW_LATENCY_COMBINE_RECV:
  if ((phases & LOW_LATENCY_RECV_PHASE) == 0) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
      printf("[combine] SEND finished\n");
    return;
  }
  // Wait all ranks to arrive
  if (responsible_expert_idx < num_experts) {
    EP_DEVICE_ASSERT(num_warps_per_group > 1);
    if (sub_warp_id == 0 and lane_id == 0) {
      auto start_time = clock64();
      while (ld_acquire_sys_global(rdma_recv_flag + responsible_expert_idx) ==
             0)
        ;
      auto wait_recv_cost = clock64() - start_time;
      if (combine_wait_recv_cost_stats != nullptr) {
        auto const& src_rank = responsible_expert_idx / num_local_experts;
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      combine_wait_recv_cost_stats + src_rank),
                  wait_recv_cost);
      }
    }
  }
  cg::this_grid().sync();

  // Reduce tokens
  EP_DEVICE_ASSERT(num_topk <= 32);
  EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerInt4) == 0,
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

    if (blockIdx.x == 0 && threadIdx.x == 0)
      printf("[combine] RECV finished\n");
  }
}

void combine(void* combined_x, void* rdma_recv_x, int* rdma_recv_flag,
             void* rdma_send_x, void const* x, int64_t const* topk_idx,
             float const* topk_weights, int const* src_info,
             int64_t const* layout_range, int64_t* combine_wait_recv_cost_stats,
             int* next_clean, int num_next_clean_int, int num_combined_tokens,
             int hidden, int num_max_dispatch_tokens_per_rank, int num_topk,
             int num_experts, int rank, int num_ranks, bool use_logfmt,
             void* workspace, int num_device_sms, cudaStream_t stream,
             int phases, bool zero_copy, uint64_t const* ring_addrs,
             int num_ring_addrs, int max_nvl_peers, void** ipc_base_ptrs,
             void* rdma_buffer_ptr, void* atomic_buffer_ptr) {
  constexpr int kNumMaxTopk = 9;
  int const num_warp_groups = ceil_div(num_experts, num_device_sms);
  int const num_warps_per_group = 32 / num_warp_groups;
  EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0);

  auto const num_warps = num_warp_groups * num_warps_per_group;
  auto const num_sms = ceil_div(num_experts, num_warp_groups);

  // Check workspace
  auto atomic_clean_flag = static_cast<int*>(workspace);
  EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
  EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

  // Online cast cannot use zero-copy
  EP_HOST_ASSERT(not(zero_copy and use_logfmt));

  constexpr int kNumTMABytesPerWarp = 12 * (512 + 16);
  int const smem_size = kNumTMABytesPerWarp * num_warps;
  printf("Combine launched\n");

#define COMBINE_LAUNCH_CASE(hidden)                                            \
  {                                                                            \
    auto combine_func = use_logfmt ? combine<true, hidden, kNumMaxTopk>        \
                                   : combine<false, hidden, kNumMaxTopk>;      \
    SET_SHARED_MEMORY_FOR_TMA(combine_func);                                   \
    LAUNCH_KERNEL(&cfg, combine_func, combined_x, rdma_recv_x, rdma_recv_flag, \
                  rdma_send_x, x, topk_idx, topk_weights, src_info,            \
                  layout_range, combine_wait_recv_cost_stats, next_clean,      \
                  num_next_clean_int, atomic_clean_flag, num_combined_tokens,  \
                  hidden, num_topk, num_max_dispatch_tokens_per_rank,          \
                  num_experts, rank, num_ranks, num_warp_groups,               \
                  num_warps_per_group, phases, zero_copy, ring_addrs,          \
                  num_ring_addrs, max_nvl_peers, ipc_base_ptrs,                \
                  rdma_buffer_ptr, atomic_buffer_ptr);                         \
  }                                                                            \
  break

  SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
  SWITCH_HIDDEN(COMBINE_LAUNCH_CASE);
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("[combine] kernel launch error: %s\n", cudaGetErrorString(err));
    fflush(stdout);
  }
#undef COMBINE_LAUNCH_CASE
}

}  // namespace internode_ll
}  // namespace uccl