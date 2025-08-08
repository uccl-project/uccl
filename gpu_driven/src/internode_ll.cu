#include "ep_configs.cuh"
#include "ep_launch.cuh"
#include "ep_util.hpp"
#include "ep_utils.cuh"
#include "internode_ll.cuh"
#include "uccl_ibgda.cuh"
#include <iostream>
#include <vector>

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
    bool round_scale, int phases) {
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
  const size_t hidden_bytes =
      kHidden * (kUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));
  const size_t hidden_int4 = hidden_bytes / sizeof(int4);

  // Message package: hidden data, FP8 scales, index at source
  // NOTES: currently we have 3 reserved int fields for future use
  using vec_t = typename std::conditional<kUseFP8, int2, int4>::type;
  const size_t num_bytes_per_msg =
      sizeof(int4) + (kUseFP8 ? (kHidden + num_scales * sizeof(float))
                              : (kHidden * sizeof(nv_bfloat16)));
  const size_t num_int4_per_msg = num_bytes_per_msg / sizeof(int4);
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
    const size_t hidden_bf16_int4 = kHidden / kNumElemsPerRead;

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
        auto const dst_ptr =
            reinterpret_cast<uint64_t>(rdma_recv_x) +
            dst_expert_local_idx * num_ranks *
                num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
            rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
            slot_idx * num_bytes_per_msg;
        auto const dst_p2p_ptr =
            uccl::nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
        if (dst_p2p_ptr == 0) {
          uccl::nvshmemi_ibgda_put_nbi_warp(dst_ptr, src_ptr, num_bytes_per_msg,
                                            dst_rank, dst_expert_local_idx,
                                            lane_id, slot_idx);
        } else {
          // NOTES: only 2 load iterations for 7K hidden with 8 unrolls
          auto const* src_int4_ptr = reinterpret_cast<int4 const*>(src_ptr);
          auto const* dst_int4_ptr = reinterpret_cast<int4*>(dst_p2p_ptr);
          UNROLLED_WARP_COPY(8, lane_id, num_int4_per_msg, dst_int4_ptr,
                             src_int4_ptr, ld_nc_global, st_na_global);
        }

        // Increase counter after finishing
        __syncwarp();
        lane_id == 0 ? atomic_add_release_global(
                           atomic_finish_counter_per_expert + dst_expert_idx, 1)
                     : 0;
      }
    }
  } else if (warp_id == num_warps - 1) {
    EP_DEVICE_ASSERT(num_sms > 1);
    if (sm_id == 0) {
      // The first SM is also responsible for checking QPs
      EP_DEVICE_ASSERT(uccl::ibgda_get_state()->num_rc_per_pe >=
                       num_local_experts);

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
    auto dst_ptr = reinterpret_cast<uint64_t>(
        rdma_recv_count + dst_expert_local_idx * num_ranks + rank);
    auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
    if (dst_p2p_ptr == 0) {
      uccl::nvshmemi_ibgda_amo_nonfetch_add(reinterpret_cast<int*>(dst_ptr),
                                            -num_tokens_sent - 1, dst_rank,
                                            dst_expert_local_idx);
    } else {
      st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr),
                            -num_tokens_sent - 1);
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
  if ((phases & LOW_LATENCY_RECV_PHASE) == 0) return;

  // For send-and-recv kernels, we need a grid sync for making
  // `packed_recv_count` visible
  // TODO(MaoZiming): Fix this.
  // if (phases & LOW_LATENCY_SEND_PHASE)
  //     cg::this_grid().sync();

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
      while ((num_recv_tokens = ld_acquire_sys_global(
                  rdma_recv_count + local_expert_idx * num_ranks + src_rank)) ==
             0)
        ;
      auto wait_recv_cost = clock64() - start_time;
      num_recv_tokens = -num_recv_tokens - 1;
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
              int num_device_sms, cudaStream_t stream, int phases) {
  // TODO(MaoZiming): Fix.
  std::cout << "[uccl::internode_ll::dispatch] dummy launch invoked"
            << std::endl;

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

  // TODO(MaoZiming): Fix the launch configuration
}

void combine(void* combined_x, void* rdma_recv_x, int* rdma_recv_flag,
             void* rdma_send_x, void const* x, int64_t const* topk_idx,
             float const* topk_weights, int const* src_info,
             int64_t const* layout_range, int64_t* combine_wait_recv_cost_stats,
             int* next_clean, int num_next_clean_int, int num_combined_tokens,
             int hidden, int num_max_dispatch_tokens_per_rank, int num_topk,
             int num_experts, int rank, int num_ranks, bool use_logfmt,
             void* workspace, int num_device_sms, cudaStream_t stream,
             int phases, bool zero_copy) {
  // TODO(MaoZiming): Fix.
  std::cout << "[uccl::internode_ll::combine] dummy launch invoked"
            << std::endl;
}

// TODO(MaoZiming): This corresponds to DeepEP/csrc/kernels/runtime.cu
// They use nvshmem, but we don't have that in our environment.
int init(std::vector<uint8_t> const& root_unique_id_val, int rank,
         int num_ranks, bool low_latency_mode) {
  // TODO(MaoZiming): Fix.
  std::cout << "[uccl::internode_ll::init] dummy init invoked" << std::endl;
  return 0;  // Return success
}

void* alloc(size_t size, size_t alignment) {
  // TODO(MaoZiming): Fix.
  std::cout << "[uccl::internode_ll::alloc] dummy alloc invoked" << std::endl;
  return nullptr;
}

void barrier() {
  // TODO(MaoZiming): Fix.
  std::cout << "[uccl::internode_ll::barrier] dummy barrier invoked"
            << std::endl;
  return;
}

std::vector<uint8_t> get_unique_id() {
  // TODO(MaoZiming): Fix.
  return std::vector<uint8_t>(64, 0);  // Dummy unique ID
}

}  // namespace internode_ll
}  // namespace uccl