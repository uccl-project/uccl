// ============================================================================
// uccl.ep main class declaration.
//
// The ``Buffer`` class declaration lives here (interface only, no method
// bodies).  Implementations are in ``ep/src/uccl_ep.cc``.  Other
// translation units -- notably ``ep/src/uccl_ep_jax.cc`` which hosts the
// JAX FFI handlers -- include this header and call ``Buffer`` methods
// directly.
// ============================================================================

#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>
#include <nanobind/nanobind.h>

#include "d2h_queue_device.cuh"
#include "ep_config.hpp"
#include "ep_configs.cuh"
#include "ep_event.hpp"

namespace nb = nanobind;

class Buffer {
 public:
  Buffer(int rank, int num_ranks, long num_nvl_bytes, long num_rdma_bytes,
         bool low_latency_mode, bool explicitly_destroy, int num_local_ranks);

  std::optional<EventHandle> get_dispatch_layout(
      std::uintptr_t topk_idx_ptr, int num_tokens, int num_topk,
      int num_experts, std::uintptr_t num_tokens_per_rank_ptr,
      std::uintptr_t num_tokens_per_rdma_rank_ptr,
      std::uintptr_t num_tokens_per_expert_ptr,
      std::uintptr_t is_token_in_rank_ptr,
      std::optional<EventHandle>& previous_event, bool async,
      bool allocate_on_comm_stream, std::uintptr_t compute_stream_ptr);

  ~Buffer() noexcept(false);

  void destroy();

  std::tuple<int, std::vector<int>, std::optional<EventHandle>>
  intranode_prepare(std::uintptr_t num_tokens_per_rank_ptr,
                    std::uintptr_t is_token_in_rank_ptr,
                    std::uintptr_t num_tokens_per_expert_ptr, int num_tokens,
                    int num_experts, std::uintptr_t rank_prefix_matrix_ptr,
                    std::uintptr_t channel_prefix_matrix_ptr,
                    int expert_alignment, int num_worst_tokens,
                    uccl::Config const& config,
                    std::optional<EventHandle>& previous_event, bool async,
                    bool allocate_on_comm_stream,
                    std::uintptr_t compute_stream_ptr);

  std::optional<EventHandle> intranode_dispatch(
      std::uintptr_t x_ptr, int num_tokens, int hidden, int x_element_size,
      std::uintptr_t x_scales_ptr, int num_scales, int scale_token_stride,
      int scale_hidden_stride, std::uintptr_t topk_idx_ptr, int num_topk,
      std::uintptr_t topk_weights_ptr, std::uintptr_t is_token_in_rank_ptr,
      std::uintptr_t rank_prefix_matrix_ptr,
      std::uintptr_t channel_prefix_matrix_ptr, int num_experts,
      int num_worst_tokens, bool cached_mode, uccl::Config const& config,
      int num_recv_tokens, std::uintptr_t recv_x_ptr,
      std::uintptr_t recv_x_scales_ptr, std::uintptr_t recv_topk_idx_ptr,
      std::uintptr_t recv_topk_weights_ptr,
      std::uintptr_t recv_channel_prefix_matrix_ptr,
      std::uintptr_t recv_src_idx_ptr, std::uintptr_t send_head_ptr,
      std::optional<EventHandle>& previous_event, bool async,
      bool allocate_on_comm_stream, std::uintptr_t compute_stream_ptr);

  std::optional<EventHandle> intranode_combine(
      std::uintptr_t x_ptr, int num_tokens, int hidden, int x_dtype_code,
      int x_element_size, std::uintptr_t topk_weights_ptr, int num_topk,
      std::uintptr_t bias_0_ptr, std::uintptr_t bias_1_ptr,
      std::uintptr_t src_idx_ptr, int num_recv_tokens,
      std::uintptr_t rank_prefix_matrix_ptr,
      std::uintptr_t channel_prefix_matrix_ptr, std::uintptr_t send_head_ptr,
      uccl::Config const& config, std::uintptr_t recv_x_ptr,
      std::uintptr_t recv_topk_weights_ptr,
      std::optional<EventHandle>& previous_event, bool async,
      bool allocate_on_comm_stream, std::uintptr_t compute_stream_ptr);

  std::tuple<int, int, std::vector<int>, std::optional<EventHandle>>
  internode_prepare(std::uintptr_t num_tokens_per_rank_ptr,
                    std::uintptr_t num_tokens_per_rdma_rank_ptr,
                    std::uintptr_t num_tokens_per_expert_ptr,
                    std::uintptr_t is_token_in_rank_ptr, int num_tokens,
                    int hidden, int x_element_size, int num_scales,
                    int num_topk, int num_experts, int expert_alignment,
                    int num_worst_tokens, uccl::Config const& config,
                    std::uintptr_t rdma_channel_prefix_matrix_ptr,
                    std::uintptr_t recv_rdma_rank_prefix_sum_ptr,
                    std::uintptr_t gbl_channel_prefix_matrix_ptr,
                    std::uintptr_t recv_gbl_rank_prefix_sum_ptr,
                    std::optional<EventHandle>& previous_event, bool async,
                    bool allocate_on_comm_stream,
                    std::uintptr_t compute_stream_ptr);

  std::optional<EventHandle> internode_dispatch(
      std::uintptr_t x_ptr, int num_tokens, int hidden, int x_element_size,
      std::uintptr_t x_scales_ptr, int num_scales, int scale_token_stride,
      int scale_hidden_stride, std::uintptr_t topk_idx_ptr, int num_topk,
      std::uintptr_t topk_weights_ptr, std::uintptr_t is_token_in_rank_ptr,
      std::uintptr_t rdma_channel_prefix_matrix_ptr,
      std::uintptr_t recv_rdma_rank_prefix_sum_ptr,
      std::uintptr_t gbl_channel_prefix_matrix_ptr,
      std::uintptr_t recv_gbl_rank_prefix_sum_ptr, int num_experts,
      int num_worst_tokens, bool cached_mode, int num_rdma_recv_tokens,
      uccl::Config const& config, std::uintptr_t recv_x_ptr,
      std::uintptr_t recv_x_scales_ptr, std::uintptr_t recv_topk_idx_ptr,
      std::uintptr_t recv_topk_weights_ptr, std::uintptr_t recv_src_meta_ptr,
      std::uintptr_t recv_rdma_channel_prefix_matrix_ptr,
      std::uintptr_t recv_gbl_channel_prefix_matrix_ptr,
      std::uintptr_t send_rdma_head_ptr, std::uintptr_t send_nvl_head_ptr,
      std::optional<EventHandle>& previous_event, bool async,
      bool allocate_on_comm_stream, std::uintptr_t compute_stream_ptr);

  std::optional<EventHandle> internode_combine(
      std::uintptr_t x_ptr, int num_tokens, int hidden, int x_dtype_code,
      int x_element_size, std::uintptr_t topk_weights_ptr, int num_topk,
      std::uintptr_t bias_0_ptr, std::uintptr_t bias_1_ptr,
      std::uintptr_t src_meta_ptr, int num_combined_tokens,
      std::uintptr_t is_combined_token_in_rank_ptr,
      std::uintptr_t rdma_channel_prefix_matrix_ptr,
      std::uintptr_t rdma_rank_prefix_sum_ptr,
      std::uintptr_t gbl_channel_prefix_matrix_ptr,
      std::uintptr_t combined_rdma_head_ptr,
      std::uintptr_t combined_nvl_head_ptr, uccl::Config const& config,
      std::uintptr_t combined_x_ptr, std::uintptr_t combined_topk_weights_ptr,
      std::optional<EventHandle>& previous_event, bool async,
      bool allocate_on_comm_stream, std::uintptr_t compute_stream_ptr);

  void clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank,
                                int hidden, int num_experts,
                                std::uintptr_t stream_ptr);

  std::tuple<std::optional<EventHandle>, std::optional<std::function<void()>>>
  low_latency_dispatch(std::uintptr_t x_ptr, int x_rows, int x_cols,
                       std::uintptr_t topk_idx_ptr, int topk_rows,
                       int topk_cols, std::uintptr_t packed_recv_x_ptr,
                       std::uintptr_t packed_recv_x_scales_ptr,
                       std::uintptr_t packed_recv_count_ptr,
                       std::uintptr_t packed_recv_src_info_ptr,
                       std::uintptr_t packed_recv_layout_range_ptr,
                       std::uintptr_t cumulative_local_expert_recv_stats_ptr,
                       std::uintptr_t dispatch_wait_recv_cost_stats_ptr,
                       std::uintptr_t compute_stream_ptr,
                       int num_max_dispatch_tokens_per_rank, int num_experts,
                       bool use_fp8, bool round_scale, bool use_ue8m0,
                       bool async, bool return_recv_hook);

  std::tuple<std::optional<EventHandle>, std::optional<std::function<void()>>>
  low_latency_combine(std::uintptr_t x_ptr, int x_dim0, int x_dim1, int x_dim2,
                      std::uintptr_t topk_idx_ptr, int topk_rows, int topk_cols,
                      std::uintptr_t topk_weights_ptr,
                      std::uintptr_t src_info_ptr, int src_info_dim0,
                      int src_info_dim1, std::uintptr_t layout_range_ptr,
                      int layout_range_dim0, int layout_range_dim1,
                      std::uintptr_t combine_wait_recv_cost_stats_ptr,
                      std::uintptr_t compute_stream_ptr,
                      int num_max_dispatch_tokens_per_rank, int num_experts,
                      bool use_logfmt, bool zero_copy, bool async,
                      bool return_recv_hook, std::uintptr_t out_ptr);

  int get_local_device_id();

  nb::bytes get_local_ipc_handle() const;

  nb::bytes get_local_rdma_ipc_handle();

  nb::bytes get_local_atomics_ipc_handle();

  int get_num_rdma_ranks() const;
  int get_num_max_nvl_peers() const;
  int get_source_meta_bytes() const;
  int get_rdma_rank() const;
  int get_root_rdma_rank(bool global) const;

  nb::bytes get_local_uccl_shmem_unique_id() const;

  void reset_rdma_buffer();

  void sync(std::vector<int> const& device_ids,
            std::vector<std::optional<nb::bytes>> const& all_gathered_handles,
            std::optional<nb::bytes> const& root_unique_id_opt,
            std::optional<std::vector<std::optional<nb::bytes>>> const&
                all_gathered_rdma_handles_opt = std::nullopt);

  void set_rdma_buffer(std::uintptr_t addr, bool is_host_ptr = false);

  void set_atomic_buffer_ptr(void* ptr);

  std::uintptr_t get_local_buffer_ptr(int64_t offset,
                                      bool use_rdma_buffer) const;

  int64_t get_local_buffer_nbytes(bool use_rdma_buffer) const;

  std::uintptr_t get_comm_stream() const;

  bool is_available() const;
  bool is_internode_available() const;

 private:
  int rank{0};
  int num_ranks{1};
  long num_nvl_bytes{0};
  long num_rdma_bytes{0};
  bool low_latency_mode{false};
  bool explicitly_destroy{false};
  int device_index{0};
  std::vector<nb::object> proxies_;
  bool available{false};
  void* rdma_buffer_ptr = nullptr;
  void* atomic_buffer_ptr = nullptr;
  int low_latency_buffer_idx = 0;
  void* workspace = nullptr;

  // device / ranks
  int rdma_rank{0}, nvl_rank{0};
  int num_rdma_ranks{1}, num_nvl_ranks{1};
  int num_device_sms{0};
  int max_nvl_peers{0};

  // stream & workspace
  cudaStream_t comm_stream{nullptr};

  cudaIpcMemHandle_t ipc_handles[NUM_MAX_NVL_PEERS]{};
  void* buffer_ptrs[NUM_MAX_NVL_PEERS]{};
  int* barrier_signal_ptrs[NUM_MAX_NVL_PEERS]{};
  void** buffer_ptrs_gpu{nullptr};
  int** barrier_signal_ptrs_gpu{nullptr};
  cudaIpcMemHandle_t rdma_ipc_handles[NUM_MAX_NVL_PEERS]{};
  void* ipc_rdma_base_ptrs[NUM_MAX_NVL_PEERS]{};

  // clang-format would change to int volatile*
  // clang-format off
  // MoE counters (host mapped)
  volatile int* moe_recv_counter = nullptr;
  int* moe_recv_counter_mapped{nullptr};  // device pointer
  volatile int* moe_recv_expert_counter{nullptr};
  int* moe_recv_expert_counter_mapped{nullptr};
  volatile int* moe_recv_rdma_counter{nullptr};
  int* moe_recv_rdma_counter_mapped{nullptr};
  // clang-format on

  bool destroyed = false;

  // Ring buffers
  int num_d2h_channel_addrs{0};
  d2hq::D2HHandle* d_handle_objs{nullptr};
  uint64_t* d_handles{nullptr};

  // IPC base pointers for GPU access (for replacing nvshmemi_get_p2p_ptr)
  void** d_ipc_rdma_base_ptrs{
      nullptr};  // Device pointer to array of IPC base addresses
};

