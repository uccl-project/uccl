#pragma once
#include <cstddef>
#include <cstdint>

#define LOW_LATENCY_SEND_PHASE 1
#define LOW_LATENCY_RECV_PHASE 2

// thirdparty/DeepEP/csrc/config.hpp
// TODO(MaoZiming): dummy
struct LowLatencyBuffer {
  void* dispatch_rdma_recv_data_buffer = nullptr;
  void* dispatch_rdma_send_buffer = nullptr;
  int* dispatch_rdma_recv_count_buffer = nullptr;
  void* combine_rdma_recv_data_buffer = nullptr;
  void* combine_rdma_send_buffer = nullptr;
  int* combine_rdma_recv_flag_buffer = nullptr;
  std::size_t num_bytes_per_combine_msg = 0;
  int num_clean_int = 0;

  std::pair<int*, int> clean_meta() {
    EP_HOST_ASSERT(dispatch_rdma_recv_count_buffer ==
                   combine_rdma_recv_flag_buffer);
    return {dispatch_rdma_recv_count_buffer, num_clean_int};
  }
};

struct LowLatencyLayout {
  std::size_t total_bytes = 0;
  LowLatencyBuffer buffers[2];

  LowLatencyLayout(void* /*rdma_buffer*/,
                   int /*num_max_dispatch_tokens_per_rank*/, int /*hidden*/,
                   int /*num_ranks*/, int /*num_experts*/) {
    // TODO(MaoZiming): implement
  }
};