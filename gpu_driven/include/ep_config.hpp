#pragma once
#include "ep_util.hpp"
#include <cstddef>
#include <cstdint>
#include <utility>

#define LOW_LATENCY_SEND_PHASE 1
#define LOW_LATENCY_RECV_PHASE 2

template <typename dtype_t>
dtype_t ceil_div(dtype_t a, dtype_t b) {
  return (a + b - 1) / b;
}

template <typename dtype_t>
dtype_t align(dtype_t a, dtype_t b) {
  return ceil_div<dtype_t>(a, b) * b;
}

// thirdparty/DeepEP/csrc/config.hpp
struct LowLatencyBuffer {
  int num_clean_int = 0;

  void* dispatch_rdma_send_buffer = nullptr;
  void* dispatch_rdma_recv_data_buffer = nullptr;
  int* dispatch_rdma_recv_count_buffer = nullptr;

  void* combine_rdma_send_buffer = nullptr;
  void* combine_rdma_recv_data_buffer = nullptr;
  int* combine_rdma_recv_flag_buffer = nullptr;

  void* combine_rdma_send_buffer_data_start = nullptr;
  size_t num_bytes_per_combine_msg = 0;

  std::pair<int*, int> clean_meta() {
    EP_HOST_ASSERT(dispatch_rdma_recv_count_buffer ==
                   combine_rdma_recv_flag_buffer);
    return {dispatch_rdma_recv_count_buffer, num_clean_int};
  }
};

struct LowLatencyLayout {
  size_t total_bytes = 0;
  LowLatencyBuffer buffers[2];

  template <typename out_ptr_t = void*, typename count_ptr_t = uint8_t*,
            typename in_ptr_t = void*>
  out_ptr_t advance(in_ptr_t const& ptr, size_t count) {
    return reinterpret_cast<out_ptr_t>(reinterpret_cast<count_ptr_t>(ptr) +
                                       count);
  }

  LowLatencyLayout(void* rdma_buffer, int num_max_dispatch_tokens_per_rank,
                   int hidden, int num_ranks, int num_experts) {
    int const num_scales = hidden / 128;

    // Dispatch and combine layout:
    //  - 2 symmetric odd/even send buffer
    //  - 2 symmetric odd/even receive buffers
    //  - 2 symmetric odd/even signaling buffers

    // Message sizes
    // NOTES: you should add a control `int4` for combine messages if you want
    // to do data transformation
    EP_HOST_ASSERT(num_scales * sizeof(float) <= static_cast<size_t>(hidden));
    size_t num_bytes_per_dispatch_msg =
        sizeof(int4) + std::max(hidden * sizeof(nv_bfloat16),
                                hidden + num_scales * sizeof(float));
    size_t num_bytes_per_combine_msg = hidden * sizeof(nv_bfloat16);

    // Send buffer
    size_t dispatch_send_buffer_bytes =
        num_max_dispatch_tokens_per_rank * num_bytes_per_dispatch_msg;
    size_t combine_send_buffer_bytes = num_experts *
                                       num_max_dispatch_tokens_per_rank *
                                       num_bytes_per_combine_msg;
    size_t send_buffer_bytes =
        std::max(dispatch_send_buffer_bytes, combine_send_buffer_bytes);
    EP_HOST_ASSERT(send_buffer_bytes % sizeof(int4) == 0);
    total_bytes += send_buffer_bytes * 2;

    // Symmetric receive buffers
    // TODO: optimize memory usages
    size_t dispatch_recv_data_buffer_bytes = num_experts *
                                             num_max_dispatch_tokens_per_rank *
                                             num_bytes_per_dispatch_msg;
    size_t combine_recv_buffer_bytes = num_experts *
                                       num_max_dispatch_tokens_per_rank *
                                       num_bytes_per_combine_msg;
    size_t recv_buffer_bytes =
        std::max(dispatch_recv_data_buffer_bytes, combine_recv_buffer_bytes);
    EP_HOST_ASSERT(recv_buffer_bytes % sizeof(int4) == 0);
    total_bytes += recv_buffer_bytes * 2;

    // Symmetric signaling buffers
    size_t dispatch_recv_count_buffer_bytes = num_experts * sizeof(int);
    size_t combine_recv_flag_buffer_bytes = dispatch_recv_count_buffer_bytes;
    size_t signaling_buffer_bytes = std::max(dispatch_recv_count_buffer_bytes,
                                             combine_recv_flag_buffer_bytes);
    size_t signaling_buffer_bytes_aligned =
        align<size_t>(signaling_buffer_bytes, 128);
    total_bytes += signaling_buffer_bytes_aligned * 2;

    // Assign pointers
    // NOTES: we still leave some space for distinguishing dispatch/combine
    // buffer, so you may see some parameters are duplicated
    for (int i = 0; i < 2; ++i) {
      buffers[i] = {
          static_cast<int>(signaling_buffer_bytes / sizeof(int)),
          advance(rdma_buffer,
                  signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * i),
          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 +
                                   send_buffer_bytes * 2 +
                                   recv_buffer_bytes * i),
          advance<int*>(rdma_buffer, signaling_buffer_bytes_aligned * i),
          advance(rdma_buffer,
                  signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * i),
          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 +
                                   send_buffer_bytes * 2 +
                                   recv_buffer_bytes * i),
          advance<int*>(rdma_buffer, signaling_buffer_bytes_aligned * i),
          advance(rdma_buffer,
                  signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * i),
          num_bytes_per_combine_msg};
    }
  }
};