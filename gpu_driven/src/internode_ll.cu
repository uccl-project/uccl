#include "internode_ll.cuh"
// #include <ATen/cuda/CUDAContext.h>
#include <iostream>

namespace uccl {
namespace internode_ll {

__global__ void dummy_dispatch_kernel() {
  // Just a no-op kernel to make sure launch compiles
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

  // Launch a dummy kernel just to ensure setup works
  dummy_dispatch_kernel<<<1, 1, 0, stream>>>();
}

}  // namespace internode_ll
}  // namespace uccl