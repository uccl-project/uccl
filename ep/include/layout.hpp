#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace uccl {
namespace layout {

void get_dispatch_layout(int64_t const* topk_idx, int* num_tokens_per_rank,
                         int* num_tokens_per_rdma_rank,
                         int* num_tokens_per_expert, bool* is_token_in_rank,
                         int num_tokens, int num_topk, int num_ranks,
                         int num_experts, cudaStream_t stream);

}  // namespace layout
}  // namespace uccl