#include "pack_unpack_kernels.cuh"
#include <torch/extension.h>
#include <cuda_runtime.h>

namespace moe_pack_unpack {

#define CUDA_CHECK(call)                                     \
  do {                                                       \
    cudaError_t err = call;                                  \
    if (err != cudaSuccess) {                                \
      throw std::runtime_error(std::string("CUDA error: ") + \
                               cudaGetErrorString(err));     \
    }                                                        \
  } while (0)

// Kernel to pack MoE data
// Each thread processes one (token, expert) pair
template <typename scalar_t, typename idx_t, typename weight_t>
__global__ void pack_moe_kernel(
    scalar_t const* __restrict__ x,      // (num_tokens, hidden_dim)
    idx_t const* __restrict__ topk_idx,  // (num_tokens, experts_per_token)
    weight_t const* __restrict__ topk_weights,  // (num_tokens,
                                                // experts_per_token)
    uint8_t** __restrict__ buffers,             // world_size buffer pointers
    int* __restrict__ per_rank_item_counts,     // (world_size,) atomic counters
    int num_tokens, int hidden_dim, int experts_per_token,
    int num_local_experts, int world_size) {
  // Global thread ID represents (token_id, topk_pos)
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_items = num_tokens * experts_per_token;

  if (tid >= total_items) return;

  int token_id = tid / experts_per_token;
  int topk_pos = tid % experts_per_token;

  // Get expert ID
  idx_t expert_id = topk_idx[token_id * experts_per_token + topk_pos];

  // Skip padding (-1)
  if (expert_id == -1) return;

  // Determine target rank
  int target_rank = expert_id / num_local_experts;
  int local_expert_id = expert_id % num_local_experts;

  // Calculate sizes
  int bytes_per_token = hidden_dim * sizeof(scalar_t);
  int bytes_per_idx = sizeof(idx_t);
  int bytes_per_weight = sizeof(weight_t);
  int bytes_per_item = bytes_per_token + bytes_per_idx + bytes_per_weight;

  // Atomically get offset in target buffer
  int item_offset = atomicAdd(&per_rank_item_counts[target_rank], 1);
  int byte_offset = item_offset * bytes_per_item;

  uint8_t* target_buf = buffers[target_rank];

  // Pack token data using byte-wise copy to avoid alignment issues
  scalar_t const* token_data = x + token_id * hidden_dim;
  uint8_t* out_ptr = target_buf + byte_offset;
  uint8_t const* in_ptr = reinterpret_cast<uint8_t const*>(token_data);
  for (int i = 0; i < bytes_per_token; i++) {
    out_ptr[i] = in_ptr[i];
  }
  byte_offset += bytes_per_token;

  // Pack local expert ID using byte-wise copy
  idx_t local_expert_id_typed = static_cast<idx_t>(local_expert_id);
  out_ptr = target_buf + byte_offset;
  in_ptr = reinterpret_cast<uint8_t const*>(&local_expert_id_typed);
  for (int i = 0; i < bytes_per_idx; i++) {
    out_ptr[i] = in_ptr[i];
  }
  byte_offset += bytes_per_idx;

  // Pack weight using byte-wise copy
  weight_t weight_val = topk_weights[token_id * experts_per_token + topk_pos];
  out_ptr = target_buf + byte_offset;
  in_ptr = reinterpret_cast<uint8_t const*>(&weight_val);
  for (int i = 0; i < bytes_per_weight; i++) {
    out_ptr[i] = in_ptr[i];
  }
}

// Kernel to unpack MoE data
// Each thread processes one item
template <typename scalar_t, typename idx_t, typename weight_t>
__global__ void unpack_moe_kernel(
    uint8_t const* const* __restrict__ buffers,   // world_size buffer pointers
    int const* __restrict__ per_rank_recv_bytes,  // (world_size,)
    scalar_t* __restrict__ recv_x,                // (total_items, hidden_dim)
    idx_t* __restrict__ recv_topk_idx,            // (total_items,)
    weight_t* __restrict__ recv_topk_weights,     // (total_items,)
    int* __restrict__ recv_item_offsets,  // (world_size,) cumulative offsets
    int hidden_dim, int world_size) {
  // Each block processes one sender rank
  int sender_rank = blockIdx.x;
  if (sender_rank >= world_size) return;

  int recv_bytes = per_rank_recv_bytes[sender_rank];
  if (recv_bytes == 0) return;

  int bytes_per_token = hidden_dim * sizeof(scalar_t);
  int bytes_per_idx = sizeof(idx_t);
  int bytes_per_weight = sizeof(weight_t);
  int bytes_per_item = bytes_per_token + bytes_per_idx + bytes_per_weight;

  int num_items = recv_bytes / bytes_per_item;
  int base_item_id = recv_item_offsets[sender_rank];

  uint8_t const* buf = buffers[sender_rank];

  // Each thread unpacks one item
  for (int item_idx = threadIdx.x; item_idx < num_items;
       item_idx += blockDim.x) {
    int byte_offset = item_idx * bytes_per_item;
    int global_item_id = base_item_id + item_idx;

    // Unpack token data using byte-wise copy to avoid alignment issues
    scalar_t* token_out = recv_x + global_item_id * hidden_dim;
    uint8_t const* in_ptr = buf + byte_offset;
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(token_out);
    for (int i = 0; i < bytes_per_token; i++) {
      out_ptr[i] = in_ptr[i];
    }
    byte_offset += bytes_per_token;

    // Unpack expert ID using byte-wise copy
    idx_t idx_val;
    in_ptr = buf + byte_offset;
    out_ptr = reinterpret_cast<uint8_t*>(&idx_val);
    for (int i = 0; i < bytes_per_idx; i++) {
      out_ptr[i] = in_ptr[i];
    }
    recv_topk_idx[global_item_id] = idx_val;
    byte_offset += bytes_per_idx;

    // Unpack weight using byte-wise copy
    weight_t weight_val;
    in_ptr = buf + byte_offset;
    out_ptr = reinterpret_cast<uint8_t*>(&weight_val);
    for (int i = 0; i < bytes_per_weight; i++) {
      out_ptr[i] = in_ptr[i];
    }
    recv_topk_weights[global_item_id] = weight_val;
  }
}

// Kernel to count tokens per expert
template <typename idx_t>
__global__ void count_expert_tokens_kernel(
    idx_t const* __restrict__ recv_topk_idx,  // (total_items,)
    int* __restrict__ expert_counts,          // (num_local_experts,)
    int total_items) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total_items) return;

  idx_t expert_id = recv_topk_idx[tid];
  if (expert_id >= 0) {
    atomicAdd(&expert_counts[expert_id], 1);
  }
}

void pack_moe_data_cuda(torch::Tensor const& x, torch::Tensor const& topk_idx,
                        torch::Tensor const& topk_weights,
                        std::vector<torch::Tensor> const& buffers,
                        int num_experts, int num_local_experts, int world_size,
                        torch::Tensor& per_rank_bytes) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(topk_idx.is_cuda(), "topk_idx must be a CUDA tensor");
  TORCH_CHECK(topk_weights.is_cuda(), "topk_weights must be a CUDA tensor");

  int num_tokens = x.size(0);
  int hidden_dim = x.size(1);
  int experts_per_token = topk_idx.size(1);

  // Allocate per-rank item counters (atomic)
  auto per_rank_item_counts = torch::zeros(
      {world_size},
      torch::TensorOptions().dtype(torch::kInt32).device(x.device()));

  // Create buffer pointer array on GPU
  std::vector<uint8_t*> h_buffer_ptrs(world_size);
  for (int i = 0; i < world_size; i++) {
    h_buffer_ptrs[i] = buffers[i].data_ptr<uint8_t>();
  }

  uint8_t** d_buffer_ptrs;
  CUDA_CHECK(cudaMalloc(&d_buffer_ptrs, world_size * sizeof(uint8_t*)));
  CUDA_CHECK(cudaMemcpy(d_buffer_ptrs, h_buffer_ptrs.data(),
                        world_size * sizeof(uint8_t*), cudaMemcpyHostToDevice));

  // Launch kernel
  int total_items = num_tokens * experts_per_token;
  int threads = 256;
  int blocks = (total_items + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
      "pack_moe_kernel", [&] {
        using scalar_t = scalar_t;
        AT_DISPATCH_INDEX_TYPES(
            topk_idx.scalar_type(), "pack_moe_kernel_idx", [&] {
              using idx_t = index_t;
              using weight_t = float;  // topk_weights is always float32

              pack_moe_kernel<scalar_t, idx_t, weight_t><<<blocks, threads>>>(
                  x.data_ptr<scalar_t>(), topk_idx.data_ptr<idx_t>(),
                  topk_weights.data_ptr<weight_t>(), d_buffer_ptrs,
                  per_rank_item_counts.data_ptr<int>(), num_tokens, hidden_dim,
                  experts_per_token, num_local_experts, world_size);
            });
      });

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Calculate bytes from item counts
  int bytes_per_token = hidden_dim * x.element_size();
  int bytes_per_idx = topk_idx.element_size();
  int bytes_per_weight = topk_weights.element_size();
  int bytes_per_item = bytes_per_token + bytes_per_idx + bytes_per_weight;

  per_rank_bytes = per_rank_item_counts * bytes_per_item;

  CUDA_CHECK(cudaFree(d_buffer_ptrs));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
unpack_moe_data_cuda(std::vector<torch::Tensor> const& buffers,
                     torch::Tensor const& per_rank_recv_bytes,
                     int num_local_experts, int hidden_dim, int world_size,
                     torch::ScalarType x_dtype, torch::ScalarType idx_dtype,
                     torch::ScalarType weight_dtype) {
  TORCH_CHECK(per_rank_recv_bytes.is_cuda(),
              "per_rank_recv_bytes must be a CUDA tensor");

  auto device = per_rank_recv_bytes.device();

  // Calculate total items and create offset array
  auto per_rank_bytes_cpu = per_rank_recv_bytes.to(torch::kCPU);
  auto per_rank_bytes_acc = per_rank_bytes_cpu.accessor<int, 1>();

  int bytes_per_token = hidden_dim * torch::elementSize(x_dtype);
  int bytes_per_idx = torch::elementSize(idx_dtype);
  int bytes_per_weight = torch::elementSize(weight_dtype);
  int bytes_per_item = bytes_per_token + bytes_per_idx + bytes_per_weight;

  std::vector<int> h_item_offsets(world_size);
  int total_items = 0;
  for (int i = 0; i < world_size; i++) {
    h_item_offsets[i] = total_items;
    int num_items = per_rank_bytes_acc[i] / bytes_per_item;
    total_items += num_items;
  }

  if (total_items == 0) {
    total_items = 1;  // Avoid empty tensors
  }

  // Allocate output tensors
  auto recv_x =
      torch::empty({total_items, hidden_dim},
                   torch::TensorOptions().dtype(x_dtype).device(device));
  auto recv_topk_idx = torch::empty(
      {total_items}, torch::TensorOptions().dtype(idx_dtype).device(device));
  auto recv_topk_weights = torch::empty(
      {total_items}, torch::TensorOptions().dtype(weight_dtype).device(device));

  // Copy offsets to GPU
  auto d_item_offsets =
      torch::from_blob(h_item_offsets.data(), {world_size}, torch::kInt32)
          .to(device);

  // Create buffer pointer array on GPU
  std::vector<uint8_t const*> h_buffer_ptrs(world_size);
  for (int i = 0; i < world_size; i++) {
    h_buffer_ptrs[i] = buffers[i].data_ptr<uint8_t>();
  }

  uint8_t const** d_buffer_ptrs;
  CUDA_CHECK(cudaMalloc(&d_buffer_ptrs, world_size * sizeof(uint8_t const*)));
  CUDA_CHECK(cudaMemcpy(d_buffer_ptrs, h_buffer_ptrs.data(),
                        world_size * sizeof(uint8_t const*),
                        cudaMemcpyHostToDevice));

  // Launch unpack kernel (one block per sender rank)
  int threads = 256;
  int blocks = world_size;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x_dtype,
      "unpack_moe_kernel", [&] {
        using scalar_t = scalar_t;
        AT_DISPATCH_INDEX_TYPES(idx_dtype, "unpack_moe_kernel_idx", [&] {
          using idx_t = index_t;
          using weight_t = float;  // weights are always float32

          unpack_moe_kernel<scalar_t, idx_t, weight_t><<<blocks, threads>>>(
              d_buffer_ptrs, per_rank_recv_bytes.data_ptr<int>(),
              recv_x.data_ptr<scalar_t>(), recv_topk_idx.data_ptr<idx_t>(),
              recv_topk_weights.data_ptr<weight_t>(),
              d_item_offsets.data_ptr<int>(), hidden_dim, world_size);
        });
      });

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Count tokens per expert
  auto expert_counts =
      torch::zeros({num_local_experts},
                   torch::TensorOptions().dtype(torch::kInt32).device(device));

  if (total_items > 1) {  // Skip if empty
    threads = 256;
    blocks = (total_items + threads - 1) / threads;

    AT_DISPATCH_INDEX_TYPES(idx_dtype, "count_expert_tokens_kernel", [&] {
      using idx_t = index_t;
      count_expert_tokens_kernel<idx_t>
          <<<blocks, threads>>>(recv_topk_idx.data_ptr<idx_t>(),
                                expert_counts.data_ptr<int>(), total_items);
    });

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  CUDA_CHECK(cudaFree(d_buffer_ptrs));

  return std::make_tuple(recv_x, recv_topk_idx, recv_topk_weights,
                         expert_counts);
}

}  // namespace moe_pack_unpack
