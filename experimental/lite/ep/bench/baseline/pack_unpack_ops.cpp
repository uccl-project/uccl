#include "pack_unpack_kernels.cuh"
#include <torch/extension.h>

namespace moe_pack_unpack {

// Python-facing wrapper for pack
torch::Tensor pack_moe_data(torch::Tensor const& x,
                            torch::Tensor const& topk_idx,
                            torch::Tensor const& topk_weights,
                            std::vector<torch::Tensor> const& buffers,
                            int num_experts, int world_size) {
  int num_local_experts = num_experts / world_size;
  auto per_rank_bytes = torch::zeros(
      {world_size},
      torch::TensorOptions().dtype(torch::kInt32).device(x.device()));

  pack_moe_data_cuda(x, topk_idx, topk_weights, buffers, num_experts,
                     num_local_experts, world_size, per_rank_bytes);

  return per_rank_bytes;
}

// Python-facing wrapper for unpack
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::vector<int64_t>>
unpack_moe_data(std::vector<torch::Tensor> const& buffers,
                torch::Tensor const& per_rank_recv_bytes, int num_local_experts,
                int hidden_dim, int world_size, torch::ScalarType x_dtype,
                torch::ScalarType idx_dtype, torch::ScalarType weight_dtype) {
  auto [recv_x, recv_topk_idx, recv_topk_weights, expert_counts] =
      unpack_moe_data_cuda(buffers, per_rank_recv_bytes, num_local_experts,
                           hidden_dim, world_size, x_dtype, idx_dtype,
                           weight_dtype);

  // Convert expert_counts tensor to vector
  auto expert_counts_cpu = expert_counts.to(torch::kCPU);
  auto counts_acc = expert_counts_cpu.accessor<int, 1>();
  std::vector<int64_t> counts_vec;
  for (int i = 0; i < num_local_experts; i++) {
    counts_vec.push_back(counts_acc[i]);
  }

  return std::make_tuple(recv_x, recv_topk_idx, recv_topk_weights, counts_vec);
}

}  // namespace moe_pack_unpack

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_moe_data", &moe_pack_unpack::pack_moe_data,
        "Pack MoE data into per-rank buffers (CUDA)", py::arg("x"),
        py::arg("topk_idx"), py::arg("topk_weights"), py::arg("buffers"),
        py::arg("num_experts"), py::arg("world_size"));

  m.def("unpack_moe_data", &moe_pack_unpack::unpack_moe_data,
        "Unpack MoE data from per-rank buffers (CUDA)", py::arg("buffers"),
        py::arg("per_rank_recv_bytes"), py::arg("num_local_experts"),
        py::arg("hidden_dim"), py::arg("world_size"), py::arg("x_dtype"),
        py::arg("idx_dtype"), py::arg("weight_dtype"));
}
