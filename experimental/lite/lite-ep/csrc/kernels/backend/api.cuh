#pragma once

#include <memory>
#include <optional>
#include <vector>
#include <nccl.h>
#include <nccl_device.h>
#include <pybind11/pytypes.h>

namespace deep_ep::nccl {

pybind11::bytearray get_local_unique_id();

int64_t create_nccl_comm(pybind11::bytearray const& root_unique_id_bytes,
                         int const& num_ranks, int const& rank_idx);

void destroy_nccl_comm(int64_t const& nccl_comm);

std::tuple<int, int> get_physical_domain_size(int64_t const& nccl_comm);

std::tuple<int, int> get_logical_domain_size(int64_t const& nccl_comm,
                                             bool const& allow_hybrid_mode);

// TODO: make it header only?
struct NCCLSymmetricMemoryContext {
 private:
  // Can not use this unmapped pointer from outside
  void* raw_window_ptr;

 public:
  // Global
  int rank_idx;
  int num_ranks;

  // Logical
  int num_scaleout_ranks, num_scaleup_ranks;
  int scaleout_rank_idx, scaleup_rank_idx;

  // Physical
  int num_rdma_ranks, num_nvl_ranks;
  int rdma_rank_idx, nvl_rank_idx;
  bool is_scaleup_nvlink;

  // NCCL handles
  ncclComm_t comm;
  ncclDevComm_t dev_comm;
  ncclWindow_t window;
  void* mapped_window_ptr;
  std::vector<void*> nvl_window_ptrs;

  // Configs
  int num_allocated_qps;

  NCCLSymmetricMemoryContext(int64_t const& nccl_comm, int const& num_ranks,
                             int const& rank_idx, size_t const& size,
                             size_t const& alignment,
                             bool const& allow_hybrid_mode, int const& sl_idx,
                             int const& num_allocated_qps);

  // TODO: finish this with `explicit_destroy`
  // ~NCCLSymmetricMemoryContext();

  void* get_sym_ptr(void* ptr, int const& dst_rank_idx) const;

  void finalize() const;
};

}  // namespace deep_ep::nccl

namespace deep_ep::cuda_driver {

void batched_write(CUstream stream, std::vector<void*> const& ptrs,
                   int const& value);

void batched_wait(CUstream stream, std::vector<void*> const& ptrs,
                  int const& value);

void batched_write_and_wait(CUstream stream,
                            std::vector<void*> const& write_ptrs,
                            std::vector<void*> const& wait_ptrs,
                            int const& value);

}  // namespace deep_ep::cuda_driver
