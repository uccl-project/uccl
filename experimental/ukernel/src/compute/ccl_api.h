#pragma once

#include "../ccl/executor.h"
#include "ccl_backend.h"
#include "persistent.h"
#include <memory>

namespace UKernel {
namespace Compute {

struct CollectiveHostApiConfig {
  PersistentKernel<Task>* kernel = nullptr;
  CollectiveBuffers buffers{};
  DataType dtype = DataType::Fp32;
  ReduceType reduce_type = ReduceType::Sum;
  TransferPath pk_transfer_path = TransferPath::Auto;
  uint32_t num_blocks = 1;
  bool enable_copy_engine = false;
  int dst_device = -1;
  int src_device = -1;
  gpuStream_t ce_stream = nullptr;
};

class CollectiveHostApi {
 public:
  explicit CollectiveHostApi(CollectiveHostApiConfig const& config);
  ~CollectiveHostApi();

  CollectiveHostApi(CollectiveHostApi const&) = delete;
  CollectiveHostApi& operator=(CollectiveHostApi const&) = delete;

  UKernel::CCL::CollectiveOpHandle submit_allgather(
      UKernel::CCL::CollectiveConfig const& config);
  UKernel::CCL::CollectiveOpHandle submit_allreduce(
      UKernel::CCL::CollectiveConfig const& config);
  bool poll(UKernel::CCL::CollectiveOpHandle handle);
  void wait(UKernel::CCL::CollectiveOpHandle handle);
  void release(UKernel::CCL::CollectiveOpHandle handle);
  UKernel::CCL::CollectiveOpStatus status(
      UKernel::CCL::CollectiveOpHandle handle) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace Compute
}  // namespace UKernel
