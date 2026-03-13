#include "ccl_api.h"
#include <stdexcept>
#include <utility>

namespace UKernel {
namespace Compute {

struct CollectiveHostApi::Impl {
  explicit Impl(CollectiveHostApiConfig const& config)
      : pk_backend(build_pk_backend(config)),
        ce_backend(build_ce_backend(config)),
        executor(build_backends()) {}

  static std::unique_ptr<ComputePersistentKernelBackend> build_pk_backend(
      CollectiveHostApiConfig const& config) {
    if (config.kernel == nullptr) {
      throw std::invalid_argument("CollectiveHostApi requires a persistent kernel");
    }
    return std::make_unique<ComputePersistentKernelBackend>(
        *config.kernel, config.buffers, config.dtype, config.reduce_type,
        config.pk_transfer_path, config.num_blocks);
  }

  static std::unique_ptr<ComputeCopyEngineBackend> build_ce_backend(
      CollectiveHostApiConfig const& config) {
    if (!config.enable_copy_engine) return nullptr;
    return std::make_unique<ComputeCopyEngineBackend>(
        config.buffers, config.dst_device, config.src_device, config.ce_stream);
  }

  UKernel::CCL::ExecutorBackends build_backends() {
    UKernel::CCL::ExecutorBackends backends{};
    backends.persistent = pk_backend.get();
    backends.ce = ce_backend.get();
    return backends;
  }

  std::unique_ptr<ComputePersistentKernelBackend> pk_backend;
  std::unique_ptr<ComputeCopyEngineBackend> ce_backend;
  UKernel::CCL::Executor executor;
};

CollectiveHostApi::CollectiveHostApi(CollectiveHostApiConfig const& config)
    : impl_(std::make_unique<Impl>(config)) {}

CollectiveHostApi::~CollectiveHostApi() = default;

UKernel::CCL::CollectiveOpHandle CollectiveHostApi::submit_allgather(
    UKernel::CCL::CollectiveConfig const& config) {
  return impl_->executor.submit_allgather(config);
}

UKernel::CCL::CollectiveOpHandle CollectiveHostApi::submit_allreduce(
    UKernel::CCL::CollectiveConfig const& config) {
  return impl_->executor.submit_allreduce(config);
}

bool CollectiveHostApi::poll(UKernel::CCL::CollectiveOpHandle handle) {
  return impl_->executor.poll(handle);
}

void CollectiveHostApi::wait(UKernel::CCL::CollectiveOpHandle handle) {
  impl_->executor.wait(handle);
}

void CollectiveHostApi::release(UKernel::CCL::CollectiveOpHandle handle) {
  impl_->executor.release(handle);
}

UKernel::CCL::CollectiveOpStatus CollectiveHostApi::status(
    UKernel::CCL::CollectiveOpHandle handle) const {
  return impl_->executor.status(handle);
}

}  // namespace Compute
}  // namespace UKernel
