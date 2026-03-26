#include "executor.h"
#include "executor_impl.h"
#include "backend/device_backend.h"
#include "backend/transport_backend.h"
#include <memory>
#include <utility>

namespace UKernel {
namespace CCL {

Executor::Executor(CollectiveMemory memory, ExecutorConfig const& config)
    : impl_(nullptr) {
  auto shared_memory =
      std::make_shared<CollectiveMemory>(std::move(memory));
  auto transport_backend = std::make_unique<CommunicatorTransportBackend>(
      TransportBackendConfig{config.gpu_id, config.rank, config.world_size,
                             config.communicator_config},
      shared_memory);
  auto* communicator = &transport_backend->communicator();
  impl_ = std::make_unique<Impl>(
      std::move(transport_backend),
      std::make_unique<DeviceBackend>(
          shared_memory,
          DeviceBackendConfig{config.device_task_capacity,
                              config.max_device_fifos,
                              config.threads_per_block, config.fifo_capacity,
                              config.smem_size}));
  impl_->runtime_memory = std::move(shared_memory);
  impl_->resolve_remote_buffer_ptr =
      [communicator](int remote_rank, uint32_t mr_id, size_t offset,
                     size_t bytes, void** out_ptr, int* out_device_idx) {
        return communicator->resolve_remote_buffer_pointer(
            remote_rank, mr_id, offset, bytes, out_ptr, out_device_idx);
      };
}

UKernel::Transport::Communicator* Executor::communicator() {
  if (!impl_ || !impl_->owned_transport_backend) return nullptr;
  auto* backend = dynamic_cast<CommunicatorTransportBackend*>(
      impl_->owned_transport_backend.get());
  return backend != nullptr ? &backend->communicator() : nullptr;
}

UKernel::Transport::Communicator const* Executor::communicator() const {
  if (!impl_ || !impl_->owned_transport_backend) return nullptr;
  auto const* backend = dynamic_cast<CommunicatorTransportBackend const*>(
      impl_->owned_transport_backend.get());
  return backend != nullptr ? &backend->communicator() : nullptr;
}

CollectiveMemory* Executor::memory() {
  if (!impl_ || !impl_->owned_transport_backend) return nullptr;
  auto* backend = dynamic_cast<CommunicatorTransportBackend*>(
      impl_->owned_transport_backend.get());
  return backend != nullptr ? &backend->memory() : nullptr;
}

CollectiveMemory const* Executor::memory() const {
  if (!impl_ || !impl_->owned_transport_backend) return nullptr;
  auto const* backend = dynamic_cast<CommunicatorTransportBackend const*>(
      impl_->owned_transport_backend.get());
  return backend != nullptr ? &backend->memory() : nullptr;
}

}  // namespace CCL
}  // namespace UKernel
