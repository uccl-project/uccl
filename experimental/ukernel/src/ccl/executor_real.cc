#include "../transport/communicator.h"
#include "backend/device_backend.h"
#include "backend/transport_backend.h"
#include "executor.h"
#include "executor_impl.h"
#include <cstdint>
#include <memory>
#include <utility>

namespace UKernel {
namespace CCL {

Executor::Executor(ExecutorConfig const& config) : impl_(nullptr) {
  auto transport_backend = std::make_unique<CommunicatorTransportBackend>(
      TransportBackendConfig{config.gpu_id, config.rank, config.world_size,
                             config.communicator_config});
  auto* communicator = &transport_backend->communicator();
  impl_ = std::make_unique<Impl>(
      std::move(transport_backend),
      std::make_unique<DeviceBackend>(DeviceBackendConfig{
          config.device_task_capacity,
          config.max_device_fifos,
          config.threads_per_block,
          config.fifo_capacity,
          config.smem_size,
      }));
  impl_->resolve_ipc_buffer_pointer =
      [communicator](int remote_rank, uint32_t remote_buffer_id, size_t offset,
                     size_t bytes, void** out_ptr, int* out_device_idx) {
        if (out_ptr == nullptr) return false;
        UKernel::Transport::IPCItem ipc{};
        try {
          ipc = communicator->get_ipc(remote_rank, remote_buffer_id);
        } catch (...) {
          return false;
        }
        if (!ipc.valid || ipc.direct_ptr == nullptr) return false;
        if (offset > ipc.bytes || bytes > ipc.bytes - offset) return false;
        *out_ptr = reinterpret_cast<void*>(
            reinterpret_cast<uintptr_t>(ipc.direct_ptr) + ipc.base_offset +
            offset);
        if (out_device_idx != nullptr) {
          *out_device_idx = ipc.device_idx;
        }
        return true;
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

}  // namespace CCL
}  // namespace UKernel
