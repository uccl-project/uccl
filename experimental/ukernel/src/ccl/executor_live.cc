#include "executor.h"
#include "backend/device_backend.h"
#include "backend/transport_backend.h"
#include "../include/transport.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace UKernel {
namespace CCL {

namespace {

std::vector<Backend*> live_backend_sources(ExecutorBackends const& backends) {
  std::vector<Backend*> out;
  if (backends.transport != nullptr) out.push_back(backends.transport);
  if (backends.device != nullptr && backends.device != backends.transport)
    out.push_back(backends.device);
  if (backends.fallback != nullptr && backends.fallback != backends.transport &&
      backends.fallback != backends.device)
    out.push_back(backends.fallback);
  return out;
}

}  // namespace

Executor::Executor(ExecutorConfig const& config) {
  owned_transport_backend_ =
      std::make_unique<CommunicatorTransportBackend>(TransportBackendConfig{
          config.gpu_id, config.rank, config.world_size,
          config.communicator_config});
  auto* communicator =
      &static_cast<CommunicatorTransportBackend*>(
           owned_transport_backend_.get())
           ->communicator();
  owned_device_backend_ = std::make_unique<DeviceBackend>(DeviceBackendConfig{
      config.device_task_capacity, config.max_device_fifos,
      config.threads_per_block, config.fifo_capacity, config.smem_size});
  backends_.transport = owned_transport_backend_.get();
  backends_.device = owned_device_backend_.get();
  completion_sources_ = live_backend_sources(backends_);
  resolve_ipc_buffer_pointer_ =
      [communicator](int remote_rank, uint32_t remote_buffer_id, size_t offset,
                     size_t bytes, void** out_ptr, int* out_device_idx) {
        return communicator->try_resolve_remote_ipc_pointer(
            remote_rank, remote_buffer_id, offset, bytes, out_ptr,
            out_device_idx);
      };
}

UKernel::Transport::Communicator* Executor::communicator() {
  if (backends_.transport == nullptr) return nullptr;
  return &static_cast<CommunicatorTransportBackend*>(backends_.transport)
              ->communicator();
}

UKernel::Transport::Communicator const* Executor::communicator() const {
  if (backends_.transport == nullptr) return nullptr;
  return &static_cast<CommunicatorTransportBackend const*>(backends_.transport)
              ->communicator();
}

}  // namespace CCL
}  // namespace UKernel
