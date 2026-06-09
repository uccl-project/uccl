#include "../include/gpu_rt.h"
#include "../include/transport.h"
#include "backend/device_backend.h"
#include "backend/rdma_local_copy_backend.h"
#include "backend/transport_backend.h"
#include "executor.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

namespace UKernel {
namespace CCL {

Executor::Executor(ExecutorConfig const& config) {
  owned_transport_backend_ = std::make_unique<CommunicatorTransportBackend>(
      TransportBackendConfig{config.gpu_id, config.rank, config.world_size,
                             config.communicator_config});
  auto* comm = &static_cast<CommunicatorTransportBackend*>(
                    owned_transport_backend_.get())
                    ->communicator();
  owned_device_backend_ = std::make_unique<DeviceBackend>(DeviceBackendConfig{
      config.device_task_capacity, config.max_device_fifos,
      config.threads_per_block, config.fifo_capacity, config.smem_size});
  backends_.transport = owned_transport_backend_.get();
  backends_.device = owned_device_backend_.get();

  if (config.enable_rdma_copy) {
    owned_rdma_copy_backend_ = std::make_unique<RdmaLocalCopyBackend>(
        RdmaLocalCopyBackendConfig{config.gpu_id});
    backends_.rdma_copy = owned_rdma_copy_backend_.get();
  }
  resolve_ipc_buffer_pointer_ = [comm](int remote_rank,
                                       uint32_t remote_buffer_id, size_t offset,
                                       size_t bytes, void** out_ptr,
                                       int* out_device_idx) {
    return comm->try_resolve_remote_ipc_pointer(
        remote_rank, remote_buffer_id, offset, bytes, out_ptr, out_device_idx);
  };

  // SM IPC: allocate GPU completion buffers and exchange IPC handles.
  // Each rank registers its buffer with a deterministic ID so the peer can
  // find it via wait_ipc.  Base 0x40000000 avoids collisions with user IDs.
  static constexpr uint32_t kCompIdBase = 0x40000000u;
  int n = comm->world_size();
  gpu_comp_.resize(n);
  for (int peer = 0; peer < n; ++peer) {
    if (peer == config.rank) continue;
    GPU_RT_CHECK(gpuMalloc(&gpu_comp_[peer].local, 16));
    GPU_RT_CHECK(gpuMemset(gpu_comp_[peer].local, 0, 16));
    uint32_t comp_id = kCompIdBase + peer;
    comm->reg_ipc(comp_id, gpu_comp_[peer].local, 16, true);
  }
  for (int peer = 0; peer < n; ++peer) {
    if (peer == config.rank) continue;
    uint32_t wait_id = kCompIdBase + config.rank;
    if (!comm->wait_ipc(peer, wait_id, 30000))
      throw std::runtime_error("GPU comp buffer IPC exchange failed for peer " +
                               std::to_string(peer));
    int remote_dev = -1;
    comm->try_resolve_remote_ipc_pointer(peer, wait_id, 0, 16,
                                         &gpu_comp_[peer].remote, &remote_dev);
  }
  static_cast<DeviceBackend*>(owned_device_backend_.get())
      ->set_signal_buffers(gpu_comp_);
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
