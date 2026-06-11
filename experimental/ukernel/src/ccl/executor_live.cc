#include "executor.h"
#include "../../include/gpu_rt.h"
#include "../../include/transport.h"
#include "backend/backend.h"
#include "backend/device_backend.h"
#include "backend/transport_backend.h"
#include <memory>
#include <stdexcept>
#include <vector>

namespace UKernel {
namespace CCL {

static constexpr uint32_t kCompIdBase = 0x40000000u;

std::unique_ptr<SprayExecutor> SprayExecutor::create(
    SprayExecutorConfig const& config) {
  auto comm = std::make_shared<UKernel::Transport::Communicator>(
      config.gpu_id, config.rank, config.world_size,
      config.communicator_config);

  auto dev_be = std::make_unique<DeviceBackend>(DeviceBackendConfig{
      .task_capacity = static_cast<uint32_t>(config.device_task_capacity),
      .max_fifos = static_cast<uint32_t>(config.max_device_fifos),
      .threads_per_block = static_cast<uint32_t>(config.threads_per_block),
      .fifo_capacity = static_cast<uint32_t>(config.fifo_capacity),
      .smem_size = static_cast<uint32_t>(config.smem_size),
  });
  auto tpt_be = std::make_unique<TransportBackend>(comm.get());

  int n = comm->world_size();
  std::vector<GpuSignalPeer> gpu_comp(n);
  for (int peer = 0; peer < n; ++peer) {
    if (peer == config.rank) continue;
    GPU_RT_CHECK(gpuMalloc(&gpu_comp[peer].local, 16));
    GPU_RT_CHECK(gpuMemset(gpu_comp[peer].local, 0, 16));
    comm->reg_ipc(kCompIdBase + peer, gpu_comp[peer].local, 16, true);
  }
  for (int peer = 0; peer < n; ++peer) {
    if (peer == config.rank) continue;
    if (!comm->wait_ipc(peer, kCompIdBase + config.rank, 30000))
      throw std::runtime_error("GPU comp buffer IPC exchange failed for peer " +
                               std::to_string(peer));
    int remote_dev = -1;
    comm->try_resolve_remote_ipc_pointer(peer, kCompIdBase + config.rank,
                                         0, 16, reinterpret_cast<void**>(&gpu_comp[peer].remote),
                                         &remote_dev);
  }
  dev_be->set_signal_buffers(gpu_comp);

  auto ex = std::make_unique<SprayExecutor>(dev_be.get(), tpt_be.get());
  ex->owned_device_ = std::move(dev_be);
  ex->owned_transport_ = std::move(tpt_be);
  ex->owned_comm_ = std::move(comm);

  return ex;
}

}  // namespace CCL
}  // namespace UKernel
