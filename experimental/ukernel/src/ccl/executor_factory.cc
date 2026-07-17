#include "../../include/transport.h"
#include "backend/backend.h"
#include "backend/device_backend.h"
#include "backend/signal_backend.h"
#include "backend/transport_backend.h"
#include "executor.h"
#include <memory>
#include <stdexcept>
#include <vector>

namespace UKernel {
namespace CCL {

std::unique_ptr<SprayExecutor> SprayExecutor::create(
    SprayExecutorConfig const& config) {
  fprintf(stderr, "[FACTORY] creating Communicator rank=%d gpu=%d\n", config.rank, config.gpu_id);
  auto comm_cfg = std::make_shared<Transport::CommunicatorConfig>();
  comm_cfg->exchanger_ip = config.exchanger_ip;
  comm_cfg->exchanger_port = config.exchanger_port;
  comm_cfg->local_id = config.local_id;
  auto comm = std::make_shared<UKernel::Transport::Communicator>(
      config.gpu_id, config.rank, config.world_size, comm_cfg);
  fprintf(stderr, "[FACTORY] Communicator done\n");
  auto dev_be = std::make_unique<DeviceBackend>(DeviceBackendConfig{
      .task_capacity = static_cast<uint32_t>(config.device_task_capacity),
      .max_fifos = static_cast<uint32_t>(config.max_device_fifos),
      .threads_per_block = static_cast<uint32_t>(config.threads_per_block),
      .blocks_per_worker = static_cast<uint32_t>(config.blocks_per_worker),
      .fifo_capacity = static_cast<uint32_t>(config.fifo_capacity),
      .smem_size = config.smem_size,
  });
  auto tpt_be = std::make_unique<TransportBackend>(comm.get());
  auto sig_be = std::make_unique<SignalBackend>();

  auto ex = std::make_unique<SprayExecutor>(dev_be.get(), tpt_be.get(),
                                            sig_be.get(), config.world_size);
  ex->max_concurrent_runs_ = config.max_concurrent_runs;
  ex->owned_device_ = std::move(dev_be);
  ex->owned_transport_ = std::move(tpt_be);
  ex->owned_signal_ = std::move(sig_be);
  ex->owned_comm_ = std::move(comm);

  ex->device_be_->set_comm(ex->owned_comm_.get());
  ex->tpt_be_->set_comm(ex->owned_comm_.get());
  ex->signal_be_->set_comm(ex->owned_comm_.get());

  // Enable P2P access for same-host peers so the DeviceBackend SM kernel
  // can directly read/write remote GPU memory.
  for (int p = 0; p < config.world_size; ++p) {
    if (p == config.rank) continue;
    if (ex->owned_comm_->same_host(p)) {
      int peer_gpu = ex->owned_comm_->peer_gpu_idx(p);
      if (peer_gpu >= 0) {
        gpuError_t err = gpuDeviceEnablePeerAccess(peer_gpu, 0);
        if (err != gpuSuccess && err != gpuErrorPeerAccessAlreadyEnabled)
          std::cerr << "[FACTORY] gpuDeviceEnablePeerAccess failed gpu="
                    << config.gpu_id << " peer=" << peer_gpu << std::endl;
      }
    }
  }

  ex->register_buf_fn_ = [](Transport::Communicator* comm, uint32_t id,
                            void* ptr, size_t len) {
    comm->register_buffer(id, ptr, len);
  };
  ex->peer_setup_fn_ = [](Transport::Communicator* comm, int rank,
                          std::vector<int> const& peers) {
    for (int p : peers) {
      if (p == rank) continue;
      bool same = comm->same_host(p);

      if (same) {
        if (rank < p) {
          comm->connect(p, Transport::PeerTransportKind::Ipc);
          comm->accept(p, Transport::PeerTransportKind::Ipc);
        } else {
          comm->accept(p, Transport::PeerTransportKind::Ipc);
          comm->connect(p, Transport::PeerTransportKind::Ipc);
        }
      }
      if (rank < p) {
        comm->connect(p, Transport::PeerTransportKind::Rdma);
        comm->accept(p, Transport::PeerTransportKind::Rdma);
      } else {
        comm->accept(p, Transport::PeerTransportKind::Rdma);
        comm->connect(p, Transport::PeerTransportKind::Rdma);
      }
    }
  };
  ex->resolve_buf_fn_ = [](Transport::Communicator* comm, int peer,
                           int /*world_size*/, uint32_t buf_id) {
    comm->resolve_remote_buffer(peer, buf_id, 30000);
  };
  ex->same_host_fn_ = [](Transport::Communicator* comm, int peer) {
    return comm->same_host(peer);
  };

  ex->start();
  fprintf(stderr, "[FACTORY] done rank=%d\n", config.rank);
  return ex;
}

}  // namespace CCL
}  // namespace UKernel
