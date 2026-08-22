#include "../../include/transport.h"
#include "backend/backend.h"
#include "backend/device_backend.h"
#include "backend/signal_backend.h"
#include "backend/transport_backend.h"
#include "executor.h"
#include "gpu_rt.h"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sched.h>
#include <stdexcept>
#include <unistd.h>
#include <vector>

namespace UKernel {
namespace CCL {

namespace {

// ukernel relies on host threads doing tight polling for completion and
// signal propagation. MPI launchers that bind each rank to a single CPU
// (or a small subset) can starve those pollers and cause 5-10x slowdowns
// (observed with OpenMPI 5 even when hwloc binding seemed disabled).
//
// This guard does two things:
//   1. Warns when the process affinity is narrower than the host.
//   2. If UK_CCL_UNBIND=1 is set, tries to widen the process affinity to
//      all online CPUs so the same binary works under restrictive MPI
//      launchers without requiring launcher-specific flags.
void maybe_handle_cpu_binding() {
  cpu_set_t cur;
  CPU_ZERO(&cur);
  if (sched_getaffinity(0, sizeof(cur), &cur) != 0) return;
  int cur_count = CPU_COUNT(&cur);
  long online = sysconf(_SC_NPROCESSORS_ONLN);
  if (cur_count <= 0 || online <= 0 || cur_count >= online) return;

  char const* env = std::getenv("UK_CCL_UNBIND");
  bool unbind = env && std::string(env) != "0";
  if (unbind) {
    cpu_set_t all;
    CPU_ZERO(&all);
    long max_cpu = online < CPU_SETSIZE ? online : CPU_SETSIZE;
    for (long i = 0; i < max_cpu; ++i) CPU_SET((int)i, &all);
    if (sched_setaffinity(0, sizeof(all), &all) == 0) {
      std::fprintf(
          stderr,
          "[ukernel] widened CPU affinity %d -> %ld CPUs (UK_CCL_UNBIND=1)\n",
          cur_count, online);
    } else {
      std::fprintf(stderr,
                   "[ukernel] WARN UK_CCL_UNBIND=1 but sched_setaffinity "
                   "failed: %s\n",
                   std::strerror(errno));
    }
  } else {
    std::fprintf(
        stderr,
        "[ukernel] WARN process bound to %d/%ld CPUs; ukernel host polling "
        "may be very slow. Use --bind-to none (or set UK_CCL_UNBIND=1)\n",
        cur_count, online);
  }
}

// Default device blocks_per_worker from the GPU's compute capability.
// Measured sweet spots: A40-class (sm_86/89) 8 blocks; Hopper (sm_90)
// 16; Blackwell (sm_100/103) 32 — with the auto REDUCE_ILP=16 build,
// 32 blocks reaches ~76% of native on the reduce bench and is the
// few-SM-friendly pick (64 = ~95%, override via UK_CCL_DEV_BLOCKS).
// Capped at the SM count.
uint32_t auto_device_blocks(int gpu_id) {
  gpuDeviceProp prop{};
  GPU_RT_CHECK(gpuGetDeviceProperties(&prop, gpu_id));
  uint32_t blocks = 8;
  if (prop.major == 9) {
    blocks = 16;  // Hopper
  } else if (prop.major == 10) {
    blocks = 32;  // Blackwell
  }
  if (blocks > static_cast<uint32_t>(prop.multiProcessorCount)) {
    blocks = static_cast<uint32_t>(prop.multiProcessorCount);
  }
  UK_DBG(UK_DBG_LVL_EXEC,
         "[factory] auto device blocks=%u (compute=%u.%u, sm_count=%d)",
         blocks, prop.major, prop.minor, prop.multiProcessorCount);
  return blocks;
}

}  // namespace

std::unique_ptr<SprayExecutor> SprayExecutor::create(
    SprayExecutorConfig const& config) {
  maybe_handle_cpu_binding();
  auto comm_cfg = std::make_shared<Transport::CommunicatorConfig>();
  comm_cfg->exchanger_ip = config.exchanger_ip;
  comm_cfg->exchanger_port = config.exchanger_port;
  comm_cfg->local_id = config.local_id;
  auto comm = std::make_shared<UKernel::Transport::Communicator>(
      config.gpu_id, config.rank, config.world_size, comm_cfg);
  DeviceBackendConfig dev_cfg{
      .task_capacity = static_cast<uint32_t>(config.device_task_capacity),
      .max_fifos = static_cast<uint32_t>(config.max_device_fifos),
      .threads_per_block = static_cast<uint32_t>(config.threads_per_block),
      .blocks_per_worker = 0,  // resolved below (env / config / auto)
      .fifo_capacity = static_cast<uint32_t>(config.fifo_capacity),
      .smem_size = static_cast<uint32_t>(config.smem_size),
      .idle_exit_after_us = config.device_idle_exit_us,
  };
  // Optional env overrides for benchmarking (win over config values):
  // UK_CCL_DEV_FIFOS / UK_CCL_DEV_BLOCKS / UK_CCL_DEV_THREADS.
  if (char const* v = std::getenv("UK_CCL_DEV_FIFOS"))
    dev_cfg.max_fifos = static_cast<uint32_t>(std::stoul(v));
  if (char const* v = std::getenv("UK_CCL_DEV_BLOCKS"))
    dev_cfg.blocks_per_worker = static_cast<uint32_t>(std::stoul(v));
  else if (config.blocks_per_worker > 0)
    dev_cfg.blocks_per_worker = static_cast<uint32_t>(config.blocks_per_worker);
  else
    dev_cfg.blocks_per_worker = auto_device_blocks(config.gpu_id);
  if (char const* v = std::getenv("UK_CCL_DEV_THREADS"))
    dev_cfg.threads_per_block = static_cast<uint32_t>(std::stoul(v));
  if (char const* v = std::getenv("UK_CCL_DEV_IDLE_EXIT_US"))
    dev_cfg.idle_exit_after_us = static_cast<uint32_t>(std::stoul(v));
  auto dev_be = std::make_unique<DeviceBackend>(dev_cfg);
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

  ex->register_buf_fn_ = [](Transport::Communicator* comm, uint32_t id,
                            void* ptr, size_t len) {
    comm->register_buffer(id, ptr, len);
  };
  ex->deregister_buf_fn_ = [](Transport::Communicator* comm, uint32_t id) {
    comm->dereg_mr(id);
    comm->dereg_ipc(id);
  };
  ex->peer_setup_fn_ = [](Transport::Communicator* comm, int rank,
                          std::vector<int> const& peers) {
    for (int p : peers) {
      if (p == rank) continue;
      bool same = comm->same_host(p);

      if (same) {
        if (rank < p) {
          UK_DBG(UK_DBG_LVL_EXEC, "[peer_setup r%d] IPC connect to p%d ...",
                 rank, p);
          comm->connect(p, Transport::PeerTransportKind::Ipc);
          UK_DBG(UK_DBG_LVL_EXEC, "[peer_setup r%d] IPC connect to p%d done",
                 rank, p);
          UK_DBG(UK_DBG_LVL_EXEC, "[peer_setup r%d] IPC accept from p%d ...",
                 rank, p);
          comm->accept(p, Transport::PeerTransportKind::Ipc);
          UK_DBG(UK_DBG_LVL_EXEC, "[peer_setup r%d] IPC accept from p%d done",
                 rank, p);
        } else {
          UK_DBG(UK_DBG_LVL_EXEC, "[peer_setup r%d] IPC accept from p%d ...",
                 rank, p);
          comm->accept(p, Transport::PeerTransportKind::Ipc);
          UK_DBG(UK_DBG_LVL_EXEC, "[peer_setup r%d] IPC accept from p%d done",
                 rank, p);
          UK_DBG(UK_DBG_LVL_EXEC, "[peer_setup r%d] IPC connect to p%d ...",
                 rank, p);
          comm->connect(p, Transport::PeerTransportKind::Ipc);
          UK_DBG(UK_DBG_LVL_EXEC, "[peer_setup r%d] IPC connect to p%d done",
                 rank, p);
        }
      }
      if (rank < p) {
        UK_DBG(UK_DBG_LVL_EXEC, "[peer_setup r%d] RDMA connect to p%d ...",
               rank, p);
        comm->connect(p, Transport::PeerTransportKind::Rdma);
        UK_DBG(UK_DBG_LVL_EXEC, "[peer_setup r%d] RDMA connect to p%d done",
               rank, p);
        UK_DBG(UK_DBG_LVL_EXEC, "[peer_setup r%d] RDMA accept from p%d ...",
               rank, p);
        comm->accept(p, Transport::PeerTransportKind::Rdma);
        UK_DBG(UK_DBG_LVL_EXEC, "[peer_setup r%d] RDMA accept from p%d done",
               rank, p);
      } else {
        UK_DBG(UK_DBG_LVL_EXEC, "[peer_setup r%d] RDMA accept from p%d ...",
               rank, p);
        comm->accept(p, Transport::PeerTransportKind::Rdma);
        UK_DBG(UK_DBG_LVL_EXEC, "[peer_setup r%d] RDMA accept from p%d done",
               rank, p);
        UK_DBG(UK_DBG_LVL_EXEC, "[peer_setup r%d] RDMA connect to p%d ...",
               rank, p);
        comm->connect(p, Transport::PeerTransportKind::Rdma);
        UK_DBG(UK_DBG_LVL_EXEC, "[peer_setup r%d] RDMA connect to p%d done",
               rank, p);
      }
    }
    // Enable P2P access now that peer GPU indices are known (after RDMA
    // exchange).
    for (int p : peers) {
      if (p == rank) continue;
      if (comm->same_host(p)) {
        int peer_gpu = comm->peer_gpu_idx(p);
        if (peer_gpu >= 0) {
          UK_DBG(UK_DBG_LVL_EXEC, "[peer_setup r%d] enable P2P to gpu=%d", rank,
                 peer_gpu);
          gpuError_t err = gpuDeviceEnablePeerAccess(peer_gpu, 0);
          if (err == gpuErrorPeerAccessAlreadyEnabled) {
            // Tolerated — but clear the sticky per-thread error so later
            // CUDA users on this thread (e.g. torch) don't trip over it.
            (void)gpuGetLastError();
          } else if (err != gpuSuccess) {
            std::cerr << "[peer_setup r" << rank
                      << "] gpuDeviceEnablePeerAccess failed gpu=" << peer_gpu
                      << " err=" << err << std::endl;
          }
        }
      }
    }
  };
  ex->resolve_buf_fn_ = [](Transport::Communicator* comm, int peer,
                           int /*world_size*/, uint32_t buf_id) {
    // Fail fast: a failed resolve means the peer never published (or
    // explicitly failed) the buffer — silently continuing leaves the
    // executor to crash later at enqueue with a confusing error.
    // propagate as an exception so prepare() surfaces it immediately.
    if (!comm->resolve_remote_buffer(peer, buf_id, 30000))
      throw std::runtime_error("resolve_remote_buffer failed peer=" +
                               std::to_string(peer) + " buf=" +
                               std::to_string(buf_id));
  };
  ex->same_host_fn_ = [](Transport::Communicator* comm, int peer) {
    return comm->same_host(peer);
  };

  ex->start();
  return ex;
}

}  // namespace CCL
}  // namespace UKernel
