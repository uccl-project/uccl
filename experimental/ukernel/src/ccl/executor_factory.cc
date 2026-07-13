#include "../../include/transport.h"
#include "backend/backend.h"
#include "backend/device_backend.h"
#include "backend/signal_backend.h"
#include "backend/transport_backend.h"
#include "executor.h"
#include <memory>
#if !defined(__HIP_PLATFORM_AMD__)
#include <gdrapi.h>
#include <mutex>
#include <unordered_map>
#endif
#include <cstring>

#if !defined(__HIP_PLATFORM_AMD__)
namespace {
struct GdrSlot {
  gdr_mh_t mh{};
  void* map_ptr = nullptr;
};
std::mutex g_gdr_mu;
std::unordered_map<void*, GdrSlot> g_gdr_slots;
gdr_t g_gdr = nullptr;
}  // namespace
#endif

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

  ex->register_buf_fn_ = [](Transport::Communicator* comm, uint32_t id,
                            void* ptr, size_t len) {
    comm->register_buffer(id, ptr, len);
  };
  ex->peer_setup_fn_ = [](Transport::Communicator* comm, int rank,
                          int world_size) {
    for (int p = 0; p < world_size; ++p) {
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
      comm->connect(p, Transport::PeerTransportKind::Rdma);
      comm->accept(p, Transport::PeerTransportKind::Rdma);
    }
  };
  ex->resolve_buf_fn_ = [](Transport::Communicator* comm, int peer,
                           int /*world_size*/, uint32_t buf_id) {
    comm->resolve_remote_buffer(peer, buf_id, 30000);
  };
  ex->same_host_fn_ = [](Transport::Communicator* comm, int peer) {
    return comm->same_host(peer);
  };
  // GDR read-tail: pin GPU output buffer once at registration time via
  // GDRCopy BAR1 mapping.  At each RDMA-Put completion the drain thread
  // reads the tail of the written range from the already-mapped pointer
  // — the PCIe read forces GPU L2 invalidation on pre-Hopper hardware.
  ex->pin_buf_fn_ = [](void* gpu_ptr, size_t bytes) {
#if !defined(__HIP_PLATFORM_AMD__)
    if (!gpu_ptr || !bytes) return;
    {
      std::lock_guard<std::mutex> lk(g_gdr_mu);
      if (g_gdr_slots.count(gpu_ptr)) return;
      if (!g_gdr) {
        g_gdr = gdr_open();
        if (!g_gdr) return;
      }
    }
    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(gpu_ptr);
    gdr_mh_t mh{};
    if (gdr_pin_buffer(g_gdr, dptr, bytes, 0, 0, &mh) != 0) return;
    void* map_ptr = nullptr;
    if (gdr_map(g_gdr, mh, &map_ptr, bytes) != 0) {
      gdr_unpin_buffer(g_gdr, mh);
      return;
    }
    std::lock_guard<std::mutex> lk(g_gdr_mu);
    g_gdr_slots[gpu_ptr] = {mh, map_ptr};
#else
    (void)gpu_ptr; (void)bytes;
#endif
  };

  ex->flush_rdma_fn_ = [](void* gpu_buf_ptr, size_t offset, size_t bytes) {
#if !defined(__HIP_PLATFORM_AMD__)
    void* map_ptr = nullptr;
    {
      std::lock_guard<std::mutex> lk(g_gdr_mu);
      auto it = g_gdr_slots.find(gpu_buf_ptr);
      if (it != g_gdr_slots.end()) map_ptr = it->second.map_ptr;
    }
    if (!map_ptr) return;

    size_t tail_off = offset + bytes;
    size_t read_sz = 32;
    if (tail_off < read_sz) {
      tail_off = offset;
      read_sz = (bytes < 32) ? bytes : 32;
    } else {
      tail_off -= read_sz;
    }
    volatile char sink[32];
    std::memcpy(const_cast<char*>(sink),
                static_cast<const char*>(map_ptr) + tail_off, read_sz);
#else
    (void)gpu_buf_ptr; (void)offset; (void)bytes;
#endif
  };

  fprintf(stderr, "[FACTORY] done rank=%d\n", config.rank);
  return ex;
}

}  // namespace CCL
}  // namespace UKernel
