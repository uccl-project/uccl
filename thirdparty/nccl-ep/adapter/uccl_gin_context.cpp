#include "uccl_gin_context.hpp"

#include "transport/uccl_proxy.hpp"
#include "transport/common.hpp"
#include <cuda_runtime.h>
#include <mpi.h>

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

#define ADAPTER_CUDA_OK(x)                                                 \
  do {                                                                     \
    cudaError_t e = (x);                                                   \
    if (e != cudaSuccess) {                                                \
      throw std::runtime_error(std::string("[UCCL-GIN] ") + __FILE__ +     \
                               ":" + std::to_string(__LINE__) + " " +     \
                               cudaGetErrorString(e));                     \
    }                                                                      \
  } while (0)

std::string iface_ip(const char* ifname) {
  struct ifaddrs* ifa = nullptr;
  getifaddrs(&ifa);
  std::string out;
  for (auto* p = ifa; p; p = p->ifa_next) {
    if (!p->ifa_addr || p->ifa_addr->sa_family != AF_INET) continue;
    if (std::strcmp(p->ifa_name, ifname) != 0) continue;
    char buf[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &((struct sockaddr_in*)p->ifa_addr)->sin_addr, buf,
              sizeof(buf));
    out = buf;
    break;
  }
  if (ifa) freeifaddrs(ifa);
  return out;
}

struct WirePeer {
  int rank;
  uintptr_t ptr;
  size_t nbytes;
  int ports[kNumProxyThs];
  char ip[64];
};

}  // namespace

namespace nccl_ep_adapter {

UcclGinContext::UcclGinContext(Config cfg) { setup(cfg); }

UcclGinContext::~UcclGinContext() { teardown(); }

void UcclGinContext::setup(Config cfg) {
  if (cfg.gpu_buffer_addr == 0 || cfg.gpu_buffer_bytes == 0) {
    throw std::invalid_argument("gpu_buffer_addr and gpu_buffer_bytes required");
  }

  const int local_rank = cfg.rank % cfg.local_world_size;
  const int node_idx = cfg.rank / cfg.local_world_size;
  const int num_nodes = cfg.world_size / cfg.local_world_size;

  ADAPTER_CUDA_OK(cudaSetDevice(local_rank));

  // Create proxies using the external GPU buffer
  for (int t = 0; t < kNumProxyThs; ++t) {
    auto p = std::make_unique<UcclProxy>(
        /*thread_idx=*/t, /*gpu_buffer_addr=*/cfg.gpu_buffer_addr,
        /*total_size=*/cfg.gpu_buffer_bytes,
        /*rank=*/cfg.rank, /*node_idx=*/node_idx,
        /*local_rank=*/local_rank, /*num_experts=*/0,
        /*num_ranks=*/cfg.world_size, /*num_nodes=*/num_nodes,
        /*use_normal_mode=*/true, /*is_intranode=*/(num_nodes <= 1),
        /*gpu_buffer_is_host_allocated=*/false, /*barrier_local_rank=*/-1,
        /*owns_gpu_buffer=*/false);
    proxies_.push_back(std::move(p));
  }

  // Exchange peer metadata via MPI
  PeerMeta me{};
  me.rank = cfg.rank;
  me.ptr = cfg.gpu_buffer_addr;
  me.nbytes = cfg.gpu_buffer_bytes;
  me.ip = iface_ip(cfg.ifname);
  if (me.ip.empty()) {
    throw std::runtime_error(std::string("interface has no IPv4 address: ") +
                             cfg.ifname);
  }
  for (int t = 0; t < kNumProxyThs; ++t) {
    me.listen_ports[t] = proxies_[t]->get_listen_port();
  }

  WirePeer mine{};
  mine.rank = me.rank;
  mine.ptr = me.ptr;
  mine.nbytes = me.nbytes;
  std::memcpy(mine.ports, me.listen_ports, sizeof(mine.ports));
  std::strncpy(mine.ip, me.ip.c_str(), sizeof(mine.ip) - 1);

  std::vector<WirePeer> all(cfg.world_size);
  MPI_Allgather(&mine, sizeof(WirePeer), MPI_BYTE, all.data(),
                sizeof(WirePeer), MPI_BYTE, MPI_COMM_WORLD);

  std::vector<PeerMeta> peers(cfg.world_size);
  for (int r = 0; r < cfg.world_size; ++r) {
    peers[r].rank = all[r].rank;
    peers[r].ptr = all[r].ptr;
    peers[r].nbytes = all[r].nbytes;
    peers[r].ip = all[r].ip;
    std::memcpy(peers[r].listen_ports, all[r].ports,
                sizeof(peers[r].listen_ports));
  }
  for (auto& p : proxies_) p->set_peers_meta(peers);

  MPI_Barrier(MPI_COMM_WORLD);
  for (auto& p : proxies_) p->start_dual();
  MPI_Barrier(MPI_COMM_WORLD);

  // Collect D2H ring handles into device array
  std::vector<uint64_t> dev_ring_addrs;
  for (auto& p : proxies_) {
    for (auto a : p->get_d2h_channel_handle_addrs()) {
      dev_ring_addrs.push_back(a);
    }
  }
  num_queues_ = static_cast<int>(dev_ring_addrs.size());
  std::vector<d2hq::D2HHandle*> h_handles(num_queues_);
  for (int i = 0; i < num_queues_; ++i) {
    h_handles[i] = reinterpret_cast<d2hq::D2HHandle*>(dev_ring_addrs[i]);
  }
  ADAPTER_CUDA_OK(
      cudaMalloc(&d_handles_, num_queues_ * sizeof(d2hq::D2HHandle*)));
  ADAPTER_CUDA_OK(cudaMemcpy(d_handles_, h_handles.data(),
                              num_queues_ * sizeof(d2hq::D2HHandle*),
                              cudaMemcpyHostToDevice));

  // Set up shared atomic buffer and resources
  uintptr_t atomic_base = proxies_[0]->get_atomic_buffer_addr();
  for (auto& p : proxies_) p->set_atomic_buffer_addr(atomic_base);

  resources_.d2h_queues = d_handles_;
  resources_.num_queues = static_cast<uint32_t>(num_queues_);
  resources_.window_base = cfg.gpu_buffer_addr;
  resources_.window_bytes = static_cast<uint64_t>(cfg.gpu_buffer_bytes);
  resources_.atomic_tail_base = static_cast<uint64_t>(atomic_base);
  resources_.num_scaleout_ranks = num_nodes;
  resources_.num_scaleup_ranks = cfg.local_world_size;
  resources_.scaleout_rank = node_idx;
  resources_.scaleup_rank = local_rank;
  resources_.num_lanes = static_cast<uint32_t>(kNumProxyThs);
}

void UcclGinContext::teardown() {
  for (auto& p : proxies_) {
    if (p->is_running()) p->stop();
  }
  proxies_.clear();
  if (d_handles_) {
    cudaFree(d_handles_);
    d_handles_ = nullptr;
  }
  // NOTE: window buffer is owned by NCCL-EP — we never free it
}

}  // namespace nccl_ep_adapter
