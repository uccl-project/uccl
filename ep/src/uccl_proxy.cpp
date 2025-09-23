#include "uccl_proxy.hpp"
#include <cstdio>
#include <stdexcept>

UcclProxy::UcclProxy(uintptr_t rb_addr, int thread_idx,
                     uintptr_t gpu_buffer_addr, size_t total_size, int rank,
                     int node_idx, int local_rank, std::string const& peer_ip,
                     int num_experts, int num_ranks)
    : peer_ip_{peer_ip}, thread_{}, mode_{Mode::None}, running_{false} {
  if (peer_ip.empty()) {
    printf("Intranode mode. UcclProxy returns\n");
    return;
  }

  // Allocate multiple ring buffers for this proxy
  ring_buffer_addrs_.reserve(kRingsPerProxy);
  printf("Allocating %zu ring buffers for thread %d\n", kRingsPerProxy,
         thread_idx);

  for (size_t i = 0; i < kRingsPerProxy; ++i) {
    uintptr_t ring_addr = alloc_cmd_ring();
    ring_buffer_addrs_.push_back(ring_addr);
  }

  Proxy::Config cfg;
  thread_idx_ = thread_idx;
  gpu_buffer_addr_ = reinterpret_cast<void*>(gpu_buffer_addr);

  // Convert addresses to DeviceToHostCmdBuffer pointers
  cfg.ring_buffers.reserve(kRingsPerProxy);
  for (auto addr : ring_buffer_addrs_) {
    cfg.ring_buffers.push_back(reinterpret_cast<DeviceToHostCmdBuffer*>(addr));
  }

  cfg.thread_idx = thread_idx;
  cfg.gpu_buffer = reinterpret_cast<void*>(gpu_buffer_addr);
  cfg.total_size = total_size;
  cfg.rank = rank;
  cfg.local_rank = local_rank;
  cfg.peer_ip = peer_ip_.empty() ? nullptr : peer_ip_.c_str();
  cfg.num_experts = num_experts;
  cfg.num_ranks = num_ranks;
  proxy_ = std::make_unique<Proxy>(cfg);
  local_rank_ = local_rank;
  node_idx_ = node_idx;

  if (thread_idx == 0) {
    // size_t atomic_buffer_bytes = 2 * align<size_t>(num_experts * sizeof(int),
    // 128);
    // TODO(MaoZiming)
#ifdef USE_GRACE_HOPPER
    cudaMallocManaged(&atomic_buffer_ptr_, kAtomicBufferSize);
#else
    cudaHostAlloc(&atomic_buffer_ptr_, kAtomicBufferSize,
                  cudaHostAllocMapped | cudaHostAllocWriteCombined);
#endif
    cudaMemset(atomic_buffer_ptr_, 0, kAtomicBufferSize);
    proxy_->set_atomic_buffer_ptr(atomic_buffer_ptr_);
  }
}

UcclProxy::~UcclProxy() {
  try {
    stop();
  } catch (...) {
  }

  // Free all allocated ring buffers
  for (auto ring_addr : ring_buffer_addrs_) {
    free_cmd_ring(ring_addr);
  }
  ring_buffer_addrs_.clear();
}

std::vector<uint64_t> UcclProxy::get_ring_buffer_addrs() const {
  std::vector<uint64_t> addrs;
  addrs.reserve(ring_buffer_addrs_.size());
  for (auto addr : ring_buffer_addrs_) {
    addrs.push_back(static_cast<uint64_t>(addr));
  }
  return addrs;
}

void UcclProxy::set_peers_meta(std::vector<PeerMeta> const& peers) {
  peers_ = peers;
  proxy_->set_peers_meta(peers);
}

void UcclProxy::start_sender() { start(Mode::Sender); }
void UcclProxy::start_remote() { start(Mode::Remote); }
void UcclProxy::start_local() { start(Mode::Local); }
void UcclProxy::start_dual() { start(Mode::Dual); }

void UcclProxy::stop() {
  if (!running_.load(std::memory_order_acquire)) {
    throw std::runtime_error("Proxy already stopped");
  }
  proxy_->set_progress_run(false);
  if (thread_.joinable()) thread_.join();
  running_.store(false, std::memory_order_release);
  // Because proxies share the gpu_buffer, only destroy gpu_buffer for the first
  // proxy.
  proxy_->destroy(thread_idx_ == 0);
}

void UcclProxy::start(Mode m) {
  if (running_.load(std::memory_order_acquire)) {
    throw std::runtime_error("Proxy already running");
  }
  mode_ = m;
  proxy_->set_progress_run(true);
  running_.store(true, std::memory_order_release);

  thread_ = std::thread([this]() {
    if (peer_ip_.empty()) {
      std::printf("UcclProxy: no peer IP set, running in local mode\n");
      proxy_->run_local();
      return;
    }
    switch (mode_) {
      case Mode::Sender:
        proxy_->run_sender();
        break;
      case Mode::Remote:
        proxy_->run_remote();
        break;
      case Mode::Local:
        proxy_->run_local();
        break;
      case Mode::Dual:
        proxy_->run_dual();
        break;
      default:
        break;
    }
  });
}