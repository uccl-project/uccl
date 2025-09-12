#include "uccl_proxy.hpp"
#include <cstdio>
#include <stdexcept>

UcclProxy::UcclProxy(uintptr_t rb_addr, int block_idx,
                     uintptr_t gpu_buffer_addr, size_t total_size, int rank,
                     std::string const& peer_ip)
    : peer_ip_storage_{peer_ip}, thread_{}, mode_{Mode::None}, running_{false} {
  Proxy::Config cfg;
  // cfg.rb = reinterpret_cast<DeviceToHostCmdBuffer*>(rb_addr);
  rb_ = rb_addr;
  block_idx_ = block_idx;
  gpu_buffer_addr_ = reinterpret_cast<void*>(gpu_buffer_addr);
  cfg.rb = reinterpret_cast<DeviceToHostCmdBuffer*>(rb_);
  cfg.block_idx = block_idx;
  cfg.gpu_buffer = reinterpret_cast<void*>(gpu_buffer_addr);
  cfg.total_size = total_size;
  cfg.rank = rank;
  cfg.peer_ip = peer_ip_storage_.empty() ? nullptr : peer_ip_storage_.c_str();
  proxy_ = std::make_unique<Proxy>(cfg);
  local_rank_ = rank;

  if (block_idx == 0) {
    // size_t atomic_buffer_bytes = 2 * align<size_t>(num_experts * sizeof(int),
    // 128);
    // TODO(MaoZiming)
    cudaHostAlloc(&atomic_buffer_ptr_, kAtomicBufferSize,
                  cudaHostAllocMapped | cudaHostAllocWriteCombined);
    cudaMemset(atomic_buffer_ptr_, 0, kAtomicBufferSize);
    proxy_->set_atomic_buffer_ptr(atomic_buffer_ptr_);
  }
}

UcclProxy::~UcclProxy() {
  try {
    stop();
  } catch (...) {
  }
}

void UcclProxy::set_peers_meta(std::vector<PeerMeta> const& peers) {
  peers_ = peers;
  proxy_->set_peers_meta(peers);
}

void UcclProxy::start_sender() {
  start(Mode::Sender);
  std::printf("UcclProxy started as Sender\n");
}
void UcclProxy::start_remote() {
  start(Mode::Remote);
  std::printf("UcclProxy started as Remote\n");
}
void UcclProxy::start_local() {
  start(Mode::Local);
  std::printf("UcclProxy started as Local\n");
}
void UcclProxy::start_dual() {
  start(Mode::Dual);
  std::printf("UcclProxy started as Dual\n");
}

void UcclProxy::stop() {
  if (!running_.load(std::memory_order_acquire)) return;
  proxy_->set_progress_run(false);
  if (thread_.joinable()) thread_.join();
  running_.store(false, std::memory_order_release);
  // Because proxies share the gpu_buffer, only destroy gpu_buffer for the first
  // proxy.
  std::printf("UcclProxy destroying\n");
  proxy_->destroy(block_idx_ == 0);
  std::printf("UcclProxy destroyed\n");
}

void UcclProxy::start(Mode m) {
  if (running_.load(std::memory_order_acquire)) {
    throw std::runtime_error("Proxy already running");
  }
  mode_ = m;
  proxy_->set_progress_run(true);
  running_.store(true, std::memory_order_release);

  thread_ = std::thread([this]() {
    if (peer_ip_storage_.empty()) {
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