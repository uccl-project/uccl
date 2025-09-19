#include "peer_copy_manager.hpp"
#include "uccl_proxy.hpp"

PeerCopyManager::PeerCopyManager(int src_device) {
  shared_.src_device = src_device;
  shared_.run.store(true, std::memory_order_release);
}
PeerCopyManager::~PeerCopyManager() { stop(); }

void PeerCopyManager::start_for_proxies(
    std::vector<UcclProxy*> const& proxies) {
  int const n = static_cast<int>(proxies.size());
  if (n <= 0) return;
  ctxs_.resize(n);
  threads_.reserve(n);
  for (int i = 0; i < n; ++i) {
    threads_.emplace_back(peer_copy_worker, std::ref(shared_),
                          std::ref(ctxs_[i]),
                          std::ref(proxies[i]->proxy_->ring), i);
  }
}

void PeerCopyManager::stop() {
  if (threads_.empty()) return;
  shared_.run.store(false, std::memory_order_release);
  for (auto& t : threads_)
    if (t.joinable()) t.join();
  threads_.clear();
  ctxs_.clear();
}
