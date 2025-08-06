#pragma once

#include <thread>
#include <vector>
#include "peer_copy_worker.hpp"
class UcclProxy;

class PeerCopyManager {
 public:
  explicit PeerCopyManager(int src_device = 0);
  ~PeerCopyManager();

  void start_for_proxies(const std::vector<UcclProxy*>& proxies);
  void stop();

 private:
  PeerCopyShared shared_;
  std::vector<PeerWorkerCtx> ctxs_;
  std::vector<std::thread> threads_;
};
