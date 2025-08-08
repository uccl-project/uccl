#pragma once

#include "peer_copy_worker.hpp"
#include <thread>
#include <vector>
class UcclProxy;

class PeerCopyManager {
 public:
  explicit PeerCopyManager(int src_device = 0);
  ~PeerCopyManager();

  void start_for_proxies(std::vector<UcclProxy*> const& proxies);
  void stop();

 private:
  PeerCopyShared shared_;
  std::vector<PeerWorkerCtx> ctxs_;
  std::vector<std::thread> threads_;
};
