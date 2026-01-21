// cq_poller.h
#pragma once
#include <infiniband/verbs.h>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

namespace UKernel {
namespace Transport {

class Communicator;

class CQPoller {
 public:
  CQPoller(Communicator* comm, ibv_cq* assign_cq, int poll_batch = 16);
  ~CQPoller();

  void start();
  void stop();
  bool running() const { return running_.load(std::memory_order_acquire); }

  void process_pending();

 private:
  void run_loop();

  Communicator* comm_;
  ibv_cq* cq_;
  std::thread thr_;
  std::atomic<bool> running_{false};
  int poll_batch_;
};

}  // namespace Transport
}  // namespace UKernel
