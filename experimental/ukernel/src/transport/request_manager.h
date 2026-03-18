#pragma once

#include "request.h"
#include "transport_engine.h"
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace Transport {

class RequestManager {
 public:
  using CompletionCallback =
      std::function<void(unsigned, std::chrono::steady_clock::time_point)>;

  RequestManager();
  ~RequestManager() = default;

  unsigned allocate_request_id();

  void add_request(unsigned id, int peer_rank, PeerTransportKind kind,
                  std::shared_ptr<Request> req);

  bool poll_completion(unsigned id);
  bool wait_completion(unsigned id, bool blocking);
  bool wait_all(std::vector<unsigned> const& reqs);
  bool check_failed(unsigned id) const;

  void release(unsigned id);
  void clear();

  void set_completion_callback(CompletionCallback cb);
  void start_notifier_loop();
  void stop_notifier_loop();

  size_t pending_count() const;

 private:
  struct TrackedRequest {
    int peer_rank = -1;
    PeerTransportKind kind = PeerTransportKind::Ipc;
    std::shared_ptr<Request> ipc_request;
    bool completed = false;
    bool failed = false;
    bool notified = false;
  };

  bool poll_request_completion(unsigned id, bool blocking);
  void notifier_loop();

  std::unordered_map<unsigned, TrackedRequest> requests_;
  mutable std::mutex mu_;
  std::atomic<unsigned> next_request_id_{1};

  std::atomic<bool> notifier_running_{false};
  std::atomic<bool> notifier_started_{false};
  std::thread notifier_thread_;
  std::condition_variable notifier_cv_;
  std::mutex notifier_mu_;
  CompletionCallback callback_;
};

}  // namespace Transport
}  // namespace UKernel
