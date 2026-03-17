#pragma once

#include "config.h"
#include "gpu_rt.h"
#include "oob.h"
#include "request.h"
#include "util/jring.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace UKernel {
namespace Transport {

class Communicator;

static constexpr size_t kTaskRingSize = 1024;
static constexpr size_t kIpcSizePerEngine = 1ul << 20;

class IpcChannel {
 public:
  explicit IpcChannel(Communicator* comm);
  ~IpcChannel();

  bool connect_to(int rank);
  bool accept_from(int rank);
  bool send_async(int to_rank, std::shared_ptr<Request> creq);
  bool recv_async(int from_rank, std::shared_ptr<Request> creq);

 private:
  enum class IpcTaskType : uint8_t { SEND, RECV };

  struct IpcTask {
    IpcTaskType type;
    int peer_rank;
    std::shared_ptr<Request> req;
  };

  bool send_(int to_rank, std::shared_ptr<Request> creq);
  bool recv_(int from_rank, std::shared_ptr<Request> creq);
  void proxy_thread_func();

  jring_t* task_ring_;
  std::atomic<bool> stop_{false};
  std::thread proxy_thread_;
  std::mutex cv_mu_;
  std::condition_variable cv_;
  std::atomic<int> pending_{0};
  std::vector<gpuStream_t> ipc_streams_;

  Communicator* comm_;
};

}  // namespace Transport
}  // namespace UKernel
