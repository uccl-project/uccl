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
static constexpr int kIpcControlTimeoutMs = 50000;

class IpcChannel {
 public:
  explicit IpcChannel(Communicator* comm);
  ~IpcChannel();
  void shutdown();

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

  bool send_one(int to_rank, Request* creq);
  bool recv_one(int from_rank, Request* creq);
  void send_thread_func();
  void recv_thread_func();
  void complete_task(std::shared_ptr<Request> const& req, bool ok);

  jring_t* send_task_ring_;
  jring_t* recv_task_ring_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> shutdown_started_{false};
  std::thread send_thread_;
  std::thread recv_thread_;
  std::mutex cv_mu_;
  std::condition_variable cv_;
  std::atomic<int> pending_send_{0};
  std::atomic<int> pending_recv_{0};
  std::vector<gpuStream_t> ipc_streams_;

  Communicator* comm_;
};

}  // namespace Transport
}  // namespace UKernel
