#pragma once

#include "../oob/oob.h"
#include "config.h"
#include "gpu_rt.h"
#include "request.h"
#include "transport_adapter.h"
#include "../util/jring.h"
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace Transport {

class Communicator;

static constexpr size_t kTaskRingSize = 1024;
static constexpr size_t kIpcSizePerEngine = 1ul << 20;
static constexpr int kIpcControlTimeoutMs = 50000;

class IpcChannel final : public TransportAdapter {
 public:
  explicit IpcChannel(Communicator* comm);
  ~IpcChannel() override;
  void shutdown();

  bool connect_to(int rank);
  bool accept_from(int rank);

  bool connect(int peer_rank) override { return connect_to(peer_rank); }
  bool accept(int peer_rank) override { return accept_from(peer_rank); }
  bool has_send_path(int peer_rank) const override;
  bool has_recv_path(int peer_rank) const override;

  unsigned send_async(int peer_rank, void* local_ptr, size_t len,
                      uint64_t local_mr_id,
                      std::optional<RemoteSlice> remote_hint,
                      BounceBufferProvider bounce_provider = nullptr) override;
  unsigned recv_async(int peer_rank, void* local_ptr, size_t len,
                      uint64_t local_mr_id,
                      BounceBufferProvider bounce_provider = nullptr) override;

  bool poll_completion(unsigned id) override;
  bool wait_completion(unsigned id) override;
  bool request_failed(unsigned id) override;
  void release_request(unsigned id) override;

  int peer_count() const override;

  bool send_async_ipc(int to_rank, std::shared_ptr<Request> creq,
                      void* bounce_ptr = nullptr, size_t bounce_len = 0,
                      std::string bounce_shm_name = "",
                      BounceBufferProvider bounce_provider = nullptr);
  bool recv_async_ipc(int from_rank, std::shared_ptr<Request> creq,
                      void* bounce_ptr = nullptr, size_t bounce_len = 0,
                      std::string bounce_shm_name = "");
  uint64_t next_match_seq(int rank, RequestType type);

 private:
  enum class IpcTaskType : uint8_t { SEND, RECV };

  struct IpcTask {
    IpcTaskType type;
    int peer_rank;
    std::shared_ptr<Request> req;
    void* bounce_ptr = nullptr;
    size_t bounce_len = 0;
    std::string bounce_shm_name;
    BounceBufferProvider bounce_provider = nullptr;
  };

  bool send_one(int to_rank, Request* creq, void* bounce_ptr, size_t bounce_len,
                std::string const& bounce_shm_name,
                BounceBufferProvider bounce_provider);
  bool recv_one(int from_rank, Request* creq, void* bounce_ptr,
                size_t bounce_len, std::string const& bounce_shm_name);
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

  std::mutex match_seq_mu_;
  // Two directed-edge counters per peer:
  // dir=0 -> low-rank to high-rank, dir=1 -> high-rank to low-rank.
  std::vector<std::array<uint64_t, 2>> next_match_seq_per_peer_;
  std::atomic<unsigned> next_request_id_{1};
  mutable std::mutex req_mu_;
  std::unordered_map<unsigned, std::shared_ptr<Request>> pending_requests_;

  Communicator* comm_;
};

}  // namespace Transport
}  // namespace UKernel
