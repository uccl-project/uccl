#pragma once

#include "../../include/gpu_rt.h"
#include "../memory/ipc_manager.h"
#include "../util/jring.h"
#include "transport_adapter.h"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace UKernel {
namespace Transport {

class Communicator;

// Signal and DataPut share the same SHM region but use independent
// completion counters to prevent tag reuse or sequence interference
// across ping-pong iterations where user tags (e.g., 1, 2) alternate.
struct IpcDataCompletion {
  std::atomic<uint64_t> last_completed[2];  // [0] = dir 0, [1] = dir 1
  std::atomic<uint64_t> last_signal[2];     // independent signal counter
};

class IpcAdapter final : public TransportAdapter {
 public:
  IpcAdapter(Communicator* comm, std::string ring_namespace, int gpu_id);
  ~IpcAdapter() override;
  void shutdown();

  uint64_t next_send_match_seq(int peer);
  uint64_t next_recv_match_seq(int peer);
  uint64_t next_send_signal_seq(int peer);
  uint64_t next_recv_signal_seq(int peer);

  bool ensure_put_path(PeerConnectSpec const&) override;
  bool ensure_wait_path(PeerConnectSpec const&) override;
  bool has_put_path(int peer) const override;
  bool has_wait_path(int peer) const override;

  unsigned put_async(int peer, void* local_ptr, uint32_t local_buf,
                     void* remote_ptr, uint32_t remote_buf, size_t len,
                     unsigned comm_rid) override;
  unsigned signal_async(int peer, uint64_t tag, unsigned comm_rid) override;
  unsigned wait_async(int peer, uint64_t tag, std::optional<WaitTarget>,
                      unsigned comm_rid) override;

  void close_comp(int peer_rank);

 private:
  enum class ReqType : uint8_t { DataPut, DataWait, Signal, SignalWait };

  struct RingElem {
    unsigned comm_rid;
    int peer;
    ReqType type;
    uint64_t seq;
    void* local_ptr;
    void* remote_ptr;
    size_t bytes;
  };

  struct PeerComp {
    IpcDataCompletion* local = nullptr;
    IpcDataCompletion* remote = nullptr;
    int shm_fd = -1;
    size_t shm_size = 0;
    std::string shm_name;
  };

  void send_worker();
  void recv_worker();
  bool send_one(RingElem* e);
  bool recv_one(RingElem* e);

  bool connect_to(int rank);
  bool accept_from(int rank);

  std::string comp_shm_name(int peer) const;
  bool ensure_local_comp(int peer);
  bool ensure_remote_comp(int peer);

  jring_t* send_ring_ = nullptr;
  jring_t* recv_ring_ = nullptr;
  std::atomic<bool> stop_{false};
  std::thread send_th_;
  std::thread recv_th_;
  std::vector<std::pair<gpuStream_t, gpuEvent_t>> ipc_ctx_;

  std::mutex seq_mu_;
  std::vector<std::array<uint64_t, 2>> seqs_;  // [peer][0]=send, [1]=recv
  std::vector<std::array<uint64_t, 2>> signal_seqs_;  // [peer][0]=send, [1]=recv

  std::string ns_;
  mutable std::mutex dir_mu_;
  std::vector<std::pair<bool, bool>> dir_state_;  // {put_ready, wait_ready}
  std::vector<PeerComp> comps_;
  Communicator* comm_;
  int gpu_id_ = -1;
};
}  // namespace Transport
}  // namespace UKernel
