#pragma once

#include "transport_adapter.h"
#include <infiniband/verbs.h>
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

struct RdmaTransportConfig {
  int num_qps = 4;
  int chunk_size_kb = 512;
};

class RdmaTransportAdapter final : public TransportAdapter {
 public:
  RdmaTransportAdapter(int local_gpu_idx, RdmaTransportConfig config = {});
  ~RdmaTransportAdapter() override;

  bool is_initialized() const { return ctx_handle_ != nullptr; }
  RdmaPeerConnectSpec get_connect_init(int peer_rank);

  bool ensure_put_path(PeerConnectSpec const& spec) override;
  bool ensure_wait_path(PeerConnectSpec const& spec) override;
  bool has_put_path(int peer_rank) const override;
  bool has_wait_path(int peer_rank) const override;

  unsigned put_async(int peer_rank, void* local_ptr, uint32_t local_buffer_id,
                     void* remote_ptr, uint32_t remote_buffer_id, size_t len,
                     unsigned comm_rid) override;
  unsigned signal_async(int peer_rank, uint64_t tag,
                        unsigned comm_rid) override;
  unsigned wait_async(int peer_rank, uint64_t expected_tag,
                      std::optional<WaitTarget> target,
                      unsigned comm_rid) override;


  bool register_memory(uint32_t buffer_id, void* ptr, size_t len);
  void deregister_memory(uint32_t buffer_id);
  bool is_memory_registered(uint32_t buffer_id) const;
  uint32_t get_memory_rkey(uint32_t buffer_id) const;

  void register_remote_buffer(int peer_rank, uint32_t buffer_id, uint64_t addr,
                              uint32_t rkey);

 private:
  static constexpr int kMaxQPs = 4;
  static constexpr uint32_t kMaxChunks = 128;  // BDP: 64 MB
  static constexpr int kMaxMsgId = 128;
  static constexpr int kMsgIdMask = kMaxMsgId - 1;

  static constexpr uint32_t kCacheSizeThresh = 8192;
  static constexpr uint32_t kCacheConsecutiveThresh = 16384;
  static constexpr int kMaxInflightWrs = 128;

  enum class Kind : uint8_t { DataPut, Signal, SignalWait };

  struct RingElem {
    unsigned comm_rid;
    int peer_rank;
    Kind kind;
    void* local_ptr;
    void* remote_ptr;
    uint32_t local_buf_id;
    uint32_t remote_buf_id;
    size_t len;
    uint64_t tag;          // for signal / signal wait
    uint64_t remote_addr;  // resolved RDMA remote address
    uint32_t remote_rkey;  // resolved RDMA remote rkey
    uint32_t local_lkey;   // local MR lkey for RDMA WR posting
  };

  struct RemoteBufInfo {
    uint64_t addr = 0;
    uint32_t rkey = 0;
  };

  struct ChunkResult {
    uint32_t count;
    uint32_t chunk_size;
    uint32_t last_size;
  };

  struct RecvPool {
    std::vector<uint8_t> buffer;
    ibv_mr* mr = nullptr;
    std::vector<ibv_sge> sges;
    std::vector<ibv_recv_wr> wrs;
  };

  struct QpState {
    std::atomic<uint64_t> last_send_ns{0};
    double ewma_rtt_ns = 1'000'000.0;
    std::atomic<uint32_t> unacked_wrs{0};
  };

  struct ChunkTracker {
    std::atomic<bool> completed{false};
    std::atomic<unsigned> wait_comm_rid{0};
  };

  struct PendingSend {
    unsigned comm_rid;
    uint32_t total_chunks;
    std::atomic<uint32_t> completed_chunks{0};

    PendingSend() = default;
    PendingSend(unsigned rid, uint32_t tc)
        : comm_rid(rid), total_chunks(tc), completed_chunks(0) {}
  };

  struct RdmaPeer {
    ibv_qp* data_qps[kMaxQPs] = {};
    ibv_cq* data_cq = nullptr;

    uint32_t remote_data_qpns[kMaxQPs] = {};

    ibv_qp* signal_qp = nullptr;
    ibv_cq* signal_cq = nullptr;
    uint32_t remote_signal_qpn = 0;
    std::unique_ptr<RecvPool> signal_pool;

    uint8_t num_qps = kMaxQPs;
    bool put_ready = false;
    bool wait_ready = false;
    bool qps_created = false;

    uint16_t remote_lid = 0;
    union ibv_gid remote_gid = {};

    std::unordered_map<uint32_t, RemoteBufInfo> remote_buffers;

    QpState qp_state[kMaxQPs];

    std::atomic<int> last_qp_{0};
    std::atomic<bool> cached_qp_valid_{false};
    std::atomic<uint32_t> consecutive_cached_bytes_{0};

    std::atomic<unsigned> next_msg_id{0};

    ChunkTracker trackers[kMaxMsgId];
    std::atomic<uint32_t> next_expected_dispatch{0};
    uint32_t dispatch_cursor = 0;
  };

  static uint64_t now_ns();

  int select_qp(RdmaPeer& p, uint32_t msize);
  int find_qp_idx(ibv_qp* const* qps, int count, uint32_t qp_num);
  void check_dispatch(RdmaPeer& p, int rank);

  bool create_qp_set(ibv_qp** qps, ibv_cq** cq, int count, int cq_size,
                     int max_recv_wr = 1);
  bool qps_to_init(ibv_qp** qps, int count);
  bool qps_to_rtr(ibv_qp** qps, int count, RdmaPeerConnectSpec const& remote);
  bool qps_to_rts(ibv_qp** qps, int count);

  bool setup_peer_path(int peer_rank, RdmaPeerConnectSpec const& remote);
  bool init_peer_qps(RdmaPeer& peer);
  void destroy_peer_qps(RdmaPeer& peer);

  bool init_signal_pool(RdmaPeer& p);
  bool repost_signal_recv(RdmaPeer& p);

  ChunkResult chunk_split(size_t len) const;

  void send_worker();
  void recv_worker();
  void poll_loop();
  bool poll_cq_set(RdmaPeer& peer, int rank, ibv_cq* cq, ibv_qp* const* qps,
                   int qp_count);
  bool poll_signal_cq(RdmaPeer& peer, int rank);

  ibv_context* ctx_ = nullptr;
  ibv_pd* pd_ = nullptr;
  ibv_device_attr dev_attr_ = {};
  uint16_t lid_ = 0;
  union ibv_gid gid_ = {};
  uint8_t gid_index_ = 0;

  std::shared_ptr<ibv_context> ctx_handle_;

  int local_gpu_idx_ = -1;
  int local_dev_idx_ = -1;
  RdmaTransportConfig config_;

  mutable std::mutex mu_;
  std::unordered_map<int, std::unique_ptr<RdmaPeer>> peers_;
  std::unordered_map<uint32_t, ibv_mr*> mr_map_;

  // Jring-based task queues
  jring_t* send_ring_ = nullptr;
  jring_t* recv_ring_ = nullptr;

  // Worker threads
  std::atomic<bool> stop_{false};
  std::thread send_worker_;
  std::thread recv_worker_;
  std::thread poll_thread_;

  // send_id → PendingSend mapping (for CQ completion lookup)
  std::atomic<uint64_t> next_send_id_{0};
  std::mutex pending_mu_;
  std::unordered_map<uint32_t, PendingSend> pending_sends_;

  // Condition variable for back-pressure
  std::mutex cv_mu_;
  std::condition_variable cv_;
};

}  // namespace Transport
}  // namespace UKernel
