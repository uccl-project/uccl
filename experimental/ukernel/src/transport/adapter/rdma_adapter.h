#pragma once

#include "transport_adapter.h"
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <infiniband/verbs.h>

namespace UKernel {
namespace Transport {

struct RdmaTransportConfig {
  int num_qps = 4;
  int chunk_size_kb = 512;
  int max_chunks = 16;
  int recv_wr_pool_per_qp = 128;
};

struct RdmaConnectInit {
  uint32_t qpns[4] = {};
  uint8_t num_qps = 4;
  uint16_t lid = 0;
  uint8_t gid_raw[16] = {};
  uint8_t gid_index = 0;
  uint64_t signal_addr = 0;
  uint32_t signal_rkey = 0;
  int dev_idx = -1;
  int gpu_idx = -1;
};

class RdmaTransportAdapter final : public TransportAdapter {
 public:
  RdmaTransportAdapter(int local_gpu_idx, RdmaTransportConfig config = {});
  ~RdmaTransportAdapter() override;

  bool is_initialized() const { return ctx_handle_ != nullptr; }
  RdmaConnectInit get_connect_init(int peer_rank);

  bool ensure_put_path(PeerConnectSpec const& spec) override;
  bool ensure_wait_path(PeerConnectSpec const& spec) override;
  bool has_put_path(int peer_rank) const override;
  bool has_wait_path(int peer_rank) const override;

  unsigned put_async(int peer_rank, void* local_ptr, uint32_t local_buffer_id,
                     void* remote_ptr, uint32_t remote_buffer_id,
                     size_t len) override;
  unsigned signal_async(int peer_rank, uint64_t tag) override;
  unsigned wait_async(int peer_rank, uint64_t expected_tag,
                      std::optional<WaitTarget> target = std::nullopt) override;

  bool poll_completion(unsigned id) override;
  bool wait_completion(unsigned id) override;
  bool request_failed(unsigned id) override;
  void release_request(unsigned id) override;

  bool register_memory(uint32_t buffer_id, void* ptr, size_t len);
  void deregister_memory(uint32_t buffer_id);
  bool is_memory_registered(uint32_t buffer_id) const;

  void register_remote_buffer(int peer_rank, uint32_t buffer_id,
                              uint64_t addr, uint32_t rkey);
  void set_remote_signal_info(int peer_rank, uint64_t addr, uint32_t rkey);

 private:
  static constexpr int kMaxQPs = 4;
  static constexpr uint32_t kMaxChunks = 16;
  static constexpr uint32_t kSignalImmBit = 0x80000000u;

  static constexpr uint32_t kCacheSizeThresh = 8192;
  static constexpr uint32_t kCacheConsecutiveThresh = 16384;
  static constexpr uint64_t kMaxInflightBytes =
      static_cast<uint64_t>(4) * 512 * 1024;

  struct RemoteBufInfo {
    uint64_t addr = 0;
    uint32_t rkey = 0;
  };

  struct ChunkResult {
    uint32_t count;
    uint32_t regular_size;
    uint32_t last_size;
  };

  struct RecvPool {
    std::vector<uint8_t> buffer;
    ibv_mr* mr = nullptr;
    std::vector<ibv_sge> sges;
    std::vector<ibv_recv_wr> wrs;
  };

  struct QpRttTracker {
    std::atomic<uint64_t> last_send_ns{0};
    double ewma_rtt_ns = 1'000'000.0;
  };

  struct RdmaPeer {
    ibv_qp* qps[kMaxQPs] = {};
    ibv_cq* cq = nullptr;
    uint8_t num_qps = kMaxQPs;
    bool put_ready = false;
    bool wait_ready = false;
    bool qps_in_init = false;

    uint32_t remote_qpns[kMaxQPs] = {};
    uint16_t remote_lid = 0;
    union ibv_gid remote_gid = {};
    uint8_t remote_gid_index = 0;

    uint64_t remote_signal_addr = 0;
    uint32_t remote_signal_rkey = 0;

    std::unordered_map<uint32_t, RemoteBufInfo> remote_buffers;

    std::unique_ptr<RecvPool> recv_pools[kMaxQPs];
    int recv_post_idx[kMaxQPs] = {};

    QpRttTracker rtt_tracker[kMaxQPs];

    std::atomic<int> last_qp_{0};
    std::atomic<bool> cached_qp_valid_{false};
    std::atomic<uint32_t> consecutive_cached_bytes_{0};

    std::atomic<uint64_t> unacked_bytes_{0};

    std::atomic<unsigned> pending_data_wait_id{0};
    uint32_t pending_data_chunks_total = 0;
    uint32_t pending_data_chunks_done = 0;

    std::atomic<unsigned> pending_signal_wait_id{0};
  };

  enum class RequestKind : uint8_t {
    DataPut = 0,
    DataWait = 1,
    Signal = 2,
    SignalWait = 3,
  };

  struct RequestSlot {
    std::atomic<uint8_t> state{0};
    std::atomic<uint32_t> generation{1};
    RequestKind kind = RequestKind::DataPut;
    int peer_rank = -1;
    uint64_t expected_tag = 0;
    std::atomic<bool> completed{false};
    std::atomic<bool> failed{false};
    uint32_t total_chunks = 0;
    std::atomic<uint32_t> completed_chunks{0};
  };

  static constexpr uint32_t kSlotBits = 13;
  static constexpr uint32_t kSlotCount = 1u << kSlotBits;
  static constexpr uint32_t kSlotMask = kSlotCount - 1u;

  static unsigned make_request_id(uint32_t idx, uint32_t gen) {
    return ((gen << kSlotBits) | idx);
  }
  static uint32_t slot_index(unsigned id) { return id & kSlotMask; }
  static uint32_t slot_generation(unsigned id) { return id >> kSlotBits; }

  static uint32_t imm_data_chunks(uint16_t total, uint16_t index) {
    return (static_cast<uint32_t>(total) << 16) | index;
  }
  static uint16_t imm_total(uint32_t imm) {
    return static_cast<uint16_t>(imm >> 16);
  }
  static uint32_t imm_signal(uint64_t tag) {
    return kSignalImmBit | static_cast<uint32_t>(tag & 0x7FFFFFFFULL);
  }
  static uint64_t imm_to_tag(uint32_t imm) {
    return imm & 0x7FFFFFFFULL;
  }

  static uint64_t now_ns();

  RequestSlot* acquire_slot(unsigned* out_id);
  RequestSlot* resolve_slot(unsigned id);
  const RequestSlot* resolve_const(unsigned id) const;
  void free_slot(unsigned id);

  int select_qp(RdmaPeer& p, uint32_t msize);
  int find_qp_idx(RdmaPeer& p, uint32_t qp_num);

  bool do_connect(int peer_rank, RdmaPeerConnectSpec const& remote);
  bool do_accept(int peer_rank, RdmaPeerConnectSpec const& remote);
  bool init_qps_to_init(RdmaPeer& peer);
  bool qps_to_rtr(RdmaPeer& peer, RdmaPeerConnectSpec const& remote);
  bool qps_to_rts(RdmaPeer& peer);
  bool init_recv_pools(RdmaPeer& peer);
  void destroy_peer_qps(RdmaPeer& peer);
  bool repost_one_recv(RdmaPeer& peer, int qp_idx);

  ChunkResult chunk_split(size_t len) const;

  void poll_loop();
  bool poll_cq(RdmaPeer& peer, int rank);

  ibv_context* ctx_ = nullptr;
  ibv_pd* pd_ = nullptr;
  ibv_device_attr dev_attr_ = {};
  uint16_t lid_ = 0;
  union ibv_gid gid_ = {};
  uint8_t gid_index_ = 0;

  std::shared_ptr<ibv_context> ctx_handle_;
  uint64_t signal_addr_ = 0;
  uint32_t signal_rkey_ = 0;
  ibv_mr* signal_mr_ = nullptr;

  int local_gpu_idx_ = -1;
  int local_dev_idx_ = -1;
  RdmaTransportConfig config_;

  mutable std::mutex mu_;
  std::unordered_map<int, std::unique_ptr<RdmaPeer>> peers_;
  std::unordered_map<uint32_t, ibv_mr*> mr_map_;

  std::unique_ptr<RequestSlot[]> slots_;
  std::atomic<uint32_t> cursor_{0};

  std::atomic<bool> stop_{false};
  std::thread poll_thread_;
  std::mutex cv_mu_;
  std::condition_variable cv_;
};

}  // namespace Transport
}  // namespace UKernel
