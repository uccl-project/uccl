#pragma once

#include "common.h"
#include "epoll_client.h"
#include "epoll_server.h"
#include "util/debug.h"
#include "util/gpu_rt.h"
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Between nodes in different switch groups, a CXI flow can be capped at about
// 18 GB/s per target endpoint (seen on some node pairs). Each CxiEndpoint
// therefore opens several endpoints ("lanes") on its one domain, and a large
// RMA is split over the peer's lanes. CXI provider MR keys use all 64 bits, so
// the FIFO metadata (32 bytes) has room for 2 lane keys.
constexpr int kCxiMaxLanes = 2;

struct CxiMemoryRegion {
  void* addr = nullptr;
  size_t len = 0;
  // One MR per local lane: with FI_MR_ENDPOINT, an MR binds to one endpoint.
  std::vector<fid_mr*> mrs;
  std::vector<uint64_t> keys;
};

struct CxiFifoMetadata {
  uint64_t base = 0;
  uint64_t len = 0;
  // keys[i] is the MR key on the owner's lane i. The lane count comes from
  // the handshake; unused entries are 0.
  uint64_t keys[kCxiMaxLanes] = {};
};

static_assert(sizeof(CxiFifoMetadata) <= sizeof(FifoItem::padding),
              "CXI FIFO metadata must fit in FifoItem padding");

struct UcclRequest;

class CxiEndpoint {
 public:
  explicit CxiEndpoint(int gpu_index = INVALID_GPU, uint64_t port = 0);
  ~CxiEndpoint();

  bool initialize_rdma_ctx_for_gpu(
      int gpu_index,
      std::vector<size_t> const& device_ids = std::vector<size_t>());
  void create_unified_p2p_socket() {}

  int gpu_index() const { return gpu_index_; }
  // NUMA node of the CXI NIC this endpoint opened, or -1 if it is unknown.
  int nic_numa_node() const;
  // True if remote accesses to this endpoint's MRs make events that need
  // progress here (MR match events, set by FI_CXI_MR_MATCH_EVENTS).
  bool needs_target_progress() const { return mr_match_events_; }
  uint16_t get_p2p_listen_port();
  int get_p2p_listen_fd();
  std::shared_ptr<EpollClient> get_oob_client();
  std::string get_oob_conn_key(uint64_t peer_id) const;

  ConnID uccl_connect(int remote_gpuidx, std::string remote_ip,
                      uint16_t remote_port);
  ConnID uccl_accept(std::string& remote_ip, int* remote_gpuidx);
  void stop_accept();
  bool accept_stopped() const {
    return stop_accept_.load(std::memory_order_acquire);
  }

  int uccl_regmr(void* data, size_t len,
                 std::shared_ptr<CxiMemoryRegion>& region);
  void uccl_deregmr(std::shared_ptr<CxiMemoryRegion> const& region);

  int uccl_read_async(ConnID const& conn,
                      std::shared_ptr<CxiMemoryRegion> const& local_mr,
                      void* dst, size_t size, FifoItem const& fifo_item,
                      UcclRequest* ureq);
  int uccl_write_async(ConnID const& conn,
                       std::shared_ptr<CxiMemoryRegion> const& local_mr,
                       void* src, size_t size, FifoItem const& fifo_item,
                       UcclRequest* ureq);

  bool check_send_complete_once(uint64_t peer_id, int64_t request_id);
  void send_routine();
  void recv_routine();
  void flush_all_sends() {}

  int send_notification(uint64_t peer_id, NotifyMsg const& notification) const;

 private:
  struct Peer {
    std::vector<fi_addr_t> addrs;  // one per peer lane
    std::string ip;
    uint16_t port = 0;
    int gpu_index = -1;
  };

  // One per request. A striped request posts its pieces to the peer lanes in
  // turn; each piece has its own fi_context2, and the request is done when
  // all pieces have completed.
  struct OpContext {
    std::vector<fi_context2> ctxs;  // sized before posting, never resized
    std::vector<int> lanes;         // remote lane of each piece
    int remaining = 0;
    int64_t id = -1;
    bool done = false;
    bool failed = false;
    int err = 0;
    int prov_errno = 0;
    std::string error_message;
  };

  struct EndpointInfo {
    std::vector<std::vector<uint8_t>> ep_names;  // one per lane
    uint16_t oob_port = 0;
    int gpu_index = -1;
  };

  void init_fabric(int gpu_index);
  EndpointInfo local_endpoint_info() const;
  std::vector<fi_addr_t> insert_peer(EndpointInfo const& info);
  void process_meta(std::string const& input, std::string& output,
                    std::string const& client_ip, int client_port);
  std::string serialize_endpoint_info(EndpointInfo const& info) const;
  EndpointInfo deserialize_endpoint_info(std::string const& bytes) const;
  int post_rma(bool is_read, ConnID const& conn,
               std::shared_ptr<CxiMemoryRegion> const& local_mr, void* local,
               size_t size, FifoItem const& fifo_item, UcclRequest* ureq);
  void poll_cq();
  void poll_cq_locked();

  int gpu_index_ = INVALID_GPU;
  bool fabric_initialized_ = false;
  bool mr_match_events_ = true;  // true until the domain says otherwise
  std::atomic<bool> stop_accept_{false};
  std::atomic<int32_t> next_send_peer_id_{0};
  std::atomic<int32_t> next_recv_peer_id_{0};
  std::atomic<int64_t> next_request_id_{0};

  fi_info* info_ = nullptr;
  fid_fabric* fabric_ = nullptr;
  fid_domain* domain_ = nullptr;
  std::vector<fid_ep*> eps_;    // lanes; RMA is posted from lane 0
  size_t max_piece_bytes_ = 0;  // max_msg_size, rounded down to 4 KiB
  fid_cq* cq_ = nullptr;
  fid_av* av_ = nullptr;

  std::shared_ptr<EpollClient> oob_client_;
  std::shared_ptr<EpollServer> oob_server_;

  mutable std::shared_mutex peer_mutex_;
  std::unordered_map<uint64_t, Peer> peers_;
  mutable std::shared_mutex peer_oob_conn_keys_mutex_;
  std::unordered_map<uint64_t, std::string> peer_oob_conn_keys_;

  mutable std::shared_mutex accepted_meta_mutex_;
  std::unordered_map<uint64_t, AcceptedMeta> accepted_meta_;

  mutable std::mutex fabric_mutex_;
  std::mutex op_mutex_;
  std::unordered_map<int64_t, std::unique_ptr<OpContext>> inflight_ops_;
};

void encode_cxi_fifo_metadata(CxiMemoryRegion const& region, FifoItem& item);
bool decode_cxi_fifo_metadata(FifoItem const& item, CxiFifoMetadata& metadata);
