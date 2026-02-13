#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#if defined(UCCL_P2P_USE_RCCL)
#include <rccl/rccl.h>
#else
#include <nccl.h>
#endif

#include "include/common.h"

namespace uccl {

using FlowID = uint64_t;
using PeerID = uint64_t;

struct ConnID {
  void* context;
  int sock_fd;
  FlowID flow_id;
  PeerID peer_id;
  int dev;
};

struct Mhandle;
class UcclFlow;

struct FifoItem {
  uint64_t addr;
  uint32_t size;
  uint32_t rkey;
  uint32_t nmsgs;
  uint32_t rid;
  uint64_t idx;
  uint32_t engine_offset;
  char padding[28];
};
static_assert(sizeof(struct FifoItem) == 64, "FifoItem must be 64 bytes");

enum ReqType { ReqTx, ReqRx, ReqRead, ReqWrite };

struct ucclRequest {
  enum ReqType type;
  uint32_t n;
  void* context;
  uint32_t engine_idx;
};

}  // namespace uccl

class EpollClient;

namespace tcp {

int get_tcp_numa_node_from_iface();

// Placeholder for RDMA-style MR arrays. TCP does not register memory, but the
// unified interface expects the type.
struct MRArray {
  void* dummy = nullptr;
};

class TCPEndpoint {
 public:
  // NCCL-over-TCP endpoint: TCP control-plane + NCCL send/recv data-plane.
  explicit TCPEndpoint(int gpu_index, uint16_t port = 0);
  ~TCPEndpoint();

  // GPU index selected by the engine; may be overridden by uccl_connect/accept.
  int gpuIndex() const { return gpu_index_; }

  // RDMA endpoint exposes these for OOB metadata exchange; TCP path doesn't
  // use EpollClient but we keep stubs so engine code can compile unchanged.
  std::shared_ptr<EpollClient> get_oob_client() const { return nullptr; }
  std::string get_oob_conn_key(uint64_t rank_id) const {
    (void)rank_id;
    return "";
  }

  // Establish a TCP control connection, exchange NCCL IDs, and start a control
  // thread for one-sided read/write requests.
  uccl::ConnID uccl_connect(int dev, int local_gpuidx, int remote_dev,
                            int remote_gpuidx, std::string remote_ip,
                            uint16_t remote_port);
  // Listen socket metadata. dev is unused for TCP but required by the API.
  uint16_t get_p2p_listen_port(int dev) { return listen_port_; }
  int get_p2p_listen_fd(int dev) { return listen_fd_; }
  // Accept a TCP control connection and initialize NCCL communicators.
  uccl::ConnID uccl_accept(int dev, int listen_fd, int local_gpuidx,
                           std::string& remote_ip, int* remote_dev,
                           int* remote_gpuidx);

  // Memory registration is a no-op for TCP; we keep the interface.
  int uccl_regmr(uccl::UcclFlow* flow, void* data, size_t len, int type,
                 struct uccl::Mhandle** mhandle);
  int uccl_regmr(void* data, size_t len, MRArray& mr_array);
  int uccl_regmr(int dev, void* data, size_t len, int type,
                 struct uccl::Mhandle** mhandle);
  void uccl_deregmr(struct uccl::Mhandle* mhandle);
  void uccl_deregmr(MRArray const& mr_array);

  // Two-sided NCCL send/recv (async) used by the engine.
  int uccl_send_async(uccl::UcclFlow* flow, struct uccl::Mhandle* mh,
                      void const* data, size_t size,
                      struct uccl::ucclRequest* ureq);
  int uccl_recv_async(uccl::UcclFlow* flow, struct uccl::Mhandle** mhandles,
                      void** data, int* sizes, int n,
                      struct uccl::ucclRequest* ureq);
  // One-sided semantics built on a control message + NCCL send/recv.
  int uccl_read_async(uccl::UcclFlow* flow, struct uccl::Mhandle* mh, void* dst,
                      size_t size, uccl::FifoItem const& slot_item,
                      uccl::ucclRequest* ureq);
  int uccl_write_async(uccl::UcclFlow* flow, struct uccl::Mhandle* mh,
                       void* src, size_t size, uccl::FifoItem const& slot_item,
                       uccl::ucclRequest* ureq);
  // Poll a ucclRequest created by send/recv/read/write.
  bool uccl_poll_ureq_once(struct uccl::ucclRequest* ureq);
  // Serialize a FIFO descriptor (addr/size) for control-plane exchange.
  int prepare_fifo_metadata(uccl::UcclFlow* flow,
                            struct uccl::Mhandle** mhandle, void const* data,
                            size_t size, char* out_buf);

  // TCP has no device selection or unified socket; keep these as no-ops.
  int get_best_dev_idx(int gpu_idx) { return 0; }

  // Get the socket file descriptor for a connection.
  int get_sock_fd(uint64_t flow_id);

  // Send a notification message to a peer (uses NotifyMsg from common.h)
  int send_notification(uint64_t flow_id, struct ::NotifyMsg const& notification);

  bool initialize_engine_by_dev(int dev, bool enable_p2p_listen) {
    if (dev >= 0) {
      gpu_index_ = dev;
    }
    (void)enable_p2p_listen;
    return true;
  }

  void create_unified_p2p_socket() {}
  
  void stop_accept() {}


 private:
  struct Conn;
  struct AsyncHandle;

  // Control-plane helpers.
  bool setup_listener_(uint16_t port);
  bool send_all_(int fd, void const* buf, size_t len) const;
  bool recv_all_(int fd, void* buf, size_t len) const;
  // NCCL communicator setup. Two communicators split send/recv directions.
  bool init_comm_(Conn& conn, ncclUniqueId const& uid, int comm_index);
  bool init_comms_(Conn& conn, ncclUniqueId const& uid_rank0,
                   ncclUniqueId const& uid_rank1);
  // Per-connection control thread to service one-sided requests.
  void control_loop_(Conn* conn);
  int comm_index_for_send_(Conn const& conn) const;
  int comm_index_for_recv_(Conn const& conn) const;
  // Internal NCCL send/recv with completion events in ucclRequest.
  bool send_internal_(Conn& conn, void const* data, size_t size, int comm_index,
                      uccl::ucclRequest* ureq);
  bool recv_internal_(Conn& conn, void* data, size_t size, int comm_index,
                      uccl::ucclRequest* ureq);
  void cleanup_conn_(Conn& conn);

  // GPU index the endpoint binds to by default.
  int gpu_index_;
  // Control-plane listening socket.
  uint16_t listen_port_;
  int listen_fd_;
  std::atomic<uint64_t> next_flow_id_{1};
  std::mutex conn_mu_;
  std::unordered_map<uint64_t, std::unique_ptr<Conn>> conn_map_;
};

}  // namespace tcp
