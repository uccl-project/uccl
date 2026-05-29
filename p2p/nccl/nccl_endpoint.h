#pragma once

#include "common.h"
#include "nccl_types.h"
#include "util/gpu_rt.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

using NcclPeerID = uint64_t;

struct NcclConnID {
  void* context;
  int sock_fd;
  NcclPeerID peer_id;
  int dev;
};

struct NcclMhandle;
class NcclFlow;

struct NcclFifoItem {
  uint64_t addr;
  uint32_t size;
  uint32_t rkey;
  uint32_t nmsgs;
  uint32_t rid;
  uint64_t idx;
  uint32_t engine_offset;
  char padding[28];
};
static_assert(sizeof(struct NcclFifoItem) == 64,
              "NcclFifoItem must be 64 bytes");

enum class NcclReqType { ReqRead, ReqWrite };

struct NcclRequest {
  NcclReqType type;
  uint32_t peer_id;
  void* context;
  uint32_t engine_idx;
};

class EpollClient;

int get_numa_node_from_iface();

// Placeholder for RDMA-style MR arrays. NCCL does not register memory, but the
// unified interface expects the type.
struct NcclMRArray {
  void* dummy = nullptr;
};

class NCCLEndpoint {
 public:
  // NCCL endpoint: NCCL control-plane + NCCL send/recv data-plane.
  explicit NCCLEndpoint(int gpu_index, uint16_t port = 0);
  ~NCCLEndpoint();

  // GPU index selected by the engine; may be overridden by uccl_connect/accept.
  int gpuIndex() const { return gpu_index_; }

  // RDMA endpoint exposes these for OOB metadata exchange; NCCL path doesn't
  // use EpollClient but we keep stubs so engine code can compile unchanged.
  std::shared_ptr<EpollClient> get_oob_client() const { return nullptr; }
  std::string get_oob_conn_key(uint64_t peer_id) const {
    (void)peer_id;
    return "";
  }

  // Establish a NCCL control connection, exchange NCCL IDs, and start a control
  // thread for one-sided read/write requests.
  NcclConnID uccl_connect(int dev, int local_gpuidx, int remote_dev,
                          int remote_gpuidx, std::string remote_ip,
                          uint16_t remote_port);
  // Listen socket metadata. dev is unused for NCCL but required by the API.
  uint16_t get_p2p_listen_port(int dev) { return listen_port_; }
  int get_p2p_listen_fd(int dev) { return listen_fd_; }
  // Accept a NCCL control connection and initialize NCCL communicators.
  NcclConnID uccl_accept(int dev, int listen_fd, int local_gpuidx,
                         std::string& remote_ip, int* remote_dev,
                         int* remote_gpuidx);

  // Memory registration is a no-op for NCCL; we keep the interface.
  int uccl_regmr(NcclFlow* flow, void* data, size_t len, int type,
                 NcclMhandle** mhandle);
  int uccl_regmr(void* data, size_t len, NcclMRArray& mr_array);
  int uccl_regmr(int dev, void* data, size_t len, int type,
                 NcclMhandle** mhandle);
  void uccl_deregmr(NcclMhandle* mhandle);
  void uccl_deregmr(NcclMRArray const& mr_array);

  // One-sided semantics built on a control message + NCCL send/recv.
  int uccl_read_async(NcclFlow* flow, NcclMhandle* mh, void* dst, size_t size,
                      NcclFifoItem const& slot_item, NcclRequest* ureq);
  int uccl_write_async(NcclFlow* flow, NcclMhandle* mh, void* src, size_t size,
                       NcclFifoItem const& slot_item, NcclRequest* ureq);
  // Poll a NcclRequest created by send/recv/read/write.
  bool uccl_poll_ureq_once(NcclRequest* ureq);
  // Serialize a FIFO descriptor (addr/size) for control-plane exchange.
  int prepare_fifo_metadata(NcclFlow* flow, NcclMhandle** mhandle,
                            void const* data, size_t size, char* out_buf);

  // NCCL has no device selection or unified socket; keep these as no-ops.
  int get_best_dev_idx(int gpu_idx) { return 0; }

  // Get the socket file descriptor for a connection.
  int get_sock_fd(uint64_t peer_id);

  // Send a notification message to a peer (uses NotifyMsg from common.h)
  int send_notification(uint64_t peer_id,
                        struct ::NotifyMsg const& notification);

  bool initialize_engine_by_dev(int dev, bool enable_p2p_listen) {
    if (dev >= 0) {
      gpu_index_ = dev;
    }
    (void)enable_p2p_listen;
    return true;
  }

  void create_unified_p2p_socket() {}

  void stop_accept() { stop_accept_.store(true, std::memory_order_release); }

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
  // Internal NCCL send/recv with completion events in NcclRequest.
  bool send_internal_(Conn& conn, void const* data, size_t size, int comm_index,
                      NcclRequest* ureq);
  bool recv_internal_(Conn& conn, void* data, size_t size, int comm_index,
                      NcclRequest* ureq);
  void cleanup_conn_(Conn& conn);

  // GPU index the endpoint binds to by default.
  int gpu_index_;
  // Control-plane listening socket.
  uint16_t listen_port_;
  int listen_fd_;
  std::atomic<uint64_t> next_peer_id_{1};
  std::atomic<bool> stop_accept_{false};
  std::mutex conn_mu_;
  std::unordered_map<uint64_t, std::unique_ptr<Conn>> conn_map_;
};
