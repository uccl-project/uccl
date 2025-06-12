#pragma once

#include "eqds.h"
#include "transport_config.h"
#include "util/latency.h"
#include "util/shared_pool.h"
#include "util/util.h"
#include "util_rdma.h"
#include "util_timer.h"
#include <glog/logging.h>
#include <infiniband/verbs.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <netdb.h>

namespace uccl {

struct ConnID {
  void* context;
  FlowID flow_id;  // Used for RDMAEndpoint to look up UcclFlow.
  PeerID peer_id;  // Used for UcclEngine to look up RDMAContext.
  int dev;
};

struct Mhandle {
  struct ibv_mr* mr;
};

/**
 * @class Channel
 * @brief A channel is a command queue for application threads to submit rx and
 * tx requests to the UcclFlow. A channel is only served by one UcclFlow, but
 * could be shared by multiple app threads if needed.
 */
class Channel {
  constexpr static uint32_t kChannelSize = 1024;

 public:
  struct Msg {
    enum Op : uint8_t { kTx, kRx };
    Op opcode;
    PeerID peer_id;
    struct ucclRequest* ureq;
    PollCtx* poll_ctx;
  };
  static_assert(sizeof(Msg) % 4 == 0, "channelMsg must be 32-bit aligned");

  struct CtrlMsg {
    enum Op : uint8_t {
      // Endpoint --> Engine
      kInstallCtx = 0,
      kInstallFlow,
    };
    Op opcode;
    PeerID peer_id;

    union CtrlMeta meta;
    // Wakeup handler
    PollCtx* poll_ctx;
  };
  static_assert(sizeof(CtrlMsg) % 4 == 0, "channelMsg must be 32-bit aligned");

  Channel() {
    tx_cmdq_ = create_ring(sizeof(Msg), kChannelSize);
    rx_cmdq_ = create_ring(sizeof(Msg), kChannelSize);
    ctrl_cmdq_ = create_ring(sizeof(CtrlMsg), kChannelSize);
  }

  ~Channel() {
    free(tx_cmdq_);
    free(rx_cmdq_);
    free(ctrl_cmdq_);
  }

  jring_t* tx_cmdq_;
  jring_t* rx_cmdq_;
  jring_t* ctrl_cmdq_;
};

class UcclFlow;
class UcclRDMAEngine;
class RDMAEndpoint;

/**
 * @brief Class `UcclRDMAEngine' abstracts the main Uccl engine which supports
 * RDMA. This engine contains all the functionality need to be run by the
 * stack's threads.
 */
class UcclRDMAEngine {
 public:
  TimerManager rto_tm_;
  uint32_t engine_outstanding_bytes_ = 0;

  UcclRDMAEngine() = delete;
  UcclRDMAEngine(UcclRDMAEngine const&) = delete;

  /**
   * @brief Construct a new UcclRDMAEngine object.
   * @param dev           Device index.
   * @param engine_id     Engine index.
   * @param channel       Uccl channel the engine will be responsible for.
   * For now, we assume an engine is responsible for a single channel, but
   * future it may be responsible for multiple channels.
   */
  UcclRDMAEngine(int dev, int engine_id, Channel* channel, eqds::EQDS* eqds)
      : engine_idx_(engine_id),
        dev_(dev),
        channel_(channel),
        eqds_(eqds),
        last_periodic_tsc_(rdtsc()),
        last_sync_clock_tsc_(rdtsc()),
        rto_tm_(kRTOUSec),
        kSlowTimerIntervalTsc_(us_to_cycles(kSlowTimerIntervalUs, freq_ghz)) {
    auto context = RDMAFactory::get_factory_dev(dev_)->context;
    struct ibv_values_ex values;
    values.comp_mask = IBV_VALUES_MASK_RAW_CLOCK;
    ibv_query_rt_values_ex(context, &values);
    auto nic_clock = values.raw_clock.tv_sec * 1e9 + values.raw_clock.tv_nsec;
    last_nic_clock_ = nic_clock;
    last_host_clock_ = rdtsc();
  }

  /**
   * @brief Handling aysnc send requests from Endpoint for all flows.
   */
  void handle_tx_work(void);

  /**
   * @brief Handling aysnc recv requests from Endpoint for all flows.
   */
  void handle_rx_work(void);

  /**
   * @brief Handling all completion events for all RDMAContexts, including:
   * High-priority completion events from all Ctrl QPs.
   * Datapath completion events from all data path QPs.
   */
  void uc_handle_completion(void);

  void rc_handle_completion(void);

  inline void handle_completion(void) {
    if constexpr (!kRCMode)
      uc_handle_completion();
    else
      rc_handle_completion();
  }

  /**
   * @brief Handle all timing wheel events for all RDMAContexts.
   *
   */
  void handle_timing_wheel(void);

  /**
   * @brief This is the main event cycle of the Uccl engine.
   * It is called by a separate thread running the Uccl engine.
   * On each iteration, the engine processes incoming packets in the RX
   * queue and enqueued messages in all channels that it is responsible
   * for. This method is not thread-safe.
   */
  void run();

  /**
   * @brief Method to perform periodic processing. This is called by the
   * main engine cycle (see method `Run`).
   */
  void periodic_process();

  /**
   * @brief Install a new RDMA context on the engine.
   * @param ctrl_work
   */
  void handle_install_ctx_on_engine(Channel::CtrlMsg& ctrl_work);

  /**
   * @brief Install a new UcclFlow on the engine.
   *
   * @param ctrl_work
   */
  void handle_install_flow_on_engine(Channel::CtrlMsg& ctrl_work);

  inline bool need_sync(uint64_t now) {
    return now - last_sync_clock_tsc_ >
           ns_to_cycles(kSyncClockIntervalNS, freq_ghz);
  }

  /**
   * @brief Synchronize the clock between host and NIC every
   * kSyncClockIntervalNS according to Flor[OSDI'23]
   */
  inline void handle_clock_synchronization(void) {
    auto host_clock = rdtsc();
    if (need_sync(host_clock)) {
      auto context = RDMAFactory::get_factory_dev(dev_)->context;
      struct ibv_values_ex values;
      values.comp_mask = IBV_VALUES_MASK_RAW_CLOCK;
      ibv_query_rt_values_ex(context, &values);

      auto nic_clock = values.raw_clock.tv_sec * 1e9 + values.raw_clock.tv_nsec;

      // Update ratio and offset
      ratio_ = (1.0 * (int64_t)host_clock - (int64_t)last_host_clock_) /
               ((int64_t)nic_clock - (int64_t)last_nic_clock_);
      offset_ = host_clock - ratio_ * nic_clock;

      last_sync_clock_tsc_ = host_clock;
    }
  }

  // Called by application to shutdown the engine. App will need to join
  // the engine thread.
  inline void shutdown() { shutdown_ = true; }

  void release();

  std::string status_to_string();

 protected:
  /**
   * @brief Iterate throught the list of flows, check and handle RTOs.
   */
  void handle_rto();

  /**
   * @brief This method polls active channels for all control plane
   * requests and processes them. It is called periodically.
   */
  void process_ctl_reqs();

 private:
  // Device index
  int dev_;
  // Engine index
  uint32_t engine_idx_;
  // RDMAContext map
  std::unordered_map<PeerID, RDMAContext*> rdma_ctx_map_;
  // Control plane channel with RDMAEndpoint.
  Channel* channel_;

  eqds::EQDS* eqds_;

  // Pending rx work due to no available request.
  std::deque<std::pair<RDMAContext*, struct ucclRequest*>> pending_rx_works_;
  // Pending tx work due to reaching the max outstanding bytes.
  std::deque<std::pair<RDMAContext*, struct ucclRequest*>> pending_tx_works_;

  // Timestamp of last periodic process execution.
  uint64_t last_periodic_tsc_;
  // Slow timer interval in TSC.
  uint64_t kSlowTimerIntervalTsc_;

  // Timestamp of last clock synchronization.
  uint64_t last_sync_clock_tsc_;
  uint64_t last_host_clock_;
  uint64_t last_nic_clock_;
  double ratio_ = 0;
  double offset_ = 0;

  // Whether shutdown is requested.
  std::atomic<bool> shutdown_{false};

  friend class UcclFlow;
};

/**
 * @brief A peer is identified by its IP address and device index and GPU index.
 */
struct UcclPeer {
  std::string remote_ip;
  int remote_dev;
};

static bool operator==(UcclPeer const& lhs, UcclPeer const& rhs) {
  return lhs.remote_ip == rhs.remote_ip && lhs.remote_dev == rhs.remote_dev;
}

struct UcclPeerHash {
  std::size_t operator()(UcclPeer const& peer) const {
    return std::hash<std::string>()(peer.remote_ip) ^
           std::hash<int>()(peer.remote_dev);
  }
};

struct PeerInfo {
  PeerID peer_id;
  ibv_gid remote_gid;
  struct ibv_port_attr remote_port_attr;
  uint32_t flow_cnt;
};

/**
 * @class RDMAEndpoint
 * @brief application-facing interface, communicating with `UcclRDMAEngine'
 * through `Channel'. Each connection is identified by a unique flow_id, and
 * uses multiple src+dst port combinations to leverage multiple paths. Under the
 * hood, we leverage TCP to boostrap our connections. We do not consider
 * multi-tenancy for now, assuming this endpoint exclusively uses the NIC and
 * its all queues. Note that all IB devices are managed by a single
 * RDMAEndpoint.
 */
class RDMAEndpoint {
  constexpr static uint32_t kMaxInflightMsg = 1024 * 256;
  constexpr static uint16_t kTestListenPort = 30000;
  constexpr static uint32_t kStatsTimerIntervalSec = 2;
  constexpr static uint32_t RC_MAGIC = 0x12345678;
  constexpr static uint16_t kBootstrapPort = 5000;

  std::shared_ptr<RDMAFactory> rdma_ctl_;

  // RDMA devices.
  int num_devices_;

  int num_engines_per_dev_;
  // Per-engine communication channel
  Channel* channel_vec_[NUM_ENGINES * NUM_DEVICES];
  std::vector<std::unique_ptr<UcclRDMAEngine>> engine_vec_;
  std::unordered_map<int, std::unique_ptr<UcclRDMAEngine>>
      engine_id_to_engine_map_;
  std::vector<std::unique_ptr<std::thread>> engine_th_vec_;

  // Number of outstanding messages for each engine.
  std::array<std::atomic<uint32_t>, NUM_ENGINES* NUM_DEVICES> engine_load_vec_ =
      {};

  // Receiver-driven congestion control.
  eqds::EQDS* eqds_[NUM_DEVICES] = {};

  SharedPool<PollCtx*, true>* ctx_pool_;
  uint8_t* ctx_pool_buf_;

  int test_listen_fds_[NUM_DEVICES];

  std::mutex fd_vec_mu_;
  // Mapping from unique (within this engine) flow_id to the boostrap fd.
  std::vector<int> fd_vec_;

  // Peer map for connecting/accepting
  std::unordered_map<UcclPeer, PeerInfo, UcclPeerHash> peer_map_[NUM_DEVICES];
  std::mutex peer_map_mu_[NUM_DEVICES];

  std::unordered_map<UcclPeer, PeerInfo, UcclPeerHash>
      peer_same_dev_map_[NUM_DEVICES][2];
  std::mutex peer_same_dev_map_mu_[NUM_DEVICES][2];

  std::atomic<PeerID> next_peer_id_[NUM_DEVICES] = {};

  Spin flow_id_spin_[NUM_DEVICES][MAX_PEER];
  FlowID next_flow_id_[NUM_DEVICES][MAX_PEER] = {};

  std::vector<UcclFlow*> active_flows_vec_[NUM_DEVICES];
  Spin active_flows_spin_[NUM_DEVICES];

 public:
  RDMAEndpoint(uint8_t const* devname_suffix_list, int num_devices,
               int num_engines_per_dev);

  RDMAEndpoint(int num_devices, int num_engines_per_dev);
  ~RDMAEndpoint();

  bool initialize_engine_by_dev(int dev, std::atomic<uint16_t>& port);

  /// For testing easily.
  ConnID test_uccl_connect(int dev, std::string remote_ip, int remote_dev) {
    return uccl_connect(dev, dev, remote_dev, remote_dev, remote_ip,
                        kTestListenPort + remote_dev);
  }
  ConnID test_uccl_accept(int dev, std::string& remote_ip, int* remote_dev) {
    return uccl_accept(dev, test_listen_fds_[dev], dev, remote_ip, remote_dev);
  }
  /// For testing easily.

  // Connect to a remote peer <remote_ip, remote_dev> with the given dev, who
  // is listening on the given listen_port. This function is thread-safe.
  ConnID uccl_connect(int dev, int local_gpuidx, int remote_dev,
                      int remote_gpuidx, std::string remote_ip,
                      uint16_t remote_port);

  // Accept a connection using the given listen_fd. <remote_ip, remote_dev> is
  // returned. This function is thread-safe.
  ConnID uccl_accept(int dev, int listen_fd, int local_gpuidx,
                     std::string& remote_ip, int* remote_dev);

  // Register a memory region.
  int uccl_regmr(UcclFlow* flow, void* data, size_t len, int type,
                 struct Mhandle** mhandle);
  // Register a DMA-BUF memory region.
  int uccl_regmr_dmabuf(UcclFlow* flow, void* data, size_t len, int type,
                        int offset, int fd, struct Mhandle** mhandle);
  // Deregister a memory region.
  void uccl_deregmr(struct Mhandle* mhandle);

  // Post a buffer to engine for sending data asynchronously.
  int uccl_send_async(UcclFlow* flow, struct Mhandle* mhandle, void const* data,
                      size_t const size, struct ucclRequest* ureq);

  // Post n buffers to engine for receiving data asynchronously.
  int uccl_recv_async(UcclFlow* flow, struct Mhandle** mhandles, void** data,
                      int* size, int n, struct ucclRequest* ureq);

  // Ensure that all received data is visible to GPU.
  int uccl_flush(UcclFlow* flow, struct Mhandle** mhandles, void** data,
                 int* size, int n, struct ucclRequest* ureq);

  bool uccl_poll_ureq_once(struct ucclRequest* ureq);

  inline bool uccl_poll_ureq(struct ucclRequest* ureq) {
    while (!uccl_poll_ureq_once(ureq)) {
    }
    return true;
  }

  inline bool uccl_wait(PollCtx* ctx) {
    std::unique_lock<std::mutex> lock(ctx->mu);
    ctx->cv.wait(lock, [&ctx] { return ctx->done.load(); });
    fence_and_clean_ctx(ctx);
    return true;
  }

  inline bool uccl_poll_once(PollCtx* ctx) {
    if (!ctx->done.load()) return false;
    fence_and_clean_ctx(ctx);
    return true;
  }

  inline bool uccl_poll(PollCtx* ctx) {
    while (!uccl_poll_once(ctx)) {
    }
    return true;
  }

  inline PeerID alloc_peer_id(int dev) {
    return next_peer_id_[dev].fetch_add(1);
  }

 private:
  PollCtx* install_ctx_on_engine(uint32_t engine_idx, PeerID peer_id,
                                 union CtrlMeta meta);

  PollCtx* install_flow_on_engine(uint32_t engine_idx, PeerID peer_id,
                                  union CtrlMeta meta);

  /**
   * @brief Safely install context on all engines serving the device.
   * When local_lock_first is true, the function will acquire local lock
   * first, and then acquire remote lock. Otherwise, it will acquire remote
   * lock first. When holding the two locks, and no context is installed for
   * the remote peer before, the function will install the context on all
   * engines serving the device. peer_id and remote_ctx are returned for
   * creating UcclFlow.
   *
   * @param dev
   * @param bootstrap_fd
   * @param local_lock_first
   * @param remote_ip
   * @param remote_dev
   * @param peer_id
   * @param remote_ctx
   */
  void safe_install_ctx(int dev, int bootstrap_fd, bool local_lock_first,
                        std::string& remote_ip, int remote_dev, PeerID* peer_id,
                        struct RemoteRDMAContext* remote_ctx);

  void same_dev_install_ctx(int dev, int bootstrap_fd, bool local_lock_first,
                            bool is_send, std::string& remote_ip,
                            int remote_dev, PeerID* peer_id,
                            struct RemoteRDMAContext* remote_ctx);

  void install_ctx_on_engines(int fd, int dev, PeerID peer_id,
                              struct RemoteRDMAContext* remote_ctx);

  void install_flow_on_engines(int dev, PeerID peer_id, FlowID flow_id,
                               UcclFlow* flow, bool is_send);

  inline void inc_load_on_engine(int engine_id) {
    std::atomic_fetch_add(&engine_load_vec_[engine_id], 1);
  }

  inline void dec_load_on_engine(int engine_id) {
    std::atomic_fetch_sub(&engine_load_vec_[engine_id], 1);
  }

  // Find a least loaded engine and update the load for the given device.
  inline uint32_t find_least_loaded_engine_idx(int dev);

  inline uint32_t find_pot_load_engine_idx(int dev);

  // Find an engine in a round-robin manner.
  inline uint32_t find_rr_engine_idx(int dev, uint32_t* next_candidate);

  // Find an engine in an oblivious manner.
  inline uint32_t find_oblivious_engine_idx(int dev);

  inline int find_first_engine_idx_on_dev(int dev) {
    return dev * num_engines_per_dev_;
  }

  inline void fence_and_clean_ctx(PollCtx* ctx) {
    // Make the data written by the engine thread visible to the app thread.
    std::ignore =
        std::atomic_load_explicit(&ctx->fence, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);

    ctx->clear();
    ctx_pool_->push(ctx);
  }

  std::thread stats_thread_;
  void stats_thread_fn();
  std::mutex stats_mu_;
  std::condition_variable stats_cv_;
  std::atomic<bool> shutdown_{false};

  friend class UcclFlow;
};

/**
 * @class UcclFlow, a connection between a local and a remote NIC.
 * @brief Class to abstract the components and functionality of a single flow.
 * A flow is a **unidirectional** connection between two NICs, uniquely
 * identified by a TCP-negotiated `FlowID'.
 */
class UcclFlow {
  static constexpr int kFifoMRSize = sizeof(struct RemFifo);
  static constexpr int kFifoCQSize = 4096;

 public:
  SubUcclFlow* sub_flows_[NUM_ENGINES];

  UcclFlow(RDMAEndpoint* ep, int bootstrap_fd, int dev, PeerID peer_id,
           FlowID flow_id, struct RemoteRDMAContext remote_ctx,
           std::string remote_ip, int remote_dev, bool is_send)
      : ep_(ep),
        dev_(dev),
        peer_id_(peer_id),
        flow_id_(flow_id),
        remote_ctx_(remote_ctx),
        remote_ip_(remote_ip),
        remote_dev_(remote_dev),
        is_send_(is_send) {
    for (int i = 0; i < NUM_ENGINES; i++) {
      sub_flows_[i] = new SubUcclFlow(flow_id);
    }

    memset(&send_comm_, 0, sizeof(send_comm_));
    memset(&recv_comm_, 0, sizeof(recv_comm_));

    auto comm_base = is_send_ ? &send_comm_.base : &recv_comm_.base;

    auto factory_dev = RDMAFactory::get_factory_dev(dev);

    // Fifo QP.
    comm_base->fifo_local_psn = BASE_PSN;
    util_rdma_create_qp(factory_dev->context, &comm_base->fifo_qp, IBV_QPT_RC,
                        false, false, &comm_base->flow_cq, false, kFifoCQSize,
                        factory_dev->pd, &comm_base->fifo_mr, nullptr,
                        kFifoMRSize, kMaxReq * kMaxRecv, kMaxReq * kMaxRecv, 1,
                        1);
    comm_base->fifo =
        reinterpret_cast<struct RemFifo*>(comm_base->fifo_mr->addr);

    // Exchange local PSN, QPN for Fifo QP with remote peer.
    char buf[2 * sizeof(uint32_t)];
    auto fifo_lpsn = comm_base->fifo_local_psn;
    auto fifo_lqpn = comm_base->fifo_qp->qp_num;
    memcpy(buf, &fifo_lpsn, sizeof(uint32_t));
    memcpy(buf + sizeof(uint32_t), &fifo_lqpn, sizeof(uint32_t));

    UCCL_INIT_CHECK(send_message(bootstrap_fd, buf, 2 * sizeof(uint32_t)) ==
                        2 * sizeof(uint32_t),
                    "uccl_connect: send_message()");

    UCCL_INIT_CHECK(receive_message(bootstrap_fd, buf, 2 * sizeof(uint32_t)) ==
                        2 * sizeof(uint32_t),
                    "uccl_connect: receive_message()");

    auto fifo_rpsn = *reinterpret_cast<uint32_t*>(buf);
    auto fifo_rqpn = *reinterpret_cast<uint32_t*>(buf + sizeof(uint32_t));

    UCCL_INIT_CHECK(modify_qp_rtr(comm_base->fifo_qp, dev, &remote_ctx_,
                                  fifo_rqpn, fifo_rpsn) == 0,
                    "Failed to modify Fifo QP to RTR");
    UCCL_INIT_CHECK(modify_qp_rts(comm_base->fifo_qp, fifo_lpsn, true) == 0,
                    "Failed to modify Fifo QP to RTS");

    // Exchange addr and rkey for Fifo MR with remote peer.
    char buf2[sizeof(uint64_t) + sizeof(uint32_t)];
    auto fifo_laddr = reinterpret_cast<uint64_t>(comm_base->fifo_mr->addr);
    auto fifo_lrkey = comm_base->fifo_mr->rkey;
    memcpy(buf2, &fifo_laddr, sizeof(uint64_t));
    memcpy(buf2 + sizeof(uint64_t), &fifo_lrkey, sizeof(uint32_t));

    UCCL_INIT_CHECK(
        send_message(bootstrap_fd, buf2, sizeof(uint64_t) + sizeof(uint32_t)) ==
            sizeof(uint64_t) + sizeof(uint32_t),
        "uccl_connect: send_message()");

    UCCL_INIT_CHECK(receive_message(bootstrap_fd, buf2,
                                    sizeof(uint64_t) + sizeof(uint32_t)) ==
                        sizeof(uint64_t) + sizeof(uint32_t),
                    "uccl_connect: receive_message()");

    comm_base->remote_fifo_addr = *reinterpret_cast<uint64_t*>(buf2);
    comm_base->remote_fifo_rkey =
        *reinterpret_cast<uint32_t*>(buf2 + sizeof(uint64_t));

    // RC QP
    struct ibv_qp_init_attr qp_init_attr;
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    qp_init_attr.qp_context = this;
    qp_init_attr.send_cq = comm_base->flow_cq;
    qp_init_attr.recv_cq = comm_base->flow_cq;
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.cap.max_send_wr = kMaxReq * kMaxRecv;
    qp_init_attr.cap.max_recv_wr = kMaxReq * kMaxRecv;
    qp_init_attr.cap.max_send_sge = kMaxSge;
    qp_init_attr.cap.max_recv_sge = kMaxSge;
    qp_init_attr.cap.max_inline_data = 0;

    struct ibv_qp_attr qpAttr;
    memset(&qpAttr, 0, sizeof(qpAttr));
    qpAttr.qp_state = IBV_QPS_INIT;
    qpAttr.pkey_index = 0;
    qpAttr.port_num = IB_PORT_NUM;
    qpAttr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;

    comm_base->rc_qp = ibv_create_qp(factory_dev->pd, &qp_init_attr);
    UCCL_INIT_CHECK(comm_base->rc_qp != nullptr, "Failed to create RC QP");

    UCCL_INIT_CHECK(ibv_modify_qp(comm_base->rc_qp, &qpAttr,
                                  IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                                      IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0,
                    "Failed to modify RC QP to INIT");

    comm_base->rc_local_psn = BASE_PSN;

    // Send PSN, QPN to remote peer.
    memcpy(buf, &comm_base->rc_local_psn, sizeof(uint32_t));
    memcpy(buf + sizeof(uint32_t), &comm_base->rc_qp->qp_num, sizeof(uint32_t));
    int ret = send_message(bootstrap_fd, buf, 2 * sizeof(uint32_t));
    DCHECK(ret == 2 * sizeof(uint32_t));

    // Receive PSN, QPN from remote peer.
    ret = receive_message(bootstrap_fd, buf, 2 * sizeof(uint32_t));
    DCHECK(ret == 2 * sizeof(uint32_t));

    auto rc_rpsn = *reinterpret_cast<uint32_t*>(buf);
    auto rc_rqpn = *reinterpret_cast<uint32_t*>(buf + sizeof(uint32_t));

    UCCL_INIT_CHECK(modify_qp_rtr(comm_base->rc_qp, dev, &remote_ctx_, rc_rqpn,
                                  rc_rpsn) == 0,
                    "Failed to modify RC QP to RTR");

    UCCL_INIT_CHECK(
        modify_qp_rts(comm_base->rc_qp, comm_base->rc_local_psn, true) == 0,
        "Failed to modify RC QP to RTS");

    // GPU flush QP for receiver.
    if (!is_send_) {
      util_rdma_create_qp(
          factory_dev->context, &recv_comm_.gpu_flush_qp, IBV_QPT_RC, false,
          false, &comm_base->flow_cq, true, 0, factory_dev->pd,
          &recv_comm_.gpu_flush_mr, &recv_comm_.gpu_flush, sizeof(int),
          kMaxReq * kMaxRecv, kMaxReq * kMaxRecv, kMaxSge, kMaxSge);

      recv_comm_.gpu_flush_sge.addr = (uint64_t)&recv_comm_.gpu_flush;
      recv_comm_.gpu_flush_sge.length = 1;
      recv_comm_.gpu_flush_sge.lkey = recv_comm_.gpu_flush_mr->lkey;

      UCCL_INIT_CHECK(modify_qp_rtr_gpuflush(recv_comm_.gpu_flush_qp, dev) == 0,
                      "Failed to modify GPU flush QP to RTR");
      UCCL_INIT_CHECK(modify_qp_rts(recv_comm_.gpu_flush_qp, 0, true) == 0,
                      "Failed to modify GPU flush QP to RTS");
    }
    // Avoid all flows using the same initial engine offset.
    static std::atomic<uint32_t> off[NUM_DEVICES] = {};
    next_engine_offset_ = off[dev].fetch_add(1) % NUM_ENGINES;
  }

  ~UcclFlow() {
    auto comm_base = is_send_ ? &send_comm_.base : &recv_comm_.base;

    munmap(comm_base->fifo_mr->addr, comm_base->fifo_mr->length);
    ibv_dereg_mr(comm_base->fifo_mr);
    ibv_destroy_qp(comm_base->fifo_qp);

    ibv_destroy_cq(comm_base->flow_cq);

    if (!is_send_) {
      munmap(recv_comm_.gpu_flush_mr->addr, recv_comm_.gpu_flush_mr->length);
      ibv_dereg_mr(recv_comm_.gpu_flush_mr);
      ibv_destroy_qp(recv_comm_.gpu_flush_qp);
    }

    for (int i = 0; i < NUM_ENGINES; i++) {
      delete sub_flows_[i];
    }
  }

  inline FlowID flowid() const { return flow_id_; }

  friend class UcclRDMAEngine;

  void release() {}

 private:
  inline int check_need_flush(int* size, int n) {
    // Only flush once using the last non-zero receive
    int last = -1;
    for (int i = 0; i < n; i++)
      if (size[i]) last = i;
    return last;
  }
  /**
   * @brief Post a RDMA READ operation to GPU flush QP. This operation
   * bypasses the UcclEngine.
   */
  void post_flush(struct Mhandle** mhandles, void** data, int* size, int n,
                  uint64_t* flush_done, int last);

  /**
   * @brief Post multiple recv requests to a FIFO queue for remote peer to use
   * RDMA WRITE. These requests are transmitted through the underlyding fifo
   * QP (RC).
   * @param data Array of data buffers.
   * @param size Array of buffer sizes.
   * @param n Number of buffers.
   */
  struct FifoItem* post_fifo(uint32_t engine_idx, void** data, int* size, int n,
                             struct Mhandle** mhandle, struct ibv_send_wr* wr,
                             struct ibv_sge* sge);

  void rc_recv(void* data, int size, struct Mhandle* mhandle,
               struct ibv_send_wr* wr, struct ibv_sge* sge,
               struct ucclRequest* ureq);

  /**
   * @brief Poll the completion queue for the Fifo/GPU flush QP.
   */
  void poll_flow_cq(void);

  /**
   * @brief This function is called by uccl_send_async to check if the
   * receiver has posted buffers.
   */
  bool check_fifo_ready(int* ret_slot, int* ret_nmsgs);

  /**
   * @brief This function is called by uccl_send_async to post multiple send
   * requests to UcclEngine.
   * @param engine_offset The engine offset to use, which is determined by the
   * receiver. 0 <= engine_offset < num_engines_per_dev_.
   */
  void post_multi_send(struct ucclRequest** ureqs, uint32_t engine_offset);

  void rc_send(struct ucclRequest* ureq);

  inline bool check_room(void) { return outstanding_reqs_ < kMaxReq; }

  inline void inc_outstanding_reqs(void) { outstanding_reqs_++; }

  inline void dec_outstanding_reqs(void) { outstanding_reqs_--; }

  inline uint32_t get_last_rc_size(void) { return last_rc_recv_; }

  inline void set_last_rc_size(uint32_t size) { last_rc_recv_ = size; }

  RDMAEndpoint* ep_;

  PeerID peer_id_;
  FlowID flow_id_;

  int dev_;
  int remote_dev_;
  struct RemoteRDMAContext remote_ctx_;
  std::string remote_ip_;

  /**
   * @brief Next engine offset to use for receving.
   */
  uint32_t next_engine_offset_ = 0;

  /**
   * @brief # of CQEs need to be polled for Fifo/GPU flush QP.
   */
  uint32_t flow_cq_cnt_ = 0;

  /**
   * @brief Communication abstraction for sending and receiving.
   * Since data flow is unidirectional in NCCL, we use two different
   * structures.
   */
  union {
    // For connection setup by connect().
    struct SendComm send_comm_;
    // For connection setup by accept().
    struct RecvComm recv_comm_;
  };

  uint32_t outstanding_reqs_ = 0;

  uint32_t last_rc_recv_ = 0;

  // Whether this context is for sending or receiving.
  bool is_send_;

  // Measure the distribution of probed RTT.
  Latency rtt_stats_;
  uint64_t rtt_probe_count_ = 0;

  friend class UcclRDMAEngine;
  friend class RDMAEndpoint;
};

}  // namespace uccl
