#pragma once

#include "transport_config.h"
#include "transport_header.h"
#include "util/list.h"
#include "util/util.h"
#include "util_buffpool.h"
#include "util_flock.h"
#include "util_timer.h"
#include <arpa/inet.h>
#include <glog/logging.h>
#include <infiniband/efadv.h>
#include <infiniband/verbs.h>
#include <linux/if_ether.h>
#include <linux/if_link.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <atomic>
#include <bitset>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <assert.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <ifaddrs.h>
#include <inttypes.h>
#include <memory.h>
#include <poll.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace uccl {

class PktHdrBuffPool : public BuffPool {
 public:
  // This should cover EFA_UD_ADDITION + data packet hdr
  // or EFA_UD_ADDITION + ack packet hdr + sack packet hdr.
  static constexpr uint32_t kPktHdrSize = 256;
  static constexpr uint32_t kNumPktHdr = NUM_FRAMES;
  static_assert((kNumPktHdr & (kNumPktHdr - 1)) == 0,
                "kNumPktHdr must be power of 2");

  PktHdrBuffPool(struct ibv_mr* mr) : BuffPool(kNumPktHdr, kPktHdrSize, mr) {}

  ~PktHdrBuffPool() = default;
};
static_assert(EFA_UD_ADDITION + kUcclPktHdrLen + kUcclSackHdrLen <
                  PktHdrBuffPool::kPktHdrSize,
              "uccl pkt hdr and sack hdr too large");

class PktDataBuffPool : public BuffPool {
 public:
  static constexpr uint32_t kPktDataSize = EFA_MTU;
  static constexpr uint32_t kNumPktData = NUM_FRAMES;
  static_assert((kNumPktData & (kNumPktData - 1)) == 0,
                "kNumPkt must be power of 2");

  PktDataBuffPool(struct ibv_mr* mr)
      : BuffPool(kNumPktData, kPktDataSize, mr) {}

  ~PktDataBuffPool() = default;
};

class PullHdrBuffPool : public BuffPool {
 public:
  static constexpr uint32_t kPullHdrSize = EFA_UD_ADDITION + kUcclPullHdrLen;
  static constexpr uint32_t kNumPktHdr = NUM_FRAMES;
  static_assert((kNumPktHdr & (kNumPktHdr - 1)) == 0,
                "kNumPktHdr must be power of 2");

  PullHdrBuffPool(struct ibv_mr* mr) : BuffPool(kNumPktHdr, kPullHdrSize, mr) {}

  ~PullHdrBuffPool() = default;
};

class FrameDesc {
  uint64_t pkt_hdr_addr_;      // in CPU memory.
  uint32_t pkt_hdr_len_;       // the length of packet hdr.
  uint64_t pkt_data_addr_;     // in GPU memory.
  uint32_t pkt_data_len_;      // the length of packet data.
  uint32_t pkt_data_lkey_tx_;  // Tx buff may come from app.

  uint16_t src_qp_idx_;     // src QP to use for this frame.
  uint16_t dest_qpn_;       // dest QPN to use for this frame.
  struct ibv_ah* dest_ah_;  // dest ah to use for this frame.

  uint64_t cpe_time_tsc_;  // Completion event time.

  // Flags to denote the message buffer state.
  static uint8_t const UCCL_MSGBUF_FLAGS_SYN = (1 << 0);
  static uint8_t const UCCL_MSGBUF_FLAGS_FIN = (1 << 1);
  uint8_t msg_flags_;

  // Pointing to the next message buffer in the chain.
  FrameDesc* next_;
  // Used for tx to wakeup the app thread when finishing a message.
  void* poll_ctx_;

  FrameDesc(uint64_t pkt_hdr_addr, uint32_t pkt_hdr_len, uint64_t pkt_data_addr,
            uint32_t pkt_data_len, uint32_t pkt_data_lkey_tx, uint8_t msg_flags)
      : pkt_hdr_addr_(pkt_hdr_addr),
        pkt_hdr_len_(pkt_hdr_len),
        pkt_data_addr_(pkt_data_addr),
        pkt_data_len_(pkt_data_len),
        pkt_data_lkey_tx_(pkt_data_lkey_tx),
        msg_flags_(msg_flags) {
    src_qp_idx_ = UINT16_MAX;
    dest_qpn_ = UINT16_MAX;
    dest_ah_ = nullptr;
    next_ = nullptr;
    poll_ctx_ = nullptr;
    cpe_time_tsc_ = 0;
  }

 public:
  static FrameDesc* Create(uint64_t frame_desc_addr, uint64_t pkt_hdr_addr,
                           uint32_t pkt_hdr_len, uint64_t pkt_data_addr,
                           uint32_t pkt_data_len, uint32_t pkt_data_lkey,
                           uint8_t msg_flags) {
    return new (reinterpret_cast<void*>(frame_desc_addr))
        FrameDesc(pkt_hdr_addr, pkt_hdr_len, pkt_data_addr, pkt_data_len,
                  pkt_data_lkey, msg_flags);
  }
  uint64_t get_pkt_hdr_addr() const { return pkt_hdr_addr_; }
  void set_pkt_hdr_addr(uint64_t pkt_hdr_addr) { pkt_hdr_addr_ = pkt_hdr_addr; }

  uint32_t get_pkt_hdr_len() const { return pkt_hdr_len_; }
  void set_pkt_hdr_len(uint32_t pkt_hdr_len) { pkt_hdr_len_ = pkt_hdr_len; }

  uint64_t get_pkt_data_addr() const { return pkt_data_addr_; }
  void set_pkt_data_addr(uint64_t pkt_data_addr) {
    pkt_data_addr_ = pkt_data_addr;
  }

  uint32_t get_pkt_data_len() const { return pkt_data_len_; }
  void set_pkt_data_len(uint32_t pkt_data_len) { pkt_data_len_ = pkt_data_len; }

  uint32_t get_pkt_data_lkey_tx() const { return pkt_data_lkey_tx_; }
  void set_pkt_data_lkey_tx(uint32_t pkt_data_lkey_tx) {
    pkt_data_lkey_tx_ = pkt_data_lkey_tx;
  }

  uint16_t get_msg_flags() const { return msg_flags_; }
  void set_msg_flags(uint16_t flags) { msg_flags_ = flags; }
  void add_msg_flags(uint16_t flags) { msg_flags_ |= flags; }

  uint16_t get_src_qp_idx() const { return src_qp_idx_; }
  void set_src_qp_idx(uint16_t src_qp_idx) { src_qp_idx_ = src_qp_idx; }

  uint16_t get_dest_qpn() const { return dest_qpn_; }
  void set_dest_qpn(uint16_t dest_qpn) { dest_qpn_ = dest_qpn; }

  struct ibv_ah* get_dest_ah() const {
    return dest_ah_;
  }
  void set_dest_ah(struct ibv_ah* dest_ah) { dest_ah_ = dest_ah; }

  uint64_t get_cpe_time_tsc() const { return cpe_time_tsc_; }
  void set_cpe_time_tsc(uint64_t cpe_time_tsc) { cpe_time_tsc_ = cpe_time_tsc; }

  // Returns true if this is the first in a message.
  bool is_first() const { return (msg_flags_ & UCCL_MSGBUF_FLAGS_SYN) != 0; }
  // Returns true if this is the last in a message.
  bool is_last() const { return (msg_flags_ & UCCL_MSGBUF_FLAGS_FIN) != 0; }

  // Returns the next message buffer index in the chain.
  FrameDesc* next() const { return next_; }
  // Set the next message buffer index in the chain.
  void set_next(FrameDesc* next) { next_ = next; }

  void mark_first() { add_msg_flags(UCCL_MSGBUF_FLAGS_SYN); }
  void mark_last() { add_msg_flags(UCCL_MSGBUF_FLAGS_FIN); }

  void* get_poll_ctx() const { return poll_ctx_; }
  void set_poll_ctx(void* poll_ctx) { poll_ctx_ = poll_ctx; }

  void clear_fields() {
    pkt_hdr_addr_ = 0;
    pkt_data_addr_ = 0;
    pkt_hdr_len_ = 0;
    pkt_data_len_ = 0;
    pkt_data_lkey_tx_ = 0;
    msg_flags_ = 0;
    src_qp_idx_ = UINT16_MAX;
    dest_qpn_ = UINT16_MAX;
    dest_ah_ = nullptr;
    next_ = nullptr;
    poll_ctx_ = nullptr;
    cpe_time_tsc_ = 0;
  }

  std::string to_string() {
    std::stringstream s;
    s << "pkt_hdr_addr: 0x" << std::hex << pkt_hdr_addr_ << " pkt_data_addr: 0x"
      << std::hex << pkt_data_addr_ << " pkt_hdr_len: " << std::dec
      << pkt_hdr_len_ << " pkt_data_len: " << std::dec << pkt_data_len_
      << " pkt_data_lkey_tx: " << std::hex << pkt_data_lkey_tx_
      << " src_qp_idx: " << std::dec << src_qp_idx_ << " dest_qpn: " << std::dec
      << dest_qpn_ << " dest_ah_: " << std::hex << dest_ah_
      << " msg_flags: " << std::dec << std::bitset<8>(msg_flags_)
      << " cpe_time_tsc: " << std::dec << cpe_time_tsc_;
    return s.str();
  }

  std::string print_chain() {
    std::stringstream s;
    auto* cur = this;
    while (cur && !cur->is_last()) {
      s << cur->to_string() << "\n";
      cur = cur->next_;
    }
    return s.str();
  }
};

class FrameDescBuffPool : public BuffPool {
 public:
  static constexpr uint32_t kFrameDescSize = sizeof(FrameDesc);
  static constexpr uint32_t kNumFrameDesc = NUM_FRAMES;
  static_assert((kNumFrameDesc & (kNumFrameDesc - 1)) == 0,
                "kNumFrameDesc must be power of 2");

  FrameDescBuffPool() : BuffPool(kNumFrameDesc, kFrameDescSize, nullptr) {}

  ~FrameDescBuffPool() = default;
};

struct EFADevice {
  char ib_name[64];
  std::string local_ip_str;

  struct ibv_context* context;
  struct ibv_device_attr dev_attr;
  struct ibv_port_attr port_attr;

  uint8_t dev_idx;
  uint8_t efa_port_num;
  union ibv_gid gid;

  struct ibv_pd* pd;

  // DMA-BUF support
  bool dma_buf_support;

  // Each is NIC-specific and binded with local pd_. The transport must call
  // this function to create ah and attached the ah to frame_desc before
  // sending these frames.
  struct ibv_ah* create_ah(union ibv_gid remote_gid) {
    struct ibv_ah_attr ah_attr = {};

    ah_attr.is_global = 1;  // Enable Global Routing Header (GRH)
    ah_attr.port_num = EFA_PORT_NUM;
    ah_attr.grh.sgid_index = EFA_GID_IDX;  // Local GID index
    ah_attr.grh.dgid = remote_gid;         // Destination GID
    ah_attr.grh.flow_label = 0;
    ah_attr.grh.hop_limit = 255;
    ah_attr.grh.traffic_class = 0;

    struct ibv_ah* ah = ibv_create_ah(pd, &ah_attr);
    if (!ah) {
      perror("Failed to create AH");
      exit(1);
    }

    return ah;
  }
};

class EFASocket;
class EFAFactory;
extern EFAFactory efa_ctl;

class EFAFactory {
 public:
  // dev_idx --> EFADevice pointer.
  std::unordered_map<int, struct EFADevice*> dev_map;

  std::mutex socket_q_lock_;
  std::deque<EFASocket*> socket_q_;

  // Not thread-safe; should be called just once.
  static void Init(int gpu);

  static void Init();

  // dev_idx from [1, ..., NUM_DEVICES];
  // socket_idx from [0, ..., kNumEngines - 1].
  static EFASocket* CreateSocket(int gpu_idx, int dev_idx, int socket_idx);

  // Getting a pointer to the struct EFADevice.
  static struct EFADevice* GetEFADevice(int dev_idx);

  static void Shutdown();

 private:
  static void InitDev(int dev_idx);
};

struct ConnMeta {
  // data qps + ctrl qp.
  uint32_t qpn_list[kMaxSrcDstQP];
  uint32_t qpn_list_ctrl[kMaxSrcDstQPCtrl];
  uint32_t qpn_list_credit[kMaxSrcDstQPCredit];
  union ibv_gid gid;
};

class EFASocket {
  struct ibv_context* context_;
  struct ibv_pd* pd_;
  union ibv_gid gid_;

  // Separating send and recv cq to allow better budget ctrl
  struct ibv_cq* send_cq_;
  struct ibv_cq* recv_cq_;
  struct ibv_qp* qp_list_[kMaxSrcDstQP];

  // Separating ctrl and data qp, so we can have different sizes for data
  // pkthdr and ack pkthdr.
  struct ibv_cq* ctrl_cq_;
  struct ibv_qp* ctrl_qp_list_[kMaxSrcDstQPCtrl];

  uint32_t gpu_idx_;
  uint32_t dev_idx_;
  uint32_t socket_idx_;

  PktHdrBuffPool* pkt_hdr_pool_;
  PktDataBuffPool* pkt_data_pool_;
  FrameDescBuffPool* frame_desc_pool_;

  // For fast CQ polling.
  struct ibv_wc wc_[kMaxPollBatch];

  struct ibv_send_wr send_wr_vec_[kMaxChainedWr];
  struct ibv_sge send_sge_vec_[kMaxChainedWr][2];

  struct ibv_recv_wr recv_wr_vec_[kMaxChainedWr];
  struct ibv_sge recv_sge_vec_[kMaxChainedWr][2];

  // How many recv_wrs are lacking for each qp to be full?
  uint16_t deficit_cnt_recv_wrs_[kMaxSrcDstQP];
  uint16_t deficit_cnt_recv_wrs_for_ctrl_[kMaxSrcDstQPCtrl];

  EFASocket(int gpu_idx, int dev_idx, int socket_idx);

  struct ibv_qp* create_qp(struct ibv_cq* send_cq, struct ibv_cq* recv_cq,
                           uint32_t send_cq_size, uint32_t recv_cq_size);
  struct ibv_qp* create_srd_qp(struct ibv_cq* send_cq, struct ibv_cq* recv_cq,
                               uint32_t send_cq_size, uint32_t recv_cq_size);

 public:
  inline uint32_t gpu_idx() const { return gpu_idx_; }
  inline uint32_t dev_idx() const { return dev_idx_; }
  inline uint32_t socket_idx() const { return socket_idx_; }

  // dest_qpn and dest_gid_idx is specified in FrameDesc; src_qp is determined
  // by EFASocket internally to evenly spread the load. This function also
  // dynamically chooses normal qp and ctrl_qp based on pkt_data_len_.
  uint32_t post_send_wr(FrameDesc* frame, uint16_t src_qp_idx);
  uint32_t post_send_wrs(std::vector<FrameDesc*>& frames, uint16_t src_qp_idx);
  uint32_t post_send_wrs_for_ctrl(std::vector<FrameDesc*>& frames,
                                  uint16_t src_qp_idx);

  // To do chaining while using different paths, we need to use fixed
  // src_qp_idx for all frames, and let the transport choose different dest
  // qpns. The transport needs to supply the src_qp_idx so it can maintain
  // per-path state. Here path_id = <src_qpn, dst_qpn>.
  void post_recv_wrs(uint32_t budget, uint16_t qp_idx);
  void post_recv_wrs_for_ctrl(uint32_t budget, uint16_t qp_idx);

  // This polls send_cq_ for data qps; wr_id is FrameDesc*.
  std::vector<FrameDesc*> poll_send_cq(uint32_t bugget);  // only for test
  void poll_send_cq();

  // This polls recv_cq_ for data qps; wr_id is FrameDesc*.
  std::vector<FrameDesc*> poll_recv_cq(uint32_t budget);
  // This returns 1) received acks and 2) num of polled sent acks (it
  // internally frees FrameDesc for sent acks)
  std::tuple<std::vector<FrameDesc*>, uint32_t> poll_ctrl_cq(uint32_t budget);

  // Round-robin index for choosing src_qp to send data/acl packets; this
  // is to fully utilize the micro-cores on EFA NICs.
  uint16_t next_qp_idx_for_send_ = 0;
  uint16_t next_qp_idx_for_send_ctrl_ = 0;
  inline uint16_t get_next_src_qp_idx_for_send() {
    return (next_qp_idx_for_send_++) % kMaxSrcQP;
  }
  inline uint16_t get_next_src_qp_idx_for_send_ctrl() {
    return (next_qp_idx_for_send_ctrl_++) % kMaxSrcQPCtrl;
  }

  std::string to_string();
  void shutdown();
  ~EFASocket();

  // Return kMaxSrcDstQP + kMaxSrcDstQPCtrl QP numbers for client to use.
  void get_conn_metadata(ConnMeta* meta) {
    for (int i = 0; i < kMaxSrcDstQP; i++) {
      meta->qpn_list[i] = qp_list_[i]->qp_num;
    }
    for (int i = 0; i < kMaxSrcDstQPCtrl; i++) {
      meta->qpn_list_ctrl[i] = ctrl_qp_list_[i]->qp_num;
    }
    meta->gid = gid_;
  }

  inline uint64_t pop_pkt_hdr() {
    uint64_t pkt_hdr_addr;
    auto ret = pkt_hdr_pool_->alloc_buff(&pkt_hdr_addr);
    DCHECK(ret == 0);
    return pkt_hdr_addr;
  }
  inline void push_pkt_hdr(uint64_t pkt_hdr_addr) {
    pkt_hdr_pool_->free_buff(pkt_hdr_addr);
  }
  inline uint32_t get_pkt_hdr_lkey() { return pkt_hdr_pool_->get_lkey(); }

  inline uint64_t pop_pkt_data() {
    uint64_t pkt_data_addr;
    auto ret = pkt_data_pool_->alloc_buff(&pkt_data_addr);
    DCHECK(ret == 0);
    return pkt_data_addr;
  }
  inline void push_pkt_data(uint64_t pkt_data_addr) {
    pkt_data_pool_->free_buff(pkt_data_addr);
  }
  inline uint32_t get_pkt_data_lkey() { return pkt_data_pool_->get_lkey(); }

  inline uint64_t pop_frame_desc() {
    uint64_t pkt_frame_desc;
    auto ret = frame_desc_pool_->alloc_buff(&pkt_frame_desc);
    DCHECK(ret == 0);
    return pkt_frame_desc;
  }
  inline void push_frame_desc(uint64_t pkt_frame_desc) {
    frame_desc_pool_->free_buff(pkt_frame_desc);
  }

  inline uint32_t send_queue_free_space_per_qp(int src_qp_id) {
    return kMaxSendWr - send_queue_wrs_per_qp_[src_qp_id];
  }
  inline uint32_t send_queue_free_space() {
    return kMaxSendWr * kMaxSrcQP - send_queue_wrs_;
  }
  inline uint64_t send_queue_estimated_latency_ns() {
    return send_queue_wrs_ * EFA_MTU * 1000000000UL / kLinkBandwidth;
  }
  inline uint32_t ctrl_send_queue_wrs() { return ctrl_send_queue_wrs_; }
  inline uint32_t send_queue_wrs() { return send_queue_wrs_; }
  inline uint32_t recv_queue_wrs() { return recv_queue_wrs_; }

 private:  // for stats
  uint32_t send_queue_wrs_per_qp_[kMaxSrcDstQP] = {0};
  uint32_t send_queue_wrs_ = 0;
  uint32_t recv_queue_wrs_ = 0;
  uint32_t ctrl_send_queue_wrs_ = 0;

  std::chrono::time_point<std::chrono::high_resolution_clock> last_stat_;
  std::atomic<uint64_t> out_packets_ = 0;
  std::atomic<uint64_t> out_bytes_ = 0;
  std::atomic<uint64_t> in_packets_ = 0;
  std::atomic<uint64_t> in_bytes_ = 0;

  friend class EFAFactory;
};

/**
 * @brief This helper function gets the Infiniband name from the dev index.
 *
 * @param dev_idx
 * @param ib_name
 * @return int
 */
static inline int util_efa_get_ib_name_from_dev_idx(int dev_idx,
                                                    char* ib_name) {
  sprintf(ib_name, "%s", EFA_DEVICE_NAME_LIST[dev_idx].c_str());
  return 0;
}

/**
 * @brief This helper function gets the IP address of the device from dev index.
 *
 * @param dev_idx
 * @return int
 */
static inline int util_efa_get_ip_from_dev_idx(int dev_idx, std::string* ip) {
  *ip = get_dev_ip(ENA_DEVICE_NAME_LIST[dev_idx].c_str());
  return *ip == "" ? -1 : 0;
}

}  // namespace uccl