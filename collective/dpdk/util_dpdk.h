#include "packet.h"
#include "packet_pool.h"
#include "pmd_port.h"
#include "rx_ring.h"
#include "tx_ring.h"
#include <glog/logging.h>
#include <linux/ethtool.h>
#include <cstdint>
#include <deque>
#include <mutex>
#include <new>

namespace uccl {

class DPDKSocket {
 public:
  DPDKSocket(int queue_id, RxRing* rx_ring_, TxRing* tx_ring_,
             PacketPool* packet_pool_) {
    this->queue_id_ = queue_id;
    this->rx_ring_ = rx_ring_;
    this->tx_ring_ = tx_ring_;
    this->packet_pool_ = packet_pool_;
    this->total_recv_packets_ = 0;
    this->total_sent_packets_ = 0;
    this->unsent_packets_ = 0;
  }

  inline uint32_t avail_packets() {
    std::lock_guard<std::mutex> lock(mutex_);
    return packet_pool_->AvailPacketsCount();
  }

  inline uint32_t in_use_packets() {
    std::lock_guard<std::mutex> lock(mutex_);
    return packet_pool_->InUsePacketsCount();
  }

  inline void push_packet(Packet* pkt) {
    // LOG(INFO) << "push_packet packet: " << pkt;
    Packet::Free(pkt);
  }

  inline Packet* pop_packet() {
    // std::lock_guard<std::mutex> lock(mutex_);
    return packet_pool_->PacketAlloc();
  }

  inline Packet* pop_packet(uint16_t pkt_len) {
    Packet* pkt = nullptr;
    do {
      // std::lock_guard<std::mutex> lock(mutex_);
      pkt = packet_pool_->PacketAlloc();
    } while (pkt == nullptr);

    // LOG(INFO) << "pop_packet(uint16_t) avail_packets: "
    //           << packet_pool_->AvailPacketsCount();
    pkt->append<void*>(pkt_len);
    return pkt;
  }

  uint32_t send_packets(Packet** pkts, uint32_t nb_pkts) {
    // for (uint32_t i = 0; i < nb_pkts; i++) {
    //   LOG(INFO) << "send_packets [" << i << "] packet: " << pkts[i];
    // }
    // LOG(INFO) << "send_packets: " << nb_pkts;

    uint32_t sent = tx_ring_->TrySendPackets(pkts, nb_pkts);
    total_sent_packets_ += sent;
    // LOG(INFO) << "send_packets sent: " << sent;
    return sent;
  }

  uint32_t send_packet(Packet* pkt) { return send_packets(&pkt, 1); }

  uint32_t recv_packets(Packet** pkts, uint32_t nb_pkts) {
    uint32_t rcvd = rx_ring_->RecvPackets(pkts, nb_pkts);
    // total_recv_packets_ += rcvd;
    return rcvd;
  }

  void increment_recv() { total_recv_packets_++; }

  int get_queue_id() const { return queue_id_; }

  inline uint64_t send_queue_estimated_latency_ns() { return 0; }

  std::string to_string() {
    struct rte_eth_stats stats;
    rte_eth_stats_get(0, &stats);
    LOG(INFO) << "rx: " << stats.ipackets << " rx_dropped: " << stats.imissed
              << " rx_nombuf: " << stats.rx_nombuf;

    return Format("[TX] %u [RX] %u [POOL] (%u, %u)", total_sent_packets_,
                  total_recv_packets_, avail_packets(), in_use_packets());
  }

 private:
  int queue_id_;
  // Designated RX queue for this engine (not shared).
  RxRing* rx_ring_;
  // Designated TX queue for this engine (not shared).
  TxRing* tx_ring_;
  // The following packet pool is used for all TX packets;
  // should not be shared with other engines/threads.
  PacketPool* packet_pool_;

  uint32_t total_recv_packets_;
  uint32_t total_sent_packets_;

  uint32_t unsent_packets_;
  std::mutex mutex_;
};

class DPDKFactory {
 public:
  DPDKFactory(uint16_t port_id, uint16_t rx_rings_nr, uint16_t tx_rings_nr)
      : initialized_(false), pmd_port_(port_id, rx_rings_nr, tx_rings_nr) {}

  ~DPDKFactory() { DeInit(); }

  void Init() {
    if (!initialized_) {
      pmd_port_.InitDriver();
      initialized_ = true;
    }
  }

  DPDKSocket* CreateSocket(int queue_id) {
    if (!initialized_) {
      LOG(WARNING) << "DPDKFactory is not initialized";
      return nullptr;
    }

    std::lock_guard<std::mutex> lock(socket_q_lock_);

    RxRing* rx_ring = pmd_port_.GetRing<RxRing>(queue_id);
    TxRing* tx_ring = pmd_port_.GetRing<TxRing>(queue_id);
    PacketPool* packet_pool = tx_ring->GetPacketPool();

    return new DPDKSocket(queue_id, rx_ring, tx_ring, packet_pool);
  }

  void DeInit() {
    if (!initialized_) {
      LOG(WARNING) << "DPDKFactory is not initialized";
      return;
    }
  }

 private:
  bool initialized_;
  PmdPort pmd_port_;
  std::mutex socket_q_lock_;
};

class PacketBuf {
  // Pointing to the next message buffer in the chain.
  PacketBuf* next_;
  // Describing the packet frame address and length.
  Packet* pkt_;
  // The length of the payload in the packet.
  uint32_t payload_len_;
  // The address of the payload in the packet.
  uint8_t* payload_addr_;
  // The sequence number of the packet.
  uint32_t seqno_;
  // Flags to denote the message buffer state.
#define UCCL_MSGBUF_FLAGS_SYN (1 << 0)
#define UCCL_MSGBUF_FLAGS_FIN (1 << 1)
#define UCCL_MSGBUF_FLAGS_TXPULLTIME_FREE (1 << 2)
  uint8_t msg_flags_;

  PacketBuf(Packet* pkt, uint8_t msg_flags) {
    next_ = nullptr;
    pkt_ = pkt;
    msg_flags_ = msg_flags;
    payload_addr_ = nullptr;
    payload_len_ = 0;
    seqno_ = -1;
  }

 public:
  ~PacketBuf() {
    if (payload_addr_) {
      delete[] payload_addr_;
    }
  }

  static PacketBuf* Allocate(Packet* pkt, uint8_t msg_flags) {
    return new PacketBuf(pkt, msg_flags);
  }

  static void Release(PacketBuf* pkt_buf) { delete pkt_buf; }

  Packet* get_pkt() const { return pkt_; }
  void set_pkt(Packet* pkt) { pkt_ = pkt; }
  Packet* pop_pkt() {
    auto res = pkt_;
    pkt_ = nullptr;
    return res;
  }
  uint8_t* get_pkt_addr() const { return pkt_->head_data<uint8_t*>(); }

  uint32_t get_packet_len() const { return pkt_->length(); }

  void set_payload(uint8_t* payload_addr, uint32_t payload_len) {
    DCHECK(payload_addr_ == nullptr);
    payload_addr_ = new uint8_t[payload_len];
    memcpy(payload_addr_, payload_addr, payload_len);
    payload_len_ = payload_len;
  }

  uint8_t* get_payload() const { return payload_addr_; }
  uint32_t get_payload_len() const { return payload_len_; }

  void set_seqno(uint32_t seqno) { seqno_ = seqno; }
  uint32_t get_seqno() const { return seqno_; }

  uint16_t msg_flags() const { return msg_flags_; }

  // Returns true if this is the first in a message.
  bool is_first() const { return (msg_flags_ & UCCL_MSGBUF_FLAGS_SYN) != 0; }
  // Returns true if this is the last in a message.
  bool is_last() const { return (msg_flags_ & UCCL_MSGBUF_FLAGS_FIN) != 0; }
  static bool is_last(uint8_t msg_flags) {
    return (msg_flags & UCCL_MSGBUF_FLAGS_FIN) != 0;
  }

  // Returns the next message buffer index in the chain.
  PacketBuf* next() const { return next_; }
  // Set the next message buffer index in the chain.
  void set_next(PacketBuf* next) { next_ = next; }

  void mark_first() { add_msg_flags(UCCL_MSGBUF_FLAGS_SYN); }
  void mark_last() { add_msg_flags(UCCL_MSGBUF_FLAGS_FIN); }

  void mark_txpulltime_free() {
    add_msg_flags(UCCL_MSGBUF_FLAGS_TXPULLTIME_FREE);
  }
  void mark_not_txpulltime_free() {
    msg_flags_ &= ~UCCL_MSGBUF_FLAGS_TXPULLTIME_FREE;
  }
  bool is_txpulltime_free() {
    return (msg_flags_ & UCCL_MSGBUF_FLAGS_TXPULLTIME_FREE) != 0;
  }

  static uint32_t get_packet_len(Packet* pkt) { return pkt->length(); }

  void clear_fields() {
    next_ = nullptr;
    pkt_ = nullptr;
    msg_flags_ = 0;
  }

  void set_msg_flags(uint16_t flags) { msg_flags_ = flags; }
  void add_msg_flags(uint16_t flags) { msg_flags_ |= flags; }

  std::string to_string() {
    std::stringstream s;
    s << "pkt: 0x" << std::hex << pkt_->head_data<void*>()
      << " packet_len: " << std::dec << pkt_->length()
      << " msg_flags: " << std::dec << std::bitset<8>(msg_flags_);
    return s.str();
  }

  std::string print_chain() {
    std::stringstream s;
    auto* cur = this;
    while (cur && !cur->is_last()) {
      s << cur->to_string();
      cur = cur->next_;
    }
    return s.str();
  }
};
}  // namespace uccl