#include <cstdint>
#include <deque>
#include <mutex>
#include <new>

#include "packet_pool.h"
#include "pmd_port.h"
#include "rx_ring.h"
#include "tx_ring.h"

namespace uccl {

class DPDKSocket {
public:
  DPDKSocket(int queue_id, RxRing *rx_ring_, TxRing *tx_ring_,
             PacketPool *packet_pool_) {
    this->queue_id_ = queue_id;
    this->rx_ring_ = rx_ring_;
    this->tx_ring_ = tx_ring_;
    this->packet_pool_ = packet_pool_;
  }

  inline void push_packet(Packet *pkt) { Packet::Free(pkt); }
  inline Packet *pop_packet() { return packet_pool_->PacketAlloc(); }

  uint32_t send_packets(Packet **pkts, uint32_t nb_pkts) {
    return tx_ring_->TrySendPackets(pkts, nb_pkts);
  }

  uint32_t recv_packets(Packet **pkts, uint32_t nb_pkts) {
    return rx_ring_->RecvPackets(pkts, nb_pkts);
  }

  int get_queue_id() const { return queue_id_; }

  // unpulled_tx_pkts
  // send_queue_free_entries
  // send_packets
  // kick_tx_and_pull
  // send_queue_estimated_latency_ns
  // pop_frame
  // recv_packets
  // shutdown
  // recv_packets

private:
  int queue_id_;
  // Designated RX queue for this engine (not shared).
  RxRing *rx_ring_;
  // Designated TX queue for this engine (not shared).
  TxRing *tx_ring_;
  // The following packet pool is used for all TX packets;
  // should not be shared with other engines/threads.
  PacketPool *packet_pool_;
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

  DPDKSocket *CreateSocket(int queue_id) {
    if (!initialized_) {
      LOG(WARNING) << "DPDKFactory is not initialized";
      return nullptr;
    }

    std::lock_guard<std::mutex> lock(socket_q_lock_);

    RxRing *rx_ring = pmd_port_.GetRing<RxRing>(queue_id);
    TxRing *tx_ring = pmd_port_.GetRing<TxRing>(queue_id);
    PacketPool *packet_pool = tx_ring->GetPacketPool();

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

} // namespace uccl