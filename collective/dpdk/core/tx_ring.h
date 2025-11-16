#ifndef SRC_INCLUDE_TX_RING_H_
#define SRC_INCLUDE_TX_RING_H_

#include "pmd_ring.h"
#include <rte_ethdev.h>

namespace uccl {

class TxRing : public PmdRing {
public:
  TxRing(uint8_t port_id, uint16_t ring_id, uint16_t ndesc)
      : PmdRing(port_id, ring_id, ndesc) {}

  TxRing(uint8_t port_id, uint16_t ring_id, uint16_t ndesc,
         struct rte_eth_txconf txconf)
      : PmdRing(port_id, ring_id, ndesc), conf_(txconf) {}

  TxRing(uint8_t port_id, uint16_t ring_id, uint16_t ndesc,
         struct rte_eth_txconf txconf, uint32_t nmbufs, uint32_t mbuf_sz)
      : PmdRing(port_id, ring_id, ndesc, nmbufs, mbuf_sz), conf_(txconf) {}

  TxRing(TxRing const &) = delete;
  TxRing &operator=(TxRing const &) = delete;

  void Init() {
    int ret = rte_eth_tx_queue_setup(this->GetPortId(), this->GetRingId(),
                                     this->GetDescNum(), SOCKET_ID_ANY, &conf_);
    if (ret != 0) {
      LOG(FATAL) << "rte_eth_tx_queue_setup() faled. Cannot setup TX queue.";
    }
  }

  /**
   * @brief Tries to send a burst of packets through this TX ring.
   *
   * @param pkts Array of packet pointers to send.
   * @param nb_pkts Number of packets to send.
   * @return Number of packets successfully sent.
   */
  uint16_t TrySendPackets(Packet **pkts, uint16_t nb_pkts) const {
    const uint16_t nb_success =
        rte_eth_tx_burst(this->GetPortId(), this->GetRingId(),
                         reinterpret_cast<struct rte_mbuf **>(pkts), nb_pkts);

    // Free not-sent packets. TODO (ilias): This drops packets!
    for (auto i = nb_success; i < nb_pkts; ++i)
      Packet::Free(pkts[i]);
    return nb_success;
  }

  /**
   * @brief Tries to send a burst of packets through this TX ring.
   *
   * @param batch Batch of packets to send.
   */
  uint16_t TrySendPackets(PacketBatch *batch) const {
    const uint16_t ret = TrySendPackets(batch->pkts(), batch->GetSize());
    batch->Clear();
    return ret;
  }

  /**
   * @brief Sends all packets from a PacketBatch through this TX ring. Retries
   * until all are sent.
   *
   * @param pkts Array of packet pointers to send.
   * @param nb_pkts Number of packets to send.
   */
  void SendPackets(Packet **pkts, uint16_t nb_pkts) const {
    uint16_t nb_remaining = nb_pkts;

    do {
      auto index = nb_pkts - nb_remaining;
      auto nb_success = rte_eth_tx_burst(
          this->GetPortId(), this->GetRingId(),
          reinterpret_cast<struct rte_mbuf **>(&pkts[index]), nb_remaining);
      nb_remaining -= nb_success;
    } while (nb_remaining);
  }

  /**
   * @brief Sends all packets from a PacketBatch through this TX ring. Retries
   * until all are sent.
   *
   * @param batch Pointer to the PacketBatch.
   */
  void SendPackets(PacketBatch *batch) const {
    SendPackets(batch->pkts(), batch->GetSize());
    batch->Clear();
  }

private:
  struct rte_eth_txconf conf_;
};
} // namespace uccl
#endif // SRC_INCLUDE_TX_RING_H_