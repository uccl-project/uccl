#ifndef SRC_INCLUDE_RX_RING_H_
#define SRC_INCLUDE_RX_RING_H_

#include "pmd_ring.h"
#include <rte_ethdev.h>

namespace uccl {
/**
 * @brief Represents a RX ring in DPDK.
 *
 * Provides functionalities specific to RX rings in DPDK.
 */
class RxRing : public PmdRing {
 public:
  RxRing(uint8_t port_id, uint16_t ring_id, uint16_t ndesc)
      : PmdRing(port_id, ring_id, ndesc) {}

  RxRing(uint8_t port_id, uint16_t ring_id, uint16_t ndesc,
         struct rte_eth_rxconf rxconf, uint32_t nmbufs, uint32_t mbuf_sz)
      : PmdRing(port_id, ring_id, ndesc, nmbufs, mbuf_sz), conf_(rxconf) {}

  RxRing(RxRing const&) = delete;
  RxRing& operator=(RxRing const&) = delete;

  void Init() {
    int ret = rte_eth_rx_queue_setup(this->GetPortId(), this->GetRingId(),
                                     this->GetDescNum(), SOCKET_ID_ANY, &conf_,
                                     this->GetPacketMemPool());
    if (ret != 0) {
      LOG(FATAL) << "rte_eth_rx_queue_setup() faled. Cannot setup RX queue.";
    }
  }

  /**
   * @brief Receives a burst of packets from this RX ring.
   *
   * This method fetches packets that have arrived on the RX ring
   * up to the number specified by nb_pkts. The received packets are
   * stored in the provided pkts array.
   *
   * @param pkts Array of packet pointers to store the received packets.
   * @param nb_pkts Maximum number of packets to receive.
   * @return Number of packets successfully received.
   */
  uint16_t RecvPackets(Packet** pkts, uint16_t nb_pkts) {
    return rte_eth_rx_burst(this->GetPortId(), this->GetRingId(),
                            reinterpret_cast<struct rte_mbuf**>(pkts), nb_pkts);
  }

  /**
   * @brief Receives a burst of packets from this RX ring.
   *
   * This method fetches packets that have arrived on the RX ring
   * up to the number specified by nb_pkts. The received packets are
   * stored in the provided batch; the number of packets added is at most the
   * amount of available slots remaining in the batch.
   *
   * @param batch Batch of packets to store the received packets.
   * @return Number of packets successfully received.
   */
  uint16_t RecvPackets(PacketBatch* batch) {
    const uint16_t nb_rx = RecvPackets(batch->pkts(), batch->GetRoom());
    batch->IncrCount(nb_rx);
    return nb_rx;
  }

 private:
  struct rte_eth_rxconf conf_;
};
}  // namespace uccl

#endif  // SRC_INCLUDE_RX_RING_H_