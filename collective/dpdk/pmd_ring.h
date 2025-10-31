#ifndef SRC_INCLUDE_PMD_RING_H_
#define SRC_INCLUDE_PMD_RING_H_

#include "packet.h"
#include "packet_pool.h"

#include <cstdint>
#include <memory>

#include <rte_bus_pci.h>
#include <rte_ethdev.h>

namespace uccl {

class PmdRing {
public:
  static const inline uint16_t kDefaultFrameSize = 1500;
  static const inline uint16_t kJumboFrameSize =
      9000 - RTE_ETHER_HDR_LEN - RTE_ETHER_CRC_LEN;
  static const inline uint16_t kDefaultRingDescNr = 512;

  /**
   * @brief Default and copy constructors are deleted to prevent instantiation
   * and copying.
   */
  PmdRing() = delete;
  PmdRing(PmdRing const &) = delete;
  PmdRing &operator=(PmdRing const &) = delete;

  virtual ~PmdRing() = default;

  PacketPool *GetPacketPool() const { return ppool_.get(); }
  uint16_t GetDescNum() const { return ndesc_; }
  uint8_t GetPortId() const { return port_id_; }
  uint16_t GetRingId() const { return ring_id_; }

protected:
  // Only TX rings can be initialized without a packetpool attached.
  PmdRing(uint8_t port_id, uint16_t ring_id, uint16_t ndesc)
      : port_id_(port_id), ring_id_(ring_id), ndesc_(ndesc), ppool_(nullptr) {}

  PmdRing(uint8_t port_id, uint16_t ring_id, uint16_t ndesc, uint32_t nmbufs,
          uint32_t mbuf_sz)
      : port_id_(port_id), ring_id_(ring_id), ndesc_(ndesc),
        ppool_(std::unique_ptr<PacketPool>(new PacketPool(nmbufs, mbuf_sz))) {}

  rte_mempool *GetPacketMemPool() const { return ppool_.get()->GetMemPool(); }

private:
  const uint8_t port_id_;
  const uint16_t ring_id_;
  const uint16_t ndesc_;
  const std::unique_ptr<PacketPool> ppool_;
};
} // namespace uccl

#endif // SRC_INCLUDE_PMD_RING_H_