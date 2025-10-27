#ifndef SRC_INCLUDE_PACKET_POOL_H_
#define SRC_INCLUDE_PACKET_POOL_H_

#include "packet.h"

#include <rte_common.h>
#include <rte_config.h>
#include <rte_errno.h>
#include <rte_mbuf.h>
#include <rte_mbuf_core.h>
#include <rte_mempool.h>

#include <cstdint>

namespace uccl {

/**
 * @brief A packet pool class implementation, wrapping around DPDK's mbuf pool.
 */
class PacketPool {
public:
  static constexpr uint32_t kRteDefaultMbufsNum_ =
      2048 - 1; //!< Default number of mbufs.
  static constexpr uint16_t kRteDefaultMbufDataSz_ =
      RTE_MBUF_DEFAULT_BUF_SIZE; //!< Default mbuf data size.
  static constexpr char *kRteDefaultMempoolName =
      nullptr; //!< Default mempool name.

  /** @brief Default constructor is deleted to prevent instantiation. */
  PacketPool() = delete;

  /** @brief Copy constructor is deleted to prevent copying. */
  PacketPool(const PacketPool &) = delete;

  /** @brief Assignment operator is deleted to prevent assignment. */
  PacketPool &operator=(const PacketPool &) = delete;

  /**
   * @brief Initializes the packet pool.
   * @param nmbufs Number of mbufs.
   * @param mbuf_size Size of each mbuf.
   * @param mempool_name Name of the mempool.
   */
  PacketPool(uint32_t nmbufs = kRteDefaultMbufsNum_,
             uint16_t mbuf_size = kRteDefaultMbufDataSz_,
             const char *mempool_name = kRteDefaultMempoolName);
  ~PacketPool();

  /**
   * @return The name of the packet pool.
   */
  const char *GetPacketPoolName() { return mpool_->name; }

  /**
   * @return The data room size of the packet.
   */
  uint32_t GetPacketDataRoomSize() const {
    return rte_pktmbuf_data_room_size(mpool_);
  }

  /**
   * @return The underlying memory pool.
   */
  rte_mempool *GetMemPool() { return mpool_; }

  /**
   * @brief Allocates a packet from the pool.
   * @return Pointer to the allocated packet.
   */
  Packet *PacketAlloc() {
    return reinterpret_cast<Packet *>(rte_pktmbuf_alloc(mpool_));
  }

  /**
   * @brief Allocates multiple packets in bulk.
   * @param pkts Array to store the pointers to allocated packets.
   * @param cnt Count of packets to allocate.
   * @return True if allocation succeeds, false otherwise.
   */
  bool PacketBulkAlloc(Packet **pkts, uint16_t cnt) {
    int ret = rte_pktmbuf_alloc_bulk(
        mpool_, reinterpret_cast<struct rte_mbuf **>(pkts), cnt);
    if (ret == 0) [[likely]]
      return true;
    return false;
  }

  /**
   * @brief Allocates multiple packets in bulk to a batch.
   * @param batch Batch to store the allocated packets.
   * @param cnt Count of packets to allocate.
   * @return True if allocation succeeds, false otherwise.
   */
  bool PacketBulkAlloc(PacketBatch *batch, uint16_t cnt) {
    (void)DCHECK_NOTNULL(batch);
    int ret = rte_pktmbuf_alloc_bulk(
        mpool_, reinterpret_cast<struct rte_mbuf **>(batch->pkts()), cnt);
    if (ret != 0) [[unlikely]]
      return false;

    batch->IncrCount(cnt);
    return true;
  }

  /**
   * @return The total capacity (number of packets) in the pool.
   */
  uint32_t Capacity() { return mpool_->populated_size; }

  /**
   * @return The count of available packets in the pool.
   */
  uint32_t AvailPacketsCount() { return rte_mempool_avail_count(mpool_); }

  uint32_t InUsePacketsCount() {return rte_mempool_in_use_count(mpool_);}
  
private:
  const bool
      is_dpdk_primary_process_; //!< Indicates if it's a DPDK primary process.
  static uint16_t next_id_; //!< Static ID for the next packet pool instance.
  rte_mempool *mpool_;      //!< Underlying rte mbuf pool.
  uint16_t id_;             //!< Unique ID for this packet pool instance.
};

} // namespace uccl

#endif // SRC_INCLUDE_PACKET_POOL_H_