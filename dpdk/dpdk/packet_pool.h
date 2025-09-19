#ifndef SRC_INCLUDE_PACKET_POOL_H_
#define SRC_INCLUDE_PACKET_POOL_H_

#include <cstdint>
#include <packet.h>
#include <rte_common.h>
#include <rte_config.h>
#include <rte_errno.h>
#include <rte_mbuf.h>
#include <rte_mbuf_core.h>

namespace juggler {
namespace dpdk {

[[maybe_unused]] static rte_mempool* CreateSpScPacketPool(
    std::string const& name, uint32_t nmbufs, uint16_t mbuf_data_size) {
  struct rte_mempool* mp;
  struct rte_pktmbuf_pool_private mbp_priv;

  uint16_t const priv_size = 0;
  size_t const elt_size = sizeof(struct rte_mbuf) + priv_size + mbuf_data_size;
  memset(&mbp_priv, 0, sizeof(mbp_priv));
  mbp_priv.mbuf_data_room_size = mbuf_data_size;
  mbp_priv.mbuf_priv_size = priv_size;

  unsigned int const kMemPoolFlags =
      RTE_MEMPOOL_F_SC_GET | RTE_MEMPOOL_F_SP_PUT;
  mp = rte_mempool_create(name.c_str(), nmbufs, elt_size, 0, sizeof(mbp_priv),
                          rte_pktmbuf_pool_init, &mbp_priv, rte_pktmbuf_init,
                          NULL, rte_socket_id(), kMemPoolFlags);
  if (mp == nullptr) {
    LOG(ERROR) << "rte_mempool_create() failed. ";
    return nullptr;
  }

  return mp;
}

/**
 * @brief A packet pool class implementation, wrapping around DPDK's mbuf pool.
 */
class PacketPool {
 public:
  static constexpr uint32_t kRteDefaultMbufsNum_ =
      2048 - 1;  //!< Default number of mbufs.
  static constexpr uint16_t kRteDefaultMbufDataSz_ =
      RTE_MBUF_DEFAULT_BUF_SIZE;  //!< Default mbuf data size.
  static constexpr char* kRteDefaultMempoolName =
      nullptr;  //!< Default mempool name.

  /** @brief Default constructor is deleted to prevent instantiation. */
  PacketPool() = delete;

  /** @brief Copy constructor is deleted to prevent copying. */
  PacketPool(PacketPool const&) = delete;

  /** @brief Assignment operator is deleted to prevent assignment. */
  PacketPool& operator=(PacketPool const&) = delete;

  /**
   * @brief Initializes the packet pool.
   * @param nmbufs Number of mbufs.
   * @param mbuf_size Size of each mbuf.
   * @param mempool_name Name of the mempool.
   */
  PacketPool(uint32_t nmbufs = kRteDefaultMbufsNum_,
             uint16_t mbuf_size = kRteDefaultMbufDataSz_,
             char const* mempool_name = kRteDefaultMempoolName)
      : is_dpdk_primary_process_(rte_eal_process_type() == RTE_PROC_PRIMARY) {
    if (is_dpdk_primary_process_) {
      // Create mempool here, choose the name automatically
      id_ = ++next_id_;
      std::string mpool_name = "mbufpool" + std::to_string(id_);
      LOG(INFO) << "[ALLOC] [type:mempool, name:" << mpool_name
                << ", nmbufs:" << nmbufs << ", mbuf_size:" << mbuf_size << "]";
      // mpool_ = rte_pktmbuf_pool_create(mpool_name.c_str(), nmbufs, 0, 0,
      //                                  mbuf_size, SOCKET_ID_ANY);
      mpool_ = CreateSpScPacketPool(mpool_name, nmbufs, mbuf_size);
      CHECK(mpool_) << "Failed to create packet pool.";
    } else {
      // Lookup mempool created earlier by the primary
      mpool_ = rte_mempool_lookup(mempool_name);
      if (mpool_ == nullptr) {
        LOG(FATAL) << "[LOOKUP] [type: mempool, name: " << mempool_name
                   << "] failed. rte_errno = " << rte_errno << " ("
                   << rte_strerror(rte_errno) << ")";
      } else {
        LOG(INFO) << "[LOOKUP] [type: mempool, name " << mempool_name
                  << "] successful. num mbufs " << mpool_->size
                  << ", mbuf size " << mpool_->elt_size;
      }
    }
  }

  ~PacketPool() {
    LOG(INFO) << "[FREE] [type:mempool, name:" << this->GetPacketPoolName()
              << "]";
    if (is_dpdk_primary_process_) rte_mempool_free(mpool_);
  }

  /**
   * @return The name of the packet pool.
   */
  char const* GetPacketPoolName() { return mpool_->name; }

  /**
   * @return The data room size of the packet.
   */
  uint32_t GetPacketDataRoomSize() const {
    return rte_pktmbuf_data_room_size(mpool_);
  }

  /**
   * @return The underlying memory pool.
   */
  rte_mempool* GetMemPool() { return mpool_; }

  /**
   * @brief Allocates a packet from the pool.
   * @return Pointer to the allocated packet.
   */
  Packet* PacketAlloc() {
    return reinterpret_cast<Packet*>(rte_pktmbuf_alloc(mpool_));
  }

  /**
   * @brief Allocates multiple packets in bulk.
   * @param pkts Array to store the pointers to allocated packets.
   * @param cnt Count of packets to allocate.
   * @return True if allocation succeeds, false otherwise.
   */
  bool PacketBulkAlloc(Packet** pkts, uint16_t cnt) {
    int ret = rte_pktmbuf_alloc_bulk(
        mpool_, reinterpret_cast<struct rte_mbuf**>(pkts), cnt);
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
  bool PacketBulkAlloc(PacketBatch* batch, uint16_t cnt) {
    (void)DCHECK_NOTNULL(batch);
    int ret = rte_pktmbuf_alloc_bulk(
        mpool_, reinterpret_cast<struct rte_mbuf**>(batch->pkts()), cnt);
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

 private:
  bool const
      is_dpdk_primary_process_;  //!< Indicates if it's a DPDK primary process.
  static uint16_t next_id_;  //!< Static ID for the next packet pool instance.
  rte_mempool* mpool_;       //!< Underlying rte mbuf pool.
  uint16_t id_;              //!< Unique ID for this packet pool instance.
};

uint16_t PacketPool::next_id_ = 0;

}  // namespace dpdk
}  // namespace juggler

#endif  // SRC_INCLUDE_PACKET_POOL_H_
