#include "packet_pool.h"
#include <glog/logging.h>
#include <string>
#include <rte_mbuf.h>
#include <rte_mbuf_pool_ops.h>
#include <rte_mempool.h>

namespace uccl {

[[maybe_unused]] static rte_mempool* CreateSpScPacketPool(
    std::string const& name, uint32_t nmbufs, uint16_t mbuf_data_size) {
  struct rte_mempool* mp;
  struct rte_pktmbuf_pool_private mbp_priv;

  const uint16_t priv_size = 0;
  const size_t elt_size = sizeof(struct rte_mbuf) + priv_size + mbuf_data_size;
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

uint16_t PacketPool::next_id_ = 0;

// 'id' of the PacketPool usually refers to the thread id.
// 'nmbufs' is the number of mbufs to allocate in the backing pool.
// 'mbuf_size' the size of an mbuf buffer. (MBUF_DATASZ_DEFAULT is the minimum)
PacketPool::PacketPool(uint32_t nmbufs, uint16_t mbuf_size,
                       char const* mempool_name)
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
                << "] successful. num mbufs " << mpool_->size << ", mbuf size "
                << mpool_->elt_size;
    }
  }
}

PacketPool::~PacketPool() {
  LOG(INFO) << "[FREE] [type:mempool, name:" << this->GetPacketPoolName()
            << "]";
  if (is_dpdk_primary_process_) rte_mempool_free(mpool_);
}

}  // namespace uccl