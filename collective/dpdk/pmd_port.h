#ifndef SRC_INCLUDE_PMD_PORT_H_
#define SRC_INCLUDE_PMD_PORT_H_

#include "pmd_ring.h"
#include "rx_ring.h"
#include "tx_ring.h"
#include "util/util.h"

namespace uccl {
/**
 * @brief Represents a DPDK Port with functionalities to manage RX and TX rings.
 *
 * Provides a higher-level abstraction over a DPDK port, encapsulating
 * operations such as initialization, deinitialization, statistics gathering,
 * and other related functions.
 */
class PmdPort {
public:
  static const uint16_t kDefaultRingNr_ = 1;

  /**
   * @brief Constructor to initialize a PmdPort with given parameters.
   *
   * @param id Port identifier.
   * @param rx_rings_nr (Optional) Number of RX rings. Default is
   * kDefaultRingNr_.
   * @param tx_rings_nr (Optional) Number of TX rings. Default is
   * kDefaultRingNr_.
   * @param rx_desc_nr (Optional) Number of RX descriptors. Default is
   * PmdRing::kDefaultRingDescNr.
   * @param tx_desc_nr (Optional) Number of TX descriptors. Default is
   * PmdRing::kDefaultRingDescNr.
   */
  PmdPort(uint16_t id, uint16_t rx_rings_nr = kDefaultRingNr_,
          uint16_t tx_rings_nr = kDefaultRingNr_,
          uint16_t rx_desc_nr = PmdRing::kDefaultRingDescNr,
          uint16_t tx_desc_nr = PmdRing::kDefaultRingDescNr)
      : is_dpdk_primary_process_(rte_eal_process_type() == RTE_PROC_PRIMARY),
        port_id_(id), tx_rings_nr_(tx_rings_nr), rx_rings_nr_(rx_rings_nr),
        tx_ring_desc_nr_(tx_desc_nr), rx_ring_desc_nr_(rx_desc_nr),
        initialized_(false) {
    // Get L2 address.
    rte_ether_addr temp;
    CHECK_EQ(rte_eth_macaddr_get(port_id_, &temp), 0);
    l2_addr_.FromUint8(temp.addr_bytes);
  }

  /**
   * @brief Deleted copy constructor and assignment operator.
   */
  PmdPort(PmdPort const &) = delete;
  PmdPort &operator=(PmdPort const &) = delete;

  /**
   * @brief Destructor. Deinitializes the port if it was initialized.
   */
  ~PmdPort() { DeInit(); }

  /**
   * @brief Initializes the driver.
   *
   * @param mtu (Optional) Maximum Transmission Unit to set for the port.
   * Default is PmdRing::kDefaultFrameSize.
   */
  void InitDriver(uint16_t mtu = PmdRing::kDefaultFrameSize);

  /**
   * @brief Deinitializes the port.
   */
  void DeInit();

  /**
   * @brief Checks if the port has been initialized.
   *
   * @return True if the port is initialized, false otherwise.
   */
  bool IsInitialized() const { return initialized_; }

  uint16_t GetPortId() const { return port_id_; }

  std::string GetDriverName() const {
    return Format("%s", devinfo_.driver_name);
  }

  /**
   * @brief Retrieves the associated device for this port.
   *
   * @return Pointer to the associated device.
   */
  rte_device *GetDevice() const { return device_; }

  template <typename T> decltype(auto) GetRing(uint16_t id) const {
    constexpr bool is_tx_ring = std::is_same<T, TxRing>::value;
    CHECK_LT(id, (is_tx_ring ? tx_rings_nr_ : rx_rings_nr_)) << "Out-of-bounds";
    return is_tx_ring ? static_cast<T *>(tx_rings_.at(id).get())
                      : static_cast<T *>(rx_rings_.at(id).get());
  }

  /**
   * @brief Retrieves the Maximum Transmission Unit (MTU) for the port.
   *
   * @return An optional containing the MTU value if successful, or std::nullopt
   * if there was an error.
   */
  std::optional<uint16_t> GetMTU() const {
    uint16_t mtu;
    int ret = rte_eth_dev_get_mtu(port_id_, &mtu);
    if (ret != 0)
      return std::nullopt; // Error (wrong port id?)
    return mtu;
  }

  /**
   * @brief Retrieves the MAC address of the port as set in the hardware.
   *
   * @return `Ethernet' address representing the MAC address.
   */
  Ethernet::Address GetL2Addr() const { return l2_addr_; }

  /**
   * @brief Retrieves the port's RSS (Receive Side Scaling) key.
   *
   * @return A vector containing the RSS hash key bytes.
   */
  const std::vector<uint8_t> &GetRSSKey() const { return rss_hash_key_; }

  /**
   * @brief Calculates the landing RX queue for a given RSS hash.
   *
   * @param rss_hash The RSS hash value.
   * @return The index of the RX queue.
   */
  uint16_t GetRSSRxQueue(uint32_t rss_hash) const {
    auto lsb = rss_hash & (devinfo_.reta_size - 1);
    auto index = lsb / RTE_ETH_RETA_GROUP_SIZE;
    auto shift = lsb % RTE_ETH_RETA_GROUP_SIZE;
    LOG(INFO) << "index: " << index << " shift: " << shift
              << "rss_hash: " << rss_hash
              << " reta_size: " << devinfo_.reta_size
              << " reta_group_size: " << RTE_ETH_RETA_GROUP_SIZE
              << " reta: " << rss_reta_conf_[index].reta[shift]
              << " lsb: " << lsb;
    return rss_reta_conf_[index].reta[shift];
  }

  /**
   * @brief Retrieves the number of RX queues for the port.
   *
   * @return Number of RX queues.
   */
  uint16_t GetRxQueuesNr() const { return rx_rings_nr_; }

  /**
   * @brief Updates the statistics for this port.
   */
  void UpdatePortStats();

  uint64_t GetPortRxPkts() const { return port_stats_.ipackets; }

  uint64_t GetPortRxBytes() const { return port_stats_.ibytes; }

  uint64_t GetPortTxPkts() const { return port_stats_.opackets; }

  uint64_t GetPortTxBytes() const { return port_stats_.obytes; }

  uint64_t GetPortRxDrops() const { return port_stats_.imissed; }

  uint64_t GetPortTxDrops() const { return port_stats_.oerrors; }

  uint64_t GetPortRxNoMbufErr() const { return port_stats_.rx_nombuf; }

  uint64_t GetPortQueueRxPkts(uint16_t queue_id) const {
    CHECK_LT(queue_id,
             std::min(rx_rings_.size(),
                      static_cast<size_t>(RTE_ETHDEV_QUEUE_STAT_CNTRS)));
    return port_stats_.q_ipackets[queue_id];
  }

  uint64_t GetPortQueueRxBytes(uint16_t queue_id) const {
    CHECK_LT(queue_id,
             std::min(rx_rings_.size(),
                      static_cast<size_t>(RTE_ETHDEV_QUEUE_STAT_CNTRS)));
    return port_stats_.q_ibytes[queue_id];
  }

  uint64_t GetPortQueueTxPkts(uint16_t queue_id) const {
    CHECK_LT(queue_id,
             std::min(tx_rings_.size(),
                      static_cast<size_t>(RTE_ETHDEV_QUEUE_STAT_CNTRS)));
    return port_stats_.q_opackets[queue_id];
  }

  uint64_t GetPortQueueTxBytes(uint16_t queue_id) const {
    CHECK_LT(queue_id,
             std::min(tx_rings_.size(),
                      static_cast<size_t>(RTE_ETHDEV_QUEUE_STAT_CNTRS)));
    return port_stats_.q_obytes[queue_id];
  }

  void DumpStats() {
    UpdatePortStats();
    LOG(INFO) << Format(
        "[STATS - Port: %u] [TX] Pkts: %lu, Bytes: %lu, Drops: %lu [RX] Pkts: "
        "%lu, Bytes: %lu, Drops: %lu, NoRXMbufs: %lu",
        port_id_, GetPortTxPkts(), GetPortTxBytes(), GetPortTxDrops(),
        GetPortRxPkts(), GetPortRxBytes(), GetPortRxDrops(),
        GetPortRxNoMbufErr());

    for (uint16_t i = 0; i < tx_rings_nr_; i++) {
      LOG(INFO) << Format(
          "[STATS - Port: %u, Queue: %u] [TX] Pkts: %lu, Bytes: %lu", port_id_,
          i, GetPortQueueTxPkts(i), GetPortQueueTxBytes(i));
    }

    for (uint16_t i = 0; i < rx_rings_nr_; i++) {
      LOG(INFO) << Format(
          "[STATS - Port: %u, Queue: %u] [RX] Pkts: %lu, Bytes: %lu", port_id_,
          i, GetPortQueueRxPkts(i), GetPortQueueRxBytes(i));
    }
  }

private:
  const bool is_dpdk_primary_process_;
  const uint16_t port_id_;
  const uint16_t tx_rings_nr_, rx_rings_nr_;
  uint16_t tx_ring_desc_nr_, rx_ring_desc_nr_;
  std::vector<std::unique_ptr<PmdRing>> tx_rings_, rx_rings_;

  Ethernet::Address l2_addr_;
  struct rte_eth_dev_info devinfo_;
  rte_device *device_;
  std::vector<rte_eth_rss_reta_entry64> rss_reta_conf_;
  struct rte_eth_stats port_stats_;
  std::vector<uint8_t> rss_hash_key_;
  bool initialized_;
};
} // namespace uccl
#endif // SRC_INCLUDE_PMD_PORT_H_