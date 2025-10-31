#include "pmd_port.h"
#include "ether.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace uccl {
[[maybe_unused]] static void FetchDpdkPortInfo(uint8_t port_id,
                                               struct rte_eth_dev_info *devinfo,
                                               Ethernet::Address *lladdr) {
  if (!rte_eth_dev_is_valid_port(port_id)) {
    LOG(INFO) << "Port id " << static_cast<int>(port_id) << " is not valid.";
    return;
  }

  int ret = rte_eth_dev_info_get(port_id, devinfo);
  if (ret != 0) {
    LOG(WARNING) << "rte_eth_dev_info() failed. Cannot retrieve eth device "
                    "contextual info for port "
                 << static_cast<int>(port_id);
    return;
  }
  CHECK_NOTNULL(devinfo->device);

  rte_eth_macaddr_get(port_id,
                      reinterpret_cast<rte_ether_addr *>(lladdr->bytes));

  LOG(INFO) << "[PMDPORT] [port_id: " << static_cast<uint32_t>(port_id)
            << ", driver: " << devinfo->driver_name
            << ", RXQ: " << devinfo->max_rx_queues
            << ", TXQ: " << devinfo->max_tx_queues
            << ", l2addr: " << lladdr->ToString() << "]";
}

static rte_eth_conf DefaultEthConf(const rte_eth_dev_info *devinfo) {
  CHECK_NOTNULL(devinfo);

  struct rte_eth_conf port_conf = rte_eth_conf();

  // The `net_null' driver is only used for testing, and it does not support
  // offloads so return a very basic ethernet configuration.
  if (std::string(devinfo->driver_name) == "net_null")
    return port_conf;

  port_conf.link_speeds = RTE_ETH_LINK_SPEED_AUTONEG;
  uint64_t rss_hf =
      RTE_ETH_RSS_IP | RTE_ETH_RSS_UDP | RTE_ETH_RSS_TCP | RTE_ETH_RSS_SCTP;
  if (devinfo->flow_type_rss_offloads) {
    rss_hf &= devinfo->flow_type_rss_offloads;
  }

  port_conf.lpbk_mode = 1;
  port_conf.rxmode.mq_mode = RTE_ETH_MQ_RX_RSS;

  port_conf.rxmode.mtu = PmdRing::kDefaultFrameSize;
  port_conf.rxmode.max_lro_pkt_size = PmdRing::kDefaultFrameSize;
  const auto rx_offload_capa = devinfo->rx_offload_capa;
  port_conf.rxmode.offloads |= ((RTE_ETH_RX_OFFLOAD_CHECKSUM)&rx_offload_capa);

  port_conf.rx_adv_conf.rss_conf = {
      .rss_key = nullptr,
      .rss_key_len = devinfo->hash_key_size,
      .rss_hf = rss_hf,
      .algorithm = RTE_ETH_HASH_FUNCTION_DEFAULT,
  };

  const auto tx_offload_capa = devinfo->tx_offload_capa;
  if (!(tx_offload_capa & RTE_ETH_TX_OFFLOAD_IPV4_CKSUM) ||
      !(tx_offload_capa & RTE_ETH_TX_OFFLOAD_UDP_CKSUM)) {
    // Making this fatal; not sure what NIC does not support checksum offloads.
    LOG(FATAL) << "Hardware does not support checksum offloads.";
  }

  port_conf.txmode.mq_mode = RTE_ETH_MQ_TX_NONE;
  port_conf.txmode.offloads =
      (RTE_ETH_TX_OFFLOAD_IPV4_CKSUM | RTE_ETH_TX_OFFLOAD_UDP_CKSUM);

  if (tx_offload_capa & RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE) {
    // TODO(ilias): Add option to the constructor to enable this offload.
    LOG(WARNING)
        << "Enabling FAST FREE: use always the same mempool for each queue.";
    port_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE;
  }

  return port_conf;
}

/**
 * @brief Factory function to create a ring.
 * @tparam T Type of the ring to be created.
 * @tparam Args Types of the arguments.
 * @param params Arguments for ring's constructor.
 * @return A unique pointer to the created ring.
 */
template <typename T, typename... Args>
decltype(auto) makeRing(Args &&...params) {
  std::unique_ptr<T> ptr(nullptr);
  ptr.reset(new T(std::forward<Args>(params)...));
  return ptr;
}

void PmdPort::InitDriver(uint16_t mtu) {
  if (is_dpdk_primary_process_) {
    // Get DPDK port info.
    FetchDpdkPortInfo(port_id_, &devinfo_, &l2_addr_);
    device_ = devinfo_.device;

    LOG(INFO) << "Rings nr: " << rx_rings_nr_;
    const rte_eth_conf portconf = DefaultEthConf(&devinfo_);
    int ret =
        rte_eth_dev_configure(port_id_, rx_rings_nr_, tx_rings_nr_, &portconf);
    if (ret != 0) {
      LOG(FATAL) << "rte_eth_dev_configure() failed. Cannot configure port id: "
                 << static_cast<int>(port_id_);
    }

    // Check if the MTU is set correctly.
    CHECK(GetMTU().has_value())
        << "Failed to get MTU for port " << static_cast<int>(port_id_);
    if (mtu != GetMTU().value()) {
      // If there is a mismatch, try to set the MTU.
      ret = rte_eth_dev_set_mtu(port_id_, mtu);
      if (ret != 0) {
        LOG(FATAL) << "Failed to set MTU for port "
                   << static_cast<int>(port_id_) << ". Error "
                   << rte_strerror(ret);
      }
    }

    // Try to get the RSS configuration from the device.
    rss_hash_key_.resize(devinfo_.hash_key_size, 0);
    struct rte_eth_rss_conf rss_conf;
    rss_conf.rss_key = rss_hash_key_.data();
    rss_conf.rss_key_len = devinfo_.hash_key_size;
    ret = rte_eth_dev_rss_hash_conf_get(port_id_, &rss_conf);
    if (ret != 0) {
      LOG(WARNING) << "Failed to get RSS configuration for port "
                   << static_cast<int>(port_id_) << ". Error "
                   << rte_strerror(ret);
    }

    rss_reta_conf_.resize(devinfo_.reta_size / RTE_ETH_RETA_GROUP_SIZE,
                          {-1ull, {0}});

    for (auto i = 0u; i < devinfo_.reta_size; i++) {
      // Initialize the RETA table in a round-robin fashion.
      auto index = i / RTE_ETH_RETA_GROUP_SIZE;
      auto shift = i % RTE_ETH_RETA_GROUP_SIZE;
      rss_reta_conf_[index].reta[shift] = i % rx_rings_nr_;
      rss_reta_conf_[index].mask |= (1 << shift);
    }

    ret = rte_eth_dev_rss_reta_update(port_id_, rss_reta_conf_.data(),
                                      devinfo_.reta_size);
    if (ret != 0) {
      // By default the RSS RETA table is configured and it works when the
      // number of RX queues is a power of two. In case of non-power-of-two it
      // seems that RSS is not behaving as expected, although it should be
      // supported by 'mlx5' drivers according to the documentation.
      //
      // Explicitly updating the RSS RETA table with the default configuration
      // seems to fix the issue.
      LOG(WARNING) << "Failed to update RSS RETA configuration for port "
                   << static_cast<int>(port_id_) << ". Error "
                   << rte_strerror(ret);
    }

    ret = rte_eth_dev_rss_reta_query(port_id_, rss_reta_conf_.data(),
                                     devinfo_.reta_size);
    if (ret != 0) {
      LOG(WARNING) << "Failed to get RSS RETA configuration for port "
                   << static_cast<int>(port_id_) << ". Error "
                   << rte_strerror(ret);
    }

    LOG(INFO) << Format("RSS indirection table (size %d):\n",
                        devinfo_.reta_size);
    for (auto i = 0u; i < devinfo_.reta_size; i++) {
      const auto kColumns = 8;
      auto index = i / RTE_ETH_RETA_GROUP_SIZE;
      auto shift = i % RTE_ETH_RETA_GROUP_SIZE;
      if (!(rss_reta_conf_[index].mask & (1 << shift))) {
        LOG(WARNING) << "Rss reta conf mask is not set for index " << index
                     << " and shift " << shift;
        continue;
      }

      std::string reta_table;
      if (i % kColumns == 0) {
        reta_table += std::to_string(i) + ":\t" +
                      std::to_string(rss_reta_conf_[index].reta[shift]);
      } else if (i % kColumns == kColumns - 1) {
        reta_table +=
            "\t" + std::to_string(rss_reta_conf_[index].reta[shift]) + "\n";
      } else {
        reta_table += "\t" + std::to_string(rss_reta_conf_[index].reta[shift]);
      }

      std::cout << reta_table;
    }

    ret = rte_eth_dev_adjust_nb_rx_tx_desc(port_id_, &rx_ring_desc_nr_,
                                           &tx_ring_desc_nr_);
    if (ret != 0) {
      LOG(FATAL)
          << "rte_eth_dev_adjust_nb_rx_tx_desc() failed for port with id: "
          << static_cast<int>(port_id_);
    }

    const auto mbuf_data_size =
        mtu + RTE_ETHER_HDR_LEN + RTE_ETHER_CRC_LEN + RTE_PKTMBUF_HEADROOM;

    // Setup the TX queues.
    for (auto q = 0; q < tx_rings_nr_; q++) {
      LOG(INFO) << "Initializing TX ring: " << q;
      auto tx_ring = makeRing<TxRing>(port_id_, q, tx_ring_desc_nr_,
                                      devinfo_.default_txconf,
                                      2 * tx_ring_desc_nr_ - 1, mbuf_data_size);
      // auto tx_ring = makeRing<TxRing>(this, port_id_, q, tx_ring_desc_nr_,
      //                                 devinfo_.default_txconf);
      tx_ring.get()->Init();
      tx_rings_.emplace_back(std::move(tx_ring));
    }

    // Setup the RX queues.
    for (auto q = 0; q < rx_rings_nr_; q++) {
      LOG(INFO) << "Initializing RX ring: " << q;
      auto rx_ring = makeRing<RxRing>(port_id_, q, rx_ring_desc_nr_,
                                      devinfo_.default_rxconf,
                                      2 * rx_ring_desc_nr_ - 1, mbuf_data_size);
      rx_ring.get()->Init();
      rx_rings_.emplace_back(std::move(rx_ring));
    }

    // TODO(Nelson): Cannot Enable Promiscuous Mode.
    LOG(INFO) << "Promiscuous mode: " << rte_eth_promiscuous_get(port_id_);
    // ret = rte_eth_promiscuous_enable(port_id_);
    // if (ret != 0)
    //   LOG(WARNING) << "rte_eth_promiscuous_enable() failed.";
    // else
    //   LOG(INFO) << "Promiscuous mode enabled.";

    ret = rte_eth_stats_reset(port_id_);
    if (ret != 0)
      LOG(WARNING) << "Failed to reset port statistics.";

    ret = rte_eth_dev_set_link_up(port_id_);
    if (ret != 0)
      LOG(WARNING) << "rte_eth_dev_set_link_up() failed.";

    ret = rte_eth_dev_start(port_id_);
    if (ret != 0) {
      LOG(FATAL) << "rte_eth_dev_start() failed.";
    }

    LOG(INFO) << "Waiting for link to get up...";
    struct rte_eth_link link;
    memset(&link, '0', sizeof(link));
    int nsecs = 30;
    while (nsecs-- && link.link_status == RTE_ETH_LINK_DOWN) {
      memset(&link, '0', sizeof(link));
      int ret = rte_eth_link_get_nowait(port_id_, &link);
      if (ret != 0) {
        LOG(WARNING) << "rte_eth_link_get_nowait() failed.";
      }

      sleep(1);
    }

    if (link.link_status == RTE_ETH_LINK_UP) {
      LOG(INFO) << "[PMDPORT: " << static_cast<int>(port_id_) << "] "
                << "Link is UP " << link.link_speed
                << (link.link_autoneg ? " (AutoNeg)" : " (Fixed)")
                << (link.link_duplex ? " Full Duplex" : " Half Duplex");
    } else {
      LOG(INFO) << "[PMDPORT: " << static_cast<int>(port_id_) << "] "
                << "Link is DOWN.";
    }
  } else {
    FetchDpdkPortInfo(port_id_, &devinfo_, &l2_addr_);

    // For the rings, just set port and queue IDs here, which have been
    // pre-initialized by the DPDK primary process
    for (auto q = 0; q < tx_rings_nr_; q++) {
      tx_rings_.emplace_back(
          makeRing<TxRing>(port_id_, q, tx_ring_desc_nr_));
    }

    for (auto q = 0; q < rx_rings_nr_; q++) {
      rx_rings_.emplace_back(
          makeRing<RxRing>(port_id_, q, rx_ring_desc_nr_));
    }
  }

  // Mark port as initialized.
  initialized_ = true;
}

void PmdPort::UpdatePortStats() {
  int ret = rte_eth_stats_get(port_id_, &port_stats_);
  if (ret != 0) {
    LOG(WARNING) << "Failed to retrieve DPDK port stats.";
    memset(&port_stats_, 0, sizeof(port_stats_));
  }
}

void PmdPort::DeInit() {
  if (!initialized_ || !is_dpdk_primary_process_)
    return;
  rte_eth_dev_stop(port_id_);
  rte_eth_dev_close(port_id_);
  LOG(INFO) << Format("[PMDPORT: %u closed.]", port_id_);
  initialized_ = false;
}
} // namespace uccl