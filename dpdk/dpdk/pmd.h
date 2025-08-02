#ifndef SRC_INCLUDE_PMD_H_
#define SRC_INCLUDE_PMD_H_

#include "dpdk.h"
#include "ether.h"
#include "packet.h"
#include "packet_pool.h"
#include <glog/logging.h>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <rte_bus_pci.h>
#include <rte_ethdev.h>

namespace juggler {
namespace dpdk {

static rte_eth_conf DefaultEthConf(rte_eth_dev_info const* devinfo) {
  CHECK_NOTNULL(devinfo);

  struct rte_eth_conf port_conf = rte_eth_conf();

  // The `net_null' driver is only used for testing, and it does not support
  // offloads so return a very basic ethernet configuration.
  if (std::string(devinfo->driver_name) == "net_null") return port_conf;

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
  auto const rx_offload_capa = devinfo->rx_offload_capa;
  port_conf.rxmode.offloads |= ((RTE_ETH_RX_OFFLOAD_CHECKSUM)&rx_offload_capa);

  port_conf.rx_adv_conf.rss_conf = {
      .rss_key = nullptr,
      .rss_key_len = devinfo->hash_key_size,
      .rss_hf = rss_hf,
  };

  auto const tx_offload_capa = devinfo->tx_offload_capa;
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

// Forward declarations.
class PmdPort;

/**
 * @brief Base class for RX and TX rings.
 *
 * Represents an abstraction for RX and TX rings used in DPDK.
 */
class PmdRing {
 public:
  static inline uint16_t const kDefaultFrameSize = 1500;
  static inline uint16_t const kJumboFrameSize =
      9000 - RTE_ETHER_HDR_LEN - RTE_ETHER_CRC_LEN;
  static inline uint16_t const kDefaultRingDescNr = 512;

  /**
   * @brief Default and copy constructors are deleted to prevent instantiation
   * and copying.
   */
  PmdRing() = delete;
  PmdRing(PmdRing const&) = delete;
  PmdRing& operator=(PmdRing const&) = delete;

  virtual ~PmdRing() = default;

  PmdPort const* GetPmdPort() const { return pmd_port_; }
  PacketPool* GetPacketPool() const { return ppool_.get(); }
  uint16_t GetDescNum() const { return ndesc_; }
  uint8_t GetPortId() const { return port_id_; }
  uint16_t GetRingId() const { return ring_id_; }

 protected:
  // Only TX rings can be initialized without a packetpool attached.
  PmdRing(PmdPort const* port, uint8_t port_id, uint16_t ring_id,
          uint16_t ndesc)
      : pmd_port_(port),
        port_id_(port_id),
        ring_id_(ring_id),
        ndesc_(ndesc),
        ppool_(nullptr) {}
  PmdRing(PmdPort const* port, uint8_t port_id, uint16_t ring_id,
          uint16_t ndesc, uint32_t nmbufs, uint32_t mbuf_sz)
      : pmd_port_(port),
        port_id_(port_id),
        ring_id_(ring_id),
        ndesc_(ndesc),
        ppool_(std::unique_ptr<PacketPool>(new PacketPool(nmbufs, mbuf_sz))) {}

  rte_mempool* GetPacketMemPool() const { return ppool_.get()->GetMemPool(); }

 private:
  PmdPort const* pmd_port_;
  uint8_t const port_id_;
  uint16_t const ring_id_;
  uint16_t const ndesc_;
  std::unique_ptr<PacketPool> const ppool_;
};

/*
 * @brief Represents a TX ring in DPDK.
 *
 * Provides functionalities specific to TX rings (e.g., sending packets).
 */
class TxRing : public PmdRing {
 public:
  TxRing(PmdPort const* pmd_port, uint8_t port_id, uint16_t ring_id,
         uint16_t ndesc)
      : PmdRing(pmd_port, port_id, ring_id, ndesc) {}

  TxRing(PmdPort const* pmd_port, uint8_t port_id, uint16_t ring_id,
         uint16_t ndesc, struct rte_eth_txconf txconf)
      : PmdRing(pmd_port, port_id, ring_id, ndesc), conf_(txconf) {}

  TxRing(PmdPort const* pmd_port, uint8_t port_id, uint16_t ring_id,
         uint16_t ndesc, struct rte_eth_txconf txconf, uint32_t nmbufs,
         uint32_t mbuf_sz)
      : PmdRing(pmd_port, port_id, ring_id, ndesc, nmbufs, mbuf_sz),
        conf_(txconf) {}

  TxRing(TxRing const&) = delete;
  TxRing& operator=(TxRing const&) = delete;

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
  uint16_t TrySendPackets(Packet** pkts, uint16_t nb_pkts) const {
    uint16_t const nb_success =
        rte_eth_tx_burst(this->GetPortId(), this->GetRingId(),
                         reinterpret_cast<struct rte_mbuf**>(pkts), nb_pkts);

    // Free not-sent packets. TODO (ilias): This drops packets!
    for (auto i = nb_success; i < nb_pkts; ++i) Packet::Free(pkts[i]);
    return nb_success;
  }

  /**
   * @brief Tries to send a burst of packets through this TX ring.
   *
   * @param batch Batch of packets to send.
   */
  uint16_t TrySendPackets(PacketBatch* batch) const {
    uint16_t const ret = TrySendPackets(batch->pkts(), batch->GetSize());
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
  void SendPackets(Packet** pkts, uint16_t nb_pkts) const {
    uint16_t nb_remaining = nb_pkts;

    do {
      auto index = nb_pkts - nb_remaining;
      auto nb_success = rte_eth_tx_burst(
          this->GetPortId(), this->GetRingId(),
          reinterpret_cast<struct rte_mbuf**>(&pkts[index]), nb_remaining);
      nb_remaining -= nb_success;
    } while (nb_remaining);
  }

  /**
   * @brief Sends all packets from a PacketBatch through this TX ring. Retries
   * until all are sent.
   *
   * @param batch Pointer to the PacketBatch.
   */
  void SendPackets(PacketBatch* batch) const {
    SendPackets(batch->pkts(), batch->GetSize());
    batch->Clear();
  }

  /**
   * @brief Explicitly reclaims the memory buffers (mbufs) used by sent packets
   * in the TX ring.
   */
  int ReclaimTxMbufs() const {
    return rte_eth_tx_done_cleanup(this->GetPortId(), this->GetRingId(), 0);
  }

 private:
  struct rte_eth_txconf conf_;
};

/**
 * @brief Represents a RX ring in DPDK.
 *
 * Provides functionalities specific to RX rings in DPDK.
 */
class RxRing : public PmdRing {
 public:
  RxRing(PmdPort const* pmd_port, uint8_t port_id, uint16_t ring_id,
         uint16_t ndesc)
      : PmdRing(pmd_port, port_id, ring_id, ndesc) {}

  RxRing(PmdPort const* pmd_port, uint8_t port_id, uint16_t ring_id,
         uint16_t ndesc, struct rte_eth_rxconf rxconf, uint32_t nmbufs,
         uint32_t mbuf_sz)
      : PmdRing(pmd_port, port_id, ring_id, ndesc, nmbufs, mbuf_sz),
        conf_(rxconf) {}

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
    uint16_t const nb_rx = RecvPackets(batch->pkts(), batch->GetRoom());
    batch->IncrCount(nb_rx);
    return nb_rx;
  }

 private:
  struct rte_eth_rxconf conf_;
};

/**
 * @brief Factory function to create a ring.
 * @tparam T Type of the ring to be created.
 * @tparam Args Types of the arguments.
 * @param params Arguments for ring's constructor.
 * @return A unique pointer to the created ring.
 */
template <typename T, typename... Args>
decltype(auto) makeRing(Args&&... params) {
  std::unique_ptr<T> ptr(nullptr);
  ptr.reset(new T(std::forward<Args>(params)...));
  return ptr;
}

/**
 * @brief Represents a DPDK Port with functionalities to manage RX and TX rings.
 *
 * Provides a higher-level abstraction over a DPDK port, encapsulating
 * operations such as initialization, deinitialization, statistics gathering,
 * and other related functions.
 */
class PmdPort {
 public:
  static uint16_t const kDefaultRingNr_ = 1;

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
        port_id_(id),
        tx_rings_nr_(tx_rings_nr),
        rx_rings_nr_(rx_rings_nr),
        tx_ring_desc_nr_(tx_desc_nr),
        rx_ring_desc_nr_(rx_desc_nr),
        initialized_(false) {
    // Get L2 address.
    rte_ether_addr temp;
    CHECK_EQ(rte_eth_macaddr_get(port_id_, &temp), 0);
    l2_addr_.FromUint8(temp.addr_bytes);
  }

  /**
   * @brief Deleted copy constructor and assignment operator.
   */
  PmdPort(PmdPort const&) = delete;
  PmdPort& operator=(PmdPort const&) = delete;

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
  void InitDriver(uint16_t mtu = PmdRing::kDefaultFrameSize) {
    if (is_dpdk_primary_process_) {
      // Get DPDK port info.
      FetchDpdkPortInfo(port_id_, &devinfo_, &l2_addr_);
      device_ = devinfo_.device;

      if (std::string(devinfo_.driver_name) == "net_netvsc") {
        // For Azure's 'netvsc' driver, we need to find the associated VF so
        // that we can register DMA memory directly with the NIC. Otherwise DMA
        // memory registration is not supported by the synthetic driver.
        auto vf_port_id = dpdk::FindSlaveVfPortId(port_id_);
        if (vf_port_id.has_value()) {
          LOG(INFO) << "Found VF port id: "
                    << static_cast<int>(vf_port_id.value())
                    << " for port id: " << static_cast<int>(port_id_);
          // Get DPDK port info.
          struct rte_eth_dev_info vf_devinfo_;
          struct net::Ethernet::Address vf_l2_addr_;
          FetchDpdkPortInfo(vf_port_id.value(), &vf_devinfo_, &vf_l2_addr_);

          // If the VF is using an 'mlx4*' driver, we need extra checks.
          if (std::string(vf_devinfo_.driver_name).find("mlx4") !=
              std::string::npos) {
            // Mellanox CX3 and CX3-Pro NICs do not support non-power-of-two RSS
            // queues. Furthermore, RETA table cannot be configured.
            if (!utils::is_power_of_two(rx_rings_nr_)) {
              LOG(FATAL)
                  << "Mellanox CX3 and CX3-Pro NICs do not support "
                     "non-power-of-two RSS queues. Please use a power of "
                     "two number of engines in the configuration.";
            }
          }
          device_ = vf_devinfo_.device;
        }
      }

      LOG(INFO) << "Rings nr: " << rx_rings_nr_;
      rte_eth_conf const portconf = DefaultEthConf(&devinfo_);
      int ret = rte_eth_dev_configure(port_id_, rx_rings_nr_, tx_rings_nr_,
                                      &portconf);
      if (ret != 0) {
        LOG(FATAL)
            << "rte_eth_dev_configure() failed. Cannot configure port id: "
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

      LOG(INFO) << utils::Format("RSS indirection table (size %d):\n",
                                 devinfo_.reta_size);
      for (auto i = 0u; i < devinfo_.reta_size; i++) {
        auto const kColumns = 8;
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
          reta_table +=
              "\t" + std::to_string(rss_reta_conf_[index].reta[shift]);
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

      auto const mbuf_data_size =
          mtu + RTE_ETHER_HDR_LEN + RTE_ETHER_CRC_LEN + RTE_PKTMBUF_HEADROOM;

      // Setup the TX queues.
      for (auto q = 0; q < tx_rings_nr_; q++) {
        LOG(INFO) << "Initializing TX ring: " << q;
        auto tx_ring = makeRing<TxRing>(
            this, port_id_, q, tx_ring_desc_nr_, devinfo_.default_txconf,
            2 * tx_ring_desc_nr_ - 1, mbuf_data_size);
        // auto tx_ring = makeRing<TxRing>(this, port_id_, q, tx_ring_desc_nr_,
        //                                 devinfo_.default_txconf);
        tx_ring.get()->Init();
        tx_rings_.emplace_back(std::move(tx_ring));
      }

      // Setup the RX queues.
      for (auto q = 0; q < rx_rings_nr_; q++) {
        LOG(INFO) << "Initializing RX ring: " << q;
        auto rx_ring = makeRing<RxRing>(
            this, port_id_, q, rx_ring_desc_nr_, devinfo_.default_rxconf,
            2 * rx_ring_desc_nr_ - 1, mbuf_data_size);
        rx_ring.get()->Init();
        rx_rings_.emplace_back(std::move(rx_ring));
      }

      ret = rte_eth_promiscuous_enable(port_id_);
      if (ret != 0)
        LOG(WARNING) << "rte_eth_promiscuous_enable() failed.";
      else
        LOG(INFO) << "Promiscuous mode enabled.";

      ret = rte_eth_stats_reset(port_id_);
      if (ret != 0) LOG(WARNING) << "Failed to reset port statistics.";

      ret = rte_eth_dev_set_link_up(port_id_);
      if (ret != 0) LOG(WARNING) << "rte_eth_dev_set_link_up() failed.";

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
            makeRing<TxRing>(this, port_id_, q, tx_ring_desc_nr_));
      }

      for (auto q = 0; q < rx_rings_nr_; q++) {
        rx_rings_.emplace_back(
            makeRing<RxRing>(this, port_id_, q, rx_ring_desc_nr_));
      }
    }

    // Mark port as initialized.
    initialized_ = true;
  }

  /**
   * @brief Deinitializes the port.
   */
  void DeInit() {
    if (!initialized_ || !is_dpdk_primary_process_) return;
    rte_eth_dev_stop(port_id_);
    rte_eth_dev_close(port_id_);
    LOG(INFO) << juggler::utils::Format("[PMDPORT: %u closed.]", port_id_);
    initialized_ = false;
  }

  /**
   * @brief Checks if the port has been initialized.
   *
   * @return True if the port is initialized, false otherwise.
   */
  bool IsInitialized() const { return initialized_; }

  uint16_t GetPortId() const { return port_id_; }

  std::string GetDriverName() const {
    return juggler::utils::Format("%s", devinfo_.driver_name);
  }

  /**
   * @brief Retrieves the associated device for this port.
   *
   * @return Pointer to the associated device.
   */
  rte_device* GetDevice() const { return device_; }

  template <typename T>
  decltype(auto) GetRing(uint16_t id) const {
    constexpr bool is_tx_ring = std::is_same<T, TxRing>::value;
    CHECK_LT(id, (is_tx_ring ? tx_rings_nr_ : rx_rings_nr_)) << "Out-of-bounds";
    return is_tx_ring ? static_cast<T*>(tx_rings_.at(id).get())
                      : static_cast<T*>(rx_rings_.at(id).get());
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
    if (ret != 0) return std::nullopt;  // Error (wrong port id?)
    return mtu;
  }

  /**
   * @brief Retrieves the MAC address of the port as set in the hardware.
   *
   * @return `Ethernet' address representing the MAC address.
   */
  net::Ethernet::Address GetL2Addr() const { return l2_addr_; }

  /**
   * @brief Retrieves the port's RSS (Receive Side Scaling) key.
   *
   * @return A vector containing the RSS hash key bytes.
   */
  std::vector<uint8_t> const& GetRSSKey() const { return rss_hash_key_; }

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
    LOG(INFO) << juggler::utils::Format(
        "[STATS - Port: %u] [TX] Pkts: %lu, Bytes: %lu, Drops: %lu [RX] Pkts: "
        "%lu, Bytes: %lu, Drops: %lu, NoRXMbufs: %lu",
        port_id_, GetPortTxPkts(), GetPortTxBytes(), GetPortTxDrops(),
        GetPortRxPkts(), GetPortRxBytes(), GetPortRxDrops(),
        GetPortRxNoMbufErr());

    for (uint16_t i = 0; i < tx_rings_nr_; i++) {
      LOG(INFO) << juggler::utils::Format(
          "[STATS - Port: %u, Queue: %u] [TX] Pkts: %lu, Bytes: %lu", port_id_,
          i, GetPortQueueTxPkts(i), GetPortQueueTxBytes(i));
    }

    for (uint16_t i = 0; i < rx_rings_nr_; i++) {
      LOG(INFO) << juggler::utils::Format(
          "[STATS - Port: %u, Queue: %u] [RX] Pkts: %lu, Bytes: %lu", port_id_,
          i, GetPortQueueRxPkts(i), GetPortQueueRxBytes(i));
    }
  }

 private:
  bool const is_dpdk_primary_process_;
  uint16_t const port_id_;
  uint16_t const tx_rings_nr_, rx_rings_nr_;
  uint16_t tx_ring_desc_nr_, rx_ring_desc_nr_;
  std::vector<std::unique_ptr<PmdRing>> tx_rings_, rx_rings_;

  juggler::net::Ethernet::Address l2_addr_;
  struct rte_eth_dev_info devinfo_;
  rte_device* device_;
  std::vector<rte_eth_rss_reta_entry64> rss_reta_conf_;
  struct rte_eth_stats port_stats_;
  std::vector<uint8_t> rss_hash_key_;
  bool initialized_;
};
}  // namespace dpdk
}  // namespace juggler

#endif  // SRC_INCLUDE_PMD_H_
