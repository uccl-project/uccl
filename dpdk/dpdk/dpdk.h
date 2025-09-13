#ifndef SRC_INCLUDE_DPDK_H_
#define SRC_INCLUDE_DPDK_H_

#include <optional>
#include <ether.h>
#include <packet_pool.h>
#include <rte_bus.h>
#include <rte_bus_pci.h>
#include <rte_common.h>
#include <rte_config.h>
#include <rte_cycles.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_ether.h>
#include <utils.h>
#include <worker.h>

namespace juggler {
namespace dpdk {

[[maybe_unused]] static void FetchDpdkPortInfo(
    uint8_t port_id, struct rte_eth_dev_info* devinfo,
    juggler::net::Ethernet::Address* lladdr) {
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
                      reinterpret_cast<rte_ether_addr*>(lladdr->bytes));

  LOG(INFO) << "[PMDPORT] [port_id: " << static_cast<uint32_t>(port_id)
            << ", driver: " << devinfo->driver_name
            << ", RXQ: " << devinfo->max_rx_queues
            << ", TXQ: " << devinfo->max_tx_queues
            << ", l2addr: " << lladdr->ToString() << "]";
}

[[maybe_unused]] static std::optional<uint16_t> FindSlaveVfPortId(
    uint16_t port_id) {
  struct rte_eth_dev_info devinfo;
  juggler::net::Ethernet::Address lladdr;

  FetchDpdkPortInfo(port_id, &devinfo, &lladdr);

  uint16_t slave_port_id = 0;
  while (slave_port_id < RTE_MAX_ETHPORTS) {
    if (slave_port_id == port_id) {
      slave_port_id++;
      continue;
    }

    if (!rte_eth_dev_is_valid_port(slave_port_id)) {
      break;
    }

    struct rte_eth_dev_info slave_devinfo;
    juggler::net::Ethernet::Address slave_lladdr;
    FetchDpdkPortInfo(slave_port_id, &slave_devinfo, &slave_lladdr);
    if (slave_lladdr == lladdr) {
      return slave_port_id;
    }

    slave_port_id++;
  }

  return std::nullopt;
}

[[maybe_unused]] static void ScanDpdkPorts() {
  // This iteration is *required* to expose the net failsafe interface in Azure
  // VMs. Without this, the application is going to bind on top of the mlx5
  // driver. Worse TX is going to work, but nothing will appear on the RX side.
  uint16_t port_id;
  RTE_ETH_FOREACH_DEV(port_id) {
    struct rte_eth_dev_info devinfo;
    juggler::net::Ethernet::Address lladdr;

    FetchDpdkPortInfo(port_id, &devinfo, &lladdr);
  }
}

// Default EAL init arguments.
static auto kDefaultEalOpts =
    juggler::utils::CmdLineOpts({"", "--log-level=eal,8", "--proc-type=auto"});

class Dpdk {
 public:
  Dpdk() : initialized_(false) {}
  ~Dpdk() { DeInitDpdk(); }

  void InitDpdk(juggler::utils::CmdLineOpts copts = kDefaultEalOpts) {
    if (initialized_) {
      LOG(WARNING) << "DPDK is already initialized.";
      return;
    }

    LOG(INFO) << "Initializing DPDK with args: " << rte_args.ToString();
    int ret = rte_eal_init(rte_args.GetArgc(), rte_args.GetArgv());
    if (ret < 0) {
      LOG(FATAL) << "rte_eal_init() failed: ret = " << ret
                 << " rte_errno = " << rte_errno << " ("
                 << rte_strerror(rte_errno) << ")";
    }

    // Check if DPDK runs in PA or VA mode.
    if (rte_eal_iova_mode() == RTE_IOVA_VA) {
      LOG(INFO) << "DPDK runs in VA mode.";
    } else {
      LOG(INFO) << "DPDK runs in PA mode.";
    }

    ScanDpdkPorts();
    initialized_ = true;
  }

  void DeInitDpdk() {
    int ret = rte_eal_cleanup();
    if (ret != 0) {
      LOG(FATAL) << "rte_eal_cleanup() failed: ret = " << ret
                 << " rte_errno = " << rte_errno << " ("
                 << rte_strerror(rte_errno) << ")";
    }

    initialized_ = false;
  }

  bool const isInitialized() { return initialized_; }
  size_t GetNumPmdPortsAvailable() { return rte_eth_dev_count_avail(); }
  std::optional<uint16_t> GetPmdPortIdByMac(
      juggler::net::Ethernet::Address const& l2_addr) const {
    if (!initialized_) {
      LOG(WARNING) << "DPDK is not initialized. Cannot retrieve eth device "
                      "contextual info.";
      return std::nullopt;
    }

    std::optional<uint16_t> p_id = std::nullopt;
    uint16_t port_id;
    RTE_ETH_FOREACH_DEV(port_id) {
      std::string pci_info;
      juggler::net::Ethernet::Address lladdr;

      int ret = rte_eth_macaddr_get(
          port_id, reinterpret_cast<rte_ether_addr*>(lladdr.bytes));
      if (ret != 0) {
        LOG(WARNING)
            << "rte_eth_macaddr_get() failed. Cannot retrieve eth device "
               "contextual info for port "
            << static_cast<int>(port_id);
        break;
      }
      LOG(INFO) << "looking for " << l2_addr.ToString() << " found "
                << lladdr.ToString() << " port " << static_cast<int>(port_id);

      if (lladdr == l2_addr) {
        p_id = port_id;
      }
    }

    return p_id;
  }

 private:
  bool initialized_;
};
}  // namespace dpdk
}  // namespace juggler

#endif  // SRC_INCLUDE_DPDK_H_
