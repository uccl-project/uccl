#ifndef SRC_INCLUDE_DPDK_H_
#define SRC_INCLUDE_DPDK_H_

#include "ether.h"
#include <glog/logging.h>
#include <rte_common.h>
#include <rte_config.h>
#include <rte_cycles.h>
#include <rte_eal.h>
#include <rte_errno.h>
#include <rte_ethdev.h>

namespace uccl {
class Dpdk {
 public:
  Dpdk() : initialized_(false) {}
  ~Dpdk() { DeInitDpdk(); }

  void InitDpdk(int argc, char** argv) {
    if (initialized_) {
      LOG(WARNING) << "DPDK is already initialized.";
      return;
    }

    LOG(INFO) << "Initializing DPDK with args";

    int ret = rte_eal_init(argc, argv);
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

    initialized_ = true;
    LOG(INFO) << "DPDK initialized successfully";
  }

  void DeInitDpdk() {
    if (initialized_) {
      int ret = rte_eal_cleanup();
      if (ret != 0) {
        LOG(FATAL) << "rte_eal_cleanup() failed: ret = " << ret
                   << " rte_errno = " << rte_errno << " ("
                   << rte_strerror(rte_errno) << ")";
      }

      initialized_ = false;
    }
  }

  bool isInitialized() const { return initialized_; }

  size_t GetNumPmdPortsAvailable() { return rte_eth_dev_count_avail(); }

  uint16_t GetPmdPortIdByMac(char const* l2_addr) const {
    uint16_t nb_ports = rte_eth_dev_count_avail();
    struct rte_ether_addr mac;
    char buf[32];

    LOG(INFO) << "Checking " << nb_ports << " ports";
    for (uint16_t pid = 0; pid < nb_ports; pid++) {
      if (rte_eth_macaddr_get(pid, &mac) < 0) continue;

      mac_to_str(&mac, buf, sizeof(buf));
      LOG(INFO) << "Checking port " << pid << " with MAC " << buf;
      if (strcasecmp(buf, l2_addr) == 0) {
        return pid;
      }
    }
    return -1;  // 没找到
  }

 private:
  void mac_to_str(struct rte_ether_addr* mac, char* buf, size_t size) const {
    snprintf(buf, size, "%02x:%02x:%02x:%02x:%02x:%02x", mac->addr_bytes[0],
             mac->addr_bytes[1], mac->addr_bytes[2], mac->addr_bytes[3],
             mac->addr_bytes[4], mac->addr_bytes[5]);
  }

  bool initialized_;
};
}  // namespace uccl

#endif  // SRC_INCLUDE_DPDK_H_