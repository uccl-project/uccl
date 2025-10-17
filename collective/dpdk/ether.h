#ifndef SRC_INCLUDE_ETHER_H_
#define SRC_INCLUDE_ETHER_H_

#include "util/endian.h"
#include "util/util.h"

#include <cstdint>
#include <string>

namespace uccl {

struct __attribute__((packed)) Ethernet {
  struct __attribute__((packed)) Address {
    static const uint8_t kSize = 6;
    Address() = default;
    Address(const Address &other) = default;
    Address(const uint8_t *addr) {
      bytes[0] = addr[0];
      bytes[1] = addr[1];
      bytes[2] = addr[2];
      bytes[3] = addr[3];
      bytes[4] = addr[4];
      bytes[5] = addr[5];
    }

    Address(const std::string mac_addr) { FromString(mac_addr); }

    void FromUint8(const uint8_t *addr) {
      bytes[0] = addr[0];
      bytes[1] = addr[1];
      bytes[2] = addr[2];
      bytes[3] = addr[3];
      bytes[4] = addr[4];
      bytes[5] = addr[5];
    }

    bool FromString(std::string str) {
      return kSize == sscanf(str.c_str(), "%hhx:%hhx:%hhx:%hhx:%hhx:%hhx",
                             &bytes[0], &bytes[1], &bytes[2], &bytes[3],
                             &bytes[4], &bytes[5]);
    }

    std::string ToString() const {
      std::string ret;
      char addr[18];

      if (17 == snprintf(addr, sizeof(addr),
                         "%02hhx:%02hhx:%02hhx:%02hhx:%02hhx:%02hhx", bytes[0],
                         bytes[1], bytes[2], bytes[3], bytes[4], bytes[5])) {
        ret = std::string(addr);
      }

      return ret;
    }

    Address &operator=(const Address &rhs) {
      bytes[0] = rhs.bytes[0];
      bytes[1] = rhs.bytes[1];
      bytes[2] = rhs.bytes[2];
      bytes[3] = rhs.bytes[3];
      bytes[4] = rhs.bytes[4];
      bytes[5] = rhs.bytes[5];
      return *this;
    }
    bool operator==(const Address &rhs) const {
      return bytes[0] == rhs.bytes[0] && bytes[1] == rhs.bytes[1] &&
             bytes[2] == rhs.bytes[2] && bytes[3] == rhs.bytes[3] &&
             bytes[4] == rhs.bytes[4] && bytes[5] == rhs.bytes[5];
    }
    bool operator!=(const Address &rhs) const { return !operator==(rhs); }

    uint8_t bytes[kSize];
  };
  inline static const Address kBroadcastAddr{"ff:ff:ff:ff:ff:ff"};
  inline static const Address kZeroAddr{"00:00:00:00:00:00"};

  enum EthType : uint16_t {
    kArp = 0x806,
    kIpv4 = 0x800,
    kIpv6 = 0x86DD,
  };

  std::string ToString() const {
    return Format("[Eth: dst %s, src %s, eth_type %u]",
                                  dst_addr.ToString().c_str(),
                                  src_addr.ToString().c_str(), eth_type.value());
  }

  Address dst_addr;
  Address src_addr;
  be16_t eth_type;
};
} // namespace uccl

#endif // SRC_INCLUDE_ETHER_H_