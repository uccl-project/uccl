#ifndef SRC_INCLUDE_IPV4_H_
#define SRC_INCLUDE_IPV4_H_

#include "util/endian.h"
#include "util/util.h"
#include <arpa/inet.h>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <string>

namespace uccl {

struct __attribute__((packed)) Ipv4 {
  static const uint8_t kDefaultTTL = 64;
  struct __attribute__((packed)) Address {
    static const uint8_t kSize = 4;
    Address() = default;
    Address(uint8_t const* addr) {
      std::memcpy(&address, addr, sizeof(address));
    }
    Address(uint32_t addr) { address = be32_t(addr); }

    static bool IsValid(std::string const& addr) {
      struct sockaddr_in sa;
      int result = inet_pton(AF_INET, addr.c_str(), &(sa.sin_addr));
      return result != 0;
    }

    /// If addr is not a valid IPv4 address, return a zero-valued IP address
    static std::optional<Address> MakeAddress(std::string const& addr) {
      Address ret;
      if (!ret.FromString(addr)) return std::nullopt;
      return ret;
    }

    Address& operator=(Address const& rhs) {
      address = rhs.address;
      return *this;
    }
    bool operator==(Address const& rhs) const { return address == rhs.address; }
    bool operator!=(Address const& rhs) const { return address != rhs.address; }
    bool operator==(be32_t const& rhs) const { return rhs == address; }
    bool operator!=(be32_t const& rhs) const { return rhs != address; }
    bool operator!=(be32_t rhs) const { return rhs != address; }
    bool operator==(uint32_t const& rhs) const {
      return be32_t(rhs) == address;
    }
    bool operator!=(uint32_t const& rhs) const {
      return be32_t(rhs) != address;
    }

    bool FromString(std::string str) {
      if (!Ipv4::Address::IsValid(str)) return false;
      unsigned char bytes[4];
      uint8_t len = sscanf(str.c_str(), "%hhu.%hhu.%hhu.%hhu", &bytes[0],
                           &bytes[1], &bytes[2], &bytes[3]);
      if (len != Ipv4::Address::kSize) return false;
      address = be32_t((uint32_t)(bytes[0]) << 24 | (uint32_t)(bytes[1]) << 16 |
                       (uint32_t)(bytes[2]) << 8 | (uint32_t)(bytes[3]));

      return true;
    }

    std::string ToString() const {
      const std::vector<uint8_t> bytes(address.ToByteVector());
      CHECK_EQ(bytes.size(), 4);
      return Format("%hhu.%hhu.%hhu.%hhu", bytes[0], bytes[1], bytes[2],
                    bytes[3]);
    }

    be32_t address;
  };

  enum Proto : uint8_t {
    kIcmp = 1,
    kTcp = 6,
    kUdp = 17,
    kRaw = 255,
  };

  std::string ToString() const {
    return Format(
        "[IPv4: src %s, dst %s, ihl %u, ToS %u, tot_len %u, ID %u, "
        "frag_off %u, "
        "TTL %u, proto %u, check %u]",
        src_addr.ToString().c_str(), dst_addr.ToString().c_str(), version_ihl,
        type_of_service, total_length.value(), packet_id.value(),
        fragment_offset.value(), time_to_live, next_proto_id, hdr_checksum);
  }

  uint8_t version_ihl;
  uint8_t type_of_service;
  be16_t total_length;
  be16_t packet_id;
  be16_t fragment_offset;
  uint8_t time_to_live;
  uint8_t next_proto_id;
  uint16_t hdr_checksum;
  Address src_addr;
  Address dst_addr;
};

}  // namespace uccl
#endif  // SRC_INCLUDE_IPV4_H_