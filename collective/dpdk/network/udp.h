#ifndef SRC_INCLUDE_UDP_H_
#define SRC_INCLUDE_UDP_H_

#include "util/endian.h"
#include "util/util.h"
#include <cstdint>

namespace uccl {

struct __attribute__((packed)) Udp {
  struct __attribute__((packed)) Port {
    static const uint8_t kSize = 2;
    Port() = default;
    Port(uint16_t udp_port) { port = be16_t(udp_port); }
    bool operator==(Port const& rhs) const { return port == rhs.port; }
    bool operator==(be16_t rhs) const { return rhs == port; }
    bool operator!=(Port const& rhs) const { return port != rhs.port; }
    bool operator!=(be16_t rhs) const { return rhs != port; }

    be16_t port;
  };

  std::string ToString() const {
    return Format("[UDP: src_port %zu, dst_port %zu, len %zu, csum %zu]",
                  src_port.port.value(), dst_port.port.value(), len.value(),
                  cksum.value());
  }

  Port src_port;
  Port dst_port;
  be16_t len;
  be16_t cksum;
};

}  // namespace uccl

#endif  // SRC_INCLUDE_UDP_H_