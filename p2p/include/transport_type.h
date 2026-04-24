#pragma once
// Runtime transport type detection.
// Reads UCCL_P2P_TRANSPORT env var once and caches the result.

#include <cstdlib>
#include <cstring>

namespace uccl {

enum class TransportType { RDMA, NCCL, EFA };

inline TransportType get_transport_type() {
  static TransportType t = [] {
    char const* env = std::getenv("UCCL_P2P_TRANSPORT");
    if (!env) return TransportType::RDMA;
    if (std::strcmp(env, "nccl") == 0 || std::strcmp(env, "tcp") == 0 ||
        std::strcmp(env, "tcpx") == 0)
      return TransportType::NCCL;
    if (std::strcmp(env, "efa") == 0) return TransportType::EFA;
    return TransportType::RDMA;
  }();
  return t;
}

inline bool is_nccl_transport() {
  return get_transport_type() == TransportType::NCCL;
}

inline bool is_efa_transport() {
  return get_transport_type() == TransportType::EFA;
}

}  // namespace uccl
