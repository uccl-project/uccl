#pragma once
// Runtime transport type detection.
// Reads UCCL_P2P_TRANSPORT env var once and caches the result.

#include <cstdlib>
#include <cstring>
#if defined(UCCL_USE_MUSA)
#include <stdexcept>
#endif

enum class TransportType { RDMA, NCCL, EFA, CXI };

inline TransportType get_transport_type() {
  static TransportType t = [] {
    char const* env = std::getenv("UCCL_P2P_TRANSPORT");
#if defined(UCCL_USE_MUSA)
    if (env && std::strcmp(env, "rdma") != 0 && std::strcmp(env, "ib") != 0)
      throw std::invalid_argument(
          "MUSA P2P supports only UCCL_P2P_TRANSPORT=rdma/ib (including IPC)");
    return TransportType::RDMA;
#else
    if (!env) return TransportType::RDMA;
    if (std::strcmp(env, "nccl") == 0 || std::strcmp(env, "tcp") == 0 ||
        std::strcmp(env, "tcpx") == 0)
      return TransportType::NCCL;
    if (std::strcmp(env, "efa") == 0) return TransportType::EFA;
    if (std::strcmp(env, "cxi") == 0) return TransportType::CXI;
    return TransportType::RDMA;
#endif
  }();
  return t;
}

inline bool is_nccl_transport() {
  return get_transport_type() == TransportType::NCCL;
}

inline bool is_efa_transport() {
  return get_transport_type() == TransportType::EFA;
}

inline bool is_cxi_transport() {
  return get_transport_type() == TransportType::CXI;
}
