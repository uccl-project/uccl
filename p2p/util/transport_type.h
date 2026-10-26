#pragma once
// Runtime transport selection, validated against this build before GPU setup.
#include "transport/config.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

enum class TransportType { RDMA, NCCL, EFA, CXI };

inline bool transport_available(TransportType type) {
  switch (type) {
    case TransportType::RDMA:
      return true;
    case TransportType::NCCL:
      return UCCL_P2P_ENABLE_NCCL;
    case TransportType::EFA:
      return UCCL_P2P_ENABLE_EFA;
    case TransportType::CXI:
      return UCCL_P2P_ENABLE_CXI;
  }
  return false;
}

inline TransportType get_transport_type() {
  static TransportType t = [] {
    char const* env = std::getenv("UCCL_P2P_TRANSPORT");
    // Retain the strict RDMA-only profile's accepted aliases.
    if (UCCL_DEVICE_RDMA_ONLY && env && std::strcmp(env, "rdma") != 0 &&
        std::strcmp(env, "ib") != 0)
      throw std::invalid_argument(
          std::string(uccl::device::Backend::name) +
          " P2P supports only UCCL_P2P_TRANSPORT=rdma/ib (including IPC)");
    TransportType selected = TransportType::RDMA;
    if (env) {
      if (std::strcmp(env, "nccl") == 0 || std::strcmp(env, "tcp") == 0 ||
          std::strcmp(env, "tcpx") == 0)
        selected = TransportType::NCCL;
      else if (std::strcmp(env, "efa") == 0)
        selected = TransportType::EFA;
      else if (std::strcmp(env, "cxi") == 0)
        selected = TransportType::CXI;
    }
    if (!transport_available(selected))
      throw std::invalid_argument(std::string("Transport ") + env +
                                  " is not enabled in this P2P build");
    return selected;
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
