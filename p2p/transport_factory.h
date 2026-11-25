#pragma once
#include "engine.h"
#include "transport_type.h"

inline GenericEndpoint uccl_create_endpoint(int gpu_idx) {
  switch (get_transport_type()) {
#if UCCL_P2P_ENABLE_NCCL
    case TransportType::NCCL:
      return std::make_shared<NCCLEndpoint>(gpu_idx, 0);
#endif
#if UCCL_P2P_ENABLE_CXI
    case TransportType::CXI:
      return std::make_shared<CxiEndpoint>(gpu_idx, 0);
#endif
    case TransportType::RDMA:
    case TransportType::EFA:
      return std::make_shared<RDMAEndpoint>(gpu_idx, 0);
    default:
      throw std::invalid_argument("Transport is not enabled in this P2P build");
  }
}
