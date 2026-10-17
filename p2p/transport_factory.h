#pragma once

#include "engine.h"
#include "transport_type.h"

inline GenericEndpoint uccl_create_endpoint(int gpu_idx) {
#if defined(UCCL_USE_MUSA)
  return std::make_shared<RDMAEndpoint>(gpu_idx, 0);
#else
  switch (get_transport_type()) {
    case TransportType::NCCL:
      return std::make_shared<NCCLEndpoint>(gpu_idx, 0);
    case TransportType::CXI:
      return std::make_shared<CxiEndpoint>(gpu_idx, 0);
    case TransportType::RDMA:
    case TransportType::EFA:
      return std::make_shared<RDMAEndpoint>(gpu_idx, 0);
  }
  return std::make_shared<RDMAEndpoint>(gpu_idx, 0);
#endif
}
