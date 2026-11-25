#pragma once
#include "rdma/rdma_endpoint.h"
#include "transport/config.h"
#if UCCL_P2P_ENABLE_NCCL
#include "nccl/nccl_endpoint.h"
#endif
#if UCCL_P2P_ENABLE_CXI
#include "cxi/cxi_endpoint.h"
#endif
#include <memory>
#include <variant>

// Memory handles may hold an empty CXI reference in builds without CXI.
struct CxiMemoryRegion;

// This is the compile-time transport registration boundary.
// Runtime-polymorphic endpoint.
using GenericEndpoint = std::variant<std::shared_ptr<RDMAEndpoint>
#if UCCL_P2P_ENABLE_NCCL
                                     ,
                                     std::shared_ptr<NCCLEndpoint>
#endif
#if UCCL_P2P_ENABLE_CXI
                                     ,
                                     std::shared_ptr<CxiEndpoint>
#endif
                                     >;
