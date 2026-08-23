// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include "copy_primitives.hpp"
#include "rdma_primitives.hpp"
#include "reduce_primitives.hpp"
#include "sync_primitives.hpp"

namespace mscclpp::lite {

/*
Unified host-side CPU switch primitive, invoked by collective operations directly, 
in the same style as NCCL Primitives.
*/
template <typename T, typename RedOp = Sum<T>>
class CpuSwitch : public CopyPrimitives<T>,
                  public ReducePrimitives<T, RedOp>,
                  public RdmaPrimitives,
                  public SyncPrimitives {
 public:
  CpuSwitch() = default;
  ~CpuSwitch() = default;

  CpuSwitch(CpuSwitch const&) = delete;
  CpuSwitch& operator=(CpuSwitch const&) = delete;
};

}  // namespace mscclpp::lite
