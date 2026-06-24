#pragma once

#include <cstddef>
#include <cstdint>

namespace UKernel {
namespace CCL {

enum class CollKind : uint32_t {
  AllReduceRing,
  AllToAllPairwise,
};

enum class OpKind : uint32_t {
  Send,
  Recv,
  Copy,
  Reduce,
  RecvReduce,
  Signal,  // send_signal_async(peer, tag) — completes via completion_ring_
  SignalWait,  // wait_signal_async(peer, tag) — completes via signal_ring_
};

enum class ScalarType : uint32_t {
  UInt8,
  Int8,
  Int16,
  Int32,
  Int64,
  Float16,
  Float32,
  Float64,
  BFloat16,
  Bool,
};

inline constexpr size_t scalar_type_size(ScalarType dtype) {
  switch (dtype) {
    case ScalarType::UInt8:
    case ScalarType::Int8:
    case ScalarType::Bool:
      return 1;
    case ScalarType::Int16:
    case ScalarType::Float16:
    case ScalarType::BFloat16:
      return 2;
    case ScalarType::Int32:
    case ScalarType::Float32:
      return 4;
    case ScalarType::Int64:
    case ScalarType::Float64:
      return 8;
  }
  return 0;
}

enum class ReductionKind : uint32_t {
  None,
  Sum,
  Prod,
  Max,
  Min,
  BitwiseAnd,
};

enum class CollectiveBufferRole : uint32_t {
  Input,
  Output,
  Scratch,
};

}  // namespace CCL
}  // namespace UKernel
