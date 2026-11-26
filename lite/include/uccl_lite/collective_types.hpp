#pragma once

#include <cstddef>

namespace uccl_lite {

enum class Collective { AllReduce, AllGather, ReduceScatter };
enum class ScalarType { BFloat16, Float16, Float32 };
constexpr size_t scalarBytes(ScalarType type) {
  return type == ScalarType::Float32                                   ? 4
         : type == ScalarType::BFloat16 || type == ScalarType::Float16 ? 2
                                                                       : 0;
}

struct Workload {
  Collective collective = Collective::AllReduce;
  ScalarType dtype = ScalarType::BFloat16;
  // Complete per-rank output for AG, complete input for RS, tensor size for AR.
  size_t bytes = 0;
  bool inPlace = false;
  // Minimum input/output pointer alignment, capped at 256 bytes. Zero denotes
  // a caller-defined/legacy key; live dispatcher calls always record alignment.
  unsigned alignment = 0;
};

}  // namespace uccl_lite
