#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace UKernel {
namespace CCL {

// Matches the usual torch-style scalar type split at the tensor metadata layer.
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

struct TensorLayout {
  // Torch-style tensor metadata: sizes/strides/storage_offset are all
  // expressed in elements rather than bytes.
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  int64_t storage_offset = 0;
  ScalarType dtype = ScalarType::UInt8;
};

enum class BufferKind : uint32_t {
  Tensor,
  Staging,
  PeerTensor,
};

// BufferRef is the shared buffer binding used by both the algorithm plan and
// the lowered execution plan. It identifies one concrete tensor region.
struct BufferRef {
  BufferKind kind = BufferKind::Tensor;
  int peer_rank = -1;
  size_t offset_bytes = 0;
};

struct PeerTensorView {
  void* ptr = nullptr;
  uint32_t mr_id = 0;
  bool same_node = false;
  bool peer_accessible = false;
};

struct SymmetricTensor {
  int local_rank = 0;
  void* local_ptr = nullptr;
  uint32_t local_mr_id = 0;
  size_t bytes = 0;
  TensorLayout layout{};
  // Rank-indexed remote tensor table: peer_views[rank] is the memory view for
  // that rank's symmetric tensor.
  std::vector<PeerTensorView> peer_views;
};

struct StagingTensor {
  void* local_ptr = nullptr;
  size_t bytes = 0;
  TensorLayout layout{};
};

struct CollectiveMemory {
  SymmetricTensor tensor;
  // Staging is temporary rank-local tensor storage used for recv/reduce/copy
  // staging during collective execution.
  StagingTensor staging;
};

}  // namespace CCL
}  // namespace UKernel
