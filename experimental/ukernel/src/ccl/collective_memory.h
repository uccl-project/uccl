#pragma once

#include "collective_types.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace UKernel {
namespace CCL {

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
  PeerStaging,
};

// BufferRef is the shared buffer binding used by both the algorithm plan and
// the lowered execution plan. Local kinds address this rank's tensor/staging;
// peer kinds address a specific peer rank's tensor/staging view.
struct BufferRef {
  BufferKind kind = BufferKind::Tensor;
  size_t offset_bytes = 0;
  int rank = -1;
};

struct PeerBufferView {
  uint32_t mr_id = 0;
  bool same_node = false;
};

struct SymmetricTensor {
  int local_rank = 0;
  void* local_ptr = nullptr;
  uint32_t local_mr_id = 0;
  size_t bytes = 0;
  TensorLayout layout{};
  // Rank-indexed remote tensor table: peer_views[rank] is the memory view for
  // that rank's symmetric tensor.
  std::vector<PeerBufferView> peer_views;
};

struct StagingTensor {
  void* local_ptr = nullptr;
  uint32_t local_mr_id = 0;
  size_t bytes = 0;
  TensorLayout layout{};
  std::vector<PeerBufferView> peer_views;
};

struct CollectiveMemory {
  SymmetricTensor tensor;
  // Staging is temporary rank-local tensor storage used for recv/reduce/copy
  // staging during collective execution.
  StagingTensor staging;
};

}  // namespace CCL
}  // namespace UKernel
