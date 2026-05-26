#pragma once

#include "collective_memory.h"
#include "collective_types.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace UKernel {
namespace CCL {

enum class CollectiveKind : uint32_t {
  AllReduce,
  AllToAll,
};

enum class AlgorithmKind : uint32_t {
  Ring,
  Pairwise,
};

enum class OpKind : uint32_t {
  TransportSend,
  TransportRecv,
  DeviceCopy,
  DeviceReduce,
  DeviceSend,
  DeviceRecv,
  DeviceRecvReduce,
};

struct Op {
  OpKind kind = OpKind::DeviceCopy;
  uint32_t stream_index = 0;
  size_t bytes = 0;
  size_t src_off = 0;
  size_t dst_off = 0;
  uint32_t src_peer = 0;
  uint32_t dst_peer = 0;
  uint64_t seq = 0;
  bool copy_from_staging = false;
  std::vector<uint32_t> deps;
};

struct CollectiveConfig {
  CollectiveKind collective = CollectiveKind::AllReduce;
  int nranks = 1;
  int rank = 0;
  uint32_t num_streams = 1;
  size_t input_bytes = 0;
  size_t output_bytes = 0;
  size_t tile_bytes = 0;
  std::vector<size_t> input_split_bytes;
  std::vector<size_t> output_split_bytes;
  AlgorithmKind algorithm = AlgorithmKind::Ring;
  ScalarType dtype = ScalarType::Float32;
  ReductionKind reduction = ReductionKind::Sum;
  bool use_sm_ipc = true;
};

struct CollectivePlan {
  CollectiveKind collective = CollectiveKind::AllReduce;
  int nranks = 1;
  int rank = 0;
  uint32_t num_streams = 1;
  size_t input_bytes = 0;
  size_t output_bytes = 0;
  size_t tile_bytes = 0;
  size_t staging_bytes_required = 0;
  ReductionKind reduction = ReductionKind::None;
  std::vector<Op> ops;
  std::vector<std::vector<uint32_t>> stream_ops;
};

uint32_t normalized_num_streams(CollectiveKind collective, int nranks,
                               size_t input_bytes, size_t tile_bytes,
                               ScalarType dtype, uint32_t requested_streams);

CollectivePlan build_plan(CollectiveConfig const& config, bool inplace);
std::string to_string(CollectivePlan const& plan);

}  // namespace CCL
}  // namespace UKernel
