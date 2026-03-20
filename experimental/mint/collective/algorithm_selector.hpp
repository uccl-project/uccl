// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_NCCL_ALGORITHM_SELECTOR_HPP_
#define MSCCLPP_EXT_NCCL_ALGORITHM_SELECTOR_HPP_

#include "algorithm.hpp"
#include "core.hpp"
#include <memory>
#include <unordered_map>

namespace mscclpp {
namespace nccl {

/// Configuration for algorithm selection
struct AlgorithmSelectorConfig {
  bool symmetricMemory;
  bool nvlsSupported;
  bool isCuMemMapAllocated;
  bool inCaptureMode;
  std::pair<int, int> computeCapability;
  bool ncclDlopenSharedLib;
};

/// Select an algorithm for single-node allreduce
std::shared_ptr<Algorithm> selectSingleNodeAllreduce(
    std::unordered_map<std::string, std::shared_ptr<Algorithm>> const& algoMap,
    CollectiveRequest const& request, AlgorithmSelectorConfig const& config);

/// Select an algorithm for single-node allgather
std::shared_ptr<Algorithm> selectSingleNodeAllgather(
    std::unordered_map<std::string, std::shared_ptr<Algorithm>> const& algoMap,
    CollectiveRequest const& request, AlgorithmSelectorConfig const& config);

/// Select an algorithm for multi-node collective operations
/// Currently returns nullptr to fallback to NCCL/RCCL
/// TODO: Implement multi-node NVLS and multi-node IB algorithms
std::shared_ptr<Algorithm> selectMultiNodeAlgorithm(
    std::unordered_map<std::string, std::shared_ptr<Algorithm>> const& algoMap,
    CollectiveRequest const& request, AlgorithmSelectorConfig const& config);

/// Check if an execution plan matches the request
bool matchExecutionPlan(std::shared_ptr<DslAlgorithm> algo,
                        CollectiveRequest const& request);

}  // namespace nccl
}  // namespace mscclpp

#endif  // MSCCLPP_EXT_NCCL_ALGORITHM_SELECTOR_HPP_
