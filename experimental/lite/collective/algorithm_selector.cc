// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "algorithm_selector.hpp"
#include "debug.h"
#include "env.hpp"
#include "utils.hpp"

namespace mscclpp {
namespace nccl {

static bool isNvlsSupportedForDataType(AlgorithmSelectorConfig const& config,
                                       DataType dtype) {
  bool nvlsSupported = config.nvlsSupported;

  // NVLS does not support uint8_t (no hardware support for byte-level
  // reduction)
  if (dtype == DataType::UINT8) {
    return false;
  }

  bool const isFp8 =
      dtype == DataType::FLOAT8_E4M3 || dtype == DataType::FLOAT8_E5M2;

  if (!isFp8) {
    return nvlsSupported;
  }

  // FP8 handling
#if !defined(__HIP_PLATFORM_AMD__)
  // NVLS does not support FP8 on devices with compute capability < 10
  if (config.computeCapability.first < 10) {
    return false;
  }
#if (defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__))
  return true;
#else
  return false;
#endif
#else
  return nvlsSupported;
#endif
}

bool matchExecutionPlan(std::shared_ptr<DslAlgorithm> algo,
                        CollectiveRequest const& request) {
  bool worldSizeMatch = algo->constraint().worldSize == request.worldSize;
  bool ranksPerNodeMatch =
      algo->constraint().nRanksPerNode == request.nRanksPerNode;
  bool collectiveMatch = algo->collective() == request.collective;
  bool bufferModeMatch = algo->bufferMode() == CollectiveBufferMode::Any ||
                         request.bufferMode() == algo->bufferMode();
  size_t effectiveSize = (request.collective == "allgather")
                             ? (request.messageSize * request.worldSize)
                             : request.messageSize;
  bool minSizeMatch = effectiveSize >= algo->messageRange().first;
  bool maxSizeMatch = effectiveSize <= algo->messageRange().second;
  bool result = worldSizeMatch && ranksPerNodeMatch && collectiveMatch &&
                bufferModeMatch && minSizeMatch && maxSizeMatch;
  return result;
}

static std::shared_ptr<Algorithm> selectSingleNodeAllreduceBlackwell(
    std::unordered_map<std::string, std::shared_ptr<Algorithm>> const& algoMap,
    CollectiveRequest const& request, AlgorithmSelectorConfig const& config) {
  const size_t messageSize = request.messageSize;

  bool const nvlsSupported = isNvlsSupportedForDataType(config, request.dtype);

  // Small messages always use NVLS packet algorithm
  if (messageSize <= (1 << 15)) {  // <= 32KB
    return algoMap.at("default_allreduce_nvls_packet");
  }

  if (!config.symmetricMemory) {
    if (messageSize <= (1 << 21)) {  // <= 2MB
      return algoMap.at("default_allreduce_packet");
    }
    if (config.inCaptureMode) {
      // CUDA graph mode: setup new connections each time (zero-copy for graph)
      return algoMap.at("default_allreduce_rsag_zero_copy");
    }
    // Non-graph mode: use non-zero-copy algorithms
    if (messageSize <= (1 << 23)) {  // <= 8MB
      return algoMap.at("default_allreduce_rsag");
    }
    return algoMap.at("default_allreduce_rsag_pipeline");
  }

  // Symmetric memory path: can use cached memory handles
  bool const useNvlsWithZeroCopy = nvlsSupported && config.isCuMemMapAllocated;
  if (messageSize <= (1 << 16) ||
      (messageSize <= (1 << 20) &&
       !useNvlsWithZeroCopy)) {  // <= 64KB or <= 1MB
    return algoMap.at("default_allreduce_packet");
  }
  if (useNvlsWithZeroCopy) {
    return algoMap.at("default_allreduce_nvls_zero_copy");
  }

  return algoMap.at("default_allreduce_rsag_zero_copy");
}

std::shared_ptr<Algorithm> selectSingleNodeAllreduce(
    std::unordered_map<std::string, std::shared_ptr<Algorithm>> const& algoMap,
    CollectiveRequest const& request, AlgorithmSelectorConfig const& config) {
  // Use Blackwell-specific selection for compute capability 10.x
  if (config.computeCapability.first == 10) {
    return selectSingleNodeAllreduceBlackwell(algoMap, request, config);
  }

  const size_t messageSize = request.messageSize;

  // Determine NVLS availability based on data type and device capability
  bool const nvlsSupported = isNvlsSupportedForDataType(config, request.dtype);

  bool const useNvlsWithZeroCopy =
      nvlsSupported && config.symmetricMemory && config.isCuMemMapAllocated;

  // Very small messages: use allpair packet algorithm
  if (messageSize <= (1 << 14)) {  // <= 16KB
    return algoMap.at("default_allreduce_allpair_packet");
  }
  // Small messages with NVLS support
  if (messageSize <= (1 << 15) && nvlsSupported) {  // <= 32KB
    return algoMap.at("default_allreduce_nvls_packet");
  }
  // Medium messages: use packet algorithm
  if (messageSize <= (1 << 16) ||
      (messageSize <= (1 << 20) &&
       !useNvlsWithZeroCopy)) {  // <= 64KB or <= 1MB
    return algoMap.at("default_allreduce_packet");
  }
  // Large messages with NVLS zero-copy support
  if (nvlsSupported && useNvlsWithZeroCopy) {
    return algoMap.at("default_allreduce_nvls_zero_copy");
  }
  // Large messages with NVLS but without zero-copy
  if (nvlsSupported) {
    if (messageSize < (1 << 24)) {  // < 16MB
      return algoMap.at("default_allreduce_nvls_warp_pipeline");
    }
    return algoMap.at("default_allreduce_nvls_block_pipeline");
  }
#if defined(__HIP_PLATFORM_AMD__)
  // AMD platform: use fullmesh algorithm
  return algoMap.at("default_allreduce_fullmesh");
#else
  // NVIDIA without NVLS: use RSAG pipeline if no NCCL fallback
  if (!config.ncclDlopenSharedLib) {
    return algoMap.at("default_allreduce_fullmesh");
  }
  return nullptr;
#endif
}

std::shared_ptr<Algorithm> selectSingleNodeAllgather(
    std::unordered_map<std::string, std::shared_ptr<Algorithm>> const& algoMap,
    CollectiveRequest const& request,
    [[maybe_unused]] AlgorithmSelectorConfig const& config) {
  const size_t messageSize = request.messageSize;

  // For messages up to 32MB, use fullmesh2 algorithm
  if (messageSize <= 32 * (1 << 20)) {
    return algoMap.at("default_allgather_fullmesh2");
  }

#if defined(__HIP_PLATFORM_AMD__)
  // AMD platform always uses fullmesh2
  return algoMap.at("default_allgather_fullmesh2");
#else
  // NVIDIA: use fullmesh for large messages if no NCCL fallback is available
  if (!config.ncclDlopenSharedLib) {
    return algoMap.at("default_allgather_fullmesh");
  }
  return nullptr;
#endif
}

std::shared_ptr<Algorithm> selectMultiNodeAlgorithm(
    std::unordered_map<std::string, std::shared_ptr<Algorithm>> const& algoMap
    [[maybe_unused]],
    CollectiveRequest const& request [[maybe_unused]],
    AlgorithmSelectorConfig const& config [[maybe_unused]]) {
  // TODO: Implement multi-node algorithm selection
  // Multi-node scenarios will need to consider:
  // 1. Multi-node NVLS (if supported by hardware)
  // 2. Multi-node IB (InfiniBand)
  // 3. Hierarchical algorithms (intra-node + inter-node)
  // 4. Network topology awareness

  // For now, return nullptr to fallback to NCCL/RCCL
  INFO(MSCCLPP_NCCL,
       "Multi-node collective not yet supported, fallback to nccl/rccl");
  return nullptr;
}

}  // namespace nccl
}  // namespace mscclpp
