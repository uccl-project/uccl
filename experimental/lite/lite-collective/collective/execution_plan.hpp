// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EXECUTOR_PLAN_HPP_
#define MSCCLPP_EXECUTOR_PLAN_HPP_

#include "core.hpp"
#include "execution_common.hpp"
#include "executor.hpp"
#include "utils.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

namespace mscclpp {

struct ChannelKey {
  BufferType bufferType;
  ChannelType channelType;
  bool operator==(ChannelKey const& other) const {
    return bufferType == other.bufferType && channelType == other.channelType;
  }
};

struct NvlsInfo {
  std::vector<int> ranks;
  uint32_t nChunks;
  BufferType bufferType;
};
}  // namespace mscclpp

namespace std {
template <>
struct hash<mscclpp::ChannelKey> {
  std::size_t operator()(mscclpp::ChannelKey const& key) const {
    std::size_t seed = 0;
    mscclpp::detail::hashCombine(seed, static_cast<int>(key.bufferType));
    mscclpp::detail::hashCombine(seed, static_cast<int>(key.channelType));
    return seed;
  }
};

template <>
struct hash<std::pair<int, mscclpp::ChannelType>> {
  std::size_t operator()(
      std::pair<int, mscclpp::ChannelType> const& key) const {
    std::size_t seed = 0;
    mscclpp::detail::hashCombine(seed, key.first);
    mscclpp::detail::hashCombine(seed, static_cast<int>(key.second));
    return seed;
  }
};
}  // namespace std

namespace mscclpp {

struct ChannelInfo {
  ChannelType channelType;
  std::vector<int> connectedPeers;
};

struct BufferInfo {
  int rank;
  int accessRank;
  BufferType bufferType;
  std::vector<ChannelType> accessChannelTypes;
};

struct SemaphoreInfo {
  int initValue;
};

struct ExecutionPlan::Impl {
 public:
  Impl(std::string const& planPath, int rank);
  ~Impl() = default;

  void loadExecutionPlan(size_t inputSize, size_t outputSize,
                         size_t contsSrcOffset, size_t constDstOffset);
  void lightLoadExecutionPlan(size_t inputSize, size_t outputSize,
                              size_t contsSrcOffset, size_t constDstOffset);
  size_t calScratchBufferSize(size_t inputSize, size_t outputSize) const;
  size_t calMaxScratchChunkSize(size_t scratchSize) const;

  std::vector<ChannelInfo> getChannelInfos(ChannelType channelType) const;
  std::vector<ChannelInfo> getUnpairedChannelInfos(int worldSize,
                                                   ChannelType channelType);
  std::vector<int> getConnectedPeers() const;
  std::vector<BufferInfo> getRemoteBufferInfos() const;
  std::vector<BufferInfo> getLocalBufferToSend() const;
  std::vector<Operation> getOperations(int threadblock) const;
  int getThreadblockCount() const;

  void reset();
  void operationsReset();

  std::string name;
  std::string collective;
  const std::string planPath;
  bool isUsingPacket;
  bool reuseResources;
  int rank;

  // operations for current ranks [threadblock] = list[operations]
  std::vector<std::vector<Operation>> operations;
  std::vector<SemaphoreInfo> semaphoreInfos;
  // for nvls channels
  std::unordered_map<int, std::vector<NvlsInfo>> nvlsInfos;

  // threadblockChannels[threadblock] = channelIndexes
  std::vector<std::vector<int>> threadblockMemoryChannels;
  std::vector<std::vector<int>> threadblockPortChannels;
  std::vector<std::vector<int>> threadblockNvlsChannels;
  // threadblockBuffers[threadblock] = {bufferIndexes, bufferType}
  std::vector<std::vector<std::pair<int, BufferType>>>
      threadblockMemoryChannelBuffers;
  std::vector<std::vector<std::pair<int, BufferType>>>
      threadblockPortChannelBuffers;

  uint32_t inputChunks;
  uint32_t outputChunks;
  uint32_t scratchChunks;
  size_t inputSize;
  size_t outputSize;
  int nThreadsPerBlock;
  size_t minMessageSize;
  size_t maxMessageSize;
  uint32_t bufferAlignment;
  bool isInPlace;
  bool doubleScratchBuffer;

 private:
  std::pair<size_t, uint32_t> getSizeAndChunks(size_t inputSize,
                                               size_t outputSize) const;
  size_t calcOffset(size_t size, uint32_t index, uint32_t slices) const;
  size_t calcSize(size_t size, uint32_t index, uint32_t slices) const;
  size_t getOffset(size_t inputSize, size_t outputSize, uint32_t chunkIndex,
                   BufferType bufferType = BufferType::NONE) const;
  size_t getBufferSize(size_t inputSize, size_t outputSize, uint32_t index,
                       uint32_t nChunks) const;
  size_t getUpperBoundChunkSize(size_t inputSize, size_t outputSize) const;

  void setupChannels(nlohmann::json const& gpus);
  void setupRemoteBuffers(nlohmann::json const& gpus);
  void setupSemaphores(nlohmann::json const& gpu);
  void setupOperations(nlohmann::json const& gpu, size_t contsSrcOffset,
                       size_t constDstOffset);
  void setupOperation(nlohmann::json const& op, Operation& operation, int rank,
                      int threadBlockId, size_t constSrcOffset,
                      size_t constDstOffset);
  // helper functions to setup the channels
  void parseChannels(nlohmann::json const& gpu,
                     std::vector<ChannelInfo>& channelInfos,
                     std::vector<NvlsInfo>& nvlsInfos,
                     std::map<std::pair<int, ChannelType>, std::vector<int>>&
                         chanConnectedPeersMap,
                     int rank);
  void parseRemoteBuffer(nlohmann::json const& gpus);
  void checkMessageSize() const;

  std::unordered_map<std::pair<int, ChannelType>, std::unordered_map<int, int>>
      channelCountMap_;
  std::unordered_map<int, std::unordered_map<std::pair<int, ChannelType>, int>>
      bufferIndexMap_;
  std::unordered_map<int, std::vector<ChannelInfo>> channelInfos_;
  std::unordered_map<int, std::vector<ChannelInfo>> channelInfosByDstRank_;
  std::unordered_map<int, std::vector<BufferInfo>> remoteBufferInfos_;
  std::unordered_map<int, std::vector<BufferInfo>> localBufferToSend_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_EXECUTOR_PLAN_HPP_
