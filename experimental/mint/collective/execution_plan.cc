// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "execution_plan.hpp"
#include "debug.h"
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>

namespace {

template <typename T, typename Predicate>
std::vector<T> filter(std::vector<T> const& vec, Predicate pred) {
  std::vector<T> filtered;
  std::copy_if(vec.begin(), vec.end(), std::back_inserter(filtered), pred);
  return filtered;
}

auto getOpType = [](std::string const& str) {
  if (str == "nop") {
    return mscclpp::OperationType::NOP;
  } else if (str == "barrier") {
    return mscclpp::OperationType::BARRIER;
  } else if (str == "put") {
    return mscclpp::OperationType::PUT;
  } else if (str == "pws") {
    return mscclpp::OperationType::PUT_WITH_SIGNAL;
  } else if (str == "pwsf") {
    return mscclpp::OperationType::PUT_WITH_SIGNAL_AND_FLUSH;
  } else if (str == "get") {
    return mscclpp::OperationType::GET;
  } else if (str == "copy") {
    return mscclpp::OperationType::COPY;
  } else if (str == "signal") {
    return mscclpp::OperationType::SIGNAL;
  } else if (str == "wait") {
    return mscclpp::OperationType::WAIT;
  } else if (str == "flush") {
    return mscclpp::OperationType::FLUSH;
  } else if (str == "re") {
    return mscclpp::OperationType::REDUCE;
  } else if (str == "res") {
    return mscclpp::OperationType::REDUCE_SEND;
  } else if (str == "rre") {
    return mscclpp::OperationType::READ_REDUCE;
  } else if (str == "rres") {
    return mscclpp::OperationType::READ_REDUCE_SEND;
  } else if (str == "ppkt") {
    return mscclpp::OperationType::PUT_PACKETS;
  } else if (str == "rppkt") {
    return mscclpp::OperationType::READ_PUT_PACKETS;
  } else if (str == "respkt") {
    return mscclpp::OperationType::REDUCE_SEND_PACKETS;
  } else if (str == "cpkt") {
    return mscclpp::OperationType::COPY_PACKETS;
  } else if (str == "upkt") {
    return mscclpp::OperationType::UNPACK_PACKETS;
  } else if (str == "repkt") {
    return mscclpp::OperationType::REDUCE_PACKETS;
  } else if (str == "recpkt") {
    return mscclpp::OperationType::REDUCE_COPY_PACKETS;
  } else if (str == "recspkt") {
    return mscclpp::OperationType::REDUCE_COPY_SEND_PACKETS;
  } else if (str == "glres") {
    return mscclpp::OperationType::MULTI_LOAD_REDUCE_STORE;
  } else if (str == "rlxsignal") {
    return mscclpp::OperationType::RELAXED_SIGNAL;
  } else if (str == "rlxwait") {
    return mscclpp::OperationType::RELAXED_WAIT;
  } else if (str == "pipeline") {
    return mscclpp::OperationType::PIPELINE;
  } else if (str == "sem_acquire") {
    return mscclpp::OperationType::SEM_ACQUIRE;
  } else if (str == "sem_release") {
    return mscclpp::OperationType::SEM_RELEASE;
  } else {
    throw mscclpp::Error("Invalid operation type: " + str,
                         mscclpp::ErrorCode::ExecutorError);
  }
};

auto convertToBufferType = [](std::string const& str) {
  if (str == "i") {
    return mscclpp::BufferType::INPUT;
  } else if (str == "o") {
    return mscclpp::BufferType::OUTPUT;
  } else if (str == "s") {
    return mscclpp::BufferType::SCRATCH;
  } else {
    throw mscclpp::Error("Invalid buffer type",
                         mscclpp::ErrorCode::ExecutorError);
  }
};

auto convertToChannelType = [](std::string const& str) {
  if (str == "memory") {
    return mscclpp::ChannelType::MEMORY;
  } else if (str == "port") {
    return mscclpp::ChannelType::PORT;
  } else if (str == "none") {
    return mscclpp::ChannelType::NONE;
  } else if (str == "switch") {
    return mscclpp::ChannelType::SWITCH;
  } else {
    throw mscclpp::Error("Invalid channel type",
                         mscclpp::ErrorCode::ExecutorError);
  }
};

}  // namespace

namespace mscclpp {
using json = nlohmann::json;

ExecutionPlan::Impl::Impl(std::string const& planPath, int rank)
    : planPath(planPath), isUsingPacket(false), rank(rank) {
  std::ifstream file(this->planPath);
  json obj = json::parse(file);
  this->name = obj["name"];
  this->collective = obj["collective"];
  this->isInPlace = obj["inplace"];
  this->reuseResources = obj.value("reuse_resources", false);
  this->doubleScratchBuffer = obj.value("use_double_scratch_buffer", false);
  this->bufferAlignment = obj.value("buffer_alignment", 16);
  this->minMessageSize = obj.value("min_message_size", 0);
  this->maxMessageSize =
      obj.value("max_message_size", std::numeric_limits<uint64_t>::max());
}

std::vector<ChannelInfo> ExecutionPlan::Impl::getChannelInfos(
    ChannelType channelType) const {
  auto pred = [channelType](ChannelInfo const& info) {
    return info.channelType == channelType;
  };
  return filter(this->channelInfos_.at(rank), pred);
}

std::vector<ChannelInfo> ExecutionPlan::Impl::getUnpairedChannelInfos(
    int worldSize, ChannelType channelType) {
  std::vector<ChannelInfo> unpaired;
  for (int peer = 0; peer < worldSize; peer++) {
    if (peer == rank) {
      continue;
    }
    if (this->channelCountMap_[{rank, channelType}][peer] <
        this->channelCountMap_[{peer, channelType}][rank]) {
      int count = this->channelCountMap_[{peer, channelType}][rank] -
                  this->channelCountMap_[{rank, channelType}][peer];
      for (int i = 0; i < count; i++) {
        ChannelInfo info;
        info.channelType = channelType;
        info.connectedPeers.push_back(peer);
        unpaired.push_back(info);
      }
    }
  }
  return unpaired;
}

std::vector<int> ExecutionPlan::Impl::getConnectedPeers() const {
  std::set<int> peers;
  for (auto const& info : this->channelInfos_.at(rank)) {
    for (int peer : info.connectedPeers) {
      peers.insert(peer);
    }
  }
  if (this->channelInfosByDstRank_.find(rank) !=
      this->channelInfosByDstRank_.end()) {
    for (auto const& info : this->channelInfosByDstRank_.at(rank)) {
      for (int peer : info.connectedPeers) {
        peers.insert(peer);
      }
    }
  }
  return std::vector<int>(peers.begin(), peers.end());
}

std::vector<BufferInfo> ExecutionPlan::Impl::getRemoteBufferInfos() const {
  if (this->remoteBufferInfos_.find(rank) == this->remoteBufferInfos_.end()) {
    return std::vector<BufferInfo>();
  }
  return this->remoteBufferInfos_.at(rank);
}

std::vector<BufferInfo> ExecutionPlan::Impl::getLocalBufferToSend() const {
  if (this->localBufferToSend_.find(rank) == this->localBufferToSend_.end()) {
    return std::vector<BufferInfo>();
  }
  return this->localBufferToSend_.at(rank);
}

size_t ExecutionPlan::Impl::calScratchBufferSize(size_t inputSize,
                                                 size_t outputSize) const {
  if (reuseResources && this->scratchChunks > 0) {
    return PREDFINED_SCRATCH_SIZE;
  }

  size_t sizePerChunk = 0;
  size_t size = 0;
  if (this->inputChunks != 0) {
    sizePerChunk = (inputSize + this->inputChunks - 1) / this->inputChunks;
  } else if (this->outputChunks != 0) {
    sizePerChunk = (outputSize + this->outputChunks - 1) / this->outputChunks;
  } else {
    throw mscclpp::Error("Output or Input chunks must be greater than 0",
                         mscclpp::ErrorCode::ExecutorError);
  }

  if (this->isUsingPacket) {
    size = sizePerChunk * this->scratchChunks * 2; /* data + flag*/
  } else {
    size = sizePerChunk * this->scratchChunks;
  }

  if (this->doubleScratchBuffer) {
    size = size * 2;
  }
  return (size + this->bufferAlignment - 1) / this->bufferAlignment *
         this->bufferAlignment;
}

size_t ExecutionPlan::Impl::calMaxScratchChunkSize(size_t scratchSize) const {
  if (this->scratchChunks == 0) {
    return 0;
  }
  if (this->doubleScratchBuffer) {
    scratchSize = scratchSize / 2;
  }
  size_t size = (scratchSize + this->scratchChunks - 1) / this->scratchChunks;
  return (size + this->bufferAlignment - 1) / this->bufferAlignment *
         this->bufferAlignment;
}

std::vector<Operation> ExecutionPlan::Impl::getOperations(
    int threadblock) const {
  return this->operations[threadblock];
}

int ExecutionPlan::Impl::getThreadblockCount() const {
  return this->operations.size();
}

void ExecutionPlan::Impl::loadExecutionPlan(size_t inputSize, size_t outputSize,
                                            size_t contsSrcOffset,
                                            size_t constDstOffset) {
  std::ifstream file(this->planPath);
  json obj = json::parse(file);
  if (this->name != obj["name"]) {
    throw Error("Plan name does not match", ErrorCode::ExecutorError);
  }
  this->collective = obj["collective"];
  std::string protocol = obj["protocol"];
  if (protocol == "LL") {
    this->isUsingPacket = true;
  }
  this->inputSize = inputSize;
  this->outputSize = outputSize;
  this->nThreadsPerBlock = obj.value("num_threads_per_block", 1024);
  this->minMessageSize = obj.value("min_message_size", 0);
  this->maxMessageSize =
      obj.value("max_message_size", std::numeric_limits<uint64_t>::max());

  this->isInPlace = obj["inplace"];
  auto const& gpus = obj["gpus"];

  auto& gpu = gpus[rank];
  if (gpu["id"] != rank) {
    throw Error("GPU rank does not match", ErrorCode::ExecutorError);
  }
  this->inputChunks = gpu["input_chunks"];
  this->outputChunks = gpu["output_chunks"];
  this->scratchChunks = gpu["scratch_chunks"];
  checkMessageSize();

  this->setupChannels(gpus);
  this->setupRemoteBuffers(gpus);
  this->setupSemaphores(gpu);
  this->setupOperations(gpu, contsSrcOffset, constDstOffset);
}

void ExecutionPlan::Impl::lightLoadExecutionPlan(size_t inputSize,
                                                 size_t outputSize,
                                                 size_t contsSrcOffset,
                                                 size_t constDstOffset) {
  std::ifstream file(this->planPath);
  json obj = json::parse(file);
  if (this->name != obj["name"]) {
    throw Error("Plan name does not match", ErrorCode::ExecutorError);
  }
  std::string protocol = obj["protocol"];
  if (protocol == "LL") {
    this->isUsingPacket = true;
  }
  auto const& gpus = obj["gpus"];
  auto const& gpu = gpus[rank];
  if (gpu["id"] != rank) {
    throw Error("GPU rank does not match", ErrorCode::ExecutorError);
  }

  this->inputChunks = gpu["input_chunks"];
  this->outputChunks = gpu["output_chunks"];
  this->scratchChunks = gpu["scratch_chunks"];

  this->inputSize = inputSize;
  this->outputSize = outputSize;

  checkMessageSize();
  this->setupOperations(gpu, contsSrcOffset, constDstOffset);
}

void ExecutionPlan::Impl::checkMessageSize() const {
  size_t size = inputSize;
  if (inputSize % bufferAlignment != 0 || outputSize % bufferAlignment != 0 ||
      (inputSize / bufferAlignment) % inputChunks != 0 ||
      (outputSize / bufferAlignment) % outputChunks != 0) {
    throw Error(
        "Input or output size is not aligned with buffer alignment or chunks",
        ErrorCode::ExecutorError);
  }
  if (this->collective == "allgather") {
    size = outputSize;
  }
  if (size < this->minMessageSize || size > this->maxMessageSize) {
    throw Error("Input or output size is not within the valid range",
                ErrorCode::ExecutorError);
  }
}

void ExecutionPlan::Impl::parseChannels(
    json const& gpu, std::vector<ChannelInfo>& channelInfos,
    std::vector<NvlsInfo>& nvlsInfos,
    std::map<std::pair<int, ChannelType>, std::vector<int>>&
        chanConnectedPeersMap,
    int rank) {
  for (auto const& channel : gpu["channels"]) {
    ChannelType chanType = convertToChannelType(channel["channel_type"]);

    if (chanType == ChannelType::SWITCH) {
      NvlsInfo info;
      info.bufferType = convertToBufferType(channel["buffer_type"]);
      for (auto const& group : channel["rank_groups"]) {
        info.nChunks = (int)group["size"];
        info.ranks.clear();
        for (int rank : group["ranks"]) {
          info.ranks.push_back(rank);
        }
        nvlsInfos.push_back(info);
      }
    } else {
      ChannelInfo info;
      info.channelType = chanType;
      for (auto const& peer : channel["connected_to"]) {
        info.connectedPeers.push_back(peer);
        chanConnectedPeersMap[{peer, info.channelType}].push_back(rank);
        this->channelCountMap_[{rank, info.channelType}][peer]++;
      }
      channelInfos.push_back(info);
    }
  }
}

void ExecutionPlan::Impl::parseRemoteBuffer(nlohmann::json const& gpus) {
  for (auto const& gpu : gpus) {
    std::unordered_map<ChannelType, int> channelCountMap;
    int gpuRank = gpu["id"];
    auto& bufferInfos = this->remoteBufferInfos_[gpuRank];
    auto& bufferIndexMap = this->bufferIndexMap_[gpuRank];
    for (auto& remoteBuffer : gpu["remote_buffers"]) {
      int bufferId = bufferInfos.size();
      int oriRank = remoteBuffer["rank"];
      BufferType bufferType = convertToBufferType(remoteBuffer["type"]);
      std::vector<ChannelType> accessChannels;
      for (auto const& channel : remoteBuffer["access_channel_types"]) {
        ChannelType chanType = convertToChannelType(channel);
        accessChannels.push_back(chanType);
        bufferIndexMap[{bufferId, chanType}] = channelCountMap[chanType]++;
      }
      BufferInfo info{oriRank, gpuRank, bufferType, accessChannels};
      bufferInfos.push_back(info);
      this->localBufferToSend_[oriRank].push_back(info);
    }
  }
}

// Construct the channel info. Step 1. Flatten MEMORY and PORT channels into
// separate vectors. Step 2. For each threadblock, construct a vector of channel
// indexes and keys.
void ExecutionPlan::Impl::setupChannels(json const& gpus) {
  using mapKey = std::pair<int, ChannelType>;
  std::map<mapKey, std::vector<int>> chanConnectedPeersMap;
  for (auto const& gpu : gpus) {
    int rank = gpu["id"];
    std::vector<ChannelInfo> channelInfos;
    std::vector<NvlsInfo> nvlsInfos;
    this->parseChannels(gpu, channelInfos, nvlsInfos, chanConnectedPeersMap,
                        rank);
    this->channelInfos_[rank] = channelInfos;
    this->nvlsInfos[rank] = nvlsInfos;
  }

  for (auto const& [key, connectedFrom] : chanConnectedPeersMap) {
    auto [peer, channelType] = key;
    ChannelInfo info;
    info.channelType = channelType;
    info.connectedPeers = connectedFrom;
    this->channelInfosByDstRank_[peer].push_back(info);
  }

  // setup threadblockChannels
  auto const& gpu = gpus[rank];
  int nthreadblocks = gpu["threadblocks"].size();
  this->threadblockMemoryChannels.resize(nthreadblocks);
  this->threadblockPortChannels.resize(nthreadblocks);
  this->threadblockNvlsChannels.resize(nthreadblocks);
  for (auto const& threadblock : gpu["threadblocks"]) {
    for (auto const& channel : threadblock["channels"]) {
      ChannelType channelType = convertToChannelType(channel["channel_type"]);
      ChannelKey key = {BufferType::NONE, channelType};
      if (channel.contains("buff")) {
        key = {convertToBufferType(channel["buff"]), channelType};
      }
      for (int id : channel["channel_ids"]) {
        if (channelType == ChannelType::MEMORY) {
          this->threadblockMemoryChannels[threadblock["id"]].emplace_back(id);
        } else if (channelType == ChannelType::PORT) {
          this->threadblockPortChannels[threadblock["id"]].emplace_back(id);
        } else if (channelType == ChannelType::SWITCH) {
          this->threadblockNvlsChannels[threadblock["id"]].emplace_back(id);
        }
      }
    }
  }
}

void ExecutionPlan::Impl::setupRemoteBuffers(json const& gpus) {
  this->parseRemoteBuffer(gpus);

  // setup threadblockBuffers
  auto const& gpu = gpus[rank];
  int nthreadblocks = gpu["threadblocks"].size();
  this->threadblockMemoryChannelBuffers.resize(nthreadblocks);
  this->threadblockPortChannelBuffers.resize(nthreadblocks);
  for (auto const& threadblock : gpu["threadblocks"]) {
    if (!threadblock.contains("remote_buffer_refs")) {
      continue;
    }
    for (auto const& remoteBuffRef : threadblock["remote_buffer_refs"]) {
      ChannelType accessChanType =
          convertToChannelType(remoteBuffRef["access_channel_type"]);
      if (accessChanType == ChannelType::PORT) {
        for (auto const& bufferId : remoteBuffRef["remote_buffer_ids"]) {
          BufferType type = this->remoteBufferInfos_[rank][bufferId].bufferType;
          this->threadblockPortChannelBuffers[threadblock["id"]].push_back(
              {this->bufferIndexMap_[rank][{bufferId, accessChanType}], type});
        }
      } else if (accessChanType == ChannelType::MEMORY) {
        for (auto const& bufferId : remoteBuffRef["remote_buffer_ids"]) {
          BufferType type = this->remoteBufferInfos_[rank][bufferId].bufferType;
          this->threadblockMemoryChannelBuffers[threadblock["id"]].push_back(
              {this->bufferIndexMap_[rank][{bufferId, accessChanType}], type});
        }
      }
    }
  }
}

void ExecutionPlan::Impl::setupSemaphores(json const& gpu) {
  if (!gpu.contains("semaphores")) {
    return;
  }
  for (auto const& sem : gpu["semaphores"]) {
    SemaphoreInfo info;
    info.initValue = sem["init_value"];
    this->semaphoreInfos.push_back(info);
  }
}

void ExecutionPlan::Impl::setupOperations(json const& gpu,
                                          size_t constSrcOffset,
                                          size_t constDstOffset) {
  // setup threadblocks and operations
  for (auto const& threadblock : gpu["threadblocks"]) {
    int threadBlockId = threadblock["id"];
    std::vector<Operation> ops;
    for (auto const& op : threadblock["ops"]) {
      Operation operation = {};
      this->setupOperation(op, operation, rank, threadBlockId, constSrcOffset,
                           constDstOffset);
      ops.push_back(operation);
      if (operation.type == OperationType::PIPELINE) {
        for (auto const& innerOp : op["ops"]) {
          Operation pipelineOp = {};
          this->setupOperation(innerOp, pipelineOp, rank, threadBlockId,
                               constSrcOffset, constDstOffset);
          ops.push_back(pipelineOp);
        }
      }
    }
    this->operations.push_back(ops);
  }
}

void ExecutionPlan::Impl::setupOperation(nlohmann::json const& op,
                                         Operation& operation, int rank,
                                         int threadBlockId,
                                         size_t constSrcOffset,
                                         size_t constDstOffset) {
  auto getConstOffset = [&](BufferType type) -> size_t {
    switch (type) {
      case BufferType::INPUT:
        return constSrcOffset;
      case BufferType::OUTPUT:
        return constDstOffset;
      case BufferType::SCRATCH:
        return 0;
      default:
        throw Error("Invalid buffer type", ErrorCode::ExecutorError);
    }
  };

  auto getRemoteBufferTypeWithId = [&](int bufferId, int threadBlockId,
                                       ChannelType channelType) -> BufferType {
    if (channelType == ChannelType::MEMORY) {
      return this->threadblockMemoryChannelBuffers[threadBlockId][bufferId]
          .second;
    }
    if (channelType == ChannelType::PORT) {
      return this->threadblockPortChannelBuffers[threadBlockId][bufferId]
          .second;
    }
    throw Error("Invalid channel type", ErrorCode::ExecutorError);
  };

  uint32_t tbId = 0;
  uint32_t tbgSize = 1;

  operation.type = static_cast<mscclpp::OperationType>(getOpType(op["name"]));
  if (op.contains("channel_type")) {
    operation.channelType = convertToChannelType(op["channel_type"]);
  }
  if (op.contains("channel_ids")) {
    operation.nChannels = op["channel_ids"].size();
    for (uint32_t i = 0; i < op["channel_ids"].size(); i++) {
      operation.channelIndexes[i] = op["channel_ids"][i];
    }
  }
  if (op.contains("tbg_info")) {
    tbId = op["tbg_info"]["tb_id"];
    tbgSize = op["tbg_info"]["tbg_size"];
  }
  if (op.contains("src_buff")) {
    operation.nInputs = op["src_buff"].size();
    for (int i = 0; i < operation.nInputs; i++) {
      auto& buff = op["src_buff"][i];
      size_t constOffset = 0;
      BufferType bufferType = BufferType::NONE;
      if (buff.contains("type")) {
        bufferType = convertToBufferType(buff["type"]);
        operation.inputBufferRefs[i].type = bufferType;
      }
      if (buff.contains("buffer_id")) {
        operation.inputBufferRefs[i].id = buff["buffer_id"];
        bufferType = getRemoteBufferTypeWithId(buff["buffer_id"], threadBlockId,
                                               operation.channelType);
        constOffset = getConstOffset(bufferType);
      }
      if (buff.contains("switch_channel_id")) {
        int switchChannelIdx =
            this->threadblockNvlsChannels[threadBlockId]
                                         [buff["switch_channel_id"]];
        bufferType = this->nvlsInfos[rank][switchChannelIdx].bufferType;
        constOffset = getConstOffset(bufferType);
        operation.nvlsInputBufferType = bufferType;
        operation.nvlsInputIndex = buff["switch_channel_id"];
      }
      size_t inputOffset = this->getOffset(this->inputSize, this->outputSize,
                                           buff["index"], bufferType) +
                           constOffset;
      size_t inputBufferSize = this->getBufferSize(
          this->inputSize, this->outputSize, buff["index"], buff["size"]);
      inputOffset += calcOffset(inputBufferSize, tbId, tbgSize);
      inputBufferSize = calcSize(inputBufferSize, tbId, tbgSize);
      operation.inputOffsets[i] = inputOffset;
      operation.inputBufferSizes[i] = inputBufferSize;
    }
  }
  if (op.contains("dst_buff")) {
    operation.nOutputs = op["dst_buff"].size();
    for (int i = 0; i < operation.nOutputs; i++) {
      auto& buff = op["dst_buff"][i];
      size_t constOffset = 0;
      BufferType bufferType = BufferType::NONE;
      if (buff.contains("type")) {
        bufferType = convertToBufferType(buff["type"]);
        operation.outputBufferRefs[i].type = bufferType;
      }
      if (buff.contains("buffer_id")) {
        operation.outputBufferRefs[i].id = buff["buffer_id"];
        bufferType = getRemoteBufferTypeWithId(buff["buffer_id"], threadBlockId,
                                               operation.channelType);
        constOffset = getConstOffset(bufferType);
      }
      if (buff.contains("switch_channel_id")) {
        int switchChannelIdx =
            this->threadblockNvlsChannels[threadBlockId]
                                         [buff["switch_channel_id"]];
        bufferType = this->nvlsInfos[rank][switchChannelIdx].bufferType;
        constOffset = getConstOffset(bufferType);
        operation.nvlsOutputBufferType = bufferType;
        operation.nvlsOutputIndex = buff["switch_channel_id"];
      }
      size_t outputOffset = this->getOffset(this->inputSize, this->outputSize,
                                            buff["index"], bufferType) +
                            constOffset;
      size_t outputBufferSize = this->getBufferSize(
          this->inputSize, this->outputSize, buff["index"], buff["size"]);
      outputOffset += calcOffset(outputBufferSize, tbId, tbgSize);
      outputBufferSize = calcSize(outputBufferSize, tbId, tbgSize);
      operation.outputOffsets[i] = outputOffset;
      operation.outputBufferSizes[i] = outputBufferSize;
    }
  }
  if (op.contains("barrier_id")) {
    operation.deviceSyncerIndex = op["barrier_id"];
  }
  if (op.contains("num_threadblocks")) {
    operation.nThreadBlocks = op["num_threadblocks"];
  }
  if (op.contains("semaphore_ids")) {
    operation.nDeviceSemaphores = op["semaphore_ids"].size();
    for (uint32_t id = 0; id < operation.nDeviceSemaphores; id++) {
      operation.deviceSemaphoreIds[id] = op["semaphore_ids"][id];
    }
  }
  if (op.contains("iter_context")) {
    operation.unitSize = op["iter_context"]["unit_size"];
    operation.nOperations = op["ops"].size();
    int nChunks = op["iter_context"]["num_chunks"];
    size_t sizes =
        nChunks * getUpperBoundChunkSize(this->inputSize, this->outputSize);
    operation.nIterations =
        (sizes + (operation.unitSize - 1)) / operation.unitSize;
  }
}

std::pair<size_t, uint32_t> ExecutionPlan::Impl::getSizeAndChunks(
    size_t inputSize, size_t outputSize) const {
  std::pair<size_t, uint32_t> sizePerRank;
  if (this->inputChunks == 0 && this->outputChunks == 0) {
    throw mscclpp::Error("Output or Input chunks must be greater than 0",
                         mscclpp::ErrorCode::ExecutorError);
  } else if (this->inputChunks != 0 && this->outputChunks != 0) {
    if (inputSize / this->inputChunks != outputSize / this->outputChunks)
      throw mscclpp::Error("Size per chunks inconsistent: inputSize " +
                               std::to_string(inputSize) + " inputChunks " +
                               std::to_string(this->inputChunks) +
                               " outputSize " + std::to_string(outputSize) +
                               " outputChunks " +
                               std::to_string(this->outputChunks),
                           mscclpp::ErrorCode::ExecutorError);
    else
      sizePerRank = std::make_pair(inputSize, this->inputChunks);
  } else if (this->inputChunks != 0) {
    sizePerRank = std::make_pair(inputSize, this->inputChunks);
  } else if (this->outputChunks != 0) {
    sizePerRank = std::make_pair(outputSize, this->outputChunks);
  }
  return sizePerRank;
}

size_t ExecutionPlan::Impl::calcOffset(size_t size, uint32_t index,
                                       uint32_t slices) const {
  uint32_t nelems = size / (this->bufferAlignment * sizeof(uint8_t));
  uint32_t minNelems = nelems / slices;
  uint32_t remainder = nelems % slices;
  uint32_t offset = index * minNelems +
                    (index % nelems < remainder ? index % nelems : remainder);
  return static_cast<size_t>(offset) * this->bufferAlignment;
}

size_t ExecutionPlan::Impl::calcSize(size_t size, uint32_t index,
                                     uint32_t slices) const {
  uint32_t beginOff = calcOffset(size, index, slices);
  uint32_t endOff = calcOffset(size, index + 1, slices);
  return endOff - beginOff;
}

size_t ExecutionPlan::Impl::getOffset(size_t inputSize, size_t outputSize,
                                      uint32_t chunkIndex,
                                      BufferType bufferType) const {
  auto rankSizeAndChunks = getSizeAndChunks(inputSize, outputSize);
  uint32_t nChunks = rankSizeAndChunks.second;
  uint32_t chunkSize = (rankSizeAndChunks.first + nChunks - 1) / nChunks;
  uint32_t scratchChunkSize =
      this->calMaxScratchChunkSize(PREDFINED_SCRATCH_SIZE);

  // Reuse scratch buffer for large input/output chunks
  if (bufferType == BufferType::SCRATCH && this->reuseResources &&
      scratchChunkSize < chunkSize) {
    return chunkIndex * this->calMaxScratchChunkSize(PREDFINED_SCRATCH_SIZE);
  }

  return calcOffset(rankSizeAndChunks.first, chunkIndex, nChunks);
}

size_t ExecutionPlan::Impl::getBufferSize(size_t inputSize, size_t outputSize,
                                          uint32_t index,
                                          uint32_t nChunks) const {
  uint32_t beginOff = getOffset(inputSize, outputSize, index);
  uint32_t endOff = getOffset(inputSize, outputSize, index + nChunks);
  return endOff - beginOff;
}

size_t ExecutionPlan::Impl::getUpperBoundChunkSize(size_t inputSize,
                                                   size_t outputSize) const {
  size_t nInputChunks = this->inputChunks;
  size_t nOutputChunks = this->outputChunks;
  if (nInputChunks != 0) {
    size_t nelems = inputSize / this->bufferAlignment;
    return (nelems + nInputChunks - 1) / nInputChunks * this->bufferAlignment;
  }
  if (nOutputChunks != 0) {
    size_t nelems = outputSize / this->bufferAlignment;
    return (nelems + nOutputChunks - 1) / nOutputChunks * this->bufferAlignment;
  }
  throw mscclpp::Error("Output or Input chunks must be greater than 0",
                       mscclpp::ErrorCode::ExecutorError);
}

void ExecutionPlan::Impl::reset() {
  this->operations.clear();
  this->nvlsInfos.clear();
  this->threadblockMemoryChannels.clear();
  this->threadblockPortChannels.clear();
  this->threadblockNvlsChannels.clear();
  this->threadblockMemoryChannelBuffers.clear();
  this->threadblockPortChannelBuffers.clear();
  this->semaphoreInfos.clear();

  this->channelInfos_.clear();
  this->channelCountMap_.clear();
  this->channelInfosByDstRank_.clear();
  this->remoteBufferInfos_.clear();
  this->localBufferToSend_.clear();
  this->bufferIndexMap_.clear();
}

void ExecutionPlan::Impl::operationsReset() { this->operations.clear(); }

ExecutionPlan::ExecutionPlan(std::string const& planPath, int rank)
    : impl_(std::make_shared<Impl>(planPath, rank)) {}

std::string const& ExecutionPlan::name() const { return this->impl_->name; }

std::string const& ExecutionPlan::collective() const {
  return this->impl_->collective;
}

size_t ExecutionPlan::minMessageSize() const {
  return this->impl_->minMessageSize;
}

size_t ExecutionPlan::maxMessageSize() const {
  return this->impl_->maxMessageSize;
}

bool ExecutionPlan::isInPlace() const { return this->impl_->isInPlace; }

}  // namespace mscclpp
