// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "allreduce_allpair_packet.hpp"
#include "collective_utils.hpp"
#include "common.hpp"
#include "debug.h"
#include <collective_utils.hpp>

namespace mscclpp {
namespace collective {

template <ReduceOp OpType, typename T>
__global__ void allreduceAllPairs(T* buff, T* scratch, T* resultBuff,
                                  DeviceHandle<MemoryChannel>* memoryChannels,
                                  size_t channelDataOffset,
                                  size_t scratchBufferSize, int rank,
                                  int nRanksPerNode, int worldSize,
                                  size_t nelems, uint32_t numScratchBuff,
                                  void* flags, uint32_t flagSize) {
  // This version of allreduce only works for single nodes
  if (worldSize != nRanksPerNode) return;

  if (sizeof(T) == 2 || sizeof(T) == 1)
    nelems = (nelems * sizeof(T) + sizeof(T)) / sizeof(int);
  int const nPeers = nRanksPerNode - 1;

  uint32_t flag = ((uint32_t*)flags)[blockIdx.x];
  size_t scratchBaseOffset =
      (flag % numScratchBuff) ? (scratchBufferSize / numScratchBuff) : 0;
  size_t channelScratchOffset = scratchBaseOffset;

  int const nBlocksPerPeer = gridDim.x / nPeers;
  int const localBlockIdx = blockIdx.x % nBlocksPerPeer;
  int const tid = threadIdx.x + localBlockIdx * blockDim.x;
  int const peerIdx = blockIdx.x / nBlocksPerPeer;
  size_t srcOffset = channelDataOffset;
  size_t scratchOffset =
      channelScratchOffset + rank * nelems * sizeof(LL8Packet);
  void* scratchBuff = (void*)((char*)scratch + channelScratchOffset);
  uint32_t* src = (uint32_t*)((char*)buff);
  uint32_t* dst = (uint32_t*)((char*)resultBuff);

  // step 1: write data to each peer's scratch buffer
  memoryChannels[peerIdx].putPackets<LL8Packet>(
      scratchOffset, srcOffset, nelems * sizeof(uint32_t), tid,
      blockDim.x * nBlocksPerPeer, flag);

  // step 2: Reduce Data
  for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nelems;
       idx += blockDim.x * gridDim.x) {
    uint32_t data = src[idx];
    for (int index = 0; index < nPeers; index++) {
      int const remoteRank = index < rank ? index : index + 1;
      LL8Packet* dstPkt = (LL8Packet*)scratchBuff + remoteRank * nelems;
      uint32_t val = dstPkt[idx].read(flag, -1);
      data = cal_vector<T, OpType>(val, data);
    }
    dst[idx] = data;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    ((uint32_t*)flags)[blockIdx.x] = flag + 1;
  }
  if (blockIdx.x == 0 && threadIdx.x >= gridDim.x &&
      threadIdx.x < flagSize / sizeof(uint32_t)) {
    ((uint32_t*)flags)[threadIdx.x] = flag + 1;
  }
}

inline std::pair<int, int> getDefaultBlockNumAndThreadNum(size_t inputSize,
                                                          int worldSize) {
  if (inputSize < worldSize * sizeof(int)) {
    return {worldSize - 1, 32};
  }
  return {(worldSize - 1) * 4, 512};
}

template <ReduceOp OpType, typename T>
struct AllpairAdapter {
  static cudaError_t call(void const* buff, void* scratch, void* resultBuff,
                          void* memoryChannels, void*,
                          DeviceHandle<SwitchChannel>*,
                          DeviceHandle<SwitchChannel>*, size_t channelInOffset,
                          size_t, size_t scratchBufferSize, int rank,
                          int nRanksPerNode, int worldSize, size_t inputSize,
                          cudaStream_t stream, void* flags, uint32_t flagSize,
                          uint32_t numScratchBuff, int nBlocks = 0,
                          int nThreadsPerBlock = 0) {
    using ChannelType = DeviceHandle<MemoryChannel>;
    const size_t nelems = inputSize / sizeof(T);
    allreduceAllPairs<OpType, T><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
        (T*)buff, (T*)scratch, (T*)resultBuff, (ChannelType*)memoryChannels,
        channelInOffset, scratchBufferSize, rank, nRanksPerNode, worldSize,
        nelems, numScratchBuff, flags, flagSize);
    return cudaGetLastError();
  }
};

void AllreduceAllpairPacket::initialize(std::shared_ptr<Communicator> comm) {
  conns_ = setupConnections(comm);
  memorySemaphores_ = setupMemorySemaphores(comm, conns_, maxBlockNum_);
  RegisteredMemory scratchMemory = comm->registerMemory(
      scratchBuffer_, scratchBufferSize_, Transport::CudaIpc);
  registeredMemories_ =
      setupRemoteMemories(comm, comm->bootstrap()->getRank(), scratchMemory);
  registeredMemories_.push_back(scratchMemory);
}

CommResult AllreduceAllpairPacket::allreduceKernelFunc(
    const std::shared_ptr<void> ctx, void const* input, void* output,
    size_t inputSize, [[maybe_unused]] DataType dtype, ReduceOp op,
    cudaStream_t stream, int nBlocks, int nThreadsPerBlock,
    std::unordered_map<std::string, uintptr_t> const&) {
  auto algoCtx = std::static_pointer_cast<AlgorithmCtx>(ctx);
  std::pair<int, int> blockAndThreadNum{nBlocks, nThreadsPerBlock};
  if (blockAndThreadNum.first == 0 || blockAndThreadNum.second == 0) {
    blockAndThreadNum =
        getDefaultBlockNumAndThreadNum(inputSize, algoCtx->workSize);
  }
  size_t sendBytes;
  CUdeviceptr sendBasePtr;
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  size_t channelInOffset = (char*)input - (char*)sendBasePtr;

  AllreduceFunc allreduce = dispatch<AllpairAdapter>(op, dtype);
  if (!allreduce) {
    WARN("Unsupported operation or data type for allreduce: op=%d, dtype=%d",
         op, static_cast<int>(dtype));
    return CommResult::CommInvalidArgument;
  }
  cudaError_t error =
      allreduce(input, this->scratchBuffer_, output,
                algoCtx->memoryChannelDeviceHandles.get(), nullptr, nullptr,
                nullptr, channelInOffset, 0, this->scratchBufferSize_,
                algoCtx->rank, algoCtx->nRanksPerNode, algoCtx->workSize,
                inputSize, stream, (void*)flagBuffer_,
                (uint32_t)flagBufferSize_, this->nSegmentsForScratchBuffer_,
                blockAndThreadNum.first, blockAndThreadNum.second);
  if (error != cudaSuccess) {
    WARN("AllreducePacket failed with error: %s", cudaGetErrorString(error));
    return CommResult::CommUnhandledCudaError;
  }
  return CommResult::CommSuccess;
}

std::shared_ptr<void> AllreduceAllpairPacket::initAllreduceContext(
    std::shared_ptr<Communicator> comm, void const* input, void*, size_t,
    DataType) {
  auto ctx = std::make_shared<AlgorithmCtx>();
  int const nChannelsPerConnection = maxBlockNum_;
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();
  ctx->memorySemaphores = this->memorySemaphores_;
  ctx->registeredMemories = this->registeredMemories_;
  ctx->registeredMemories
      .pop_back();  // remove the local memory from previous context

  size_t sendBytes;
  CUdeviceptr sendBasePtr;
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  RegisteredMemory localMemory =
      comm->registerMemory((void*)sendBasePtr, sendBytes, Transport::CudaIpc);

  // setup channels
  ctx->memoryChannels = setupMemoryChannels(
      this->conns_, ctx->memorySemaphores, ctx->registeredMemories, localMemory,
      nChannelsPerConnection);
  ctx->memoryChannelDeviceHandles =
      setupMemoryChannelDeviceHandles(ctx->memoryChannels);
  ctx->registeredMemories.emplace_back(localMemory);
  return ctx;
}

AlgorithmCtxKey AllreduceAllpairPacket::generateAllreduceContextKey(
    void const* input, void*, size_t, DataType, bool) {
  size_t sendBytes;
  CUdeviceptr sendBasePtr;
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  return AlgorithmCtxKey{(void*)sendBasePtr, nullptr, sendBytes, 0, 0};
}

std::shared_ptr<Algorithm> AllreduceAllpairPacket::build() {
  auto self = std::make_shared<AllreduceAllpairPacket>(
      reinterpret_cast<uintptr_t>(scratchBuffer_), scratchBufferSize_,
      flagBuffer_, flagBufferSize_);
  return std::make_shared<NativeAlgorithm>(
      "default_allreduce_allpair_packet", "allreduce",
      [self](std::shared_ptr<Communicator> comm) { self->initialize(comm); },
      [self](const std::shared_ptr<void> ctx, void const* input, void* output,
             size_t inputSize, [[maybe_unused]] size_t outputSize,
             DataType dtype, ReduceOp op, cudaStream_t stream, int nBlocks,
             int nThreadsPerBlock,
             std::unordered_map<std::string, uintptr_t> const& extras) {
        return self->allreduceKernelFunc(ctx, input, output, inputSize, dtype,
                                         op, stream, nBlocks, nThreadsPerBlock,
                                         extras);
      },
      [self](std::shared_ptr<Communicator> comm, void const* input,
             void* output, size_t inputSize, [[maybe_unused]] size_t outputSize,
             DataType dtype) {
        return self->initAllreduceContext(comm, input, output, inputSize,
                                          dtype);
      },
      [self](void const* input, void* output, size_t inputSize,
             [[maybe_unused]] size_t outputSize, DataType dtype,
             bool symmetricMemory) {
        return self->generateAllreduceContextKey(input, output, inputSize,
                                                 dtype, symmetricMemory);
      });
}
}  // namespace collective
}  // namespace mscclpp