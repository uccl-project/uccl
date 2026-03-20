// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "allreduce_fullmesh.hpp"
#include "collective_utils.hpp"
#include "common.hpp"
#include "debug.h"

namespace mscclpp {
namespace collective {

template <ReduceOp OpType, typename T>
__global__ void __launch_bounds__(512, 1)
    allreduceFullmesh(T* buff, T* scratch, T* resultBuff,
                      DeviceHandle<MemoryChannel>* memoryChannels,
                      DeviceHandle<MemoryChannel>* memoryOutChannels,
                      size_t channelOutDataOffset, int rank, int nRanksPerNode,
                      int worldSize, size_t nelems) {
  int const nPeer = nRanksPerNode - 1;
  const size_t chanOffset = nPeer * blockIdx.x;
  // assume (nelems * sizeof(T)) is divisible by (16 * worldSize)
  const size_t nInt4 = nelems * sizeof(T) / sizeof(int4);
  const size_t nInt4PerRank = nInt4 / worldSize;
  auto memoryChans = memoryChannels + chanOffset;
  auto memoryOutChans = memoryOutChannels + chanOffset;

  int4* buff4 = reinterpret_cast<int4*>(buff);
  int4* scratch4 = reinterpret_cast<int4*>((char*)scratch);
  int4* resultBuff4 = reinterpret_cast<int4*>(resultBuff);

  // Distribute `nInt4PerRank` across all blocks with the unit size `unitNInt4`
  constexpr size_t unitNInt4 = 512;
  const size_t maxNInt4PerBlock =
      (((nInt4PerRank + gridDim.x - 1) / gridDim.x) + unitNInt4 - 1) /
      unitNInt4 * unitNInt4;
  size_t offsetOfThisBlock = maxNInt4PerBlock * blockIdx.x;
  size_t nInt4OfThisBlock = maxNInt4PerBlock;
  size_t nNeededBlocks =
      (nInt4PerRank + maxNInt4PerBlock - 1) / maxNInt4PerBlock;
  constexpr size_t nInt4PerChunk = 1024 * 256 / sizeof(int4);  // 256KB
  if (blockIdx.x >= nNeededBlocks) {
    nInt4OfThisBlock = 0;
  } else if (blockIdx.x == nNeededBlocks - 1) {
    nInt4OfThisBlock = nInt4PerRank - maxNInt4PerBlock * (nNeededBlocks - 1);
  }
  const size_t nItrs = nInt4OfThisBlock / nInt4PerChunk;
  const size_t restNInt4 = nInt4OfThisBlock % nInt4PerChunk;
  const size_t chunkSizePerRank = nNeededBlocks * nInt4PerChunk;
  const size_t blockOffset = nInt4PerChunk * blockIdx.x;
  const size_t scratchChunkRankOffset = chunkSizePerRank * rank;

  __shared__ DeviceHandle<MemoryChannel> channels[MAX_NRANKS_PER_NODE - 1];
  __shared__ DeviceHandle<MemoryChannel> outChannels[MAX_NRANKS_PER_NODE - 1];
  int const lid = threadIdx.x % WARP_SIZE;
  if (lid < nPeer) {
    channels[lid] = memoryChans[lid];
    outChannels[lid] = memoryOutChans[lid];
  }
  __syncwarp();

  // we can use double buffering to hide synchronization overhead
  for (size_t itr = 0; itr < nItrs; itr++) {
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      outChannels[threadIdx.x].signal();
      outChannels[threadIdx.x].wait();
    }
    __syncthreads();
    // Starts allgather
    for (size_t idx = threadIdx.x; idx < nInt4PerChunk; idx += blockDim.x) {
      for (int i = 0; i < nPeer; i++) {
        int const peerIdx = (i + blockIdx.x) % nPeer;
        int const remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
        int4 val = buff4[nInt4PerRank * remoteRank + idx + offsetOfThisBlock];
        channels[peerIdx].write(scratchChunkRankOffset + blockOffset + idx,
                                val);
      }
    }

    // Starts reduce-scatter
    // Ensure that all writes of this block have been issued before issuing the
    // signal
    __syncthreads();
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      outChannels[threadIdx.x].signal();
      outChannels[threadIdx.x].wait();
    }
    __syncthreads();

    for (size_t idx = threadIdx.x; idx < nInt4PerChunk; idx += blockDim.x) {
      int4 data = buff4[nInt4PerRank * rank + idx + offsetOfThisBlock];
      for (int peerIdx = 0; peerIdx < nPeer; peerIdx++) {
        int const remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
        int4 val = scratch4[chunkSizePerRank * remoteRank + blockOffset + idx];
        data = cal_vector<T, OpType>(val, data);
      }
      resultBuff4[nInt4PerRank * rank + idx + offsetOfThisBlock] = data;
      for (int peerIdx = 0; peerIdx < nPeer; peerIdx++) {
        outChannels[peerIdx].write(nInt4PerRank * rank + idx +
                                       offsetOfThisBlock +
                                       channelOutDataOffset / sizeof(int4),
                                   data);
      }
    }
    offsetOfThisBlock += nInt4PerChunk;
    // Ensure all threads have consumed data from scratch buffer before
    // signaling re-use in next iteration
    __syncthreads();
  }
  if (restNInt4 > 0) {
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      outChannels[threadIdx.x].signal();
      outChannels[threadIdx.x].wait();
    }
    __syncthreads();
    for (size_t idx = threadIdx.x; idx < restNInt4; idx += blockDim.x) {
      for (int i = 0; i < nPeer; i++) {
        int const peerIdx = (i + blockIdx.x) % nPeer;
        int const remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
        int4 val = buff4[nInt4PerRank * remoteRank + idx + offsetOfThisBlock];
        channels[peerIdx].write(scratchChunkRankOffset + blockOffset + idx,
                                val);
      }
    }

    // Ensure that all writes of this block have been issued before issuing the
    // signal
    __syncthreads();
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      outChannels[threadIdx.x].signal();
      outChannels[threadIdx.x].wait();
    }
    __syncthreads();

    for (size_t idx = threadIdx.x; idx < restNInt4; idx += blockDim.x) {
      int4 data = buff4[nInt4PerRank * rank + idx + offsetOfThisBlock];
      for (int peerIdx = 0; peerIdx < nPeer; peerIdx++) {
        int const remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
        int4 val = scratch4[chunkSizePerRank * remoteRank + blockOffset + idx];
        data = cal_vector<T, OpType>(val, data);
      }
      resultBuff4[nInt4PerRank * rank + idx + offsetOfThisBlock] = data;
      for (int peerIdx = 0; peerIdx < nPeer; peerIdx++) {
        outChannels[peerIdx].write(nInt4PerRank * rank + idx +
                                       offsetOfThisBlock +
                                       channelOutDataOffset / sizeof(int4),
                                   data);
      }
    }
    // Ensure all threads have issued writes to outChannel
    __syncthreads();
  }
  // Threads are already synchronized
  // So all writes to outChannel have been issued before signal is being issued
  if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
    outChannels[threadIdx.x].signal();
    outChannels[threadIdx.x].wait();
  }
}

template <ReduceOp OpType, typename T>
struct AllreduceAllconnectAdapter {
  static cudaError_t call(void const* input, void* scratch, void* output,
                          void* memoryChannels, void* memoryOutChannels,
                          DeviceHandle<SwitchChannel>*,
                          DeviceHandle<SwitchChannel>*, size_t,
                          size_t channelOutDataOffset, size_t, int rank,
                          int nRanksPerNode, int worldSize, size_t inputSize,
                          cudaStream_t stream, void*, uint32_t, uint32_t,
                          int nBlocks, int nThreadsPerBlock) {
    using ChannelType = DeviceHandle<MemoryChannel>;
    size_t nelems = inputSize / sizeof(T);
    if (nBlocks == 0) nBlocks = 35;
    if (nThreadsPerBlock == 0) nThreadsPerBlock = 512;
    allreduceFullmesh<OpType, T><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
        (T*)input, (T*)scratch, (T*)output, (ChannelType*)memoryChannels,
        (ChannelType*)memoryOutChannels, channelOutDataOffset, rank,
        nRanksPerNode, worldSize, nelems);
    return cudaGetLastError();
  }
};

void AllreduceFullmesh::initialize(std::shared_ptr<Communicator> comm) {
  this->conns_ = setupConnections(comm);
  nChannelsPerConnection_ = 64;
  comm_ = comm;
  // setup semaphores
  this->outputSemaphores_ =
      setupMemorySemaphores(comm, this->conns_, nChannelsPerConnection_);
  this->inputScratchSemaphores_ =
      setupMemorySemaphores(comm, this->conns_, nChannelsPerConnection_);
  RegisteredMemory localMemory = comm->registerMemory(
      scratchBuffer_, scratchBufferSize_, Transport::CudaIpc);
  this->remoteScratchMemories_ =
      setupRemoteMemories(comm, comm->bootstrap()->getRank(), localMemory);
  localScratchMemory_ = std::move(localMemory);
}

CommResult AllreduceFullmesh::allreduceKernelFunc(
    const std::shared_ptr<void> ctx_void, void const* input, void* output,
    size_t inputSize, DataType dtype, ReduceOp op, cudaStream_t stream,
    int nBlocks, int nThreadsPerBlock,
    std::unordered_map<std::string, uintptr_t> const&) {
  auto ctx = std::static_pointer_cast<AlgorithmCtx>(ctx_void);
  size_t recvBytes;
  CUdeviceptr recvBasePtr;
  size_t channelOutOffset = 0;
  if (symmetricMemory_) {
    MSCCLPP_CUTHROW(
        cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));
    channelOutOffset = (char*)output - (char*)recvBasePtr;
  }
  std::shared_ptr<DeviceHandle<MemoryChannel>> inputChannelHandles;
  if (this->memoryChannelsMap_.find(input) != this->memoryChannelsMap_.end()) {
    inputChannelHandles = this->memoryChannelsMap_[input].second;
  } else {
    RegisteredMemory localMemory = comm_->registerMemory(
        const_cast<void*>(input), inputSize, Transport::CudaIpc);
    std::vector<MemoryChannel> channels = setupMemoryChannels(
        this->conns_, this->inputScratchSemaphores_,
        this->remoteScratchMemories_, localMemory, nChannelsPerConnection_);
    this->memoryChannelsMap_[input] =
        std::make_pair(channels, setupMemoryChannelDeviceHandles(channels));
  }
  inputChannelHandles = this->memoryChannelsMap_[input].second;

  AllreduceFunc allreduce = dispatch<AllreduceAllconnectAdapter>(op, dtype);
  if (!allreduce) {
    WARN("Unsupported operation or data type for allreduce: op=%d, dtype=%d",
         static_cast<int>(op), static_cast<int>(dtype));
    return CommResult::CommInvalidArgument;
  }
  std::pair<int, int> numBlocksAndThreads = {nBlocks, nThreadsPerBlock};
  cudaError_t error =
      allreduce(input, this->scratchBuffer_, output, inputChannelHandles.get(),
                ctx->memoryChannelDeviceHandles.get(), nullptr, nullptr, 0,
                channelOutOffset, 0, ctx->rank, ctx->nRanksPerNode,
                ctx->workSize, inputSize, stream, nullptr, 0, 0,
                numBlocksAndThreads.first, numBlocksAndThreads.second);
  if (error != cudaSuccess) {
    WARN("AllreduceAllconnect failed with error: %s",
         cudaGetErrorString(error));
    return CommResult::CommUnhandledCudaError;
  }
  return CommResult::CommSuccess;
}

AlgorithmCtxKey AllreduceFullmesh::generateAllreduceContextKey(
    void const*, void* output, size_t, DataType, bool symmetricMemory) {
  static int tag = 0;
  size_t recvBytes;
  CUdeviceptr recvBasePtr;
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));
  symmetricMemory_ = symmetricMemory;
  if (!symmetricMemory_) {
    return AlgorithmCtxKey{nullptr, (void*)recvBasePtr, 0, recvBytes, tag++};
  }
  return AlgorithmCtxKey{nullptr, (void*)recvBasePtr, 0, recvBytes, 0};
}

std::shared_ptr<void> AllreduceFullmesh::initAllreduceContext(
    std::shared_ptr<Communicator> comm, void const*, void* output, size_t size,
    DataType) {
  auto ctx = std::make_shared<AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

  // setup semaphores
  ctx->memorySemaphores = this->outputSemaphores_;
  // setup memories and channels
  size_t recvBytes;
  CUdeviceptr recvBasePtr;
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));
  if (!symmetricMemory_) {
    recvBytes = size;
    recvBasePtr = (CUdeviceptr)output;
  }
  RegisteredMemory localMemory =
      comm->registerMemory((void*)recvBasePtr, recvBytes, Transport::CudaIpc);
  ctx->registeredMemories = setupRemoteMemories(comm, ctx->rank, localMemory);
  ctx->memoryChannels = setupMemoryChannels(
      this->conns_, ctx->memorySemaphores, ctx->registeredMemories, localMemory,
      nChannelsPerConnection_);
  ctx->memoryChannelDeviceHandles =
      setupMemoryChannelDeviceHandles(ctx->memoryChannels);
  return ctx;
}

std::shared_ptr<Algorithm> AllreduceFullmesh::build() {
  auto self = std::make_shared<AllreduceFullmesh>((uintptr_t)scratchBuffer_,
                                                  scratchBufferSize_);
  return std::make_shared<NativeAlgorithm>(
      "default_allreduce_fullmesh", "allreduce",
      [self](std::shared_ptr<mscclpp::Communicator> comm) {
        self->initialize(comm);
      },
      [self](const std::shared_ptr<void> ctx, void const* input, void* output,
             size_t inputSize, [[maybe_unused]] size_t outputSize,
             DataType dtype, ReduceOp op, cudaStream_t stream, int nBlocks,
             int nThreadsPerBlock,
             std::unordered_map<std::string, uintptr_t> const& extras)
          -> CommResult {
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