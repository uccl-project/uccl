// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "allgather_fullmesh.hpp"
#include "collective_utils.hpp"
#include "debug.h"

namespace mscclpp {
namespace collective {

template <bool IsOutOfPlace>
__global__ void __launch_bounds__(1024, 1)
    allgatherFullmesh(void* buff, void* scratch, void* resultBuff,
                      DeviceHandle<MemoryChannel>* memoryChannels, int rank,
                      int nRanksPerNode, [[maybe_unused]] int worldSize,
                      size_t nelems) {
  int const nPeer = nRanksPerNode - 1;
  const size_t chanOffset = nPeer * blockIdx.x;
  // assume (nelems * sizeof(T)) is divisible by 16
  const size_t nInt4 = nelems * sizeof(int) / sizeof(int4);
  auto memoryChans = memoryChannels + chanOffset;

  int4* buff4 = reinterpret_cast<int4*>(buff);
  int4* scratch4 = reinterpret_cast<int4*>(scratch);
  int4* resultBuff4 = reinterpret_cast<int4*>(resultBuff);

  const size_t unitNInt4 =
      blockDim.x * gridDim.x;  // The number of int4 transfers at once
  const size_t nInt4PerChunk =
      unitNInt4 * 4;  // 4 instructions per thread to make it more efficient
  const size_t nItrs = nInt4 / nInt4PerChunk;
  const size_t restNInt4 = nInt4 % nInt4PerChunk;
  const size_t scratchChunkRankOffset = nInt4PerChunk * rank;

  __shared__ DeviceHandle<MemoryChannel> channels[MAX_NRANKS_PER_NODE - 1];
  int const lid = threadIdx.x % WARP_SIZE;
  if (lid < nPeer) {
    channels[lid] = memoryChans[lid];
  }
  __syncwarp();
  int const tid = threadIdx.x + blockIdx.x * blockDim.x;
  // we can use double buffering to hide synchronization overhead
  for (size_t itr = 0; itr < nItrs; itr++) {
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      channels[threadIdx.x].signal();
      channels[threadIdx.x].wait();
    }
    __syncthreads();
    // Starts allgather
    for (size_t idx = tid; idx < nInt4PerChunk; idx += blockDim.x * gridDim.x) {
      int4 val = buff4[itr * nInt4PerChunk + idx];
      for (int i = 0; i < nPeer; i++) {
        int const peerIdx = (i + rank) % nPeer;
        channels[peerIdx].write(idx + scratchChunkRankOffset, val);
      }
      if constexpr (IsOutOfPlace) {
        resultBuff4[nInt4 * rank + idx + itr * nInt4PerChunk] = val;
      }
    }
    // Ensure that all writes of this block have been issued before issuing the
    // signal
    __syncthreads();
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      channels[threadIdx.x].signal();
      channels[threadIdx.x].wait();
    }
    __syncthreads();
    for (int peerIdx = 0; peerIdx < nPeer; peerIdx++) {
      int const remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
      int const resultOffset = nInt4 * remoteRank + itr * nInt4PerChunk;
      for (size_t idx = tid; idx < nInt4PerChunk;
           idx += blockDim.x * gridDim.x) {
        int4 val = scratch4[nInt4PerChunk * remoteRank + idx];
        resultBuff4[resultOffset + idx] = val;
      }
    }
  }

  if (restNInt4 > 0) {
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      channels[threadIdx.x].signal();
      channels[threadIdx.x].wait();
    }
    __syncthreads();
    for (size_t idx = tid; idx < restNInt4; idx += blockDim.x * gridDim.x) {
      int4 val = buff4[nItrs * nInt4PerChunk + idx];
      for (int i = 0; i < nPeer; i++) {
        int const peerIdx = (i + rank) % nPeer;
        channels[peerIdx].write(idx + scratchChunkRankOffset, val);
      }
      if constexpr (IsOutOfPlace) {
        resultBuff4[nInt4 * rank + idx + nItrs * nInt4PerChunk] = val;
      }
    }
    // Ensure that all writes of this block have been issued before issuing the
    // signal
    __syncthreads();
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      channels[threadIdx.x].signal();
      channels[threadIdx.x].wait();
    }
    __syncthreads();
    for (int peerIdx = 0; peerIdx < nPeer; peerIdx++) {
      int const remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
      int const resultOffset = nInt4 * remoteRank + nItrs * nInt4PerChunk;
      for (size_t idx = tid; idx < restNInt4; idx += blockDim.x * gridDim.x) {
        int4 val = scratch4[nInt4PerChunk * remoteRank + idx];
        resultBuff4[resultOffset + idx] = val;
      }
    }
  }
}

void AllgatherFullmesh::initialize(
    std::shared_ptr<mscclpp::Communicator> comm) {
  this->conns_ = setupConnections(comm);
}

CommResult AllgatherFullmesh::allgatherKernelFunc(
    const std::shared_ptr<void> ctx_void, void const* input, void* output,
    size_t inputSize, cudaStream_t stream, int nBlocks, int nThreadsPerBlock,
    std::unordered_map<std::string, uintptr_t> const&) {
  auto ctx = std::static_pointer_cast<AlgorithmCtx>(ctx_void);
  int rank = ctx->rank;
  const size_t nElem = inputSize / sizeof(int);
  std::pair<int, int> numBlocksAndThreads = {nBlocks, nThreadsPerBlock};
  if (numBlocksAndThreads.first > 56) {
    WARN(
        "AllgatherFullmesh: number of blocks exceeds maximum supported blocks, "
        "which is 56");
    return mscclpp::CommResult::CommInvalidArgument;
  }
  if (numBlocksAndThreads.first == 0 || numBlocksAndThreads.second == 0) {
    numBlocksAndThreads = {56, 1024};
  }
  if ((char*)input == (char*)output + rank * inputSize) {
    allgatherFullmesh<false>
        <<<numBlocksAndThreads.first, numBlocksAndThreads.second, 0, stream>>>(
            (void*)input, this->scratchBuffer_, (void*)output,
            ctx->memoryChannelDeviceHandles.get(), rank, ctx->nRanksPerNode,
            ctx->workSize, nElem);
  } else {
    allgatherFullmesh<true>
        <<<numBlocksAndThreads.first, numBlocksAndThreads.second, 0, stream>>>(
            (void*)input, this->scratchBuffer_, (void*)output,
            ctx->memoryChannelDeviceHandles.get(), rank, ctx->nRanksPerNode,
            ctx->workSize, nElem);
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    WARN("AllgatherFullmesh failed with error %d", err);
    return mscclpp::CommResult::CommInternalError;
  }
  return mscclpp::CommResult::CommSuccess;
}

std::shared_ptr<void> AllgatherFullmesh::initAllgatherContext(
    std::shared_ptr<Communicator> comm, void const* input, void*,
    size_t inputSize, DataType) {
  constexpr int nChannelsPerConnection = 56;

  auto ctx = std::make_shared<AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

  // setup semaphores
  ctx->memorySemaphores =
      setupMemorySemaphores(comm, this->conns_, nChannelsPerConnection);

  // register the memory for the broadcast operation
  RegisteredMemory localMemory =
      comm->registerMemory((void*)input, inputSize, Transport::CudaIpc);
  RegisteredMemory scratchMemory = comm->registerMemory(
      this->scratchBuffer_, scratchBufferSize_, Transport::CudaIpc);
  std::vector<RegisteredMemory> remoteMemories =
      setupRemoteMemories(comm, ctx->rank, scratchMemory);

  // setup channels
  ctx->memoryChannels =
      setupMemoryChannels(this->conns_, ctx->memorySemaphores, remoteMemories,
                          localMemory, nChannelsPerConnection);
  ctx->memoryChannelDeviceHandles =
      setupMemoryChannelDeviceHandles(ctx->memoryChannels);

  // keep registered memories reference
  ctx->registeredMemories = std::move(remoteMemories);
  ctx->registeredMemories.push_back(localMemory);
  ctx->registeredMemories.push_back(scratchMemory);

  return ctx;
}

AlgorithmCtxKey AllgatherFullmesh::generateAllgatherContextKey(void const*,
                                                               void*, size_t,
                                                               DataType, bool) {
  // always return same key, non-zero copy algo
  return AlgorithmCtxKey{nullptr, nullptr, 0, 0, 0};
}

std::shared_ptr<Algorithm> AllgatherFullmesh::build() {
  auto self = std::make_shared<AllgatherFullmesh>(
      reinterpret_cast<uintptr_t>(scratchBuffer_), scratchBufferSize_);
  return std::make_shared<mscclpp::NativeAlgorithm>(
      "default_allgather_fullmesh", "allgather",
      [self](std::shared_ptr<mscclpp::Communicator> comm) {
        self->initialize(comm);
      },
      [self](const std::shared_ptr<void> ctx, void const* input, void* output,
             size_t inputSize, [[maybe_unused]] size_t outputSize,
             [[maybe_unused]] DataType dtype, [[maybe_unused]] ReduceOp op,
             cudaStream_t stream, int nBlocks, int nThreadsPerBlock,
             std::unordered_map<std::string, uintptr_t> const& extras)
          -> CommResult {
        return self->allgatherKernelFunc(ctx, input, output, inputSize, stream,
                                         nBlocks, nThreadsPerBlock, extras);
      },
      [self](std::shared_ptr<mscclpp::Communicator> comm, void const* input,
             void* output, size_t inputSize, [[maybe_unused]] size_t outputSize,
             DataType dtype) {
        return self->initAllgatherContext(comm, input, output, inputSize,
                                          dtype);
      },
      [self](void const* input, void* output, size_t inputSize,
             [[maybe_unused]] size_t outputSize, DataType dtype,
             bool symmetricMemory) {
        return self->generateAllgatherContextKey(input, output, inputSize,
                                                 dtype, symmetricMemory);
      });
}
}  // namespace collective
}  // namespace mscclpp