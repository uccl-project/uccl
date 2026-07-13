// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "allgather_fullmesh_2.hpp"
#include "collective_utils.hpp"
#include "debug.h"

namespace mscclpp {
namespace collective {

__device__ DeviceSyncer deviceSyncer;
template <bool IsOutOfPlace>
__global__ void __launch_bounds__(1024, 1) allgatherFullmesh2(
    void* sendbuff,
    mscclpp::DeviceHandle<mscclpp::MemoryChannel>* memoryChannels,
    size_t channelOutOffset, size_t rank, [[maybe_unused]] size_t worldSize,
    size_t nRanksPerNode, size_t nelemsPerGPU) {
  const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t lid = tid % WARP_SIZE;
  const size_t wid = tid / WARP_SIZE;

  const size_t nThread = blockDim.x * gridDim.x;
  const size_t nWarp = nThread / WARP_SIZE;
  const size_t nPeer = nRanksPerNode - 1;
  const size_t chanOffset = nPeer * blockIdx.x;
  auto memChans = memoryChannels + chanOffset;

  if (threadIdx.x < nPeer) {
    memChans[threadIdx.x].relaxedSignal();
    memChans[threadIdx.x].wait();
  }
  __syncthreads();

  const size_t bytesPerGPU = nelemsPerGPU * sizeof(int);
  const size_t bytes = bytesPerGPU * nPeer;
  size_t unitBytesPerThread;
  if (bytes >= nThread * 64) {
    unitBytesPerThread = 64;
  } else {
    unitBytesPerThread = 16;
  }
  const size_t unitBytesPerWarp = unitBytesPerThread * WARP_SIZE;
  const size_t unitBytes = unitBytesPerWarp * nWarp;
  const size_t nLoop = bytes / unitBytes;

  if (nLoop > 0) {
    // First loop unrolling
    const size_t peerIdx = wid % nPeer;
    const size_t offset = bytesPerGPU * rank + (wid / nPeer) * unitBytesPerWarp;
    if constexpr (IsOutOfPlace) {
      char* dst = reinterpret_cast<char*>(memChans[peerIdx].dst_);
      char* src = reinterpret_cast<char*>(memChans[peerIdx].src_);
      char* buff = reinterpret_cast<char*>(sendbuff);
      const size_t offsetWithinRank = (wid / nPeer) * unitBytesPerWarp;
      mscclpp::copy<16, false>(src + offset + channelOutOffset,
                               buff + offsetWithinRank, unitBytesPerWarp, lid,
                               WARP_SIZE);
      mscclpp::copy<16, false>(dst + offset + channelOutOffset,
                               buff + offsetWithinRank, unitBytesPerWarp, lid,
                               WARP_SIZE);
    } else {
      memChans[peerIdx].put<16, false>(offset + channelOutOffset,
                                       unitBytesPerWarp, lid, WARP_SIZE);
    }
  }

  for (size_t i = 1; i < nLoop; ++i) {
    const size_t gWid = wid + i * nWarp;
    const size_t peerIdx = gWid % nPeer;
    const size_t offset =
        bytesPerGPU * rank + (gWid / nPeer) * unitBytesPerWarp;
    if constexpr (IsOutOfPlace) {
      char* dst = reinterpret_cast<char*>(memChans[peerIdx].dst_);
      char* src = reinterpret_cast<char*>(memChans[peerIdx].src_);
      char* buff = reinterpret_cast<char*>(sendbuff);
      const size_t offsetWithinRank = (gWid / nPeer) * unitBytesPerWarp;
      mscclpp::copy<16, false>(src + offset + channelOutOffset,
                               buff + offsetWithinRank, unitBytesPerWarp, lid,
                               WARP_SIZE);
      mscclpp::copy<16, false>(dst + offset + channelOutOffset,
                               buff + offsetWithinRank, unitBytesPerWarp, lid,
                               WARP_SIZE);
    } else {
      memChans[peerIdx].put<16, false>(offset + channelOutOffset,
                                       unitBytesPerWarp, lid, WARP_SIZE);
    }
  }

  if (bytes % unitBytes > 0) {
    const size_t gWid = wid + nLoop * nWarp;
    const size_t peerIdx = gWid % nPeer;
    const size_t offsetWithinRank = (gWid / nPeer) * unitBytesPerWarp;
    const size_t offset = bytesPerGPU * rank + offsetWithinRank;
    const size_t remainBytes =
        (offsetWithinRank + unitBytesPerWarp > bytesPerGPU)
            ? ((bytesPerGPU > offsetWithinRank)
                   ? (bytesPerGPU - offsetWithinRank)
                   : 0)
            : unitBytesPerWarp;
    if (remainBytes > 0) {
      if constexpr (IsOutOfPlace) {
        char* dst = reinterpret_cast<char*>(memChans[peerIdx].dst_);
        char* src = reinterpret_cast<char*>(memChans[peerIdx].src_);
        char* buff = reinterpret_cast<char*>(sendbuff);
        mscclpp::copy<16, true>(src + offset + channelOutOffset,
                                buff + offsetWithinRank, remainBytes, lid,
                                WARP_SIZE);
        mscclpp::copy<16, true>(dst + offset + channelOutOffset,
                                buff + offsetWithinRank, remainBytes, lid,
                                WARP_SIZE);
      } else {
        memChans[peerIdx].put<16, true>(offset + channelOutOffset, remainBytes,
                                        lid, WARP_SIZE);
      }
    }
  }

  deviceSyncer.sync(gridDim.x);

  if (threadIdx.x < nPeer) {
    memChans[threadIdx.x].signal();
    memChans[threadIdx.x].wait();
  }
}

void AllgatherFullmesh2::initialize(std::shared_ptr<Communicator> comm) {
  this->conns_ = setupConnections(comm);
  this->memorySemaphores_ =
      setupMemorySemaphores(comm, this->conns_, nChannelsPerConnection_);
}

CommResult AllgatherFullmesh2::allgatherKernelFunc(
    const std::shared_ptr<void> ctx_void, void const* input, void* output,
    size_t inputSize, cudaStream_t stream, int nBlocks, int nThreadsPerBlock,
    std::unordered_map<std::string, uintptr_t> const&) {
  auto ctx = std::static_pointer_cast<AlgorithmCtx>(ctx_void);
  std::pair<int, int> numBlocksAndThreads = {nBlocks, nThreadsPerBlock};
  const size_t nElem = inputSize / sizeof(int);
  int rank = ctx->rank;
  if (numBlocksAndThreads.first == 0 || numBlocksAndThreads.second == 0) {
    numBlocksAndThreads.second = 1024;
    numBlocksAndThreads.first = 28;
    if (inputSize <= 32 * (1 << 20)) {
      if (nElem <= 4096) {
        numBlocksAndThreads.first = 7;
      } else if (nElem <= 32768) {
        numBlocksAndThreads.first = 14;
      } else if (nElem >= 2097152) {
        numBlocksAndThreads.first = 35;
      }
    } else {
      numBlocksAndThreads.first = 35;
    }
  }

  size_t channelOutOffset =
      *static_cast<size_t*>(ctx->extras["channel_out_offset"].get());
  if ((char*)input == (char*)output + rank * inputSize) {
    allgatherFullmesh2<false>
        <<<numBlocksAndThreads.first, numBlocksAndThreads.second, 0, stream>>>(
            (void*)input, ctx->memoryChannelDeviceHandles.get(),
            channelOutOffset, ctx->rank, ctx->workSize, ctx->nRanksPerNode,
            nElem);
  } else {
    allgatherFullmesh2<true>
        <<<numBlocksAndThreads.first, numBlocksAndThreads.second, 0, stream>>>(
            (void*)input, ctx->memoryChannelDeviceHandles.get(),
            channelOutOffset, ctx->rank, ctx->workSize, ctx->nRanksPerNode,
            nElem);
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    WARN("AllgatherFullmesh2 failed with error %d", err);
    return CommResult::CommUnhandledCudaError;
  }
  return CommResult::CommSuccess;
}

std::shared_ptr<void> AllgatherFullmesh2::initAllgatherContext(
    std::shared_ptr<mscclpp::Communicator> comm, void const*, void* output,
    size_t inputSize, mscclpp::DataType) {
  auto ctx = std::make_shared<AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

  // setup semaphores
  ctx->memorySemaphores = this->memorySemaphores_;

  size_t recvBytes;
  CUdeviceptr recvBasePtr;
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));
  size_t channelOutOffset = (char*)output - (char*)recvBasePtr;
  if (!symmetricMemory_) {
    channelOutOffset = 0;
    recvBytes = inputSize * comm->bootstrap()->getNranks();
    recvBasePtr = (CUdeviceptr)output;
  }
  ctx->extras.insert(
      {"channel_out_offset", std::make_shared<size_t>(channelOutOffset)});

  // register the memory for the broadcast operation
  mscclpp::RegisteredMemory localMemory = comm->registerMemory(
      (void*)recvBasePtr, recvBytes, mscclpp::Transport::CudaIpc);
  std::vector<mscclpp::RegisteredMemory> remoteMemories =
      setupRemoteMemories(comm, ctx->rank, localMemory);
  ctx->memoryChannels =
      setupMemoryChannels(this->conns_, ctx->memorySemaphores, remoteMemories,
                          localMemory, nChannelsPerConnection_);
  ctx->memoryChannelDeviceHandles =
      setupMemoryChannelDeviceHandles(ctx->memoryChannels);

  // keep registered memories reference
  ctx->registeredMemories = std::move(remoteMemories);
  ctx->registeredMemories.push_back(localMemory);

  return ctx;
}

mscclpp::AlgorithmCtxKey AllgatherFullmesh2::generateAllgatherContextKey(
    void const*, void* output, size_t, mscclpp::DataType,
    bool symmetricMemory) {
  static int tag = 0;
  symmetricMemory_ = symmetricMemory;
  if (!symmetricMemory_) {
    // always return a new key if symmetric memory is not enabled.
    return mscclpp::AlgorithmCtxKey{nullptr, nullptr, 0, 0, tag++};
  }
  size_t recvBytes;
  CUdeviceptr recvBasePtr;
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));
  return mscclpp::AlgorithmCtxKey{nullptr, (void*)recvBasePtr, 0, recvBytes, 0};
}

std::shared_ptr<Algorithm> AllgatherFullmesh2::build() {
  auto self = std::make_shared<AllgatherFullmesh2>();
  return std::make_shared<NativeAlgorithm>(
      "default_allgather_fullmesh2", "allgather",
      [self](std::shared_ptr<Communicator> comm) { self->initialize(comm); },
      [self](const std::shared_ptr<void> ctx, void const* input, void* output,
             size_t inputSize, [[maybe_unused]] size_t outputSize,
             [[maybe_unused]] mscclpp::DataType dtype,
             [[maybe_unused]] ReduceOp op, cudaStream_t stream, int nBlocks,
             int nThreadsPerBlock,
             std::unordered_map<std::string, uintptr_t> const& extras)
          -> mscclpp::CommResult {
        return self->allgatherKernelFunc(ctx, input, output, inputSize, stream,
                                         nBlocks, nThreadsPerBlock, extras);
      },
      [self](std::shared_ptr<mscclpp::Communicator> comm, void const* input,
             void* output, size_t inputSize, [[maybe_unused]] size_t outputSize,
             mscclpp::DataType dtype) {
        return self->initAllgatherContext(comm, input, output, inputSize,
                                          dtype);
      },
      [self](void const* input, void* output, size_t inputSize,
             [[maybe_unused]] size_t outputSize, mscclpp::DataType dtype,
             bool symmetricMemory) {
        return self->generateAllgatherContextKey(input, output, inputSize,
                                                 dtype, symmetricMemory);
      });
}

}  // namespace collective
}  // namespace mscclpp