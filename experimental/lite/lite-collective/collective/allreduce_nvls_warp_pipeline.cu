// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "algorithm.hpp"
#include "allreduce_nvls_warp_pipeline.hpp"
#include "collective_utils.hpp"
#include "common.hpp"
#include "debug.h"

namespace mscclpp {
namespace collective {

template <typename T>
__global__ void __launch_bounds__(1024, 1) allreduceNvlsWarpPipeline(
    [[maybe_unused]] void const* src, [[maybe_unused]] void* scratch,
    [[maybe_unused]] void* dst,
    [[maybe_unused]] DeviceHandle<BaseMemoryChannel>* memoryChannels,
    [[maybe_unused]] DeviceHandle<SwitchChannel>* multicast,
    [[maybe_unused]] size_t size, [[maybe_unused]] size_t scratchBufferSize,
    [[maybe_unused]] int rank, [[maybe_unused]] int nRanksPerNode) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr int alignment = 16;
  int nPeers = nRanksPerNode - 1;
  int nBlocks = gridDim.x;
  int nBlocksPerNvlsConn = nBlocks / NUM_NVLS_CONNECTION;
  int bid = blockIdx.x;
  size_t sizePerRank = size / nRanksPerNode;
  size_t scratchSizePerRank = scratchBufferSize / nRanksPerNode;
  const size_t maxSizePerBlock =
      ((sizePerRank + nBlocks - 1) / nBlocks + alignment - 1) / alignment *
      alignment;
  size_t start = bid * maxSizePerBlock;
  size_t end = min(start + maxSizePerBlock, sizePerRank);
  size_t sizePerBlock = end - start;
  auto* multicastPtr = multicast + bid / nBlocksPerNvlsConn;
  size_t copyPerIter = 1024 * 16;
  if (sizePerBlock >= 1024 * 64) {
    copyPerIter = 1024 * 32;
  }
  size_t scratchSizePerBlock =
      (scratchSizePerRank / nBlocks) / copyPerIter * copyPerIter;
  size_t blockScratchOffset =
      scratchSizePerBlock * bid + scratchSizePerRank * rank;
  constexpr int NCOPY_WARPS = 14;
  constexpr int NREDUCE_WARPS = 4;
  constexpr int NRECV_COPY_WARPS = 14;
  constexpr int endCopyWid = NCOPY_WARPS;
  constexpr int startRecvCopyWid = NCOPY_WARPS;
  constexpr int endRecvCopyWid = NCOPY_WARPS + NRECV_COPY_WARPS;
  constexpr int endReduceWid = NCOPY_WARPS + NREDUCE_WARPS + NRECV_COPY_WARPS;
  int const warpId = threadIdx.x / WARP_SIZE;
  size_t nIter = sizePerBlock / copyPerIter;
  size_t lastIterSize = copyPerIter;
  if (sizePerBlock % copyPerIter != 0) {
    nIter += 1;
    lastIterSize = sizePerBlock % copyPerIter;
  }

  const size_t chanOffset = (nRanksPerNode - 1) * blockIdx.x * 2;
  auto memoryChans = memoryChannels + chanOffset;
  __shared__ DeviceHandle<BaseMemoryChannel>
      channels[(MAX_NRANKS_PER_NODE - 1) * 2];
  int const lid = threadIdx.x % WARP_SIZE;
  if (lid < nPeers * 2) {
    channels[lid] = memoryChans[lid];
  }
  __syncwarp();
  for (int it = 0; it < nIter; it++) {
    const size_t iterSize = (it == nIter - 1) ? lastIterSize : copyPerIter;
    if (warpId < endCopyWid) {
      int tidInCopy = threadIdx.x;
      for (int i = 0; i < nRanksPerNode; i++) {
        size_t offset =
            i * sizePerRank + maxSizePerBlock * bid + it * copyPerIter;
        size_t offsetScratch = i * scratchSizePerRank +
                               scratchSizePerBlock * bid +
                               (it * copyPerIter) % scratchSizePerBlock;
        char* srcData = (char*)src + offset;
        char* dstData = (char*)scratch + offsetScratch;
        mscclpp::copy(dstData, srcData, iterSize, tidInCopy,
                      NCOPY_WARPS * WARP_SIZE);
      }
      asm volatile("bar.sync %0, %1;" ::"r"(0), "r"(NCOPY_WARPS * WARP_SIZE)
                   : "memory");
      if (tidInCopy < nPeers) {
        channels[tidInCopy].signal();
        channels[tidInCopy].wait();
      }
      asm volatile("bar.sync %0, %1;" ::"r"(1),
                   "r"((NCOPY_WARPS + NREDUCE_WARPS) * WARP_SIZE)
                   : "memory");
    }
    if (warpId >= endRecvCopyWid && warpId < endReduceWid) {
      int tidInReduce = threadIdx.x - endRecvCopyWid * WARP_SIZE;
      asm volatile("bar.sync %0, %1;" ::"r"(1),
                   "r"((NCOPY_WARPS + NREDUCE_WARPS) * WARP_SIZE)
                   : "memory");
      T* mcBuff = (T*)multicastPtr->mcPtr;
      size_t offset =
          blockScratchOffset + (it * copyPerIter) % scratchSizePerBlock;
      handleMultiLoadReduceStore(mcBuff, mcBuff, offset, offset, iterSize,
                                 tidInReduce, NREDUCE_WARPS * WARP_SIZE);
      asm volatile("bar.sync %0, %1;" ::"r"(2),
                   "r"((NRECV_COPY_WARPS + NREDUCE_WARPS) * WARP_SIZE)
                   : "memory");
    }
    if (warpId >= startRecvCopyWid && warpId < endRecvCopyWid) {
      int tidInRecvCopy = threadIdx.x - startRecvCopyWid * WARP_SIZE;
      asm volatile("bar.sync %0, %1;" ::"r"(2),
                   "r"((NRECV_COPY_WARPS + NREDUCE_WARPS) * WARP_SIZE)
                   : "memory");
      if (tidInRecvCopy < nPeers) {
        channels[tidInRecvCopy + nPeers].signal();
        channels[tidInRecvCopy + nPeers].wait();
      }
      asm volatile("bar.sync %0, %1;" ::"r"(3),
                   "r"((NRECV_COPY_WARPS)*WARP_SIZE)
                   : "memory");
      for (int i = 0; i < nRanksPerNode; i++) {
        size_t offset =
            i * sizePerRank + maxSizePerBlock * bid + it * copyPerIter;
        size_t offsetScratch = i * scratchSizePerRank +
                               scratchSizePerBlock * bid +
                               (it * copyPerIter) % scratchSizePerBlock;
        char* srcData = (char*)scratch + offsetScratch;
        char* dstData = (char*)dst + offset;
        mscclpp::copy(dstData, srcData, iterSize, tidInRecvCopy,
                      NRECV_COPY_WARPS * WARP_SIZE);
      }
    }
  }
#endif
}

template <ReduceOp OpType, typename T>
struct NvlsWarpPipelineAdapter {
  static cudaError_t call(void const* input, void* scratch, void* output,
                          void* memoryChannels, void*,
                          DeviceHandle<SwitchChannel>* nvlsChannels,
                          DeviceHandle<SwitchChannel>*, size_t, size_t,
                          size_t scratchBufferSize, int rank, int nRanksPerNode,
                          int, size_t inputSize, cudaStream_t stream, void*,
                          uint32_t, uint32_t, int nBlocks,
                          int nThreadsPerBlock) {
    // uint8_t is not supported for NVLS (no hardware support for byte-level
    // reduction)
    if constexpr (std::is_same_v<T, uint8_t>) {
      return cudaErrorNotSupported;
    } else
#if defined(__CUDA_ARCH__)  // Skip the __CUDA_ARCH__ < 1000 since FP8 has not
                            // been supported for NVLS
      if constexpr (std::is_same_v<T, __fp8_e4m3> ||
                    std::is_same_v<T, __fp8_e5m2>) {
        return cudaErrorNotSupported;
      } else
#endif
      {
        using ChannelType = DeviceHandle<BaseMemoryChannel>;
        allreduceNvlsWarpPipeline<T><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
            input, scratch, output, (ChannelType*)memoryChannels, nvlsChannels,
            inputSize, scratchBufferSize, rank, nRanksPerNode);
        return cudaGetLastError();
      }
  }
};

void AllreduceNvlsWarpPipeline::initialize(std::shared_ptr<Communicator> comm) {
  nSwitchChannels_ = 8;
  int nBaseChannels = 64;
  this->conns_ = setupConnections(comm);
  // setup semaphores
  std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>> memorySemaphores =
      setupMemorySemaphores(comm, this->conns_, nBaseChannels);
  // setup base memory channels
  this->baseChannels_ =
      setupBaseMemoryChannels(this->conns_, memorySemaphores, nBaseChannels);
  this->memoryChannelsDeviceHandle_ =
      setupBaseMemoryChannelDeviceHandles(this->baseChannels_);
  this->nvlsConnections_ =
      setupNvlsConnections(comm, nvlsBufferSize_, nSwitchChannels_);
}

CommResult AllreduceNvlsWarpPipeline::allreduceKernelFunc(
    const std::shared_ptr<void> ctx_void, void const* input, void* output,
    size_t inputSize, DataType dtype, ReduceOp op, cudaStream_t stream,
    int nBlocks, int nThreadsPerBlock,
    std::unordered_map<std::string, uintptr_t> const&) {
  auto ctx = std::static_pointer_cast<AlgorithmCtx>(ctx_void);
  AllreduceFunc allreduce = dispatch<NvlsWarpPipelineAdapter>(op, dtype);
  if (!allreduce) {
    WARN("Unsupported operation or data type for allreduce, dtype=%d",
         static_cast<int>(dtype));
    return CommResult::CommInvalidArgument;
  }
  std::pair<int, int> blockAndThreadNum = {nBlocks, nThreadsPerBlock};
  if (blockAndThreadNum.first == 0 || blockAndThreadNum.second == 0) {
    blockAndThreadNum = {ctx->nRanksPerNode * 4, 1024};
  }
  cudaError_t error =
      allreduce(input, this->scratchBuffer_, output,
                this->memoryChannelsDeviceHandle_.get(), nullptr,
                ctx->switchChannelDeviceHandles.get(), nullptr, 0, 0,
                this->scratchBufferSize_, ctx->rank, ctx->nRanksPerNode,
                ctx->workSize, inputSize, stream, nullptr, 0, 0,
                blockAndThreadNum.first, blockAndThreadNum.second);
  if (error != cudaSuccess) {
    WARN("AllreduceNvlsWarpPipeline failed with error: %s",
         cudaGetErrorString(error));
    return CommResult::CommUnhandledCudaError;
  }
  return CommResult::CommSuccess;
}

AlgorithmCtxKey AllreduceNvlsWarpPipeline::generateAllreduceContextKey(
    void const*, void*, size_t, DataType, bool) {
  return AlgorithmCtxKey{nullptr, nullptr, 0, 0, 0};
}

std::shared_ptr<void> AllreduceNvlsWarpPipeline::initAllreduceContext(
    std::shared_ptr<Communicator> comm, void const*, void*, size_t, DataType) {
  auto ctx = std::make_shared<AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

  // setup channels
  ctx->switchChannels =
      setupNvlsChannels(this->nvlsConnections_, this->scratchBuffer_,
                        scratchBufferSize_, nSwitchChannels_);
  ctx->switchChannelDeviceHandles =
      setupNvlsChannelDeviceHandles(ctx->switchChannels);
  return ctx;
}

std::shared_ptr<Algorithm> AllreduceNvlsWarpPipeline::build() {
  auto self = std::make_shared<AllreduceNvlsWarpPipeline>(
      reinterpret_cast<uintptr_t>(scratchBuffer_), scratchBufferSize_);
  return std::make_shared<NativeAlgorithm>(
      "default_allreduce_nvls_warp_pipeline", "allreduce",
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
