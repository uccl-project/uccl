// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "algorithm.hpp"
#include "allreduce_nvls_block_pipeline.hpp"
#include "collective_utils.hpp"
#include "common.hpp"
#include "debug.h"

namespace mscclpp {
namespace collective {

__device__ DeviceSemaphore deviceSemaphore[NUM_SEMAPHORES];

template <typename T>
__global__ void __launch_bounds__(1024, 1) allreduceNvlsBlockPipeline(
    [[maybe_unused]] void const* src, [[maybe_unused]] void* scratch,
    [[maybe_unused]] void* dst,
    [[maybe_unused]] DeviceHandle<BaseMemoryChannel>* memoryChannels,
    [[maybe_unused]] DeviceHandle<SwitchChannel>* switchChannels,
    [[maybe_unused]] size_t size, [[maybe_unused]] size_t scratchBufferSize,
    [[maybe_unused]] int rank, [[maybe_unused]] int nRanksPerNode) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr int alignment = 16;
  int nPeers = nRanksPerNode - 1;
  int nBlocksForCopy = nRanksPerNode * 2;
  int nBlocksForReduce = nRanksPerNode;
  int copyReduceRatio = nBlocksForCopy / nBlocksForReduce;
  size_t scratchSizePerRank = scratchBufferSize / nRanksPerNode;
  size_t sizePerRank = size / nRanksPerNode;
  assert(sizePerRank % alignment == 0);
  uint32_t sizePerBlock =
      ((sizePerRank + (nBlocksForCopy - 1)) / nBlocksForCopy + alignment - 1) /
      alignment * alignment;
  uint32_t lastBlockSize = sizePerRank - (nBlocksForCopy - 1) * sizePerBlock;
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  uint32_t unitSize = 1 << 17;
  if (size <= 1024 * 1024 * 128) {
    unitSize = 1 << 16;
  }
  int nIter = sizePerBlock / unitSize;
  int nIterLastBlock = lastBlockSize / unitSize;
  uint32_t lastIterSize = unitSize;
  uint32_t lastBlockIterSize = unitSize;
  if (sizePerBlock % unitSize != 0) {
    nIter += 1;
    lastIterSize = sizePerBlock % unitSize;
  }
  if (lastBlockSize % unitSize != 0) {
    nIterLastBlock += 1;
    lastBlockIterSize = lastBlockSize % unitSize;
  }
  if (bid == nBlocksForCopy - 1 ||
      bid == 2 * nBlocksForCopy + nBlocksForReduce - 1) {
    lastIterSize = lastBlockIterSize;
    nIter = nIterLastBlock;
  }
  size_t scratchSizePerBlock =
      (scratchSizePerRank / nBlocksForCopy) / unitSize * unitSize;
  size_t maxItersForScratch = scratchSizePerBlock / unitSize;
  if (bid < nBlocksForCopy && tid == 0) {
    deviceSemaphore[bid + 2 * nBlocksForCopy].set(maxItersForScratch);
  }
  for (int it = 0; it < nIter; it++) {
    const uint32_t iterSize = (it == nIter - 1) ? lastIterSize : unitSize;
    const uint32_t scratchIt = it % maxItersForScratch;
    if (bid < nBlocksForCopy) {
      if (tid == 0) {
        deviceSemaphore[bid + 2 * nBlocksForCopy].acquire();
      }
      __syncthreads();
      for (int i = 0; i < nRanksPerNode; i++) {
        size_t blockOffset =
            it * unitSize + bid * sizePerBlock + i * sizePerRank;
        uint32_t scratchOffset = scratchIt * unitSize +
                                 bid * scratchSizePerBlock +
                                 i * scratchSizePerRank;
        char* srcData = (char*)src + blockOffset;
        char* dstData = (char*)scratch + scratchOffset;
        mscclpp::copy(dstData, srcData, iterSize, tid, blockDim.x);
      }
      __syncthreads();
      if (tid < nPeers) {
        int chanId = bid * nPeers + tid;
        mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>* channels =
            memoryChannels + chanId;
        channels->signal();
        channels->wait();
      }
      __syncthreads();
      if (tid == 0) {
        deviceSemaphore[bid].release();
      }
    }
    if (bid >= nBlocksForCopy && bid < nBlocksForCopy + nBlocksForReduce) {
      int bidForReduce = bid - nBlocksForCopy;
      auto switchChannel = switchChannels + bidForReduce;
      T* mcBuff = (T*)switchChannel->mcPtr;
      for (int i = 0; i < copyReduceRatio; i++) {
        int oriBid = bidForReduce * copyReduceRatio + i;
        uint32_t offset = rank * scratchSizePerRank + scratchIt * unitSize +
                          oriBid * scratchSizePerBlock;
        uint32_t reduceIterSize = iterSize;
        if ((oriBid == nBlocksForCopy - 1) && (it >= nIterLastBlock - 1)) {
          if (it > nIterLastBlock - 1) {
            continue;
          }
          reduceIterSize = lastBlockIterSize;
        }
        if (tid == 0) {
          deviceSemaphore[oriBid].acquire();
        }
        __syncthreads();
        handleMultiLoadReduceStore(mcBuff, mcBuff, offset, offset,
                                   reduceIterSize, tid, blockDim.x);
        __syncthreads();
        if (tid == 0) {
          deviceSemaphore[nBlocksForCopy + bidForReduce * copyReduceRatio + i]
              .release();
        }
      }
    }
    if (bid >= nBlocksForCopy + nBlocksForReduce &&
        bid < nBlocksForCopy + nBlocksForReduce + nBlocksForCopy) {
      int bidForCopy = bid - nBlocksForCopy - nBlocksForReduce;
      if (tid == 0) {
        deviceSemaphore[bid - nBlocksForReduce].acquire();
      }
      __syncthreads();
      if (tid < nPeers) {
        int chanId = (bid - nBlocksForReduce) * nPeers + tid;
        DeviceHandle<BaseMemoryChannel>* channels = memoryChannels + chanId;
        channels->signal();
        channels->wait();
      }
      __syncthreads();
      for (int i = 0; i < nRanksPerNode; i++) {
        size_t blockOffset =
            it * unitSize +
            (bid - nBlocksForCopy - nBlocksForReduce) * sizePerBlock +
            i * sizePerRank;
        uint32_t scratchOffset =
            scratchIt * unitSize +
            (bid - nBlocksForCopy - nBlocksForReduce) * scratchSizePerBlock +
            i * scratchSizePerRank;
        char* srcData = (char*)scratch + scratchOffset;
        char* dstData = (char*)dst + blockOffset;
        mscclpp::copy(dstData, srcData, iterSize, tid, blockDim.x);
      }
      __syncthreads();
      if (tid == 0) {
        deviceSemaphore[bidForCopy + 2 * nBlocksForCopy].release();
      }
    }
  }
  if (bid < nBlocksForCopy && tid == 0) {
    deviceSemaphore[bid + 2 * nBlocksForCopy].set(0);
  }
#endif
}

template <ReduceOp OpType, typename T>
struct NvlsBlockPipelineAdapter {
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
        allreduceNvlsBlockPipeline<T><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
            input, scratch, output, (ChannelType*)memoryChannels, nvlsChannels,
            inputSize, scratchBufferSize, rank, nRanksPerNode);
        return cudaGetLastError();
      }
  }
};

void AllreduceNvlsBlockPipeline::initialize(
    std::shared_ptr<Communicator> comm) {
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

CommResult AllreduceNvlsBlockPipeline::allreduceKernelFunc(
    const std::shared_ptr<void> ctx_void, void const* input, void* output,
    size_t inputSize, DataType dtype, ReduceOp op, cudaStream_t stream,
    int nBlocks, int nThreadsPerBlock,
    std::unordered_map<std::string, uintptr_t> const&) {
  auto ctx = std::static_pointer_cast<AlgorithmCtx>(ctx_void);
  AllreduceFunc allreduce = dispatch<NvlsBlockPipelineAdapter>(op, dtype);
  if (!allreduce) {
    WARN("Unsupported operation or data type for allreduce, dtype=%d",
         static_cast<int>(dtype));
    return CommResult::CommInvalidArgument;
  }
  std::pair<int, int> blockAndThreadNum = {nBlocks, nThreadsPerBlock};
  if (blockAndThreadNum.first == 0 || blockAndThreadNum.second == 0) {
    blockAndThreadNum = {ctx->nRanksPerNode * 5, 1024};
  }
  cudaError_t error =
      allreduce(input, this->scratchBuffer_, output,
                this->memoryChannelsDeviceHandle_.get(), nullptr,
                ctx->switchChannelDeviceHandles.get(), nullptr, 0, 0,
                this->scratchBufferSize_, ctx->rank, ctx->nRanksPerNode,
                ctx->workSize, inputSize, stream, nullptr, 0, 0,
                blockAndThreadNum.first, blockAndThreadNum.second);
  if (error != cudaSuccess) {
    WARN("AllreduceNvlsBlockPipeline failed with error: %s",
         cudaGetErrorString(error));
    return CommResult::CommUnhandledCudaError;
  }
  return CommResult::CommSuccess;
}

AlgorithmCtxKey AllreduceNvlsBlockPipeline::generateAllreduceContextKey(
    void const*, void*, size_t, DataType, bool) {
  return AlgorithmCtxKey{nullptr, nullptr, 0, 0, 0};
}

std::shared_ptr<void> AllreduceNvlsBlockPipeline::initAllreduceContext(
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

std::shared_ptr<Algorithm> AllreduceNvlsBlockPipeline::build() {
  auto self = std::make_shared<AllreduceNvlsBlockPipeline>(
      reinterpret_cast<uintptr_t>(scratchBuffer_), scratchBufferSize_);
  return std::make_shared<NativeAlgorithm>(
      "default_allreduce_nvls_block_pipeline", "allreduce",
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
