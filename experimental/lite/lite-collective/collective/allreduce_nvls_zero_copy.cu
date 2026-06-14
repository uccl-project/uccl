// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "allreduce_nvls_zero_copy.hpp"
#include "collective_utils.hpp"
#include "common.hpp"
#include "core.hpp"
#include "debug.h"

namespace mscclpp {
namespace collective {

constexpr int MAX_NBLOCKS = 32;

template <typename T>
__global__ void __launch_bounds__(1024, 1) allreduceNvls(
    [[maybe_unused]] mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>*
        memoryChannels,
    [[maybe_unused]] mscclpp::DeviceHandle<mscclpp::SwitchChannel>* multicast,
    [[maybe_unused]] mscclpp::DeviceHandle<mscclpp::SwitchChannel>*
        multicastOut,
    [[maybe_unused]] size_t channelInOffset,
    [[maybe_unused]] size_t channelOutOffset, [[maybe_unused]] size_t size,
    [[maybe_unused]] int rank, [[maybe_unused]] int nRanksPerNode) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  int nPeers = nRanksPerNode - 1;
  int nBlocks = gridDim.x;
  int bid = blockIdx.x;
  size_t sizePerRank = size / nRanksPerNode;
  const size_t minAlign = 16;
  // Align sizePerBlock to 16 bytes to ensure aligned vector access in
  // handleMultiLoadReduceStore
  size_t sizePerBlock = (sizePerRank + nBlocks - 1) / nBlocks;
  sizePerBlock = (sizePerBlock + minAlign - 1) / minAlign * minAlign;

  size_t rankOffset = sizePerRank * rank;
  size_t blockOffset = sizePerBlock * bid + rankOffset;
  size_t curBlockSize = 0;
  if (sizePerBlock * bid < sizePerRank) {
    curBlockSize = min(sizePerBlock, sizePerRank - sizePerBlock * bid);
  }

  mscclpp::DeviceHandle<mscclpp::SwitchChannel>* multicastPtr = multicast + bid;
  mscclpp::DeviceHandle<mscclpp::SwitchChannel>* multicastOutPtr =
      multicastOut + bid;

  const size_t chanOffset = (nRanksPerNode - 1) * blockIdx.x;
  auto memoryChans = memoryChannels + chanOffset;
  __shared__ mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>
      channels[MAX_NRANKS_PER_NODE - 1];
  int const lid = threadIdx.x % WARP_SIZE;
  if (lid < nRanksPerNode - 1) {
    channels[lid] = memoryChans[lid];
  }
  __syncwarp();
  if (threadIdx.x < nPeers) {
    channels[threadIdx.x].relaxedSignal();
    channels[threadIdx.x].relaxedWait();
  }
  __syncthreads();
  T* src = (T*)multicastPtr->mcPtr;
  T* dst = (T*)multicastOutPtr->mcPtr;
  if (curBlockSize > 0) {
    handleMultiLoadReduceStore(src, dst, blockOffset + channelInOffset,
                               blockOffset + channelOutOffset, curBlockSize,
                               threadIdx.x, blockDim.x);
  }
  __syncthreads();
  if (threadIdx.x < nPeers) {
    channels[threadIdx.x].relaxedSignal();
    channels[threadIdx.x].relaxedWait();
  }
#endif
}

template <ReduceOp OpType, typename T>
struct NvlsAdapter {
  static cudaError_t call(
      void const*, void*, void*, void* memoryChannels, void*,
      mscclpp::DeviceHandle<mscclpp::SwitchChannel>* nvlsChannels,
      mscclpp::DeviceHandle<mscclpp::SwitchChannel>* nvlsOutChannels,
      size_t channelInOffset, size_t channelOutOffset, size_t, int rank,
      int nRanksPerNode, int, size_t inputSize, cudaStream_t stream, void*,
      uint32_t, uint32_t, int nBlocks, int nThreadsPerBlock) {
    // uint8_t is not supported for NVLS (no hardware support for byte-level
    // reduction)
    if constexpr (std::is_same_v<T, uint8_t>) {
      return cudaErrorNotSupported;
    } else
#if (!defined(__CUDA_ARCH_SPECIFIC__) &&         \
     !defined(__CUDA_ARCH_FAMILY_SPECIFIC__)) || \
    (__CUDA_ARCH__ < 1000)
        if constexpr (std::is_same_v<T, __fp8_e4m3> ||
                      std::is_same_v<T, __fp8_e5m2>) {
      return cudaErrorNotSupported;
    } else
#endif
    {
      using ChannelType = DeviceHandle<mscclpp::BaseMemoryChannel>;
      allreduceNvls<T><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
          (ChannelType*)memoryChannels, nvlsChannels, nvlsOutChannels,
          channelInOffset, channelOutOffset, inputSize, rank, nRanksPerNode);
      return cudaGetLastError();
    }
  }
};

void AllreduceNvls::initialize(std::shared_ptr<mscclpp::Communicator> comm) {
  int device;
  MSCCLPP_CUDATHROW(cudaGetDevice(&device));
  cudaDeviceProp deviceProp;
  MSCCLPP_CUDATHROW(cudaGetDeviceProperties(&deviceProp, device));
  computeCapabilityMajor_ = deviceProp.major;
  nSwitchChannels_ = 32;
  this->conns_ = setupConnections(comm);
  // setup semaphores
  std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>>
      memorySemaphores =
          setupMemorySemaphores(comm, this->conns_, nSwitchChannels_);
  // setup base memory channels
  this->baseChannels_ =
      setupBaseMemoryChannels(this->conns_, memorySemaphores, nSwitchChannels_);
  this->memoryChannelsDeviceHandle_ =
      setupBaseMemoryChannelDeviceHandles(this->baseChannels_);
  this->nvlsConnections_ =
      setupNvlsConnections(comm, nvlsBufferSize_, nSwitchChannels_);
  this->nvlsOutConnections_ =
      setupNvlsConnections(comm, nvlsBufferSize_, nSwitchChannels_);
}

CommResult AllreduceNvls::allreduceKernelFunc(
    const std::shared_ptr<void> ctx_void, void const* input, void* output,
    size_t inputSize, mscclpp::DataType dtype, ReduceOp op, cudaStream_t stream,
    int nBlocks, int nThreadsPerBlock,
    std::unordered_map<std::string, uintptr_t> const&) {
  if (!symmetricMemory_) {
    WARN("AllreduceNvls requires symmetric memory for now.");
    return CommResult::CommInvalidArgument;
  }
  auto ctx = std::static_pointer_cast<AlgorithmCtx>(ctx_void);
  AllreduceFunc allreduce = dispatch<NvlsAdapter>(op, dtype);
  if (!allreduce) {
    WARN("Unsupported operation or data type for allreduce, dtype=%d",
         static_cast<int>(dtype));
    return CommResult::CommInvalidArgument;
  }
  size_t sendBytes, recvBytes;
  CUdeviceptr sendBasePtr, recvBasePtr;
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));
  size_t channelInOffset = (char*)input - (char*)sendBasePtr;
  size_t channelOutOffset = (char*)output - (char*)recvBasePtr;
  mscclpp::DeviceHandle<mscclpp::SwitchChannel>* nvlsChannels =
      ctx->switchChannelDeviceHandles.get();
  mscclpp::DeviceHandle<mscclpp::SwitchChannel>* nvlsOutChannels =
      ctx->switchChannelDeviceHandles.get();
  if (input != output) {
    nvlsOutChannels = nvlsOutChannels + nSwitchChannels_;
  }
  std::pair<int, int> numBlocksAndThreads = {nBlocks, nThreadsPerBlock};
  if (numBlocksAndThreads.first == 0 || numBlocksAndThreads.second == 0) {
    numBlocksAndThreads = {::min(ctx->nRanksPerNode, MAX_NBLOCKS), 1024};
    // For GB200 devices with MNNVLS (Multi-Node NVLink Sharp), scale the number
    // of blocks inversely with the number of GPUs. Empirically, 32 blocks works
    // well for 4 GPUs and 16 for 8 GPUs, which follows the formula 128 / nGPUs,
    // clamped to [1, MAX_NBLOCKS].
    if (computeCapabilityMajor_ == 10) {
      numBlocksAndThreads.first =
          ::max(1, ::min(128 / ctx->workSize, MAX_NBLOCKS));
    }
  }
  if (numBlocksAndThreads.first > MAX_NBLOCKS) {
    WARN("Number of blocks exceeds maximum supported value of %d", MAX_NBLOCKS);
    return CommResult::CommInvalidArgument;
  }
  cudaError_t error = allreduce(
      nullptr, nullptr, nullptr, this->memoryChannelsDeviceHandle_.get(),
      nullptr, nvlsChannels, nvlsOutChannels, channelInOffset, channelOutOffset,
      0, ctx->rank, ctx->nRanksPerNode, ctx->workSize, inputSize, stream,
      nullptr, 0, 0, numBlocksAndThreads.first, numBlocksAndThreads.second);
  if (error != cudaSuccess) {
    WARN("AllreduceNvls failed with error: %s", cudaGetErrorString(error));
    return CommResult::CommUnhandledCudaError;
  }
  return CommResult::CommSuccess;
}

mscclpp::AlgorithmCtxKey AllreduceNvls::generateAllreduceContextKey(
    void const* input, void* output, size_t, mscclpp::DataType,
    bool symmetricMemory) {
  symmetricMemory_ = symmetricMemory;
  size_t sendBytes, recvBytes;
  CUdeviceptr sendBasePtr, recvBasePtr;
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));
  return mscclpp::AlgorithmCtxKey{(void*)sendBasePtr, (void*)recvBasePtr,
                                  sendBytes, recvBytes, 0};
}

std::shared_ptr<void> AllreduceNvls::initAllreduceContext(
    std::shared_ptr<mscclpp::Communicator> comm, void const* input,
    void* output, size_t, mscclpp::DataType) {
  auto ctx = std::make_shared<AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

  size_t sendBytes, recvBytes;
  CUdeviceptr sendBasePtr, recvBasePtr;
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));

  // setup channels
  ctx->switchChannels = setupNvlsChannels(
      this->nvlsConnections_, (void*)sendBasePtr, sendBytes, nSwitchChannels_);
  if (input != output) {
    auto nvlsOutConnections = this->nvlsOutConnections_;
    std::vector<mscclpp::SwitchChannel> outChannels =
        setupNvlsChannels(this->nvlsOutConnections_, (void*)recvBasePtr,
                          recvBytes, nSwitchChannels_);
    ctx->switchChannels.insert(ctx->switchChannels.end(), outChannels.begin(),
                               outChannels.end());
  }

  ctx->switchChannelDeviceHandles =
      setupNvlsChannelDeviceHandles(ctx->switchChannels);
  return ctx;
}

std::shared_ptr<mscclpp::Algorithm> AllreduceNvls::build() {
  auto self = std::make_shared<AllreduceNvls>();
  return std::make_shared<mscclpp::NativeAlgorithm>(
      "default_allreduce_nvls_zero_copy", "allreduce",
      [self](std::shared_ptr<mscclpp::Communicator> comm) {
        self->initialize(comm);
      },
      [self](const std::shared_ptr<void> ctx, void const* input, void* output,
             size_t inputSize, [[maybe_unused]] size_t outputSize,
             mscclpp::DataType dtype, ReduceOp op, cudaStream_t stream,
             int nBlocks, int nThreadsPerBlock,
             std::unordered_map<std::string, uintptr_t> const& extras) {
        return self->allreduceKernelFunc(ctx, input, output, inputSize, dtype,
                                         op, stream, nBlocks, nThreadsPerBlock,
                                         extras);
      },
      [self](std::shared_ptr<mscclpp::Communicator> comm, void const* input,
             void* output, size_t inputSize, [[maybe_unused]] size_t outputSize,
             mscclpp::DataType dtype) {
        return self->initAllreduceContext(comm, input, output, inputSize,
                                          dtype);
      },
      [self](void const* input, void* output, size_t inputSize,
             [[maybe_unused]] size_t outputSize, mscclpp::DataType dtype,
             bool symmetricMemory) {
        return self->generateAllreduceContextKey(input, output, inputSize,
                                                 dtype, symmetricMemory);
      });
}
}  // namespace collective
}  // namespace mscclpp
