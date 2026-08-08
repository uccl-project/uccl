// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "algorithm.hpp"
#include "allreduce.hpp"
#include "common.hpp"
#include "datatype_conversion.hpp"
#include "debug.h"
#include "env.hpp"
#include "gpu.hpp"
#include "gpu_utils.hpp"
#include "nccl.h"

using AllreduceFunc = std::function<cudaError_t(
    void const*, void*, void*, void*, void*,
    mscclpp::DeviceHandle<mscclpp::SwitchChannel>*,
    mscclpp::DeviceHandle<mscclpp::SwitchChannel>*, size_t, size_t, size_t, int,
    int, int, size_t, cudaStream_t, uint32_t*, uint32_t*, uint32_t*, uint32_t)>;

namespace {

template <Op OpType, typename T>
struct AllpairAdapter {
  static cudaError_t call(void const* buff, void* scratch, void* resultBuff,
                          void* memoryChannels, void*,
                          mscclpp::DeviceHandle<mscclpp::SwitchChannel>*,
                          mscclpp::DeviceHandle<mscclpp::SwitchChannel>*,
                          size_t channelInOffset, size_t,
                          size_t scratchBufferSize, int rank, int nRanksPerNode,
                          int worldSize, size_t nelems, cudaStream_t stream,
                          uint32_t* deviceFlag7, uint32_t* deviceFlag28,
                          uint32_t* deviceFlag56, uint32_t numScratchBuff) {
    using ChannelType = mscclpp::DeviceHandle<mscclpp::MemoryChannel>;
    if (sizeof(T) * nelems < worldSize * sizeof(int)) {
      int nBlocks = worldSize - 1;
      int nThreadsPerBlock = 32;
      allreduceAllPairs<OpType><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
          (T*)buff, (T*)scratch, (T*)resultBuff, (ChannelType*)memoryChannels,
          channelInOffset, scratchBufferSize, rank, nRanksPerNode, worldSize,
          nelems, deviceFlag7, numScratchBuff);
    } else if (sizeof(T) * nelems <= (1 << 14)) {
      int nBlocks = (worldSize - 1) * 4;
      int nThreadsPerBlock = 512;
      allreduceAllPairs<OpType><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
          (T*)buff, (T*)scratch, (T*)resultBuff, (ChannelType*)memoryChannels,
          channelInOffset, scratchBufferSize, rank, nRanksPerNode, worldSize,
          nelems, deviceFlag28, numScratchBuff);
    } else if (sizeof(T) * nelems <= (1 << 20)) {
      int nBlocks = (nRanksPerNode - 1) * 4;
      int nThreadsPerBlock = 1024;
      uint32_t* deviceFlag = deviceFlag28;
      if (nelems >= 8192) {
        nBlocks = (worldSize - 1) * 8;
        nThreadsPerBlock = (nelems <= 76800) ? 512 : 1024;
        deviceFlag = deviceFlag56;

#if defined(__HIP_PLATFORM_AMD__)
        size_t sizeBytes = sizeof(T) * nelems;
        if constexpr (std::is_same_v<T, __half>) {
          // Half-specific tuning for 32KB-256KB range
          if (sizeBytes == (32 << 10)) {
            nThreadsPerBlock = 64;
          } else if (sizeBytes >= (64 << 10) && sizeBytes <= (256 << 10)) {
            nThreadsPerBlock = 128;
          }
        }

#if defined(__FP8_TYPES_EXIST__)
        // FP8-specific tuning for 32KB-256KB range
        if constexpr (std::is_same_v<T, __fp8_e4m3> ||
                      std::is_same_v<T, __fp8_e5m2>) {
          if (sizeBytes == (32 << 10)) {
            nThreadsPerBlock = 64;
          } else if (sizeBytes == (64 << 10)) {
            nThreadsPerBlock = 128;
          } else if (sizeBytes >= (128 << 10) && sizeBytes <= (256 << 10)) {
            nThreadsPerBlock = 256;
          }
        }
#endif
#endif
      }
#if defined(ENABLE_NPKIT)
      size_t NpkitSharedMemSize = NPKIT_SHM_NUM_EVENTS * sizeof(NpKitEvent);
      allreduce7<OpType>
          <<<nBlocks, nThreadsPerBlock, NpkitSharedMemSize, stream>>>(
              (T*)buff, (T*)scratch, (T*)resultBuff,
              (ChannelType*)memoryChannels, channelInOffset, scratchBufferSize,
              rank, nRanksPerNode, worldSize, nelems, deviceFlag,
              numScratchBuff, NpKit::GetGpuEventCollectContexts(),
              NpKit::GetCpuTimestamp());
#else
      allreduce7<OpType><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
          (T*)buff, (T*)scratch, (T*)resultBuff, (ChannelType*)memoryChannels,
          channelInOffset, scratchBufferSize, rank, nRanksPerNode, worldSize,
          nelems, deviceFlag, numScratchBuff);
#endif
    }
    return cudaGetLastError();
  }
};

template <Op OpType, typename T>
struct NvlsAdapter {
  static cudaError_t call(
      void const*, void*, void*, void* memoryChannels, void*,
      mscclpp::DeviceHandle<mscclpp::SwitchChannel>* nvlsChannels,
      mscclpp::DeviceHandle<mscclpp::SwitchChannel>* nvlsOutChannels,
      size_t channelInOffset, size_t channelOutOffset, size_t, int rank,
      int nRanksPerNode, int, size_t nelems, cudaStream_t stream, uint32_t*,
      uint32_t*, uint32_t*, uint32_t) {
#if defined(__CUDA_ARCH__)  // Skip the __CUDA_ARCH__ < 1000 since FP8 has not
                            // been supported for NVLS
    if constexpr (std::is_same_v<T, __fp8_e4m3> ||
                  std::is_same_v<T, __fp8_e5m2>) {
      return cudaErrorNotSupported;
    } else
#endif
    {
      using ChannelType = mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>;
      int nBlocks = nRanksPerNode;
      int nThreadsPerBlock = 1024;
      allreduce9<T><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
          (ChannelType*)memoryChannels, nvlsChannels, nvlsOutChannels,
          channelInOffset, channelOutOffset, nelems * sizeof(T), rank,
          nRanksPerNode);
      return cudaGetLastError();
    }
  }
};

template <Op OpType, typename T>
struct NvlsWithCopyAdapter {
  static cudaError_t call(
      void const* input, void* scratch, void* output, void* memoryChannels,
      void*, mscclpp::DeviceHandle<mscclpp::SwitchChannel>* nvlsChannels,
      mscclpp::DeviceHandle<mscclpp::SwitchChannel>*, size_t, size_t,
      size_t scratchBufferSize, int rank, int nRanksPerNode, int, size_t nelems,
      cudaStream_t stream, uint32_t*, uint32_t*, uint32_t*, uint32_t) {
#if defined(__CUDA_ARCH__)  // Skip the __CUDA_ARCH__ < 1000 since FP8 has not
                            // been supported for NVLS
    if constexpr (std::is_same_v<T, __fp8_e4m3> ||
                  std::is_same_v<T, __fp8_e5m2>) {
      return cudaErrorNotSupported;
    } else
#endif
    {
      using ChannelType = mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>;
      if (sizeof(T) * nelems < (1 << 24)) {
        int nBlocks = nRanksPerNode * 4;
        int nThreadsPerBlock = 1024;
        allreduce10<T><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
            input, scratch, output, (ChannelType*)memoryChannels, nvlsChannels,
            nelems * sizeof(T), scratchBufferSize, rank, nRanksPerNode);
      } else {
        int nBlocks = nRanksPerNode * 5;
        int nThreadsPerBlock = 1024;
        allreduce11<T><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
            input, scratch, output, (ChannelType*)memoryChannels, nvlsChannels,
            nelems * sizeof(T), scratchBufferSize, rank, nRanksPerNode);
      }
      return cudaGetLastError();
    }
  }
};

template <Op OpType, typename T>
struct Allreduce8Adapter {
  static cudaError_t call(void const* buff, void* scratch, void* resultBuff,
                          void* memoryChannels, void* memoryOutChannels,
                          mscclpp::DeviceHandle<mscclpp::SwitchChannel>*,
                          mscclpp::DeviceHandle<mscclpp::SwitchChannel>*,
                          size_t, size_t channelOutOffset, size_t, int rank,
                          int nRanksPerNode, int worldSize, size_t nelems,
                          cudaStream_t stream, uint32_t*, uint32_t*, uint32_t*,
                          uint32_t) {
    using ChannelType = mscclpp::DeviceHandle<mscclpp::MemoryChannel>;
    int nBlocks = (nRanksPerNode - 1) * 5;
    int nThreadsPerBlock = 512;
    allreduce8<OpType><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
        (T*)buff, (T*)scratch, (T*)resultBuff, (ChannelType*)memoryChannels,
        (ChannelType*)memoryOutChannels, channelOutOffset, 0, rank,
        nRanksPerNode, worldSize, nelems);
    return cudaGetLastError();
  }
};

template <Op OpType, typename T>
struct AllreduceNvlsPacketAdapter {
  static cudaError_t call(
      void const* input, void* scratch, void* output, void*, void*,
      mscclpp::DeviceHandle<mscclpp::SwitchChannel>* nvlsChannels,
      mscclpp::DeviceHandle<mscclpp::SwitchChannel>*, size_t, size_t,
      size_t scratchBufferSize, int rank, int, int worldSize, size_t nelems,
      cudaStream_t stream, uint32_t* deviceFlag, uint32_t*, uint32_t*,
      uint32_t) {
    size_t size = nelems * sizeof(T);
    int nBlocks = 8;
    int nThreadsPerBlock = 1024;
    if (size <= (1 << 13)) {
      nBlocks = 4;
      nThreadsPerBlock = 512;
    }
    allreduceNvlsPacket<OpType, T><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
        (const T*)input, (T*)scratch, (T*)output, nvlsChannels, nelems,
        scratchBufferSize, rank, worldSize, deviceFlag);
    return cudaGetLastError();
  }
};

template <template <Op, typename> class Adapter>
AllreduceFunc dispatch(ncclRedOp_t op, mscclpp::DataType dtype) {
  Op reduceOp = getReduceOp(op);

  if (reduceOp == SUM) {
    if (dtype == mscclpp::DataType::FLOAT16) {
      return Adapter<SUM, half>::call;
    } else if (dtype == mscclpp::DataType::FLOAT32) {
      return Adapter<SUM, float>::call;
#if defined(__CUDA_BF16_TYPES_EXIST__)
    } else if (dtype == mscclpp::DataType::BFLOAT16) {
      return Adapter<SUM, __bfloat16>::call;
#endif
#if defined(__FP8_TYPES_EXIST__)
    } else if (dtype == mscclpp::DataType::FLOAT8_E4M3) {
      return Adapter<SUM, __fp8_e4m3>::call;
    } else if (dtype == mscclpp::DataType::FLOAT8_E5M2) {
      return Adapter<SUM, __fp8_e5m2>::call;
#endif
    } else if (dtype == mscclpp::DataType::INT32 ||
               dtype == mscclpp::DataType::UINT32) {
      return Adapter<SUM, int>::call;
    } else {
      return nullptr;
    }
  } else if (reduceOp == MIN) {
    if (dtype == mscclpp::DataType::FLOAT16) {
      return Adapter<MIN, half>::call;
    } else if (dtype == mscclpp::DataType::FLOAT32) {
      return Adapter<MIN, float>::call;
#if defined(__CUDA_BF16_TYPES_EXIST__)
    } else if (dtype == mscclpp::DataType::BFLOAT16) {
      return Adapter<MIN, __bfloat16>::call;
#endif
#if defined(__FP8_TYPES_EXIST__)
    } else if (dtype == mscclpp::DataType::FLOAT8_E4M3) {
      return Adapter<MIN, __fp8_e4m3>::call;
    } else if (dtype == mscclpp::DataType::FLOAT8_E5M2) {
      return Adapter<MIN, __fp8_e5m2>::call;
#endif
    } else if (dtype == mscclpp::DataType::INT32 ||
               dtype == mscclpp::DataType::UINT32) {
      return Adapter<MIN, int>::call;
    } else {
      return nullptr;
    }
  }
  return nullptr;
}
}  // namespace

enum Op getReduceOp(ncclRedOp_t op) {
  switch (op) {
    case ncclSum:
      return SUM;
    case ncclMin:
      return MIN;
    default:
      WARN("op is invalid, op: %d", op);
      throw mscclpp::Error("Invalid operation",
                           mscclpp::ErrorCode::InternalError);
  }
}

AllreducePacket::AllreducePacket(uintptr_t scratchBuffer,
                                 size_t scratchBufferSize)
    : scratchPtr_(reinterpret_cast<char*>(scratchBuffer)),
      scratchBufferSize_(scratchBufferSize) {}

void AllreducePacket::initialize(
    std::shared_ptr<mscclpp::Communicator> comm,
    std::unordered_map<std::string, std::shared_ptr<void>>&) {
  deviceFlag7_ = mscclpp::detail::gpuCallocShared<uint32_t>(7);
  deviceFlag28_ = mscclpp::detail::gpuCallocShared<uint32_t>(28);
  deviceFlag56_ = mscclpp::detail::gpuCallocShared<uint32_t>(56);
  std::vector<uint32_t> initFlag(56);
  for (int i = 0; i < 56; ++i) {
    initFlag[i] = 1;
  }
  mscclpp::gpuMemcpy<uint32_t>(deviceFlag7_.get(), initFlag.data(), 7,
                               cudaMemcpyHostToDevice);
  mscclpp::gpuMemcpy<uint32_t>(deviceFlag28_.get(), initFlag.data(), 28,
                               cudaMemcpyHostToDevice);
  mscclpp::gpuMemcpy<uint32_t>(deviceFlag56_.get(), initFlag.data(), 56,
                               cudaMemcpyHostToDevice);
  this->conns_ = setupConnections(comm);
}

ncclResult_t AllreducePacket::allreduceKernelFunc(
    const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, void const* input,
    void* output, size_t count, mscclpp::DataType dtype, cudaStream_t stream,
    std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
  ncclRedOp_t op = *static_cast<ncclRedOp_t*>(extras.at("op").get());

  size_t sendBytes;
  CUdeviceptr sendBasePtr;
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  size_t channelInOffset = (char*)input - (char*)sendBasePtr;

  AllreduceFunc allreduce = dispatch<AllpairAdapter>(op, dtype);
  if (!allreduce) {
    WARN("Unsupported operation or data type for allreduce: op=%d, dtype=%d",
         op, static_cast<int>(dtype));
    return ncclInvalidArgument;
  }
  cudaError_t error = allreduce(
      input, scratchPtr_, output, ctx->memoryChannelDeviceHandles.get(),
      nullptr, nullptr, nullptr, channelInOffset, 0, this->scratchBufferSize_,
      ctx->rank, ctx->nRanksPerNode, ctx->workSize, count, stream,
      deviceFlag7_.get(), deviceFlag28_.get(), deviceFlag56_.get(),
      this->nSegmentsForScratchBuffer_);
  if (error != cudaSuccess) {
    WARN("AllreducePacket failed with error: %s", cudaGetErrorString(error));
    return ncclUnhandledCudaError;
  }
  return ncclSuccess;
}

std::shared_ptr<mscclpp::AlgorithmCtx> AllreducePacket::initAllreduceContext(
    std::shared_ptr<mscclpp::Communicator> comm, void const* input, void*,
    size_t, mscclpp::DataType) {
  auto ctx = std::make_shared<mscclpp::AlgorithmCtx>();
  int const nChannelsPerConnection = 56;
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();
  if (this->ctx_ == nullptr) {
    // setup semaphores
    ctx->memorySemaphores =
        setupMemorySemaphores(comm, this->conns_, nChannelsPerConnection);
    // setup registered memories
    mscclpp::RegisteredMemory scratchMemory = comm->registerMemory(
        scratchPtr_, this->scratchBufferSize_, mscclpp::Transport::CudaIpc);
    std::vector<mscclpp::RegisteredMemory> remoteMemories =
        setupRemoteMemories(comm, ctx->rank, scratchMemory);
    ctx->registeredMemories = std::move(remoteMemories);
    ctx->registeredMemories.push_back(scratchMemory);
  } else {
    ctx->memorySemaphores = ctx_->memorySemaphores;
    ctx->registeredMemories = ctx_->registeredMemories;
    ctx->registeredMemories
        .pop_back();  // remove the local memory from previous context
  }

  size_t sendBytes;
  CUdeviceptr sendBasePtr;
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  mscclpp::RegisteredMemory localMemory = comm->registerMemory(
      (void*)sendBasePtr, sendBytes, mscclpp::Transport::CudaIpc);

  // setup channels
  ctx->memoryChannels = setupMemoryChannels(
      this->conns_, ctx->memorySemaphores, ctx->registeredMemories, localMemory,
      nChannelsPerConnection);
  ctx->memoryChannelDeviceHandles =
      setupMemoryChannelDeviceHandles(ctx->memoryChannels);
  ctx->registeredMemories.emplace_back(localMemory);

  this->ctx_ = ctx;

  return ctx;
}

mscclpp::AlgorithmCtxKey AllreducePacket::generateAllreduceContextKey(
    void const* input, void*, size_t, mscclpp::DataType) {
  size_t sendBytes;
  CUdeviceptr sendBasePtr;
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  return mscclpp::AlgorithmCtxKey{(void*)sendBasePtr, nullptr, sendBytes, 0, 0};
}

std::shared_ptr<mscclpp::Algorithm> AllreducePacket::build() {
  auto self = shared_from_this();
  return std::make_shared<mscclpp::NativeAlgorithm>(
      "default_allreduce_packet", "allreduce",
      [self](std::shared_ptr<mscclpp::Communicator> comm) {
        std::unordered_map<std::string, std::shared_ptr<void>> e;
        self->initialize(comm, e);
      },
      [self](std::shared_ptr<void> const& ctxPtr, void const* input,
             void* output, size_t inputBytes, size_t /*outBytes*/,
             mscclpp::DataType dtype, mscclpp::ReduceOp rop,
             cudaStream_t stream, int, int,
             std::unordered_map<std::string, uintptr_t> const&) {
        auto ctx = std::static_pointer_cast<mscclpp::AlgorithmCtx>(ctxPtr);
        size_t const count = inputBytes / getDataTypeSize(dtype);
        auto ex = mscclpp::nccl_port::legacyExtrasWithOp(rop);
        return mscclpp::nccl_port::ncclResultToCommResult(
            self->allreduceKernelFunc(ctx, input, output, count, dtype, stream,
                                      ex));
      },
      [self](std::shared_ptr<mscclpp::Communicator> comm, void const* input,
             void* output, size_t inputBytes, size_t /*outBytes*/,
             mscclpp::DataType dtype) {
        size_t const count = inputBytes / getDataTypeSize(dtype);
        return std::static_pointer_cast<void>(
            self->initAllreduceContext(comm, input, output, count, dtype));
      },
      [self](void const* input, void* output, size_t inputBytes,
             size_t /*outBytes*/, mscclpp::DataType dtype, bool) {
        size_t const count = inputBytes / getDataTypeSize(dtype);
        return self->generateAllreduceContextKey(input, output, count, dtype);
      });
}

void AllreduceNvls::initialize(
    std::shared_ptr<mscclpp::Communicator> comm,
    std::unordered_map<std::string, std::shared_ptr<void>>&) {
  nSwitchChannels_ = 8;
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
}

ncclResult_t AllreduceNvls::allreduceKernelFunc(
    const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, void const* input,
    void* output, size_t count, mscclpp::DataType dtype, cudaStream_t stream,
    std::unordered_map<std::string, std::shared_ptr<void>>&) {
  AllreduceFunc allreduce = dispatch<NvlsAdapter>(ncclSum, dtype);
  if (!allreduce) {
    WARN("Unsupported operation or data type for allreduce, dtype=%d",
         static_cast<int>(dtype));
    return ncclInvalidArgument;
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
  cudaError_t error = allreduce(
      nullptr, nullptr, nullptr, this->memoryChannelsDeviceHandle_.get(),
      nullptr, nvlsChannels, nvlsOutChannels, channelInOffset, channelOutOffset,
      0, ctx->rank, ctx->nRanksPerNode, ctx->workSize, count, stream, nullptr,
      nullptr, nullptr, 0);
  if (error != cudaSuccess) {
    WARN("AllreduceNvls failed with error: %s", cudaGetErrorString(error));
    return ncclUnhandledCudaError;
  }
  return ncclSuccess;
}

mscclpp::AlgorithmCtxKey AllreduceNvls::generateAllreduceContextKey(
    void const* input, void* output, size_t, mscclpp::DataType) {
  size_t sendBytes, recvBytes;
  CUdeviceptr sendBasePtr, recvBasePtr;
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));
  return mscclpp::AlgorithmCtxKey{(void*)sendBasePtr, (void*)recvBasePtr,
                                  sendBytes, recvBytes, 0};
}

std::shared_ptr<mscclpp::AlgorithmCtx> AllreduceNvls::initAllreduceContext(
    std::shared_ptr<mscclpp::Communicator> comm, void const* input,
    void* output, size_t, mscclpp::DataType) {
  auto ctx = std::make_shared<mscclpp::AlgorithmCtx>();
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
  ctx->nvlsConnections =
      setupNvlsConnections(comm, nvlsBufferSize_, nSwitchChannels_);
  ctx->switchChannels = setupNvlsChannels(
      ctx->nvlsConnections, (void*)sendBasePtr, sendBytes, nSwitchChannels_);
  if (input != output) {
    auto nvlsOutConnections =
        setupNvlsConnections(comm, nvlsBufferSize_, nSwitchChannels_);
    std::vector<mscclpp::SwitchChannel> outChannels = setupNvlsChannels(
        nvlsOutConnections, (void*)recvBasePtr, recvBytes, nSwitchChannels_);
    ctx->nvlsConnections.insert(ctx->nvlsConnections.end(),
                                nvlsOutConnections.begin(),
                                nvlsOutConnections.end());
    ctx->switchChannels.insert(ctx->switchChannels.end(), outChannels.begin(),
                               outChannels.end());
  }

  ctx->switchChannelDeviceHandles =
      setupNvlsChannelDeviceHandles(ctx->switchChannels);
  return ctx;
}

std::shared_ptr<mscclpp::Algorithm> AllreduceNvls::build() {
  auto self = shared_from_this();
  return std::make_shared<mscclpp::NativeAlgorithm>(
      "default_allreduce_nvls", "allreduce",
      [self](std::shared_ptr<mscclpp::Communicator> comm) {
        std::unordered_map<std::string, std::shared_ptr<void>> e;
        self->initialize(comm, e);
      },
      [self](std::shared_ptr<void> const& ctxPtr, void const* input,
             void* output, size_t inputBytes, size_t /*ob*/,
             mscclpp::DataType dtype, mscclpp::ReduceOp /*rop*/,
             cudaStream_t stream, int, int,
             std::unordered_map<std::string, uintptr_t> const&) {
        auto ctx = std::static_pointer_cast<mscclpp::AlgorithmCtx>(ctxPtr);
        size_t const count = inputBytes / getDataTypeSize(dtype);
        std::unordered_map<std::string, std::shared_ptr<void>> e;
        return mscclpp::nccl_port::ncclResultToCommResult(
            self->allreduceKernelFunc(ctx, input, output, count, dtype, stream,
                                      e));
      },
      [self](std::shared_ptr<mscclpp::Communicator> comm, void const* input,
             void* output, size_t inputBytes, size_t /*ob*/,
             mscclpp::DataType dtype) {
        size_t const count = inputBytes / getDataTypeSize(dtype);
        return std::static_pointer_cast<void>(
            self->initAllreduceContext(comm, input, output, count, dtype));
      },
      [self](void const* input, void* output, size_t inputBytes, size_t /*ob*/,
             mscclpp::DataType dtype, bool) {
        size_t const count = inputBytes / getDataTypeSize(dtype);
        return self->generateAllreduceContextKey(input, output, count, dtype);
      });
}

AllreduceNvlsWithCopy::AllreduceNvlsWithCopy(uintptr_t scratchBuffer,
                                             size_t scratchBufferSize)
    : scratchPtr_(reinterpret_cast<char*>(scratchBuffer)),
      scratchBufferSize_(scratchBufferSize) {}

void AllreduceNvlsWithCopy::initialize(
    std::shared_ptr<mscclpp::Communicator> comm,
    std::unordered_map<std::string, std::shared_ptr<void>>&) {
  nSwitchChannels_ = 8;
  int nBaseChannels = 64;
  this->conns_ = setupConnections(comm);
  // setup semaphores
  std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>>
      memorySemaphores =
          setupMemorySemaphores(comm, this->conns_, nBaseChannels);
  // setup base memory channels
  this->baseChannels_ =
      setupBaseMemoryChannels(this->conns_, memorySemaphores, nBaseChannels);
  this->memoryChannelsDeviceHandle_ =
      setupBaseMemoryChannelDeviceHandles(this->baseChannels_);
}

ncclResult_t AllreduceNvlsWithCopy::allreduceKernelFunc(
    const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, void const* input,
    void* output, size_t count, mscclpp::DataType dtype, cudaStream_t stream,
    std::unordered_map<std::string, std::shared_ptr<void>>&) {
  AllreduceFunc allreduce = dispatch<NvlsWithCopyAdapter>(ncclSum, dtype);
  if (!allreduce) {
    WARN("Unsupported operation or data type for allreduce, dtype=%d",
         static_cast<int>(dtype));
    return ncclInvalidArgument;
  }
  cudaError_t error = allreduce(
      input, scratchPtr_, output, this->memoryChannelsDeviceHandle_.get(),
      nullptr, ctx->switchChannelDeviceHandles.get(), nullptr, 0, 0,
      this->scratchBufferSize_, ctx->rank, ctx->nRanksPerNode, ctx->workSize,
      count, stream, nullptr, nullptr, nullptr, 0);
  if (error != cudaSuccess) {
    WARN("AllreduceNvlsWithCopy failed with error: %s",
         cudaGetErrorString(error));
    return ncclUnhandledCudaError;
  }
  return ncclSuccess;
}

mscclpp::AlgorithmCtxKey AllreduceNvlsWithCopy::generateAllreduceContextKey(
    void const*, void*, size_t, mscclpp::DataType) {
  return mscclpp::AlgorithmCtxKey{nullptr, nullptr, 0, 0, 0};
}

std::shared_ptr<mscclpp::AlgorithmCtx>
AllreduceNvlsWithCopy::initAllreduceContext(
    std::shared_ptr<mscclpp::Communicator> comm, void const*, void*, size_t,
    mscclpp::DataType) {
  auto ctx = std::make_shared<mscclpp::AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

  // setup channels
  ctx->nvlsConnections =
      setupNvlsConnections(comm, nvlsBufferSize_, nSwitchChannels_);
  ctx->switchChannels = setupNvlsChannels(ctx->nvlsConnections, scratchPtr_,
                                          scratchBufferSize_, nSwitchChannels_);
  ctx->switchChannelDeviceHandles =
      setupNvlsChannelDeviceHandles(ctx->switchChannels);
  return ctx;
}

std::shared_ptr<mscclpp::Algorithm> AllreduceNvlsWithCopy::build() {
  auto self = shared_from_this();
  return std::make_shared<mscclpp::NativeAlgorithm>(
      "default_allreduce_nvls_with_copy", "allreduce",
      [self](std::shared_ptr<mscclpp::Communicator> comm) {
        std::unordered_map<std::string, std::shared_ptr<void>> e;
        self->initialize(comm, e);
      },
      [self](std::shared_ptr<void> const& ctxPtr, void const* input,
             void* output, size_t inputBytes, size_t, mscclpp::DataType dtype,
             mscclpp::ReduceOp, cudaStream_t stream, int, int,
             std::unordered_map<std::string, uintptr_t> const&) {
        auto ctx = std::static_pointer_cast<mscclpp::AlgorithmCtx>(ctxPtr);
        size_t const count = inputBytes / getDataTypeSize(dtype);
        std::unordered_map<std::string, std::shared_ptr<void>> e;
        return mscclpp::nccl_port::ncclResultToCommResult(
            self->allreduceKernelFunc(ctx, input, output, count, dtype, stream,
                                      e));
      },
      [self](std::shared_ptr<mscclpp::Communicator> comm, void const* input,
             void* output, size_t inputBytes, size_t, mscclpp::DataType dtype) {
        size_t const count = inputBytes / getDataTypeSize(dtype);
        return std::static_pointer_cast<void>(
            self->initAllreduceContext(comm, input, output, count, dtype));
      },
      [self](void const* input, void* output, size_t inputBytes, size_t,
             mscclpp::DataType dtype, bool) {
        size_t const count = inputBytes / getDataTypeSize(dtype);
        return self->generateAllreduceContextKey(input, output, count, dtype);
      });
}

Allreduce8::Allreduce8(uintptr_t scratchBuffer, size_t scratchBufferSize)
    : scratchPtr_(reinterpret_cast<char*>(scratchBuffer)),
      scratchBufferSize_(scratchBufferSize) {}

void Allreduce8::initialize(
    std::shared_ptr<mscclpp::Communicator> comm,
    std::unordered_map<std::string, std::shared_ptr<void>>&) {
  this->conns_ = setupConnections(comm);
  nChannelsPerConnection_ = 64;
  comm_ = comm;
  // setup semaphores
  this->outputSemaphores_ =
      setupMemorySemaphores(comm, this->conns_, nChannelsPerConnection_);
  this->inputScratchSemaphores_ =
      setupMemorySemaphores(comm, this->conns_, nChannelsPerConnection_);
  mscclpp::RegisteredMemory localMemory = comm->registerMemory(
      scratchPtr_, scratchBufferSize_, mscclpp::Transport::CudaIpc);
  this->remoteScratchMemories_ =
      setupRemoteMemories(comm, comm->bootstrap()->getRank(), localMemory);
  localScratchMemory_ = std::move(localMemory);
}

ncclResult_t Allreduce8::allreduceKernelFunc(
    const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, void const* input,
    void* output, size_t count, mscclpp::DataType dtype, cudaStream_t stream,
    std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
  const size_t bytes = count * getDataTypeSize(dtype);
  ncclRedOp_t op = *static_cast<ncclRedOp_t*>(extras.at("op").get());

  size_t recvBytes;
  CUdeviceptr recvBasePtr;
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));
  size_t channelOutOffset = (char*)output - (char*)recvBasePtr;
  std::shared_ptr<mscclpp::MemoryChannel::DeviceHandle> inputChannelHandles;
  if (this->memoryChannelsMap_.find(input) != this->memoryChannelsMap_.end()) {
    inputChannelHandles = this->memoryChannelsMap_[input].second;
  } else {
    mscclpp::RegisteredMemory localMemory = comm_->registerMemory(
        const_cast<void*>(input), bytes, mscclpp::Transport::CudaIpc);
    std::vector<mscclpp::MemoryChannel> channels = setupMemoryChannels(
        this->conns_, this->inputScratchSemaphores_,
        this->remoteScratchMemories_, localMemory, nChannelsPerConnection_);
    this->memoryChannelsMap_[input] =
        std::make_pair(channels, setupMemoryChannelDeviceHandles(channels));
  }
  inputChannelHandles = this->memoryChannelsMap_[input].second;

  AllreduceFunc allreduce = dispatch<Allreduce8Adapter>(op, dtype);
  if (!allreduce) {
    WARN("Unsupported operation or data type for allreduce: op=%d, dtype=%d",
         op, static_cast<int>(dtype));
    return ncclInvalidArgument;
  }
  cudaError_t error =
      allreduce(input, scratchPtr_, output, inputChannelHandles.get(),
                ctx->memoryChannelDeviceHandles.get(), nullptr, nullptr, 0,
                channelOutOffset, 0, ctx->rank, ctx->nRanksPerNode,
                ctx->workSize, count, stream, nullptr, nullptr, nullptr, 0);
  if (error != cudaSuccess) {
    WARN("Allreduce8 failed with error: %s", cudaGetErrorString(error));
    return ncclUnhandledCudaError;
  }
  return ncclSuccess;
}

mscclpp::AlgorithmCtxKey Allreduce8::generateAllreduceContextKey(
    void const*, void* output, size_t, mscclpp::DataType) {
  static int tag = 0;
  size_t recvBytes;
  CUdeviceptr recvBasePtr;
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));
  if (mscclpp::env()->disableChannelCache) {
    return mscclpp::AlgorithmCtxKey{nullptr, (void*)recvBasePtr, 0, recvBytes,
                                    tag++};
  }
  return mscclpp::AlgorithmCtxKey{nullptr, (void*)recvBasePtr, 0, recvBytes, 0};
}

std::shared_ptr<mscclpp::AlgorithmCtx> Allreduce8::initAllreduceContext(
    std::shared_ptr<mscclpp::Communicator> comm, void const*, void* output,
    size_t, mscclpp::DataType) {
  auto ctx = std::make_shared<mscclpp::AlgorithmCtx>();
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
  mscclpp::RegisteredMemory localMemory = comm->registerMemory(
      (void*)recvBasePtr, recvBytes, mscclpp::Transport::CudaIpc);
  ctx->registeredMemories = setupRemoteMemories(comm, ctx->rank, localMemory);
  ctx->memoryChannels = setupMemoryChannels(
      this->conns_, ctx->memorySemaphores, ctx->registeredMemories, localMemory,
      nChannelsPerConnection_);
  ctx->memoryChannelDeviceHandles =
      setupMemoryChannelDeviceHandles(ctx->memoryChannels);
  return ctx;
}

std::shared_ptr<mscclpp::Algorithm> Allreduce8::build() {
  auto self = shared_from_this();
  return std::make_shared<mscclpp::NativeAlgorithm>(
      "default_allreduce_allreduce8", "allreduce",
      [self](std::shared_ptr<mscclpp::Communicator> comm) {
        std::unordered_map<std::string, std::shared_ptr<void>> e;
        self->initialize(comm, e);
      },
      [self](std::shared_ptr<void> const& ctxPtr, void const* input,
             void* output, size_t inputBytes, size_t, mscclpp::DataType dtype,
             mscclpp::ReduceOp rop, cudaStream_t stream, int, int,
             std::unordered_map<std::string, uintptr_t> const&) {
        auto ctx = std::static_pointer_cast<mscclpp::AlgorithmCtx>(ctxPtr);
        size_t const count = inputBytes / getDataTypeSize(dtype);
        auto ex = mscclpp::nccl_port::legacyExtrasWithOp(rop);
        return mscclpp::nccl_port::ncclResultToCommResult(
            self->allreduceKernelFunc(ctx, input, output, count, dtype, stream,
                                      ex));
      },
      [self](std::shared_ptr<mscclpp::Communicator> comm, void const* input,
             void* output, size_t inputBytes, size_t, mscclpp::DataType dtype) {
        size_t const count = inputBytes / getDataTypeSize(dtype);
        return std::static_pointer_cast<void>(
            self->initAllreduceContext(comm, input, output, count, dtype));
      },
      [self](void const* input, void* output, size_t inputBytes, size_t,
             mscclpp::DataType dtype, bool) {
        size_t const count = inputBytes / getDataTypeSize(dtype);
        return self->generateAllreduceContextKey(input, output, count, dtype);
      });
}

AllreduceNvlsPacket::AllreduceNvlsPacket(uintptr_t scratchBuffer,
                                         size_t scratchBufferSize)
    : scratchPtr_(reinterpret_cast<char*>(scratchBuffer)),
      scratchBufferSize_(scratchBufferSize) {}

void AllreduceNvlsPacket::initialize(
    std::shared_ptr<mscclpp::Communicator>,
    std::unordered_map<std::string, std::shared_ptr<void>>&) {
  deviceFlag_ = mscclpp::detail::gpuCallocShared<uint32_t>(16);
  std::vector<uint32_t> initFlag(16);
  for (int i = 0; i < 16; ++i) {
    initFlag[i] = 1;
  }
  mscclpp::gpuMemcpy<uint32_t>(deviceFlag_.get(), initFlag.data(), 16,
                               cudaMemcpyHostToDevice);
}

mscclpp::AlgorithmCtxKey AllreduceNvlsPacket::generateAllreduceContextKey(
    void const*, void*, size_t, mscclpp::DataType) {
  return mscclpp::AlgorithmCtxKey{nullptr, nullptr, 0, 0, 0};
}

std::shared_ptr<mscclpp::AlgorithmCtx>
AllreduceNvlsPacket::initAllreduceContext(
    std::shared_ptr<mscclpp::Communicator> comm, void const*, void*, size_t,
    mscclpp::DataType) {
  auto ctx = std::make_shared<mscclpp::AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

  // setup channels
  int nSwitchChannels = 1;
  ctx->nvlsConnections =
      setupNvlsConnections(comm, nvlsBufferSize_, nSwitchChannels);
  ctx->switchChannels =
      setupNvlsChannels(ctx->nvlsConnections, scratchPtr_,
                        this->scratchBufferSize_, nSwitchChannels);
  ctx->switchChannelDeviceHandles =
      setupNvlsChannelDeviceHandles(ctx->switchChannels);
  return ctx;
}

ncclResult_t AllreduceNvlsPacket::allreduceKernelFunc(
    const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, void const* input,
    void* output, size_t count, mscclpp::DataType dtype, cudaStream_t stream,
    std::unordered_map<std::string, std::shared_ptr<void>>& extra) {
  int op = *static_cast<int*>(extra.at("op").get());
  AllreduceFunc allreduce =
      dispatch<AllreduceNvlsPacketAdapter>(static_cast<ncclRedOp_t>(op), dtype);
  if (!allreduce) {
    WARN("Unsupported operation or data type for allreduce, dtype=%d",
         static_cast<int>(dtype));
    return ncclInvalidArgument;
  }
  cudaError_t error = allreduce(
      input, scratchPtr_, output, nullptr, nullptr,
      ctx->switchChannelDeviceHandles.get(), nullptr, 0, 0,
      this->scratchBufferSize_, ctx->rank, ctx->nRanksPerNode, ctx->workSize,
      count, stream, this->deviceFlag_.get(), nullptr, nullptr, 0);
  if (error != cudaSuccess) {
    WARN("AllreduceNvlsPacket failed with error: %s",
         cudaGetErrorString(error));
    return ncclUnhandledCudaError;
  }
  return ncclSuccess;
}

std::shared_ptr<mscclpp::Algorithm> AllreduceNvlsPacket::build() {
  auto self = shared_from_this();
  return std::make_shared<mscclpp::NativeAlgorithm>(
      "default_allreduce_nvls_packet", "allreduce",
      [self](std::shared_ptr<mscclpp::Communicator> comm) {
        std::unordered_map<std::string, std::shared_ptr<void>> e;
        self->initialize(comm, e);
      },
      [self](std::shared_ptr<void> const& ctxPtr, void const* input,
             void* output, size_t inputBytes, size_t, mscclpp::DataType dtype,
             mscclpp::ReduceOp rop, cudaStream_t stream, int, int,
             std::unordered_map<std::string, uintptr_t> const&) {
        auto ctx = std::static_pointer_cast<mscclpp::AlgorithmCtx>(ctxPtr);
        size_t const count = inputBytes / getDataTypeSize(dtype);
        auto ex = mscclpp::nccl_port::legacyExtrasWithOp(rop);
        return mscclpp::nccl_port::ncclResultToCommResult(
            self->allreduceKernelFunc(ctx, input, output, count, dtype, stream,
                                      ex));
      },
      [self](std::shared_ptr<mscclpp::Communicator> comm, void const* input,
             void* output, size_t inputBytes, size_t, mscclpp::DataType dtype) {
        size_t const count = inputBytes / getDataTypeSize(dtype);
        return std::static_pointer_cast<void>(
            self->initAllreduceContext(comm, input, output, count, dtype));
      },
      [self](void const* input, void* output, size_t inputBytes, size_t,
             mscclpp::DataType dtype, bool) {
        size_t const count = inputBytes / getDataTypeSize(dtype);
        return self->generateAllreduceContextKey(input, output, count, dtype);
      });
}