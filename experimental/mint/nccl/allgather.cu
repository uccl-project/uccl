// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "algorithm.hpp"
#include "allgather.hpp"
#include "common.hpp"
#include "datatype_conversion.hpp"
#include "debug.h"
#include "env.hpp"
#include "gpu_utils.hpp"
#include "nccl.h"

AllgatherAlgo6::AllgatherAlgo6() : disableChannelCache_(false) {
  if (mscclpp::env()->disableChannelCache) {
    disableChannelCache_ = true;
  }
}

void AllgatherAlgo6::initialize(
    std::shared_ptr<mscclpp::Communicator> comm,
    std::unordered_map<std::string, std::shared_ptr<void>>&) {
  this->conns_ = setupConnections(comm);
  this->memorySemaphores_ = std::move(
      setupMemorySemaphores(comm, this->conns_, nChannelsPerConnection_));
}

ncclResult_t AllgatherAlgo6::allgatherKernelFunc(
    const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, void const* input,
    void* output, size_t count, mscclpp::DataType dtype, cudaStream_t stream,
    std::unordered_map<std::string, std::shared_ptr<void>>&) {
  int nBlocks = 28;
  const size_t bytes = count * getDataTypeSize(dtype);
  const size_t nElem = bytes / sizeof(int);
  int rank = ctx->rank;
  if (bytes <= 32 * (1 << 20)) {
    if (nElem <= 4096) {
      nBlocks = 7;
    } else if (nElem <= 32768) {
      nBlocks = 14;
    } else if (nElem >= 2097152) {
      nBlocks = 35;
    }
  } else {
    nBlocks = 35;
  }

  size_t channelOutOffset =
      *static_cast<size_t*>(ctx->extras["channel_out_offset"].get());
  if ((char*)input == (char*)output + rank * bytes) {
    allgather6<false><<<nBlocks, 1024, 0, stream>>>(
        (void*)input, ctx->memoryChannelDeviceHandles.get(), channelOutOffset,
        ctx->rank, ctx->workSize, ctx->nRanksPerNode, nElem);
  } else {
    allgather6<true><<<nBlocks, 1024, 0, stream>>>(
        (void*)input, ctx->memoryChannelDeviceHandles.get(), channelOutOffset,
        ctx->rank, ctx->workSize, ctx->nRanksPerNode, nElem);
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    WARN("AllgatherAlgo6 failed with error %d", err);
    return ncclInternalError;
  }
  return ncclSuccess;
}

std::shared_ptr<mscclpp::AlgorithmCtx> AllgatherAlgo6::initAllgatherContext(
    std::shared_ptr<mscclpp::Communicator> comm, void const*, void* output,
    size_t count, mscclpp::DataType dtype) {
  auto ctx = std::make_shared<mscclpp::AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

  // setup semaphores
  ctx->memorySemaphores = this->memorySemaphores_;

  size_t bytes = count * getDataTypeSize(dtype);
  size_t recvBytes;
  CUdeviceptr recvBasePtr;
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));
  size_t channelOutOffset = (char*)output - (char*)recvBasePtr;
  if (disableChannelCache_) {
    channelOutOffset = 0;
    recvBytes = bytes;
    recvBasePtr = (CUdeviceptr)output;
  }
  ctx->extras.insert(
      {"channel_out_offset", std::make_shared<size_t>(channelOutOffset)});

  // register the memory for the broadcast operation
  mscclpp::RegisteredMemory localMemory = comm->registerMemory(
      (void*)recvBasePtr, recvBytes, mscclpp::Transport::CudaIpc);
  std::vector<mscclpp::RegisteredMemory> remoteMemories =
      setupRemoteMemories(comm, ctx->rank, localMemory);
  ctx->memoryChannels = std::move(
      setupMemoryChannels(this->conns_, ctx->memorySemaphores, remoteMemories,
                          localMemory, nChannelsPerConnection_));
  ctx->memoryChannelDeviceHandles =
      setupMemoryChannelDeviceHandles(ctx->memoryChannels);

  // keep registered memories reference
  ctx->registeredMemories = std::move(remoteMemories);
  ctx->registeredMemories.push_back(localMemory);

  return ctx;
}

mscclpp::AlgorithmCtxKey AllgatherAlgo6::generateAllgatherContextKey(
    void const*, void* output, size_t, mscclpp::DataType) {
  static int tag = 0;
  if (disableChannelCache_) {
    // always return a new key if channel cache is disabled
    return mscclpp::AlgorithmCtxKey{nullptr, nullptr, 0, 0, tag++};
  }
  size_t recvBytes;
  CUdeviceptr recvBasePtr;
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));
  return mscclpp::AlgorithmCtxKey{nullptr, (void*)recvBasePtr, 0, recvBytes, 0};
}

std::shared_ptr<mscclpp::Algorithm> AllgatherAlgo6::build() {
  auto self = shared_from_this();
  return std::make_shared<mscclpp::NativeAlgorithm>(
      "default_allgather6", "allgather",
      [self](std::shared_ptr<mscclpp::Communicator> comm) {
        std::unordered_map<std::string, std::shared_ptr<void>> e;
        self->initialize(comm, e);
      },
      [self](std::shared_ptr<void> const& ctxPtr, void const* input,
             void* output, size_t inputBytes, size_t /*outputBytes*/,
             mscclpp::DataType dtype, mscclpp::ReduceOp /*op*/,
             cudaStream_t stream, int /*nb*/, int /*nt*/,
             std::unordered_map<std::string, uintptr_t> const&) {
        auto ctx = std::static_pointer_cast<mscclpp::AlgorithmCtx>(ctxPtr);
        size_t const count = inputBytes / getDataTypeSize(dtype);
        std::unordered_map<std::string, std::shared_ptr<void>> e;
        return mscclpp::nccl_port::ncclResultToCommResult(
            self->allgatherKernelFunc(ctx, input, output, count, dtype, stream,
                                      e));
      },
      [self](std::shared_ptr<mscclpp::Communicator> comm, void const* input,
             void* output, size_t inputBytes, size_t /*outputBytes*/,
             mscclpp::DataType dtype) {
        size_t const count = inputBytes / getDataTypeSize(dtype);
        return std::static_pointer_cast<void>(
            self->initAllgatherContext(comm, input, output, count, dtype));
      },
      [self](void const* input, void* output, size_t inputBytes,
             size_t /*outputBytes*/, mscclpp::DataType dtype, bool /*sym*/) {
        size_t const count = inputBytes / getDataTypeSize(dtype);
        return self->generateAllgatherContextKey(input, output, count, dtype);
      });
}

AllgatherAlgo8::AllgatherAlgo8(uintptr_t scratchBuffer,
                               size_t scratchBufferSize)
    : scratchPtr_(reinterpret_cast<char*>(scratchBuffer)),
      scratchBufferSize_(scratchBufferSize) {}

void AllgatherAlgo8::initialize(
    std::shared_ptr<mscclpp::Communicator> comm,
    std::unordered_map<std::string, std::shared_ptr<void>>&) {
  this->conns_ = setupConnections(comm);
}

ncclResult_t AllgatherAlgo8::allgatherKernelFunc(
    const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, void const* input,
    void* output, size_t count, mscclpp::DataType dtype, cudaStream_t stream,
    std::unordered_map<std::string, std::shared_ptr<void>>&) {
  int rank = ctx->rank;
  const size_t bytes = count * getDataTypeSize(dtype);
  const size_t nElem = bytes / sizeof(int);
  if ((char*)input == (char*)output + rank * bytes) {
    allgather8<false>
        <<<56, 1024, 0, stream>>>((void*)input, scratchPtr_, (void*)output,
                                  ctx->memoryChannelDeviceHandles.get(), rank,
                                  ctx->nRanksPerNode, ctx->workSize, nElem);
  } else {
    allgather8<true>
        <<<56, 1024, 0, stream>>>((void*)input, scratchPtr_, (void*)output,
                                  ctx->memoryChannelDeviceHandles.get(), rank,
                                  ctx->nRanksPerNode, ctx->workSize, nElem);
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    WARN("AllgatherAlgo8 failed with error %d", err);
    return ncclInternalError;
  }
  return ncclSuccess;
}

std::shared_ptr<mscclpp::AlgorithmCtx> AllgatherAlgo8::initAllgatherContext(
    std::shared_ptr<mscclpp::Communicator> comm, void const* input, void*,
    size_t count, mscclpp::DataType dtype) {
  constexpr int nChannelsPerConnection = 56;

  auto ctx = std::make_shared<mscclpp::AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

  // setup semaphores
  ctx->memorySemaphores = std::move(
      setupMemorySemaphores(comm, this->conns_, nChannelsPerConnection));

  size_t bytes = count * getDataTypeSize(dtype);
  // register the memory for the broadcast operation
  mscclpp::RegisteredMemory localMemory =
      comm->registerMemory((void*)input, bytes, mscclpp::Transport::CudaIpc);
  mscclpp::RegisteredMemory scratchMemory = comm->registerMemory(
      scratchPtr_, scratchBufferSize_, mscclpp::Transport::CudaIpc);
  std::vector<mscclpp::RegisteredMemory> remoteMemories =
      setupRemoteMemories(comm, ctx->rank, scratchMemory);

  // setup channels
  ctx->memoryChannels = std::move(
      setupMemoryChannels(this->conns_, ctx->memorySemaphores, remoteMemories,
                          localMemory, nChannelsPerConnection));
  ctx->memoryChannelDeviceHandles =
      setupMemoryChannelDeviceHandles(ctx->memoryChannels);

  // keep registered memories reference
  ctx->registeredMemories = std::move(remoteMemories);
  ctx->registeredMemories.push_back(localMemory);
  ctx->registeredMemories.push_back(scratchMemory);

  return ctx;
}

mscclpp::AlgorithmCtxKey AllgatherAlgo8::generateAllgatherContextKey(
    void const*, void*, size_t, mscclpp::DataType) {
  // always return same key, non-zero copy algo
  return mscclpp::AlgorithmCtxKey{nullptr, nullptr, 0, 0, 0};
}

std::shared_ptr<mscclpp::Algorithm> AllgatherAlgo8::build() {
  auto self = shared_from_this();
  return std::make_shared<mscclpp::NativeAlgorithm>(
      "default_allgather8", "allgather",
      [self](std::shared_ptr<mscclpp::Communicator> comm) {
        std::unordered_map<std::string, std::shared_ptr<void>> e;
        self->initialize(comm, e);
      },
      [self](std::shared_ptr<void> const& ctxPtr, void const* input,
             void* output, size_t inputBytes, size_t /*outputBytes*/,
             mscclpp::DataType dtype, mscclpp::ReduceOp /*op*/,
             cudaStream_t stream, int /*nb*/, int /*nt*/,
             std::unordered_map<std::string, uintptr_t> const&) {
        auto ctx = std::static_pointer_cast<mscclpp::AlgorithmCtx>(ctxPtr);
        size_t const count = inputBytes / getDataTypeSize(dtype);
        std::unordered_map<std::string, std::shared_ptr<void>> e;
        return mscclpp::nccl_port::ncclResultToCommResult(
            self->allgatherKernelFunc(ctx, input, output, count, dtype, stream,
                                      e));
      },
      [self](std::shared_ptr<mscclpp::Communicator> comm, void const* input,
             void* output, size_t inputBytes, size_t /*outputBytes*/,
             mscclpp::DataType dtype) {
        size_t const count = inputBytes / getDataTypeSize(dtype);
        return std::static_pointer_cast<void>(
            self->initAllgatherContext(comm, input, output, count, dtype));
      },
      [self](void const* input, void* output, size_t inputBytes,
             size_t /*outputBytes*/, mscclpp::DataType dtype, bool /*sym*/) {
        size_t const count = inputBytes / getDataTypeSize(dtype);
        return self->generateAllgatherContextKey(input, output, count, dtype);
      });
}