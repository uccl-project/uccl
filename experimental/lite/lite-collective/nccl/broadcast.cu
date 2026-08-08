// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "algorithm.hpp"
#include "broadcast.hpp"
#include "common.hpp"
#include "datatype_conversion.hpp"
#include "gpu_utils.hpp"
#include "nccl.h"

BroadcastAlgo6::BroadcastAlgo6(uintptr_t scratchBuffer,
                               size_t scratchBufferSize)
    : scratchPtr_(reinterpret_cast<char*>(scratchBuffer)),
      scratchMemSize_(scratchBufferSize) {}

void BroadcastAlgo6::initialize(
    std::shared_ptr<mscclpp::Communicator> comm,
    std::unordered_map<std::string, std::shared_ptr<void>>&) {
  this->conns_ = setupConnections(comm);
}

ncclResult_t BroadcastAlgo6::broadcastKernelFunc(
    const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, void const* input,
    void* output, size_t count, mscclpp::DataType dtype, cudaStream_t stream,
    std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
  int root = *(int*)extras.at("root").get();
  const size_t elemSize = getDataTypeSize(dtype);
  cudaError_t err;
  if (input == output) {
    err = broadcast<false>((int*)input, (int*)scratchPtr_, (int*)output,
                           ctx->memoryChannelDeviceHandles.get(), 0, ctx->rank,
                           ctx->nRanksPerNode, root, ctx->workSize,
                           count * elemSize / sizeof(int), stream);
  } else {
    err = broadcast<true>((int*)input, (int*)scratchPtr_, (int*)output,
                          ctx->memoryChannelDeviceHandles.get(), 0, ctx->rank,
                          ctx->nRanksPerNode, root, ctx->workSize,
                          count * elemSize / sizeof(int), stream);
  }
  if (err != cudaSuccess) {
    return ncclInternalError;
  }
  return ncclSuccess;
}

std::shared_ptr<mscclpp::AlgorithmCtx> BroadcastAlgo6::initBroadcastContext(
    std::shared_ptr<mscclpp::Communicator> comm, void const*, void* output,
    size_t, mscclpp::DataType) {
  constexpr int nChannelsPerConnection = 8;

  auto ctx = std::make_shared<mscclpp::AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

  // setup semaphores
  ctx->memorySemaphores =
      setupMemorySemaphores(comm, this->conns_, nChannelsPerConnection);
  size_t recvBytes;
  CUdeviceptr recvBasePtr;
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));

  // register the memory for the broadcast operation
  mscclpp::RegisteredMemory localMemory = comm->registerMemory(
      (void*)recvBasePtr, recvBytes, mscclpp::Transport::CudaIpc);
  mscclpp::RegisteredMemory localScratchMemory = comm->registerMemory(
      scratchPtr_, scratchMemSize_, mscclpp::Transport::CudaIpc);
  std::vector<mscclpp::RegisteredMemory> remoteMemories =
      setupRemoteMemories(comm, ctx->rank, localScratchMemory);
  ctx->memoryChannels =
      setupMemoryChannels(this->conns_, ctx->memorySemaphores, remoteMemories,
                          localMemory, nChannelsPerConnection);
  ctx->memoryChannelDeviceHandles =
      setupMemoryChannelDeviceHandles(ctx->memoryChannels);

  // keep registered memories reference
  ctx->registeredMemories = std::move(remoteMemories);
  ctx->registeredMemories.push_back(localMemory);
  ctx->registeredMemories.push_back(localScratchMemory);

  return ctx;
}

mscclpp::AlgorithmCtxKey BroadcastAlgo6::generateBroadcastContextKey(
    void const*, void*, size_t, mscclpp::DataType) {
  // always use same context
  return mscclpp::AlgorithmCtxKey{nullptr, nullptr, 0, 0, 0};
}

std::shared_ptr<mscclpp::Algorithm> BroadcastAlgo6::build() {
  auto self = shared_from_this();
  return std::make_shared<mscclpp::NativeAlgorithm>(
      "default_broadcast6", "broadcast",
      [self](std::shared_ptr<mscclpp::Communicator> comm) {
        std::unordered_map<std::string, std::shared_ptr<void>> e;
        self->initialize(comm, e);
      },
      [self](std::shared_ptr<void> const& ctxPtr, void const* input,
             void* output, size_t inputBytes, size_t /*outputBytes*/,
             mscclpp::DataType dtype, mscclpp::ReduceOp /*op*/,
             cudaStream_t stream, int /*nb*/, int /*nt*/,
             std::unordered_map<std::string, uintptr_t> const& xextras) {
        auto ctx = std::static_pointer_cast<mscclpp::AlgorithmCtx>(ctxPtr);
        size_t const count = inputBytes / getDataTypeSize(dtype);
        auto lex = mscclpp::nccl_port::legacyExtrasWithRoot(
            *reinterpret_cast<int const*>(xextras.at("root")));
        return mscclpp::nccl_port::ncclResultToCommResult(
            self->broadcastKernelFunc(ctx, input, output, count, dtype, stream,
                                      lex));
      },
      [self](std::shared_ptr<mscclpp::Communicator> comm, void const* input,
             void* output, size_t inputBytes, size_t /*outputBytes*/,
             mscclpp::DataType dtype) {
        size_t const count = inputBytes / getDataTypeSize(dtype);
        return std::static_pointer_cast<void>(
            self->initBroadcastContext(comm, input, output, count, dtype));
      },
      [self](void const* input, void* output, size_t inputBytes,
             size_t /*outputBytes*/, mscclpp::DataType dtype, bool /*sym*/) {
        size_t const count = inputBytes / getDataTypeSize(dtype);
        return self->generateBroadcastContextKey(input, output, count, dtype);
      });
}
