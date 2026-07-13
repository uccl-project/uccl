// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "allreduce_nvls_packet.hpp"
#include "collective_utils.hpp"
#include "common.hpp"
#include "debug.h"

namespace mscclpp {
namespace collective {

template <ReduceOp OpType, typename T>
__global__ void __launch_bounds__(1024, 1) allreduceNvlsPacket(
    [[maybe_unused]] const T* input, [[maybe_unused]] T* scratch,
    [[maybe_unused]] T* output,
    [[maybe_unused]] mscclpp::DeviceHandle<mscclpp::SwitchChannel>* multicast,
    [[maybe_unused]] size_t nelems, [[maybe_unused]] size_t scratchBufferSize,
    [[maybe_unused]] int rank, [[maybe_unused]] int worldSize,
    [[maybe_unused]] void* flags, [[maybe_unused]] uint32_t flagBufferSize) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t flag = ((uint32_t*)flags)[blockIdx.x];
  size_t scratchBaseOffset = (flag % 2) ? scratchBufferSize / 2 : 0;
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t nPktPerRank =
      nelems / worldSize / (sizeof(mscclpp::LL8Packet::Payload) / sizeof(T));
  mscclpp::LL8Packet* multiPkt =
      (mscclpp::LL8Packet*)((char*)multicast->mcPtr + scratchBaseOffset) +
      rank * worldSize * nPktPerRank;
  uint* src = (uint*)(input);
  uint* dst = (uint*)(output);
  mscclpp::LL8Packet* scratchPkt =
      (mscclpp::LL8Packet*)((char*)scratch + scratchBaseOffset);
  for (uint32_t i = tid; i < nPktPerRank * worldSize;
       i += blockDim.x * gridDim.x) {
    mscclpp::LL8Packet pkt(src[i], flag);
    mscclpp::SwitchChannelDeviceHandle::multimemStore(*(mscclpp::f32x2*)(&pkt),
                                                      multiPkt + i);
  }
  for (uint32_t i = tid; i < nPktPerRank * worldSize;
       i += blockDim.x * gridDim.x) {
    uint data = src[i];
    for (int peer = 0; peer < worldSize; peer++) {
      if (peer == rank) {
        continue;
      }
      uint val = scratchPkt[peer * worldSize * nPktPerRank + i].read(flag);
      data = cal_vector<T, OpType>(data, val);
    }
    dst[i] = data;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    ((uint32_t*)flags)[blockIdx.x] = flag + 1;
  }
  // update other flags in-case using different number of blocks in next launch
  if (blockIdx.x == 0 && (threadIdx.x > gridDim.x - 1) &&
      (threadIdx.x < flagBufferSize / sizeof(uint32_t))) {
    ((uint32_t*)flags)[threadIdx.x] = flag + 1;
  }
#endif
}

inline std::pair<int, int> getDefaultBlockNumAndThreadNum(size_t inputSize) {
  int blockNum = 8;
  int threadNum = 1024;
  if (inputSize <= (1 << 13)) {
    blockNum = 4;
    threadNum = 512;
  }
  return {blockNum, threadNum};
}

template <ReduceOp OpType, typename T>
struct AllreduceNvlsPacketAdapter {
  static cudaError_t call(void const* input, void* scratch, void* output, void*,
                          void*, DeviceHandle<SwitchChannel>* nvlsChannels,
                          DeviceHandle<SwitchChannel>*, size_t, size_t,
                          size_t scratchBufferSize, int rank, int,
                          int worldSize, size_t inputSize, cudaStream_t stream,
                          void* flags, uint32_t flagBufferSize, uint32_t,
                          int nBlocks, int nThreadsPerBlock) {
    allreduceNvlsPacket<OpType, T><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
        (const T*)input, (T*)scratch, (T*)output, nvlsChannels,
        inputSize / sizeof(T), scratchBufferSize, rank, worldSize, flags,
        flagBufferSize);
    return cudaGetLastError();
  }
};

void AllreduceNvlsPacket::initialize(std::shared_ptr<Communicator> comm) {
  int nSwitchChannels = 1;
  this->nvlsConnections_ =
      setupNvlsConnections(comm, nvlsBufferSize_, nSwitchChannels);
}

AlgorithmCtxKey AllreduceNvlsPacket::generateAllreduceContextKey(void const*,
                                                                 void*, size_t,
                                                                 DataType,
                                                                 bool) {
  return AlgorithmCtxKey{nullptr, nullptr, 0, 0, 0};
}

std::shared_ptr<void> AllreduceNvlsPacket::initAllreduceContext(
    std::shared_ptr<Communicator> comm, void const*, void*, size_t, DataType) {
  auto ctx = std::make_shared<AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

  // setup channels
  int nSwitchChannels = 1;
  ctx->switchChannels =
      setupNvlsChannels(this->nvlsConnections_, this->scratchBuffer_,
                        this->scratchBufferSize_, nSwitchChannels);
  ctx->switchChannelDeviceHandles =
      setupNvlsChannelDeviceHandles(ctx->switchChannels);
  return ctx;
}

CommResult AllreduceNvlsPacket::allreduceKernelFunc(
    const std::shared_ptr<void> ctx_void, void const* input, void* output,
    size_t inputSize, mscclpp::DataType dtype, ReduceOp op, cudaStream_t stream,
    int nBlocks, int nThreadsPerBlock,
    std::unordered_map<std::string, uintptr_t> const&) {
  auto ctx = std::static_pointer_cast<AlgorithmCtx>(ctx_void);
  std::pair<int, int> blockAndThreadNum = {nBlocks, nThreadsPerBlock};
  if (blockAndThreadNum.first == 0 || blockAndThreadNum.second == 0) {
    blockAndThreadNum = getDefaultBlockNumAndThreadNum(inputSize);
  }
  if (blockAndThreadNum.first > maxBlockNum_) {
    WARN("Block number %d exceeds the maximum limit %d",
         blockAndThreadNum.first, maxBlockNum_);
    return CommResult::CommInvalidArgument;
  }
  AllreduceFunc allreduce = dispatch<AllreduceNvlsPacketAdapter>(op, dtype);
  if (!allreduce) {
    WARN("Unsupported operation or data type for allreduce, dtype=%d",
         static_cast<int>(dtype));
    return CommResult::CommInvalidArgument;
  }
  cudaError_t error = allreduce(
      input, this->scratchBuffer_, output, nullptr, nullptr,
      ctx->switchChannelDeviceHandles.get(), nullptr, 0, 0,
      this->scratchBufferSize_, ctx->rank, ctx->nRanksPerNode, ctx->workSize,
      inputSize, stream, (void*)flagBuffer_, (uint32_t)flagBufferSize_, 0,
      blockAndThreadNum.first, blockAndThreadNum.second);
  if (error != cudaSuccess) {
    WARN("AllreduceNvlsPacket failed with error: %s",
         cudaGetErrorString(error));
    return CommResult::CommUnhandledCudaError;
  }
  return CommResult::CommSuccess;
}

std::shared_ptr<mscclpp::Algorithm> AllreduceNvlsPacket::build() {
  auto self = std::make_shared<AllreduceNvlsPacket>(
      (uintptr_t)scratchBuffer_, scratchBufferSize_, flagBuffer_,
      flagBufferSize_);
  return std::make_shared<mscclpp::NativeAlgorithm>(
      "default_allreduce_nvls_packet", "allreduce",
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