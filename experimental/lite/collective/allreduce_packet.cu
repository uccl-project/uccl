// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "algorithm.hpp"
#include "allreduce_packet.hpp"
#include "collective_utils.hpp"
#include "common.hpp"
#include "debug.h"

namespace mscclpp {
namespace collective {

template <ReduceOp OpType, typename T>
__global__ void __launch_bounds__(1024, 1) allreducePacket(
    T* buff, T* scratch, T* resultBuff,
    mscclpp::DeviceHandle<mscclpp::MemoryChannel>* memoryChannels,
    size_t channelDataOffset, size_t scratchBufferSize, int rank,
    int nRanksPerNode, int worldSize, size_t nelems, void* flags,
    uint32_t flagBufferSize, uint32_t numScratchBuff
#if defined(ENABLE_NPKIT)
    ,
    NpKitEventCollectContext* npKitEventCollectContexts,
    uint64_t* cpuTimestamp) {
#else
) {
#endif
  // This version of allreduce only works for single nodes
  if (worldSize != nRanksPerNode) return;

#if defined(ENABLE_NPKIT)
  extern __shared__ int4 NpkitSharedMem[];
  NpKitEvent* event_buffer = (NpKitEvent*)((char*)NpkitSharedMem);
  uint64_t event_buffer_head = 0;
#if defined(ENABLE_NPKIT_EVENT_KERNEL_ALLREDUCE_ENTRY) && \
    defined(ENABLE_NPKIT_EVENT_KERNEL_ALLREDUCE_EXIT)
  uint64_t npkit_timestamp_entry = 0;
  if (threadIdx.x == 0) {
    npkit_timestamp_entry = NPKIT_GET_GPU_TIMESTAMP();
  }
#endif
#endif
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_CPU)
#if defined(MSCCLPP_DEVICE_HIP)
  NpKit::CollectGpuEventShm(
      NPKIT_EVENT_TIME_SYNC_CPU, 0, 0,
      NPKIT_LOAD_CPU_TIMESTAMP_PER_BLOCK(cpuTimestamp, blockIdx.x),
#else
  NpKit::CollectGpuEventShm(NPKIT_EVENT_TIME_SYNC_CPU, 0, 0, *cpuTimestamp,
#endif
      event_buffer, &event_buffer_head);
#endif
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_GPU)
  NpKit::CollectGpuEventShm(NPKIT_EVENT_TIME_SYNC_GPU, 0, 0,
                            NPKIT_GET_GPU_TIMESTAMP(), event_buffer,
                            &event_buffer_head);
#endif

  if (sizeof(T) == 2 || sizeof(T) == 1)
    nelems = (nelems * sizeof(T) + sizeof(T)) / sizeof(int);
  else
    nelems = nelems / (sizeof(int) / sizeof(T));

  int const nPeers = nRanksPerNode - 1;
  const size_t nPkts = nelems / 2;

  uint32_t flag = ((uint32_t*)flags)[blockIdx.x];
  size_t channelScratchOffset =
      (flag % numScratchBuff) ? scratchBufferSize / numScratchBuff : 0;

  int nelemsPerRank = nelems / worldSize;
  if ((nelemsPerRank % 2))
    nelemsPerRank = (nelemsPerRank * sizeof(T) + sizeof(T)) / sizeof(T);

  int const nPktsPerRank = nelemsPerRank / 2;
  // thread block & channel info
  int const nBlocksPerPeer = gridDim.x / nPeers;
  int const localBlockIdx = blockIdx.x % nBlocksPerPeer;
  int const peerIdx = blockIdx.x / nBlocksPerPeer;
  int const remoteRank = peerIdx < rank ? peerIdx : peerIdx + 1;
  int const tid = threadIdx.x + localBlockIdx * blockDim.x;
  void* scratchBuff = (void*)((char*)scratch + channelScratchOffset);
  size_t scratchOffset =
      channelScratchOffset + rank * nPktsPerRank * sizeof(mscclpp::LLPacket);
  size_t scratchResultOffset =
      channelScratchOffset + 2 * nPkts * sizeof(mscclpp::LLPacket);
  size_t srcOffset =
      remoteRank * nelemsPerRank * sizeof(int) + channelDataOffset;

  uint2* src = (uint2*)((char*)buff + rank * nelemsPerRank * sizeof(int));
  uint2* dst = (uint2*)((char*)resultBuff + rank * nelemsPerRank * sizeof(int));

  // Put channels into shared memory, read channel info from global memory is
  // unexpectable slow.
  __shared__ mscclpp::DeviceHandle<mscclpp::MemoryChannel>
      channels[MAX_NRANKS_PER_NODE - 1];
  int const lid = tid % WARP_SIZE;
  if (lid < nPeers) {
    channels[lid] = memoryChannels[lid];
  }
  __syncwarp();
  // step 1: write to scratch buffer
  channels[peerIdx].putPackets<mscclpp::LLPacket>(
      scratchOffset, srcOffset, nelemsPerRank * sizeof(int), tid,
      blockDim.x * nBlocksPerPeer, flag);
  // step 2: get data from scratch buffer, reduce data and write result to
  // remote scratch buffer
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPktsPerRank;
       idx += blockDim.x * gridDim.x) {
    uint2 data = src[idx];
    for (int index = 0; index < nPeers; index++) {
      int const remoteRank = index < rank ? index : index + 1;
      mscclpp::LLPacket* dstPkt =
          (mscclpp::LLPacket*)scratchBuff + remoteRank * nPktsPerRank;
      uint2 val = dstPkt[idx].read(flag);
      data.x = cal_vector<T, OpType>(val.x, data.x);
      data.y = cal_vector<T, OpType>(val.y, data.y);
    }

    dst[idx].x = data.x;
    dst[idx].y = data.y;

    mscclpp::LLPacket packet;
    packet.data1 = data.x;
    packet.flag1 = flag;
    packet.data2 = data.y;
    packet.flag2 = flag;
    size_t offset = scratchResultOffset / sizeof(mscclpp::LLPacket) +
                    (idx + rank * nPktsPerRank);
    for (int index = 0; index < nPeers; index++) {
      channels[index].write(offset, packet);
    }
  }
  // step 3: get data result from scratch buffer
  mscclpp::LLPacket* dstPkt =
      (mscclpp::LLPacket*)((char*)scratch + scratchResultOffset);
  int const dstOffset = remoteRank * nPktsPerRank;
  uint2* result =
      (uint2*)((char*)resultBuff + remoteRank * nelemsPerRank * sizeof(int));
  for (int idx = threadIdx.x + localBlockIdx * blockDim.x; idx < nPktsPerRank;
       idx += blockDim.x * nBlocksPerPeer) {
    uint2 data = dstPkt[idx + dstOffset].read(flag, -1);
    result[idx].x = data.x;
    result[idx].y = data.y;
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    ((uint32_t*)flags)[blockIdx.x] = flag + 1;
  }
  if (blockIdx.x == 0 && (threadIdx.x > gridDim.x - 1) &&
      (threadIdx.x < flagBufferSize / sizeof(uint32_t))) {
    ((uint32_t*)flags)[threadIdx.x] = flag + 1;
  }
#if defined(ENABLE_NPKIT) &&                              \
    defined(ENABLE_NPKIT_EVENT_KERNEL_ALLREDUCE_ENTRY) && \
    defined(ENABLE_NPKIT_EVENT_KERNEL_ALLREDUCE_EXIT)
  NpKit::CollectGpuEventShm(NPKIT_EVENT_KERNEL_ALLREDUCE_ENTRY, 0, 0,
                            npkit_timestamp_entry, event_buffer,
                            &event_buffer_head);
  NpKit::CollectGpuEventShm(NPKIT_EVENT_KERNEL_ALLREDUCE_EXIT, 0, 0,
                            NPKIT_GET_GPU_TIMESTAMP(), event_buffer,
                            &event_buffer_head);
#endif
#if defined(ENABLE_NPKIT)
  NpKit::StoreGpuEventShm(npKitEventCollectContexts, event_buffer,
                          event_buffer_head);
#endif
}

template <ReduceOp OpType, typename T>
struct PacketAdapter {
  static cudaError_t call(void const* buff, void* scratch, void* resultBuff,
                          void* memoryChannels, void*,
                          DeviceHandle<SwitchChannel>*,
                          DeviceHandle<SwitchChannel>*, size_t channelInOffset,
                          size_t, size_t scratchBufferSize, int rank,
                          int nRanksPerNode, int worldSize, size_t inputSize,
                          cudaStream_t stream, void* flags,
                          uint32_t flagBufferSize, uint32_t numScratchBuff,
                          int nBlocks = 0, int nThreadsPerBlock = 0) {
    using ChannelType = DeviceHandle<MemoryChannel>;
    const size_t nelems = inputSize / sizeof(T);
    // Optimize the number of blocks to be multiple of (worldSize - 1)
    nBlocks = nBlocks / (worldSize - 1) * (worldSize - 1);
#if defined(ENABLE_NPKIT)
    size_t sharedMemSize = sizeof(NpKitEvent) * NPKIT_SHM_NUM_EVENTS;
    allreducePacket<OpType>
        <<<nBlocks, nThreadsPerBlock, sharedMemSize, stream>>>(
            (T*)buff, (T*)scratch, (T*)resultBuff, (ChannelType*)memoryChannels,
            channelInOffset, scratchBufferSize, rank, nRanksPerNode, worldSize,
            nelems, flags, flagBufferSize, numScratchBuff,
            NpKit::GetGpuEventCollectContexts(), NpKit::GetCpuTimestamp());
#else
    allreducePacket<OpType><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
        (T*)buff, (T*)scratch, (T*)resultBuff, (ChannelType*)memoryChannels,
        channelInOffset, scratchBufferSize, rank, nRanksPerNode, worldSize,
        nelems, flags, flagBufferSize, numScratchBuff);
#endif
    return cudaGetLastError();
  }
};

inline std::pair<int, int> getDefaultBlockNumAndThreadNum(
    size_t inputSize, int nRanksPerNode, int worldSize,
    [[maybe_unused]] DataType dtype) {
  int nBlocks = (nRanksPerNode - 1) * 4;
  int nThreadsPerBlock = 1024;
  if (inputSize >= 32768) {
    nBlocks = (worldSize - 1) * 8;
    nThreadsPerBlock = (inputSize <= 153600) ? 512 : 1024;
  }
#if defined(__HIP_PLATFORM_AMD__)
  if (dtype == DataType::FLOAT16) {
    // Half-specific tuning for 32KB-256KB range
    if (inputSize < (64 << 10)) {
      nThreadsPerBlock = 64;
    } else if (inputSize >= (64 << 10) && inputSize <= (256 << 10)) {
      nThreadsPerBlock = 128;
    }
  }

#if defined(__FP8_TYPES_EXIST__)
  // FP8-specific tuning for 32KB-256KB range
  if (dtype == DataType::FLOAT8_E4M3 || dtype == DataType::FLOAT8_E5M2) {
    if (inputSize < (64 << 10)) {
      nThreadsPerBlock = 64;
    } else if (inputSize >= (64 << 10) && inputSize <= (128 << 10)) {
      nThreadsPerBlock = 128;
    } else if (inputSize >= (128 << 10) && inputSize <= (256 << 10)) {
      nThreadsPerBlock = 256;
    }
  }
#endif
#endif
  return {nBlocks, nThreadsPerBlock};
}

void AllreducePacket::initialize(std::shared_ptr<Communicator> comm) {
  conns_ = setupConnections(comm);
  memorySemaphores_ = setupMemorySemaphores(comm, conns_, maxBlockNum_);
  RegisteredMemory scratchMemory = comm->registerMemory(
      scratchBuffer_, scratchBufferSize_, Transport::CudaIpc);
  registeredMemories_ =
      setupRemoteMemories(comm, comm->bootstrap()->getRank(), scratchMemory);
  registeredMemories_.push_back(scratchMemory);
}

CommResult AllreducePacket::allreduceKernelFunc(
    const std::shared_ptr<void> ctx_void, void const* input, void* output,
    size_t inputSize, [[maybe_unused]] DataType dtype, ReduceOp op,
    cudaStream_t stream, int nBlocks, int nThreadsPerBlock,
    std::unordered_map<std::string, uintptr_t> const&) {
  auto ctx = std::static_pointer_cast<AlgorithmCtx>(ctx_void);
  std::pair<int, int> blockAndThreadNum = {nBlocks, nThreadsPerBlock};
  if (blockAndThreadNum.first == 0 || blockAndThreadNum.second == 0) {
    blockAndThreadNum = getDefaultBlockNumAndThreadNum(
        inputSize, ctx->workSize, ctx->nRanksPerNode, dtype);
  }

  size_t sendBytes;
  CUdeviceptr sendBasePtr;
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  size_t channelInOffset = (char*)input - (char*)sendBasePtr;

  AllreduceFunc allreduce = dispatch<PacketAdapter>(op, dtype);
  if (!allreduce) {
    WARN("Unsupported operation or data type for allreduce: op=%d, dtype=%d",
         op, static_cast<int>(dtype));
    return CommResult::CommInvalidArgument;
  }
  cudaError_t error = allreduce(
      input, this->scratchBuffer_, output,
      ctx->memoryChannelDeviceHandles.get(), nullptr, nullptr, nullptr,
      channelInOffset, 0, this->scratchBufferSize_, ctx->rank,
      ctx->nRanksPerNode, ctx->workSize, inputSize, stream, (void*)flagBuffer_,
      (uint32_t)flagBufferSize_, this->nSegmentsForScratchBuffer_,
      blockAndThreadNum.first, blockAndThreadNum.second);
  if (error != cudaSuccess) {
    WARN("AllreducePacket failed with error: %s", cudaGetErrorString(error));
    return CommResult::CommUnhandledCudaError;
  }
  return CommResult::CommSuccess;
}

std::shared_ptr<void> AllreducePacket::initAllreduceContext(
    std::shared_ptr<Communicator> comm, void const* input, void*, size_t,
    DataType) {
  auto ctx = std::make_shared<AlgorithmCtx>();
  int const nChannelsPerConnection = maxBlockNum_;
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();
  ctx->memorySemaphores = this->memorySemaphores_;
  ctx->registeredMemories = this->registeredMemories_;
  ctx->registeredMemories
      .pop_back();  // remove the local memory from previous context

  size_t sendBytes;
  CUdeviceptr sendBasePtr;
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  RegisteredMemory localMemory =
      comm->registerMemory((void*)sendBasePtr, sendBytes, Transport::CudaIpc);

  // setup channels
  ctx->memoryChannels = setupMemoryChannels(
      this->conns_, ctx->memorySemaphores, ctx->registeredMemories, localMemory,
      nChannelsPerConnection);
  ctx->memoryChannelDeviceHandles =
      setupMemoryChannelDeviceHandles(ctx->memoryChannels);
  ctx->registeredMemories.emplace_back(localMemory);
  return ctx;
}

AlgorithmCtxKey AllreducePacket::generateAllreduceContextKey(void const* input,
                                                             void*, size_t,
                                                             DataType, bool) {
  size_t sendBytes;
  CUdeviceptr sendBasePtr;
  MSCCLPP_CUTHROW(
      cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  return AlgorithmCtxKey{(void*)sendBasePtr, nullptr, sendBytes, 0, 0};
}

std::shared_ptr<Algorithm> AllreducePacket::build() {
  auto self = std::make_shared<AllreducePacket>(
      reinterpret_cast<uintptr_t>(scratchBuffer_), scratchBufferSize_,
      flagBuffer_, flagBufferSize_);
  return std::make_shared<NativeAlgorithm>(
      "default_allreduce_packet", "allreduce",
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