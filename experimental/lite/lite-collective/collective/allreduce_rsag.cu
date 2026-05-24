// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "allreduce_rsag.hpp"
#include "collective_utils.hpp"
#include "common.hpp"
#include "logger.hpp"

namespace mscclpp {
namespace collective {

// Allreduce using the Reduce-Scatter + All-Gather (RSAG) pattern.
//
// This algorithm performs allreduce in three phases over intra-node peers
// connected via CudaIpc memory channels:
//
//   1. Scatter: Each rank copies its input data into a scratch buffer, then
//      signals peers and waits for all peers to do the same.
//
//   2. Reduce-Scatter: Each rank reduces its assigned chunk by reading the
//      corresponding chunks from all peers' scratch buffers (via remote memory
//      handles) and applying the reduction op. The reduced result is written
//      back to both the local result buffer and peers' scratch buffers.
//
//   3. All-Gather: After a second signal/wait barrier, each rank copies the
//      reduced chunks produced by other ranks from the scratch buffer into its
//      result buffer, completing the allreduce.
//
// Data is processed in int4-sized (16-byte) units for coalesced memory access,
// with special handling for any remainder elements at the tail.
template <ReduceOp OpType, typename T>
__global__ void __launch_bounds__(1024, 1)
    allreduceRsAg(T* buff, T* scratch, T* resultBuff,
                  DeviceHandle<BaseMemoryChannel>* memoryChannels,
                  DeviceHandle<SwitchChannel>* switchChannels,
                  void* remoteMemories, int rank, int nRanksPerNode,
                  int worldSize, size_t nelems) {
  int blockId = blockIdx.x;
  uint32_t nPeers = nRanksPerNode - 1;

  assert((uintptr_t)buff % sizeof(int4) == 0);
  assert((uintptr_t)resultBuff % sizeof(int4) == 0);

  constexpr uint32_t nelemsPerInt4 = sizeof(int4) / sizeof(T);
  uint32_t alignedNelems =
      ((nelems + nRanksPerNode - 1) / nRanksPerNode + nelemsPerInt4 - 1) /
      nelemsPerInt4 * nelemsPerInt4 * nRanksPerNode;
  uint32_t nelemsPerRank = alignedNelems / nRanksPerNode;
  uint32_t nInt4PerRank = nelemsPerRank / nelemsPerInt4;
  uint32_t lastInt4Index = nelems / nelemsPerInt4;
  uint32_t remainder = nelems % nelemsPerInt4;

  int4* scratch4 = reinterpret_cast<int4*>((char*)scratch);
  int4* resultBuff4 = reinterpret_cast<int4*>((char*)resultBuff);
  int4* buff4 = reinterpret_cast<int4*>((char*)buff);
  DeviceHandle<BaseMemoryChannel>* memoryChannelsLocal =
      memoryChannels + blockId * nPeers;

  uint32_t nInt4PerBlock = nInt4PerRank / gridDim.x;
  uint32_t remainderForBlock = nInt4PerRank % gridDim.x;
  uint32_t offset4 = blockId * nInt4PerBlock;
  if (blockId == (int)(gridDim.x - 1)) {
    nInt4PerBlock += remainderForBlock;
  }
  if (nInt4PerBlock == 0) return;
  uint32_t nInt4ForCopy = nInt4PerBlock * nRanksPerNode;

  for (uint32_t idx = threadIdx.x; idx < nInt4ForCopy; idx += blockDim.x) {
    int rankIdx = idx / nInt4PerBlock;
    uint32_t offsetIdx =
        rankIdx * nInt4PerRank + offset4 + (idx % nInt4PerBlock);
    if (offsetIdx > lastInt4Index) continue;
    if (offsetIdx == lastInt4Index && remainder != 0) {
      for (uint32_t i = 0; i < remainder; i++) {
        ((T*)&scratch4[offsetIdx])[i] = ((T*)&buff4[offsetIdx])[i];
      }
      continue;
    }
    scratch4[offsetIdx] = buff4[offsetIdx];
  }
  __syncthreads();
  if (threadIdx.x < nPeers) {
    memoryChannelsLocal[threadIdx.x].signal();
    memoryChannelsLocal[threadIdx.x].wait();
  }
  __syncthreads();
  for (uint32_t idx = threadIdx.x; idx < nInt4PerBlock; idx += blockDim.x) {
    uint32_t offset = idx + offset4 + rank * nInt4PerRank;
    if (offset > lastInt4Index) continue;
    int4 tmp = scratch4[offset];
    for (uint32_t i = 0; i < nPeers; i++) {
      int rankIdx = (rank + i + 1) % nRanksPerNode;
      int peerIdx = rankIdx < rank ? rankIdx : rankIdx - 1;
      int4 data =
          mscclpp::read<int4>(((void**)remoteMemories)[peerIdx], offset);
      tmp = cal_vector<T, OpType>(data, tmp);
    }
    for (uint32_t i = 0; i < nPeers; i++) {
      int rankIdx = (rank + i + 1) % nRanksPerNode;
      int peerIdx = rankIdx < rank ? rankIdx : rankIdx - 1;
      mscclpp::write<int4>(((void**)remoteMemories)[peerIdx], offset, tmp);
    }
    if (offset == lastInt4Index && remainder != 0) {
      for (uint32_t i = 0; i < remainder; i++) {
        ((T*)&resultBuff4[offset])[i] = ((T*)&tmp)[i];
      }
      continue;
    }
    resultBuff4[offset] = tmp;
  }
  __syncthreads();
  if (threadIdx.x < nPeers) {
    memoryChannelsLocal[threadIdx.x].signal();
    memoryChannelsLocal[threadIdx.x].wait();
  }
  __syncthreads();
  for (uint32_t idx = threadIdx.x; idx < nInt4ForCopy; idx += blockDim.x) {
    int rankIdx = idx / nInt4PerBlock;
    if (rankIdx == rank) continue;
    uint32_t offsetIdx =
        rankIdx * nInt4PerRank + offset4 + (idx % nInt4PerBlock);
    if (offsetIdx > lastInt4Index) continue;
    if (offsetIdx == lastInt4Index && remainder != 0) {
      for (uint32_t i = 0; i < remainder; i++) {
        ((T*)&resultBuff4[offsetIdx])[i] = ((T*)&scratch4[offsetIdx])[i];
      }
      continue;
    }
    resultBuff4[offsetIdx] = scratch4[offsetIdx];
  }
}

template <ReduceOp OpType, typename T>
struct AllreduceRsAgAdapter {
  static cudaError_t call(void const* input, void* scratch, void* output,
                          void* memoryChannels, void* remoteMemories,
                          DeviceHandle<SwitchChannel>* switchChannel,
                          DeviceHandle<SwitchChannel>*, size_t, size_t, size_t,
                          int rank, int nRanksPerNode, int worldSize,
                          size_t inputSize, cudaStream_t stream, void*,
                          uint32_t, uint32_t, int nBlocks,
                          int nThreadsPerBlock) {
    using ChannelType = DeviceHandle<BaseMemoryChannel>;
    size_t nelems = inputSize / sizeof(T);
    if (nBlocks == 0 || nThreadsPerBlock == 0) {
      nThreadsPerBlock = 1024;
      nBlocks = 64;
    }
    allreduceRsAg<OpType, T><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
        (T*)input, (T*)scratch, (T*)output, (ChannelType*)memoryChannels,
        switchChannel, remoteMemories, rank, nRanksPerNode, worldSize, nelems);
    return cudaGetLastError();
  }
};

void AllreduceRsAg::initialize(std::shared_ptr<Communicator> comm) {
  this->conns_ = setupConnections(comm);
  nChannelsPerConnection_ = 64;
  comm_ = comm;
  // setup semaphores
  this->scratchSemaphores_ =
      setupMemorySemaphores(comm, this->conns_, nChannelsPerConnection_);
  RegisteredMemory localMemory = comm->registerMemory(
      scratchBuffer_, scratchBufferSize_, Transport::CudaIpc);
  this->remoteScratchMemories_ =
      setupRemoteMemories(comm, comm->bootstrap()->getRank(), localMemory);
  localScratchMemory_ = std::move(localMemory);

  this->baseChannels_ = setupBaseMemoryChannels(
      this->conns_, this->scratchSemaphores_, nChannelsPerConnection_);
  this->baseMemoryChannelHandles_ =
      setupBaseMemoryChannelDeviceHandles(baseChannels_);
  std::vector<void*> remoteMemoryHandles;
  for (auto const& remoteMemory : this->remoteScratchMemories_) {
    remoteMemoryHandles.push_back(remoteMemory.data());
  }
  this->remoteMemoryHandles_ =
      detail::gpuCallocShared<void*>(remoteMemoryHandles.size());
  gpuMemcpy(this->remoteMemoryHandles_.get(), remoteMemoryHandles.data(),
            remoteMemoryHandles.size(), cudaMemcpyHostToDevice);
}

CommResult AllreduceRsAg::allreduceKernelFunc(
    const std::shared_ptr<void> ctx, void const* input, void* output,
    size_t inputSize, DataType dtype, ReduceOp op, cudaStream_t stream,
    int nBlocks, int nThreadsPerBlock,
    std::unordered_map<std::string, uintptr_t> const&) {
  auto algoCtx = std::static_pointer_cast<AlgorithmCtx>(ctx);
  AllreduceFunc allreduce = dispatch<AllreduceRsAgAdapter>(op, dtype);
  if (!allreduce) {
    WARN(ALGO, "Unsupported operation or data type for allreduce: op=",
         static_cast<int>(op), ", dtype=", static_cast<int>(dtype));
    return CommResult::CommInvalidArgument;
  }
  if (inputSize > this->scratchBufferSize_) {
    WARN(ALGO, "Input size ", inputSize, " exceeds scratch buffer size ",
         this->scratchBufferSize_);
    return CommResult::CommInvalidArgument;
  }
  std::pair<int, int> numBlocksAndThreads = {nBlocks, nThreadsPerBlock};
  cudaError_t error = allreduce(
      input, this->scratchBuffer_, output,
      this->baseMemoryChannelHandles_.get(), this->remoteMemoryHandles_.get(),
      nullptr, nullptr, 0, 0, 0, algoCtx->rank, algoCtx->nRanksPerNode,
      algoCtx->workSize, inputSize, stream, nullptr, 0, 0,
      numBlocksAndThreads.first, numBlocksAndThreads.second);
  if (error != cudaSuccess) {
    WARN(ALGO, "Allreduce kernel launch failed with error: ",
         cudaGetErrorString(error));
    return CommResult::CommUnhandledCudaError;
  }
  return CommResult::CommSuccess;
}

AlgorithmCtxKey AllreduceRsAg::generateAllreduceContextKey(void const*, void*,
                                                           size_t, DataType,
                                                           bool) {
  return AlgorithmCtxKey{nullptr, nullptr, 0, 0, 0};
}

std::shared_ptr<void> AllreduceRsAg::initAllreduceContext(
    std::shared_ptr<Communicator> comm, void const*, void*, size_t, DataType) {
  auto ctx = std::make_shared<AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

  ctx->memorySemaphores = this->scratchSemaphores_;
  ctx->registeredMemories = this->remoteScratchMemories_;
  return ctx;
}

std::shared_ptr<Algorithm> AllreduceRsAg::build() {
  auto self = std::make_shared<AllreduceRsAg>((uintptr_t)scratchBuffer_,
                                              scratchBufferSize_);
  return std::make_shared<NativeAlgorithm>(
      "default_allreduce_rsag", "allreduce",
      [self](std::shared_ptr<mscclpp::Communicator> comm) {
        self->initialize(comm);
      },
      [self](const std::shared_ptr<void> ctx, void const* input, void* output,
             size_t inputSize, [[maybe_unused]] size_t outputSize,
             DataType dtype, ReduceOp op, cudaStream_t stream, int nBlocks,
             int nThreadsPerBlock,
             std::unordered_map<std::string, uintptr_t> const& extras)
          -> CommResult {
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
