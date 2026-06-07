// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "reducescatter_rs.hpp"
#include "collective_utils.hpp"
#include "common.hpp"
#include "logger.hpp"

namespace mscclpp {
namespace collective {

size_t reduceScatterDataTypeSize(DataType dtype) {
  switch (dtype) {
    case DataType::UINT8:
    case DataType::FLOAT8_E4M3:
    case DataType::FLOAT8_E5M2:
      return 1;
    case DataType::FLOAT16:
    case DataType::BFLOAT16:
      return 2;
    case DataType::INT32:
    case DataType::UINT32:
    case DataType::FLOAT32:
      return 4;
    default:
      return 0;
  }
}

template <typename T>
__device__ __forceinline__ int4 rsLoadVecAt(T const* buff, size_t elemOffset,
                                            size_t nelems) {
  constexpr size_t elemsPerInt4 = sizeof(int4) / sizeof(T);
  if (elemOffset % elemsPerInt4 == 0 && elemOffset + elemsPerInt4 <= nelems) {
    return reinterpret_cast<int4 const*>(buff)[elemOffset / elemsPerInt4];
  }

  union {
    int4 i;
    T t[elemsPerInt4];
  } vec;
  vec.i = make_int4(0, 0, 0, 0);
  for (size_t i = 0; i < elemsPerInt4 && elemOffset + i < nelems; ++i) {
    vec.t[i] = buff[elemOffset + i];
  }
  return vec.i;
}

template <typename T>
__device__ __forceinline__ void rsStoreVecAt(T* buff, size_t elemOffset,
                                             int4 value, size_t nelems) {
  constexpr size_t elemsPerInt4 = sizeof(int4) / sizeof(T);
  if (elemOffset % elemsPerInt4 == 0 && elemOffset + elemsPerInt4 <= nelems) {
    reinterpret_cast<int4*>(buff)[elemOffset / elemsPerInt4] = value;
    return;
  }

  union {
    int4 i;
    T t[elemsPerInt4];
  } vec;
  vec.i = value;
  for (size_t i = 0; i < elemsPerInt4 && elemOffset + i < nelems; ++i) {
    buff[elemOffset + i] = vec.t[i];
  }
}

template <ReduceOp OpType, typename T>
__global__ void __launch_bounds__(1024, 1)
    reduceScatterRs(T const* input, T* scratch, T* output,
                    DeviceHandle<BaseMemoryChannel>* memoryChannels,
                    void* remoteMemories, int rank, int nRanksPerNode,
                    int worldSize, size_t nelems) {
  if (worldSize != nRanksPerNode) return;

  const int nPeers = nRanksPerNode - 1;
  if (nPeers <= 0) return;

  const size_t elemsPerRank = nelems / static_cast<size_t>(nRanksPerNode);
  constexpr size_t elemsPerInt4 = sizeof(int4) / sizeof(T);
  const size_t nInt4PerRank =
      (elemsPerRank + elemsPerInt4 - 1) / elemsPerInt4;
  const size_t nThreads =
      static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x);
  const size_t tid =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
      static_cast<size_t>(threadIdx.x);
  auto* scratch4 = reinterpret_cast<int4*>(scratch);
  auto** remoteScratch = reinterpret_cast<void**>(remoteMemories);
  DeviceHandle<BaseMemoryChannel>* memoryChannelsLocal =
      memoryChannels + static_cast<size_t>(blockIdx.x) * nPeers;

  for (size_t vecIdx = tid; vecIdx < nInt4PerRank; vecIdx += nThreads) {
    for (int targetRank = 0; targetRank < nRanksPerNode; ++targetRank) {
      if (targetRank == rank) continue;
      const int peerIdx = targetRank < rank ? targetRank : targetRank - 1;
      const size_t srcElemOffset =
          static_cast<size_t>(targetRank) * elemsPerRank +
          vecIdx * elemsPerInt4;
      const size_t dstVecIdx =
          static_cast<size_t>(rank) * nInt4PerRank + vecIdx;
      int4 value = rsLoadVecAt(input, srcElemOffset, nelems);
      mscclpp::write<int4>(remoteScratch[peerIdx], dstVecIdx, value);
    }
  }

  __syncthreads();
  if (threadIdx.x < static_cast<uint32_t>(nPeers)) {
    memoryChannelsLocal[threadIdx.x].signal();
    memoryChannelsLocal[threadIdx.x].wait();
  }
  __syncthreads();

  const size_t myShardBase =
      static_cast<size_t>(rank) * elemsPerRank;
  for (size_t vecIdx = tid; vecIdx < nInt4PerRank; vecIdx += nThreads) {
    int4 acc = rsLoadVecAt(input, myShardBase + vecIdx * elemsPerInt4, nelems);
    for (int peer = 0; peer < nPeers; ++peer) {
      int remoteRank = peer < rank ? peer : peer + 1;
      int4 value =
          scratch4[static_cast<size_t>(remoteRank) * nInt4PerRank + vecIdx];
      acc = cal_vector<T, OpType>(value, acc);
    }
    rsStoreVecAt(output, vecIdx * elemsPerInt4, acc, elemsPerRank);
  }

  __syncthreads();
  if (threadIdx.x < static_cast<uint32_t>(nPeers)) {
    memoryChannelsLocal[threadIdx.x].signal();
    memoryChannelsLocal[threadIdx.x].wait();
  }
}

template <ReduceOp OpType, typename T>
struct ReduceScatterRsAdapter {
  static cudaError_t call(void const* input, void* scratch, void* output,
                          void* memoryChannels, void* remoteMemories,
                          DeviceHandle<SwitchChannel>*,
                          DeviceHandle<SwitchChannel>*, size_t, size_t, size_t,
                          int rank, int nRanksPerNode, int worldSize,
                          size_t inputSize, cudaStream_t stream, void*,
                          uint32_t, uint32_t, int nBlocks,
                          int nThreadsPerBlock) {
    using ChannelType = DeviceHandle<BaseMemoryChannel>;
    if (nBlocks == 0 || nThreadsPerBlock == 0) {
      nBlocks = 64;
      nThreadsPerBlock = 1024;
    }
    const size_t nelems = inputSize / sizeof(T);
    reduceScatterRs<OpType, T>
        <<<nBlocks, nThreadsPerBlock, 0, stream>>>(
            static_cast<T const*>(input), static_cast<T*>(scratch),
            static_cast<T*>(output), static_cast<ChannelType*>(memoryChannels),
            remoteMemories, rank, nRanksPerNode, worldSize, nelems);
    return cudaGetLastError();
  }
};

void ReduceScatterRs::initialize(std::shared_ptr<Communicator> comm) {
  conns_ = setupConnections(comm);
  nChannelsPerConnection_ = 64;
  scratchSemaphores_ =
      setupMemorySemaphores(comm, conns_, nChannelsPerConnection_);
  RegisteredMemory localMemory = comm->registerMemory(
      scratchBuffer_, scratchBufferSize_, Transport::CudaIpc);
  remoteScratchMemories_ =
      setupRemoteMemories(comm, comm->bootstrap()->getRank(), localMemory);
  localScratchMemory_ = std::move(localMemory);

  baseChannels_ =
      setupBaseMemoryChannels(conns_, scratchSemaphores_,
                              nChannelsPerConnection_);
  baseMemoryChannelHandles_ = setupBaseMemoryChannelDeviceHandles(baseChannels_);

  std::vector<void*> remoteMemoryHandles;
  remoteMemoryHandles.reserve(remoteScratchMemories_.size());
  for (auto const& remoteMemory : remoteScratchMemories_) {
    remoteMemoryHandles.push_back(remoteMemory.data());
  }
  remoteMemoryHandles_ =
      detail::gpuCallocShared<void*>(remoteMemoryHandles.size());
  gpuMemcpy(remoteMemoryHandles_.get(), remoteMemoryHandles.data(),
            remoteMemoryHandles.size(), cudaMemcpyHostToDevice);
}

CommResult ReduceScatterRs::reduceScatterKernelFunc(
    const std::shared_ptr<void> ctx, void const* input, void* output,
    size_t inputSize, size_t, DataType dtype, ReduceOp op, cudaStream_t stream,
    int nBlocks, int nThreadsPerBlock,
    std::unordered_map<std::string, uintptr_t> const&) {
  auto algoCtx = std::static_pointer_cast<AlgorithmCtx>(ctx);
  if (algoCtx->workSize != algoCtx->nRanksPerNode) {
    return CommResult::CommInvalidUsage;
  }
  size_t typeSize = reduceScatterDataTypeSize(dtype);
  if (typeSize == 0) {
    WARN(ALGO, "Unsupported data type for reducescatter: dtype=",
         static_cast<int>(dtype));
    return CommResult::CommInvalidArgument;
  }
  size_t worldSize = static_cast<size_t>(algoCtx->workSize);
  if (inputSize % worldSize != 0 || (inputSize / worldSize) % typeSize != 0) {
    WARN(ALGO, "ReduceScatter input size ", inputSize,
         " is not divisible by world size ", algoCtx->workSize);
    return CommResult::CommInvalidArgument;
  }
  size_t shardBytes = inputSize / worldSize;
  size_t paddedShardBytes =
      ((shardBytes + sizeof(int4) - 1) / sizeof(int4)) * sizeof(int4);
  if (paddedShardBytes > scratchBufferSize_ / worldSize) {
    WARN(ALGO, "ReduceScatter requires ", paddedShardBytes * worldSize,
         " scratch bytes, but only ", scratchBufferSize_, " are available");
    return CommResult::CommInvalidArgument;
  }

  AllreduceFunc reduceScatter = dispatch<ReduceScatterRsAdapter>(op, dtype);
  if (!reduceScatter) {
    WARN(ALGO, "Unsupported operation or data type for reducescatter: op=",
         static_cast<int>(op), ", dtype=", static_cast<int>(dtype));
    return CommResult::CommInvalidArgument;
  }

  cudaError_t error = reduceScatter(
      input, scratchBuffer_, output, baseMemoryChannelHandles_.get(),
      remoteMemoryHandles_.get(), nullptr, nullptr, 0, 0, 0, algoCtx->rank,
      algoCtx->nRanksPerNode, algoCtx->workSize, inputSize, stream, nullptr, 0,
      0, nBlocks, nThreadsPerBlock);
  if (error != cudaSuccess) {
    WARN(ALGO, "ReduceScatter kernel launch failed with error: ",
         cudaGetErrorString(error));
    return CommResult::CommUnhandledCudaError;
  }
  return CommResult::CommSuccess;
}

std::shared_ptr<void> ReduceScatterRs::initReduceScatterContext(
    std::shared_ptr<Communicator> comm, void const*, void*, size_t, size_t,
    DataType) {
  auto ctx = std::make_shared<AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();
  ctx->memorySemaphores = scratchSemaphores_;
  ctx->registeredMemories = remoteScratchMemories_;
  return ctx;
}

AlgorithmCtxKey ReduceScatterRs::generateReduceScatterContextKey(
    void const*, void*, size_t, size_t, DataType, bool) {
  return AlgorithmCtxKey{nullptr, nullptr, 0, 0, 0};
}

std::shared_ptr<Algorithm> ReduceScatterRs::build() {
  auto self = std::make_shared<ReduceScatterRs>(
      (uintptr_t)scratchBuffer_, scratchBufferSize_);
  return std::make_shared<NativeAlgorithm>(
      "default_reducescatter_rs", "reducescatter",
      [self](std::shared_ptr<Communicator> comm) { self->initialize(comm); },
      [self](const std::shared_ptr<void> ctx, void const* input, void* output,
             size_t inputSize, size_t outputSize, DataType dtype, ReduceOp op,
             cudaStream_t stream, int nBlocks, int nThreadsPerBlock,
             std::unordered_map<std::string, uintptr_t> const& extras) {
        return self->reduceScatterKernelFunc(ctx, input, output, inputSize,
                                             outputSize, dtype, op, stream,
                                             nBlocks, nThreadsPerBlock, extras);
      },
      [self](std::shared_ptr<Communicator> comm, void const* input, void* output,
             size_t inputSize, size_t outputSize, DataType dtype) {
        return self->initReduceScatterContext(comm, input, output, inputSize,
                                              outputSize, dtype);
      },
      [self](void const* input, void* output, size_t inputSize,
             size_t outputSize, DataType dtype, bool symmetricMemory) {
        return self->generateReduceScatterContextKey(
            input, output, inputSize, outputSize, dtype, symmetricMemory);
      });
}

}  // namespace collective
}  // namespace mscclpp
