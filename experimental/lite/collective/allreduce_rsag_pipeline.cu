// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "allreduce_rsag_pipeline.hpp"
#include "collective_utils.hpp"
#include "common.hpp"
#include "logger.hpp"

namespace mscclpp {
namespace collective {
constexpr int MAX_NBLOCKS_FOR_PUT = 32;
constexpr int MAX_NBLOCKS_FOR_RECV = 32;
constexpr int MAX_NBLOCKS_FOR_REDUCE = 64;
constexpr int REDUCE_COPY_RATIO = 2;
__device__ DeviceSemaphore semaphoreForSend[MAX_NBLOCKS_FOR_REDUCE];
__device__ DeviceSemaphore semaphoreForRecv[MAX_NBLOCKS_FOR_REDUCE];
__device__ DeviceSemaphore semaphoreForReduce[MAX_NBLOCKS_FOR_REDUCE];

// TODO: move it to a common header file
template <typename T>
__device__ __forceinline__ int4 loadVec(const T* buff, size_t i,
                                        size_t nelems) {
  constexpr size_t ElemsPerInt4 = sizeof(int4) / sizeof(T);
  size_t offset = i * ElemsPerInt4;
  if (offset + ElemsPerInt4 <= nelems) {
    return reinterpret_cast<int4 const*>(buff)[i];
  } else {
    union {
      int4 i;
      T t[ElemsPerInt4];
    } vec;
    vec.i = make_int4(0, 0, 0, 0);
    for (size_t j = 0; j < ElemsPerInt4 && offset + j < nelems; ++j) {
      vec.t[j] = buff[offset + j];
    }
    return vec.i;
  }
}

template <typename T>
__device__ __forceinline__ void storeVec(T* buff, size_t i, int4 val,
                                         size_t nelems) {
  constexpr size_t ElemsPerInt4 = sizeof(int4) / sizeof(T);
  size_t offset = i * ElemsPerInt4;
  if (offset + ElemsPerInt4 <= nelems) {
    reinterpret_cast<int4*>(buff)[i] = val;
  } else {
    union {
      int4 i;
      T t[ElemsPerInt4];
    } vec;
    vec.i = val;
    for (size_t j = 0; j < ElemsPerInt4 && offset + j < nelems; ++j) {
      buff[offset + j] = vec.t[j];
    }
  }
}

// Pipelined Reduce-Scatter + All-Gather (RSAG) allreduce.
//
// This is a pipelined variant of the basic RSAG allreduce that overlaps
// communication and computation by splitting the data into chunks processed
// across multiple iterations. Three groups of thread blocks run concurrently
// with different roles, synchronized via device semaphores:
//
//   PUT blocks  — Read local input chunks and write them into peers' scratch
//                 buffers via remote memory handles (CudaIpc).
//
//   REDUCE blocks — After a signal/wait barrier confirming PUT completion,
//                   reduce the local chunk with data received from all peers
//                   in the scratch buffer. Write the reduced result to both
//                   the local output and peers' scratch (for the AG phase).
//
//   RECV blocks — After a signal/wait barrier confirming REDUCE completion,
//                 copy other ranks' reduced chunks from scratch into the
//                 local result buffer, completing the all-gather.
//
// Pipelining is achieved by using a circular scratch buffer (pipelineDepth
// stages). PUT blocks wait on a semaphore before reusing a scratch slot,
// allowing the next iteration's PUT to overlap with the current iteration's
// REDUCE and RECV. Each REDUCE block handles a subset of the PUT block's
// data (controlled by REDUCE_COPY_RATIO), enabling finer-grained overlap.
//
// Data is processed in int4-sized (16-byte) units with vectorized load/store
// helpers that handle tail elements.

template <ReduceOp OpType, typename T>
__global__ void __launch_bounds__(1024, 1)
    allreduceRsAgPipeline(T* buff, T* scratch, T* resultBuff,
                          DeviceHandle<BaseMemoryChannel>* memoryChannels,
                          DeviceHandle<SwitchChannel>* switchChannels,
                          void* remoteMemories, int rank, int nRanksPerNode,
                          int worldSize, size_t nelems, size_t scratchSize,
                          uint32_t nblocksForPut, uint32_t nblocksForReduce,
                          uint32_t nblocksForRecv) {
  uint32_t bid = blockIdx.x;
  constexpr uint32_t nStepsPerIter = 4;
  uint32_t nInt4 = (nelems * sizeof(T) + sizeof(int4) - 1) / sizeof(int4);
  uint32_t nInt4PerIter = nblocksForReduce * blockDim.x * nStepsPerIter;
  const uint32_t chunkSize = nInt4PerIter * worldSize;
  uint32_t nIters = (nInt4 + chunkSize - 1) / chunkSize;
  uint32_t nPeers = nRanksPerNode - 1;
  int4* scratch4 = reinterpret_cast<int4*>((char*)scratch);
  const uint32_t scratchIterStride = 2 * chunkSize;  // one for AS, one for AG
  const uint32_t pipelineDepth = scratchSize / sizeof(int4) / scratchIterStride;
  assert(pipelineDepth >= 1);

  if (bid < nblocksForPut) {
    if (threadIdx.x == 0) {
      semaphoreForSend[bid].set(pipelineDepth);
    }
    for (uint32_t iter = 0; iter < nIters; iter++) {
      if (threadIdx.x == 0) {
        semaphoreForSend[bid].acquire();
      }
      __syncthreads();
      uint32_t threadIdInPut = bid * blockDim.x + threadIdx.x;
      for (uint32_t peer = 0; peer < nPeers; peer++) {
        int remoteRankId = (rank + peer + 1) % nRanksPerNode;
        int peerId = remoteRankId < rank ? remoteRankId : remoteRankId - 1;
        // Read chunk[remoteRankId] from local buff, write to peer's
        // scratch[rank] (sender's slot)
        uint32_t srcOffset = iter * chunkSize + remoteRankId * nInt4PerIter;
        uint32_t dstOffset =
            (iter % pipelineDepth) * scratchIterStride + rank * nInt4PerIter;
        int4 tmp[nStepsPerIter * REDUCE_COPY_RATIO];
#pragma unroll
        for (uint32_t step = 0; step < nStepsPerIter * REDUCE_COPY_RATIO;
             step++) {
          uint32_t offset =
              srcOffset + threadIdInPut + step * blockDim.x * nblocksForPut;
          tmp[step] = loadVec(buff, offset, nelems);
        }
#pragma unroll
        for (uint32_t step = 0; step < nStepsPerIter * REDUCE_COPY_RATIO;
             step++) {
          uint32_t offset =
              dstOffset + threadIdInPut + step * blockDim.x * nblocksForPut;
          mscclpp::write<int4>(((void**)remoteMemories)[peerId], offset,
                               tmp[step]);
        }
      }
      __syncthreads();
      if (threadIdx.x < REDUCE_COPY_RATIO) {
        semaphoreForReduce[bid * REDUCE_COPY_RATIO + threadIdx.x].release();
      }
    }
  } else if (bid < nblocksForPut + nblocksForReduce) {
    uint32_t bidInReduce = bid - nblocksForPut;
    DeviceHandle<BaseMemoryChannel>* localMemoryChannels =
        memoryChannels + bidInReduce * nPeers;
    // Map REDUCE blocks to PUT blocks: REDUCE blocks 0,1 handle PUT block 0's
    // data
    uint32_t putBlockId = bidInReduce / REDUCE_COPY_RATIO;
    uint32_t subBlockId = bidInReduce % REDUCE_COPY_RATIO;
    for (uint32_t iter = 0; iter < nIters; iter++) {
      if (threadIdx.x == 0) {
        semaphoreForReduce[bidInReduce].acquire();
      }
      uint32_t baseOffset = (iter % pipelineDepth) * scratchIterStride;
      uint32_t baseSrcOffset = iter * chunkSize;

      // Use same thread mapping as PUT: putBlockId * blockDim.x + threadIdx.x
      uint32_t threadIdInPut = putBlockId * blockDim.x + threadIdx.x;
      __syncthreads();
      if (threadIdx.x < nPeers) {
        localMemoryChannels[threadIdx.x].signal();
        localMemoryChannels[threadIdx.x].wait();
      }
      __syncthreads();
#pragma unroll nStepsPerIter
      for (uint32_t step = 0; step < nStepsPerIter; step++) {
        // Map to PUT's step pattern: each REDUCE block handles nStepsPerIter
        // steps subBlockId determines which subset of the REDUCE_COPY_RATIO *
        // nStepsPerIter steps
        uint32_t putStep = subBlockId * nStepsPerIter + step;
        uint32_t myChunkOffset = baseSrcOffset + rank * nInt4PerIter +
                                 threadIdInPut +
                                 putStep * blockDim.x * nblocksForPut;
        int4 tmp = loadVec(buff, myChunkOffset, nelems);
        // Add data from each peer's slot in scratch (peer sent their
        // chunk[rank] to our scratch[peer])
        for (uint32_t peer = 0; peer < nPeers; peer++) {
          int remoteRankId = (rank + peer + 1) % nRanksPerNode;
          uint32_t peerSlotOffset = baseOffset + remoteRankId * nInt4PerIter +
                                    threadIdInPut +
                                    putStep * blockDim.x * nblocksForPut;
          int4 data = scratch4[peerSlotOffset];
          tmp = cal_vector<T, OpType>(data, tmp);
        }
        storeVec(resultBuff, myChunkOffset, tmp, nelems);
        // Broadcast reduced result to all peers' scratch at SCATTER_AG_OFFSET +
        // rank * nInt4PerIter
        uint32_t dstOffset = baseOffset + chunkSize + rank * nInt4PerIter +
                             threadIdInPut +
                             putStep * blockDim.x * nblocksForPut;
        for (uint32_t i = 0; i < nPeers; i++) {
          int peerIdx = (rank + i + 1) % nRanksPerNode;
          int index = peerIdx < rank ? peerIdx : peerIdx - 1;
          mscclpp::write<int4>(((void**)remoteMemories)[index], dstOffset, tmp);
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        semaphoreForRecv[bidInReduce].release();
      }
    }
  } else if (bid < nblocksForPut + nblocksForReduce + nblocksForRecv) {
    uint32_t bidInRecv = bid - nblocksForPut - nblocksForReduce;
    DeviceHandle<BaseMemoryChannel>* localMemoryChannels =
        memoryChannels + (nblocksForReduce + bidInRecv) * nPeers;
    for (uint32_t iter = 0; iter < nIters; iter++) {
      if (threadIdx.x < REDUCE_COPY_RATIO) {
        semaphoreForRecv[bidInRecv * REDUCE_COPY_RATIO + threadIdx.x].acquire();
      }
      uint32_t baseOffset = scratchIterStride * (iter % pipelineDepth);
      uint32_t baseDstOffset = chunkSize * iter;
      int threadIdInRecv = bidInRecv * blockDim.x + threadIdx.x;
      __syncthreads();
      if (threadIdx.x < nPeers) {
        localMemoryChannels[threadIdx.x].signal();
        localMemoryChannels[threadIdx.x].wait();
      }
      __syncthreads();
      // Copy other ranks' reduced chunks from scratch to result
      for (uint32_t peer = 0; peer < nPeers; peer++) {
        int remoteRankId = (rank + peer + 1) % nRanksPerNode;
        for (uint32_t step = 0; step < nStepsPerIter * REDUCE_COPY_RATIO;
             step++) {
          uint32_t offset = baseOffset + chunkSize +
                            remoteRankId * nInt4PerIter + threadIdInRecv +
                            step * blockDim.x * nblocksForRecv;
          uint32_t dstOffset = baseDstOffset + remoteRankId * nInt4PerIter +
                               threadIdInRecv +
                               step * blockDim.x * nblocksForRecv;
          storeVec(resultBuff, dstOffset, scratch4[offset], nelems);
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        semaphoreForSend[bidInRecv].release();
      }
    }
  }
}

template <ReduceOp OpType, typename T>
struct AllreduceRsAgPipelineAdapter {
  static cudaError_t call(void const* input, void* scratch, void* output,
                          void* memoryChannels, void* remoteMemories,
                          DeviceHandle<SwitchChannel>* switchChannel,
                          DeviceHandle<SwitchChannel>*, size_t, size_t,
                          size_t scratchSize, int rank, int nRanksPerNode,
                          int worldSize, size_t inputSize, cudaStream_t stream,
                          void*, uint32_t, uint32_t, int nBlocks,
                          int nThreadsPerBlock) {
    using ChannelType = DeviceHandle<BaseMemoryChannel>;
    size_t nelems = inputSize / sizeof(T);
    uint32_t nblocksForPut = MAX_NBLOCKS_FOR_PUT;
    uint32_t nblocksForReduce = MAX_NBLOCKS_FOR_REDUCE;
    uint32_t nblocksForRecv = MAX_NBLOCKS_FOR_RECV;
    int maxNblocks = nblocksForPut + nblocksForReduce + nblocksForRecv;
    if (nBlocks == 0 || nThreadsPerBlock == 0) {
      nThreadsPerBlock = 1024;
      nBlocks = maxNblocks;
    } else {
      nBlocks = nBlocks / (REDUCE_COPY_RATIO + 2) * (REDUCE_COPY_RATIO + 2);
      if (nBlocks > maxNblocks) {
        WARN(ALGO,
             "The number of blocks is too large for the allreduce pipeline "
             "algorithm, reducing it to ",
             maxNblocks);
        nBlocks = maxNblocks;
      }
      nblocksForPut = nBlocks / (REDUCE_COPY_RATIO + 2);
      nblocksForReduce = nblocksForPut * REDUCE_COPY_RATIO;
      nblocksForRecv = nblocksForPut;
    }
    allreduceRsAgPipeline<OpType, T><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
        (T*)input, (T*)scratch, (T*)output, (ChannelType*)memoryChannels,
        switchChannel, remoteMemories, rank, nRanksPerNode, worldSize, nelems,
        scratchSize, nblocksForPut, nblocksForReduce, nblocksForRecv);
    return cudaGetLastError();
  }
};

void AllreduceRsAgPipeline::initialize(std::shared_ptr<Communicator> comm) {
  this->conns_ = setupConnections(comm);
  nChannelsPerConnection_ = MAX_NBLOCKS_FOR_REDUCE + MAX_NBLOCKS_FOR_RECV;
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

CommResult AllreduceRsAgPipeline::allreduceKernelFunc(
    const std::shared_ptr<void> ctx, void const* input, void* output,
    size_t inputSize, DataType dtype, ReduceOp op, cudaStream_t stream,
    int nBlocks, int nThreadsPerBlock,
    std::unordered_map<std::string, uintptr_t> const&) {
  auto algoCtx = std::static_pointer_cast<AlgorithmCtx>(ctx);
  AllreduceFunc allreduce = dispatch<AllreduceRsAgPipelineAdapter>(op, dtype);
  if (!allreduce) {
    WARN(ALGO, "Unsupported operation or data type for allreduce: op=",
         static_cast<int>(op), ", dtype=", static_cast<int>(dtype));
    return CommResult::CommInvalidArgument;
  }
  std::pair<int, int> numBlocksAndThreads = {nBlocks, nThreadsPerBlock};
  cudaError_t error = allreduce(
      input, this->scratchBuffer_, output,
      this->baseMemoryChannelHandles_.get(), this->remoteMemoryHandles_.get(),
      nullptr, nullptr, 0, 0, this->scratchBufferSize_, algoCtx->rank,
      algoCtx->nRanksPerNode, algoCtx->workSize, inputSize, stream, nullptr, 0,
      0, numBlocksAndThreads.first, numBlocksAndThreads.second);
  if (error != cudaSuccess) {
    WARN(ALGO, "Allreduce kernel launch failed with error: ",
         cudaGetErrorString(error));
    return CommResult::CommUnhandledCudaError;
  }
  return CommResult::CommSuccess;
}

AlgorithmCtxKey AllreduceRsAgPipeline::generateAllreduceContextKey(
    void const*, void*, size_t, DataType, bool) {
  return AlgorithmCtxKey{nullptr, nullptr, 0, 0, 0};
}

std::shared_ptr<void> AllreduceRsAgPipeline::initAllreduceContext(
    std::shared_ptr<Communicator> comm, void const*, void*, size_t, DataType) {
  auto ctx = std::make_shared<AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

  ctx->memorySemaphores = this->scratchSemaphores_;
  ctx->registeredMemories = this->remoteScratchMemories_;
  return ctx;
}

std::shared_ptr<Algorithm> AllreduceRsAgPipeline::build() {
  auto self = std::make_shared<AllreduceRsAgPipeline>((uintptr_t)scratchBuffer_,
                                                      scratchBufferSize_);
  return std::make_shared<NativeAlgorithm>(
      "default_allreduce_rsag_pipeline", "allreduce",
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
