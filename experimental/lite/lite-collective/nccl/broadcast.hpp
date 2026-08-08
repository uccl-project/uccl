// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef BROADCAST_HPP_
#define BROADCAST_HPP_

#include "algorithm.hpp"
#include "common.hpp"
#include "concurrency_device.hpp"
#include "core.hpp"
#include "executor.hpp"
#include "gpu.hpp"
#include "memory_channel.hpp"
#include "memory_channel_device.hpp"
#include "nccl.h"
#include <cstdint>
#include <memory>

template <bool IsOutOfPlace>
__global__ void __launch_bounds__(1024, 1)
    broadcast6(void* sendbuff, void* scratchbuff, void* recvbuff,
               mscclpp::DeviceHandle<mscclpp::MemoryChannel>* memoryChannels,
               [[maybe_unused]] size_t channelOutOffset, size_t rank,
               [[maybe_unused]] size_t worldSize, size_t root,
               size_t nRanksPerNode, size_t nelemsPerGPU) {
  const size_t nThread = blockDim.x * gridDim.x;
  const size_t nPeer = nRanksPerNode - 1;
  const size_t chanOffset = nPeer * blockIdx.x;

  __shared__ mscclpp::DeviceHandle<mscclpp::MemoryChannel>
      memChans[MAX_NRANKS_PER_NODE - 1];
  if (threadIdx.x < nPeer) {
    memChans[threadIdx.x] = memoryChannels[chanOffset + threadIdx.x];
    memChans[threadIdx.x].relaxedSignal();
    memChans[threadIdx.x].wait();
  }
  __syncthreads();

  const size_t peerRootIdx =
      (root == rank) ? nPeer : ((root < rank) ? root : (root - 1));

  const size_t bytesPerGPU = nelemsPerGPU * sizeof(int);
  const size_t bytes = bytesPerGPU;
  size_t unitBytesPerThread;
  if (bytes * nPeer >= nThread * 64) {
    unitBytesPerThread = 64;
  } else {
    unitBytesPerThread = 16;
  }
  const size_t unitBytesPerBlock = unitBytesPerThread * blockDim.x;
  const size_t unitBytes = unitBytesPerBlock * gridDim.x;
  const size_t nLoop = bytes / unitBytes;

  const size_t maxScratchSizeToUse = (SCRATCH_SIZE - unitBytes);
  const size_t nLoopToSync = (maxScratchSizeToUse / unitBytes) + 1;

  size_t scratchSub = 0;

  // First loop will always fit the scratch size.
  if (nLoop > 0) {
    // First loop unrolling
    const size_t offset = blockIdx.x * unitBytesPerBlock;
    if (rank == root) {
      char* send_ = reinterpret_cast<char*>(sendbuff);
      for (size_t peerIdx = 0; peerIdx < nPeer; peerIdx++) {
        char* dst = reinterpret_cast<char*>(
            memChans[peerIdx].dst_);  // Peer's scratchbuff.
        mscclpp::copy<16, false>(dst + offset, send_ + offset,
                                 unitBytesPerBlock, threadIdx.x, blockDim.x);
        __syncthreads();
        if (threadIdx.x == peerIdx) memChans[peerIdx].signal();
      }
      if constexpr (IsOutOfPlace) {
        char* recv_ = reinterpret_cast<char*>(recvbuff);
        mscclpp::copy<16, false>(recv_ + offset, send_ + offset,
                                 unitBytesPerBlock, threadIdx.x, blockDim.x);
      }

    } else {  // rank != root.
      if (threadIdx.x == peerRootIdx) memChans[peerRootIdx].wait();
      __syncthreads();
      char* recv_ = reinterpret_cast<char*>(recvbuff);
      char* scratch_ = reinterpret_cast<char*>(scratchbuff);  // My scratchbuff.
      mscclpp::copy<16, false>(recv_ + offset, scratch_ + offset,
                               unitBytesPerBlock, threadIdx.x, blockDim.x);
    }
  }

  for (size_t i = 1; i < nLoop; ++i) {
    const size_t offset = blockIdx.x * unitBytesPerBlock + i * unitBytes;
    if (i % nLoopToSync == 0) {  // Sync to reuse scratch buff
      scratchSub = -i * unitBytes;
      deviceSyncer.sync(gridDim.x);
      if (threadIdx.x < nPeer) {
        memChans[threadIdx.x].relaxedSignal();
        memChans[threadIdx.x].wait();
      }
    }
    if (rank == root) {
      char* send_ = reinterpret_cast<char*>(sendbuff);
      for (size_t peerIdx = 0; peerIdx < nPeer; peerIdx++) {
        char* dst = reinterpret_cast<char*>(
            memChans[peerIdx].dst_);  // Peer's scratchbuff.
        mscclpp::copy<16, false>(dst + offset + scratchSub, send_ + offset,
                                 unitBytesPerBlock, threadIdx.x, blockDim.x);
        __syncthreads();
        if (threadIdx.x == peerIdx) memChans[peerIdx].signal();
      }
      if constexpr (IsOutOfPlace) {
        char* recv_ = reinterpret_cast<char*>(recvbuff);
        mscclpp::copy<16, false>(recv_ + offset, send_ + offset,
                                 unitBytesPerBlock, threadIdx.x, blockDim.x);
      }
    } else {  // rank != root.
      if (threadIdx.x == peerRootIdx) memChans[peerRootIdx].wait();
      __syncthreads();
      char* recv_ = reinterpret_cast<char*>(recvbuff);
      char* scratch_ = reinterpret_cast<char*>(scratchbuff);  // My scratchbuff.
      mscclpp::copy<16, false>(recv_ + offset, scratch_ + offset + scratchSub,
                               unitBytesPerBlock, threadIdx.x, blockDim.x);
    }
  }

  // Remainder loop will also fit the scratch buff since we subtract unitBytes
  // from SCRATCH_SIZE.
  if (bytes % unitBytes > 0) {  // remainder.
    const size_t offset = blockIdx.x * unitBytesPerBlock + nLoop * unitBytes;
    const size_t remainBytes = (offset < bytes) ? (bytes - offset) : 0;
    if (remainBytes > 0) {
      if (rank == root) {
        char* send_ = reinterpret_cast<char*>(sendbuff);
        for (size_t peerIdx = 0; peerIdx < nPeer; peerIdx++) {
          char* dst = reinterpret_cast<char*>(
              memChans[peerIdx].dst_);  // Peer's scratchbuff.
          mscclpp::copy<16, true>(dst + offset + scratchSub, send_ + offset,
                                  remainBytes, threadIdx.x, blockDim.x);
          __syncthreads();
          if (threadIdx.x == peerIdx) memChans[peerIdx].signal();
        }
        if constexpr (IsOutOfPlace) {
          char* recv_ = reinterpret_cast<char*>(recvbuff);
          mscclpp::copy<16, true>(recv_ + offset, send_ + offset, remainBytes,
                                  threadIdx.x, blockDim.x);
        }
      } else {  // rank != root.
        if (threadIdx.x == peerRootIdx) memChans[peerRootIdx].wait();
        __syncthreads();
        char* recv_ = reinterpret_cast<char*>(recvbuff);
        char* scratch_ =
            reinterpret_cast<char*>(scratchbuff);  // My scratchbuff.
        mscclpp::copy<16, true>(recv_ + offset, scratch_ + offset + scratchSub,
                                remainBytes, threadIdx.x, blockDim.x);
      }
    }  // remainBytes > 0.
  }

  deviceSyncer.sync(gridDim.x);

  if (threadIdx.x < nPeer) {
    memChans[threadIdx.x].relaxedSignal();
    memChans[threadIdx.x].wait();
  }
}

template <bool IsOutOfPlace, typename T>
cudaError_t broadcast(
    T* buff, T* scratch, T* resultBuff,
    mscclpp::DeviceHandle<mscclpp::MemoryChannel>* memoryChannels,
    size_t channelOutOffset, int rank, int nRanksPerNode, int root,
    int worldSize, size_t nelems, cudaStream_t stream) {
  int nBlocks = 7;
  broadcast6<IsOutOfPlace><<<nBlocks, 1024, 0, stream>>>(
      (void*)buff, (void*)scratch, (void*)resultBuff, memoryChannels,
      channelOutOffset, rank, worldSize, root, nRanksPerNode,
      nelems * sizeof(T) / sizeof(int));
  return cudaGetLastError();
}

class BroadcastAlgo6 : public mscclpp::AlgorithmBuilder,
                       public std::enable_shared_from_this<BroadcastAlgo6> {
 public:
  BroadcastAlgo6(uintptr_t scratchBuffer, size_t scratchBufferSize);
  std::shared_ptr<mscclpp::Algorithm> build() override;

 private:
  void initialize(
      std::shared_ptr<mscclpp::Communicator> comm,
      std::unordered_map<std::string, std::shared_ptr<void>>& extras);
  ncclResult_t broadcastKernelFunc(
      const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, void const* input,
      void* output, size_t count, mscclpp::DataType dtype, cudaStream_t stream,
      std::unordered_map<std::string, std::shared_ptr<void>>& extras);

  std::shared_ptr<mscclpp::AlgorithmCtx> initBroadcastContext(
      std::shared_ptr<mscclpp::Communicator> comm, void const*, void* output,
      size_t, mscclpp::DataType);
  mscclpp::AlgorithmCtxKey generateBroadcastContextKey(void const*, void*,
                                                       size_t,
                                                       mscclpp::DataType);

  std::vector<mscclpp::Connection> conns_;
  char* scratchPtr_{nullptr};
  size_t scratchMemSize_{0};
};

#endif  // BROADCAST_HPP_
