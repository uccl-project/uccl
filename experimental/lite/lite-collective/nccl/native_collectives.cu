#include "native_collectives.hpp"
#include "debug.h"
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

namespace mscclpp {
namespace nccl {
namespace {

ncclResult_t cudaResult(cudaError_t error, char const* operation) {
  if (error == cudaSuccess) return ncclSuccess;
  WARN("%s failed with CUDA error: %s", operation, cudaGetErrorString(error));
  if (error == cudaErrorInvalidValue) return ncclInvalidArgument;
  return ncclUnhandledCudaError;
}

template <typename T>
__global__ void reduceRowsKernel(char const* rows, void* output, size_t count,
                                 size_t rowBytes, int nRows, ncclRedOp_t op) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;

  auto const* first = reinterpret_cast<T const*>(rows);
  T acc = first[idx];
  for (int row = 1; row < nRows; ++row) {
    auto const* shard =
        reinterpret_cast<T const*>(rows + static_cast<size_t>(row) * rowBytes);
    T value = shard[idx];
    if (op == ncclSum) {
      acc = acc + value;
    } else if (op == ncclProd) {
      acc = acc * value;
    } else if (op == ncclMax) {
      acc = value > acc ? value : acc;
    } else if (op == ncclMin) {
      acc = value < acc ? value : acc;
    }
  }
  reinterpret_cast<T*>(output)[idx] = acc;
}

ncclResult_t launchReduceRows(void* rows, void* output, size_t elemOffset,
                              size_t count, size_t rowBytes,
                              ncclDataType_t datatype,
                              ncclRedOp_t op, int nRows, cudaStream_t stream) {
  if (op != ncclSum && op != ncclProd && op != ncclMax && op != ncclMin) {
    WARN("unsupported native reduction op %d", op);
    return ncclInvalidArgument;
  }

  constexpr int threads = 256;
  int blocks = static_cast<int>((count + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;

  switch (datatype) {
    case ncclFloat32:
      reduceRowsKernel<float><<<blocks, threads, 0, stream>>>(
          static_cast<char const*>(rows),
          static_cast<float*>(output) + elemOffset, count, rowBytes, nRows, op);
      break;
    case ncclFloat64:
      reduceRowsKernel<double><<<blocks, threads, 0, stream>>>(
          static_cast<char const*>(rows),
          static_cast<double*>(output) + elemOffset, count, rowBytes, nRows, op);
      break;
    case ncclInt32:
      reduceRowsKernel<int32_t><<<blocks, threads, 0, stream>>>(
          static_cast<char const*>(rows),
          static_cast<int32_t*>(output) + elemOffset, count, rowBytes, nRows,
          op);
      break;
    case ncclUint32:
      reduceRowsKernel<uint32_t><<<blocks, threads, 0, stream>>>(
          static_cast<char const*>(rows),
          static_cast<uint32_t*>(output) + elemOffset, count, rowBytes, nRows,
          op);
      break;
    default:
      WARN("unsupported native reduction datatype %d", datatype);
      return ncclInvalidArgument;
  }

  return cudaResult(cudaGetLastError(), "native reduction kernel launch");
}

size_t maxChunkBytes(size_t scratchBufferSize, int nRows, size_t typeSize) {
  if (nRows <= 0 || typeSize == 0) return 0;
  size_t rowBytes = scratchBufferSize / static_cast<size_t>(nRows);
  rowBytes = std::min(rowBytes, static_cast<size_t>(2 * 1024 * 1024));
  return rowBytes / typeSize * typeSize;
}

ncclResult_t finishGroup(ncclResult_t enqueueResult) {
  ncclResult_t groupResult = ncclGroupEnd();
  if (enqueueResult != ncclSuccess) return enqueueResult;
  return groupResult;
}

}  // namespace

ncclResult_t runSendRecvAllGather(void const* sendbuff, void* recvbuff,
                                  size_t sendcount, size_t bytesPerRank,
                                  ncclDataType_t datatype, ncclComm_t comm,
                                  cudaStream_t stream, int rank, int nRanks,
                                  std::shared_ptr<Communicator> bootstrapComm) {
  size_t typeSize = ncclTypeSize(datatype);
  if (typeSize == 0) return ncclInvalidArgument;
  size_t maxChunkCount = std::max(static_cast<size_t>(1),
                                  static_cast<size_t>(2 * 1024 * 1024) /
                                      typeSize);
  auto const* send = static_cast<char const*>(sendbuff);
  auto* recv = static_cast<char*>(recvbuff);
  for (size_t elemOffset = 0; elemOffset < sendcount;
       elemOffset += maxChunkCount) {
    size_t chunkCount = std::min(maxChunkCount, sendcount - elemOffset);
    size_t chunkBytes = chunkCount * typeSize;
    size_t offsetBytes = elemOffset * typeSize;
    size_t selfOffset = static_cast<size_t>(rank) * bytesPerRank + offsetBytes;
    if (send + offsetBytes != recv + selfOffset) {
      ncclResult_t result =
          cudaResult(cudaMemcpyAsync(recv + selfOffset, send + offsetBytes,
                                     chunkBytes, cudaMemcpyDeviceToDevice,
                                     stream),
                     "allgather self copy");
      if (result != ncclSuccess) return result;
    }

    ncclResult_t result = ncclGroupStart();
    if (result != ncclSuccess) return result;

    ncclResult_t enqueueResult = ncclSuccess;
    for (int peer = 0; peer < nRanks; ++peer) {
      if (peer == rank) continue;
      enqueueResult =
          ncclSend(send + offsetBytes, chunkCount, datatype, peer, comm, stream);
      if (enqueueResult != ncclSuccess) break;
    }
    if (enqueueResult == ncclSuccess) {
      for (int peer = 0; peer < nRanks; ++peer) {
        if (peer == rank) continue;
        enqueueResult =
            ncclRecv(recv + static_cast<size_t>(peer) * bytesPerRank +
                         offsetBytes,
                     chunkCount, datatype, peer, comm, stream);
        if (enqueueResult != ncclSuccess) break;
      }
    }
    result = finishGroup(enqueueResult);
    if (result != ncclSuccess) return result;
    if (elemOffset + chunkCount < sendcount) {
      result =
          cudaResult(cudaStreamSynchronize(stream), "allgather chunk sync");
      if (result != ncclSuccess) return result;
      bootstrapComm->bootstrap()->barrier();
    }
  }
  return ncclSuccess;
}

ncclResult_t runSendRecvReduceScatter(void const* sendbuff, void* recvbuff,
                                      size_t recvcount, size_t bytesPerRank,
                                      ncclDataType_t datatype, ncclRedOp_t op,
                                      ncclComm_t comm, cudaStream_t stream,
                                      int rank, int nRanks, void* scratchBuffer,
                                      size_t scratchBufferSize,
                                      std::shared_ptr<Communicator> bootstrapComm) {
  size_t typeSize = ncclTypeSize(datatype);
  size_t rowBytes = maxChunkBytes(scratchBufferSize, nRanks, typeSize);
  if (rowBytes == 0) return ncclInvalidUsage;

  size_t maxChunkCount = rowBytes / typeSize;
  auto const* send = static_cast<char const*>(sendbuff);
  auto* scratch = static_cast<char*>(scratchBuffer);
  for (size_t elemOffset = 0; elemOffset < recvcount;
       elemOffset += maxChunkCount) {
    size_t chunkCount = std::min(maxChunkCount, recvcount - elemOffset);
    size_t chunkBytes = chunkCount * typeSize;
    size_t offsetBytes = elemOffset * typeSize;

    ncclResult_t result = cudaResult(
        cudaMemcpyAsync(scratch + static_cast<size_t>(rank) * rowBytes,
                        send + static_cast<size_t>(rank) * bytesPerRank +
                            offsetBytes,
                        chunkBytes, cudaMemcpyDeviceToDevice, stream),
        "reducescatter self shard copy");
    if (result != ncclSuccess) return result;

    result = ncclGroupStart();
    if (result != ncclSuccess) return result;

    ncclResult_t enqueueResult = ncclSuccess;
    for (int peer = 0; peer < nRanks; ++peer) {
      if (peer == rank) continue;
      enqueueResult =
          ncclSend(send + static_cast<size_t>(peer) * bytesPerRank + offsetBytes,
                   chunkCount, datatype, peer, comm, stream);
      if (enqueueResult != ncclSuccess) break;
    }
    if (enqueueResult == ncclSuccess) {
      for (int peer = 0; peer < nRanks; ++peer) {
        if (peer == rank) continue;
        enqueueResult = ncclRecv(scratch + static_cast<size_t>(peer) * rowBytes,
                                 chunkCount, datatype, peer, comm, stream);
        if (enqueueResult != ncclSuccess) break;
      }
    }

    result = finishGroup(enqueueResult);
    if (result != ncclSuccess) return result;
    result = launchReduceRows(scratch, recvbuff, elemOffset, chunkCount,
                              rowBytes, datatype, op, nRanks, stream);
    if (result != ncclSuccess) return result;
    if (elemOffset + chunkCount < recvcount) {
      result = cudaResult(cudaStreamSynchronize(stream),
                          "reducescatter chunk synchronization");
      if (result != ncclSuccess) return result;
      bootstrapComm->bootstrap()->barrier();
    }
  }
  ncclResult_t result = cudaResult(cudaStreamSynchronize(stream),
                                   "reducescatter final synchronization");
  if (result != ncclSuccess) return result;
  bootstrapComm->bootstrap()->barrier();
  return ncclSuccess;
}

ncclResult_t runSendRecvAllReduce(void const* sendbuff, void* recvbuff,
                                  size_t count, ncclDataType_t datatype,
                                  ncclRedOp_t op, ncclComm_t comm,
                                  cudaStream_t stream, int rank, int nRanks,
                                  void* scratchBuffer,
                                  size_t scratchBufferSize,
                                  std::shared_ptr<Communicator> bootstrapComm) {
  size_t typeSize = ncclTypeSize(datatype);
  if (typeSize == 0 || count % static_cast<size_t>(nRanks) != 0) {
    return ncclInvalidArgument;
  }

  if (sendbuff != recvbuff) {
    size_t bytes = count * typeSize;
    ncclResult_t result = cudaResult(cudaMemcpyAsync(recvbuff, sendbuff, bytes,
                                        cudaMemcpyDeviceToDevice, stream),
                                      "allreduce out-of-place input copy");
    if (result != ncclSuccess) return result;
    return runSendRecvAllReduce(recvbuff, recvbuff, count, datatype, op, comm,
                                stream, rank, nRanks, scratchBuffer,
                                scratchBufferSize, bootstrapComm);
  }

  size_t shardCount = count / static_cast<size_t>(nRanks);
  size_t shardBytes = shardCount * typeSize;
  auto* recv = static_cast<char*>(recvbuff);
  void* localShard = recv + static_cast<size_t>(rank) * shardBytes;

  ncclResult_t result = runSendRecvReduceScatter(
      sendbuff, localShard, shardCount, shardBytes, datatype, op, comm, stream,
      rank, nRanks, scratchBuffer, scratchBufferSize, bootstrapComm);
  if (result != ncclSuccess) return result;

  return runSendRecvAllGather(localShard, recvbuff, shardCount, shardBytes,
                              datatype, comm, stream, rank, nRanks,
                              bootstrapComm);
}

}  // namespace nccl
}  // namespace mscclpp
