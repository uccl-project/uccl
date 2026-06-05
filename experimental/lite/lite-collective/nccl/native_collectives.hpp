#ifndef MSCCLPP_NCCL_NATIVE_COLLECTIVES_HPP_
#define MSCCLPP_NCCL_NATIVE_COLLECTIVES_HPP_

#include "nccl.h"
#include "core.hpp"
#include <cstddef>
#include <memory>

namespace mscclpp {
namespace nccl {

ncclResult_t runSendRecvAllGather(void const* sendbuff, void* recvbuff,
                                  size_t sendcount, size_t bytesPerRank,
                                  ncclDataType_t datatype, ncclComm_t comm,
                                  cudaStream_t stream, int rank, int nRanks,
                                  std::shared_ptr<Communicator> bootstrapComm);

ncclResult_t runSendRecvReduceScatter(void const* sendbuff, void* recvbuff,
                                      size_t recvcount, size_t bytesPerRank,
                                      ncclDataType_t datatype, ncclRedOp_t op,
                                      ncclComm_t comm, cudaStream_t stream,
                                      int rank, int nRanks, void* scratchBuffer,
                                      size_t scratchBufferSize,
                                      std::shared_ptr<Communicator> bootstrapComm);

ncclResult_t runSendRecvAllReduce(void const* sendbuff, void* recvbuff,
                                  size_t count, ncclDataType_t datatype,
                                  ncclRedOp_t op, ncclComm_t comm,
                                  cudaStream_t stream, int rank, int nRanks,
                                  void* scratchBuffer,
                                  size_t scratchBufferSize,
                                  std::shared_ptr<Communicator> bootstrapComm);

}  // namespace nccl
}  // namespace mscclpp

#endif  // MSCCLPP_NCCL_NATIVE_COLLECTIVES_HPP_
