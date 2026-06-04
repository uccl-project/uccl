#ifndef MSCCLPP_NCCL_ALLTOALL_HPP_
#define MSCCLPP_NCCL_ALLTOALL_HPP_

#include "core.hpp"
#include "nccl.h"
#include <memory>
#include <vector>

struct AllToAllCommView {
  ncclComm_t handle = nullptr;
  std::shared_ptr<mscclpp::Communicator> comm;
  int worldSize = 0;
  int nRanksPerNode = 0;
  int cudaDevice = -1;
  bool hasIB = false;
};

enum class GroupedP2POpKind { Send, Recv };

struct GroupedP2POp {
  GroupedP2POpKind kind;
  void const* sendbuff = nullptr;
  void* recvbuff = nullptr;
  size_t count = 0;
  ncclDataType_t datatype = ncclFloat32;
  int peer = -1;
  ncclComm_t comm = nullptr;
  cudaStream_t stream = nullptr;
};

ncclResult_t executeSelfGroupedP2POp(GroupedP2POp const& sendOp,
                                     GroupedP2POp const& recvOp);

bool tryExecuteOptimizedGroupedAllToAll(AllToAllCommView const& commView,
                                        std::vector<GroupedP2POp>& ops,
                                        ncclResult_t& result);

void cleanupAllToAllContexts(ncclComm_t comm);

#endif  // MSCCLPP_NCCL_ALLTOALL_HPP_
