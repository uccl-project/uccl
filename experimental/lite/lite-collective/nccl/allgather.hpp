// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ALLGATHER_HPP_
#define ALLGATHER_HPP_

#include "algorithm.hpp"
#include "core.hpp"
#include "nccl.h"
#include <memory>
#include <string>

enum class LiteAllGatherPath {
  SingleNodeCudaIpc,
  SingleNodeShm,
  MultiNode,
};

class LiteAllgatherAlgo : public mscclpp::AlgorithmBuilder {
 public:
  LiteAllgatherAlgo(std::string name, LiteAllGatherPath path);
  std::shared_ptr<mscclpp::Algorithm> build() override;

 private:
  std::string name_;
  LiteAllGatherPath path_;
};

using LiteAllgatherP2pFn = ncclResult_t (*)(void const* input, void* output,
                                            size_t bytesPerRank,
                                            ncclComm_t comm,
                                            cudaStream_t stream, int rank,
                                            int nRanks);

using LiteAllgatherHostFn = ncclResult_t (*)(
    void const* input, void* output, size_t bytesPerRank, ncclComm_t comm,
    cudaStream_t stream, int rank, int nRanks, int nRanksPerNode,
    std::shared_ptr<mscclpp::Communicator> bootstrapComm, int cudaDevice);

#endif  // ALLGATHER_HPP_
