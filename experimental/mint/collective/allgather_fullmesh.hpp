// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_ALLGATHER_FULLMESH_HPP_
#define MSCCLPP_EXT_ALLGATHER_FULLMESH_HPP_

#include "algorithm.hpp"

namespace mscclpp {
namespace collective {

class AllgatherFullmesh : public AlgorithmBuilder {
 public:
  AllgatherFullmesh(uintptr_t scratchBuffer, size_t scratchBufferSize)
      : scratchBuffer_(reinterpret_cast<void*>(scratchBuffer)),
        scratchBufferSize_(scratchBufferSize) {}
  std::shared_ptr<mscclpp::Algorithm> build() override;

 private:
  std::vector<mscclpp::Connection> conns_;

  void initialize(std::shared_ptr<mscclpp::Communicator> comm);
  CommResult allgatherKernelFunc(
      const std::shared_ptr<void> ctx, void const* input, void* output,
      size_t inputSize, cudaStream_t stream, int nBlocks, int nThreadsPerBlock,
      std::unordered_map<std::string, uintptr_t> const& extras);

  std::shared_ptr<void> initAllgatherContext(
      std::shared_ptr<mscclpp::Communicator> comm, void const*, void* output,
      size_t, mscclpp::DataType);
  mscclpp::AlgorithmCtxKey generateAllgatherContextKey(void const*, void*,
                                                       size_t,
                                                       mscclpp::DataType, bool);

  void* scratchBuffer_;
  size_t scratchBufferSize_;
};
}  // namespace collective
}  // namespace mscclpp
#endif  // MSCCLPP_EXT_ALLGATHER_FULLMESH_HPP_
