// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_ALLGATHER_FULLMESH_2_HPP_
#define MSCCLPP_EXT_ALLGATHER_FULLMESH_2_HPP_

#include "algorithm.hpp"

namespace mscclpp {
namespace collective {

class AllgatherFullmesh2 : public AlgorithmBuilder {
 public:
  AllgatherFullmesh2() = default;
  std::shared_ptr<Algorithm> build() override;

 private:
  bool symmetricMemory_;
  std::vector<Connection> conns_;
  std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>> memorySemaphores_;
  int const nChannelsPerConnection_ = 35;

  void initialize(std::shared_ptr<Communicator> comm);
  CommResult allgatherKernelFunc(
      const std::shared_ptr<void> ctx, void const* input, void* output,
      size_t inputSize, cudaStream_t stream, int nBlocks, int nThreadsPerBlock,
      std::unordered_map<std::string, uintptr_t> const& extras);

  std::shared_ptr<void> initAllgatherContext(std::shared_ptr<Communicator> comm,
                                             void const*, void* output, size_t,
                                             DataType);
  AlgorithmCtxKey generateAllgatherContextKey(void const*, void*, size_t,
                                              DataType, bool);
};

}  // namespace collective
}  // namespace mscclpp
#endif  // MSCCLPP_EXT_ALLGATHER_FULLMESH_2_HPP_