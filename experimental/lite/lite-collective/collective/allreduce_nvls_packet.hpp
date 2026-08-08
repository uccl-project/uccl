// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_ALLREDUCE_NVLS_PACKET_HPP_
#define MSCCLPP_ALLREDUCE_NVLS_PACKET_HPP_

#include "algorithm.hpp"

namespace mscclpp {
namespace collective {
class AllreduceNvlsPacket : public mscclpp::AlgorithmBuilder {
 public:
  AllreduceNvlsPacket(uintptr_t scratchBuffer, size_t scratchBufferSize,
                      uintptr_t flagBuffer, size_t flagBufferSize)
      : scratchBuffer_((void*)scratchBuffer),
        scratchBufferSize_(scratchBufferSize),
        flagBuffer_(flagBuffer),
        flagBufferSize_(flagBufferSize){};
  std::shared_ptr<mscclpp::Algorithm> build() override;

 private:
  void initialize(std::shared_ptr<mscclpp::Communicator> comm);
  CommResult allreduceKernelFunc(
      const std::shared_ptr<void> ctx, void const* input, void* output,
      size_t inputSize, mscclpp::DataType dtype, ReduceOp op,
      cudaStream_t stream, int nBlocks, int nThreadsPerBlock,
      std::unordered_map<std::string, uintptr_t> const& extras);

  std::shared_ptr<void> initAllreduceContext(
      std::shared_ptr<mscclpp::Communicator> comm, void const*, void* output,
      size_t, mscclpp::DataType);
  mscclpp::AlgorithmCtxKey generateAllreduceContextKey(void const*, void*,
                                                       size_t,
                                                       mscclpp::DataType, bool);

  void* scratchBuffer_;
  size_t scratchBufferSize_;
  const size_t nvlsBufferSize_ = (1 << 30);
  int const maxBlockNum_ = 16;
  uintptr_t flagBuffer_;
  size_t flagBufferSize_;
  std::vector<std::shared_ptr<NvlsConnection>> nvlsConnections_;
};
}  // namespace collective
}  // namespace mscclpp
#endif