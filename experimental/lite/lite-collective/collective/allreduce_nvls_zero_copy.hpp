// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_ALLREDUCE_NVLS_ZERO_COPY_HPP_
#define MSCCLPP_ALLREDUCE_NVLS_ZERO_COPY_HPP_

#include "algorithm.hpp"

namespace mscclpp {
namespace collective {

class AllreduceNvls : public AlgorithmBuilder {
 public:
  AllreduceNvls() = default;
  std::shared_ptr<Algorithm> build() override;

 private:
  bool symmetricMemory_ = false;
  void initialize(std::shared_ptr<Communicator> comm);
  CommResult allreduceKernelFunc(
      const std::shared_ptr<void> ctx, void const* input, void* output,
      size_t inputSize, DataType dtype, ReduceOp op, cudaStream_t stream,
      int nBlocks, int nThreadsPerBlock,
      std::unordered_map<std::string, uintptr_t> const& extras);

  std::shared_ptr<void> initAllreduceContext(std::shared_ptr<Communicator> comm,
                                             void const*, void* output, size_t,
                                             DataType);
  AlgorithmCtxKey generateAllreduceContextKey(void const*, void*, size_t,
                                              DataType, bool);

  // Large buffer size because cuMemMap requires offset=0 for multicast handles,
  // so the entire user allocation must be mapped. This only reserves virtual
  // address space; no physical memory is consumed beyond what is actually
  // bound.
  const size_t nvlsBufferSize_ = (1UL << 34);
  uint32_t nSwitchChannels_;
  std::shared_ptr<DeviceHandle<BaseMemoryChannel>> memoryChannelsDeviceHandle_;
  std::vector<BaseMemoryChannel> baseChannels_;
  std::vector<Connection> conns_;
  std::vector<std::shared_ptr<NvlsConnection>> nvlsConnections_;
  std::vector<std::shared_ptr<NvlsConnection>> nvlsOutConnections_;
  int computeCapabilityMajor_{0};
};

}  // namespace collective
}  // namespace mscclpp

#endif  // MSCCLPP_ALLREDUCE_NVLS_ZERO_COPY_HPP_