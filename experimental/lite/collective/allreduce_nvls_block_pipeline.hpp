// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_ALLREDUCE_NVLS_BLOCK_PIPELINE_HPP_
#define MSCCLPP_EXT_ALLREDUCE_NVLS_BLOCK_PIPELINE_HPP_

#include "algorithm.hpp"

namespace mscclpp {
namespace collective {

class AllreduceNvlsBlockPipeline : public AlgorithmBuilder {
 public:
  AllreduceNvlsBlockPipeline(uintptr_t scratchBuffer, size_t scratchBufferSize)
      : scratchBuffer_(reinterpret_cast<void*>(scratchBuffer)),
        scratchBufferSize_(scratchBufferSize){};
  std::shared_ptr<Algorithm> build() override;

 private:
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

  const size_t nvlsBufferSize_ = (1 << 30);
  void* scratchBuffer_;
  size_t scratchBufferSize_;
  uint32_t nSwitchChannels_;
  std::shared_ptr<DeviceHandle<BaseMemoryChannel>> memoryChannelsDeviceHandle_;
  std::vector<BaseMemoryChannel> baseChannels_;
  std::vector<Connection> conns_;
  std::vector<std::shared_ptr<NvlsConnection>> nvlsConnections_;
};
}  // namespace collective
}  // namespace mscclpp

#endif  // MSCCLPP_EXT_ALLREDUCE_NVLS_BLOCK_PIPELINE_HPP_
