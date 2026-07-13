// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_ALLREDUCE_RSAG_ZERO_COPY_HPP_
#define MSCCLPP_EXT_ALLREDUCE_RSAG_ZERO_COPY_HPP_

#include "algorithm.hpp"

namespace mscclpp {
namespace collective {

class AllreduceRsAgZeroCopy : public mscclpp::AlgorithmBuilder {
 public:
  AllreduceRsAgZeroCopy() = default;
  std::shared_ptr<mscclpp::Algorithm> build() override;

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
  std::shared_ptr<Communicator> comm_;
  int nChannelsPerConnection_;
  std::vector<Connection> conns_;
  std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>> semaphores_;
  std::vector<RegisteredMemory> inputMemories_;
  std::vector<RegisteredMemory> outputMemories_;

  std::vector<BaseMemoryChannel> baseChannels_;
  std::shared_ptr<DeviceHandle<BaseMemoryChannel>> baseMemoryChannelHandles_;
};
}  // namespace collective
}  // namespace mscclpp

#endif  // MSCCLPP_EXT_ALLREDUCE_RSAG_ZERO_COPY_HPP_