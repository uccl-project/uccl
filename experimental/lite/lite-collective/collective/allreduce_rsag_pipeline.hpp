// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_ALLREDUCE_RSAG_PIPELINE_HPP_
#define MSCCLPP_EXT_ALLREDUCE_RSAG_PIPELINE_HPP_

#include "algorithm.hpp"

namespace mscclpp {
namespace collective {

class AllreduceRsAgPipeline : public mscclpp::AlgorithmBuilder {
 public:
  AllreduceRsAgPipeline(uintptr_t scratchBuffer, size_t scratchBufferSize)
      : scratchBuffer_((void*)scratchBuffer),
        scratchBufferSize_(scratchBufferSize){};
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
  void* scratchBuffer_;
  size_t scratchBufferSize_;
  std::shared_ptr<Communicator> comm_;
  int nChannelsPerConnection_;
  std::vector<Connection> conns_;
  std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>> scratchSemaphores_;
  std::vector<RegisteredMemory> remoteScratchMemories_;
  RegisteredMemory localScratchMemory_;

  std::vector<BaseMemoryChannel> baseChannels_;
  std::shared_ptr<DeviceHandle<BaseMemoryChannel>> baseMemoryChannelHandles_;
  std::shared_ptr<void*> remoteMemoryHandles_;
};
}  // namespace collective
}  // namespace mscclpp

#endif  // MSCCLPP_EXT_ALLREDUCE_RSAG_PIPELINE_HPP_