// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "algorithm.hpp"

namespace mscclpp {
namespace collective {

class AllreduceFullmesh : public mscclpp::AlgorithmBuilder {
 public:
  AllreduceFullmesh(uintptr_t scratchBuffer, size_t scratchBufferSize)
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
  std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>> outputSemaphores_;
  std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>>
      inputScratchSemaphores_;
  std::vector<RegisteredMemory> remoteScratchMemories_;
  RegisteredMemory localScratchMemory_;
  std::unordered_map<void const*,
                     std::pair<std::vector<MemoryChannel>,
                               std::shared_ptr<DeviceHandle<MemoryChannel>>>>
      memoryChannelsMap_;
  bool symmetricMemory_ = false;
};
}  // namespace collective
}  // namespace mscclpp