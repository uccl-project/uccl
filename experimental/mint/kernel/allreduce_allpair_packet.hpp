// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "algorithm.hpp"
#include "common.hpp"

namespace mscclpp {
namespace collective {
class AllreduceAllpairPacket : public AlgorithmBuilder {
 public:
  AllreduceAllpairPacket(uintptr_t scratchBuffer, size_t scratchBufferSize,
                         uintptr_t flagBuffer, size_t flagBufferSize)
      : scratchBuffer_((void*)scratchBuffer),
        scratchBufferSize_(scratchBufferSize),
        flagBuffer_(flagBuffer),
        flagBufferSize_(flagBufferSize){};
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

  void* scratchBuffer_;
  size_t scratchBufferSize_;
  int const nSegmentsForScratchBuffer_ = 2;
  int const maxBlockNum_ = 28;
  std::vector<Connection> conns_;
  std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>> memorySemaphores_;
  std::vector<RegisteredMemory> registeredMemories_;
  uintptr_t flagBuffer_;
  size_t flagBufferSize_;
};
}  // namespace collective
}  // namespace mscclpp