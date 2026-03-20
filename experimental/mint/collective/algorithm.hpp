// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_ALGORITHM_HPP_
#define MSCCLPP_ALGORITHM_HPP_

#include "executor.hpp"
#include "memory_channel.hpp"
#include "port_channel.hpp"
#include "switch_channel.hpp"
#include "utils.hpp"
#include <memory>
#include <vector>

namespace mscclpp {

/// Capsule name for native algorithm pointers used in Python bindings.
constexpr char ALGORITHM_NATIVE_CAPSULE_NAME[] = "mscclpp::AlgorithmPtr";

enum class CollectiveBufferMode {
  Any = 0,
  InPlace,
  OutOfPlace,
};

enum class AlgorithmType {
  Native = 0,
  DSL,
};

enum class CommResult {
  CommSuccess = 0,
  CommUnhandledCudaError = 1,
  CommSystemError = 2,
  CommInternalError = 3,
  CommInvalidArgument = 4,
  CommInvalidUsage = 5,
  CommRemoteError = 6,
  CommInProgress = 7,
  CommNumResults = 8
};

enum ReduceOp { SUM = 0, MIN = 3, NOP = 255 };

/// Base class for collective communication algorithms.
///
/// This abstract class defines the interface for implementing collective
/// communication algorithms such as allreduce, allgather, and reduce-scatter.
/// Concrete implementations can be either native C++/CUDA algorithms or
/// DSL-defined algorithms.
class Algorithm {
 public:
  struct Constraint {
    int worldSize;
    int nRanksPerNode;
  };

  virtual ~Algorithm() = default;

  /// Get the name of the algorithm.
  /// @return A reference to the algorithm name string.
  virtual std::string const& name() const = 0;

  /// Get the collective operation this algorithm implements.
  /// @return A reference to the collective name (e.g., "allreduce",
  /// "allgather").
  virtual std::string const& collective() const = 0;

  /// Get the valid message size range for this algorithm.
  /// @return A pair of (minMessageSize, maxMessageSize) in bytes.
  virtual std::pair<size_t, size_t> const& messageRange() const = 0;

  /// Get the tags associated with this algorithm.
  /// @return An unordered map of tag names to tag values.
  virtual std::unordered_map<std::string, uint64_t> const& tags() const = 0;

  /// Get the buffer mode supported by this algorithm.
  /// @return The CollectiveBufferMode indicating in-place, out-of-place, or
  /// any.
  virtual CollectiveBufferMode const& bufferMode() const = 0;

  /// Get the type of this algorithm.
  /// @return AlgorithmType::Native or AlgorithmType::DSL.
  virtual AlgorithmType type() const = 0;

  /// Get the execution constraints for this algorithm.
  /// @return The Constraint struct specifying worldSize and nRanksPerNode
  /// requirements.
  virtual Constraint constraint() const = 0;

  /// Set the valid message size range for this algorithm.
  /// @param minMessageSize Minimum supported message size in bytes.
  /// @param maxMessageSize Maximum supported message size in bytes.
  virtual void setMessageSizeRange(size_t minMessageSize,
                                   size_t maxMessageSize) = 0;

  /// Execute the algorithm.
  virtual CommResult execute(
      std::shared_ptr<Communicator> comm, void const* input, void* output,
      size_t inputSize, size_t outputSize, DataType dtype, ReduceOp op,
      cudaStream_t stream, std::shared_ptr<Executor> executor, int nBlocks = 0,
      int nThreadsPerBlock = 0, bool symmetricMemory = false,
      std::unordered_map<std::string, uintptr_t> const& extras = {}) = 0;

  /// Reset the algorithm state, clearing any cached contexts.
  virtual void reset() = 0;
};

/// Interface for building Algorithm instances.
class AlgorithmBuilder {
 public:
  virtual ~AlgorithmBuilder() = default;
  virtual std::shared_ptr<Algorithm> build() = 0;
};

/// Key for identifying cached AlgorithmCtx instances.
struct AlgorithmCtxKey {
  void* baseSendBuff;
  void* baseRecvBuff;
  size_t baseSendSize;
  size_t baseRecvSize;
  int tag;

  bool operator==(AlgorithmCtxKey const& other) const {
    return baseSendBuff == other.baseSendBuff &&
           baseRecvBuff == other.baseRecvBuff &&
           baseSendSize == other.baseSendSize &&
           baseRecvSize == other.baseRecvSize && tag == other.tag;
  }
};

}  // namespace mscclpp

namespace std {

template <>
struct hash<mscclpp::AlgorithmCtxKey> {
  std::size_t operator()(mscclpp::AlgorithmCtxKey const& key) const {
    std::size_t seed = 42;
    mscclpp::detail::hashCombine(seed, key.baseSendBuff);
    mscclpp::detail::hashCombine(seed, key.baseRecvBuff);
    mscclpp::detail::hashCombine(seed, key.baseSendSize);
    mscclpp::detail::hashCombine(seed, key.baseRecvSize);
    mscclpp::detail::hashCombine(seed, key.tag);
    return seed;
  }
};
}  // namespace std

namespace mscclpp {

/// Native C++/CUDA implementation of a collective algorithm.
class NativeAlgorithm : public Algorithm {
 public:
  using InitFunc = std::function<void(std::shared_ptr<Communicator>)>;
  using KernelFunc = std::function<CommResult(
      const std::shared_ptr<void>, void const*, void*, size_t, size_t, DataType,
      ReduceOp, cudaStream_t, int, int,
      std::unordered_map<std::string, uintptr_t> const&)>;
  using ContextInitFunc = std::function<std::shared_ptr<void>(
      std::shared_ptr<Communicator>, void const*, void*, size_t, size_t,
      DataType)>;
  using ContextKeyGenFunc = std::function<AlgorithmCtxKey(
      void const* input, void* output, size_t inputSize, size_t outputSize,
      DataType dtype, bool symmetricMemory)>;

  NativeAlgorithm(std::string name, std::string collective, InitFunc initFunc,
                  KernelFunc kernelFunc, ContextInitFunc contextInitFunc,
                  ContextKeyGenFunc contextKeyGenFunc,
                  size_t minMessageSize = 0, size_t maxMessageSize = UINT64_MAX,
                  CollectiveBufferMode bufferMode = CollectiveBufferMode::Any,
                  std::unordered_map<std::string, uint64_t> tags = {},
                  Constraint constraint = {});

  CommResult execute(
      std::shared_ptr<Communicator> comm, void const* input, void* output,
      size_t inputSize, size_t outputSize, DataType dtype, ReduceOp op,
      cudaStream_t stream, std::shared_ptr<Executor> executor, int nBlocks = 0,
      int nThreadsPerBlock = 0, bool symmetricMemory = false,
      std::unordered_map<std::string, uintptr_t> const& extras = {}) override;
  std::string const& name() const override;
  std::string const& collective() const override;
  std::pair<size_t, size_t> const& messageRange() const override;
  void setMessageSizeRange(size_t minMessageSize,
                           size_t maxMessageSize) override;
  std::unordered_map<std::string, uint64_t> const& tags() const override;
  CollectiveBufferMode const& bufferMode() const override;
  AlgorithmType type() const override { return AlgorithmType::Native; }
  Constraint constraint() const override;
  void reset() override;

 private:
  std::string name_;
  std::string collective_;
  NativeAlgorithm::InitFunc initFunc_;
  NativeAlgorithm::KernelFunc kernelLaunchFunc_;
  NativeAlgorithm::ContextInitFunc contextInitFunc_;
  NativeAlgorithm::ContextKeyGenFunc contextKeyGenFunc_;
  size_t minMessageSize_;
  size_t maxMessageSize_;
  CollectiveBufferMode bufferMode_;
  std::unordered_map<std::string, uint64_t> tags_;
  Constraint constraint_;
  std::unordered_map<AlgorithmCtxKey, std::shared_ptr<void>> contexts_;

  bool initialized_ = false;
};

/// DSL-based implementation of a collective algorithm.
class DslAlgorithm : public Algorithm,
                     public AlgorithmBuilder,
                     public std::enable_shared_from_this<DslAlgorithm> {
 public:
  DslAlgorithm(std::string id, ExecutionPlan plan,
               std::unordered_map<std::string, uint64_t> tags = {},
               Constraint constraint = {});
  std::string const& name() const override;
  std::string const& collective() const override;
  std::pair<size_t, size_t> const& messageRange() const override;
  void setMessageSizeRange(size_t minMessageSize,
                           size_t maxMessageSize) override;
  std::unordered_map<std::string, uint64_t> const& tags() const override;
  CollectiveBufferMode const& bufferMode() const override;
  CommResult execute(
      std::shared_ptr<Communicator> comm, void const* input, void* output,
      size_t inputSize, size_t outputSize, DataType dtype, ReduceOp op,
      cudaStream_t stream, std::shared_ptr<Executor> executor, int nBlocks = 0,
      int nThreadsPerBlock = 0, bool symmetricMemory = false,
      std::unordered_map<std::string, uintptr_t> const& extras = {}) override;
  AlgorithmType type() const override { return AlgorithmType::DSL; }
  Constraint constraint() const override;
  void reset() override;

  std::shared_ptr<Algorithm> build() override;

 private:
  ExecutionPlan plan_;
  std::string id_;
  std::unordered_map<std::string, uint64_t> tags_;
  Constraint constraint_;
};

/// Request parameters for selecting and executing a collective operation.
struct CollectiveRequest {
  int worldSize;
  int nRanksPerNode;
  int rank;
  void const* inputBuffer;
  void* outputBuffer;
  size_t messageSize;
  cudaStream_t stream;
  std::string const& collective;
  const DataType dtype;
  std::unordered_map<std::string, std::vector<uint64_t>> const& hints;

  CollectiveBufferMode bufferMode() const;
};

using AlgoSelectFunc = std::function<std::shared_ptr<Algorithm>(
    std::unordered_map<
        std::string,
        std::unordered_map<std::string, std::shared_ptr<Algorithm>>> const&
        algoMapByCollective,
    CollectiveRequest const& request)>;

/// Collection of algorithms for collective operations.
class AlgorithmCollection {
 public:
  AlgorithmCollection() = default;

  std::shared_ptr<Algorithm> selectAlgorithm(CollectiveRequest const& request);

  void registerAlgorithm(const std::string collective,
                         const std::string algoName,
                         std::shared_ptr<Algorithm> algorithm);

  std::unordered_map<std::string, std::shared_ptr<Algorithm>>
  getAlgorithmsByCollective(std::string const& collective) const;

  std::vector<std::shared_ptr<Algorithm>> getAllAlgorithms() const;

  void extend(AlgorithmCollection const& other);

  void setSelectors(AlgoSelectFunc algoSelector,
                    AlgoSelectFunc fallbackAlgoSelector);

 private:
  std::unordered_map<
      std::string, std::unordered_map<std::string, std::shared_ptr<Algorithm>>>
      algoMapByCollective_;
  AlgoSelectFunc algoSelector_ = nullptr;
  AlgoSelectFunc fallbackAlgoSelector_ = nullptr;
};

std::pair<std::shared_ptr<void>, size_t> getFlagBuffer();

}  // namespace mscclpp

#endif  // MSCCLPP_ALGORITHM_HPP_
