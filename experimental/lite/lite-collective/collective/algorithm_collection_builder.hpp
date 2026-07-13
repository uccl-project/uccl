// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_COLLECTIVES_ALGORITHM_COLLECTION_BUILDER_HPP_
#define MSCCLPP_EXT_COLLECTIVES_ALGORITHM_COLLECTION_BUILDER_HPP_

#include "algorithm.hpp"

namespace mscclpp {
namespace collective {

/// Builder for creating AlgorithmCollection instances.
///
/// AlgorithmCollectionBuilder provides a singleton interface for registering
/// algorithm builders and configuring algorithm selection functions. It can
/// build both default algorithms and custom algorithms registered by users.
///
/// Typical usage:
///   1. Get the singleton instance with getInstance()
///   2. Add algorithm builders with addAlgorithmBuilder()
///   3. Optionally set custom selectors with setAlgorithmSelector()
///   4. Build the collection with build() or buildDefaultAlgorithms()
class AlgorithmCollectionBuilder {
 public:
  /// Get the singleton instance of the builder.
  /// @return A shared pointer to the singleton instance.
  static std::shared_ptr<AlgorithmCollectionBuilder> getInstance();

  /// Reset the singleton instance.
  static void reset();

  /// Add a new algorithm builder.
  /// @param builder The algorithm builder to add.
  void addAlgorithmBuilder(std::shared_ptr<AlgorithmBuilder> builder);

  /// Set a custom algorithm selection function.
  /// @param selector The algorithm selection function.
  void setAlgorithmSelector(AlgoSelectFunc selector);

  /// Set a fallback algorithm selection function.
  /// @param selector The fallback algorithm selection function.
  /// @note The fallback selector is used if the primary selector returns
  /// nullptr.
  ///       MSCCL++ assigns a predefined selector as the fallback by default.
  void setFallbackAlgorithmSelector(AlgoSelectFunc selector);

  /// Build the AlgorithmCollection instance.
  /// @return The built AlgorithmCollection containing all registered
  /// algorithms.
  AlgorithmCollection build();

  AlgorithmCollection buildDefaultAlgorithms(uintptr_t scratchBuffer,
                                             size_t scratchBufferSize,
                                             uintptr_t flagBuffer,
                                             size_t flagBufferSize, int rank);

 private:
  AlgorithmCollectionBuilder() = default;
  std::vector<std::shared_ptr<AlgorithmBuilder>> algoBuilders_;
  AlgoSelectFunc algoSelector_ = nullptr;
  AlgoSelectFunc fallbackAlgoSelector_ = nullptr;

  AlgorithmCollection buildDefaultNativeAlgorithms(uintptr_t scratchBuffer,
                                                   size_t scratchBufferSize,
                                                   uintptr_t flagBuffer,
                                                   size_t flagBufferSize);
  AlgorithmCollection buildDefaultDslAlgorithms(int rank);

  static std::shared_ptr<AlgorithmCollectionBuilder>
      gAlgorithmCollectionBuilder_;
};

}  // namespace collective
}  // namespace mscclpp

#endif  // MSCCLPP_EXT_COLLECTIVES_ALGORITHM_COLLECTION_BUILDER_HPP_