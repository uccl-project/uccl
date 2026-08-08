// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "algorithm.hpp"
#include "gpu_utils.hpp"
#include "logger.hpp"
#include <filesystem>

namespace mscclpp {

CollectiveBufferMode CollectiveRequest::bufferMode() const {
  if (inputBuffer == outputBuffer) return CollectiveBufferMode::InPlace;
  if (collective == "allgather") {
    size_t rankOffset = rank * messageSize;
    char const* expectedInput =
        static_cast<char const*>(outputBuffer) + rankOffset;
    if (static_cast<void const*>(expectedInput) == inputBuffer) {
      return CollectiveBufferMode::InPlace;
    }
    return CollectiveBufferMode::OutOfPlace;
  }
  return CollectiveBufferMode::OutOfPlace;
}

NativeAlgorithm::NativeAlgorithm(std::string name, std::string collective,
                                 InitFunc initFunc, KernelFunc kernelFunc,
                                 ContextInitFunc contextInitFunc,
                                 ContextKeyGenFunc contextKeyGenFunc,
                                 size_t minMessageSize, size_t maxMessageSize,
                                 CollectiveBufferMode bufferMode,
                                 std::unordered_map<std::string, uint64_t> tags,
                                 Constraint constraint)
    : name_(name),
      collective_(collective),
      initFunc_(initFunc),
      kernelLaunchFunc_(kernelFunc),
      contextInitFunc_(contextInitFunc),
      contextKeyGenFunc_(contextKeyGenFunc),
      minMessageSize_(minMessageSize),
      maxMessageSize_(maxMessageSize),
      bufferMode_(bufferMode),
      tags_(tags),
      constraint_(constraint) {}

CommResult NativeAlgorithm::execute(
    std::shared_ptr<Communicator> comm, void const* input, void* output,
    size_t inputSize, size_t outputSize, DataType dtype, ReduceOp op,
    cudaStream_t stream, std::shared_ptr<Executor>, int nBlocks,
    int nThreadsPerBlock, bool symmetricMemory,
    std::unordered_map<std::string, uintptr_t> const& extras) {
  if (!initialized_) {
    initFunc_(comm);
    initialized_ = true;
  }
  AlgorithmCtxKey ctxKey = contextKeyGenFunc_(
      input, output, inputSize, outputSize, dtype, symmetricMemory);
  auto it = contexts_.find(ctxKey);
  if (it == contexts_.end()) {
    auto ctx =
        contextInitFunc_(comm, input, output, inputSize, outputSize, dtype);
    contexts_[ctxKey] = ctx;
  }
  return kernelLaunchFunc_(contexts_[ctxKey], input, output, inputSize,
                           outputSize, dtype, op, stream, nBlocks,
                           nThreadsPerBlock, extras);
}

std::string const& NativeAlgorithm::name() const { return name_; }

std::string const& NativeAlgorithm::collective() const { return collective_; }

std::pair<size_t, size_t> const& NativeAlgorithm::messageRange() const {
  static std::pair<size_t, size_t> range;
  range = {minMessageSize_, maxMessageSize_};
  return range;
}

void NativeAlgorithm::setMessageSizeRange(size_t minMessageSize,
                                          size_t maxMessageSize) {
  minMessageSize_ = minMessageSize;
  maxMessageSize_ = maxMessageSize;
}

std::unordered_map<std::string, uint64_t> const& NativeAlgorithm::tags() const {
  return tags_;
}

CollectiveBufferMode const& NativeAlgorithm::bufferMode() const {
  return bufferMode_;
}

Algorithm::Constraint NativeAlgorithm::constraint() const {
  return constraint_;
}

void NativeAlgorithm::reset() {
  contexts_.clear();
  initialized_ = false;
}

void AlgorithmCollection::registerAlgorithm(
    const std::string collective, const std::string algoName,
    std::shared_ptr<Algorithm> algorithm) {
  this->algoMapByCollective_[collective][algoName] = algorithm;
}

std::shared_ptr<Algorithm> AlgorithmCollection::selectAlgorithm(
    CollectiveRequest const& request) {
  std::shared_ptr<Algorithm> algo;
  if (!algoSelector_ && !fallbackAlgoSelector_) {
    THROW(ALGO, Error, ErrorCode::InvalidUsage,
          "No algorithm selector is set in AlgorithmCollection.");
  }
  if (algoSelector_) {
    algo = algoSelector_(algoMapByCollective_, request);
  }
  if (!algo) {
    algo = fallbackAlgoSelector_(algoMapByCollective_, request);
  }
  return algo;
}

void AlgorithmCollection::extend(AlgorithmCollection const& other) {
  for (auto const& [collective, algoMap] : other.algoMapByCollective_) {
    for (auto const& [algoName, algorithm] : algoMap) {
      this->registerAlgorithm(collective, algoName, algorithm);
    }
  }
}

void AlgorithmCollection::setSelectors(AlgoSelectFunc algoSelector,
                                       AlgoSelectFunc fallbackAlgoSelector) {
  algoSelector_ = algoSelector;
  fallbackAlgoSelector_ = fallbackAlgoSelector;
}

std::vector<std::shared_ptr<Algorithm>> AlgorithmCollection::getAllAlgorithms()
    const {
  std::vector<std::shared_ptr<Algorithm>> allAlgos;
  for (auto const& [collective, algoMap] : algoMapByCollective_) {
    for (auto const& [algoName, algorithm] : algoMap) {
      allAlgos.push_back(algorithm);
    }
  }
  return allAlgos;
}

std::unordered_map<std::string, std::shared_ptr<Algorithm>>
AlgorithmCollection::getAlgorithmsByCollective(
    std::string const& collective) const {
  auto it = algoMapByCollective_.find(collective);
  if (it != algoMapByCollective_.end()) {
    return it->second;
  } else {
    return {};
  }
}

DslAlgorithm::DslAlgorithm(std::string id, ExecutionPlan plan,
                           std::unordered_map<std::string, uint64_t> tags,
                           Constraint constraint)
    : plan_(plan), id_(id), tags_(tags), constraint_(constraint) {}

std::string const& DslAlgorithm::name() const { return plan_.name(); }

std::string const& DslAlgorithm::collective() const {
  return plan_.collective();
}

std::pair<size_t, size_t> const& DslAlgorithm::messageRange() const {
  static std::pair<size_t, size_t> range;
  range = {plan_.minMessageSize(), plan_.maxMessageSize()};
  return range;
}

void DslAlgorithm::setMessageSizeRange(size_t, size_t) {
  THROW(EXEC, Error, ErrorCode::InvalidUsage,
        "setMessageSizeRange is only supported for native algorithms");
}

std::unordered_map<std::string, uint64_t> const& DslAlgorithm::tags() const {
  return tags_;
}

CollectiveBufferMode const& DslAlgorithm::bufferMode() const {
  // TODO: need to fix
  static CollectiveBufferMode mode = plan_.isInPlace()
                                         ? CollectiveBufferMode::InPlace
                                         : CollectiveBufferMode::OutOfPlace;
  return mode;
}

Algorithm::Constraint DslAlgorithm::constraint() const { return constraint_; }

CommResult DslAlgorithm::execute(
    std::shared_ptr<Communicator> comm, void const* input, void* output,
    size_t inputSize, size_t outputSize, DataType dtype, ReduceOp,
    cudaStream_t stream, std::shared_ptr<Executor> executor, int, int, bool,
    std::unordered_map<std::string, uintptr_t> const&) {
  if (!executor) {
    THROW(EXEC, Error, ErrorCode::InvalidUsage,
          "Executor is null in DslAlgorithm::execute");
  }
  int rank = comm->bootstrap()->getRank();
  switch (dtype) {
    case DataType::FLOAT16:
      executor->execute(rank, (half*)input, (half*)output, inputSize,
                        outputSize, DataType::FLOAT16, plan_, stream);
      break;
    case DataType::FLOAT32:
      executor->execute(rank, (float*)input, (float*)output, inputSize,
                        outputSize, DataType::FLOAT32, plan_, stream);
      break;
    case DataType::BFLOAT16:
      executor->execute(rank, (__bfloat16*)input, (__bfloat16*)output,
                        inputSize, outputSize, DataType::BFLOAT16, plan_,
                        stream);
      break;
#if defined(__FP8_TYPES_EXIST__)
    case DataType::FLOAT8_E4M3:
      executor->execute(rank, (__fp8_e4m3*)input, (__fp8_e4m3*)output,
                        inputSize, outputSize, DataType::FLOAT8_E4M3, plan_,
                        stream);
      break;
    case DataType::FLOAT8_E5M2:
      executor->execute(rank, (__fp8_e5m2*)input, (__fp8_e5m2*)output,
                        inputSize, outputSize, DataType::FLOAT8_E5M2, plan_,
                        stream);
      break;
#endif
    case DataType::INT32:
    case DataType::UINT32:
      executor->execute(rank, (int*)input, (int*)output, inputSize, outputSize,
                        DataType::UINT32, plan_, stream);
      break;
    default:
      WARN(ALGO, "Unsupported data type: ", static_cast<int>(dtype),
           " in DslAlgorithm");
      return CommResult::CommInvalidArgument;
  }
  return CommResult::CommSuccess;
}

std::shared_ptr<Algorithm> DslAlgorithm::build() { return shared_from_this(); }

// TODO: implement this
void DslAlgorithm::reset() {}

static uint32_t* gDefaultFlagBuffer = nullptr;
static std::weak_ptr<void> gDefaultFlagBufferWeak;
static size_t gDefaultFlagCount = 128;

std::pair<std::shared_ptr<void>, size_t> getFlagBuffer() {
  auto ptr = gDefaultFlagBufferWeak.lock();
  if (!ptr) {
    if (!gDefaultFlagBuffer) {
      // Intentionally never freed — CUDA driver reclaims GPU memory at process
      // exit.
      gDefaultFlagBuffer = static_cast<uint32_t*>(
          mscclpp::detail::gpuCalloc(gDefaultFlagCount * sizeof(uint32_t)));
      std::vector<uint32_t> initFlags(gDefaultFlagCount, 1);
      mscclpp::gpuMemcpy(gDefaultFlagBuffer, initFlags.data(),
                         gDefaultFlagCount, cudaMemcpyHostToDevice);
    }
    ptr = std::shared_ptr<void>(gDefaultFlagBuffer, [](void*) {});
    gDefaultFlagBufferWeak = ptr;
  }
  return {ptr, gDefaultFlagCount * sizeof(uint32_t)};
}

}  // namespace mscclpp